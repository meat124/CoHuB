import torch
import yaml
import cv2
import numpy as np
import hydra
import dill
import sys, os
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner
from baselines.utils.model_template import ModelTemplate
from baselines.utils.process_data import pack_robot_state, unpack_robot_state, get_robot_action_dim_info
from baselines.utils.checkpoint_resolver import resolve_checkpoint_root, build_run_dir_name

# Env camera slot that carries each robot's ego view.
COHUB_CAMERA_SLOT = {"robot_a": "cam_left_wrist", "robot_b": "cam_right_wrist", "robot_c": "cam_third_view"}
# robot_c exists on three-robot tasks only. Its checkpoint is loaded when
# model_dir_robot_c is set or its default run directory exists.
COHUB_OPTIONAL_ROBOTS = ("robot_c",)


def _prep_camera(color: np.ndarray) -> np.ndarray:
    """XPolicyLab vision obs (HWC uint8) -> DP's CHW float32 input, 320x240."""
    img = np.moveaxis(color, -1, 0) / 255.0
    return np.transpose(
        cv2.resize(np.transpose(img, (1, 2, 0)), (320, 240), interpolation=cv2.INTER_AREA), (2, 0, 1)
    )


DDPM_ONLY_CONFIG_KEYS = ("variance_type",)


def configure_sampler(policy, scheduler_name, num_inference_steps):
    """Choose the denoiser used at evaluation. The weights are unchanged.

    With both arguments None the checkpoint's own sampler is kept. `ddim` is a
    different solver that allows fewer inference steps. DDPM-only settings
    are filtered out because DDIM's `step` takes no `**kwargs`.
    """
    if scheduler_name is None and num_inference_steps is None:
        return policy

    name = (scheduler_name or "ddpm").lower()
    if name not in ("ddpm", "ddim"):
        raise ValueError(f"inference_scheduler must be 'ddpm' or 'ddim', got {scheduler_name!r}")

    if name == "ddim" and not isinstance(policy.noise_scheduler, DDIMScheduler):
        import inspect

        trained = dict(policy.noise_scheduler.config)
        carried = {
            key: value
            for key, value in trained.items()
            if key not in DDPM_ONLY_CONFIG_KEYS and not key.startswith("_")
        }
        # The Diffusion Policy release's DDIM settings.
        carried.setdefault("set_alpha_to_one", True)
        carried.setdefault("steps_offset", 0)
        policy.noise_scheduler = DDIMScheduler(**carried)

        allowed = set(inspect.signature(DDIMScheduler.step).parameters)
        dropped = {k: v for k, v in policy.kwargs.items() if k not in allowed}
        if dropped:
            print(f"[DP][sampler] dropping step kwargs DDIM does not take: {sorted(dropped)}")
            policy.kwargs = {k: v for k, v in policy.kwargs.items() if k in allowed}

    if num_inference_steps is not None:
        policy.num_inference_steps = int(num_inference_steps)

    print(
        f"[DP][sampler] {type(policy.noise_scheduler).__name__} "
        f"num_inference_steps={policy.num_inference_steps} "
        f"(trained: {policy.noise_scheduler.config.num_train_timesteps} steps)"
    )
    return policy


def _pack_single_robot_action(flat_action: np.ndarray) -> dict:
    """One robot's 35D DP action -> CoHuBTaskEnv.take_action's
    {joint_targets, base_vel, height}. Same 35D layout as ACT's model.py."""
    assert flat_action.shape[-1] == 35, f"expected 35D per-robot action, got {flat_action.shape}"
    return {
        "joint_targets": flat_action[0:31],
        "height": flat_action[31:32],
        "base_vel": flat_action[32:35],
    }


def _pack_dual_arm_action(flat_action: np.ndarray) -> dict:
    """This adapter's 70D flat action -> one dict per robot."""
    assert flat_action.shape[-1] == 70, f"expected 70D dual-robot action, got {flat_action.shape}"
    return {
        robot: _pack_single_robot_action(flat_action[i * 35 : (i + 1) * 35])
        for i, robot in enumerate(("robot_a", "robot_b"))
    }


_CLIP_OFF = ("", "none", "null", "off", "false", "0")
_CLIP_ON = ("fitted", "true", "1")


def clip_enabled(value, name):
    """Whether `value` asks for clamping to the checkpoint's fitted range.

    Off is None or one of the strings in _CLIP_OFF (e.g. "None", "null").
    """
    if value is None:
        return False
    token = str(value).strip().lower()
    if token in _CLIP_OFF:
        return False
    if token in _CLIP_ON:
        return True
    raise ValueError(f"{name} must be 'fitted' or None, got {value!r}")


def fitted_obs_bounds(policy):
    """The per-dimension [min, max] of `agent_pos` that the checkpoint's
    normalizer was fitted on."""
    stats = policy.normalizer.params_dict["agent_pos"]["input_stats"]
    return (
        stats["min"].detach().cpu().numpy().astype(np.float32),
        stats["max"].detach().cpu().numpy().astype(np.float32),
    )


def fitted_action_bounds(policy):
    """The per-dimension [min, max] of the action that the checkpoint's
    normalizer was fitted on, as numpy."""
    stats = policy.normalizer.params_dict["action"]["input_stats"]
    return (
        stats["min"].detach().cpu().numpy().astype(np.float32),
        stats["max"].detach().cpu().numpy().astype(np.float32),
    )


class Model(ModelTemplate):

    def __init__(self, model_cfg):
        # Fallback values. `_load_policy` overwrites both from the checkpoint.
        load_config_path = os.path.join(parent_dir, 'diffusion_policy/config/robot_dp.yaml')
        with open(load_config_path, "r", encoding="utf-8") as f:
            model_training_config = yaml.safe_load(f)
        self.n_obs_steps = model_training_config['n_obs_steps']
        self.n_action_steps = model_training_config['n_action_steps']
        self.action_type = model_cfg['action_type']
        self._dump_calls = 0
        self._dump_episode = -1
        self._dump_last_obs = None
        # Evaluation-time sampler. None keeps the checkpoint's own.
        self._inference_scheduler = model_cfg.get('inference_scheduler')
        self._num_inference_steps = model_cfg.get('num_inference_steps')
        # 'fitted' clamps the action to the checkpoint's fitted range. Must be
        # set before _init_cohub_decentralized, which loads the policies.
        self._action_clip = model_cfg.get('action_clip')
        # 'fitted' clamps agent_pos to the checkpoint's fitted range.
        self._obs_clip = model_cfg.get('obs_clip')
        self._obs_bounds_of = {}
        # Keyed by policy: each checkpoint has its own fitted ranges.
        self._bounds_of = {}

        self._cohub_decentralized = (
            str(model_cfg.get("bench_name") or "") == "cohub"
            and model_cfg.get("env_cfg_type") == "unitree_g1x2_decentralized"
        )
        if self._cohub_decentralized:
            self._init_cohub_decentralized(model_cfg)
            return

        self._cohub_dual_robot = False  # set per-batch in update_obs_batch
        self.model = self.get_model(model_cfg=model_cfg)
        self.runner = DPRunner(n_obs_steps=self.n_obs_steps, n_action_steps=self.n_action_steps)
        try:
            self.robot_action_dim_info = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        except FileNotFoundError:
            # CoHuB env_cfg_types have no env_cfg/<type>.yml. The CoHuB path
            # does not use this.
            self.robot_action_dim_info = None
        self._latest_env_idx_list = None

    def _init_cohub_decentralized(self, model_cfg):
        """Load single-robot DP checkpoints, one policy instance per robot.

        `ckpt_name` is the task (e.g. "cocarry"), not a run dir. Each robot's
        checkpoint sees its own ego view (head_cam) and its own 43 joints and
        predicts its own 35-D action; the policies are loaded once per robot
        and each instance is fed its own robot's obs.
        """
        task = str(model_cfg.get('ckpt_name') or '').strip()
        if not task:
            raise ValueError("cohub decentralized eval needs ckpt_name=<task> (e.g. cocarry)")

        checkpoints_dir = os.path.join(parent_dir, "checkpoints")
        self._sub_policies = {}
        self._sub_runners = {}
        for robot, camera_name in COHUB_CAMERA_SLOT.items():
            explicit = model_cfg.get(f'model_dir_{robot}')
            if explicit:
                ckpt_dir = explicit
            else:
                run_cfg = dict(model_cfg)
                run_cfg['ckpt_name'] = f"{task}_{robot}"
                run_name = build_run_dir_name(run_cfg)
                if run_name is None:
                    raise ValueError("bench_name/ckpt_name/env_cfg_type/action_type/seed required to name the run dir")
                ckpt_dir = os.path.join(checkpoints_dir, run_name)
            if not os.path.isdir(ckpt_dir):
                if robot in COHUB_OPTIONAL_ROBOTS and not explicit:
                    continue
                raise FileNotFoundError(f"{robot} checkpoint not found: {ckpt_dir}")

            self._sub_policies[robot] = self._load_policy(ckpt_dir, model_cfg.get('checkpoint_num', 'latest'))
            # After _load_policy, which sets the step counts from the checkpoint.
            self._sub_runners[robot] = DPRunner(n_obs_steps=self.n_obs_steps, n_action_steps=self.n_action_steps)
            print(f"[DP][cohub] {robot}: {ckpt_dir} (camera={camera_name})")

    def _load_policy(self, ckpt_dir, checkpoint_num):
        ckpt_file = self._resolve_checkpoint_file(ckpt_dir, checkpoint_num)

        # load checkpoint and workspace
        payload = torch.load(open(ckpt_file, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=None)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        configure_sampler(policy, self._inference_scheduler, self._num_inference_steps)

        # The checkpoint decides the observation window and the chunk length.
        # With n_obs_steps > 1 the rollout loop must observe every step
        # (deploy.py's OBS_STRIDE).
        served_obs = int(getattr(policy, "n_obs_steps", self.n_obs_steps))
        served_action = int(getattr(policy, "n_action_steps", self.n_action_steps))
        if (served_obs, served_action) != (self.n_obs_steps, self.n_action_steps):
            print(f"[DP] checkpoint was trained with n_obs_steps={served_obs} "
                  f"n_action_steps={served_action}; robot_dp.yaml says "
                  f"{self.n_obs_steps}/{self.n_action_steps} -- serving the checkpoint's")
        self.n_obs_steps, self.n_action_steps = served_obs, served_action
        if served_obs > 1:
            print(f"[DP] n_obs_steps={served_obs}: the rollout loop must observe every step "
                  f"(EVAL_OBS_STRIDE=1), or the observation window is padded, not real")

        device = torch.device("cuda:0")
        policy.to(device)
        policy.eval()

        if clip_enabled(self._action_clip, "action_clip"):
            low, high = fitted_action_bounds(policy)
            print(f"[DP][action_clip] clamping to the fitted action range, "
                  f"{int(((high - low) < 1e-4).sum())} of {low.size} dims have zero range")
            self._bounds_of[id(policy)] = (low, high)

        if clip_enabled(self._obs_clip, "obs_clip"):
            low, high = fitted_obs_bounds(policy)
            gain = 2.0 / np.maximum(high - low, 1e-9)
            print(f"[DP][obs_clip] clamping agent_pos to the fitted range; "
                  f"largest normalization gain {gain.max():.0f} per unit on dim {int(gain.argmax())}")
            self._obs_bounds_of[id(policy)] = (low, high)

        return policy

    def get_model(self, model_cfg):
        ckpt_dir = resolve_checkpoint_root(
            model_cfg,
            os.path.join(parent_dir, "checkpoints"),
            policy_dir=parent_dir,
            must_exist=False,
        )
        return self._load_policy(ckpt_dir, model_cfg.get('checkpoint_num', 'latest'))

    def _resolve_checkpoint_file(self, ckpt_dir, checkpoint_num):
        ckpt_dir = os.fspath(ckpt_dir)
        checkpoint_num = "latest" if checkpoint_num is None else str(checkpoint_num)

        if checkpoint_num.lower() not in {"", "latest", "none"}:
            ckpt_file = os.path.join(ckpt_dir, f"{checkpoint_num}.ckpt")
            if not os.path.isfile(ckpt_file):
                raise FileNotFoundError(f"DP checkpoint not found: {ckpt_file}")
            return ckpt_file

        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"DP checkpoint directory not found: {ckpt_dir}")

        # `latest` is the highest-numbered checkpoint, not the trainer's `latest.ckpt`.
        candidates = []
        for name in os.listdir(ckpt_dir):
            if not name.endswith(".ckpt"):
                continue
            stem = name[:-5]
            if stem.isdigit():
                candidates.append((int(stem), os.path.join(ckpt_dir, name)))

        if not candidates:
            raise FileNotFoundError(f"No numeric DP checkpoints found under: {ckpt_dir}")

        return max(candidates, key=lambda item: item[0])[1]

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        env_idx_list = [obs["env_idx"] for obs in obs_list]

        if self._cohub_decentralized:
            unserved = [r for r in (obs_list[0].get("cohub_state") or {}) if r not in self._sub_runners] if obs_list else []
            if unserved:
                raise FileNotFoundError(
                    f"the scene has {unserved} but no checkpoint was loaded for it "
                    f"(loaded: {list(self._sub_runners)}); pass model_dir_{unserved[0]}")
            merged = {}
            for robot, runner in self._sub_runners.items():
                encoded = [self._encode_cohub_robot_obs(obs, robot) for obs in obs_list]
                for item in encoded:
                    item["agent_pos"] = self._clip_obs(item["agent_pos"], self._sub_policies[robot])
                if encoded:
                    merged.update({f"{robot}_{k}": v for k, v in encoded[0].items()})
                runner.update_obs(encoded, env_idx_list)
            self._dump_last_obs = merged or None
            self._latest_env_idx_list = env_idx_list
            return

        self._cohub_dual_robot = bool(obs_list) and "cohub_state" in obs_list[0]
        encoded_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, self._cohub_dual_robot)
            for obs in obs_list
        ]
        for encoded in encoded_list:
            encoded["agent_pos"] = self._clip_obs(encoded["agent_pos"], self.model)
        self._dump_last_obs = encoded_list[0] if encoded_list else None
        self.runner.update_obs(encoded_list, env_idx_list)
        self._latest_env_idx_list = env_idx_list

    def _encode_cohub_robot_obs(self, observation, robot):
        """One per-robot policy's obs: head_cam is the robot's own ego view
        and agent_pos its own joints."""
        camera_name = COHUB_CAMERA_SLOT[robot]
        head_img = _prep_camera(observation["vision"][camera_name]["color"])
        agent_pos = np.asarray(observation["cohub_state"][robot]["joint_pos"], dtype=np.float32)
        return dict(head_cam=head_img, agent_pos=agent_pos)

    def get_action(self):
        if not self._latest_env_idx_list:
            raise RuntimeError("get_action() called before update_obs().")

        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])

        return action_list[0]

    def _clip_obs(self, agent_pos, policy):
        """Clamp agent_pos to the range `policy`'s normalizer was fitted on,
        when `obs_clip` asks for it."""
        bounds = self._obs_bounds_of.get(id(policy))
        if bounds is None:
            return agent_pos
        low, high = bounds
        if low.size != agent_pos.shape[-1]:
            raise ValueError(
                f"obs_clip bounds are {low.size}D but agent_pos is {agent_pos.shape[-1]}D"
            )
        return np.clip(agent_pos, low, high)

    def _clip_action(self, actions, policy):
        """Clamp a (..., D) action to the range `policy`'s own normalizer was
        fitted on, when `action_clip` asks for it."""
        bounds = self._bounds_of.get(id(policy))
        if bounds is None:
            return actions
        low, high = bounds
        if low.size != actions.shape[-1]:
            raise ValueError(
                f"action_clip bounds are {low.size}D but the action is {actions.shape[-1]}D"
            )
        return np.clip(actions, low, high)

    # Diagnostics. COHUB_DP_DUMP=<dir> writes the inputs and outputs of every
    # get_action call. COHUB_DP_DUMP_LIMIT caps the calls dumped per episode
    # (default 400).
    def _dump_dir(self):
        root = os.environ.get("COHUB_DP_DUMP")
        if not root:
            return None
        os.makedirs(root, exist_ok=True)
        return root

    def _dump(self, obs_encoded, actions, action_pred=None):
        root = self._dump_dir()
        if root is None:
            return
        limit = int(os.environ.get("COHUB_DP_DUMP_LIMIT", "400"))
        if self._dump_calls >= limit:
            return
        payload = {"call": np.int64(self._dump_calls), "episode": np.int64(self._dump_episode)}
        for key, value in obs_encoded.items():
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[0] == 3:           # CHW float image
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            payload[f"obs_{key}"] = arr
        payload["action"] = np.asarray(actions, dtype=np.float32)
        if action_pred is not None:
            payload["action_pred"] = np.asarray(action_pred, dtype=np.float32)
        np.savez_compressed(
            os.path.join(root, f"ep{self._dump_episode:03d}_call{self._dump_calls:04d}.npz"),
            **payload,
        )
        self._dump_calls += 1

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        if not env_idx_list:
            raise RuntimeError("get_action_batch() called before update_obs_batch().")

        if self._cohub_decentralized:
            per_robot = {
                robot: self._clip_action(
                    runner.get_action(self._sub_policies[robot], env_idx_list),
                    self._sub_policies[robot],
                )
                for robot, runner in self._sub_runners.items()
            }  # each: (len(env_idx_list), n_action_steps, 35)
            if self._dump_last_obs is not None:
                self._dump(
                    self._dump_last_obs,
                    np.concatenate([chunk[0] for chunk in per_robot.values()], axis=-1),
                )
            steps = min(arr.shape[1] for arr in per_robot.values())
            return [
                [
                    {
                        "cohub_raw_action": {
                            robot: _pack_single_robot_action(per_robot[robot][env_i, t])
                            for robot in per_robot
                        }
                    }
                    for t in range(steps)
                ]
                for env_i in range(len(env_idx_list))
            ]

        actions = self.runner.get_action(self.model, env_idx_list)  # (len(env_idx_list), n_action_steps, action_dim)
        actions = self._clip_action(actions, self.model)
        if self._dump_last_obs is not None:
            self._dump(self._dump_last_obs, actions[0])

        if self._cohub_dual_robot:
            return [
                [{"cohub_raw_action": _pack_dual_arm_action(actions[i, t])} for t in range(actions.shape[1])]
                for i in range(len(env_idx_list))
            ]

        return [
            unpack_robot_state(actions[i], self.action_type, self.robot_action_dim_info, source_type='obs')
            for i in range(len(env_idx_list))
        ]

    def reset(self):
        self._dump_episode += 1
        self._dump_calls = 0
        if self._cohub_decentralized:
            for runner in self._sub_runners.values():
                runner.reset_obs()
            self._latest_env_idx_list = None
            return
        self.runner.reset_obs()
        self._latest_env_idx_list = None

def encode_obs(observation, action_type, robot_action_dim_info, cohub_dual_robot=False):
    left_cam = _prep_camera(observation["vision"]["cam_left_wrist"]["color"])
    right_cam = _prep_camera(observation["vision"]["cam_right_wrist"]["color"])

    if cohub_dual_robot:
        # CoHuB state is 86D and does not fit pack_robot_state. Only the two
        # ego cameras are used; cam_head is the scene camera.
        cohub_state = observation["cohub_state"]
        agent_pos = np.concatenate([
            np.asarray(cohub_state["robot_a"]["joint_pos"], dtype=np.float32),
            np.asarray(cohub_state["robot_b"]["joint_pos"], dtype=np.float32),
        ])
        return dict(left_cam=left_cam, right_cam=right_cam, agent_pos=agent_pos)

    head_img = _prep_camera(observation["vision"]["cam_head"]["color"])
    agent_pos = pack_robot_state(observation, action_type, robot_action_dim_info, source_type='obs')
    return dict(head_cam=head_img, left_cam=left_cam, right_cam=right_cam, agent_pos=agent_pos)
