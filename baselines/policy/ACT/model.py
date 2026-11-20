import cv2
import numpy as np
import torch
from .detr.act_policy import ACT
from argparse import Namespace

from baselines.utils.model_template import ModelTemplate
from baselines.utils.process_data import pack_robot_state, unpack_robot_state, get_robot_action_dim_info
from baselines.utils.checkpoint_resolver import build_run_dir_name
import os
import pickle

# Env camera slot that carries each robot's ego view. cam_head is the fixed
# third-person camera.
COHUB_CAMERA_SLOT = {"robot_a": "cam_left_wrist", "robot_b": "cam_right_wrist", "robot_c": "cam_third_view"}
# robot_c exists on three-robot tasks only.
COHUB_OPTIONAL_ROBOTS = ("robot_c",)
# Both ego views in training order. A centralized checkpoint is served these.
COHUB_JOINT_CAMERAS = [COHUB_CAMERA_SLOT["robot_a"], COHUB_CAMERA_SLOT["robot_b"]]
COHUB_STATE_DIM = 43
COHUB_ACTION_DIM = 35


def _checkpoint_dims(ckpt_dir):
    """(state_dim, action_dim) a checkpoint was trained at, read from its dataset_stats.pkl."""
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"ACT dataset stats not found: {stats_path}")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    return int(np.size(stats["qpos_mean"])), int(np.size(stats["action_mean"]))


def _set_act_dims(cfg, state_dim, action_dim):
    """Export the widths detr's build() reads from the environment.

    `action_dim` in the config sizes the temporal aggregation buffer.
    """
    os.environ["ACT_STATE_DIM"] = str(state_dim)
    os.environ["ACT_ACTION_DIM"] = str(action_dim)
    cfg["action_dim"] = action_dim

# cv2.resize (width, height) of the training frames.
COHUB_IMAGE_SIZE = (320, 240)
UPSTREAM_IMAGE_SIZE = (640, 480)

class Model(ModelTemplate):

    def __init__(self, model_cfg):
        self._cohub_decentralized = (
            str(model_cfg.get("bench_name") or "") == "cohub"
            and model_cfg.get("env_cfg_type") == "unitree_g1x2_decentralized"
        )
        if self._cohub_decentralized:
            self._init_cohub_decentralized(model_cfg)
            return

        self.camera_names = model_cfg.get('camera_names', [])
        if str(model_cfg.get("bench_name") or "") == "cohub":
            self.camera_names = list(COHUB_JOINT_CAMERAS)
        model_cfg['camera_names'] = self.camera_names

        self.model = self.get_model(model_cfg=model_cfg)
        self.action_type = model_cfg['action_type']
        try:
            self.robot_action_dim_info = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        except FileNotFoundError:
            # CoHuB env_cfg_types have no env_cfg/<type>.yml. The CoHuB path
            # does not use this.
            self.robot_action_dim_info = None

    def get_model(self, model_cfg):
        if not model_cfg.get('ckpt_dir'):
            if not model_cfg.get('ckpt_name'):
                raise ValueError("ACT requires ckpt_name or ckpt_dir during evaluation.")
            # ckpt_name is the full run directory name under checkpoints/.
            model_cfg['ckpt_dir'] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'checkpoints', str(model_cfg['ckpt_name']))
        if str(model_cfg.get("bench_name") or "") == "cohub":
            state_dim, action_dim = _checkpoint_dims(model_cfg['ckpt_dir'])
            if (state_dim, action_dim) != (2 * COHUB_STATE_DIM, 2 * COHUB_ACTION_DIM):
                raise ValueError(
                    f"{model_cfg['ckpt_dir']} is {state_dim}D state / {action_dim}D action, not a centralized "
                    f"checkpoint -- a per-robot checkpoint is served with cohub_mode=decentralized")
            _set_act_dims(model_cfg, state_dim, action_dim)
        return ACT(model_cfg, Namespace(**model_cfg))

    def _init_cohub_decentralized(self, model_cfg):
        """Load one single-robot ACT checkpoint per robot, one policy instance each.

        `ckpt_name` is the task (e.g. "cocarry"), not a run dir. Each instance
        keeps its own observation and chunk position.
        """
        self.action_type = model_cfg['action_type']
        task = str(model_cfg.get('ckpt_name') or '').strip()
        if not task:
            raise ValueError("cohub decentralized eval needs ckpt_name=<task> (e.g. cocarry)")

        checkpoints_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        self._sub_models = {}
        for robot, camera_name in COHUB_CAMERA_SLOT.items():
            # An explicit deploy.yml override comes first.
            explicit = model_cfg.get(f'model_dir_{robot}')
            if explicit:
                ckpt_dir = explicit
            else:
                run_cfg = dict(model_cfg)
                run_cfg['ckpt_name'] = f"{task}_{robot}"
                run_name = build_run_dir_name(run_cfg)
                if run_name is None:
                    raise ValueError("bench_name/ckpt_name/env_cfg_type/action_type/seed required to name the run dir")
                # train.sh's ckpt_dir ends in the tag.
                tag = str(model_cfg.get('ckpt_tag') or '').strip()
                ckpt_dir = os.path.join(checkpoints_dir, f"{run_name}-{tag}" if tag else run_name)
            if not os.path.isdir(ckpt_dir):
                if robot in COHUB_OPTIONAL_ROBOTS and not explicit:
                    continue
                raise FileNotFoundError(f"{robot} checkpoint not found: {ckpt_dir}")

            sub_cfg = dict(model_cfg)
            sub_cfg['ckpt_dir'] = ckpt_dir
            sub_cfg['camera_names'] = [camera_name]
            state_dim, action_dim = _checkpoint_dims(ckpt_dir)
            if (state_dim, action_dim) != (COHUB_STATE_DIM, COHUB_ACTION_DIM):
                raise ValueError(
                    f"{robot}: {ckpt_dir} is {state_dim}D state / {action_dim}D action, not a per-robot "
                    f"checkpoint ({COHUB_STATE_DIM}/{COHUB_ACTION_DIM})")
            _set_act_dims(sub_cfg, state_dim, action_dim)
            self._sub_models[robot] = ACT(sub_cfg, Namespace(**sub_cfg))
            print(f"[ACT][cohub] {robot}: {ckpt_dir} (camera={camera_name})")

    def update_obs(self, obs):
        if self._cohub_decentralized:
            unserved = [r for r in (obs.get("cohub_state") or {}) if r not in self._sub_models]
            if unserved:
                raise FileNotFoundError(
                    f"the scene has {unserved} but no checkpoint was loaded for it "
                    f"(loaded: {list(self._sub_models)}); pass model_dir_{unserved[0]}")
            for robot, sub_model in self._sub_models.items():
                sub_model.update_obs(self._encode_cohub_robot_obs(obs, robot))
            return

        self._cohub_dual_robot = "cohub_state" in obs
        encoded_obs = self.encode_obs(obs, self.action_type, self.robot_action_dim_info)
        self.model.update_obs(encoded_obs)

    def _encode_cohub_robot_obs(self, observation, robot):
        camera_name = COHUB_CAMERA_SLOT[robot]
        color = cv2.resize(observation["vision"][camera_name]["color"], COHUB_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        color = np.moveaxis(color, -1, 0) / 255.0
        joint_pos = np.asarray(observation["cohub_state"][robot]["joint_pos"], dtype=np.float32)
        return {camera_name: color, "qpos": joint_pos}


    @staticmethod
    def _act_chunk(act) -> np.ndarray:
        """Return the action chunk as ``(n, action_dim)``.

        Under temporal aggregation the chunk is one action per call.
        """
        if act.temporal_agg:
            return np.atleast_2d(np.asarray(act.get_action()))
        return np.asarray(act.get_action_chunk())

    def get_action(self):
        if self._cohub_decentralized:
            per_robot = {
                robot: self._act_chunk(sub_model)
                for robot, sub_model in self._sub_models.items()
            }
            # Take the shortest chunk so the robots stay step-aligned.
            steps = min(chunk.shape[0] for chunk in per_robot.values())
            return [
                {
                    "cohub_raw_action": {
                        robot: self._pack_single_robot_action(chunk[t])
                        for robot, chunk in per_robot.items()
                    }
                }
                for t in range(steps)
            ]

        if getattr(self, "_cohub_dual_robot", False):
            return [
                {"cohub_raw_action": self._pack_dual_arm_action(a)}
                for a in self._act_chunk(self.model)
            ]

        # Other benches return one action per call.
        actions = self.model.get_action()
        return unpack_robot_state(actions, self.action_type, self.robot_action_dim_info, source_type='obs')

    @staticmethod
    def _pack_single_robot_action(flat_action: np.ndarray) -> dict:
        """One robot's 35D ACT action -> CoHuBTaskEnv.take_action's
        {joint_targets, base_vel, height}.

        Layout: [left_arm(7) right_arm(7) left_hand(7) right_hand(7) waist(3)
        base_height_command(1) navigate_command(3)]. The first 31 are the joint
        targets in cohub.keys.action_order() order.
        """
        assert flat_action.shape[-1] == 35, f"expected 35D per-robot action, got {flat_action.shape}"
        return {
            "joint_targets": flat_action[0:31],
            "height": flat_action[31:32],
            "base_vel": flat_action[32:35],
        }

    def _pack_dual_arm_action(self, flat_action: np.ndarray) -> dict:
        """70D flat action -> one dict per robot, each from its 35D slice."""
        assert flat_action.shape[-1] == 70, f"expected 70D dual-robot action, got {flat_action.shape}"
        return {
            robot: self._pack_single_robot_action(flat_action[i * 35 : (i + 1) * 35])
            for i, robot in enumerate(("robot_a", "robot_b"))
        }


    def reset(self):
        if self._cohub_decentralized:
            for sub_model in self._sub_models.values():
                self._reset_act_instance(sub_model)
            return
        self._reset_act_instance(self.model)

    @staticmethod
    def _reset_act_instance(act):
        if act.temporal_agg:
            act.all_time_actions = torch.zeros([
                act.max_timesteps,
                act.max_timesteps + act.num_queries,
                act.state_dim,
            ]).to(act.device)
        act.t = 0

    def encode_obs(self, observation, action_type, robot_action_dim_info):
        res_dict = dict()
        cohub_state = observation.get("cohub_state")
        image_size = COHUB_IMAGE_SIZE if cohub_state is not None else UPSTREAM_IMAGE_SIZE

        for camera_name in self.camera_names:
            if camera_name not in observation["vision"]:
                raise ValueError(f"Expected camera '{camera_name}' not found in observation['vision']")
            color = cv2.resize(observation["vision"][camera_name]["color"], image_size, interpolation=cv2.INTER_LINEAR)
            color = np.moveaxis(color, -1, 0) / 255.0
            res_dict[camera_name] = color

        if cohub_state is not None:
            # CoHuB state is 86D: all 7 URDF joint groups per robot, in
            # cohub.keys.urdf_order() order.
            res_dict["qpos"] = np.concatenate([
                np.asarray(cohub_state["robot_a"]["joint_pos"], dtype=np.float32),
                np.asarray(cohub_state["robot_b"]["joint_pos"], dtype=np.float32),
            ])
        else:
            res_dict["qpos"] = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs")

        return res_dict