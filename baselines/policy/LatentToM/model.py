"""XPolicyLab adapter for LatentToM (StanfordMSL, CoRL 2025).

LatentToM is two policies, a `DiffusionSheafSplitPolicy` per arm. `arm1` sees
`camera_1` (private) and `camera_3` (shared); `arm2` sees `camera_3` and
`camera_4` (private). Each arm is loaded from its own checkpoint
(`arm1_latest.ckpt` / `arm2_latest.ckpt`).

Proprio (43D per arm) and action (35D per arm) are joint-space, in the layout
`convert_to_replay_buffer.py` writes.
"""

import sys
from pathlib import Path

import dill
import hydra
import numpy as np
import torch

# The repository root: `baselines` imports from this checkout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_POLICY_DIR = Path(__file__).resolve().parent
if str(_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_DIR))

from diffusion_policy.env_runner.sheaf_runner import SheafRunner  # noqa: E402
from diffusion_policy.policy.diffusion_sheaf_split_policy import DiffusionSheafSplitPolicy  # noqa: E402
from baselines.utils.model_template import ModelTemplate  # noqa: E402
from baselines.utils.checkpoint_resolver import resolve_checkpoint_root  # noqa: E402
from baselines.utils.process_data import (  # noqa: E402
    get_action_dim,
    get_batch_size,
    get_robot_action_dim_info,
)

IMAGE_SIZE = (240, 320)  # (H, W) -- LatentToM's trained resolution.

_PROPRIO_DIM = 43  # cohub_keys.JOINTS_PER_ROBOT

# One robot's 35D action: left_arm(7) right_arm(7) left_hand(7) right_hand(7)
# waist(3) base_height(1) navigate(3). The first 31 are in
# cohub.keys.action_order() order.
_ROBOT_ACTION_DIM = 35
_JOINT_TARGETS = slice(0, 31)
_BASE_HEIGHT = slice(31, 32)
_NAVIGATE = slice(32, 35)


def _checkpoint_file_suffix(checkpoint_num) -> str:
    """eval's `checkpoint_num` -> which arm{1,2}_<suffix>.ckpt to serve.

    Empty or `latest` is the resumable checkpoint. A number is a weights-only
    snapshot. Anything else names the file outright (`best_val`).
    """
    value = "" if checkpoint_num is None else str(checkpoint_num).strip()
    if value.lower() in {"", "latest", "none"}:
        return "latest"
    if value.isdigit():
        return f"epoch={int(value):04d}"
    return value


def _load_arm_policy(ckpt_path: Path, arm_id: int, device: torch.device, use_ema: bool) -> DiffusionSheafSplitPolicy:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"LatentToM arm{arm_id} checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location=device)
    cfg = payload["cfg"]
    policy: DiffusionSheafSplitPolicy = hydra.utils.instantiate(cfg.policy)

    state_dicts = payload["state_dicts"]
    key_order = [f"arm{arm_id}_ema_model", f"arm{arm_id}_model"] if use_ema else [f"arm{arm_id}_model"]
    for key in key_order:
        if key in state_dicts:
            policy.load_state_dict(state_dicts[key])
            break
    else:
        raise KeyError(f"{ckpt_path} has none of {key_order} -- got {sorted(state_dicts)}")

    policy.to(device)
    policy.eval()
    return policy


def _predict_arm_action(policy: DiffusionSheafSplitPolicy, obs_dict: dict) -> torch.Tensor:
    """Two-stage predict_action.

    The first call returns the embeddings. The sheaf embedding is concatenated
    into `global_cond`, as training does, before the second call samples.
    """
    encoded = policy.predict_action(obs_dict, return_embedding_only=True, encoded_obs=None)
    encoded = dict(encoded)
    encoded["global_cond"] = torch.cat([encoded["global_cond"], encoded["sheaf_embedding"]], dim=-1)
    result = policy.predict_action(obs_dict=None, return_embedding_only=False, encoded_obs=encoded)
    return result["action"]  # (B, n_action_steps, _ROBOT_ACTION_DIM)


def _to_chw_float(color: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (3, h, w) float32 in [0, 1], resized to IMAGE_SIZE."""
    import cv2

    resized = cv2.resize(color, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)
    return np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0


def _robot_proprio_joint43(observation: dict, robot: str) -> np.ndarray:
    """This robot's 43 joint angles [rad] in cohub.keys.urdf_order() order.

    Read from `observation["cohub_state"]`. Zero-filled when absent.
    """
    cohub_state = observation.get("cohub_state")
    if cohub_state is not None and robot in cohub_state:
        joint_pos = cohub_state[robot].get("joint_pos")
        if joint_pos is not None:
            return np.asarray(joint_pos, dtype=np.float32)[:_PROPRIO_DIM]
    return np.zeros(_PROPRIO_DIM, dtype=np.float32)


def _encode_arm_obs(observation: dict, arm: int) -> dict:
    """XPolicyLab's standard obs dict -> one arm's LatentToM obs_dict.

    Cameras: `cam_left_wrist` -> `camera_1` (arm1 private), `cam_head` ->
    `camera_3` (shared), `cam_right_wrist` -> `camera_4` (arm2 private).
    `arm{N}_proprio` is that robot's 43 joint angles.
    """
    vision = observation["vision"]
    robot = "robot_a" if arm == 1 else "robot_b"
    proprio = _robot_proprio_joint43(observation, robot)

    obs = {}
    if arm == 1:
        obs["camera_1"] = _to_chw_float(vision["cam_left_wrist"]["color"])
        obs["camera_3"] = _to_chw_float(vision["cam_head"]["color"])
        obs["arm1_proprio"] = proprio
    else:
        obs["camera_3"] = _to_chw_float(vision["cam_head"]["color"])
        obs["camera_4"] = _to_chw_float(vision["cam_right_wrist"]["color"])
        obs["arm2_proprio"] = proprio
    return obs


def _unpack_robot_action(action: np.ndarray) -> dict:
    """One robot's raw 35D model output -> `{joint_targets(31), base_vel(3), height(1)}`."""
    return {
        "joint_targets": np.asarray(action[_JOINT_TARGETS], dtype=np.float32),
        "base_vel": np.asarray(action[_NAVIGATE], dtype=np.float32),
        "height": np.asarray(action[_BASE_HEIGHT], dtype=np.float32),
    }


def _pack_dual_arm_action(arm1_action: np.ndarray, arm2_action: np.ndarray) -> dict:
    """Repack one timestep's arm1 (robot_a) and arm2 (robot_b) output under `cohub_raw_action`."""
    return {
        "cohub_raw_action": {
            "robot_a": _unpack_robot_action(arm1_action),
            "robot_b": _unpack_robot_action(arm2_action),
        }
    }


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        if self.action_type != "joint":
            raise NotImplementedError(
                "LatentToM is joint-space only -- set deploy.yml's action_type: joint "
                f"(got {self.action_type!r}). See README.md's Model contract details."
            )

        self.action_dim = get_action_dim(self.env_cfg_type)
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.batch_size = get_batch_size(self.env_cfg_type)
        num_arms = len(self.robot_action_dim_info["arm_dim"])
        if num_arms != 2:
            raise NotImplementedError(
                f"LatentToM is a two-arm sheaf-split policy (arm1/arm2); env_cfg_type="
                f"{self.env_cfg_type!r} registers {num_arms} arm(s)."
            )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        policy_dir = Path(__file__).resolve().parent
        ckpt_root = resolve_checkpoint_root(
            model_cfg, policy_dir / "checkpoints", policy_dir=policy_dir, must_exist=True
        )
        use_ema = bool(model_cfg.get("use_ema", True))
        which = _checkpoint_file_suffix(model_cfg.get("checkpoint_num"))
        self.arm1_policy = _load_arm_policy(ckpt_root / "checkpoints" / f"arm1_{which}.ckpt", 1, self.device, use_ema)
        self.arm2_policy = _load_arm_policy(ckpt_root / "checkpoints" / f"arm2_{which}.ckpt", 2, self.device, use_ema)

        for arm_id, policy in ((1, self.arm1_policy), (2, self.arm2_policy)):
            if policy.action_dim != _ROBOT_ACTION_DIM:
                raise ValueError(
                    f"arm{arm_id} checkpoint has action_dim={policy.action_dim}, expected "
                    f"{_ROBOT_ACTION_DIM} -- this checkpoint wasn't trained on the "
                    "joint-space GR00T-convention data."
                )

        n_obs_steps = self.arm1_policy.n_obs_steps
        self.n_action_steps = self.arm1_policy.n_action_steps
        self.runner = SheafRunner(n_obs_steps=n_obs_steps)
        self._env_idx_list = None

        print(f"[LatentToM] loaded arm1/arm2 from {ckpt_root} (device={self.device}, use_ema={use_ema})")

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        env_idx_list = [obs["env_idx"] for obs in obs_list]
        arm1_obs_list = [_encode_arm_obs(obs, arm=1) for obs in obs_list]
        arm2_obs_list = [_encode_arm_obs(obs, arm=2) for obs in obs_list]
        self.runner.update_obs(arm1_obs_list, arm2_obs_list, env_idx_list)
        self._env_idx_list = env_idx_list

    def get_action(self):
        if not self._env_idx_list:
            raise RuntimeError("get_action() called before update_obs().")
        return self.get_action_batch(env_idx_list=[self._env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._env_idx_list
        if not env_idx_list:
            raise RuntimeError("get_action_batch() called before update_obs_batch().")

        arm1_obs, arm2_obs = self.runner.get_arm_obs_batch(env_idx_list, self.device)
        with torch.no_grad():
            arm1_action = _predict_arm_action(self.arm1_policy, arm1_obs).cpu().numpy()
            arm2_action = _predict_arm_action(self.arm2_policy, arm2_obs).cpu().numpy()

        num_envs, chunk_len = arm1_action.shape[0], arm1_action.shape[1]
        action_batch = []
        for b in range(num_envs):
            chunk = [_pack_dual_arm_action(arm1_action[b, t], arm2_action[b, t]) for t in range(chunk_len)]
            action_batch.append(chunk)
        return action_batch

    def reset(self):
        self.runner.reset()
        self._env_idx_list = None
