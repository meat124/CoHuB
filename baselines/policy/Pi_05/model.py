#!/usr/bin/env python
# -- coding: UTF-8
"""pi0.5 adapter for the policy server."""
from pathlib import Path
import os
import sys
from typing import Any

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

from baselines.utils.model_template import ModelTemplate
from baselines.utils.checkpoint_resolver import build_run_dir_name, candidate_checkpoint_roots
from baselines.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"


# CoHuB: Unitree G1 humanoids, 43 joint angles and one ego camera per robot.
COHUB_BENCH_NAME = "cohub"

# env_cfg_type the checkpoints were trained under; run directories are named with it.
COHUB_CENTRALIZED_ENV_CFG_TYPE = "unitree_g1x2_centralized"
COHUB_DECENTRALIZED_ENV_CFG_TYPE = "unitree_g1x2_decentralized"

COHUB_ROBOTS = ("robot_a", "robot_b")
COHUB_ROBOTS3 = ("robot_a", "robot_b", "robot_c")
"""Every robot name a task may have. The observation says which are present."""


def _cohub_robots(observation: dict) -> tuple:
    present = observation.get("cohub_state") or {}
    return tuple(robot for robot in COHUB_ROBOTS3 if robot in present) or COHUB_ROBOTS

# Vision slot that carries each robot's ego camera.
COHUB_VIDEO_SLOT = {"ego_a": "cam_left_wrist", "ego_b": "cam_right_wrist",
                      "ego_c": "cam_third_view"}
COHUB_EGO_VIEW = {"robot_a": "ego_a", "robot_b": "ego_b", "robot_c": "ego_c"}

# Run name of the shared decentralized policy.
COHUB_MULTITASK_CKPT = "multitask"
COHUB_MULTITASK_CKPTS = ("multitask", "multitask3")
"""Shared runs: the two-robot task set and the three-robot task set."""

# Per-robot action: 31 joint targets, 1 height, 3 base velocities.
COHUB_JOINT_TARGET_DIM = 31
COHUB_ACTION_DIM = 35


def _cohub_mode(model_cfg: dict[str, Any]) -> str:
    mode = str(model_cfg.get("cohub_mode") or "decentralized").strip().lower()
    if mode not in ("centralized", "decentralized"):
        raise ValueError(f"cohub_mode must be 'centralized' or 'decentralized', got {mode!r}")
    return mode


def _cohub_train_config_name(task: str, robot: str | None) -> str:
    """Name of the TrainConfig the checkpoint was trained under, derived from task and target."""
    if task in COHUB_MULTITASK_CKPTS:
        return f"pi05_cohub_{task}_decentralized"
    return f"pi05_cohub_{task}_{robot or 'centralized'}"


def _cohub_model_dir(model_cfg: dict[str, Any], task: str, robot: str | None) -> Path:
    """`checkpoints/cohub-<task>[_<robot>]-<env_cfg_type>-joint-<seed>/<step>`."""
    explicit = model_cfg.get(
        "model_dir" if robot is None or task in COHUB_MULTITASK_CKPTS else f"model_dir_{robot}"
    )
    if explicit:
        return _resolve_pi05_model_root({**model_cfg, "model_path": explicit})

    run_cfg = dict(model_cfg)
    if robot is None:
        run_cfg["ckpt_name"] = task
        run_cfg["env_cfg_type"] = COHUB_CENTRALIZED_ENV_CFG_TYPE
    elif task in COHUB_MULTITASK_CKPTS:
        run_cfg["ckpt_name"] = task
        run_cfg["env_cfg_type"] = COHUB_DECENTRALIZED_ENV_CFG_TYPE
    else:
        run_cfg["ckpt_name"] = f"{task}_{robot}"
        run_cfg["env_cfg_type"] = COHUB_DECENTRALIZED_ENV_CFG_TYPE
    run_name = build_run_dir_name(run_cfg)
    if run_name is None:
        raise ValueError("bench_name/ckpt_name/env_cfg_type/action_type/seed are required to name the run dir")

    run_dir = _CHECKPOINTS_DIR / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{robot or 'centralized'} checkpoint not found: {run_dir}")
    return _resolve_pi05_model_root({**model_cfg, "model_path": str(run_dir)})


def _encode_cohub_obs(observation: dict[str, Any]) -> dict[str, Any]:
    """CoHuBTaskEnv observation -> the keys `CoHuBInputs` reads (post-repack names).

    The state is every robot's joint vector concatenated; `cohub_policy` slices it per target.
    """
    state = observation["cohub_state"]
    robots = _cohub_robots(observation)
    encoded: dict[str, Any] = {
        "observation/state": np.concatenate(
            [np.asarray(state[robot]["joint_pos"], dtype=np.float32) for robot in robots]
        )
    }
    # Only the views this observation carries.
    vision = observation.get("vision") or {}
    for camera, slot in COHUB_VIDEO_SLOT.items():
        view = vision.get(slot)
        if view is None:
            continue
        encoded[f"observation/{camera}"] = np.asarray(view["color"])
    return encoded


def _encode_cohub_shared_obs(observation: dict[str, Any], robot: str) -> dict[str, Any]:
    """One robot's 43 joint angles and ego camera, as `CoHuBSharedInputs` reads them."""
    return {
        "observation/state": np.asarray(
            observation["cohub_state"][robot]["joint_pos"], dtype=np.float32
        ),
        "observation/ego": np.asarray(
            observation["vision"][COHUB_VIDEO_SLOT[COHUB_EGO_VIEW[robot]]]["color"]
        ),
    }


def _pack_cohub_robot_action(flat: np.ndarray) -> dict[str, np.ndarray]:
    if flat.shape[-1] != COHUB_ACTION_DIM:
        raise ValueError(f"expected a {COHUB_ACTION_DIM}D per-robot action, got {flat.shape}")
    return {
        "joint_targets": flat[:COHUB_JOINT_TARGET_DIM],
        "height": flat[COHUB_JOINT_TARGET_DIM : COHUB_JOINT_TARGET_DIM + 1],
        "base_vel": flat[COHUB_JOINT_TARGET_DIM + 1 :],
    }


def _extract_step_number(value: Any) -> int | None:
    matches = [part for part in str(value).split("/") if part]
    if not matches:
        return None
    digits = "".join(ch for ch in matches[-1] if ch.isdigit())
    return int(digits) if digits else None


def _resolve_pi05_model_root(model_cfg: dict[str, Any]) -> Path:
    # Precedence: model_path/checkpoint_path > ckpt_name as a path >
    # {bench}-{ckpt}-{env}-{action}-{seed} > checkpoints/<ckpt_name>.
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name or model_path is required for Pi_05.")
    checkpoint_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_dirs = []
    if (checkpoint_root / "params").exists() or (checkpoint_root / "assets").exists():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and ((child / "params").exists() or (child / "assets").exists())
    )
    if not candidate_dirs:
        return checkpoint_root

    checkpoint_num = model_cfg.get("checkpoint_num")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        normalized = str(desired_step)
        for candidate in candidate_dirs:
            name = candidate.name.lstrip("0") or "0"
            if name == normalized:
                return candidate

        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step is None:
                continue
            scaled_step = desired_step
            while len(str(scaled_step)) < len(str(candidate_step)):
                scaled_step *= 10
            if candidate_step in {desired_step, scaled_step}:
                return candidate

    numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_dirs[0]


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]

        self._cohub = str(model_cfg.get("bench_name") or "") == COHUB_BENCH_NAME
        if self._cohub:
            self._init_cohub(model_cfg)
            return

        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg_type"]) if model_cfg.get("env_cfg_type") is not None else None
        )
        self.policy = self.get_model(model_cfg=model_cfg)
        self.model = self.policy

    # CoHuB

    def _init_cohub(self, model_cfg: dict[str, Any]) -> None:
        """Load the policies that drive the robots.

        centralized: one policy with all 70 action dimensions.
        decentralized, `ckpt_name: multitask`: one shared policy, queried once
        per agent with its own camera, state and instruction.
        decentralized, `ckpt_name: <task>`: one policy per robot.
        """
        self.robot_action_dim_info = None
        self._cohub_mode = _cohub_mode(model_cfg)
        task = str(model_cfg.get("ckpt_name") or "").strip()
        if not task:
            raise ValueError("cohub eval needs ckpt_name=<task> (e.g. cocarry)")

        self._cohub_shared = task in COHUB_MULTITASK_CKPTS
        targets: tuple[str | None, ...] = (
            (None,) if self._cohub_mode == "centralized" else COHUB_ROBOTS
        )
        # The shared checkpoint is loaded once.
        load_targets = targets[:1] if self._cohub_shared else targets
        self._XPolicyLab: dict[str | None, Any] = {}
        for robot in load_targets:
            config_name = _cohub_train_config_name(task, robot)
            model_dir = _cohub_model_dir(model_cfg, task, robot)
            config = _config.get_config(config_name)
            # Norm stats load from <model_dir>/assets/<asset_id>.
            self._XPolicyLab[robot] = _policy_config.create_trained_policy(config, str(model_dir))
            print(f"[Pi_05][cohub] {robot or 'centralized'}: {config_name} <- {model_dir}")
        if self._cohub_shared:
            for robot in COHUB_ROBOTS3:
                self._XPolicyLab[robot] = self._XPolicyLab[load_targets[0]]

        horizon = _config.get_config(_cohub_train_config_name(task, targets[0])).model.action_horizon
        exec_horizon = model_cfg.get("exec_horizon")
        self._cohub_exec_horizon = min(int(exec_horizon), horizon) if exec_horizon else horizon

        self.policy = self._XPolicyLab[targets[0]]
        self.model = self.policy

    def _cohub_update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self.observation_window = [_encode_cohub_obs(obs) for obs in obs_list]
        # The shared policy encodes per agent from the raw observation.
        self.observation_window_raw = list(obs_list)

    def _cohub_get_action_batch(self, env_idx_list: list[int] | None) -> list[list[dict[str, Any]]]:
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")
        env_idx_list = env_idx_list or self._latest_env_idx_list

        batch: list[list[dict[str, Any]]] = []
        for index in range(len(env_idx_list)):
            observation = self.observation_window[index]
            if self._cohub_mode == "centralized":
                actions = np.asarray(self._XPolicyLab[None].infer(observation)["actions"])
                per_robot = {
                    robot: actions[:, i * COHUB_ACTION_DIM : (i + 1) * COHUB_ACTION_DIM]
                    for i, robot in enumerate(COHUB_ROBOTS)
                }
            elif self._cohub_shared:
                source = self.observation_window_raw[index]
                per_robot = {}
                for robot in _cohub_robots(source):
                    encoded = _encode_cohub_shared_obs(source, robot)
                    prompt = (source.get("cohub_instruction") or {}).get(robot)
                    if prompt:
                        encoded["prompt"] = str(prompt)
                    per_robot[robot] = np.asarray(
                        self._XPolicyLab[robot].infer(encoded)["actions"]
                    )
            else:
                per_robot = {
                    robot: np.asarray(policy.infer(observation)["actions"])
                    for robot, policy in self._XPolicyLab.items()
                }

            steps = min(self._cohub_exec_horizon, min(a.shape[0] for a in per_robot.values()))
            batch.append(
                [
                    {
                        "cohub_raw_action": {
                            robot: _pack_cohub_robot_action(chunk[step])
                            for robot, chunk in per_robot.items()
                        }
                    }
                    for step in range(steps)
                ]
            )
        return batch

    def get_model(self, model_cfg: dict[str, Any]):
        train_config_name = model_cfg.get("train_config_name", "pi05_aloha")
        repo_id = model_cfg.get("repo_id", "1118")
        model_root = _resolve_pi05_model_root(model_cfg)

        config = _config.get_config(train_config_name)
        norm_stats = None
        if repo_id is not None:
            norm_stats = _normalize.load(model_root / "assets" / str(repo_id))

        return _policy_config.create_trained_policy(config, str(model_root), norm_stats=norm_stats)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if self._cohub:
            self._cohub_update_obs_batch(obs_list)
            return
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info) for obs in obs_list
        ]
        self.observation_window = stack_obs(encoded_obs_list)

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self._cohub:
            return self._cohub_get_action_batch(env_idx_list)

        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            single_observation = slice_stacked_obs(self.observation_window, batch_index)
            actions = self.policy.infer(single_observation, **kwargs)["actions"]
            if self.robot_action_dim_info is None:
                action_list.append(actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]

    def reset_obsrvationwindows(self):
        self.reset()

    def seed(self, seed: int) -> None:
        """Reset every openpi policy's JAX sampling key to `seed`."""
        import jax

        policies = list(getattr(self, "_XPolicyLab", {}).values()) or [getattr(self, "policy", None)]
        seen: set[int] = set()
        for policy in policies:
            if policy is None or id(policy) in seen or not hasattr(policy, "_rng"):
                continue
            seen.add(id(policy))
            policy._rng = jax.random.key(seed)


def encode_obs(observation, action_type, robot_action_dim_info):
    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        images = {
            "cam_high": ensure_chw_uint8(observation["images"]["cam_high"]),
            "cam_left_wrist": ensure_chw_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_chw_uint8(observation["images"]["cam_right_wrist"]),
        }
        prompt = observation.get("instruction")
        return {"state": state, "images": images, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    images = {
        "cam_high": ensure_chw_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])),
        "cam_left_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("instruction")
    return {"state": state, "images": images, "prompt": prompt}


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            "cam_high": np.stack([obs["images"]["cam_high"] for obs in obs_list], axis=0),
            "cam_left_wrist": np.stack([obs["images"]["cam_left_wrist"] for obs in obs_list], axis=0),
            "cam_right_wrist": np.stack([obs["images"]["cam_right_wrist"] for obs in obs_list], axis=0),
        },
        "prompt": [obs["prompt"] for obs in obs_list],
    }


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    return {
        "state": obs["state"][batch_index],
        "images": {
            "cam_high": obs["images"]["cam_high"][batch_index],
            "cam_left_wrist": obs["images"]["cam_left_wrist"][batch_index],
            "cam_right_wrist": obs["images"]["cam_right_wrist"][batch_index],
        },
        "prompt": obs["prompt"][batch_index],
    }


def extract_image(observation, candidate_names):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_chw_uint8(image):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.transpose(image_hwc, (2, 0, 1))
