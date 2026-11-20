"""Ψ₀ adapter for the CoHuB policy server.

Supports `cohub_mode=decentralized` only: the policy answers once per agent,
from that agent's ego camera, 43 joint angles and instruction.
`psi0_layout.py` maps CoHuB's action and state order to Ψ₀'s.
Serving follows `psi0/src/psi/deploy/psi0_serve_simple.py`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"
_PSI_ROOT = _POLICY_DIR / "psi0"

if str(_PSI_ROOT) not in sys.path:
    sys.path.insert(0, str(_PSI_ROOT))
if str(_PSI_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PSI_ROOT / "src"))

from baselines.utils.model_template import ModelTemplate  # noqa: E402
from baselines.utils.checkpoint_resolver import (  # noqa: E402
    build_run_dir_name,
    candidate_checkpoint_roots,
)

COHUB_BENCH_NAME = "cohub"
COHUB_CENTRALIZED_ENV_CFG_TYPE = "unitree_g1x2_centralized"
COHUB_DECENTRALIZED_ENV_CFG_TYPE = "unitree_g1x2_decentralized"
COHUB_ROBOTS = ("robot_a", "robot_b")
COHUB_ROBOTS3 = ("robot_a", "robot_b", "robot_c")
"""Every robot name a task may have. The observation says which are present."""


def _cohub_robots(obs: dict) -> tuple[str, ...]:
    present = obs.get("cohub_state") or {}
    return tuple(robot for robot in COHUB_ROBOTS3 if robot in present) or COHUB_ROBOTS
COHUB_MULTITASK_CKPT = "multitask"
COHUB_MULTITASK_CKPTS = ("multitask", "multitask3")
"""Shared runs: the two-robot task set and the three-robot task set."""

COHUB_VIDEO_SLOT = {"ego_a": "cam_left_wrist", "ego_b": "cam_right_wrist",
                      "ego_c": "cam_third_view"}
"""Vision slot that carries each robot's head camera."""

COHUB_EGO_VIEW = {"robot_a": "ego_a", "robot_b": "ego_b", "robot_c": "ego_c"}

COHUB_JOINT_TARGET_DIM = 31
COHUB_ACTION_DIM = 35


def _cohub_layout():
    """Import `psi0_layout.py` from this directory."""
    if str(_POLICY_DIR) not in sys.path:
        sys.path.insert(0, str(_POLICY_DIR))
    import psi0_layout

    return psi0_layout


layout = _cohub_layout()


def _is_cohub(model_cfg: dict[str, Any]) -> bool:
    return str(model_cfg.get("bench_name") or "") == COHUB_BENCH_NAME


def _cohub_mode(model_cfg: dict[str, Any]) -> str:
    mode = str(model_cfg.get("cohub_mode") or "decentralized").strip().lower()
    if mode == "centralized":
        raise ValueError(
            "Psi0 has no centralized form: it reads one camera and emits one "
            "robot's 36 dimensions. Evaluate it with cohub_mode=decentralized, "
            "which is the benchmark's default."
        )
    if mode != "decentralized":
        raise ValueError(f"cohub_mode must be 'decentralized', got {mode!r}")
    return mode


def _decentralized_style(model_cfg: dict[str, Any]) -> str:
    style = str(model_cfg.get("cohub_decentralized_style") or "shared").strip().lower()
    if style not in ("shared", "per_robot"):
        raise ValueError(
            f"cohub_decentralized_style must be 'shared' or 'per_robot', got {style!r}")
    return style


def _run_root(model_cfg: dict[str, Any], robot: str | None) -> Path:
    """`checkpoints/<run-dir name>` for one target."""
    explicit = "model_dir" if robot is None else f"model_dir_{robot}"
    run_cfg = dict(model_cfg)
    # Run directories are named with the training env_cfg_type.
    run_cfg["env_cfg_type"] = COHUB_DECENTRALIZED_ENV_CFG_TYPE
    if robot is not None:
        run_cfg["ckpt_name"] = f"{model_cfg.get('ckpt_name')}_{robot}"
    candidates = candidate_checkpoint_roots(
        run_cfg, _CHECKPOINTS_DIR, policy_dir=_POLICY_DIR, explicit_keys=(explicit, "model_dir"),
    )
    if not candidates:
        raise ValueError("bench_name/ckpt_name/env_cfg_type/action_type/seed are required "
                         "to name the run dir")
    ckpt_tag = str(model_cfg.get("ckpt_tag") or "").strip()
    if ckpt_tag and ckpt_tag != "-":
        tagged = build_run_dir_name(run_cfg)
        if tagged:
            candidates.insert(0, _CHECKPOINTS_DIR / f"{tagged}-{ckpt_tag}")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"no Psi0 checkpoint for {robot or 'shared'}; looked at "
        + ", ".join(str(c) for c in candidates))


def _psi_run_dir(root: Path) -> Path:
    """The Ψ₀ run directory (holds `run_config.json`), at `root` or up to two levels below."""
    if (root / "run_config.json").is_file():
        return root
    found = sorted(p.parent for p in root.glob("*/*/run_config.json"))
    found += sorted(p.parent for p in root.glob("*/run_config.json"))
    if not found:
        raise FileNotFoundError(
            f"{root} holds no Psi0 run (no run_config.json in it or two levels below). "
            f"An interrupted first job can leave the directory empty.")
    if len(found) > 1:
        print(f"[Psi0] {len(found)} runs under {root}; taking {found[-1].name}")
    return found[-1]


def _ckpt_step(run_dir: Path, checkpoint_num: Any) -> str:
    """Step of the `checkpoints/ckpt_<step>` to serve. None/'last'/'latest' = newest."""
    steps: dict[int, str] = {}
    for path in (run_dir / "checkpoints").glob("ckpt_*"):
        suffix = path.name[len("ckpt_"):]
        if suffix.isdigit():
            steps[int(suffix)] = suffix
    if not steps:
        raise FileNotFoundError(f"{run_dir}/checkpoints holds no ckpt_<step> directory")
    wanted = str(checkpoint_num or "last").strip().lower()
    if wanted in ("", "last", "latest", "none"):
        return steps[max(steps)]
    digits = "".join(ch for ch in wanted if ch.isdigit())
    if digits and int(digits) in steps:
        return steps[int(digits)]
    raise FileNotFoundError(
        f"checkpoint_num={checkpoint_num!r} is not among {sorted(steps)} in {run_dir}")


def _load_launch_config(run_dir: Path):
    """Rebuild the training LaunchConfig from `argv.txt` and `run_config.json`."""
    from psi.utils import apply_legacy_model_config_defaults, parse_args_to_tyro_config

    for name in ("argv.txt", "run_config.json"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"{run_dir} has no {name}; Psi0 cannot rebuild its config")
    config = parse_args_to_tyro_config(run_dir / "argv.txt")
    return config.model_validate(
        apply_legacy_model_config_defaults(json.loads((run_dir / "run_config.json").read_text())))


def _pack_cohub_robot_action(flat: np.ndarray) -> dict[str, np.ndarray]:
    if flat.shape[-1] != COHUB_ACTION_DIM:
        raise ValueError(f"expected a {COHUB_ACTION_DIM}D per-robot action, got {flat.shape}")
    return {
        "joint_targets": flat[:COHUB_JOINT_TARGET_DIM],
        "height": flat[COHUB_JOINT_TARGET_DIM:COHUB_JOINT_TARGET_DIM + 1],
        "base_vel": flat[COHUB_JOINT_TARGET_DIM + 1:],
    }


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        if not _is_cohub(model_cfg):
            raise NotImplementedError(
                "policy/Psi0 is an CoHuB adapter: the only Psi0 checkpoints here are "
                "CoHuB ones and its action space is a Unitree G1's. Pass bench_name=cohub.")
        self.task_name = model_cfg.get("task_name")
        self.action_type = str(model_cfg.get("action_type") or "joint")
        self.robot_action_dim_info = None
        self.observation_window: list[dict[str, Any]] | None = None
        self._latest_env_idx_list: list[int] = [0]

        self._mode = _cohub_mode(model_cfg)
        self._style = _decentralized_style(model_cfg)
        self._prompt_override = {
            robot: model_cfg.get(f"prompt_{robot}") for robot in COHUB_ROBOTS3}

        task = str(model_cfg.get("ckpt_name") or "").strip()
        if not task:
            raise ValueError("cohub eval needs ckpt_name (multitask for the shared policy)")
        shared = self._style == "shared"
        if shared and task not in COHUB_MULTITASK_CKPTS:
            print(f"[Psi0] decentralized_style=shared with ckpt_name={task!r}; "
                  f"the benchmark's shared runs are {COHUB_MULTITASK_CKPTS}")

        import torch

        self._torch = torch
        gpu_id = model_cfg.get("gpu_id")
        self._device = torch.device(
            f"cuda:{gpu_id}" if gpu_id not in (None, "", "-") and torch.cuda.is_available()
            else ("cuda:0" if torch.cuda.is_available() else "cpu"))

        targets: tuple[str | None, ...] = (None,) if shared else COHUB_ROBOTS
        self._runs: dict[str, Any] = {}
        loaded: dict[str | None, Any] = {}
        for target in targets:
            loaded[target] = self._load(model_cfg, target)
        # Shared weights drive every robot; per_robot has one checkpoint for each of the pair.
        self._per_robot = {robot: loaded[None] for robot in COHUB_ROBOTS3} if shared else {
            robot: loaded[robot] for robot in COHUB_ROBOTS}

        first = next(iter(loaded.values()))
        self.model = first["model"]
        self.policy = self.model
        chunk = int(first["config"].model.action_chunk_size)
        horizon = int(model_cfg.get("exec_horizon") or first["config"].model.action_exec_horizon)
        self._exec_horizon = max(1, min(horizon, chunk))
        self._steps = int(model_cfg.get("num_inference_steps")
                          or getattr(first["config"].model, "eval_diffusion_steps", 10))
        self._last_height = {robot: layout.NOMINAL_HEIGHT for robot in COHUB_ROBOTS3}
        print(f"[Psi0][cohub] style={self._style} chunk={chunk} exec={self._exec_horizon} "
              f"steps={self._steps} device={self._device}")

    # Loading

    def _load(self, model_cfg: dict[str, Any], robot: str | None) -> dict[str, Any]:
        from torchvision.transforms import v2

        from psi.models.psi0 import Psi0Model

        root = _run_root(model_cfg, robot)
        run_dir = _psi_run_dir(root)
        step = _ckpt_step(run_dir, model_cfg.get("checkpoint_num"))
        config = _load_launch_config(run_dir)
        print(f"[Psi0][cohub] {robot or 'shared'}: ckpt_{step} <- {run_dir}")

        model = Psi0Model.from_pretrained(run_dir, step, config, device=str(self._device))
        model.to(self._device)
        model.eval()

        field = config.data.transform.field
        model_transform = config.data.transform.model
        if field.action_min is None:
            raise RuntimeError(
                f"{run_dir} rebuilt a config with no normalisation bounds -- its "
                f"{field.stat_path} was unreadable at load time. Actions would be "
                f"denormalised against nothing.")
        return {
            "model": model,
            "config": config,
            "field": field,
            "image": v2.Compose([model_transform.resize(), model_transform.center_crop()]),
            "step": step,
        }

    # Observation

    def _frame(self, obs: dict[str, Any], robot: str) -> np.ndarray:
        slot = COHUB_VIDEO_SLOT[COHUB_EGO_VIEW[robot]]
        try:
            color = obs["vision"][slot]["color"]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"observation carries no {slot} view for {robot}; the eval runner sends "
                f"ego_a and ego_b (and ego_c on a three-robot task) by default") from exc
        # Copy: the decoded array is a read-only view.
        return np.array(color, dtype=np.uint8, copy=True)

    def _prompt(self, obs: dict[str, Any], robot: str) -> str:
        """Instruction for this agent, lowercased as in training."""
        override = self._prompt_override.get(robot)
        if override:
            return str(override).lower()
        wire = (obs.get("cohub_instruction") or {}).get(robot)
        if wire:
            return str(wire).lower()
        instruction = obs.get("instruction")
        if instruction:
            return str(instruction).lower()
        raise KeyError(
            f"no instruction for {robot}: the shared policy is told the two agents apart "
            f"by language alone, so serving it without one is meaningless")

    def _states(self, obs: dict[str, Any], robot: str, entry: dict[str, Any]):
        from psi.utils import pad_to_len

        joints = np.asarray(obs["cohub_state"][robot]["joint_pos"], dtype=np.float32)
        states = layout.state_to_psi0(joints, self._last_height[robot]).astype(np.float32)
        states = states[None, :]  # (To, Ds), one observation step
        field = entry["field"]
        if field.normalize_state:
            states = field.normalize_state_func(pad_to_len(states, field.pad_state_dim, dim=1)[0])
        else:
            states = pad_to_len(states, field.pad_state_dim, dim=1)[0]
        return self._torch.from_numpy(np.ascontiguousarray(states)).to(self._device)

    # Inference

    def _chunk_for(self, obs: dict[str, Any], robot: str) -> np.ndarray:
        from PIL import Image

        entry = self._per_robot[robot]
        image = entry["image"](Image.fromarray(self._frame(obs, robot)))
        with self._torch.inference_mode():
            raw = entry["model"].predict_action(
                observations=[[image]],
                states=self._states(obs, robot, entry).unsqueeze(0),  # (B, To, Ds)
                instructions=[self._prompt(obs, robot)],
                num_inference_steps=self._steps,
                traj2ds=None,
            )
        raw = raw.reshape(-1, int(entry["config"].model.action_dim)).float().cpu().numpy()
        chunk = entry["field"].denormalize(raw)[: self._exec_horizon]
        cohub = layout.action_from_psi0(np.asarray(chunk, dtype=np.float32))
        # The state's height slot holds the last commanded base height.
        self._last_height[robot] = float(cohub[-1, COHUB_JOINT_TARGET_DIM])
        return cohub

    def _actions_for(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        robots = _cohub_robots(obs)
        per_robot = {robot: self._chunk_for(obs, robot) for robot in robots}
        steps = min(len(chunk) for chunk in per_robot.values())
        return [
            {"cohub_raw_action": {
                robot: _pack_cohub_robot_action(per_robot[robot][step])
                for robot in robots}}
            for step in range(steps)
        ]

    # ModelTemplate interface

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self.observation_window = list(obs_list)

    def get_action(self, **kwargs):
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")
        env_idx_list = env_idx_list or self._latest_env_idx_list
        return [self._actions_for(self.observation_window[index])
                for index in range(len(env_idx_list))]

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]
        self._last_height = {robot: layout.NOMINAL_HEIGHT for robot in COHUB_ROBOTS3}

    def reset_obsrvationwindows(self):
        self.reset()
