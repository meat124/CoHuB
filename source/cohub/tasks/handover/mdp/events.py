"""Reset-time randomisation of the handover target.

The target pad's own pose is the target: ``randomize_target`` moves the pad and
``target`` reads the goal back from it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def target(
    env: ManagerBasedEnv,
    centre: tuple[float, float, float],
    marker_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
) -> torch.Tensor:
    """This episode's target per env, env-local ``(num_envs, 3)``.

    Args:
        centre: The nominal target. Only its Z is used: the height of the
            bottle's centre when it stands on the pad.
        marker_cfg: The pad whose pose is the target.
    """
    marker: RigidObject = env.scene[marker_cfg.name]
    goal = (marker.data.root_pos_w.torch - env.scene.env_origins).clone()
    goal[:, 2] = centre[2]
    return goal


def randomize_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    centre: tuple[float, float, float],
    marker_z: float,
    marker_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
) -> None:
    """Draw this episode's target by moving the pad that shows it.

    ``pose_range`` holds ``x``/``y`` offsets from ``centre`` (env-local);
    ``marker_z`` is the pad's own centre height.
    """
    marker: RigidObject = env.scene[marker_cfg.name]
    device = marker.device

    ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y")], device=device, dtype=torch.float32
    )
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 2), device=device)

    base = torch.tensor(centre[:2], device=device, dtype=torch.float32)
    pose = marker.data.default_root_pose.torch[env_ids].clone()
    pose[:, 0:2] = base + samples + env.scene.env_origins[env_ids][:, 0:2]
    pose[:, 2] = marker_z + env.scene.env_origins[env_ids][:, 2]
    marker.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
