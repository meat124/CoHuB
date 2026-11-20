"""Reset-time randomisation for pouring.

The cup and the mug are each moved on the worktop and turned about Z. The
bead is drawn inside the cup, on top of the cup's own draw.
"""

from __future__ import annotations

import math

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul

from cohub.tasks._base.mdp import randomize_scene_to_default, draw_xy


__all__ = ["randomize_scene_to_default", "randomize_pour_station", "mug_frame"]


def _draw_yaw(env, env_ids: torch.Tensor, yaw_range: tuple[float, float]) -> torch.Tensor:
    """One yaw quaternion per environment, ``(x, y, z, w)``."""
    angle = torch.empty(len(env_ids), device=env.device).uniform_(
        math.radians(yaw_range[0]), math.radians(yaw_range[1])
    )
    half = angle / 2.0
    quat = torch.zeros((len(env_ids), 4), device=env.device)
    quat[:, 2] = torch.sin(half)
    quat[:, 3] = torch.cos(half)
    return quat


def _place(env, env_ids: torch.Tensor, name: str, offset: torch.Tensor,
           yaw: torch.Tensor | None = None) -> None:
    """Move one rigid object to its default pose, plus an xy offset and a yaw.

    The velocity write is skipped for a kinematic body.
    """
    asset: RigidObject = env.scene[name]
    pose = asset.data.default_root_pose.torch[env_ids].clone()
    pose[:, :2] += offset
    pose[:, :3] += env.scene.env_origins[env_ids]
    if yaw is not None:
        # the drawn yaw turns the object about its own axis
        pose[:, 3:] = quat_mul(yaw, pose[:, 3:])
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    if not getattr(asset.cfg.spawn.rigid_props, "kinematic_enabled", False):
        asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
        )


def randomize_pour_station(
    env,
    env_ids: torch.Tensor,
    cup_cfg: SceneEntityCfg,
    mug_cfg: SceneEntityCfg,
    bead_cfg: SceneEntityCfg,
    cup_pose_range: dict,
    bead_pose_range: dict,
    yaw_range: tuple[float, float],
) -> None:
    """Draw one pour station per environment: the cup, the mug, the bead in it.

    The bead's offset is added to the cup's.
    """
    # ``env_ids`` must be a tensor of indices, not ``None``
    if env_ids is None or len(env_ids) == 0:
        return

    cup = draw_xy(env, env_ids, cup_pose_range)
    _place(env, env_ids, cup_cfg.name, cup, _draw_yaw(env, env_ids, yaw_range))
    _place(env, env_ids, mug_cfg.name, draw_xy(env, env_ids, cup_pose_range),
           _draw_yaw(env, env_ids, yaw_range))
    # on top of the cup's draw, not rotated with the cup
    _place(env, env_ids, bead_cfg.name, cup + draw_xy(env, env_ids, bead_pose_range))


def mug_frame(env, mug_cfg: SceneEntityCfg = SceneEntityCfg("catch_mug")):
    """This episode's target: the receiving mug's live pose, env-local.

    Returns ``(position, quaternion)``, both ``(num_envs, ...)``.
    """
    mug: RigidObject = env.scene[mug_cfg.name]
    position = (mug.data.root_pos_w.torch - env.scene.env_origins).clone()
    return position, mug.data.root_quat_w.torch.clone()
