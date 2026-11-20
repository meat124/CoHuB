"""Terms the tasks share: what 'placed' means, and the scene reset."""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul
import torch

from cohub.runtime import is_kinematic


# --- Placement ---

MAX_LINEAR_SPEED = 0.05
"""Linear speed below which an object is at rest [m/s]."""
MAX_ANGULAR_SPEED = 0.15
"""Angular speed below which an object is at rest [rad/s]."""
RELEASE_SETTLE_S = 0.20
"""Seconds an object must be out of every hand before its placement counts."""


def up_axis(quat: torch.Tensor) -> torch.Tensor:
    """Each body's own +Z in world axes, from its ``(N, 4)`` orientation.

    ``up_axis(q)[:, 2]`` is the cosine of the tilt from upright.
    """
    up = torch.tensor((0.0, 0.0, 1.0), device=quat.device, dtype=quat.dtype)
    return quat_apply(quat, up.expand(quat.shape[0], 3))


def tilt_from_upright(quat: torch.Tensor) -> torch.Tensor:
    """Angle between a body's own +Z and the world's [rad]."""
    return torch.acos(up_axis(quat)[:, 2].clamp(-1.0, 1.0))


# --- Reset events ---

def draw_xy(env, env_ids: torch.Tensor, pose_range: dict) -> torch.Tensor:
    """One (dx, dy) per environment, uniform in ``pose_range``.

    ``pose_range`` is ``{"x": (lo, hi), "y": (lo, hi)}``. A missing axis is zero.
    """
    offset = torch.zeros((len(env_ids), 2), device=env.device)
    for axis, key in enumerate(("x", "y")):
        low, high = pose_range.get(key, (0.0, 0.0))
        offset[:, axis].uniform_(low, high)
    return offset


def randomize_scene_to_default(env, env_ids: torch.Tensor) -> None:
    """Isaac Lab's own term, minus the velocity write on kinematic bodies.

    PhysX rejects a velocity write on a kinematic body. Everything else
    matches ``isaaclab.envs.mdp.reset_scene_to_default``.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    for rigid_object in env.scene.rigid_objects.values():
        pose = rigid_object.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += env.scene.env_origins[env_ids]
        rigid_object.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        if is_kinematic(rigid_object):
            continue
        vel = rigid_object.data.default_root_vel.torch[env_ids].clone()
        rigid_object.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)

    for articulation in env.scene.articulations.values():
        pose = articulation.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += env.scene.env_origins[env_ids]
        vel = articulation.data.default_root_vel.torch[env_ids].clone()
        articulation.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        articulation.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)
        articulation.write_joint_position_to_sim_index(
            position=articulation.data.default_joint_pos.torch[env_ids].clone(), env_ids=env_ids
        )
        articulation.write_joint_velocity_to_sim_index(
            velocity=articulation.data.default_joint_vel.torch[env_ids].clone(), env_ids=env_ids
        )


REGION_AXES = ("x", "y", "z", "yaw")
"""The axes of a region: ``x``/``y``/``z`` [m] and ``yaw`` [deg]."""

_MAX_REDRAWS = 200


def draw_region(env, count: int, region: dict) -> dict[str, torch.Tensor]:
    """One offset per axis, ``{axis: (count,)}``, uniform over a box with holes.

    ``region`` is ``{"outer": {axis: (lo, hi)}, "exclude": [{axis: (lo, hi)}, ...]}``.
    A draw inside an ``exclude`` box is drawn again. An axis missing from
    ``outer`` is zero; one missing from an ``exclude`` box is unbounded.
    """
    outer = region.get("outer", {})
    unknown = set(outer) - set(REGION_AXES)
    if unknown:
        raise KeyError(f"region names axes {sorted(unknown)}; expected a subset of {REGION_AXES}")
    excludes = region.get("exclude", ())

    def sample(n: int) -> dict[str, torch.Tensor]:
        out = {}
        for axis in REGION_AXES:
            low, high = outer.get(axis, (0.0, 0.0))
            out[axis] = torch.empty(n, device=env.device).uniform_(low, high)
        return out

    def excluded(values: dict[str, torch.Tensor]) -> torch.Tensor:
        bad = torch.zeros(count, dtype=torch.bool, device=env.device)
        for box in excludes:
            inside = torch.ones(count, dtype=torch.bool, device=env.device)
            for axis, (low, high) in box.items():
                inside &= (values[axis] > low) & (values[axis] < high)
            bad |= inside
        return bad

    values = sample(count)
    for _ in range(_MAX_REDRAWS):
        bad = excluded(values)
        if not bool(bad.any()):
            return values
        again = sample(int(bad.sum()))
        for axis in REGION_AXES:
            values[axis][bad] = again[axis]
    raise RuntimeError(f"region leaves nothing to draw from: {region}")


def _place(env, env_ids: torch.Tensor, name: str, offset: dict[str, torch.Tensor], about: torch.Tensor) -> None:
    """Move a rigid object to its default pose plus ``offset``, its yaw turning it about ``about``."""
    asset = env.scene[name]
    pose = asset.data.default_root_pose.torch[env_ids].clone()
    yaw = torch.deg2rad(offset["yaw"])
    rel = pose[:, :2] - about
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    pose[:, 0] = about[:, 0] + cos * rel[:, 0] - sin * rel[:, 1] + offset["x"]
    pose[:, 1] = about[:, 1] + sin * rel[:, 0] + cos * rel[:, 1] + offset["y"]
    pose[:, 2] += offset["z"]
    # (x, y, z, w)
    turn = torch.zeros((len(env_ids), 4), device=pose.device, dtype=pose.dtype)
    turn[:, 2] = torch.sin(yaw / 2.0)
    turn[:, 3] = torch.cos(yaw / 2.0)
    pose[:, 3:] = quat_mul(turn, pose[:, 3:])
    pose[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    if not is_kinematic(asset):
        asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
        )


def reset_region(
    env,
    env_ids: torch.Tensor,
    region: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    riders: tuple[str, ...] = (),
) -> None:
    """Offset ``asset_cfg`` from its default pose by a :func:`draw_region` draw.

    ``riders`` stand on the drawn object and take the same draw.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    offset = draw_region(env, len(env_ids), region)
    about = env.scene[asset_cfg.name].data.default_root_pose.torch[env_ids, :2].clone()
    for name in (asset_cfg.name, *riders):
        _place(env, env_ids, name, offset, about)
