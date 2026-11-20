"""Reset events for FrameHang: draw the hang station's position each episode.

The wall moves in y. The peg moves along the wall and down. The picture moves
with the wall and gets its own floor jitter. The wall and the peg are
kinematic, so they take pose writes only.
"""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

# Kinematic-safe scene reset: kinematic bodies reject velocity writes.
from cohub.tasks._base.mdp import randomize_scene_to_default

__all__ = ["randomize_scene_to_default", "randomize_station"]


def _uniform(env, env_ids: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return torch.empty(len(env_ids), device=env.device).uniform_(low, high)


def _place(env, env_ids: torch.Tensor, name: str, offset: torch.Tensor) -> None:
    """Move one rigid object to its default pose plus a per-env xyz offset."""
    asset: RigidObject = env.scene[name]
    pose = asset.data.default_root_state.torch[env_ids, :7].clone()
    pose[:, :3] += offset + env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    return asset


def randomize_station(
    env,
    env_ids: torch.Tensor,
    wall_parts: tuple[str, ...],
    hook_cfg: SceneEntityCfg,
    frame_cfg: SceneEntityCfg,
    x_jitter: float,
    z_range: tuple[float, float],
    wall_y_jitter: float,
    frame_x_jitter: float,
    frame_y_jitter: float,
) -> None:
    """Draw one hang station -- wall, peg and picture -- per environment.

    Everything in ``wall_parts`` moves with the wall in y. The peg also moves
    along the wall (``x_jitter``) and in height (``z_range``). The picture gets
    its own floor jitter on top of the station's offset.
    """
    # ``None`` is a no-op here. It does not mean every env.
    if env_ids is None or len(env_ids) == 0:
        return

    station = torch.zeros((len(env_ids), 3), device=env.device)
    station[:, 0] = _uniform(env, env_ids, -x_jitter, x_jitter)
    station[:, 1] = _uniform(env, env_ids, -wall_y_jitter, wall_y_jitter)

    wall = torch.zeros_like(station)
    wall[:, 1] = station[:, 1]
    for name in wall_parts:
        _place(env, env_ids, name, wall)

    hook = station.clone()
    hook[:, 2] = _uniform(env, env_ids, *z_range)
    _place(env, env_ids, hook_cfg.name, hook)

    picture = station.clone()
    picture[:, 0] += _uniform(env, env_ids, -frame_x_jitter, frame_x_jitter)
    picture[:, 1] += _uniform(env, env_ids, -frame_y_jitter, frame_y_jitter)
    frame = _place(env, env_ids, frame_cfg.name, picture)
    # The picture is the only dynamic body here, so its velocity is cleared.
    frame.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
    )
