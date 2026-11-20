"""CoPouring's reset draw: the tub, its apples and the target crate.

Each apple's draw is added to the tub's. Kinematic bodies get a pose write
only, the others get pose and zero velocity.
"""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from cohub.tasks._base.mdp import randomize_scene_to_default, draw_xy


__all__ = ["randomize_scene_to_default", "randomize_pour_station", "bin_target"]


def _place(env, env_ids: torch.Tensor, name: str, offset: torch.Tensor, kinematic: bool):
    """Move one rigid object to its default pose plus a per-env xy offset."""
    asset: RigidObject = env.scene[name]
    pose = asset.data.default_root_pose.torch[env_ids].clone()
    pose[:, :2] += offset
    pose[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    if not kinematic:
        asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
        )
    return asset


def randomize_pour_station(
    env,
    env_ids: torch.Tensor,
    tote_cfg: SceneEntityCfg,
    apple_names: tuple,
    bin_cfg: SceneEntityCfg,
    tote_pose_range: dict,
    apple_pose_range: dict,
    bin_pose_range: dict,
    tote_stand_names: tuple = (),
    bin_stand_names: tuple = (),
) -> None:
    """Draw one pour station per environment: the crate, its fruit, and the target.

    Each apple's offset is added to the tub's, so the apples move with it.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    tote = draw_xy(env, env_ids, tote_pose_range)
    _place(env, env_ids, tote_cfg.name, tote, kinematic=False)
    # The stack moves with the crate standing on it.
    for name in tote_stand_names:
        _place(env, env_ids, name, tote, kinematic=True)
    for name in apple_names:
        _place(env, env_ids, name, tote + draw_xy(env, env_ids, apple_pose_range), kinematic=False)
    catch = draw_xy(env, env_ids, bin_pose_range)
    _place(env, env_ids, bin_cfg.name, catch, kinematic=True)
    for name in bin_stand_names:
        _place(env, env_ids, name, catch, kinematic=True)


def bin_target(env, bin_cfg: SceneEntityCfg = SceneEntityCfg("catch_bin")) -> torch.Tensor:
    """This episode's target per env, env-local ``(num_envs, 3)``.

    The target crate's base centre, read from the scene.
    """
    catch_bin: RigidObject = env.scene[bin_cfg.name]
    return (catch_bin.data.root_pos_w.torch - env.scene.env_origins).clone()
