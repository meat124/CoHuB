"""TrashCollection's terminations: litter in the bin, litter on the floor, bin tipped.

The bin's mouth is the cylinder from the bin's origin (its base centre) up to
``bin_height``, ``mouth_radius`` wide, measured against the bin's live pose.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from cohub.subtasks import publish_checks
from cohub.g1.grasp import grasp_term


def _litter_in_bin(
    env,
    bin_pos: torch.Tensor,
    litter_names: tuple[str, ...],
    mouth_radius: float,
    bin_height: float,
) -> dict[str, torch.Tensor]:
    """Which pieces are inside the bin's mouth cylinder right now, by name.

    Measured in world axes, so it is exact only while the bin is upright.
    """
    in_bin: dict[str, torch.Tensor] = {}
    for name in litter_names:
        offset = env.scene[name].data.root_pos_w[:, :3] - bin_pos
        radial = torch.linalg.norm(offset[:, :2], dim=-1)
        in_bin[name] = (
            (radial < mouth_radius) & (offset[:, 2] > 0.0) & (offset[:, 2] < bin_height)
        )
    return in_bin


class TrashCollectionSuccess(ManagerTermBase):
    """Every piece of litter inside the bin -- the verdict -- plus the route.

    Published for the eval's subtask chain. "Held" is Robot A holding the bin
    with at least one hand. "Aloft" is the bin's base ``off_floor_m`` above
    the floor.

    * ``lifted``: a step with the bin held and aloft.
    * ``delivered``: after that, held and aloft, the centre within
      ``deliver_radius`` of ``deliver_at`` and the rim no more than
      ``rim_tolerance`` above ``bench_top_z``.
    * ``bin_held_aloft``: held and aloft at this step.
    * ``bin_at_lip``: the delivery geometry at this step.
    """

    checks_key = "trashcollection_success"

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        self._bin = env.scene[object_cfg.name]
        self._grasp = grasp_term(env)
        self._holder = self._grasp.hand_slice(cfg.params.get("holder", "robot_a"))
        self._lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._delivered = torch.zeros_like(self._lifted)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._lifted[env_ids] = False
        self._delivered[env_ids] = False

    def __call__(
        self,
        env,
        object_cfg: SceneEntityCfg,
        litter_names: tuple[str, ...],
        mouth_radius: float,
        bin_height: float,
        deliver_at: tuple[float, float],
        deliver_radius: float,
        bench_top_z: float,
        rim_tolerance: float,
        off_floor_m: float = 0.02,
        holder: str = "robot_a",
        require_bin_held: bool = True,
    ) -> torch.Tensor:
        """Every piece of litter is inside the bin.

        ``litter_names`` are scene entity names. ``mouth_radius`` and ``bin_height``
        describe the bin's mouth [m]; ``deliver_at``, ``deliver_radius``, ``bench_top_z``
        and ``rim_tolerance`` the delivery geometry. ``require_bin_held`` off lets a
        scripted run move the bin without hands.
        """
        bin_pos = self._bin.data.root_pos_w[:, :3]
        in_bin = _litter_in_bin(env, bin_pos, litter_names, mouth_radius, bin_height)

        local = bin_pos - env.scene.env_origins
        target = torch.tensor(deliver_at, device=local.device, dtype=local.dtype)
        at_lip = torch.linalg.norm(local[:, :2] - target, dim=-1) <= deliver_radius
        rim_low = local[:, 2] + bin_height <= bench_top_z + rim_tolerance

        held = self._grasp.latched[:, self._holder].any(dim=1)
        if not require_bin_held:
            held = torch.ones_like(held)
        held_aloft = held & (local[:, 2] >= off_floor_m)
        self._lifted |= held_aloft
        self._delivered |= self._lifted & at_lip & rim_low & held_aloft

        # One check per piece.
        publish_checks(
            env,
            self.checks_key,
            **in_bin,
            bin_at_lip=at_lip & rim_low,
            # Cloned: the checks store keeps references and reset() clears in place.
            lifted=self._lifted.clone(),
            delivered=self._delivered.clone(),
            bin_held_aloft=held_aloft,
        )

        done = torch.ones_like(next(iter(in_bin.values())))
        for flag in in_bin.values():
            done &= flag
        return done


def litter_dropped(
    env,
    object_cfg: SceneEntityCfg,
    litter_names: tuple[str, ...],
    mouth_radius: float,
    bin_height: float,
    floor_z: float,
) -> torch.Tensor:
    """A piece of litter is below ``floor_z`` and not inside the bin.

    Publishes one ``<name>_off_floor`` check per piece.
    """
    bin_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3]
    in_bin = _litter_in_bin(env, bin_pos, litter_names, mouth_radius, bin_height)
    dropped: dict[str, torch.Tensor] = {}
    for name in litter_names:
        z = env.scene[name].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        dropped[name] = (z < floor_z) & ~in_bin[name]
    publish_checks(
        env, "litter_dropped", **{f"{name}_off_floor": ~flag for name, flag in dropped.items()}
    )
    done = torch.zeros_like(next(iter(dropped.values())))
    for flag in dropped.values():
        done |= flag
    return done


def bin_tipped(env, object_cfg: SceneEntityCfg, max_tilt_deg: float) -> torch.Tensor:
    """The bin's own up-axis has leaned more than ``max_tilt_deg`` from vertical.

    Reads ``root_quat_w`` as (x, y, z, w).
    """
    quat = env.scene[object_cfg.name].data.root_quat_w
    up_z = 1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
    tilted = up_z < math.cos(math.radians(max_tilt_deg))
    publish_checks(env, "bin_tipped", upright=~tilted)
    return tilted
