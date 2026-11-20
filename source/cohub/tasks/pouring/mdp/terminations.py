"""Pouring's success predicate: the marble went into the held glass, by being tipped.

The scene entity ``bead`` is the marble. Success needs the bead inside the mug
for ``dwell_s``, the mug off the worktop, and two sticky milestones in order:

1. ``lift_both``: cup and mug both off the worktop at the same instant.
2. ``tip``: after that, the cup tipped past ``pour_tilt_deg`` while Robot A
   holds it off the worktop.

"Off the worktop" is measured at the cup's lowest rim point. The bead is
scored in the mug's own frame. Robot B's contact with the mug (``mug_in_hand``)
is published, not gated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply_inverse

from cohub.tasks._base.mdp import tilt_from_upright

from cohub.subtasks import publish_checks, publish_metrics
from cohub.g1.grasp import grasp_term

from .events import mug_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

MUG_CONTACT_SENSOR = "mug_contact"
"""Scene sensor watching Robot B's hands against the receiving mug."""

MUG_CONTACT_FORCE = 0.5
MUG_CONTACT_LINKS = 2
"""Force on one finger link that counts as touching [N], and how many links make a hold."""


def bead_in_mug(
    env: ManagerBasedRLEnv,
    bead_cfg: SceneEntityCfg,
    mug_cfg: SceneEntityCfg,
    inner_radius: float,
    inner_top: float,
    floor_thickness: float,
) -> tuple[torch.Tensor, dict, dict]:
    """Is the bead inside the mug's own walls? With the numbers that say so.

    Asked in the mug's frame, against the cylinder its walls enclose.
    Returns ``(inside, checks, metrics)``.
    """
    bead = env.scene[bead_cfg.name]
    position = bead.data.root_pos_w.torch - env.scene.env_origins
    goal, orientation = mug_frame(env, mug_cfg)
    local = quat_apply_inverse(orientation, position - goal.to(position))

    radial = torch.linalg.norm(local[:, :2], dim=-1)
    inside_radius = radial <= inner_radius
    below_rim = local[:, 2] <= inner_top
    above_floor = local[:, 2] >= floor_thickness

    inside = inside_radius & below_rim & above_floor
    return (
        inside,
        {
            "bead_over_mug": inside_radius,
            "bead_below_rim": below_rim,
            "bead_above_floor": above_floor,
        },
        {
            "bead_radial_m": radial,
            "bead_depth_m": local[:, 2],
        },
    )


def bead_in_cup(
    env: ManagerBasedRLEnv,
    bead_cfg: SceneEntityCfg,
    cup_cfg: SceneEntityCfg,
    inner_radius: float,
    rim_z: float,
) -> torch.Tensor:
    """Is the marble still in the cup it started in? Asked in the cup's frame."""
    bead = env.scene[bead_cfg.name]
    cup = env.scene[cup_cfg.name]
    delta = bead.data.root_pos_w.torch - cup.data.root_pos_w.torch
    local = quat_apply_inverse(cup.data.root_quat_w.torch, delta)
    radial = torch.linalg.norm(local[:, :2], dim=-1)
    return (radial <= inner_radius) & (local[:, 2] >= -0.01) & (local[:, 2] <= rim_z + 0.02)


class BeadLost(ManagerTermBase):
    """The marble is out of both vessels and is not going back into either.

    The term fires when the marble is in neither vessel and either its centre
    is within ``lost_height`` of the worktop or below it, or it has been
    slower than ``max_linear_speed`` for ``settle_hold_s``.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        bead_cfg: SceneEntityCfg = cfg.params.get("bead_cfg", SceneEntityCfg("bead"))
        self._bead = env.scene[bead_cfg.name]
        self._slow_for = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._slow_for[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        bead_cfg: SceneEntityCfg = SceneEntityCfg("bead"),
        mug_cfg: SceneEntityCfg = SceneEntityCfg("catch_mug"),
        cup_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        inner_radius: float = 0.030,
        inner_top: float = 0.117,
        floor_thickness: float = 0.007,
        cup_inner_radius: float = 0.0378,
        cup_rim_z: float = 0.089,
        worktop_z: float = 0.72,
        lost_height: float = 0.025,
        max_linear_speed: float = 0.05,
        settle_hold_s: float = 0.25,
    ) -> torch.Tensor:
        in_mug, _, _ = bead_in_mug(
            env, bead_cfg, mug_cfg, inner_radius, inner_top, floor_thickness
        )
        in_cup = bead_in_cup(env, bead_cfg, cup_cfg, cup_inner_radius, cup_rim_z)
        loose = (~in_mug) & (~in_cup)

        height = self._bead.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        on_table = height <= worktop_z + lost_height

        slow = torch.linalg.norm(self._bead.data.root_lin_vel_w.torch, dim=-1) <= max_linear_speed
        # Seconds, not steps. Updated in place so the buffer stays writable outside inference mode.
        self._slow_for += float(env.step_dt)
        self._slow_for[~(loose & slow)] = 0.0
        settled_out = self._slow_for >= settle_hold_s

        return loose & (on_table | settled_out)


class PourSuccess(ManagerTermBase):
    """The bead stayed in the lifted mug, poured from the cup Robot A tipped.

    Success at a step = bead inside the mug for ``dwell_s`` without leaving,
    mug off the worktop, and both milestones latched:

    1. ``lift_both``: cup and mug off the worktop by ``lift_margin`` at the
       same instant.
    2. ``tip``: after ``lift_both``, the cup past ``pour_tilt_deg`` while A
       holds it off the worktop.

    Milestones are sticky and cleared by :meth:`reset`. They are published as
    ``lift_both_ok`` and ``tip_ok``.
    """

    checks_key = "pouring_success"

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        cup_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        mug_cfg: SceneEntityCfg = cfg.params.get("mug_cfg", SceneEntityCfg("catch_mug"))
        bead_cfg: SceneEntityCfg = cfg.params.get("bead_cfg", SceneEntityCfg("bead"))
        self._cup = env.scene[cup_cfg.name]
        self._mug = env.scene[mug_cfg.name]
        self._bead = env.scene[bead_cfg.name]
        self._grasp = grasp_term(env)
        self._cup_hands = self._grasp.hand_slice(self._grasp.cfg.robot_names[0])
        self._mug_sensor = env.scene.sensors.get(MUG_CONTACT_SENSOR)
        self._lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._mug_lifted = torch.zeros_like(self._lifted)
        self._both_lifted = torch.zeros_like(self._lifted)
        self._tipped = torch.zeros_like(self._lifted)
        self._mug_held = torch.zeros_like(self._lifted)
        self._landed = torch.zeros_like(self._lifted)
        self._inside_for = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._lifted[env_ids] = False
        self._mug_lifted[env_ids] = False
        self._both_lifted[env_ids] = False
        self._tipped[env_ids] = False
        self._mug_held[env_ids] = False
        self._landed[env_ids] = False
        self._inside_for[env_ids] = 0.0

    def _mug_in_hand(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Robot B's hand on the mug this step. Diagnostic.

        The force matrix is ``(num_envs, 1, links, 3)``.
        """
        if self._mug_sensor is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        matrix = self._mug_sensor.data.force_matrix_w
        if matrix is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        forces = torch.linalg.norm(matrix.reshape(env.num_envs, -1, 3), dim=-1)
        return (forces > MUG_CONTACT_FORCE).sum(dim=1) >= MUG_CONTACT_LINKS

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        mug_cfg: SceneEntityCfg = SceneEntityCfg("catch_mug"),
        bead_cfg: SceneEntityCfg = SceneEntityCfg("bead"),
        inner_radius: float = 0.055,
        inner_top: float = 0.051,
        floor_thickness: float = 0.003,
        worktop_z: float = 0.83,
        lift_margin: float = 0.03,
        pour_tilt_deg: float = 35.0,
        dwell_s: float = 0.5,
        cup_outer_radius: float = 0.0425,
        cup_height: float = 0.089,
        require_cup_in_hand: bool = True,
        debug: bool = False,
    ) -> torch.Tensor:
        origin_z = env.scene.env_origins[:, 2]
        # heights above the worktop, measured at the base centres
        cup_rise = self._cup.data.root_pos_w.torch[:, 2] - origin_z - worktop_z
        mug_rise = self._mug.data.root_pos_w.torch[:, 2] - origin_z - worktop_z

        # the cup's lowest rim point above the worktop
        tilt = tilt_from_upright(self._cup.data.root_quat_w.torch)
        cup_clearance = (
            cup_rise
            + torch.clamp(cup_height * torch.cos(tilt), max=0.0)
            - cup_outer_radius * torch.sin(tilt)
        )
        lifted_now = cup_clearance >= lift_margin
        self._lifted |= lifted_now

        mug_off_worktop = mug_rise >= lift_margin
        self._mug_lifted |= mug_off_worktop

        # lift_both: both vessels in the air at the same instant
        both_now = lifted_now & mug_off_worktop
        newly_lifted = ~self._both_lifted & both_now
        self._both_lifted |= both_now

        cup_in_hand = self._grasp.latched[:, self._cup_hands].any(dim=1)
        mug_in_hand = self._mug_in_hand(env)
        self._mug_held |= mug_in_hand
        cup_held = cup_in_hand if require_cup_in_hand else torch.ones_like(cup_in_hand)

        # tip: past the angle while A holds the cup off the worktop
        cup_held_aloft = cup_held & lifted_now
        tipped_now = self._both_lifted & cup_held_aloft & (tilt >= math.radians(pour_tilt_deg))
        newly_tipped = ~self._tipped & tipped_now
        self._tipped |= tipped_now

        inside, geometry_checks, geometry_metrics = bead_in_mug(
            env, bead_cfg, mug_cfg, inner_radius, inner_top, floor_thickness
        )

        # seconds the bead has been inside without leaving, updated in place
        self._inside_for += float(env.step_dt)
        self._inside_for[~inside] = 0.0
        stayed = self._inside_for >= dwell_s

        # diagnostic only, relative to the mug
        relative_speed = torch.linalg.norm(
            self._bead.data.root_lin_vel_w.torch - self._mug.data.root_lin_vel_w.torch, dim=-1
        )

        success = stayed & mug_off_worktop & self._tipped

        publish_checks(
            env,
            self.checks_key,
            **geometry_checks,
            bead_in_mug=inside,
            bead_stayed=stayed,
            mug_off_worktop=mug_off_worktop,
            cup_lifted=self._lifted,
            mug_lifted=self._mug_lifted,
            cup_tipped=self._tipped,
            cup_in_hand=cup_in_hand,
            cup_held_aloft=cup_held_aloft,
            mug_in_hand=mug_in_hand,
            mug_ever_held=self._mug_held,
            # read by the subtask tracker
            lift_both_ok=self._both_lifted,
            tip_ok=self._tipped,
        )
        publish_metrics(
            env,
            self.checks_key,
            **geometry_metrics,
            cup_rise_m=cup_rise,
            cup_clearance_m=cup_clearance,
            mug_rise_m=mug_rise,
            cup_tilt_deg=torch.rad2deg(tilt),
            bead_in_mug_s=self._inside_for,
            bead_speed_mps=relative_speed,
        )

        for index in newly_lifted.nonzero(as_tuple=False).flatten().tolist():
            print(
                f"[pouring] env {index}: lift_both achieved "
                f"(cup {float(cup_rise[index]):.3f} m, mug {float(mug_rise[index]):.3f} m "
                f"above the worktop)"
            )
        for index in newly_tipped.nonzero(as_tuple=False).flatten().tolist():
            angle = float(torch.rad2deg(tilt[index]))
            print(f"[pouring] env {index}: tip achieved ({angle:.0f} deg from upright)")

        # one line the first time the marble has stayed in the glass, saying whether it scores
        newly_landed = ~self._landed & stayed
        self._landed |= stayed
        for index in newly_landed.nonzero(as_tuple=False).flatten().tolist():
            if bool(success[index]):
                print(f"[pouring] env {index}: the marble is in the glass and the pour scores")
                continue
            missing = []
            if not bool(mug_off_worktop[index]):
                print_rise = float(mug_rise[index])
                missing.append(
                    f"the glass is not off the worktop (its base is {print_rise * 1000:+.0f} mm "
                    f"above it, and {lift_margin * 1000:.0f} mm is what counts as held)"
                )
            if not bool(self._both_lifted[index]):
                missing.append(
                    "the two vessels were never off the worktop at the same time "
                    f"(cup {float(cup_clearance[index]) * 1000:+.0f} mm, "
                    f"glass {float(mug_rise[index]) * 1000:+.0f} mm now)"
                )
            elif not bool(self._tipped[index]):
                missing.append(
                    f"the cup was never tipped past {pour_tilt_deg:.0f} deg while Robot A "
                    f"held it off the worktop (it has reached "
                    f"{float(torch.rad2deg(tilt[index])):.0f})"
                )
            print(
                f"[pouring] env {index}: the marble is in the glass but the pour "
                f"does NOT score -- " + "; ".join(missing)
            )

        if debug and bool(inside.any()) and not bool(success.any()):
            for index in inside.nonzero(as_tuple=False).flatten().tolist():
                print(
                    f"[pouring] env {index}: in the mug but not scored -- "
                    f"stayed={bool(stayed[index])} "
                    f"(inside for {float(self._inside_for[index]):.2f}/{dwell_s} s, "
                    f"rel {float(relative_speed[index]):.3f} m/s) "
                    f"mug_off_worktop={bool(mug_off_worktop[index])} "
                    f"(rise {float(mug_rise[index]):.3f}) "
                    f"lift_both={bool(self._both_lifted[index])} "
                    f"(cup rise {float(cup_rise[index]):.3f}) "
                    f"tipped={bool(self._tipped[index])} "
                    f"tilt={float(torch.rad2deg(tilt[index])):.0f} deg"
                )
        return success
