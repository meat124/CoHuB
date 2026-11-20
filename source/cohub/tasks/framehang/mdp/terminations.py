"""When the picture counts as hung: the wall peg threaded through its loop."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from cohub.tasks._base.mdp import up_axis

from cohub.subtasks import publish_checks, publish_metrics
from cohub.g1.grasp import grasp_term

from ..geometry import (
    FRAME_DROP_HEIGHT,
    FRAME_HALF_HEIGHT,
    FRAME_START_Z,
    HOOK_KNOB_RADIUS,
    HOOK_SHANK_LENGTH,
    LOOP_INNER_WIDTH,
    LOOP_RISE,
)


class HangSuccess(ManagerTermBase):
    """The peg through the picture's loop, the picture still and let go of.

    Threaded: the peg's axis, in the picture's frame, crosses the loop's
    mid-plane inside the opening (``|x| <= loop_half_width``, between
    ``rail_z`` and ``rail_z + loop_rise``) at ``0 <= s <= peg_length`` from the
    wall. The peg points -Y out of the wall in the hook's frame.

    Success: threaded, upright and facing the wall within ``max_tilt_deg``, at
    rest, every hand off for ``release_settle_time``, and both robots having
    held the picture at the same instant.

    Sticky route flags for ``FrameHangTracker``: ``held_together``,
    ``lifted_together`` (centre ``lift_margin_m`` above its spawn height) and
    ``hooked_together`` (threaded while both robots hold).
    """

    checks_key = "framehang_success"

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        self._frame = env.scene[cfg.params.get("object_cfg", SceneEntityCfg("object")).name]
        self._hook = env.scene[cfg.params.get("hook_cfg", SceneEntityCfg("wall_hook")).name]
        self._grasp = grasp_term(env)
        self._held_together = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._lifted_together = torch.zeros_like(self._held_together)
        self._hooked_together = torch.zeros_like(self._held_together)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._held_together[env_ids] = False
        self._lifted_together[env_ids] = False
        self._hooked_together[env_ids] = False

    def __call__(
        self,
        env,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        hook_cfg: SceneEntityCfg = SceneEntityCfg("wall_hook"),
        loop_half_width: float = LOOP_INNER_WIDTH / 2,
        rail_z: float = FRAME_HALF_HEIGHT,
        loop_rise: float = LOOP_RISE,
        peg_length: float = HOOK_SHANK_LENGTH + HOOK_KNOB_RADIUS,
        max_tilt_deg: float = 35.0,
        max_linear_speed: float = 0.20,
        max_angular_speed: float = 0.80,
        release_settle_time: float = 0.20,
        spawn_height: float = FRAME_START_Z,
        lift_margin_m: float = 0.10,
    ) -> torch.Tensor:
        """Score the picture against the peg.

        ``loop_half_width``, ``rail_z`` and ``loop_rise`` describe the loop's opening in
        the picture's frame [m]; ``peg_length`` is the peg's reach from the wall.
        ``lift_margin_m`` is the rise above ``spawn_height`` for ``lifted_together``.
        """
        pos = self._frame.data.root_pos_w.torch - env.scene.env_origins
        quat = self._frame.data.root_quat_w.torch
        hook_pos = self._hook.data.root_pos_w.torch - env.scene.env_origins
        hook_quat = self._hook.data.root_quat_w.torch

        # The peg's root and direction in the picture's frame, and where its
        # axis crosses the loop's mid-plane (local y = 0).
        out_of_wall = torch.zeros_like(pos)
        out_of_wall[:, 1] = -1.0
        root_local = quat_apply_inverse(quat, hook_pos - pos)
        dir_local = quat_apply_inverse(quat, quat_apply(hook_quat, out_of_wall))
        crosses = dir_local[:, 1].abs() >= 0.5
        s = -root_local[:, 1] / torch.where(crosses, dir_local[:, 1], torch.ones_like(dir_local[:, 1]))
        at_plane = root_local + s.unsqueeze(-1) * dir_local
        peg_z_above_rail = at_plane[:, 2] - rail_z
        threaded = (
            crosses
            & (s >= 0.0)
            & (s <= peg_length)
            & (at_plane[:, 0].abs() <= loop_half_width)
            & (peg_z_above_rail >= 0.0)
            & (peg_z_above_rail <= loop_rise)
        )

        # Upright and facing the wall.
        upright = up_axis(quat)[:, 2]
        back = torch.zeros_like(pos)
        back[:, 1] = 1.0
        facing = quat_apply(quat, back)[:, 1]
        limit = torch.cos(torch.deg2rad(torch.tensor(max_tilt_deg, device=pos.device)))
        hanging = (upright >= limit) & (facing >= limit)

        lin_speed = torch.linalg.norm(self._frame.data.root_lin_vel_w.torch, dim=-1)
        ang_speed = torch.linalg.norm(self._frame.data.root_ang_vel_w.torch, dim=-1)
        still = (lin_speed <= max_linear_speed) & (ang_speed <= max_angular_speed)

        grasp = self._grasp
        latched = grasp.latched
        carried_together = grasp.ever_both_robots_latched
        released = ~latched.any(dim=1) & (grasp.release_age >= release_settle_time)

        held_by_both = latched[:, :2].any(dim=1) & latched[:, 2:].any(dim=1)
        self._held_together |= held_by_both
        self._lifted_together |= held_by_both & (pos[:, 2] - spawn_height >= lift_margin_m)
        self._hooked_together |= held_by_both & threaded

        publish_checks(
            env,
            self.checks_key,
            threaded=threaded,
            hanging_square=hanging,
            at_rest=still,
            carried_together=carried_together.clone(),
            released=released,
            # Cloned: the checks store keeps references and reset() clears in place.
            held_together=self._held_together.clone(),
            lifted_together=self._lifted_together.clone(),
            hooked_together=self._hooked_together.clone(),
        )
        # Where the peg crosses the loop.
        publish_metrics(
            env,
            self.checks_key,
            peg_x_in_loop_m=at_plane[:, 0],
            peg_z_above_rail_m=peg_z_above_rail,
            peg_s_from_wall_m=s,
            tilt_upright_deg=torch.rad2deg(torch.acos(upright.clamp(-1.0, 1.0))),
            tilt_facing_deg=torch.rad2deg(torch.acos(facing.clamp(-1.0, 1.0))),
            linear_speed_mps=lin_speed,
            angular_speed_radps=ang_speed,
        )
        return threaded & hanging & still & carried_together & released


def frame_dropped(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    minimum_height: float = FRAME_DROP_HEIGHT,
) -> torch.Tensor:
    """Terminate once the picture is on its way to the floor.

    True when the picture's centre is below ``minimum_height``.
    """
    frame = env.scene[object_cfg.name]
    return (frame.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]) < minimum_height
