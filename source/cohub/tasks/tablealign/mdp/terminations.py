"""TableAlign's success predicate: the table squared to the chair.

The placement is scored in the chair's frame:

* ``in_front``: the table's centre is inside ``forward_band`` along the chair's forward,
* ``centred``: within ``lateral_tolerance`` of the line the chair looks down,
* ``square``: its long axis runs across that line, within ``square_tolerance_deg``.

The turn from the spawn is published as the metric ``yaw_from_spawn_deg`` and
is not scored. Two sticky flags are published through :mod:`cohub.subtasks` for
the eval's subtask chain:

* ``lifted``: both robots holding while the table is ``lift_margin_m`` up.
* ``carried_to_goal``: the centre inside the ``in_front`` x ``centred`` box,
  each robot holding, and every foot bar ``carry_clearance_m`` off the floor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply

from cohub.tasks._base.mdp import up_axis

from cohub.subtasks import publish_checks, publish_metrics
from cohub.g1.grasp import grasp_term

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _chair_frame(
    chair_at: tuple[float, float],
    chair_forward: tuple[float, float],
    device,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The chair's seat frame: its origin, its forward, and its right.

    ``right`` is ``(fy, -fx)``, the right-hand side of a person sitting in the chair.
    """
    origin = torch.tensor(chair_at, device=device, dtype=dtype)
    forward = torch.tensor(chair_forward, device=device, dtype=dtype)
    forward = forward / torch.linalg.norm(forward)
    right = torch.stack((forward[1], -forward[0]))
    return origin, forward, right


class TableAlignSuccess(ManagerTermBase):
    """The table set down in front of the chair, square to it, and let go of."""

    checks_key = "tablealign_success"

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self._object = env.scene[object_cfg.name]
        self._grasp = grasp_term(env)
        self._ever_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._carried_to_goal = torch.zeros_like(self._ever_lifted)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._ever_lifted[env_ids] = False
        self._carried_to_goal[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        chair_at: tuple[float, float] = (0.40, -2.10),
        chair_forward: tuple[float, float] = (-1.0, 0.0),
        forward_band: tuple[float, float] = (0.335, 0.465),
        lateral_tolerance: float = 0.13,
        standing_height: float = 0.68,
        height_tolerance: float = 0.04,
        square_tolerance_deg: float = 4.0,
        max_tilt_deg: float = 15.0,
        max_linear_speed: float = 0.05,
        max_angular_speed: float = 0.15,
        release_settle_time: float = 0.20,
        lift_margin_m: float = 0.03,
        frame_x: float = 0.632,
        frame_depth: float = 0.071,
        frame_width: float = 0.626,
        carry_clearance_m: float = 0.01,
        require_release: bool = True,
        require_lifted: bool = False,
    ) -> torch.Tensor:
        """Score the table against the chair.

        ``chair_at`` and ``chair_forward`` are in the task frame. ``forward_band`` and
        ``lateral_tolerance`` bound the table's centre in the chair's frame [m];
        ``square_tolerance_deg`` its long axis. ``frame_*`` describe the foot bars,
        whose lowest corner must clear ``carry_clearance_m`` for ``carried_to_goal``.
        """
        pos = self._object.data.root_pos_w.torch - env.scene.env_origins
        quat = self._object.data.root_quat_w.torch
        origin, forward, right = _chair_frame(chair_at, chair_forward, pos.device, pos.dtype)

        # Position in the chair's frame
        delta = pos[:, :2] - origin
        forward_m = delta @ forward
        lateral_m = delta @ right
        dz = pos[:, 2] - standing_height

        in_front = (forward_m >= forward_band[0]) & (forward_m <= forward_band[1])
        centred = lateral_m.abs() <= lateral_tolerance
        at_table_height = dz.abs() <= height_tolerance

        # Orientation. The long axis is the table's local X.
        long_local = torch.tensor((1.0, 0.0, 0.0), device=quat.device, dtype=quat.dtype)
        long_axis = quat_apply(quat, long_local.expand_as(pos))
        up = up_axis(quat)

        long_xy = long_axis[:, :2]
        long_xy = long_xy / torch.linalg.norm(long_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        off_square = (long_xy @ forward).abs().clamp(max=1.0)
        square = off_square <= math.sin(math.radians(square_tolerance_deg))
        level = up[:, 2] >= math.cos(math.radians(max_tilt_deg))

        lin_speed = torch.linalg.norm(self._object.data.root_lin_vel_w.torch, dim=-1)
        ang_speed = torch.linalg.norm(self._object.data.root_ang_vel_w.torch, dim=-1)
        at_rest = (lin_speed <= max_linear_speed) & (ang_speed <= max_angular_speed)

        grasp = self._grasp
        latched = grasp.latched
        carried_together = getattr(grasp, "ever_both_robots_latched", None)
        if carried_together is None:
            carried_together = latched[:, :2].any(dim=1) & latched[:, 2:].any(dim=1)
        released = ~latched.any(dim=1)
        settled = getattr(grasp, "release_age", torch.zeros_like(dz)) >= release_settle_time
        final_hand_state = released & settled if require_release else torch.ones_like(released)

        held_by_both = latched[:, :2].any(dim=1) & latched[:, 2:].any(dim=1)
        self._ever_lifted |= held_by_both & (dz >= lift_margin_m)

        # Lowest foot-bar corner above the floor.
        corners = torch.tensor(
            [
                (sx * x, sy * frame_width / 2, -standing_height)
                for x in (frame_x - frame_depth / 2, frame_x + frame_depth / 2)
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ],
            device=pos.device,
            dtype=pos.dtype,
        )
        corners_w = quat_apply(
            quat.unsqueeze(1).expand(-1, corners.shape[0], -1),
            corners.unsqueeze(0).expand(pos.shape[0], -1, -1),
        )
        foot_clearance = pos[:, 2] + corners_w[..., 2].amin(dim=1)

        self._carried_to_goal |= (
            in_front & centred & held_by_both & (foot_clearance >= carry_clearance_m)
        )

        publish_checks(
            env,
            "tablealign_success",
            in_front=in_front,
            centred=centred,
            at_table_height=at_table_height,
            square=square,
            level=level,
            at_rest=at_rest,
            released=final_hand_state,
            carried_together=carried_together,
            # Cloned: the checks store keeps references and reset() clears in place.
            lifted=self._ever_lifted.clone(),
            carried_to_goal=self._carried_to_goal.clone(),
        )
        publish_metrics(
            env,
            "tablealign_success",
            forward_m=forward_m,
            lateral_m=lateral_m,
            dz_m=dz,
            foot_clearance_m=foot_clearance,
            square_deg=torch.rad2deg(torch.asin(off_square)),
            tilt_deg=torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0))),
            linear_speed=lin_speed,
            angular_speed=ang_speed,
            # Modulo 180: the table is the same turned half a circle.
            yaw_from_spawn_deg=torch.rad2deg(
                torch.atan2(long_xy[:, 1], long_xy[:, 0])
            )
            % 180.0,
        )

        success = (
            in_front & centred & at_table_height & square & level & at_rest & final_hand_state
        )
        if require_lifted:
            success = success & self._ever_lifted
        return success
