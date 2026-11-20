"""MoveHouse's success predicate: the table put down in the living room.

The verdict is the placement: every foot on a pad, level, at rest, and both
carriers' hands off. Robot A's door is published but not part of the verdict.

Sticky route flags, published for the subtask chain and not gating the verdict.
"Holding" is per carrier: at least one of its two hands. "Clearance" is the
lowest of the four feet above the floor.

* ``lifted``: a step with B and C holding and clearance ``lift_margin_m``.
* ``carried_through_door``: a step with the table, B and C all at
  ``x >= door_clear_x``, B and C holding, and clearance ``carry_clearance_m``.
* ``carried_to_goal``: a step with the table's centre within
  ``carry_radius_m`` of ``goal``, B and C holding, and clearance
  ``carry_clearance_m``.
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


class MoveHouseSuccess(ManagerTermBase):
    """The table set down with each foot on a mark, and let go of.

    Each foot has to stand within ``mark_half_size`` of one of ``marks`` on both
    axes. A foot may stand on any pad. The route flags in the module docstring
    are published and do not gate the verdict.
    """

    checks_key = "movehouse_success"

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self._object = env.scene[object_cfg.name]
        self._grasp = grasp_term(env)
        self._ever_carried = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._ever_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._through = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._carried_through_door = torch.zeros_like(self._through)
        self._carried_to_goal = torch.zeros_like(self._through)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._ever_carried[env_ids] = False
        self._ever_lifted[env_ids] = False
        self._through[env_ids] = False
        self._carried_through_door[env_ids] = False
        self._carried_to_goal[env_ids] = False

    def _door_angle(self, env: ManagerBasedRLEnv, hinge_joint: str) -> torch.Tensor | None:
        """The hinge's angle, or ``None`` in a scene with no door."""
        door = env.scene.articulations.get("door")
        if door is None:
            return None
        names = list(door.data.joint_names)
        if hinge_joint not in names:
            return None
        return door.data.joint_pos.torch[:, names.index(hinge_joint)]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        carriers: tuple[str, ...] = ("robot_b", "robot_c"),
        goal: tuple[float, float] = (2.60, 1.378),
        feet: tuple[tuple[float, float, float], ...] = (),
        marks: tuple[tuple[float, float], ...] = (),
        mark_half_size: float = 0.05,
        standing_height: float = 0.8855,
        height_tolerance: float = 0.04,
        max_tilt_deg: float = 15.0,
        max_linear_speed: float = 0.05,
        max_angular_speed: float = 0.15,
        release_settle_time: float = 0.20,
        lift_margin_m: float = 0.03,
        door_clear_x: float = 0.05,
        carry_radius_m: float = 0.15,
        carry_clearance_m: float = 0.01,
        hinge_joint: str = "hinge_joint",
        require_release: bool = True,
        require_carried: bool = False,
    ) -> torch.Tensor:
        """Score the table against the mark.

        ``feet`` are the table's feet in its own frame and ``marks`` the pads in the
        task frame [m]; a foot may be ``mark_half_size`` from its pad on each axis.
        ``goal`` is only published (``dx_m``/``dy_m``). ``door_clear_x``,
        ``carry_radius_m`` and the two clearances define the route flags.
        """
        pos = self._object.data.root_pos_w.torch - env.scene.env_origins
        quat = self._object.data.root_quat_w.torch

        # every foot against every pad: (envs, feet, pads)
        feet_local = torch.tensor(feet, device=pos.device, dtype=pos.dtype)
        pads = torch.tensor(marks, device=pos.device, dtype=pos.dtype)
        n_env, n_feet = pos.shape[0], feet_local.shape[0]
        feet_w = pos.unsqueeze(1) + quat_apply(
            quat.unsqueeze(1).expand(n_env, n_feet, 4),
            feet_local.unsqueeze(0).expand(n_env, n_feet, 3),
        )
        # lowest foot above the floor
        foot_clearance = feet_w[..., 2].amin(dim=1)
        off = (feet_w[:, :, None, :2] - pads[None, None, :, :]).abs().amax(dim=-1)
        foot_off = off.amin(dim=-1)  # each foot's distance to its nearest pad
        feet_on_marks = foot_off <= mark_half_size
        on_marks = feet_on_marks.all(dim=-1)

        dx = pos[:, 0] - goal[0]
        dy = pos[:, 1] - goal[1]
        dz = pos[:, 2] - standing_height
        at_table_height = dz.abs() <= height_tolerance

        # heading and tilt. The heading is published only. The long axis is the table's local X.
        long_local = torch.tensor((1.0, 0.0, 0.0), device=quat.device, dtype=quat.dtype)
        long_axis = quat_apply(quat, long_local.expand_as(pos))
        up = up_axis(quat)
        long_xy = long_axis[:, :2]
        long_xy = long_xy / torch.linalg.norm(long_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        off_line = long_xy[:, 1].abs().clamp(max=1.0)
        level = up[:, 2] >= math.cos(math.radians(max_tilt_deg))

        # at rest and released
        lin_speed = torch.linalg.norm(self._object.data.root_lin_vel_w.torch, dim=-1)
        ang_speed = torch.linalg.norm(self._object.data.root_ang_vel_w.torch, dim=-1)
        at_rest = (lin_speed <= max_linear_speed) & (ang_speed <= max_angular_speed)

        grasp = self._grasp
        held = [grasp.latched[:, grasp.hand_slice(name)].any(dim=1) for name in carriers]
        held_by_both = held[0]
        for other in held[1:]:
            held_by_both = held_by_both & other
        self._ever_carried |= held_by_both
        self._ever_lifted |= held_by_both & (foot_clearance >= lift_margin_m)

        carried_now = held_by_both & (foot_clearance >= carry_clearance_m)
        all_through = pos[:, 0] >= door_clear_x
        for name in carriers:
            carrier_x = env.scene[name].data.root_pos_w.torch[:, 0] - env.scene.env_origins[:, 0]
            all_through = all_through & (carrier_x >= door_clear_x)
        self._carried_through_door |= all_through & carried_now
        goal_xy = torch.tensor(goal, device=pos.device, dtype=pos.dtype)
        near_goal = torch.linalg.norm(pos[:, :2] - goal_xy, dim=-1) <= carry_radius_m
        self._carried_to_goal |= near_goal & carried_now

        released = ~torch.stack(held, dim=0).any(dim=0)
        settled = torch.ones_like(released)
        for name in carriers:
            index = grasp.cfg.robot_names.index(name)
            settled = settled & (grasp.robot_release_age[:, index] >= release_settle_time)
        final_hand_state = (
            released & settled if require_release else torch.ones_like(released)
        )

        # door and doorway, published only
        hinge = self._door_angle(env, hinge_joint)
        if hinge is None:
            hinge = torch.zeros_like(dz)
        door_open = hinge.abs() >= 1.0
        # the doorway plane is x = 0 and the living room is +x
        self._through |= pos[:, 0] > 0.0

        publish_checks(
            env,
            "movehouse_success",
            on_marks=on_marks,
            at_table_height=at_table_height,
            level=level,
            at_rest=at_rest,
            released=final_hand_state,
            # cloned: reset() clears these in place
            carried_together=self._ever_carried.clone(),
            lifted=self._ever_lifted.clone(),
            through_doorway=self._through.clone(),
            carried_through_door=self._carried_through_door.clone(),
            carried_to_goal=self._carried_to_goal.clone(),
            door_open=door_open,
        )
        publish_metrics(
            env,
            "movehouse_success",
            feet_on_marks=feet_on_marks.sum(dim=-1).float(),
            worst_foot_off_m=foot_off.amax(dim=-1),
            dx_m=dx,
            dy_m=dy,
            dz_m=dz,
            foot_clearance_m=foot_clearance,
            square_deg=torch.rad2deg(torch.asin(off_line)),
            tilt_deg=torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0))),
            linear_speed=lin_speed,
            angular_speed=ang_speed,
            hinge_rad=hinge,
        )

        success = on_marks & at_table_height & level & at_rest & final_hand_state
        if require_carried:
            success = success & self._ever_carried
        return success
