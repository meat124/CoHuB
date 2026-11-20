"""BigTable's success predicate and drop termination.

``BigTableSuccess`` publishes two sticky flags through :mod:`cohub.subtasks`
for the eval's subtask chain. A robot holds the table when at least one of
its hands does.

* ``lifted_by_three``: all three robots holding, foot ``lift_margin`` off the floor.
* ``carried_to_goal``: whole foot over the pad, all three holding, foot
  ``carry_clearance_m`` off the floor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from cohub.tasks._base.mdp import up_axis

from cohub.subtasks import publish_checks, publish_metrics
from cohub.g1.grasp import grasp_term

from ..assets import (
    CARRY_FOOT_CLEARANCE,
    GOAL_AT,
    GOAL_TOLERANCE,
    LIFT_MARGIN,
    PEDESTAL_FOOT_RADIUS,
    STANDING_Z,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BigTableSuccess(ManagerTermBase):
    """The table carried by three and set down on the pad, level and let go of."""

    checks_key = "bigtable_success"

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self._object = env.scene[object_cfg.name]
        self._grasp = grasp_term(env)
        self._lifted_by_three = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._held_by_three = torch.zeros_like(self._lifted_by_three)
        self._carried_to_goal = torch.zeros_like(self._lifted_by_three)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._lifted_by_three[env_ids] = False
        self._held_by_three[env_ids] = False
        self._carried_to_goal[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        goal: tuple[float, float] = GOAL_AT[:2],
        tolerance: float = GOAL_TOLERANCE,
        standing_height: float = STANDING_Z,
        height_tolerance: float = 0.05,
        max_tilt_deg: float = 15.0,
        max_linear_speed: float = 0.05,
        max_angular_speed: float = 0.15,
        release_settle_time: float = 0.20,
        lift_margin: float = LIFT_MARGIN,
        foot_radius: float = PEDESTAL_FOOT_RADIUS,
        carry_clearance_m: float = CARRY_FOOT_CLEARANCE,
        require_release: bool = True,
        require_lifted_by_three: bool = True,
    ) -> torch.Tensor:
        """Score the table against the pad.

        ``goal`` is the pad centre (x, y) and ``tolerance`` its radius [m]: the foot's
        centre may be ``tolerance - foot_radius`` from it. ``lift_margin`` and
        ``carry_clearance_m`` are the foot clearances for the two route flags.
        """
        pos = self._object.data.root_pos_w.torch - env.scene.env_origins
        quat = self._object.data.root_quat_w.torch

        up = up_axis(quat)
        level = up[:, 2] >= math.cos(math.radians(max_tilt_deg))

        # Pedestal foot centre: ``standing_height`` down the table's axis.
        foot_centre = pos - standing_height * up
        goal_xy = torch.tensor(goal, device=pos.device, dtype=pos.dtype)
        offset = torch.linalg.norm(foot_centre[:, :2] - goal_xy, dim=-1)
        dz = pos[:, 2] - standing_height
        on_the_pad = offset <= tolerance - foot_radius
        at_table_height = dz.abs() <= height_tolerance

        # Lowest point of the foot above the floor.
        sin_tilt = torch.sqrt((1.0 - up[:, 2] ** 2).clamp(min=0.0))
        foot_clearance = pos[:, 2] - standing_height * up[:, 2] - foot_radius * sin_tilt

        lin_speed = torch.linalg.norm(self._object.data.root_lin_vel_w.torch, dim=-1)
        ang_speed = torch.linalg.norm(self._object.data.root_ang_vel_w.torch, dim=-1)
        at_rest = (lin_speed <= max_linear_speed) & (ang_speed <= max_angular_speed)

        grasp = self._grasp
        latched = grasp.latched
        per_robot = latched.view(env.num_envs, len(grasp.cfg.robot_names), -1).any(dim=2)
        all_three_now = per_robot.all(dim=1)
        self._held_by_three |= all_three_now
        self._lifted_by_three |= all_three_now & (foot_clearance >= lift_margin)
        self._carried_to_goal |= on_the_pad & all_three_now & (foot_clearance >= carry_clearance_m)

        released = ~latched.any(dim=1)
        settled = grasp.release_age >= release_settle_time
        final_hand_state = released & settled if require_release else torch.ones_like(released)

        publish_checks(
            env,
            "bigtable_success",
            on_the_pad=on_the_pad,
            at_table_height=at_table_height,
            level=level,
            at_rest=at_rest,
            released=final_hand_state,
            # Cloned: the checks store keeps references and reset() clears in place.
            held_by_three=self._held_by_three.clone(),
            lifted_by_three=self._lifted_by_three.clone(),
            carried_to_goal=self._carried_to_goal.clone(),
        )
        publish_metrics(
            env,
            "bigtable_success",
            foot_offset_m=offset,
            dz_m=dz,
            foot_clearance_m=foot_clearance,
            tilt_deg=torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0))),
            linear_speed=lin_speed,
            angular_speed=ang_speed,
            robots_holding=per_robot.sum(dim=1).to(pos.dtype),
        )

        success = on_the_pad & at_table_height & level & at_rest & final_hand_state
        if require_lifted_by_three:
            success = success & self._lifted_by_three
        return success


def table_dropped(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    minimum_height: float = STANDING_Z - 0.22,
    max_tilt_deg: float = 50.0,
) -> torch.Tensor:
    """Terminate once the table is on the floor or on its side.

    A tipped table keeps its origin high, so tilt is tested as well as height.
    """
    table = env.scene[object_cfg.name]
    pos = table.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    quat = table.data.root_quat_w.torch
    up = up_axis(quat)
    return (pos < minimum_height) | (up[:, 2] < math.cos(math.radians(max_tilt_deg)))
