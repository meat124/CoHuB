"""Success condition for the cooperative carry.

``CarrySuccess`` publishes two sticky flags for the eval's subtask chain:

* ``lifted``: each robot holding and every base corner at least
  ``lift_margin_m`` above the shelf top.
* ``carried_to_goal``: the basket centre inside the goal box, each robot
  holding, and every base corner at least ``carry_clearance_m`` above the
  shelf top.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply

from cohub.tasks._base.mdp import MAX_ANGULAR_SPEED, MAX_LINEAR_SPEED, RELEASE_SETTLE_S, up_axis

from cohub.subtasks import publish_checks, publish_metrics
from cohub.g1.grasp import grasp_term

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CarrySuccess(ManagerTermBase):
    """Basket released and stably resting on the goal shelf."""

    checks_key = "carry_success"

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self._object = env.scene[object_cfg.name]
        self._grasp = grasp_term(env)
        self._lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._carried_to_goal = torch.zeros_like(self._lifted)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._lifted[env_ids] = False
        self._carried_to_goal[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        goal_distance: float = -1.0,
        goal_axis: int = 1,
        goal_axis_tolerance: float = 0.15,
        goal_cross_axis_tolerance: float = 0.15,
        goal_height: float = 0.314,
        goal_height_tolerance: float = 0.06,
        max_linear_speed: float = MAX_LINEAR_SPEED,
        max_angular_speed: float = MAX_ANGULAR_SPEED,
        max_tilt_deg: float = 15.0,
        require_release: bool = True,
        release_settle_time: float = RELEASE_SETTLE_S,
        require_carried_together: bool = True,
        base_half_size: tuple[float, float] = (0.160, 0.275),
        lift_margin_m: float = 0.03,
        carry_clearance_m: float = 0.01,
    ) -> torch.Tensor:
        """Score the basket against the goal shelf.

        ``goal_distance`` is the goal coordinate along ``goal_axis`` (0 X, 1 Y) [m];
        ``goal_height`` the basket root height with its base on a shelf top.
        ``base_half_size`` is the base footprint in the basket's frame. ``lift_margin_m``
        and ``carry_clearance_m`` are base clearances above the shelf top for the route flags.
        """
        obj = self._object
        pos = obj.data.root_pos_w.torch - env.scene.env_origins
        quat = obj.data.root_quat_w.torch
        cross_axis = 1 - goal_axis

        at_goal_distance = (pos[:, goal_axis] - goal_distance).abs() <= goal_axis_tolerance
        on_centreline = pos[:, cross_axis].abs() <= goal_cross_axis_tolerance
        at_stand_height = (pos[:, 2] - goal_height).abs() <= goal_height_tolerance
        on_stands = at_goal_distance & on_centreline & at_stand_height

        long_vector = torch.tensor((0.0, 1.0, 0.0), device=quat.device, dtype=quat.dtype).expand_as(pos)
        up = up_axis(quat)
        long_axis = quat_apply(quat, long_vector)
        tilt_limit = math.cos(math.radians(max_tilt_deg))
        oriented = (up[:, 2] >= tilt_limit) & (long_axis[:, cross_axis].abs() >= tilt_limit)

        stable = (
            torch.linalg.norm(obj.data.root_lin_vel_w.torch, dim=-1) <= max_linear_speed
        ) & (torch.linalg.norm(obj.data.root_ang_vel_w.torch, dim=-1) <= max_angular_speed)

        grasp = self._grasp
        latched = grasp.latched
        carried_together = grasp.ever_both_robots_latched
        released = ~latched.any(dim=1)
        settled = grasp.release_age >= release_settle_time
        final_hand_state = released & settled if require_release else torch.ones_like(released)

        # Lowest base corner above the shelf top.
        corners = torch.tensor(
            [
                (sx * base_half_size[0], sy * base_half_size[1], 0.0)
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
        base_clearance = pos[:, 2] + corners_w[..., 2].amin(dim=1) - goal_height

        held_by_both = latched[:, :2].any(dim=1) & latched[:, 2:].any(dim=1)
        self._lifted |= held_by_both & (base_clearance >= lift_margin_m)
        self._carried_to_goal |= (
            at_goal_distance & on_centreline & held_by_both & (base_clearance >= carry_clearance_m)
        )

        publish_checks(
            env,
            self.checks_key,
            at_goal_distance=at_goal_distance,
            on_centreline=on_centreline,
            at_stand_height=at_stand_height,
            oriented=oriented,
            at_rest=stable,
            carried_together=carried_together,
            released=final_hand_state,
            # Cloned: the checks store keeps references and reset() clears in place.
            lifted=self._lifted.clone(),
            carried_to_goal=self._carried_to_goal.clone(),
        )
        publish_metrics(env, self.checks_key, base_clearance_m=base_clearance)

        success = on_stands & oriented & stable & final_hand_state
        if require_carried_together:
            success = success & carried_together
        return success
