"""CartService termination terms.

:class:`CartServiceSuccess` latches two ordered milestones, ``cart_parked`` then
``bottle_lifted``, and then requires the bottle on the goal, upright, at rest
and released. :func:`bottle_toppled` ends the episode when a free bottle falls
over. All conditions are published through :mod:`cohub.subtasks`.
"""

from __future__ import annotations

import math

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from cohub.tasks._base.mdp import MAX_ANGULAR_SPEED, MAX_LINEAR_SPEED, RELEASE_SETTLE_S, up_axis

from cohub.subtasks import publish_checks, publish_metrics
from cohub.g1.grasp import grasp_term

from ..assets import (
    BOTTLE_GOAL_Z,
    BOTTLE_LIFT_MARGIN,
    BOTTLE_MAX_TILT_DEG,
    BOTTLE_START,
    BOTTLE_TOPPLED_DEG,
    CART_PARKED,
    CART_PARK_RADIUS,
    GOAL_XY_TOL,
    GOAL_Z_TOL,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bottle_toppled(
    env: ManagerBasedRLEnv,
    bottle_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    max_tilt_deg: float = BOTTLE_TOPPLED_DEG,
) -> torch.Tensor:
    """The bottle has gone over, and nothing in this scene stands it back up.

    Fires only while no hand holds the bottle and its tilt exceeds
    ``max_tilt_deg``.
    """
    bottle = env.scene[bottle_cfg.name]
    quat = bottle.data.root_quat_w.torch
    up = up_axis(quat)
    over = up[:, 2] < math.cos(math.radians(max_tilt_deg))
    free = ~grasp_term(env).latched.any(dim=1)
    toppled = over & free
    publish_checks(env, "bottle_toppled", bottle_standing=~toppled, bottle_in_hand=~free)
    publish_metrics(
        env,
        "bottle_toppled",
        bottle_tilt_deg=torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0))),
    )
    return toppled


class CartServiceSuccess(ManagerTermBase):
    """The ordered serving protocol, latched, followed by the end state.

    Milestones are sticky for the episode. ``bottle_lifted`` cannot latch
    before ``cart_parked``. Publishes under ``cartservice_success``.
    """

    checks_key = "cartservice_success"

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        bottle_cfg: SceneEntityCfg = cfg.params.get("bottle_cfg", SceneEntityCfg("object"))
        cart_cfg: SceneEntityCfg = cfg.params.get("cart_cfg", SceneEntityCfg("cart"))
        self._bottle = env.scene[bottle_cfg.name]
        self._cart = env.scene[cart_cfg.name]
        self._grasp = grasp_term(env)
        zeros = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._parked = zeros.clone()
        self._lifted = zeros.clone()
        self._carried_to_goal = zeros.clone()

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._parked[env_ids] = False
        self._lifted[env_ids] = False
        self._carried_to_goal[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        bottle_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        cart_cfg: SceneEntityCfg = SceneEntityCfg("cart"),
        goal_marker_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
        goal_z: float = BOTTLE_GOAL_Z,
        xy_tolerance: float = GOAL_XY_TOL,
        height_tolerance: float = GOAL_Z_TOL,
        max_tilt_deg: float = BOTTLE_MAX_TILT_DEG,
        max_linear_speed: float = MAX_LINEAR_SPEED,
        max_angular_speed: float = MAX_ANGULAR_SPEED,
        release_settle_time: float = RELEASE_SETTLE_S,
        lift_margin_m: float = BOTTLE_LIFT_MARGIN,
        park_at: tuple[float, float] = CART_PARKED,
        park_radius: float = CART_PARK_RADIUS,
        lifter: str = "robot_b",
        require_lifter_hold: bool = True,
        carry_radius_m: float = 0.15,
    ) -> torch.Tensor:
        origin = env.scene.env_origins
        bottle_pos = self._bottle.data.root_pos_w.torch[:, :3] - origin
        bottle_quat = self._bottle.data.root_quat_w.torch
        cart_pos = self._cart.data.root_pos_w.torch[:, :3] - origin

        # Milestone 1: the cart is within park_radius of park_at.
        park_offset = cart_pos[:, :2] - torch.tensor(
            park_at, device=cart_pos.device, dtype=cart_pos.dtype
        )
        park_range = torch.linalg.norm(park_offset, dim=-1)
        self._parked |= park_range <= park_radius

        # Milestone 2: after the park, Robot B holds the bottle lift_margin_m
        # above its spawn height.
        rise = bottle_pos[:, 2] - BOTTLE_START[2]
        lifter_holds = self._grasp.latched[:, self._grasp.hand_slice(lifter)].any(dim=1)
        if not require_lifter_hold:
            lifter_holds = torch.ones_like(lifter_holds)
        self._lifted |= self._parked & lifter_holds & (rise >= lift_margin_m)
        ever_held = self._grasp.ever_latched.any(dim=1)

        # End state.
        # The goal is where the patch is.
        target = (env.scene[goal_marker_cfg.name].data.root_pos_w.torch[:, :3] - origin).clone()
        target[:, 2] = goal_z
        delta = bottle_pos - target
        x_in_range = delta[:, 0].abs() <= xy_tolerance
        y_in_range = delta[:, 1].abs() <= xy_tolerance
        height_in_range = delta[:, 2].abs() <= height_tolerance
        on_goal = x_in_range & y_in_range & height_in_range
        near_goal = torch.linalg.norm(delta[:, :2], dim=-1) <= carry_radius_m
        self._carried_to_goal |= lifter_holds & near_goal

        up = up_axis(bottle_quat)
        upright = up[:, 2] >= math.cos(math.radians(max_tilt_deg))

        lin = torch.linalg.norm(self._bottle.data.root_lin_vel_w.torch, dim=-1)
        ang = torch.linalg.norm(self._bottle.data.root_ang_vel_w.torch, dim=-1)
        at_rest = (lin <= max_linear_speed) & (ang <= max_angular_speed)

        hands_off = ~self._grasp.latched.any(dim=1)
        settled = self._grasp.release_age >= release_settle_time
        released = hands_off & settled

        served = on_goal & upright & at_rest & released
        success = served & self._lifted

        publish_checks(
            env,
            self.checks_key,
            # Cloned: the checks store keeps references and reset() clears in place.
            cart_parked=self._parked.clone(),
            bottle_lifted=self._lifted.clone(),
            carried_to_goal=self._carried_to_goal.clone(),
            bottle_ever_held=ever_held,
            bottle_x_in_range=x_in_range,
            bottle_y_in_range=y_in_range,
            bottle_height_in_range=height_in_range,
            bottle_upright=upright,
            bottle_at_rest=at_rest,
            all_hands_released=hands_off,
            release_settled=settled,
        )
        publish_metrics(
            env,
            self.checks_key,
            bottle_dx_m=delta[:, 0],
            bottle_dy_m=delta[:, 1],
            bottle_dz_m=delta[:, 2],
            bottle_tilt_deg=torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0))),
            bottle_rise_m=rise,
            cart_to_park_m=park_range,
            bottle_linear_speed_mps=lin,
            bottle_angular_speed_radps=ang,
        )
        return success
