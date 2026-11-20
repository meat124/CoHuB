"""Success condition for the kitchen handover.

:func:`bottle_placement_success` is the geometric predicate.
:class:`BottlePlacementSuccess` wraps it with the ordered A-right-to-B-right handoff.
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

from .events import target as episode_target

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bottle_placement_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    x_tolerance: float = 0.15,
    y_tolerance: float = 0.15,
    height_tolerance: float = 0.06,
    max_linear_speed: float = MAX_LINEAR_SPEED,
    max_angular_speed: float = MAX_ANGULAR_SPEED,
    max_tilt_deg: float = 15.0,
    release_settle_time: float = RELEASE_SETTLE_S,
    debug: bool = False,
    *,
    handoff_sequence: torch.Tensor,
    grasp_lift_sequence: torch.Tensor,
) -> torch.Tensor:
    """Bottle handed from Robot A to Robot B, released, and resting on the
    table surface.

    Only ``target``'s Z is used; the goal's X and Y are read from the marker pad.
    ``handoff_sequence`` (per env) gates the result; ``grasp_lift_sequence`` is
    published as a diagnostic.
    """
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w.torch - env.scene.env_origins
    goal = episode_target(env, target).to(device=pos.device, dtype=pos.dtype)

    dx = pos[:, 0] - goal[:, 0]
    dy = pos[:, 1] - goal[:, 1]
    dz = pos[:, 2] - goal[:, 2]
    x_in_range = dx.abs() <= x_tolerance
    y_in_range = dy.abs() <= y_tolerance
    height_in_range = dz.abs() <= height_tolerance
    on_target = x_in_range & y_in_range & height_in_range

    # upright only, yaw is free
    quat = obj.data.root_quat_w.torch
    up = up_axis(quat)
    tilt_limit = math.cos(math.radians(max_tilt_deg))
    oriented = up[:, 2] >= tilt_limit

    lin_speed = torch.linalg.norm(obj.data.root_lin_vel_w.torch, dim=-1)
    ang_speed = torch.linalg.norm(obj.data.root_ang_vel_w.torch, dim=-1)
    linear_speed_ok = lin_speed <= max_linear_speed
    angular_speed_ok = ang_speed <= max_angular_speed
    stable = linear_speed_ok & angular_speed_ok

    grasp = grasp_term(env)

    # whether each robot held the bottle at some point (published, does not gate)
    ever_latched = getattr(grasp, "ever_latched", grasp.latched)
    handed_off = torch.ones_like(ever_latched[:, 0])
    for robot in grasp.cfg.robot_names:
        handed_off = handed_off & ever_latched[:, grasp.hand_slice(robot)].any(dim=1)
    handoff_sequence_ok = handoff_sequence.to(device=pos.device, dtype=torch.bool)

    released = ~grasp.latched.any(dim=1)
    release_age = getattr(grasp, "release_age", torch.zeros_like(pos[:, 0]))
    settled = release_age >= release_settle_time
    final_hand_state = released & settled

    tilt_deg = torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))
    published_checks = {
        "x_in_range": x_in_range,
        "y_in_range": y_in_range,
        "height_in_range": height_in_range,
        "upright": oriented,
        "linear_speed_ok": linear_speed_ok,
        "angular_speed_ok": angular_speed_ok,
        "all_hands_released": released,
        "release_settled": settled,
    }
    published_metrics = {
        "dx_m": dx,
        "dy_m": dy,
        "dz_m": dz,
        "tilt_deg": tilt_deg,
        "linear_speed_mps": lin_speed,
        "angular_speed_radps": ang_speed,
        "release_age_s": release_age,
        "handoff_history_info": handed_off.float(),
    }
    published_checks["grasp_lift_ok"] = grasp_lift_sequence
    published_checks["handoff_sequence_ok"] = handoff_sequence_ok
    published_checks["transfer_ok"] = handoff_sequence_ok
    published_metrics["grasp_lift_info"] = grasp_lift_sequence.float()
    published_metrics["handoff_sequence_info"] = handoff_sequence_ok.float()
    published_metrics["transfer_info"] = handoff_sequence_ok.float()
    publish_checks(env, "bottle_placement_success", **published_checks)
    publish_metrics(env, "bottle_placement_success", **published_metrics)

    success = on_target & oriented & stable & final_hand_state & handoff_sequence_ok

    if debug and bool(final_hand_state.any()) and not bool(success.any()):
        for i in final_hand_state.nonzero(as_tuple=False).flatten().tolist():
            print(
                f"[bottle_placement_success] env {i}: "
                f"on_target={bool(on_target[i])} "
                f"(target={float(goal[i,0]):+.3f},{float(goal[i,1]):+.3f},{float(goal[i,2]):+.3f} "
                f"dx={float(dx[i]):+.3f} dy={float(dy[i]):+.3f} "
                f"dz={float(dz[i]):+.3f} / tol x={x_tolerance} y={y_tolerance} z={height_tolerance}) "
                f"oriented={bool(oriented[i])} (up_z={float(up[i,2]):.3f}, "
                f"need >= {tilt_limit:.3f}) "
                f"stable={bool(stable[i])} (lin={float(lin_speed[i]):.3f}/{max_linear_speed} "
                f"ang={float(ang_speed[i]):.3f}/{max_angular_speed}) "
                f"handed_off={bool(handed_off[i])} (informational) "
                f"handoff_sequence={bool(handoff_sequence_ok[i])} "
                f"release_age={float(release_age[i]):.2f}s"
            )

    return success


class BottlePlacementSuccess(ManagerTermBase):
    """Ordered right-hand handoff followed by :func:`bottle_placement_success`.

    Two sticky milestones, in order, and the second gates the placement:

    1. ``grasp_lift``: Robot A's right hand holds the bottle ``lift_margin_m``
       above its resting height.
    2. ``transfer``: a step with A's right and B's right hands both holding it,
       then a step with B's right hand holding and both of A's hands off.

    A step with no hand on the bottle before ``transfer`` clears the handoff
    in progress.
    """

    # name the delegated predicate publishes its checks under
    checks_key = "bottle_placement_success"

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self._object = env.scene[object_cfg.name]
        self._grasp = grasp_term(env)
        self._a_right = self._grasp.hand_index("robot_a", "right")
        self._b_right = self._grasp.hand_index("robot_b", "right")
        self._a_hands = (
            self._grasp.hand_index("robot_a", "left"),
            self._a_right,
        )
        self._grasp_lift = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._transfer = torch.zeros_like(self._grasp_lift)
        self._carrying = torch.zeros_like(self._grasp_lift)  # held since a lift, no drop
        self._hand_to_hand = torch.zeros_like(self._grasp_lift)  # A right and B right at once

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._grasp_lift[env_ids] = False
        self._transfer[env_ids] = False
        self._carrying[env_ids] = False
        self._hand_to_hand[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        x_tolerance: float = 0.15,
        y_tolerance: float = 0.15,
        height_tolerance: float = 0.06,
        max_linear_speed: float = 0.05,
        max_angular_speed: float = 0.15,
        max_tilt_deg: float = 15.0,
        release_settle_time: float = RELEASE_SETTLE_S,
        lift_margin_m: float = 0.03,
        debug: bool = False,
    ) -> torch.Tensor:
        height = self._object.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        # ``target[2]`` is the bottle's resting height on the counter
        resting_height = torch.full_like(height, float(target[2]))
        rise_above_counter = height - resting_height
        lifted_now = rise_above_counter >= lift_margin_m

        latched = self._grasp.latched
        a_right_holds = latched[:, self._a_right]
        b_right_holds = latched[:, self._b_right]
        a_holds_any = latched[:, self._a_hands].any(dim=1)
        grasp_lift_now = a_right_holds & lifted_now
        newly_grasp_lifted = ~self._grasp_lift & grasp_lift_now
        self._grasp_lift |= grasp_lift_now

        # A drop before the transfer clears the handoff in progress.
        dropped = ~latched.any(dim=1) & ~self._transfer & (self._carrying | self._hand_to_hand)
        self._carrying &= ~dropped
        self._hand_to_hand &= ~dropped
        self._carrying |= grasp_lift_now
        self._hand_to_hand |= self._carrying & a_right_holds & b_right_holds
        transfer_now = self._hand_to_hand & b_right_holds & ~a_holds_any
        newly_transferred = ~self._transfer & transfer_now
        self._transfer |= transfer_now

        for i in newly_grasp_lifted.nonzero(as_tuple=False).flatten().tolist():
            rise = float(rise_above_counter[i])
            print(f"[handover] env {i}: grasp_lift achieved (rise {rise:.3f} m)")
        for i in dropped.nonzero(as_tuple=False).flatten().tolist():
            print(f"[handover] env {i}: bottle let go before the transfer -- lift it again")
        for i in newly_transferred.nonzero(as_tuple=False).flatten().tolist():
            print(f"[handover] env {i}: transfer achieved (A released, B right holds)")

        return bottle_placement_success(
            env,
            object_cfg=object_cfg,
            target=target,
            x_tolerance=x_tolerance,
            y_tolerance=y_tolerance,
            height_tolerance=height_tolerance,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
            max_tilt_deg=max_tilt_deg,
            release_settle_time=release_settle_time,
            # cloned: reset() clears these in place
            handoff_sequence=self._transfer.clone(),
            grasp_lift_sequence=self._grasp_lift.clone(),
            debug=debug,
        )
