"""CoPouring's success predicate: both apples in the empty crate.

Success is every apple's centre inside the target crate's interior.
``PourSuccess`` also publishes route flags through :mod:`cohub.subtasks` for
the eval's subtask chain. A robot holds the tub when at least one of its
hands is latched on it.

* ``lifted_together``: both robots holding, the tub ``lift_margin`` off its stack.
* ``tipped_together``: after that, both robots holding, the tub past ``pour_tilt_deg``.
* ``held_now``: both robots holding at this step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from cohub.tasks._base.mdp import MAX_LINEAR_SPEED, tilt_from_upright

from cohub.subtasks import publish_checks, publish_metrics
from cohub.g1.grasp import grasp_term

from .events import bin_target

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apples_in_bin(
    env: ManagerBasedRLEnv,
    apple_names: Sequence[str],
    bin_cfg: SceneEntityCfg,
    bin_inner: tuple[float, float, float],
    bin_rim_z: float,
    max_linear_speed: float,
    max_angular_speed: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """How many apples have landed in the empty crate, and how many have settled.

    The box is the crate's interior, taken from its live pose.

    Returns ``(landed, resting)`` per environment: apples inside the crate,
    and apples at rest anywhere.
    """
    goal = bin_target(env, bin_cfg)
    landed = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    resting = torch.zeros_like(landed)
    for name in apple_names:
        apple = env.scene[name]
        delta = (apple.data.root_pos_w.torch - env.scene.env_origins) - goal
        inside = (
            (delta[:, 0].abs() <= bin_inner[0] / 2)
            & (delta[:, 1].abs() <= bin_inner[1] / 2)
            & (delta[:, 2] <= bin_rim_z)
            & (delta[:, 2] >= 0.0)
        )
        still = (
            torch.linalg.norm(apple.data.root_lin_vel_w.torch, dim=-1) <= max_linear_speed
        ) & (torch.linalg.norm(apple.data.root_ang_vel_w.torch, dim=-1) <= max_angular_speed)
        # Landing does not require the apple to be at rest.
        landed += inside.long()
        resting += still.long()
    return landed, resting


class PourMissed(ManagerTermBase):
    """An apple is on the floor. The episode is over the moment one is."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        crate_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self._crate = env.scene[crate_cfg.name]
        self._tipped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._tipped[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        apple_names: Sequence[str] = (),
        bin_cfg: SceneEntityCfg = SceneEntityCfg("catch_bin"),
        bin_inner: tuple[float, float, float] = (0.635, 0.5, 0.319),
        bin_rim_z: float = 0.343,
        pour_tilt_deg: float = 25.0,
        max_linear_speed: float = MAX_LINEAR_SPEED,
        max_angular_speed: float = 1.0,
        floor_z: float = 0.12,
    ) -> torch.Tensor:
        """True as soon as any apple is on the floor rather than in the target.

        "On the floor" is below ``floor_z`` and outside the target's footprint.
        """
        goal = bin_target(env, bin_cfg)
        spilled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for name in apple_names:
            apple = env.scene[name]
            local = apple.data.root_pos_w.torch - env.scene.env_origins
            delta = local - goal
            over_target = (
                (delta[:, 0].abs() <= bin_inner[0] / 2)
                & (delta[:, 1].abs() <= bin_inner[1] / 2)
            )
            spilled |= (local[:, 2] < floor_z) & ~over_target
        return spilled


class PourSuccess(ManagerTermBase):
    """Every apple in the empty crate; the route flags are published, not gated."""

    checks_key = "copouring_success"

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        crate_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self._crate = env.scene[crate_cfg.name]
        self._grasp = grasp_term(env)
        self._hands = [self._grasp.hand_slice(robot) for robot in self._grasp.cfg.robot_names]
        self._lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._tipped = torch.zeros_like(self._lifted)
        self._held_together = torch.zeros_like(self._lifted)
        self._lifted_together = torch.zeros_like(self._lifted)
        self._tipped_together = torch.zeros_like(self._lifted)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._lifted[env_ids] = False
        self._tipped[env_ids] = False
        self._held_together[env_ids] = False
        self._lifted_together[env_ids] = False
        self._tipped_together[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        apple_names: Sequence[str] = (),
        bin_cfg: SceneEntityCfg = SceneEntityCfg("catch_bin"),
        bin_inner: tuple[float, float, float] = (0.635, 0.5, 0.319),
        bin_rim_z: float = 0.343,
        stand_top_z: float = 0.254,
        lift_margin: float = 0.03,
        pour_tilt_deg: float = 25.0,
        min_apples: int = 1,
        max_linear_speed: float = MAX_LINEAR_SPEED,
        max_angular_speed: float = 1.0,
    ) -> torch.Tensor:
        height = self._crate.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        # The tub's origin is its base centre, so this is the gap under it.
        rise = height - stand_top_z
        lifted_now = rise >= lift_margin
        newly_lifted = ~self._lifted & lifted_now
        self._lifted |= lifted_now

        tilt = tilt_from_upright(self._crate.data.root_quat_w.torch)
        tipped_now = self._lifted & (tilt >= math.radians(pour_tilt_deg))
        newly_tipped = ~self._tipped & tipped_now
        self._tipped |= tipped_now

        latched = self._grasp.latched
        holding = [latched[:, hands].any(dim=1) for hands in self._hands]
        together = holding[0]
        for other in holding[1:]:
            together = together & other
        self._held_together |= together
        self._lifted_together |= together & lifted_now
        self._tipped_together |= (
            self._lifted_together & together & (tilt >= math.radians(pour_tilt_deg))
        )

        landed, resting = apples_in_bin(
            env, apple_names, bin_cfg, bin_inner, bin_rim_z, max_linear_speed, max_angular_speed
        )
        enough = landed >= min_apples
        success = enough

        # Cloned: the store keeps references and reset() clears in place.
        publish_checks(
            env,
            self.checks_key,
            held_together=self._held_together.clone(),
            lifted_together=self._lifted_together.clone(),
            tipped_together=self._tipped_together.clone(),
            held_now=together,
            apples_in_bin=enough,
        )
        publish_metrics(
            env,
            self.checks_key,
            apples_landed=landed.float(),
            apples_at_rest=resting.float(),
            crate_rise_m=rise,
            crate_tilt_deg=torch.rad2deg(tilt),
            held_together_info=self._held_together.float(),
        )

        for index in newly_lifted.nonzero(as_tuple=False).flatten().tolist():
            print(f"[copouring] env {index}: lift achieved (rise {float(rise[index]):.3f} m)")
        for index in newly_tipped.nonzero(as_tuple=False).flatten().tolist():
            angle = float(torch.rad2deg(tilt[index]))
            print(f"[copouring] env {index}: tip achieved ({angle:.0f} deg from upright)")
        # Goal state reached without the full chain.
        shortcut = enough & ~(self._tipped_together & together)
        if bool(shortcut.any()):
            for index in shortcut.nonzero(as_tuple=False).flatten().tolist():
                print(
                    f"[copouring] env {index}: {int(landed[index])} apple(s) in, but "
                    f"lifted_together={bool(self._lifted_together[index])} "
                    f"tipped_together={bool(self._tipped_together[index])} "
                    f"held_now={bool(together[index])}; this scores as a failure"
                )
        return success
