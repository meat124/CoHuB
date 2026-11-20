"""Sub-conditions of a task: the checks a success predicate publishes, and the trackers that read them.

A tracker watches an episode and latches each milestone the first time it
holds, so an evaluation can score partial progress. The last milestone is the
episode's verdict, gated on its prerequisites.
"""

from __future__ import annotations

import torch

# --- Published checks ---
# Tensors are stored as published and converted to bools or floats only when read.

_ATTR = "_cohub_checks"
_METRICS_ATTR = "_cohub_check_metrics"


def publish_checks(env, key: str, **checks: torch.Tensor) -> None:
    """Record the sub-checks ``key`` evaluated, replacing its last set.

    Args:
        key: Names the predicate. Terms sharing one function must qualify it,
            or the second call overwrites the first.
    """
    store = getattr(env, _ATTR, None)
    if store is None:
        store = {}
        setattr(env, _ATTR, store)
    store[key] = checks


def read_checks(env, key: str, env_id: int = 0) -> dict[str, bool]:
    """The last published breakdown of ``key``, as plain bools. Empty if none."""
    store = getattr(env, _ATTR, None) or {}
    out: dict[str, bool] = {}
    for name, value in store.get(key, {}).items():
        try:
            out[name] = bool(value.reshape(-1)[env_id])
        except (AttributeError, IndexError, RuntimeError):
            out[name] = bool(value)
    return out


def publish_metrics(env, key: str, **metrics: torch.Tensor) -> None:
    """Record numeric diagnostics computed by a predicate."""
    store = getattr(env, _METRICS_ATTR, None)
    if store is None:
        store = {}
        setattr(env, _METRICS_ATTR, store)
    store[key] = metrics


def read_metrics(env, key: str, env_id: int = 0) -> dict[str, float]:
    """Return the last numeric diagnostics for ``key`` as plain floats."""
    store = getattr(env, _METRICS_ATTR, None) or {}
    out: dict[str, float] = {}
    for name, value in store.get(key, {}).items():
        try:
            item = value.reshape(-1)[env_id]
            out[name] = float(item.item() if hasattr(item, "item") else item)
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            try:
                out[name] = float(value)
            except (TypeError, ValueError):
                continue
    return out


# --- Subtask trackers ---

# Seconds the hinge must stay past its open angle for the door to count as opened.
DOOR_OPEN_HOLD_S = 0.5


def _door_hinge_angle(env, joint_name: str) -> float:
    """How far the door panel has swung, either way [rad]."""
    door = env.scene["door"]
    index = list(door.data.joint_names).index(joint_name)
    return abs(float(door.data.joint_pos.torch[0, index]))


class SubtaskTracker:
    """Sticky milestone accumulation for one episode of one env (env 0).

    ``names`` is the canonical order. The last name is the final-success
    subtask: :meth:`result` fills it from the verdict, never :meth:`update`.
    ``checks_key`` names the task's success predicate in the
    :mod:`cohub.subtasks` store.
    """

    checks_key: str = ""
    names: tuple[str, ...] = ()
    # A subtask only counts once its prerequisites have. Default: the previous name.
    prerequisites: dict[str, tuple[str, ...]] = {}
    details: tuple[str, ...] = ()
    """Milestones recorded beside the chain, reported as ``subtask_details``."""

    def __init__(self) -> None:
        self._prereqs: dict[str, tuple[str, ...]] = {}
        for i, name in enumerate(self.names):
            default = (self.names[i - 1],) if i else ()
            self._prereqs[name] = self.prerequisites.get(name, default)
        self.reset()

    def reset(self) -> None:
        self._achieved: dict[str, int | None] = {name: None for name in self.names}
        self._detail_steps: dict[str, int | None] = {name: None for name in self.details}

    def update(self, env, step: int, terminal: bool) -> None:
        """Called once per executed control step, after ``env.step``.

        On a ``terminal`` step the scene already belongs to the next episode:
        read only the checks store, or skip the sample.
        """
        raise NotImplementedError

    def snapshot(self, env) -> tuple[dict[str, bool], dict[str, float]]:
        """The success predicate's own published breakdown, as plain types."""
        return read_checks(env, self.checks_key), read_metrics(env, self.checks_key)

    def result(self, final_success: bool, final_step: int) -> tuple[dict[str, bool], dict[str, int | None]]:
        """``({name: achieved}, {name: first step achieved})`` in canonical order.

        The last name is the verdict, gated by its prerequisites.
        """
        if not self.names:
            return {}, {}
        subtasks: dict[str, bool] = {}
        steps: dict[str, int | None] = {}
        for name in self.names[:-1]:
            steps[name] = self._achieved[name]
            subtasks[name] = self._achieved[name] is not None
        final = self.names[-1]
        ok = bool(final_success) and all(
            self._achieved[p] is not None for p in self._prereqs[final]
        )
        subtasks[final] = ok
        steps[final] = final_step if ok else None
        return subtasks, steps

    def detail_result(self) -> tuple[dict[str, bool], dict[str, int | None]]:
        """``({detail: achieved}, {detail: first step})`` in :attr:`details` order."""
        steps = dict(self._detail_steps)
        return {name: step is not None for name, step in steps.items()}, steps

    def _note(self, name: str, step: int) -> None:
        """Record a detail milestone's first step. No prerequisites."""
        if self._detail_steps[name] is None:
            self._detail_steps[name] = step

    def _achieve(self, name: str, step: int) -> None:
        """Record a milestone's first step. Refused while a prerequisite is unachieved."""
        if self._achieved[name] is not None:
            return
        if any(self._achieved[p] is None for p in self._prereqs[name]):
            return
        self._achieved[name] = step

    def achieved(self) -> dict[str, int | None]:
        """``{name: first step achieved}`` for the chain so far, the verdict excluded."""
        return {name: self._achieved[name] for name in self.names[:-1]}

    def force(self, name: str, step: int) -> None:
        """Mark ``name`` and everything before it achieved, prerequisites or not.

        For an episode that does not begin at the beginning. Steps already
        recorded are kept.
        """
        for link in self.names[: self.names.index(name) + 1]:
            if self._achieved[link] is None:
                self._achieved[link] = step


class HandoverTracker(SubtaskTracker):
    """grasp_lift -> transfer -> place, using only the two right hands.

    Reads the success term's sticky ``grasp_lift_ok`` and ``transfer_ok``
    flags. ``place`` is the env's own placement verdict.
    """

    checks_key = "bottle_placement_success"
    names = ("grasp_lift", "transfer", "place")

    def update(self, env, step: int, terminal: bool) -> None:
        # Fall back to the numeric metrics when the boolean flags are absent.
        published = read_checks(env, self.checks_key)
        metrics = read_metrics(env, self.checks_key)
        grasp_lift_ok = published.get(
            "grasp_lift_ok", metrics.get("grasp_lift_info", 0.0) > 0.5
        )
        transfer_ok = published.get(
            "transfer_ok",
            published.get(
                "handoff_sequence_ok",
                metrics.get("transfer_info", metrics.get("handoff_sequence_info", 0.0)) > 0.5,
            ),
        )
        if grasp_lift_ok:
            self._achieve("grasp_lift", step)
        if transfer_ok:
            self._achieve("transfer", step)


class PouringTracker(SubtaskTracker):
    """lift_both -> tip -> pour, straight off the success term's own latches.

    ``lift_both``: both vessels off the worktop at once.
    ``tip``: after that, the cup rolled past ``assets.POUR_TILT_DEG``.
    ``pour``: the env verdict.
    """

    checks_key = "pouring_success"
    names = ("lift_both", "tip", "pour")

    def update(self, env, step: int, terminal: bool) -> None:
        published = read_checks(env, self.checks_key)
        if published.get("lift_both_ok"):
            self._achieve("lift_both", step)
        if published.get("tip_ok"):
            self._achieve("tip", step)


class CoPouringTracker(SubtaskTracker):
    """grasp_lift -> tip -> pour, all read off ``PourSuccess``'s store.

    ``grasp_lift``: both robots holding with the tub ``assets.LIFT_MARGIN`` off its stack.
    ``tip``: after that, both robots holding with the tub past ``assets.POUR_TILT_DEG``.
    ``pour``: the env verdict, with both robots holding at that step.
    """

    checks_key = "copouring_success"
    names = ("grasp_lift", "tip", "pour")

    def reset(self) -> None:
        super().reset()
        self._held_now = False

    def update(self, env, step: int, terminal: bool) -> None:
        published = read_checks(env, self.checks_key)
        if published.get("lifted_together"):
            self._achieve("grasp_lift", step)
        if published.get("tipped_together"):
            self._achieve("tip", step)
        self._held_now = bool(published.get("held_now", False))

    def result(self, final_success: bool, final_step: int):
        return super().result(bool(final_success) and self._held_now, final_step)


class FrameHangTracker(SubtaskTracker):
    """grasp -> lift -> hang, with hook as a detail, all read off ``HangSuccess``'s store.

    ``grasp``: both robots holding at once.
    ``lift``: both robots holding with the picture's centre 0.10 m above its spawn height.
    ``hang``: the env verdict, and the ``hook`` detail achieved (the wall peg
    threaded through the loop with both robots holding).
    """

    checks_key = "framehang_success"
    names = ("grasp", "lift", "hang")
    details = ("hook",)

    def update(self, env, step: int, terminal: bool) -> None:
        published = read_checks(env, self.checks_key)
        if published.get("held_together"):
            self._achieve("grasp", step)
        if published.get("lifted_together"):
            self._achieve("lift", step)
        if published.get("hooked_together"):
            self._note("hook", step)

    def result(self, final_success: bool, final_step: int):
        hooked, _ = self.detail_result()
        return super().result(bool(final_success) and hooked["hook"], final_step)


class MoveHouseTracker(SubtaskTracker):
    """lift_open -> pass_door -> place, with grasp_lift, open_door and carry as details.

    ``lift_open``: the carriers' ``grasp_lift`` and Robot A's ``open_door``, in
    either order. ``open_door`` is the hinge at or past ``open_angle`` for
    :data:`DOOR_OPEN_HOLD_S`, read live and skipped on the terminal step.
    ``pass_door``: the table, B and C past ``assets.DOOR_CLEAR_X`` while carrying.
    ``place``: the env verdict, and the ``carry`` detail achieved.
    """

    checks_key = "movehouse_success"
    names = ("lift_open", "pass_door", "place")
    details = ("grasp_lift", "open_door", "carry")

    def __init__(
        self,
        hinge_joint: str = "hinge_joint",
        open_angle: float = 1.0,
        open_hold_s: float = DOOR_OPEN_HOLD_S,
    ) -> None:
        self._hinge_joint = hinge_joint
        self._open_angle = open_angle
        self._open_hold_s = open_hold_s
        super().__init__()

    def reset(self) -> None:
        super().reset()
        self._open_held_s = 0.0

    def update(self, env, step: int, terminal: bool) -> None:
        published = read_checks(env, self.checks_key)
        if published.get("lifted"):
            self._note("grasp_lift", step)
        if not terminal:
            if _door_hinge_angle(env, self._hinge_joint) >= self._open_angle:
                self._open_held_s += float(env.step_dt)
            else:
                self._open_held_s = 0.0
            if self._open_held_s >= self._open_hold_s:
                self._note("open_door", step)
        if published.get("carried_to_goal"):
            self._note("carry", step)
        done, _ = self.detail_result()
        if done["grasp_lift"] and done["open_door"]:
            self._achieve("lift_open", step)
        if published.get("carried_through_door"):
            self._achieve("pass_door", step)

    def result(self, final_success: bool, final_step: int):
        carried, _ = self.detail_result()
        return super().result(bool(final_success) and carried["carry"], final_step)


class CoCarryTracker(SubtaskTracker):
    """grasp_lift -> carry -> release, all read off ``CarrySuccess``'s store.

    ``grasp_lift``: each robot holding with the basket's base 3 cm above the shelf top.
    ``carry``: the basket centre inside the goal box while held and
    ``assets.CARRY_BASE_CLEARANCE`` above the shelf top.
    ``release``: the env verdict.
    """

    checks_key = "carry_success"
    names = ("grasp_lift", "carry", "release")

    def update(self, env, step: int, terminal: bool) -> None:
        published = read_checks(env, self.checks_key)
        if published.get("lifted"):
            self._achieve("grasp_lift", step)
        if published.get("carried_to_goal"):
            self._achieve("carry", step)


class TableAlignTracker(SubtaskTracker):
    """grasp_lift -> carry -> align, all read off ``TableAlignSuccess``'s store.

    ``grasp_lift``: both robots holding with the table 0.03 m above its standing height.
    ``carry``: the table's centre inside the goal box while held, every foot
    bar ``assets.CARRY_FOOT_CLEARANCE`` off the floor.
    ``align``: the env verdict.
    """

    checks_key = "tablealign_success"
    names = ("grasp_lift", "carry", "align")

    def update(self, env, step: int, terminal: bool) -> None:
        published = read_checks(env, self.checks_key)
        if published.get("lifted"):
            self._achieve("grasp_lift", step)
        if published.get("carried_to_goal"):
            self._achieve("carry", step)


class TrashCollectionTracker(SubtaskTracker):
    """lift -> deliver -> sweep, all read off ``TrashCollectionSuccess``'s store.

    ``lift``: Robot A holding the bin with its base ``assets.BIN_OFF_FLOOR`` off the floor.
    ``deliver``: after that, the bin held aloft within ``assets.DELIVER_RADIUS``
    of the carry point, its rim at or under the bench top.
    ``sweep``: the env verdict, with the bin held aloft at that step.
    """

    checks_key = "trashcollection_success"
    names = ("lift", "deliver", "sweep")

    def reset(self) -> None:
        super().reset()
        self._held_aloft = False

    def update(self, env, step: int, terminal: bool) -> None:
        published = read_checks(env, self.checks_key)
        if published.get("lifted"):
            self._achieve("lift", step)
        if published.get("delivered"):
            self._achieve("deliver", step)
        self._held_aloft = bool(published.get("bin_held_aloft", False))

    def result(self, final_success: bool, final_step: int):
        return super().result(bool(final_success) and self._held_aloft, final_step)


class CartServiceTracker(SubtaskTracker):
    """park -> lift -> serve, with carry as a detail, all read off ``CartServiceSuccess``'s store.

    ``park``: the trolley within ``assets.CART_PARK_RADIUS`` of ``assets.CART_PARKED``.
    ``lift``: after the park, Robot B holding the bottle
    ``assets.BOTTLE_LIFT_MARGIN`` above its spawn.
    ``serve``: the env verdict, and the ``carry`` detail achieved (Robot B
    holding the bottle within ``assets.CARRY_GOAL_RADIUS`` of the goal).
    """

    checks_key = "cartservice_success"
    names = ("park", "lift", "serve")
    details = ("carry",)

    def update(self, env, step: int, terminal: bool) -> None:
        published = read_checks(env, self.checks_key)
        if published.get("cart_parked"):
            self._achieve("park", step)
        if published.get("bottle_lifted"):
            self._achieve("lift", step)
        if published.get("carried_to_goal"):
            self._note("carry", step)

    def result(self, final_success: bool, final_step: int):
        carried, _ = self.detail_result()
        return super().result(bool(final_success) and carried["carry"], final_step)


class BigTableTracker(SubtaskTracker):
    """grasp_lift -> carry -> release, all read off ``BigTableSuccess``'s store.

    ``grasp_lift``: all three robots holding with the pedestal foot
    ``assets.LIFT_MARGIN`` off the floor.
    ``carry``: the whole pedestal foot over the pad while held,
    ``assets.CARRY_FOOT_CLEARANCE`` off the floor.
    ``release``: the env verdict.
    """

    checks_key = "bigtable_success"
    names = ("grasp_lift", "carry", "release")

    def update(self, env, step: int, terminal: bool) -> None:
        published = read_checks(env, self.checks_key)
        if published.get("lifted_by_three"):
            self._achieve("grasp_lift", step)
        if published.get("carried_to_goal"):
            self._achieve("carry", step)


def make_tracker(gym_id: str) -> SubtaskTracker | None:
    """The tracker for ``gym_id``, or None for an id this file does not know."""
    if gym_id == "Isaac-Handover-G1x2-v0":
        return HandoverTracker()
    if gym_id == "Isaac-FrameHang-G1x2-v0":
        return FrameHangTracker()
    if gym_id == "Isaac-CoCarry-G1x2-v0":
        return CoCarryTracker()
    if gym_id == "Isaac-BigTable-G1x3-v0":
        return BigTableTracker()
    if gym_id == "Isaac-Pouring-G1x2-v0":
        return PouringTracker()
    if gym_id == "Isaac-MoveHouse-G1x3-v0":
        # Imported lazily: the assets module needs Isaac Lab at import.
        from cohub.tasks.movehouse.assets import DOOR_CLEAR_ANGLE

        return MoveHouseTracker(open_angle=DOOR_CLEAR_ANGLE)
    if gym_id == "Isaac-TableAlign-G1x2-v0":
        return TableAlignTracker()
    if gym_id == "Isaac-TrashCollection-G1x2-v0":
        return TrashCollectionTracker()
    if gym_id == "Isaac-CartService-G1x2-v0":
        return CartServiceTracker()
    if gym_id == "Isaac-CoPouring-G1x2-v0":
        return CoPouringTracker()
    return None
