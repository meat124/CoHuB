"""One row per task: the facts about a task that are not its env cfg, and its sentences.

``TASKS`` stays a pure literal: ``baselines/common.sh`` reads it with
``ast.literal_eval`` on a bare interpreter.

    slug        dataset spelling: ``datasets/<slug>/``, checkpoint names
    scene       the ``cohub.tasks`` package and ``baselines/env_cfg/<scene>.yml``; equals ``slug``
    robots      robots in the scene; the teleop action is ``32 * robots`` wide
    grasp       ``"physical"``: fingers closed on the object
    bimanual    both robots hold the one object at the same time
    evaluated   has an env_cfg yml and a subtask tracker
    multitask   its episodes go into the multitask LeRobot export
    step_limit  evaluation budget in control steps (50 Hz)
"""

from __future__ import annotations


TASKS = {
    "Isaac-CoCarry-G1x2-v0": {
        "slug": "cocarry",
        "scene": "cocarry",
        "robots": 2,
        "grasp": "physical",
        "bimanual": True,
        "evaluated": True,
        "multitask": True,
        "step_limit": 1000,
    },
    "Isaac-Handover-G1x2-v0": {
        "slug": "handover",
        "scene": "handover",
        "robots": 2,
        "grasp": "physical",
        "bimanual": False,
        "evaluated": True,
        "multitask": True,
        "step_limit": 1000,
    },
    "Isaac-FrameHang-G1x2-v0": {
        "slug": "framehang",
        "scene": "framehang",
        "robots": 2,
        "grasp": "physical",
        "bimanual": True,
        "evaluated": True,
        "multitask": True,
        "step_limit": 1000,
    },
    "Isaac-Pouring-G1x2-v0": {
        "slug": "pouring",
        "scene": "pouring",
        "robots": 2,
        "grasp": "physical",
        "bimanual": False,
        "evaluated": True,
        "multitask": True,
        "step_limit": 1000,
    },
    "Isaac-CoPouring-G1x2-v0": {
        "slug": "copouring",
        "scene": "copouring",
        "robots": 2,
        "grasp": "physical",
        "bimanual": True,
        "evaluated": True,
        "multitask": True,
        "step_limit": 1000,
    },
    "Isaac-TrashCollection-G1x2-v0": {
        "slug": "trashcollection",
        "scene": "trashcollection",
        "robots": 2,
        "grasp": "physical",
        "bimanual": False,
        "evaluated": True,
        "multitask": True,
        "step_limit": 1000,
    },
    "Isaac-CartService-G1x2-v0": {
        "slug": "cartservice",
        "scene": "cartservice",
        "robots": 2,
        "grasp": "physical",
        "bimanual": False,
        "evaluated": True,
        "multitask": True,
        "step_limit": 2000,
    },
    "Isaac-TableAlign-G1x2-v0": {
        "slug": "tablealign",
        "scene": "tablealign",
        "robots": 2,
        "grasp": "physical",
        "bimanual": True,
        "evaluated": True,
        "multitask": True,
        "step_limit": 1100,
    },
    "Isaac-MoveHouse-G1x3-v0": {
        "slug": "movehouse",
        "scene": "movehouse",
        "robots": 3,
        "grasp": "physical",
        "bimanual": True,
        "evaluated": True,
        "multitask": False,
        "step_limit": 1500,
    },
    "Isaac-BigTable-G1x3-v0": {
        "slug": "bigtable",
        "scene": "bigtable",
        "robots": 3,
        "grasp": "physical",
        "bimanual": True,
        "evaluated": True,
        "multitask": False,
        "step_limit": 1000,
    },
}
"""gym id -> the facts above. Ordered as the benchmark presents its tasks."""


def gym_ids(**where):
    """Registered ids, optionally filtered: ``gym_ids(evaluated=True)``."""
    return tuple(
        gym_id
        for gym_id, row in TASKS.items()
        if all(row[key] == value for key, value in where.items())
    )


def gym_id(task: str) -> str:
    """The gym id of a task given by its name (``cocarry``) or by its id."""
    if task in TASKS:
        return task
    for candidate, row in TASKS.items():
        if row["slug"] == task:
            return candidate
    raise SystemExit(f"unknown task {task!r}: one of {', '.join(row['slug'] for row in TASKS.values())}")


def slugs(**where):
    """Dataset spellings, deduplicated, in registry order."""
    seen = []
    for gym_id in gym_ids(**where):
        slug = TASKS[gym_id]["slug"]
        if slug not in seen:
            seen.append(slug)
    return tuple(seen)


# --- What each task tells each robot to do ---

TASK_TEXT: dict[str, tuple[str, ...]] = {
    "Isaac-CoCarry-G1x2-v0": (
        "Carry the laundry basket together and set it down level on the green patch on the far shelf.",
        "Hold your side of the basket with both hands and side-step to your right, keeping it level, until it rests on the green patch.",
        "Hold your side of the basket with both hands and side-step to your left, keeping it level, until it rests on the green patch.",
    ),
    "Isaac-Handover-G1x2-v0": (
        "Pass the bottle directly from Robot A's right hand to Robot B's right hand, then place it on the target.",
        "Pick up the bottle directly in front of you with your right hand and hand it to your partner's right hand.",
        "Receive the bottle with your right hand, place it on the target directly in front of you, and release it.",
    ),
    "Isaac-FrameHang-G1x2-v0": (
        "Lift the framed painting off the floor together, hang the loop on its top rail on the wall peg, and let go once it is hanging square.",
        "Take the left edge of the frame's moulding with one hand, lift level with your partner, guide the loop onto the peg, and release once it is seated.",
        "Take the right edge of the frame's moulding with one hand, lift level with your partner, guide the loop onto the peg, and release once it is seated.",
    ),
    "Isaac-Pouring-G1x2-v0": (
        "Pour the marble from the small cup into the glass the other robot is holding in the air.",
        "Take the squat drinking cup in front of you round its body, hold it over the mouth of your partner's glass and tip it until the marble drops into the glass, then set the cup back on the table.",
        "Take the tall glass in front of you round its body, lift it off the table, and hold it out upright and still under your partner's cup until the marble is in it.",
    ),
    # The target crate is on A's left and B's right.
    "Isaac-CoPouring-G1x2-v0": (
        "Lift the tub of apples together and pour all of them into the empty crate beside you.",
        "Take the bar at your end of the tub with one hand, lift it level with your partner, and tip it together into the empty crate on your left until all the apples are in it.",
        "Take the bar at your end of the tub with one hand, lift it level with your partner, and tip it together into the empty crate on your right until all the apples are in it.",
    ),
    "Isaac-TrashCollection-G1x2-v0": (
        "Bring the bin to the bench and sweep the bottle off the bench into it.",
        "Bend and take the tall bin in both hands, carry it upright straight ahead to the end of your partner's bench, and hold it there with its rim just below the bench top.",
        "Wait at the bench. When the bin is at the end of it, sweep the plastic bottle lying between your hands along the surface and over that end into the bin; it rolls, so push straight and keep it off the near edge.",
    ),
    "Isaac-CartService-G1x2-v0": (
        "Push the loaded service trolley down the aisle to the green mark on the floor, then take the wine off it and stand the bottle on the green patch on the table.",
        "Take the push bar of the trolley in front of you with both hands and push it straight down the aisle until it stands on the green rectangle painted on the floor.",
        "Walk up the aisle to meet the trolley and stand square in front of its near edge; take the wine bottle around the glass just above the steel ring it stands in, lift it straight up out of the ring, then step round to your right to the near edge of the table and stand it on the green patch.",
    ),
    # A faces +X and B faces -X, so the -Y lane is A's right and B's left.
    "Isaac-TableAlign-G1x2-v0": (
        "Carry the table to the chair together and set it down square to it.",
        "Take your end of the table with both hands, lift level with your partner, and side-step to your right down the room, turning the table a quarter circle together as you go, until it stands on the green patch with the chair tucked into its long edge.",
        "Take your end of the table with both hands, lift level with your partner, and side-step to your left down the room, turning the table a quarter circle together as you go, until it stands on the green patch with the chair tucked into its long edge.",
    ),
    # B leads through the doorway walking backwards; C follows walking forwards.
    "Isaac-MoveHouse-G1x3-v0": (
        "One robot opens the door and stands clear while the other two carry the table out of the bedroom, through the doorway, and into the living room.",
        "Walk to the door, take the lever, push the door wide open, and stand out of the lane so your partners and their table can pass.",
        "Take the two legs at your end of the table, one in each hand, lift level with your partner, and lead it backwards through the open doorway until its four legs stand on the green squares.",
        "Take the two legs at your end of the table, one in each hand, lift level with your partner, and follow them forwards through the open doorway until its four legs stand on the green squares.",
    ),
    # The carry is straight ahead for A; B and C walk it backwards.
    "Isaac-BigTable-G1x3-v0": (
        "All three robots lift the round table together and carry it to the marked spot.",
        "Take the edge of the table in front of you with both hands, lift on the count, and walk it straight forward to the green disc, keeping it level.",
        "Take the edge of the table in front of you with both hands, lift on the count, and walk it backwards and to your right to the green disc, keeping it level.",
        "Take the edge of the table in front of you with both hands, lift on the count, and walk it backwards and to your left to the green disc, keeping it level.",
    ),
}
"""env id -> (shared sentence, then one sentence per robot in scene order)."""


def sentences(gym_id: str) -> tuple[str, ...] | None:
    """``(shared, robot_a, robot_b, ...)`` for ``gym_id``, or None if it has none."""
    return TASK_TEXT.get(gym_id)
