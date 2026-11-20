"""The key names CoHuB's LeRobot export and every policy's data path share.

Pure data, no third-party imports. The layout follows NVIDIA's G1 embodiment
(``unitree_g1_full_body_with_waist_height_nav_cmd``), except that every key
carries a ``robot_a_`` / ``robot_b_`` prefix.
"""

from __future__ import annotations

ROBOTS = ("robot_a", "robot_b")
"""The robots of a two-robot task."""

ROBOTS3 = ("robot_a", "robot_b", "robot_c")
"""Every robot name the platform knows, in slice order."""

ALL_ROBOTS = ROBOTS3
"""Every robot name the per-robot maps are built for."""

TELEOP_ACTION_DIM = 32
"""One robot's teleop action slice: wrist poses 7+7, hand joints 14, locomotion 4.
A robot's slice starts at ``ROBOTS3.index(name) * TELEOP_ACTION_DIM``."""


def robots(count: int = 2) -> tuple[str, ...]:
    """The robot names of a task with ``count`` of them."""
    if count not in (2, 3):
        raise ValueError(f"CoHuB tasks have two or three robots, not {count}")
    return ALL_ROBOTS[:count]


SHARED = "shared"
"""The one policy that drives either robot. Each dataset row is one robot,
with unprefixed keys; the instruction says which robot it is."""

# -- joint names ------------------------------------------------------------
# The G1's joints per group, in the two orders the benchmark depends on.

LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
WAIST = ("waist_yaw", "waist_roll", "waist_pitch")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
HAND = (
    "hand_thumb_0",
    "hand_thumb_1",
    "hand_thumb_2",
    "hand_middle_0",
    "hand_middle_1",
    "hand_index_0",
    "hand_index_1",
)
"""Per-group joint order, as templates: both sides use the same names."""


def left(group) -> list[str]:
    """``left_<joint>_joint`` for each name in ``group``."""
    return [f"left_{joint}_joint" for joint in group]


def right(group) -> list[str]:
    """``right_<joint>_joint`` for each name in ``group``."""
    return [f"right_{joint}_joint" for joint in group]


def unsided(group) -> list[str]:
    """``<joint>_joint`` for each name in ``group``."""
    return [f"{joint}_joint" for joint in group]


URDF_GROUPS = (
    ("left_leg", left(LEG)),
    ("right_leg", right(LEG)),
    ("waist", unsided(WAIST)),
    ("left_arm", left(ARM)),
    ("left_hand", left(HAND)),
    ("right_arm", right(ARM)),
    ("right_hand", right(HAND)),
)
"""(group, joint names in order). Concatenated, the 43 names of one G1 in
``observation.state`` order, which is the G1 URDF's own order. Do not sort."""

ACTION_GROUPS = (
    ("left_arm", left(ARM)),
    ("right_arm", right(ARM)),
    ("left_hand", left(HAND)),
    ("right_hand", right(HAND)),
    ("waist", unsided(WAIST)),
)
"""The 31 joints an upper-body action drives, in GR00T's action-column order.
Legs are absent: locomotion is commanded as a velocity and a height."""


def urdf_order() -> list[str]:
    """All 43 joint names of one robot, in ``observation.state`` order."""
    return [name for _, names in URDF_GROUPS for name in names]


def action_order() -> list[str]:
    """The 31 upper-body joint names, in action-column order."""
    return [name for _, names in ACTION_GROUPS for name in names]


def hand_order() -> list[str]:
    """The 14 hand joints as they sit inside :func:`action_order`."""
    return left(HAND) + right(HAND)


# -- state ------------------------------------------------------------------

JOINT_GROUPS = tuple(group for group, _ in URDF_GROUPS)
"""The seven groups of ``observation.state``, in the order they are stored."""

JOINT_GROUP_WIDTHS = {group: len(names) for group, names in URDF_GROUPS}
"""Width of each group. Sums to 43, one robot's joint count."""

JOINTS_PER_ROBOT = sum(JOINT_GROUP_WIDTHS.values())
assert JOINTS_PER_ROBOT == 43

STATE_JOINT_KEYS = {
    robot: [f"{robot}_{group}" for group in JOINT_GROUPS] for robot in ALL_ROBOTS
}
STATE_JOINT_KEYS[SHARED] = list(JOINT_GROUPS)

# -- action -----------------------------------------------------------------

ACTION_JOINT_GROUPS = tuple(group for group, _ in ACTION_GROUPS)
"""The joint groups a policy commands, in GR00T's own listing order."""

ACTION_COMMAND_KEYS = ("base_height_command", "navigate_command")

ACTION_KEYS = {
    robot: [f"{robot}_{name}" for name in (*ACTION_JOINT_GROUPS, *ACTION_COMMAND_KEYS)]
    for robot in ALL_ROBOTS
}
ACTION_KEYS[SHARED] = [*ACTION_JOINT_GROUPS, *ACTION_COMMAND_KEYS]


ACTION_DIMS_PER_ROBOT = len(action_order()) + len(("base_height",)) + len(("vx", "vy", "vyaw"))
"""31 joint targets, base_height(1), navigate(3)."""
assert ACTION_DIMS_PER_ROBOT == 35

# -- cameras, video, language ----------------------------------------------

CAMERAS = (
    # (dataset key, scene sensor, demo mp4 stem or None, records depth)
    ("ego_a", "ego_cam_a", "ego_robot_a", True),
    ("ego_b", "ego_cam_b", "ego_robot_b", True),
    ("ego_c", "ego_cam_c", "ego_robot_c", True),
    ("scene", "scene_cam", "scene", False),
    ("scene_2", "scene_cam_2", None, False),
)
"""Every camera a task may build. Readers skip a sensor the scene does not have."""

CAMERA_SENSORS = tuple((key, sensor) for key, sensor, _, _ in CAMERAS)
"""Dataset camera key -> scene sensor name."""

DEMO_VIDEOS = tuple((key, f"{stem}.mp4") for key, _, stem, _ in CAMERAS if stem)
"""Dataset camera key -> demo mp4 file name. Depth streams add a ``_depth`` suffix."""

DEPTH_CAMERAS = tuple(key for key, _, _, depth in CAMERAS if depth)
"""Which cameras also record depth."""

VIDEO_KEYS = {"robot_a": ["ego_a"], "robot_b": ["ego_b"], "robot_c": ["ego_c"], SHARED: ["ego"]}
"""Each robot's own head camera. The shared policy's single camera is ``ego``."""

VIDEO_KEYS_DUO = ["ego_a", "ego_b", "scene"]
"""Both heads plus the fixed third-person camera, for a policy driving both."""

LANGUAGE_KEYS = {
    "duo": "annotation.human.task_description",
    "robot_a": "annotation.human.task_description_robot_a",
    "robot_b": "annotation.human.task_description_robot_b",
    "robot_c": "annotation.human.task_description_robot_c",
    # One column, holding whichever robot's sentence that row is.
    SHARED: "annotation.human.task_description",
}
"""One instruction for the pair, and one per robot."""

# -- horizons ---------------------------------------------------------------

ACTION_HORIZON = 50
"""Action chunk length in steps: 1.0 s at CoHuB's 50 Hz. Same as upstream's G1.

Isaac-GR00T's ``scripts/launch_finetune.py`` rejects chunks above 40, so train
through ``gr00t/experiment/launch_train.py``, or set this to 40 and regenerate
``meta/relative_stats.json``.
"""

ACTION_HORIZON_FINETUNE_SAFE = 40
"""The largest chunk ``launch_finetune.py`` accepts without an override."""


# -- assembled key sets -----------------------------------------------------


def state_keys(robot: str | None = None) -> list[str]:
    """Joint groups only: 86 dims for the pair, 43 for one robot.

    Args:
        robot: ``"robot_a"``/``"robot_b"`` for that robot alone, ``None`` for both.
    """
    chosen = ROBOTS if robot is None else (robot,)
    return [k for r in chosen for k in STATE_JOINT_KEYS[r]]


def action_keys(robot: str | None = None) -> list[str]:
    """35 dims per robot, in upstream's listing order. 70 for the pair."""
    chosen = ROBOTS if robot is None else (robot,)
    return [k for r in chosen for k in ACTION_KEYS[r]]


def video_keys(robot: str | None = None) -> list[str]:
    return VIDEO_KEYS_DUO if robot is None else list(VIDEO_KEYS[robot])


def language_key(robot: str | None = None) -> str:
    return LANGUAGE_KEYS["duo" if robot is None else robot]
