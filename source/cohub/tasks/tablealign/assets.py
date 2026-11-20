"""TableAlign's table, the chair it is squared to and the layout constants.

Task frame: the robots face each other across world X, the table's long axis
is world X at spawn and world Y at the goal, and the carry runs along world
-Y. The chair belongs to the downloaded room. Its pose is given here as
constants in the task frame.
"""

from pathlib import Path

from isaaclab.assets import AssetBaseCfg

from cohub.tasks._base.props import (
    DECAL_LIFT,
    DOOR_FILL_COLOUR,
    GOAL_PAINT_COLOUR,
    GOAL_PAINT_EMISSIVE,
    GOAL_PAINT_THICKNESS,
    WINDOW_FILL_COLOUR,
    box_cfg,
    carried_table_cfg,
)

# Table
TOP_THICKNESS = 0.044
TOP_Z = 0.702
"""Top slab thickness, and the height of its upper face [m]."""

TABLE_MASS = 4.0
"""Table mass [kg]."""

FRAME_X = 0.632
FRAME_DEPTH = 0.071
FRAME_WIDTH = 0.626
POST_WIDTH = 0.045
POST_Y = 0.292
"""The frame at each short end [m]. ``FRAME_X`` is its centre along the long
axis, ``FRAME_DEPTH`` its depth along it, ``FRAME_WIDTH`` its length across it,
and ``POST_Y`` the posts' offset across the table."""

# Geometry
STANDING_Z = round(TOP_Z - TOP_THICKNESS / 2, 4)
"""Height of the table's origin, the centre of the top slab, when the table
stands on its legs [m]."""

# Layout
STANDOFF = 0.931
"""Each robot's pelvis from the table's centre along the long axis [m]."""

TABLE_SPAWN = (0.0, 0.0, STANDING_Z)
"""Where the table starts: the task origin, standing on its legs [m]."""

TABLE_SPAWN_JITTER = (0.05, 0.05)
"""Half-width of the square the table's start is drawn from, (x, y) [m]."""

TABLE_SPAWN_POSE_RANGE = {
    "x": (-TABLE_SPAWN_JITTER[0], TABLE_SPAWN_JITTER[0]),
    "y": (-TABLE_SPAWN_JITTER[1], TABLE_SPAWN_JITTER[1]),
}

CARRY_DISTANCE = 2.10
"""Spawn to goal along world -Y [m]."""

DROP_MIN_HEIGHT = 0.50
"""Origin height below which the table counts as dropped [m]."""

# Chair, in the task frame. These move if the room is placed differently.
CHAIR_AT = (0.40, -2.10)
"""The chair's origin in the task frame [m]."""

CHAIR_FORWARD = (-1.0, 0.0)
"""The direction the chair faces, in the task frame."""

CHAIR_TO_TABLE_CENTRE = 0.40
"""Chair origin to table centre at the goal, along the chair's forward [m]."""

GOAL_AT = (
    round(CHAIR_AT[0] + CHAIR_FORWARD[0] * CHAIR_TO_TABLE_CENTRE, 4),
    round(CHAIR_AT[1] + CHAIR_FORWARD[1] * CHAIR_TO_TABLE_CENTRE, 4),
)
"""Where the table's centre has to end up, in the task frame [m]."""

# Success thresholds
FORWARD_BAND = (0.335, 0.465)
"""Allowed chair-to-table distance along the chair's forward, (min, max) [m]."""

LATERAL_TOLERANCE = 0.13
"""Allowed offset across the chair's forward [m]."""

HEIGHT_TOLERANCE = 0.04
"""Allowed error in the table's origin height [m]."""

SQUARE_TOLERANCE_DEG = 4.0
"""How far the table's long axis may be off square to the chair [deg]."""

MAX_TILT_DEG = 15.0
MAX_LINEAR_SPEED = 0.05
MAX_ANGULAR_SPEED = 0.15
RELEASE_SETTLE_TIME = 0.20
"""Placement limits: tilt [deg], linear speed [m/s], angular speed [rad/s] and
seconds since release."""

CARRY_FOOT_CLEARANCE = 0.01
"""How far the lowest point of both foot bars must be off the floor for the
table to count as carried into the goal box [m]."""

# Cfgs
TABLE_CFG = carried_table_cfg(Path(__file__).parent / "data" / "align_table.usda", TABLE_SPAWN, TABLE_MASS)
OBJECT_CFG = TABLE_CFG
"""Alias of ``TABLE_CFG``."""

# Foot marks: where the table's feet have to land.
FOOT_MARK_GROWTH = 2.0
"""Size of each foot mark relative to the post that lands on it."""

FOOT_MARK_SIZE = (
    round(POST_WIDTH * FOOT_MARK_GROWTH, 4),
    round(FRAME_DEPTH * FOOT_MARK_GROWTH, 4),
)
"""One foot mark in world (x, y) at the goal pose [m]."""

FOOT_MARK_AT = tuple(
    (round(GOAL_AT[0] + sy * POST_Y, 4), round(GOAL_AT[1] + sx * FRAME_X, 4))
    for sx in (-1.0, 1.0)
    for sy in (-1.0, 1.0)
)
"""The four foot marks' centres in the task frame [m]."""

FOOT_MARK_CFGS: tuple[AssetBaseCfg, ...] = tuple(
    box_cfg(
        f"FootMark{index}",
        (FOOT_MARK_SIZE[0], FOOT_MARK_SIZE[1], GOAL_PAINT_THICKNESS),
        (x, y, GOAL_PAINT_THICKNESS / 2 + DECAL_LIFT),
        GOAL_PAINT_COLOUR,
        collision=False,
        emissive=GOAL_PAINT_EMISSIVE,
    )
    for index, (x, y) in enumerate(FOOT_MARK_AT)
)
"""The four foot marks. Visual only."""

# Fills for the room's openings
DOOR_COLOUR = DOOR_FILL_COLOUR
"""Fill colour for the doorways."""

WINDOW_COLOUR = WINDOW_FILL_COLOUR
"""Fill colour for the east window."""

DOOR_SOUTH_CFG: AssetBaseCfg = box_cfg(
    "DoorSouth", (0.06, 0.98, 2.08), (-2.12, 0.10, 1.04), DOOR_COLOUR
)
"""Fill for the south doorway."""

DOOR_WEST_CFG: AssetBaseCfg = box_cfg(
    "DoorWest", (0.98, 0.06, 2.08), (0.94, 3.70, 1.04), DOOR_COLOUR
)
"""Fill for the west doorway."""

WINDOW_EAST_CFG: AssetBaseCfg = box_cfg(
    "WindowEast", (0.86, 0.06, 1.06), (1.33, -4.30, 1.75), WINDOW_COLOUR
)
"""Fill for the east window."""
