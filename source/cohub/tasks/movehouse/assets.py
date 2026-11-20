"""MoveHouse's door, its console table, and their dimensions.

Everything is in the task frame:

* the doorway plane is ``x = 0``, the bedroom is -X and the living room +X,
* the carry runs along +X and the table's long axis lies along the carry,
* the door's hinge is at ``y = -0.44`` and its lever at ``y = +0.33``.

The room is placed by ``scenesmith_room.py``.
"""

from pathlib import Path

from isaaclab.assets import AssetBaseCfg

from cohub.tasks._base.props import (
    DECAL_LIFT,
    DOOR_FILL_COLOUR,
    GOAL_PAINT_COLOUR,
    GOAL_PAINT_EMISSIVE,
    GOAL_PAINT_THICKNESS,
    box_cfg,
    carried_table_cfg,
)

# ------------------------------------------------------------- the doorway
WALL_THICKNESS = 0.100
"""Thickness of the partition the door hangs in [m]."""

# ------------------------------------------------------------------ the door
# Joint and body names of the DoorMan asset, read by ``mdp.DoorOpeningAction``.
DOOR_HINGE_JOINT = "hinge_joint"
DOOR_HANDLE_JOINT = "handle_joint"
DOOR_HANDLE_BODY = "door_handle"

# Defaults of the action term [rad, m]. The env cfg overrides the hinge threshold with DOOR_CLEAR_ANGLE.
DOOR_HINGE_OPEN_ANGLE = 1.0
DOOR_HANDLE_OPEN_ANGLE = 0.12
DOOR_HANDLE_GRASP_OFFSET = (-0.15, -0.068, 0.02)

PASSAGE_SUCCESS_X = 0.60
"""Task x past which the leading carrier is through the doorway [m]. Read by the action term's stage 5."""

DOOR_CLEAR_ANGLE = 1.45
"""How far the leaf has to swing before the table fits through [rad]."""

# ---------------------------------------------------------------- the table
# Sizes of the room's ``hallway_console_table_0`` [m]. The carried table is a scaled copy.
SOURCE_LENGTH = 0.882
SOURCE_TOP_Z = 0.780
SOURCE_TOP_THICKNESS = 0.020

TABLE_SCALE = 1.15
"""Scale of the carried table against the room's console table, along its length and height."""

TABLE_LENGTH = round(SOURCE_LENGTH * TABLE_SCALE, 3)
TOP_Z = round(SOURCE_TOP_Z * TABLE_SCALE, 3)
TOP_THICKNESS = round(SOURCE_TOP_THICKNESS * TABLE_SCALE, 3)
"""The scaled table [m]: length, top face height and slab thickness."""

TABLE_DEPTH = 0.400
"""The table's depth, across the carry [m]. Not scaled."""

LEG_INSET = 0.022
"""Each leg's centre from the top's two edges [m]."""

LEG_X = round(TABLE_LENGTH / 2 - LEG_INSET, 4)
LEG_Y = round(TABLE_DEPTH / 2 - LEG_INSET, 4)
"""The four legs' centres from the table's centre [m]: (+/-0.485, +/-0.178)."""

TABLE_MASS = 4.0
"""Mass of the carried table [kg]."""

REST_FINGER_REACH = 0.361
"""How far in front of its spawn pelvis a resting G1's fingers are [m]."""

SPAWN_HAND_CLEAR = 0.18
"""Air between the resting fingers and the top's short edge at spawn [m]."""

# --------------------------------------------------------------- geometry
STANDING_Z = round(TOP_Z - TOP_THICKNESS / 2, 4)
"""The table's origin height with it standing on its legs [m]. The origin is the top slab's centre."""

DROP_MIN_HEIGHT = round(STANDING_Z - 0.18, 4)
"""Below this origin height the table has been dropped [m]."""

# --------------------------------------------------------------- the layout
STANDOFF = round(TABLE_LENGTH / 2 + REST_FINGER_REACH + SPAWN_HAND_CLEAR, 3)
"""Each carrier's pelvis from the table's centre, along the carry [m]."""

SPAWN_LANE_Y = -0.15
"""How far off the doorway's axis the carry starts [m]."""

TABLE_SPAWN = (-2.05, SPAWN_LANE_Y, STANDING_Z)
"""Where the table starts, in the task frame [m]."""

TABLE_SPAWN_JITTER = (0.05, 0.05)
TABLE_SPAWN_POSE_RANGE = {
    "x": (-TABLE_SPAWN_JITTER[0], TABLE_SPAWN_JITTER[0]),
    "y": (-TABLE_SPAWN_JITTER[1], TABLE_SPAWN_JITTER[1]),
}
"""Spawn offsets from ``TABLE_SPAWN`` [m]. Offsets, not positions. No yaw."""

GOAL_AT = (2.60, 1.378)
"""Where the table's centre has to end up, in the task frame [m]."""

CARRY_DISTANCE = round(
    ((GOAL_AT[0] - TABLE_SPAWN[0]) ** 2 + (GOAL_AT[1] - TABLE_SPAWN[1]) ** 2) ** 0.5, 4
)
"""Straight-line distance from the spawn to the goal [m]."""

# ------------------------------------------------------------- the predicate

HEIGHT_TOLERANCE = 0.04
"""Allowed error in the table's origin height [m]."""

MAX_TILT_DEG = 15.0
MAX_LINEAR_SPEED = 0.05
MAX_ANGULAR_SPEED = 0.15
RELEASE_SETTLE_TIME = 0.20
"""Placement limits: tilt [deg], linear speed [m/s], angular speed [rad/s] and time released [s]."""

LIFT_MARGIN_M = 0.03
"""How far the lowest foot must be off the floor, with both carriers holding, for the lift to count [m]."""

DOOR_CLEAR_X = round(WALL_THICKNESS / 2, 3)
"""Task x of the living-room face of the doorway wall [m]."""

CARRY_GOAL_RADIUS = 0.15
CARRY_FOOT_CLEARANCE = 0.01
"""Distance to ``GOAL_AT`` and foot clearance for the carry to the goal to count [m]."""

# ------------------------------------------------------------------ the cfgs
TABLE_CFG = carried_table_cfg(Path(__file__).parent / "data" / "movehouse_table.usda", TABLE_SPAWN, TABLE_MASS)
OBJECT_CFG = TABLE_CFG
"""The table, under the name the scene cfg and the recorders use."""


def _door_cfg():
    """The door: the baked ``data/movehouse_door.usda``."""
    from pathlib import Path

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

    usd = Path(__file__).parent / "data" / "movehouse_door.usda"
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Door",
        articulation_root_prim_path="/root",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={DOOR_HINGE_JOINT: 0.0, DOOR_HANDLE_JOINT: 0.0},
            joint_vel={DOOR_HINGE_JOINT: 0.0, DOOR_HANDLE_JOINT: 0.0},
        ),
        spawn=UsdFileCfg(
            usd_path=str(usd),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        actuators={
            "door_hinge": ImplicitActuatorCfg(
                joint_names_expr=[DOOR_HINGE_JOINT],
                effort_limit_sim=400.0,
                velocity_limit_sim=3.0,
                stiffness=0.0,
                damping=4.0,
            ),
            "door_handle": ImplicitActuatorCfg(
                joint_names_expr=[DOOR_HANDLE_JOINT],
                effort_limit_sim=12.0,
                velocity_limit_sim=3.0,
                stiffness=5.0,
                damping=0.5,
            ),
        },
    )


DOOR_CFG = _door_cfg()

# ------------------------------------------------------- closing the house
# Panel that fills the front door, the only opening of the house with nothing behind it.
FRONT_DOOR_CFG: AssetBaseCfg = box_cfg(
    "FrontDoor", (0.93, 0.03, 2.08), (1.006, 3.293, 1.04), DOOR_FILL_COLOUR
)


# ------------------------------------------------------------- the foot marks
# Where the table's four feet have to land. The pads are what is scored.
FOOT_MARK_SIZE = 0.100
"""Side of one square pad [m]. A foot scores when its centre is within half of it on both axes."""

FOOT_MARK_AT = tuple(
    (round(GOAL_AT[0] + sx * LEG_X, 4), round(GOAL_AT[1] + sy * LEG_Y, 4))
    for sx in (-1.0, 1.0)
    for sy in (-1.0, 1.0)
)
"""The four pads' centres in the task frame [m]."""

LEG_FEET = tuple(
    (sx * LEG_X, sy * LEG_Y, round(-STANDING_Z, 4))
    for sx in (-1.0, 1.0)
    for sy in (-1.0, 1.0)
)
"""Each foot in the table's own frame [m]."""

FOOT_MARK_CFGS: tuple[AssetBaseCfg, ...] = tuple(
    box_cfg(
        f"FootMark{index}",
        (FOOT_MARK_SIZE, FOOT_MARK_SIZE, GOAL_PAINT_THICKNESS),
        (x, y, GOAL_PAINT_THICKNESS / 2 + DECAL_LIFT),
        GOAL_PAINT_COLOUR,
        collision=False,
        emissive=GOAL_PAINT_EMISSIVE,
    )
    for index, (x, y) in enumerate(FOOT_MARK_AT)
)
"""Four collision-free pads, one cfg per prim."""
