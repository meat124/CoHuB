"""CoCarry's laundry basket, its shelves and the room fittings.

The basket's long axis is local Y. The task rotates it onto world X, the
robot-robot axis, so each robot works at its own end. It has no handles.
Geometry here must match ``data/laundry_basket.usdc``. The visual mesh is
a CC BY 4.0 scan, see THIRD_PARTY_ASSETS.md.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from cohub.tasks._base.props import (
    DECAL_LIFT,
    GOAL_PAINT_COLOUR,
    GOAL_PAINT_EMISSIVE,
    GOAL_PAINT_THICKNESS,
    box_cfg,
)

BOARD_MASS = 3.0
"""Basket mass [kg]."""

BASE_HALF_SIZE = (0.160, 0.275)
"""Half-size (x, y) of the basket's base footprint [m]."""

ROBOT_STANDOFF_X = 0.52
"""Each robot's pelvis from the basket's centre, along world X [m]."""

STAND_TOP_Z = 0.324
"""Top of the shelf [m], where the basket's base rests."""
BOARD_SPAWN_HEIGHT = STAND_TOP_Z
"""Spawn height of the basket's origin, which is its base [m]."""

TASK_YAW_ROT = (0.0, 0.0, -0.7071, 0.7071)
"""Clockwise quarter turn, (x, y, z, w). Puts the basket's long axis along world X."""

PICKUP_Y = 0.50
"""World-Y coordinate of the pickup shelf and basket centre [m]."""

CARRY_DISTANCE = 2.10
"""Distance carried from the pickup shelf to the goal shelf, along world -Y [m]."""

_DATA = Path(__file__).parent / "data"
_BASKET_USD_PATH = str(_DATA / "laundry_basket.usdc")

BOARD_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Board",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, PICKUP_Y, BOARD_SPAWN_HEIGHT),
        rot=TASK_YAW_ROT,
    ),
    spawn=UsdFileCfg(
        usd_path=_BASKET_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            # A sleeping body ignores the wrench the grasp applies.
            sleep_threshold=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=BOARD_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
    ),
)


# Doorway fills
ROOM_ROUGHNESS = 0.5
"""Roughness of every surface this task adds to the room."""

DOORWAY_FILL_COLOUR = (0.882, 0.886, 0.894)
"""Wall colour, for the doorway fills."""

DOORWAY_SOUTH_CFG: AssetBaseCfg = box_cfg(
    "DoorwaySouth", (1.18, 0.03, 2.30), (-1.189, -2.975, 1.15), DOORWAY_FILL_COLOUR,
    roughness=ROOM_ROUGHNESS,
)
"""Fill for the south doorway."""

DOORWAY_WEST_CFG: AssetBaseCfg = box_cfg(
    "DoorwayWest", (0.03, 0.98, 2.30), (-2.325, -1.004, 1.15), DOORWAY_FILL_COLOUR,
    roughness=ROOM_ROUGHNESS,
)
"""Fill for the west doorway."""

# Washing machines. They replace the room's own appliances.
WASHING_MACHINE_SIZE = (0.600, 0.601, 0.873)
"""One machine, (x, y, z) in the task frame [m]."""

WASHING_MACHINE_PITCH = round(WASHING_MACHINE_SIZE[0] + 0.06, 4)
"""Centre to centre along the wall [m]."""

WASHING_MACHINE_Y = -2.6355
"""The south row's centreline in task y [m]."""

WASHING_MACHINE_COUNT = 6
"""Number of machines in the south row."""

WASHING_MACHINE_AT = tuple(
    (round(index * WASHING_MACHINE_PITCH, 4), WASHING_MACHINE_Y)
    for index in range(WASHING_MACHINE_COUNT)
)
"""Each machine's centre (x, y) [m]. The first is on the shelves' axis at x = 0."""

DRYER_BAY_Y = 2.6355
"""The north pair's centreline in task y [m]."""

DRYER_BAY_CENTRE_X = 3.4473
"""Midpoint of the north pair in task x [m]."""

DRYER_BAY_AT = tuple(
    (round(DRYER_BAY_CENTRE_X + (index - 0.5) * WASHING_MACHINE_PITCH, 4), DRYER_BAY_Y)
    for index in range(2)
)
"""Each machine's centre (x, y) on the north wall [m]."""

DRYER_BAY_ROT = (0.0, 0.0, 0.7071, 0.7071)
"""Counter-clockwise quarter turn, (x, y, z, w). The north wall's machines face -Y."""

_WASHING_MACHINE_SCALE = (0.9024, 0.9967, 1.0271)
"""Scale that fits the vendored mesh to ``WASHING_MACHINE_SIZE``."""


def _machine_cfgs(
    prim_name: str,
    places: tuple[tuple[float, float], ...],
    rot: tuple[float, float, float, float],
) -> tuple[tuple[AssetBaseCfg, AssetBaseCfg], ...]:
    """Each machine as (mesh, collider).

    The vendored mesh is visual only. An invisible box is the collider.
    """
    out = []
    for index, (x, y) in enumerate(places):
        mesh = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/%s%d" % (prim_name, index),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, 0.0), rot=rot),
            spawn=UsdFileCfg(
                usd_path=str(_DATA / "washing_machine.usdc"), scale=_WASHING_MACHINE_SCALE
            ),
        )
        collider = box_cfg(
            f"{prim_name}Collider{index}",
            WASHING_MACHINE_SIZE,
            (x, y, WASHING_MACHINE_SIZE[2] / 2),
            (0.0, 0.0, 0.0),
            visible=False,
        )
        out.append((mesh, collider))
    return tuple(out)


WASHING_MACHINE_CFGS = _machine_cfgs("WashingMachine", WASHING_MACHINE_AT, TASK_YAW_ROT)
"""``(mesh, collider)`` per machine, in ``WASHING_MACHINE_AT`` order."""

DRYER_BAY_CFGS = _machine_cfgs("DryerBay", DRYER_BAY_AT, DRYER_BAY_ROT)
"""``(mesh, collider)`` per machine on the north wall, in ``DRYER_BAY_AT`` order."""

# Shelf

STAND_TOP_SIZE = (0.62, 0.56, 0.06)
"""The shelf's top slab, (x, y, z) [m]."""

STAND_COLUMN_SIZE = (0.44, 0.30, STAND_TOP_Z - STAND_TOP_SIZE[2])
"""The column under the slab, (x, y, z) [m]."""

STAND_BASE_SIZE = (0.62, 0.42, 0.03)
"""The floor plate, (x, y, z) [m]."""


SHELF_TOP_COLOUR = (0.84, 0.84, 0.84)
SHELF_FRAME_COLOUR = (0.565, 0.56, 0.56)
SHELF_ROUGHNESS = ROOM_ROUGHNESS
"""The shelves' colours and roughness."""


def _shelf_cfgs(prefix: str, y: float) -> tuple[AssetBaseCfg, ...]:
    """One shelf, as its three boxes, centred on the carry line at ``y``."""
    top = box_cfg(
        f"{prefix}Top",
        STAND_TOP_SIZE,
        (0.0, y, STAND_TOP_Z - STAND_TOP_SIZE[2] / 2),
        SHELF_TOP_COLOUR,
        roughness=SHELF_ROUGHNESS,
    )
    column = box_cfg(
        f"{prefix}Column",
        STAND_COLUMN_SIZE,
        (0.0, y, STAND_COLUMN_SIZE[2] / 2),
        SHELF_FRAME_COLOUR,
        roughness=SHELF_ROUGHNESS,
    )
    base = box_cfg(
        f"{prefix}Base",
        STAND_BASE_SIZE,
        (0.0, y, STAND_BASE_SIZE[2] / 2),
        SHELF_FRAME_COLOUR,
        roughness=SHELF_ROUGHNESS,
    )
    return top, column, base


(
    STAND_TOP_CFG,
    STAND_COLUMN_CFG,
    STAND_BASE_CFG,
) = _shelf_cfgs("Stand", PICKUP_Y)


BOARD_SPAWN_JITTER = (0.05, 0.05)
"""Half-width of the square the basket's start is drawn from, (x, y) [m]."""

BOARD_SPAWN_POSE_RANGE = {
    "x": (-BOARD_SPAWN_JITTER[0], BOARD_SPAWN_JITTER[0]),
    "y": (-BOARD_SPAWN_JITTER[1], BOARD_SPAWN_JITTER[1]),
}
"""Pose range for ``reset_root_state_uniform``. Position only."""


# Goal
GOAL_Y = PICKUP_Y - CARRY_DISTANCE
"""World-Y coordinate of the goal shelf [m]."""
GOAL_Y_TOLERANCE = 0.15
GOAL_TOLERANCE_X = 0.15
GOAL_BOARD_HEIGHT = BOARD_SPAWN_HEIGHT
GOAL_HEIGHT_TOLERANCE = 0.06

CARRY_BASE_CLEARANCE = 0.01
"""How far the lowest base corner must be above the shelf top for the basket to
count as carried into the goal box [m]."""

(
    GOAL_STAND_TOP_CFG,
    GOAL_STAND_COLUMN_CFG,
    GOAL_STAND_BASE_CFG,
) = _shelf_cfgs("GoalStand", GOAL_Y)



GOAL_MARKER_CFG = box_cfg(
    "GoalMarker",
    (2 * GOAL_TOLERANCE_X, 2 * GOAL_Y_TOLERANCE, GOAL_PAINT_THICKNESS),
    (0.0, GOAL_Y, STAND_TOP_Z + DECAL_LIFT - GOAL_PAINT_THICKNESS / 2),
    GOAL_PAINT_COLOUR,
    collision=False,
    emissive=GOAL_PAINT_EMISSIVE,
)
"""The green target on the goal shelf's top face, sized as the box
``CarrySuccess`` scores. Visual only."""
