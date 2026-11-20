"""CartService props and layout constants.

Robot A pushes a loaded service cart down the aisle and parks it beside the
goal table. Robot B lifts the wine bottle out of the cart's collar and stands
it on the table. Only the bottle is scored. Positions are in the task frame.
"""

import functools
import math

from pathlib import Path

import isaaclab.sim as sim_utils

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from cohub.tasks._base.props import (
    DECAL_LIFT,
    GOAL_PAINT_COLOUR,
    GOAL_PAINT_EMISSIVE,
    GOAL_PAINT_THICKNESS,
    IDENTITY_ROT,
    box_cfg,
    manipuland_collision_props,
    manipuland_rigid_props,
)

_DATA = Path(__file__).resolve().parent / "data"

# Room measurements in the task frame, and the G1's reach

TABLE_TOP_Z = 0.813
"""Top of the room's dining tables [m]. Keep the millimetre: the goal decal is placed against it."""

TABLE_HALF = 0.425
"""Half the side of a dining table [m]."""

BODY_AHEAD_OF_PELVIS = 0.165
"""How far a G1's body reaches in front of its pelvis, arms excluded [m]."""

EDGE_CLEARANCE = 0.035
"""Gap between the robot's body and the edge it stands at [m]."""

GOAL_TABLE_AT = (1.405, 0.0)
"""Centre of the goal table, ``dining_room_dining_table_2``, in the task frame [m]."""

# Layout

CART_LANE = 0.0
"""The x of the lane the cart is pushed down [m]. The task's origin is the parked cart."""

CART_START_Y = 2.415
"""The cart's y at the start of the episode [m]."""

CART_PARKED_Y = GOAL_TABLE_AT[1]
"""The cart's y when parked beside the goal table [m]."""

CART_PARKED = (CART_LANE, CART_PARKED_Y)
CART_START = (CART_LANE, CART_START_Y)

PUSH_LENGTH = round(CART_START_Y - CART_PARKED_Y, 3)
"""Distance Robot A pushes the cart [m]."""

# Service cart

CART_DECK_X = 0.55
"""The cart's width, across the push direction [m]."""

CART_DECK_Y = 0.85
"""The cart's length, along the push direction [m]."""

CART_TOP_Z = 0.75
"""Height of the top deck's surface [m]."""

CART_POST_SECTION = 0.030
CART_POST_Y = round(CART_DECK_Y / 2 - CART_POST_SECTION / 2 - 0.010, 4)
"""Section of the four corner posts, and their offset from the cart's centre along the push direction [m]."""

CART_BAR_Y = CART_POST_Y
"""The push bar's y in the cart's frame [m], over the rear posts."""

CART_MASS = 12.0
"""Mass of the cart without its load [kg]."""

CART_ASSET = "service_cart.usda"

# Wine bottle, the only scored object

BOTTLE_SCALE = 1.3
BOTTLE_HEIGHT = 0.286
BOTTLE_HALF_HEIGHT = round(BOTTLE_HEIGHT / 2, 4)
BOTTLE_MASS = 1.5
"""The wine bottle as spawned: scale, height and half height [m], mass [kg]. The scale must be uniform."""

BOTTLE_MAX_TILT_DEG = 15.0
"""Largest tilt at which the bottle counts as upright [deg]."""

BOTTLE_TOPPLED_DEG = 60.0
"""Tilt past which a released bottle counts as toppled [deg]."""

BOTTLE_ASSET = "wine_bottle.usdc"

# Plate and cheese board: cargo, not scored. Both assets are CC BY 4.0, see
# THIRD_PARTY_ASSETS.md.

PLATE_ASSET = "sandwich_plate.usdc"
PLATE_MASS = 0.5
"""The plate of sandwiches: its USD file and mass [kg]."""

BOARD_ASSET = "cheese_board.usdc"
BOARD_MASS = 0.7
"""The cheese board: its USD file and mass [kg]."""

# Positions on the deck

WORK_STRIP_Y = -0.29
"""The y of the row the bottle and the plate stand in, in the cart's frame [m]."""

COLLAR_AT = (0.14, WORK_STRIP_Y)
PLATE_AT = (-0.14, WORK_STRIP_Y)
BOARD_AT = (0.00, -0.02)
"""Positions of the collar, the plate and the board on the deck, in the cart's frame [m]."""

# Robots

ROBOT_A_SPAWN_GAP = 0.55
"""Distance from Robot A's pelvis to the push bar at spawn [m].
Larger than the 0.36 m it pushes from, so the hands do not spawn inside the bar."""

ROBOT_A_START = (CART_LANE, round(CART_START_Y + CART_BAR_Y + ROBOT_A_SPAWN_GAP, 4))
"""Robot A's spawn, (x, y) [m]: behind the cart, facing -Y."""

ROBOT_B_START = (CART_LANE, -1.80)
"""Robot B's spawn, (x, y) [m]: at the far end of the aisle, facing +Y."""

ROBOT_B_PICK = (
    CART_LANE,
    round(CART_PARKED_Y - CART_DECK_Y / 2 - BODY_AHEAD_OF_PELVIS - EDGE_CLEARANCE, 4),
)
"""Where Robot B stands to take the bottle off the parked cart, (x, y) [m]."""

ROBOT_B_SERVE = (
    round(GOAL_TABLE_AT[0] - TABLE_HALF - BODY_AHEAD_OF_PELVIS - EDGE_CLEARANCE, 4),
    -0.085,
)
"""Where Robot B stands to place the bottle on the goal table, (x, y) [m]."""

SERVE_WALK = round(math.dist(ROBOT_B_PICK, ROBOT_B_SERVE), 3)
"""Distance Robot B carries the bottle [m]."""

# Success thresholds

SPAWN_GAP = 0.005
"""Height above its surface at which each prop is released at reset [m]."""

TABLE_INSET = 0.16
"""Distance from the goal table's west edge to the goal [m]."""

GOAL_AT = (
    round(GOAL_TABLE_AT[0] - TABLE_HALF + TABLE_INSET, 4),
    -0.145,
    TABLE_TOP_Z,
)
"""Goal position of the bottle on the table top, (x, y, z) [m]."""

GOAL_XY_TOL = 0.06
"""Tolerance on the bottle's resting position, per horizontal axis [m]."""

GOAL_Z_TOL = 0.06
"""Tolerance on the height of the bottle's centre [m]."""

CART_PARK_RADIUS = 0.35
"""Radius round ``CART_PARKED`` inside which the cart counts as parked [m]."""

BOTTLE_LIFT_MARGIN = 0.10
"""Rise above its spawn height at which the held bottle counts as lifted [m]."""

CARRY_GOAL_RADIUS = 0.15
"""Radius round the goal, in the plane, the held bottle must enter for the carry to count [m]."""

BOTTLE_GOAL_Z = round(TABLE_TOP_Z + BOTTLE_HALF_HEIGHT, 4)
"""Height of the bottle's origin when it stands on the goal table [m].
The origin is the bottle's geometric centre."""

DROPPED_BELOW_Z = 0.50
"""Height below which the bottle counts as dropped [m]."""

GOAL_MARKER_CFG = box_cfg(
    "GoalMarker",
    (2 * GOAL_XY_TOL, 2 * GOAL_XY_TOL, GOAL_PAINT_THICKNESS),
    (GOAL_AT[0], GOAL_AT[1], TABLE_TOP_Z + DECAL_LIFT + GOAL_PAINT_THICKNESS / 2),
    GOAL_PAINT_COLOUR,
    collision=False,
    emissive=GOAL_PAINT_EMISSIVE,
    movable=True,
)
"""Green goal patch on the table, visual only. Its size is the placement
tolerance. A reset moves it, and its pose is the goal."""

CART_MARK_CFG = box_cfg(
    "CartMark",
    (CART_DECK_X, CART_DECK_Y, GOAL_PAINT_THICKNESS),
    (CART_PARKED[0], CART_PARKED[1], DECAL_LIFT + GOAL_PAINT_THICKNESS / 2),
    GOAL_PAINT_COLOUR,
    collision=False,
    emissive=GOAL_PAINT_EMISSIVE,
)
"""Green parking mark for the cart on the floor, visual only."""

# Spawn poses

CART_START_POS = (CART_START[0], CART_START[1], 0.0)
"""The cart's reset position [m]. Its origin is the centre of its footprint, on the floor."""

BOTTLE_START = (
    round(CART_START[0] + COLLAR_AT[0], 4),
    round(CART_START[1] + COLLAR_AT[1], 4),
    round(CART_TOP_Z + SPAWN_GAP + BOTTLE_HALF_HEIGHT, 4),
)
PLATE_START = (
    round(CART_START[0] + PLATE_AT[0], 4),
    round(CART_START[1] + PLATE_AT[1], 4),
    round(CART_TOP_Z + SPAWN_GAP, 4),
)
BOARD_START = (
    round(CART_START[0] + BOARD_AT[0], 4),
    round(CART_START[1] + BOARD_AT[1], 4),
    round(CART_TOP_Z + SPAWN_GAP, 4),
)
"""Reset positions of the bottle, the plate and the board [m].
The bottle's origin is its centre. The other two have theirs at the base."""

# Reset draws

CART_SPAWN_REGION = {
    "outer": {"x": (-0.10, 0.10), "y": (-0.30, 0.08)},
}
"""Where a reset puts the cart, as offsets from ``CART_START`` [m]: across
the aisle, and up to 0.30 m further from Robot A or 0.08 m nearer."""

CART_RIDERS = ("object", "plate", "board")
"""What rides the cart and moves with it."""

GOAL_SPAWN_REGION = {"outer": {"y": (-0.16, 0.0)}}
"""Where a reset puts the goal patch, as an offset from ``GOAL_AT`` [m]:
up to 0.16 m along the table edge, away from the napkin setting."""

# Asset cfgs


def pin_caster_combine_mode(prim_path_expr: str, mode: str = "min") -> int:
    """Make the casters' friction ``min(caster, floor)`` instead of the average. Returns how many.

    Applied at spawn: the PhysX material schema needs Kit running.
    """
    from pxr import PhysxSchema

    from isaaclab.sim.utils.prims import find_matching_prim_paths, get_current_stage

    stage = get_current_stage()
    pinned = 0
    for cart_path in find_matching_prim_paths(prim_path_expr):
        material = stage.GetPrimAtPath(f"{cart_path}/Looks/CasterPhysics")
        if not material.IsValid():
            continue
        PhysxSchema.PhysxMaterialAPI.Apply(material)
        PhysxSchema.PhysxMaterialAPI(material).CreateFrictionCombineModeAttr(mode)
        pinned += 1
    return pinned


def _with_rolling_casters(cfg: RigidObjectCfg) -> RigidObjectCfg:
    """Wrap the cart's spawner so :func:`pin_caster_combine_mode` runs after spawn."""
    original = cfg.spawn.func

    @functools.wraps(original)
    def wrapper(prim_path, spawn_cfg, *args, **kwargs):
        prim = original(prim_path, spawn_cfg, *args, **kwargs)
        pin_caster_combine_mode(prim_path)
        return prim

    cfg.spawn.func = wrapper
    return cfg


CART_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cart",
    init_state=RigidObjectCfg.InitialStateCfg(pos=CART_START_POS, rot=IDENTITY_ROT),
    spawn=UsdFileCfg(
        usd_path=str(_DATA / CART_ASSET),
        # Set here because `graspable()` would replace the USD's two physics materials.
        activate_contact_sensors=True,
        rigid_props=manipuland_rigid_props(),
        collision_props=manipuland_collision_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=CART_MASS),
    ),
)
"""The service cart. No ``physics_material``: the USD binds its own two, per prim."""

CART_CFG = _with_rolling_casters(CART_CFG)
"""The cart with its casters set to the ``min`` friction combine mode at spawn."""

BOTTLE_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Bottle",
    init_state=RigidObjectCfg.InitialStateCfg(pos=BOTTLE_START, rot=IDENTITY_ROT),
    spawn=UsdFileCfg(
        usd_path=str(_DATA / BOTTLE_ASSET),
        scale=(BOTTLE_SCALE, BOTTLE_SCALE, BOTTLE_SCALE),
        rigid_props=manipuland_rigid_props(),
        collision_props=manipuland_collision_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=BOTTLE_MASS),
    ),
)
"""The wine bottle, the scene's ``object``.
The prim name must match the one given to ``physical_scene``."""

PLATE_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Plate",
    init_state=RigidObjectCfg.InitialStateCfg(pos=PLATE_START, rot=IDENTITY_ROT),
    spawn=UsdFileCfg(
        usd_path=str(_DATA / PLATE_ASSET),
        rigid_props=manipuland_rigid_props(),
        collision_props=manipuland_collision_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=PLATE_MASS),
    ),
)
"""The plate of sandwiches. Cargo."""

BOARD_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Board",
    init_state=RigidObjectCfg.InitialStateCfg(pos=BOARD_START, rot=IDENTITY_ROT),
    spawn=UsdFileCfg(
        usd_path=str(_DATA / BOARD_ASSET),
        rigid_props=manipuland_rigid_props(),
        collision_props=manipuland_collision_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=BOARD_MASS),
    ),
)
"""The cheese board. Cargo, loose on the deck."""
