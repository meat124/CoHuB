"""Pouring's two vessels, the marble, the table, and their dimensions.

Robot A tips one marble out of a cup into a glass Robot B holds. Both vessels
are SceneSmith assets at their own size, baked into ``data/``. Neither robot
can reach the other's vessel.
``mdp.PourSuccess`` scores the marble against the glass's interior, in the
glass's own frame.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from cohub.g1.grasp import GRIP_FRICTION
from cohub.scenesmith.cfg import yaw_quat
from cohub.tasks._base.props import box_cfg

_DATA = Path(__file__).parent / "data"

# ---------------------------------------------------------------- the marble
BEAD_RADIUS = 0.015
"""Marble radius [m]."""

BEAD_MASS = 0.035
"""Marble mass [kg]."""

# --------------------------------------------------------------- the vessels
CUP_RAW = (0.0850, 0.0850, 0.0890)
"""Robot A's vessel: a tumbler cup, with its bounding size [m]."""

MUG_RAW = (0.0720, 0.0720, 0.1190)
"""Robot B's vessel: a water tumbler, with its bounding size [m]."""

POUR_CUP_OUTER_R = round(CUP_RAW[0] / 2, 4)
POUR_CUP_HEIGHT = round(CUP_RAW[2], 4)
CATCH_MUG_HEIGHT = round(MUG_RAW[2], 4)
"""The cup's outer radius, and each vessel's height [m]."""

BOWL_FLOOR_Z = 0.011
"""The inside of the cup's floor, above its own base [m]."""

POUR_CUP_INNER_R = 0.0378
"""The cup's interior radius at its rim [m]."""

CATCH_MUG_INNER_R = 0.030
CATCH_MUG_FLOOR_Z = 0.007
CATCH_MUG_INNER_TOP = round(CATCH_MUG_HEIGHT - 0.002, 4)
"""The tumbler's interior, as the cylinder the success predicate scores [m]."""

POUR_ESCAPE_DEG = 110.0
"""About where the marble leaves the cup [degrees from upright]."""

BEAD_START_Z = round(BOWL_FLOOR_Z + BEAD_RADIUS + 0.002, 4)
"""Marble centre above the cup's base at reset [m]."""

POUR_CUP_MASS = 0.30
CATCH_MUG_MASS = 0.30
"""Vessel masses [kg]."""

# ------------------------------------------------------------------ the room
WORKTOP_Z = 0.72
"""Height of the work table's top, where both vessels start [m]."""

TABLE_SIZE = (0.55, 1.20)
TABLE_SLAB = 0.05
TABLE_LEG = 0.06
TABLE_LEG_INSET = 0.05
"""The work table [m]: top size, slab thickness, leg width and leg inset."""

# ------------------------------------------------------------------- layout
ROBOT_STANDOFF_X = 0.455
"""Distance from the table's centre line to each robot's pelvis at spawn [m]. Solved together with :data:`CUP_AT`."""

CUP_AT = (-0.17, 0.0)
MUG_AT = (0.17, 0.0)
"""Where each vessel stands on the table at reset, env-local [m]."""

CUP_A_YAW_DEG = 180.0
MUG_B_YAW_DEG = 0.0
"""Which way each vessel is turned at reset [degrees about Z]."""

# --------------------------------------------------- reset randomisation
CUP_POSE_RANGE = {"x": (-0.03, 0.03), "y": (-0.06, 0.06)}
"""How far each vessel is drawn from its nominal spot on the table [m]."""

CUP_YAW_RANGE = (-30.0, 30.0)
"""How far each vessel is turned from its nominal facing [degrees]."""

BEAD_POSE_RANGE = {"x": (-0.006, 0.006), "y": (-0.006, 0.006)}
"""Where in the cup the marble starts, relative to the cup's own centre [m]."""

# ---------------------------------------------------------------- success
POUR_TILT_DEG = 60.0
"""How far the cup must be tipped for the tip to count [degrees from upright]."""

LIFT_MARGIN = 0.03
"""How far a vessel's base must rise above the table for the lift to count [m]."""

POUR_DWELL_S = 0.5
"""How long the marble must stay inside the glass to count as landed [s]."""

BEAD_LOST_HEIGHT = round(BEAD_RADIUS + 0.010, 3)
BEAD_SETTLE_SPEED = 0.05
BEAD_SETTLE_HOLD_S = 0.25
"""Thresholds of ``mdp.BeadLost``: height above the worktop [m], speed [m/s] and hold time [s]."""


def _vessel_spawn(usd: str, mass: float) -> UsdFileCfg:
    """The spawn both vessels share.

    ``sleep_threshold=0.0`` is required: a sleeping body reports no contact to
    the grasp term.
    """
    return UsdFileCfg(
        usd_path=str(_DATA / usd),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5,
            disable_gravity=False,
            sleep_threshold=0.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        # small, because the cup's wall and floor are only 7-8 mm thick
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=GRIP_FRICTION, dynamic_friction=GRIP_FRICTION
        ),
        activate_contact_sensors=True,
    )


# ------------------------------------------------------------------ the cfgs
POUR_CUP_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/PourCup",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(CUP_AT[0], CUP_AT[1], WORKTOP_Z), rot=yaw_quat(CUP_A_YAW_DEG)
    ),
    spawn=_vessel_spawn("pour_cup.usdc", POUR_CUP_MASS),
)
"""Robot A's cup, with the marble in it. The scene's ``object``."""

CATCH_MUG_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/CatchMug",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(MUG_AT[0], MUG_AT[1], WORKTOP_Z), rot=yaw_quat(MUG_B_YAW_DEG)
    ),
    spawn=_vessel_spawn("catch_mug.usdc", CATCH_MUG_MASS),
)
"""Robot B's tumbler, empty. A dynamic body."""

BEAD_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Bead",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(CUP_AT[0], CUP_AT[1], WORKTOP_Z + BEAD_START_Z)
    ),
    spawn=UsdFileCfg(
        usd_path=str(_DATA / "bead.usdc"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5,
            disable_gravity=False,
            sleep_threshold=0.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
            # PhysX models no rolling resistance, so this is what slows a rolling marble
            angular_damping=4.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=BEAD_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.7, dynamic_friction=0.6
        ),
    ),
)
"""The marble."""

# ------------------------------------------------------------------ the table
TABLE_TOP_WOOD = (0.42, 0.33, 0.22)
TABLE_LEG_WOOD = (0.29, 0.21, 0.13)
"""The table's two wood colours."""

TABLE_CFGS: tuple[AssetBaseCfg, ...] = (
    box_cfg(
        "TableTop",
        (TABLE_SIZE[0], TABLE_SIZE[1], TABLE_SLAB),
        (0.0, 0.0, WORKTOP_Z - TABLE_SLAB / 2),
        TABLE_TOP_WOOD,
    ),
) + tuple(
    box_cfg(
        f"TableLeg{index}",
        (TABLE_LEG, TABLE_LEG, WORKTOP_Z - TABLE_SLAB),
        (
            sx * (TABLE_SIZE[0] / 2 - TABLE_LEG_INSET - TABLE_LEG / 2),
            sy * (TABLE_SIZE[1] / 2 - TABLE_LEG_INSET - TABLE_LEG / 2),
            (WORKTOP_Z - TABLE_SLAB) / 2,
        ),
        TABLE_LEG_WOOD,
    )
    for index, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))
)
"""The work table: a slab on four legs."""

# ------------------------------------------------------------------- the door
DOOR_AT = (-0.131, -1.525)
DOOR_OPENING = (0.90, 2.10)
DOOR_LEAF_COLOUR = (0.88, 0.87, 0.85)
DOOR_CASING_COLOUR = (0.79, 0.77, 0.74)
DOOR_HANDLE_COLOUR = (0.58, 0.56, 0.52)
"""The doorway in the kitchen's south wall: centre (x, y) [m], opening size [m] and the door's colours."""

_LEAF_T = 0.045
_CASING_W = 0.075
_CASING_T = 0.025
_WALL_HALF = 0.03

DOOR_CFGS: tuple[AssetBaseCfg, ...] = (
    box_cfg(
        "DoorLeaf",
        (DOOR_OPENING[0] - 0.02, _LEAF_T, DOOR_OPENING[1] - 0.02),
        (DOOR_AT[0], DOOR_AT[1], (DOOR_OPENING[1] - 0.02) / 2),
        DOOR_LEAF_COLOUR,
        roughness=0.45,
    ),
    # casing: two jambs and a head, on the room's side of the wall
    box_cfg(
        "DoorCasingLeft",
        (_CASING_W, _CASING_T, DOOR_OPENING[1] + _CASING_W),
        (DOOR_AT[0] - (DOOR_OPENING[0] + _CASING_W) / 2,
         DOOR_AT[1] + _WALL_HALF + _CASING_T / 2,
         (DOOR_OPENING[1] + _CASING_W) / 2),
        DOOR_CASING_COLOUR,
    ),
    box_cfg(
        "DoorCasingRight",
        (_CASING_W, _CASING_T, DOOR_OPENING[1] + _CASING_W),
        (DOOR_AT[0] + (DOOR_OPENING[0] + _CASING_W) / 2,
         DOOR_AT[1] + _WALL_HALF + _CASING_T / 2,
         (DOOR_OPENING[1] + _CASING_W) / 2),
        DOOR_CASING_COLOUR,
    ),
    box_cfg(
        "DoorCasingHead",
        (DOOR_OPENING[0] + 2 * _CASING_W, _CASING_T, _CASING_W),
        (DOOR_AT[0], DOOR_AT[1] + _WALL_HALF + _CASING_T / 2,
         DOOR_OPENING[1] + _CASING_W / 2),
        DOOR_CASING_COLOUR,
    ),
    box_cfg(
        "DoorHandle",
        (0.11, 0.035, 0.022),
        (DOOR_AT[0] + DOOR_OPENING[0] / 2 - 0.09,
         DOOR_AT[1] + _LEAF_T / 2 + 0.017, 1.02),
        DOOR_HANDLE_COLOUR,
        roughness=0.3,
    ),
)
"""The door: a leaf, a three-piece casing and a handle, all boxes. It collides."""

# --------------------------------------------------------------- the backdrop
BACKDROP_AT = ((-2.60, 2.30), (-2.10, 3.20))
BACKDROP_HEIGHT = 5.0
BACKDROP_COLOUR = (0.86, 0.85, 0.83)
"""The four backdrop slabs outside the walls: positions [m], height [m] and colour."""

BACKDROP_CFGS: tuple[AssetBaseCfg, ...] = (
    box_cfg("BackdropXNeg", (0.1, 9.0, BACKDROP_HEIGHT),
            (BACKDROP_AT[0][0], 0.55, BACKDROP_HEIGHT / 2), BACKDROP_COLOUR, collision=False),
    box_cfg("BackdropXPos", (0.1, 9.0, BACKDROP_HEIGHT),
            (BACKDROP_AT[0][1], 0.55, BACKDROP_HEIGHT / 2), BACKDROP_COLOUR, collision=False),
    box_cfg("BackdropYNeg", (8.0, 0.1, BACKDROP_HEIGHT),
            (-0.15, BACKDROP_AT[1][0], BACKDROP_HEIGHT / 2), BACKDROP_COLOUR, collision=False),
    box_cfg("BackdropYPos", (8.0, 0.1, BACKDROP_HEIGHT),
            (-0.15, BACKDROP_AT[1][1], BACKDROP_HEIGHT / 2), BACKDROP_COLOUR, collision=False),
)
