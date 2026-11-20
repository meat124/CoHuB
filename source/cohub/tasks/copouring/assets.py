"""CoPouring's tub, apples, target crate and the layout constants.

``mdp.PourSuccess`` scores the apples against the target crate's interior,
read from its live pose. The USDs in ``data/`` are baked: a dimension changed
here does not change them.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from cohub.tasks._base.props import box_cfg

_DATA = Path(__file__).parent / "data"

# Crate
CRATE_SIZE = (0.506, 0.406, 0.254)
"""The crate asset's size, (x, y, z) [m]. Z up, base at z=0."""

# Tub
TOTE_MASS = 2.5
"""Tub mass [kg]."""

# Apples
APPLE_SIZE = (0.079, 0.080, 0.086)
"""The baked apple's natural size, (x, y, z) [m]."""

APPLE_SCALE = 1.5
APPLE_RADIUS = 0.06
APPLE_MASS = 0.22
"""Apple scale, collision sphere radius [m] and mass [kg]."""

APPLE_GRID = (2, 1)
APPLE_COUNT = APPLE_GRID[0] * APPLE_GRID[1]
APPLE_SPACING = (0.150, 0.000)
"""Apple layout at reset: grid (columns, rows), count and spacing (x, y) [m]."""

TOTE_FLOOR_Z = 0.015
"""Height of the tub's inner floor above its base [m]."""

APPLE_REST_Z = round(
    TOTE_FLOOR_Z + APPLE_RADIUS - APPLE_SIZE[2] * APPLE_SCALE / 2, 4
)
"""Height of an apple's origin above the tub's base at reset [m].
The origin is the apple's bottom."""

# Stand
STAND_CRATES = 3
STAND_TOP_Z = round(CRATE_SIZE[2] * STAND_CRATES, 3)
"""Number of crates stacked under the tub, and the height of the stack's top [m]."""

STAND_AT = (0.0, 0.0, 0.0)
"""Where the pickup stack stands, env-local [m]. The task's origin."""

# Target crate
BIN_SCALE = (1.35, 1.35, 1.35)
"""Scale of the target crate relative to the crate asset."""

BIN_SIZE = tuple(round(s * f, 3) for s, f in zip(CRATE_SIZE, BIN_SCALE))
BIN_WALL = 0.018
BIN_FLOOR = 0.011
"""The target crate's outer size (x, y, z), wall thickness and floor thickness [m]."""

BIN_INNER = (
    round(BIN_SIZE[0] - 2 * BIN_WALL, 3),
    round(BIN_SIZE[1] - 2 * BIN_WALL, 3),
    round(BIN_SIZE[2] - BIN_FLOOR, 3),
)
"""The target crate's interior, (x, y, z) [m]. The success box."""

BIN_MASS = 2.0

BIN_STAND_CRATES = 0
BIN_STAND_TOP_Z = round(CRATE_SIZE[2] * BIN_STAND_CRATES, 3)
"""Number of crates stacked under the target crate, and the height of that
stack's top [m]."""

BIN_AT = (0.0, 0.65, 0.0)
"""Where the target crate stands at reset, env-local [m]."""

# Reset randomisation
TOTE_POSE_RANGE = {"x": (-0.02, 0.02), "y": (-0.010, 0.010)}
"""How far the tub's start is drawn from the stack's centre [m]."""

APPLE_POSE_RANGE = {"x": (-0.008, 0.008), "y": (-0.008, 0.008)}
"""How far each apple is drawn from its slot [m]. Must stay inside the pocket's
clearance."""

BIN_POSE_RANGE = {"x": (-0.06, 0.06), "y": (-0.005, 0.005)}
"""How far the target crate is drawn from :data:`BIN_AT` [m]."""

# Success
POUR_MIN_APPLES = APPLE_COUNT
"""How many apples must be in the target crate for success."""

POUR_TILT_DEG = 25.0
"""Tilt from upright at which the tub counts as tipped [deg]."""

LIFT_MARGIN = 0.03
"""How far the tub's base must rise above the stack for the lift to count [m]."""

SPILL_FLOOR_Z = round(APPLE_RADIUS + 0.06, 3)
"""Env-local height below which an apple outside the target crate counts as on
the floor [m]."""

APPLE_MAX_SPEED = 0.05
APPLE_MAX_ANGULAR_SPEED = 1.0
"""Linear [m/s] and angular [rad/s] speed below which an apple is at rest."""

# Cfgs
_RIGID = dict(
    max_depenetration_velocity=1.0,
    disable_gravity=False,
    # A sleeping body reports no contact, so the grasp would never latch.
    sleep_threshold=0.0,
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=2,
)

TOTE_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Tote",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(STAND_AT[0], STAND_AT[1], STAND_TOP_Z)),
    spawn=UsdFileCfg(
        usd_path=str(_DATA / "pour_tote.usda"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(**_RIGID),
        mass_props=sim_utils.MassPropertiesCfg(mass=TOTE_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.006, rest_offset=0.0),
        # The colliders are authored in ``pour_tote.usda``.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8, dynamic_friction=0.7
        ),
    ),
)


def apple_slot(index: int) -> tuple[float, float]:
    """Where apple ``index`` sits in the layer, in the crate's own frame [m]."""
    columns, rows = APPLE_GRID
    column, row = index % columns, index // columns
    return (
        (column - (columns - 1) / 2) * APPLE_SPACING[0],
        (row - (rows - 1) / 2) * APPLE_SPACING[1],
    )


APPLE_CFGS: tuple[RigidObjectCfg, ...] = tuple(
    RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + f"Apple{index}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                STAND_AT[0] + apple_slot(index)[0],
                STAND_AT[1] + apple_slot(index)[1],
                STAND_TOP_Z + APPLE_REST_Z,
            )
        ),
        spawn=UsdFileCfg(
            usd_path=str(_DATA / "apple_fruit.usda"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                **{**_RIGID, "max_depenetration_velocity": 0.5},
                # Stands in for rolling resistance, which PhysX does not model.
                angular_damping=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=APPLE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7, dynamic_friction=0.6
            ),
        ),
    )
    for index in range(APPLE_COUNT)
)
"""The apples, one rigid body each."""

APPLE_NAMES = tuple(f"apple_{index}" for index in range(APPLE_COUNT))
"""Their scene-entity names, in the order :func:`apple_slot` lays them out."""

BIN_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/CatchBin",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(BIN_AT[0], BIN_AT[1], BIN_STAND_TOP_Z)),
    spawn=UsdFileCfg(
        usd_path=str(_DATA / "catch_bin.usda"),
        # Kinematic, so the pour cannot move it. A RigidObject, so an event
        # term can write its pose at reset.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True, disable_gravity=True
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=BIN_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.006, rest_offset=0.0),
    ),
)


STAND_SCALE = (1.0, 1.0, round(1.0 / BIN_SCALE[2], 4))
"""Scale of a stacking crate relative to ``catch_bin.usda``: the same footprint,
the crate asset's height."""


def _stack_crate(name: str, at: tuple[float, float], index: int, crates: int,
                 yaw90: bool = False) -> RigidObjectCfg:
    """One empty crate of the stack under a working crate.

    Kinematic, and a ``RigidObjectCfg`` so the reset can move it with the
    crate on top.
    """
    # Every crate of a stack is upside down, so the top is a flat surface.
    flipped = True
    z = (index + 1) * CRATE_SIZE[2]
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + f"{name}{index}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(at[0], at[1], z),
            # Upside down. ``yaw90`` adds a quarter turn about Z, so the tub
            # overhangs the stand on each robot's side.
            rot=((0.7071068, 0.7071068, 0.0, 0.0) if yaw90 else (0.0, 1.0, 0.0, 0.0))
            if flipped else (1.0, 0.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(
            usd_path=str(_DATA / "catch_bin.usda"),
            # x and y keep the target crate's footprint. z restores the
            # crate's own height.
            scale=STAND_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BIN_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.006, rest_offset=0.0
            ),
        ),
    )


STAND_CRATE_CFGS: tuple[RigidObjectCfg, ...] = tuple(
    _stack_crate("StandCrate", STAND_AT[:2], index, STAND_CRATES, yaw90=True)
    for index in range(STAND_CRATES)
)
"""The stack under the tub."""

# Backdrop
BACKDROP_HEIGHT = 4.5
BACKDROP_COLOUR = (0.62, 0.63, 0.64)
BACKDROP_CFGS: tuple[AssetBaseCfg, ...] = (
    box_cfg("BackdropXNeg", (0.1, 13.0, BACKDROP_HEIGHT), (-3.05, 3.0, BACKDROP_HEIGHT / 2),
            BACKDROP_COLOUR, collision=False),
    box_cfg("BackdropXPos", (0.1, 13.0, BACKDROP_HEIGHT), (6.25, 3.0, BACKDROP_HEIGHT / 2),
            BACKDROP_COLOUR, collision=False),
    box_cfg("BackdropYNeg", (10.0, 0.1, BACKDROP_HEIGHT), (1.6, -2.65, BACKDROP_HEIGHT / 2),
            BACKDROP_COLOUR, collision=False),
    box_cfg("BackdropYPos", (10.0, 0.1, BACKDROP_HEIGHT), (1.6, 8.65, BACKDROP_HEIGHT / 2),
            BACKDROP_COLOUR, collision=False),
)
"""Four slabs outside the store's walls, seen through its openings. Visual only."""

SHUTTER_AT = (5.62, 2.36)
SHUTTER_OPENING = (1.44, 2.03)
SHUTTER_CFG: AssetBaseCfg = box_cfg(
    "ServiceDoor",
    (0.12, SHUTTER_OPENING[0] + 0.10, SHUTTER_OPENING[1] + 0.10),
    (SHUTTER_AT[0] - 0.02, SHUTTER_AT[1], (SHUTTER_OPENING[1] + 0.10) / 2),
    BACKDROP_COLOUR,
    collision=False,
)
"""A slab filling the store's floor-level opening. Visual only."""
