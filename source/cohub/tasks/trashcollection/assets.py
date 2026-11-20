"""TrashCollection's bin, litter and the layout constants.

The bin and the bottle are baked USDs in ``data/``.
The bench belongs to the room. Every position is in the task frame, which is
the scene's frame shifted by ``scenesmith_room.ROOM_ORIGIN_AT``.
"""

from pathlib import Path

import isaaclab.sim as sim_utils

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from cohub.tasks._base.props import (
    IDENTITY_ROT,
    manipuland_collision_props,
    manipuland_rigid_props,
)

_DATA = Path(__file__).resolve().parent / "data"

FLOOR_Z = 0.0
"""Height of the floor the props stand on [m]."""

# Bin. See THIRD_PARTY_ASSETS.md.

BIN_SIZE = (0.362, 0.362, 0.60)
"""The baked bin's size [m]: diameter, diameter, height."""

BIN_HEIGHT = BIN_SIZE[2]
BIN_RIM_INNER = 0.173
"""The bin's height, and the inner radius of its rim [m]: the mouth the success
predicate measures against."""

BIN_MASS = 2.5
"""Bin mass [kg]."""

BIN_TIPPED_DEG = 60.0
"""How far the bin's own up-axis may lean from vertical before the episode
ends on ``bin_tipped`` [deg]."""

# The room's bench that Robot B sweeps, in the task frame

BENCH_TOP_Z = 0.830
"""The bench's work surface [m]: the room's bench lowered by
``scenesmith_room.LOWERED_BY``."""

# Litter

LITTER_CONTACT_OFFSET = 0.004
"""PhysX contact offset for a piece of litter [m]."""

LITTER_FRICTION = (0.8, 0.7)
"""Static and dynamic friction of a piece of litter."""

LITTER_LIE_ROT = (0.70710678, 0.0, 0.0, 0.70710678)
"""A quarter turn about X, in (x, y, z, w). Lays an upright prop down with its
long axis along Y."""

BOTTLE_LENGTH = 0.16
"""The bottle's length [m]."""

BOTTLE_MASS = 0.05
"""Bottle mass [kg]."""

LITTER_KINDS = ("plastic_bottle",)
"""Which prop each of ``LITTER_START``'s positions gets."""

LITTER_START = ((0.41, -0.10),)
"""Where each piece's centre sits on the bench, (x, y) in the task frame [m]."""

LITTER_NAMES = ("litter_0",)
"""The scene entity names of the pieces. Terms take them as plain strings."""

LITTER_FLOOR_Z = 0.10
"""Below this, a piece of litter that is not inside the bin has fallen to the
floor and the episode ends on ``litter_dropped`` [m]."""

_LITTER_SPEC = {
    # kind: (length, thickness once laid down, mass)
    "plastic_bottle": (BOTTLE_LENGTH, 0.0547, BOTTLE_MASS),
}
"""Baked dimensions per kind. Update them after re-running ``build_trash_props.py``."""


def _litter_cfg(index: int, kind: str) -> RigidObjectCfg:
    """One piece of litter, lying on its side with its centre on its spot.

    The baked USD has its origin at the base of the standing prop, so the
    pose is offset by half the thickness in z and half the length in y.
    """
    length, across, mass = _LITTER_SPEC[kind]
    x, y = LITTER_START[index]
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + f"Litter{index}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(x, y + length / 2.0, BENCH_TOP_Z + across / 2.0), rot=LITTER_LIE_ROT
        ),
        spawn=UsdFileCfg(
            usd_path=str(_DATA / f"{kind}.usdc"),
            rigid_props=manipuland_rigid_props(),
            collision_props=manipuland_collision_props(LITTER_CONTACT_OFFSET),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=LITTER_FRICTION[0], dynamic_friction=LITTER_FRICTION[1]
            ),
        ),
    )


LITTER_CFGS = tuple(_litter_cfg(i, kind) for i, kind in enumerate(LITTER_KINDS))
"""The pieces, in ``LITTER_START`` order."""

# Layout. The task origin is where the bin is held. +X runs east along the bench.

APPROACH_X = -0.42
"""The x of the line Robot A picks up on, walks down and delivers on [m]."""

OBJECT_START = (APPROACH_X, 1.57, FLOOR_Z)
"""Where the bin stands at reset [m]. Its origin is its base centre."""

BIN_CARRY_AT = (0.03, LITTER_START[0][1])
"""Where Robot A holds the bin for the sweep, (x, y) in the task frame [m].
The centre of the ``deliver`` milestone. Not scored by ``success``."""

DELIVER_RADIUS = 0.25
"""How far the bin's centre may stand from ``BIN_CARRY_AT`` in the plane and
still count as delivered [m]."""

BIN_OFF_FLOOR = 0.02
"""How far the bin's base centre must be off the floor, with Robot A holding
it, for the bin to count as held aloft [m]."""

RIM_ABOVE_BENCH_TOLERANCE = 0.01
"""How far the rim may stand above the bench top and the bin still count as
delivered [m]."""

# Reset draws

BIN_SPAWN_REGION = {
    "outer": {"x": (-0.15, 0.15), "y": (-0.15, 0.0)},
    "exclude": [
        {"x": (0.08, 0.15), "y": (-0.04, 0.0)},  # on one of Robot A's feet
        {"x": (-0.15, -0.08), "y": (-0.04, 0.0)},  # on the other
    ],
}
"""Where a reset puts the bin, as offsets from ``OBJECT_START`` [m]:
to either side and up to 0.15 m further from Robot A."""

LITTER_SPAWN_REGION = {
    "outer": {"x": (-0.10, 0.10), "y": (-0.04, 0.12)},
    "exclude": [
        {"x": (0.02, 0.11), "y": (-0.05, 0.10)},  # under Robot B's right hand
        {"x": (-0.11, -0.02), "y": (-0.05, 0.10)},  # under its left hand
    ],
}
"""Where a reset puts the bottle, as offsets from ``LITTER_START`` [m]."""

# Robot poses

ROBOT_A_START = (APPROACH_X, 1.867)
"""Robot A's spawn, (x, y) [m]. It faces -Y."""

ROBOT_B_POS = (0.41, -0.41)
"""Robot B's spawn, (x, y) [m]. It faces +Y, into the bench."""

# Bin cfg

OBJECT_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Bin",
    init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_START, rot=IDENTITY_ROT),
    spawn=UsdFileCfg(
        usd_path=str(_DATA / "trash_bin.usdc"),
        rigid_props=manipuland_rigid_props(),
        collision_props=manipuland_collision_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=BIN_MASS),
    ),
)
"""The bin. Its collision is authored in the USD: 32 convex wall pieces on the
visual wall, thickened outward to the rim's outer radius, and a floor slab. The
visual meshes do not collide."""
