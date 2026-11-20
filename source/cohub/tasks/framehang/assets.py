"""Scene assets for the gallery picture-hang task.

The room is a SceneSmith gallery, see ``scenesmith_room.py``. This file holds
the hang station (picture, peg, hang wall, neighbouring paintings), the wall
art and a few props. Poly Haven models are CC0. No decorative prop is within
1.2 m of either robot.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from cohub.tasks._base.props import (
    COPLANAR_LIFT,
    YAW_90,
    YAW_180,
    YAW_270,
    box_cfg,
    polyhaven_cfg,
)

from .geometry import (
    BASEBOARD_HEIGHT,
    FRAME_CONTACT_OFFSET,
    FRAME_MASS,
    FRAME_START_ROT,
    FRAME_START_Y,
    FRAME_START_Z,
    HOOK_HEIGHT,
    ROOM_HALF_X,
    ROOM_HEIGHT,
    WALL_THICKNESS,
    WALL_Y,
)

_DATA = Path(__file__).parent / "data"
_POLYHAVEN = _DATA / "polyhaven"
_FRAME_USD = str(_DATA / "grande_jatte_frame.usda")
_HOOK_USD = str(_DATA / "wall_hook.usda")
_WALL_ART_USD = {
    "left": str(_DATA / "wall_painting_left.usda"),
    "right": str(_DATA / "wall_painting_right.usda"),
    "far": str(_DATA / "wall_painting_far.usda"),
}

HANG_X = 0.0
"""The x of the peg and of the pair's centreline [m]."""

FRAME_START = (HANG_X, FRAME_START_Y, FRAME_START_Z)
"""Start position of the painting [m]: standing on the floor under the peg,
``geometry.FRAME_STANDOFF`` clear of the wall."""


# Wall finish
_GALLERY_ROUGHNESS = 0.85
"""Roughness of the gallery's plaster."""


def _box(name, size, pos, colour, **kwargs):
    """:func:`~cohub.tasks._base.props.box_cfg` at this room's finish."""
    return box_cfg(name, size, pos, colour, **{"roughness": _GALLERY_ROUGHNESS, **kwargs})


# Floor: a visual-only slab laid over the gallery's own floor
_GALLERY_FLOOR_COLOUR = (0.20, 0.19, 0.175)
"""Floor colour. Linear, not sRGB."""

_FLOOR_CENTRE = (-0.50, -2.53)
"""Centre of ``scene_136`` in the task frame [m]."""

_GALLERY_FLOOR_THICKNESS = 0.02

GALLERY_FLOOR_CFG = _box(
    "GalleryFloor",
    (12.2, 9.2, _GALLERY_FLOOR_THICKNESS),
    (
        _FLOOR_CENTRE[0],
        _FLOOR_CENTRE[1],
        COPLANAR_LIFT - _GALLERY_FLOOR_THICKNESS / 2,
    ),
    _GALLERY_FLOOR_COLOUR,
    collision=False,
    roughness=0.55,
)
"""Visual-only floor slab. Its top face is at ``props.COPLANAR_LIFT``."""


_WALL_COLOUR = (0.80, 0.79, 0.76)
"""Colour of the hang wall. Linear, not sRGB."""

# Hang wall and its baseboard. Kinematic: ``mdp.randomize_station`` moves them at reset.
ROOM_HANG_WALL_CFG = _box(
    "GalleryHangWall",
    (2 * ROOM_HALF_X + 2 * WALL_THICKNESS, WALL_THICKNESS, ROOM_HEIGHT),
    (0.0, WALL_Y + WALL_THICKNESS / 2, ROOM_HEIGHT / 2),
    _WALL_COLOUR,
    movable=True,
)

_BASEBOARD_COLOUR = (0.90, 0.89, 0.87)
_BASEBOARD_DEPTH = 0.02
ROOM_BASEBOARD_HANG_CFG = _box(
    "GalleryBaseboardHang",
    (2 * ROOM_HALF_X, _BASEBOARD_DEPTH, BASEBOARD_HEIGHT),
    (0.0, WALL_Y - _BASEBOARD_DEPTH / 2, BASEBOARD_HEIGHT / 2),
    _BASEBOARD_COLOUR,
    movable=True,
)

# West door: closes the doorway in the gallery's west wall.
# The room is at yaw 180, so into the room is decreasing x.
_DOOR_X = 5.45
"""The x of the west wall's inside face, in the task frame [m]."""
_DOOR_Y = -3.12
"""The y of the centre of the door opening [m]."""
_DOOR_H = 2.40
_DOOR_W = 1.20

_DOOR_LEAF_COLOUR = (0.34, 0.32, 0.30)
"""Colour of the door leaf. Linear, not sRGB."""
_DOOR_CASING_COLOUR = (0.55, 0.54, 0.52)
_DOOR_HANDLE_COLOUR = (0.45, 0.36, 0.16)

# The leaf is the only part that collides.
DOOR_LEAF_CFG = _box(
    "WestDoorLeaf",
    (0.045, _DOOR_W - 0.04, _DOOR_H - 0.04),
    (_DOOR_X + 0.05, _DOOR_Y, (_DOOR_H - 0.04) / 2),
    _DOOR_LEAF_COLOUR,
    roughness=0.45,
)

# Architrave: two jambs and a head, on the room side, visual only.
_CASING = 0.03
DOOR_JAMB_LEFT_CFG = _box(
    "WestDoorJambLeft", (_CASING, 0.09, _DOOR_H + 0.09),
    (_DOOR_X - _CASING / 2, _DOOR_Y + _DOOR_W / 2 + 0.045, (_DOOR_H + 0.09) / 2),
    _DOOR_CASING_COLOUR, collision=False,
)
DOOR_JAMB_RIGHT_CFG = _box(
    "WestDoorJambRight", (_CASING, 0.09, _DOOR_H + 0.09),
    (_DOOR_X - _CASING / 2, _DOOR_Y - _DOOR_W / 2 - 0.045, (_DOOR_H + 0.09) / 2),
    _DOOR_CASING_COLOUR, collision=False,
)
DOOR_HEAD_CFG = _box(
    "WestDoorHead", (_CASING, _DOOR_W + 0.18, 0.09),
    (_DOOR_X - _CASING / 2, _DOOR_Y, _DOOR_H + 0.045),
    _DOOR_CASING_COLOUR, collision=False,
)
# Handle, 1.05 m up.
DOOR_HANDLE_CFG = _box(
    "WestDoorHandle", (0.07, 0.11, 0.025),
    (_DOOR_X + 0.005, _DOOR_Y - _DOOR_W / 2 + 0.18, 1.05),
    _DOOR_HANDLE_COLOUR, collision=False, roughness=0.30,
)


# Wall peg. Kinematic so the reset can move it. Pose writes only.
WALL_HOOK_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/WallHook",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(HANG_X, WALL_Y, HOOK_HEIGHT)),
    spawn=UsdFileCfg(
        usd_path=_HOOK_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
)

FRAME_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Frame",
    # (x, y, z, w). Identity is (0, 0, 0, 1).
    init_state=RigidObjectCfg.InitialStateCfg(pos=FRAME_START, rot=FRAME_START_ROT),
    spawn=UsdFileCfg(
        usd_path=_FRAME_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # Never sleeps: a sleeping body ignores an external wrench.
            sleep_threshold=0.0,
            max_depenetration_velocity=0.5,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=FRAME_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=FRAME_CONTACT_OFFSET, rest_offset=0.0
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
    ),
)


# Props
def _polyhaven(prim_name, asset_dir, pos, **kwargs):
    """:func:`~cohub.tasks._base.props.polyhaven_cfg` out of this task's data dir."""
    return polyhaven_cfg(_POLYHAVEN, prim_name, asset_dir, pos, **kwargs)


# Every prop is at least 1.2 m from either robot.
_DISPLAY_PLINTH_COLOUR = (0.84, 0.83, 0.80)


def _display_plinth(name: str, pos: tuple[float, float], height: float, half_x: float = 0.16):
    return _box(
        name,
        (2 * half_x, 2 * half_x, height),
        (pos[0], pos[1], height / 2),
        _DISPLAY_PLINTH_COLOUR,
        roughness=0.6,
    )


# An unrotated Poly Haven scan faces -Y.
BUST_PLINTH_CFG = _display_plinth("BustPlinth", (-5.85, -6.45), 0.95)
BUST_CFG = _polyhaven("MarbleBust", "marble_bust_01", (-5.85, -6.45, 0.95), rot=YAW_90)

VASE_PLINTH_CFG = _display_plinth("VasePlinth", (4.60, -6.45), 0.85, half_x=0.14)
VASE_CFG = _polyhaven("CeramicVase", "antique_ceramic_vase_01", (4.60, -6.45, 0.85))

STOOL_CFG = _polyhaven("GuardStool", "metal_stool_02", (1.60, -1.45, 0.0), rot=YAW_180)

# Two paintings on the hang wall, 1.80 m either side of the peg. Kinematic, so
# they move with the wall. Their depth is on the -Y side of their origin, so
# they sit on the wall plane unrotated.
_NEIGHBOUR_Z = 1.55


def _wall_painting(name: str, which: str, pos: tuple[float, float, float], rot=(0.0, 0.0, 0.0, 1.0)):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=UsdFileCfg(
            usd_path=_WALL_ART_USD[which],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )


FRAME_NEIGHBOUR_LEFT_CFG = _wall_painting(
    "NeighbourPaintingLeft", "left", (-1.80, WALL_Y, _NEIGHBOUR_Z)
)
FRAME_NEIGHBOUR_RIGHT_CFG = _wall_painting(
    "NeighbourPaintingRight", "right", (1.80, WALL_Y, _NEIGHBOUR_Z)
)

# Wall art replacing the room's own, which is hidden (``scenesmith_room.HIDDEN``).
_ROOM_ART = str(_DATA)

YAW = {0.0: (0.0, 0.0, 0.0, 1.0), 90.0: YAW_90, 180.0: YAW_180, 270.0: YAW_270}
"""(x, y, z, w) quaternions for the four wall facings."""


def _canvas(name: str, asset: str, pos, yaw_deg: float = 0.0):
    """One flat canvas on a wall, facing the room.

    The asset faces -Y. ``yaw_deg`` is 0 for the south wall, 180 for the north,
    90 for the east and 270 for the west.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=YAW[yaw_deg]),
        spawn=UsdFileCfg(
            usd_path=f"{_ROOM_ART}/{asset}.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )


# Wall faces in the task frame [m]: south y = +1.80 (the hang wall),
# north y = -6.97, east x = -6.44, west x = +5.44.
_N_Y, _E_X, _W_X = -6.97, -6.44, 5.44

GALLERY_CANVAS_CFGS = {
    # 2.53 x 1.30 m, on the hang wall left of the pair.
    "canvas_south": _canvas("CanvasSouth", "canvas_south", (-4.75, WALL_Y, 1.55), 0.0),
    "canvas_north_a": _canvas("CanvasNorthA", "canvas_north_a", (2.77, _N_Y, 1.55), 180.0),
    "canvas_north_b": _canvas("CanvasNorthB", "canvas_north_b", (-0.77, _N_Y, 1.55), 180.0),
    "print_north": _canvas("PrintNorth", "print_north", (-4.04, _N_Y, 1.55), 180.0),
    "canvas_east": _canvas("CanvasEast", "canvas_east", (_E_X, -2.53, 1.55), 90.0),
    "print_east": _canvas("PrintEast", "print_east", (_E_X, -0.39, 1.55), 90.0),
    "print_east_b": _canvas("PrintEastB", "print_east_b", (_E_X, -4.68, 1.55), 90.0),
    "canvas_west": _canvas("CanvasWest", "canvas_west", (_W_X, -5.43, 1.55), 270.0),
    "print_west": _canvas("PrintWest", "print_west", (_W_X, -0.28, 1.55), 270.0),
}
"""The wall art, keyed by the slot it fills. Centres are 1.55 m up."""
