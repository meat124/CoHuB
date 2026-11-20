"""The room for CartService: SceneSmith's restaurant, ``Room/scene_110``.

The scene is not vendored. Fetch and prepare it once:

    python scripts/setup_assets.py rooms

Add ``ROOM_ORIGIN_AT`` to a task coordinate to get the scene coordinate.
"""

from cohub.scenesmith.cfg import scene_data_dir, scenery_cfg_at
from cohub.tasks._base.props import doorway_panel_cfg

SCENE_ID = "Room/scene_110"
"""A restaurant: one hall, 12 x 8 x 3.6 m inside, with a bar counter and four dining tables.
Listed in ``cohub.scenesmith.REQUIRED_SCENES``."""

SCENE_DIR = scene_data_dir(SCENE_ID)
"""Where ``scripts/setup_assets.py rooms`` puts the scene."""

ROOM_YAW = 0.0
"""Yaw of the room in the task frame [deg]. The robots face each other along Y, the room's aisle."""

ROOM_ORIGIN_AT = (2.40, 2.305)
"""Scene coordinate (x, y) that becomes the task's origin [m]: where the cart is parked."""

HIDDEN = (
    # The chairs on the goal table's west and south sides, where Robot B serves.
    "dining_chair_b$",
    "dining_chair_9$",
)
"""Objects hidden from the room, as regular expressions. The ``$`` anchors make each entry
match one object. An entry that matches nothing is ignored silently."""

DOORWAYS = {
    # name: ((x0, x1), (y0, y1), (z0, z1)) in scene coordinates, spanning both
    # wall layers on the thin axis.
    # north wall
    "DoorNorth": ((4.398, 5.598), (7.95, 8.05), (0.0, 2.2)),
    # south wall, the entrance
    "DoorSouth": ((10.051, 11.651), (-0.05, 0.05), (0.0, 2.2)),
}
"""The room's two doorways. The scene leaves them open, so ``DOORS`` closes them."""

DOORS = {
    name: doorway_panel_cfg(name, opening, ROOM_ORIGIN_AT, ROOM_YAW)
    for name, opening in DOORWAYS.items()
}
"""One kinematic panel per doorway, in the task frame."""

ROOM_CFG = scenery_cfg_at(
    SCENE_DIR, ROOM_ORIGIN_AT, yaw_deg=ROOM_YAW, hide=HIDDEN, scene_id=SCENE_ID
)


def _look_at(eye, target):
    """Quaternion (x, y, z, w) aiming a ``scene_camera_cfg`` from ``eye`` at ``target``.

    The camera's local +X is forward. Yaw about Z, then pitch about Y, no roll.
    """
    import math

    dx, dy, dz = (target[i] - eye[i] for i in range(3))
    yaw, pitch = math.atan2(dy, dx), math.atan2(-dz, math.hypot(dx, dy))
    cz, sz = math.cos(yaw / 2), math.sin(yaw / 2)
    cy, sy = math.cos(pitch / 2), math.sin(pitch / 2)
    # (0, 0, sz, cz) * (0, sy, 0, cy)
    return (-sz * sy, cz * sy, sz * cy, cz * cy)


SCENE_CAM_AIM = (0.35, 0.45, 0.55)
"""Point the third-person camera looks at, in the task frame [m].
Low enough that the frame stays below the top of the walls: the room has no ceiling."""

SCENE_CAM_POS = (3.90, -1.705, 2.85)
"""Position of the third-person camera, in the task frame [m]."""

SCENE_CAM_ROT = _look_at(SCENE_CAM_POS, SCENE_CAM_AIM)
"""Rotation of the third-person camera, (x, y, z, w)."""
