"""The gallery for FrameHang: SceneSmith's ``Room/scene_136``.

The room supplies the floor, the walls and the furniture. The task keeps its
own hang wall, peg and picture, because the hang wall moves at reset. The scene
has no lights and no ceiling. The shared dome light illuminates it.

    python scripts/setup_assets.py rooms    # fetches and prepares it
"""

from cohub.scenesmith.cfg import scene_data_dir, scenery_cfg_at

SCENE_ID = "Room/scene_136"
"""An art gallery: one hall, 12.1 x 9.1 x 4.1 m, with its art on the walls.
Listed in ``cohub.scenesmith.REQUIRED_SCENES``."""

SCENE_DIR = scene_data_dir(SCENE_ID)
"""Where ``scripts/setup_assets.py rooms`` puts the scene."""

ROOM_YAW = 180.0
"""Yaw of the room in the task frame [deg]. It puts the gallery's south wall behind the hang wall."""

ROOM_ORIGIN_AT = (5.50, 1.97)
"""Scene coordinate (x, y) that becomes the task's origin [m].
It puts the back of the hang wall on the gallery's south wall."""

HIDDEN = (
    "gallery_painting",
    "gallery_print",
    "gallery_photograph",
    "gallery_partition_panel",
)
"""Objects hidden from the room, matched by substring: its artworks and its partition panels.
``assets.GALLERY_CANVAS_CFGS`` replaces the artworks."""

ROOM_CFG = scenery_cfg_at(
    SCENE_DIR, ROOM_ORIGIN_AT, yaw_deg=ROOM_YAW, hide=HIDDEN, scene_id=SCENE_ID
)

SCENE_CAM_POS = (0.0, -1.45, 2.05)
"""Position of the third-person camera, in the task frame [m]."""
