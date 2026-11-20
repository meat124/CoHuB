"""The room this task stands in: a downloaded SceneSmith scene.

The scene is not vendored. Fetch and prepare it once:

    python scripts/setup_assets.py rooms
"""

from cohub.scenesmith.cfg import scene_data_dir, scenery_cfg_at
from cohub.tasks._base.props import box_cfg

SCENE_ID = "Room/scene_084"
"""A modern kitchen: 49 objects, 4.6 x 4.1 x 2.9 m, floor top at z=0."""

SCENE_DIR = scene_data_dir(SCENE_ID)
"""Where ``scripts/setup_assets.py rooms`` puts the scene."""

ROOM_YAW = 90.0
"""Room yaw [degrees]. Turns the kitchen's long axis onto the counter's."""

ROOM_ORIGIN_AT = (2.25, 2.50)
"""The scene coordinate that becomes the task's origin [m]."""

HIDDEN = (
    "bar_height_table",
    "bar_stool",
    # objects that rested on the table
    "fruit_bowl",
    "banana",
    "apple",
    "pear",
    "tumbler",
    "serving_tray",
    # two of the three identical prints
    "line_art_print_0$",
    "line_art_print_2$",
)
"""Prims removed from the room. A trailing ``$`` anchors an entry to the end of the name."""

ROOM_CFG = scenery_cfg_at(
    SCENE_DIR, ROOM_ORIGIN_AT, yaw_deg=ROOM_YAW, hide=HIDDEN, scene_id=SCENE_ID
)

DOOR_CFG = box_cfg(
    "Door",
    size=(0.08, 0.89, 2.09),
    pos=(2.50, 0.253, 1.05),
    color=(0.45, 0.33, 0.24),
)
"""A panel closing the room's one doorway. It collides."""

SCENE_CAM_POS = (-1.30, 0.0, 2.60)
"""Third-person camera position, inside the room [m]."""
