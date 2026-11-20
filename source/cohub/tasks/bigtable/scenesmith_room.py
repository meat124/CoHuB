"""BigTable's room: a downloaded SceneSmith dining room and its placement.

    python scripts/setup_assets.py rooms    # fetches and prepares it
"""

from cohub.scenesmith.cfg import scene_data_dir, scenery_cfg_at
from cohub.tasks._base.props import box_cfg

SCENE_ID = "Room/scene_046"
"""A SceneSmith dining room: 58 objects, 6.1 x 5.1 x 3.3 m.
Listed in ``cohub.scenesmith.REQUIRED_SCENES``."""

SCENE_DIR = scene_data_dir(SCENE_ID)
"""Where ``setup_assets.py rooms`` puts the scene."""

ROOM_YAW = 90.0
"""Room yaw [deg]. Puts the room's long axis along the carry."""

ROOM_ORIGIN_AT = (4.20, 3.00)
"""The scene coordinate that becomes the task's origin, where the table spawns [m]."""

DOOR_CFG = box_cfg(
    "Door",
    size=(0.08, 0.99, 2.19),
    pos=(3.00, -1.318, 1.095),
    color=(0.45, 0.33, 0.24),
)
"""A panel closing the room's doorway, in the task frame. It collides."""

HIDDEN = (
    # The room's own table. The task spawns its copy in its place.
    "round_dining_table",
    # Its chairs stand where the robots do.
    "dining_chair",
    # Everything that rests on the table.
    "dinner_plate",
    "shallow_ceramic_bowl",
    "eucalyptus_stem",
    "white_flower",
    "rose_stem",
    "votive_candle_holder",
    "candle_1",
)
"""Objects hidden from the room, matched by substring."""

ROOM_CFG = scenery_cfg_at(
    SCENE_DIR, ROOM_ORIGIN_AT, yaw_deg=ROOM_YAW, hide=HIDDEN, scene_id=SCENE_ID
)
