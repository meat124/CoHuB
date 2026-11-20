"""CoPouring's room: a downloaded SceneSmith grocery store and its placement.

    python scripts/setup_assets.py rooms    # fetches and prepares it
"""

from cohub.scenesmith.cfg import scene_data_dir, scenery_cfg_at

SCENE_ID = "Room/scene_102"
"""A SceneSmith grocery store: 276 objects, 10.1 x 8.1 x 3.7 m."""

SCENE_DIR = scene_data_dir(SCENE_ID)
"""Where ``setup_assets.py rooms`` puts the scene."""

ROOM_YAW = 90.0
"""Room yaw [deg]."""

ROOM_ORIGIN_AT = (2.05, 5.65)
"""The scene coordinate that becomes the task's origin [m]."""

HIDDEN = ()
"""Objects hidden from the room: none."""

ROOM_CFG = scenery_cfg_at(
    SCENE_DIR, ROOM_ORIGIN_AT, yaw_deg=ROOM_YAW, hide=HIDDEN, scene_id=SCENE_ID
)
