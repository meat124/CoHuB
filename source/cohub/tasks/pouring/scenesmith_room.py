"""The room this task pours in: a downloaded SceneSmith scene.

    python scripts/setup_assets.py rooms    # fetches and prepares it
"""

from cohub.scenesmith.cfg import scene_data_dir, scenery_cfg_at

SCENE_ID = "Room/scene_085"
"""A cozy kitchen: 45 objects, 3.9 x 4.3 x 2.65 m, floor top at z=0."""

SCENE_DIR = scene_data_dir(SCENE_ID)
"""Where ``scripts/setup_assets.py rooms`` puts the scene."""

ROOM_YAW = 0.0
"""Room yaw [degrees]."""

ROOM_ORIGIN_AT = (2.05, 1.55)
"""The scene coordinate that becomes the task's origin [m]."""

HIDDEN = ()
"""Prims removed from the room: none."""

ROOM_CFG = scenery_cfg_at(
    SCENE_DIR, ROOM_ORIGIN_AT, yaw_deg=ROOM_YAW, hide=HIDDEN, scene_id=SCENE_ID
)
