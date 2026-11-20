"""TableAlign's room: a downloaded SceneSmith office and its placement.

The room is placed so that its chair ``office_office_chair_2``, the goal
frame, lands 2.10 m down the carry lane.

    python scripts/setup_assets.py rooms    # fetches and prepares it
"""

from cohub.tasks._base.props import ceiling_cfg
from cohub.scenesmith.cfg import scene_data_dir, scenery_cfg_at

SCENE_ID = "Room/scene_115"
"""A SceneSmith open-plan office: 78 objects, 8.10 x 6.10 x 3.10 m.
Listed in ``cohub.scenesmith.REQUIRED_SCENES``."""

SCENE_DIR = scene_data_dir(SCENE_ID)
"""Where ``setup_assets.py rooms`` puts the scene."""

ROOM_YAW = 270.0
"""Room yaw [deg]. Turns the chair's forward onto task -X."""

ROOM_ORIGIN_AT = (3.70, 2.12)
"""The scene coordinate that becomes the task's origin [m]. Puts the goal on
the carry lane, ``assets.CARRY_DISTANCE`` from the spawn."""

HIDDEN = (
    "office_office_desk_2",
    "office_office_divider_panel_0",
    "office_office_divider_panel_1",
    "office_spiral_notebook_1",
    "office_wireless_keyboard_2",
    "office_all_in_one_desktop_computer_2",
    "office_computer_mouse_2",
    "office_smartphone_0",
)
"""Objects hidden from the room: the desk at the goal, the items on it and both
divider panels."""

ROOM_CFG = scenery_cfg_at(
    SCENE_DIR, ROOM_ORIGIN_AT, yaw_deg=ROOM_YAW, hide=HIDDEN, scene_id=SCENE_ID
)

SCENE_CAM_POS = (-1.95, -1.05, 2.60)
SCENE_CAM_ROT = (0.0, 0.35172667, 0.0, 0.93610274)
SCENE_CAM_FOCAL = 10.5
"""Third-person camera position [m], rotation (x, y, z, w) and focal length [mm].
Broadside to the carry, aimed at its middle."""

CEILING_SIZE = (6.1, 8.1)
CEILING_CENTRE = (0.88, -0.3)
CEILING_TOP_Z = 3.0
"""The room's footprint (x, y), its centre and its wall height, in the task frame [m]."""

CEILING_CFG = ceiling_cfg(CEILING_SIZE, CEILING_CENTRE, CEILING_TOP_Z)

