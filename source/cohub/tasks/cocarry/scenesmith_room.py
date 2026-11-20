"""CoCarry's room: a downloaded SceneSmith laundromat and its placement.

The scene is not vendored. Fetch and prepare it once:

    python scripts/setup_assets.py rooms
"""

from cohub.tasks._base.props import ceiling_cfg
from cohub.scenesmith.cfg import scene_data_dir, scenery_cfg_at

SCENE_ID = "Room/scene_143"
"""A SceneSmith laundromat: 47 objects, 8.1 x 6.1 x 3.7 m.
Listed in ``cohub.scenesmith.REQUIRED_SCENES``."""

SCENE_DIR = scene_data_dir(SCENE_ID)
"""Where ``setup_assets.py rooms`` puts the scene."""

ROOM_YAW = 0.0
"""Room yaw [deg]."""

ROOM_ORIGIN_AT = (2.35, 3.00)
"""The scene coordinate that becomes the task's origin [m]."""

HIDDEN = (
    "laundry_room_washing_machine_0",
    "laundry_room_washing_machine_1",
    "laundry_room_washing_machine_2",
    "laundry_room_washing_machine_3",
    "laundry_room_tumble_dryer_0",
    "laundry_room_tumble_dryer_1",
    "laundry_room_laundry_detergent_bottle_1",
    "laundry_room_dryer_sheets_box_1",
    "laundry_room_dryer_sheets_box_2",
    "laundry_room_lint_roller_0",
)
"""Objects hidden from the room. ``assets`` replaces the appliances with a
vendored mesh."""

ROOM_CFG = scenery_cfg_at(
    SCENE_DIR, ROOM_ORIGIN_AT, yaw_deg=ROOM_YAW, hide=HIDDEN, scene_id=SCENE_ID
)

SCENE_CAM_POS = (-2.00, -0.55, 2.60)
SCENE_CAM_ROT = (0.0, 0.35172667, 0.0, 0.93610274)
SCENE_CAM_FOCAL = 10.5
"""Third-person camera position [m], rotation (x, y, z, w) and focal length [mm].
Broadside to the carry, aimed at its middle."""

CEILING_SIZE = (8.1, 6.1)
CEILING_CENTRE = (1.65, 0.0)
CEILING_TOP_Z = 3.6
"""The room's footprint (x, y), its centre and its wall height, in the task frame [m]."""

CEILING_CFG = ceiling_cfg(CEILING_SIZE, CEILING_CENTRE, CEILING_TOP_Z)

