"""TrashCollection's room: a downloaded SceneSmith laboratory and its placement.

The scene is not vendored. Fetch and prepare it once:

    python scripts/setup_assets.py rooms
"""

from cohub.scenesmith.cfg import scene_data_dir, scenery_cfg_at
from cohub.tasks._base.props import doorway_panel_cfg

SCENE_ID = "Room/scene_147"
"""A SceneSmith laboratory: 67 objects, 8.00 x 6.00 x 3.60 m.
Listed in ``cohub.scenesmith.REQUIRED_SCENES``."""

SCENE_DIR = scene_data_dir(SCENE_ID)
"""Where ``setup_assets.py rooms`` puts the scene."""

ROOM_YAW = 0.0
"""Room yaw [deg]. The task frame is the room frame shifted by ``ROOM_ORIGIN_AT``."""

ROOM_ORIGIN_AT = (2.87, 1.85)
"""The scene coordinate that becomes the task's origin [m]: the west end of the
bench, where Robot A holds the bin."""

HIDDEN = (
    # Stands on the bench in the strip Robot B sweeps.
    "microscope_0",
    # Stands where Robot B does.
    "lab_stool_2",
)
"""Objects hidden from the room."""

LOWERED_BY = 0.068
"""How far the two workbenches and everything on them are lowered [m].
Keep in step with ``assets.BENCH_TOP_Z``."""

LOWERED = (
    # Both benches
    "workbench_table",
    # Everything standing on them
    "spiral_bound_lab_notebook_0$",
    "clear_plastic_transfer_pipette",
    "petri_dish_0_s0_0",
    "lid_0_s0_1",
    "microscope_1",
    "glove_dispenser_box_0",
    "reagent_bottle_0$",
    "test_tube_stand",
    "beaker_0",
    "beaker_1",
)
"""Objects lowered with the benches. A trailing ``$`` anchors the name, so
objects with a longer name are left alone."""

ROOM_CFG = scenery_cfg_at(
    SCENE_DIR,
    ROOM_ORIGIN_AT,
    yaw_deg=ROOM_YAW,
    hide=HIDDEN,
    lower=LOWERED,
    lower_by=LOWERED_BY,
    scene_id=SCENE_ID,
)

DOORWAYS = {
    # name: the hole's extent in scene coordinates, ((x0, x1), (y0, y1), (z0, z1))
    "DoorSouth": ((5.994, 7.094), (-0.05, 0.05), (0.0, 2.2)),
    "DoorWest": ((-0.05, 0.05), (2.745, 3.845), (0.0, 2.2)),
}
"""The room's two doorways, in scene coordinates [m]."""

DOORS = {
    name: doorway_panel_cfg(name, opening, ROOM_ORIGIN_AT, ROOM_YAW)
    for name, opening in DOORWAYS.items()
}
"""One door leaf per doorway, in the task frame."""

SCENE_CAM_POS = (-2.50, 0.80, 2.75)
"""Third-person camera position in the task frame [m]."""
