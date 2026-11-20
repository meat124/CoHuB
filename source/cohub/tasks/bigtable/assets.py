"""BigTable's round table, its goal pad and the layout constants.

Task frame: the table spawns at the origin and the carry runs along world -Y.
The robots stand at :data:`STATION_BEARINGS` on a circle of radius
:data:`STANDOFF`, facing the table's centre. ``data/round_table.usda`` is baked:
a table constant changed here does not change it.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from pathlib import Path

from cohub.tasks._base.props import (
    GOAL_PAINT_COLOUR,
    GOAL_PAINT_EMISSIVE,
    GOAL_PAINT_THICKNESS,
    manipuland_collision_props,
    manipuland_rigid_props,
)

# Robot measurements

EGO_HEIGHT_ABOVE_PELVIS = 0.4739
"""Ego camera mount above the pelvis [m]."""

# Table
TABLE_DIAMETER = 1.36
"""Diameter of the table top [m]."""
R_TOP = TABLE_DIAMETER / 2

TOP_THICKNESS = 0.046
TOP_Z = 0.746
"""Top slab thickness, and the height of its upper face [m]."""

STANDING_Z = round(TOP_Z - TOP_THICKNESS / 2, 4)
"""Height of the table's origin, the centre of the top slab, when the table
stands on the floor [m]."""

TABLE_MASS = 7.5
"""Table mass [kg]."""

PEDESTAL_FOOT_RADIUS = 0.33
"""Radius of the pedestal's foot [m]."""

STATION_BEARINGS = (90.0, 210.0, 330.0)
"""Bearings of robots A, B and C around the table, in degrees ccw from +X in
the task frame."""


# Layout
STANDOFF = 0.98
"""Each robot's pelvis from the table's centre [m]."""

# Reset draw
SPAWN_JITTER = (0.04, 0.04)
SPAWN_YAW_JITTER = 4.0
"""Half-width of the table's spawn position draw (x, y) [m], and of its yaw
draw [deg]."""

SPAWN_POSE_RANGE = {
    "x": (-SPAWN_JITTER[0], SPAWN_JITTER[0]),
    "y": (-SPAWN_JITTER[1], SPAWN_JITTER[1]),
    "yaw": (-math.radians(SPAWN_YAW_JITTER), math.radians(SPAWN_YAW_JITTER)),
}
"""Pose range for ``reset_root_state_uniform``. Yaw in radians."""

# Goal
CARRY_DISTANCE = 1.60
"""How far the table is carried, along world -Y [m]."""

GOAL_AT = (0.0, -CARRY_DISTANCE, 0.0)
"""The pad's centre in the task frame [m]."""

GOAL_TOLERANCE = round(R_TOP, 4)
"""The green pad's radius [m]. Success needs the whole pedestal foot on the pad."""

LIFT_MARGIN = 0.03
"""How far the pedestal foot's lowest point must be off the floor, with all
three robots holding, for the lift to count [m]."""

CARRY_FOOT_CLEARANCE = 0.01
"""How far the pedestal foot's lowest point must be off the floor, with all
three robots holding and the centre on the pad, for ``carried_to_goal`` [m]."""

# Cfgs
_DATA = Path(__file__).parent / "data"
_TABLE_USD = str(_DATA / "round_table.usda")

OBJECT_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/RoundTable",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, STANDING_Z)),
    spawn=UsdFileCfg(
        usd_path=_TABLE_USD,
        rigid_props=manipuland_rigid_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=TABLE_MASS),
        collision_props=manipuland_collision_props(),
    ),
)
"""The table. The scene entity is named ``object`` and the prim ``RoundTable``."""


def _disc_cfg(prim_name, radius, height, pos, color, *, emissive=None, collision=False):
    """One kinematic disc, positioned by its centre.

    A mesh cylinder: the analytic ``UsdGeom.Cylinder`` renders as a 12-gon.
    """
    surface = {"diffuse_color": color, "roughness": 0.72}
    if emissive is not None:
        surface["emissive_color"] = emissive
    spawn_kwargs = {
        "radius": radius,
        "height": height,
        "visual_material": sim_utils.PreviewSurfaceCfg(**surface),
    }
    if collision:
        spawn_kwargs.update(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_name,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.MeshCylinderCfg(**spawn_kwargs),
    )


GOAL_MARKER_CFG: AssetBaseCfg = _disc_cfg(
    "GoalMarker",
    GOAL_TOLERANCE,
    GOAL_PAINT_THICKNESS,
    (GOAL_AT[0], GOAL_AT[1], GOAL_PAINT_THICKNESS / 2 + 0.004),
    GOAL_PAINT_COLOUR,
    emissive=GOAL_PAINT_EMISSIVE,
)
"""The green target disc on the floor, ``GOAL_TOLERANCE`` in radius. Visual only."""

SCENE_CAM_POS = (2.70, -0.80, 2.55)
SCENE_CAM_TARGET = (0.0, -0.80, 0.85)
SCENE_CAM_FOCAL = 12.0
"""Third-person camera position and target [m], and focal length [mm]."""
