"""Scene props for the one-arm handover task: the counter, the bottle and the target.

Robot A picks the bottle up and offers it over the counter. Robot B takes it
and places it on the target. The robots stand on opposite sides of the counter:

    Y=+0.25  Robot A and the bottle start
    Y=-0.25  Robot B and the target

Bottle and target are each drawn from a ``SPAWN_SQUARE_SIDE`` square every reset.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

_DATA = Path(__file__).parent / "data"

# ---- counter
COUNTER_BASE_Z = 0.0
"""Height of the counter's base: the room's floor [m]."""

COUNTER_TOP_Z = 0.72
"""Height of the worktop [m]. Reach, grasp and spawn heights are solved against it."""

COUNTER_HEIGHT = COUNTER_TOP_Z - COUNTER_BASE_Z

# L-shaped corner counter, static, with triangle-mesh collision. See THIRD_PARTY_ASSETS.md.
COUNTER_SCALE = 0.008395
"""Base scale of the counter model. The three axis factors multiply it."""

COUNTER_HEIGHT_SCALE = COUNTER_HEIGHT / (92.91 * COUNTER_SCALE)
"""Z factor: brings the model's height to :data:`COUNTER_HEIGHT`."""
COUNTER_DEPTH_SQUEEZE = 0.415
"""Y factor: gives 0.230 m of worktop depth."""

COUNTER_LENGTH_SQUEEZE = 0.80
"""X factor: gives a 2.42 m long arm."""
COUNTER_ROT_DEG = (0.0, 0.0, 90.0)
"""Rotation about (X, Y, Z) [degrees]. Turns the long arm onto the task's Y."""
LAYOUT_X = 0.50
"""X of the counter's centreline [m]. Robots, bottle and target are placed relative to it."""

COUNTER_OFFSET = (LAYOUT_X, 0.0)
"""(X, Y) of the long arm's midpoint [m]."""

_COUNTER_Z = COUNTER_BASE_Z


def _counter_quat() -> tuple[float, float, float, float]:
    """(x, y, z, w) for COUNTER_ROT_DEG."""
    from pxr import Gf

    rx, ry, rz = COUNTER_ROT_DEG
    # composed Z, then Y, then X
    q = (
        Gf.Rotation(Gf.Vec3d(0, 0, 1), rz)
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry)
        * Gf.Rotation(Gf.Vec3d(1, 0, 0), rx)
    ).GetQuat()
    i = q.GetImaginary()
    return (i[0], i[1], i[2], q.GetReal())


COUNTER_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Counter",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(COUNTER_OFFSET[0], COUNTER_OFFSET[1], _COUNTER_Z),
        rot=_counter_quat(),
    ),
    spawn=UsdFileCfg(
        usd_path=str(_DATA / "corner_counter.usdc"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        scale=(
            COUNTER_SCALE * COUNTER_LENGTH_SQUEEZE,
            COUNTER_SCALE * COUNTER_DEPTH_SQUEEZE,
            COUNTER_SCALE * COUNTER_HEIGHT_SCALE,
        ),
    ),
)

# ---- layout (robots, bottle start, target)
# A is the west robot, B the east one. Each object shares its robot's Y.
ROBOT_A_Y = 0.25
ROBOT_B_Y = -0.25

SPAWN_SQUARE_SIDE = 0.05
"""Side of the square the bottle and the target are each drawn from [m]."""

SPAWN_SQUARE_HALF = SPAWN_SQUARE_SIDE / 2

ROBOT_X_OFFSET = 0.35  # counter centreline to each robot's pelvis [m]
ROBOT_A_POS = (LAYOUT_X - ROBOT_X_OFFSET, ROBOT_A_Y)  # west of the counter, faces +X
ROBOT_B_POS = (LAYOUT_X + ROBOT_X_OFFSET, ROBOT_B_Y)  # east of the counter, faces -X

# ---- bottle
# The visual meshes carry no collider. Two convex cylinders inside the USD collide.
BOTTLE_SCALE = 1.3
"""Uniform spawn scale on wine_bottle.usdc. The only number to change to resize the bottle."""

_BOTTLE_MODEL_HEIGHT = 0.22

BOTTLE_HEIGHT = _BOTTLE_MODEL_HEIGHT * BOTTLE_SCALE  # 0.286 m
BOTTLE_HALF_HEIGHT = BOTTLE_HEIGHT / 2
BOTTLE_MASS = 1.5
"""Bottle mass [kg]. Not scaled with BOTTLE_SCALE."""

_BOTTLE_USD_PATH = str(_DATA / "wine_bottle.usdc")

# Bottle centre at spawn, 2 cm above its resting height so it settles onto the counter.
BOTTLE_START = (LAYOUT_X, ROBOT_A_Y, COUNTER_TOP_Z + BOTTLE_HALF_HEIGHT + 0.02)

BOTTLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Bottle",
    init_state=RigidObjectCfg.InitialStateCfg(pos=BOTTLE_START, rot=(0.0, 0.0, 0.0, 1.0)),
    spawn=UsdFileCfg(
        usd_path=_BOTTLE_USD_PATH,
        # uniform: PhysX rejects a non-uniform scale on cylinder colliders
        scale=(BOTTLE_SCALE, BOTTLE_SCALE, BOTTLE_SCALE),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.25,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.0,
            disable_gravity=False,
            enable_gyroscopic_forces=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=BOTTLE_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.006, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
    ),
)

# Centre of the target square. ``mdp.randomize_target`` draws the episode's target around it.
TARGET_CENTER = (LAYOUT_X, ROBOT_B_Y, COUNTER_TOP_Z + BOTTLE_HALF_HEIGHT)

SPAWN_POSE_RANGE = {
    "x": (-SPAWN_SQUARE_HALF, SPAWN_SQUARE_HALF),
    "y": (-SPAWN_SQUARE_HALF, SPAWN_SQUARE_HALF),
}
"""Uniform offsets applied to the bottle's and the target's nominal centres [m]. Position only."""
# Per-axis placement tolerance [m]. The target pad is drawn from these, so the pad is the success box.
X_TOLERANCE = 0.06
Y_TOLERANCE = X_TOLERANCE
HEIGHT_TOLERANCE = 0.06

MARKER_Z = COUNTER_TOP_Z + 0.003
"""Height of the pad's own centre [m]."""

# Flat pad that shows the target. A RigidObject, so the reset event can move it and its pose is recorded.
# No collider. Dynamic with gravity off: a kinematic body rejects the velocity writes of a reset.
TARGET_MARKER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/TargetMarker",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(TARGET_CENTER[0], TARGET_CENTER[1], MARKER_Z)),
    spawn=sim_utils.CuboidCfg(
        # same size as the success box
        size=(2 * X_TOLERANCE, 2 * Y_TOLERANCE, 0.005),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        # no collider to derive a mass from
        mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.1, 0.75, 0.25), emissive_color=(0.05, 0.35, 0.1)
        ),
    ),
)
