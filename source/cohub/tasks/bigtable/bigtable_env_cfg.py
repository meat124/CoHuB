"""BigTable: three G1s lift one round table together and carry it across a dining room.

Action layout (96D, fixed by action-term declaration order):

    [ A: left_wrist(7) right_wrist(7) hands(14) | A: loco(4) |
      B: left_wrist(7) right_wrist(7) hands(14) | B: loco(4) |
      C: left_wrist(7) right_wrist(7) hands(14) | C: loco(4) ]
"""

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from cohub.g1.cameras import look_at_rot, scene_camera_cfg
from cohub.g1.robots import PELVIS_SPAWN_Z, make_robot
from cohub.scenesmith.cfg import yaw_quat
from cohub.tasks._base.props import hidden_grid_cfg
from cohub.tasks._base.env_cfg import (
    TRIPLE_ROBOT_PRIMS,
    G1TripleActionsCfg,
    G1TripleEnvCfg,
    G1TripleObservationsCfg,
    G1TripleSceneCfg,
    G1TripleTerminationsCfg,
    physical_scene,
)

from . import mdp
from .assets import (
    CARRY_FOOT_CLEARANCE,
    GOAL_AT,
    SCENE_CAM_FOCAL,
    GOAL_MARKER_CFG,
    GOAL_TOLERANCE,
    LIFT_MARGIN,
    OBJECT_CFG,
    PEDESTAL_FOOT_RADIUS,
    SCENE_CAM_POS,
    SCENE_CAM_TARGET,
    SPAWN_POSE_RANGE,
    STANDING_Z,
    STANDOFF,
    STATION_BEARINGS,
)
from .scenesmith_room import DOOR_CFG, ROOM_CFG


def _station(prim_name: str, bearing_deg: float) -> ArticulationCfg:
    """One robot at ``bearing_deg`` on the ring, facing the table."""
    import math

    b = math.radians(bearing_deg)
    return make_robot(
        prim_name,
        pos=(STANDOFF * math.cos(b), STANDOFF * math.sin(b), PELVIS_SPAWN_Z),
        rot=yaw_quat(bearing_deg + 180.0),
    )


G1_ROBOT_A_CFG: ArticulationCfg = _station("Robot_A", STATION_BEARINGS[0])
G1_ROBOT_B_CFG: ArticulationCfg = _station("Robot_B", STATION_BEARINGS[1])
G1_ROBOT_C_CFG: ArticulationCfg = _station("Robot_C", STATION_BEARINGS[2])
"""A stands at 90 degrees, B at 210, C at 330. A walks the carry forwards,
B and C walk it backwards."""


@configclass
class BigTableSceneCfg(G1TripleSceneCfg):
    """Three G1s around a round table, in a downloaded dining room."""

    robot_a: ArticulationCfg = G1_ROBOT_A_CFG
    robot_b: ArticulationCfg = G1_ROBOT_B_CFG
    robot_c: ArticulationCfg = G1_ROBOT_C_CFG

    object: RigidObjectCfg = OBJECT_CFG

    # Green disc on the rug, visual only.
    goal_marker: AssetBaseCfg = GOAL_MARKER_CFG

    # SceneSmith dining room. Its own table is hidden.
    room: AssetBaseCfg = ROOM_CFG

    # Closes the doorway in the south wall.
    door: AssetBaseCfg = DOOR_CFG

    # The platform ground plane, not drawn. Its collider stays.
    ground: AssetBaseCfg = hidden_grid_cfg()

    scene_cam = scene_camera_cfg(
        pos=SCENE_CAM_POS,
        rot=look_at_rot(SCENE_CAM_POS, SCENE_CAM_TARGET),
        focal_length=SCENE_CAM_FOCAL,
    )


BigTablePhysicalSceneCfg = physical_scene(
    BigTableSceneCfg, OBJECT_CFG, "RoundTable", robot_prims=TRIPLE_ROBOT_PRIMS
)
"""``robot_prims`` names all three robots, so the grasp sensor covers Robot C's hands."""


@configclass
class BigTableEventsCfg:
    """Draws the table's start pose at each reset.

    ``reset_scene_to_default`` must come first, or it would undo the draw.
    """

    reset_scene_to_default = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    randomize_table = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": SPAWN_POSE_RANGE,
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class BigTableTerminationsCfg(G1TripleTerminationsCfg):
    """The task's own terminations: table dropped and success."""

    object_dropped = DoneTerm(
        func=mdp.table_dropped,
        params={"object_cfg": SceneEntityCfg("object")},
    )
    """Height or tilt. See ``mdp.table_dropped``."""

    success = DoneTerm(
        func=mdp.BigTableSuccess,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "goal": GOAL_AT[:2],
            "tolerance": GOAL_TOLERANCE,
            "standing_height": STANDING_Z,
            "lift_margin": LIFT_MARGIN,
            "foot_radius": PEDESTAL_FOOT_RADIUS,
            "carry_clearance_m": CARRY_FOOT_CLEARANCE,
            "require_release": True,
            # Without it two robots could deliver the table alone.
            "require_lifted_by_three": True,
        },
    )


@configclass
class BigTableG1x3EnvCfg(G1TripleEnvCfg):
    """Three robots lift one round table together and carry it to the marked spot."""

    scene: BigTablePhysicalSceneCfg = BigTablePhysicalSceneCfg(
        num_envs=1, env_spacing=8.0, replicate_physics=True
    )
    observations: G1TripleObservationsCfg = G1TripleObservationsCfg()
    actions: G1TripleActionsCfg = G1TripleActionsCfg()
    events: BigTableEventsCfg = BigTableEventsCfg()
    terminations: BigTableTerminationsCfg = BigTableTerminationsCfg()

    episode_length_s: float = 120.0

    operator_briefing: str = """\
  The task: all three robots lift the round table together and carry it 1.6 m
  across the room, onto the green disc on the rug.

  Take the edge of the table in front of you in both hands and say when you
  are ready. Lift on a count, together: the table has to come off the floor while
  all three of you are holding it, or the placement does not score however
  well it lands.

  The top is frosted rather than solid, so look through it: the green disc you
  are aiming at stays visible under the table for the whole carry, and so do
  the other two robots' feet.

  Put it down on the disc. A table left anywhere else does not count, and
  neither does one that came off the floor without all three of you holding
  it.

  Robot A walks straight forwards, facing the green disc, and sets the pace.
  B and C walk backwards and to the side -- B to its right, C to its left --
  and A can see where it is going when they cannot. Keep the top level, walk it
  onto the green disc, set it down together, and only then open your hands.
  Nobody lets go before the others."""
