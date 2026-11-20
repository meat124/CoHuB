"""TableAlign: both robots carry one table to the chair and turn it square to it.

Action layout (64D, fixed by action-term declaration order):

    [ A: left_wrist(7) right_wrist(7) hands(14) | A: loco(4) |
      B: left_wrist(7) right_wrist(7) hands(14) | B: loco(4) ]
"""

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from cohub.g1.cameras import scene_camera_cfg
from cohub.g1.robots import PELVIS_SPAWN_Z, make_robot
from cohub.tasks._base.props import hidden_grid_cfg
from cohub.tasks._base.env_cfg import (
    G1DualActionsCfg,
    G1DualEnvCfg,
    G1DualObservationsCfg,
    G1DualSceneCfg,
    G1DualTerminationsCfg,
    physical_scene,
)

from . import mdp
from .assets import (
    CARRY_FOOT_CLEARANCE,
    CHAIR_AT,
    CHAIR_FORWARD,
    DOOR_SOUTH_CFG,
    DOOR_WEST_CFG,
    DROP_MIN_HEIGHT,
    FOOT_MARK_CFGS,
    FORWARD_BAND,
    FRAME_DEPTH,
    FRAME_WIDTH,
    FRAME_X,
    HEIGHT_TOLERANCE,
    LATERAL_TOLERANCE,
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_SPEED,
    MAX_TILT_DEG,
    OBJECT_CFG,
    RELEASE_SETTLE_TIME,
    SQUARE_TOLERANCE_DEG,
    STANDING_Z,
    STANDOFF,
    TABLE_SPAWN_POSE_RANGE,
    WINDOW_EAST_CFG,
)
from .scenesmith_room import (
    CEILING_CFG,
    ROOM_CFG,
    SCENE_CAM_FOCAL,
    SCENE_CAM_POS,
    SCENE_CAM_ROT,
)

# A stands on the -X end facing +X, B on the +X end facing -X.
# ``rot`` is (x, y, z, w).
G1_ROBOT_A_CFG: ArticulationCfg = make_robot(
    "Robot_A", pos=(-STANDOFF, 0.0, PELVIS_SPAWN_Z), rot=(0.0, 0.0, 0.0, 1.0)
)
G1_ROBOT_B_CFG: ArticulationCfg = make_robot(
    "Robot_B", pos=(STANDOFF, 0.0, PELVIS_SPAWN_Z), rot=(0.0, 0.0, 1.0, 0.0)
)


@configclass
class TableAlignSceneCfg(G1DualSceneCfg):
    """Two G1s at the ends of a table, in a downloaded office."""

    robot_a: ArticulationCfg = G1_ROBOT_A_CFG
    robot_b: ArticulationCfg = G1_ROBOT_B_CFG

    object: RigidObjectCfg = OBJECT_CFG

    # Painted marks where the table's four feet have to land.
    foot_mark_0: AssetBaseCfg = FOOT_MARK_CFGS[0]
    foot_mark_1: AssetBaseCfg = FOOT_MARK_CFGS[1]
    foot_mark_2: AssetBaseCfg = FOOT_MARK_CFGS[2]
    foot_mark_3: AssetBaseCfg = FOOT_MARK_CFGS[3]

    # SceneSmith office. Its chair ``office_office_chair_2`` is the goal frame.
    room: AssetBaseCfg = ROOM_CFG

    # The room has no ceiling of its own.
    ceiling: AssetBaseCfg = CEILING_CFG

    # Fills for the room's openings.
    door_south: AssetBaseCfg = DOOR_SOUTH_CFG
    door_west: AssetBaseCfg = DOOR_WEST_CFG
    window_east: AssetBaseCfg = WINDOW_EAST_CFG

    # The platform ground plane, not drawn. Its collider stays.
    ground: AssetBaseCfg = hidden_grid_cfg()

    scene_cam = scene_camera_cfg(
        pos=SCENE_CAM_POS, rot=SCENE_CAM_ROT, focal_length=SCENE_CAM_FOCAL
    )


TableAlignPhysicalSceneCfg = physical_scene(TableAlignSceneCfg, OBJECT_CFG, "Table")


@configclass
class TableAlignEventsCfg:
    """Draws the table's start pose at each reset.

    ``reset_scene_to_default`` must come first, or it would undo the draw.
    """

    reset_scene_to_default = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    randomize_table = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": TABLE_SPAWN_POSE_RANGE,
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class TableAlignTerminationsCfg(G1DualTerminationsCfg):
    """The task's own terminations: table dropped and success."""

    object_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={
            "minimum_height": DROP_MIN_HEIGHT,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=mdp.TableAlignSuccess,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "chair_at": CHAIR_AT,
            "chair_forward": CHAIR_FORWARD,
            "forward_band": FORWARD_BAND,
            "lateral_tolerance": LATERAL_TOLERANCE,
            "standing_height": STANDING_Z,
            "height_tolerance": HEIGHT_TOLERANCE,
            "square_tolerance_deg": SQUARE_TOLERANCE_DEG,
            "max_tilt_deg": MAX_TILT_DEG,
            "max_linear_speed": MAX_LINEAR_SPEED,
            "max_angular_speed": MAX_ANGULAR_SPEED,
            "release_settle_time": RELEASE_SETTLE_TIME,
            "frame_x": FRAME_X,
            "frame_depth": FRAME_DEPTH,
            "frame_width": FRAME_WIDTH,
            "carry_clearance_m": CARRY_FOOT_CLEARANCE,
            "require_release": True,
            # ``lifted`` and ``carried_to_goal`` are scored by the eval's subtask chain.
            "require_lifted": False,
        },
    )


@configclass
class TableAlignG1x2EnvCfg(G1DualEnvCfg):
    """Two-operator cooperative carry with a quarter turn at the end of it.

    ``released`` is read from finger contact: a hand that rests on the table
    still counts as holding.
    """

    scene: TableAlignPhysicalSceneCfg = TableAlignPhysicalSceneCfg(
        num_envs=1, env_spacing=8.0, replicate_physics=True
    )
    observations: G1DualObservationsCfg = G1DualObservationsCfg()
    actions: G1DualActionsCfg = G1DualActionsCfg()
    events: TableAlignEventsCfg = TableAlignEventsCfg()
    terminations: TableAlignTerminationsCfg = TableAlignTerminationsCfg()

    episode_length_s: float = 60.0

    operator_briefing: str = """\
  The task: carry the table to the chair and set it down square to it.

  Both robots: take your own end of the table with both hands and lift
  together. Strafe 2.1 m in -Y, as in the basket carry --
  you face each other, so "the same way in the world" means OPPOSITE
  controls: Robot A strafes to its right, Robot B to its left.

  Then turn. The table has to end up across the lane rather than along
  it, with the office chair tucked into the middle of the long edge
  nearest it. Success is the table's position and angle RELATIVE TO THE
  CHAIR -- not the amount you turned it, and not the route you took.

  TURN EARLY. Stop about 1.5 m short of the chair and turn there, in the
  open aisle, then walk the table the rest of the way in -- one of you
  forward, one backward. Turning it on the spot at the goal swings a
  robot into the desk behind it.

  Set it down on its own four legs, centred on the green patch on the
  floor. Then open all four hands AND LIFT THEM CLEAR of the table --
  a hand that opens but stays resting on it still counts as holding,
  and the episode will not score. Success requires the released
  table to stay standing where you put it."""
