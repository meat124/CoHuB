"""MoveHouse: Robot A opens the door; Robots B and C carry the table through it.

High locomotion, high coupling, three robots, in a downloaded house
(SceneSmith ``House/scene_098``).

* The doorway plane is x = 0, the bedroom is -X and the carry runs +X.
* B and C stand at the table's two short ends and carry it end-on through
  the 1.10 m doorway.
"""

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils.configclass import configclass

from cohub.g1.actions import upper_ik_action_cfg
from cohub.g1.cameras import scene_camera_cfg, VideoCutCfg
from cohub.g1.mdp import body_pos_w, G1TriplePolicyObsCfg
from cohub.g1.robots import PELVIS_SPAWN_Z, make_robot
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
    CARRY_GOAL_RADIUS,
    DOOR_CFG,
    DOOR_CLEAR_X,
    DOOR_CLEAR_ANGLE,
    DOOR_HANDLE_JOINT,
    DOOR_HINGE_JOINT,
    DROP_MIN_HEIGHT,
    FOOT_MARK_CFGS,
    FRONT_DOOR_CFG,
    FOOT_MARK_AT,
    FOOT_MARK_SIZE,
    GOAL_AT,
    HEIGHT_TOLERANCE,
    LEG_FEET,
    LIFT_MARGIN_M,
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_SPEED,
    MAX_TILT_DEG,
    OBJECT_CFG,
    RELEASE_SETTLE_TIME,
    STANDING_Z,
    STANDOFF,
    TABLE_SPAWN,
    TABLE_SPAWN_POSE_RANGE,
)
from .scenesmith_room import (
    CEILING_CFG,
    ROOM_CFG,
    SCENE_CAM_FOCAL,
    SCENE_CAM_2_POS,
    SCENE_CAM_2_ROT,
    SCENE_CAM_POS,
    SCENE_CAM_ROT,
)

# Robot A stands in the bedroom beside the door. B and C stand at the two ends of the table.
# ``rot`` is (x, y, z, w): identity faces world +X and (0, 0, 1, 0) faces -X.
G1_ROBOT_A_CFG: ArticulationCfg = make_robot(
    "Robot_A", pos=(-0.55, 0.65, PELVIS_SPAWN_Z), rot=(0.0, 0.0, 0.0, 1.0)
)
G1_ROBOT_B_CFG: ArticulationCfg = make_robot(
    "Robot_B",
    pos=(TABLE_SPAWN[0] + STANDOFF, TABLE_SPAWN[1], PELVIS_SPAWN_Z),
    rot=(0.0, 0.0, 1.0, 0.0),
)
G1_ROBOT_C_CFG: ArticulationCfg = make_robot(
    "Robot_C",
    pos=(TABLE_SPAWN[0] - STANDOFF, TABLE_SPAWN[1], PELVIS_SPAWN_Z),
    rot=(0.0, 0.0, 0.0, 1.0),
)
"""B stands at the table's +x end facing -X. C stands at the -x end facing +X."""


@configclass
class MoveHouseSceneCfg(G1TripleSceneCfg):
    """Three G1s, a console table, a door, and a downloaded house."""

    robot_a: ArticulationCfg = G1_ROBOT_A_CFG
    robot_b: ArticulationCfg = G1_ROBOT_B_CFG
    robot_c: ArticulationCfg = G1_ROBOT_C_CFG

    object: RigidObjectCfg = OBJECT_CFG

    # hangs at the task's origin, the doorway's centre
    door: ArticulationCfg = DOOR_CFG

    # Where the table's four feet have to land. One field per pad.
    foot_mark_0: AssetBaseCfg = FOOT_MARK_CFGS[0]
    foot_mark_1: AssetBaseCfg = FOOT_MARK_CFGS[1]
    foot_mark_2: AssetBaseCfg = FOOT_MARK_CFGS[2]
    foot_mark_3: AssetBaseCfg = FOOT_MARK_CFGS[3]

    # fills the front door, which opens onto nothing
    front_door: AssetBaseCfg = FRONT_DOOR_CFG

    room: AssetBaseCfg = ROOM_CFG

    ceiling: AssetBaseCfg = CEILING_CFG

    # hidden but still collided with: the room brings its own floor
    ground: AssetBaseCfg = hidden_grid_cfg()

    # bedroom view
    scene_cam = scene_camera_cfg(
        pos=SCENE_CAM_POS, rot=SCENE_CAM_ROT, focal_length=SCENE_CAM_FOCAL
    )

    # living room view, recorded for the whole episode as ``scene_2``
    scene_cam_2 = scene_camera_cfg(
        pos=SCENE_CAM_2_POS,
        rot=SCENE_CAM_2_ROT,
        focal_length=SCENE_CAM_FOCAL,
        prim_name="SceneCamera2",
    )


MoveHousePhysicalSceneCfg = physical_scene(
    MoveHouseSceneCfg, OBJECT_CFG, "Table", robot_prims=TRIPLE_ROBOT_PRIMS
)


@configclass
class MoveHouseActionsCfg(G1TripleActionsCfg):
    """The shared 96D layout, plus the zero-width door term after it.

    Both carriers get a raised posture cost to keep their elbows in through the doorway.
    """

    upper_body_b = upper_ik_action_cfg("robot_b", posture_cost=0.10)
    upper_body_c = upper_ik_action_cfg("robot_c", posture_cost=0.10)

    door_opening = mdp.DoorOpeningActionCfg(
        # Robot B leads the table through the doorway
        passage_robot_name="robot_b",
        passage_threshold=0.60,
        # the 1.15 rad default does not open this doorway wide enough for the table
        hinge_gesture_angle=1.55,
        hinge_open_threshold=DOOR_CLEAR_ANGLE,
    )


@configclass
class MoveHousePolicyObsCfg(G1TriplePolicyObsCfg):
    """The shared three-G1 group, plus six terms about the door."""

    door_joint_pos = ObsTerm(
        func=base_mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg("door", joint_names=[DOOR_HINGE_JOINT, DOOR_HANDLE_JOINT])},
    )
    door_joint_vel = ObsTerm(
        func=base_mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("door", joint_names=[DOOR_HINGE_JOINT, DOOR_HANDLE_JOINT])},
    )
    door_handle_pos = ObsTerm(
        func=body_pos_w,
        params={"asset_cfg": SceneEntityCfg("door"), "body_name": "door_handle"},
    )
    handle_latched = ObsTerm(func=mdp.door_handle_latched)
    door_open_state = ObsTerm(func=mdp.door_open_state)
    door_stage = ObsTerm(func=mdp.door_stage)


@configclass
class MoveHouseObservationsCfg(G1TripleObservationsCfg):

    policy: MoveHousePolicyObsCfg = MoveHousePolicyObsCfg()


@configclass
class MoveHouseEventsCfg:
    """Where the table starts, drawn fresh each reset.

    Terms run in declaration order. The scene reset must come first. The door
    is not randomised: it goes back to shut and latched.
    """

    reset_scene_to_default = EventTerm(func=mdp.randomize_scene_to_default, mode="reset")

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
class MoveHouseTerminationsCfg(G1TripleTerminationsCfg):
    """``time_out`` and the fallen terms come from the base. These two are the task."""

    object_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={
            "minimum_height": DROP_MIN_HEIGHT,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=mdp.MoveHouseSuccess,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "carriers": ("robot_b", "robot_c"),
            "goal": GOAL_AT,
            "feet": LEG_FEET,
            "marks": FOOT_MARK_AT,
            "mark_half_size": FOOT_MARK_SIZE / 2,
            "standing_height": STANDING_Z,
            "height_tolerance": HEIGHT_TOLERANCE,
            "max_tilt_deg": MAX_TILT_DEG,
            "max_linear_speed": MAX_LINEAR_SPEED,
            "max_angular_speed": MAX_ANGULAR_SPEED,
            "release_settle_time": RELEASE_SETTLE_TIME,
            "lift_margin_m": LIFT_MARGIN_M,
            "door_clear_x": DOOR_CLEAR_X,
            "carry_radius_m": CARRY_GOAL_RADIUS,
            "carry_clearance_m": CARRY_FOOT_CLEARANCE,
            "hinge_joint": DOOR_HINGE_JOINT,
            "require_release": True,
            "require_carried": False,
        },
    )


@configclass
class MoveHouseG1x3EnvCfg(G1TripleEnvCfg):
    """Two operators carry a console table; a third holds the door for them.

    The table is carried end-on, with one carrier walking backwards. Each carrier
    holds the two legs at its own end, under the apron. The carriers spawn a step
    back from the table. ``released`` is read from finger contact, so a hand that
    opens while it still touches a leg counts as holding.
    """

    scene: MoveHousePhysicalSceneCfg = MoveHousePhysicalSceneCfg(
        num_envs=1, env_spacing=10.0, replicate_physics=True
    )

    # Eval video cut: the bedroom view until Robot A is past the doorway wall, the living room after.
    video_cut: VideoCutCfg = VideoCutCfg(
        before="scene",
        after="scene_2",
        entity="robot_a",
        axis=0,
        past=DOOR_CLEAR_X,
    )
    observations: MoveHouseObservationsCfg = MoveHouseObservationsCfg()
    actions: MoveHouseActionsCfg = MoveHouseActionsCfg()
    events: MoveHouseEventsCfg = MoveHouseEventsCfg()
    terminations: MoveHouseTerminationsCfg = MoveHouseTerminationsCfg()

    episode_length_s: float = 120.0

    operator_briefing: str = """\
  The task: carry the table out of the bedroom, through the doorway, and set
  it down with its four legs on the four green squares in the living room.

  Robot A: you are at the door. Take the lever with one hand, push it down,
  and push the door AWAY from you into the living room -- not toward you: the
  other two are standing where it would swing. Push it all the way back
  against the wall. Half open is not open here: the doorway is 1.10 m and the
  table is 0.40 m deep with a robot walking it through, so a door that is
  only part way back leaves a gap it will not go through. Then stand out of
  the lane and stay there.

  Robots B and C: you start a step back from the table -- step in first. Then
  take the two legs at your own end, one in each hand, just under the
  tabletop: reach down beside the top, turn your fingers in toward the leg and
  close round it. Do not grab the tabletop. Lift together. You face each other
  along the line of the carry, so B walks backwards and C walks forwards -- B
  goes through the doorway first.

  You start half a pace to the far side of the doorway's centre line from A,
  so that A has room to work the lever. Drift back onto the centre line as you
  come up to the door -- 0.15 m over the length of the bedroom, so it is a
  lean rather than a turn. Do not carry that offset through the doorway: the
  open door lies back along the side you started on, and the gap it leaves you
  is on A's side of the opening. Keep the table level.

  Once you are through, the green squares are NOT straight ahead. They are
  against the far wall of the living room, to B's left and C's right -- about
  a metre and a half off the line you came in on, on the wall the bookshelf is
  not on. Side-step to them without turning the table: it goes down the way it
  was carried, long side along the wall.

  Set it down with each leg on its own green square -- that is what is
  scored: every foot inside its square, the table standing level. Then open
  all four hands AND PULL THEM CLEAR of the legs -- a hand that opens but
  still touches a leg counts as holding, and the episode will not score.
  Robot A's hand on the lever does not matter to the score; the two carriers'
  hands do."""
