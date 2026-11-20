"""Pouring: Robot A tips a marble out of a cup into the glass Robot B holds.

Low locomotion, low coupling (Easy). The two robots face each other across a
0.55 m table. Each picks up its own vessel.

Action layout (64D, fixed by action-term declaration order):

    [ A: left_wrist(7) right_wrist(7) hands(14) | A: loco(4) |
      B: left_wrist(7) right_wrist(7) hands(14) | B: loco(4) ]
"""

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from cohub.g1.cameras import look_at_rot, scene_camera_cfg
from cohub.g1.grasp import dex3_contact_filters
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
    BACKDROP_CFGS,
    DOOR_CFGS,
    BEAD_CFG,
    BEAD_LOST_HEIGHT,
    BEAD_POSE_RANGE,
    BEAD_SETTLE_HOLD_S,
    BEAD_SETTLE_SPEED,
    CATCH_MUG_CFG,
    CATCH_MUG_FLOOR_Z,
    CATCH_MUG_INNER_R,
    CATCH_MUG_INNER_TOP,
    CUP_POSE_RANGE,
    CUP_YAW_RANGE,
    LIFT_MARGIN,
    POUR_CUP_CFG,
    POUR_CUP_HEIGHT,
    POUR_CUP_INNER_R,
    POUR_CUP_OUTER_R,
    POUR_DWELL_S,
    POUR_TILT_DEG,
    ROBOT_STANDOFF_X,
    TABLE_CFGS,
    WORKTOP_Z,
)
from .scenesmith_room import ROOM_CFG

# A stands at -X and faces +X, B stands at +X and faces -X.
# ``rot`` is (x, y, z, w): identity faces world +X and (0, 0, 1, 0) faces -X.
G1_ROBOT_A_CFG: ArticulationCfg = make_robot(
    "Robot_A", pos=(-ROBOT_STANDOFF_X, 0.0, PELVIS_SPAWN_Z), rot=(0.0, 0.0, 0.0, 1.0)
)
G1_ROBOT_B_CFG: ArticulationCfg = make_robot(
    "Robot_B", pos=(ROBOT_STANDOFF_X, 0.0, PELVIS_SPAWN_Z), rot=(0.0, 0.0, 1.0, 0.0)
)

SCENE_CAM_POS = (0.52, -0.82, 1.26)
SCENE_CAM_AT = (0.0, 0.0, 0.84)
"""Where the third-person camera stands and what it looks at [m], env-local."""


@configclass
class PouringSceneCfg(G1DualSceneCfg):
    """Two G1s facing each other across a kitchen work table."""

    robot_a: ArticulationCfg = G1_ROBOT_A_CFG
    robot_b: ArticulationCfg = G1_ROBOT_B_CFG

    # ``object`` is Robot A's cup. Robot B never touches it, so its ``obs/grasp_latched`` stays false.
    object: RigidObjectCfg = POUR_CUP_CFG
    catch_mug: RigidObjectCfg = CATCH_MUG_CFG
    bead: RigidObjectCfg = BEAD_CFG

    # Robot B's hands against the mug. Publishes ``mug_in_hand``, gates nothing.
    mug_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/CatchMug",
        filter_prim_paths_expr=dex3_contact_filters(("Robot_B",)),
    )

    scene_cam = scene_camera_cfg(pos=SCENE_CAM_POS, rot=look_at_rot(SCENE_CAM_POS, SCENE_CAM_AT))

    table_top: AssetBaseCfg = TABLE_CFGS[0]
    table_leg_0: AssetBaseCfg = TABLE_CFGS[1]
    table_leg_1: AssetBaseCfg = TABLE_CFGS[2]
    table_leg_2: AssetBaseCfg = TABLE_CFGS[3]
    table_leg_3: AssetBaseCfg = TABLE_CFGS[4]

    room: AssetBaseCfg = ROOM_CFG

    # hidden but still collided with: the room brings its own floor
    ground: AssetBaseCfg = hidden_grid_cfg()

    # slabs outside the four walls, seen through the window and door openings
    backdrop_x_neg: AssetBaseCfg = BACKDROP_CFGS[0]
    backdrop_x_pos: AssetBaseCfg = BACKDROP_CFGS[1]
    backdrop_y_neg: AssetBaseCfg = BACKDROP_CFGS[2]
    backdrop_y_pos: AssetBaseCfg = BACKDROP_CFGS[3]

    # closes the doorway in the room's south wall
    door_leaf: AssetBaseCfg = DOOR_CFGS[0]
    door_casing_left: AssetBaseCfg = DOOR_CFGS[1]
    door_casing_right: AssetBaseCfg = DOOR_CFGS[2]
    door_casing_head: AssetBaseCfg = DOOR_CFGS[3]
    door_handle: AssetBaseCfg = DOOR_CFGS[4]


PouringPhysicalSceneCfg = physical_scene(PouringSceneCfg, POUR_CUP_CFG, "PourCup")


@configclass
class PouringEventsCfg:
    """The pour station, drawn fresh each reset.

    Terms run in declaration order. The scene reset must come first.
    """

    reset_scene_to_default = EventTerm(func=mdp.randomize_scene_to_default, mode="reset")

    randomize_station = EventTerm(
        func=mdp.randomize_pour_station,
        mode="reset",
        params={
            "cup_cfg": SceneEntityCfg("object"),
            "mug_cfg": SceneEntityCfg("catch_mug"),
            "bead_cfg": SceneEntityCfg("bead"),
            "cup_pose_range": CUP_POSE_RANGE,
            "bead_pose_range": BEAD_POSE_RANGE,
            "yaw_range": CUP_YAW_RANGE,
        },
    )


@configclass
class PouringTerminationsCfg(G1DualTerminationsCfg):
    """``time_out`` and the fallen terms come from the base. These are the task."""

    object_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={
            # the cup's origin is its base, which rests at WORKTOP_Z
            "minimum_height": round(WORKTOP_Z - 0.25, 3),
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    mug_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={
            "minimum_height": round(WORKTOP_Z - 0.25, 3),
            "asset_cfg": SceneEntityCfg("catch_mug"),
        },
    )

    bead_lost = DoneTerm(
        func=mdp.BeadLost,
        params={
            "bead_cfg": SceneEntityCfg("bead"),
            "mug_cfg": SceneEntityCfg("catch_mug"),
            "cup_cfg": SceneEntityCfg("object"),
            "inner_radius": CATCH_MUG_INNER_R,
            "inner_top": CATCH_MUG_INNER_TOP,
            "floor_thickness": CATCH_MUG_FLOOR_Z,
            "cup_inner_radius": POUR_CUP_INNER_R,
            "cup_rim_z": POUR_CUP_HEIGHT,
            "worktop_z": WORKTOP_Z,
            "lost_height": BEAD_LOST_HEIGHT,
            "max_linear_speed": BEAD_SETTLE_SPEED,
            "settle_hold_s": BEAD_SETTLE_HOLD_S,
        },
    )

    success = DoneTerm(
        func=mdp.PourSuccess,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "mug_cfg": SceneEntityCfg("catch_mug"),
            "bead_cfg": SceneEntityCfg("bead"),
            # the mug's interior
            "inner_radius": CATCH_MUG_INNER_R,
            "inner_top": CATCH_MUG_INNER_TOP,
            "floor_thickness": CATCH_MUG_FLOOR_Z,
            "worktop_z": WORKTOP_Z,
            "lift_margin": LIFT_MARGIN,
            "pour_tilt_deg": POUR_TILT_DEG,
            "dwell_s": POUR_DWELL_S,
            "cup_outer_radius": POUR_CUP_OUTER_R,
            "cup_height": POUR_CUP_HEIGHT,
            "require_cup_in_hand": True,
            # prints a line per env per step while the bead is in the mug but not scored
            "debug": False,
        },
    )


@configclass
class PouringG1x2EnvCfg(G1DualEnvCfg):
    """Robot A tips a marble out of a cup into the glass Robot B is holding.

    Both vessels are held by finger contact and have no handle. The cup is
    straight-sided, so the marble leaves it only when the cup is turned past
    upside down (``assets.POUR_ESCAPE_DEG``).
    """

    scene: PouringPhysicalSceneCfg = PouringPhysicalSceneCfg(
        num_envs=1, env_spacing=8.0, replicate_physics=True
    )
    observations: G1DualObservationsCfg = G1DualObservationsCfg()
    actions: G1DualActionsCfg = G1DualActionsCfg()
    events: PouringEventsCfg = PouringEventsCfg()
    terminations: PouringTerminationsCfg = PouringTerminationsCfg()

    episode_length_s: float = 90.0

    operator_briefing: str = """\
  The task: tip the marble out of your cup into the glass your partner is
  holding.

  Robot A: the squat drinking cup in front of you has one amber marble in
  it. There is no handle -- close your hand round the cup itself, the way you
  would round a mug. It stands in a different place on the table every episode,
  so look before you reach.

  Robot B: the tall glass in front of you is empty. Take it round its body,
  lift it off the table and hold it out over the middle of the table, upright
  and still. Keep it in the air: a marble that lands in a glass standing on the
  table does not count.

  Robot A: bring the cup over the mouth of your partner's glass, a hand's
  width above the rim, and turn your wrist over. The cup's sides are
  straight, so the marble does not come out until it is past upside down --
  keep turning until it has gone. Then bring the cup back level and set it
  down.

  The glass's mouth is 66 mm across and the marble is 30, so there is under
  twenty millimetres of aim either side. Line it up before you turn -- and hold
  the cup a little *short* of the glass, not over its middle: the marble rolls
  down the inside of the cup before it clears the lip, so it comes out moving
  away from you and lands about five centimetres beyond where the lip is
  pointing.

  Neither of you can reach the other's vessel: that is the point of the task,
  and it is measured.

  What is scored, in order: both vessels off the table at the same moment, then
  the cup turned past sixty degrees, then the marble in the glass -- and still
  in it half a second later -- while the glass is still in the air. It does
  not have to lie still -- a marble rolling round the bottom of the glass you
  are holding counts -- but keep the glass up and level for that half second,
  because a marble that bounces or is tipped back out before then has not
  landed. Robot B lifts first and keeps holding: a marble poured into a glass
  that has been set back down does not count, and the session log says so in
  as many words the moment the marble lands.

  If the marble ends up anywhere but in one of the two vessels -- on the table
  or on the floor -- the episode is over and you will get a fresh one."""
