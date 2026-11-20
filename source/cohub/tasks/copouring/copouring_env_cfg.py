"""CoPouring: two G1s lift a tub of apples and tip it into the empty crate beside it.

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

from cohub.g1.cameras import look_at_rot, scene_camera_cfg
from cohub.g1.robots import PELVIS_SPAWN_Z, make_robot
from cohub.scenesmith.cfg import yaw_quat
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
    APPLE_CFGS,
    APPLE_MAX_ANGULAR_SPEED,
    APPLE_MAX_SPEED,
    APPLE_NAMES,
    APPLE_POSE_RANGE,
    BACKDROP_CFGS,
    BIN_CFG,
    BIN_INNER,
    BIN_POSE_RANGE,
    BIN_SIZE,
    LIFT_MARGIN,
    POUR_MIN_APPLES,
    POUR_TILT_DEG,
    SPILL_FLOOR_Z,
    SHUTTER_CFG,
    STAND_CRATE_CFGS,
    BIN_STAND_CRATES,
    STAND_CRATES,
    STAND_TOP_Z,
    TOTE_CFG,
    TOTE_POSE_RANGE,
)
from .scenesmith_room import ROOM_CFG

ROBOT_STANDOFF_X = 0.78
"""Distance from the tub's centre to each robot's pelvis, along world X [m]."""

ROBOT_ALONG_Y = 0.30
"""How far each robot stands toward the target crate, along world Y [m]."""

ROBOT_YAW_TOWARD_BIN = 0.0
"""How far each robot is turned toward the target crate, from facing straight
across the tub [deg]."""


G1_ROBOT_A_CFG: ArticulationCfg = make_robot(
    "Robot_A",
    pos=(-ROBOT_STANDOFF_X, ROBOT_ALONG_Y, PELVIS_SPAWN_Z),
    rot=yaw_quat(ROBOT_YAW_TOWARD_BIN),
)
G1_ROBOT_B_CFG: ArticulationCfg = make_robot(
    "Robot_B",
    pos=(ROBOT_STANDOFF_X, ROBOT_ALONG_Y, PELVIS_SPAWN_Z),
    rot=yaw_quat(180.0 - ROBOT_YAW_TOWARD_BIN),
)

SCENE_CAM_POS = (-1.75, 1.65, 1.75)
SCENE_CAM_AT = (0.0, 0.30, 0.40)
"""Third-person camera position and target, env-local [m]."""


# A configclass cannot loop: the scene declares one field per crate, and the
# counts must match ``assets.py``.
_DECLARED_STAND_CRATES = 3
_DECLARED_BIN_STAND_CRATES = 0
if (_DECLARED_STAND_CRATES, _DECLARED_BIN_STAND_CRATES) != (STAND_CRATES, BIN_STAND_CRATES):
    raise AssertionError(
        f"copouring scene declares {_DECLARED_STAND_CRATES} stand crates and "
        f"{_DECLARED_BIN_STAND_CRATES} bin-stand crates, but assets.py asks for "
        f"{STAND_CRATES} and {BIN_STAND_CRATES}. Add or remove the "
        f"`stand_crate_N` / `bin_stand_crate_N` fields in CoPouringSceneCfg to match, "
        f"and update the two counts above."
    )


@configclass
class CoPouringSceneCfg(G1DualSceneCfg):
    """Two G1s over a crate of apples, with the empty crate beside them."""

    robot_a: ArticulationCfg = G1_ROBOT_A_CFG
    robot_b: ArticulationCfg = G1_ROBOT_B_CFG

    # The tub, which the hands hold.
    object: RigidObjectCfg = TOTE_CFG

    # Apples, one rigid body each.
    apple_0: RigidObjectCfg = APPLE_CFGS[0]
    apple_1: RigidObjectCfg = APPLE_CFGS[1]

    # Kinematic target crate. The reset writes its pose.
    catch_bin: RigidObjectCfg = BIN_CFG

    # Stack under the tub. One field per crate, matching assets.STAND_CRATES.
    stand_crate_0: RigidObjectCfg = STAND_CRATE_CFGS[0]
    stand_crate_1: RigidObjectCfg = STAND_CRATE_CFGS[1]
    stand_crate_2: RigidObjectCfg = STAND_CRATE_CFGS[2]

    # SceneSmith grocery store.
    room: AssetBaseCfg = ROOM_CFG

    # The platform ground plane, not drawn. Its collider stays.
    ground: AssetBaseCfg = hidden_grid_cfg()

    # Slabs outside the walls and a fill for the floor-level opening.
    backdrop_x_neg: AssetBaseCfg = BACKDROP_CFGS[0]
    backdrop_x_pos: AssetBaseCfg = BACKDROP_CFGS[1]
    backdrop_y_neg: AssetBaseCfg = BACKDROP_CFGS[2]
    backdrop_y_pos: AssetBaseCfg = BACKDROP_CFGS[3]
    service_door: AssetBaseCfg = SHUTTER_CFG

    scene_cam = scene_camera_cfg(pos=SCENE_CAM_POS, rot=look_at_rot(SCENE_CAM_POS, SCENE_CAM_AT))


CoPouringPhysicalSceneCfg = physical_scene(CoPouringSceneCfg, TOTE_CFG, "Tote")


@configclass
class CoPouringEventsCfg:
    """Draws the pour station at each reset.

    The reset-to-spawn term must come first, or it would undo the draw. It is
    the kinematic-safe ``randomize_scene_to_default``.
    """

    reset_scene_to_default = EventTerm(func=mdp.randomize_scene_to_default, mode="reset")

    # One term for the tub, its apples and the target crate.
    randomize_station = EventTerm(
        func=mdp.randomize_pour_station,
        mode="reset",
        params={
            "tote_cfg": SceneEntityCfg("object"),
            "apple_names": APPLE_NAMES,
            "bin_cfg": SceneEntityCfg("catch_bin"),
            "tote_pose_range": TOTE_POSE_RANGE,
            "apple_pose_range": APPLE_POSE_RANGE,
            "tote_stand_names": tuple(f"stand_crate_{i}" for i in range(STAND_CRATES)),
            "bin_stand_names": tuple(f"bin_stand_crate_{i}" for i in range(BIN_STAND_CRATES)),
            "bin_pose_range": BIN_POSE_RANGE,
        },
    )


_SCORING = {
    "object_cfg": SceneEntityCfg("object"),
    "apple_names": APPLE_NAMES,
    "bin_cfg": SceneEntityCfg("catch_bin"),
    # The target crate's interior.
    "bin_inner": BIN_INNER,
    "bin_rim_z": BIN_SIZE[2],
    "pour_tilt_deg": POUR_TILT_DEG,
    "max_linear_speed": APPLE_MAX_SPEED,
    "max_angular_speed": APPLE_MAX_ANGULAR_SPEED,
}
"""Parameters shared by the success and the miss term."""


@configclass
class CoPouringTerminationsCfg(G1DualTerminationsCfg):
    """The task's own terminations.

    * ``success``: every apple is inside the target crate.
    * ``pour_missed``: an apple is on the floor outside the target crate.
    * ``object_dropped``: the tub fell more than 0.15 m below its stack.
    """

    object_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={
            # The tub's origin is its base, which rests at STAND_TOP_Z.
            "minimum_height": round(STAND_TOP_Z - 0.15, 3),
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    pour_missed = DoneTerm(
        func=mdp.PourMissed, params=dict(_SCORING, floor_z=SPILL_FLOOR_Z)
    )

    success = DoneTerm(
        func=mdp.PourSuccess,
        params={
            **_SCORING,
            "stand_top_z": STAND_TOP_Z,
            "lift_margin": LIFT_MARGIN,
            "min_apples": POUR_MIN_APPLES,
        },
    )


@configclass
class CoPouringG1x2EnvCfg(G1DualEnvCfg):
    """Two G1s lift a crate of apples and tip it into the empty crate beside it."""

    scene: CoPouringPhysicalSceneCfg = CoPouringPhysicalSceneCfg(
        num_envs=1, env_spacing=8.0, replicate_physics=True
    )
    observations: G1DualObservationsCfg = G1DualObservationsCfg()
    actions: G1DualActionsCfg = G1DualActionsCfg()
    events: CoPouringEventsCfg = CoPouringEventsCfg()
    terminations: CoPouringTerminationsCfg = CoPouringTerminationsCfg()

    episode_length_s: float = 90.0

    operator_briefing: str = """\
  The task: lift the tub of apples together and pour all of them into the
  empty crate beside you.

  Both robots, the same hand each: take the bar at your own end of the tub
  with one hand and close on it. It lies level across the end of the tub, so
  come down onto it from above, palm down. The tub is at hip height, just
  under where your wrists rest, so reach down for it rather than squatting
  to it.

  Lift together, level, until it is clear of the stack, and tip it into the
  empty crate on your side -- A's left, B's right. It is full to the rim, so
  it does not need turning over: about a quarter turn and the apples start to
  go. One of you leads the tip and the other follows it down.

  Every apple in the empty crate is the task: keep tipping until yours is
  empty, and the success check counts them.

  The tub starts in a different place on its stack every episode, and so does
  the empty one. Look before you reach, and look again before you tip: the
  target moves further than the crate does."""
