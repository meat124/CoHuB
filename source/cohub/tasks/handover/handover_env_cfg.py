"""Two-G1 one-arm kitchen handover.

Robot A picks the bottle up with its right hand and offers it over the counter.
Robot B takes it with its right hand and places it on the target. Only one robot
holds the bottle at a time. The bottle start and the target are drawn from a
0.05 m square every reset. See ``assets.py`` for the geometry.
"""

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from cohub.g1.cameras import scene_camera_cfg
from cohub.g1.robots import PELVIS_SPAWN_Z, make_robot
from cohub.tasks._base.env_cfg import (
    G1DualActionsCfg,
    G1DualEnvCfg,
    G1DualObservationsCfg,
    G1DualSceneCfg,
    G1DualTerminationsCfg,
    physical_scene,
)
from cohub.tasks._base.props import hidden_grid_cfg
from .scenesmith_room import DOOR_CFG, ROOM_CFG, SCENE_CAM_POS

from . import mdp
from .assets import (
    BOTTLE_CFG,
    COUNTER_CFG,
    COUNTER_TOP_Z,
    HEIGHT_TOLERANCE,
    MARKER_Z,
    ROBOT_A_POS,
    ROBOT_B_POS,
    SPAWN_POSE_RANGE,
    TARGET_CENTER,
    TARGET_MARKER_CFG,
    X_TOLERANCE,
    Y_TOLERANCE,
)


# Robot A stands west of the counter and faces +X. Robot B stands east and faces -X.
G1_ROBOT_A_CFG: ArticulationCfg = make_robot(
    "Robot_A", pos=(ROBOT_A_POS[0], ROBOT_A_POS[1], PELVIS_SPAWN_Z), rot=(0.0, 0.0, 0.0, 1.0)
)
G1_ROBOT_B_CFG: ArticulationCfg = make_robot(
    "Robot_B", pos=(ROBOT_B_POS[0], ROBOT_B_POS[1], PELVIS_SPAWN_Z), rot=(0.0, 0.0, 1.0, 0.0)
)


@configclass
class HandoverSceneCfg(G1DualSceneCfg):
    """Two G1s standing on either side of a single island counter, facing inward."""

    robot_a: ArticulationCfg = G1_ROBOT_A_CFG
    robot_b: ArticulationCfg = G1_ROBOT_B_CFG

    object: RigidObjectCfg = BOTTLE_CFG
    counter: AssetBaseCfg = COUNTER_CFG
    target_marker: RigidObjectCfg = TARGET_MARKER_CFG

    room: AssetBaseCfg = ROOM_CFG
    # closes the room's open doorway
    door: AssetBaseCfg = DOOR_CFG

    # hidden but still collided with: the room brings its own floor
    ground: AssetBaseCfg = hidden_grid_cfg()

    scene_cam = scene_camera_cfg(pos=SCENE_CAM_POS)


@configclass
class HandoverEventsCfg:
    """Where the bottle starts and where it has to end up, drawn fresh each reset.

    Terms run in declaration order. ``reset_scene_to_default`` must come first.
    """

    reset_scene_to_default = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    randomize_bottle = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": SPAWN_POSE_RANGE,
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    randomize_target = EventTerm(
        func=mdp.randomize_target,
        mode="reset",
        params={
            "pose_range": SPAWN_POSE_RANGE,
            "centre": TARGET_CENTER,
            "marker_z": MARKER_Z,
            "marker_cfg": SceneEntityCfg("target_marker"),
        },
    )


@configclass
class HandoverTerminationsCfg(G1DualTerminationsCfg):

    object_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": COUNTER_TOP_Z - 0.35, "asset_cfg": SceneEntityCfg("object")},
    )

    # ``target`` is the centre of the target square. The episode's target is read from the marker pad.
    success = DoneTerm(
        func=mdp.BottlePlacementSuccess,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target": TARGET_CENTER,
            "x_tolerance": X_TOLERANCE,
            "y_tolerance": Y_TOLERANCE,
            "height_tolerance": HEIGHT_TOLERANCE,
            "max_linear_speed": 0.05,
            "max_angular_speed": 0.15,
            "max_tilt_deg": 15.0,
            "release_settle_time": 0.20,
            # measured from the bottle's resting height on the counter
            "lift_margin_m": 0.03,
            # prints a per-step breakdown during the release window
            "debug": False,
        },
    )


HandoverPhysicalSceneCfg = physical_scene(HandoverSceneCfg, BOTTLE_CFG, "Bottle")


@configclass
class HandoverG1x2EnvCfg(G1DualEnvCfg):
    """One-arm bottle pick, pass and place between two standing G1s.

    A right pick, B right receive, B right place. The left arms are unused.
    The bottle is held by finger contact.
    """

    scene: HandoverPhysicalSceneCfg = HandoverPhysicalSceneCfg(
        num_envs=1, env_spacing=6.0, replicate_physics=True
    )

    observations: G1DualObservationsCfg = G1DualObservationsCfg()
    actions: G1DualActionsCfg = G1DualActionsCfg()
    events: HandoverEventsCfg = HandoverEventsCfg()
    terminations: HandoverTerminationsCfg = HandoverTerminationsCfg()

    episode_length_s: float = 60.0

    operator_briefing: str = """\
  The task: pass the bottle once, from A's RIGHT hand to B's RIGHT hand,
  then place it on the target with that same B right hand.

  Robot A: the bottle starts directly in front of you. Pick it up with your
  RIGHT hand, lift it at least 3 cm clear of the counter, and offer it over
  the middle. Release only after Robot B's right hand has taken it.
  Robot B: receive with your RIGHT hand and keep that hand on the bottle.
  Move it straight to the target directly in front of you, stand it upright,
  then open your right hand and leave it settled on the pad.

  The left arms are not used. The start and target still move every reset,
  within their own 5 cm squares, so look before reaching."""
