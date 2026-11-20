"""Two-G1 cooperative basket-carry environment.

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
from cohub.tasks._base.env_cfg import (
    G1DualActionsCfg,
    G1DualEnvCfg,
    G1DualObservationsCfg,
    G1DualSceneCfg,
    G1DualTerminationsCfg,
    physical_scene,
)
from cohub.tasks._base.props import hidden_grid_cfg
from .scenesmith_room import (
    CEILING_CFG,
    ROOM_CFG,
    SCENE_CAM_FOCAL,
    SCENE_CAM_POS,
    SCENE_CAM_ROT,
)

from . import mdp
from .assets import (
    BASE_HALF_SIZE,
    BOARD_CFG,
    CARRY_BASE_CLEARANCE,
    DOORWAY_SOUTH_CFG,
    DOORWAY_WEST_CFG,
    DRYER_BAY_CFGS,
    BOARD_SPAWN_POSE_RANGE,
    GOAL_BOARD_HEIGHT,
    GOAL_HEIGHT_TOLERANCE,
    GOAL_MARKER_CFG,
    GOAL_STAND_BASE_CFG,
    GOAL_STAND_COLUMN_CFG,
    GOAL_STAND_TOP_CFG,
    GOAL_TOLERANCE_X,
    GOAL_Y,
    GOAL_Y_TOLERANCE,
    PICKUP_Y,
    ROBOT_STANDOFF_X,
    STAND_BASE_CFG,
    STAND_COLUMN_CFG,
    STAND_TOP_CFG,
    WASHING_MACHINE_CFGS,
)

# A stands on the -X end facing +X, B on the +X end facing -X.
G1_ROBOT_A_CFG: ArticulationCfg = make_robot(
    "Robot_A",
    pos=(-ROBOT_STANDOFF_X, PICKUP_Y, PELVIS_SPAWN_Z),
    rot=(0.0, 0.0, 0.0, 1.0),
)
G1_ROBOT_B_CFG: ArticulationCfg = make_robot(
    "Robot_B",
    pos=(ROBOT_STANDOFF_X, PICKUP_Y, PELVIS_SPAWN_Z),
    rot=(0.0, 0.0, 1.0, 0.0),
)


@configclass
class CoCarrySceneCfg(G1DualSceneCfg):
    """Two G1s facing each other across a basket resting on a shelf."""

    robot_a: ArticulationCfg = G1_ROBOT_A_CFG
    robot_b: ArticulationCfg = G1_ROBOT_B_CFG

    object: RigidObjectCfg = BOARD_CFG
    # Pickup shelf
    stand_top: AssetBaseCfg = STAND_TOP_CFG
    stand_column: AssetBaseCfg = STAND_COLUMN_CFG
    stand_base: AssetBaseCfg = STAND_BASE_CFG

    # Goal shelf, with the green target painted on its top. The paint is visual only.
    goal_stand_top: AssetBaseCfg = GOAL_STAND_TOP_CFG
    goal_stand_column: AssetBaseCfg = GOAL_STAND_COLUMN_CFG
    goal_stand_base: AssetBaseCfg = GOAL_STAND_BASE_CFG
    goal_marker: AssetBaseCfg = GOAL_MARKER_CFG

    # SceneSmith laundromat. Its own appliances are hidden.
    room: AssetBaseCfg = ROOM_CFG

    # The room has no ceiling of its own.
    ceiling: AssetBaseCfg = CEILING_CFG

    # Washing machines along the south wall, as mesh and collider.
    washing_machine_0: AssetBaseCfg = WASHING_MACHINE_CFGS[0][0]
    washing_machine_1: AssetBaseCfg = WASHING_MACHINE_CFGS[1][0]
    washing_machine_2: AssetBaseCfg = WASHING_MACHINE_CFGS[2][0]
    washing_machine_3: AssetBaseCfg = WASHING_MACHINE_CFGS[3][0]
    washing_machine_4: AssetBaseCfg = WASHING_MACHINE_CFGS[4][0]
    washing_machine_5: AssetBaseCfg = WASHING_MACHINE_CFGS[5][0]
    washing_machine_collider_0: AssetBaseCfg = WASHING_MACHINE_CFGS[0][1]
    washing_machine_collider_1: AssetBaseCfg = WASHING_MACHINE_CFGS[1][1]
    washing_machine_collider_2: AssetBaseCfg = WASHING_MACHINE_CFGS[2][1]
    washing_machine_collider_3: AssetBaseCfg = WASHING_MACHINE_CFGS[3][1]
    washing_machine_collider_4: AssetBaseCfg = WASHING_MACHINE_CFGS[4][1]
    washing_machine_collider_5: AssetBaseCfg = WASHING_MACHINE_CFGS[5][1]

    # Two more on the north wall.
    dryer_bay_0: AssetBaseCfg = DRYER_BAY_CFGS[0][0]
    dryer_bay_1: AssetBaseCfg = DRYER_BAY_CFGS[1][0]
    dryer_bay_collider_0: AssetBaseCfg = DRYER_BAY_CFGS[0][1]
    dryer_bay_collider_1: AssetBaseCfg = DRYER_BAY_CFGS[1][1]

    # Fills for the room's two doorways.
    doorway_south: AssetBaseCfg = DOORWAY_SOUTH_CFG
    doorway_west: AssetBaseCfg = DOORWAY_WEST_CFG

    # The platform ground plane, not drawn. Its collider stays.
    ground: AssetBaseCfg = hidden_grid_cfg()

    scene_cam = scene_camera_cfg(
        pos=SCENE_CAM_POS, rot=SCENE_CAM_ROT, focal_length=SCENE_CAM_FOCAL
    )


@configclass
class CoCarryEventsCfg:
    """Draws the basket's start pose at each reset.

    ``reset_scene_to_default`` must come first, or it would undo the draw.
    """

    reset_scene_to_default = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    randomize_board = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": BOARD_SPAWN_POSE_RANGE,
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class CoCarryTerminationsCfg(G1DualTerminationsCfg):

    # Fires when the basket drops 0.139 m below the shelf top.
    object_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={
            "minimum_height": round(GOAL_BOARD_HEIGHT - 0.139, 3),
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=mdp.CarrySuccess,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "goal_distance": GOAL_Y,
            "goal_axis": 1,
            "goal_axis_tolerance": GOAL_Y_TOLERANCE,
            "goal_cross_axis_tolerance": GOAL_TOLERANCE_X,
            "goal_height": GOAL_BOARD_HEIGHT,
            "goal_height_tolerance": GOAL_HEIGHT_TOLERANCE,
            "max_linear_speed": 0.05,
            "max_angular_speed": 0.15,
            "max_tilt_deg": 15.0,
            "require_release": True,
            "release_settle_time": 0.20,
            "base_half_size": BASE_HALF_SIZE,
            "lift_margin_m": 0.03,
            "carry_clearance_m": CARRY_BASE_CLEARANCE,
            # ``lifted`` and ``carried_to_goal`` are scored by the eval's subtask chain.
            "require_carried_together": False,
        },
    )


CoCarryPhysicalSceneCfg = physical_scene(CoCarrySceneCfg, BOARD_CFG, "Board")


@configclass
class CoCarryG1x2EnvCfg(G1DualEnvCfg):
    """Two-operator cooperative carry: Quest-teleoped leader + follower."""

    scene: CoCarryPhysicalSceneCfg = CoCarryPhysicalSceneCfg(
        num_envs=1, env_spacing=6.0, replicate_physics=True
    )

    observations: G1DualObservationsCfg = G1DualObservationsCfg()
    actions: G1DualActionsCfg = G1DualActionsCfg()
    events: CoCarryEventsCfg = CoCarryEventsCfg()
    terminations: CoCarryTerminationsCfg = CoCarryTerminationsCfg()

    episode_length_s: float = 40.0

    operator_briefing: str = """\
  The task: carry the basket together to the green goal marker.

  Walk both robots up to the basket until the palms are within 10 cm of
  it, grasp with both hands, raise it a little, then strafe together
  2.1 m in -Y to the receiving shelf, lower it onto the green patch on
  its top, then open all four hands. Success requires the released
  basket to remain stable on the shelf.

  The robots face each other, so "strafe the same way in the world"
  means OPPOSITE controls: Robot A strafes to its right, Robot B to its
  left. That carries both toward the receiving shelf while keeping them
  clear of the two shelves."""
