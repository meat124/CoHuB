"""Two-G1 gallery picture hang: lift a framed painting and hook it on the wall.

Each robot takes one side rail of the frame with one hand. Together they lift
the picture and thread the loop on its top rail onto a wall peg. Nobody walks.
The hang wall, the peg and the picture are redrawn at every reset.
"""

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from cohub.g1.cameras import scene_camera_cfg
from cohub.g1.robots import PELVIS_SPAWN_Z, make_robot
from cohub.scenesmith.cfg import yaw_quat
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
    BUST_CFG,
    BUST_PLINTH_CFG,
    DOOR_HANDLE_CFG,
    DOOR_HEAD_CFG,
    DOOR_JAMB_LEFT_CFG,
    DOOR_JAMB_RIGHT_CFG,
    DOOR_LEAF_CFG,
    FRAME_CFG,
    GALLERY_CANVAS_CFGS,
    GALLERY_FLOOR_CFG,
    FRAME_NEIGHBOUR_LEFT_CFG,
    FRAME_NEIGHBOUR_RIGHT_CFG,
    HANG_X,
    ROOM_BASEBOARD_HANG_CFG,
    ROOM_HANG_WALL_CFG,
    STOOL_CFG,
    VASE_CFG,
    VASE_PLINTH_CFG,
    WALL_HOOK_CFG,
)
from cohub.tasks._base.props import hidden_grid_cfg

from .scenesmith_room import ROOM_CFG, SCENE_CAM_POS
from .geometry import (
    FRAME_CLEAR_RISE,
    FRAME_DROP_HEIGHT,
    FRAME_RESET_X_JITTER,
    FRAME_RESET_Y_JITTER,
    HOOK_X_JITTER,
    HOOK_Z_RANGE,
    WALL_Y_JITTER,
    ROBOT_Y,
    STANCE_DX,
    STANCE_YAW_DEG,
)


# Both robots face the hang wall (+Y), turned inward by STANCE_YAW_DEG.
G1_ROBOT_A_CFG: ArticulationCfg = make_robot(
    "Robot_A",
    pos=(HANG_X - STANCE_DX, ROBOT_Y, PELVIS_SPAWN_Z),
    rot=yaw_quat(90.0 - STANCE_YAW_DEG),
)
G1_ROBOT_B_CFG: ArticulationCfg = make_robot(
    "Robot_B",
    pos=(HANG_X + STANCE_DX, ROBOT_Y, PELVIS_SPAWN_Z),
    rot=yaw_quat(90.0 + STANCE_YAW_DEG),
)


@configclass
class FrameHangSceneCfg(G1DualSceneCfg):
    """A generated art gallery, one peg, and two G1s standing in front of the picture."""

    robot_a: ArticulationCfg = G1_ROBOT_A_CFG
    robot_b: ArticulationCfg = G1_ROBOT_B_CFG

    object: RigidObjectCfg = FRAME_CFG

    # Hang station. Kinematic rigid bodies, so the reset can move them.
    wall_hook: RigidObjectCfg = WALL_HOOK_CFG
    room_hang_wall: RigidObjectCfg = ROOM_HANG_WALL_CFG
    room_baseboard_hang: RigidObjectCfg = ROOM_BASEBOARD_HANG_CFG
    # On the hang wall, so they move with it.
    neighbour_painting_left: RigidObjectCfg = FRAME_NEIGHBOUR_LEFT_CFG
    neighbour_painting_right: RigidObjectCfg = FRAME_NEIGHBOUR_RIGHT_CFG

    # SceneSmith gallery, see scenesmith_room.py.
    room: AssetBaseCfg = ROOM_CFG

    # The room has its own floor, so the ground plane is hidden. It still collides.
    ground: AssetBaseCfg = hidden_grid_cfg()
    # Visual-only floor slab over the gallery's own floor.
    room_floor: AssetBaseCfg = GALLERY_FLOOR_CFG

    # Wall art replacing the room's own. ``canvas_south`` is on the hang wall.
    canvas_south: RigidObjectCfg = GALLERY_CANVAS_CFGS["canvas_south"]
    canvas_north_a: RigidObjectCfg = GALLERY_CANVAS_CFGS["canvas_north_a"]
    canvas_north_b: RigidObjectCfg = GALLERY_CANVAS_CFGS["canvas_north_b"]
    print_north: RigidObjectCfg = GALLERY_CANVAS_CFGS["print_north"]
    canvas_east: RigidObjectCfg = GALLERY_CANVAS_CFGS["canvas_east"]
    print_east: RigidObjectCfg = GALLERY_CANVAS_CFGS["print_east"]
    print_east_b: RigidObjectCfg = GALLERY_CANVAS_CFGS["print_east_b"]
    canvas_west: RigidObjectCfg = GALLERY_CANVAS_CFGS["canvas_west"]
    print_west: RigidObjectCfg = GALLERY_CANVAS_CFGS["print_west"]

    # Door closing the gallery's west doorway.
    west_door_leaf: AssetBaseCfg = DOOR_LEAF_CFG
    west_door_jamb_left: AssetBaseCfg = DOOR_JAMB_LEFT_CFG
    west_door_jamb_right: AssetBaseCfg = DOOR_JAMB_RIGHT_CFG
    west_door_head: AssetBaseCfg = DOOR_HEAD_CFG
    west_door_handle: AssetBaseCfg = DOOR_HANDLE_CFG

    # Props, all at least 1.2 m from either robot.
    bust_plinth: AssetBaseCfg = BUST_PLINTH_CFG
    bust: AssetBaseCfg = BUST_CFG
    vase_plinth: AssetBaseCfg = VASE_PLINTH_CFG
    vase: AssetBaseCfg = VASE_CFG
    stool: AssetBaseCfg = STOOL_CFG

    # Third-person camera behind the pair, on the hook's centreline, looking at
    # the hang wall.
    scene_cam = scene_camera_cfg(
        pos=SCENE_CAM_POS, rot=(-0.13052619, 0.13052619, 0.69636424, 0.69636424)
    )


@configclass
class FrameHangEventsCfg:
    """Reset events: restore the scene, then draw the hang station.

    ``reset_scene_to_default`` must run before ``randomize_station``. It uses
    the kinematic-safe term from ``_base.mdp``, because the peg and the hang
    wall reject velocity writes.
    """

    reset_scene_to_default = EventTerm(func=mdp.randomize_scene_to_default, mode="reset")

    randomize_station = EventTerm(
        func=mdp.randomize_station,
        mode="reset",
        params={
            # Everything mounted on the hang wall. A part left out stays behind
            # when the wall moves.
            "wall_parts": (
                "room_hang_wall",
                "room_baseboard_hang",
                "neighbour_painting_left",
                "neighbour_painting_right",
                "canvas_south",
            ),
            "hook_cfg": SceneEntityCfg("wall_hook"),
            "frame_cfg": SceneEntityCfg("object"),
            "x_jitter": HOOK_X_JITTER,
            "z_range": HOOK_Z_RANGE,
            "wall_y_jitter": WALL_Y_JITTER,
            "frame_x_jitter": FRAME_RESET_X_JITTER,
            "frame_y_jitter": FRAME_RESET_Y_JITTER,
        },
    )


@configclass
class FrameHangTerminationsCfg(G1DualTerminationsCfg):

    object_dropped = DoneTerm(
        func=mdp.frame_dropped,
        params={"object_cfg": SceneEntityCfg("object"), "minimum_height": FRAME_DROP_HEIGHT},
    )

    success = DoneTerm(
        func=mdp.HangSuccess,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "hook_cfg": SceneEntityCfg("wall_hook"),
            "max_tilt_deg": 35.0,
            "max_linear_speed": 0.20,
            "max_angular_speed": 0.80,
            "release_settle_time": 0.20,
            "lift_margin_m": 0.10,
        },
    )


FrameHangPhysicalSceneCfg = physical_scene(FrameHangSceneCfg, FRAME_CFG, "Frame")


@configclass
class FrameHangG1x2EnvCfg(G1DualEnvCfg):
    """Two operators lift a framed picture, carry it to the +Y wall, and hang it.

    The picture is held by the fingers, see ``cohub.g1.grasp``.

    Known limitation: the picture slides out of a closed hand on two of four hands.
    """

    scene: FrameHangPhysicalSceneCfg = FrameHangPhysicalSceneCfg(
        num_envs=1, env_spacing=8.0, replicate_physics=True
    )

    grip_combine: str = "max"
    """How PhysX combines the Dex3 pad's friction with the picture's. ``"max"`` uses the pad's own."""

    observations: G1DualObservationsCfg = G1DualObservationsCfg()
    actions: G1DualActionsCfg = G1DualActionsCfg()
    events: FrameHangEventsCfg = FrameHangEventsCfg()
    terminations: FrameHangTerminationsCfg = FrameHangTerminationsCfg()

    episode_length_s: float = 180.0

    operator_briefing: str = f"""\
  The task: the painting is standing on the floor in front of you, on its own
  feet a hand's width clear of the wall, under the bronze peg.  Between the two
  of you, lift it and hang it on that peg.

  There are no handles on it.  Take the edge of the frame on your own side
  with one hand -- the gilt moulding itself, high up rather than at the
  middle -- and close your hand onto the rail from outside it.  Nothing else on
  the painting can be held: not the face of it, not the canvas.

  One hand each is enough.  Hold it high on the picture, above the middle: it
  then hangs below your grips and stays level by itself.  Put your free hand on
  the front if you want it steadier, but the task does not need it.

  Lift about {FRAME_CLEAR_RISE * 100:.0f} cm -- enough for the loop on top of
  the frame to clear the knob on the end of the peg -- move it in toward the
  wall until the loop is over the peg, and lower it.  You will feel it drop the
  last few millimetres into the groove; that is what stops it sliding along.
  Then open both hands and let go: it only counts once the peg is holding it
  and neither of you is.

  Nobody walks and nobody turns.  You are both already where you need to be,
  facing the wall and turned slightly toward each other, and that is the stance
  that keeps the peg, your partner and both of your hands in your view for the
  whole job.  The peg is not in the same place every time -- it moves along the
  wall and up and down between attempts -- so look at where it actually is
  before you lift.  If one side goes up faster than the other the painting
  tilts, and you will see it tilt: that is the signal to wait rather than to
  push.
"""
