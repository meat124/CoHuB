"""CartService: Robot A pushes a loaded service cart to a dining table, Robot B serves the wine from it.

High locomotion, low coupling (Hard), in SceneSmith's restaurant. The bottle is
``object`` and the only scored body. ``grasp_cart`` is a zero-width grasp term
on the cart's handle, so the 64D action layout is unchanged.
"""

import isaaclab.envs.mdp as base_mdp

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from cohub.g1.cameras import scene_camera_cfg
from cohub.g1.grasp import GraspActionCfg, grasp_contact_sensor
from cohub.g1.robots import PELVIS_SPAWN_Z, make_robot
from cohub.tasks._base.props import hidden_grid_cfg, YAW_90, YAW_270
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
    BOARD_CFG,
    BOTTLE_CFG,
    BOTTLE_TOPPLED_DEG,
    CARRY_GOAL_RADIUS,
    CART_CFG,
    CART_MARK_CFG,
    CART_RIDERS,
    CART_SPAWN_REGION,
    DROPPED_BELOW_Z,
    GOAL_MARKER_CFG,
    GOAL_SPAWN_REGION,
    PLATE_CFG,
    PUSH_LENGTH,
    ROBOT_A_START,
    ROBOT_B_START,
    SERVE_WALK,
)
from .scenesmith_room import DOORS, ROOM_CFG, SCENE_CAM_POS, SCENE_CAM_ROT

# ``rot`` is (x, y, z, w). ``YAW_90`` faces +Y and ``YAW_270`` faces -Y.
# The robots face each other along the aisle, which runs along Y.
G1_ROBOT_A_CFG: ArticulationCfg = make_robot(
    "Robot_A", pos=(*ROBOT_A_START, PELVIS_SPAWN_Z), rot=YAW_270
)
G1_ROBOT_B_CFG: ArticulationCfg = make_robot(
    "Robot_B", pos=(*ROBOT_B_START, PELVIS_SPAWN_Z), rot=YAW_90
)


@configclass
class CartServiceSceneCfg(G1DualSceneCfg):
    """A loaded trolley in the aisle, a waiter at each end of it."""

    robot_a: ArticulationCfg = G1_ROBOT_A_CFG
    robot_b: ArticulationCfg = G1_ROBOT_B_CFG

    # The wine bottle. ``physical_scene`` below replaces it with the graspable
    # version and adds its sensor.
    object: RigidObjectCfg = BOTTLE_CFG

    cart: RigidObjectCfg = CART_CFG

    # Cargo: recorded, not scored.
    plate: RigidObjectCfg = PLATE_CFG
    board: RigidObjectCfg = BOARD_CFG

    # Goal patch on the dining table. Its size is the tolerance.
    goal_marker: RigidObjectCfg = GOAL_MARKER_CFG

    # Parking mark for the cart, painted on the floor.
    cart_mark: AssetBaseCfg = CART_MARK_CFG

    # SceneSmith restaurant, see scenesmith_room.py.
    room: AssetBaseCfg = ROOM_CFG

    # Panels closing the room's two doorways.
    door_north: AssetBaseCfg = DOORS["DoorNorth"]
    door_south: AssetBaseCfg = DOORS["DoorSouth"]

    # The room has its own floor, so the ground plane is hidden. It still collides.
    ground: AssetBaseCfg = hidden_grid_cfg()

    scene_cam = scene_camera_cfg(pos=SCENE_CAM_POS, rot=SCENE_CAM_ROT)


CartServicePhysicalSceneCfg = physical_scene(CartServiceSceneCfg, BOTTLE_CFG, "Bottle")
"""The scene with the bottle made graspable and the ``grasp_contact`` sensor on it.
The prim name must match ``BOTTLE_CFG``'s ``prim_path``."""


@configclass
class CartServiceFullSceneCfg(CartServicePhysicalSceneCfg):
    """Adds the contact sensor on the cart's handle.

    The sensor name must differ from the shared ``grasp_contact``, or the
    second grasp term reads the bottle's contacts.
    """

    cart_contact: ContactSensorCfg = grasp_contact_sensor("Cart")


@configclass
class CartServiceActionsCfg(G1DualActionsCfg):
    """The platform's 64D layout, plus a zero-width grasp term for the cart's handle.

    ``grasp_cart`` must come after ``grasp``. It commands nothing. It records
    whether Robot A's hands were on the handle.
    """

    grasp_cart = GraspActionCfg(
        asset_name="cart",
        contact_sensor_name="cart_contact",
    )


@configclass
class CartServiceEventsCfg:
    """Draws the cart's and the goal patch's pose at each reset.

    ``reset_scene_to_default`` must come first, or it would undo the draws.
    It is the kinematic-safe term, because the goal patch is kinematic.
    """

    reset_scene_to_default = EventTerm(func=mdp.randomize_scene_to_default, mode="reset")

    randomize_cart = EventTerm(
        func=mdp.reset_region,
        mode="reset",
        params={"region": CART_SPAWN_REGION, "asset_cfg": SceneEntityCfg("cart"), "riders": CART_RIDERS},
    )

    randomize_goal = EventTerm(
        func=mdp.reset_region,
        mode="reset",
        params={"region": GOAL_SPAWN_REGION, "asset_cfg": SceneEntityCfg("goal_marker")},
    )


@configclass
class CartServiceTerminationsCfg(G1DualTerminationsCfg):
    """The base terms plus two bottle failures and success.

    ``bottle_toppled`` catches a bottle lying on the deck or the table, which
    never falls below ``DROPPED_BELOW_Z``.
    """

    object_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": DROPPED_BELOW_Z, "asset_cfg": SceneEntityCfg("object")},
    )

    bottle_toppled = DoneTerm(
        func=mdp.bottle_toppled,
        params={"bottle_cfg": SceneEntityCfg("object"), "max_tilt_deg": BOTTLE_TOPPLED_DEG},
    )

    success = DoneTerm(
        func=mdp.CartServiceSuccess,
        params={"lifter": "robot_b", "require_lifter_hold": True, "carry_radius_m": CARRY_GOAL_RADIUS},
    )


@configclass
class CartServiceG1x2EnvCfg(G1DualEnvCfg):
    """CartService with two G1s: Robot A brings the cart, Robot B serves the wine from it."""

    scene: CartServiceFullSceneCfg = CartServiceFullSceneCfg(
        num_envs=1, env_spacing=14.0, replicate_physics=True
    )
    observations: G1DualObservationsCfg = G1DualObservationsCfg()
    actions: CartServiceActionsCfg = CartServiceActionsCfg()
    events: CartServiceEventsCfg = CartServiceEventsCfg()
    terminations: CartServiceTerminationsCfg = CartServiceTerminationsCfg()

    episode_length_s: float = 150.0

    operator_briefing: str = f"""\
  The task: a restaurant. Robot A is bringing a loaded service trolley down the
aisle, Robot B is the waiter meeting it, and between you a bottle of wine has
to end up on a diner's table.

A (the trolley): you start behind the cart, facing down the aisle, with the
push bar directly in front of you at about the height your hands are already
at. Take one step in and close a fist round the bar, one hand either side of
the middle -- there are 470 mm of clear steel between the two uprights, so
anywhere near the middle is the same grip, and the bar stands 130 mm above the
deck so there is air all round it. Then walk. Push it straight down the aisle
to the dining table on your left, about {PUSH_LENGTH:.1f} m, and stop when the
cart is standing on the **green rectangle painted on the floor** -- that is
your stop mark, and it is the whole of the aiming you have to do. Do not turn
the cart; the aisle is straight, the mark is straight ahead of you and 0.71 m
of the aisle is clear on the table side. Park it on the paint and stand there;
that is your whole job, and the episode cannot succeed until the trolley is
there.

B (the waiter): you start at the far end of the same aisle facing A. Walk up to
meet the trolley and stand square in front of its near end. Two things are
within reach on it, side by side: a plate of sandwiches on your left and the
wine on your right, standing in a steel collar. (A cheese board rides further
down the deck, out of reach and meant to stay there.)

**The wine is what counts.** Take it around the body just above the collar --
it is 75 mm across and there are 60 mm of straight glass between the collar and
the shoulder, so close your fist in that band -- and lift it straight up out of
the collar before you move it sideways. Then step round to your right to the near
edge of the table and stand it on the **green patch** painted on it, let it
settle and let go.

Leave the plate and the board where they are: neither has anything on it a
hand can close round, and the episode is scored on the wine.

Both of you: **nothing here has to stay balanced.** The load rides the cart,
the bottle sits in a collar that will not let it lean more than about 11 degrees,
and the only thing carried by hand is the bottle for the last {SERVE_WALK:.2f} m. What
does end an episode is the bottle on the floor, or the bottle on its side once
you have let go of it. The episode succeeds when the bottle is standing on the
green patch, upright, at rest, and both of you have let go."""
