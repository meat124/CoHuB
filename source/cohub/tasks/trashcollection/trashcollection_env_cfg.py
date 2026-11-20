"""TrashCollection: Robot A carries the bin to the bench, Robot B sweeps the
bottle off it into the bin.

1. Robot A picks the bin up off the floor.
2. A carries it to the west end of B's bench and turns to face along the bench.
3. A holds it with the mouth under the bench's lip (``assets.BIN_CARRY_AT``).
4. B sweeps the bottle west along the bench and over the lip into the bin.

The eval's subtask chain is ``lift -> deliver -> sweep``.
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

from . import mdp
from .assets import (
    BENCH_TOP_Z,
    BIN_CARRY_AT,
    BIN_HEIGHT,
    BIN_OFF_FLOOR,
    BIN_RIM_INNER,
    BIN_SPAWN_REGION,
    BIN_TIPPED_DEG,
    DELIVER_RADIUS,
    LITTER_CFGS,
    LITTER_FLOOR_Z,
    LITTER_NAMES,
    LITTER_SPAWN_REGION,
    OBJECT_CFG,
    RIM_ABOVE_BENCH_TOLERANCE,
    ROBOT_A_START,
    ROBOT_B_POS,
)
from cohub.tasks._base.props import hidden_grid_cfg
from .scenesmith_room import DOORS, ROOM_CFG, SCENE_CAM_POS

# ``rot`` is (x, y, z, w). A faces -Y down the approach line, B faces +Y into
# the bench.
G1_ROBOT_A_CFG: ArticulationCfg = make_robot(
    "Robot_A",
    pos=(ROBOT_A_START[0], ROBOT_A_START[1], PELVIS_SPAWN_Z),
    rot=(0.0, 0.0, -0.70710678, 0.70710678),
)
G1_ROBOT_B_CFG: ArticulationCfg = make_robot(
    "Robot_B",
    pos=(ROBOT_B_POS[0], ROBOT_B_POS[1], PELVIS_SPAWN_Z),
    rot=(0.0, 0.0, 0.70710678, 0.70710678),
)


@configclass
class TrashCollectionSceneCfg(G1DualSceneCfg):
    """2 G1s in a generated laboratory."""

    robot_a: ArticulationCfg = G1_ROBOT_A_CFG
    robot_b: ArticulationCfg = G1_ROBOT_B_CFG
    object: RigidObjectCfg = OBJECT_CFG

    # Litter: one field per piece, in ``assets.LITTER_NAMES`` order.
    litter_0: RigidObjectCfg = LITTER_CFGS[0]

    # SceneSmith laboratory. It brings the bench B sweeps.
    room: AssetBaseCfg = ROOM_CFG

    # Fills for the room's two doorways.
    door_south: AssetBaseCfg = DOORS["DoorSouth"]
    door_west: AssetBaseCfg = DOORS["DoorWest"]

    # The platform ground plane, not drawn. Its collider stays.
    ground: AssetBaseCfg = hidden_grid_cfg()

    scene_cam = scene_camera_cfg(pos=SCENE_CAM_POS)


TrashCollectionPhysicalSceneCfg = physical_scene(
    TrashCollectionSceneCfg, OBJECT_CFG, "Bin"
)

@configclass
class TrashCollectionEventsCfg:
    """Draws the bin's and the bottle's start pose at each reset.

    ``reset_scene_to_default`` must come first, or it would undo the draws.
    """

    reset_scene_to_default = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    randomize_bin = EventTerm(
        func=mdp.reset_region,
        mode="reset",
        params={"region": BIN_SPAWN_REGION, "asset_cfg": SceneEntityCfg("object")},
    )

    randomize_litter = EventTerm(
        func=mdp.reset_region,
        mode="reset",
        params={"region": LITTER_SPAWN_REGION, "asset_cfg": SceneEntityCfg("litter_0")},
    )


_IN_BIN = {
    "object_cfg": SceneEntityCfg("object"),
    "litter_names": LITTER_NAMES,
    "mouth_radius": BIN_RIM_INNER,
    "bin_height": BIN_HEIGHT,
}
"""The bin's mouth geometry, shared by ``success`` and ``litter_dropped``."""


@configclass
class TrashCollectionTerminationsCfg(G1DualTerminationsCfg):
    """The task's own terminations: bin dropped, litter dropped, bin tipped, success."""

    # The bin starts on the floor, so this only catches a fall through it.
    object_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": -0.30, "asset_cfg": SceneEntityCfg("object")},
    )

    litter_dropped = DoneTerm(
        func=mdp.litter_dropped, params={**_IN_BIN, "floor_z": LITTER_FLOOR_Z}
    )

    bin_tipped = DoneTerm(
        func=mdp.bin_tipped,
        params={"object_cfg": SceneEntityCfg("object"), "max_tilt_deg": BIN_TIPPED_DEG},
    )

    # The delivery and hold params feed the route flags, not the verdict.
    success = DoneTerm(
        func=mdp.TrashCollectionSuccess,
        params={
            **_IN_BIN,
            "deliver_at": BIN_CARRY_AT,
            "deliver_radius": DELIVER_RADIUS,
            "bench_top_z": BENCH_TOP_Z,
            "rim_tolerance": RIM_ABOVE_BENCH_TOLERANCE,
            "off_floor_m": BIN_OFF_FLOOR,
            "holder": "robot_a",
            "require_bin_held": True,
        },
    )


@configclass
class TrashCollectionG1x2EnvCfg(G1DualEnvCfg):
    """A carries the bin to B's bench and holds it under the lip; B sweeps."""

    scene: TrashCollectionPhysicalSceneCfg = TrashCollectionPhysicalSceneCfg(
        num_envs=1, env_spacing=8.0, replicate_physics=True
    )
    observations: G1DualObservationsCfg = G1DualObservationsCfg()
    actions: G1DualActionsCfg = G1DualActionsCfg()
    events: TrashCollectionEventsCfg = TrashCollectionEventsCfg()
    terminations: TrashCollectionTerminationsCfg = TrashCollectionTerminationsCfg()

    episode_length_s: float = 90.0

    operator_briefing: str = """\
  The task: Robot A brings the bin to the end of the bench and holds it there;
  Robot B sweeps the bottle off the bench into it.

  Robot A: the bin is on the floor straight ahead of you, in the open strip
  west of both benches. It is a tall one -- its rim is about at your hip. Step
  up to it, bend, and take it in both hands, one on each side.
  Then carry it straight ahead -- the way you are already facing, about two
  metres, past the near end of your partner's bench -- and turn a quarter
  LEFT, so you are looking back along the bench with the bin between you and
  its end.
  Hold the bin out in front of you with its rim a hand's width BELOW the
  bench top, and slide it in under the edge of the top until it touches the
  bench's frame -- the mouth is then under the lip, where the bottle will
  come off. Keep it upright: if it goes over the run ends. Then keep still
  until your partner says it is in.
  Robot B: wait at the bench. A plastic bottle lies on its side in front of
  you, between your hands. When the bin is at the end of the bench, sweep it
  ALONG the bench and over that end -- push it sideways, do not pick it up
  and do not pull it towards yourself. It ROLLS, so push straight and keep
  your hand behind it; if it goes off the near edge onto the floor the run
  ends. Tell your partner when it is in."""
