"""The cfg classes every G1 task subclasses.

``configclass`` keeps base fields first, in declaration order, then the
subclass's. The action layout depends on that.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, JointWrenchSensorCfg
from isaaclab.utils.configclass import configclass

from cohub.g1.actions import lower_body_action_cfg, upper_ik_action_cfg
from cohub.g1.cameras import ego_camera_cfg, scene_camera_cfg
from cohub.g1.grasp import GraspActionCfg, graspable, grasp_contact_sensor, use_physical_grasp
from cohub.g1.mdp import (
    AgileObsRobotACfg,
    AgileObsRobotBCfg,
    AgileObsRobotCCfg,
    G1DualPolicyObsCfg,
    G1TriplePolicyObsCfg,
    apply_g1_dual_defaults,
    robot_fallen,
)

from cohub.keys import ROBOTS, ROBOTS3

from .props import GROUND_CFG, LIGHT_CFG

TRIPLE_ROBOT_PRIMS = ("Robot_A", "Robot_B", "Robot_C")
"""Prim names, in ``keys.ROBOTS3`` order."""


@configclass
class G1DualSceneCfg(InteractiveSceneCfg):
    """Sensors, cameras, ground and light.

    A subclass must fill in ``robot_a``, ``robot_b`` and ``object``.
    """

    robot_a: ArticulationCfg = None
    robot_b: ArticulationCfg = None
    object: RigidObjectCfg = None

    # Joint wrenches and foot contact, recorded in the dataset.
    wrench_a = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot_A")
    wrench_b = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot_B")
    contact_a = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot_A/.*_ankle_roll_link")
    contact_b = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot_B/.*_ankle_roll_link")

    # Per-hand contact, on the wrist link: PhysX reports no rigid body at
    # ``*_hand_palm_link``.
    hand_contact_a = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot_A/.*_wrist_yaw_link")
    hand_contact_b = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot_B/.*_wrist_yaw_link")

    # Removed by ``apply_g1_dual_defaults`` unless --enable_cameras.
    ego_cam_a = ego_camera_cfg("Robot_A")
    ego_cam_b = ego_camera_cfg("Robot_B")
    scene_cam = scene_camera_cfg()

    ground: AssetBaseCfg = GROUND_CFG
    light: AssetBaseCfg = LIGHT_CFG


@configclass
class G1DualActionsCfg:
    """Declaration order defines the 64D layout: A upper, A loco, B upper, B loco.

    A subclass may append a zero-width term but must not insert one among these.
    """

    upper_body_a = upper_ik_action_cfg("robot_a")
    lower_body_a = lower_body_action_cfg("robot_a", "lower_body_a")
    upper_body_b = upper_ik_action_cfg("robot_b")
    lower_body_b = lower_body_action_cfg("robot_b", "lower_body_b")

    # Zero-width: driven by the hand joints already in the action vector.
    grasp = GraspActionCfg()


@configclass
class G1DualObservationsCfg:
    """The recorded observation groups.

    A task that observes more subclasses ``G1DualPolicyObsCfg`` and overrides
    ``policy``.
    """

    policy: G1DualPolicyObsCfg = G1DualPolicyObsCfg()
    lower_body_a: AgileObsRobotACfg = AgileObsRobotACfg()
    lower_body_b: AgileObsRobotBCfg = AgileObsRobotBCfg()


@configclass
class G1DualTerminationsCfg:
    """Timeout and robot falls. A task must set ``object_dropped`` and ``success``."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    robot_a_fallen = DoneTerm(func=robot_fallen, params={"asset_cfg": SceneEntityCfg("robot_a")})
    robot_b_fallen = DoneTerm(func=robot_fallen, params={"asset_cfg": SceneEntityCfg("robot_b")})

    object_dropped: DoneTerm = None
    success: DoneTerm = None


def physical_scene(scene_cls, object_cfg, prim_name: str, robot_prims=None):
    """Return ``scene_cls`` with a graspable object and its grasp contact sensor.

    ``robot_prims`` must name every robot of a three-robot task. ``None``
    keeps the two-robot default.
    """

    @configclass
    class PhysicalSceneCfg(scene_cls):
        object: RigidObjectCfg = graspable(object_cfg)
        grasp_contact = (
            grasp_contact_sensor(prim_name)
            if robot_prims is None
            else grasp_contact_sensor(prim_name, robot_prims=robot_prims)
        )

    PhysicalSceneCfg.__name__ = f"{scene_cls.__name__}Physical"
    PhysicalSceneCfg.__qualname__ = PhysicalSceneCfg.__name__
    return PhysicalSceneCfg


@configclass
class G1DualEnvCfg(ManagerBasedRLEnvCfg):
    """The fields and ``__post_init__`` every two-G1 task cfg shares.

    A subclass sets ``scene``, ``observations``, ``actions``, ``events``,
    ``terminations``, ``episode_length_s`` and ``operator_briefing``.
    """

    commands = None
    rewards = None
    curriculum = None

    episode_length_s: float = 60.0
    """Seconds per episode."""

    lock_locomotion: bool = False
    """``True`` zeroes the robots' velocity commands so they stand still."""

    grip_combine: str | None = None
    """How PhysX combines the hand pad's friction with the object's.
    ``None`` keeps the platform default."""

    operator_briefing: str = ""
    """Printed verbatim to the operators by ``record_demos.py``. Never formatted."""

    robots: tuple[str, ...] = ROBOTS
    """The robots ``use_physical_grasp`` applies to."""

    def __post_init__(self):
        apply_g1_dual_defaults(self, episode_length_s=self.episode_length_s)
        use_physical_grasp(self, robots=self.robots, grip_combine=self.grip_combine)


# Three robots


@configclass
class G1TripleSceneCfg(G1DualSceneCfg):
    """:class:`G1DualSceneCfg` plus Robot C's articulation, sensors and camera.

    A subclass also fills in ``robot_c``. Build the grasp contact sensor with
    :func:`physical_scene` and ``robot_prims=TRIPLE_ROBOT_PRIMS``.
    """

    robot_c: ArticulationCfg = None

    wrench_c = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot_C")
    contact_c = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot_C/.*_ankle_roll_link")
    hand_contact_c = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot_C/.*_wrist_yaw_link")

    ego_cam_c = ego_camera_cfg("Robot_C")


@configclass
class G1TripleActionsCfg:
    """96D: three robots x 32, in declaration order.

    Not a subclass of :class:`G1DualActionsCfg`: base fields come first, which
    would put ``grasp`` ahead of Robot C's terms. A task may append a
    zero-width term but must not insert one among these six.
    """

    upper_body_a = upper_ik_action_cfg("robot_a")
    lower_body_a = lower_body_action_cfg("robot_a", "lower_body_a")
    upper_body_b = upper_ik_action_cfg("robot_b")
    lower_body_b = lower_body_action_cfg("robot_b", "lower_body_b")
    upper_body_c = upper_ik_action_cfg("robot_c")
    lower_body_c = lower_body_action_cfg("robot_c", "lower_body_c")

    # Zero-width. Sizes the hand tensors for three robots.
    grasp = GraspActionCfg(
        robot_names=ROBOTS3,
        hand_term_names=("upper_body_a", "upper_body_b", "upper_body_c"),
    )


@configclass
class G1TripleObservationsCfg:
    """The recorded observation groups at three robots."""

    policy: G1TriplePolicyObsCfg = G1TriplePolicyObsCfg()
    lower_body_a: AgileObsRobotACfg = AgileObsRobotACfg()
    lower_body_b: AgileObsRobotBCfg = AgileObsRobotBCfg()
    lower_body_c: AgileObsRobotCCfg = AgileObsRobotCCfg()


@configclass
class G1TripleTerminationsCfg(G1DualTerminationsCfg):
    """The two-robot terms plus Robot C's fall."""

    robot_c_fallen = DoneTerm(func=robot_fallen, params={"asset_cfg": SceneEntityCfg("robot_c")})


@configclass
class G1TripleEnvCfg(G1DualEnvCfg):
    """:class:`G1DualEnvCfg` with the grasp applied to all three robots."""

    robots: tuple[str, ...] = ROBOTS3
