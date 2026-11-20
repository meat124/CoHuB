"""Contact grasp: the fingers hold the object, the term reports it, and what a task needs for that to work."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass
import torch

from cohub.g1.robots import with_grip_material
from cohub.keys import ROBOTS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# --- Grasp term ---
# GraspAction applies nothing and has action_dim 0: the hands are driven by the upper-body terms. It
# reads a filtered contact sensor and keeps the `latched` history success terms use.

FINGERS = ("thumb", "index", "middle")
"""The Dex3's three fingers, in the order ``finger_closed`` stores them."""

PINCH_PAIRS = (("thumb", "index"), ("thumb", "middle"))
"""The two thumb pinch pairs, in the order ``obs/*_grasp`` stores them."""


DEX3_CONTACT_LINKS = (
    "palm",
    "index_0",
    "index_1",
    "middle_0",
    "middle_1",
    "thumb_0",
    "thumb_1",
    "thumb_2",
)
"""Every body of one Dex3 hand, as the suffix of ``<side>_hand_<name>_link``."""


DEX3_GRASP_QPOS = {
    "thumb_0": 0.00,
    "thumb_1": 1.00,
    "thumb_2": 1.25,
    "index_0": 1.20,
    "index_1": 1.20,
    "middle_0": 1.20,
    "middle_1": 1.20,
}
"""Joint angles of a closed Dex3 hand, as magnitudes per joint [rad]."""


def use_grasp_qpos_retargeting() -> None:
    """Raise the Quest retargeter's thumb scales to :data:`DEX3_GRASP_QPOS`.

    Sets class attributes of IsaacTeleop's
    ``TriHandMotionControllerRetargeter``, so it affects every pipeline built
    afterwards. Does nothing when ``isaacteleop`` is not installed.
    """
    try:
        from isaacteleop.retargeters.G1.trihand_motion_controller import (
            TriHandMotionControllerRetargeter,
        )
    except ModuleNotFoundError as exc:
        # ``isaacteleop`` is optional outside teleoperation.
        if exc.name == "isaacteleop":
            return
        raise

    TriHandMotionControllerRetargeter.THUMB_PROXIMAL_SCALE = DEX3_GRASP_QPOS["thumb_1"]
    TriHandMotionControllerRetargeter.THUMB_DISTAL_SCALE = DEX3_GRASP_QPOS["thumb_2"]


def dex3_contact_filters(
    robot_prims: tuple[str, ...] = ("Robot_A", "Robot_B"),
    sides: tuple[str, ...] = ("left", "right"),
) -> list[str]:
    """Contact-sensor filter paths for every Dex3 link, grouped hand by hand.

    Ordered by robot, side, then :data:`DEX3_CONTACT_LINKS`, so the force matrix
    reshapes into ``(envs, hands, links, 3)``. ``robot_prims`` are prim names
    (``Robot_A``). Paths are written in full: a wildcard does not match the
    instanced hand links in PhysX.
    """
    return [
        f"{{ENV_REGEX_NS}}/{prim}/{side}_hand/{side}_hand_{link}_link"
        for prim in robot_prims
        for side in sides
        for link in DEX3_CONTACT_LINKS
    ]


@configclass
class GraspActionCfg(ActionTermCfg):
    """Configuration for :class:`GraspAction`."""

    class_type: type[ActionTerm] | str = f"{__name__}:GraspAction"

    asset_name: str = "object"
    """The grasped asset. Also the asset the base class binds to."""

    contact_sensor_name: str = "grasp_contact"
    """Scene contact sensor on the object, filtered by :func:`dex3_contact_filters`."""

    contact_force_threshold: float = 0.5
    """Force on one hand link above which it counts as touching [N]."""

    min_contact_links: int = 2
    """How many of a hand's links must be touching to count as a grip."""

    grip_time: float = 0.04
    release_time: float = 0.08
    """Seconds of continuous contact before a grip latches, and of continuous
    non-contact before it releases."""

    robot_names: tuple[str, ...] = ROBOTS
    hand_term_names: tuple[str, ...] = ("upper_body_a", "upper_body_b")
    """Action terms carrying each robot's hand joint commands, in :attr:`robot_names` order."""

    attach_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    """One wrist body per hand, in the order every per-hand tensor uses."""

    close_threshold: float = 0.30
    open_threshold: float = 0.15
    """Hysteresis on one finger's mean |joint angle| [rad], for the ``obs/*_grasp`` observation."""

    verbose: bool = True


class GraspAction(ActionTerm):
    """Whether each hand is holding the object, read off finger contact.

    Applies nothing. Keeps the ``latched`` / ``ever_latched`` /
    ``release_age`` history that success terms read, and the measured grip
    force.
    """

    cfg: GraspActionCfg

    def __init__(self, cfg: GraspActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._object = env.scene[cfg.asset_name]
        self._robots = [env.scene[name] for name in cfg.robot_names]
        self._hands_per_robot = len(cfg.attach_body_names)
        self._num_hands = len(self._robots) * self._hands_per_robot

        n, h = self.num_envs, self._num_hands
        self.latched = torch.zeros(n, h, dtype=torch.bool, device=self.device)
        # Which fingers are closed, from measured joint angles. Ordered as FINGERS.
        self.finger_closed = torch.zeros(n, h, len(FINGERS), dtype=torch.bool, device=self.device)
        # Joint columns per (hand, finger), resolved lazily off the asset.
        self._finger_joint_ids: list[list[torch.Tensor]] | None = None
        # Episode-local hold history.
        self.ever_latched = torch.zeros(n, h, dtype=torch.bool, device=self.device)
        self.ever_both_robots_latched = torch.zeros(n, dtype=torch.bool, device=self.device)
        # Seconds since every hand in the scene let go.
        self.release_age = torch.zeros(n, device=self.device)
        # Seconds since each robot's own hands let go.
        self.robot_release_age = torch.zeros(n, len(self._robots), device=self.device)
        # Hardest grip since the hand took hold.
        self.peak_force = torch.zeros(n, h, device=self.device)
        self.peak_torque = torch.zeros(n, h, device=self.device)
        self.grasp_force = torch.zeros(n, h, 3, device=self.device)
        self.grasp_torque = torch.zeros(n, h, 3, device=self.device)

        self._empty = torch.zeros(n, 0, device=self.device)

        # How long each hand has been in, or out of, contact.
        self._contact_time = torch.zeros(n, h, device=self.device)
        self._no_contact_time = torch.zeros(n, h, device=self.device)
        self._links_per_hand = 0
        sensor = env.scene.sensors.get(cfg.contact_sensor_name)
        if sensor is None:
            raise RuntimeError(
                f"the grasp term needs a contact sensor named {cfg.contact_sensor_name!r} in the scene; "
                f"the scene has {sorted(env.scene.sensors)}. See GraspActionCfg.contact_sensor_name."
            )
        self._contact_sensor = sensor
        filters = len(sensor.cfg.filter_prim_paths_expr)
        if filters == 0 or filters % self._num_hands:
            raise RuntimeError(
                f"{cfg.contact_sensor_name!r} filters against {filters} paths, which does not divide "
                f"among this term's {self._num_hands} hands. Build the list with "
                "cohub.g1.grasp.dex3_contact_filters(), which groups them hand by hand in the "
                "order every per-hand tensor here uses."
            )

    # Properties

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._empty

    def hand_index(self, robot: str, side: str) -> int:
        """Column of one hand in the per-hand tensors.

        Args:
            robot: Scene entity name, as listed in ``cfg.robot_names``.
            side: ``"left"`` or ``"right"``, the prefix of a
                ``cfg.attach_body_names`` entry.
        """
        sides = [name.split("_")[0] for name in self.cfg.attach_body_names]
        if robot not in self.cfg.robot_names:
            raise ValueError(f"{robot!r} is not one of {self.cfg.robot_names}")
        if side not in sides:
            raise ValueError(f"{side!r} is not one of {sides}")
        return self.cfg.robot_names.index(robot) * self._hands_per_robot + sides.index(side)

    def hand_slice(self, robot: str) -> slice:
        """Columns of one robot's hands in the per-hand tensors."""
        if robot not in self.cfg.robot_names:
            raise ValueError(f"{robot!r} is not one of {self.cfg.robot_names}")
        start = self.cfg.robot_names.index(robot) * self._hands_per_robot
        return slice(start, start + self._hands_per_robot)

    # Operations

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.latched[env_ids] = False
        self.finger_closed[env_ids] = False
        self.ever_latched[env_ids] = False
        self.ever_both_robots_latched[env_ids] = False
        self.release_age[env_ids] = 0.0
        self.robot_release_age[env_ids] = 0.0
        self.peak_force[env_ids] = 0.0
        self.peak_torque[env_ids] = 0.0
        self.grasp_force[env_ids] = 0.0
        self.grasp_torque[env_ids] = 0.0
        self._contact_time[env_ids] = 0.0
        self._no_contact_time[env_ids] = 0.0
        return {}

    def process_actions(self, actions: torch.Tensor):
        """Refresh the grip/release decisions. Once per environment step."""
        self._update_contact_latches()

    def apply_actions(self):
        pass

    # Internals

    def _resolve_finger_joints(self) -> list[list[torch.Tensor]]:
        """Articulation joint ids per hand, per finger: ``[hand][finger]``."""
        if self._finger_joint_ids is None:
            sides = [name.split("_")[0] for name in self.cfg.attach_body_names]
            self._finger_joint_ids = []
            for robot in self._robots:
                names = list(robot.data.joint_names)
                for side in sides:
                    self._finger_joint_ids.append([
                        torch.tensor(
                            [i for i, n in enumerate(names) if n.startswith(f"{side}_hand_{finger}_")],
                            dtype=torch.long,
                            device=self.device,
                        )
                        for finger in FINGERS
                    ])
        return self._finger_joint_ids

    def update_finger_closure(self, hand: int) -> None:
        """Latch one hand's fingers open or closed from the measured joint angles.

        Called from :func:`grasp_closed` so the value matches the observation
        step. Idempotent. Updates ``finger_closed`` in place.
        """
        joints = self._resolve_finger_joints()
        q = self._robots[hand // self._hands_per_robot].data.joint_pos.torch
        for finger, ids in enumerate(joints[hand]):
            if ids.numel() == 0:
                continue
            bend = q[:, ids].abs().mean(dim=1)
            self.finger_closed[:, hand, finger] |= bend > self.cfg.close_threshold
            self.finger_closed[:, hand, finger] &= ~(bend < self.cfg.open_threshold)

    def _update_contact_latches(self) -> None:
        """Read the grip off the fingers, and apply nothing.

        A hand holds the object once ``min_contact_links`` of its links push harder
        than ``contact_force_threshold`` for ``grip_time``, and stops after
        ``release_time`` without that.
        """
        sensor = self._contact_sensor
        matrix = sensor.data.force_matrix_w
        if matrix is None:
            raise RuntimeError(
                f"{self.cfg.contact_sensor_name!r} reports no force matrix. Its "
                "filter_prim_paths_expr is empty, or its prim_path matches more than one prim "
                "per environment -- filtered reporting is one-to-many, so the sensor goes on "
                "the object and the fingers are the filters."
            )
        # (envs, 1 sensor body, filter shapes, 3), shapes grouped hand by hand.
        forces = matrix.torch[:, 0]
        if self._links_per_hand == 0:
            shapes = forces.shape[1]
            if shapes % self._num_hands:
                raise RuntimeError(
                    f"{self.cfg.contact_sensor_name!r} reports {shapes} filter shapes, which does not "
                    f"divide evenly among {self._num_hands} hands. Each hand's filter expression must "
                    "match the same number of collision shapes."
                )
            self._links_per_hand = shapes // self._num_hands
        per_hand = forces.view(self.num_envs, self._num_hands, self._links_per_hand, 3)
        magnitude = torch.linalg.norm(per_hand, dim=-1)

        touching = (magnitude > self.cfg.contact_force_threshold).sum(dim=-1)
        gripping = touching >= self.cfg.min_contact_links

        dt = self._env.step_dt
        self._contact_time = torch.where(gripping, self._contact_time + dt, torch.zeros_like(self._contact_time))
        self._no_contact_time = torch.where(
            gripping, torch.zeros_like(self._no_contact_time), self._no_contact_time + dt
        )

        new_latch = ~self.latched & (self._contact_time >= self.cfg.grip_time)
        released = self.latched & (self._no_contact_time >= self.cfg.release_time)
        if new_latch.any():
            self.latched |= new_latch
            self.ever_latched |= new_latch
            self._log(new_latch, "gripped")
        if released.any():
            self.latched &= ~released
            self._log(released, "let go")

        # One force per hand. The sensor reports no torque, so it stays zero.
        self.grasp_force = per_hand.sum(dim=2) * self.latched.unsqueeze(-1)
        self.grasp_torque = torch.zeros_like(self.grasp_torque)
        self.peak_force = torch.maximum(self.peak_force, torch.linalg.norm(self.grasp_force, dim=-1))

        self._update_hold_history()

    def _update_hold_history(self) -> None:
        """Update the episode-local hold history from ``latched``."""
        per_robot_latched = self.latched.view(
            self.num_envs, len(self._robots), self._hands_per_robot
        ).any(dim=2)
        # True once two or more robots hold the object at the same time.
        both_now = per_robot_latched.sum(dim=1) >= 2
        self.ever_both_robots_latched |= both_now
        fully_released = ~self.latched.any(dim=1)
        self.release_age = torch.where(
            fully_released,
            self.release_age + self._env.step_dt,
            torch.zeros_like(self.release_age),
        )
        self.robot_release_age = torch.where(
            ~per_robot_latched,
            self.robot_release_age + self._env.step_dt,
            torch.zeros_like(self.robot_release_age),
        )

    def _hand_name(self, hand: int) -> str:
        robot = self.cfg.robot_names[hand // self._hands_per_robot]
        side = self.cfg.attach_body_names[hand % self._hands_per_robot].split("_")[0]
        return f"{robot} {side} hand"

    def _log(self, mask: torch.Tensor, what: str) -> None:
        # The peak is cleared when a hand takes hold.
        if what == "gripped":
            self.peak_force = torch.where(mask, torch.zeros_like(self.peak_force), self.peak_force)
            self.peak_torque = torch.where(mask, torch.zeros_like(self.peak_torque), self.peak_torque)
        if not self.cfg.verbose:
            return
        for env_id, hand in mask.nonzero(as_tuple=False).tolist():
            detail = "" if what == "gripped" else f" (peak {self.peak_force[env_id, hand]:.0f} N)"
            print(f"[grasp] env {env_id}: {self._hand_name(hand)} {what}{detail}")


def grasp_term(env: ManagerBasedEnv) -> "GraspAction":
    """The task's grasp term. Raises when the task has no ``"grasp"`` action term."""
    return env.action_manager.get_term("grasp")


def grasp_latched(env: ManagerBasedEnv) -> torch.Tensor:
    """Observation term: per-hand hold state as floats. Shape (num_envs, 4).

    One channel per hand, in ``robot_names`` x ``attach_body_names`` order.
    ``1.0`` means the fingers have been on the object for at least
    :attr:`~GraspActionCfg.grip_time`.
    """
    return grasp_term(env).latched.float()


def grasp_closed(env: ManagerBasedEnv, robot: str, side: str) -> torch.Tensor:
    """Observation term: which pinch this hand is making. Shape (num_envs, 2).

    Column 0 is thumb-against-index, column 1 thumb-against-middle, each 1.0 or
    0.0, read from the measured joint angles.
    """
    term = env.action_manager.get_term("grasp")
    hand = term.hand_index(robot, side)
    term.update_finger_closure(hand)
    closed = term.finger_closed[:, hand]
    order = {name: i for i, name in enumerate(FINGERS)}
    return torch.stack(
        [closed[:, order[a]] & closed[:, order[b]] for a, b in PINCH_PAIRS], dim=-1
    ).float()


# --- Scene and actuator wiring ---
# A task needs a filtered contact sensor on the object, a graspable object, and use_physical_grasp
# (gains, pad material, thumb mapping).

REFERENCE_HAND_GAINS = {
    "stiffness": 15.0,
    "damping": 0.5,
    "armature": 0.01,
    "friction": 0.01,
}
"""Dex3 actuator gains, from GR00T-WholeBodyControl's Dex3 model.

:func:`use_physical_grasp` writes them onto every robot.
"""

DEX3_EFFORT_LIMIT = {
    ".*_thumb_0_joint": 5.0,
    ".*_(thumb_[12]|index_[01]|middle_[01])_joint": 8.0,
}
"""Torque limit per Dex3 finger joint [N.m].

Higher than the hardware's 1.4 N.m (2.45 on ``thumb_0``), because the
benchmark's objects are heavier than the real hand's 500 g rating.
"""

GRIP_FRICTION = 1.0
"""Default sliding friction for an object held by the fingers."""


def graspable(object_cfg, friction: float = GRIP_FRICTION):
    """The object cfg with contact reporting and grip friction, as :class:`GraspAction` needs."""
    spawn = object_cfg.spawn.replace(
        activate_contact_sensors=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=friction, dynamic_friction=friction
        ),
    )
    return object_cfg.replace(spawn=spawn)


def grasp_contact_sensor(object_prim: str, robot_prims=("Robot_A", "Robot_B")) -> ContactSensorCfg:
    """The contact sensor the grasp term reads: on the object, filtered by hand link.

    Filtered contact reporting needs one sensing prim per environment, so the
    sensor sits on the object. ``robot_prims`` are in the grasp term's own order.
    """
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/" + object_prim,
        filter_prim_paths_expr=dex3_contact_filters(robot_prims),
    )


def use_physical_grasp(env_cfg, robots=ROBOTS, grip_combine: str | None = None) -> None:
    """Set up the robots' hands to hold the task's object by contact.

    Call last from ``__post_init__``, on a scene that already has the sensor
    (:func:`grasp_contact_sensor`) and a :func:`graspable` object. ``grip_combine``
    is PhysX's friction combine mode for the pads (``None``: average).
    """
    for name in robots:
        robot = getattr(env_cfg.scene, name)
        hands = robot.actuators["hands"]
        for key, value in REFERENCE_HAND_GAINS.items():
            setattr(hands, key, value)
        # The two limits must be set together and be equal.
        hands.effort_limit = dict(DEX3_EFFORT_LIMIT)
        hands.effort_limit_sim = dict(DEX3_EFFORT_LIMIT)
        with_grip_material(robot, combine_mode=grip_combine)
    use_grasp_qpos_retargeting()
