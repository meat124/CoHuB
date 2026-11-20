"""Per-robot G1 action terms and their env-cfg factories."""

from __future__ import annotations

import copy
from dataclasses import MISSING
import os

from isaaclab.controllers.pink_ik import PinkIKController, NullSpacePostureTaskCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import (
    AgileBasedLowerBodyActionCfg,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.pink_controller_cfg import (
    G1_UPPER_BODY_IK_ACTION_CFG,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.mdp.actions import AgileBasedLowerBodyAction
import numpy as np
from pink.tasks import Task
import torch

from cohub.keys import action_order, hand_order


# --- Elbow clearance IK task ---
# Acts on each elbow's lateral offset from the torso midplane. cfg.elbow_clearance None or 0 turns it
# off.

TORSO_LINK = "torso_link"
ELBOW_LINKS = (("left_elbow_link", +1.0), ("right_elbow_link", -1.0))
"""Elbow link per side and the sign of its lateral (y) coordinate in the torso frame."""


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _resolve_frame(names: list[str], link: str) -> str:
    """URDF frame for a scene link name, matched on the suffix."""
    for name in names:
        if name == link or name.endswith("_" + link):
            return name
    raise ValueError(f"{link!r} is not a frame of the IK model")


class ElbowLateralTask(Task):
    r"""One-sided 1-D task on ``sign * y_torso(elbow)``, active only inside ``d_min``.

    Error :math:`e = s\,[R_t^T (p_f - p_t)]_y - d_{min}` while negative, zero
    otherwise. Frame Jacobians use Pink's LOCAL convention.
    """

    def __init__(self, frame: str, root: str, sign: float, d_min: float, cost: float, gain: float = 1.0,
                 lm_damping: float = 0.0) -> None:
        super().__init__(cost=float(cost), gain=float(gain), lm_damping=float(lm_damping))
        self.frame, self.root, self.sign, self.d_min = frame, root, float(sign), float(d_min)
        self.enabled = True

    def __repr__(self) -> str:
        return f"ElbowLateralTask({self.frame}, d_min={self.d_min}, cost={self.cost}, enabled={self.enabled})"

    def _relative(self, configuration):
        T_r = configuration.get_transform_frame_to_world(self.root)
        T_f = configuration.get_transform_frame_to_world(self.frame)
        p_rel = T_r.rotation.T @ (T_f.translation - T_r.translation)
        return T_r, T_f, p_rel

    def compute_error(self, configuration) -> np.ndarray:
        _, _, p_rel = self._relative(configuration)
        y_out = self.sign * p_rel[1]
        active = self.enabled and y_out < self.d_min
        return np.array([y_out - self.d_min if active else 0.0])

    def compute_jacobian(self, configuration) -> np.ndarray:
        T_r, T_f, p_rel = self._relative(configuration)
        if not (self.enabled and self.sign * p_rel[1] < self.d_min):
            return np.zeros((1, configuration.model.nv))
        J_f = configuration.get_frame_jacobian(self.frame)
        J_r = configuration.get_frame_jacobian(self.root)
        J_rel = T_r.rotation.T @ T_f.rotation @ J_f[:3] - J_r[:3] + _skew(p_rel) @ J_r[3:]
        return (self.sign * J_rel[1])[None, :]


def add_soft_elbow_clearance(controller: PinkIKController, d_min: float, cost: float) -> list[ElbowLateralTask]:
    """Append one :class:`ElbowLateralTask` per elbow to a live controller.

    The tasks must come after the frame tasks: the action term indexes its
    wrist poses by position in ``variable_input_tasks``.
    """
    frames = [f.name for f in controller.pink_configuration.full_model.frames]
    torso = _resolve_frame(frames, TORSO_LINK)
    gain = controller._variable_input_tasks[0].gain
    tasks = [ElbowLateralTask(_resolve_frame(frames, link), torso, sign, d_min, cost, gain=gain, lm_damping=75.0)
             for link, sign in ELBOW_LINKS]
    controller._variable_input_tasks.extend(tasks)
    controller.cfg.variable_input_tasks.extend(tasks)
    return tasks


# --- Action terms ---
# Subclasses of Isaac Lab's Agile lower-body and Pink IK upper-body actions that read the configured
# obs group and asset, so one term exists per robot. The lower-body term also implements the crouch.

SQUAT_CALIBRATION = (
    # (bias angle [rad], settled pelvis height [m])
    (0.00, 0.7225),
    (0.10, 0.6757),
    (0.15, 0.6494),
    (0.20, 0.6212),
    (0.25, 0.5913),
    (0.30, 0.5601),
    (0.35, 0.5278),
    (0.40, 0.4947),
    (0.45, 0.4608),
    (0.50, 0.4264),
    (0.55, 0.3925),
    (0.60, 0.3568),
)

SQUAT_BIAS_MAX = 0.55
"""Largest leg bias allowed [rad]."""

WALK_LOCK_BIAS = 0.05
"""Bias past which the velocity command is held at zero [rad].

The robot does not walk while crouched."""

SQUAT_SLEW = 0.6
"""Fastest the bias may change [rad/s]."""

SAFE_HIP_HEIGHT = 0.60
"""Lowest pelvis height at which the robot stays upright while reaching [m]."""


def squat_bias_for_height(height: float | np.ndarray) -> np.ndarray:
    """Leg bias angle [rad] that settles the pelvis at ``height`` [m].

    Interpolated from :data:`SQUAT_CALIBRATION` and clamped to
    :data:`SQUAT_BIAS_MAX`.
    """
    biases = np.array([b for b, _ in SQUAT_CALIBRATION])
    heights = np.array([h for _, h in SQUAT_CALIBRATION])
    # np.interp needs ascending x.
    bias = np.interp(height, heights[::-1], biases[::-1])
    return np.clip(bias, 0.0, SQUAT_BIAS_MAX)


class MultiRobotAgileLowerBodyAction(AgileBasedLowerBodyAction):
    """Agile lower-body action, per robot, with a crouch.

    Reads ``cfg.obs_group_name``. The hip-height command (action column 3) is
    implemented as a symmetric squat offset added to the policy's leg targets.
    The offset stops at :data:`SQUAT_BIAS_MAX`, and the velocity command is
    zeroed past :data:`WALK_LOCK_BIAS`.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        names = list(self._joint_names)

        def columns(pattern: str) -> list[int]:
            return [i for i, name in enumerate(names) if pattern in name]

        self._hip_pitch_cols = columns("hip_pitch")
        self._knee_cols = columns("knee")
        self._ankle_pitch_cols = columns("ankle_pitch")
        self._squat_bias = torch.zeros(self.num_envs, device=self.device)
        self._squat_enabled = bool(getattr(cfg, "enable_squat_bias", True))
        if self._squat_enabled and not (self._hip_pitch_cols and self._knee_cols):
            print(
                f"[{cfg.asset_name}] squat bias OFF - no hip_pitch/knee joints in "
                f"{names}"
            )
            self._squat_enabled = False

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            self._squat_bias.zero_()
        else:
            self._squat_bias[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        base_command = actions
        bias = None

        if self._squat_enabled:
            target = squat_bias_for_height(actions[:, 3].detach().cpu().numpy())
            target = torch.as_tensor(target, dtype=self._squat_bias.dtype, device=self.device)
            slew = SQUAT_SLEW * self._env.step_dt
            self._squat_bias += torch.clamp(target - self._squat_bias, -slew, slew)
            bias = self._squat_bias

            crouching = bias > WALK_LOCK_BIAS
            if bool(crouching.any()):
                # Zero the velocity command the policy sees.
                base_command = actions.clone()
                base_command[crouching, 0:3] = 0.0

        obs_tensor = self._env.obs_buf[self._obs_group_name]

        policy_input = self._compose_policy_input(base_command, obs_tensor)
        joint_actions = self._policy.forward(policy_input)

        self._raw_actions[:] = joint_actions
        self._processed_actions = joint_actions * self._policy_output_scale + self._policy_output_offset

        if bias is not None:
            self._processed_actions = self._processed_actions + self._squat_offset(bias)

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    # Internals

    def _squat_offset(self, bias: torch.Tensor) -> torch.Tensor:
        """Per-joint offset for a symmetric squat of ``bias`` radians."""
        offset = torch.zeros_like(self._processed_actions)
        column = bias.unsqueeze(-1)
        offset[:, self._hip_pitch_cols] = -column
        offset[:, self._knee_cols] = 2.0 * column
        offset[:, self._ankle_pitch_cols] = -column
        return offset

    @property
    def squat_bias(self) -> torch.Tensor:
        """Current leg bias [rad]."""
        return self._squat_bias


@configclass
class MultiRobotAgileLowerBodyActionCfg(AgileBasedLowerBodyActionCfg):
    class_type: type[MultiRobotAgileLowerBodyAction] | str = f"{__name__}:MultiRobotAgileLowerBodyAction"

    enable_squat_bias: bool = True
    """Implement the hip-height command with a leg offset. Turn off for upstream behaviour."""


IK_TIME_CONSTANT = 0.090
"""Time the arms take to close 63% of a pose error, in simulated seconds.

The Pink task gain is derived from it by :func:`normalised_task_gain`."""


def normalised_task_gain(
    step_dt: float,
    time_constant: float = IK_TIME_CONSTANT,
    solves_per_step: int = 4,
) -> float:
    """Per-solve Pink task gain giving a ``time_constant`` [s] arm response.

    Each solve multiplies the remaining error by ``1 - gain``, so over
    ``solves_per_step`` solves in a step of ``step_dt`` we want

        (1 - gain) ** solves_per_step == exp(-step_dt / time_constant)
    """
    solves = max(1, int(solves_per_step))
    return float(1.0 - np.exp(-step_dt / (solves * time_constant)))


_LIMIT_TOL = 1e-6
"""How near zero a joint limit still counts as zero [rad]."""


class MultiRobotPinkIKAction(PinkInverseKinematicsAction):
    """Pink IK action, per robot, with a time-constant gain and a tracking readout.

    Differs from upstream: the articulation comes from ``cfg.asset_name``, the task
    gain from the control period (:data:`IK_TIME_CONSTANT`), :attr:`tracking_error`
    reports each wrist's distance from its command, and hand commands are signed
    by each joint's travel direction.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self._step_dt = float(env.step_dt)
        # Solves per control step. Counted at run time; decimation is the first estimate.
        self._solves_per_step = max(1, int(getattr(env.cfg, "decimation", 1)))
        self._solves_this_step = 0
        self._gain_solves = 0

        # Wrist bodies, in the order of the action's 7D pose blocks.
        body_names = self._asset.data.body_names
        self._eef_body_ids = [body_names.index(link) for link in self.cfg.target_eef_link_names.values()]
        self._commanded_eef_pos = torch.zeros(self.num_envs, len(self._eef_body_ids), 3, device=self.device)
        self._tracking_error = torch.zeros(self.num_envs, len(self._eef_body_ids), device=self.device)
        self._have_commanded = False
        # Resolved on first use, once the joint limits are available.
        self._hand_signs: torch.Tensor | None = None

    def _initialize_ik_controllers(self) -> None:
        assert self._env.num_envs > 0, "Number of environments specified are less than 1."

        robot_cfg = getattr(self._env.scene.cfg, self.cfg.asset_name)
        clearance = getattr(self.cfg, "elbow_clearance", None)
        self._ik_controllers = []
        for _ in range(self._env.num_envs):
            kwargs = dict(
                cfg=self.cfg.controller.copy(),
                robot_cfg=robot_cfg,
                device=self.device,
                controlled_joint_indices=self._isaaclab_controlled_joint_ids,
            )
            controller = PinkIKController(**kwargs)
            if clearance is not None and clearance > 0:

                add_soft_elbow_clearance(controller, clearance, getattr(self.cfg, "elbow_clearance_cost", 10.0))
            self._ik_controllers.append(controller)

    # Operations

    def process_actions(self, actions: torch.Tensor) -> None:
        self._measure_tracking()
        if self.cfg.constant_time_constant:
            self._retune_gains()
        self._solves_this_step = 0
        super().process_actions(actions)
        # Rebind, do not write in place: the parent's tensor is a view on ``actions``.
        self._target_hand_joint_positions = self._orient_hand_targets(self._target_hand_joint_positions)
        for slot in range(len(self._eef_body_ids)):
            start = slot * self.pose_dim
            self._commanded_eef_pos[:, slot] = actions[:, start : start + self.position_dim]
        self._have_commanded = True

    def commanded_wrist_positions(self) -> torch.Tensor:
        """(num_envs, num_wrists, 3) commanded wrist positions, pelvis frame."""
        return self._commanded_eef_pos

    def apply_actions(self) -> None:
        self._solves_this_step += 1
        super().apply_actions()

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        # Do not compare a command from before the reset with the reset pose.
        self._have_commanded = False
        if env_ids is None:
            self._tracking_error.zero_()
        else:
            self._tracking_error[env_ids] = 0.0

    # Readout

    @property
    def tracking_error(self) -> torch.Tensor:
        """Per-wrist distance from the last commanded position [m].

        Shape ``(num_envs, 2)``, left then right, in the order of
        ``cfg.target_eef_link_names``. Zero after a reset until a command has
        been issued and one step has passed.
        """
        return self._tracking_error

    # Internals

    def _hand_closing_signs(self) -> torch.Tensor:
        """Which way each hand joint has to move to curl, from its own limits.

        ``+1`` or ``-1`` where the joint can travel in one direction only,
        ``0`` where it can travel both ways. The G1's hands are mirrored, so
        the same joint closes in opposite directions on the two hands.
        """
        if self._hand_signs is None:
            limits = self._asset.data.joint_pos_limits.torch[0, self._hand_joint_ids]
            lower, upper = limits[:, 0], limits[:, 1]
            signs = torch.zeros(len(self._hand_joint_ids), device=self.device)
            signs = torch.where(upper <= _LIMIT_TOL, torch.full_like(signs, -1.0), signs)
            signs = torch.where(lower >= -_LIMIT_TOL, torch.full_like(signs, 1.0), signs)
            self._hand_signs = signs.unsqueeze(0)
        return self._hand_signs

    def _orient_hand_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Point each hand joint's command down the direction it can travel.

        Only the sign changes, and only for joints with a one-sided limit.
        """
        signs = self._hand_closing_signs()
        return torch.where(signs != 0.0, signs * targets.abs(), targets)

    def _measure_tracking(self) -> None:
        if not self._have_commanded:
            return
        pose_w = self._asset.data.body_link_pose_w.torch[:, self._eef_body_ids, :3]
        # Commanded poses are in the env frame; measured ones are in the world.
        achieved = pose_w - self._env.scene.env_origins.unsqueeze(1)
        self._tracking_error[:] = torch.linalg.norm(achieved - self._commanded_eef_pos, dim=-1)

    def _retune_gains(self) -> None:
        """Set the task gain from the control period and the solves per step."""
        if self._solves_this_step > 0:
            self._solves_per_step = self._solves_this_step
        if self._solves_per_step == self._gain_solves:
            return
        self._gain_solves = self._solves_per_step

        gain = normalised_task_gain(self._step_dt, self.cfg.ik_time_constant, self._solves_per_step)
        for controller in self._ik_controllers:
            for task in controller.cfg.variable_input_tasks:
                task.gain = gain
            for task in controller.cfg.fixed_input_tasks:
                task.gain = gain


@configclass
class MultiRobotPinkIKActionCfg(PinkInverseKinematicsActionCfg):
    class_type: type[MultiRobotPinkIKAction] | str = f"{__name__}:MultiRobotPinkIKAction"

    constant_time_constant: bool = True
    """Derive the task gain from the control period and the solve count.

    Turn off for upstream's fixed gain."""

    ik_time_constant: float = IK_TIME_CONSTANT
    """Target 63% settling time for the arms [s]."""

    elbow_clearance: float | None = 0.24
    """Minimum lateral distance of each elbow from the torso midplane [m].

    ``None`` or 0 disables it."""

    elbow_clearance_cost: float = 10.0
    """Weight of the elbow clearance task."""



# Env-cfg factories

_IK_URDF_PATH = os.environ.get(
    "COHUB_G1_IK_URDF",
    f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/"
    "unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf",
)

_AGILE_POLICY_PATH = os.environ.get(
    "COHUB_AGILE_POLICY",
    f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt",
)

_LOWER_BODY_JOINT_NAMES = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
]


WAIST_MODES = {
    "all": ("waist_.*_joint",),
    "yaw": ("waist_yaw_joint",),
    "none": (),
}
"""Which waist joints the IK may use, per mode.

Yaw turns the torso and the ego camera with it; pitch and roll lean it."""


def upper_ik_action_cfg(
    asset_name: str,
    waist: str = "all",
    posture_cost: float | None = None,
    elbow_clearance: float | None = 0.24,
    elbow_clearance_cost: float = 10.0,
) -> MultiRobotPinkIKActionCfg:
    """Per-robot deep copy of the upstream G1 upper-body IK action config.

    ``waist`` is a key of :data:`WAIST_MODES` (upstream: ``"all"``). ``elbow_clearance``
    is the minimum distance [m] of each elbow from the torso's midplane; 0 disables it.
    """
    if waist not in WAIST_MODES:
        raise ValueError(f"waist must be one of {sorted(WAIST_MODES)}, got {waist!r}")

    base = copy.deepcopy(G1_UPPER_BODY_IK_ACTION_CFG)
    allowed = WAIST_MODES[waist]
    joint_names = [name for name in base.pink_controlled_joint_names if "waist" not in name]
    joint_names.extend(allowed)

    cfg = MultiRobotPinkIKActionCfg(
        pink_controlled_joint_names=joint_names,
        hand_joint_names=base.hand_joint_names,
        target_eef_link_names=base.target_eef_link_names,
        asset_name=asset_name,
        controller=base.controller,
    )
    cfg.controller.articulation_name = asset_name
    cfg.controller.urdf_path = _IK_URDF_PATH
    cfg.elbow_clearance = elbow_clearance
    cfg.elbow_clearance_cost = elbow_clearance_cost
    posture_tasks = [
        task
        for task in cfg.controller.variable_input_tasks
        if isinstance(task, NullSpacePostureTaskCfg)
    ]
    if len(posture_tasks) != 1:
        raise RuntimeError(f"Expected one Pink null-space posture task for {asset_name}, got {len(posture_tasks)}")
    posture = posture_tasks[0]
    for elbow_joint in ("left_elbow_joint", "right_elbow_joint"):
        if elbow_joint not in posture.controlled_joints:
            posture.controlled_joints.append(elbow_joint)
    if posture_cost is not None:
        posture.cost = posture_cost

    # Must run after the posture task's joints are edited above.
    if waist != "all":
        # Drop the waist joints the solver does not control from every task.
        keep = {"waist_yaw_joint"} if waist == "yaw" else set()
        for task in cfg.controller.variable_input_tasks:
            controlled = getattr(task, "controlled_joints", None)
            if controlled:
                task.controlled_joints = [
                    name for name in controlled if "waist" not in name or name in keep
                ]
    return cfg


def lower_body_action_cfg(asset_name: str, obs_group_name: str) -> MultiRobotAgileLowerBodyActionCfg:
    """Per-robot Agile locomotion action config, wired to its own obs group."""
    return MultiRobotAgileLowerBodyActionCfg(
        asset_name=asset_name,
        joint_names=list(_LOWER_BODY_JOINT_NAMES),
        policy_output_scale=0.25,
        obs_group_name=obs_group_name,
        policy_path=_AGILE_POLICY_PATH,
    )


# Eval-only joint-space upper-body action, no Pink IK

class SmoothedJointTargetAction(JointPositionAction):
    """:class:`JointPositionAction` whose arm and waist targets are low-pass filtered.

    Arm and waist columns approach the target at the Pink IK's rate
    (:data:`IK_TIME_CONSTANT`). Hand columns are applied immediately.
    """

    cfg: GR00TJointTargetActionCfg

    def __init__(self, cfg: GR00TJointTargetActionCfg, env) -> None:
        super().__init__(cfg, env)
        hand_set = set(cfg.hand_joint_names)
        self._hand_cols = [i for i, name in enumerate(self._joint_names) if name in hand_set]
        self._arm_cols = [i for i in range(self._num_joints) if i not in self._hand_cols]
        self._step_dt = float(env.step_dt)
        solves_per_step = max(1, int(getattr(env.cfg, "decimation", 1)))
        self._gain = normalised_task_gain(self._step_dt, IK_TIME_CONSTANT, solves_per_step)
        self._smoothed = torch.zeros_like(self._processed_actions)
        self._seeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Built on first use.
        self._wrist_fk: _WristForwardKinematics | None = None
        self._wrist_cache: torch.Tensor | None = None

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._wrist_cache = None

    def commanded_wrist_positions(self) -> torch.Tensor:
        """(num_envs, num_wrists, 3) commanded wrist positions, pelvis frame.

        Computed by forward kinematics from the commanded joint targets.
        """
        if self._wrist_cache is None:
            if self._wrist_fk is None:
                links = self.cfg.target_eef_link_names or G1_UPPER_BODY_IK_ACTION_CFG.target_eef_link_names
                self._wrist_fk = _WristForwardKinematics(
                    list(self._joint_names),
                    list(links.values()),
                    self.cfg.urdf_path or _IK_URDF_PATH,
                )
            self._wrist_cache = self._wrist_fk(self._processed_actions)
        return self._wrist_cache

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._wrist_cache = None
        if env_ids is None:
            self._seeded[:] = False
        else:
            self._seeded[env_ids] = False

    def apply_actions(self) -> None:
        # Seed the filter from the measured joint positions, once per episode.
        unseeded = ~self._seeded
        if bool(unseeded.any()):
            self._smoothed[unseeded] = self._asset.data.joint_pos.torch[unseeded][:, self._joint_ids]
            self._seeded[unseeded] = True

        self._smoothed[:, self._arm_cols] += self._gain * (
            self.processed_actions[:, self._arm_cols] - self._smoothed[:, self._arm_cols]
        )
        self._smoothed[:, self._hand_cols] = self.processed_actions[:, self._hand_cols]
        self._asset.set_joint_position_target_index(target=self._smoothed, joint_ids=self._joint_ids)


class _WristForwardKinematics:
    """Pelvis-frame wrist positions from a set of commanded joint targets."""

    def __init__(self, joint_names: list[str], link_names: list[str], urdf_path: str) -> None:
        import pinocchio as pin

        self._pin = pin
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()
        # URDF link names carry a model-name prefix, so match on the suffix.
        frames = [f.name for f in self._model.frames]

        def frame_id(link: str) -> int:
            exact = [n for n in frames if n == link or n.endswith("_" + link)]
            if not exact:
                raise ValueError(f"{link!r} is not a frame in {urdf_path}")
            return self._model.getFrameId(exact[0])

        self._pelvis_fid = frame_id("pelvis")
        self._wrist_fids = [frame_id(link) for link in link_names]
        # action column -> configuration index, for the joints this term drives
        self._cols: list[tuple[int, int]] = []
        for joint_index, name in enumerate(self._model.names):
            if name in joint_names:
                self._cols.append((self._model.idx_qs[joint_index], joint_names.index(name)))
        self._q = pin.neutral(self._model)

    def __call__(self, targets: torch.Tensor) -> torch.Tensor:
        """``targets`` (num_envs, num_joints) -> (num_envs, num_wrists, 3)."""
        joints = targets.detach().cpu().numpy()
        out = np.zeros((joints.shape[0], len(self._wrist_fids), 3), dtype=np.float32)
        for env in range(joints.shape[0]):
            for q_index, column in self._cols:
                self._q[q_index] = joints[env, column]
            self._pin.forwardKinematics(self._model, self._data, self._q)
            self._pin.updateFramePlacements(self._model, self._data)
            pelvis_inv = self._data.oMf[self._pelvis_fid].inverse()
            for slot, fid in enumerate(self._wrist_fids):
                out[env, slot] = (pelvis_inv * self._data.oMf[fid]).translation
        return torch.from_numpy(out).to(targets.device)


@configclass
class GR00TJointTargetActionCfg(JointPositionActionCfg):
    """:class:`JointPositionActionCfg` plus ``hand_joint_names``, which locates
    the hand columns for :class:`SmoothedJointTargetAction`."""

    class_type: type[SmoothedJointTargetAction] | str = f"{__name__}:SmoothedJointTargetAction"

    hand_joint_names: list[str] = MISSING

    # Optional. When unset, the upstream G1 wrist links and IK URDF are used.
    target_eef_link_names: dict[str, str] | None = None
    urdf_path: str | None = None


def joint_target_action_cfg(asset_name: str) -> GR00TJointTargetActionCfg:
    """Eval-only upper-body action: absolute joint targets, Pink IK bypassed.

    The action is the absolute joint-position target for the 31 arm, hand and
    waist joints, in ``keys.action_order()`` order. Not for recording: teleop
    commands wrist poses and uses :func:`upper_ik_action_cfg`.
    """
    return GR00TJointTargetActionCfg(
        asset_name=asset_name,
        joint_names=action_order(),
        preserve_order=True,
        use_default_offset=False,
        hand_joint_names=hand_order(),
        target_eef_link_names=copy.deepcopy(G1_UPPER_BODY_IK_ACTION_CFG.target_eef_link_names),
        urdf_path=_IK_URDF_PATH,
    )
