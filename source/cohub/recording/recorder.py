"""The demonstration recorder: its terms and cfg."""

from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import dataclass, field
import inspect
from typing import Any

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import string_to_callable
from isaaclab.utils.configclass import configclass
import torch

from cohub.keys import CAMERA_SENSORS, ROBOTS3


# --- State shared with the record script ---
# The record script stamps operator sample times each loop iteration; the terms read them.

@dataclass
class _SyncState:
    teleop_sample_time: float = 0.0
    follower_sample_time: float = 0.0
    step_wall_time: float = 0.0
    # Federated mode: when the second process sampled operator 2's headset.
    secondary_sample_time: float = 0.0
    # The same for operator 3, on a three-robot task.
    tertiary_sample_time: float = 0.0
    # Round trip of the state/action exchange [s].
    link_rtt: float = 0.0
    # Raw operator input before retargeting. Empty when the headset is not connected.
    operator_raw: dict[str, list[float]] = field(default_factory=dict)
    # Operator 2's raw input, from the federation link. Empty with a local follower device.
    operator_raw_b: list[float] = field(default_factory=list)
    # Operator 1's raw input as the wire vector, when operator 1 is the remote one.
    operator_raw_a: list[float] = field(default_factory=list)
    # Operator 3's raw input as the wire vector, on a three-robot task.
    operator_raw_c: list[float] = field(default_factory=list)
    # Numeric episode metadata.
    episode_meta: dict[str, float] = field(
        default_factory=lambda: {
            "partner_id": 0,
            "operator_id_a": 0,
            "operator_id_b": 0,
            "operator_device_b": 0,  # 0=constant, 1=keyboard, 2=gamepad, 3=quest
            "role_a_leader": 1,
            "terrain_id": 0,  # 0=flat
            # 0=held by the fingers, 1=held by a proximity latch.
            "grasp_abstraction": 0,
        }
    )


SYNC_STATE = _SyncState()


# --- Recorder terms ---
# Keys are nested HDF5 paths. State records at env rate, EEF wrenches at sim rate, episode metadata
# once per episode, cameras on record_pre_step.

_WRIST_JOINTS = ["left_wrist_yaw_joint", "right_wrist_yaw_joint"]




def robots_present(env) -> tuple[str, ...]:
    """The robots this task declares, in slice order."""
    return tuple(name for name in ROBOTS3 if name in env.scene.articulations)


def robots_in_cfg(scene_cfg) -> tuple[str, ...]:
    """The robots a scene cfg declares, in slice order. Usable before ``gym.make``."""
    return tuple(name for name in ROBOTS3 if getattr(scene_cfg, name, None) is not None)


def _sensor_key(prefix: str, robot_key: str) -> str:
    """``("contact", "robot_a") -> "contact_a"`` -- the scene's naming."""
    return f"{prefix}_{robot_key.rsplit('_', 1)[1]}"


def camera_frame(output) -> dict[str, torch.Tensor]:
    """One camera's annotator outputs, named and typed as the dataset stores them.

    Shared with ``scripts/dataset/render_dataset_images.py``. The batch dimension is kept.
    """
    frame: dict[str, torch.Tensor] = {}
    for data_type, tensor in output.items():
        values = tensor.torch if hasattr(tensor, "torch") else tensor
        if data_type == "rgb":
            # Drop the constant alpha channel.
            frame["rgb"] = values[..., :3].to(torch.uint8).clone()
        elif data_type == "distance_to_image_plane":
            frame["depth"] = values.to(torch.float16).clone()
        elif data_type == "rgba":
            continue  # identical to rgb plus a constant alpha
        else:
            frame[data_type] = values.clone()
    return frame


def _robot_state(env, robot_name: str) -> dict[str, torch.Tensor]:
    robot = env.scene[robot_name]
    data = robot.data
    state = {
        "q": data.joint_pos.torch.clone(),
        "dq": data.joint_vel.torch.clone(),
        "root_pose": torch.cat([data.root_pos_w.torch, data.root_quat_w.torch], dim=-1),
        "root_vel": torch.cat([data.root_lin_vel_w.torch, data.root_ang_vel_w.torch], dim=-1),
    }
    com = getattr(data, "root_com_pos_w", None)
    if com is not None:
        state["com"] = com.torch.clone()
    return state


class PerRobotStateRecorder(RecorderTerm):
    """q, dq, root pose/vel, CoM for every robot at env rate."""

    def record_post_step(self):
        return "states/robots", {name: _robot_state(self._env, name) for name in robots_present(self._env)}


class FootContactRecorder(RecorderTerm):
    """Net contact force on each foot, from the per-robot contact sensors."""

    def record_post_step(self):
        out = {}
        for robot_key in robots_present(self._env):
            sensor = self._env.scene.sensors.get(_sensor_key("contact", robot_key))
            if sensor is None:
                continue
            out[robot_key] = sensor.data.net_forces_w.torch.clone()
        return ("states/foot_contact", out) if out else (None, None)


class EEFWrenchRecorder(RecorderTerm):
    """Wrist joint reaction wrenches at sim rate (real F/T sensor analog)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._joint_ids: dict[str, list[int]] = {}

    def _wrist_ids(self, robot_key: str) -> list[int]:
        if robot_key not in self._joint_ids:
            robot = self._env.scene[robot_key]
            ids, _ = robot.find_joints(_WRIST_JOINTS, preserve_order=True)
            self._joint_ids[robot_key] = ids
        return self._joint_ids[robot_key]

    def record_post_physics_decimation_step(self):
        out = {}
        for robot_key in robots_present(self._env):
            sensor = self._env.scene.sensors.get(_sensor_key("wrench", robot_key))
            if sensor is None or sensor.data.force is None:
                continue
            ids = self._wrist_ids(robot_key)
            force = sensor.data.force.torch[:, ids, :].clone()
            torque = sensor.data.torque.torch[:, ids, :].clone()
            out[robot_key] = {"force": force, "torque": torque}
        return ("eef_wrench", out) if out else (None, None)


class ObjectStateRecorder(RecorderTerm):
    def record_post_step(self):
        obj = self._env.scene["object"]
        return "states/object", {
            "pose": torch.cat([obj.data.root_pos_w.torch, obj.data.root_quat_w.torch], dim=-1),
            "vel": torch.cat([obj.data.root_lin_vel_w.torch, obj.data.root_ang_vel_w.torch], dim=-1),
        }


class CameraRecorder(RecorderTerm):
    """Ego RGB-D for both robots plus the scene view, at env rate.

    Stored under ``obs/images`` on ``record_pre_step``: ``obs/images/*[k]`` is
    the frame rendered at the end of step ``k-1``, the one seen when
    ``actions[k]`` was made. Depth is stored as float16 metres.
    """

    _CAMERAS = CAMERA_SENSORS

    def record_pre_step(self):
        out = {}
        for key, sensor_name in self._CAMERAS:
            sensor = self._env.scene.sensors.get(sensor_name)
            if sensor is None:
                continue
            frame = camera_frame(sensor.data.output)
            if frame:
                out[key] = frame
        return ("obs/images", out) if out else (None, None)


MOUNT_DRIFT_LIMIT_M = 0.25
"""How far a camera's reported pose may sit from its mount before it is wrong [m]."""

_WARNED: set[str] = set()


def camera_pose(key: str, sensor) -> torch.Tensor:
    """One camera's world pose as the dataset stores it, xyz + xyzw.

    Warns once per camera when the pose sits more than
    :data:`MOUNT_DRIFT_LIMIT_M` from its mount. Raises ``RuntimeError`` when
    the camera exposes no pose at all.
    """
    data = sensor.data
    pos = getattr(data, "pos_w", None)
    quat = getattr(data, "quat_w_world", None)
    if quat is None:
        quat = getattr(data, "quat_w_ros", None)
    if pos is None or quat is None:
        raise RuntimeError(
            f"camera '{key}' exposes no world pose (pos_w/quat_w_world), so its "
            "images could not be located in the scene"
        )
    pos_t = pos.torch if hasattr(pos, "torch") else pos
    quat_t = quat.torch if hasattr(quat, "torch") else quat

    mount = getattr(sensor, "mount_pos_w", None)
    if mount is not None and key not in _WARNED:
        drift = float((pos_t - mount).abs().max())
        if drift > MOUNT_DRIFT_LIMIT_M:
            _WARNED.add(key)
            print(
                f"[camera] WARNING: '{key}' reports a pose {drift:.2f} m from the link it is "
                f"bolted to. It is not tracking its mount, so the extrinsics recorded beside "
                f"its images are wrong. The session continues -- the states are unaffected -- "
                f"but verify_dataset.py will fail this file."
            )
    return torch.cat([pos_t, quat_t], dim=-1)


class CameraInfoRecorder(RecorderTerm):
    """Intrinsics and world pose per camera.

    Recorded on ``record_pre_step`` so ``camera_info/*/pose[k]`` describes
    ``obs/images/*[k]``.
    """

    _CAMERAS = CAMERA_SENSORS

    def record_pre_step(self):
        out = {}
        for key, sensor_name in self._CAMERAS:
            sensor = self._env.scene.sensors.get(sensor_name)
            if sensor is None:
                continue
            out[key] = {
                "intrinsics": sensor.data.intrinsic_matrices.torch.clone(),
                "pose": camera_pose(key, sensor),
            }
        return ("camera_info", out) if out else (None, None)


class JointTorqueRecorder(RecorderTerm):
    """Applied joint torques -- the proprioception channel q and dq leave out."""

    def record_post_step(self):
        out = {}
        for robot_key in robots_present(self._env):
            data = self._env.scene[robot_key].data
            torque = getattr(data, "applied_torque", None)
            if torque is not None:
                out[robot_key] = torque.torch.clone()
        return ("states/joint_torque", out) if out else (None, None)


class HandContactRecorder(RecorderTerm):
    """Net contact force on each palm, separate from the grasp force."""

    def record_post_step(self):
        out = {}
        for robot_key in robots_present(self._env):
            sensor = self._env.scene.sensors.get(_sensor_key("hand_contact", robot_key))
            if sensor is None:
                continue
            out[robot_key] = sensor.data.net_forces_w.torch.clone()
        return ("states/hand_contact", out) if out else (None, None)


class OperatorInputRecorder(RecorderTerm):
    """Raw operator input before retargeting: controller poses, buttons,
    triggers and the headset pose.

    A local operator's groups are named as ``OperatorInputCapture`` produced
    them. A federated operator appears as one fixed-width ``operator/raw_a``,
    ``raw_b`` or ``raw_c`` vector. A robot on a local follower has no entry.
    """

    def record_post_step(self):
        raw = SYNC_STATE.operator_raw
        federated = {
            "raw_a": SYNC_STATE.operator_raw_a,
            "raw_b": SYNC_STATE.operator_raw_b,
            "raw_c": SYNC_STATE.operator_raw_c,
        }
        if not raw and not any(federated.values()):
            return None, None
        n, device = self._env.num_envs, self._env.device

        def _t(values) -> torch.Tensor:
            tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
            return tensor.unsqueeze(0).expand(n, -1).clone()

        out = {key: _t(value) for key, value in raw.items()}
        out.update({key: _t(value) for key, value in federated.items() if value})
        return "operator", out


class GraspStateRecorder(RecorderTerm):
    """Per-hand hold state and the grip force measured with it.

    Hand order is ``GraspActionCfg.robot_names`` x ``attach_body_names``:
    (robot_a left, robot_a right, robot_b left, robot_b right, ...). Read the
    width from the array.
    """

    def record_post_step(self):
        term = self._env.action_manager.get_term("grasp")
        return "states/grasp", {
            "latched": term.latched.float().clone(),
            "ever_latched": term.ever_latched.float().clone(),
            "ever_both_robots_latched": term.ever_both_robots_latched.float().clone(),
            "release_age": term.release_age.clone(),
            "force": term.grasp_force.clone(),
            "torque": term.grasp_torque.clone(),
        }


class DerivedQuantitiesRecorder(RecorderTerm):
    """Internal-wrench decomposition, load share, inter-robot distance, for the A-B pair.

    ``squeeze`` is the differential grasp force along the A→B axis;
    ``load_share_ratio`` is robot A's share of the vertical support force.
    """

    def _net_eef_force(self, robot_key: str, sensor_key: str) -> torch.Tensor | None:
        sensor = self._env.scene.sensors.get(sensor_key)
        if sensor is None or sensor.data.force is None:
            return None
        robot = self._env.scene[robot_key]
        ids, _ = robot.find_joints(_WRIST_JOINTS, preserve_order=True)
        return sensor.data.force.torch[:, ids, :].sum(dim=1)

    def record_post_step(self):
        env = self._env
        pelvis_a = env.scene["robot_a"].data.root_pos_w.torch
        pelvis_b = env.scene["robot_b"].data.root_pos_w.torch
        dist = torch.linalg.norm(pelvis_b - pelvis_a, dim=-1, keepdim=True)

        out = {"inter_robot_distance": dist}

        f_a = self._net_eef_force("robot_a", "wrench_a")
        f_b = self._net_eef_force("robot_b", "wrench_b")
        if f_a is not None and f_b is not None:
            axis = pelvis_b - pelvis_a
            axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1e-6)
            squeeze = 0.5 * ((f_a - f_b) * axis).sum(dim=-1, keepdim=True)
            net = f_a + f_b
            share_a = f_a[:, 2:3].abs()
            share_b = f_b[:, 2:3].abs()
            out["internal_wrench_squeeze"] = squeeze
            out["net_wrench"] = net
            out["load_share_ratio"] = share_a / torch.clamp(share_a + share_b, min=1e-6)
        return "derived", out


class SyncRecorder(RecorderTerm):
    """Wall-clock stamps of the freshest operator samples vs sim time.

    ``follower_sample_time``: when a local follower device was read.
    ``secondary_sample_time``: when a second headset's process sampled its operator.
    ``tertiary_sample_time``: the same for a third headset, three-robot tasks only.
    ``link_rtt``: the federation round trip.
    """

    def record_pre_step(self):
        env = self._env
        device = env.device
        n = env.num_envs

        def _t(value: float) -> torch.Tensor:
            return torch.full((n, 1), value, dtype=torch.float64, device=device)

        out = {
            "sim_time": _t(float(env.common_step_counter) * env.step_dt),
            "teleop_sample_time": _t(SYNC_STATE.teleop_sample_time),
            "follower_sample_time": _t(SYNC_STATE.follower_sample_time),
            "secondary_sample_time": _t(SYNC_STATE.secondary_sample_time),
            "link_rtt": _t(SYNC_STATE.link_rtt),
            "step_wall_time": _t(SYNC_STATE.step_wall_time),
        }
        if "robot_c" in env.scene.articulations:
            out["tertiary_sample_time"] = _t(SYNC_STATE.tertiary_sample_time)
        return "sync", out


class EpisodeMetaRecorder(RecorderTerm):
    """Per-episode operator/session metadata: the dataset's ``meta`` block.

    The legend for the int codes is ``recording/operators.yaml``. Fires on
    ``record_pre_reset``, so every reset produces an episode carrying this
    block even when nothing was stepped; see
    :func:`~cohub.recording.dataset_writer.is_empty_episode`.
    """

    def record_pre_reset(self, env_ids: Sequence[int] | None):
        env = self._env
        n = env.num_envs if env_ids is None else len(env_ids)
        meta = SYNC_STATE.episode_meta

        def _t(value: float) -> torch.Tensor:
            return torch.full((n, 1), value, dtype=torch.float32, device=env.device)

        return "meta", {key: _t(float(value)) for key, value in meta.items()}


# --- Recorder cfg ---
# Wired to each task through demo_recorder_cfg_entry_point in cohub.tasks.

_TERMS = __name__


@configclass
class PerRobotStateRecorderCfg(RecorderTermCfg):
    class_type: str = f"{_TERMS}:PerRobotStateRecorder"


@configclass
class FootContactRecorderCfg(RecorderTermCfg):
    class_type: str = f"{_TERMS}:FootContactRecorder"


@configclass
class EEFWrenchRecorderCfg(RecorderTermCfg):
    class_type: str = f"{_TERMS}:EEFWrenchRecorder"


@configclass
class ObjectStateRecorderCfg(RecorderTermCfg):
    class_type: str = f"{_TERMS}:ObjectStateRecorder"


@configclass
class CameraRecorderCfg(RecorderTermCfg):
    class_type: str = f"{_TERMS}:CameraRecorder"


@configclass
class CameraInfoRecorderCfg(RecorderTermCfg):
    class_type: str = f"{_TERMS}:CameraInfoRecorder"


@configclass
class JointTorqueRecorderCfg(RecorderTermCfg):
    class_type: str = f"{_TERMS}:JointTorqueRecorder"


@configclass
class HandContactRecorderCfg(RecorderTermCfg):
    class_type: str = f"{_TERMS}:HandContactRecorder"


@configclass
class OperatorInputRecorderCfg(RecorderTermCfg):
    class_type: str = f"{_TERMS}:OperatorInputRecorder"


@configclass
class GraspStateRecorderCfg(RecorderTermCfg):
    class_type: str = f"{_TERMS}:GraspStateRecorder"


@configclass
class DerivedQuantitiesRecorderCfg(RecorderTermCfg):
    class_type: str = f"{_TERMS}:DerivedQuantitiesRecorder"


@configclass
class SyncRecorderCfg(RecorderTermCfg):
    class_type: str = f"{_TERMS}:SyncRecorder"


@configclass
class EpisodeMetaRecorderCfg(RecorderTermCfg):
    class_type: str = f"{_TERMS}:EpisodeMetaRecorder"


@configclass
class DemoRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_per_robot_state = PerRobotStateRecorderCfg()
    record_foot_contact = FootContactRecorderCfg()
    record_eef_wrench = EEFWrenchRecorderCfg()
    record_object_state = ObjectStateRecorderCfg()
    record_grasp = GraspStateRecorderCfg()
    record_joint_torque = JointTorqueRecorderCfg()
    record_hand_contact = HandContactRecorderCfg()
    record_operator_input = OperatorInputRecorderCfg()
    record_images = CameraRecorderCfg()
    record_camera_info = CameraInfoRecorderCfg()
    record_derived = DerivedQuantitiesRecorderCfg()
    record_sync = SyncRecorderCfg()
    record_episode_meta = EpisodeMetaRecorderCfg()


# --- A term run outside its manager ---
# The record script removes the task's success term and polls it itself; this instantiates a term as
# its manager would.

class DetachedManagerTerm:
    """A configured manager term callable outside an Isaac Lab manager."""

    def __init__(self, env, term_cfg: ManagerTermBaseCfg):
        # Run on a copy; the caller keeps the unresolved config for logs.
        self.config = term_cfg
        self.runtime_config = copy.deepcopy(term_cfg)
        _prepare_term_config(env, self.runtime_config)
        self.func = self.runtime_config.func

    def __call__(self, env):
        """Evaluate with the runtime-resolved parameters from the config."""
        return self.func(env, **self.runtime_config.params)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Reset stateful terms at the same episode boundaries as a manager."""
        if isinstance(self.func, ManagerTermBase):
            self.func.reset(env_ids)


def _prepare_term_config(env, term_cfg: ManagerTermBaseCfg) -> None:
    """Resolve and instantiate one config as ``ManagerBase`` does at play."""
    for value in term_cfg.params.values():
        _resolve_param_value(env, value)

    func = term_cfg.func
    if isinstance(func, str):
        func = string_to_callable(func)
    if inspect.isclass(func):
        if not issubclass(func, ManagerTermBase):
            raise TypeError(
                "A class-based detached manager term must inherit ManagerTermBase; "
                f"received {func!r}."
            )
        func = func(cfg=term_cfg, env=env)
    if not callable(func):
        raise TypeError(f"Detached manager term is not callable: {func!r}")
    term_cfg.func = func


def _resolve_param_value(env, value: Any) -> None:
    """Recursively resolve scene and nested term configs like ManagerBase."""
    if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
    elif isinstance(value, ManagerTermBaseCfg):
        _prepare_term_config(env, value)
    elif isinstance(value, dict):
        for child in value.values():
            _resolve_param_value(env, child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _resolve_param_value(env, child)
