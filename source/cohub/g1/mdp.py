"""The robot's manager terms: observations, the shared observation group, and the fall termination."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg, ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.agile_locomotion_observation_cfg import (
    AgileTeacherPolicyObservationsCfg,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
    _build_g1_locomanipulation_pipeline,
)
from isaaclab_teleop import IsaacTeleopCfg, XrAnchorRotationMode, XrCfg
import torch

from cohub.subtasks import publish_checks
from cohub.g1.cameras import cameras_enabled
from cohub.g1.grasp import grasp_closed, grasp_latched
from cohub.g1.geometry import anchor_rot_for_spawn

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --- Observation terms ---

_AGILE_JOINT_NAMES = [
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "waist_.*_joint",
]


def body_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, body_name: str) -> torch.Tensor:
    """World position of a named body of the given articulation."""
    asset = env.scene[asset_cfg.name]
    idx = asset.data.body_names.index(body_name)
    return asset.data.body_pos_w.torch[:, idx, :]


def body_quat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, body_name: str) -> torch.Tensor:
    """World orientation (quaternion) of a named body of the given articulation."""
    asset = env.scene[asset_cfg.name]
    idx = asset.data.body_names.index(body_name)
    return asset.data.body_quat_w.torch[:, idx, :]


def _retarget_agile_group(group: AgileTeacherPolicyObservationsCfg, asset_name: str, action_name: str) -> None:
    """Point every term of an Agile observation group at the given robot."""
    group.base_lin_vel.params = {"asset_cfg": SceneEntityCfg(asset_name)}
    group.base_ang_vel.params = {"asset_cfg": SceneEntityCfg(asset_name)}
    group.projected_gravity.params = {"asset_cfg": SceneEntityCfg(asset_name)}
    group.joint_pos.params = {"asset_cfg": SceneEntityCfg(asset_name, joint_names=list(_AGILE_JOINT_NAMES))}
    group.joint_vel.params = {"asset_cfg": SceneEntityCfg(asset_name, joint_names=list(_AGILE_JOINT_NAMES))}
    group.actions.params = {"action_name": action_name}


@configclass
class AgileObsRobotACfg(AgileTeacherPolicyObservationsCfg):
    def __post_init__(self):
        super().__post_init__()
        _retarget_agile_group(self, "robot_a", "lower_body_a")


@configclass
class AgileObsRobotBCfg(AgileTeacherPolicyObservationsCfg):
    def __post_init__(self):
        super().__post_init__()
        _retarget_agile_group(self, "robot_b", "lower_body_b")


@configclass
class AgileObsRobotCCfg(AgileTeacherPolicyObservationsCfg):
    def __post_init__(self):
        super().__post_init__()
        _retarget_agile_group(self, "robot_c", "lower_body_c")


# --- Terminations ---

def robot_fallen(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_height: float = 0.25,
    max_tilt_deg: float = 45.0,
) -> torch.Tensor:
    """Robot on the floor -- as distinct from deliberately crouched.

    True when the pelvis tilts past ``max_tilt_deg`` or drops below
    ``minimum_height`` [m], which must stay below the 0.40 m crouch floor.
    """
    asset = env.scene[asset_cfg.name]
    too_low = asset.data.root_pos_w.torch[:, 2] < minimum_height
    # Body up-axis z in world = R[2, 2] = 1 - 2(qx^2 + qy^2), for xyzw.
    quat = asset.data.root_quat_w.torch
    up_z = 1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
    tilted = up_z < math.cos(math.radians(max_tilt_deg))
    # Keyed by asset so each robot's checks stay separate.
    publish_checks(env, f"robot_fallen[{asset_cfg.name}]", above_floor=~too_low, upright=~tilted)
    return too_low | tilted


# --- Observation group and env defaults ---
# Term order is the recorded order: base fields first, then a subclass's own.

@configclass
class G1DualPolicyObsCfg(ObsGroup):
    """Low-dim state observations recorded for imitation learning."""

    actions = ObsTerm(func=base_mdp.last_action)

    robot_a_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_a")})
    robot_a_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot_a")})
    robot_a_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot_a")})
    robot_b_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_b")})
    robot_b_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot_b")})
    robot_b_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot_b")})

    robot_a_left_eef_pos = ObsTerm(
        func=body_pos_w, params={"asset_cfg": SceneEntityCfg("robot_a"), "body_name": "left_wrist_yaw_link"}
    )
    robot_a_right_eef_pos = ObsTerm(
        func=body_pos_w, params={"asset_cfg": SceneEntityCfg("robot_a"), "body_name": "right_wrist_yaw_link"}
    )
    robot_b_left_eef_pos = ObsTerm(
        func=body_pos_w, params={"asset_cfg": SceneEntityCfg("robot_b"), "body_name": "left_wrist_yaw_link"}
    )
    robot_b_right_eef_pos = ObsTerm(
        func=body_pos_w, params={"asset_cfg": SceneEntityCfg("robot_b"), "body_name": "right_wrist_yaw_link"}
    )

    robot_a_left_eef_rot = ObsTerm(
        func=body_quat_w, params={"asset_cfg": SceneEntityCfg("robot_a"), "body_name": "left_wrist_yaw_link"}
    )
    robot_a_right_eef_rot = ObsTerm(
        func=body_quat_w, params={"asset_cfg": SceneEntityCfg("robot_a"), "body_name": "right_wrist_yaw_link"}
    )
    robot_b_left_eef_rot = ObsTerm(
        func=body_quat_w, params={"asset_cfg": SceneEntityCfg("robot_b"), "body_name": "left_wrist_yaw_link"}
    )
    robot_b_right_eef_rot = ObsTerm(
        func=body_quat_w, params={"asset_cfg": SceneEntityCfg("robot_b"), "body_name": "right_wrist_yaw_link"}
    )

    object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
    object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})

    # Which hands currently hold the object.
    grasp_latched = ObsTerm(func=grasp_latched)

    # Each hand's own open/close state.
    robot_a_left_grasp = ObsTerm(func=grasp_closed, params={"robot": "robot_a", "side": "left"})
    robot_a_right_grasp = ObsTerm(func=grasp_closed, params={"robot": "robot_a", "side": "right"})
    robot_b_left_grasp = ObsTerm(func=grasp_closed, params={"robot": "robot_b", "side": "left"})
    robot_b_right_grasp = ObsTerm(func=grasp_closed, params={"robot": "robot_b", "side": "right"})

    def __post_init__(self):
        self.enable_corruption = False
        # The dataset records terms by name.
        self.concatenate_terms = False


@configclass
class G1TriplePolicyObsCfg(G1DualPolicyObsCfg):
    """The two-G1 terms plus a third robot's own, which come after the shared ones."""

    robot_c_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_c")})
    robot_c_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot_c")})
    robot_c_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot_c")})

    robot_c_left_eef_pos = ObsTerm(
        func=body_pos_w, params={"asset_cfg": SceneEntityCfg("robot_c"), "body_name": "left_wrist_yaw_link"}
    )
    robot_c_right_eef_pos = ObsTerm(
        func=body_pos_w, params={"asset_cfg": SceneEntityCfg("robot_c"), "body_name": "right_wrist_yaw_link"}
    )
    robot_c_left_eef_rot = ObsTerm(
        func=body_quat_w, params={"asset_cfg": SceneEntityCfg("robot_c"), "body_name": "left_wrist_yaw_link"}
    )
    robot_c_right_eef_rot = ObsTerm(
        func=body_quat_w, params={"asset_cfg": SceneEntityCfg("robot_c"), "body_name": "right_wrist_yaw_link"}
    )

    robot_c_left_grasp = ObsTerm(func=grasp_closed, params={"robot": "robot_c", "side": "left"})
    robot_c_right_grasp = ObsTerm(func=grasp_closed, params={"robot": "robot_c", "side": "right"})


def apply_g1_dual_defaults(cfg, episode_length_s: float) -> None:
    """Set the rates, camera handling and XR anchor every G1 task shares, in place.

    Call it from the task's own ``__post_init__``.
    """
    cfg.decimation = 4
    cfg.episode_length_s = episode_length_s
    cfg.sim.dt = 1 / 200
    # One render per control step.
    cfg.sim.render_interval = cfg.decimation

    # Camera sensors need --enable_cameras; drop them when it is not set.
    if not cameras_enabled():
        for name, value in vars(cfg.scene).items():
            if isinstance(value, (TiledCameraCfg, CameraCfg)):
                setattr(cfg.scene, name, None)

    # The XR anchor follows Robot A; ``anchor_rot`` carries its spawn heading.
    cfg.xr = XrCfg(
        anchor_pos=(0.0, 0.0, -0.95),
        anchor_rot=anchor_rot_for_spawn(cfg.scene.robot_a.init_state.rot),
    )
    cfg.xr.anchor_prim_path = "/World/envs/env_0/Robot_A/pelvis"
    cfg.xr.fixed_anchor_height = True
    cfg.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
    cfg.xr.anchor_rotation_smoothing_time = 0.3  # seconds
    cfg.isaac_teleop = IsaacTeleopCfg(
        pipeline_builder=_build_g1_locomanipulation_pipeline,
        sim_device=cfg.sim.device,
        xr_cfg=cfg.xr,
    )
