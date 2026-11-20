"""Action term for the door interaction.

The term is zero-width: the robot still receives the unchanged 64D G1 action
vector. It reads Robot A's Dex3 hand commands, latches a hand to the handle
when it is close enough, and applies bounded joint efforts to the handle and hinge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from ..assets import (
    DOOR_HANDLE_BODY,
    DOOR_HANDLE_GRASP_OFFSET,
    DOOR_HANDLE_JOINT,
    DOOR_HANDLE_OPEN_ANGLE,
    DOOR_HINGE_JOINT,
    DOOR_HINGE_OPEN_ANGLE,
    PASSAGE_SUCCESS_X,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_STAGE_MESSAGES = {
    2: "{robot} has the lever",
    3: "lever depressed - the latch is released",
    4: "door open - {robot} has cleared the passage",
    5: "{passer} is through the doorway",
}
"""Message printed when the door sequence reaches each stage."""


@configclass
class DoorOpeningActionCfg(ActionTermCfg):
    """Configuration for the zero-width articulated-door action."""

    class_type: type[ActionTerm] | str = "cohub.tasks.movehouse.mdp.actions:DoorOpeningAction"

    asset_name: str = "door"
    robot_name: str = "robot_a"
    hand_term_name: str = "upper_body_a"
    hand_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")

    hinge_joint_name: str = DOOR_HINGE_JOINT
    handle_joint_name: str = DOOR_HANDLE_JOINT
    handle_body_name: str = DOOR_HANDLE_BODY

    # grasp target relative to the handle body: the end of the lever
    handle_grasp_offset: tuple[float, float, float] = DOOR_HANDLE_GRASP_OFFSET
    max_grip_distance: float = 0.18
    close_threshold: float = 0.30
    open_threshold: float = 0.15

    # After latching, a downward wrist motion turns the lever and a forward/backward
    # motion swings the leaf. These gains turn the motions into bounded joint efforts.
    lever_arm_length: float = 0.136
    handle_position_gain: float = 28.0
    handle_velocity_gain: float = 1.2
    handle_torque_limit: float = 5.0
    hinge_push_deadband: float = 0.015
    hinge_full_motion: float = 0.10
    hinge_gesture_angle: float = 1.15
    hinge_position_gain: float = 28.0
    hinge_velocity_gain: float = 5.0
    hinge_torque_limit: float = 24.0

    # Holds the leaf at q=0 until the lever has been depressed.
    latch_lock_stiffness: float = 4000.0
    latch_lock_damping: float = 320.0
    latch_lock_torque_limit: float = 350.0
    handle_open_threshold: float = DOOR_HANDLE_OPEN_ANGLE
    hinge_open_threshold: float = DOOR_HINGE_OPEN_ANGLE

    # robot whose passage reports stage 5, and the x it has to reach [m]
    passage_robot_name: str = "robot_b"
    passage_threshold: float = PASSAGE_SUCCESS_X


class DoorOpeningAction(ActionTerm):
    """Sticky handle latch followed by a bounded handle/hinge effort."""

    cfg: DoorOpeningActionCfg

    def __init__(self, cfg: DoorOpeningActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._door = env.scene[cfg.asset_name]
        self._robot = env.scene[cfg.robot_name]

        joint_ids, joint_names = self._door.find_joints(
            [cfg.hinge_joint_name, cfg.handle_joint_name], preserve_order=True
        )
        if len(joint_ids) != 2:
            raise RuntimeError(
                "DoorOpeningAction expected the hinge and handle joints, "
                f"got {joint_names} from {cfg.asset_name!r}"
            )
        # Plain ints for the indexed setters, which reject int64 tensors. The tensor is for torch indexing.
        self._joint_ids = [int(joint_id) for joint_id in joint_ids]
        self._joint_ids_tensor = torch.as_tensor(self._joint_ids, dtype=torch.long, device=self.device)
        self._hinge_index = 0
        self._handle_index = 1

        body_ids, body_names = self._robot.find_bodies(list(cfg.hand_body_names), preserve_order=True)
        if len(body_ids) != len(cfg.hand_body_names):
            raise RuntimeError(
                "DoorOpeningAction could not find both Robot A wrist bodies: "
                f"requested={cfg.hand_body_names}, resolved={body_names}"
            )
        self._hand_body_ids = torch.as_tensor(body_ids, dtype=torch.long, device=self.device)

        handle_ids, resolved_handle_names = self._door.find_bodies([cfg.handle_body_name], preserve_order=True)
        if len(handle_ids) != 1:
            raise RuntimeError(
                "DoorOpeningAction could not find the articulated handle body: "
                f"requested={cfg.handle_body_name!r}, resolved={resolved_handle_names}"
            )
        self._handle_body_id = int(handle_ids[0])

        self._handle_grasp_offset = torch.tensor(cfg.handle_grasp_offset, device=self.device)
        self._empty = torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device)
        self.handle_latched = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.handle_opened = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.opened_by_a = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stage = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._announced_stage = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._latch_hand_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self._latch_command_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self._hand_columns: list[torch.Tensor] | None = None

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._empty

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.handle_latched[env_ids] = False
        self.handle_opened[env_ids] = False
        self.opened_by_a[env_ids] = False
        self.stage[env_ids] = 0
        self._announced_stage[env_ids] = 0
        self._latch_hand_pos[env_ids] = 0.0
        self._latch_command_pos[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        del actions  # the term has no action columns

        hand_term = self._env.action_manager.get_term(self.cfg.hand_term_name)
        hand_names = list(hand_term.cfg.hand_joint_names)
        hand_actions = self._hand_actions(hand_term, hand_names)
        wrist_commands = self._wrist_commands(hand_term)
        columns = self._resolve_hand_columns(hand_names)
        closure = torch.stack(
            [hand_actions[:, col].abs().mean(dim=1) for col in columns],
            dim=1,
        )

        hand_pos = self._robot.data.body_link_pos_w.torch[:, self._hand_body_ids]
        handle_data = self._door.data
        handle_pos = handle_data.body_link_pos_w.torch[:, self._handle_body_id]
        handle_quat = handle_data.body_link_quat_w.torch[:, self._handle_body_id]
        grasp_pos = handle_pos + quat_apply(handle_quat, self._handle_grasp_offset.expand_as(handle_pos))
        in_range = torch.linalg.norm(hand_pos - grasp_pos.unsqueeze(1), dim=-1) <= self.cfg.max_grip_distance

        want_close = closure > self.cfg.close_threshold
        want_open = closure < self.cfg.open_threshold
        new_latch = want_close & in_range & ~self.handle_latched
        if new_latch.any():
            self._latch_hand_pos = torch.where(
                new_latch.unsqueeze(-1), hand_pos, self._latch_hand_pos
            )
            self._latch_command_pos = torch.where(
                new_latch.unsqueeze(-1), wrist_commands, self._latch_command_pos
            )
            self.handle_latched |= new_latch
        self.handle_latched &= ~want_open

        hinge_q = handle_data.joint_pos.torch[:, self._joint_ids_tensor[self._hinge_index]]
        handle_q = handle_data.joint_pos.torch[:, self._joint_ids_tensor[self._handle_index]]
        holding = self.handle_latched.any(dim=1)
        self.handle_opened |= holding & (handle_q >= self.cfg.handle_open_threshold)
        # credit is independent of the swing direction
        self.opened_by_a |= holding & self.handle_opened & (
            hinge_q.abs() >= self.cfg.hinge_open_threshold
        )

        # Diagnostic stage: 0 approach, 2 handle grasp, 3 handle turned, 4 door open, 5 passage.
        self.stage.zero_()
        self.stage = torch.where(self.handle_latched.any(dim=1), torch.full_like(self.stage, 2), self.stage)
        self.stage = torch.where(self.handle_opened, torch.full_like(self.stage, 3), self.stage)
        self.stage = torch.where(self.opened_by_a, torch.full_like(self.stage, 4), self.stage)
        passer = self._env.scene[self.cfg.passage_robot_name]
        passer_pos = passer.data.root_pos_w.torch - self._env.scene.env_origins
        self.stage = torch.where(
            passer_pos[:, 0] >= self.cfg.passage_threshold,
            torch.full_like(self.stage, 5),
            self.stage,
        )
        self._announce_progress()

    def _announce_progress(self) -> None:
        """Print to the terminal when the door sequence advances to a new stage."""
        advanced = self.stage > self._announced_stage
        if not bool(advanced.any()):
            return
        for env_id in advanced.nonzero(as_tuple=False).flatten().tolist():
            message = _STAGE_MESSAGES.get(int(self.stage[env_id]))
            if message:
                message = message.format(
                    robot=self.cfg.robot_name, passer=self.cfg.passage_robot_name
                )
                print(f"[door] env {env_id}: {message}")
        self._announced_stage = torch.maximum(self._announced_stage, self.stage)

    def apply_actions(self):
        door_data = self._door.data
        handle_q = door_data.joint_pos.torch[:, self._joint_ids_tensor[self._handle_index]]
        hinge_q = door_data.joint_pos.torch[:, self._joint_ids_tensor[self._hinge_index]]
        handle_qd = door_data.joint_vel.torch[:, self._joint_ids_tensor[self._handle_index]]
        hinge_qd = door_data.joint_vel.torch[:, self._joint_ids_tensor[self._hinge_index]]

        hand_term = self._env.action_manager.get_term(self.cfg.hand_term_name)
        wrist_commands = self._wrist_commands(hand_term)
        active = self.handle_latched.float()
        active_count = active.sum(dim=1).clamp(min=1.0)

        # positive handle q lowers the lever, so the wrist has to move down
        downward_motion = self._latch_command_pos[..., 2] - wrist_commands[..., 2]
        mean_down = (downward_motion * active).sum(dim=1) / active_count
        handle_target = torch.clamp(
            mean_down / self.cfg.lever_arm_length,
            min=0.0,
            max=0.55,
        )
        handle_effort = self.cfg.handle_position_gain * (handle_target - handle_q)
        handle_effort -= self.cfg.handle_velocity_gain * handle_qd
        handle_effort = torch.clamp(
            handle_effort,
            min=-self.cfg.handle_torque_limit,
            max=self.cfg.handle_torque_limit,
        )
        handle_effort *= self.handle_latched.any(dim=1)

        # unlock only after a downward lever gesture
        self.handle_opened |= self.handle_latched.any(dim=1) & (handle_q >= self.cfg.handle_open_threshold)

        push_motion = wrist_commands[..., 0] - self._latch_command_pos[..., 0]
        mean_push = (push_motion * active).sum(dim=1) / active_count
        signed_push = torch.sign(mean_push) * torch.clamp(
            mean_push.abs() - self.cfg.hinge_push_deadband,
            min=0.0,
        )
        hinge_target = torch.clamp(
            signed_push / max(self.cfg.hinge_full_motion, 1e-6),
            min=-1.0,
            max=1.0,
        ) * self.cfg.hinge_gesture_angle
        gesture_effort = self.cfg.hinge_position_gain * (hinge_target - hinge_q)
        gesture_effort -= self.cfg.hinge_velocity_gain * hinge_qd
        gesture_effort = torch.clamp(
            gesture_effort,
            min=-self.cfg.hinge_torque_limit,
            max=self.cfg.hinge_torque_limit,
        )
        holding = self.handle_latched.any(dim=1)
        gesture_effort *= holding & self.handle_opened

        lock_effort = -(self.cfg.latch_lock_stiffness * hinge_q + self.cfg.latch_lock_damping * hinge_qd)
        lock_effort = torch.clamp(
            lock_effort,
            min=-self.cfg.latch_lock_torque_limit,
            max=self.cfg.latch_lock_torque_limit,
        )
        # Releasing the lever while the leaf is seated re-engages the latch.
        # Once the leaf is clear of the strike it stays free to swing.
        door_unlatched = (holding & (handle_q >= self.cfg.handle_open_threshold)) | (
            self.handle_opened & (hinge_q.abs() >= 0.08)
        )
        hinge_effort = torch.where(door_unlatched, gesture_effort, lock_effort)

        target = torch.stack((hinge_effort, handle_effort), dim=-1).float()
        self._door.set_joint_effort_target_index(target=target, joint_ids=self._joint_ids)

        # also updated at physics rate so the threshold is not missed
        self.opened_by_a |= holding & self.handle_opened & (
            hinge_q.abs() >= self.cfg.hinge_open_threshold
        )

    def _wrist_commands(self, hand_term) -> torch.Tensor:
        """Commanded left/right wrist positions, (num_envs, 2, 3), in metres.

        The upper-body term must provide ``commanded_wrist_positions``.
        """
        reader = getattr(hand_term, "commanded_wrist_positions", None)
        if reader is None:
            raise RuntimeError(
                f"{self.cfg.hand_term_name!r} ({type(hand_term).__name__}) cannot report commanded "
                "wrist positions. DoorOpeningAction reads them in metres, so the upper-body term "
                "has to be one that either receives them (Pink IK) or reconstructs them from the "
                "joint targets it receives (SmoothedJointTargetAction)."
            )
        wrists = reader()
        if wrists.shape[1:] != (2, 3):
            raise RuntimeError(
                f"{self.cfg.hand_term_name!r} reported wrist positions shaped {tuple(wrists.shape)}; "
                "DoorOpeningAction expects (num_envs, 2, 3)."
            )
        return wrists

    def _hand_actions(self, hand_term, hand_names: list[str]) -> torch.Tensor:
        """The two Dex3 command blocks, in ``hand_names`` order.

        Terms that declare ``joint_names`` have their hand columns resolved by name.
        Otherwise the hand block is the trailing columns of the raw action.
        """
        raw = hand_term.raw_actions
        action_names = getattr(hand_term.cfg, "joint_names", None)
        if action_names is None:
            if raw.shape[1] < len(hand_names):
                raise RuntimeError(
                    f"{self.cfg.hand_term_name!r} has {raw.shape[1]} raw action columns, "
                    f"fewer than its {len(hand_names)} hand joints"
                )
            return raw[:, -len(hand_names) :]

        action_names = list(action_names)
        if len(action_names) != raw.shape[1]:
            raise RuntimeError(
                f"{self.cfg.hand_term_name!r} declares {len(action_names)} joint_names but "
                f"has {raw.shape[1]} raw action columns"
            )
        column_by_name = {name: index for index, name in enumerate(action_names)}
        missing = [name for name in hand_names if name not in column_by_name]
        if missing:
            raise RuntimeError(
                f"{self.cfg.hand_term_name!r} is missing hand action columns {missing}; "
                f"joint_names={action_names}"
            )
        columns = torch.as_tensor(
            [column_by_name[name] for name in hand_names],
            dtype=torch.long,
            device=raw.device,
        )
        return raw.index_select(1, columns)

    def _resolve_hand_columns(self, names: list[str]) -> list[torch.Tensor]:
        if self._hand_columns is None:
            self._hand_columns = [
                torch.as_tensor(
                    [i for i, name in enumerate(names) if name.startswith(f"{side}_")],
                    dtype=torch.long,
                    device=self.device,
                )
                for side in ("left", "right")
            ]
            if any(column.numel() == 0 for column in self._hand_columns):
                raise RuntimeError(f"Could not resolve left/right Dex3 joints from {names}")
        return self._hand_columns
