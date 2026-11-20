"""A robot without a headset: the 32D action slice from a constant pose or the keyboard."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
import weakref

import carb
import numpy as np
import omni

from cohub.keys import TELEOP_ACTION_DIM
from cohub.teleop.controls import quat_mul, shape_yaw


# --- Follower device ---
# The 32D slice: left wrist pose (7, env-frame xyz + xyzw) | right wrist pose (7) | hand joints (14) |
# loco (vx, vy, wz, hip height).

# Nominal carry pose in the pelvis-local frame (x forward, y left, z up) [m].
NOMINAL_LEFT_POS = np.array([0.25, 0.15, 0.12], dtype=np.float32)
NOMINAL_RIGHT_POS = np.array([0.25, -0.15, 0.12], dtype=np.float32)
# Wrist orientation of that pose, pelvis-local (x, y, z, w).
NOMINAL_LEFT_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
NOMINAL_RIGHT_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def quat_apply(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by an (x, y, z, w) quaternion."""
    xyz, w = quat_xyzw[:3], quat_xyzw[3]
    t = 2.0 * np.cross(xyz, vec)
    return (vec + w * t + np.cross(xyz, t)).astype(np.float32)

HAND_OPEN = np.zeros(14, dtype=np.float32)
# Closed-hand joint targets for both Dex3 hands. A magnitude: the action term
# applies the closing sign per joint.
HAND_CLOSED = np.full(14, 0.6, dtype=np.float32)

DEFAULT_HIP_HEIGHT = 0.72


@dataclass
class FollowerState:
    """Operator-adjustable offsets on top of the nominal pose."""

    height_offset: float = 0.0  # dpad up/down -> wrist z offset [m]
    forward_offset: float = 0.0  # dpad left/right -> wrist x offset [m]
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    hip_height: float = DEFAULT_HIP_HEIGHT
    grasp: bool = False
    hand_joints: np.ndarray = field(default_factory=lambda: HAND_OPEN.copy())


class FollowerDevice(ABC):
    """Produces a robot's 32D action slice from operator input.

    Also carries session-control flags (start/stop, reset).
    """

    def __init__(self):
        self.state = FollowerState()
        self._pending_toggle_run = False
        self._pending_reset = False
        # See cohub.teleop.controls.
        self.shape_yaw_command = True

    def consume_control_flags(self) -> tuple[bool, bool]:
        """Return (toggle_run, reset) requested since the last call."""
        toggle, reset = self._pending_toggle_run, self._pending_reset
        self._pending_toggle_run = False
        self._pending_reset = False
        return toggle, reset

    @abstractmethod
    def update(self) -> None:
        """Refresh ``self.state`` from the input device."""

    def reset(self) -> None:
        self.state = FollowerState()

    def advance(self, root_pos: np.ndarray | None = None, root_quat: np.ndarray | None = None) -> np.ndarray:
        """Return the 32D slice for the current frame.

        Args:
            root_pos: The robot's pelvis position in the env frame [m].
            root_quat: The pelvis orientation as (x, y, z, w).
        """
        self.update()
        s = self.state

        if root_pos is None:
            root_pos = np.zeros(3, dtype=np.float32)
        if root_quat is None:
            root_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        offset = np.array([s.forward_offset, 0.0, s.height_offset], dtype=np.float32)
        left_pos = root_pos + quat_apply(root_quat, NOMINAL_LEFT_POS + offset)
        right_pos = root_pos + quat_apply(root_quat, NOMINAL_RIGHT_POS + offset)
        left_quat = quat_mul(root_quat, NOMINAL_LEFT_QUAT)
        right_quat = quat_mul(root_quat, NOMINAL_RIGHT_QUAT)
        hands = HAND_CLOSED if s.grasp else s.hand_joints

        action = np.zeros(TELEOP_ACTION_DIM, dtype=np.float32)
        action[0:3] = left_pos
        action[3:7] = left_quat
        action[7:10] = right_pos
        action[10:14] = right_quat
        action[14:28] = hands
        action[28] = s.vx
        action[29] = s.vy
        action[30] = shape_yaw(s.wz) if self.shape_yaw_command else s.wz
        action[31] = s.hip_height
        return action


class ConstantFollower(FollowerDevice):
    """Frozen nominal pose, standing in place."""

    def update(self) -> None:
        pass


def follower_action(env, robot_name: str, follower: FollowerDevice) -> np.ndarray:
    """One robot's 32D slice, anchored to that robot's current pelvis pose."""
    data = env.scene[robot_name].data
    return follower.advance(
        data.root_pos_w.torch[0].cpu().numpy(),
        data.root_quat_w.torch[0].cpu().numpy(),
    )


# --- Keyboard follower ---
# I/K vx, J/L vy, U/O wz; arrows move the wrists; SPACE grasp; R reset offsets; P start/stop
# recording; N reset the episode.

_VEL_STEP = 0.3
# Wrist offset per update [m].
_OFFSET_STEP = 0.006
# Per-update exponential ramp toward the commanded velocity.
_VEL_SMOOTHING = 0.12

# A key can arrive under several names: the key id, the character it
# produces, or the Hangul character of the same key under a Korean IME.
_KEY_ALIASES = {
    " ": "SPACE",
    "ARROW_UP": "UP",
    "ARROW_DOWN": "DOWN",
    "ARROW_LEFT": "LEFT",
    "ARROW_RIGHT": "RIGHT",
    "ㅑ": "I",
    "ㅓ": "J",
    "ㅏ": "K",
    "ㅣ": "L",
    "ㅕ": "U",
    "ㅐ": "O",
    "ㄱ": "R",
}

# Keys that act once per press rather than while held.
_TOGGLE_KEYS = ("SPACE", "R", "P", "N", "M")
_TOGGLE_DEBOUNCE_S = 0.25


class KeyboardFollower(FollowerDevice):
    def __init__(self):
        super().__init__()
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_key(event, *args),
        )
        self._pressed: set[str] = set()
        self._seen: set[str] = set()
        self._last_toggle: dict[str, float] = {}

    def __del__(self):
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None

    def _on_key(self, event, *args) -> bool:
        # event.input is a string or a carb enum, depending on the build.
        raw = event.input
        name = (raw if isinstance(raw, str) else raw.name).upper()
        name = _KEY_ALIASES.get(name, name)
        etype = event.type
        is_press = etype == carb.input.KeyboardEventType.KEY_PRESS or str(etype).endswith("KEY_PRESS")
        is_release = etype == carb.input.KeyboardEventType.KEY_RELEASE or str(etype).endswith("KEY_RELEASE")
        if name not in self._seen:
            self._seen.add(name)
            print(f"[KeyboardFollower] key seen: {name!r}")
        if is_press:
            self._pressed.add(name)
            # Debounced on time: a duplicate delivery can arrive after the release.
            if name in _TOGGLE_KEYS:
                now = time.monotonic()
                if now - self._last_toggle.get(name, 0.0) < _TOGGLE_DEBOUNCE_S:
                    return True
                self._last_toggle[name] = now
            if name == "SPACE":
                self.state.grasp = not self.state.grasp
                print(f"[follower] grasp {'CLOSED' if self.state.grasp else 'OPEN'}")
            elif name == "R":
                self.state.height_offset = 0.0
                self.state.forward_offset = 0.0
                self.state.hip_height = DEFAULT_HIP_HEIGHT
            elif name == "P":
                self._pending_toggle_run = True
            elif name == "N":
                self._pending_reset = True
        elif is_release:
            self._pressed.discard(name)
        return True

    def update(self) -> None:
        s = self.state
        target_vx = _VEL_STEP * (("I" in self._pressed) - ("K" in self._pressed))
        target_vy = _VEL_STEP * (("J" in self._pressed) - ("L" in self._pressed))
        target_wz = _VEL_STEP * (("U" in self._pressed) - ("O" in self._pressed))
        s.vx += _VEL_SMOOTHING * (target_vx - s.vx)
        s.vy += _VEL_SMOOTHING * (target_vy - s.vy)
        s.wz += _VEL_SMOOTHING * (target_wz - s.wz)
        if "UP" in self._pressed:
            s.height_offset = min(s.height_offset + _OFFSET_STEP, 0.3)
        if "DOWN" in self._pressed:
            s.height_offset = max(s.height_offset - _OFFSET_STEP, -0.3)
        if "RIGHT" in self._pressed:
            s.forward_offset = min(s.forward_offset + _OFFSET_STEP, 0.2)
        if "LEFT" in self._pressed:
            s.forward_offset = max(s.forward_offset - _OFFSET_STEP, -0.2)
