"""Ψ₀'s action and state layout, and the permutation from CoHuB's.

Shared by ``scripts/export/convert_lerobot.py --format psi0`` (dataset builder) and the adapter
``model.py``. Imports numpy only, so both environments can load it.

Ψ₀ order (left before right inside ``hand`` and ``arm``)::

    action  ->  hand(14) + arm(14) + rpy(3) + height(1) + vx + vy + vyaw + target_yaw   # 36
    state   ->  hand(14) + arm(14) + torso_rpy(3) + torso_height(1)                     # 32

CoHuB order (``source/cohub/keys.py``)::

    action  ->  left_arm(7) right_arm(7) left_hand(7) right_hand(7) waist(3)
                base_height(1) navigate(3)                                  # 35
    state   ->  left_leg(6) right_leg(6) waist(3)
                left_arm(7) left_hand(7) right_arm(7) right_hand(7)         # 43

``waist`` is ``(yaw, roll, pitch)`` in CoHuB and ``(roll, pitch, yaw)`` in Ψ₀.
``torso_height`` holds the last commanded base height. ``target_yaw`` stays zero.
"""

from __future__ import annotations

import numpy as np

# CoHuB layout

COHUB_ACTION_DIM = 35
COHUB_STATE_DIM = 43

COHUB_ACTION_SLICES = {
    "left_arm": (0, 7),
    "right_arm": (7, 14),
    "left_hand": (14, 21),
    "right_hand": (21, 28),
    "waist": (28, 31),          # (yaw, roll, pitch)
    "base_height": (31, 32),
    "base_vel": (32, 35),       # (vx, vy, vyaw)
}
"""Columns of ``action``."""

COHUB_STATE_SLICES = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),          # (yaw, roll, pitch)
    "left_arm": (15, 22),
    "left_hand": (22, 29),
    "right_arm": (29, 36),
    "right_hand": (36, 43),
}
"""Columns of ``observation.state``."""

# Ψ₀ layout

ACTION_DIM = 36

ACTION_COLUMNS = 35
STATE_COLUMNS = 32
"""Columns this module writes. Ψ₀ pads both to 36."""

NOMINAL_HEIGHT = 0.72
"""Metres. Fills the state's ``torso_height`` slot before the policy has commanded a height."""

_WAIST_YAW, _WAIST_ROLL, _WAIST_PITCH = 0, 1, 2
_RPY_FROM_WAIST = (_WAIST_ROLL, _WAIST_PITCH, _WAIST_YAW)
"""``(roll, pitch, yaw)`` picked out of CoHuB's ``(yaw, roll, pitch)``."""


def _span(slices: dict[str, tuple[int, int]], group: str) -> list[int]:
    start, end = slices[group]
    return list(range(start, end))


def _waist(slices: dict[str, tuple[int, int]]) -> list[int]:
    start, _ = slices["waist"]
    return [start + offset for offset in _RPY_FROM_WAIST]


ACTION_PERM: list[int] = (
    _span(COHUB_ACTION_SLICES, "left_hand")
    + _span(COHUB_ACTION_SLICES, "right_hand")
    + _span(COHUB_ACTION_SLICES, "left_arm")
    + _span(COHUB_ACTION_SLICES, "right_arm")
    + _waist(COHUB_ACTION_SLICES)
    + _span(COHUB_ACTION_SLICES, "base_height")
    + _span(COHUB_ACTION_SLICES, "base_vel")
)
"""``psi_action[i] = cohub_action[ACTION_PERM[i]]`` for the 35 columns."""

STATE_PERM: list[int] = (
    _span(COHUB_STATE_SLICES, "left_hand")
    + _span(COHUB_STATE_SLICES, "right_hand")
    + _span(COHUB_STATE_SLICES, "left_arm")
    + _span(COHUB_STATE_SLICES, "right_arm")
    + _waist(COHUB_STATE_SLICES)
)
"""``psi_state[i] = cohub_state[STATE_PERM[i]]`` for the first 31 columns. The legs are dropped."""

assert len(ACTION_PERM) == ACTION_COLUMNS
assert len(STATE_PERM) == STATE_COLUMNS - 1        # minus the height slot
assert sorted(ACTION_PERM) == list(range(COHUB_ACTION_DIM))

_ACTION_PERM = np.asarray(ACTION_PERM)
_ACTION_INVERSE = np.argsort(_ACTION_PERM)
_STATE_PERM = np.asarray(STATE_PERM)


def action_to_psi0(action: np.ndarray) -> np.ndarray:
    """CoHuB's 35 action columns -> Ψ₀'s first 35. Trailing axis, any rank."""
    action = np.asarray(action)
    if action.shape[-1] != COHUB_ACTION_DIM:
        raise ValueError(f"expected a {COHUB_ACTION_DIM}D CoHuB action, got {action.shape}")
    return action[..., _ACTION_PERM]


def action_from_psi0(action: np.ndarray) -> np.ndarray:
    """Ψ₀'s action -> CoHuB's 35 columns. A 36-wide input drops ``target_yaw``."""
    action = np.asarray(action)
    if action.shape[-1] == ACTION_DIM:
        action = action[..., :ACTION_COLUMNS]
    if action.shape[-1] != ACTION_COLUMNS:
        raise ValueError(f"expected a {ACTION_COLUMNS}D or {ACTION_DIM}D Ψ₀ action, got {action.shape}")
    return action[..., _ACTION_INVERSE]


def state_to_psi0(state: np.ndarray, height: float | np.ndarray = NOMINAL_HEIGHT) -> np.ndarray:
    """CoHuB's 43 joint angles -> Ψ₀'s 32 state columns.

    ``height`` is the last commanded base height. A scalar broadcasts over a stack.
    """
    state = np.asarray(state)
    if state.shape[-1] != COHUB_STATE_DIM:
        raise ValueError(f"expected a {COHUB_STATE_DIM}D CoHuB state, got {state.shape}")
    joints = state[..., _STATE_PERM]
    tail = np.broadcast_to(np.asarray(height, dtype=joints.dtype).reshape(-1)[..., None]
                           if np.ndim(height) else np.asarray(height, dtype=joints.dtype),
                           joints.shape[:-1] + (1,))
    return np.concatenate([joints, tail], axis=-1)
