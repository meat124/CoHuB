"""Low-dimensional observations of the door, read from the ``door_opening`` action term."""

from __future__ import annotations

import torch


def door_handle_latched(env) -> torch.Tensor:
    """Whether either of Robot A's hands is latched to the handle."""
    return env.action_manager.get_term("door_opening").handle_latched.float()


def door_open_state(env) -> torch.Tensor:
    """Stateful flags: handle turned and door opened by Robot A."""
    term = env.action_manager.get_term("door_opening")
    return torch.stack((term.handle_opened, term.opened_by_a), dim=-1).float()


def door_stage(env) -> torch.Tensor:
    """DoorMan-style diagnostic stage, shaped ``(num_envs, 1)``."""
    return env.action_manager.get_term("door_opening").stage.float().unsqueeze(-1)
