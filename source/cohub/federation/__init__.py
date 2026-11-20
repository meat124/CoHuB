"""Two-process federation for running two headsets against one simulation."""

from .link import (
    DEFAULT_ACTION_PORT,
    DEFAULT_STALE_AFTER,
    DEFAULT_STATE_PORT,
    ROBOT_A_ACTION_PORT,
    ROBOT_A_STATE_PORT,
    ports_for_robot,
    PrimaryLink,
    ResetRelay,
    SecondaryLink,
    TELEOP_ACTION_DIM,
    OPERATOR_RAW_DIM,
    ActionPacket,
    ProtocolError,
    StatePacket,
    pack_operator_raw,
)

__all__ = [
    "TELEOP_ACTION_DIM",
    "DEFAULT_ACTION_PORT",
    "DEFAULT_STALE_AFTER",
    "DEFAULT_STATE_PORT",
    "ROBOT_A_ACTION_PORT",
    "ROBOT_A_STATE_PORT",
    "ports_for_robot",
    "OPERATOR_RAW_DIM",
    "ActionPacket",
    "PrimaryLink",
    "ProtocolError",
    "ResetRelay",
    "SecondaryLink",
    "StatePacket",
    "pack_operator_raw",
]
