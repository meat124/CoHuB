"""WebSocket env↔policy transport."""

from __future__ import annotations

from typing import Any

from baselines.client_server.ws.model_client import WsModelClient
from baselines.client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
from baselines.client_server.ws.protocol.exceptions import (
    ErrorCode,
    ServerRestartedError,
    WsError,
)
from baselines.client_server.ws.protocol.messages import MessageType
from baselines.client_server.ws.protocol.schemas import Frame

__all__ = [
    "ErrorCode",
    "Frame",
    "MessageType",
    "PolicyEvalClient",
    "PolicyEvalClientConfig",
    "PolicyServer",
    "PolicyServerConfig",
    "ServerRestartedError",
    "WsError",
    "WsModelClient",
]


def __getattr__(name: str) -> Any:
    if name == "PolicyServer":
        from baselines.client_server.ws.model_server import PolicyServer

        return PolicyServer
    if name == "PolicyServerConfig":
        from baselines.client_server.ws.model_server import PolicyServerConfig

        return PolicyServerConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
