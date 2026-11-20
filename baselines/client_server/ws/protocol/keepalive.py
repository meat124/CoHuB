"""Validation of the websocket keepalive settings shared by server and client."""

from __future__ import annotations

import math
from typing import Any


def normalize_ws_ping(value: Any, *, field_name: str) -> float | None:
    """Return the interval in seconds, or None when keepalive is disabled.

    `null` turns keepalive off. Booleans are rejected.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be null or a positive number, got {value!r}"
        )
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(
            f"{field_name} must be null or a positive finite number, got {value!r}"
        )
    return seconds
