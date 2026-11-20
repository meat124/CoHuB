"""Builds a GR00T modality config over CoHuB's exported LeRobot dataset.

Imported by the config files beside this one. Each registers one variant, and
only one config may register against ``NEW_EMBODIMENT`` in a process.
"""

import importlib.util
import sys
from pathlib import Path

# Load source/cohub/keys.py by path: the policy venv has no cohub package.
_KEYS_PY = Path(__file__).resolve().parents[4] / "source" / "cohub" / "keys.py"
_spec = importlib.util.spec_from_file_location("cohub_keys", _KEYS_PY)
cohub_keys = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("cohub_keys", cohub_keys)
_spec.loader.exec_module(cohub_keys)

from gr00t.data.types import (  # noqa: E402
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


def _action_config(key: str) -> ActionConfig:
    """Action representation of one key: arms RELATIVE, everything else ABSOLUTE.

    A relative action key needs a state key of the same name and width.
    """
    # Matches both `left_arm` (shared dataset) and `robot_a_left_arm`.
    relative = key.endswith("left_arm") or key.endswith("right_arm")
    return ActionConfig(
        rep=ActionRepresentation.RELATIVE if relative else ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


def build(robot: str | None = None, action_horizon: int | None = None) -> dict:
    """The four modality configs for one CoHuB policy.

    Args:
        robot: ``"robot_a"``/``"robot_b"`` for a policy driving one robot,
            ``None`` for one policy driving both, ``"shared"`` for the
            multitask policy whose dataset keys carry no robot prefix.
        action_horizon: Chunk length. Defaults to
            :data:`cohub_keys.ACTION_HORIZON`.

    State and video use no history. The action chunk starts at index 0, where
    relative actions read their reference state.
    """
    horizon = cohub_keys.ACTION_HORIZON if action_horizon is None else action_horizon
    action_keys = cohub_keys.action_keys(robot)
    return {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=cohub_keys.video_keys(robot),
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=cohub_keys.state_keys(robot),
        ),
        "action": ModalityConfig(
            delta_indices=list(range(horizon)),
            modality_keys=action_keys,
            action_configs=[_action_config(key) for key in action_keys],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[cohub_keys.language_key(robot)],
        ),
    }
