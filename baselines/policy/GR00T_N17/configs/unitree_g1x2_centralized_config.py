"""GR00T modality config for the CoHuB centralized policy: one policy, both robots.

Video ``ego_a`` + ``ego_b``, state 86, action 70, the pair's shared instruction.

Environment:
    COHUB_SCENE_CAMERA    1 adds the room camera ``scene`` (default 0).
    COHUB_ACTION_HORIZON  chunk length (default 40). ``meta/relative_stats.json``
                          must be generated at the same horizon.
"""

import os
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import cohub_modality  # noqa: E402
from cohub_modality import cohub_keys  # noqa: E402

from gr00t.configs.data.embodiment_configs import register_modality_config  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.data.types import ModalityConfig  # noqa: E402


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"{name} must be a boolean flag (0/1), got {os.environ[name]!r}")


scene_camera = _env_flag("COHUB_SCENE_CAMERA")

action_horizon = int(
    os.environ.get("COHUB_ACTION_HORIZON", cohub_keys.ACTION_HORIZON_FINETUNE_SAFE)
)

unitree_g1x2_centralized_config = cohub_modality.build(
    robot=None, action_horizon=action_horizon
)

if not scene_camera:
    ego_views = [key for robot in ("robot_a", "robot_b") for key in cohub_keys.video_keys(robot)]
    duo_views = unitree_g1x2_centralized_config["video"].modality_keys
    missing = [key for key in ego_views if key not in duo_views]
    if missing:
        raise ValueError(f"ego views {missing} are not in the duo camera set {duo_views}")
    unitree_g1x2_centralized_config["video"] = ModalityConfig(
        delta_indices=unitree_g1x2_centralized_config["video"].delta_indices,
        modality_keys=ego_views,
    )

register_modality_config(
    unitree_g1x2_centralized_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
