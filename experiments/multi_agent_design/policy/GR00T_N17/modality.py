import os
import sys
from pathlib import Path

_CORE_CONFIGS = Path(__file__).resolve().parents[4] / "baselines" / "policy" / "GR00T_N17" / "configs"
if str(_CORE_CONFIGS) not in sys.path:
    sys.path.insert(0, str(_CORE_CONFIGS))

import cohub_modality  # noqa: E402
from cohub_modality import cohub_keys  # noqa: E402

from gr00t.configs.data.embodiment_configs import register_modality_config  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.data.types import ModalityConfig  # noqa: E402

PARTNER_PREFIX = "partner_"
robot = os.environ.get("COHUB_ROBOT", cohub_keys.SHARED)
action_horizon = int(os.environ.get("COHUB_ACTION_HORIZON", cohub_keys.ACTION_HORIZON_FINETUNE_SAFE))

if robot == cohub_keys.SHARED:
    unitree_g1x2_decentralized_config = cohub_modality.build(robot=cohub_keys.SHARED, action_horizon=action_horizon)
elif robot == "shared_global":
    unitree_g1x2_decentralized_config = cohub_modality.build(robot=cohub_keys.SHARED, action_horizon=action_horizon)
    _state = unitree_g1x2_decentralized_config["state"]
    unitree_g1x2_decentralized_config["state"] = ModalityConfig(
        delta_indices=_state.delta_indices,
        modality_keys=[*_state.modality_keys, *(f"{PARTNER_PREFIX}{key}" for key in _state.modality_keys)],
    )
    _video = unitree_g1x2_decentralized_config["video"]
    unitree_g1x2_decentralized_config["video"] = ModalityConfig(
        delta_indices=_video.delta_indices, modality_keys=["ego", "ego_partner"],
    )
elif robot == "duo":
    unitree_g1x2_decentralized_config = cohub_modality.build(robot=None, action_horizon=action_horizon)
    _video = unitree_g1x2_decentralized_config["video"]
    unitree_g1x2_decentralized_config["video"] = ModalityConfig(
        delta_indices=_video.delta_indices,
        modality_keys=[key for r in cohub_keys.ROBOTS for key in cohub_keys.video_keys(r)],
    )
else:
    raise ValueError(f"COHUB_ROBOT must be shared, shared_global or duo here, got {robot!r}")

register_modality_config(unitree_g1x2_decentralized_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
