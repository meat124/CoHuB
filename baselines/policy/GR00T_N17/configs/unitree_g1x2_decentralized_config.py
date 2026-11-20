"""GR00T modality config for the CoHuB decentralized and multitask policies.

COHUB_ROBOT selects the target (default ``shared``):

    shared             one multitask policy for either robot, keys without a
                       robot prefix: video ``ego``, state 43, action 35.
    robot_a / robot_b  one single-task policy per robot, prefixed keys.

COHUB_ACTION_HORIZON sets the chunk length (default 40).
``meta/relative_stats.json`` must be generated at the same horizon.
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

robot = os.environ.get("COHUB_ROBOT", cohub_keys.SHARED)
if robot not in ("robot_a", "robot_b", cohub_keys.SHARED):
    raise ValueError(
        f"COHUB_ROBOT must be 'robot_a', 'robot_b' or '{cohub_keys.SHARED}', got {robot!r}"
    )

action_horizon = int(
    os.environ.get("COHUB_ACTION_HORIZON", cohub_keys.ACTION_HORIZON_FINETUNE_SAFE)
)

unitree_g1x2_decentralized_config = cohub_modality.build(
    robot=robot, action_horizon=action_horizon
)

register_modality_config(
    unitree_g1x2_decentralized_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)