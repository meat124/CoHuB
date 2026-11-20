"""Rollout loop for GauDP.

`OBS_STRIDE = 1` observes after every executed step, so each prediction uses a
fresh render. `scripts/eval/eval_policy_client.py` also reads this value.
"""

from baselines.utils.rollout import bind

OBS_STRIDE = 1

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
