"""Rollout loop for pi0.5. The policy reads only the current frame, so it observes once per chunk.

`exec_horizon` in `deploy.yml` sets how much of the chunk runs before re-observing.
"""

from baselines.utils.rollout import bind

OBS_STRIDE = None

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
