"""Rollout loop for LatentToM.

Both arms condition on a window of `n_obs_steps: 2` frames, so the loop
observes after every executed step.
"""

from baselines.utils.rollout import bind

OBS_STRIDE = 1

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
