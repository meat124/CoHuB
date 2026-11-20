"""Rollout loop for Diffusion Policy.

`n_obs_steps` is 1, so the loop observes once per action chunk. Set
`OBS_STRIDE = 1` for a policy trained with `n_obs_steps > 1`.
"""

from baselines.utils.rollout import bind

OBS_STRIDE = None

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
