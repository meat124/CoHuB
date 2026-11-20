"""Rollout loop for GR00T N1.7.

`OBS_STRIDE = None` observes once per action chunk: the model conditions on the
current frame only.
"""

from baselines.utils.rollout import bind

OBS_STRIDE = None

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
