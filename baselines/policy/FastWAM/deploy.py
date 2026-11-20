"""Rollout loop for FastWAM. The policy reads only the current frame, so it observes once per chunk."""

from baselines.utils.rollout import bind

OBS_STRIDE = None

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
