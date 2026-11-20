"""Rollout loop for ACT.

ACT reads a single frame, so the loop observes once per action chunk. With
`temporal_agg` on, `model.py` returns one action per call and the loop
observes every step.
"""

from baselines.utils.rollout import bind

OBS_STRIDE = None

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
