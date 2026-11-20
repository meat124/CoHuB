"""Rollout loop shared by every policy, parameterized by the observation rate.

    obs_stride=1     observe after every executed step (default; needed by
                     history-conditioned policies)
    obs_stride=None  observe once per action chunk
    obs_stride=k     observe every k steps inside the chunk

Each `deploy.py` declares its stride as a module-level `OBS_STRIDE`.
"""

from __future__ import annotations


def _observe(TASK_ENV, model_client) -> None:
    model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())


def _should_observe(index: int, obs_stride: int | None) -> bool:
    """Whether to re-observe after the `index`-th action of a chunk (0-based).

    `None` never re-observes inside a chunk; `k` does so every k steps.
    """
    if not obs_stride:
        return False
    return (index + 1) % obs_stride == 0


def eval_one_episode(TASK_ENV, model_client, obs_stride: int | None = 1) -> None:
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        _observe(TASK_ENV, model_client)
        actions = model_client.call(func_name="get_action")

        for index, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or index + 1 == len(actions):
                break

            if _should_observe(index, obs_stride):
                _observe(TASK_ENV, model_client)


def eval_one_episode_batch(TASK_ENV, model_client, obs_stride: int | None = 1) -> None:
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        chunk_size = len(actions[0])
        for index in range(chunk_size):
            TASK_ENV.take_action_batch(
                [env_actions[index] for env_actions in actions], env_idx_list
            )

            if TASK_ENV.is_episode_end() or index + 1 == chunk_size:
                break

            # Environments that finished mid-chunk drop out.
            running = set(TASK_ENV.get_running_env_idx_list())
            keep = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            if len(keep) != len(env_idx_list):
                actions = [actions[i] for i in keep]
                env_idx_list = [env_idx_list[i] for i in keep]

            if _should_observe(index, obs_stride):
                model_client.call(
                    func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list)
                )


def bind(obs_stride: int | None):
    """Return the two loop entry points a `deploy.py` exports, bound to one stride."""

    def eval_one_episode_bound(TASK_ENV, model_client):
        return eval_one_episode(TASK_ENV, model_client, obs_stride=obs_stride)

    def eval_one_episode_batch_bound(TASK_ENV, model_client):
        return eval_one_episode_batch(TASK_ENV, model_client, obs_stride=obs_stride)

    return eval_one_episode_bound, eval_one_episode_batch_bound
