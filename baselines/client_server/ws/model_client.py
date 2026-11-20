"""Synchronous environment-side adapter for the policy websocket protocol."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

from baselines.client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig


class WsModelClient:
    """Synchronous env-side client for the websocket policy protocol.

    The event loop runs on a background thread, so keepalive pings are
    answered while the caller blocks. Not thread-safe: call from one thread.
    `step` counts inference calls, not environment control steps.
    """

    def __init__(
        self,
        *,
        url: str,
        evaluation_id: str,
        trial_id: str,
        action_case_id: str | None = None,
        repeat_index: int | None = None,
        ws_ping_interval_s: float | None = 20.0,
        ws_ping_timeout_s: float | None = 20.0,
        connect_timeout_s: float | None = None,
        handshake_timeout_s: float | None = None,
        request_timeout_s: float | None = None,
        max_connect_attempts: int | None = None,
        connect_retry_delay_s: float | None = None,
        max_connect_seconds: float | None = None,
        close_timeout_s: float | None = None,
        client: Any | None = None,
    ):
        self.action_case_id = action_case_id
        self.trial_id = trial_id
        self.repeat_index = repeat_index
        self._step = 0
        # None keeps the PolicyEvalClientConfig default.
        tuning = {
            "connect_timeout_s": connect_timeout_s,
            "handshake_timeout_s": handshake_timeout_s,
            "request_timeout_s": request_timeout_s,
            "max_connect_attempts": max_connect_attempts,
            "connect_retry_delay_s": connect_retry_delay_s,
            "max_connect_seconds": max_connect_seconds,
            "close_timeout_s": close_timeout_s,
        }
        self._client = client or PolicyEvalClient(
            PolicyEvalClientConfig(
                url=url,
                evaluation_id=evaluation_id,
                ws_ping_interval_s=ws_ping_interval_s,
                ws_ping_timeout_s=ws_ping_timeout_s,
                **{k: v for k, v in tuning.items() if v is not None},
            )
        )
        # Start the loop thread only after the config is validated.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="ws-model-client", daemon=True
        )
        self._loop_thread.start()
        try:
            self._run(self._client.connect(handshake=True))
        except BaseException:
            self._shutdown_loop()
            raise

    def _run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run a client coroutine on the background loop and wait for it."""
        if self._loop.is_closed() or not self._loop_thread.is_alive():
            coro.close()
            raise RuntimeError("websocket client event loop is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _shutdown_loop(self) -> None:
        if self._loop.is_closed():
            return
        if self._loop_thread.is_alive():
            # Cancel and drain leftover tasks before closing the loop.
            async def _drain() -> None:
                tasks = [
                    t
                    for t in asyncio.all_tasks()
                    if t is not asyncio.current_task()
                ]
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            try:
                asyncio.run_coroutine_threadsafe(_drain(), self._loop).result(
                    timeout=5.0
                )
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5.0)
        if not self._loop_thread.is_alive():
            self._loop.close()

    def call(self, func_name: str | None = None, obs: Any = None, **kwargs: Any) -> Any:
        # `obs` is the only payload channel; extra kwargs are rejected.
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"WsModelClient.call() only accepts func_name and obs; "
                f"unsupported keyword argument(s): {unexpected}. "
                f"For get_action_batch, pass env indices as obs=env_idx_list."
            )

        if func_name == "prepare_case":
            if self.action_case_id is None:
                raise ValueError("prepare_case requires action_case_id")
            response = self._run(
                self._client.prepare_case(
                    self.action_case_id,
                    case_meta=self._dict_payload(func_name, obs),
                    repeat_index=self.repeat_index,
                )
            )
            return response.payload.get("result")

        if func_name == "reset":
            # model.reset() takes no arguments.
            if obs is not None:
                raise TypeError("reset takes no obs payload")
            self._step = 0
            response = self._run(
                self._client.reset(
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    repeat_index=self.repeat_index,
                )
            )
            return response.payload.get("result")

        if func_name == "trial_end":
            response = self._run(
                self._client.trial_end(
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    result=self._dict_payload(func_name, obs),
                    repeat_index=self.repeat_index,
                )
            )
            return response.payload.get("result")

        if not isinstance(func_name, str) or not func_name or func_name.startswith("_"):
            raise NotImplementedError(f"unsupported websocket model call: {func_name!r}")

        # Any other func_name is dispatched to the model method of that name.
        response = self._run(
            self._client.call(
                func_name,
                obs,
                trial_id=self.trial_id,
                action_case_id=self.action_case_id,
                repeat_index=self.repeat_index,
                step=self._step,
            )
        )
        if func_name in {"get_action", "get_action_batch"}:
            self._step += 1
        return response.payload.get("result")

    @staticmethod
    def _dict_payload(func_name: str, obs: Any) -> dict | None:
        if obs is not None and not isinstance(obs, dict):
            raise TypeError(
                f"{func_name} payload must be a dict or None, "
                f"got {type(obs).__name__}"
            )
        return obs

    def close(self) -> None:
        if self._loop.is_closed():
            return
        try:
            if self._loop_thread.is_alive():
                self._run(self._client.close())
        finally:
            self._shutdown_loop()

    def __enter__(self) -> WsModelClient:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
