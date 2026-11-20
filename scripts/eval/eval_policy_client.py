"""The simulator side of an evaluation: a policy's episode loop against `CoHuBTaskEnv`.

    python scripts/eval/eval_policy_client.py --bench_name cohub \\
        --task_name smoketest --env_cfg_type cocarry --policy_name LatentToM \\
        --host localhost --port 8765 --root_dir baselines --device_id 0 --seed 0

Usually started by scripts/eval/runner.sh.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import socket
import sys
import time
from pathlib import Path


from cohub.results import subtask_summary, write_json_atomically  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _write_results(
    args: argparse.Namespace, results: list[dict], *, final: bool, provenance: dict | None = None
) -> None:
    if not args.results_path:
        return
    successes = sum(1 for r in results if r.get("success"))
    payload = {
        "bench_name": args.bench_name,
        "task_name": args.task_name,
        "env_cfg_type": args.env_cfg_type,
        "policy_name": args.policy_name,
        "seed": args.seed,
        "provenance": provenance,
        "episode_step_limit": args.episode_step_limit,
        "eval_episode_num": args.eval_episode_num,
        "episodes_run": len(results),
        "successes": successes,
        "success_rate": round(successes / len(results), 4) if results else None,
        **subtask_summary(results),
        "final": final,
        "episodes": results,
    }
    write_json_atomically(Path(args.results_path), payload)


def seed_policy_server(model_client, seed: int) -> bool | None:
    """Reseed the policy server for the episode about to run.

    Returns what the server reported, or None when it has no `seed_episode` RPC.
    """
    try:
        reply = model_client.call(func_name="seed_episode", obs={"seed": int(seed)})
    except Exception as exc:  # noqa: BLE001 -- the failure mode is "no such RPC"
        if not getattr(seed_policy_server, "_warned", False):
            seed_policy_server._warned = True
            print(
                f"[eval_policy_client] server has no seed_episode ({type(exc).__name__}: {exc}); "
                "its noise is not seeded per episode",
                flush=True,
            )
        return None
    return bool(isinstance(reply, dict) and reply.get("seeded"))


def build_provenance(args: argparse.Namespace, *, obs_stride: int, obs_cameras: list[str],
                     video_cameras: list[str]) -> dict:
    """What this client evaluated under, for the results file.

    Merges the runner's JSON in `COHUB_EVAL_PROVENANCE` with the host, GPU,
    render and observation settings of this process.
    """
    from cohub.runtime import RENDER_ANTIALIASING, RENDER_SPP
    from cohub.policy_env import SETTLE_RENDERS_MAX

    provenance: dict = {}
    raw = os.environ.get("COHUB_EVAL_PROVENANCE")
    if raw:
        try:
            provenance.update(json.loads(raw))
        except ValueError:
            provenance["runner_provenance_unparsed"] = raw[:200]
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 -- provenance must never fail the run
        pass
    from cohub.policy_env import obs_codec_setting

    provenance.update(
        host=socket.gethostname().split(".")[0],
        gpu=gpu,
        obs_stride=obs_stride,
        obs_cameras=obs_cameras,
        video_cameras=video_cameras,
        obs_codec=obs_codec_setting() or "none",
        settle_renders_max=int(os.environ.get("COHUB_SETTLE_RENDERS", SETTLE_RENDERS_MAX)),
        render_spp=int(os.environ.get("COHUB_RENDER_SPP", RENDER_SPP)),
        antialiasing=os.environ.get("COHUB_AA", RENDER_ANTIALIASING),
        episode_step_limit=args.episode_step_limit,
        upper_body_mode="joint",
        scene_camera=args.scene_camera,
        smooth_video=args.smooth_video,
        video_stride=args.video_stride,
        physics_device=args.device or f"cuda:{args.device_id}",
    )
    return provenance


def str2bool(v: str) -> bool:
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"boolean value expected, got {v!r}")


def build_parser() -> argparse.ArgumentParser:
    """The client's flags, as a parser another client can add to."""
    parser = argparse.ArgumentParser()
    # The flags scripts/eval/runner.sh passes.
    parser.add_argument("--bench_name", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--env_cfg_type", required=True, help="baselines/env_cfg/<env_cfg_type>.yml")
    parser.add_argument("--policy_name", required=True, help="policy module under baselines/policy, e.g. LatentToM")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--protocol", choices=("legacy_tcp", "ws"), default="ws")
    parser.add_argument("--eval_batch", type=str2bool, default=False)
    parser.add_argument("--root_dir", required=True, help="baselines/, the package the policies import from")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--additional_info", default="")
    parser.add_argument("--seed", type=int, default=0)
    # Extras with defaults.
    parser.add_argument("--eval_episode_num", type=int, default=1)
    parser.add_argument("--evaluation_id", default="sim-eval")
    parser.add_argument("--trial_id", default="sim-trial")
    parser.add_argument("--action_case_id", default=None)
    parser.add_argument("--repeat_index", type=int, default=None)
    parser.add_argument("--policy_server_url", default=None)
    # 1000 steps = 20 s at 50 Hz.
    parser.add_argument("--episode_step_limit", type=int, default=1000)
    parser.add_argument("--headless", type=str2bool, default=True)
    parser.add_argument("--visualizer", default=None, help="Pass 'kit' for a real Isaac Sim window. Omit for headless.")
    parser.add_argument(
        "--video_dir", default=None, help="Write one mp4 per episode per camera, under <dir>/<camera>/."
    )
    parser.add_argument(
        "--video_cameras",
        default="scene",
        help="Cameras to record (scene, ego_a, ego_b; separators ',:+ '). An observed one reuses the policy's frame.",
    )
    parser.add_argument(
        "--video_scale",
        type=int,
        default=1,
        help="Integer upsampling factor for the written videos (done by ffmpeg). 1 = native.",
    )
    parser.add_argument(
        "--scene_camera",
        choices=("fixed", "follow"),
        default="fixed",
        help="'fixed': the authored pose. 'follow': track the robots (refused when 'scene' is in --obs_cameras).",
    )
    parser.add_argument(
        "--follow_distance",
        type=float,
        default=2.0,
        help="Distance the following scene camera keeps from the robots' centroid [m].",
    )
    parser.add_argument(
        "--smooth_video",
        type=str2bool,
        default=False,
        help="Record policy-input cameras from twin sensors at the recording rate, not the policy's held frame.",
    )
    parser.add_argument(
        "--follow_max_dolly",
        type=float,
        default=1.5,
        help="Furthest the following camera may travel from the authored pose [m].",
    )
    parser.add_argument(
        "--obs_cameras",
        default="scene,ego_a,ego_b",
        help="Cameras get_obs() reads (scene, ego_a, ego_b; separators ',:+ '). Each one read costs a render.",
    )
    parser.add_argument(
        "--obs_stride",
        type=int,
        default=None,
        help="Observe every N control steps. Default: the policy's OBS_STRIDE, else COHUB_OBS_STRIDE, else 1.",
    )
    parser.add_argument(
        "--video_stride",
        type=int,
        default=1,
        help="Record one video frame every N steps; the file's fps is divided to keep playback real time.",
    )
    parser.add_argument(
        "--ego_depth",
        type=str2bool,
        default=True,
        help="Keep the ego cameras' depth annotator. Nothing in eval reads depth, but the "
        "dataset was rendered with it on.",
    )
    parser.add_argument(
        "--drop_unused_cameras",
        type=str2bool,
        default=False,
        help="Remove sensors that are in neither --obs_cameras nor --video_cameras from the scene.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Physics device ('cpu' or 'cuda:N'); overrides --device_id when given.",
    )
    parser.add_argument("--results_path", default=None, help="Write per-episode results + summary JSON here.")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


class TimedModelClient:
    """A model client that times every RPC.

    Everything that is not `call` is forwarded to the wrapped client.
    """

    def __init__(self, inner):
        self.__dict__["_inner"] = inner
        self.__dict__["timing"] = {}
        self.__dict__["counts"] = {}

    def call(self, *call_args, **call_kwargs):
        name = call_kwargs.get("func_name") or (call_args[0] if call_args else "?")
        t0 = time.perf_counter()
        try:
            return self._inner.call(*call_args, **call_kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            self.timing[name] = self.timing.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    def reset_timing(self) -> None:
        self.timing.clear()
        self.counts.clear()

    def snapshot(self) -> dict:
        return {
            "rpc_s": {name: round(seconds, 3) for name, seconds in sorted(self.timing.items())},
            "rpc_n": dict(sorted(self.counts.items())),
        }

    def __getattr__(self, name):
        return getattr(self.__dict__["_inner"], name)


def resolve_obs_stride(args: argparse.Namespace, eval_module) -> int:
    """How many control steps one observation is reused for.

    Order: `--obs_stride`, then `COHUB_OBS_STRIDE`, then the policy module's
    `OBS_STRIDE`, then 1.
    """
    if args.obs_stride is not None:
        return max(1, args.obs_stride)
    from_env = os.environ.get("COHUB_OBS_STRIDE")
    if from_env:
        return max(1, int(from_env))
    declared = getattr(eval_module, "OBS_STRIDE", None)
    if isinstance(declared, int) and declared >= 1:
        return declared
    return 1


_CAMERA_SEPARATORS = ",:+ \t"


def camera_list(value: str) -> list[str]:
    """Split a camera list on any of `,`, `:`, `+` or whitespace."""
    out: list[str] = []
    for chunk in re.split(f"[{re.escape(_CAMERA_SEPARATORS)}]+", value or ""):
        if chunk:
            out.append(chunk)
    return out


def resolve_rollout(args: argparse.Namespace, eval_module, obs_stride: int):
    """The episode loop to drive: (run_episode, run_episode_batch, shared).

    `shared` is True when the policy's `deploy.py` declares `OBS_STRIDE`. With
    an explicit `--obs_stride`, a shared-loop policy runs the shared loop at
    that stride. Otherwise the policy's own loop is used.
    """
    shared = hasattr(eval_module, "OBS_STRIDE")
    if args.obs_stride is None or not shared:
        return eval_module.eval_one_episode, eval_module.eval_one_episode_batch, shared

    from functools import partial

    from baselines.utils.rollout import eval_one_episode, eval_one_episode_batch

    return (
        partial(eval_one_episode, obs_stride=obs_stride),
        partial(eval_one_episode_batch, obs_stride=obs_stride),
        shared,
    )


def build_model_client(args: argparse.Namespace):
    deploy_cfg = {
        "protocol": args.protocol,
        "policy_server_url": args.policy_server_url,
        "host": args.host,
        "port": args.port,
        "evaluation_id": args.evaluation_id,
        "trial_id": args.trial_id,
        "action_case_id": args.action_case_id,
        "repeat_index": args.repeat_index,
    }
    if args.protocol == "ws":
        from baselines.client_server.ws import WsModelClient

        policy_server_url = deploy_cfg["policy_server_url"] or f"ws://{args.host}:{args.port}"
        # EVAL_RPC_TIMEOUT_S overrides the client's request timeout (seconds).
        rpc_timeout = os.environ.get("EVAL_RPC_TIMEOUT_S", "").strip()
        return WsModelClient(
            url=policy_server_url,
            evaluation_id=deploy_cfg["evaluation_id"],
            trial_id=deploy_cfg["trial_id"],
            action_case_id=deploy_cfg["action_case_id"],
            repeat_index=deploy_cfg["repeat_index"],
            **({"request_timeout_s": float(rpc_timeout)} if rpc_timeout else {}),
        )
    from baselines.client_server.tcp.model_client import ModelClient

    return ModelClient(host=args.host, port=args.port)


def main() -> None:
    args = parse_args()

    # --root_dir is baselines/, which is the package: its parent goes on the path.
    package_parent = str(Path(args.root_dir).resolve().parent)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    # AppLauncher must exist before anything that imports Isaac Sim.
    # --enable_cameras is required: CoHuBTaskEnv reads the cameras.
    from isaaclab.app import AppLauncher

    boot_t0 = time.monotonic()
    provenance: dict | None = None
    device = args.device or (f"cuda:{args.device_id}" if args.device_id is not None else "cpu")
    app_launcher = AppLauncher(
        headless=args.headless,
        visualizer=args.visualizer,
        enable_cameras=True,
        device=device,
    )
    simulation_app = app_launcher.app

    from cohub.policy_env import CoHuBTaskEnv, load_env_cfg_type

    eval_module = importlib.import_module(f"baselines.policy.{args.policy_name}.deploy")
    obs_stride = resolve_obs_stride(args, eval_module)
    run_episode, run_episode_batch, shared_loop = resolve_rollout(args, eval_module, obs_stride)

    # No video directory means no video cameras.
    video_cameras = camera_list(args.video_cameras) if args.video_dir else []
    obs_cameras = camera_list(args.obs_cameras)

    gym_id = load_env_cfg_type(args.env_cfg_type)["gym_id"]
    env_client = CoHuBTaskEnv(
        gym_id=gym_id,
        device=device,
        episode_step_limit=args.episode_step_limit,
        video_dir=args.video_dir,
        video_cameras=video_cameras,
        obs_cameras=obs_cameras,
        obs_stride=obs_stride,
        video_stride=args.video_stride,
        ego_depth=args.ego_depth,
        drop_unused_cameras=args.drop_unused_cameras,
        scene_camera=args.scene_camera,
        follow_distance=args.follow_distance,
        follow_max_dolly=args.follow_max_dolly,
        smooth_video=args.smooth_video,
        video_scale=args.video_scale,
    )
    boot_s = round(time.monotonic() - boot_t0, 2)
    provenance = build_provenance(args, obs_stride=obs_stride, obs_cameras=obs_cameras,
                                  video_cameras=video_cameras)
    print(
        f"[eval_policy_client] env ready in {boot_s}s | obs_cameras={obs_cameras} "
        f"obs_stride={obs_stride} video_cameras={video_cameras} video_stride={args.video_stride} "
        f"video_scale={args.video_scale} scene_camera={args.scene_camera} "
        f"smooth_video={args.smooth_video} "
        f"ego_depth={args.ego_depth} drop_unused_cameras={args.drop_unused_cameras} "
        f"upper_body=joint "
        f"loop={'shared@' + str(obs_stride) if (shared_loop and args.obs_stride is not None) else 'policy'}"
    )

    model_client = TimedModelClient(build_model_client(args))

    results: list[dict] = []
    try:
        for episode_idx in range(args.eval_episode_num):
            print(f"[eval_policy_client] episode {episode_idx + 1}/{args.eval_episode_num}")
            episode_seed = args.seed + episode_idx
            _seed_everything(episode_seed)
            env_client.reset(seed=episode_seed)
            model_client.reset_timing()
            server_seeded = seed_policy_server(model_client, episode_seed)
            episode_t0 = time.monotonic()
            if args.eval_batch:
                run_episode_batch(TASK_ENV=env_client, model_client=model_client)
            else:
                run_episode(TASK_ENV=env_client, model_client=model_client)
            result = env_client.episode_result()
            result.update(episode=episode_idx, seed=episode_seed,
                          wall_s=round(time.monotonic() - episode_t0, 2),
                          host=provenance["host"], server_seeded=server_seeded)
            # Env timing plus RPC timing, per episode.
            result["timing"] = {
                **result.get("timing", {}),
                **model_client.snapshot(),
                "boot_s": boot_s if episode_idx == 0 else 0.0,
            }
            results.append(result)
            done = sum(1 for r in results if r["success"])
            achieved = [name for name, ok in (result.get("subtasks") or {}).items() if ok]
            subtask_note = f" | subtasks: {', '.join(achieved) if achieved else 'none'}"
            print(f"[eval_policy_client] episode {episode_idx + 1}: "
                  f"{result['reason']} after {result['steps']} steps "
                  f"({result['wall_s']}s) | running success {done}/{len(results)}"
                  f"{subtask_note}")
            _write_results(args, results, final=False, provenance=provenance)
            env_client.finish_episode()
    except BaseException:
        # simulation_app.close() exits the process with rc 0: print the traceback
        # and exit non-zero first.
        import os
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        _write_results(args, results, final=True, provenance=provenance)
        os._exit(3)
    finally:
        _write_results(args, results, final=True, provenance=provenance)
        close = getattr(model_client, "close", None)
        if callable(close):
            close()
        env_client.close()
        simulation_app.update()
        simulation_app.close()


if __name__ == "__main__":
    main()
