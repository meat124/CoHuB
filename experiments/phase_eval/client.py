from __future__ import annotations

import importlib
import json
import sys
import time

from pathlib import Path

_WT = Path(__file__).resolve().parents[2]
for _p in (str(_WT / "experiments"), str(_WT / "scripts" / "eval"), str(_WT / "source")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cohub.results import write_json_atomically  # noqa: E402
from phase_eval.report import summarize, table  # noqa: E402
from phase_eval.phase_starts.plan import (  # noqa: E402
    DEFAULT_START,
    START_MODES,
    dataset_root,
    labels_path,
    load_labels,
    plan_trials,
    trials_digest,
)

import eval_policy_client as base  # noqa: E402


def parse_args():
    parser = base.build_parser()
    parser.add_argument("--dataset", default=None,
                        help="the task's dataset root, holding data/ and lerobot/; default datasets/<task_name>")
    parser.add_argument("--phase_starts", default=None,
                        help="labels file; default experiments/phase_eval/phase_starts/states/<task>.json")
    parser.add_argument("--phase", default="all", help="comma-separated chain links to evaluate, or 'all'")
    parser.add_argument("--demos", default="", help='episode indices, e.g. "0-49". Default: every labelled demo')
    parser.add_argument("--start", choices=START_MODES, default=DEFAULT_START,
                        help="auto (the protocol): a trial for every demo, started `lived` -- the demo's actions "
                             "replayed in this env until the previous phase is achieved -- where the labels say the "
                             "replay gets there, and `restored` -- the demo's recorded state written in -- where it "
                             "does not. lived / restored: that one kind only. see plan.DEFAULT_START.")
    parser.add_argument("--states", choices=("auto", "pack", "shards"), default="auto",
                        help="where a demo's recorded states are read from: the tracked states pack beside the "
                             "labels (what `auto` takes when it exists) or the state shards "
                             "under <dataset>/data")
    parser.add_argument("--warm_steps", type=int, default=None,
                        help="recorded steps replayed before the handover; default: the labels' own (0)")
    parser.add_argument("--phase_step_limit", type=int, default=None,
                        help="override every phase's step budget (default: the labels' per-phase budgets)")
    return parser.parse_args()


def write_results(args, labels: dict, trials: list[dict], missing: list[dict], provenance: dict, final: bool) -> None:
    if not args.results_path:
        return
    payload = {
        "bench_name": args.bench_name, "task_name": args.task_name, "env_cfg_type": args.env_cfg_type,
        "policy_name": args.policy_name, "gym_id": labels["gym_id"], "protocol": "phase_starts",
        "chain": labels["chain"], "collaborative": labels.get("collaborative"), "budgets": labels["budgets"],
        "start": args.start,
        "labels": {"file": Path(args.phase_starts).name, "trials_digest": trials_digest(labels, args.start),
                   "source": labels.get("source")},
        "seed": args.seed, "final": final, "provenance": provenance,
        **summarize(trials, labels["chain"], labels.get("collaborative"), missing),
        "unstartable": missing, "trials": trials,
    }
    write_json_atomically(args.results_path, payload)


def main() -> None:
    args = parse_args()
    dataset = dataset_root(args.dataset or args.task_name)
    args.phase_starts = Path(args.phase_starts) if args.phase_starts else labels_path(str(dataset))
    labels = load_labels(args.phase_starts)
    wanted = None if args.phase == "all" else [s for s in args.phase.split(",") if s]

    package_parent = str(Path(args.root_dir).resolve().parent)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    from isaaclab.app import AppLauncher

    boot_t0 = time.monotonic()
    device = args.device or (f"cuda:{args.device_id}" if args.device_id is not None else "cpu")
    simulation_app = AppLauncher(headless=args.headless, visualizer=args.visualizer, enable_cameras=True,
                                 device=device).app

    import torch

    from cohub.g1.grasp import grasp_term
    from cohub.policy_env import load_env_cfg_type
    from phase_eval.env import PhaseEvalEnv

    from cohub.recording.demo_source import parse_selection
    from phase_eval.env import DemoCommands, check_restorable, open_states

    trials, missing = plan_trials(labels, wanted, parse_selection(args.demos), args.warm_steps, args.start)
    trials.sort(key=lambda t: (t.phase_index, t.episode_index))

    eval_module = importlib.import_module(f"baselines.policy.{args.policy_name}.deploy")
    obs_stride = base.resolve_obs_stride(args, eval_module)
    run_episode, _, _ = base.resolve_rollout(args, eval_module, obs_stride)
    video_cameras = base.camera_list(args.video_cameras) if args.video_dir else []
    obs_cameras = base.camera_list(args.obs_cameras)

    gym_id = load_env_cfg_type(args.env_cfg_type)["gym_id"]
    if gym_id != labels["gym_id"]:
        raise SystemExit(f"{args.phase_starts} labels {labels['gym_id']}, but --env_cfg_type names {gym_id}")
    env_client = PhaseEvalEnv(
        gym_id=gym_id, device=device, episode_step_limit=max(labels["budgets"].values()),
        video_dir=args.video_dir, video_cameras=video_cameras, obs_cameras=obs_cameras, obs_stride=obs_stride,
        video_stride=args.video_stride, ego_depth=args.ego_depth,
        drop_unused_cameras=args.drop_unused_cameras, scene_camera=args.scene_camera,
        follow_distance=args.follow_distance, follow_max_dolly=args.follow_max_dolly,
        smooth_video=args.smooth_video, video_scale=args.video_scale,
    )
    env = env_client.env
    boot_s = round(time.monotonic() - boot_t0, 2)
    provenance = base.build_provenance(args, obs_stride=obs_stride, obs_cameras=obs_cameras, video_cameras=video_cameras)
    provenance.update(boot_s=boot_s, warm_steps=args.warm_steps if args.warm_steps is not None
                      else labels.get("warm_steps", 0))
    model_client = base.TimedModelClient(base.build_model_client(args))

    results: list[dict] = []
    if args.results_path and Path(args.results_path).is_file():
        previous = json.loads(Path(args.results_path).read_text())
        results = [t for t in previous.get("trials", []) if not str(t.get("reason", "")).startswith("crash")]
    finished = {f"{t['demo']}:{t['phase']}" for t in results}
    todo = [t for t in trials if t.key not in finished]
    print(f"[phase_eval] env ready in {boot_s}s | {len(todo)} trials to run, {len(finished)} already in "
          f"{args.results_path} | budgets {labels['budgets']} | obs_cameras={obs_cameras} obs_stride={obs_stride}")

    grip_time = float(grasp_term(env).cfg.grip_time)
    try:
        with open_states(dataset, labels, args.states) as states:
            env_args = states.env_args
            provenance["states"] = {"kind": states.kind, "file": Path(states.path).name}
            print(f"[phase_eval] recorded states from the {states.kind}: {states.path}")
            for index, trial in enumerate(todo):
                frames = states.frames(trial.demo, env.device, float(env.step_dt), grip_time, len(env_client.robots))
                commands = DemoCommands(dataset / "lerobot", trial.episode_index, env_client.robots, float(env.step_dt))
                trial_seed = args.seed + trial.episode_index
                base._seed_everything(trial_seed)
                env_client.reset(seed=trial_seed, settle=False,
                                 video_stem=f"{trial.phase_index}_{trial.phase}_{trial.demo}")
                problems = check_restorable(env, frames, env_args)
                if problems:
                    raise SystemExit(f"{trial.demo} cannot be restored into {gym_id}:\n  " + "\n  ".join(problems))
                how = "restored" if trial.start != "lived" else ("auto" if args.start == "auto" else "lived")
                with torch.no_grad():
                    start = env_client.start_from_demo(
                        frames, trial.phase, trial.boundary, trial.restore, commands,
                        args.phase_step_limit or trial.step_budget, start=how,
                    )
                start["start_planned"] = trial.start
                model_client.reset_timing()
                server_seeded = base.seed_policy_server(model_client, trial_seed)
                t0 = time.monotonic()
                run_episode(TASK_ENV=env_client, model_client=model_client)
                result = env_client.episode_result()
                result.update(start)
                result.update(
                    demo=trial.demo, episode_index=trial.episode_index, phase=trial.phase,
                    phase_index=trial.phase_index, seed=trial_seed, server_seeded=server_seeded,
                    wall_s=round(time.monotonic() - t0, 2), host=provenance["host"],
                )
                result["timing"] = {**result.get("timing", {}), **model_client.snapshot()}
                results.append(result)
                mine = [r for r in results if r["phase"] == trial.phase]
                print(f"[phase_eval] {index + 1}/{len(todo)} {trial.demo} {trial.phase}: "
                      f"{'SUCCESS' if result['target_success'] else 'fail'} ({result['reason']}) after "
                      f"{result['steps']}/{result['step_budget']} steps, {result['wall_s']}s"
                      f" | start {start['start_mode']}"
                      f"{'' if start['start_valid'] else ' INVALID: ' + str(start['start_reason'])} | "
                      f"{trial.phase} so far {sum(1 for r in mine if r['target_success'])}/{len(mine)}", flush=True)
                write_results(args, labels, results, missing, provenance, final=False)
                env_client.finish_episode()
    except BaseException:
        import os
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        write_results(args, labels, results, missing, provenance, final=False)
        os._exit(3)
    write_results(args, labels, results, missing, provenance, final=True)
    print("\n" + table(summarize(results, labels["chain"], labels.get("collaborative"), missing)))
    close = getattr(model_client, "close", None)
    if callable(close):
        close()
    env_client.close()
    simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
