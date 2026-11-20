from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from pathlib import Path

_WT = Path(__file__).resolve().parents[3]
for _p in (str(_WT / "experiments"), str(_WT / "scripts" / "eval"), str(_WT / "source")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cohub.runtime import exit_gate, fail_gate_on_exception  # noqa: E402

fail_gate_on_exception()

NUM_SUCCESS_STEPS = 10

UNREPRODUCED_LIMIT = 0.10


def label(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="build.py label")
    ap.add_argument("dataset", help="a task (cocarry) or its dataset root (datasets/cocarry: holds data/ and lerobot/)")
    ap.add_argument("--env_cfg_type", default=None, help="baselines/env_cfg/<this>.yml; default: the dataset dir's name")
    ap.add_argument("--demos", default="", help='e.g. "0-49". Default: the LeRobot export\'s train split')
    ap.add_argument("--out", default=None, help="default: experiments/phase_eval/phase_starts/states/<task>.json")
    ap.add_argument("--num_success_steps", type=int, default=NUM_SUCCESS_STEPS,
                    help="closing success frames every demo must show (record_demos.py's own flag)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--headless", action="store_true", default=True)
    args = ap.parse_args(argv)

    from phase_eval.phase_starts.plan import dataset_root, labels_path

    dataset = dataset_root(args.dataset)
    lerobot = dataset / "lerobot"
    provenance = json.loads(next(p for p in (lerobot / "meta" / n for n in ("cohub_provenance.json", "mhbench_provenance.json")) if p.is_file()).read_text())
    info = json.loads((lerobot / "meta" / "info.json").read_text())
    episode_of = {row["demo"]: int(row["episode_index"]) for row in provenance["episodes"]}
    exported_length = {row["demo"]: int(row["num_samples"]) for row in provenance["episodes"]}

    from cohub.recording.demo_source import DemoSource, parse_selection

    selection = parse_selection(args.demos)
    split = "custom"
    if selection is None:
        lo, hi = (int(v) for v in info["splits"]["train"].split(":"))
        selection, split = set(range(lo, hi)), "train"

    from isaaclab.app import AppLauncher

    AppLauncher(headless=args.headless, enable_cameras=True, device=args.device)

    import numpy as np
    import torch

    from cohub.g1.grasp import grasp_term
    from cohub.policy_env import load_env_cfg_type
    from phase_eval.env import PhaseEvalEnv

    from phase_eval.env import DemoCommands, DemoFrames, check_restorable, episode_column, sweep_terms
    from phase_eval.phase_starts.plan import (
        COLLABORATIVE,
        LIVED_SLACK,
        SCHEMA,
        lived_available,
        offline_cocarry_labels,
        phase_budgets,
        phase_lengths,
    )

    gym_id = load_env_cfg_type(args.env_cfg_type or dataset.name)["gym_id"]
    recorded_id = provenance.get("task", {}).get("env_name")
    if recorded_id and recorded_id != gym_id:
        raise SystemExit(f"{dataset} was recorded as {recorded_id}, but the env cfg names {gym_id}")

    wrapper = PhaseEvalEnv(gym_id=gym_id, device=args.device, obs_cameras=[], video_cameras=[],
                             drop_unused_cameras=True)
    env, tracker = wrapper.env, wrapper._tracker
    if tracker is None or not tracker.names:
        raise SystemExit(f"{gym_id} has no phase chain in cohub.subtasks")
    chain = list(tracker.names)
    wrapper.reset(seed=0)
    env_ids = torch.tensor([0], device=env.device)
    grasp = grasp_term(env)
    success_params = dict(env.termination_manager.get_term_cfg("success").params)

    rows, failures, lengths, unreproduced = [], [], [], []
    steps_stepped = 0
    t0 = time.perf_counter()
    frames_seen = 0
    with DemoSource(dataset / "data") as demos, torch.no_grad():
        env_args = demos.attrs.get("env_args")
        for name in demos.names:
            episode = episode_of.get(name)
            if episode is None or episode not in selection:
                continue
            group = demos[name]
            frames = DemoFrames(group, env.device, float(env.step_dt), float(grasp.cfg.grip_time), len(wrapper.robots))
            problems = check_restorable(env, frames, env_args)
            if frames.steps != exported_length[name]:
                problems.append(f"{frames.steps} steps in the shard, {exported_length[name]} in the LeRobot export")
            else:
                obs = group["obs"]
                recorded = {
                    "observation.object_state": np.concatenate([obs["object_pos"], obs["object_rot"]], axis=1),
                    "observation.grasp_latched": np.asarray(obs["grasp_latched"]),
                    "observation.robots_state": np.concatenate(
                        [obs[f"{robot}_root_{part}"] for robot in wrapper.robots for part in ("pos", "rot")], axis=1),
                }
                for column, shard_values in recorded.items():
                    gap = float(np.abs(episode_column(lerobot, episode, column) - shard_values).max())
                    if gap > 0.0:
                        problems.append(f"{column} differs from the LeRobot export by {gap:.2e}: not the same recording")
            if problems:
                raise SystemExit(f"{name} cannot be restored into {gym_id}:\n  " + "\n  ".join(problems))

            env.termination_manager.reset(env_ids)
            tracker.reset()
            fired: dict[str, int] = {}
            succeeded: list[bool] = []

            def on_frame(k: int) -> None:
                tracker.update(env, step=k, terminal=False)
                for term in env.termination_manager.active_terms:
                    if term not in fired and bool(env.termination_manager.get_term(term)[0]):
                        fired[term] = k
                succeeded.append(bool(env.termination_manager.get_term("success")[0]))

            sweep_terms(env, frames, frames.steps - 1, on_frame)
            frames_seen += frames.steps

            achieved = tracker.achieved()
            details = tracker.detail_result()[1]

            lived: dict[str, int | None] = {link: None for link in chain[:-1]}
            reachable = [frame for frame in achieved.values() if frame is not None]
            if reachable:
                commands = DemoCommands(lerobot, episode, wrapper.robots, float(env.step_dt))
                wrapper.start_from_demo(frames, chain[0], -1, -1, commands, 10**6, start="restored")
                for row_index in range(min(frames.steps, max(reachable) + LIVED_SLACK + 1)):
                    if wrapper._warm_step(commands.actions[row_index]):
                        break
                    tracker.update(env, step=row_index, terminal=False)
                    for link, step in tracker.achieved().items():
                        if step is not None and lived[link] is None:
                            lived[link] = row_index
                    if all(step is not None for step in lived.values()):
                        break
                steps_stepped += row_index + 1
            success_frame = fired.get("success")
            closing = next((i for i, ok in enumerate(reversed(succeeded)) if not ok), len(succeeded))
            row = {"episode_index": episode, "demo": name, "T": frames.steps, "achieved": achieved, "lived": lived,
                   "details": details, "env_success_frame": success_frame, "closing_success_frames": closing,
                   "other_terms_fired": {k: v for k, v in fired.items() if k != "success"}}
            rows.append(row)
            lengths.append(phase_lengths(chain, achieved, success_frame))

            if closing < args.num_success_steps:
                unreproduced.append(f"{name}: the swept success term holds on the last {closing} of "
                                    f"{frames.steps} frames (first true at {success_frame})")
                if success_frame is None:
                    failures.append(f"{name}: recorded as a success, but the swept success term never fires")
            early = {k: v for k, v in row["other_terms_fired"].items() if success_frame is None or v < success_frame}
            if early:
                failures.append(f"{name}: a failure term fired during a successful demo: {early}")
            if any(frame is None for frame in achieved.values()):
                failures.append(f"{name}: chain not walked: {achieved}")
            if gym_id == "Isaac-CoCarry-G1x2-v0":
                offline = offline_cocarry_labels(np.asarray(group["states/grasp/latched"]),
                                                 np.asarray(group["states/rigid_object/object/root_pose"]),
                                                 success_params)
                if offline != achieved:
                    failures.append(f"{name}: swept {achieved} but the arrays alone say {offline}")
            print(f"[label] {name:>8} T={frames.steps:4d} {achieved} success@{success_frame} "
                  f"closing={closing} lived={lived}", flush=True)

    elapsed = time.perf_counter() - t0
    if not rows:
        raise SystemExit("no demos selected")
    if len(unreproduced) > UNREPRODUCED_LIMIT * len(rows):
        failures.append(f"the closing success run is not reproduced on {len(unreproduced)} of {len(rows)} demos "
                        f"(limit {UNREPRODUCED_LIMIT:.0%}): " + "; ".join(unreproduced[:5]))
    budgets = phase_budgets(chain, lengths)
    lived_trials = {chain[0]: len(rows)}
    for index, link in enumerate(chain[:-1]):
        lived_trials[chain[index + 1]] = sum(
            1 for r in rows if r["achieved"][link] is not None and lived_available(r, link, r["achieved"][link]))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_WT, capture_output=True, text=True).stdout.strip()
    revision = None
    marker = lerobot.resolve().parent / ".cache" / "huggingface" / "download" / lerobot.resolve().name / "meta" / "info.json.metadata"
    if marker.is_file():
        revision = (marker.read_text().splitlines() or [None])[0]
    labels = {
        "schema": SCHEMA, "task": dataset.name, "gym_id": gym_id, "chain": chain,
        "details": list(tracker.details), "collaborative": dict(COLLABORATIVE.get(dataset.name, {})),
        "checks_key": tracker.checks_key, "step_dt": float(env.step_dt), "budgets": budgets,
        "phase_length_max": {n: max(r[n] for r in lengths if r[n] is not None) for n in chain},
        "lived_slack": LIVED_SLACK, "lived_trials": lived_trials,
        "source": {"split": split, "demos": len(rows), "frames": frames_seen, "mhbench_sha": sha,
                   "shards_match_export": "exact", "unreproduced_success": unreproduced,
                   "export_revision": revision},
        "demos": rows,
    }
    out = Path(args.out) if args.out else labels_path(args.dataset)
    from cohub.results import write_json_atomically

    write_json_atomically(out, labels)
    print(f"\n[label] {len(rows)} demos, {frames_seen} frames in {elapsed:.1f}s "
          f"({1e3 * elapsed / max(1, frames_seen):.2f} ms/frame)")
    for index, link in enumerate(chain[:-1]):
        at = [r["achieved"][link] for r in rows if r["achieved"][link] is not None]
        print(f"[label] {link:<12} frames {min(at)}..{max(at)}  ({len(at)}/{len(rows)} demos); "
              f"a replay reaches it in {lived_trials[chain[index + 1]]}")
    print(f"[label] trials started lived, of {len(rows)} a phase (the rest are restored): {lived_trials}  "
          f"({steps_stepped} steps replayed)")
    print(f"[label] phase length max {labels['phase_length_max']} -> budgets {budgets}")
    for note in unreproduced:
        print(f"[label] note: {note}")
    print(f"[label] wrote {out}")
    print(f"[label] now re-cut the states pack an evaluation reads: python experiments/phase_eval/phase_starts/build.py pack {dataset.name}")
    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
    exit_gate(failures)


def pack(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="build.py pack")
    ap.add_argument("dataset", help="a task (cocarry) or its dataset root (holds data/ and lerobot/)")
    ap.add_argument("--labels", default=None, help="default: experiments/phase_eval/phase_starts/states/<task>.json")
    ap.add_argument("--out", default=None, help="default: experiments/phase_eval/phase_starts/states/<task>.states.npz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    from phase_eval.phase_starts.plan import PACK_SCHEMA, StartPack, pack_path, state_layout, state_rows
    from phase_eval.phase_starts.plan import dataset_root, labels_path, load_labels, trials_digest

    dataset = dataset_root(args.dataset)
    labels_file = Path(args.labels) if args.labels else labels_path(args.dataset)
    labels = load_labels(labels_file)
    chain = list(labels["chain"])
    out = Path(args.out) if args.out else pack_path(args.dataset)

    from isaaclab.app import AppLauncher

    AppLauncher(headless=True, enable_cameras=True, device=args.device)

    import numpy as np
    import torch

    from cohub.g1.grasp import grasp_term
    from phase_eval.env import PhaseEvalEnv

    from cohub.recording.demo_source import DemoSource
    from phase_eval.env import DemoFrames, check_restorable, memory_differs, sweep_terms, term_memory

    wrapper = PhaseEvalEnv(gym_id=labels["gym_id"], device=args.device, obs_cameras=[], video_cameras=[],
                             drop_unused_cameras=True)
    env, tracker = wrapper.env, wrapper._tracker
    wrapper.reset(seed=0)
    env_ids = torch.tensor([0], device=env.device)
    grasp = grasp_term(env)
    shape = (env.device, float(env.step_dt), float(grasp.cfg.grip_time), len(wrapper.robots))

    def term_values() -> dict:
        manager = env.termination_manager
        return {name: bool(manager.get_term(name)[0]) for name in manager.active_terms}

    arrays: dict[str, np.ndarray] = {}
    demos_meta: dict[str, dict] = {}
    whole: dict[str, dict] = {}
    shard_rows: dict[str, dict] = {}
    layout = None
    t0 = time.perf_counter()
    swept = 0
    with DemoSource(dataset / "data") as demos, torch.no_grad():
        env_args = demos.attrs.get("env_args")
        env_args = env_args.decode() if isinstance(env_args, bytes) else env_args
        for row in labels["demos"]:
            name = row["demo"]
            group = demos[name]
            frames = DemoFrames(group, *shape)
            problems = check_restorable(env, frames, env_args)
            if problems:
                raise SystemExit(f"{name} cannot be restored into {labels['gym_id']}:\n  " + "\n  ".join(problems))
            mine = state_layout(group["states"])
            if layout is None:
                layout = mine
            elif mine != layout:
                raise SystemExit(f"{name}: its states are laid out differently from the first demo's")
            if state_layout(group["initial_state"]) != layout:
                raise SystemExit(f"{name}: initial_state and states do not hold the same fields")

            starts = sorted({row["achieved"][link] for link in chain[:-1] if row["achieved"].get(link) is not None})
            kept: set[int] = set(starts)
            whole[name] = {}
            if starts:
                env.termination_manager.reset(env_ids)
                tracker.reset()
                before = [term_memory(env, tracker)]

                def on_frame(k: int) -> None:
                    tracker.update(env, step=0, terminal=False)
                    now = term_memory(env, tracker)
                    if memory_differs(before[0], now):
                        kept.add(k)
                    before[0] = now
                    if k in starts:
                        whole[name][k] = (now, term_values())

                sweep_terms(env, frames, starts[-1], on_frame)
                swept += starts[-1] + 1

            index = np.asarray(sorted(kept), dtype=np.int64)
            packed = np.concatenate([state_rows(group["initial_state"], layout, None)[:1],
                                     state_rows(group["states"], layout, index)], axis=0)
            recorded = group["states"]["grasp"]
            arrays[f"{name}/frames"] = index.astype(np.int32)
            arrays[f"{name}/states"] = packed.astype(np.float32)
            arrays[f"{name}/grasp"] = np.concatenate(
                [np.asarray(recorded["latched"], np.float32), np.asarray(recorded["ever_latched"], np.float32),
                 np.asarray(recorded["ever_both_robots_latched"], np.float32).reshape(-1, 1),
                 np.asarray(recorded["release_age"], np.float32).reshape(-1, 1)], axis=1)
            arrays[f"{name}/object_pose"] = np.asarray(
                group["states"]["rigid_object"][grasp.cfg.asset_name]["root_pose"], np.float32)
            demos_meta[name] = {"episode_index": row["episode_index"], "T": frames.steps, "starts": starts,
                                "kept": len(index)}
            shard_rows[name] = {frame: {key: value.clone() for key, value in _flat(frames.state(frame)).items()}
                                for frame in [-1, *starts]}
            shard_rows[name]["grasp"] = {frame: frames.grasp_seed(frame) for frame in starts}
            print(f"[pack] {name:>8} T={frames.steps:4d} starts={starts} kept {len(index)} frames", flush=True)

    meta = {"schema": PACK_SCHEMA, "task": dataset.name, "gym_id": labels["gym_id"], "env_args": env_args,
            "grasp_asset": grasp.cfg.asset_name, "layout": layout, "demos": demos_meta,
            "labels": {"file": labels_file.name, "trials_digest": trials_digest(labels)},
            "source": labels.get("source")}
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = out.with_suffix(".building.npz")
    np.savez_compressed(scratch, meta=np.array(json.dumps(meta)), **arrays)

    failures: list[str] = []
    checked = 0
    with StartPack(scratch) as pack, torch.no_grad():
        failures += pack.check_labels(labels)
        for name, per_start in whole.items():
            frames = pack.frames(name, *shape)
            for frame in [-1, *per_start]:
                packed_state = _flat(frames.state(frame))
                for key, value in shard_rows[name][frame].items():
                    if not torch.equal(packed_state[key], value):
                        failures.append(f"{name}: packed state {key} at frame {frame} is not the shard's")
            for frame, (memory, values) in per_start.items():
                seed, recorded_seed = frames.grasp_seed(frame), shard_rows[name]["grasp"][frame]
                if any(not np.array_equal(seed[key], recorded_seed[key]) for key in recorded_seed):
                    failures.append(f"{name}: packed grasp seed at frame {frame} is not the shard's")
                env.termination_manager.reset(env_ids)
                tracker.reset()
                sweep_terms(env, frames, frame, lambda k: tracker.update(env, step=0, terminal=False))
                differs = memory_differs(memory, term_memory(env, tracker))
                if differs:
                    failures.append(f"{name}: swept from the pack to frame {frame}, the terms remember {differs} differently")
                if term_values() != values:
                    failures.append(f"{name}: swept from the pack to frame {frame}, a termination term reads differently")
                checked += 1

    total = sum(d["T"] for d in demos_meta.values())
    kept_total = sum(d["kept"] for d in demos_meta.values())
    if failures:
        scratch.unlink(missing_ok=True)
        print("\nFAILURES:")
        for failure in failures[:40]:
            print(f"  - {failure}")
    else:
        scratch.replace(out)
        print(f"\n[pack] {dataset.name}: {len(demos_meta)} demos, {kept_total} of {total} frames kept "
              f"({swept} swept), {checked} starts proven equal to the whole sweep, "
              f"{out.stat().st_size / 1e6:.2f} MB -> {out}  ({time.perf_counter() - t0:.0f}s)")
    exit_gate(failures)


def _flat(state: dict) -> dict:
    return {f"{family}/{name}/{field}": value for family, names in state.items()
            for name, fields in names.items() for field, value in fields.items()}


if __name__ == "__main__":
    steps = {"label": label, "pack": pack}
    if len(sys.argv) < 2 or sys.argv[1] not in steps:
        sys.exit("usage: build.py <label|pack> <task>")
    steps[sys.argv[1]](sys.argv[2:])
