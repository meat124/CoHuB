"""Log an evaluation's results.json into the wandb run that trained the policy.

A decentralized policy is one evaluation of several checkpoints, so the same
rollout numbers go into each robot's run.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from pathlib import Path


def wandb_runs_for(mode: str, task: str, seed: int, variant: str = "obs1",
                   robots: tuple[str, ...] = ("robot_a", "robot_b")) -> list[str]:
    """Return the DP training run name(s) this evaluation belongs to.

    `variant` is what policy/DP/train.sh puts between the task and the seed
    (e.g. `obs1-b128-e1000`). It must match the training run name exactly,
    because wandb resumes by id.
    """
    if mode == "decentralized":
        return [f"dp-decentralized-{task}-{robot}-{variant}-seed{seed}" for robot in robots]
    return [f"dp-centralized-{task}-{variant}-seed{seed}"]


def run_ids_for_name(entity: str, project: str, name: str) -> list[str]:
    """Return the ids to try for a run displayed under `name`, best first.

    The id normally equals the name, but a run can carry the name under a
    generated id.
    """
    import wandb

    try:
        matches = [r for r in wandb.Api().runs(f"{entity}/{project}") if r.name == name]
    except Exception as error:  # noqa: BLE001 - offline, unauthenticated, project gone
        print(f"[report] could not list {entity}/{project} ({error}); assuming id == name")
        return [name]
    # Newest first: a name can belong to more than one run.
    ids = [r.id for r in sorted(matches, key=lambda r: r.created_at, reverse=True)]
    if name not in ids:
        ids.append(name)  # nothing carries it yet; create it
    return ids


def rollout_metrics(summary: dict, mode: str, task: str | None = None) -> dict:
    """Return the `rollout/*` series an evaluation charts: the success rate and
    the subtask milestones."""
    metrics = {"rollout/success_rate": summary["success_rate"]}
    rates = dict(summary["subtask_success_rate"] or {})
    # Detail milestones chart beside the chain.
    rates.update(summary.get("subtask_detail_success_rate") or {})
    for name, rate in rates.items():
        metrics[f"rollout/subtask/{name}"] = rate
    return metrics


def rollout_facts(summary: dict, mode: str, ckpt_step: int | None = None) -> dict:
    """Return the summary-only facts: sample size, checkpoint step and mode."""
    facts = {
        "rollout/successes": summary["successes"],
        "rollout/episodes": summary["episodes_run"],
        "rollout/steps_median": summary["steps_median"],
        "rollout/mode": mode,
    }
    if ckpt_step is not None:
        facts["rollout/ckpt_step"] = ckpt_step
    return facts


def scoped(metrics: dict, task: str) -> dict:
    """Rename `rollout/<key>` to `rollout/<task>/<key>`."""
    return {key.replace("rollout/", f"rollout/{task}/", 1): value
            for key, value in metrics.items()}


def eval_keys(summary: dict, mode: str, task: str, ckpt_step: int | None = None
              ) -> tuple[dict, dict]:
    """Return (charted, facts) for this task in the eval run's key space.

    wandb sections a run's panels by the first segment of the key:

        overall/<task>          success rate per task
        <task>/<milestone>      that task's subtask chain
    """
    metrics = rollout_metrics(summary, mode, task)
    charted = {f"overall/{task}": metrics["rollout/success_rate"]}
    charted.update({f"{task}/{key.split('/', 2)[2]}": value
                    for key, value in metrics.items()
                    if key.startswith("rollout/subtask/")})
    # Summary only.
    facts = {f"{task}/{key.split('/', 1)[1]}": value
             for key, value in rollout_facts(summary, mode, ckpt_step).items()}
    return charted, facts


# Name suffix of the per-policy eval run that holds the checkpoint sweep.
EVAL_RUN_SUFFIX = "-eval"


def scorable(summary: dict, label: str, min_episodes: int) -> bool:
    """Whether this results.json is a point on a curve: enough episodes and no
    client crash. Same conditions as `upload.main`."""
    n = summary["episodes_run"]
    if min_episodes and n < min_episodes:
        print(f"[report] {label}: {n} episodes, fewer than {min_episodes} -- "
              f"not a point on the curve")
        return False
    crashed = sum(1 for e in summary["episodes"]
                  if str(e.get("reason", "")).startswith("client_crash"))
    if crashed:
        print(f"[report] {label}: {crashed} of {n} episodes are client crashes -- "
              f"rerun them before charting")
        return False
    return True


def sweep_results(eval_results: Path, policy: str, mode: str, seed: int,
                  ckpt_tag: str | None = None, min_episodes: int = 50,
                  run_dir_name: str | None = None,
                  tasks: set[str] | None = None) -> list[tuple[str, int, dict]]:
    """Return every scored checkpoint of this sweep as (task, step, summary), in
    checkpoint order.

    Directory names are `scripts/eval/runner.sh`'s RUN_ID, matched whole, so a run with
    another suffix (e.g. `-exec<N>`) is not part of the sweep.

    Args:
        run_dir_name: directory of one evaluation of the sweep. Its siblings
            differ only in `-ckpt<N>`. Without it the name is built from
            policy, mode, ckpt_tag and seed.
        tasks: restricts the sweep to one task set.
    """
    if run_dir_name is not None:
        stem = re.escape(re.sub(r"-ckpt\d+$", "", run_dir_name))
    else:
        stem = re.escape(f"{policy}-{mode}{f'-{ckpt_tag}' if ckpt_tag else ''}-seed{seed}")
    pattern = re.compile(rf"^{stem}-ckpt(\d+)$")
    found = []
    for task_dir in sorted(d for d in eval_results.iterdir()
                           if d.is_dir() and (tasks is None or d.name in tasks)):
        for run_dir in sorted(task_dir.iterdir()):
            match = pattern.match(run_dir.name)
            results = run_dir / "results.json"
            if not (match and results.is_file()):
                continue
            step = int(match.group(1))
            summary = json.loads(results.read_text())
            if scorable(summary, f"{task_dir.name} @{step}", min_episodes):
                found.append((task_dir.name, step, summary))
    return sorted(found, key=lambda row: (row[1], row[0]))


def rebuild_eval_run(eval_results: Path, policy: str, mode: str, seed: int,
                     project: str, run_name: str, ckpt_tag: str | None = None,
                     min_episodes: int = 50, run_dir_name: str | None = None,
                     tasks: set[str] | None = None) -> None:
    """Replay the whole sweep into the policy's eval run, in checkpoint order.

    The run is rebuilt, not appended to: `step=` is the checkpoint step, and
    wandb drops a point logged out of order. The run keeps its name across
    rebuilds and gets a new id each time.
    """
    import wandb

    entity = os.environ.get("WANDB_ENTITY") or wandb.Api().default_entity
    points = sweep_results(eval_results, policy, mode, seed, ckpt_tag, min_episodes,
                           run_dir_name=run_dir_name, tasks=tasks)
    if not points:
        print(f"[report] no scored checkpoints under {eval_results} for {policy}; "
              f"nothing to rebuild")
        return

    # One writer at a time: concurrent rebuilds would delete each other's run.
    eval_results.mkdir(parents=True, exist_ok=True)
    lock = (eval_results / f".{run_name}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        stale = [r.id for r in _runs_named(entity, project, run_name)]

        run = wandb.init(entity=entity, project=project, name=run_name, reinit=True)
        by_step: dict[int, dict] = {}
        facts: dict[str, object] = {}
        steps_seen: dict[str, set[int]] = {}
        for task, step, summary in points:
            charted, task_facts = eval_keys(summary, mode, task, step)
            by_step.setdefault(step, {}).update(charted)
            facts.update(task_facts)
            steps_seen.setdefault(task, set()).add(step)
        for step in sorted(by_step):
            run.log(by_step[step], step=step)
        run.summary.update(facts)
        for task, steps in steps_seen.items():
            run.summary[f"{task}/ckpt_steps"] = sorted(steps)
        run_id, run_url = run.id, run.url
        run.finish()

        for old_id in stale:
            try:
                wandb.Api().run(f"{entity}/{project}/{old_id}").delete()
            except Exception as error:  # noqa: BLE001 - the new run is already written
                print(f"[report] could not delete the previous {run_name} ({old_id}): {error}")
        print(f"[report] rebuilt {run_name} from {len(points)} evaluation(s) "
              f"over {len(by_step)} checkpoint(s): {run_url}")
        if stale:
            print(f"[report]   replaced {len(stale)} earlier run(s) of that name "
                  f"-- the id changes, the name does not")
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _runs_named(entity: str, project: str, name: str) -> list:
    """The runs currently carrying this display name."""
    import wandb

    try:
        return [r for r in wandb.Api().runs(f"{entity}/{project}") if r.name == name]
    except Exception as error:  # noqa: BLE001 - offline, unauthenticated, project gone
        print(f"[report] could not list {entity}/{project} ({error}); "
              f"not replacing any earlier run")
        return []


def log_to_wandb(summary: dict, video_root: Path, mode: str, task: str, seed: int,
                 variant: str = "obs1",
                 runs: list[str] | None = None, project: str | None = None,
                 ckpt_step: int | None = None, scope_by_task: bool = False,
                 label: str | None = None) -> None:
    """Write this evaluation into the training run(s) that produced the policy.

    Args:
        runs, project: run ids and project, for policies whose runs this module
            does not name (DP's are built from the task and the variant).
        scope_by_task: put every `rollout/*` key under `rollout/<task>/`, for a
            shared multitask run that holds every task's evaluation.
        ckpt_step: training step of the evaluated checkpoint, stored in the
            summary. It is not wandb's `step=`, which must be monotonic.
        label: re-evaluation label; keys go under `<task>-<label>`.
    """
    import wandb

    entity = os.environ.get("WANDB_ENTITY") or wandb.Api().default_entity
    project = project or os.environ.get("WANDB_PROJECT", "CoHuB")
    video_cap = int(os.environ.get("EVAL_WANDB_VIDEO_MAX", "50"))

    # One mp4 per episode per camera: videos/<camera>/seed<NNN>_<VERDICT>_<N>steps.mp4.
    # Scene-camera clips sort first.
    videos: dict[int, list[Path]] = {}
    for path in sorted(video_root.rglob("*.mp4"), key=lambda p: (p.parent.name != "scene", str(p))):
        match = re.search(r"seed(\d+)", path.stem)
        index = int(match.group(1)) if match else None
        if index is None:
            digits = "".join(c for c in path.stem if c.isdigit())
            index = int(digits) if digits else None
        if index is not None:
            videos.setdefault(index, []).append(path)

    metrics = rollout_metrics(summary, mode, task)
    facts = rollout_facts(summary, mode, ckpt_step)
    # `label` keeps a re-evaluation's keys apart from the originals.
    scope = f"{task}-{label}" if label else task
    if scope_by_task:
        metrics, facts = scoped(metrics, scope), scoped(facts, scope)

    # `runs` holds ids. An id passed as `name=` would rename the run to its id.
    if runs is not None:
        targets = [(run_id, None) for run_id in runs]      # id known, name is the run's own
    else:
        # DP: the id equals the name wandb_runs_for builds.
        targets = [(name, None) for name in wandb_runs_for(mode, task, seed, variant)]

    for known_id, display_name in targets:
        run = None
        label = known_id or display_name
        candidates = [known_id] if known_id else run_ids_for_name(entity, project, display_name)
        for run_id in candidates:
            named = {"name": display_name} if display_name else {}
            try:
                run = wandb.init(entity=entity, project=project, id=run_id, **named,
                                 resume="allow", reinit=True)
                break
            except Exception as error:  # noqa: BLE001 - the results file is already on disk
                print(f"[report] could not resume wandb run {label} (id {run_id}): {error}")
        if run is None:
            continue
        # The run's display name, for the messages below.
        run_name = run.name

        # A rollout the run already carries is replaced, as the results on disk and the
        # eval run are: the table and the summary show the newest. A run trained again
        # under the same name keeps its id, and its earlier rollout would otherwise stay.
        steps_key = f"rollout/{scope}/ckpt_steps" if scope_by_task else "rollout/ckpt_steps"
        seen = run.summary.get(steps_key) or []
        if ckpt_step is not None and int(ckpt_step) in {int(x) for x in seen}:
            print(f"[report] {run_name}: replacing its {task} rollout of step {ckpt_step}")

        table = wandb.Table(columns=["episode", "success", "reason", "steps", "subtasks", "video"])
        uploaded = 0
        for entry in summary["episodes"]:
            index = entry["episode"]
            achieved = ", ".join(k for k, v in (entry.get("subtasks") or {}).items() if v) or "none"
            clip = None
            # The file is named for the seed the episode ran at.
            video_key = entry.get("seed")
            if video_key is None:
                video_key = index
            # Pick the clip whose verdict and step count match this episode record.
            verdict = "SUCCESS" if entry.get("success") else f"FAIL-{entry.get('reason')}"
            expected_stem = f"seed{int(video_key):03d}_{verdict}_{entry.get('steps')}steps"
            candidates = videos.get(video_key, [])
            video_path = next((path for path in candidates if path.stem == expected_stem), None)
            if uploaded < video_cap and video_path is not None:
                clip = wandb.Video(str(video_path), format="mp4")
                uploaded += 1
            table.add_data(index, bool(entry.get("success")), entry.get("reason"),
                           entry.get("steps"), achieved, clip)
        table_key = f"rollout/{scope}/episodes_table" if scope_by_task else "rollout/episodes_table"
        if ckpt_step is None:
            # Single rollout: chart the metrics.
            run.log({**metrics, table_key: table})
        else:
            # Sweep: log only the episode table. rebuild_eval_run draws the curve.
            run.log({table_key: table})
        run.summary.update({**metrics, **facts})
        if ckpt_step is not None:
            # Read by the duplicate guard above.
            run.summary[steps_key] = sorted({int(x) for x in seen} | {int(ckpt_step)})
        run.finish()
        at = f" at step {ckpt_step}" if ckpt_step is not None else ""
        print(f"[report] logged rollout{at} into wandb run {run_name} "
              f"({entity}/{project}/{run.id}, {uploaded} videos)")


