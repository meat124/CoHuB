#!/usr/bin/env python3
"""Log a finished evaluation's results.json into the wandb run that trained the policy.

    python3 baselines/logging/upload.py eval_results/<task>/<run-id>

README.md says what is logged where.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import (  # noqa: E402
    EVAL_RUN_SUFFIX,
    log_to_wandb,
    rebuild_eval_run,
    wandb_runs_for,
)

# Task spelling overrides. Empty: slug and scene are the same for every task.
TASK_ALIASES: dict[str, str] = {}

# Task -> the word in checkpoint directory names.
CKPT_TASK_ALIASES: dict[str, str] = {}

# wandb project(s) per policy. The first entry is where a new run is created and
# must equal the WANDB_PROJECT of policy/<POLICY>/train_hook.sh. Further entries are
# fallbacks for lookups.
#
# How a policy's runs are addressed:
#   DP                  by name (policy/DP/train.sh sets name == id), one per robot
#   Pi_05               by id from wandb_id.txt beside the checkpoint
#   GR00T_N17/FastWAM/  by id == the checkpoint run-dir name
#   Psi0
#   ACT                 by id == the run-dir name, one per robot
#   GauDP/LatentToM     by id == the run-dir name, one per task
POLICY_PROJECTS = {
    "DP": ["CoHuB_DP"],
    "Pi_05": ["CoHuB_Pi05"],
    "GR00T_N17": ["CoHuB_GR00T"],
    "ACT": ["CoHuB_ACT"],
    "FastWAM": ["CoHuB_FastWAM"],
    "Psi0": ["CoHuB_Psi0"],
    "GauDP": ["CoHuB_GauDP"],
    "LatentToM": ["CoHuB_LatentToM"],
}

# What a decentralized run of each policy is: `shared` is one multitask
# checkpoint driving every agent, `per_robot` one checkpoint per robot. Must
# match the policy's train/ and eval/ hooks.
DECENTRALIZED_STYLE = {
    "DP": "per_robot",
    "ACT": "per_robot",
    "Pi_05": "shared",
    "GR00T_N17": "shared",
    "FastWAM": "shared",
    "Psi0": "shared",
    # GauDP is centralized only.
    "GauDP": "shared",
    # LatentToM: two arms in one run directory.
    "LatentToM": "shared",
}

PER_TASK = {"GauDP", "LatentToM"}
"""One training run per task, every robot in it."""

MULTITASK_CKPT = "multitask3" if os.environ.get("COHUB_TASK_SET") == "all3" else "multitask"
"""The shared policy's checkpoint word (common.sh: cohub_multitask_word)."""


def eval_run_name(ckpt_tag: str | None, label: str | None = None) -> str:
    """Return the shared policy's eval run name: `multitask[-<tag>]-eval`.

    The tag is dropped when it is one of the benchmark's own (`cohub`, `cohub2`, ...).
    """
    tag = "" if not ckpt_tag or re.fullmatch(r"cohub\d*", ckpt_tag) else f"-{ckpt_tag}"
    return f"{MULTITASK_CKPT}{tag}{f'-{label}' if label else ''}-eval"


def trained_label(entity: str, project: str, run_id: str) -> str | None:
    """Return `<budget>[-<tag>]` as the training run's display name carries it.

    The name is common.sh's cohub_wandb_name, `<ckpt>-<budget>[-<tag>]-train`. None when the
    run cannot be read or is named otherwise.
    """
    import wandb

    try:
        name = wandb.Api().run(f"{entity}/{project}/{run_id}").name
    except Exception as error:  # noqa: BLE001 - offline, unauthenticated, no such run
        print(f"[wandb] could not read the training run {run_id} ({error}); "
              f"naming the eval run from this shell's settings", file=sys.stderr)
        return None
    match = re.fullmatch(r".+?-(b\d+-.+)-train", name or "")
    return match.group(1) if match else None


def task_table(wt: Path) -> list[dict]:
    """The rows of source/cohub/task_registry.py, read as baselines/common.sh reads them."""
    tree = ast.parse((wt / "source" / "cohub" / "task_registry.py").read_text())
    return list(ast.literal_eval(next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign) and node.targets[0].id == "TASKS")).values())


def task_robots(wt: Path, task: str) -> tuple[str, ...]:
    """The task's robots; two when the registry does not know it."""
    count = next((row["robots"] for row in task_table(wt) if row["slug"] == task), 2)
    return ("robot_a", "robot_b", "robot_c")[:count]


def run_dir_names(task: str, mode: str, seed: int, ckpt_tag: str | None,
                  style: str = "shared", robots: tuple[str, ...] = ("robot_a", "robot_b")) -> list[str]:
    """Return the checkpoint run-directory names this eval served.

    Same naming rule as `mh_run_name` in baselines/common.sh. Several runners
    use the name as the wandb run id. A shared decentralized policy is one run
    for every task.
    """
    ckpt_task = CKPT_TASK_ALIASES.get(task, task)
    suffix = f"-{ckpt_tag}" if ckpt_tag else ""
    if mode == "centralized":
        return [f"cohub-{ckpt_task}-unitree_g1x2_centralized-joint-{seed}{suffix}"]
    if style == "shared":
        return [f"cohub-{MULTITASK_CKPT}-unitree_g1x2_decentralized-joint-{seed}{suffix}"]
    return [f"cohub-{ckpt_task}_{robot}-unitree_g1x2_decentralized-joint-{seed}{suffix}"
            for robot in robots]


def pi05_run_ids(wt: Path, task: str, mode: str, seed: int, ckpt_tag: str | None,
                 style: str = "shared") -> list[str]:
    """Return the wandb ids openpi recorded (wandb_id.txt) for the checkpoints
    this eval served."""
    base = wt / "baselines/policy/Pi_05/checkpoints"
    ids = []
    for run in run_dir_names(task, mode, seed, ckpt_tag, style):
        marker = base / run / "wandb_id.txt"
        if not marker.is_file():
            print(f"[wandb] {marker} is missing -- that checkpoint's run cannot be resumed",
                  file=sys.stderr)
            continue
        ids.append(marker.read_text().strip())
    return ids


def resolve_project(entity: str, projects: list[str], names: list[str]) -> str:
    """Return the first project that already carries one of these runs (by name
    or id), else the first project."""
    if len(projects) == 1:
        return projects[0]
    import wandb

    for project in projects:
        try:
            runs = list(wandb.Api().runs(f"{entity}/{project}"))
        except Exception as error:  # noqa: BLE001 - offline, unauthenticated, project gone
            print(f"[wandb] could not list {entity}/{project} ({error}); trying the next project",
                  file=sys.stderr)
            continue
        known = {r.name for r in runs} | {r.id for r in runs}
        if any(name in known for name in names):
            return project
    return projects[0]


def variant_of(run_id: str) -> str:
    """Return the tag policy/DP/train.sh puts between the task and the seed.

    Read from the eval directory name `DP-<mode>-<tag>-seed<n>`. The training
    run is named `obs1-<tag>`, so `obs1-` is added when missing.
    """
    match = re.match(r"^DP-(?:de)?centralized-(.+)-seed\d+(?:-\w+)*$", run_id)
    if not match:
        return "obs1"
    tag = match.group(1)
    return tag if tag.startswith("obs") else f"obs1-{tag}"


def ckpt_step_of(run_id: str) -> int | None:
    """Return the training step from the run id's `-ckpt<N>` suffix, or None
    without one."""
    match = re.search(r"-ckpt(\d+)(?:-|$)", run_id)
    return int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="eval_results/<task>/<run-id>")
    parser.add_argument("--mode", choices=("decentralized", "centralized"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", help="override the task name in results.json")
    parser.add_argument("--variant", help="override the run tag taken from the directory name")
    parser.add_argument("--min-episodes", type=int, default=100,
                        help="refuse to log an aggregate smaller than this (0 disables). The "
                             "benchmark's own run is 100 episodes; scripts/eval/runner.sh passes whatever "
                             "the run asked for, so a deliberately smaller one is not refused")
    parser.add_argument("--force", action="store_true",
                        help="log a partial aggregate or one with client crashes")
    parser.add_argument("--policy", default="DP", choices=sorted(POLICY_PROJECTS),
                        help="which baseline's training runs this evaluation belongs to")
    parser.add_argument("--ckpt-tag", default=None,
                        help="checkpoint directory suffix, for policies whose runs carry one")
    parser.add_argument("--decentralized-style", choices=("shared", "per_robot"), default=None,
                        help="override what a decentralized run of this policy is; the default is "
                             "the policy's own (DECENTRALIZED_STYLE), and matches its serving hook")
    parser.add_argument("--ckpt-step", type=int, default=None,
                        help="training step of the evaluated checkpoint; the default reads the "
                             "`-ckpt<N>` the eval runner put in the directory name. It is the "
                             "x-coordinate of the point this writes, so a sweep over "
                             "10k/20k/30k/40k lands as four points in one run")
    parser.add_argument("--eval-run-name", default=None,
                        help="display name of the policy's -eval run (common.sh: cohub_wandb_name, e.g. "
                             "multitask-b32-40k-eval), used when the training run's own name does not "
                             "give the budget; the default is multitask[-<tag>]-eval")
    parser.add_argument("--label", default=None,
                        help="a re-evaluation of the same checkpoints (common.sh: COHUB_EVAL_LABEL, e.g. "
                             "after_revise): rollout keys go under <task>-<label> and the eval run is "
                             "rebuilt from this run's own results root (eval_results_<label>/)")
    parser.add_argument("--worktree", default=None, help="repo root (default: two levels up)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results = run_dir / "results.json"
    if not results.is_file():
        print(f"[wandb] no results.json under {run_dir} -- did the aggregate step run?", file=sys.stderr)
        return 1
    summary = json.loads(results.read_text())
    if not summary.get("episodes"):
        print(f"[wandb] {results} carries no episodes; nothing to log", file=sys.stderr)
        return 1

    # Refuse a partial aggregate or one with client crashes: wandb history
    # cannot be edited afterwards.
    n = summary["episodes_run"]
    if not args.force and args.min_episodes and n < args.min_episodes:
        print(f"[wandb] {results} has {n} episodes, fewer than --min-episodes "
              f"{args.min_episodes}; refusing to log a partial aggregate", file=sys.stderr)
        return 1
    crashed = sum(1 for e in summary["episodes"] if str(e.get("reason", "")).startswith("client_crash"))
    if not args.force and crashed:
        print(f"[wandb] {crashed} of {n} episodes in {results} are client crashes; "
              f"rerun them before logging (--force to log anyway)", file=sys.stderr)
        return 1

    task = args.task or summary.get("task_name") or run_dir.parent.name
    task = TASK_ALIASES.get(task, task)
    # Match the whole word: "centralized" is a substring of "decentralized".
    mode = args.mode or ("decentralized" if "-decentralized-" in f"-{run_dir.name}-" else "centralized")

    wt = Path(args.worktree) if args.worktree else Path(__file__).resolve().parents[2]
    variant = args.variant or variant_of(run_dir.name)
    ckpt_step = args.ckpt_step if args.ckpt_step is not None else ckpt_step_of(run_dir.name)
    style = args.decentralized_style or DECENTRALIZED_STYLE[args.policy]
    import wandb
    entity = os.environ.get("WANDB_ENTITY") or wandb.Api().default_entity
    projects = POLICY_PROJECTS[args.policy]
    # A three-robot task has a third per-robot run.
    robots = task_robots(wt, task)
    third = {"robots": robots} if len(robots) > 2 else {}
    per_task = args.policy in PER_TASK
    if args.policy == "DP":
        # Addressed by name, which is the id.
        runs = names = wandb_runs_for(mode, task, args.seed, variant, **third)
    elif args.policy == "ACT":
        runs = names = run_dir_names(task, mode, args.seed, args.ckpt_tag, style, **third)
    elif args.policy == "GauDP":
        runs = names = run_dir_names(task, "centralized", args.seed, args.ckpt_tag)
    elif args.policy == "LatentToM":
        # Its run directory carries the task where the others carry the profile.
        suffix = f"-{args.ckpt_tag}" if args.ckpt_tag else ""
        runs = names = [f"cohub-{task}-{task}-joint-{args.seed}{suffix}"]
    elif args.policy == "Pi_05":
        runs = pi05_run_ids(wt, task, mode, args.seed, args.ckpt_tag, style)
        names = runs
    elif args.policy in ("GR00T_N17", "FastWAM", "Psi0"):
        # The wandb id is the checkpoint run-dir name.
        runs = run_dir_names(task, mode, args.seed, args.ckpt_tag, style)
        names = runs
    if not runs:
        print(f"[wandb] no {args.policy} run ids resolved; nothing to log", file=sys.stderr)
        return 1
    # A shared decentralized run holds every task's rollout, so its keys carry the task.
    scope_by_task = mode == "decentralized" and style == "shared" and not per_task
    # The sweep is this evaluation's siblings under the same results root.
    results_root = run_dir.resolve().parent.parent
    project = resolve_project(entity, projects, names)
    # The eval run is named for the budget the policy was trained with, not this shell's.
    trained = trained_label(entity, project, runs[0])
    label = f"-{args.label}" if args.label else ""
    log_to_wandb(summary, run_dir / "videos", mode, task, args.seed, variant,
                 runs=runs, project=project,
                 ckpt_step=ckpt_step, scope_by_task=scope_by_task, label=args.label)
    # Rebuild the policy's eval run, where the sweep reads as a curve.
    if ckpt_step is not None and scope_by_task:
        # The two-robot and the three-robot policy share a directory stem: keep to this task's set.
        rebuild_eval_run(results_root, args.policy, mode, args.seed, tasks=per_robot_task_set(wt, task)[0],
                         project=project,
                         run_name=(f"{MULTITASK_CKPT}-{trained}{label}-eval" if trained
                                   else args.eval_run_name or eval_run_name(args.ckpt_tag, args.label)),
                         ckpt_tag=args.ckpt_tag, min_episodes=args.min_episodes,
                         run_dir_name=run_dir.name)
    elif ckpt_step is not None and (per_task or (mode == "decentralized" and style == "per_robot")):
        # A policy trained per task: one eval run per task set, rebuilt from
        # the sibling evaluations on disk.
        task_set, set_word = per_robot_task_set(wt, task)
        if trained:
            budget = f"{trained}{label}"
        elif args.policy == "DP":
            budget = f"{variant_budget(variant)}{label}"
        else:
            # cohub_wandb_name's `multitask-<budget>[-<tag>][-<label>]-eval`, without its ends.
            budget = re.sub(r"^multitask3?-|-eval$", "", args.eval_run_name or f"{args.policy}-eval")
        rebuild_eval_run(results_root, args.policy, mode, args.seed,
                         project=project,
                         run_name=f"{set_word}-{budget}-eval",
                         ckpt_tag=args.ckpt_tag, min_episodes=args.min_episodes,
                         run_dir_name=run_dir.name, tasks=task_set)
    return 0


def per_robot_task_set(wt: Path, task: str) -> tuple[set[str], str]:
    """Return the task set `task` belongs to and its word: the two-robot
    multitask set or the three-robot one, as common.sh's MH_ALL_TASKS and
    MH_ALL3_TASKS."""
    rows = task_table(wt)
    sets = {"cohub2": {row["slug"] for row in rows if row["multitask"]},
            "cohub3": {row["slug"] for row in rows if row["robots"] == 3}}
    for word, tasks in sets.items():
        if task in tasks:
            return tasks, word
    return {task}, task


def variant_budget(variant: str) -> str:
    """`obs1-b256-e600` -> `b256-e600`: the training budget the run names carry."""
    return re.sub(r"^obs\d+-?", "", variant) or variant


if __name__ == "__main__":
    raise SystemExit(main())
