#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "scripts" / "export"))
sys.path.insert(0, str(WT / "source"))
from cohub import keys as cohub_keys  # noqa: E402
import convert_lerobot  # noqa: E402
import convert_lerobot as core  # noqa: E402
from convert_lerobot import LeRobotDataset, write_flat_lerobot  # noqa: E402

VIEWS = ("local", "global")
ROWS = ("robot", "pair")
PARTNER_PREFIX = "partner_"
GLOBAL_VIDEO = ("ego", "ego_partner")
GLOBAL_TOKEN = "/global"


def partner_of(robot: str) -> str:
    return next(r for r in cohub_keys.ROBOTS if r != robot)


class GlobalRows:

    def __init__(self, dataset: LeRobotDataset, robot: str):
        self._dataset = dataset
        self.robot = robot
        own = cohub_keys.video_keys(robot)[0]
        partner = cohub_keys.video_keys(partner_of(robot))[0]
        self.modality = copy.deepcopy(dataset.modality)
        self.modality["video"] = {
            "ego": dataset.modality["video"][own],
            "ego_partner": dataset.modality["video"][partner],
        }
        self.info = copy.deepcopy(dataset.info)
        self.info["features"]["observation.images.ego"] = dataset.info["features"][f"observation.images.{own}"]
        self.info["features"]["observation.images.ego_partner"] = dataset.info["features"][f"observation.images.{partner}"]

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def task_text(self, columns, robot: str) -> str:
        return self._dataset.task_text(columns, self.robot)


def keys_for(robot: str) -> tuple[list[str], list[str], list[str]]:
    if not robot.endswith(GLOBAL_TOKEN):
        return _core_keys_for(robot)
    own = robot[: -len(GLOBAL_TOKEN)]
    return (
        [*cohub_keys.state_keys(own), *cohub_keys.state_keys(partner_of(own))],
        cohub_keys.action_keys(own),
        list(GLOBAL_VIDEO),
    )


_core_keys_for = convert_lerobot.keys_for
convert_lerobot.keys_for = keys_for


def modality_for(sources: dict[str, LeRobotDataset], first: str, view: str, rows: str) -> dict:
    local = core.shared_modality(sources, first)
    if rows == "pair":
        dataset = sources[first]
        modality: dict[str, dict] = {"state": {}, "action": {}}
        for name, keys in (("state", cohub_keys.state_keys(None)), ("action", cohub_keys.action_keys(None))):
            offset = 0
            for key in keys:
                entry = dataset.modality[name][key]
                width = entry["end"] - entry["start"]
                modality[name][key] = {"start": offset, "end": offset + width}
                offset += width
        videos = [v for v in cohub_keys.video_keys(None) if v != "scene"]
        return {
            **modality,
            "video": {v: {"original_key": f"observation.images.{v}"} for v in videos},
            "annotation": {cohub_keys.language_key(None).removeprefix("annotation."): {"original_key": "annotation.tasks"}},
        }
    if view == "local":
        return local
    width = max(entry["end"] for entry in local["state"].values())
    state = dict(local["state"])
    for key, entry in local["state"].items():
        state[f"{PARTNER_PREFIX}{key}"] = {"start": entry["start"] + width, "end": entry["end"] + width}
    return {
        **local,
        "state": state,
        "video": {v: {"original_key": f"observation.images.{v}"} for v in GLOBAL_VIDEO},
    }


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default=str(WT / "datasets"), help="Directory holding <task>/lerobot for every task.")
    parser.add_argument("--out", required=True, help="Dataset directory to write.")
    parser.add_argument("--split", default="all", choices=("all", "train", "val"))
    parser.add_argument("--tasks", default=",".join(core.TASKS), help="Comma-separated subset.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Only the first N episodes of each (task, split).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--view", default=None, choices=VIEWS, help="global: a row also carries its partner's camera and state.")
    parser.add_argument("--role", default=None, choices=cohub_keys.ROBOTS, help="That robot's rows only.")
    parser.add_argument("--rows", default="robot", choices=ROWS, help="pair: one row per source episode holding both robots.")
    args = parser.parse_args()
    if args.rows == "pair":
        if args.role:
            raise SystemExit("--role keeps one robot's rows; --rows pair has no per-robot rows")
        if args.view == "local":
            raise SystemExit("--rows pair holds both robots' cameras and state: it is the global view")
        args.view = "global"
    args.view = args.view or "local"
    if (args.view, args.role, args.rows) == ("local", None, "robot"):
        raise SystemExit("both roles in the local view is the benchmark's own dataset, built by baselines/policy/GR00T_N17/train_hook.sh data")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    datasets_dir = Path(args.datasets).expanduser().resolve()
    sources: dict[str, LeRobotDataset] = {}
    for task in tasks:
        root = datasets_dir / task / "lerobot"
        if not (root / "meta" / "info.json").exists():
            root = datasets_dir / task
        if not (root / "meta" / "info.json").exists():
            raise SystemExit(f"no LeRobot export for {task!r} under {datasets_dir}/{task}")
        sources[task] = LeRobotDataset(root)
        robots = core.ROBOTS_BY_SLUG.get(task, core.ROBOTS)
        if tuple(robots) != tuple(cohub_keys.ROBOTS):
            raise SystemExit(f"the cells are defined for two-robot tasks; {task} has {len(robots)}")

    def block(split: str) -> list[tuple]:
        streams = []
        for task in tasks:
            indices = sources[task].episode_indices_for_split(split)
            if args.max_episodes:
                indices = indices[: args.max_episodes]
            if args.rows == "pair":
                streams.append((sources[task], "duo", indices))
                continue
            for robot in cohub_keys.ROBOTS:
                if args.role and robot != args.role:
                    continue
                if args.view == "global":
                    streams.append((GlobalRows(sources[task], robot), f"{robot}{GLOBAL_TOKEN}", indices))
                else:
                    streams.append((sources[task], robot, indices))
        return streams

    if args.split == "all":
        train, val = block("train"), block("val")
        streams = train + val
        train_count = sum(len(s[2]) for s in train)
        splits = {"train": f"0:{train_count}", "val": f"{train_count}:{train_count + sum(len(s[2]) for s in val)}"}
    else:
        streams = block(args.split)
        splits = None

    out = Path(args.out).expanduser().resolve()
    write_flat_lerobot(
        streams,
        out,
        cameras=list(GLOBAL_VIDEO) if args.view == "global" and args.rows == "robot" else (
            [v for v in cohub_keys.video_keys(None) if v != "scene"] if args.rows == "pair" else None),
        overwrite=args.overwrite,
        label=f"multitask/{args.split}",
        splits=splits,
        modality=modality_for(sources, tasks[0], args.view, args.rows),
        stats=args.split != "all",
        codebase_version="v2.1" if args.split == "all" else "v2.0",
    )

    provenance, index = [], 0
    for dataset, robot, indices in streams:
        for episode_index in indices:
            provenance.append({
                "episode_index": index,
                "source": str(dataset.root),
                "source_episode": episode_index,
                "robot": robot.removesuffix(GLOBAL_TOKEN),
            })
            index += 1
    (out / "meta" / "cohub_multitask_sources.json").write_text(json.dumps(
        {"split": args.split, "view": args.view, "role": args.role, "rows": args.rows, "tasks": tasks, "episodes": provenance},
        indent=2) + "\n")
    print(f"[multitask] {index} episodes from {len(tasks)} task(s): view={args.view} role={args.role or 'both'} rows={args.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
