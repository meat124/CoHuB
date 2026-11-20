#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "scripts" / "export"))
sys.path.insert(0, str(WT / "source"))
import h5py  # noqa: E402
import numpy as np  # noqa: E402
from cohub import keys as cohub_keys  # noqa: E402
from convert_lerobot import (  # noqa: E402
    DP_CAMERA_KEYS,
    DP_DEPTH_KEYS,
    LeRobotDataset,
    _append,
    _open_zarr,
    describe,
    keys_for,
    robots_in,
)

VIEWS = ("local", "global")
SHARED_CAMERA = cohub_keys.VIDEO_KEYS[cohub_keys.SHARED][0]


def ego_views() -> list[str]:
    return [camera for robot in cohub_keys.ROBOTS for camera in cohub_keys.video_keys(robot)]


def keys_for_view(robot: str, view: str) -> tuple[list[str], list[str], list[str]]:
    if view == "global":
        return cohub_keys.state_keys(None), cohub_keys.action_keys(robot), ego_views()
    return keys_for(robot)


def output_suffix(robot: str, view: str) -> str:
    return f"_{robot}" if view == "local" else f"_{view}_{robot}"


def require_pair(dataset: LeRobotDataset, what: str) -> None:
    present = robots_in(dataset)
    if present != list(cohub_keys.ROBOTS):
        raise SystemExit(f"{what} is defined for a two-robot export; {dataset.root} has {present}")


def members_of(robot: str) -> tuple[str, ...]:
    return cohub_keys.ROBOTS if robot == cohub_keys.SHARED else (robot,)


def plan_for(args, robot: str, stream_names: tuple[str, ...] | None = None):
    plans = []
    for member in members_of(robot):
        state_keys, action_keys, cameras = keys_for_view(member, args.view)
        if stream_names is not None and len(cameras) > len(stream_names):
            raise SystemExit(f"diffusion_policy's loader takes {len(stream_names)} image streams at most; got {len(cameras)}")
        depth_cameras = [c for c in cameras if c in cohub_keys.DEPTH_CAMERAS] if args.depth else []
        if args.depth and not depth_cameras:
            raise SystemExit(f"--depth was asked for, but none of {cameras} records depth")
        plans.append((member, state_keys, action_keys, list(cameras), depth_cameras))
    return plans


def convert_act(dataset: LeRobotDataset, indices: list[int], out_path: Path, args, robot: str) -> None:
    shared = robot == cohub_keys.SHARED
    plans = plan_for(args, robot)
    suffix = "_shared" if shared else output_suffix(robot, args.view)
    root = out_path / f"{args.stem}_act{suffix}{args.split_suffix}"
    if root.exists() and args.overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for member, state_keys, action_keys, _, _ in plans:
        describe(dataset, member, state_keys, action_keys)

    total = len(plans) * len(indices)
    lengths = []
    for member, state_keys, action_keys, cameras, depth_cameras in plans:
        for episode_index in indices:
            position = len(lengths)
            columns = dataset.table(episode_index)
            qpos = dataset.select(columns, "state", state_keys)
            action = dataset.select(columns, "action", action_keys)
            qvel = np.zeros_like(qpos)
            path = root / f"episode_{position}.hdf5"
            with h5py.File(path, "w") as handle:
                handle.attrs["sim"] = True
                handle.attrs["task"] = dataset.task_text(columns, member)
                handle.attrs["cohub_view"] = args.view
                handle.attrs["cohub_robot"] = member
                handle.attrs["state_keys"] = list(state_keys)
                handle.attrs["action_keys"] = list(action_keys)
                handle.create_dataset("action", data=action)
                observations = handle.create_group("observations")
                observations.create_dataset("qpos", data=qpos)
                observations.create_dataset("qvel", data=qvel)
                images = observations.create_group("images")
                for camera in cameras:
                    frames = dataset.video(episode_index, camera)
                    if len(frames) != len(action):
                        raise SystemExit(f"episode {episode_index}: {camera} has {len(frames)} frames for {len(action)} rows")
                    images.create_dataset(SHARED_CAMERA if shared else camera, data=frames)
                if args.depth:
                    depths = observations.create_group("depth")
                    for camera in depth_cameras:
                        frames = dataset.depth(episode_index, camera)
                        if len(frames) != len(action):
                            raise SystemExit(f"episode {episode_index}: {camera} depth has {len(frames)} frames for {len(action)} rows")
                        stream = depths.create_dataset(
                            SHARED_CAMERA if shared else camera, data=frames, compression="gzip", compression_opts=4)
                        stream.attrs["units_per_metre"] = dataset.depth_units(camera)
                        stream.attrs["invalid_value"] = 0
                        stream.attrs["note"] = "metres = value / units_per_metre; 0 means the camera got no return"
            lengths.append(len(action))
            print(f"[act{suffix}] {position + 1}/{total} episode {episode_index}: {len(action)} steps -> {path.name}")
    if shared:
        cameras = [SHARED_CAMERA]
    print(f"[act{suffix}] wrote {total} episode file(s) to {root} (cameras={list(cameras)})")
    print(f"[act{suffix}] num_episodes={total} max_episode_len={max(lengths)}")


def convert_diffusion_policy(dataset: LeRobotDataset, indices: list[int], out_path: Path, args, robot: str) -> None:
    zarr = _open_zarr()
    shared = robot == cohub_keys.SHARED
    plans = plan_for(args, robot, DP_CAMERA_KEYS)
    suffix = "_shared" if shared else output_suffix(robot, args.view)
    store = out_path / f"{args.stem}_diffusion_policy{suffix}{args.split_suffix}.zarr"
    if store.exists():
        if not args.overwrite:
            raise SystemExit(f"{store} exists; --overwrite to replace it")
        shutil.rmtree(store)
    for member, state_keys, action_keys, _, _ in plans:
        describe(dataset, member, state_keys, action_keys)

    root = zarr.open(str(store), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    ends, total = [], 0
    count = len(plans) * len(indices)
    for member, state_keys, action_keys, cameras, depth_cameras in plans:
        for episode_index in indices:
            columns = dataset.table(episode_index)
            state = dataset.select(columns, "state", state_keys)
            action = dataset.select(columns, "action", action_keys)
            _append(data, "state", state)
            _append(data, "action", action)
            for key, camera in zip(DP_CAMERA_KEYS, cameras):
                frames = dataset.video(episode_index, camera)
                if len(frames) != len(action):
                    raise SystemExit(f"episode {episode_index}: {camera} has {len(frames)} frames for {len(action)} rows")
                _append(data, key, frames)
            for key, camera in zip(DP_DEPTH_KEYS, depth_cameras):
                frames = dataset.depth(episode_index, camera)
                if len(frames) != len(action):
                    raise SystemExit(f"episode {episode_index}: {camera} depth has {len(frames)} frames for {len(action)} rows")
                _append(data, key, frames)
            task_text = dataset.task_text(columns, "duo" if shared else member)
            total += len(action)
            ends.append(total)
            print(f"[diffusion_policy{suffix}] {len(ends)}/{count} episode {episode_index}: {len(action)} steps")

    depth_units = dataset.depth_units(depth_cameras[0]) if depth_cameras else None
    if shared:
        state_keys = cohub_keys.state_keys(cohub_keys.SHARED)
        action_keys = cohub_keys.action_keys(cohub_keys.SHARED)
        cameras = [SHARED_CAMERA]
        depth_cameras = [SHARED_CAMERA] if depth_cameras else []
    meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
    root.attrs["state_keys"] = list(state_keys)
    root.attrs["action_keys"] = list(action_keys)
    root.attrs["camera_map"] = dict(zip(DP_CAMERA_KEYS, cameras))
    root.attrs["task"] = task_text
    root.attrs["fps"] = dataset.fps
    root.attrs["cohub_view"] = args.view
    root.attrs["cohub_robot"] = "+".join(members_of(robot))
    if depth_cameras:
        root.attrs["depth_map"] = dict(zip(DP_DEPTH_KEYS, depth_cameras))
        root.attrs["depth_units_per_metre"] = depth_units
        root.attrs["depth_invalid_value"] = 0
    print(f"[diffusion_policy{suffix}] wrote {store} ({total} steps, {count} episodes)")
    print(f"[diffusion_policy{suffix}] camera map: {dict(zip(DP_CAMERA_KEYS, cameras))}")


CONVERTERS = {"act": convert_act, "diffusion_policy": convert_diffusion_policy}


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-path", required=True, help="A LeRobot export: datasets/<task>/lerobot.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--format", required=True, choices=sorted(CONVERTERS))
    parser.add_argument("--robot", default="shared",
                        help="shared: one dataset over both robots' episodes. robot_a / robot_b, both or each: per-robot outputs.")
    parser.add_argument("--view", default="local", choices=VIEWS,
                        help="global: a per-robot output observes both robots (86D, both ego cameras).")
    parser.add_argument("--depth", action="store_true", help="Also repack the ego cameras' depth.")
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--split", default=None, help="A name from meta/info.json's splits; a non-train split's output is suffixed _<split>.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.split and args.max_demos:
        raise SystemExit("--split and --max-demos are mutually exclusive")

    source = Path(args.source_path).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"source path not found: {source}")
    dataset = LeRobotDataset(source)
    args.stem = source.parent.name if source.name == "lerobot" else source.name
    args.split_suffix = f"_{args.split}" if args.split and args.split != "train" else ""
    out_path = Path(args.output_path).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    choice = args.robot.strip().lower()
    if choice == cohub_keys.SHARED:
        if args.view != "local":
            raise SystemExit("--robot shared is the local view. One policy observing and driving both robots is scripts/export/convert_lerobot.py --robot duo")
        require_pair(dataset, "--robot shared")
        targets = [cohub_keys.SHARED]
    else:
        if args.view == "local":
            raise SystemExit("the local view of one robot is scripts/export/convert_lerobot.py --robot each")
        require_pair(dataset, f"--view {args.view}")
        targets = {"both": list(cohub_keys.ROBOTS), "each": robots_in(dataset)}.get(choice, [choice])
        for robot in targets:
            if robot not in cohub_keys.ROBOTS:
                raise SystemExit(f"--robot must be shared, robot_a, robot_b, both or each; got {robot!r}")

    indices = dataset.episode_indices_for_split(args.split) if args.split else dataset.episode_indices(args.max_demos)
    print(f"{source}: {len(dataset.episodes)} episode(s) at {dataset.fps:.0f} Hz, converting {len(indices)}")
    for robot in targets:
        CONVERTERS[args.format](dataset, indices, out_path, args, robot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
