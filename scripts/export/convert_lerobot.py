"""Repack a LeRobot export into what each baseline trains on.

    python scripts/export/convert_lerobot.py --source-path datasets/cocarry/lerobot \\
        --output-path baselines/data --format act
    .venv/bin/python scripts/export/convert_lerobot.py --format multitask --out baselines/data/multitask/lerobot

Formats:

    gr00t              validates the source in place; copies nothing
    act                one episode_N.hdf5 per episode
    diffusion_policy   one Zarr replay buffer (written with zarr 2)
    fastwam            a slim LeRobot dataset with symlinked videos
    multitask          every task flattened into one LeRobot dataset
    psi0               the Ψ₀ view of that flattened dataset

`multitask` and `psi0` take flags of their own (`--format multitask --help`).
`--robot duo` writes both robots (state 86, action 70); `--robot robot_a` one
robot and its camera (state 43, action 35).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


import h5py  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from cohub import keys as cohub_keys  # noqa: E402
from cohub.task_registry import TASKS as _REGISTRY  # noqa: E402
from cohub.task_registry import slugs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

DP_CAMERA_KEYS = ("img", "img1", "img2")
"""diffusion_policy's image stream names. The first is the primary view."""

DP_DEPTH_KEYS = ("depth", "depth1", "depth2")
"""The depth stream names, in the same order."""

ROBOT_CHOICES = ("duo", "robot_a", "robot_b", "robot_c")
"""`robot_c` exists only in a three-robot export. `duo` is the two-robot centralized layout."""

TASKS = slugs(multitask=True)
"""Every collected task, in episode layout order. Must equal `MH_ALL_TASKS` in baselines/common.sh."""

ROBOTS = cohub_keys.ROBOTS
"""The default two robots. A task's own robots come from ROBOTS_BY_SLUG."""

ROBOTS_BY_SLUG = {row["slug"]: cohub_keys.robots(row["robots"]) for row in _REGISTRY.values()}

PSI0_VIDEO_KEY = "observation.images.ego"
"""The flattened dataset's camera key. Pass it to Ψ₀ as ``--data.transform.repack.image-keys``."""


def robots_in(dataset: "LeRobotDataset") -> list[str]:
    """The robots of an export: those whose ego view is in ``meta/modality.json``."""
    video = dataset.modality.get("video", {})
    return [r for r in cohub_keys.ALL_ROBOTS if cohub_keys.VIDEO_KEYS[r][0] in video]


class LeRobotDataset:
    """A read-only view of one exported dataset, resolved through ``meta/``."""

    def __init__(self, root: Path):
        self.root = root
        meta = root / "meta"
        if not (meta / "info.json").exists():
            raise SystemExit(
                f"{root} is not a LeRobot export (no meta/info.json). "
                "Run scripts/export/export_lerobot.py on the recorded shards first."
            )
        self.info = json.loads((meta / "info.json").read_text())
        self.modality = json.loads((meta / "modality.json").read_text())
        self.episodes = [json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines() if line.strip()]
        self.tasks = [json.loads(line) for line in (meta / "tasks.jsonl").read_text().splitlines() if line.strip()]
        self.fps = float(self.info.get("fps", 30))
        self.chunks_size = int(self.info["chunks_size"])

    def episode_indices(self, limit: int | None) -> list[int]:
        indices = [int(episode["episode_index"]) for episode in self.episodes]
        return indices[:limit] if limit else indices

    def episode_indices_for_split(self, name: str) -> list[int]:
        """Episodes in ``meta/info.json``'s ``splits[name]``, e.g. ``"50:65"``."""
        splits = self.info.get("splits", {})
        if name not in splits:
            raise SystemExit(f"no split {name!r} in meta/info.json; available: {sorted(splits) or 'none'}")
        start, end = (int(bound) for bound in splits[name].split(":"))
        indices = [int(episode["episode_index"]) for episode in self.episodes]
        return [index for index in indices if start <= index < end]

    def _chunk(self, episode_index: int) -> int:
        return episode_index // self.chunks_size

    def table(self, episode_index: int) -> dict[str, np.ndarray]:
        path = self.root / self.info["data_path"].format(
            episode_chunk=self._chunk(episode_index), episode_index=episode_index
        )
        table = pq.read_table(path)
        columns = {}
        for name in table.column_names:
            values = table.column(name).to_numpy(zero_copy_only=False)
            columns[name] = np.stack(values) if isinstance(values[0], np.ndarray) else values
        return columns

    def video(self, episode_index: int, camera: str) -> np.ndarray:
        """One camera's frames, decoded. (T, H, W, 3) uint8."""
        entry = self.modality["video"].get(camera)
        if entry is None:
            raise SystemExit(f"no camera {camera!r} in this dataset; it has {sorted(self.modality['video'])}")
        path = self.root / self.info["video_path"].format(
            episode_chunk=self._chunk(episode_index),
            video_key=entry["original_key"],
            episode_index=episode_index,
        )
        reader = imageio.get_reader(str(path))
        try:
            frames = np.stack([np.asarray(frame) for frame in reader])
        finally:
            reader.close()
        return frames

    def _depth_column(self, camera: str) -> str:
        entry = self.modality["video"].get(f"{camera}_depth")
        if entry is None:
            raise SystemExit(
                f"this dataset has no depth for {camera!r}. Either the recording had none, or it "
                "was exported before depth crossed over -- re-export it with a current "
                "scripts/export/export_lerobot.py."
            )
        return entry["original_key"]

    def depth_units(self, camera: str) -> float:
        """Depth units per metre, read from ``meta/info.json``."""
        return float(self.info["features"][self._depth_column(camera)]["info"]["depth.units_per_metre"])

    def depth(self, episode_index: int, camera: str) -> np.ndarray:
        """One camera's depth in the file's own units (millimetres). (T, H, W) uint16.

        Zero means the camera got no return. See :meth:`depth_units` for the scale.
        """
        column = self._depth_column(camera)
        path = self.root / self.info["video_path"].format(
            episode_chunk=self._chunk(episode_index), video_key=column, episode_index=episode_index
        )
        reader = imageio.get_reader(str(path))
        try:
            frames = np.stack([np.asarray(frame) for frame in reader])
        finally:
            reader.close()
        return (frames[..., 0].astype(np.uint16) << 8) | frames[..., 1].astype(np.uint16)

    def select(self, columns: dict[str, np.ndarray], modality: str, keys: list[str]) -> np.ndarray:
        """Concatenate the named modality keys, in the order given."""
        blocks = []
        default = "observation.state" if modality == "state" else "action"
        for key in keys:
            entry = self.modality[modality].get(key)
            if entry is None:
                raise SystemExit(f"no {modality} key {key!r} in meta/modality.json; re-export the dataset")
            column = entry.get("original_key", default)
            if column not in columns:
                raise SystemExit(f"{modality} key {key!r} needs column {column!r}, which the parquet does not have")
            blocks.append(columns[column][:, entry["start"] : entry["end"]])
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def task_text(self, columns: dict[str, np.ndarray], robot: str) -> str:
        """This robot's instruction for one episode, resolved through ``tasks.jsonl``."""
        key = cohub_keys.language_key(None if robot == "duo" else robot)
        if key not in columns:
            key = cohub_keys.language_key(None)
        index = int(columns[key][0])
        return self.tasks[index]["task"]


def keys_for(robot: str) -> tuple[list[str], list[str], list[str]]:
    """(state keys, action keys, cameras) for one training target."""
    target = None if robot == "duo" else robot
    return (
        cohub_keys.state_keys(target),
        cohub_keys.action_keys(target),
        cohub_keys.video_keys(target),
    )


def describe(dataset: LeRobotDataset, robot: str, state_keys: list[str], action_keys: list[str]) -> tuple[int, int]:
    """Print the state and action widths of this run. Returns (state_dim, action_dim)."""
    state_dim = sum(dataset.modality["state"][k]["end"] - dataset.modality["state"][k]["start"] for k in state_keys)
    action_dim = sum(dataset.modality["action"][k]["end"] - dataset.modality["action"][k]["start"] for k in action_keys)
    print(f"[{robot}] state {state_dim}D from {len(state_keys)} keys, action {action_dim}D from {len(action_keys)} keys")
    return state_dim, action_dim


# gr00t: validate only


def convert_gr00t(dataset: LeRobotDataset, indices: list[int], out_path: Path, args, robot: str) -> None:
    """Validate that the export is what the GR00T configs ask for."""
    state_keys, action_keys, cameras = keys_for(robot)
    problems = []
    for modality, keys in (("state", state_keys), ("action", action_keys)):
        missing = [k for k in keys if k not in dataset.modality[modality]]
        if missing:
            problems.append(f"{modality} keys absent from modality.json: {missing}")
    missing_video = [c for c in cameras if c not in dataset.modality["video"]]
    if missing_video:
        problems.append(f"cameras absent: {missing_video}")
    language = cohub_keys.language_key(None if robot == "duo" else robot).removeprefix("annotation.")
    if language not in dataset.modality["annotation"]:
        problems.append(f"annotation key absent: {language}")

    shortest = min(episode["length"] for episode in dataset.episodes)
    if shortest < cohub_keys.ACTION_HORIZON:
        problems.append(
            f"shortest episode is {shortest} steps, under the {cohub_keys.ACTION_HORIZON}-step action chunk"
        )
    if not (dataset.root / "meta" / "stats.json").exists():
        print("[gr00t] meta/stats.json is absent -- GR00T writes it at train start, or run gr00t/data/stats.py")

    if problems:
        for line in problems:
            print(f"  {line}")
        raise SystemExit(f"{len(problems)} problem(s): this dataset is not ready for GR00T")

    state_dim, action_dim = describe(dataset, robot, state_keys, action_keys)
    config = {"duo": "cohub_duo", "robot_a": "cohub_robot_a", "robot_b": "cohub_robot_b"}[robot]
    depth_present = [c for c in cameras if f"{c}_depth" in dataset.modality["video"]]
    if depth_present:
        print(f"[gr00t] depth is present for {depth_present}; no modality config consumes it")
    print(f"[gr00t] {dataset.root} is ready: {len(indices)} episodes, {state_dim}D state, {action_dim}D action")
    print("[gr00t] train on it in place -- nothing is copied:")
    print("  bash baselines/policy/GR00T_N17/train.sh <task>")
    print(f"  modality config: {ROOT / 'baselines/policy/GR00T_N17/configs' / (config + '.py')}")


# ACT: one HDF5 per episode


def convert_act(dataset: LeRobotDataset, indices: list[int], out_path: Path, args, robot: str) -> None:
    state_keys, action_keys, cameras = keys_for(robot)
    cameras = args.cameras or cameras
    depth_cameras = [c for c in cameras if c in cohub_keys.DEPTH_CAMERAS] if args.depth else []
    if args.depth and not depth_cameras:
        raise SystemExit(f"--depth was asked for, but none of {cameras} records depth")
    suffix = "" if robot == "duo" else f"_{robot}"
    root = out_path / f"{args.stem}_act{suffix}{args.split_suffix}"
    if root.exists() and args.overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    describe(dataset, robot, state_keys, action_keys)

    lengths = []
    for position, episode_index in enumerate(indices):
        columns = dataset.table(episode_index)
        qpos = dataset.select(columns, "state", state_keys)
        action = dataset.select(columns, "action", action_keys)
        # ACT's own dataloader never reads qvel, but its files carry the key;
        # zeros rather than a finite difference, which would claim to be a
        # measurement this dataset does not have.
        qvel = np.zeros_like(qpos)

        path = root / f"episode_{position}.hdf5"
        with h5py.File(path, "w") as handle:
            handle.attrs["sim"] = True
            handle.attrs["task"] = dataset.task_text(columns, robot)
            handle.create_dataset("action", data=action)
            observations = handle.create_group("observations")
            observations.create_dataset("qpos", data=qpos)
            observations.create_dataset("qvel", data=qvel)
            images = observations.create_group("images")
            for camera in cameras:
                frames = dataset.video(episode_index, camera)
                if len(frames) != len(action):
                    raise SystemExit(f"episode {episode_index}: {camera} has {len(frames)} frames for {len(action)} rows")
                images.create_dataset(camera, data=frames)
            if args.depth:
                depths = observations.create_group("depth")
                for camera in depth_cameras:
                    frames = dataset.depth(episode_index, camera)
                    if len(frames) != len(action):
                        raise SystemExit(f"episode {episode_index}: {camera} depth has {len(frames)} frames for {len(action)} rows")
                    # Smooth 16-bit data, so gzip earns its time here in a way
                    # it does not on the colour frames.
                    stream = depths.create_dataset(camera, data=frames, compression="gzip", compression_opts=4)
                    stream.attrs["units_per_metre"] = dataset.depth_units(camera)
                    stream.attrs["invalid_value"] = 0
                    stream.attrs["note"] = "metres = value / units_per_metre; 0 means the camera got no return"
        lengths.append(len(action))
        print(f"[act{suffix}] {position + 1}/{len(indices)} episode {episode_index}: {len(action)} steps -> {path.name}")

    depth_note = f", depth={depth_cameras}" if depth_cameras else ""
    print(f"[act{suffix}] wrote {len(indices)} episode file(s) to {root} (cameras={list(cameras)}{depth_note})")
    print(f"[act{suffix}] num_episodes={len(indices)} max_episode_len={max(lengths)}")


# Diffusion Policy: one Zarr replay buffer


def _open_zarr():
    try:
        import zarr
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise SystemExit("diffusion_policy output needs zarr 2: run this under the DP environment") from error
    # zarr 3 writes format 3 by default, which DP's zarr 2 cannot read.
    if int(zarr.__version__.split(".")[0]) >= 3:
        raise SystemExit(f"diffusion_policy output needs zarr 2, got {zarr.__version__}: run this under the DP environment")
    return zarr


def _append(group, key: str, array: np.ndarray) -> None:
    if key in group:
        group[key].append(array)
    else:
        group.create_dataset(key, data=array, chunks=(1, *array.shape[1:]))


def convert_diffusion_policy(dataset: LeRobotDataset, indices: list[int], out_path: Path, args, robot: str) -> None:
    zarr = _open_zarr()
    state_keys, action_keys, cameras = keys_for(robot)
    cameras = args.cameras or cameras
    if len(cameras) > len(DP_CAMERA_KEYS):
        raise SystemExit(
            f"diffusion_policy's loader takes {len(DP_CAMERA_KEYS)} image streams at most "
            f"({', '.join(DP_CAMERA_KEYS)}); got {len(cameras)}. Pass --cameras to choose."
        )
    depth_cameras = [c for c in cameras if c in cohub_keys.DEPTH_CAMERAS] if args.depth else []
    if args.depth and not depth_cameras:
        raise SystemExit(f"--depth was asked for, but none of {cameras} records depth")
    suffix = "" if robot == "duo" else f"_{robot}"
    store = out_path / f"{args.stem}_diffusion_policy{suffix}{args.split_suffix}.zarr"
    if store.exists():
        if not args.overwrite:
            raise SystemExit(f"{store} exists; --overwrite to replace it")
        shutil.rmtree(store)
    describe(dataset, robot, state_keys, action_keys)

    root = zarr.open(str(store), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")

    ends = []
    total = 0
    for position, episode_index in enumerate(indices):
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
        task_text = dataset.task_text(columns, robot)
        total += len(action)
        ends.append(total)
        print(f"[diffusion_policy{suffix}] {position + 1}/{len(indices)} episode {episode_index}: {len(action)} steps")

    meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
    root.attrs["state_keys"] = list(state_keys)
    root.attrs["action_keys"] = list(action_keys)
    root.attrs["camera_map"] = dict(zip(DP_CAMERA_KEYS, cameras))
    root.attrs["task"] = task_text
    root.attrs["fps"] = dataset.fps
    if depth_cameras:
        root.attrs["depth_map"] = dict(zip(DP_DEPTH_KEYS, depth_cameras))
        root.attrs["depth_units_per_metre"] = dataset.depth_units(depth_cameras[0])
        root.attrs["depth_invalid_value"] = 0
    print(f"[diffusion_policy{suffix}] wrote {store} ({total} steps, {len(indices)} episodes)")
    print(f"[diffusion_policy{suffix}] camera map: {dict(zip(DP_CAMERA_KEYS, cameras))}")
    if depth_cameras:
        print(f"[diffusion_policy{suffix}] depth map: {dict(zip(DP_DEPTH_KEYS, depth_cameras))} "
              f"(uint16, {root.attrs['depth_units_per_metre']:g} per metre, 0 = no return)")


# FastWAM: a slim LeRobot dataset


def fastwam_stats_entry(values: np.ndarray) -> dict:
    """v2.0 ``meta/stats.json`` entry for one (N, D) float feature. ``count`` is required."""
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "count": [int(len(values))],
    }


def fastwam_image_stats(dataset: LeRobotDataset, episode_index: int, camera: str) -> dict:
    """Image stats in the (3, 1, 1) normalized-[0, 1] shape stats.json uses, sampled from one episode."""
    frames = dataset.video(episode_index, camera)
    sample = frames[:: max(1, len(frames) // 64)].astype(np.float32) / 255.0  # (S, H, W, 3)
    per_channel = sample.reshape(-1, 3)
    return {
        "mean": [[[float(v)]] for v in per_channel.mean(axis=0)],
        "std": [[[float(v)]] for v in per_channel.std(axis=0)],
        "min": [[[float(v)]] for v in per_channel.min(axis=0)],
        "max": [[[float(v)]] for v in per_channel.max(axis=0)],
        "count": [int(len(sample))],
    }


def convert_fastwam(dataset: LeRobotDataset, indices: list[int], out_path: Path, args, robot: str) -> None:
    """Write one split of one training target as the LeRobot dataset FastWAM reads.

    The output is `<dir-stem>[_<robot>]/lerobot[_<split>]/`, where `dir-stem`
    defaults to `<task>_fastwam`. Train and val are separate datasets.
    """
    suffix = "" if robot == "duo" else f"_{robot}"
    split_dir = "lerobot" if not args.split or args.split == "train" else f"lerobot_{args.split}"
    write_flat_lerobot(
        [(dataset, robot, indices)],
        out_path / f"{args.dir_stem}{suffix}" / split_dir,
        cameras=args.cameras,
        overwrite=args.overwrite,
        label=f"fastwam{suffix}",
    )


def write_flat_lerobot(
    streams: list[tuple[LeRobotDataset, str, list[int]]],
    root: Path,
    *,
    cameras: list[str] | None = None,
    overwrite: bool = False,
    label: str = "flat",
    splits: dict[str, str] | None = None,
    modality: dict | None = None,
    stats: bool = True,
    codebase_version: str = "v2.0",
) -> None:
    """Write one LeRobot dataset out of one or more (dataset, robot, episodes) streams.

    MP4s are symlinked and episodes appended in stream order. ``stats`` writes the
    ``meta/stats.json`` FastWAM parses (codebase ``v2.0``); GR00T and pi0.5 take
    ``v2.1`` without it.
    """
    import pyarrow as pa

    if root.exists():
        if not overwrite:
            raise SystemExit(f"{root} exists; --overwrite to replace it")
        shutil.rmtree(root)
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    fps = {stream[0].fps for stream in streams}
    if len(fps) != 1:
        raise SystemExit(f"sources disagree on fps ({sorted(fps)}); they cannot share one dataset")

    episodes_meta, tasks, task_of = [], [], {}
    all_state, all_action = [], []
    global_index = 0
    new_index = 0
    key_of: dict[str, str] = {}
    first: tuple[LeRobotDataset, str] | None = None

    for dataset, robot, indices in streams:
        state_keys, action_keys, default_cameras = keys_for(robot)
        chosen = [c for c in (cameras or default_cameras) if cameras or c != "scene"]
        if not chosen:
            raise SystemExit("no cameras left to convert; pass --cameras")
        # A single camera is named `ego`, so both robots' rows read through one data config.
        stream_keys = {cam: cam for cam in chosen} if len(chosen) > 1 else {chosen[0]: "ego"}
        if key_of and list(stream_keys.values()) != list(key_of.values()):
            raise SystemExit(
                f"stream {robot} contributes cameras {list(stream_keys.values())}, but the "
                f"dataset already holds {list(key_of.values())}")
        key_of = stream_keys
        if first is None:
            first = (dataset, chosen[0])
        describe(dataset, robot, state_keys, action_keys)

        for episode_index in indices:
            columns = dataset.table(episode_index)
            state = dataset.select(columns, "state", state_keys)
            action = dataset.select(columns, "action", action_keys)
            n = len(action)
            task_text = dataset.task_text(columns, robot)
            if task_text not in task_of:
                task_of[task_text] = len(tasks)
                tasks.append(task_text)

            table = pa.table(
                {
                    "observation.state": pa.array([row for row in state.astype(np.float32)]),
                    "action": pa.array([row for row in action.astype(np.float32)]),
                    "timestamp": pa.array((np.arange(n) / dataset.fps).astype(np.float32)),
                    "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                    "episode_index": pa.array(np.full(n, new_index, dtype=np.int64)),
                    "index": pa.array(np.arange(global_index, global_index + n, dtype=np.int64)),
                    "task_index": pa.array(np.full(n, task_of[task_text], dtype=np.int64)),
                    # The same index under an annotation.* name, for loaders that read the
                    # language key from an annotation column.
                    "annotation.tasks": pa.array(np.full(n, task_of[task_text], dtype=np.int64)),
                }
            )
            pq.write_table(table, root / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet")

            for camera, key in key_of.items():
                src = (dataset.root / dataset.info["video_path"].format(
                    episode_chunk=dataset._chunk(episode_index),
                    video_key=dataset.modality["video"][camera]["original_key"],
                    episode_index=episode_index,
                )).resolve()
                dst_dir = root / "videos" / "chunk-000" / f"observation.images.{key}"
                dst_dir.mkdir(parents=True, exist_ok=True)
                (dst_dir / f"episode_{new_index:06d}.mp4").symlink_to(src)

            episodes_meta.append({"episode_index": new_index, "tasks": [task_text], "length": n})
            all_state.append(state)
            all_action.append(action)
            global_index += n
            new_index += 1
            print(f"[{label}] {new_index}/{sum(len(s[2]) for s in streams)} "
                  f"{dataset.root.parent.name}/{robot} episode {episode_index}: {n} steps")

    dataset, camera = first
    src_features = dataset.info["features"]
    height, width = src_features[f"observation.images.{camera}"]["shape"][:2]
    video_feature = {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": src_features[f"observation.images.{camera}"].get("info"),
    }
    features = {
        "observation.state": {"dtype": "float32", "shape": [int(all_state[0].shape[1])], "names": None},
        "action": {"dtype": "float32", "shape": [int(all_action[0].shape[1])], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "annotation.tasks": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key in key_of.values():
        features[f"observation.images.{key}"] = dict(video_feature)

    info = {
        "codebase_version": codebase_version,
        "robot_type": dataset.info.get("robot_type", "unitree_g1_x2"),
        "total_episodes": new_index,
        "total_frames": int(global_index),
        "total_tasks": len(tasks),
        "total_videos": new_index * len(key_of),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": dataset.fps,
        "splits": splits or {"train": f"0:{new_index}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in episodes_meta))
    (root / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(tasks)))
    if modality is not None:
        (root / "meta" / "modality.json").write_text(json.dumps(modality, indent=2))

    if stats:
        aggregate = {
            "observation.state": fastwam_stats_entry(np.concatenate(all_state)),
            "action": fastwam_stats_entry(np.concatenate(all_action)),
        }
        image_stats = fastwam_image_stats(dataset, streams[0][2][0], camera)
        for key in key_of.values():
            aggregate[f"observation.images.{key}"] = image_stats
        (root / "meta" / "stats.json").write_text(json.dumps(aggregate, indent=2))

    print(f"[{label}] wrote {root} ({global_index} steps, {new_index} episodes, "
          f"cameras {dict(key_of)})")


# multitask: every task in one flattened dataset

MULTITASK_HELP = """Flatten every task into the one LeRobot dataset a shared policy trains on.

    .venv/bin/python scripts/export/convert_lerobot.py --format multitask --out baselines/data/multitask/lerobot
    .venv/bin/python scripts/export/convert_lerobot.py --format multitask --split train --out <dir>

By default each source episode becomes one episode per robot, with that
robot's ego camera, 43D state, 35D action and instruction. Episodes are laid
out task by task, robot_a before robot_b. The source of every episode is
written to meta/cohub_multitask_sources.json. MP4s are symlinked.

  --split all (default)  one dataset with train/val `splits` (GR00T, pi0.5)
  --split train | val    that block alone, all-train (FastWAM)

Two-robot tasks only:

  --split all (default)  one dataset with `splits` train/val, for GR00T and
                         the pi0.5 view built from it.
  --split train | val    that block alone, all-train, for FastWAM.

Episodes are laid out task by task, robot_a's before robot_b's; the map is
written to meta/cohub_multitask_sources.json. MP4s are symlinked, not re-encoded.
"""


def shared_modality(sources: dict[str, LeRobotDataset], first: str) -> dict:
    """``meta/modality.json`` for the flattened dataset.

    Key widths are read from the source dataset ``sources[first]``. Slices
    follow the key order of `cohub_keys`' ``shared`` target, which is the order
    `write_flat_lerobot` concatenates them in.
    """
    dataset = sources[first]
    modality: dict[str, dict] = {"state": {}, "action": {}}
    for name, keys in (
        ("state", cohub_keys.state_keys("robot_a")),
        ("action", cohub_keys.action_keys("robot_a")),
    ):
        offset = 0
        for key in keys:
            entry = dataset.modality[name][key]
            width = entry["end"] - entry["start"]
            modality[name][key.removeprefix("robot_a_")] = {"start": offset, "end": offset + width}
            offset += width
    return {
        **modality,
        "video": {"ego": {"original_key": "observation.images.ego"}},
        # The column holds an index into meta/tasks.jsonl.
        "annotation": {
            cohub_keys.language_key(cohub_keys.SHARED).removeprefix("annotation."): {
                "original_key": "annotation.tasks"
            }
        },
    }


def main_multitask() -> int:
    parser = argparse.ArgumentParser(
        description=MULTITASK_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", required=True, type=str.lower, choices=("multitask",))
    parser.add_argument("--datasets", default=str(ROOT / "datasets"),
                        help="Directory holding <task>/lerobot for every task.")
    parser.add_argument("--out", required=True, help="Dataset directory to write.")
    parser.add_argument("--split", default="all", choices=("all", "train", "val"),
                        help="all (default) keeps both blocks and declares the splits; "
                             "train/val writes that block alone, all-train.")
    parser.add_argument("--tasks", default=",".join(TASKS),
                        help="Comma-separated subset, for a quick check.")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Only the first N episodes of each (task, split), for a smoke run.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output.")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    datasets_dir = Path(args.datasets).expanduser().resolve()
    sources: dict[str, LeRobotDataset] = {}
    for task in tasks:
        root = datasets_dir / task / "lerobot"
        if not (root / "meta" / "info.json").exists():
            root = datasets_dir / task
        if not (root / "meta" / "info.json").exists():
            raise SystemExit(
                f"no LeRobot export for {task!r} under {datasets_dir}/{task} -- "
                f"export it first (scripts/export/export_lerobot.py)")
        sources[task] = LeRobotDataset(root)

    def block(split: str) -> list[tuple[LeRobotDataset, str, list[int]]]:
        streams = []
        for task in tasks:
            indices = sources[task].episode_indices_for_split(split)
            if args.max_episodes:
                indices = indices[: args.max_episodes]
            for robot in ROBOTS_BY_SLUG.get(task, ROBOTS):
                streams.append((sources[task], robot, indices))
        return streams

    if args.split == "all":
        train, val = block("train"), block("val")
        streams = train + val
        train_count = sum(len(s[2]) for s in train)
        splits = {"train": f"0:{train_count}",
                  "val": f"{train_count}:{train_count + sum(len(s[2]) for s in val)}"}
    else:
        streams = block(args.split)
        splits = None

    out = Path(args.out).expanduser().resolve()
    write_flat_lerobot(
        streams,
        out,
        overwrite=args.overwrite,
        label=f"multitask/{args.split}",
        splits=splits,
        modality=shared_modality(sources, tasks[0]),
        # GR00T writes its own meta/stats.json, so `--split all` writes none.
        stats=args.split != "all",
        # v2.1 for the GR00T/pi0.5 copy, v2.0 for FastWAM.
        codebase_version="v2.1" if args.split == "all" else "v2.0",
    )

    # Record where every episode came from.
    provenance, index = [], 0
    for dataset, robot, indices in streams:
        for episode_index in indices:
            provenance.append({
                "episode_index": index,
                "source": str(dataset.root),
                "source_episode": episode_index,
                "robot": robot,
            })
            index += 1
    (out / "meta" / "cohub_multitask_sources.json").write_text(
        json.dumps({"split": args.split, "tasks": tasks, "episodes": provenance}, indent=2) + "\n")
    counts = ", ".join(f"{t} x{len(ROBOTS_BY_SLUG.get(t, ROBOTS))}" for t in tasks)
    print(f"[multitask] {index} episodes from {len(tasks)} task(s): {counts}")
    return 0


# psi0: the Ψ₀ view of the flattened dataset

PSI0_HELP = """Build the Ψ₀ view of the flattened all-task dataset.

    .venv/bin/python scripts/export/convert_lerobot.py --format psi0 --split train --out <dir>

Writes a LeRobot v2.0 dataset with an aggregate meta/stats.json. Compared with
``baselines/data/multitask/lerobot``:

- columns are permuted into Ψ₀'s order (``psi0_layout.py``); the state drops
  the twelve leg joints and gains the commanded base height;
- ``observation.state`` is renamed ``states``;
- ``meta/stats_psi0.json`` is a copy of ``meta/stats.json``.

MP4s are symlinked. Ψ₀ takes train and val as separate datasets, so write one
directory per split.
"""


def episode_range(info: dict, split: str) -> range:
    """The source's own split block, read from ``meta/info.json``."""
    if split == "all":
        return range(int(info["total_episodes"]))
    spec = info.get("splits", {}).get(split)
    if spec is None:
        raise SystemExit(f"source declares no {split!r} split (has {sorted(info.get('splits', {}))})")
    start, end = (int(part) for part in spec.split(":"))
    return range(start, end)


def main_psi0() -> int:
    parser = argparse.ArgumentParser(
        description=PSI0_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", required=True, type=str.lower, choices=("psi0",))
    parser.add_argument("--source", default=str(ROOT / "baselines" / "data" / "multitask" / "lerobot"),
                        help="The flattened all-task dataset (baselines/utils/prepare_data.sh builds it).")
    parser.add_argument("--out", required=True, help="Dataset directory to write.")
    parser.add_argument("--split", default="train", choices=("train", "val", "all"),
                        help="Which block of the source to write, all-train in the output.")
    parser.add_argument("--max-episodes", type=int, default=None, help="First N episodes, for a smoke run.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output.")
    args = parser.parse_args()

    import pyarrow as pa

    # Ψ₀'s column layout lives with its adapter.
    sys.path.insert(0, str(ROOT / "baselines" / "policy" / "Psi0"))
    import psi0_layout as layout

    source = Path(args.source).expanduser().resolve()
    if not (source / "meta" / "info.json").exists():
        raise SystemExit(f"no dataset at {source} -- build it with "
                         f"baselines/utils/prepare_data.sh gr00t")
    src = LeRobotDataset(source)
    info = src.info

    out = Path(args.out).expanduser().resolve()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; --overwrite to replace it")
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    (out / "data" / "chunk-000").mkdir(parents=True)
    video_dir = out / "videos" / "chunk-000" / PSI0_VIDEO_KEY
    video_dir.mkdir(parents=True)

    indices = list(episode_range(info, args.split))
    if args.max_episodes:
        indices = indices[: args.max_episodes]

    src_episodes = [json.loads(line) for line in
                    (source / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    src_tasks = {entry["task_index"]: entry["task"] for entry in
                 (json.loads(line) for line in
                  (source / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip())}

    episodes_meta: list[dict] = []
    tasks: list[str] = []
    task_of: dict[str, int] = {}
    all_state: list[np.ndarray] = []
    all_action: list[np.ndarray] = []
    global_index = 0

    for new_index, episode_index in enumerate(indices):
        table = pq.read_table(source / info["data_path"].format(
            episode_chunk=0, episode_index=episode_index))
        mh_state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        mh_action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        n = len(mh_action)

        action = layout.action_to_psi0(mh_action)
        # Ψ₀'s state carries the previous commanded height. Frame 0 uses its own.
        height = np.concatenate([mh_action[:1, 31], mh_action[:-1, 31]])
        state = layout.state_to_psi0(mh_state, height)

        if not np.array_equal(layout.action_from_psi0(action), mh_action):
            raise SystemExit(f"episode {episode_index}: action permutation is not invertible")

        task_text = src_tasks[int(table.column("task_index")[0].as_py())]
        if task_text not in task_of:
            task_of[task_text] = len(tasks)
            tasks.append(task_text)

        pq.write_table(
            pa.table({
                "states": pa.array([row for row in state.astype(np.float32)]),
                "action": pa.array([row for row in action.astype(np.float32)]),
                "timestamp": pa.array((np.arange(n) / src.fps).astype(np.float32)),
                "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                "episode_index": pa.array(np.full(n, new_index, dtype=np.int64)),
                "index": pa.array(np.arange(global_index, global_index + n, dtype=np.int64)),
                "task_index": pa.array(np.full(n, task_of[task_text], dtype=np.int64)),
                "annotation.tasks": pa.array(np.full(n, task_of[task_text], dtype=np.int64)),
            }),
            out / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet",
        )
        (video_dir / f"episode_{new_index:06d}.mp4").symlink_to(
            (source / info["video_path"].format(
                episode_chunk=0, video_key=PSI0_VIDEO_KEY, episode_index=episode_index)).resolve())

        episodes_meta.append({"episode_index": new_index, "tasks": [task_text], "length": n})
        all_state.append(state)
        all_action.append(action)
        global_index += n
        if new_index % 50 == 0 or new_index + 1 == len(indices):
            print(f"[psi0/{args.split}] {new_index + 1}/{len(indices)} "
                  f"(source episode {episode_index}, {n} steps)")

    if not episodes_meta:
        raise SystemExit(f"no episodes in the {args.split!r} block")

    src_video = info["features"][PSI0_VIDEO_KEY]
    features = {
        "states": {"dtype": "float32", "shape": [layout.STATE_COLUMNS], "names": None},
        "action": {"dtype": "float32", "shape": [layout.ACTION_COLUMNS], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "annotation.tasks": {"dtype": "int64", "shape": [1], "names": None},
        PSI0_VIDEO_KEY: {"dtype": "video", "shape": src_video["shape"],
                    "names": ["height", "width", "channels"], "info": src_video.get("info")},
    }
    (out / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v2.0",
        "robot_type": info.get("robot_type", "unitree_g1_x2"),
        "total_episodes": len(episodes_meta),
        "total_frames": int(global_index),
        "total_tasks": len(tasks),
        "total_videos": len(episodes_meta),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": src.fps,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": info["data_path"],
        "video_path": info["video_path"],
        "features": features,
    }, indent=2) + "\n")
    (out / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in episodes_meta))
    (out / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(tasks)))

    stats = {
        "states": fastwam_stats_entry(np.concatenate(all_state)),
        "action": fastwam_stats_entry(np.concatenate(all_action)),
        PSI0_VIDEO_KEY: fastwam_image_stats(src, indices[0], "ego"),
    }
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))
    shutil.copyfile(out / "meta" / "stats.json", out / "meta" / "stats_psi0.json")

    print(f"[psi0/{args.split}] wrote {out} "
          f"({global_index} steps, {len(episodes_meta)} episodes, {len(tasks)} sentences)")
    return 0


CONVERTERS = {
    "gr00t": convert_gr00t,
    "act": convert_act,
    "diffusion_policy": convert_diffusion_policy,
    "fastwam": convert_fastwam,
}


def _format_arg(argv: list[str]) -> str:
    """``--format``'s value, read before any parser: multitask and psi0 have flags of their own."""
    for position, arg in enumerate(argv):
        if arg == "--format" and position + 1 < len(argv):
            return argv[position + 1]
        if arg.startswith("--format="):
            return arg.split("=", 1)[1]
    return ""


def main() -> int:
    own_flags = {"multitask": main_multitask, "psi0": main_psi0}.get(_format_arg(sys.argv[1:]).strip().lower())
    if own_flags is not None:
        return own_flags()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-path", required=True, help="A LeRobot export: datasets/<task>/lerobot.")
    parser.add_argument("--output-path", default="", help="Where to write. Not used by --format gr00t.")
    parser.add_argument("--format", required=True,
                        help="gr00t (validate in place), act, diffusion_policy, or fastwam. "
                        "multitask and psi0 take their own flags.")
    parser.add_argument(
        "--robot",
        default="duo",
        help="duo: one policy for both robots. robot_x: that robot alone. both: robot_a + robot_b. each: every robot.",
    )
    parser.add_argument("--cameras", default=None, help="Comma-separated camera list. Default: the robot's own view.")
    parser.add_argument(
        "--depth",
        action="store_true",
        help="Also repack the ego cameras' depth, as uint16 millimetres beside the colour frames.",
    )
    parser.add_argument("--max-demos", type=int, default=None, help="Only the first N episodes, for a quick check.")
    parser.add_argument(
        "--split",
        default=None,
        help="A split in meta/info.json (train, val). Non-train output gets a '_<split>' suffix. Not with --max-demos.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output.")
    parser.add_argument(
        "--dir-stem",
        default=None,
        help="--format fastwam only: output directory name before the robot suffix. Default '<task>_fastwam'.",
    )
    args = parser.parse_args()

    fmt = args.format.strip().lower()
    if fmt not in CONVERTERS:
        raise SystemExit(f"unknown --format {args.format!r}; pick one of gr00t, act, diffusion_policy, fastwam, "
                         "multitask, psi0")
    args.cameras = [c.strip() for c in args.cameras.split(",") if c.strip()] if args.cameras else None
    if args.split and args.max_demos:
        raise SystemExit("--split and --max-demos are mutually exclusive")

    source = Path(args.source_path).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"source path not found: {source}")
    dataset = LeRobotDataset(source)
    # datasets/<task>/lerobot -> "<task>"
    args.stem = source.parent.name if source.name == "lerobot" else source.name
    # A non-train split gets a `_<split>` suffix.
    args.split_suffix = f"_{args.split}" if args.split and args.split != "train" else ""
    args.dir_stem = args.dir_stem or f"{args.stem}_fastwam"

    if fmt == "gr00t":
        out_path = Path(args.output_path) if args.output_path else source
    else:
        if not args.output_path:
            raise SystemExit("--output-path is required for act and diffusion_policy")
        out_path = Path(args.output_path).expanduser()
        out_path.mkdir(parents=True, exist_ok=True)

    choice = args.robot.strip().lower()
    if choice == "both":
        targets = ["robot_a", "robot_b"]
    elif choice == "each":
        targets = robots_in(dataset)
    else:
        targets = [choice]
    for robot in targets:
        if robot not in ROBOT_CHOICES:
            raise SystemExit(f"--robot must be one of {', '.join(ROBOT_CHOICES)}, 'both' or 'each'; got {robot!r}")
        if robot != "duo" and robot not in robots_in(dataset):
            raise SystemExit(f"--robot {robot}: {source} has no {cohub_keys.VIDEO_KEYS[robot][0]} camera")

    indices = (
        dataset.episode_indices_for_split(args.split) if args.split else dataset.episode_indices(args.max_demos)
    )
    print(f"{source}: {len(dataset.episodes)} episode(s) at {dataset.fps:.0f} Hz, converting {len(indices)}")
    for robot in targets:
        CONVERTERS[fmt](dataset, indices, out_path, args, robot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
