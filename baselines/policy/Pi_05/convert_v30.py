#!/usr/bin/env python
"""Present CoHuB's LeRobot v2.1 export as a LeRobot v3.0 dataset, without copying it.

Writes new v3.0 metadata beside symlinks to the original parquet and mp4
files, one file per episode. ``--src`` is only read. By default only the two
ego cameras are carried over.

    python baselines/policy/Pi_05/convert_v30.py \\
        --src datasets/cocarry/lerobot --dst "$HF_LEROBOT_HOME/cohub-cocarry"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys

# Each robot's own head view.
DEFAULT_VIDEO_KEYS = ("observation.images.ego_a", "observation.images.ego_b")

logger = logging.getLogger("convert_v30")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _relink(link: Path, target: Path) -> None:
    """Point `link` at `target` with a relative symlink, creating the parent."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target.resolve(), link.parent.resolve()))


def convert(src: Path, dst: Path, video_keys: list[str], *, force: bool) -> None:
    from lerobot.datasets.utils import (  # noqa: PLC0415
        DEFAULT_DATA_PATH,
        DEFAULT_VIDEO_PATH,
        write_episodes,
        write_info,
        write_tasks,
    )
    import datasets  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    info = json.loads((src / "meta" / "info.json").read_text())
    if info["codebase_version"] != "v2.1":
        raise SystemExit(f"{src} is {info['codebase_version']}, expected v2.1")

    missing = [key for key in video_keys if key not in info["features"]]
    if missing:
        raise SystemExit(f"{src} has no video feature(s) {missing}; it has "
                         f"{[k for k, f in info['features'].items() if f['dtype'] == 'video']}")

    if dst.exists():
        if not force:
            raise SystemExit(f"{dst} exists; pass --force to replace it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    episodes = _read_jsonl(src / "meta" / "episodes.jsonl")
    tasks = _read_jsonl(src / "meta" / "tasks.jsonl")
    fps = int(info["fps"])

    # meta/info.json: video features that are not carried over are dropped.
    new_info = dict(info)
    new_info["codebase_version"] = "v3.0"
    new_info["fps"] = fps
    new_info.pop("total_chunks", None)
    new_info.pop("total_videos", None)
    new_info["data_files_size_in_mb"] = 100
    new_info["video_files_size_in_mb"] = 200
    new_info["data_path"] = DEFAULT_DATA_PATH
    new_info["video_path"] = DEFAULT_VIDEO_PATH
    new_info["features"] = {
        key: feature
        for key, feature in info["features"].items()
        if feature["dtype"] != "video" or key in video_keys
    }
    # v3.0 stores fps on every non-video feature.
    for key, feature in new_info["features"].items():
        if feature["dtype"] != "video":
            feature["fps"] = fps
    new_info["total_videos"] = len(video_keys) * len(episodes)
    new_info["cohub_v30_view_of"] = str(src.resolve())
    write_info(new_info, dst)

    # meta/tasks.parquet: indexed by the sentence, task_index as the column.
    write_tasks(
        pd.DataFrame(
            {"task_index": [task["task_index"] for task in tasks]},
            index=[task["task"] for task in tasks],
        ),
        dst,
    )

    # data/ and videos/: chunk 0, file index == episode index, so sorted path
    # order is episode order.
    src_data = src / info["data_path"]
    src_video = src / info["video_path"]

    rows: list[dict] = []
    frame_cursor = 0
    for episode in episodes:
        index = int(episode["episode_index"])
        length = int(episode["length"])

        _relink(
            dst / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=index),
            Path(str(src_data).format(episode_chunk=0, episode_index=index)),
        )

        row = {
            "episode_index": index,
            "tasks": episode["tasks"],
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": index,
            "dataset_from_index": frame_cursor,
            "dataset_to_index": frame_cursor + length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for key in video_keys:
            _relink(
                dst / DEFAULT_VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=index),
                Path(str(src_video).format(episode_chunk=0, video_key=key, episode_index=index)),
            )
            row[f"videos/{key}/chunk_index"] = 0
            row[f"videos/{key}/file_index"] = index
            # 0.0: the episode owns the whole file.
            row[f"videos/{key}/from_timestamp"] = 0.0
            row[f"videos/{key}/to_timestamp"] = length / fps
        rows.append(row)
        frame_cursor += length

    if frame_cursor != info["total_frames"]:
        raise SystemExit(
            f"episode lengths sum to {frame_cursor}, but info.json says {info['total_frames']}"
        )

    write_episodes(datasets.Dataset.from_list(rows), dst)
    logger.info("wrote %d episodes / %d frames -> %s", len(rows), frame_cursor, dst)


def self_check(dst: Path, repo_id: str, action_horizon: int = 50) -> None:
    """Open the result the way training will, and say what came back."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    keys = ("action", "teleop.navigate_command", "teleop.base_height_command")
    dataset = LeRobotDataset(
        repo_id,
        root=dst,
        delta_timestamps={k: [t / 50 for t in range(action_horizon)] for k in keys},
        video_backend="pyav",
    )
    logger.info("episodes=%d frames=%d video_keys=%s",
                dataset.num_episodes, dataset.num_frames, dataset.meta.video_keys)
    item = dataset[dataset.num_frames // 2]
    for key in sorted(item):
        value = item[key]
        logger.info("  %-46s %s", key, tuple(value.shape) if hasattr(value, "shape") else type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="CoHuB LeRobot v2.1 export (read-only)")
    parser.add_argument("--dst", type=Path, required=True, help="where to write the v3.0 view")
    parser.add_argument("--video-keys", default=",".join(DEFAULT_VIDEO_KEYS),
                        help="comma-separated video features to carry over")
    parser.add_argument("--force", action="store_true", help="replace --dst if it exists")
    parser.add_argument("--no-check", action="store_true", help="skip opening the result afterwards")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    src, dst = args.src.resolve(), args.dst.resolve()
    if dst == src or src in dst.parents:
        raise SystemExit("--dst must not be inside --src; this tool never writes into the source")

    convert(src, dst, [k.strip() for k in args.video_keys.split(",") if k.strip()], force=args.force)
    if not args.no_check:
        self_check(dst, dst.name)


if __name__ == "__main__":
    sys.exit(main())
