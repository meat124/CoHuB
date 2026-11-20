"""Encode recorded demos to MP4. Needs no simulator.

    .venv/bin/python scripts/dataset/encode_demo_videos.py datasets/cocarry/data
    .venv/bin/python scripts/dataset/encode_demo_videos.py datasets/cocarry/data --demos 3,7-9 --depth

Takes one ``.hdf5``, a ``data/`` directory of shards, or the dataset root, and
writes one directory per demo beside it. Demos already encoded are skipped.
"""

import argparse
import pathlib
import sys


from cohub.recording.demo_source import DemoSource, demo_fps, parse_selection, video_dir  # noqa: E402

from cohub.recording.demo_video import output_names, render_demo  # noqa: E402

parser = argparse.ArgumentParser(description="Render recorded demos to MP4.")
parser.add_argument(
    "dataset",
    type=str,
    help="The .hdf5 written by record_demos.py, or a directory of per-demo shards.",
)
parser.add_argument(
    "--demos",
    type=str,
    default="",
    help="Which demos, as indices and ranges: '3', '3,7', '7-9'. Default is all of them.",
)
parser.add_argument(
    "--fps",
    type=float,
    default=0.0,
    help="Frame rate to stamp on the files. 0 reads it from each demo's own timestamps.",
)
parser.add_argument("--quality", type=int, default=6, help="imageio quality, 0-10.")
parser.add_argument(
    "--depth",
    action="store_true",
    help="Also encode the ego cameras' depth as grey video, near bright. For "
    "looking at; the metres are in the HDF5.",
)
parser.add_argument(
    "--rate",
    choices=("sim", "wall"),
    default="sim",
    help="'sim' plays at the simulated rate; 'wall' at the session's own pace, written as '_wallclock' files.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Redo demos that already have an output directory.",
)




def main() -> int:
    args = parser.parse_args()
    dataset = pathlib.Path(args.dataset).expanduser().resolve()
    if not dataset.exists():
        print(f"No such dataset: {dataset}")
        return 1

    selection = parse_selection(args.demos)

    rendered = skipped = empty = 0
    with DemoSource(dataset) as demos:
        if not len(demos):
            print(f"{dataset} holds no demos.")
            return 1

        for name in demos.names:
            index = int(name.split("_")[1])
            if selection is not None and index not in selection:
                continue

            out_dir = video_dir(dataset, name)
            # Skip only when every target file exists.
            suffix = "" if args.rate == "sim" else "_wallclock"
            targets = output_names(demos[name], depth=args.depth, suffix=suffix)
            if targets and all((out_dir / target).exists() for target in targets) and not args.overwrite:
                skipped += 1
                continue

            fps, source = (
                (args.fps, "--fps") if args.fps > 0 else demo_fps(demos[name], demos.attrs, clock=args.rate)
            )
            written = render_demo(
                demos[name], out_dir, fps=fps, quality=args.quality, depth=args.depth, suffix=suffix
            )
            if not written:
                # The demo has no camera data.
                empty += 1
                if out_dir.exists() and not any(out_dir.iterdir()):
                    out_dir.rmdir()
                continue
            rendered += 1
            summary = ", ".join(f"{k} ({v} frames)" for k, v in written.items())
            print(f"{name}/  {summary}  @ {fps:g} fps ({source})")

    print(f"\nrendered {rendered}, skipped {skipped} already done, {empty} with no camera data")
    if empty and not rendered:
        print("Nothing had images -- record with --enable_cameras to get them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
