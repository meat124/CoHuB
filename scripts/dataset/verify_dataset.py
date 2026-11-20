"""Dataset QA: a recorded HDF5 dataset, or the LeRobot export made from it. Needs no simulator.

    .venv/bin/python scripts/dataset/verify_dataset.py datasets/cocarry/data-full
    .venv/bin/python scripts/dataset/verify_dataset.py datasets/cocarry/lerobot --source datasets/cocarry/data-full

HDF5: schema, operator timing and link health, grasp sanity, camera geometry,
provenance. LeRobot export: video length against parquet rows and, with
``--source``, the depth round trip.
"""

import argparse
import json
import pathlib
import sys


import numpy as np  # noqa: E402

from cohub.keys import ROBOTS3  # noqa: E402
from cohub.g1.geometry import (  # noqa: E402
    DEFAULT_WIDTH,
    EGO_PITCH_DOWN_DEG,
    LEGACY_EGO_HFOV_DEG,
    fov_for_fx,
    horizontal_fov_deg,
    pitch_down_deg_from_quat,
)
from cohub.recording.demo_source import hidden_heads_in, DemoSource, parse_selection, shard_files  # noqa: E402

from cohub.task_registry import TASKS  # noqa: E402

REQUIRED_GROUPS = [
    "obs",
    "actions",
    "states/object/pose",
    "states/grasp/latched",
    "states/grasp/force",
    "derived/inter_robot_distance",
    "sync/teleop_sample_time",
    "meta",
]

PER_ROBOT_GROUPS = ["states/robots/{robot}/q", "eef_wrench/{robot}/force"]
"""Required once per robot the file records."""



def recorded_robots(demo) -> list[str]:
    """The robots this demonstration carries state for, in slice order."""
    return [name for name in ROBOTS3 if demo.get(f"states/robots/{name}/q") is not None]

ALIGN_LIMIT_S = {
    "follower_sample_time": 0.020,
    "secondary_sample_time": 0.060,
    "tertiary_sample_time": 0.060,
}
"""p90 budget for operator alignment, per operator channel [s]."""

RTT_LIMIT_S = 0.100
"""p90 budget for the federation round trip [s]."""

STALL_S = 0.150
"""Worst-frame value worth naming as a stall [s]. Reported, never gated."""

def ego_lens() -> tuple[float, float]:
    """(horizontal FOV [deg], mount pitch [deg]) a recording should show: every task uses one ego lens."""
    return horizontal_fov_deg(), EGO_PITCH_DOWN_DEG


FOV_TOLERANCE_DEG = 1.0
"""How far a recorded ego lens may sit from the configured one [deg]."""

PITCH_TOLERANCE_DEG = 5.0
"""How far an ego camera may sit from the mount angle, relative to the torso [deg]."""

WORLD_PITCH_TOLERANCE_DEG = 15.0
"""The same limit in the world frame, for a file with no robot state to read the lean from [deg]."""

MOVED_M = 0.05
"""How far a robot must travel before its ego camera is expected to move [m]."""


def _waist_pitch_column(attrs: dict, robot: str) -> int | None:
    """Which column of ``states/robots/<robot>/q`` is the waist pitch joint."""
    try:
        joints = json.loads(attrs.get("env_args") or "{}")["articulation"][robot]["joint_names"]
    except (TypeError, ValueError, KeyError):
        return None
    return joints.index("waist_pitch_joint") if "waist_pitch_joint" in joints else None


def torso_pitch_deg(demo, attrs: dict, robot: str, stride: int):
    """How far the robot's torso leans forward, per sampled frame [deg].

    Pelvis pitch plus the waist pitch joint. Returns ``None`` if the
    demonstration does not carry the state this needs.
    """
    column = _waist_pitch_column(attrs, robot)
    root = demo.get(f"states/robots/{robot}/root_pose")
    joints = demo.get(f"states/robots/{robot}/q")
    if column is None or root is None or joints is None:
        return None
    pelvis = np.asarray(root)[::stride, 3:]
    waist = np.asarray(joints)[::stride, column]
    count = min(len(pelvis), len(waist))
    if not count:
        return None
    return np.array([pitch_down_deg_from_quat(q) for q in pelvis[:count]]) + np.degrees(waist[:count])


def operator_b_channel(demo) -> str | None:
    """Which sync stamp carries operator 2, or ``None`` if neither moved."""
    for key in ("secondary_sample_time", "follower_sample_time"):
        path = f"sync/{key}"
        if path in demo and np.asarray(demo[path]).squeeze().max() > 0:
            return key
    return None


def _recorded_task(attrs) -> str:
    """The env name the dataset was recorded with, or "" when it never said."""
    env_args = attrs.get("env_args")
    if env_args is None:
        return ""
    try:
        return str(json.loads(env_args).get("env_name") or "")
    except (TypeError, ValueError):
        return ""


def _provenance(attrs) -> dict:
    """The provenance block inside ``env_args``, or ``{}`` if the file has none."""
    try:
        return json.loads(attrs.get("env_args") or "{}").get("provenance") or {}
    except (TypeError, ValueError):
        return {}


def _spread(values: np.ndarray) -> str:
    """Median, p90 and worst of ``values`` [s], formatted in ms."""
    return (
        f"median {np.median(values) * 1e3:.1f} ms, p90 {np.percentile(values, 90) * 1e3:.1f} ms, "
        f"worst {values.max() * 1e3:.1f} ms"
    )


def check_cameras(name: str, demo, attrs: dict) -> int:
    """Check that ``camera_info`` describes the images stored beside it.

    Fails when the ego lens or mount pitch differs from ``cohub.g1.geometry``, or
    when the ego pose stays fixed while the robot travels. Returns the number
    of failures.
    """
    failures = 0
    if "camera_info" not in demo:
        return 0
    for cam in sorted(demo["camera_info"]):
        info = demo[f"camera_info/{cam}"]
        width = DEFAULT_WIDTH
        # Images live under obs/images, or bare images/ in older files.
        for rgb in (f"obs/images/{cam}/rgb", f"images/{cam}/rgb"):
            if rgb in demo:
                width = int(demo[rgb].shape[2])
                break
        if "intrinsics" in info:
            fx = float(np.asarray(info["intrinsics"][0])[0, 0])
            fov = fov_for_fx(fx, width)
            expected_fov, expected_pitch = ego_lens()
            legacy = any(abs(fov - old) <= FOV_TOLERANCE_DEG for old in LEGACY_EGO_HFOV_DEG)
            if cam.startswith("ego") and abs(fov - expected_fov) > FOV_TOLERANCE_DEG and not legacy:
                print(
                    f"FAIL: {name} {cam} intrinsics say {fov:.1f} deg horizontal "
                    f"(fx {fx:.2f} at {width} px), not the configured "
                    f"{expected_fov:.1f} deg -- camera_info does not describe these images"
                )
                failures += 1
            elif cam.startswith("ego") and legacy and abs(fov - expected_fov) > FOV_TOLERANCE_DEG:
                # A legacy lens is reported, not failed.
                print(
                    f"  {name}: {cam} lens {fov:.1f} deg horizontal (fx {fx:.2f} at {width} px) "
                    f"-- LEGACY, pre-ZED-Mini; current config is {expected_fov:.1f} deg"
                )
            else:
                print(f"  {name}: {cam} lens {fov:.1f} deg horizontal (fx {fx:.2f} at {width} px)")

        if "pose" not in info:
            print(f"FAIL: {name} {cam} has no pose -- its images cannot be placed in the scene")
            failures += 1
            continue
        pose = np.asarray(info["pose"])
        if not cam.startswith("ego"):
            continue  # the scene camera does not move

        robot = f"robot_{cam.rsplit('_', 1)[1]}"
        if demo.get(f"states/robots/{robot}/q") is None:
            print(f"FAIL: {name} {cam} names a robot this file does not record ({robot})")
            failures += 1
            continue
        stride = max(len(pose) // 50, 1)
        looking = np.array([pitch_down_deg_from_quat(q) for q in pose[::stride, 3:]])
        lean = torso_pitch_deg(demo, attrs, robot, stride)
        if lean is None:
            mounted = float(np.median(looking))
            tolerance, how = WORLD_PITCH_TOLERANCE_DEG, " in the world (this file records no lean to subtract)"
        else:
            count = min(len(looking), len(lean))
            mounted = float(np.median(looking[:count] - lean[:count]))
            tolerance, how = PITCH_TOLERANCE_DEG, f" below its torso, which leans {np.median(lean):.1f}"
        expected_pitch = ego_lens()[1]
        if abs(mounted - expected_pitch) > tolerance:
            print(f"FAIL: {name} {cam} is aimed {mounted:.1f} deg{how}, not the mount's {expected_pitch:.1f} deg")
            failures += 1
        else:
            print(f"  {name}: {cam} aimed {mounted:.1f} deg{how}")

        # The ego camera rides the torso, so its pose must move when the robot does.
        root = f"states/robots/{robot}/root_pose"
        if root not in demo:
            continue
        travelled = float(np.linalg.norm(np.asarray(demo[root])[:, :3] - np.asarray(demo[root])[0, :3], axis=1).max())
        moved = float(np.linalg.norm(pose[:, :3] - pose[0, :3], axis=1).max())
        if travelled > MOVED_M and moved < 1e-4:
            print(
                f"FAIL: {name} {cam} pose is frozen at the reset pose while {robot} "
                f"travelled {travelled:.2f} m -- the extrinsics are wrong for every frame but the first"
            )
            failures += 1
        else:
            print(f"  {name}: {cam} pose moved {moved:.2f} m while {robot} travelled {travelled:.2f} m")
    return failures


def constant_channels(demo) -> list[str]:
    """Recorded time series that never change, as ``path (value)``.

    Reported, not failed: some channels are constant by design.
    """
    dead: list[str] = []

    def visit(path, node):
        # Images are skipped for cost. camera_info is checked in check_cameras.
        if path.startswith(("images", "obs/images", "camera_info")) or not hasattr(node, "shape"):
            return
        if len(node.shape) == 0 or node.shape[0] < 2:
            return
        values = np.asarray(node)
        if not np.issubdtype(values.dtype, np.number):
            return
        if np.all(values == values[0]):
            flat = np.ravel(values[0])
            shown = f"{flat[0]:g}" if flat.size == 1 else "constant"
            dead.append(f"{path} ({shown})")

    demo.visititems(visit)
    return dead


def verify(path: str) -> int:
    failures = 0
    with DemoSource(path) as demos:
        print(f"demos: {len(demos)}")
        if not len(demos):
            print("FAIL: no demos")
            return 1
        for problem in demos.problems:
            print(f"FAIL: {problem}")
        failures += len(demos.problems)
        # A bimanual task needs both robots holding the object at some point.
        # An id the registry does not know leaves the check off.
        recorded = _recorded_task(demos.attrs)
        carry = any(
            gym_id in recorded and row["bimanual"] for gym_id, row in TASKS.items()
        )

        first = demos[demos.names[0]]
        has_images = any(group in first for group in ("obs/images", "images"))

        stamp = _provenance(demos.attrs)
        if stamp:
            print(
                f"provenance: {stamp.get('written_by', '?')} at {stamp.get('written_at', '?')}, "
                f"commit {(stamp.get('commit') or '?')[:12]}"
                f"{' (dirty tree)' if stamp.get('dirty') else ''}"
            )
            collected = stamp.get("collected")
            if collected:
                print(
                    f"  collected by: {collected.get('written_by', '?')} at "
                    f"{collected.get('written_at', '?')}, commit "
                    f"{(collected.get('commit') or '?')[:12]}"
                    f"{' (dirty tree)' if collected.get('dirty') else ''}"
                )
            elif stamp.get("written_by") not in ("record_demos.py", "", None):
                print(
                    "  collected by: unrecorded -- this file was rewritten before the collection"
                    " stamp was preserved, so which session produced it is unrecoverable"
                )
            render = stamp.get("render") or {}
            spp = render.get("samples_per_pixel")
            mode = render.get("carb", {}).get("/rtx/rendermode")
            if spp:
                print(f"  rendered by: {mode or '?'} at {spp} samples/pixel -- matches evaluation")
            elif has_images:
                print(
                    "  rendered by: unrecorded -- these images were NOT rebuilt on the pinned"
                    " renderer, so they are a different picture than evaluation renders."
                    " See render_dataset_images.py."
                )
        else:
            print("provenance: none recorded -- the code and scene that produced this file are unknown")

        if demos.names and ("obs/images" in demos[demos.names[0]] or "images" in demos[demos.names[0]]):
            heads = hidden_heads_in(demos.attrs)
            if heads:
                print(
                    f"  NOTE recorded with {', '.join(heads)} head hidden -- USD visibility is per "
                    "stage, so ego_b and scene show it missing.\n"
                    "       Rebuild those views with scripts/dataset/render_dataset_images.py before training on them."
                )
            elif heads is None and stamp.get("written_by") == "record_demos.py":
                print(
                    "  NOTE this file predates the hidden_heads stamp; if it came from a headset "
                    "session, the partner's head is missing from ego_b and scene."
                )

        p90s = []
        for name in demos.names:
            demo = demos[name]
            robots = recorded_robots(demo)
            if len(robots) < 2:
                print(f"FAIL: {name} records {len(robots)} robots; the platform contract needs at least two")
                failures += 1
            required = list(REQUIRED_GROUPS) + [
                template.format(robot=robot) for robot in robots for template in PER_ROBOT_GROUPS
            ]
            for key in required:
                if key not in demo:
                    print(f"FAIL: {name} missing '{key}'")
                    failures += 1
            channel = operator_b_channel(demo)
            if channel is None:
                print(f"FAIL: {name} has no operator-2 sample times (neither follower nor federated)")
                failures += 1
            # Operator 3's channel is checked only when it moved.
            channels = [channel] if channel is not None else []
            if "sync/tertiary_sample_time" in demo and np.asarray(demo["sync/tertiary_sample_time"]).max() > 0:
                channels.append("tertiary_sample_time")
            if "sync/teleop_sample_time" in demo:
                t_a = np.asarray(demo["sync/teleop_sample_time"]).squeeze()
                for chan in channels:
                    t_b = np.asarray(demo[f"sync/{chan}"]).squeeze()
                    skew = np.abs(t_b - t_a)
                    p90 = float(np.percentile(skew, 90))
                    limit = ALIGN_LIMIT_S[chan]
                    p90s.append(p90)
                    over = float((skew > limit).mean())
                    print(f"  {name}: operator skew {_spread(skew)} (via {chan}, {over * 100:.0f}% over budget)")
                    if p90 > limit:
                        print(f"FAIL: {name} operator skew p90 {p90 * 1e3:.1f} ms > {limit * 1e3:.0f} ms ({chan})")
                        failures += 1
                    elif skew.max() > STALL_S:
                        print(f"  {name}: NOTE worst frame {skew.max() * 1e3:.0f} ms -- something stalled")

                if "sync/link_rtt" in demo:
                    rtt = np.asarray(demo["sync/link_rtt"]).squeeze()
                    if rtt.max() > 0:
                        rtt_p90 = float(np.percentile(rtt, 90))
                        print(f"  {name}: federation round trip {_spread(rtt)}")
                        if rtt_p90 > RTT_LIMIT_S:
                            print(
                                f"FAIL: {name} link round trip p90 {rtt_p90 * 1e3:.1f} ms > "
                                f"{RTT_LIMIT_S * 1e3:.0f} ms -- the mirror is not keeping up"
                            )
                            failures += 1
                        elif rtt.max() > STALL_S:
                            print(f"  {name}: NOTE worst round trip {rtt.max() * 1e3:.0f} ms -- the mirror stalled")

            # A carry needs both robots on the object at once. A handover needs one hand.
            if "states/grasp/latched" in demo:
                latched = np.asarray(demo["states/grasp/latched"])
                held = latched.sum(axis=1)
                needed = 2 if carry else 1
                if held.max() < needed:
                    what = "two hands on the object" if carry else "a hand on the object"
                    print(f"FAIL: {name} never had {what}")
                    failures += 1
                else:
                    both = ((latched[:, :2].any(axis=1)) & (latched[:, 2:].any(axis=1))).mean()
                    print(f"  {name}: {held.max():.0f} hands max, both robots holding {both * 100:.0f}% of steps")

            failures += check_cameras(name, demo, demos.attrs)

            dead = constant_channels(demo)
            if dead:
                print(f"  {name}: NOTE {len(dead)} channel(s) never change:")
                for path in dead:
                    print(f"      {path}")

        if p90s:
            print(f"worst operator skew p90 across demos: {max(p90s) * 1e3:.1f} ms")

    print("PASS" if failures == 0 else f"{failures} failures")
    return 0 if failures == 0 else 1


QUANTISATION_M = 0.0005
"""Metres. Depth is stored to the millimetre."""

MAX_ENCODABLE_M = 65.0
"""Metres. The two-byte packing wraps past 65.535 m, so farther pixels are not checked."""


def video_keys(info: dict) -> list[str]:
    return sorted(k for k in info["features"] if k.startswith(("observation.images", "observation.depth")))


def check_frame_counts(root: pathlib.Path, episodes: list[dict], keys: list[str]) -> list[str]:
    """Every stream's frame count against its episode's parquet row count."""
    import imageio.v2 as imageio
    import pyarrow.parquet as pq

    failures = []
    for episode in episodes:
        index = episode["episode_index"]
        parquet = root / f"data/chunk-000/episode_{index:06d}.parquet"
        rows = pq.read_metadata(parquet).num_rows
        if rows != episode["length"]:
            failures.append(f"episode {index}: {rows} parquet rows, meta/episodes.jsonl says {episode['length']}")
        for key in keys:
            path = root / f"videos/chunk-000/{key}/episode_{index:06d}.mp4"
            if not path.exists():
                failures.append(f"episode {index}: {key} is missing")
                continue
            reader = imageio.get_reader(path)
            frames = reader.count_frames()
            reader.close()
            if frames != rows:
                failures.append(f"episode {index}: {key} has {frames} frames, the parquet has {rows} rows")
    return failures


def demo_of(root: pathlib.Path) -> dict[int, str]:
    """Which HDF5 demonstration each episode index came from."""
    # Older exports name the file mhbench_provenance.json.
    path = next((root / "meta" / n for n in ("cohub_provenance.json", "mhbench_provenance.json")
                 if (root / "meta" / n).is_file()), root / "meta/cohub_provenance.json")
    provenance = json.loads(path.read_text())
    return {entry["episode_index"]: entry["demo"] for entry in provenance.get("episodes", [])}


def check_depth(root: pathlib.Path, source: pathlib.Path, episodes: list[dict], keys: list[str]) -> tuple[list[str], float]:
    """Decode each depth stream and compare it with the metres it was made from."""
    import h5py
    import imageio.v2 as imageio

    failures, worst = [], 0.0
    names = demo_of(root)
    depth_cameras = [k.rsplit(".", 1)[-1] for k in keys if k.startswith("observation.depth")]
    for episode in episodes:
        index = episode["episode_index"]
        name = names.get(index)
        shards = sorted(source.glob(f"*.{name}.hdf5")) if name else []
        if not shards:
            failures.append(f"episode {index}: no shard for {name} under {source}")
            continue
        with h5py.File(shards[0], "r") as handle:
            demo = handle["data"][name]
            steps = int(np.asarray(demo.attrs["num_samples"]).squeeze())
            for camera in depth_cameras:
                reference = np.asarray(demo[f"obs/images/{camera}/depth"][:steps], dtype=np.float32).squeeze(-1)
                stream = root / f"videos/chunk-000/observation.depth.{camera}/episode_{index:06d}.mp4"
                frames = np.stack([frame for frame in imageio.get_reader(stream)])
                metres = (((frames[..., 0].astype(np.uint16) << 8) | frames[..., 1]).astype(np.float32)) / 1000.0
                measured = np.isfinite(reference) & (reference > 0) & (reference < MAX_ENCODABLE_M)
                error = float(np.abs(metres[measured] - reference[measured]).max()) if measured.any() else 0.0
                worst = max(worst, error)
                # 1e-6 m of slack for float32 rounding.
                if error > QUANTISATION_M + 1e-6:
                    failures.append(f"episode {index}: {camera} depth off by {error * 1000:.3f} mm, over the "
                                    f"{QUANTISATION_M * 1000:.1f} mm quantisation bound")
                lost = int(((metres == 0) & measured).sum())
                if lost:
                    failures.append(f"episode {index}: {camera} has {lost} pixel(s) the renderer measured and the "
                                    "stream calls a no-return")
    return failures, worst


def is_lerobot_export(path: pathlib.Path) -> bool:
    """A directory holding ``meta/info.json`` and no HDF5 shards."""
    return (path / "meta" / "info.json").is_file() and not shard_files(path)


def verify_export(root: pathlib.Path, source: str, episode_selection: str) -> int:
    """Check a LeRobot export: video frame counts, and depth decoding when ``source`` is given."""
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    selection = parse_selection(episode_selection)
    if selection is not None:
        episodes = [e for e in episodes if e["episode_index"] in selection]
    keys = video_keys(info)

    print(f"{root}")
    print(f"  {len(episodes)} episode(s), {len(keys)} video stream(s) each -> {len(episodes) * len(keys)} files")
    print(f"  splits {info.get('splits')}, {info.get('total_frames')} frames at {info.get('fps')} Hz")

    failures = check_frame_counts(root, episodes, keys)
    print(f"  [{'ok ' if not failures else 'FAIL'}] every stream's frame count equals its episode's row count")

    if source:
        depth_failures, worst = check_depth(root, pathlib.Path(source).expanduser(), episodes, keys)
        print(f"  [{'ok ' if not depth_failures else 'FAIL'}] depth decodes to the metres it was rendered from"
              f" -- worst {worst * 1000:.3f} mm, bound {QUANTISATION_M * 1000:.1f} mm")
        failures += depth_failures
    else:
        print("  [skip] depth: pass --source <the HDF5 this was exported from> to check it")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for line in failures[:40]:
            print(f"  {line}")
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")
        return 1
    print("\nRELEASE CHECK PASS")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="A .hdf5, a directory of per-demo shards, or a LeRobot export (the directory holding meta/).")
    parser.add_argument("--source", type=str, default="", help="LeRobot export: the HDF5 it was made from, for the depth check.")
    parser.add_argument("--episodes", type=str, default="", help="LeRobot export: which episode indices to check, e.g. 0,5-7.")
    args = parser.parse_args()
    target = pathlib.Path(args.dataset).expanduser()
    if is_lerobot_export(target):
        sys.exit(verify_export(target, args.source, args.episodes))
    if args.source or args.episodes:
        parser.error("--source and --episodes apply to a LeRobot export (a directory holding meta/info.json)")
    sys.exit(verify(args.dataset))
