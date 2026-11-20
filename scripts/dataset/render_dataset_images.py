"""Render camera frames into a dataset that was collected without them.

    python scripts/dataset/render_dataset_images.py datasets/cocarry/cocarry.hdf5 \\
        --headless --enable_cameras --device cpu

Each frame's recorded state is written into the scene and rendered; nothing is
simulated. The source is not modified: the output is a new dataset with
``obs/images/<camera>/{rgb,depth}`` and ``camera_info/`` added.
"""

import argparse
import json
import pathlib
import time


from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Re-render a states-only dataset's camera frames back into it.")
parser.add_argument(
    "dataset",
    type=str,
    help="The .hdf5 written by record_demos.py, or a directory of per-demo shards.",
)
parser.add_argument(
    "--out",
    type=str,
    default="",
    help="Destination. Default: <name>.full.hdf5 beside a single file, or data-full/ beside a shard directory.",
)
parser.add_argument("--demos", type=str, default="", help="Indices and ranges: '3', '3,7', '7-9'. Default is all.")
parser.add_argument(
    "--cameras",
    type=str,
    default="ego_a,ego_b,scene",
    help="Cameras to render, from ego_a, ego_b, scene (plus ego_c and scene_2 on the tasks that have them).",
)
parser.add_argument("--no_depth", action="store_true", help="Store RGB only, dropping the ego cameras' depth.")
parser.add_argument("--width", type=int, default=0, help="Render width; 0 keeps the task's configured resolution.")
parser.add_argument("--height", type=int, default=0)
parser.add_argument(
    "--compression",
    type=str,
    default="gzip",
    help="HDF5 compression for the image datasets: gzip, lzf, or none.",
)
parser.add_argument("--compression_level", type=int, default=4, help="gzip level, 0-9.")
parser.add_argument(
    "--renders_per_frame",
    type=int,
    default=2,
    help="Render calls per frame. A second one lets the denoiser settle on a restored state.",
)
parser.add_argument("--overwrite", action="store_true", help="Replace the destination if it exists.")
parser.add_argument(
    "--resume",
    action="store_true",
    help="Skip shards whose destination already exists (a shard is renamed from .partial only when complete).",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Start even if the GPU already looks busy.",
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Task whose scene to render in. Default: the one recorded in the dataset's env_args.",
)
AppLauncher.add_app_launcher_args(parser)
# Pink IK is CPU-only.
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()

from cohub.task_registry import gym_id  # noqa: E402

if args_cli.task is not None:
    args_cli.task = gym_id(args_cli.task)

# Without cameras the task config strips its camera sensors.
args_cli.enable_cameras = True


def _refuse_a_full_card() -> None:
    """Exit if another process already holds the GPU.

    A renderer that boots into a full card hangs instead of failing.
    ``--force`` skips the check.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        used, total = (int(v.strip()) for v in out.stdout.splitlines()[0].split(","))
    except Exception:
        return

    if used <= BUSY_CARD_MIB or args_cli.force:
        return
    print(
        f"\n  The GPU is holding {used} MiB of {total} MiB before this process has started.\n"
        "  Rendering needs the card to itself: a restore that boots into a full\n"
        "  card does not fail, it hangs at 100% CPU with no frames and no error.\n\n"
        "  Close the collection session (and give Kit a few seconds to exit),\n"
        "  check with `nvidia-smi`, then run this again. `--force` skips this check.\n"
    )
    raise SystemExit(1)


BUSY_CARD_MIB = 1500
"""VRAM in use above which another process is assumed to own the card."""

_refuse_a_full_card()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import cohub.tasks  # noqa: F401, E402
from cohub.recording.demo_source import (  # noqa: E402
    restamp,
    dataset_env_name,
    dataset_root,
    parse_selection,
    shard_files,
)
from cohub.keys import CAMERA_SENSORS  # noqa: E402
from cohub.recording.recorder import camera_frame, camera_pose  # noqa: E402
from cohub.runtime import (  # noqa: E402
    configure_render,
    pin_render_products,
    render_settings,
    reset_scene_to,
    exit_gate,
)


CAMERAS = dict(CAMERA_SENSORS)


def frame_state(states, index: int, device: str) -> dict:
    """One recorded frame, in the nested form ``scene.reset_to`` wants.

    Only ``articulation`` and ``rigid_object`` are restored.
    """
    return {
        family: {
            name: {
                field: torch.tensor(states[family][name][field][index], dtype=torch.float32, device=device).reshape(
                    1, -1
                )
                for field in states[family][name]
            }
            for name in states[family]
        }
        for family in ("articulation", "rigid_object")
        if family in states
    }


def storage_kwargs(shape: tuple[int, ...], dtype) -> dict:
    """Creation arguments for one camera stream's dataset, one chunk per frame."""
    kwargs = {"shape": shape, "dtype": dtype, "chunks": (1,) + shape[1:]}
    if args_cli.compression == "gzip":
        kwargs["compression"] = "gzip"
        kwargs["compression_opts"] = args_cli.compression_level
    elif args_cli.compression not in ("", "none"):
        kwargs["compression"] = args_cli.compression
    return kwargs


def camera_streams(scene, key: str) -> dict:
    """``{dataset name: one frame}`` for one camera, named and typed as recorded."""
    sensor = scene.sensors[CAMERAS[key]]
    sensor.update(0.0, force_recompute=True)
    return {
        name: values[0].cpu().numpy()
        for name, values in camera_frame(sensor.data.output).items()
        if not (name == "depth" and args_cli.no_depth)
    }


def camera_info_streams(scene, key: str) -> dict:
    """``{dataset name: one frame}`` of ``camera_info`` for one camera."""
    sensor = scene.sensors[CAMERAS[key]]
    return {
        "intrinsics": sensor.data.intrinsic_matrices.torch[0].cpu().numpy(),
        "pose": camera_pose(key, sensor)[0].cpu().numpy(),
    }


from cohub.recording.demo_source import human  # noqa: E402

def copy_states(source_demo, target_demo, rerendering=()) -> None:
    """Copy everything the session stored, except images of re-rendered cameras.

    Kept cameras stay at the path their file used, ``obs/images`` or ``images``.
    """
    for key, value in source_demo.attrs.items():
        target_demo.attrs[key] = value
    for member in source_demo:
        if member == "images":
            continue
        if member == "obs" and "images" in source_demo["obs"]:
            obs_out = target_demo.create_group("obs")
            for sub in source_demo["obs"]:
                if sub != "images":
                    source_demo.copy(source_demo[f"obs/{sub}"], obs_out, name=sub)
            continue
        source_demo.copy(source_demo[member], target_demo, name=member)

    for group_path in ("obs/images", "images"):
        source_images = source_demo.get(group_path)
        if source_images is None:
            continue
        kept = [cam for cam in source_images if cam not in rerendering]
        if kept:
            images = target_demo.require_group(group_path)
            for cam in kept:
                source_demo.copy(source_images[cam], images, name=cam)
        break


def render_demo(name, source_demo, target_demo, scene, env, env_ids, wanted) -> int:
    """Rebuild and store one demonstration's frames. Returns frames written.

    Frame ``k`` is rendered from the state before step ``k``: ``initial_state``
    for ``k = 0``, ``states[k-1]`` after.
    """
    states = source_demo["states"]
    initial = source_demo.get("initial_state")
    count = int(source_demo.attrs["num_samples"])
    images = target_demo.require_group("obs/images")
    # camera_info is rewritten for the cameras being re-rendered.
    for key in wanted:
        if f"camera_info/{key}" in target_demo:
            del target_demo[f"camera_info/{key}"]
    info_group = target_demo.require_group("camera_info")
    started = time.perf_counter()
    stores: dict[tuple[str, str], h5py.Dataset] = {}

    for step in range(count):
        if step == 0 and initial is not None:
            state = frame_state(initial, 0, env.device)
        else:
            # No initial_state: frame 0 uses states[0].
            state = frame_state(states, max(step - 1, 0), env.device)
        # reset_scene_to skips the root velocity write that kinematic bodies refuse.
        reset_scene_to(scene, state, env_ids, is_relative=True)
        env.sim.forward()
        for _ in range(max(1, args_cli.renders_per_frame)):
            env.sim.render()
        for key in wanted:
            streams = dict(camera_streams(scene, key))
            for stream, frame in camera_info_streams(scene, key).items():
                store = stores.get((key, "info:" + stream))
                if store is None:
                    store = info_group.create_dataset(f"{key}/{stream}", shape=(count,) + frame.shape,
                                                      dtype=frame.dtype)
                    stores[(key, "info:" + stream)] = store
                store[step] = frame
            for stream, frame in streams.items():
                store = stores.get((key, stream))
                if store is None:
                    # Sized from the first frame, not from the config.
                    store = images.create_dataset(
                        f"{key}/{stream}", **storage_kwargs((count,) + frame.shape, frame.dtype)
                    )
                    stores[(key, stream)] = store
                store[step] = frame
        if step and step % 200 == 0:
            rate = step / (time.perf_counter() - started)
            remaining = (count - step) / max(rate, 1e-6)
            print(f"  {name}: {step}/{count} frames ({rate:.1f}/s, ~{remaining:.0f}s left)")

    elapsed = time.perf_counter() - started
    written = ", ".join(f"{key}/{stream}" for key, stream in sorted(stores) if not stream.startswith("info:"))
    print(f"{name}: {count} frames -> {written}  in {elapsed:.0f}s")
    return count


def main() -> int:
    source = pathlib.Path(args_cli.dataset).expanduser().resolve()
    if not source.exists():
        print(f"No such dataset: {source}")
        return 1

    # A sharded dataset restores shard by shard into a directory.
    shards = shard_files(source)
    if shards:
        out_dir = pathlib.Path(args_cli.out).expanduser() if args_cli.out else dataset_root(source) / "data-full"
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(shard, out_dir / shard.name) for shard in shards]
    else:
        # Strip only the final ".hdf5": with_suffix() would also eat ".demo_N".
        target = (
            pathlib.Path(args_cli.out).expanduser()
            if args_cli.out
            else source.with_name(source.name[: -len(".hdf5")] + ".full.hdf5")
        )
        jobs = [(source, target)]

    if args_cli.resume and args_cli.overwrite:
        print("--resume keeps finished shards and --overwrite replaces them; pick one")
        return 1
    finished = []
    for src_path, dst_path in jobs:
        if dst_path.resolve() == src_path:
            print("Destination is the source; this script does not rewrite in place")
            return 1
        if dst_path.exists() and args_cli.resume:
            finished.append((src_path, dst_path))
        elif dst_path.exists() and not args_cli.overwrite:
            print(f"{dst_path} exists; --overwrite to replace it, or --resume to keep it")
            return 1
    if finished:
        print(f"[resume] {len(finished)} of {len(jobs)} shard(s) already restored; skipping them")
        jobs = [job for job in jobs if job not in finished]
    if not jobs:
        print("[resume] nothing left to restore")
        return 0

    if args_cli.task is None:
        args_cli.task = dataset_env_name(source)
        print(f"[task] {args_cli.task}  (from the dataset's env_args)")

    wanted = [name.strip() for name in args_cli.cameras.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in CAMERAS]
    if unknown:
        print(f"Unknown camera(s) {unknown}; pick from {sorted(CAMERAS)}")
        return 1
    selection = parse_selection(args_cli.demos)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.isaac_teleop = None
    env_cfg.recorders = None
    # Same render settings as the recording.
    configure_render(env_cfg.sim.render)
    if args_cli.width or args_cli.height:
        # Only the cameras this run renders and the scene has.
        for key in wanted:
            camera = getattr(env_cfg.scene, CAMERAS[key], None)
            if camera is None:
                continue
            camera.width = args_cli.width or camera.width
            camera.height = args_cli.height or camera.height

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    # Must run after gym.make: render products exist only once the env does.
    spp = pin_render_products()
    env.reset()
    scene = env.scene
    env_ids = torch.tensor([0], device=env.device)
    # Read back from the renderer and stored in provenance.
    rendered_with = {"samples_per_pixel": spp, **render_settings()}
    print(f"[render] readback {rendered_with}")

    missing = [key for key in wanted if CAMERAS[key] not in scene.sensors]
    if missing:
        print(f"FAIL: {missing} not in the rebuilt scene; is this the task that recorded the file?")
        env.close()
        return 1

    rendered = copied = frames = 0
    before = after = 0
    for src_path, dst_path in jobs:
        # Written as .partial and renamed when complete, so an existing
        # destination is a finished shard.
        partial = dst_path.with_name(dst_path.name + ".partial")
        with h5py.File(src_path, "r") as src, h5py.File(partial, "w") as dst, torch.inference_mode():
            demos = src.get("data")
            if demos is None or len(demos) == 0:
                print(f"{src_path} holds no demos.")
                env.close()
                return 1
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            out_group = dst.create_group("data")
            for key, value in demos.attrs.items():
                out_group.attrs[key] = value
            # restamp moves the collection session's block to
            # provenance/collected. hidden_heads=[]: no headset in this process.
            stamped = restamp(
                json.loads(out_group.attrs.get("env_args", "{}") or "{}"),
                env,
                "render_dataset_images.py",
                hidden_heads=[],
            )
            # The sample count is not readable from carb, so it is recorded here.
            stamped["provenance"].setdefault("render", {})["samples_per_pixel"] = spp
            out_group.attrs["env_args"] = json.dumps(stamped)

            for name in sorted(demos, key=lambda n: int(n.split("_")[1])):
                index = int(name.split("_")[1])
                demo = demos[name]
                out_demo = out_group.create_group(name)

                skip = ""
                if selection is not None and index not in selection:
                    skip = "not selected"
                elif "states" not in demo or int(demo.attrs.get("num_samples", 0)) == 0:
                    skip = "no recorded states, nothing to rebuild from"

                # A skipped demo keeps every image it arrived with.
                copy_states(demo, out_demo, () if skip else wanted)
                if skip:
                    if skip != "not selected":
                        print(f"{name}: {skip}")
                    copied += 1
                    continue
                frames += render_demo(name, demo, out_demo, scene, env, env_ids, wanted)
                rendered += 1
        partial.replace(dst_path)
        before += src_path.stat().st_size
        after += dst_path.stat().st_size

    where = jobs[0][1] if len(jobs) == 1 else jobs[0][1].parent
    print(f"\nrestored images for {rendered} demo(s), {frames} frames; copied {copied} without")
    print(f"\n  {source}  {human(before)}")
    print(f"  {where}  {human(after)}   ({after / max(before, 1):.0f}x larger)")
    print(f"\nCheck it with:\n  python scripts/dataset/verify_dataset.py {where}")
    env.close()
    return 0


if __name__ == "__main__":
    # simulation_app.close() never returns, so exit_gate carries the status.
    exit_gate(main())
