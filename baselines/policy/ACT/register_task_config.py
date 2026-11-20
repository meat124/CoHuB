"""Register a scripts/export/convert_lerobot.py --format act output in TASK_CONFIGS.json.

Usage (run from this directory, where imitate_episodes.py and constants.py
look for TASK_CONFIGS.json):
    python register_task_config.py \
        --bench_name cohub --ckpt_name cocarry_robot_a \
        --env_cfg_type unitree_g1x2_decentralized --action_type joint \
        --dataset_dir <cohub>/baselines/data/cocarry_act_robot_a

The registry key is "{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}".
Give each robot of a decentralized pair its own --ckpt_name, or the second
registration overwrites the first.

--val_dataset_dir registers a held-out split used for validation.
"""

import argparse
import fcntl
import json
import os
import tempfile

import h5py


def episode_files_in(dataset_dir: str) -> list:
    """Return episode_0.hdf5 .. in index order. A gap in the indices is refused."""
    files = sorted(
        (f for f in os.listdir(dataset_dir) if f.startswith("episode_") and f.endswith(".hdf5")),
        key=lambda f: int(f[len("episode_"):-len(".hdf5")]),
    )
    if not files:
        raise SystemExit(f"no episode_*.hdf5 files in {dataset_dir}")
    expected = [f"episode_{i}.hdf5" for i in range(len(files))]
    if files != expected:
        raise SystemExit(
            f"{dataset_dir} episode files aren't a contiguous 0..N-1 sequence "
            f"(EpisodicDataset indexes by range(num_episodes)): got {files}"
        )
    return files


def update_registry(config_path: str, key: str, entry: dict) -> None:
    """Set one key of the registry. A file lock and an atomic rename make concurrent registrations safe."""
    directory = os.path.dirname(os.path.abspath(config_path))
    with open(f"{config_path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(config_path, "r") as fh:
                task_configs = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            task_configs = {}
        task_configs[key] = entry

        try:
            mode = os.stat(config_path).st_mode & 0o777
        except FileNotFoundError:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".TASK_CONFIGS.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(task_configs, fh, indent=4)
            os.chmod(tmp_path, mode)
            os.replace(tmp_path, config_path)
        except BaseException:
            os.unlink(tmp_path)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench_name", required=True)
    parser.add_argument("--ckpt_name", required=True, help="Run name; combined with the others into ckpt_setting.")
    parser.add_argument("--env_cfg_type", required=True, help="Must match a key in utils/robot/_robot_info.json.")
    parser.add_argument("--action_type", required=True)
    parser.add_argument("--dataset_dir", required=True, help="A scripts/export/convert_lerobot.py --format act output dir.")
    parser.add_argument("--config_path", default="./TASK_CONFIGS.json")
    parser.add_argument("--val_dataset_dir", default=None,
                        help="A held-out split's directory; validated on instead of a random 20%% carve.")
    parser.add_argument("--setting", default=None,
                        help="The registry key, in place of the one built from the four names above.")
    args = parser.parse_args()

    episode_files = episode_files_in(args.dataset_dir)

    episode_lens = []
    camera_names = None
    for f in episode_files:
        with h5py.File(os.path.join(args.dataset_dir, f), "r") as demo:
            episode_lens.append(demo["/action"].shape[0])
            cams = sorted(demo["/observations/images"].keys())
            if camera_names is None:
                camera_names = cams
            elif cams != camera_names:
                raise SystemExit(f"{f} has cameras {cams}, earlier episodes had {camera_names} -- inconsistent dataset")

    ckpt_setting = args.setting or f"{args.bench_name}-{args.ckpt_name}-{args.env_cfg_type}-{args.action_type}"

    entry = {
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "num_episodes": len(episode_files),
        "episode_len": max(episode_lens),
        "camera_names": camera_names,
    }
    if args.val_dataset_dir:
        val_files = episode_files_in(args.val_dataset_dir)
        with h5py.File(os.path.join(args.val_dataset_dir, val_files[0]), "r") as demo:
            val_cams = sorted(demo["/observations/images"].keys())
        if val_cams != camera_names:
            raise SystemExit(f"{args.val_dataset_dir} has cameras {val_cams}, the training set {camera_names}")
        entry["val_dataset_dir"] = os.path.abspath(args.val_dataset_dir)
        entry["num_val_episodes"] = len(val_files)

    update_registry(args.config_path, ckpt_setting, entry)

    print(f"[register_task_config] {args.config_path}[{ckpt_setting!r}] = {entry}")


if __name__ == "__main__":
    main()
