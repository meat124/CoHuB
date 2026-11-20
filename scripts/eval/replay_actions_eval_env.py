"""Replay a recorded demo's actions through CoHuBTaskEnv, without a policy.

    python scripts/eval/replay_actions_eval_env.py --dataset datasets/framehang --episode 0 \\
        --env_cfg_type framehang --seed 0 --video_dir <dir>

Actions come from the LeRobot export (or ``--actions <.npy>``). Each step, the
env's joint state is compared with the recorded ``observation.state``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


from cohub.keys import action_order  # noqa: E402


PER_ROBOT = 35
"""Per robot: [joint_targets(31), height(1), base_vel(3)]."""


def pack_robots(flat: np.ndarray, robots) -> dict:
    """Split a flat action into per-robot joint_targets, height and base_vel."""
    out = {}
    for i, robot in enumerate(robots):
        a = flat[i * PER_ROBOT : (i + 1) * PER_ROBOT]
        out[robot] = {"joint_targets": a[0:31], "height": a[31:32], "base_vel": a[32:35]}
    return {"cohub_raw_action": out}


def _column(table, name: str) -> np.ndarray:
    """One LeRobot list-column as a dense ``(T, width)`` float32 array."""
    return np.stack([np.asarray(row.as_py(), dtype=np.float32) for row in table.column(name)])


def _episode_table(root: Path, episode: int):
    info = json.loads((root / "meta" / "info.json").read_text())
    path = root / info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode)
    if not path.exists():
        raise SystemExit(f"no episode {episode} in {root}: {path} is missing")
    return info, pq.read_table(path)


def episode_state(root: Path, episode: int) -> np.ndarray:
    """The recorded ``(T, 43 * robots)`` joint state, in the dataset's order."""
    return _column(_episode_table(root, episode)[1], "observation.state")


def episode_actions(root: Path, episode: int) -> np.ndarray:
    """The ``(T, 35 * robots)`` stream: ``[joint_targets(31), height(1), base_vel(3)]`` per robot."""
    info, table = _episode_table(root, episode)
    names = info["features"]["action"]["names"]
    robots: list[str] = []          # dataset order
    for name in names:
        prefix = name.split("/", 1)[0]
        if prefix not in robots:
            robots.append(prefix)
    action = _column(table, "action")
    height = _column(table, "teleop.base_height_command")
    navigate = _column(table, "teleop.navigate_command")
    index = {name: i for i, name in enumerate(names)}
    out = []
    for r, robot in enumerate(robots):
        try:
            cols = [index[f"{robot}/{joint}"] for joint in action_order()]
        except KeyError as missing:
            raise SystemExit(f"{root}'s action has no column {missing} -- this dataset was not "
                             "recorded from a G1 whose upper body is cohub.keys") from None
        out.append(action[:, cols])
        out.append(height[:, r : r + 1])
        out.append(navigate[:, 3 * r : 3 * r + 3])
    return np.concatenate(out, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="a LeRobot root, e.g. datasets/framehang")
    ap.add_argument("--episode", type=int, default=0, help="which episode of --dataset to replay")
    ap.add_argument("--actions", default=None, help="instead of --dataset: .npy of shape (T, 35 * robots)")
    ap.add_argument("--state", default=None,
                    help="with --actions: .npy (T, 43 * robots) the demo recorded, to compare against")
    ap.add_argument("--env_cfg_type", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="0 = the whole action file")
    ap.add_argument("--video_dir", default=None)
    ap.add_argument("--video_cameras", default="scene",
                    help="Comma-separated cameras to film, e.g. scene,scene_2 on MoveHouse.")
    ap.add_argument("--scene_camera", default="fixed", choices=("fixed", "follow"),
                    help="fixed: the dataset's pose; follow: eval's default shot.")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if (args.dataset is None) == (args.actions is None):
        ap.error("give exactly one of --dataset or --actions")
    if args.dataset:
        actions = episode_actions(Path(args.dataset), args.episode)
        demo_state = episode_state(Path(args.dataset), args.episode)
    else:
        actions = np.load(args.actions).astype(np.float32)
        demo_state = np.load(args.state).astype(np.float32) if args.state else None
    n = args.steps or len(actions)

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, device=args.device).app

    from cohub.policy_env import CoHuBTaskEnv, load_env_cfg_type

    env = CoHuBTaskEnv(
        gym_id=load_env_cfg_type(args.env_cfg_type)["gym_id"],
        device=args.device,
        episode_step_limit=n + 10,
        video_dir=args.video_dir,
        video_cameras=[c for c in args.video_cameras.split(",") if c] if args.video_dir else [],
        obs_cameras=[],
        obs_stride=1,
        video_stride=1,
        scene_camera=args.scene_camera,
        ego_depth=False,
        drop_unused_cameras=False,
    )
    expected_width = PER_ROBOT * len(env.robots)
    if actions.shape[-1] != expected_width:
        raise SystemExit(
            f"--actions is {actions.shape[-1]}D but {env.gym_id} has {len(env.robots)} "
            f"robot(s) ({', '.join(env.robots)}), i.e. {expected_width}D "
            f"[joint_targets(31), height(1), base_vel(3)] each"
        )
    env.reset(seed=args.seed)

    observed = []
    try:
        for step in range(n):
            obs = env.get_obs()
            observed.append(
                np.concatenate([
                    obs["cohub_state"][robot]["joint_pos"] for robot in env.robots
                ])
            )
            env.take_action(pack_robots(actions[step], env.robots))
            if env.is_episode_end():
                print(f"[replay] episode ended at step {step + 1}")
                break
        result = env.episode_result()
        observed = np.stack(observed)
        print(f"[replay] steps {len(observed)} reason {result.get('reason')} "
              f"subtasks {result.get('subtasks')}")
        if demo_state is not None:
            m = min(len(observed), len(demo_state))
            err = np.abs(observed[:m] - demo_state[:m])
            print(f"[replay] |env state - demo state|: mean {err.mean():.5f} "
                  f"max {err.max():.5f}")
            for cut in (1, 10, 50, 100, 200, 400):
                if cut < m:
                    print(f"    step {cut:4d}: mean {err[cut].mean():.5f} max {err[cut].max():.5f}")
        if args.out_json:
            Path(args.out_json).write_text(json.dumps({
                "steps": int(len(observed)),
                "result": result,
                "state_err_mean": float(np.abs(observed[:m] - demo_state[:m]).mean())
                if demo_state is not None else None,
            }, indent=2, default=str))
        if args.video_dir:
            env.finish_episode()
    finally:
        env.close()
        app.update()
        app.close()


if __name__ == "__main__":
    main()
