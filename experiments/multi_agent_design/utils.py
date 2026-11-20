from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from design_config import BASE, BATCH, HERE, WT


def sh(script: str, *args: str) -> str:
    done = subprocess.run(["bash", "-c", f'WT=$0; source "$WT/baselines/common.sh"; {script}', str(WT), *args],
                          stdout=subprocess.PIPE, text=True)
    if done.returncode:
        sys.exit(done.returncode)
    return done.stdout.strip()


def say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def step(*cmd, env: dict | None = None, cwd: Path | None = None, always: bool = False) -> None:
    say("[cohub] $ " + " ".join(str(c) for c in cmd))
    if always or os.environ.get("COHUB_DRY_RUN") != "1":
        code = subprocess.run([str(c) for c in cmd], env={**os.environ, **(env or {})}, cwd=cwd).returncode
        if code:
            sys.exit(code)


def load_settings(path: Path) -> None:
    for line in path.read_text().splitlines():
        key, colon, value = line.split("#", 1)[0].partition(":")
        if colon:
            os.environ.setdefault(key.strip(), value.strip())


def setup(tagged: bool = True, ckpts: bool = False, seed: str = "SEED") -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=sorted(BATCH))
    parser.add_argument("--view", required=True, choices=("local", "global"))
    parser.add_argument("--sharing", required=True, choices=("separate", "shared"))
    parser.add_argument("--task")
    if tagged:
        parser.add_argument("--tag")
    if ckpts:
        parser.add_argument("--ckpt", nargs="*", default=[])
        parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    args.cell = f"{args.view}-{args.sharing}"
    tag = (args.tag if tagged else None) or args.cell
    if args.cell not in tag:
        sys.exit(f"the {args.cell} cell needs a tag that names it: --tag {args.cell}")
    if args.task is None and (ckpts or args.policy != "GR00T_N17"):
        parser.error("--task is required")
    if args.task is not None and not ckpts and args.policy == "GR00T_N17":
        parser.error("GR00T N1.7 trains one policy on all eight tasks and takes no --task")
    if args.task is not None:
        args.task = sh('mh_task_word "$1"', args.task)
        if "robot_c" in sh('mh_robots "$(mh_scene "$1")"', args.task).split():
            sys.exit(f"the {args.cell} cell is defined for two robots; {args.task} has three")

    os.chdir(WT)
    os.environ.update(COHUB_WT=str(WT), COHUB_OBS_VIEW=args.view, COHUB_SHARING=args.sharing)
    load_settings(HERE / "policy" / "configs" / f"{args.policy}.yaml")
    load_settings(BASE / "configs" / f"{args.policy}.yaml")
    os.environ["CKPT_TAG"] = tag
    if args.policy == "DP":
        os.environ["DP_RUN_TAG"] = tag
    seed = os.environ.get(seed, "0")
    os.environ["COHUB_RUN_SUFFIX"] = os.environ["CKPT_TAG"] + (f"-seed{seed}" if seed != "0" else "")
    return args
