#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
POLICIES = ("ACT", "DP", "LatentToM", "GauDP", "Pi_05", "GR00T_N17", "Psi0", "FastWAM")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument("--task", required=True, choices=sorted(p.stem for p in (HERE / "phase_starts" / "states").glob("*.json")))
    parser.add_argument("--phase", nargs="*", default=[])
    parser.add_argument("--ckpt")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    run = args.policy
    if args.ckpt:
        found = re.search(r"\d+", Path(args.ckpt).name)
        if found is None:
            parser.error(f"--ckpt {args.ckpt} names no epoch or step")
        ckpt = found.group(0)
        run = f"{args.policy}-ckpt{ckpt}"
    else:
        config = (WT / "baselines" / "configs" / f"{args.policy}.yaml").read_text()
        default = re.search(r"^COHUB_EVAL_CKPTS:([^#\n]*)", config, re.M)
        ckpt = (os.environ.get("COHUB_EVAL_CKPTS") or (default.group(1) if default else "")).split()[-1:]
        ckpt = ckpt[0] if ckpt else None

    env = {**os.environ, "COHUB_EVAL_CLIENT": str(HERE / "client.py"), "COHUB_EVAL_LABEL": "phase", "EVAL_RUN_ID": run}
    if args.phase:
        env["COHUB_EVAL_CLIENT_ARGS"] = "--phase " + ",".join(args.phase)
    if args.no_video:
        env["EVAL_VIDEO"] = "0"
    command = [sys.executable, str(WT / "scripts" / "eval" / "eval.py"), "--policy", args.policy, "--task", args.task]
    sys.exit(subprocess.run(command + (["--ckpt", ckpt] if ckpt else []), env=env, cwd=WT).returncode)


if __name__ == "__main__":
    main()
