#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
POLICIES = ("ACT", "DP", "LatentToM", "GauDP", "Pi_05", "GR00T_N17", "Psi0", "FastWAM")

BEFORE = {
    "DP": 'export CKPT_TAG=${CKPT_TAG:-${DP_RUN_TAG:-$(cohub_budget DP)}}',
    "GR00T_N17": 'if [ "${EVAL_DRY_RUN:-0}" != 1 ]; then '
                 'for task_set in $(for task in $(cohub_tasks "$1"); do cohub_task_set "$task"; done | sort -u); do '
                 'COHUB_TASK_SET=$task_set cohub_groot_merge "${*:2}"; done; fi',
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument("--task", required=True)
    parser.add_argument("--ckpt", nargs="*", default=[])
    parser.add_argument("--tag")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    ckpts = []
    run_tag = None   # the tag of the run a --ckpt path lies in
    for value in args.ckpt:
        # A path under the policy's checkpoints/ names its run: cohub-<ckpt>-<profile>-joint-<seed>[-<tag>].
        root = WT / "baselines" / "policy" / args.policy / "checkpoints"
        # absolute(), not resolve(): a run directory may be a link to where the checkpoints are kept.
        if Path(value).exists() and root in Path(value).absolute().parents:
            run = Path(value).absolute().relative_to(root).parts[0]
            named = re.search(r"-joint-\d+(?:-(.+))?$", run)
            if named:
                run_tag = named.group(1) or ""
        # The step or epoch in the file name: `epoch=0600` where the name spells it, else its first number.
        name = Path(value).name
        found = re.search(r"epoch=(\d+)", name) or re.search(r"(\d+)", name)
        if args.policy == "GauDP" and value in ("best", "last"):
            ckpts.append(value)
        elif found is None or args.policy == "GauDP":
            parser.error(f"--ckpt {value} names no checkpoint of {args.policy}")
        else:
            ckpts.append(str(int(found.group(1))))   # no leading zeros: the shell reads 000010 as octal

    env = dict(os.environ)
    if args.episodes:
        env["COHUB_EVAL_EPISODES"] = str(args.episodes)
    if args.no_video:
        env["EVAL_VIDEO"] = "0"
    tag = f'cohub_parse_args {args.policy} --tag "$TAG"' if args.tag else ":"
    if run_tag is not None:
        # The run directory's own tag, over --tag and the config: DP's carries the budget it was trained with.
        env["CKPT_TAG"] = run_tag
        tag = ":"
    env["TAG"] = args.tag or ""
    script = (f'set -euo pipefail; WT=$0; export COHUB_WT=$WT; cd "$WT"; source baselines/common.sh; '
              f'cohub_init {args.policy}; {tag}; cohub_require_sim; {BEFORE.get(args.policy, ":")}; '
              f'cohub_eval_main {args.policy} "$@"')
    sys.exit(subprocess.run(["bash", "-c", script, str(WT), args.task, *ckpts], env=env).returncode)


if __name__ == "__main__":
    main()
