#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from design_config import BASE, BATCH, BUILT, CONVERT, GROOT, MODALITY, POLICY_ROOT, WT, converted, env_cfg, runs, train_data
from utils import say, sh, step

CONVERTERS = Path(__file__).resolve().parent / "converters"


def data(policy: str, cell: str, task: str) -> None:
    sim = sh("cohub_sim_python")
    if policy == "GR00T_N17":
        if cell == "local-shared":
            step("bash", POLICY_ROOT / "GR00T_N17" / "process_data.sh", always=True)
        else:
            step("bash", BASE / "utils" / "prepare_data.sh", "fetch")
            data_groot(cell, sim)
    elif cell == "local-separate":
        step("bash", POLICY_ROOT / policy / "process_data.sh", task, always=True)
    else:
        step("bash", BASE / "utils" / "prepare_data.sh", "fetch")
        data_single(policy, cell, task, sim)
    say(f"[data] {policy} {cell} {task} done")


def data_single(policy: str, cell: str, task: str, sim: str) -> None:
    source = WT / "datasets" / task / "lerobot"
    python = sim
    if policy == "DP":
        python = sh("cohub_policy_python DP")
        if not (source / "meta" / "info.json").is_file():
            source = source.parent
        if not os.access(python, os.X_OK) and os.environ.get("COHUB_DRY_RUN") != "1":
            sys.exit(f"no DP python at {python}: install DP, or set COHUB_DP_PY")
    script, flags = CONVERT[cell]
    script = CONVERTERS / script if script == "convert.py" else WT / "scripts" / "export" / script
    fmt = {"ACT": "act", "DP": "diffusion_policy"}[policy]
    source_path = os.pathsep.join(filter(None, [str(WT / "source"), os.environ.get("PYTHONPATH")]))
    for split in ("train", "val"):
        step(python, script, "--source-path", source, "--output-path", BUILT / cell, "--format", fmt, *flags,
             "--split", split, "--overwrite", env={"PYTHONPATH": source_path})
    for mode, robot, ckpt in runs(policy, cell, task):
        built = converted(policy, cell, task, robot)
        if policy == "ACT":
            step(sim, POLICY_ROOT / "ACT" / "register_task_config.py", "--bench_name", "cohub", "--ckpt_name", ckpt,
                 "--env_cfg_type", env_cfg(mode), "--action_type", "joint",
                 "--dataset_dir", built, "--val_dataset_dir", f"{built}_val",
                 "--config_path", POLICY_ROOT / "ACT" / "TASK_CONFIGS.json", "--setting", f"cohub-{ckpt}-{cell}")


def data_groot(cell: str, sim: str) -> None:
    target, flags = GROOT[cell]
    tasks = sh('echo "$MH_ALL_TASKS"').replace(" ", ",")
    python = sh("cohub_policy_python GR00T_N17")
    for mode, robot, ckpt in runs("GR00T_N17", cell, ""):
        out = train_data("GR00T_N17", cell, "", mode, robot, ckpt)
        if (out / "meta" / "info.json").is_file() and os.environ.get("MULTITASK_REBUILD") != "1":
            print(f"  reusing {out.relative_to(WT)} (MULTITASK_REBUILD=1 to rebuild)", flush=True)
        else:
            step(sim, CONVERTERS / "build_multitask.py", "--split", "all", "--out", out, "--tasks", tasks,
                 "--overwrite", *(flag.format(robot=robot) for flag in flags))
        if os.access(python, os.X_OK):
            step(python, "gr00t/data/stats.py", "--dataset-path", out, "--embodiment-tag", "NEW_EMBODIMENT",
                 "--modality-config-path", MODALITY, cwd=POLICY_ROOT / "GR00T_N17" / "gr00t_n17",
                 env={"COHUB_ROBOT": target, "COHUB_ACTION_HORIZON": os.environ.get("COHUB_ACTION_HORIZON", "40")})
        else:
            say(f"  [note] no GR00T python at {python} -- training will generate meta/stats.json itself")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", nargs="*", default=sorted(BATCH), choices=sorted(BATCH))
    parser.add_argument("--view", nargs="*", default=["local", "global"], choices=("local", "global"))
    parser.add_argument("--sharing", nargs="*", default=["separate", "shared"], choices=("separate", "shared"))
    parser.add_argument("--task", nargs="*")
    args = parser.parse_args()
    tasks = [sh('mh_task_word "$1"', task) for task in args.task or sh('echo "$MH_ALL_TASKS"').split()]

    os.chdir(WT)
    os.environ["COHUB_WT"] = str(WT)
    for policy in args.policy:
        for cell in (f"{view}-{sharing}" for view in args.view for sharing in args.sharing):
            for task in tasks[:1] if policy == "GR00T_N17" else tasks:
                data(policy, cell, task)


if __name__ == "__main__":
    main()
