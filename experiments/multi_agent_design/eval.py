#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from design_config import HERE, POLICY_ROOT, ROBOTS, WT, run_name, runs
from utils import say, setup, sh, step


SAVED = {"ACT": r"policy_epoch_(\d+)_", "DP": r"^(\d+)\.ckpt$", "GR00T_N17": r"^(?:checkpoint|merged)-(\d+)$"}


def saved(policy: str, directory: Path) -> list[int]:
    found = (re.search(SAVED[policy], path.name) for path in directory.glob("*"))
    return sorted({int(match.group(1)) for match in found if match})


def checkpoints(policy: str, cell: str, task: str, given: list[str]) -> list[str]:
    folders = []
    for _, _, ckpt in runs(policy, cell, task):
        name = run_name(ckpt)
        folders.append(POLICY_ROOT / policy / "checkpoints" / name / (name if policy == "GR00T_N17" else ""))
    if all(value.isdigit() for value in given) and given:
        return given
    if not given:
        return [str(number) for number in saved(policy, folders[0])[-1:]] or os.environ.get("COHUB_EVAL_CKPTS", "").split()
    paths = [Path(value).resolve() for value in given]
    expected = "\n".join(f"  {folder}" for folder in folders)
    if sorted(path.parent for path in paths) != sorted(folder.resolve() for folder in folders):
        sys.exit(f"--ckpt takes one checkpoint from each of\n{expected}")
    points = {match.group(1) for path in paths if (match := re.search(SAVED[policy], path.name))}
    if len(points) != 1:
        sys.exit("--ckpt takes checkpoints saved at the same point of training, one per humanoid")
    return sorted(points)


def merge(cell: str, ckpts: list[str]) -> None:
    if cell.endswith("-shared"):
        step("bash", POLICY_ROOT / "GR00T_N17" / "merge_lora.sh", "all", "multitask", " ".join(ckpts), "0")
        return
    python = sh("cohub_policy_python GR00T_N17")
    base, cosmos = sh('echo "$GR00T_BASE_MODEL"; echo "$GR00T_COSMOS_MODEL"').splitlines()
    for _, _, ckpt in runs("GR00T_N17", cell, ""):
        name = run_name(ckpt)
        run = POLICY_ROOT / "GR00T_N17" / "checkpoints" / name / name
        newest = sorted((int(p.name.split("-")[1]) for p in run.glob("checkpoint-*")), reverse=True)[:1]
        for number in ckpts or newest:
            if list((run / f"merged-{number}").glob("*.safetensors")):
                print(f"[merge] {name}: merged-{number} exists, skipping", flush=True)
            elif not (run / f"checkpoint-{number}").is_dir():
                sys.exit(f"[merge] {name}: no checkpoint-{number} under {run}")
            elif not (run / f"checkpoint-{number}" / "adapter_config.json").is_file():
                print(f"[merge] {name}: checkpoint-{number} is a full model already", flush=True)
            else:
                step(python, POLICY_ROOT / "GR00T_N17" / "merge_lora.py", "--base", base, "--cosmos", cosmos,
                     "--checkpoint", run / f"checkpoint-{number}", "--out", run / f"merged-{number}")


def evaluate(policy: str, cell: str, task: str, ckpts: list[str], no_video: bool) -> None:
    ckpts = checkpoints(policy, cell, task, ckpts)
    sim = sh("cohub_sim_python")
    if not os.access(sim, os.X_OK) and os.environ.get("COHUB_DRY_RUN") != "1":
        sys.exit(f"no simulation python at {sim}: see README, Installation (or set COHUB_SIM_PY)")
    if policy == "GR00T_N17" and os.environ.get("EVAL_DRY_RUN") != "1":
        merge(cell, ckpts)
    view, sharing = cell.split("-")
    env = {
        "COHUB_CKPT_NAME": "multitask" if policy == "GR00T_N17" else task,
        "COHUB_UPLOAD_PY": str(HERE / "wandb_upload.py"),
        "EVAL_PYTHONPATH": str(HERE.parent),
        "EVAL_PROVENANCE_EXTRA": f"obs_view={view} sharing={sharing} experiment=multi_agent_design",
        "EVAL_SERVER_OVERRIDES": " ".join(filter(None, [
            os.environ.get("EVAL_SERVER_OVERRIDES"), f"policy_module=multi_agent_design.policy.{policy}.model",
            f"cohub_obs_view={view}", f"cohub_sharing={sharing}"])),
    }
    if policy == "GR00T_N17" and cell.endswith("-separate"):
        env["COHUB_DECENTRALIZED_STYLE"] = "per_robot"
    if policy != "GR00T_N17" and cell == "global-shared":
        env["COHUB_SERVE_MODE"] = "centralized"
    if policy != "GR00T_N17" and cell == "local-shared":
        shared = run_name(f"{task}_shared")
        env["COHUB_SERVE_RUNS"] = "\n".join(f"{shared}:model_dir_{robot}" for robot in ROBOTS)
    if no_video:
        env["EVAL_VIDEO"] = "0"
    episodes = os.environ.get("COHUB_EVAL_EPISODES", "100")
    seed = os.environ.get("COHUB_EVAL_SEED", "1")
    for ckpt in ckpts or [""]:
        say(f"[eval] {policy} {task}{f' ckpt {ckpt}' if ckpt else ''} cell={cell}  ({episodes} episodes, seed {seed})")
        step("bash", WT / "scripts" / "eval" / "runner.sh", policy, task, episodes, seed, env={**env, "EVAL_CKPT_NUM": ckpt})


if __name__ == "__main__":
    args = setup(ckpts=True, seed="CKPT_SEED")
    evaluate(args.policy, args.cell, args.task, args.ckpt, args.no_video)
