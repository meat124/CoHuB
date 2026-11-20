#!/usr/bin/env python3
"""Fetch the pretrained weights a policy starts from. Does nothing when they are present.

    python baselines/utils/download_weights.py GR00T_N17
    python baselines/utils/download_weights.py Psi0
    python baselines/utils/download_weights.py FastWAM

Run it with the policy's interpreter (needs `huggingface_hub`).
`baselines/policy/<POLICY>/install.sh` does that. Destinations (baselines/common.sh
sets the defaults, site.env overrides):

  GR00T_N17  GR00T_BASE_MODEL    <policy>/GR00T_N17/weights/GR00T-N1.7-3B       nvidia/GR00T-N1.7-3B
             GR00T_COSMOS_MODEL  <policy>/GR00T_N17/weights/Cosmos-Reason2-2B    nvidia/Cosmos-Reason2-2B
  Psi0       PSI0_BASE_MODEL     <policy>/Psi0/weights/psi0/pre.fast.1by1...     USC-PSI-Lab/psi-model
             PSI0_ACTION_HEADER  <policy>/Psi0/weights/psi0/postpre.1by1...      USC-PSI-Lab/psi-model
  FastWAM    DIFFSYNTH_MODEL_BASE_PATH  <policy>/FastWAM/FastWAM/checkpoints     Wan-AI/Wan2.2-TI2V-5B,
                                                                               Wan-AI/Wan2.1-T2V-1.3B

<policy> = baselines/policy. pi0.5 is not here: openpi fetches
pi05_base itself. FastWAM's ActionDiT backbone is derived from the Wan weights
by `baselines/policy/FastWAM/install.sh`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = HERE.parent / "policy"

PSI0_VLM = "psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
PSI0_HEADER = "psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"


def where(var: str, default: Path) -> Path:
    return Path(os.environ.get(var) or default).expanduser()


def wan_root() -> Path:
    return where("DIFFSYNTH_MODEL_BASE_PATH", POLICY / "FastWAM" / "FastWAM" / "checkpoints")


# Each entry: (repo_id, allow_patterns or None, repo subdirectory to keep or
# None, destination, files whose presence means "done").
def jobs(policy: str) -> list[tuple[str, list[str] | None, str | None, Path, list[str]]]:
    if policy == "GR00T_N17":
        base = where("GR00T_BASE_MODEL", POLICY / "GR00T_N17" / "weights" / "GR00T-N1.7-3B")
        cosmos = where("GR00T_COSMOS_MODEL", POLICY / "GR00T_N17" / "weights" / "Cosmos-Reason2-2B")
        return [
            ("nvidia/GR00T-N1.7-3B", None, None, base, ["config.json", "model.safetensors.index.json"]),
            ("nvidia/Cosmos-Reason2-2B", None, None, cosmos, ["config.json", "model.safetensors"]),
        ]
    if policy == "Psi0":
        weights = POLICY / "Psi0" / "weights"
        vlm = where("PSI0_BASE_MODEL", weights / PSI0_VLM)
        header = where("PSI0_ACTION_HEADER", weights / PSI0_HEADER)
        return [
            ("USC-PSI-Lab/psi-model", [f"{PSI0_VLM}/*"], PSI0_VLM, vlm, ["config.json", "model.safetensors"]),
            ("USC-PSI-Lab/psi-model", [f"{PSI0_HEADER}/*"], PSI0_HEADER, header, ["action_header.safetensors"]),
        ]
    if policy == "FastWAM":
        root = wan_root()
        return [
            ("Wan-AI/Wan2.2-TI2V-5B",
             ["diffusion_pytorch_model*.safetensors", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2_VAE.pth"],
             None, root / "Wan-AI" / "Wan2.2-TI2V-5B",
             ["diffusion_pytorch_model-00001-of-00003.safetensors", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2_VAE.pth"]),
            ("Wan-AI/Wan2.1-T2V-1.3B", ["google/umt5-xxl/*"], None, root / "Wan-AI" / "Wan2.1-T2V-1.3B",
             ["google/umt5-xxl/tokenizer_config.json"]),
        ]
    raise SystemExit(f"no weights to fetch for {policy}: one of GR00T_N17, Psi0, FastWAM")


def fetch(repo: str, patterns: list[str] | None, subdir: str | None, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    ignore = ["*.png"]  # the model cards' figures
    if subdir is None:
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, allow_patterns=patterns, ignore_patterns=ignore, local_dir=str(dest))
        return
    # Stage the snapshot beside the destination, then move the subdirectory.
    staging = dest.parent / f".hf_staging_{dest.name}"
    staging.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, allow_patterns=patterns, ignore_patterns=ignore, local_dir=str(staging))
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(staging / subdir), str(dest))
    shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", choices=["GR00T_N17", "Psi0", "FastWAM"])
    args = ap.parse_args()

    todo = [(r, p, s, d, m) for r, p, s, d, m in jobs(args.policy)
            if not all((d / f).exists() for f in m)]
    if not todo:
        print(f"[weights] {args.policy}: present", flush=True)
        return 0
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub is not importable here -- run this with the policy env's python", file=sys.stderr)
        return 2

    failed = []
    for repo, patterns, subdir, dest, _ in todo:
        t0 = time.time()
        print(f"[weights] {repo}{'/' + subdir if subdir else ''} -> {dest}", flush=True)
        # Large transfers can drop. snapshot_download resumes on retry.
        for attempt in range(1, 4):
            try:
                fetch(repo, patterns, subdir, dest)
                print(f"[weights] {repo} done in {time.time() - t0:.0f}s", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[weights] {repo} attempt {attempt}/3 failed: {exc}", flush=True)
                time.sleep(30 * attempt)
        else:
            failed.append(repo)
    print(f"[weights] {args.policy}: failed={failed or 'none'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
