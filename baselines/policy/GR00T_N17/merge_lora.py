"""Merge a LoRA finetune checkpoint into a standalone GR00T model directory.

    python baselines/policy/GR00T_N17/merge_lora.py \
        --base <GR00T-N1.7-3B snapshot> --cosmos <Cosmos-Reason2-2B snapshot> \
        --checkpoint <run>/checkpoint-20000 --out <run>/merged-20000

The processor files are copied next to the merged weights.
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

# Import gr00t from this checkout, not from wherever the venv was built.
_GR00T_ROOT = Path(__file__).resolve().parent / "gr00t_n17"
if _GR00T_ROOT.is_dir() and str(_GR00T_ROOT) not in sys.path:
    sys.path.insert(0, str(_GR00T_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model dir (config.json + safetensors)")
    ap.add_argument("--cosmos", required=True, help="local Cosmos-Reason2-2B snapshot dir")
    ap.add_argument("--checkpoint", required=True, help="PEFT checkpoint dir (checkpoint-NNNNN)")
    ap.add_argument("--out", required=True, help="output dir for the merged standalone model")
    args = ap.parse_args()

    # Registers the model with AutoConfig / AutoModel.
    import gr00t.model.gr00t_n1d7.gr00t_n1d7  # noqa: F401
    from peft import PeftModel
    from transformers import AutoModel

    ckpt = Path(args.checkpoint)
    out = Path(args.out)

    loading_kwargs = {"local_files_only": True}
    model = AutoModel.from_pretrained(
        args.base,
        model_name=args.cosmos,
        transformers_loading_kwargs=loading_kwargs,
        **loading_kwargs,
    )

    # Reference weights, to check that the merge changed them.
    ref = model.action_head.action_decoder.layer2.W.detach().float().clone()

    peft_model = PeftModel.from_pretrained(model, str(ckpt), is_trainable=False)
    merged = peft_model.merge_and_unload()

    diff = (merged.action_head.action_decoder.layer2.W.detach().float() - ref).abs().mean().item()
    print(f"[merge] action_decoder.layer2.W mean|delta| vs base: {diff:.6e}")
    if diff == 0.0:
        raise RuntimeError("merged embodiment decoder identical to base - modules_to_save not applied")

    merged = merged.to(torch.bfloat16)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)

    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        src = ckpt / name
        if src.exists():
            shutil.copy2(src, out / name)
    cfg_dir = ckpt / "experiment_cfg"
    if cfg_dir.is_dir():
        shutil.copytree(cfg_dir, out / "experiment_cfg", dirs_exist_ok=True)

    n_params = sum(p.numel() for p in merged.parameters())
    print(f"[merge] saved {n_params:,} params to {out}")


if __name__ == "__main__":
    main()
