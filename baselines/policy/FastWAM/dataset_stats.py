"""Compute FastWAM's dataset_stats.json for a converted CoHuB dataset.

Runs FastWAM's own stats computation with the hydra task config training uses.
Run inside the FastWAM conda env (CPU is fine):

    <fastwam env>/bin/python baselines/policy/FastWAM/dataset_stats.py \\
        --task cohub_uncond_2cam_384_1e-4 \\
        --train-dir baselines/policy/FastWAM/data/<id>/lerobot \\
        --out       baselines/policy/FastWAM/data/<id>/dataset_stats.json \\
        --action-dim 70 --state-dim 86

The dims must match what train.sh uses for the same env_cfg_type.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent
FASTWAM_DIR = POLICY_DIR / "FastWAM"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="FastWAM task yaml name (configs/task/<name>.yaml)")
    parser.add_argument("--train-dir", required=True, help="the converted training split (…/lerobot)")
    parser.add_argument("--out", required=True, help="where dataset_stats.json goes")
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--state-dim", type=int, required=True)
    args = parser.parse_args()

    for path in (str(FASTWAM_DIR), str(FASTWAM_DIR / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json

    train_dir = str(Path(args.train_dir).resolve())
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str((FASTWAM_DIR / "configs").resolve())):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"task={args.task}",
                f"data.train.dataset_dirs=[{train_dir}]",
                "data.train.pretrained_norm_stats=null",
                f"data.train.shape_meta.action.0.raw_shape={args.action_dim}",
                f"data.train.shape_meta.action.0.shape={args.action_dim}",
                f"data.train.shape_meta.state.0.raw_shape={args.state_dim}",
                f"data.train.shape_meta.state.0.shape={args.state_dim}",
                f"data.train.processor.action_output_dim={args.action_dim}",
                f"data.train.processor.proprio_output_dim={args.state_dim}",
            ],
        )

    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    processor_cfg = ds_cfg.processor
    # Without a processor the dataset does not write stats into hydra's work dir.
    ds_cfg.processor = None
    dataset = instantiate(ds_cfg)
    processor = instantiate(processor_cfg)
    if hasattr(processor, "action_state_merger") and hasattr(processor.action_state_merger, "set_shape_meta"):
        processor.action_state_merger.set_shape_meta(processor.shape_meta)

    stats = dataset.lerobot_dataset.get_dataset_stats(processor)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_stats_to_json(stats, str(out))
    print(f"[fastwam-stats] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
