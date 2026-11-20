# pi0.5

pi0.5, fine-tuned with LoRA from the `pi05_base` weights. One policy is trained on all tasks and shared by the humanoids: each humanoid runs it on its own egocentric view, joint state and instruction.

Run the commands below from this folder.

```bash
cd baselines/policy/Pi_05
```

## Install

```bash
bash install.sh
```

It builds a uv environment in `openpi/.venv`, so [uv](https://docs.astral.sh/uv/) is required. The pretrained weights (12 GB) are downloaded by the first training run.

## Train

```bash
bash process_data.sh
bash train.sh
```

`process_data.sh` downloads the dataset from the Hugging Face Hub when it is missing, joins the eight tasks with two humanoids into one dataset, and computes the normalization statistics. It runs in the simulation environment of the [main README](../../../README.md#installation). The dataset is written to `HF_LEROBOT_HOME`, `~/.cache/huggingface/lerobot` by default; a fast local disk is recommended.

`train.sh` trains for 40,000 steps at batch size 32 and keeps a checkpoint every 10,000 steps under `checkpoints/`. It needs a 48 GB GPU, and an interrupted run resumes from its latest checkpoint.

The two tasks with three humanoids are trained as a separate policy. Add `all3` to both commands.

```bash
bash process_data.sh all3
bash train.sh all3
```

The settings are the lines of [`baselines/configs/Pi_05.yaml`](../../configs/Pi_05.yaml). To change one, set it on the command line, and add `--tag` so that the run is saved under its own name.

```bash
PI05_PEAK_LR=5e-5 bash train.sh --tag lr5e-5
```

| Setting | Default | Meaning |
|---|---|---|
| `PI05_NUM_TRAIN_STEPS` | `40000` | training steps |
| `PI05_BATCH_SIZE` | `32` | batch size |
| `PI05_PEAK_LR` | `2.5e-5` | learning rate after the warm-up |
| `PI05_DECAY_LR` | `2.5e-6` | learning rate at the end of the cosine schedule |
| `PI05_KEEP_PERIOD` | `10000` | steps between the checkpoints that are kept |
| `SEED` | `0` | training seed |

## Evaluate

Run the evaluation from the repository root, as described in the [main README](../../../README.md#evaluation). The checkpoints of steps 20,000 and 40,000 are evaluated unless `--ckpt` names others.

```bash
python scripts/eval/eval.py --policy Pi_05 --task cocarry
```
