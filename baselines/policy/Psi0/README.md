# Psi0

Psi0, fine-tuned with LoRA from its pretrained weights. One policy is trained on all tasks and shared by the humanoids: each humanoid runs it on its own egocentric view, joint state and instruction.

Run the commands below from this folder.

```bash
cd baselines/policy/Psi0
```

## Install

```bash
bash install.sh
```

It builds a uv environment in `psi0/.venv` (Python 3.11, torch 2.7.0) and downloads the base model (4.3 GB) and the action header (2 GB). [uv](https://docs.astral.sh/uv/) is required.

## Train

```bash
bash process_data.sh
bash train.sh
```

`process_data.sh` downloads the dataset from the Hugging Face Hub when it is missing, joins the eight tasks with two humanoids into one dataset, and reorders the state and the action into Psi0's 36-D layout (`psi0_layout.py`). It runs in the simulation environment of the [main README](../../../README.md#installation).

`train.sh` trains for 40,000 steps at batch size 32 and saves a checkpoint every 10,000 steps under `checkpoints/`. LoRA is applied to the language model and the action expert, and the vision tower is trained in full. It took about 8.5 hours on one 48 GB GPU, and an interrupted run resumes from its latest checkpoint.

The two tasks with three humanoids are trained as a separate policy. Add `all3` to both commands.

```bash
bash process_data.sh all3
bash train.sh all3
```

The settings are the lines of [`baselines/configs/Psi0.yaml`](../../configs/Psi0.yaml). To change one, set it on the command line, and add `--tag` so that the run is saved under its own name.

```bash
PSI0_LR=5e-5 bash train.sh --tag lr5e-5
```

| Setting | Default | Meaning |
|---|---|---|
| `MAX_STEPS` | `40000` | training steps |
| `GLOBAL_BATCH_SIZE` | `32` | batch size per optimizer step |
| `PSI0_MICRO_BATCH` | `16` | samples per forward pass; gradient accumulation makes up the rest |
| `SAVE_STEPS` | `10000` | steps between checkpoints |
| `PSI0_LR` | `2.5e-5` | learning rate after the warm-up |
| `SEED` | `0` | training seed |

The changes made to the vendored Psi0 code are listed in `psi0/UPSTREAM.md`.

## Evaluate

Run the evaluation from the repository root, as described in the [main README](../../../README.md#evaluation). The checkpoints of steps 20,000 and 40,000 are evaluated unless `--ckpt` names others.

```bash
python scripts/eval/eval.py --policy Psi0 --task cocarry
```
