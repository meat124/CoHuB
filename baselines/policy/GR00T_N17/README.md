# GR00T N1.7

GR00T N1.7, fine-tuned with LoRA from the N1.7 base model. One policy is trained on all tasks and shared by the humanoids: each humanoid runs it on its own egocentric view, joint state and instruction.

Run the commands below from this folder.

```bash
cd baselines/policy/GR00T_N17
```

## Install

```bash
bash install.sh
```

It builds a uv environment in `gr00t_n17/.venv` and downloads the base model (6.5 GB) and the Cosmos-Reason2 backbone (4.6 GB). [uv](https://docs.astral.sh/uv/) and `ffmpeg` are required. If it stops with `CUDA_HOME is unset`, point `CUDA_HOME` at the CUDA toolkit.

## Train

```bash
bash process_data.sh
bash train.sh
```

`process_data.sh` downloads the dataset from the Hugging Face Hub when it is missing, joins the eight tasks with two humanoids into one dataset, and computes its statistics, in about ten minutes. It runs in the simulation environment of the [main README](../../../README.md#installation).

`train.sh` trains for 40,000 steps at batch size 32 and saves a checkpoint every 10,000 steps under `checkpoints/`. LoRA is applied to the action expert and the language backbone, and the vision tower is trained in full, which needs a 48 GB GPU. `TUNE_VISUAL=0` freezes the vision tower and fits a 24 GB GPU. An interrupted run resumes from its latest checkpoint.

The two tasks with three humanoids are trained as a separate policy. Add `all3` to both commands.

```bash
bash process_data.sh all3
bash train.sh all3
```

The settings are the lines of [`baselines/configs/GR00T_N17.yaml`](../../configs/GR00T_N17.yaml). To change one, set it on the command line, and add `--tag` so that the run is saved under its own name.

```bash
GR00T_LR=5e-5 bash train.sh --tag lr5e-5
```

| Setting | Default | Meaning |
|---|---|---|
| `MAX_STEPS` | `40000` | training steps |
| `GLOBAL_BATCH_SIZE` | `32` | batch size over all GPUs |
| `SAVE_STEPS` | `10000` | steps between checkpoints |
| `GR00T_LR` | `2.5e-5` | learning rate after the warm-up |
| `LORA_RANK` | `32` | LoRA rank of the action expert |
| `TUNE_VISUAL` | `1` | `0` freezes the vision tower |
| `SEED` | `0` | training seed |

## Evaluate

Run the evaluation from the repository root, as described in the [main README](../../../README.md#evaluation). The checkpoints of steps 20,000 and 40,000 are evaluated unless `--ckpt` names others. Each checkpoint is first merged with the base model, which needs a GPU and about 160 GB of host memory.

```bash
python scripts/eval/eval.py --policy GR00T_N17 --task cocarry
```
