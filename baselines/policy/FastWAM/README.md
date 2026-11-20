# Fast-WAM

Fast-WAM, a world-action model fine-tuned with LoRA from the Wan2.2 video model. One policy is trained on all tasks and shared by the humanoids: each humanoid runs it on its own egocentric view, joint state and instruction.

Run the commands below from this folder.

```bash
cd baselines/policy/FastWAM
```

## Install

```bash
bash install.sh
```

It builds the conda environment `fastwam` (Python 3.10, torch 2.7.1), downloads the Wan2.2 weights (about 34 GB) and derives the action backbone from them on the CPU. `conda` is required.

## Train

```bash
bash process_data.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train.sh
```

`process_data.sh` downloads the dataset from the Hugging Face Hub when it is missing, joins the eight tasks with two humanoids into one dataset, computes its statistics and encodes every instruction once with the T5 text encoder. It runs in the simulation environment of the [main README](../../../README.md#installation).

`train.sh` trains for 40,000 steps at batch size 32 on the GPUs in `CUDA_VISIBLE_DEVICES`, whose number must divide 32, and saves the checkpoints under `checkpoints/`. Four 48 GB GPUs are enough for the default LoRA setting.

The two tasks with three humanoids are trained as a separate policy. Add `all3` to both commands.

```bash
bash process_data.sh all3
bash train.sh all3
```

The settings are the lines of [`baselines/configs/FastWAM.yaml`](../../configs/FastWAM.yaml). To change one, set it on the command line, and add `--tag` so that the run is saved under its own name.

```bash
FASTWAM_LORA_RANK=64 FASTWAM_LORA_ALPHA=32 bash train.sh --tag r64
```

| Setting | Default | Meaning |
|---|---|---|
| `FASTWAM_MAX_STEPS` | `40000` | training steps |
| `FASTWAM_GLOBAL_BATCH` | `32` | batch size per optimizer step |
| `FASTWAM_LORA_RANK` | `128` | LoRA rank of the video model; `0` fine-tunes it in full |
| `FASTWAM_LORA_ALPHA` | `64` | LoRA alpha |
| `SEED` | `0` | training seed |

The learning rate, 1e-4, is set in `FastWAM/configs/task/`.

## Evaluate

Run the evaluation from the repository root, as described in the [main README](../../../README.md#evaluation). The checkpoint of step 40,000 is evaluated unless `--ckpt` names another.

```bash
python scripts/eval/eval.py --policy FastWAM --task cocarry
```
