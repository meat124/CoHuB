# Diffusion Policy

Diffusion Policy, trained from scratch. One policy is trained per task and per humanoid. Each policy reads the egocentric view and the joint state of its own humanoid, without language.

Run the commands below from this folder.

```bash
cd baselines/policy/DP
```

## Install

```bash
bash install.sh
```

It builds the conda environment `cohub_dp` (Python 3.10, torch 2.4.1), so `conda` is required.

## Train

```bash
bash process_data.sh cocarry
bash train.sh cocarry
```

`process_data.sh` downloads the dataset from the Hugging Face Hub when it is missing and converts the task into one zarr replay buffer per humanoid. The download uses the simulation environment of the [main README](../../../README.md#installation).

`train.sh` trains the policy of each humanoid in turn, for 600 epochs at batch size 256, and saves a checkpoint every 150 epochs under `checkpoints/`. The policy predicts 40 steps and executes 20 of them before it predicts again. Training takes 13 GB of GPU memory, and an interrupted run resumes from its latest checkpoint.

The task is any name in the [task list](../../../source/cohub/tasks), with two or three humanoids.

The settings are the lines of [`baselines/configs/DP.yaml`](../../configs/DP.yaml). To change one, set it on the command line, and add `--tag` so that the run is saved under its own name.

```bash
DP_LR=5e-5 bash train.sh cocarry --tag lr5e-5
```

| Setting | Default | Meaning |
|---|---|---|
| `DP_NUM_EPOCHS` | `600` | epochs |
| `DP_BATCH_SIZE` | `256` | batch size |
| `DP_LR` | `1e-4` | learning rate |
| `DP_CKPT_EVERY` | `150` | epochs between checkpoints |
| `DP_HORIZON` | `40` | steps predicted per chunk |
| `DP_N_ACTION_STEPS` | `20` | steps executed before the next prediction |
| `SEED` | `0` | training seed |

## Evaluate

Run the evaluation from the repository root, as described in the [main README](../../../README.md#evaluation). The checkpoint of epoch 600 is evaluated unless `--ckpt` names another.

```bash
python scripts/eval/eval.py --policy DP --task cocarry
```
