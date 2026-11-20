# LatentToM

LatentToM, trained from scratch. One run per task trains the policies of both humanoids together, under a shared sheaf loss. Each policy reads the egocentric view of its own humanoid as its private camera and the scene camera as the shared one. Only the tasks with two humanoids are supported.

Run the commands below from this folder.

```bash
cd baselines/policy/LatentToM
```

## Install

```bash
bash install.sh
```

It builds the conda environment `latenttom` (Python 3.10, torch 2.4.1), so `conda` is required.

## Train

```bash
bash process_data.sh cocarry
bash train.sh cocarry
```

`process_data.sh` downloads the dataset from the Hugging Face Hub when it is missing and converts the task into LatentToM's replay buffer. The download uses the simulation environment of the [main README](../../../README.md#installation).

`train.sh` trains for 1,000 epochs at batch size 32 and keeps a checkpoint every 250 epochs under `checkpoints/`. It fits one 24 GB GPU; FrameHang took 38 hours. An interrupted run resumes from its latest checkpoint.

The settings are the lines of [`baselines/configs/LatentToM.yaml`](../../configs/LatentToM.yaml). To change one, set it on the command line, and add `--tag` so that the run is saved under its own name.

```bash
LATENTTOM_EXTRA="optimizer.lr=5e-5" bash train.sh cocarry --tag lr5e-5
```

| Setting | Default | Meaning |
|---|---|---|
| `LATENTTOM_EPOCHS` | `1000` | epochs |
| `LATENTTOM_BATCH_SIZE` | `32` | batch size |
| `LATENTTOM_SNAPSHOT_EVERY` | `250` | epochs between the checkpoints that are kept |
| `LATENTTOM_EXTRA` | | further Hydra overrides, separated by spaces. The learning rate is `optimizer.lr`, 1e-4 by default |
| `SEED` | `0` | training seed |

## Evaluate

Run the evaluation from the repository root, as described in the [main README](../../../README.md#evaluation). The checkpoint of epoch 1,000 is evaluated unless `--ckpt` names another.

```bash
python scripts/eval/eval.py --policy LatentToM --task cocarry
```
