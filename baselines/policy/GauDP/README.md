# GauDP

GauDP, trained from scratch on top of a pretrained Gaussian-splatting encoder (NoPoSplat). One centralized policy is trained per task: a single network reads the egocentric views and joint states of all humanoids and predicts all of their actions. The encoder is trained once and shared by every task.

Run the commands below from this folder.

```bash
cd baselines/policy/GauDP
```

## Install

```bash
bash install.sh
```

It builds the conda environment `gaudp` (Python 3.10, torch 2.1.2 with CUDA 11.8) and compiles two CUDA extensions, so `conda` is required. The pretrained NoPoSplat weights are downloaded by the first training run.

## Train

Training has three steps.

```bash
# 1. convert every task with two humanoids
for task in cocarry handover framehang pouring copouring trashcollection cartservice tablealign; do
  bash process_data.sh $task
done

# 2. train the shared Gaussian encoder, once
bash train_gaussian.sh

# 3. train the policy of a task
bash train.sh cocarry
```

1. `process_data.sh` downloads the dataset from the Hugging Face Hub when it is missing and converts a task into one HDF5 file. `cocarry` took 5 minutes and 14 GB. The download uses the simulation environment of the [main README](../../../README.md#installation).
2. `train_gaussian.sh` trains one encoder on the egocentric views of all eight tasks, for 10 epochs. It took about two days on one 96 GB GPU, and resumes when it is run again.
3. `train.sh` first runs the encoder over every frame of the task and stores the features, 36 to 259 GB per task, then trains the policy for 150 epochs at batch size 128. It stops when the shared encoder is missing.

For the two tasks with three humanoids, convert `movehouse` and `bigtable` and train their encoder with `bash train_gaussian.sh all3`. The encoder is described further in [GAUSSIAN_ENCODER.md](GAUSSIAN_ENCODER.md).

The settings are the lines of [`baselines/configs/GauDP.yaml`](../../configs/GauDP.yaml). To change one, set it on the command line, and add `--tag` so that the run is saved under its own name.

```bash
GAUDP_GAUSSIAN_FEATURES=$PWD/checkpoints/cohub-cocarry-unitree_g1x2_centralized-joint-0-cohub2/gaussian/features.hdf5 \
  GAUDP_POLICY_LR=5e-5 bash train.sh cocarry --tag lr5e-5
```

A tagged run has a folder of its own, so `GAUDP_GAUSSIAN_FEATURES` names the stored features of the default run instead of extracting them again.

| Setting | Default | Meaning |
|---|---|---|
| `GAUDP_POLICY_EPOCHS` | `150` | epochs of the policy |
| `GAUDP_POLICY_BATCH_SIZE` | `128` | batch size of the policy |
| `GAUDP_POLICY_GRAD_ACCUM` | `1` | steps the batch is split over; `2` fits a 24 GB GPU |
| `GAUDP_POLICY_LR` | `1e-4` | learning rate of the policy |
| `GAUDP_SHARED_EPOCHS` | `10` | epochs of the shared encoder |
| `GAUDP_GAUSSIAN_FEATURES` | | the file of the stored features, to keep it on another disk |
| `SEED` | `0` | training seed |

## Evaluate

Run the evaluation from the repository root, as described in the [main README](../../../README.md#evaluation). The last checkpoint of the policy is evaluated; `--ckpt best` evaluates the one with the lowest validation loss.

```bash
python scripts/eval/eval.py --policy GauDP --task cocarry
```
