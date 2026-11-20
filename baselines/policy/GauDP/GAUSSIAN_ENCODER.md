# GauDP: training the Gaussian encoder

The third step of the [README](README.md). A NoPoSplat encoder is finetuned on
the benchmark's ego views (RGB, depth, camera geometry); the next step caches
its features, and the policy trains on that cache.

The encoder reads no state and no action, so it is not tied to a task: one
encoder is trained over every task of a set, and each task uses it.

Every command below runs from this directory:

```bash
cd baselines/policy/GauDP
```

## Before training

The converted data of **every** task of the set, from
[Process data](README.md#process-data):

```bash
# the two-robot set
bash process_data.sh shared
for t in cocarry handover framehang pouring copouring trashcollection cartservice tablealign; do
  bash process_data.sh $t
done

# the three-robot set
COHUB_TASK_SET=all3 bash process_data.sh shared
for t in movehouse bigtable; do bash process_data.sh $t; done
```

The pretrained NoPoSplat weights are downloaded into `weights/` by the first run.

## Train

```bash
bash train_gaussian.sh                 # the eight two-robot tasks, two ego views
bash train_gaussian.sh all3            # movehouse and bigtable, three ego views
TRAIN_DRY_RUN=1 bash train_gaussian.sh # print the resolved command and stop
```

The defaults are the benchmark's recipe. The settings worth changing, as
`NAME=value bash train_gaussian.sh`:

| Setting | Default | Three-robot set | Default | Meaning |
|---|---|---|---|---|
| `GAUDP_SHARED_EPOCHS` | `10` | `GAUDP_SHARED3_EPOCHS` | `8` | epochs |
| `GAUDP_SHARED_LR` | `5e-5` | `GAUDP_SHARED3_LR` | `5e-5` | learning rate; the pretrained backbone trains at a tenth of it |
| `GAUDP_GAUSSIAN_BATCH_SIZE` | `8` | `GAUDP_GAUSSIAN3_BATCH_SIZE` | `4` | micro-batch |
| `GAUDP_GAUSSIAN_GRAD_ACCUM` | `1` | `GAUDP_GAUSSIAN3_GRAD_ACCUM` | `2` | micro-batches per optimizer step |

The rest are lines of `baselines/configs/GauDP.yaml`.

|  |  |
|---|---|
| Writes | `checkpoints/cohub-shared-unitree_g1x2_centralized-joint-<seed>-<tag>-lr5e-5/gaussian/`; the three-view one has `-ego_a-ego_b-ego_c` appended |
| Files | `best.ckpt` (lowest `val/loss`, the one the tasks use), `last.ckpt`, `metrics.jsonl`; 7.3 GB per checkpoint |
| wandb | project `CoHuB_GauDP`, one run per encoder |

### Memory and time

Keep batch x accumulation at 8 when the micro-batch changes.

| Card | Views | Batch x accumulation | Peak memory | One epoch |
|---|---|---|---|---|
| 96 GB | 2 | 8 x 1 (the default) | 34.5 GiB | 4.2-4.9 h |
| 24 GB | 2 | 2 x 4 | 15.7 GiB | 7 h |
| 96 GB | 3 | 4 x 2 (the default) | 31.6 GiB | 3.4-4.3 h |
| 48 GB | 3 | 4 x 2 (the default) | 31.6 GiB | 6.9 h |

```bash
GAUDP_GAUSSIAN_BATCH_SIZE=2 GAUDP_GAUSSIAN_GRAD_ACCUM=4 bash train_gaussian.sh   # a 24 GB card
```

### The benchmark's encoders

| Encoder | Command | Validation PSNR |
|---|---|---|
| two robots | `bash train_gaussian.sh` | 20.43 after one epoch, 21.14 after ten |
| three robots | `bash train_gaussian.sh all3` | 14.83 after one epoch, 17.78 after eight |

### Restarts

A run that stops is started again with the same command and resumes from
`last.ckpt`, so the restarts train what one uninterrupted run would.
`GAUDP_STOP_AT_EPOCH` cuts the schedule into runs that fit a time limit:

```bash
GAUDP_STOP_AT_EPOCH=2 bash train_gaussian.sh   # epochs 1-2, then stops
GAUDP_STOP_AT_EPOCH=4 bash train_gaussian.sh   # resumes, epochs 3-4
bash train_gaussian.sh                         # resumes, the rest
```

Epochs, batch size, accumulation and learning rate are fixed once a run has
started: a restart with other values is refused. A changed recipe is a new
encoder, under its own `--tag`.

## Use it in a task

Nothing to set. [Extract features](README.md#extract-features) and
[Train](README.md#train) find the shared encoder of the task's own tag and
views.

The tag is part of the encoder's directory name, so a task run with `--tag T`
looks for the encoder trained with `--tag T`. To use an encoder of another
tag, name its directory:

```bash
GAUDP_SHARED_GAUSSIAN_DIR=$PWD/checkpoints/cohub-shared-unitree_g1x2_centralized-joint-0-cohub2-lr5e-5 \
  bash train.sh cocarry --tag h40
```

## One task's own encoder

```bash
GAUDP_SHARED_GAUSSIAN=0 bash train.sh cocarry   # the encoder, the features, the policy
```

|  |  |
|---|---|
| Epochs | `GAUDP_GAUSSIAN_EPOCHS`, 20, ended early after 3 without a `val/loss` improvement |
| Writes | `checkpoints/cohub-<task>-unitree_g1x2_centralized-joint-<seed>-<tag>/gaussian/` |
| Measured | handover on one 96 GB card: 2111 s per epoch; validation PSNR 23.24 after one epoch, 26.18 at epoch 15 |
| Restart | the step ends as soon as a `best.ckpt` exists; `GAUDP_FORCE_GAUSSIAN=1` resumes a run that was interrupted |

This is also what `bash train.sh <task>` does first when no shared encoder is
found.

## Check the reconstruction

```bash
conda activate gaudp
ENC=checkpoints/cohub-shared-unitree_g1x2_centralized-joint-0-cohub2-lr5e-5/gaussian/best.ckpt
bash eval_gaussian.sh cocarry 0 0 $ENC --dump-only     # a few keyframes per validation episode, minutes
bash eval_gaussian.sh cocarry 0 0 $ENC                 # the whole validation split
```

Writes `gaussian_eval/best/` under the task's checkpoint directory: the scores,
and under `recon/` one PNG per keyframe, ground truth beside reconstruction for
RGB and depth.

## When it stops

| Message | Cause |
|---|---|
| `no LeRobot export for: ...`, `no data at ... -- build it first` | a task of the set is not downloaded or not converted: [Before training](#before-training) |
| `datasets disagree on camera_order` | two- and three-robot tasks in one set; they are separate encoders |
| `CUDA out of memory` | lower the batch size and raise the accumulation by the same factor |
| `was trained toward N optimizer steps and this run toward M` | the directory holds a run of another budget: restore the settings, or use a new `--tag` |
