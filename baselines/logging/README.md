# Logging to wandb

Training and evaluation both log to [Weights & Biases](https://wandb.ai). Run `wandb login` once in the environment of the baseline.

Each baseline has a project of its own, `CoHuB_<policy>`, such as `CoHuB_DP`. A project holds two kinds of runs.

| Run | Holds |
|---|---|
| training | the loss, the validation loss and the learning rate. After an evaluation, the success rate of the task and its rollouts with their videos are added |
| evaluation, named `...-eval` | the success rate of each task and of each of its phases, over the checkpoints |

An evaluation uploads its result when it finishes. `EVAL_WANDB=0` skips the upload. To upload a result again, run the following from the repository root.

```bash
bash baselines/logging/log.sh DP eval_results/cocarry/<run>
```

A result with fewer rollouts than were asked for, or with a rollout that crashed, is not uploaded.
