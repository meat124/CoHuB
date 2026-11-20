# Phase-level evaluation

Each phase of a task evaluated on its own with the checkpoints of the main benchmark, starting from the state where a training demonstration had just finished the phase before.

## 1. Evaluate

```bash
python experiments/phase_eval/eval.py --policy DP --task cocarry
python experiments/phase_eval/eval.py --policy DP --task cocarry --phase carry --ckpt path/to/checkpoint.ckpt
```

The first command evaluates every phase of the task. The second evaluates one phase of a chosen checkpoint.

| Argument | Values |
|---|---|
| `--policy` | `ACT`, `DP`, `LatentToM`, `GauDP`, `Pi_05`, `GR00T_N17`, `Psi0` or `FastWAM` |
| `--task` | one of the eight two-humanoid tasks |
| `--phase` | the phases to evaluate. Every phase of the task is evaluated when it is left out |
| `--ckpt` | the path of the saved checkpoint to evaluate. The policy's final checkpoint is used when it is left out |
| `--no-video` | skips the videos |

A run that stops partway continues from the trials already written when it is started again.

## 2. Report

```bash
python experiments/phase_eval/report.py
```

It reads every run under `eval_results_phase/` and writes `eval_results_phase/summary.md` with the success rate of each phase and the collaborative and non-collaborative averages.

## Results

| What | Where |
|---|---|
| Evaluation results | `eval_results_phase/<task>/<policy>/`, holding `results.json` and `videos/` |
| Report | `eval_results_phase/summary.md` |

## Phase starts

`phase_starts/states/` holds, for each of the eight tasks, the demonstration frame every trial starts from and the simulator state at that frame. They only need to be rebuilt when the demonstrations change.

```bash
python experiments/phase_eval/phase_starts/build.py label cocarry
python experiments/phase_eval/phase_starts/build.py pack cocarry
```
