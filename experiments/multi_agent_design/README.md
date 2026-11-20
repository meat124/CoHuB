# Multi-agent design choices

ACT, Diffusion Policy and GR00T N1.7 trained under each observation scope and policy sharing choice on the eight two-humanoid tasks.

|  | **Separate Policies** | **Shared Policy** |
|---|---|---|
| **Local Observation** | each humanoid sees its own egocentric image and proprioception, with its own parameters | the same observation, with one set of parameters for every humanoid |
| **Global Observation** | each humanoid sees the images and proprioception of all humanoids, with its own parameters | one policy sees all humanoids and controls all of them |

The main benchmark runs ACT and DP as Local Observation with Separate Policies, and GR00T N1.7 as Local Observation with a Shared Policy.

## 1. Prepare the data

Each combination gives the policy a different observation, so the demonstrations are converted once per combination before training.

```bash
python experiments/multi_agent_design/prepare_data.py
python experiments/multi_agent_design/prepare_data.py --policy DP --view global --sharing separate --task cocarry
```

The first command converts everything. The second converts one combination, using the arguments described below.

## 2. Train and evaluate

```bash
python experiments/multi_agent_design/train.py --policy DP --view global --sharing shared --task cocarry
python experiments/multi_agent_design/eval.py  --policy DP --view global --sharing shared --task cocarry --ckpt path/to/checkpoint.ckpt
```

Separate Policies train one checkpoint per humanoid, so the evaluation takes both.

```bash
python experiments/multi_agent_design/eval.py  --policy DP --view global --sharing separate --task cocarry --ckpt path/to/robot_a/checkpoint.ckpt path/to/robot_b/checkpoint.ckpt
```

| Argument | Values |
|---|---|
| `--policy` | `ACT`, `DP` or `GR00T_N17` |
| `--view` | `local` or `global` |
| `--sharing` | `separate` or `shared` |
| `--task` | a two-humanoid task. GR00T N1.7 trains one policy on all eight, so its `train.py` takes no task |
| `--ckpt` | for `eval.py`. The path of the saved checkpoint to evaluate. The last saved one is used when it is left out |
| `--no-video` | for `eval.py`. Skips the episode videos |

## Results

| What | Where |
|---|---|
| Checkpoints | `baselines/policy/<POLICY>/checkpoints/cohub-<task>_<robot>-<view>-<sharing>` |
| Evaluation results | `eval_results/<task>/<POLICY>-<mode>-<view>-<sharing>-seed<n>-ckpt<c>/`, holding `results.json` and a `videos/` folder with every episode |
