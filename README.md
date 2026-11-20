<h2 align="center">CoHuB: A Simulation Benchmark for Multi-Humanoid Collaboration</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2609.34782"><img src="assets/badges/arxiv.svg" alt="arXiv" height="40"></a>&nbsp;
  <a href="https://meat124.github.io/CoHuB/"><img src="assets/badges/project.svg" alt="Project Page" height="40"></a>&nbsp;
  <img src="assets/badges/dataset2.svg" alt="2-Agent Data (coming soon)" height="40">&nbsp;
  <img src="assets/badges/dataset3.svg" alt="3-Agent Data (coming soon)" height="40">
</p>

<p align="center">
  <img src="assets/overview.jpg" alt="CoHuB overview" width="100%">
</p>

CoHuB (Collaborative Multi-Humanoid Benchmark) is a simulation benchmark for multi-humanoid collaboration under egocentric visual observations. It provides 10 tasks, eight with two humanoids and two with three humanoids, spanning diverse collaboration patterns.

## Installation

CoHuB requires Ubuntu 22.04 or 24.04, NVIDIA driver 580 or newer and [uv](https://docs.astral.sh/uv/). Install it as follows.

```bash
git clone --single-branch https://github.com/meat124/CoHuB.git && cd CoHuB
bash install.sh
source .venv/bin/activate
```

`install.sh` builds the simulation environment in `.venv` (Isaac Sim 6.0.1, Isaac Lab 3.0, Python 3.12) and downloads the assets the tasks load.

To check the installation, open the scene of a task. No headset is needed.

```bash
python scripts/scene/view_env.py --task handover
```

See the [task list](source/cohub/tasks) for the name and the number of agents of each task.

Each baseline also needs an environment of its own. How to install it is described in the [baselines folder](baselines).

## Collecting demonstrations with VR teleoperation

CoHuB provides a multi-operator VR teleoperation pipeline, in which each operator controls one humanoid from its egocentric view. It needs one Meta Quest 3 per humanoid, on the same Wi-Fi network as the PC. Start a recording session with the name of the task and its number of agents.

```bash
./scripts/teleop/collect.sh --task cocarry --agents 2
./scripts/teleop/collect.sh --task movehouse --agents 3
```

The command prints one address per headset. Open it in the Quest browser, accept the certificate, press **Start Teleop**, then **A** to start recording.

| | Left controller | Right controller |
|---|---|---|
| Thumbstick | walk forward, backward and sideways | turn left and right |
| Index trigger | closes the index finger and thumb of the left hand | the same for the right hand |
| Grip trigger | closes the middle finger and thumb of the left hand | the same for the right hand |
| Buttons | hold **X** to crouch, hold **Y** to stand back up | press **A** to start or pause recording, press **B** to reset the episode |
| Controller pose | moves the left arm | moves the right arm |

A session records only the simulator states of the successful episodes, into `datasets/<task>/data`. The images are rendered afterwards. Turn the recording into a training dataset in three steps.

```bash
# 1. render the camera images from the saved states, into datasets/cocarry/data-full
python scripts/dataset/render_dataset_images.py datasets/cocarry/data --headless --enable_cameras --device cpu

# 2. check the rendered dataset
python scripts/dataset/verify_dataset.py datasets/cocarry/data-full

# 3. export it to the LeRobot format the baselines train on
python scripts/export/export_lerobot.py datasets/cocarry/data-full --out datasets/cocarry/lerobot
```

## Training

The demonstrations are in LeRobot v2.1 format, and each baseline converts them into the format it trains on. How to train each baseline is described in the [baselines folder](baselines). For example, to train Diffusion Policy on `cocarry`, run the following.

```bash
cd baselines/policy/DP
bash process_data.sh cocarry
bash train.sh cocarry
```

The hyperparameters of each baseline are in `baselines/configs/<POLICY>.yaml`. A variable set in the shell overrides one of them for that run.

```bash
DP_NUM_EPOCHS=300 bash train.sh cocarry
```

## Evaluation

To evaluate a checkpoint of a policy on a task, run the following.

```bash
python scripts/eval/eval.py --policy DP --task cocarry --ckpt path/to/checkpoint.ckpt
```

It starts the policy server, runs 100 rollouts in the simulator and writes `results.json` and the videos to `eval_results/<task>/`.

<table>
  <tr><th>Argument</th><th>Values</th></tr>
  <tr><td nowrap><code>--policy</code></td><td><code>ACT</code>, <code>DP</code>, <code>LatentToM</code>, <code>GauDP</code>, <code>Pi_05</code>, <code>GR00T_N17</code>, <code>Psi0</code> or <code>FastWAM</code></td></tr>
  <tr><td nowrap><code>--task</code></td><td>the name of a task, such as <code>cocarry</code></td></tr>
  <tr><td nowrap><code>--ckpt</code></td><td>the path of the saved checkpoint to evaluate. The policy's default is used when it is left out</td></tr>
  <tr><td nowrap><code>--episodes</code></td><td>the number of rollouts, 100 when it is left out</td></tr>
  <tr><td nowrap><code>--no-video</code></td><td>skips the videos</td></tr>
</table>

## Acknowledgments

The policy code under `baselines` is adapted from [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab).

## Citation

```bibtex
@article{park2026cohub,
  title   = {{CoHuB}: A Simulation Benchmark for Multi-Humanoid Collaboration},
  author  = {Park, Hyunjin and Chae, Jebeom and Park, Minwoo and Park, Sunghyun and
             Yoo, Hanjun and Choi, Seoyeon and Yoo, Soochul and Seo, Joohwan and
             Idrees, Sarmad and Hyun, Jae-Sang and Lee, Jongmin and Horowitz, Roberto and
             Lee, Youngwoon and Choi, Jongeun},
  journal = {arXiv preprint arXiv:2609.34782},
  year    = {2026}
}
```

## License

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details. The third-party 3D models the tasks use are credited in [THIRD_PARTY_ASSETS.md](THIRD_PARTY_ASSETS.md).
