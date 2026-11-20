# `scripts/` — entry points

Collect demonstrations, turn them into a dataset, and evaluate a policy on it.
Every file here is an entry point; the code they share is the `cohub` package
(`source/cohub/`). Run them in the simulation environment:

    source .venv/bin/activate
    python scripts/<group>/<script>.py ...

| step | script | what it does |
|---|---|---|
| install | `setup_assets.py rooms` | download and prepare the rooms the ten tasks stand in |
| | `setup_assets.py isaac --dest <dir>` | mirror the Isaac assets an evaluation loads from disk |
| collect | `teleop/collect.sh solo\|dual\|triple` | start a teleop session: the recording process and one mirror per headset |
| | `teleop/record_demos.py` | the recording process: runs the simulation, writes one state file per demonstration |
| | `teleop/xr_mirror.py` | one headset's process: shows the scene, sends back its robot's action |
| | `teleop/cloudxr_runtime.py` | one CloudXR runtime on its own |
| look | `scene/view_env.py` | open a task's scene in a desktop window, no headset |
| dataset | `dataset/render_dataset_images.py` | render the camera frames a session did not record (`data/` → `data-full/`) |
| | `dataset/encode_demo_videos.py` | one MP4 per demonstration, to review the takes |
| | `dataset/verify_dataset.py` | check a recorded dataset, or a LeRobot export |
| export | `export/export_lerobot.py` | `data-full/` → LeRobot v2.1: the training source and the Hub release |
| | `export/convert_lerobot.py` | LeRobot → the format each baseline trains on (`--format`) |
| evaluate | `eval/eval_policy_client.py` | the simulator side of an evaluation; `scripts/eval/runner.sh` starts it |
| | `eval/replay_actions_eval_env.py` | replay a recorded episode through the evaluation env |

Each script's flags are in its `--help`.
