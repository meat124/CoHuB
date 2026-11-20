# `cohub.teleop` — the operator's side

What sits between a headset and the env during a collection session. It
imports `cohub.g1` and `cohub.keys`, never a task: anything task-specific
arrives as an argument.

| module | what it holds |
|---|---|
| `cloudxr.py` | one CloudXR runtime per headset: ports and install directory, the landing page a headset opens, cleanup of an orphaned runtime |
| `view.py` | what the operator sees: the XR anchor, ego windows on the monitor, the recorded image's boundary in the headset, their own head hidden, the outcome text |
| `controls.py` | what the controllers do: walk and turn shaping, lean compensation, start/pause and reset buttons, crouch, haptics, raw input capture |
| `follower.py` | a robot without a headset: a constant hold pose, or the PC keyboard |

The keyboard follower is also the PC-side session control in a headset
session: `P` starts or pauses, `N` resets. In `collect.sh solo` it drives
Robot B (IJKL/UO walk, arrow keys move the wrists, SPACE grasps).
