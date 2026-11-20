# `cohub.g1` — the robot

Everything about the Unitree G1 that is the same in every task. It never
imports a task.

| module | what it holds |
|---|---|
| `robots.py` | the articulation cfg (`make_robot`) |
| `actions.py` | the action terms: arm IK and joint targets for the upper body, the walking policy for the legs |
| `grasp.py` | the contact grasp: which hand holds the object, and the sensors and gains a task needs for it |
| `cameras.py` | the recorded ego and scene cameras |
| `mdp.py` | observation terms, the observation group every task shares, `robot_fallen` |
| `geometry.py` | sim-free math: the ego lens and mount, wrist orientation from joint angles, the XR anchor yaw |

Import the module you need: `__init__.py` re-exports nothing, and `geometry.py`
imports without a simulator.
