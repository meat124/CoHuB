# `cohub.recording` — the dataset a session writes

The same recorder serves every task: its terms read the scene by the shared
entity names (`robot_a`, `robot_b`, `object`, the cameras) and record what the
running task put there.

| module | what it holds |
|---|---|
| `recorder.py` | the recorder cfg and its terms: robot and object state, contacts, grasp, operator input, cameras |
| `dataset_writer.py` | how a session reaches disk: one HDF5 per demonstration (or one appended file), and `episodes.txt` |
| `demo_source.py` | reading a recorded dataset offline, and the provenance stamped in it; imports without a simulator |
| `demo_video.py` | MP4 encoding of recorded frames |
| `operators.yaml` | who operated which robot |
