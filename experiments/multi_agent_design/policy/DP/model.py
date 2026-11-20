import os

import numpy as np

from baselines.policy.DP import model as core

COHUB_PAIR = ("robot_a", "robot_b")
COHUB_GLOBAL_CAM_KEYS = {"left_cam": "robot_a", "right_cam": "robot_b"}


class Model(core.Model):
    def _init_cohub_decentralized(self, model_cfg):
        self._view = str(model_cfg.get("cohub_obs_view") or "local")
        if self._view not in ("local", "global"):
            raise ValueError(f"cohub_obs_view must be local or global, got {self._view!r}")
        super()._init_cohub_decentralized(model_cfg)
        if self._view != "global":
            return
        if set(self._sub_policies) != set(COHUB_PAIR):
            raise ValueError(f"the global view is defined over {COHUB_PAIR}; loaded {sorted(self._sub_policies)}")
        dirs = {}
        for robot in COHUB_PAIR:
            ckpt_dir = os.path.realpath(model_cfg.get(f"model_dir_{robot}") or "")
            if ckpt_dir in dirs:
                raise ValueError(f"{robot} and {dirs[ckpt_dir]} were given the same global-view checkpoint: "
                                 "it predicts one robot's action, so each robot needs its own")
            dirs[ckpt_dir] = robot
        print(f"[DP][cohub] view=global cameras={list(COHUB_GLOBAL_CAM_KEYS)}")

    def _encode_cohub_robot_obs(self, observation, robot):
        if getattr(self, "_view", "local") != "global":
            return super()._encode_cohub_robot_obs(observation, robot)
        encoded = {}
        for key, owner in COHUB_GLOBAL_CAM_KEYS.items():
            slot = core.COHUB_CAMERA_SLOT[owner]
            if slot not in observation["vision"]:
                raise ValueError(f"{robot}: camera '{slot}' ({owner}'s ego view) is not in observation['vision']; "
                                 "the global view needs both ego views rendered")
            encoded[key] = core._prep_camera(observation["vision"][slot]["color"])
        encoded["agent_pos"] = np.concatenate([
            np.asarray(observation["cohub_state"][r]["joint_pos"], dtype=np.float32) for r in COHUB_PAIR
        ])
        return encoded
