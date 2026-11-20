import os
from argparse import Namespace

import numpy as np
import cv2

from baselines.policy.ACT import model as core
from baselines.utils.checkpoint_resolver import build_run_dir_name
from baselines.policy.ACT.detr.act_policy import ACT

COHUB_PAIR = ("robot_a", "robot_b")


class Model(core.Model):
    def _init_cohub_decentralized(self, model_cfg):
        self._view = str(model_cfg.get("cohub_obs_view") or "local")
        if self._view == "local":
            return super()._init_cohub_decentralized(model_cfg)
        if self._view != "global":
            raise ValueError(f"cohub_obs_view must be local or global, got {self._view!r}")
        self.action_type = model_cfg['action_type']
        task = str(model_cfg.get('ckpt_name') or '').strip()
        if not task:
            raise ValueError("cohub decentralized eval needs ckpt_name=<task> (e.g. cocarry)")
        checkpoints_dir = os.path.join(os.path.dirname(os.path.abspath(core.__file__)), 'checkpoints')
        wide = (2 * core.COHUB_STATE_DIM, core.COHUB_ACTION_DIM)
        self._sub_models = {}
        served = {}
        for robot in COHUB_PAIR:
            explicit = model_cfg.get(f'model_dir_{robot}')
            if explicit:
                ckpt_dir = explicit
            else:
                run_cfg = dict(model_cfg)
                run_cfg['ckpt_name'] = f"{task}_{robot}"
                run_name = build_run_dir_name(run_cfg)
                if run_name is None:
                    raise ValueError("bench_name/ckpt_name/env_cfg_type/action_type/seed required to name the run dir")
                tag = str(model_cfg.get('ckpt_tag') or '').strip()
                ckpt_dir = os.path.join(checkpoints_dir, f"{run_name}-{tag}" if tag else run_name)
            if not os.path.isdir(ckpt_dir):
                raise FileNotFoundError(f"{robot} checkpoint not found: {ckpt_dir}")
            dims = core._checkpoint_dims(ckpt_dir)
            if dims != wide:
                raise ValueError(f"{robot}: {ckpt_dir} is {dims[0]}D state / {dims[1]}D action; the global view of one "
                                 f"robot is {wide[0]}/{wide[1]}")
            if os.path.realpath(ckpt_dir) in served:
                raise ValueError(f"{robot} and {served[os.path.realpath(ckpt_dir)]} were given the same global-view "
                                 f"checkpoint {ckpt_dir}: it predicts one robot's own action, so each robot needs its own")
            served[os.path.realpath(ckpt_dir)] = robot
            sub_cfg = dict(model_cfg)
            sub_cfg['ckpt_dir'] = ckpt_dir
            sub_cfg['camera_names'] = list(core.COHUB_JOINT_CAMERAS)
            core._set_act_dims(sub_cfg, *dims)
            self._sub_models[robot] = ACT(sub_cfg, Namespace(**sub_cfg))
            print(f"[ACT][cohub] {robot}: {ckpt_dir} (view=global, cameras={sub_cfg['camera_names']})")

    def _encode_cohub_robot_obs(self, observation, robot):
        if getattr(self, "_view", "local") != "global":
            return super()._encode_cohub_robot_obs(observation, robot)
        encoded = {}
        for camera_name in core.COHUB_JOINT_CAMERAS:
            if camera_name not in observation["vision"]:
                raise ValueError(f"{robot}: camera '{camera_name}' not in observation['vision']; the global view needs both ego views rendered")
            color = cv2.resize(observation["vision"][camera_name]["color"], core.COHUB_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
            encoded[camera_name] = np.moveaxis(color, -1, 0) / 255.0
        encoded["qpos"] = np.concatenate([
            np.asarray(observation["cohub_state"][r]["joint_pos"], dtype=np.float32) for r in COHUB_PAIR
        ])
        return encoded
