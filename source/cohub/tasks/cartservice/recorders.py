"""Recorder configuration for CartService.

The shared schema records one manipuland, ``states/object`` (the bottle). This
adds ``states/cart``, ``states/plate`` and ``states/board``.
"""

from __future__ import annotations

import torch

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from cohub.recording.recorder import DemoRecorderManagerCfg


class _BodyStateRecorder(RecorderTerm):
    """One rigid body's pose and velocity, in the layout ``states/object`` uses.

    Subclasses set :attr:`entity`, which is both the scene entity's name and
    the key's last segment.
    """

    entity: str

    def record_post_step(self):
        obj = self._env.scene[self.entity]
        return f"states/{self.entity}", {
            "pose": torch.cat([obj.data.root_pos_w.torch, obj.data.root_quat_w.torch], dim=-1),
            "vel": torch.cat([obj.data.root_lin_vel_w.torch, obj.data.root_ang_vel_w.torch], dim=-1),
        }


class CartStateRecorder(_BodyStateRecorder):
    """``states/cart``: the service cart."""

    entity = "cart"


class PlateStateRecorder(_BodyStateRecorder):
    """``states/plate``: the plate of sandwiches."""

    entity = "plate"


class BoardStateRecorder(_BodyStateRecorder):
    """``states/board``: the cheese board."""

    entity = "board"


@configclass
class CartStateRecorderCfg(RecorderTermCfg):
    class_type: str = "cohub.tasks.cartservice.recorders:CartStateRecorder"


@configclass
class PlateStateRecorderCfg(RecorderTermCfg):
    class_type: str = "cohub.tasks.cartservice.recorders:PlateStateRecorder"


@configclass
class BoardStateRecorderCfg(RecorderTermCfg):
    class_type: str = "cohub.tasks.cartservice.recorders:BoardStateRecorder"


@configclass
class CartServiceRecorderManagerCfg(DemoRecorderManagerCfg):
    """The shared schema plus the cart, the plate and the board."""

    record_cart_state = CartStateRecorderCfg()
    record_plate_state = PlateStateRecorderCfg()
    record_board_state = BoardStateRecorderCfg()
