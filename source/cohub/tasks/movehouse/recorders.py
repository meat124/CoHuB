"""Recorder configuration for MoveHouse: the shared schema plus the door state."""

from __future__ import annotations

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from cohub.recording.recorder import DemoRecorderManagerCfg


@configclass
class DoorStateRecorderCfg(RecorderTermCfg):
    class_type: str = "cohub.tasks.movehouse.recorders:DoorStateRecorder"


class DoorStateRecorder(RecorderTerm):
    """Articulated door joints and the stateful interaction flags."""

    def record_post_step(self):
        door = self._env.scene["door"]
        term = self._env.action_manager.get_term("door_opening")
        return "states/door", {
            "joint_pos": door.data.joint_pos.torch.clone(),
            "joint_vel": door.data.joint_vel.torch.clone(),
            "handle_latched": term.handle_latched.float().clone(),
            "handle_opened": term.handle_opened.float().unsqueeze(-1).clone(),
            "opened_by_a": term.opened_by_a.float().unsqueeze(-1).clone(),
            "stage": term.stage.float().unsqueeze(-1).clone(),
        }


@configclass
class MoveHouseRecorderManagerCfg(DemoRecorderManagerCfg):
    """The shared schema plus the articulated door state."""

    record_door_state = DoorStateRecorderCfg()
