"""Gym registrations for the CoHuB tasks, one id per task.

Entry points are lazy strings, so an import error in a task surfaces at
``gym.make`` and not at registration.
"""

import gymnasium as gym

gym.register(
    id="Isaac-CoCarry-G1x2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cocarry.cocarry_env_cfg:CoCarryG1x2EnvCfg",
        "demo_recorder_cfg_entry_point": "cohub.recording.recorder:DemoRecorderManagerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-G1x2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handover.handover_env_cfg:HandoverG1x2EnvCfg",
        "demo_recorder_cfg_entry_point": "cohub.recording.recorder:DemoRecorderManagerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-FrameHang-G1x2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.framehang.framehang_env_cfg:FrameHangG1x2EnvCfg",
        "demo_recorder_cfg_entry_point": "cohub.recording.recorder:DemoRecorderManagerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Pouring-G1x2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pouring.pouring_env_cfg:PouringG1x2EnvCfg",
        "demo_recorder_cfg_entry_point": "cohub.recording.recorder:DemoRecorderManagerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-CoPouring-G1x2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.copouring.copouring_env_cfg:CoPouringG1x2EnvCfg",
        "demo_recorder_cfg_entry_point": "cohub.recording.recorder:DemoRecorderManagerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-TrashCollection-G1x2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trashcollection.trashcollection_env_cfg:TrashCollectionG1x2EnvCfg",
        "demo_recorder_cfg_entry_point": "cohub.recording.recorder:DemoRecorderManagerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-CartService-G1x2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartservice.cartservice_env_cfg:CartServiceG1x2EnvCfg",
        # Records four rigid bodies; the shared recorder records one.
        "demo_recorder_cfg_entry_point": f"{__name__}.cartservice.recorders:CartServiceRecorderManagerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-TableAlign-G1x2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tablealign.tablealign_env_cfg:TableAlignG1x2EnvCfg",
        "demo_recorder_cfg_entry_point": "cohub.recording.recorder:DemoRecorderManagerCfg",
    },
    disable_env_checker=True,
)

# Three-robot tasks

gym.register(
    id="Isaac-MoveHouse-G1x3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.movehouse.movehouse_env_cfg:MoveHouseG1x3EnvCfg",
        "demo_recorder_cfg_entry_point": "cohub.tasks.movehouse.recorders:MoveHouseRecorderManagerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-BigTable-G1x3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bigtable.bigtable_env_cfg:BigTableG1x3EnvCfg",
        "demo_recorder_cfg_entry_point": "cohub.recording.recorder:DemoRecorderManagerCfg",
    },
    disable_env_checker=True,
)
