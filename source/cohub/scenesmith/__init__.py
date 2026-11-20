"""Use a downloaded SceneSmith scene as a CoHuB environment.

    python scripts/setup_assets.py rooms     # everything the tasks need

This package imports none of its submodules: ``prepare`` loads pxr.
"""

REQUIRED_SCENES = {
    # scene id -> object-name substrings to leave simulated. `setup` fetches
    # and prepares every row. The id includes the subset: `House/` for
    # multi-room scenes, `Room/` for single-room ones.
    "Room/scene_084": (),  # Isaac-Handover-G1x2-v0
    "Room/scene_143": (),  # Isaac-CoCarry-G1x2-v0
    "Room/scene_136": (),  # Isaac-FrameHang-G1x2-v0
    # TableAlign's chair is the goal frame, so it stays pinned.
    "Room/scene_115": (),  # Isaac-TableAlign-G1x2-v0
    # A four-room house, 2.7 GB unpacked.
    "House/scene_098": (),  # Isaac-MoveHouse-G1x3-v0
    "Room/scene_147": (),  # Isaac-TrashCollection-G1x2-v0
    "Room/scene_110": (),  # Isaac-CartService-G1x2-v0
    "Room/scene_085": (),  # Isaac-Pouring-G1x2-v0
    "Room/scene_102": (),  # Isaac-CoPouring-G1x2-v0
    "Room/scene_046": (),  # Isaac-BigTable-G1x3-v0
}

SURFACED_SCENES = frozenset(
    {"Room/scene_136", "Room/scene_115", "Room/scene_084", "Room/scene_102",
     "Room/scene_110", "Room/scene_046"}
)
"""Scenes whose materials get surface values and the window pass.
Opt-in per scene, because it changes every rendered pixel."""

__all__ = ["REQUIRED_SCENES", "SURFACED_SCENES"]
