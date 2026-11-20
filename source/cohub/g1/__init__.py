"""The Unitree G1 robot layer: articulation, cameras, actions, grasp, observations.

Modules here never import from ``cohub.tasks``, ``cohub.teleop``,
``cohub.recording`` or ``cohub.federation``. Keep this ``__init__`` free of
re-exports: ``actions`` pulls in Pink/Pinocchio.
"""
