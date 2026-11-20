"""TrashCollection's MDP terms."""

from cohub.tasks._base.mdp import reset_region  # noqa: F401

from .terminations import TrashCollectionSuccess, bin_tipped, litter_dropped  # noqa: F401
