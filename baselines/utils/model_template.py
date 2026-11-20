class ModelTemplate:
    """Base template for a policy class. Subclass it and override the methods.

    The policy server passes observations in through update_obs, asks for
    actions through get_action and calls reset between episodes.
    `self.model` can be a neural network, a planner or any policy object.
    """

    RED = "\033[31m"
    RESET = "\033[0m"

    def __init__(self):
        """Set `self.model`, the actual model or policy object."""
        self.model = None

    def _error_msg(self, msg: str) -> str:
        return f"{self.RED}[ERROR] {msg}{self.RESET}"

    def update_obs(self, obs):
        """Store the current observation.

        Camera colors are already decoded:
        obs["vision"][<camera>]["color"] is always an image array.
        """
        raise NotImplementedError(
            self._error_msg("update_obs() must be implemented by the user.")
        )

    def update_obs_batch(self, obs_list):
        """Batched update_obs. Images are already decoded."""
        raise NotImplementedError(
            self._error_msg("update_obs_batch() must be implemented by the user.")
        )

    def get_action(self):
        """Return an action computed from the current observation."""
        raise NotImplementedError(
            self._error_msg("get_action() must be implemented by the user.")
        )

    def get_action_batch(self, env_idx_list=None):
        """Batched get_action."""
        raise NotImplementedError(
            self._error_msg("get_action_batch() must be implemented by the user.")
        )

    def reset(self):
        """Reset the internal state at the start of a new episode."""
        pass

    def prepare_case(self, case_meta=None):
        """Optional hook called before an action case starts."""
        pass

    def on_trial_end(self, result=None):
        """Optional hook called after a trial finishes."""
        pass
