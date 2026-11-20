from typing import Any

from baselines.policy.GR00T_N17 import model as core
from baselines.policy.GR00T_N17.model import Gr00tPolicy

PARTNER_PREFIX = "partner_"
SCOPE_LOCAL, SCOPE_GLOBAL, SCOPE_PAIR = "local", "global", "pair"
PAIR = ("robot_a", "robot_b")


def _cohub_scope(policy: Any) -> str | None:
    configs = policy.modality_configs
    video = list(configs["video"].modality_keys)
    state = set(configs["state"].modality_keys)
    action = set(configs["action"].modality_keys)
    own = set(core.COHUB_STATE_SLICES)
    if video == ["ego"]:
        return SCOPE_LOCAL if state == own else None
    if video == ["ego", "ego_partner"]:
        partner = {f"{PARTNER_PREFIX}{group}" for group in core.COHUB_STATE_SLICES}
        return SCOPE_GLOBAL if state == own | partner else None
    if video == ["ego_a", "ego_b"]:
        commands = (*core.COHUB_JOINT_TARGET_GROUPS, "navigate_command", "base_height_command")
        both_states = {f"{robot}_{group}" for robot in PAIR for group in own}
        both_actions = {f"{robot}_{group}" for robot in PAIR for group in commands}
        return SCOPE_PAIR if state == both_states and action == both_actions else None
    return None


def _cohub_pair(obs: dict[str, Any]) -> tuple[str, str]:
    robots = core._cohub_robots(obs)
    if len(robots) != 2:
        raise RuntimeError(f"this checkpoint sees a pair of robots; the scene has {robots}")
    return robots


def _encode_global_observation(obs: dict[str, Any], robot: str, prompt: str) -> dict[str, Any]:
    pair = _cohub_pair(obs)
    if robot not in pair:
        raise RuntimeError(f"{robot} has no partner in a scene of {pair}")
    partner = pair[1] if pair[0] == robot else pair[0]
    encoded = core._encode_cohub_shared_observation(obs, robot, prompt)
    encoded["video"]["ego_partner"] = core._cohub_view(obs, core.COHUB_EGO_VIEW[partner])
    encoded["state"].update(
        {f"{PARTNER_PREFIX}{group}": value for group, value in core._cohub_shared_state(obs, partner).items()}
    )
    return encoded


def _encode_unprefixed_observation(obs: dict[str, Any], robot: str, prompt: str, scope: str) -> dict[str, Any]:
    if scope == SCOPE_GLOBAL:
        return _encode_global_observation(obs, robot, prompt)
    return core._encode_cohub_shared_observation(obs, robot, prompt)


class Model(core.Model):
    def _init_cohub(self, model_cfg: dict[str, Any]) -> None:
        mode = str(model_cfg.get("cohub_mode") or "decentralized").strip().lower()
        style = str(model_cfg.get("cohub_decentralized_style") or "shared").strip().lower()
        self._scope = None
        self._scopes: dict[str, str] = {}
        if mode != "decentralized" or style != "per_robot":
            return super()._init_cohub(model_cfg)
        self._cohub = True
        self.model_cfg = model_cfg
        self.device = model_cfg.get("device", "cuda:0" if self._has_cuda() else "cpu")
        task = str(model_cfg.get("ckpt_name") or "").strip()
        embodiment_tag = model_cfg.get("embodiment_tag", "NEW_EMBODIMENT")
        self._mode, self._style = mode, style
        default_prompt = model_cfg.get("default_prompt", "Perform the robot manipulation task.")
        task_prompts = core.COHUB_TASK_PROMPTS.get(core._cohub_task_key(task), {})
        self._prompt_overrides = {robot: model_cfg.get(f"prompt_{robot}") for robot in core.COHUB_ROBOTS}
        self._prompts = {
            robot: str(self._prompt_overrides[robot] or task_prompts.get(robot) or default_prompt)
            for robot in core.COHUB_ROBOTS
        }
        self._policies: dict[str, Gr00tPolicy] = {}
        for robot in PAIR:
            model_dir = core._resolve_cohub_model_dir(model_cfg, robot)
            policy = Gr00tPolicy(model_path=str(model_dir), embodiment_tag=embodiment_tag, device=self.device, strict=True)
            scope = _cohub_scope(policy)
            if scope not in (SCOPE_LOCAL, SCOPE_GLOBAL):
                raise RuntimeError(
                    f"{robot} checkpoint at {model_dir} was trained on video keys "
                    f"{list(policy.modality_configs['video'].modality_keys)}; a multitask policy of one robot "
                    "reads [ego] or [ego, ego_partner] -- wrong checkpoint pairing?")
            self._scopes[robot] = scope
            self._policies[robot] = policy
            print(f"[GR00T_N17][cohub] {robot}: {model_dir} (multitask, scope={scope})")
            print(f"[GR00T_N17][cohub] {robot} prompt: {self._prompts[robot]!r}")
        if len(set(self._scopes.values())) > 1:
            raise RuntimeError(f"the two robots' checkpoints were trained under different views: {self._scopes}")
        self.model = self._policies["robot_a"]
        self.action_horizon = len(self._policies["robot_a"].modality_configs["action"].delta_indices)
        self.exec_horizon = max(1, min(int(model_cfg.get("exec_horizon") or self.action_horizon), self.action_horizon))
        self._obs_list = []
        self._latest_env_idx_list = [0]
        print(f"[GR00T_N17][cohub] action_horizon={self.action_horizon} exec_horizon={self.exec_horizon}")

    def _init_cohub_shared(self, model_cfg: dict[str, Any], embodiment_tag: str) -> None:
        model_dir = core._resolve_cohub_model_dir(model_cfg, "shared")
        policy = Gr00tPolicy(model_path=str(model_dir), embodiment_tag=embodiment_tag, device=self.device, strict=True)
        scope = _cohub_scope(policy)
        if scope is None:
            raise RuntimeError(
                f"shared checkpoint at {model_dir} was trained on video keys "
                f"{list(policy.modality_configs['video'].modality_keys)}; expected [ego], [ego, ego_partner] or [ego_a, ego_b]")
        self._scope = scope
        if scope == SCOPE_PAIR:
            self._views = list(policy.modality_configs["video"].modality_keys)
            self._prompt_duo_override = model_cfg.get("prompt_duo")
            task = str(model_cfg.get("task_name") or "").strip()
            self._prompt_duo = core.COHUB_DUO_PROMPTS.get(core._cohub_task_key(task))
        self._policy_shared = policy
        self.model = policy
        self.action_horizon = len(policy.modality_configs["action"].delta_indices)
        self.exec_horizon = max(1, min(int(model_cfg.get("exec_horizon") or self.action_horizon), self.action_horizon))
        self._obs_list = []
        self._latest_env_idx_list = [0]
        print(f"[GR00T_N17][cohub] shared (multitask): {model_dir} scope={scope}")
        print(f"[GR00T_N17][cohub] action_horizon={self.action_horizon} exec_horizon={self.exec_horizon}")

    def _get_action_cohub(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        if self._mode == "centralized" or (self._style == "shared" and self._scope == SCOPE_LOCAL):
            return super()._get_action_cohub(obs)
        if self._style == "shared" and self._scope == SCOPE_PAIR:
            prompt = self._prompt_duo_override or (obs.get("cohub_instruction") or {}).get("duo") or self._prompt_duo
            if not prompt:
                raise ValueError("no instruction for the pair: the env sent no cohub_instruction['duo'] and "
                                 "COHUB_DUO_PROMPTS has none for this task (or pass prompt_duo)")
            pair = _cohub_pair(obs)
            encoded = core._encode_cohub_duo_observation(obs, str(prompt), self._views)
            action, _ = self._policy_shared.get_action(encoded)
            per_robot = {robot: self._pack_robot_action(action, robot) for robot in pair}
        elif self._style == "shared":
            per_robot = {}
            for robot in core._cohub_robots(obs):
                encoded = _encode_unprefixed_observation(obs, robot, self._cohub_prompt(obs, robot), self._scope)
                action, _ = self._policy_shared.get_action(encoded)
                per_robot[robot] = self._pack_robot_action(action, robot, prefixed=False)
        else:
            per_robot = {}
            for robot, policy in self._policies.items():
                encoded = _encode_unprefixed_observation(obs, robot, self._cohub_prompt(obs, robot), self._scopes[robot])
                action, _ = policy.get_action(encoded)
                per_robot[robot] = self._pack_robot_action(action, robot, prefixed=False)
        steps = min(len(chunk) for chunk in per_robot.values())
        return [{"cohub_raw_action": {robot: chunk[t] for robot, chunk in per_robot.items()}} for t in range(steps)]
