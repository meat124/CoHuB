import hashlib
import io
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from baselines.utils.model_template import ModelTemplate
from baselines.utils.checkpoint_resolver import build_run_dir_name
from baselines.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


POLICY_DIR = Path(__file__).resolve().parent
FASTWAM_ROOT = POLICY_DIR / "FastWAM"
FASTWAM_SRC = FASTWAM_ROOT / "src"

# Vision slot that carries each robot's head camera.
COHUB_ROBOTS = ("robot_a", "robot_b", "robot_c")
COHUB_CAMERA_SLOT = {
    "robot_a": "cam_left_wrist",
    "robot_b": "cam_right_wrist",
    "robot_c": "cam_third_view",
}


def _cohub_robots(observation: dict) -> tuple[str, ...]:
    """Robots present in the observation, in canonical order."""
    present = observation.get("cohub_state") or {}
    return tuple(robot for robot in COHUB_ROBOTS if robot in present) or COHUB_ROBOTS[:2]

# Key of the per-agent instructions inside an encoded observation. Not a policy target.
INSTRUCTIONS_KEY = "__cohub_instructions__"

# Task yaml per checkpoint profile, the same ones train.sh trains under.
COHUB_SIM_TASK = {
    "unitree_g1x2_centralized": "cohub_uncond_2cam_384_1e-4",
    "unitree_g1x2_decentralized": "cohub_uncond_1cam_192_1e-4",
}

POLICY_NOISE_SCHEDULE = "blake2b-v1"


def _derive_policy_noise_seed(
    episode_seed: int,
    target: str,
    replan_index: int,
    salt: int = 0,
) -> int:
    """Reproducible noise seed for one episode, role and replan, in the non-negative int64 range."""
    if replan_index < 0:
        raise ValueError(f"replan_index must be non-negative, got {replan_index}")
    payload = f"{int(episode_seed)}\0{int(salt)}\0{target}\0{int(replan_index)}".encode()
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"MHBFastWAMv1",
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _standardize_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] != (240, 320):
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    if image.shape != (240, 320, 3):
        raise ValueError(f"Expected standardized RGB shape (240, 320, 3), got {image.shape}")
    return image


def _get_instruction(obs: dict, fallback: str) -> str:
    value = obs.get("task_instruction")
    if value is None:
        value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else fallback
    if value is None:
        return fallback
    if hasattr(value, "item"):
        value = value.item()
    text = str(value).strip()
    return text if text else fallback


def _pack_single_robot_action(flat_action: np.ndarray) -> dict:
    """One robot's 35D action -> {joint_targets, height, base_vel}."""
    flat_action = np.asarray(flat_action, dtype=np.float32)
    assert flat_action.shape[-1] == 35, f"expected 35D per-robot action, got {flat_action.shape}"
    return {
        "joint_targets": flat_action[0:31],
        "height": flat_action[31:32],
        "base_vel": flat_action[32:35],
    }


def _pack_dual_arm_action(flat_action: np.ndarray) -> dict:
    """A centralized 70D flat action -> one dict per robot."""
    flat_action = np.asarray(flat_action, dtype=np.float32)
    assert flat_action.shape[-1] == 70, f"expected 70D dual-robot action, got {flat_action.shape}"
    return {
        robot: _pack_single_robot_action(flat_action[i * 35 : (i + 1) * 35])
        for i, robot in enumerate(("robot_a", "robot_b"))
    }


def _h264_roundtrip(frame: np.ndarray, crf: int) -> np.ndarray:
    """Encode one frame with libx264 (yuv420p) and decode it, to match the dataset's frames.

    crf 20 matches the dataset export."""
    import av

    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=50)
        stream.width, stream.height = int(frame.shape[1]), int(frame.shape[0])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(int(crf)), "threads": "1"}
        src = av.VideoFrame.from_ndarray(np.array(frame, dtype=np.uint8, copy=True), format="rgb24")
        for packet in stream.encode(src):
            out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    buf.seek(0)
    with av.open(buf) as inp:
        return next(inp.decode(video=0)).to_ndarray(format="rgb24")


def _cohub_image_tensor_method(env_cfg_type: str, codec_crf: int = 0):
    """Image-tensor builder for one CoHuB mode, bound onto the upstream policy.

    Reproduces training's centre crop: decentralized keeps rows 24:216 of the
    240x320 view, centralized stacks both views and keeps rows 48:432.
    `codec_crf` > 0 sends every view through `_h264_roundtrip` first.
    """
    import torch

    if codec_crf > 0:
        print(f"[FastWAM][cohub] observation frames pass through libx264 crf {codec_crf} (yuv420p) before the crop", flush=True)

    def build(self, observation):
        images = observation["images"]
        view = (lambda name: _h264_roundtrip(images[name], codec_crf)) if codec_crf > 0 else (lambda name: images[name])
        if env_cfg_type == "unitree_g1x2_centralized":
            canvas = np.concatenate([view("ego_a"), view("ego_b")], axis=0)[48:432]  # (384, 320, 3)
        else:
            canvas = view("ego")[24:216]  # (192, 320, 3)
        # Observation arrays are read-only views; copy before handing them to torch.
        tensor = torch.from_numpy(np.array(canvas, dtype=np.uint8, copy=True)).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        return tensor * (2.0 / 255.0) - 1.0

    return build


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg["action_type"]
        self.env_cfg_type = self.model_cfg["env_cfg_type"]
        self.action_horizon = 1
        # exec_horizon (EVAL_EXEC_HORIZON) takes precedence over deploy.yml's replan_steps.
        self.replan_steps = int(self.model_cfg.get("exec_horizon") or self.model_cfg.get("replan_steps") or 24)
        self._image_codec_crf = int(self.model_cfg.get("image_codec_crf") or 0)
        self.default_instruction = str(
            self.model_cfg.get("default_instruction")
            or self.model_cfg.get("prompt")
            or "follow the instruction"
        )
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self.model = None
        self.allow_dummy_policy = _is_true(self.model_cfg.get("allow_dummy_policy", False))

        self._cohub = str(self.model_cfg.get("bench_name") or "") == "cohub"
        if self._cohub:
            initial_episode_seed = self.model_cfg.get("eval_seed")
            if _is_none_like(initial_episode_seed):
                initial_episode_seed = self.model_cfg.get("seed")
            self._policy_episode_seed = int(initial_episode_seed or 0)
            self._policy_seed_salt = int(self.model_cfg.get("policy_seed_salt") or 0)
            self._policy_replan_index: dict[str, int] = {}
            self._init_cohub()
            return

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        checkpoint_path = self.model_cfg.get("checkpoint_path") or self.model_cfg.get("ckpt_setting")
        dataset_stats_path = self.model_cfg.get("dataset_stats_path")

        if self.allow_dummy_policy:
            print("[FastWAM] allow_dummy_policy=true; real checkpoint loading is skipped for debug flow only.")
            return

        if _is_none_like(checkpoint_path):
            raise FileNotFoundError("FastWAM requires checkpoint_path/ckpt_setting for real deployment.")
        if _is_none_like(dataset_stats_path):
            raise FileNotFoundError("FastWAM requires dataset_stats_path for real deployment.")

        self.model = self._load_upstream_policy(
            checkpoint_path, dataset_stats_path, sim_cfg_name=None, sim_task=None
        )
        self.action_horizon = int(self.model.action_horizon)
        self.replan_steps = int(self.model.replan_steps)

    # CoHuB: centralized is one two-robot 70D policy; decentralized is one 35D
    # policy per robot. State comes from obs["cohub_state"]; actions go out as
    # `cohub_raw_action`.

    def _init_cohub(self):
        sim_task = self.model_cfg.get("sim_task")
        if _is_none_like(sim_task) or str(sim_task).startswith("robotwin"):
            sim_task = COHUB_SIM_TASK.get(self.env_cfg_type)
        if sim_task is None:
            raise ValueError(f"unsupported cohub env_cfg_type for FastWAM: {self.env_cfg_type}")
        self._sim_task = str(sim_task)
        self._decentralized = self.env_cfg_type == "unitree_g1x2_decentralized"
        # shared: one multitask policy for every agent, told apart by instruction.
        # per_robot: one checkpoint per (task, robot).
        self._shared = str(
            self.model_cfg.get("cohub_decentralized_style") or "shared"
        ).strip().lower() == "shared"
        task = str(self.model_cfg.get("ckpt_name") or "").strip()
        if not task:
            raise ValueError("cohub eval needs ckpt_name=<task> (e.g. cocarry)")

        self._policies: dict[str, Any] = {}
        self._instructions: dict[str, str] = {}
        self._obs_of: dict[str, dict] = {}
        self._batch: dict[int, dict[str, dict]] = {}

        if self.allow_dummy_policy:
            print("[FastWAM][cohub] allow_dummy_policy=true; serving zero actions for debug flow only.")
            targets = COHUB_ROBOTS[:2] if self._decentralized else ("duo",)
            for target in targets:
                self._instructions[target] = self.default_instruction
            return

        if self._decentralized and self._shared:
            ckpt = self.model_cfg.get("model_dir") or self.model_cfg.get("checkpoint_path")
            run_name = build_run_dir_name(dict(self.model_cfg))
            if _is_none_like(ckpt):
                ckpt = self._newest_weights(POLICY_DIR / "checkpoints" / run_name)
            stats = self._cohub_stats_path(task)
            policy = self._load_upstream_policy(
                ckpt, stats, sim_cfg_name="sim_cohub.yaml", sim_task=self._sim_task,
                text_cache_dir=self._cohub_text_cache_dir(task),
            )
            self._preflight_text_cache(policy, task)
            fallback = self._cohub_instruction(task)
            # One policy object serves every role.
            for robot in COHUB_ROBOTS:
                self._policies[robot] = policy
                self._instructions[robot] = fallback
            print(f"[FastWAM][cohub] shared (multitask): {ckpt}")
        elif self._decentralized:
            robots = COHUB_ROBOTS if task in {"movehouse", "bigtable", "multitask3"} else COHUB_ROBOTS[:2]
            for robot in robots:
                ckpt = self.model_cfg.get(f"model_dir_{robot}")
                run_cfg = dict(self.model_cfg)
                run_cfg["ckpt_name"] = f"{task}_{robot}"
                run_name = build_run_dir_name(run_cfg)
                if _is_none_like(ckpt):
                    ckpt = self._newest_weights(POLICY_DIR / "checkpoints" / run_name)
                stats = self._cohub_stats_path(f"{task}_{robot}")
                self._policies[robot] = self._load_upstream_policy(
                    ckpt, stats, sim_cfg_name="sim_cohub.yaml", sim_task=self._sim_task,
                    text_cache_dir=self._cohub_text_cache_dir(f"{task}_{robot}"),
                )
                self._instructions[robot] = self._cohub_instruction(f"{task}_{robot}")
                self._preflight_text_cache(self._policies[robot], f"{task}_{robot}")
                print(f"[FastWAM][cohub] {robot}: {ckpt}")
        else:
            ckpt = self.model_cfg.get("model_dir") or self.model_cfg.get("checkpoint_path")
            run_cfg = dict(self.model_cfg)
            run_name = build_run_dir_name(run_cfg)
            if _is_none_like(ckpt):
                ckpt = self._newest_weights(POLICY_DIR / "checkpoints" / run_name)
            stats = self._cohub_stats_path(task)
            self._policies["duo"] = self._load_upstream_policy(
                ckpt, stats, sim_cfg_name="sim_cohub.yaml", sim_task=self._sim_task,
                text_cache_dir=self._cohub_text_cache_dir(task),
            )
            self._instructions["duo"] = self._cohub_instruction(task)
            self._preflight_text_cache(self._policies["duo"], task)
            print(f"[FastWAM][cohub] centralized: {ckpt}")

        first = next(iter(self._policies.values()))
        self.action_horizon = int(first.action_horizon)
        self.replan_steps = int(first.replan_steps)

    def _cohub_data_root(self, ckpt_name: str) -> Path:
        data_key = f"cohub-{ckpt_name}-{self.env_cfg_type}-{self.action_type}"
        return POLICY_DIR / "data" / data_key

    def _cohub_text_cache_dir(self, ckpt_name: str):
        """T5 cache dir the checkpoint trained from, or None to load the text encoder.

        `text_embedding_cache_dir` overrides it; `serve_text_encoder: true` returns None."""
        explicit = self.model_cfg.get("text_embedding_cache_dir")
        if not _is_none_like(explicit):
            return str(explicit)
        if _is_true(self.model_cfg.get("serve_text_encoder", False)):
            return None
        data_key = f"cohub-{ckpt_name}-{self.env_cfg_type}-{self.action_type}"
        cache = FASTWAM_ROOT / "data" / "text_embeds_cache" / "XPolicyLab" / data_key
        if any(cache.glob("*.pt")):
            return str(cache)
        raise FileNotFoundError(
            f"No T5 text-embedding cache at {cache}. Precompute it with "
            "FastWAM/scripts/precompute_text_embeds.py over this dataset "
            "(`bash process_data.sh` in policy/FastWAM builds it), name another "
            "with deploy.yml's `text_embedding_cache_dir`, or set "
            "`serve_text_encoder: true` to load the encoder instead."
        )

    def _cohub_stats_path(self, ckpt_name: str) -> str:
        explicit = self.model_cfg.get("dataset_stats_path")
        if not _is_none_like(explicit):
            return str(explicit)
        return str(self._cohub_data_root(ckpt_name) / "dataset_stats.json")

    def _cohub_instruction(self, ckpt_name: str) -> str:
        """Instruction the checkpoint trained with, read from the dataset's tasks.jsonl.

        `prompt` overrides it; `default_instruction` is the last resort."""
        if self.model_cfg.get("prompt"):
            return str(self.model_cfg["prompt"])
        tasks_file = self._cohub_data_root(ckpt_name) / "lerobot" / "meta" / "tasks.jsonl"
        try:
            first = json.loads(tasks_file.read_text().splitlines()[0])
            return str(first["task"])
        except (OSError, IndexError, KeyError, json.JSONDecodeError):
            print(f"[FastWAM][cohub] no readable {tasks_file}; using the default instruction")
            return self.default_instruction

    def _cohub_instruction_for(self, target: str, wire: dict[str, str]) -> str:
        """Instruction for one agent and one inference.

        Precedence: deploy.yml's `prompt`, the env's sentence (`wire`), the dataset's.
        """
        if self.model_cfg.get("prompt"):
            sentence = str(self.model_cfg["prompt"])
        else:
            sentence = wire.get(target)
            sentence = str(sentence) if sentence and str(sentence).strip() else self._instructions[target]
        # Memoised on the incoming sentence.
        seen = self.__dict__.setdefault("_instruction_seen", {})
        cached = seen.get(target)
        if cached is not None and cached[0] == sentence:
            return cached[1]
        resolved = self._resolve_cached_instruction(target, sentence)
        seen[target] = (sentence, resolved)
        print(f"[FastWAM][cohub] {target} instruction: {resolved!r}", flush=True)
        return resolved

    def _text_cache_has(self, policy, sentence: str) -> bool:
        """Whether the policy's T5 cache holds `sentence`. True when the policy has no cache."""
        if getattr(policy, "text_embedding_cache_dir", None) is None:
            return True
        from fastwam.datasets.lerobot.robot_video_dataset import (
            DEFAULT_PROMPT,
            load_cached_text_context,
        )
        try:
            load_cached_text_context(
                policy.text_embedding_cache_dir,
                DEFAULT_PROMPT.format(task=sentence),
                int(getattr(policy, "context_len", 128)),
            )
        except FileNotFoundError:
            return False
        return True

    def _preflight_text_cache(self, policy, ckpt_name: str) -> None:
        """Raise at load time if any training sentence is missing from the T5 cache."""
        if getattr(policy, "text_embedding_cache_dir", None) is None:
            return
        tasks_file = self._cohub_data_root(ckpt_name) / "lerobot" / "meta" / "tasks.jsonl"
        try:
            sentences = [
                str(json.loads(line)["task"])
                for line in tasks_file.read_text().splitlines()
                if line.strip()
            ]
        except (OSError, KeyError, json.JSONDecodeError):
            print(f"[FastWAM][cohub] no readable {tasks_file}; text-cache preflight skipped")
            return
        missing = [s for s in sentences if not self._text_cache_has(policy, s)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(sentences)} training sentences are absent from the "
                f"T5 cache {policy.text_embedding_cache_dir} (first missing: {missing[0]!r}). "
                "Rebuild it with FastWAM/scripts/precompute_text_embeds.py over this dataset."
            )
        print(
            f"[FastWAM][cohub] text-cache preflight: {len(sentences)} training sentences present",
            flush=True,
        )

    def _resolve_cached_instruction(self, target: str, sentence: str) -> str:
        """Return `sentence` if the T5 cache holds it, else raise.

        FASTWAM_UNCACHED_INSTRUCTION=dataset substitutes the trained sentence
        instead. Use it for wiring checks only, not for benchmark scores."""
        policy = self._policies.get(target)
        if policy is None or self._text_cache_has(policy, sentence):
            return sentence
        fallback = self._instructions[target]
        if os.environ.get("FASTWAM_UNCACHED_INSTRUCTION") == "dataset" and self._text_cache_has(
            policy, fallback
        ):
            print(
                f"[FastWAM][cohub] WARNING {target}: {sentence!r} is not in the T5 cache; "
                f"substituting the trained sentence {fallback!r} "
                "(FASTWAM_UNCACHED_INSTRUCTION=dataset). This score is not a benchmark result.",
                flush=True,
            )
            return fallback
        raise FileNotFoundError(
            f"{target} was told {sentence!r}, which is not in this checkpoint's T5 cache "
            f"({policy.text_embedding_cache_dir}) -- the checkpoint never trained on it. "
            "Evaluate a task whose sentences the training dataset covers, precompute the "
            "embedding for this one, or set FASTWAM_UNCACHED_INSTRUCTION=dataset to "
            "substitute a trained sentence for a wiring check (not a benchmark result)."
        )

    def _newest_weights(self, run_dir: Path) -> str:
        """Weights file under <run>/checkpoints/weights: the newest step, or the one `checkpoint_num` names."""
        ckpt_num = self.model_cfg.get("checkpoint_num")
        if not _is_none_like(ckpt_num) and str(ckpt_num) != "latest":
            path = run_dir / "checkpoints" / "weights" / f"step_{int(ckpt_num):06d}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"no FastWAM weights at {path} (checkpoint_num={ckpt_num})")
            return str(path)
        weights = sorted((run_dir / "checkpoints" / "weights").glob("step_*.pt"))
        if not weights:
            raise FileNotFoundError(
                f"no FastWAM weights under {run_dir}/checkpoints/weights "
                "(train first, or pass model_dir/checkpoint_path)"
            )
        return str(weights[-1])

    def _load_upstream_policy(self, checkpoint_path, dataset_stats_path, sim_cfg_name, sim_task, text_cache_dir=None):
        for path in (str(FASTWAM_ROOT), str(FASTWAM_SRC)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from experiments.robotwin.fastwam_policy.deploy_policy import get_model

        upstream_cfg = dict(self.model_cfg)
        upstream_cfg["replan_steps"] = self.replan_steps
        upstream_cfg["ckpt_setting"] = str(Path(str(checkpoint_path)).expanduser().resolve())
        upstream_cfg["dataset_stats_path"] = str(Path(str(dataset_stats_path)).expanduser().resolve())
        if sim_cfg_name is None:
            upstream_cfg.setdefault("sim_cfg_name", "sim_robotwin.yaml")
            upstream_cfg.setdefault("sim_task", "robotwin_uncond_3cam_384_1e-4")
        else:
            upstream_cfg["sim_cfg_name"] = sim_cfg_name
            upstream_cfg["sim_task"] = sim_task
        upstream_cfg["text_embedding_cache_dir"] = text_cache_dir
        if text_cache_dir:
            print(f"[FastWAM] text context from cache {text_cache_dir} (no T5 loaded)")
        policy = get_model(upstream_cfg)
        if self._cohub:
            # Swap the upstream three-view image builder for CoHuB's canvas.
            policy._build_robotwin_image_tensor = types.MethodType(
                _cohub_image_tensor_method(self.env_cfg_type, self._image_codec_crf), policy
            )
        return policy

    def _encode_cohub(self, obs: dict) -> dict[str, dict]:
        """One env observation -> per-policy upstream observations."""
        vision = obs["vision"]
        state = obs.get("cohub_state")
        if state is None:
            if not self.allow_dummy_policy:
                raise KeyError(
                    "obs has no 'cohub_state' -- the CoHuB env client publishes it; "
                    "this branch cannot run against a generic client"
                )
            # The debug client sends no cohub_state; use zeros.
            state = {
                robot: {"joint_pos": np.zeros(43, dtype=np.float32)}
                for robot in COHUB_ROBOTS[:2]
            }
        robots = _cohub_robots({"cohub_state": state})
        ego = {
            robot: _standardize_rgb(vision[COHUB_CAMERA_SLOT[robot]]["color"])
            for robot in robots
            if COHUB_CAMERA_SLOT[robot] in vision
        }
        encoded: dict[str, dict] = {}
        if self._decentralized:
            missing = [robot for robot in robots if robot not in ego]
            if missing:
                raise KeyError(
                    f"CoHuB observation has state but no ego camera for {missing}; "
                    f"available vision slots are {sorted(vision)}"
                )
            for robot in robots:
                encoded[robot] = {
                    "images": {"ego": ego[robot]},
                    "joint_action": {
                        # Copy: the decoded array may be a read-only view.
                        "vector": np.array(state[robot]["joint_pos"], dtype=np.float32, copy=True)
                    },
                    # not a policy input; carried for FASTWAM_DEBUG_DUMP
                    "pelvis_pose": np.asarray(state[robot].get("pelvis_pose", np.zeros(7)), dtype=np.float32),
                }
        else:
            if robots != COHUB_ROBOTS[:2]:
                raise ValueError(
                    "FastWAM centralized CoHuB checkpoints are two-robot/70D; "
                    f"the observation contains {robots}. Evaluate three-agent tasks decentralized."
                )
            encoded["duo"] = {
                "images": {"ego_a": ego["robot_a"], "ego_b": ego["robot_b"]},
                "joint_action": {
                    "vector": np.concatenate(
                        [
                            np.array(state["robot_a"]["joint_pos"], dtype=np.float32, copy=True),
                            np.array(state["robot_b"]["joint_pos"], dtype=np.float32, copy=True),
                        ]
                    )
                },
            }
        # Instructions travel with the observation so batched envs cannot mix them up.
        encoded[INSTRUCTIONS_KEY] = dict(obs.get("cohub_instruction") or {})
        return encoded

    def _cohub_chunks(self, per_policy_obs: dict[str, dict]) -> list[dict]:
        """Run every policy once and fold the chunks into `cohub_raw_action` steps."""
        if self.allow_dummy_policy:
            robots = tuple(target for target in per_policy_obs if target != INSTRUCTIONS_KEY)
            zero = {
                robot: _pack_single_robot_action(np.zeros(35, dtype=np.float32))
                for robot in robots
            }
            return [{"cohub_raw_action": dict(zero)} for _ in range(self.replan_steps)]

        wire = per_policy_obs.get(INSTRUCTIONS_KEY) or {}
        chunks = {}
        full = {}
        targets = tuple(target for target in per_policy_obs if target != INSTRUCTIONS_KEY)
        for target in targets:
            try:
                policy = self._policies[target]
            except KeyError as exc:
                raise KeyError(
                    f"FastWAM has no policy for {target}; loaded roles are {sorted(self._policies)}"
                ) from exc
            # Upstream seeds its generator from policy.seed; give each role and replan its own.
            policy.seed = self._next_policy_noise_seed(target)
            chunk = np.asarray(
                policy._infer_action_chunk(
                    per_policy_obs[target], self._cohub_instruction_for(target, wire)
                ),
                dtype=np.float32,
            )
            if chunk.ndim == 1:
                chunk = chunk[None, :]
            full[target] = chunk
            chunks[target] = chunk[: self.replan_steps]
        self._debug_dump(per_policy_obs, wire, full)

        steps = min(chunk.shape[0] for chunk in chunks.values())
        if self._decentralized:
            return [
                {
                    "cohub_raw_action": {
                        robot: _pack_single_robot_action(chunks[robot][t])
                        for robot in targets
                    }
                }
                for t in range(steps)
            ]
        return [{"cohub_raw_action": _pack_dual_arm_action(chunks["duo"][t])} for t in range(steps)]

    def seed(self, episode_seed: int) -> None:
        """Start the policy-noise schedule for an episode. Called from the client's `seed_episode` RPC."""
        self._policy_episode_seed = int(episode_seed)
        self._policy_replan_index = {}
        print(
            f"[FastWAM][cohub] policy noise: schedule={POLICY_NOISE_SCHEDULE} "
            f"episode_seed={self._policy_episode_seed} salt={self._policy_seed_salt}",
            flush=True,
        )

    def _next_policy_noise_seed(self, target: str) -> int:
        replan_index = self._policy_replan_index.get(target, 0)
        self._policy_replan_index[target] = replan_index + 1
        return _derive_policy_noise_seed(
            self._policy_episode_seed,
            target,
            replan_index,
            self._policy_seed_salt,
        )

    # RoboTwin/RoboDojo path

    def _encode_obs_for_fastwam(self, obs: dict) -> dict:
        vision = obs["vision"]
        adapted = {
            "observation": {
                "head_camera": {"rgb": _standardize_rgb(vision["cam_head"]["color"])},
                "left_camera": {"rgb": _standardize_rgb(vision["cam_left_wrist"]["color"])},
                "right_camera": {"rgb": _standardize_rgb(vision["cam_right_wrist"]["color"])},
            },
            "joint_action": {
                "vector": pack_robot_state(
                    obs,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                    state_type="state",
                ).astype(np.float32)
            },
        }
        return adapted

    def update_obs(self, obs):
        if self._cohub:
            self.last_obs = self._encode_cohub(obs)
            self._debug_dump_obs(self.last_obs)
            return
        self.last_obs = self._encode_obs_for_fastwam(obs)
        self.last_instruction = _get_instruction(obs, self.default_instruction)

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("update_obs_batch received an empty observation list.")
        if self._cohub:
            self._batch = {int(obs["env_idx"]): self._encode_cohub(obs) for obs in obs_list}
            return
        self._batch_obs = {}
        self._batch_instruction = {}
        for obs in obs_list:
            env_idx = int(obs["env_idx"])
            self._batch_obs[env_idx] = self._encode_obs_for_fastwam(obs)
            self._batch_instruction[env_idx] = _get_instruction(obs, self.default_instruction)

    def _zero_actions(self):
        dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        zeros = np.zeros((self.replan_steps, dim), dtype=np.float32)
        return unpack_robot_state(zeros, self.action_type, self.robot_action_dim_info, source_type="obs")

    def _infer_actions(self, obs, instruction):
        if self.allow_dummy_policy:
            return self._zero_actions()
        if obs is None:
            raise ValueError("No observation is available. Call update_obs() before get_action().")
        action_chunk = self.model._infer_action_chunk(obs, instruction)
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        action_chunk = action_chunk[:n_exec]
        return unpack_robot_state(action_chunk, self.action_type, self.robot_action_dim_info, source_type="obs")

    # FASTWAM_DEBUG_DUMP=<dir> writes every inference's inputs, full chunk and states as npz.
    def _debug_dump(self, per_policy_obs, wire, full):
        dump = os.environ.get("FASTWAM_DEBUG_DUMP")
        if not dump:
            return
        os.makedirs(dump, exist_ok=True)
        n = getattr(self, "_debug_calls", 0)
        self._debug_calls = n + 1
        payload = {"instruction": json.dumps(wire)}
        for target, obs in per_policy_obs.items():
            if target == INSTRUCTIONS_KEY:
                continue
            payload[f"state_{target}"] = obs["joint_action"]["vector"]
            if "pelvis_pose" in obs:
                payload[f"pelvis_{target}"] = obs["pelvis_pose"]
            for cam, img in obs["images"].items():
                payload[f"image_{target}_{cam}"] = img
            payload[f"chunk_{target}"] = full[target]
        np.savez(os.path.join(dump, f"ep{getattr(self, '_debug_episode', 0):03d}_call{n:04d}.npz"), **payload)

    def _debug_dump_obs(self, encoded):
        dump = os.environ.get("FASTWAM_DEBUG_DUMP")
        if not dump:
            return
        trace = self.__dict__.setdefault("_debug_states", [])
        trace.append({t: np.concatenate([np.asarray(o["joint_action"]["vector"]), np.asarray(o.get("pelvis_pose", np.zeros(7)))])
                      for t, o in encoded.items() if t != INSTRUCTIONS_KEY})

    def _debug_flush(self):
        dump = os.environ.get("FASTWAM_DEBUG_DUMP")
        trace = self.__dict__.pop("_debug_states", None)
        if dump and trace:
            ep = getattr(self, "_debug_episode", 0)
            np.savez(os.path.join(dump, f"ep{ep:03d}_states.npz"),
                     **{t: np.stack([row[t] for row in trace]) for t in trace[0]})
        self._debug_episode = getattr(self, "_debug_episode", 0) + 1
        self._debug_calls = 0

    def get_action(self):
        if self._cohub:
            if self.last_obs is None:
                raise ValueError("No observation is available. Call update_obs() before get_action().")
            return self._cohub_chunks(self.last_obs)
        return self._infer_actions(self.last_obs, self.last_instruction)

    def get_action_batch(self, env_idx_list):
        if self._cohub:
            if not self._batch:
                raise ValueError("No batch observation is available. Call update_obs_batch() first.")
            return [self._cohub_chunks(self._batch[int(env_idx)]) for env_idx in env_idx_list]
        if not hasattr(self, "_batch_obs"):
            raise ValueError("No batch observation is available. Call update_obs_batch() first.")
        return [
            self._infer_actions(self._batch_obs[int(env_idx)], self._batch_instruction[int(env_idx)])
            for env_idx in env_idx_list
        ]

    def reset(self):
        self.last_obs = None
        self.last_instruction = self.default_instruction
        if self._cohub:
            self._batch = {}
            self._policy_replan_index = {}
            self._debug_flush()
            for policy in self._policies.values():
                policy.reset()
            return
        if self.model is not None:
            self.model.reset()
