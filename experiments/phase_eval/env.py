from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from cohub.g1.actions import (
    SQUAT_SLEW,
    MultiRobotAgileLowerBodyAction,
    SmoothedJointTargetAction,
    squat_bias_for_height,
)
from cohub.g1.grasp import grasp_term
from cohub.policy_env import CoHuBTaskEnv
from cohub.runtime import is_kinematic, reset_scene_to

from replay_actions_eval_env import _column, _episode_table, episode_actions

from phase_eval.phase_starts.plan import (
    FAMILIES,
    INITIAL_FRAME,
    LIVED_SLACK,
    START_MODES,
    StartPack,
    grasp_seed,
    grasp_traces,
    pack_path,
    slew_trace,
)


def episode_column(root, episode: int, name: str) -> np.ndarray:
    return _column(_episode_table(root, episode)[1], name)


def episode_joint_targets(root, episode: int) -> tuple[np.ndarray, list[str]]:
    info, table = _episode_table(root, episode)
    return _column(table, "action"), list(info["features"]["action"]["names"])


PER_ROBOT = 35
JOINT_TARGETS = 31
HEIGHT_COLUMN = 31


class DemoFrames:

    def __init__(self, group, device: str, step_dt: float, grip_time: float, robots: int) -> None:
        self.name = group.name.rsplit("/", 1)[-1]
        self.steps = int(group["actions"].shape[0])
        self._states = self._read(group["states"], device)
        self._initial = self._read(group["initial_state"], device)
        recorded = group["states"]["grasp"]
        self.grasp = grasp_traces(
            np.asarray(recorded["latched"]), np.asarray(recorded["ever_latched"]),
            np.asarray(recorded["ever_both_robots_latched"]), np.asarray(recorded["release_age"]),
            step_dt=step_dt, grip_time=grip_time, robots=robots,
        )

    @staticmethod
    def _read(node, device: str) -> dict:
        return {
            family: {
                name: {
                    field: torch.as_tensor(np.asarray(node[family][name][field]), dtype=torch.float32, device=device)
                    for field in node[family][name]
                }
                for name in node[family]
            }
            for family in FAMILIES
            if family in node
        }

    def assets(self) -> dict[str, set[str]]:
        return {family: set(self._states.get(family, {})) for family in FAMILIES}

    def sweep_frames(self, stop: int) -> list[int]:
        return list(range(0, stop + 1))

    def state(self, frame: int) -> dict:
        if frame <= INITIAL_FRAME:
            return {f: {n: {k: v.reshape(1, -1) for k, v in fields.items()} for n, fields in names.items()}
                    for f, names in self._initial.items()}
        return {f: {n: {k: v[frame].reshape(1, -1) for k, v in fields.items()} for n, fields in names.items()}
                for f, names in self._states.items()}

    def root_pose(self, family: str, name: str, frame: int) -> np.ndarray:
        source = self._initial if frame <= INITIAL_FRAME else self._states
        pose = source[family][name]["root_pose"]
        return (pose.reshape(-1) if frame <= INITIAL_FRAME else pose[frame]).cpu().numpy()

    def grasp_seed(self, frame: int) -> dict[str, np.ndarray]:
        return grasp_seed(self.grasp, frame)


class ShardStates:

    kind = "shards"

    def __init__(self, data_dir) -> None:
        from cohub.recording.demo_source import DemoSource

        self.path = data_dir
        self._source = DemoSource(data_dir)
        self.env_args = None

    def __enter__(self) -> "ShardStates":
        self._source.__enter__()
        self.env_args = self._source.attrs.get("env_args")
        return self

    def __exit__(self, *exc) -> None:
        self._source.__exit__(*exc)

    def __contains__(self, name: str) -> bool:
        return name in self._source

    def frames(self, name: str, device: str, step_dt: float, grip_time: float, robots: int) -> DemoFrames:
        return DemoFrames(self._source[name], device, step_dt, grip_time, robots)


def open_states(dataset, labels: dict, prefer: str = "auto"):
    pack_file, data_dir = pack_path(str(dataset)), Path(dataset) / "data"
    have_shards = data_dir.is_dir() and any(data_dir.glob("*.hdf5"))
    if prefer not in ("auto", "pack", "shards"):
        raise ValueError(f"states={prefer!r}; expected auto, pack or shards")
    if prefer == "shards" or (prefer == "auto" and not pack_file.is_file()):
        if not have_shards:
            raise SystemExit(f"no states pack at {pack_file} and no state shards under {data_dir}: a phase-start "
                             "trial has no recorded state to begin from. `git pull` brings the pack; "
                             "experiments/phase_eval/phase_starts/build.py pack builds one where the shards are.")
        return ShardStates(data_dir)
    if not pack_file.is_file():
        raise SystemExit(f"no states pack at {pack_file}")
    pack = StartPack(pack_file)
    problems = pack.check_labels(labels)
    if problems:
        raise SystemExit(f"{pack_file.name} was not cut for these labels:\n  " + "\n  ".join(problems[:10])
                         + "\nrebuild it with experiments/phase_eval/phase_starts/build.py pack")
    return pack


def check_restorable(env, frames: DemoFrames, env_args: str | bytes | None = None) -> list[str]:
    problems = []
    recorded = frames.assets()
    scene = env.scene
    for family, assets in (("articulation", scene.articulations), ("rigid_object", scene.rigid_objects)):
        for name, asset in assets.items():
            # reset_scene_to puts an unrecorded kinematic body at its default pose.
            if name not in recorded[family] and not is_kinematic(asset):
                problems.append(f"scene has {family} {name!r}, which {frames.name} recorded no state for")
    if env_args:
        try:
            listed = (json.loads(env_args) or {}).get("articulation") or {}
        except (TypeError, ValueError):
            listed = {}
        for name, entry in listed.items():
            names = list(entry.get("joint_names") or [])
            if names and name in scene.articulations and names != list(scene.articulations[name].data.joint_names):
                problems.append(f"{name}: the dataset's joint order is not this articulation's")
    return problems


def write_grasp(term, seed: dict[str, np.ndarray]) -> None:
    device = term.device

    def put(attribute: str, key: str, dtype) -> None:
        current = getattr(term, attribute)
        value = torch.as_tensor(np.asarray(seed[key]), dtype=dtype, device=device).reshape(current.shape)
        setattr(term, attribute, value.clone())

    put("latched", "latched", torch.bool)
    put("ever_latched", "ever_latched", torch.bool)
    put("ever_both_robots_latched", "ever_both_robots_latched", torch.bool)
    put("release_age", "release_age", torch.float32)
    put("robot_release_age", "robot_release_age", torch.float32)
    put("_contact_time", "contact_time", torch.float32)
    put("_no_contact_time", "no_contact_time", torch.float32)


def lower_body_terms(env) -> dict[str, MultiRobotAgileLowerBodyAction]:
    terms = {}
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        if isinstance(term, MultiRobotAgileLowerBodyAction):
            terms[term.cfg.asset_name] = term
    return terms


def squat_biases(warm_actions: np.ndarray, robots: list[str], step_dt: float) -> dict[str, np.ndarray]:
    out = {}
    for index, robot in enumerate(robots):
        heights = warm_actions[:, index * PER_ROBOT + HEIGHT_COLUMN]
        out[robot] = slew_trace(squat_bias_for_height(heights), SQUAT_SLEW * step_dt)
    return out


def upper_body_terms(env) -> dict[str, SmoothedJointTargetAction]:
    terms = {}
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        if isinstance(term, SmoothedJointTargetAction):
            terms[term.cfg.asset_name] = term
    return terms


class DemoCommands:

    def __init__(self, lerobot_root, episode: int, robots: list[str], step_dt: float) -> None:
        self.robots = list(robots)
        self.actions = episode_actions(lerobot_root, episode)
        self._targets, names = episode_joint_targets(lerobot_root, episode)
        self._column = {name: i for i, name in enumerate(names)}
        self.biases = squat_biases(self.actions, self.robots, step_dt)

    def hidden(self, env, frame: int) -> dict:
        row = max(frame, 0)
        hidden: dict = {"squat_bias": {}, "loco_raw": {}, "joint_targets": None}
        for robot, term in lower_body_terms(env).items():
            bias = 0.0 if frame <= INITIAL_FRAME else float(self.biases[robot][frame])
            hidden["squat_bias"][robot] = bias
            columns = [self._column[f"{robot}/{joint}"] for joint in term._joint_names]
            target = torch.as_tensor(self._targets[row, columns], dtype=torch.float32, device=term.device).reshape(1, -1)
            offset = term._policy_output_offset[:1] + term._squat_offset(torch.full((1,), bias, device=term.device))[:1]
            hidden["loco_raw"][robot] = (target - offset) / term._policy_output_scale
        if frame > INITIAL_FRAME:
            hidden["joint_targets"] = {
                robot: self.actions[frame, i * PER_ROBOT : i * PER_ROBOT + JOINT_TARGETS]
                for i, robot in enumerate(self.robots)
            }
        return hidden


def write_hidden(env, hidden: dict | None) -> None:
    hidden = hidden or {}
    for robot, term in lower_body_terms(env).items():
        term._squat_bias.fill_(float((hidden.get("squat_bias") or {}).get(robot, 0.0)))
        raw = (hidden.get("loco_raw") or {}).get(robot)
        if raw is None:
            term._raw_actions.zero_()
        else:
            term._raw_actions.copy_(torch.as_tensor(raw, dtype=torch.float32, device=term.device).reshape(1, -1))
    for robot, term in upper_body_terms(env).items():
        smoothed = (hidden.get("smoothed") or {}).get(robot)
        targets = (hidden.get("joint_targets") or {}).get(robot)
        if smoothed is None and targets is not None:
            raw = torch.as_tensor(np.asarray(targets), dtype=torch.float32, device=term.device).reshape(1, -1)
            term.process_actions(raw)
            smoothed = term.processed_actions
        if smoothed is not None:
            term._smoothed.copy_(smoothed)
            term._seeded.fill_(True)


def zero_velocity_targets(scene, env_ids) -> None:
    for articulation in scene.articulations.values():
        target = torch.zeros_like(articulation.data.joint_vel.torch[env_ids])
        articulation.set_joint_velocity_target_index(target=target, env_ids=env_ids)
    scene.write_data_to_sim()


def term_memory(env, tracker=None) -> dict:
    manager = env.termination_manager
    memory: dict = {}
    for name, cfg in zip(manager._term_names, manager._term_cfgs):
        term = cfg.func
        if not hasattr(term, "__dict__") or isinstance(term, type) or callable(term) and not hasattr(term, "reset"):
            continue
        for attr, value in vars(term).items():
            if isinstance(value, torch.Tensor):
                memory[f"{name}.{attr}"] = value.detach().clone()
            elif isinstance(value, (bool, int, float, str)) or value is None:
                memory[f"{name}.{attr}"] = value
    if tracker is not None:
        memory["tracker.achieved"] = json.dumps(tracker.achieved(), sort_keys=True, default=str)
        memory["tracker.details"] = json.dumps(tracker.detail_result(), sort_keys=True, default=str)
    return memory


def memory_differs(a: dict, b: dict) -> list[str]:
    out = []
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if x.shape != y.shape or not torch.equal(x, y):
                out.append(key)
        elif isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor) or x != y:
            out.append(key)
    return out


def sweep_terms(env, frames: DemoFrames, stop: int, on_frame=None, start: int = 0) -> None:
    env_ids = torch.tensor([0], device=env.device)
    grasp = grasp_term(env)
    for frame in frames.sweep_frames(stop):
        if frame < start:
            continue
        reset_scene_to(env.scene, frames.state(frame), env_ids, is_relative=True)
        write_grasp(grasp, frames.grasp_seed(frame))
        env.termination_manager.compute()
        if on_frame is not None:
            on_frame(frame)


def restore_frame(env, frames, frame: int, hidden: dict | None = None) -> None:
    env_ids = torch.tensor([0], device=env.device)
    env.action_manager.reset(env_ids)
    reset_scene_to(env.scene, frames.state(frame), env_ids, is_relative=True)
    zero_velocity_targets(env.scene, env_ids)
    write_grasp(grasp_term(env), frames.grasp_seed(frame))
    write_hidden(env, hidden)
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    env.observation_manager.reset(env_ids)
    env.obs_buf = env.observation_manager.compute(update_history=True)


class PhaseEvalEnv(CoHuBTaskEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_phase: str | None = None
        self._start: dict | None = None

    def reset(self, seed: int | None = None, settle: bool = True, video_stem: str | None = None) -> None:
        super().reset(seed=seed, settle=settle, video_stem=video_stem)
        self._target_phase = None
        self._start = None

    def take_action(self, action: dict) -> None:
        super().take_action(action)
        if not self._done and self._target_reached():
            self._done = True
            self._done_reason = "phase_success"
            self._snapshot_final_state()

    def _target_reached(self) -> bool:
        target, tracker = self._target_phase, self._tracker
        if target is None or tracker is None or target == tracker.names[-1]:
            return False
        return tracker.achieved().get(target) is not None

    def _warm_step(self, row: np.ndarray) -> bool:
        from replay_actions_eval_env import pack_robots

        with torch.no_grad():
            _, _, terminated, truncated, _ = self.env.step(self._action_tensor(pack_robots(row, self.robots)))
        return bool(terminated[0]) or bool(truncated[0])

    def start_from_demo(
        self,
        frames,
        target: str,
        boundary: int,
        restore: int,
        commands,
        step_budget: int,
        start: str = "lived",
        slack: int = 10,
        lived_slack: int | None = None,
        position_tolerance: float = 0.05,
    ) -> dict:
        if start not in START_MODES:
            raise ValueError(f"start={start!r}; expected one of {START_MODES}")
        lived_slack = LIVED_SLACK if lived_slack is None else lived_slack
        index = list(self._tracker.names).index(target)
        if index == 0:
            modes = ["restored"]
        else:
            modes = {"auto": ["lived", "restored"], "lived": ["lived"], "restored": ["restored"]}[start]
        t0 = time.perf_counter()
        tried = []
        for mode in modes:
            info = self._establish_start(frames, target, boundary, restore, commands, mode,
                                         slack, lived_slack, position_tolerance)
            if index == 0:
                info["start_mode"] = mode = "initial"
            tried.append(f"{mode}:{info['start_reason'] or 'held'}")
            if info["start_valid"]:
                break
        invalid = info["start_reason"]

        self._obs_cache = None
        self._obs_cache_step = -1
        self._frame_memo = {}
        self._decoded_memo = {}
        self._latest_obs_frame = {}
        self._follow_states.clear()
        self._settle_render()
        if self._follow_active():
            self._update_follow_camera()

        self.episode_step = 0
        self.episode_step_limit = int(step_budget)
        self._target_phase = target
        self._done = invalid is not None
        self._done_reason = f"start_invalid:{invalid}" if invalid else None
        if invalid:
            self._snapshot_final_state()
        info.update(start_requested=start, start_attempts=tried, start_seconds=round(time.perf_counter() - t0, 2))
        self._start = info
        return info

    def _establish_start(self, frames, target, boundary, restore, commands, mode,
                         slack, lived_slack, position_tolerance) -> dict:
        env, tracker = self.env, self._tracker
        names = list(tracker.names)
        index = names.index(target)
        previous = names[index - 1] if index else None
        env_ids = torch.tensor([0], device=self.device)
        lived = mode == "lived"
        origin = -1 if lived else restore
        invalid: str | None = None
        with torch.no_grad():
            env.termination_manager.reset(env_ids)
            tracker.reset()
            env.episode_length_buf[:] = 0
            if origin >= 0:
                sweep_terms(env, frames, origin, lambda k: tracker.update(env, step=0, terminal=False))
            restore_frame(env, frames, origin, commands.hidden(env, origin))

        def reached() -> bool:
            return previous is None or tracker.achieved()[previous] is not None

        warm_actions = commands.actions
        handover = origin
        last_row = min(boundary + (lived_slack if lived else slack), len(warm_actions) - 1)
        while handover < last_row:
            if handover >= boundary and reached():
                break
            handover += 1
            if self._warm_step(warm_actions[handover]):
                invalid = f"{self._classify_done(False)}_in_replay"
                break
            tracker.update(env, step=0, terminal=False)

        live = reached()
        if previous is not None and not live:
            if lived:
                invalid = invalid or f"replay_did_not_reach_{previous}"
            else:
                tracker.force(previous, 0)
        deviation, robots_match, hands_match = 0.0, True, True
        if invalid is None:
            grasp = grasp_term(env)
            held = grasp.latched[0].cpu().numpy().astype(bool)
            recorded = frames.grasp_seed(min(handover, frames.steps - 1))["latched"].astype(bool)
            per_robot = len(held) // len(self.robots)
            robots_match = bool(
                (held.reshape(-1, per_robot).any(axis=1) == recorded.reshape(-1, per_robot).any(axis=1)).all()
            )
            hands_match = bool((held == recorded).all())
            carried = env.scene[grasp.cfg.asset_name]
            position = (carried.data.root_pos_w.torch[0] - env.scene.env_origins[0]).cpu().numpy()
            deviation = float(np.linalg.norm(
                position - frames.root_pose("rigid_object", grasp.cfg.asset_name, min(handover, frames.steps - 1))[:3]))
            if self._target_reached() or (target == names[-1] and self._success()):
                invalid = "target_achieved_before_handover"
            elif not lived and not robots_match:
                invalid = "hold_differs_from_recording"
            elif not lived and deviation > position_tolerance:
                invalid = "object_off_recording"
        return {
            "demo": frames.name,
            "start_mode": mode,
            "boundary_frame": boundary,
            "restore_frame": origin,
            "handover_frame": handover,
            "warm_steps": handover - origin,
            "start_valid": invalid is None,
            "start_reason": invalid,
            "prev_phase": previous,
            "prev_achieved_live": live,
            "object_pos_dev_m": round(deviation, 4),
            "robots_hold_match": robots_match,
            "hands_latched_match": hands_match,
        }

    def episode_result(self) -> dict:
        result = super().episode_result()
        if self._target_phase is not None:
            final = self._target_phase == self._tracker.names[-1]
            reached = bool(result["success"]) if final else bool(result["subtasks"].get(self._target_phase))
            result.update(
                target_phase=self._target_phase,
                target_success=reached,
                steps_to_target=(result["subtask_steps"].get(self._target_phase) if reached else None),
                step_budget=self.episode_step_limit,
                start=self._start,
            )
        return result

    def _close_video_writer(self) -> None:
        if not self._video_writers:
            return
        for writer in self._video_writers.values():
            writer.close()
        self._video_writers = {}
        if not self._video_paths:
            return
        result = self.episode_result()
        verdict = "SUCCESS" if result.get("target_success", result["success"]) else f"FAIL-{result['reason']}"
        for path in self._video_paths.values():
            final = path.with_name(
                f"{path.name[: -len('_RUNNING.mp4')]}_{verdict}_{result['steps']}steps.mp4"
            )
            path.replace(final)
        self._video_paths = {}
