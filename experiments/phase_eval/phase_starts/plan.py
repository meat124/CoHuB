from __future__ import annotations

import json
import math

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCHEMA = 2

REPO_ROOT = Path(__file__).resolve().parents[3]
LABELS_DIR = Path(__file__).resolve().parent / "states"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "eval_results_phase"


def dataset_root(task_or_path: str) -> Path:
    given = Path(task_or_path).expanduser()
    return given.resolve() if given.exists() or "/" in task_or_path else REPO_ROOT / "datasets" / task_or_path


def labels_path(task_or_path: str) -> Path:
    return LABELS_DIR / f"{dataset_root(task_or_path).name}.json"

COLLABORATIVE = {
    "handover": {"grasp_lift": False, "transfer": True, "place": False},
    "pouring": {"lift_both": False, "tip": False, "pour": True},
    "copouring": {"grasp_lift": True, "tip": True, "pour": True},
    "framehang": {"grasp": False, "lift": True, "hang": True},
    "cocarry": {"grasp_lift": True, "carry": True, "release": True},
    "tablealign": {"grasp_lift": True, "carry": True, "align": True},
    "trashcollection": {"lift": False, "deliver": False, "sweep": True},
    "cartservice": {"park": False, "lift": True, "serve": False},
}

BUDGET_FACTOR = 2.0

START_MODES = ("auto", "lived", "restored")
DEFAULT_START = "auto"

LIVED_SLACK = 100

BUDGET_QUANTUM = 50

DEFAULT_WARM_STEPS = 0

INITIAL_FRAME = -1


def ceil_to(value: float, quantum: int) -> int:
    return int(math.ceil(value / quantum) * quantum)


def phase_lengths(chain: list[str], achieved: dict[str, int | None], success_frame: int | None) -> dict[str, int | None]:
    frames = [achieved.get(name) for name in chain[:-1]] + [success_frame]
    out: dict[str, int | None] = {}
    previous: int | None = INITIAL_FRAME
    for name, frame in zip(chain, frames):
        out[name] = None if frame is None or previous is None else frame - previous
        previous = frame
    return out


def phase_budgets(chain: list[str], lengths: list[dict[str, int | None]]) -> dict[str, int]:
    budgets: dict[str, int] = {}
    for name in chain:
        known = [row[name] for row in lengths if row.get(name) is not None]
        if not known:
            raise ValueError(f"no demonstration walked phase {name!r}; it cannot be given a budget")
        budgets[name] = ceil_to(BUDGET_FACTOR * max(known), BUDGET_QUANTUM)
    return budgets


@dataclass(frozen=True)
class Trial:
    demo: str
    episode_index: int
    phase: str
    phase_index: int
    boundary: int
    restore: int
    warm_steps: int
    step_budget: int
    start: str = "restored"

    @property
    def key(self) -> str:
        return f"{self.demo}:{self.phase}"

    def to_dict(self) -> dict:
        return asdict(self)


def warm_window(boundary: int, warm_steps: int) -> tuple[int, int]:
    if boundary <= INITIAL_FRAME:
        return INITIAL_FRAME, 0
    restore = max(INITIAL_FRAME, boundary - max(0, warm_steps))
    return restore, boundary - restore


def lived_available(row: dict, previous: str, boundary: int) -> bool:
    step = (row.get("lived") or {}).get(previous)
    return step is not None and step <= boundary + LIVED_SLACK


def plan_trials(labels: dict, phases: list[str] | None = None, demos: set[int] | None = None,
                warm_steps: int | None = None, start: str = DEFAULT_START) -> tuple[list[Trial], list[dict]]:
    if start not in START_MODES:
        raise ValueError(f"start={start!r}; expected one of {START_MODES}")
    chain = list(labels["chain"])
    wanted = chain if not phases else [name for name in chain if name in set(phases)]
    unknown = sorted(set(phases or ()) - set(chain))
    if unknown:
        raise ValueError(f"{unknown} not in this task's chain {chain}")
    warm = labels.get("warm_steps", DEFAULT_WARM_STEPS) if warm_steps is None else warm_steps
    trials: list[Trial] = []
    missing: list[dict] = []
    for row in labels["demos"]:
        if demos is not None and row["episode_index"] not in demos:
            continue
        for name in wanted:
            index = chain.index(name)
            if index == 0:
                boundary = INITIAL_FRAME
            else:
                boundary = row["achieved"].get(chain[index - 1])
                if boundary is None:
                    missing.append({"demo": row["demo"], "phase": name,
                                    "reason": f"demo never achieved {chain[index - 1]}"})
                    continue
                replayed = lived_available(row, chain[index - 1], boundary)
                if start == "lived" and not replayed:
                    missing.append({"demo": row["demo"], "phase": name,
                                    "reason": f"replay did not reach {chain[index - 1]}"})
                    continue
            if index == 0:
                mode = "initial"
            else:
                mode = "restored" if start == "restored" or not replayed else "lived"
            restore, steps = warm_window(boundary, warm)
            trials.append(Trial(row["demo"], row["episode_index"], name, index, boundary, restore, steps,
                                int(labels["budgets"][name]), mode))
    return trials, missing


def trials_digest(labels: dict, start: str = DEFAULT_START) -> str:
    import hashlib

    content = {
        "chain": list(labels["chain"]),
        "budgets": {name: int(labels["budgets"][name]) for name in labels["chain"]},
        "demos": [[row["demo"], row["episode_index"], [row["achieved"].get(n) for n in labels["chain"][:-1]]]
                  for row in labels["demos"]],
        "start": start,
    }
    if start != "restored":
        content["lived"] = [[lived_available(row, n, row["achieved"][n]) if row["achieved"].get(n) is not None else False
                             for n in labels["chain"][:-1]] for row in labels["demos"]]
    return f"{hashlib.sha1(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]}-{start}"


def load_labels(path) -> dict:
    labels = json.loads(Path(path).read_text())
    if labels.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema {labels.get('schema')!r}, this code reads {SCHEMA}")
    return labels


def grasp_traces(latched: np.ndarray, ever_latched: np.ndarray, ever_both: np.ndarray, release_age: np.ndarray,
                 step_dt: float, grip_time: float, robots: int) -> dict[str, np.ndarray]:
    held = latched > 0.5
    frames, hands = held.shape
    per_robot = held.reshape(frames, robots, hands // robots).any(axis=2)
    robot_age = np.zeros((frames, robots), np.float32)
    age = np.zeros(robots, np.float32)
    for k, row in enumerate(per_robot):
        age = np.where(row, 0.0, age + step_dt).astype(np.float32)
        robot_age[k] = age
    return {
        "latched": held,
        "ever_latched": ever_latched > 0.5,
        "ever_both_robots_latched": np.asarray(ever_both).reshape(frames) > 0.5,
        "release_age": np.asarray(release_age, np.float32).reshape(frames),
        "robot_release_age": robot_age,
        "contact_time": np.where(held, grip_time, 0.0).astype(np.float32),
        "no_contact_time": np.zeros((frames, hands), np.float32),
    }


def grasp_seed(traces: dict[str, np.ndarray], frame: int) -> dict[str, np.ndarray]:
    if frame <= INITIAL_FRAME:
        return {key: np.zeros_like(value[0]) for key, value in traces.items()}
    return {key: value[frame] for key, value in traces.items()}


def slew_trace(targets: np.ndarray, slew_per_step: float) -> np.ndarray:
    out = np.zeros(len(targets), np.float64)
    value = 0.0
    for k, target in enumerate(targets):
        value += float(np.clip(target - value, -slew_per_step, slew_per_step))
        out[k] = value
    return out


def _rotated_z(quat_xyzw: np.ndarray, x: float, y: float) -> np.ndarray:
    qx, qy, qz, qw = (quat_xyzw[:, i] for i in range(4))
    return 2.0 * (qx * qz - qw * qy) * x + 2.0 * (qy * qz + qw * qx) * y


def offline_cocarry_labels(latched: np.ndarray, object_pose: np.ndarray, params: dict) -> dict[str, int | None]:
    pos, quat = object_pose[:, :3], object_pose[:, 3:7]
    axis = int(params.get("goal_axis", 1))
    half_x, half_y = params.get("base_half_size", (0.160, 0.275))
    lowest = np.min([_rotated_z(quat, sx * half_x, sy * half_y) for sx in (-1, 1) for sy in (-1, 1)], axis=0)
    clearance = pos[:, 2] + lowest - float(params.get("goal_height", 0.314))
    held = latched > 0.5
    half = held.shape[1] // 2
    both = held[:, :half].any(axis=1) & held[:, half:].any(axis=1)
    at_goal = np.abs(pos[:, axis] - float(params.get("goal_distance", -1.0))) <= float(params.get("goal_axis_tolerance", 0.15))
    centred = np.abs(pos[:, 1 - axis]) <= float(params.get("goal_cross_axis_tolerance", 0.15))
    lifted = both & (clearance >= float(params.get("lift_margin_m", 0.03)))
    carried = at_goal & centred & both & (clearance >= float(params.get("carry_clearance_m", 0.01)))

    def first(mask: np.ndarray, start: int = 0) -> int | None:
        hits = np.flatnonzero(mask[start:])
        return int(hits[0]) + start if len(hits) else None

    lift = first(lifted)
    carry = first(carried, lift) if lift is not None else None
    return {"grasp_lift": lift, "carry": carry}


PACK_SCHEMA = 1
FAMILIES = ("articulation", "rigid_object")


def pack_path(task_or_path: str) -> Path:
    return LABELS_DIR / f"{Path(task_or_path).name}.states.npz"


def state_layout(node) -> list[list]:
    layout, offset = [], 0
    for family in FAMILIES:
        if family not in node:
            continue
        for name in sorted(node[family]):
            for field in sorted(node[family][name]):
                width = int(node[family][name][field].shape[-1])
                layout.append([family, name, field, offset, offset + width])
                offset += width
    return layout


def state_rows(node, layout: list[list], frames: np.ndarray | None) -> np.ndarray:
    columns = []
    for family, name, field, start, stop in layout:
        values = np.asarray(node[family][name][field], dtype=np.float32).reshape(-1, stop - start)
        columns.append(values if frames is None else values[frames])
    return np.concatenate(columns, axis=1)


class PackedFrames:

    def __init__(self, name: str, pack, layout: list[list], device: str, step_dt: float, grip_time: float,
                 robots: int) -> None:
        import torch

        self.name = name
        self._torch = torch
        self._device = device
        self._layout = layout
        self._frames = np.asarray(pack[f"{name}/frames"], dtype=np.int64)
        self._row = {int(frame): index + 1 for index, frame in enumerate(self._frames)}
        self._states = np.asarray(pack[f"{name}/states"], dtype=np.float32)
        self._object_pose = np.asarray(pack[f"{name}/object_pose"], dtype=np.float32)
        recorded = np.asarray(pack[f"{name}/grasp"], dtype=np.float32)
        self.steps = int(recorded.shape[0])
        hands = (recorded.shape[1] - 2) // 2
        self.grasp = grasp_traces(recorded[:, :hands], recorded[:, hands:2 * hands], recorded[:, 2 * hands],
                                  recorded[:, 2 * hands + 1], step_dt=step_dt, grip_time=grip_time, robots=robots)

    def assets(self) -> dict[str, set[str]]:
        found: dict[str, set[str]] = {family: set() for family in FAMILIES}
        for family, name, *_ in self._layout:
            found[family].add(name)
        return found

    def sweep_frames(self, stop: int) -> list[int]:
        return [int(frame) for frame in self._frames if frame <= stop]

    def _index(self, frame: int) -> int:
        if frame <= INITIAL_FRAME:
            return 0
        try:
            return self._row[int(frame)]
        except KeyError:
            raise KeyError(f"{self.name}: frame {frame} is not in the states pack (it keeps {len(self._frames)} of "
                           f"{self.steps}); rebuild it with experiments/phase_eval/phase_starts/build.py pack if the labels changed") from None

    def state(self, frame: int) -> dict:
        row = self._torch.as_tensor(self._states[self._index(frame)], device=self._device)
        out: dict = {}
        for family, name, field, start, stop in self._layout:
            out.setdefault(family, {}).setdefault(name, {})[field] = row[start:stop].reshape(1, -1)
        return out

    def root_pose(self, family: str, name: str, frame: int) -> np.ndarray:
        if frame > INITIAL_FRAME and int(frame) not in self._row:
            return self._object_pose[frame]
        row = self._states[self._index(frame)]
        for fam, asset, field, start, stop in self._layout:
            if (fam, asset, field) == (family, name, "root_pose"):
                return row[start:stop]
        raise KeyError(f"{self.name}: no {family} {name!r} in the states pack")

    def grasp_seed(self, frame: int) -> dict[str, np.ndarray]:
        return grasp_seed(self.grasp, frame)


class StartPack:

    kind = "pack"

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._pack = np.load(self.path, allow_pickle=False)
        self.meta = json.loads(str(self._pack["meta"]))
        if self.meta.get("schema") != PACK_SCHEMA:
            raise SystemExit(f"{self.path}: states pack schema {self.meta.get('schema')}, this code reads {PACK_SCHEMA}; "
                             "rebuild it with experiments/phase_eval/phase_starts/build.py pack")
        self.env_args = self.meta.get("env_args")
        self.names = list(self.meta["demos"])

    def __enter__(self) -> "StartPack":
        return self

    def __exit__(self, *exc) -> None:
        self._pack.close()

    def __contains__(self, name: str) -> bool:
        return name in self.meta["demos"]

    def frames(self, name: str, device: str, step_dt: float, grip_time: float, robots: int) -> PackedFrames:
        if name not in self:
            raise KeyError(f"{name} is not in {self.path.name}")
        return PackedFrames(name, self._pack, self.meta["layout"], device, step_dt, grip_time, robots)

    def check_labels(self, labels: dict) -> list[str]:
        problems = []
        chain = list(labels["chain"])
        for row in labels["demos"]:
            kept = self.meta["demos"].get(row["demo"])
            if kept is None:
                problems.append(f"{row['demo']} is labelled but not packed")
                continue
            have = set(kept["starts"])
            need = {row["achieved"].get(link) for link in chain[:-1]} - {None}
            if not need <= have:
                problems.append(f"{row['demo']}: the labels start trials at {sorted(need - have)}, which the pack does not hold")
        return problems
