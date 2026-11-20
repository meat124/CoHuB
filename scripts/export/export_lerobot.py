"""Export a recorded dataset as LeRobot v2.1, the format the baselines train on. Needs no simulator.

    python scripts/export/export_lerobot.py datasets/cocarry/data
    python scripts/export/export_lerobot.py datasets/cocarry/data --demos 0-9 --out /tmp/try

Writes parquet rows, MP4 video and ``meta/``. Joints are reordered to URDF
group order, wrist poses become pelvis-relative wxyz, and depth is packed as
millimetres in a lossless MP4.
"""

import argparse
import json
import pathlib
import shutil
import sys


import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from cohub import keys as cohub_keys  # noqa: E402

from cohub.recording.demo_source import (  # noqa: E402
    DemoSource,
    dataset_root,
    demo_fps,
    parse_selection,
    collected_stamp,
    restamp,
)
from cohub.recording.demo_video import (  # noqa: E402
    DEPTH_INVALID,
    DEPTH_UNITS_PER_METRE,
    open_lossless_writer,
    open_writer,
    pack_depth,
)
from cohub.g1.geometry import WRIST_CHAIN, named_angles, wrist_quat_xyzw  # noqa: E402
from cohub.keys import URDF_GROUPS  # noqa: E402

CHUNKS_SIZE = 1000
"""Episodes per chunk directory. LeRobot's default."""

CAMERAS = ("ego_a", "ego_b", "scene")
VIDEO_COLUMN = "observation.images.{camera}"
DEPTH_COLUMN = "observation.depth.{camera}"

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

ROBOTS = cohub_keys.ROBOTS
JOINTS_PER_ROBOT = cohub_keys.JOINTS_PER_ROBOT

ROBOT_NAMES = cohub_keys.ROBOTS3
"""Every robot a recording can hold, in action vector order. See :func:`recorded_robots`."""


def recorded_robots(demo) -> tuple[str, ...]:
    """The robots one demo recorded, as a prefix of :data:`ROBOT_NAMES`."""
    robots = tuple(robot for robot in ROBOT_NAMES if f"obs/{robot}_joint_pos" in demo)
    if len(robots) < 2 or robots != ROBOT_NAMES[: len(robots)]:
        raise SystemExit(f"recorded robots {robots} are not robot_a, robot_b[, robot_c]")
    return robots


EXTRA_SCENE_CAMERAS = ("scene_2",)
"""Fixed views a task may record beside ``scene``. Exported when the first demo has them."""


def cameras_for(robots: tuple[str, ...], demo=None) -> tuple[str, ...]:
    """One ego camera per robot, the fixed scene camera, then any extra fixed view ``demo`` recorded."""
    extra = tuple(c for c in EXTRA_SCENE_CAMERAS if demo is not None and f"obs/images/{c}/rgb" in demo)
    return (*(f"ego_{robot.removeprefix('robot_')}" for robot in robots), "scene", *extra)


def annotation_columns_for(robots: tuple[str, ...]) -> tuple[str, ...]:
    """The shared instruction column, then one per robot."""
    return ("annotation.human.task_description", *(f"annotation.human.task_description_{robot}" for robot in robots))


# Per-robot slice of the recorded action (32 columns per robot).
ROBOT_STRIDE = 32
ACTION_LEFT_WRIST = slice(0, 7)
ACTION_RIGHT_WRIST = slice(7, 14)
ACTION_NAVIGATE = slice(28, 31)
ACTION_BASE_HEIGHT = slice(31, 32)

MIN_STEPS = cohub_keys.ACTION_HORIZON
"""An episode shorter than the action chunk yields no training window."""

from cohub.task_registry import TASK_TEXT  # noqa: E402


def urdf_order(joint_names: list[str]) -> tuple[np.ndarray, list[str], dict[str, tuple[int, int]]]:
    """Permutation from the recorded joint order to the URDF group order.

    Returns ``(perm, ordered_names, group_bounds)`` where
    ``reordered[:, i] == recorded[:, perm[i]]`` and ``group_bounds`` maps a
    group name to its ``(start, end)`` in the reordered vector.
    """
    lookup = {name: i for i, name in enumerate(joint_names)}
    if len(lookup) != len(joint_names):
        raise SystemExit("articulation has duplicate joint names; cannot map columns")

    perm: list[int] = []
    ordered: list[str] = []
    bounds: dict[str, tuple[int, int]] = {}
    for group, names in URDF_GROUPS:
        start = len(perm)
        for name in names:
            if name not in lookup:
                raise SystemExit(f"articulation has no joint {name!r}; is this a G1 with Dex3 hands?")
            perm.append(lookup[name])
            ordered.append(name)
        bounds[group] = (start, len(perm))

    if len(perm) != len(joint_names):
        unclaimed = sorted(set(joint_names) - set(ordered))
        raise SystemExit(f"{len(unclaimed)} joint(s) not in any URDF group: {unclaimed}")

    buckets = {"legs": 0, "waist": 0, "arms": 0, "hands": 0}
    for name in joint_names:
        if "_hand_" in name:
            buckets["hands"] += 1
        elif "hip" in name or "knee" in name or "ankle" in name:
            buckets["legs"] += 1
        elif name.startswith("waist"):
            buckets["waist"] += 1
        elif "shoulder" in name or "elbow" in name or "wrist" in name:
            buckets["arms"] += 1
        else:
            raise SystemExit(f"joint {name!r} belongs to no known group")
    if buckets != {"legs": 12, "waist": 3, "arms": 14, "hands": 14}:
        raise SystemExit(f"unexpected joint partition {buckets}; expected 12/3/14/14")

    return np.asarray(perm, dtype=np.int64), ordered, bounds


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Scalar-last to scalar-first. The recording is xyzw, GR00T's G1 is wxyz."""
    return np.concatenate([quat[..., 3:4], quat[..., 0:3]], axis=-1)


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion, xyzw."""
    out = np.array(quat, dtype=np.float64, copy=True)
    out[..., 0:3] *= -1.0
    return out


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternion arrays."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by the inverse of ``quat`` (xyzw)."""
    conj = quat_conjugate(quat)
    padded = np.concatenate([vec, np.zeros(vec.shape[:-1] + (1,))], axis=-1)
    return quat_multiply(quat_multiply(conj, padded), quat)[..., 0:3]


def continuous_sign(quat: np.ndarray) -> np.ndarray:
    """Flip whole rows so consecutive quaternions never jump hemisphere."""
    out = np.array(quat, dtype=np.float64, copy=True)
    for k in range(1, len(out)):
        if float(np.dot(out[k], out[k - 1])) < 0.0:
            out[k] *= -1.0
    return out


def normalised(quat: np.ndarray) -> np.ndarray:
    """Unit-norm a quaternion array."""
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.where(norm == 0.0, 1.0, norm)


def pelvis_relative_pose(pos_w: np.ndarray, quat_w_xyzw: np.ndarray, root_pos: np.ndarray, root_quat_xyzw: np.ndarray) -> np.ndarray:
    """World pose -> pelvis-relative pose, xyz + wxyz. Shape (T, 7)."""
    root = normalised(root_quat_xyzw)
    rel_pos = quat_rotate_inverse(root, pos_w - root_pos)
    rel_quat = normalised(quat_multiply(quat_conjugate(root), normalised(quat_w_xyzw)))
    return np.concatenate([rel_pos, quat_xyzw_to_wxyz(continuous_sign(rel_quat))], axis=-1)


def images_stamp(env_args: dict) -> dict:
    """Who rendered the frames of the source, and which heads were hidden.

    Must be read before the export restamps ``env_args``.
    """
    block = (env_args.get("provenance") or {})
    return {
        "rendered_by": block.get("written_by", ""),
        "rendered_at": block.get("written_at", ""),
        "commit": block.get("commit", ""),
        "hidden_heads": block.get("hidden_heads"),
    }


def task_sentences(env_name: str, override: str, robots: tuple[str, ...] = ROBOTS) -> tuple[str, ...]:
    """A task's instructions: the shared one, then one per robot."""
    if override:
        return (override,) * (1 + len(robots))
    if env_name in TASK_TEXT:
        sentences = TASK_TEXT[env_name]
        if len(sentences) != 1 + len(robots):
            raise SystemExit(
                f"{env_name} has {len(sentences)} instruction(s) on file for {len(robots)} recorded robots; "
                f"expected the shared one plus one per robot ({1 + len(robots)})"
            )
        return sentences
    raise SystemExit(
        f"no instruction on file for {env_name!r}. Only registered ids have sentences, and a "
        "shard collected before a rename names the old id in its own env_args -- pass "
        "--task-id with the id that task carries now to get its real per-robot sentences, or "
        "--task-text to override all of them with one."
    )


# processed_actions layout

_IK_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
_IK_HAND_JOINTS = (
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_thumb_2_joint",
)
_LEG_JOINTS = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)

def action_term_layout(robots: tuple[str, ...]) -> tuple:
    """Which joint each column of ``processed_actions`` is a target for.

    Every task declares, per robot and in robot order, an upper-body term (the
    IK joints, then the hands) and a lower-body term (the legs). The export
    checks the widths and the joint names against the file.
    """
    layout = []
    for robot in robots:
        suffix = robot.rsplit("_", 1)[1]
        layout.append((f"upper_body_{suffix}", robot, (*_IK_JOINTS, *_IK_HAND_JOINTS)))
        layout.append((f"lower_body_{suffix}", robot, _LEG_JOINTS))
    return tuple(layout)


def action_columns(layout: tuple, joint_names: dict[str, list[str]], width: int) -> dict[str, np.ndarray]:
    """Map each robot's URDF-ordered joints to their ``processed_actions`` columns.

    Returns ``{robot: (43,) column indices}`` such that
    ``processed_actions[:, columns[robot]]`` is that robot's joint targets in
    the same order :func:`urdf_order` puts its measured joint angles.
    """
    position: dict[str, dict[str, int]] = {robot: {} for robot in joint_names}
    offset = 0
    for term_name, asset, names in layout:
        if asset not in position:
            raise SystemExit(f"action term {term_name!r} drives unknown asset {asset!r}")
        for name in names:
            if name in position[asset]:
                raise SystemExit(f"{asset} joint {name!r} is driven by two action terms")
            position[asset][name] = offset
            offset += 1
    if offset != width:
        raise SystemExit(
            f"processed_actions is {width} wide but the action layout accounts for {offset}. "
            "The task's action terms are not the ones action_term_layout() describes."
        )

    columns = {}
    for robot, names in joint_names.items():
        mapped = position.get(robot, {})
        if not mapped:
            # An articulation that no action term drives is scene state (e.g. a door).
            print(f"  {robot}: no action column drives it -- scene articulation, not part of the action")
            continue
        if set(mapped) != set(names):
            missing = sorted(set(names) - set(mapped))
            extra = sorted(set(mapped) - set(names))
            raise SystemExit(
                f"{robot}: action layout and articulation disagree. "
                f"missing {missing[:4]}{'...' if len(missing) > 4 else ''}, "
                f"unexpected {extra[:4]}{'...' if len(extra) > 4 else ''}. "
                "The task's action terms are not the ones action_term_layout() describes."
            )
        ordered = urdf_order(names)[1]
        columns[robot] = np.asarray([mapped[name] for name in ordered], dtype=np.int64)
    return columns


# Reading one demo


def _demo_index(name: str) -> int:
    return int(name.split("_")[1])


def precheck(source: DemoSource, selection: set[int] | None) -> dict:
    """Check everything that would make the export wrong, before it starts.

    A failing demo stops the export. Returns the facts the export needs.
    """
    names = [n for n in source.names if selection is None or _demo_index(n) in selection]
    if not names:
        raise SystemExit("no demos selected")

    robots = recorded_robots(source[names[0]])
    cameras = cameras_for(robots, source[names[0]])
    env_args = json.loads(source.attrs.get("env_args", "{}") or "{}")
    joint_names = {
        robot: list(entry.get("joint_names", []))
        for robot, entry in (env_args.get("articulation") or {}).items()
    }
    for robot in robots:
        if len(joint_names.get(robot, [])) != JOINTS_PER_ROBOT:
            raise SystemExit(
                f"env_args['articulation']['{robot}'] does not list {JOINTS_PER_ROBOT} joints. "
                "Without the names the columns cannot be identified; migrate the dataset first."
            )

    problems: list[str] = []
    fps_seen: list[tuple[str, float, str]] = []
    shapes: set[tuple[int, int]] = set()
    has_wrench = True
    action_width = None
    depth_cameras: set[str] | None = None
    intrinsics: dict[str, list[list[float]]] = {}
    has_camera_pose = True
    eef_rot_seen: set[bool] = set()

    for name in names:
        demo = source[name]
        steps = int(demo.attrs.get("num_samples", 0))
        if steps < MIN_STEPS:
            problems.append(f"{name}: {steps} steps, fewer than the {MIN_STEPS}-step action chunk")
        found_robots = tuple(robot for robot in ROBOT_NAMES if f"obs/{robot}_joint_pos" in demo)
        if found_robots != robots:
            problems.append(f"{name}: records {found_robots}, the first demo records {robots}")
        for key, width in (("actions", ROBOT_STRIDE * len(robots)), ("processed_actions", None)):
            if key not in demo:
                problems.append(f"{name}: no {key}")
            elif width is not None and demo[key].shape[1] != width:
                problems.append(f"{name}: {key} is {demo[key].shape[1]} wide, expected {width}")
        if "processed_actions" in demo:
            found = int(demo["processed_actions"].shape[1])
            if action_width is None:
                action_width = found
            elif found != action_width:
                problems.append(f"{name}: processed_actions is {found} wide, others are {action_width}")
        for robot in robots:
            key = f"obs/{robot}_joint_pos"
            if key not in demo:
                problems.append(f"{name}: no {key}")
            elif demo[key].shape[1] != JOINTS_PER_ROBOT:
                problems.append(f"{name}: {key} is {demo[key].shape[1]} wide")
            for side in ("left", "right"):
                if f"obs/{robot}_{side}_eef_pos" not in demo:
                    problems.append(f"{name}: no obs/{robot}_{side}_eef_pos")
                # `*_eef_rot` is optional, but must be in every demo or in none.
                eef_rot_seen.add(f"obs/{robot}_{side}_eef_rot" in demo)
            if f"eef_wrench/{robot}/force" not in demo:
                has_wrench = False
        seen_depth = set()
        for camera in cameras:
            key = f"obs/images/{camera}/rgb"
            if key not in demo:
                problems.append(f"{name}: no {key}")
            else:
                shape = demo[key].shape
                if shape[0] != steps:
                    problems.append(f"{name}: {key} has {shape[0]} frames for {steps} steps")
                shapes.add((int(shape[1]), int(shape[2])))
            depth_key = f"obs/images/{camera}/depth"
            if depth_key in demo:
                seen_depth.add(camera)
                shape = demo[depth_key].shape
                if shape[0] != steps:
                    problems.append(f"{name}: {depth_key} has {shape[0]} frames for {steps} steps")
                shapes.add((int(shape[1]), int(shape[2])))
        # Depth cameras must be the same in every demo.
        if depth_cameras is None:
            depth_cameras = seen_depth
        elif seen_depth != depth_cameras:
            problems.append(
                f"{name}: depth for {sorted(seen_depth) or 'nothing'}, "
                f"other demos have {sorted(depth_cameras) or 'nothing'}"
            )

        # Intrinsics must not change within or across demos.
        if "camera_info" not in demo:
            has_camera_pose = False
        else:
            for camera in cameras:
                matrix = demo.get(f"camera_info/{camera}/intrinsics")
                if matrix is None or f"camera_info/{camera}/pose" not in demo:
                    has_camera_pose = False
                    continue
                values = np.asarray(matrix, dtype=np.float64)
                if len(values) and np.abs(values - values[0]).max() > 0:
                    problems.append(f"{name}: camera_info/{camera}/intrinsics changes during the episode")
                if len(demo[f"camera_info/{camera}/pose"]) != steps:
                    problems.append(f"{name}: camera_info/{camera}/pose has no row per step")
                first = [[float(x) for x in row] for row in values[0]]
                if camera in intrinsics and intrinsics[camera] != first:
                    problems.append(f"{name}: {camera} intrinsics differ from the earlier demos'")
                intrinsics.setdefault(camera, first)
        rate, where = demo_fps(demo, source.attrs, clock="sim")
        fps_seen.append((name, rate, where))

    if len(shapes) > 1:
        problems.append(f"cameras disagree on resolution: {sorted(shapes)}")
    rates = [r for _, r, _ in fps_seen]
    if rates and max(rates) - min(rates) > 0.01 * max(rates):
        spread = ", ".join(f"{n}={r:.2f}" for n, r, _ in fps_seen[:4])
        problems.append(f"demos disagree on rate ({spread}...); that is two datasets, not one")

    if eef_rot_seen == {True, False}:
        problems.append(
            "some demos record obs/*_eef_rot and others do not; that is two collections, "
            "and observation.eef_state would change meaning halfway through the dataset"
        )
    recorded_eef_rot = eef_rot_seen == {True}
    if not recorded_eef_rot and not problems:
        missing = sorted(
            {
                joint
                for robot in robots
                for side in WRIST_CHAIN
                for joint, _, axis in WRIST_CHAIN[side]
                if axis is not None and joint not in joint_names[robot]
            }
        )
        if missing:
            problems.append(
                f"no obs/*_eef_rot to read and the wrist chain cannot be walked either: {missing} "
                "are not among the recorded joints"
            )

    if problems:
        for line in problems:
            print(f"  {line}")
        raise SystemExit(f"{len(problems)} problem(s); nothing written")

    fps = float(round(rates[0])) if rates else 0.0
    if abs(fps - rates[0]) > 0.01 * max(rates[0], 1.0):
        raise SystemExit(f"recorded rate {rates[0]:.4f} Hz is not a whole number of frames per second")

    columns = action_columns(action_term_layout(robots), joint_names, int(action_width))
    return {
        "names": names,
        "robots": robots,
        "cameras": cameras,
        "annotations": annotation_columns_for(robots),
        "env_args": env_args,
        "joint_names": joint_names,
        "action_columns": columns,
        "fps": fps,
        "fps_source": fps_seen[0][2],
        "height": sorted(shapes)[0][0],
        "width": sorted(shapes)[0][1],
        "has_wrench": has_wrench,
        "recorded_eef_rot": recorded_eef_rot,
        "depth_cameras": tuple(c for c in cameras if c in (depth_cameras or set())),
        "has_camera_pose": has_camera_pose,
        "intrinsics": intrinsics if has_camera_pose else {},
        "steps": {name: int(source[name].attrs["num_samples"]) for name in names},
    }


def episode_columns(demo, facts: dict) -> dict[str, np.ndarray]:
    """The float columns of one episode, in GR00T's conventions."""
    robots = facts["robots"]
    perms = {robot: urdf_order(facts["joint_names"][robot])[0] for robot in robots}

    def read(key: str) -> np.ndarray:
        return np.asarray(demo[key], dtype=np.float64)

    state = np.concatenate([read(f"obs/{robot}_joint_pos")[:, perms[robot]] for robot in robots], axis=1)

    processed = read("processed_actions")
    action = np.concatenate([processed[:, facts["action_columns"][robot]] for robot in robots], axis=1)

    actions64 = read("actions")
    navigate = np.concatenate(
        [actions64[:, i * ROBOT_STRIDE + ACTION_NAVIGATE.start : i * ROBOT_STRIDE + ACTION_NAVIGATE.stop] for i in range(len(robots))],
        axis=1,
    )
    base_height = np.concatenate(
        [actions64[:, i * ROBOT_STRIDE + ACTION_BASE_HEIGHT.start : i * ROBOT_STRIDE + ACTION_BASE_HEIGHT.stop] for i in range(len(robots))],
        axis=1,
    )

    roots = {robot: (read(f"obs/{robot}_root_pos"), read(f"obs/{robot}_root_rot")) for robot in robots}

    eef_state = []
    eef_action = []
    for index, robot in enumerate(robots):
        root_pos, root_quat = roots[robot]
        angles = None
        if not facts["recorded_eef_rot"]:
            angles = named_angles(read(f"obs/{robot}_joint_pos"), facts["joint_names"][robot])
        for side, block in (("left", ACTION_LEFT_WRIST), ("right", ACTION_RIGHT_WRIST)):
            measured_pos = read(f"obs/{robot}_{side}_eef_pos")
            if angles is None:
                measured_quat = read(f"obs/{robot}_{side}_eef_rot")
            else:
                # The chain gives the wrist in the pelvis frame. The pelvis pose carries it to the world.
                measured_quat = quat_multiply(
                    normalised(root_quat), wrist_quat_xyzw(angles, side)
                )
            eef_state.append(pelvis_relative_pose(measured_pos, measured_quat, root_pos, root_quat))

            start = index * ROBOT_STRIDE + block.start
            commanded = actions64[:, start : start + 7]
            eef_action.append(pelvis_relative_pose(commanded[:, 0:3], commanded[:, 3:7], root_pos, root_quat))

    robots_state = np.concatenate([np.concatenate(roots[robot], axis=1) for robot in robots], axis=1)

    columns = {
        "observation.state": state,
        "observation.eef_state": np.concatenate(eef_state, axis=1),
        "observation.robots_state": robots_state,
        "observation.object_state": np.concatenate([read("obs/object_pos"), read("obs/object_rot")], axis=1),
        "observation.grasp_latched": read("obs/grasp_latched"),
        "action": action,
        "action.eef": np.concatenate(eef_action, axis=1),
        "teleop.navigate_command": navigate,
        "teleop.base_height_command": base_height,
    }

    if facts["has_camera_pose"]:
        # Camera poses: world frame, xyzw.
        columns["observation.camera_pose"] = np.concatenate(
            [read(f"camera_info/{camera}/pose") for camera in facts["cameras"]], axis=1
        )

    if facts["has_wrench"]:
        # Recorded at sim rate, four rows per control step. Keep the last row of each step.
        wrench = []
        for robot in robots:
            force = read(f"eef_wrench/{robot}/force")[3::4]
            torque = read(f"eef_wrench/{robot}/torque")[3::4]
            for hand in range(force.shape[1]):
                wrench.append(force[:, hand])
                wrench.append(torque[:, hand])
        columns["observation.eef_wrench"] = np.concatenate(wrench, axis=1)

    return columns


# Writing

VECTOR_COLUMNS = (
    "observation.state",
    "observation.eef_wrench",
    "observation.eef_state",
    "observation.robots_state",
    "observation.camera_pose",
    "observation.object_state",
    "observation.grasp_latched",
    "action",
    "action.eef",
    "teleop.navigate_command",
    "teleop.base_height_command",
)

ANNOTATION_COLUMNS = (
    "annotation.human.task_description",
    "annotation.human.task_description_robot_a",
    "annotation.human.task_description_robot_b",
)

VECTOR_TYPE = pa.list_(pa.field("item", pa.float64()))
"""A vector column: a variable-length list of doubles with child field ``item``, as LeRobot writes it."""

CONSUMED_COLUMNS = ("action", "action.eef", "teleop.navigate_command", "teleop.base_height_command")
"""Columns a policy is trained to produce. Scanned for dimensions that never move."""


def episode_table(columns: dict[str, np.ndarray], episode_index: int, index_start: int, fps: float,
                  annotations: tuple[str, ...] = ANNOTATION_COLUMNS) -> pa.Table:
    """One episode as a LeRobot parquet table.

    Vector columns are ``list<double>``, so each cell loads as a numpy array.
    """
    steps = len(columns["observation.state"])
    fields = []
    arrays = []
    for name in VECTOR_COLUMNS:
        if name not in columns:
            continue
        matrix = np.ascontiguousarray(columns[name], dtype=np.float64)
        fields.append(pa.field(name, VECTOR_TYPE))
        arrays.append(pa.array(list(matrix), type=VECTOR_TYPE))

    scalars = {
        "timestamp": (pa.float64(), np.arange(steps, dtype=np.float64) / fps),
        "frame_index": (pa.int64(), np.arange(steps, dtype=np.int64)),
        "episode_index": (pa.int64(), np.full(steps, episode_index, dtype=np.int64)),
        "index": (pa.int64(), np.arange(index_start, index_start + steps, dtype=np.int64)),
        "task_index": (pa.int64(), np.zeros(steps, dtype=np.int64)),
    }
    # Annotation columns hold row indices into tasks.jsonl.
    for offset, name in enumerate(annotations):
        scalars[name] = (pa.int64(), np.full(steps, offset, dtype=np.int64))
    done = np.zeros(steps, dtype=bool)
    done[-1] = True
    scalars["next.done"] = (pa.bool_(), done)

    for name, (kind, values) in scalars.items():
        fields.append(pa.field(name, kind))
        arrays.append(pa.array(values, type=kind))

    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def write_videos(demo, out_root: pathlib.Path, episode_index: int, fps: float, quality: int,
                 depth_cameras: tuple[str, ...] = (), cameras: tuple[str, ...] = CAMERAS) -> dict[str, int]:
    """Write one MP4 per camera stream. Returns the frame count per column.

    The frame count must equal the parquet's row count. Depth uses a lossless
    encoder and :func:`~cohub.recording.demo_video.pack_depth`.
    """
    chunk = episode_index // CHUNKS_SIZE
    written = {}
    for camera in cameras:
        column = VIDEO_COLUMN.format(camera=camera)
        path = out_root / VIDEO_PATH.format(episode_chunk=chunk, video_key=column, episode_index=episode_index)
        frames = demo[f"obs/images/{camera}/rgb"]
        writer = open_writer(path, fps=fps, quality=quality)
        count = 0
        try:
            for row in range(frames.shape[0]):
                writer.append_data(np.asarray(frames[row])[..., :3])
                count += 1
        finally:
            writer.close()
        written[column] = count

    for camera in depth_cameras:
        column = DEPTH_COLUMN.format(camera=camera)
        path = out_root / VIDEO_PATH.format(episode_chunk=chunk, video_key=column, episode_index=episode_index)
        frames = demo[f"obs/images/{camera}/depth"]
        writer = open_lossless_writer(path, fps=fps)
        count = 0
        try:
            for row in range(frames.shape[0]):
                writer.append_data(pack_depth(np.asarray(frames[row])))
                count += 1
        finally:
            writer.close()
        written[column] = count
    return written


def feature_names(facts: dict) -> dict[str, list[str]]:
    """The per-dimension labels that go in ``info.json``."""
    robots = facts["robots"]
    cameras = facts["cameras"]
    joint_labels = []
    for robot in robots:
        joint_labels += [f"{robot}/{name}" for name in urdf_order(facts["joint_names"][robot])[1]]

    eef_labels = []
    for robot in robots:
        for side in ("left", "right"):
            eef_labels += [f"{robot}/{side}_wrist_{axis}" for axis in ("pos_x", "pos_y", "pos_z")]
            eef_labels += [f"{robot}/{side}_wrist_quat_{axis}" for axis in ("w", "x", "y", "z")]

    root_labels = []
    for robot in robots:
        root_labels += [f"{robot}/root_{axis}" for axis in ("pos_x", "pos_y", "pos_z")]
        root_labels += [f"{robot}/root_quat_{axis}" for axis in ("x", "y", "z", "w")]

    names = {
        "observation.state": joint_labels,
        "observation.eef_state": eef_labels,
        "observation.robots_state": root_labels,
        "observation.object_state": ["object/pos_x", "object/pos_y", "object/pos_z", "object/quat_x", "object/quat_y", "object/quat_z", "object/quat_w"],
        "observation.grasp_latched": [f"{robot}/{side}" for robot in robots for side in ("left", "right")],
        "action": joint_labels,
        "action.eef": eef_labels,
        "teleop.navigate_command": [f"{robot}/{axis}" for robot in robots for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")],
        "teleop.base_height_command": [f"{robot}/base_height" for robot in robots],
    }
    if facts["has_camera_pose"]:
        names["observation.camera_pose"] = [
            f"{camera}/{axis}"
            for camera in cameras
            for axis in ("pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")
        ]
    if facts["has_wrench"]:
        names["observation.eef_wrench"] = [
            f"{robot}/{side}_{what}_{axis}"
            for robot in robots
            for side in ("left", "right")
            for what in ("force", "torque")
            for axis in ("x", "y", "z")
        ]
    return names


def build_modality(facts: dict) -> dict:
    """``meta/modality.json``: which slice of which column each key is, with a robot prefix on every key."""
    robots = facts["robots"]
    cameras = facts["cameras"]
    state: dict[str, dict] = {}
    action: dict[str, dict] = {}
    for index, robot in enumerate(robots):
        base = index * JOINTS_PER_ROBOT
        bounds = urdf_order(facts["joint_names"][robot])[2]
        for group, (start, end) in bounds.items():
            state[f"{robot}_{group}"] = {"start": base + start, "end": base + end}
            action[f"{robot}_{group}"] = {"start": base + start, "end": base + end}

        eef_base = index * 14
        for offset, side in ((0, "left"), (7, "right")):
            for column, key in (("observation.eef_state", "state"), ("action.eef", "action")):
                target = state if key == "state" else action
                target[f"{robot}_{side}_wrist_pos"] = {
                    "start": eef_base + offset,
                    "end": eef_base + offset + 3,
                    "original_key": column,
                }
                target[f"{robot}_{side}_wrist_abs_quat"] = {
                    "start": eef_base + offset + 3,
                    "end": eef_base + offset + 7,
                    "original_key": column,
                    "rotation_type": "quaternion",
                }

        action[f"{robot}_navigate_command"] = {
            "start": index * 3,
            "end": index * 3 + 3,
            "original_key": "teleop.navigate_command",
        }
        action[f"{robot}_base_height_command"] = {
            "start": index,
            "end": index + 1,
            "original_key": "teleop.base_height_command",
        }
        state[f"{robot}_root"] = {"start": index * 7, "end": index * 7 + 7, "original_key": "observation.robots_state"}

    if facts["has_camera_pose"]:
        for index, camera in enumerate(cameras):
            state[f"{camera}_camera_pose"] = {
                "start": index * 7,
                "end": index * 7 + 7,
                "original_key": "observation.camera_pose",
            }

    state["object"] = {"start": 0, "end": 7, "original_key": "observation.object_state"}
    state["grasp_latched"] = {"start": 0, "end": 2 * len(robots), "original_key": "observation.grasp_latched"}
    if facts["has_wrench"]:
        for index, robot in enumerate(robots):
            state[f"{robot}_wrench"] = {
                "start": index * 12,
                "end": index * 12 + 12,
                "original_key": "observation.eef_wrench",
            }

    return {
        "state": state,
        "action": action,
        "video": {
            **{camera: {"original_key": VIDEO_COLUMN.format(camera=camera)} for camera in cameras},
            # Depth streams hold packed bytes. Decode with unpack_depth.
            **{
                f"{camera}_depth": {"original_key": DEPTH_COLUMN.format(camera=camera)}
                for camera in facts["depth_cameras"]
            },
        },
        "annotation": {
            name.removeprefix("annotation."): {"original_key": name} for name in facts["annotations"]
        },
    }


def episode_splits(count: int, val: int) -> dict[str, str]:
    """LeRobot's ``splits`` map: everything in train, or a held-out tail.

    Args:
        count: Episodes in the export.
        val: How many episodes to hold out from the end. ``0`` writes train only.
    """
    if val < 0 or val >= count:
        raise SystemExit(f"--val {val} is not a proper held-out tail of {count} episodes")
    if not val:
        return {"train": f"0:{count}"}
    return {"train": f"0:{count - val}", "val": f"{count - val}:{count}"}


def camera_lens_info(matrix: list[list[float]] | None) -> dict:
    """The intrinsics entries of a video feature's ``info``."""
    if not matrix:
        return {}
    return {
        "camera.intrinsics": matrix,
        "camera.fx": matrix[0][0],
        "camera.fy": matrix[1][1],
        "camera.cx": matrix[0][2],
        "camera.cy": matrix[1][2],
    }


def build_info(facts: dict, columns: dict[str, np.ndarray], episodes: list[dict], total_frames: int,
               val: int = 0) -> dict:
    """``meta/info.json``.

    Every float feature listed must also be a parquet column.
    """
    labels = feature_names(facts)
    features: dict[str, dict] = {}
    for name in VECTOR_COLUMNS:
        if name not in columns:
            continue
        features[name] = {
            "dtype": "float64",
            "shape": [int(columns[name].shape[1])],
            "names": labels[name],
        }
    if not facts["recorded_eef_rot"] and "observation.eef_state" in features:
        features["observation.eef_state"]["info"] = {
            "rotation.source": "forward_kinematics",
            "rotation.note": "this task recorded no obs/*_eef_rot; the quaternion is the pinned "
                             "pelvis-to-wrist chain evaluated at the measured joint angles. "
                             "The positions are measured.",
            "rotation.agreement": "4.9e-07 max component, 7.2e-05 degrees, against a task that records it",
        }
    features["timestamp"] = {"dtype": "float64", "shape": [1], "names": None}
    for name in ("frame_index", "episode_index", "index", "task_index", *facts["annotations"]):
        features[name] = {"dtype": "int64", "shape": [1], "names": None}
    features["next.done"] = {"dtype": "bool", "shape": [1], "names": None}

    fps = int(facts["fps"])
    for camera in facts["cameras"]:
        lens = facts["intrinsics"].get(camera)
        features[VIDEO_COLUMN.format(camera=camera)] = {
            "dtype": "video",
            "shape": [facts["height"], facts["width"], 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": facts["height"],
                "video.width": facts["width"],
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
                **camera_lens_info(lens),
            },
        }

    for camera in facts["depth_cameras"]:
        features[DEPTH_COLUMN.format(camera=camera)] = {
            "dtype": "video",
            "shape": [facts["height"], facts["width"], 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": facts["height"],
                "video.width": facts["width"],
                # Lossless h264 in RGB.
                "video.codec": "h264",
                "video.pix_fmt": "rgb24",
                "video.is_lossless": True,
                "video.is_depth_map": True,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
                "depth.encoding": "uint16_hi_lo_rgb",
                "depth.units_per_metre": DEPTH_UNITS_PER_METRE,
                "depth.invalid_value": DEPTH_INVALID,
                "depth.source": "distance_to_image_plane",
                "depth.decode": "metres = ((R << 8) | G) / units_per_metre; 0 means no return",
                **camera_lens_info(facts["intrinsics"].get(camera)),
            },
        }

    count = len(episodes)
    return {
        "codebase_version": "v2.1",
        "robot_type": f"unitree_g1_x{len(facts['robots'])}",
        "total_episodes": count,
        "total_frames": total_frames,
        "total_tasks": len(facts["annotations"]),
        "total_videos": count * (len(facts["cameras"]) + len(facts["depth_cameras"])),
        "total_chunks": (count + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": episode_splits(count, val),
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
    }


def export(source_path: pathlib.Path, out_root: pathlib.Path, selection: set[int] | None,
           quality: int, task_text: str, val: int = 0,
           task_id: str = "") -> int:
    with DemoSource(source_path) as source:
        facts = precheck(source, selection)
        env_name = json.loads(source.attrs.get("env_args", "{}") or "{}").get("env_name", "")
        # --task-id overrides the env id the shard states.
        if task_id:
            print(f"  reading {env_name or '(unstamped)'} as {task_id} for its instructions")
            env_name = task_id
        sentences = task_sentences(env_name, task_text, facts["robots"])

        print(f"{len(facts['names'])} demo(s) at {facts['fps']:.0f} Hz (from {facts['fps_source']})")
        if facts["robots"] != ROBOTS:
            print(f"  robots: {', '.join(facts['robots'])} -- cameras {', '.join(facts['cameras'])}")
        if not facts["has_wrench"]:
            print("  NOTE no eef_wrench in this dataset: the force/torque channel is not exported")
        if facts["depth_cameras"]:
            print(f"  depth: {', '.join(facts['depth_cameras'])} -- lossless, "
                  f"1/{DEPTH_UNITS_PER_METRE:g} m per unit")
        else:
            print("  NOTE no depth in this dataset: only the colour cameras are exported")
        if not facts["has_camera_pose"]:
            print("  NOTE no camera_info: observation.camera_pose and the intrinsics are not exported,")
            print("       so the depth cannot be placed in the scene. Migrate the dataset first.")

        out_root.mkdir(parents=True, exist_ok=True)
        episodes = []
        index_start = 0
        spans: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        shapes: dict[str, np.ndarray] = {}

        for episode_index, name in enumerate(facts["names"]):
            demo = source[name]
            columns = episode_columns(demo, facts)
            steps = len(columns["observation.state"])
            table = episode_table(columns, episode_index, index_start, facts["fps"], facts["annotations"])

            chunk = episode_index // CHUNKS_SIZE
            data_path = out_root / DATA_PATH.format(episode_chunk=chunk, episode_index=episode_index)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, data_path)

            counts = write_videos(demo, out_root, episode_index, facts["fps"], quality,
                                  facts["depth_cameras"], facts["cameras"])
            for column, count in counts.items():
                if count != steps:
                    raise SystemExit(f"{name}: {column} wrote {count} frames for {steps} rows")

            for column in CONSUMED_COLUMNS:
                block = columns[column]
                low, high = block.min(0), block.max(0)
                if column in spans:
                    low = np.minimum(spans[column][0], low)
                    high = np.maximum(spans[column][1], high)
                spans[column] = (low, high)
            shapes = columns

            episodes.append(
                {
                    "episode_index": episode_index,
                    "demo": name,
                    "num_samples": steps,
                    "success": bool(demo.attrs.get("success", False)),
                    "meta": {
                        key: float(np.asarray(demo[f"meta/{key}"]).squeeze())
                        for key in demo["meta"].keys()
                    }
                    if "meta" in demo
                    else {},
                }
            )
            index_start += steps
            print(f"  episode {episode_index:>3} <- {name:>10}  {steps:>4} steps")

        total_frames = sum(e["num_samples"] for e in episodes)
        meta = out_root / "meta"
        meta.mkdir(parents=True, exist_ok=True)

        (meta / "info.json").write_text(
            json.dumps(build_info(facts, shapes, episodes, total_frames, val), indent=2) + "\n"
        )
        (meta / "modality.json").write_text(json.dumps(build_modality(facts), indent=2) + "\n")
        with (meta / "episodes.jsonl").open("w") as handle:
            for episode in episodes:
                handle.write(json.dumps({
                    "episode_index": episode["episode_index"],
                    "tasks": [sentences[0]],
                    "length": episode["num_samples"],
                }) + "\n")
        with (meta / "tasks.jsonl").open("w") as handle:
            for task_index, sentence in enumerate(sentences):
                handle.write(json.dumps({"task_index": task_index, "task": sentence}) + "\n")

        labels = feature_names(facts)
        constant = [
            f"{column}[{i}] = {labels[column][i]}"
            for column in CONSUMED_COLUMNS
            for i in np.flatnonzero(spans[column][1] - spans[column][0] == 0.0)
        ]
        (meta / "cohub_provenance.json").write_text(json.dumps({
            # The name only, not the local path.
            "source": source_path.name,
            "env_args": restamp(facts["env_args"], None, "export_lerobot.py"),
            "collected": collected_stamp(dict(source.attrs)),
            # Who rendered these frames, which may differ from who collected them.
            "images": images_stamp(facts["env_args"]),
            "fps": {"value": facts["fps"], "source": facts["fps_source"]},
            "task": {"env_name": env_name, "sentences": list(sentences)},
            "episodes": episodes,
            "constant_action_dims": constant,
            "wrist_rotation": {"source": "obs/*_eef_rot, measured"} if facts["recorded_eef_rot"] else {
                "source": "forward kinematics, not measured",
                "why": "this task's ObservationsCfg declares *_eef_pos but not *_eef_rot",
                "how": "cohub.g1.geometry walks the pinned URDF chain "
                       "(waist 3 + arm 7) with the measured joint angles on the same row",
                "checked_against": "CoCarry, which records the quaternion: 4.9e-07 max component "
                                   "(7.2e-05 degrees) over 94604 wrist-rows, float32's own resolution",
                "measured_and_not_reconstructed": "the wrist positions in the same column",
            },
            "depth": {
                "cameras": list(facts["depth_cameras"]),
                "encoding": "uint16_hi_lo_rgb",
                "units_per_metre": DEPTH_UNITS_PER_METRE,
                "invalid_value": DEPTH_INVALID,
                "codec": "libx264rgb -qp 0 (lossless h264, rgb24)",
                "quantisation_error_metres": 0.5 / DEPTH_UNITS_PER_METRE,
                "read_with": "cohub.recording.demo_video.unpack_depth",
            } if facts["depth_cameras"] else {},
            "excluded": {
                "obs/*_grasp": "the pinch channels' threshold is under review; grasp_latched is exported instead",
                "sync, operator, states": "archive-only channels, kept in the HDF5",
            },
        }, indent=2) + "\n")

    print(f"\nwrote {len(episodes)} episode(s), {total_frames} frames -> {out_root}")
    if not facts["recorded_eef_rot"]:
        print("  observation.eef_state's rotation is reconstructed, not recorded: this task")
        print("  declares no *_eef_rot terms, so the pelvis-to-wrist chain was walked at the")
        print("  measured joint angles (agrees with a recording task to 7.2e-05 deg). Positions")
        print("  are measured. meta/cohub_provenance.json says so too.")
    if constant:
        print(f"  {len(constant)} commanded dim(s) never change: {', '.join(constant[:3])}"
              f"{'...' if len(constant) > 3 else ''}")
        print("  GR00T's min/max normalisation maps those to 0; a loader that divides by the")
        print("  spread needs a guard.")
    if facts["robots"] != ROBOTS:
        print(f"\nNo GR00T modality config for {len(facts['robots'])} robots exists in policy/GR00T_N17/configs/ yet;")
        print("  the export is complete, but nothing there trains on it as it stands.")
        return 0
    print("\nTrain GR00T on it (it computes its statistics at train start):")
    print("  bash baselines/policy/GR00T_N17/train.sh <task>")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a recorded dataset as LeRobot v2.1 for GR00T.")
    parser.add_argument("dataset", type=str, help="A .hdf5, or a directory of per-demo shards.")
    parser.add_argument("--out", type=str, default="", help="Destination. Defaults to a sibling lerobot/.")
    parser.add_argument("--demos", type=str, default="", help="Which demo indices to export, e.g. 3,7-9.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing export.")
    parser.add_argument("--quality", type=int, default=6, help="imageio/libx264 quality, 0-10.")
    parser.add_argument("--task-text", type=str, default="", help="One instruction for both robots, instead of the task's own.")
    parser.add_argument("--task-id", type=str, default="", help="Read the shards as this env id when picking instructions, for a collection recorded before the id was renamed.")
    parser.add_argument("--val", type=int, default=0, help="Hold out this many episodes from the END as the val split. 0 writes train only.")
    args = parser.parse_args()

    source = pathlib.Path(args.dataset).expanduser().resolve()
    if not source.exists():
        print(f"No such dataset: {source}")
        return 1

    out_root = pathlib.Path(args.out).expanduser() if args.out else dataset_root(source) / "lerobot"
    if out_root.exists() and any(out_root.iterdir()):
        if not args.overwrite:
            print(f"{out_root} exists and is not empty; --overwrite to replace it")
            return 1
        # Refuse an --out that contains the dataset root or the source.
        resolved = out_root.resolve()
        root = dataset_root(source).resolve()
        for name, path in (("the dataset root", root), ("the source", source)):
            if resolved == path or path.is_relative_to(resolved):
                print(f"--out {out_root} would delete {name} ({path}); pick another --out")
                return 1
        shutil.rmtree(out_root)

    return export(source, out_root, parse_selection(args.demos), args.quality,
                  args.task_text, args.val, args.task_id)


if __name__ == "__main__":
    sys.exit(main())
