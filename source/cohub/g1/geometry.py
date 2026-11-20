"""Robot geometry that needs no simulator: the ego lens, the wrist chain, the XR anchor yaw."""

from __future__ import annotations

import math
import os

import numpy as np


# --- Ego camera lens and mount ---

PELVIS_SPAWN_Z = 0.80
"""Height the pelvis spawns at, standing [m]. The feet start 7 mm above the ground."""

EGO_MOUNT_LINK = "torso_link"
EGO_SENSOR_FRAME = "d435_link"

# RealSense D435 mount position relative to ``torso_link`` [m].
EGO_OFFSET_POS = (0.0576, 0.0175, 0.4299)

DEFAULT_APERTURE = 20.955


def focal_length_for_fov(fov_deg: float, aperture: float = DEFAULT_APERTURE) -> float:
    """Focal length [mm] that spreads ``fov_deg`` across ``aperture``."""
    return 0.5 * aperture / math.tan(math.radians(0.5 * fov_deg))


# Stereolabs ZED Mini datasheet field of view [deg]. The ego camera matches the
# horizontal only; the vertical follows the recorded frame shape.
ZED_MINI_HFOV_DEG = 102.0

LEGACY_EGO_HFOV_DEG = (90.0, 95.6)
"""Horizontal fields of view of ego lenses found in older datasets [deg]."""

# The G1 hardware's own camera tilt and lens.
D435_PITCH_DOWN_DEG = 47.6
D435_FOCAL_LENGTH = 15.1
"""RealSense D435 colour stream [mm]: 69.4 deg horizontal on the default aperture."""

# The benchmark's ego camera tilt [deg], chosen to keep the partner robot in frame.
EGO_PITCH_DOWN_DEG = 25.0
DEFAULT_FOCAL_LENGTH = focal_length_for_fov(ZED_MINI_HFOV_DEG)
"""The ZED Mini's 102 deg horizontal as a focal length on ``DEFAULT_APERTURE``: 8.4845 mm."""

# Height of the camera mount above the pelvis [m], used for the XR eye placement.
EGO_HEIGHT_ABOVE_PELVIS = 0.4739

# Recorded frame size [px]. 4:3 keeps the hands in frame in a reach-down; the
# 16:9 height matches the ZED Mini's own frame shape.
DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
ZED_MINI_HEIGHT_16_9 = 180

RGB = "rgb"
DEPTH = "distance_to_image_plane"


def pitch_down_quat_xyzw(degrees: float) -> tuple[float, float, float, float]:
    """Rotation about the link's ``+Y`` that tips ``+X`` downwards, as xyzw.

    A positive angle is a downward pitch.
    """
    half = math.radians(degrees) / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


def horizontal_fov_deg(focal_length: float = DEFAULT_FOCAL_LENGTH, aperture: float = DEFAULT_APERTURE) -> float:
    """Horizontal field of view of a pinhole with this focal length [deg]."""
    return math.degrees(2.0 * math.atan(0.5 * aperture / focal_length))


def vertical_fov_deg(
    focal_length: float = DEFAULT_FOCAL_LENGTH,
    aperture: float = DEFAULT_APERTURE,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> float:
    """Vertical field of view [deg] of that lens on a ``width`` x ``height`` frame.

    Assumes square pixels: ``vertical_aperture = aperture * height / width``.
    """
    return math.degrees(2.0 * math.atan(0.5 * aperture * (height / width) / focal_length))


def diagonal_fov_deg(
    focal_length: float = DEFAULT_FOCAL_LENGTH,
    aperture: float = DEFAULT_APERTURE,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> float:
    """Diagonal field of view [deg] of that lens on that frame."""
    half_h = 0.5 * aperture / focal_length
    return math.degrees(2.0 * math.atan(math.hypot(half_h, half_h * (height / width))))


def focal_length_for_fx(fx: float, width: int = DEFAULT_WIDTH, aperture: float = DEFAULT_APERTURE) -> float:
    """Focal length [mm] for a recorded ``fx``: ``fx = focal_length * width / aperture``."""
    return fx * aperture / float(width)


def fov_for_fx(fx: float, width: int = DEFAULT_WIDTH, aperture: float = DEFAULT_APERTURE) -> float:
    """Horizontal field of view [deg] of a camera whose intrinsics say ``fx``."""
    return horizontal_fov_deg(focal_length_for_fx(fx, width, aperture), aperture)


def pitch_down_deg_from_quat(quat_xyzw) -> float:
    """How far below horizontal a camera at this world orientation looks [deg].

    ``quat_xyzw`` uses the ``"world"`` camera convention: the view direction
    is the rotated ``+X``.
    """
    x, y, z, w = (float(v) for v in quat_xyzw)
    forward_z = 2.0 * (x * z - y * w)
    forward_x = 1.0 - 2.0 * (y * y + z * z)
    forward_y = 2.0 * (x * y + z * w)
    horizontal = math.hypot(forward_x, forward_y)
    return math.degrees(math.atan2(-forward_z, horizontal))


# --- Wrist orientation from joint angles ---
# Rotation-only forward kinematics of the pelvis-to-wrist chain. The constants are from Isaac Lab's G1
# kinematics URDF.

WRIST_CHAIN: dict[str, tuple[tuple[str, tuple[float, float, float], tuple[int, int, int] | None], ...]] = {
    "left": (
        ("imu_joint", (-0.0, 1e-07, -1e-07), None),
        ("waist_yaw_joint", (0.0, 0.0, 0.0), (0, 0, 1)),
        ("waist_roll_joint", (0.0, 0.0, 0.0), (1, 0, 0)),
        ("waist_pitch_joint", (0.0, 0.0, 0.0), (0, 1, 0)),
        ("left_shoulder_pitch_joint", (0.27931, 5.49e-05, -0.0001916), (0, 1, 0)),
        ("left_shoulder_roll_joint", (-0.2792501, 0.0, 0.0), (1, 0, 0)),
        ("left_shoulder_yaw_joint", (1e-07, 0.0, 0.0), (0, 0, 1)),
        ("left_elbow_joint", (-0.0, 1e-07, -0.0), (0, 1, 0)),
        ("left_wrist_roll_joint", (0.0, 0.0, -0.0), (1, 0, 0)),
        ("left_wrist_pitch_joint", (-0.0, 0.0, 0.0), (0, 1, 0)),
        ("left_wrist_yaw_joint", (1e-07, -0.0, 2e-07), (0, 0, 1)),
    ),
    "right": (
        ("imu_joint", (-0.0, 1e-07, -1e-07), None),
        ("waist_yaw_joint", (0.0, 0.0, 0.0), (0, 0, 1)),
        ("waist_roll_joint", (0.0, 0.0, 0.0), (1, 0, 0)),
        ("waist_pitch_joint", (0.0, 0.0, 0.0), (0, 1, 0)),
        ("right_shoulder_pitch_joint", (-0.27931, 5.49e-05, 0.0001916), (0, 1, 0)),
        ("right_shoulder_roll_joint", (0.27925, -0.0, 0.0), (1, 0, 0)),
        ("right_shoulder_yaw_joint", (-0.0, 0.0, -0.0), (0, 0, 1)),
        ("right_elbow_joint", (0.0, 1e-07, 0.0), (0, 1, 0)),
        ("right_wrist_roll_joint", (-0.0, 0.0, -0.0), (1, 0, 0)),
        ("right_wrist_pitch_joint", (-0.0, 0.0, -0.0), (0, 1, 0)),
        ("right_wrist_yaw_joint", (-0.0, -0.0, -2e-07), (0, 0, 1)),
    ),
}
"""``side -> ((joint name, fixed rpy, joint axis or None for a fixed joint), ...)``.

Pelvis outwards, in chain order.
"""


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF's fixed rotation: yaw about z, then pitch about y, then roll about x."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_matrix(axis: tuple[int, int, int], angle: np.ndarray) -> np.ndarray:
    """(T,) angles about one of the three principal axes -> (T, 3, 3)."""
    c, s = np.cos(angle), np.sin(angle)
    zero, one = np.zeros_like(c), np.ones_like(c)
    if axis == (1, 0, 0):
        rows = ((one, zero, zero), (zero, c, -s), (zero, s, c))
    elif axis == (0, 1, 0):
        rows = ((c, zero, s), (zero, one, zero), (-s, zero, c))
    elif axis == (0, 0, 1):
        rows = ((c, -s, zero), (s, c, zero), (zero, zero, one))
    else:
        raise ValueError(f"joint axis {axis} is not a principal axis")
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def wrist_rotation_matrix(angles: dict[str, np.ndarray], side: str) -> np.ndarray:
    """(T, 3, 3): the wrist's orientation in the *pelvis* frame.

    Args:
        angles: joint name -> (T,) measured angle [rad]. Every driven joint of
            :data:`WRIST_CHAIN` must be present; extra entries are ignored.
        side: ``"left"`` or ``"right"``.
    """
    chain = WRIST_CHAIN[side]
    missing = [name for name, _, axis in chain if axis is not None and name not in angles]
    if missing:
        raise KeyError(f"{side} wrist chain is missing {missing}")

    steps = len(next(iter(angles.values())))
    rotation = np.broadcast_to(np.eye(3), (steps, 3, 3)).copy()
    for name, rpy, axis in chain:
        rotation = rotation @ _rpy_matrix(*rpy)
        if axis is not None:
            rotation = rotation @ _axis_matrix(axis, np.asarray(angles[name], dtype=np.float64))
    return rotation


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) scalar-last, taking the largest component's branch."""
    m = np.asarray(matrix, dtype=np.float64)
    xx, yy, zz = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    # Four candidate reconstructions; pick the numerically stable one per element.
    candidates = np.stack(
        [
            np.stack([1.0 + xx - yy - zz, m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0], m[..., 2, 1] - m[..., 1, 2]], axis=-1),
            np.stack([m[..., 0, 1] + m[..., 1, 0], 1.0 - xx + yy - zz, m[..., 1, 2] + m[..., 2, 1], m[..., 0, 2] - m[..., 2, 0]], axis=-1),
            np.stack([m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1.0 - xx - yy + zz, m[..., 1, 0] - m[..., 0, 1]], axis=-1),
            np.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1], 1.0 + xx + yy + zz], axis=-1),
        ],
        axis=-2,
    )
    scale = np.stack([1.0 + xx - yy - zz, 1.0 - xx + yy - zz, 1.0 - xx - yy + zz, 1.0 + xx + yy + zz], axis=-1)
    branch = np.argmax(scale, axis=-1)
    chosen = np.take_along_axis(candidates, branch[..., None, None], axis=-2)[..., 0, :]
    return chosen / np.linalg.norm(chosen, axis=-1, keepdims=True)


def wrist_quat_xyzw(angles: dict[str, np.ndarray], side: str) -> np.ndarray:
    """(T, 4) scalar-last: the wrist's orientation in the pelvis frame."""
    return matrix_to_quat_xyzw(wrist_rotation_matrix(angles, side))


def named_angles(joint_pos: np.ndarray, joint_names: list[str]) -> dict[str, np.ndarray]:
    """(T, 43) recorded in PhysX's order + its names -> ``{name: (T,)}``."""
    values = np.asarray(joint_pos, dtype=np.float64)
    if values.shape[1] != len(joint_names):
        raise ValueError(f"{values.shape[1]} joint columns against {len(joint_names)} names")
    return {name: values[:, index] for index, name in enumerate(joint_names)}


# --- XR anchor rotation ---
# With an identity anchor_rot Kit starts the operator facing world +Y. The anchor rotation is the
# robot's spawn yaw minus that heading.

XR_ANCHOR_FORWARD_YAW = math.pi / 2.0
"""World yaw an identity ``anchor_rot`` starts the operator facing [rad]: +Y."""

ANCHOR_YAW_ENV = "COHUB_ANCHOR_YAW_DEG"
"""Environment variable: extra yaw [deg] added to every anchor.

Positive is counter-clockwise seen from above.
"""


def yaw_of(quat_xyzw) -> float:
    """Yaw about Z [rad] of an (x, y, z, w) quaternion; the input is normalised first."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("quaternion has zero length")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def anchor_yaw_offset() -> float:
    """:data:`ANCHOR_YAW_ENV` in radians, 0.0 if unset or unparseable."""
    raw = os.environ.get(ANCHOR_YAW_ENV)
    if not raw:
        return 0.0
    try:
        return math.radians(float(raw))
    except ValueError:
        return 0.0


def anchor_rot_for_yaw(spawn_yaw: float) -> tuple[float, float, float, float]:
    """``anchor_rot`` (x, y, z, w) that starts an operator facing ``spawn_yaw`` [rad]."""
    yaw = spawn_yaw - XR_ANCHOR_FORWARD_YAW + anchor_yaw_offset()
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def anchor_rot_for_spawn(rot) -> tuple[float, float, float, float]:
    """:func:`anchor_rot_for_yaw` off a spawn rotation quaternion (x, y, z, w)."""
    return anchor_rot_for_yaw(yaw_of(rot))
