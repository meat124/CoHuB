"""Reading the scene on the primary and re-posing it on the secondary.

Every dynamic body travels: both robots, the carried object and the task's
dynamic props. The secondary overwrites the full state on every packet and
integrates nothing.
"""

from __future__ import annotations

import numpy as np
import torch

from cohub.keys import ROBOTS
from cohub.runtime import is_kinematic

ROBOT_KEYS = ROBOTS
"""The robots with named fields in the packet. A third robot travels in the
extras block."""
OBJECT_KEY = "object"

POSE_FLOATS = 7
"""Position (3) + quaternion (4), matching ``protocol.POSE_DIM``."""


def _pose(data) -> np.ndarray:
    """Root link pose as [x, y, z, qx, qy, qz, qw] for env 0."""
    return torch.cat([data.root_pos_w.torch[0], data.root_quat_w.torch[0]]).cpu().numpy().astype(np.float32)


def extra_keys(scene) -> list[str]:
    """The task's dynamic bodies besides the robots and the object.

    Sorted, articulations before rigid objects, so both processes lay the
    extras block out identically.
    """
    keys = [name for name in sorted(scene.articulations) if name not in ROBOT_KEYS]
    keys += [name for name in sorted(scene.rigid_objects) if name != OBJECT_KEY]
    return keys


def _joint_count(scene, key: str) -> int:
    """Joints of ``key``, or 0 for a body that has none."""
    return int(scene[key].num_joints) if key in scene.articulations else 0


def extra_floats(scene) -> int:
    """How many floats :func:`extra_state` produces for this scene."""
    return sum(POSE_FLOATS + _joint_count(scene, key) for key in extra_keys(scene))


def extra_state(scene) -> np.ndarray:
    """Pack the task's dynamic props: per entity, root pose then joint positions."""
    chunks: list[np.ndarray] = []
    for key in extra_keys(scene):
        asset = scene[key]
        chunks.append(_pose(asset.data))
        if key in scene.articulations:
            chunks.append(asset.data.joint_pos.torch[0].cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def robot_root_from_state(scene, state, robot_key: str) -> np.ndarray:
    """``robot_key``'s root pose as the packet carries it: [x, y, z, qx, qy, qz, qw].

    A third robot is read from the extras block. Raises ``KeyError`` if the robot is not in the scene.
    """
    if robot_key == ROBOT_KEYS[0]:
        return np.asarray(state.root_a, dtype=np.float32)
    if robot_key == ROBOT_KEYS[1]:
        return np.asarray(state.root_b, dtype=np.float32)

    cursor = 0
    for key in extra_keys(scene):
        if key == robot_key:
            return np.asarray(state.extras[cursor : cursor + POSE_FLOATS], dtype=np.float32)
        cursor += POSE_FLOATS + _joint_count(scene, key)
    raise KeyError(f"{robot_key!r} is not in this scene: it has {extra_keys(scene) or 'no props'}")


def publish(link, scene, *, episode_id: int = 0, running: bool = False, haptic=None, task: str = "") -> int:
    """Send one frame of scene state from the primary. Returns the sequence sent.

    ``haptic`` is the left/right load [N] on the hands of the robot this link's operator drives.
    """
    robot_a = scene[ROBOT_KEYS[0]].data
    robot_b = scene[ROBOT_KEYS[1]].data
    return link.send_state(
        root_a=_pose(robot_a),
        joint_q_a=robot_a.joint_pos.torch[0].cpu().numpy().astype(np.float32),
        root_b=_pose(robot_b),
        joint_q_b=robot_b.joint_pos.torch[0].cpu().numpy().astype(np.float32),
        object_pose=_pose(scene[OBJECT_KEY].data),
        haptic=None if haptic is None else np.asarray(haptic, dtype=np.float32).reshape(2),
        extras=extra_state(scene),
        episode_id=episode_id,
        running=running,
        task=task,
    )


def apply(scene, state, device: str) -> None:
    """Re-pose the secondary's copy of the scene from a received packet.

    Writes joint positions and targets and zeroes velocities. Raises ``ValueError``
    when the sender's joint count or extras block does not match this scene.
    """
    for key, root, joint_q in (
        (ROBOT_KEYS[0], state.root_a, state.joint_q_a),
        (ROBOT_KEYS[1], state.root_b, state.joint_q_b),
    ):
        robot = scene[key]
        if joint_q.shape[0] != robot.num_joints:
            raise ValueError(
                f"{key}: peer sent {joint_q.shape[0]} joints, this process's articulation has "
                f"{robot.num_joints}. The two processes are not running the same asset."
            )
        _pin_articulation(robot, joint_q, root, device)

    _pin(scene[OBJECT_KEY], state.object_pose, device)
    _apply_extras(scene, state.extras, device)


def _apply_extras(scene, values: np.ndarray, device: str) -> None:
    """Re-pose the task's own dynamic props from the packet's extras block."""
    expected = extra_floats(scene)
    if values.shape[0] != expected:
        raise ValueError(
            f"peer sent {values.shape[0]} floats of scene props, this process's scene needs {expected} "
            f"for {extra_keys(scene) or 'no props'}. The two processes are not running the same scene."
        )
    cursor = 0
    for key in extra_keys(scene):
        asset = scene[key]
        pose = values[cursor : cursor + POSE_FLOATS]
        cursor += POSE_FLOATS
        joints = _joint_count(scene, key)
        if joints:
            _pin_articulation(asset, values[cursor : cursor + joints], pose, device)
            cursor += joints
        else:
            _pin(asset, pose, device)


def _pin_articulation(asset, joint_q: np.ndarray, pose: np.ndarray, device: str) -> None:
    """Write an articulation's joints and root, leaving it nothing to integrate."""
    q = torch.as_tensor(np.asarray(joint_q), dtype=torch.float32, device=device).unsqueeze(0)
    asset.write_joint_position_to_sim_index(position=q)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
    # Targets too, or the PD actuators pull the joints back between packets.
    asset.set_joint_position_target(q)
    _pin(asset, pose, device)


def _pin(asset, pose: np.ndarray, device: str) -> None:
    """Put a body exactly at ``pose`` and leave it with nothing to integrate."""
    asset.write_root_link_pose_to_sim_index(
        root_pose=torch.as_tensor(pose, dtype=torch.float32, device=device).unsqueeze(0)
    )
    # A kinematic body has no velocity state to write.
    if is_kinematic(asset):
        return
    asset.write_root_link_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=device)
    )


def pose_error(scene, state) -> tuple[float, float]:
    """Largest joint and root-position disagreement with a received packet.

    Covers the robots, the object and the task's props.

    Returns:
        ``(max joint error [rad or m], max root position error [m])``.
    """
    joint_err = 0.0
    root_err = 0.0
    pairs = [
        (ROBOT_KEYS[0], state.root_a, state.joint_q_a),
        (ROBOT_KEYS[1], state.root_b, state.joint_q_b),
        (OBJECT_KEY, state.object_pose, None),
    ]
    cursor = 0
    for key in extra_keys(scene):
        pose = state.extras[cursor : cursor + POSE_FLOATS]
        cursor += POSE_FLOATS
        joints = _joint_count(scene, key)
        pairs.append((key, pose, state.extras[cursor : cursor + joints] if joints else None))
        cursor += joints

    for key, root, joint_q in pairs:
        data = scene[key].data
        if joint_q is not None:
            mine = data.joint_pos.torch[0].cpu().numpy()
            joint_err = max(joint_err, float(np.abs(mine - joint_q).max()))
        root_err = max(root_err, float(np.abs(data.root_pos_w.torch[0].cpu().numpy() - root[:3]).max()))
    return joint_err, root_err
