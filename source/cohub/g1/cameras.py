"""The recorded cameras: one ego view per robot, a fixed scene view, and the eval video cut."""

from __future__ import annotations

from collections.abc import Sequence
import math

from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.camera.camera import Camera
import isaaclab.sim as sim_utils
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxManager
import torch
import warp as wp

from cohub.g1.geometry import (
    D435_FOCAL_LENGTH,
    D435_PITCH_DOWN_DEG,
    DEFAULT_APERTURE,
    DEFAULT_FOCAL_LENGTH,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEPTH,
    EGO_MOUNT_LINK,
    EGO_PITCH_DOWN_DEG,
    EGO_SENSOR_FRAME,
    RGB,
    pitch_down_quat_xyzw,
)


# --- Camera posed from its PhysX mount body ---
# Every update reads the mount body's transform from PhysX and writes the camera prim. For cameras on
# a moving articulation link.

class PhysicsMountedCamera(Camera):
    """Camera that follows the nearest PhysX rigid-body ancestor exactly."""

    def __init__(self, cfg):
        # ``_update_poses`` runs during initialization, so these must exist first.
        self._mount_body_view = None
        self._mount_pos_b = None
        self._mount_quat_b = None
        self._synced_pos_w = None
        super().__init__(cfg)

    @property
    def mount_pos_w(self):
        """Camera world position computed from the PhysX mount; ``None`` before the first sync.

        Compare it with ``CameraData.pos_w`` to check the prim followed the mount.
        """
        return self._synced_pos_w

    def _initialize_impl(self) -> None:
        # The camera prim must exist before its rigid-body ancestor is resolved.
        super()._initialize_impl()

        rigid_expr, fixed_pos_b, fixed_quat_b = self._resolve_rigid_body_ancestor_expr()
        physics_view = PhysxManager.get_physics_sim_view()
        if physics_view is None:
            raise RuntimeError("PhysX simulation view is unavailable for a mounted camera")
        self._mount_body_view = physics_view.create_rigid_body_view(rigid_expr.replace(".*", "*"))
        if self._mount_body_view.count != self._view.count:
            raise RuntimeError(
                "Mounted camera/body count mismatch: "
                f"{self._view.count} cameras vs {self._mount_body_view.count} bodies"
            )

        count = self._view.count
        pos = fixed_pos_b if fixed_pos_b is not None else (0.0, 0.0, 0.0)
        quat = fixed_quat_b if fixed_quat_b is not None else (0.0, 0.0, 0.0, 1.0)
        self._mount_pos_b = torch.tensor(pos, dtype=torch.float32, device=self._device).repeat(count, 1)
        self._mount_quat_b = torch.tensor(quat, dtype=torch.float32, device=self._device).repeat(count, 1)

        # Sync once before the first frame.
        self._sync_camera_to_mount()
        Camera._update_poses(self)

    def _sync_camera_to_mount(self) -> None:
        if self._mount_body_view is None:
            return

        transforms = self._mount_body_view.get_transforms()
        if isinstance(transforms, wp.array):
            transforms_t = wp.to_torch(transforms).reshape(-1, 7)
        else:
            transforms_t = transforms.reshape(-1, 7)
        transforms_t = transforms_t.to(device=self._device, dtype=torch.float32)

        body_pos_w = transforms_t[:, :3]
        body_quat_w = transforms_t[:, 3:7]
        camera_pos_w = body_pos_w + math_utils.quat_apply(body_quat_w, self._mount_pos_b)
        camera_quat_w = math_utils.quat_mul(body_quat_w, self._mount_quat_b)

        self._synced_pos_w = camera_pos_w.clone()
        self._view.set_world_poses(
            wp.from_torch(camera_pos_w.contiguous(), dtype=wp.vec3f),
            wp.from_torch(camera_quat_w.contiguous(), dtype=wp.quatf),
        )

    def _update_poses(
        self,
        env_ids: Sequence[int] | wp.array | None = None,
        env_mask: wp.array | None = None,
        frame_op: int = 0,
    ) -> None:
        self._sync_camera_to_mount()
        Camera._update_poses(self, env_ids=env_ids, env_mask=env_mask, frame_op=frame_op)


# --- Camera cfgs ---
# The ego camera sits on the G1's d435_link, pitched 25 degrees down, 102 degrees horizontal. Offsets
# use the "world" convention: forward +X, up +Z, rotations (x, y, z, w).

def _pinhole(focal_length: float, horizontal_aperture: float = DEFAULT_APERTURE) -> sim_utils.PinholeCameraCfg:
    return sim_utils.PinholeCameraCfg(
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        # The near plane must clear the head shell the camera sits in.
        clipping_range=(0.05, 20.0),
    )


def ego_camera_cfg(
    robot_prim: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    data_types: list[str] | None = None,
    focal_length: float = DEFAULT_FOCAL_LENGTH,
    pitch_down_deg: float = EGO_PITCH_DOWN_DEG,
) -> TiledCameraCfg:
    """The robot's head camera, at the hardware's own RealSense mount.

    ``focal_length`` [mm] defaults to the ZED Mini's 102 degrees horizontal;
    ``D435_FOCAL_LENGTH`` and ``D435_PITCH_DOWN_DEG`` are the G1 hardware's lens.
    """
    # The D435 frame is already pitched down; author only the difference.
    relative_pitch_deg = pitch_down_deg - D435_PITCH_DOWN_DEG
    cfg = TiledCameraCfg(
        prim_path=(
            f"{{ENV_REGEX_NS}}/{robot_prim}/{EGO_MOUNT_LINK}/"
            f"{EGO_SENSOR_FRAME}/EgoCamera"
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=pitch_down_quat_xyzw(relative_pitch_deg),
            convention="world",
        ),
        spawn=_pinhole(focal_length),
        width=width,
        height=height,
        data_types=data_types if data_types is not None else [RGB, DEPTH],
        update_latest_camera_pose=True,
    )
    cfg.class_type = PhysicsMountedCamera
    return cfg


def look_at_rot(
    eye: tuple[float, float, float], target: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    """The ``offset.rot`` (x, y, z, w) that points a camera at ``eye`` toward ``target``.

    For the ``"world"`` offset convention: local +X forward, +Z up.
    """
    dx, dy, dz = (t - e for t, e in zip(target, eye))
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length == 0.0:
        raise ValueError("a camera cannot look at the point it is standing on")
    dx, dy, dz = dx / length, dy / length, dz / length
    yaw = math.atan2(dy, dx)
    pitch = math.asin(max(-1.0, min(1.0, -dz)))
    # q_z(yaw) x q_y(pitch)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return (-sy * sp, cy * sp, sy * cp, cy * cp)


def scene_camera_cfg(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    data_types: list[str] | None = None,
    pos: tuple[float, float, float] = (-2.5, 0.0, 2.6),
    rot: tuple[float, float, float, float] = (0.0, 0.27855065, 0.0, 0.96042154),
    focal_length: float = 12.0,
    prim_name: str = "SceneCamera",
) -> TiledCameraCfg:
    """Fixed third-person view of the whole workspace.

    The default pose aims at (0.5, 0, 0.7) from the -X side. Give each extra
    fixed view its own ``prim_name``.
    """
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        offset=TiledCameraCfg.OffsetCfg(
            pos=pos,
            rot=rot,
            convention="world",
        ),
        spawn=_pinhole(focal_length=focal_length),
        width=width,
        height=height,
        data_types=data_types if data_types is not None else [RGB],
    )


def cameras_enabled() -> bool:
    """Whether the app was launched with ``--enable_cameras``."""
    try:
        import carb.settings

        return bool(carb.settings.get_settings().get("/isaaclab/cameras_enabled"))
    except Exception:  # noqa: BLE001
        return False


# --- Eval video cut ---
# A task declares one condition on its env cfg: a published check (checks_key + flag), or an asset's
# root past a coordinate (entity + axis + past). The eval client latches the cut the first frame it
# holds.

@configclass
class VideoCutCfg:
    """Film ``before`` until the condition holds, ``after`` from then on."""

    name: str = "scene_cut"
    """Video camera name the eval client offers."""

    before: str = "scene"
    after: str = "scene_2"

    checks_key: str = ""
    """Published-check condition: the predicate's ``cohub.subtasks`` key."""

    flag: str = ""
    """Published-check condition: the boolean that triggers the cut."""

    entity: str = ""
    """Plane condition: the scene asset whose root is watched."""

    axis: int = 0
    """Plane condition: env-frame axis (0, 1 or 2)."""

    past: float = 0.0
    """Plane condition: threshold on that axis [m]."""
