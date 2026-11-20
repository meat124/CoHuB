"""What the operator sees: the XR anchor, ego windows, the ego frame, their own hidden head, the status text."""

from __future__ import annotations

import logging
import math
import os
import time

from cohub.g1.geometry import (
    EGO_HEIGHT_ABOVE_PELVIS,
    EGO_MOUNT_LINK,
    EGO_OFFSET_POS,
    EGO_PITCH_DOWN_DEG,
    horizontal_fov_deg,
    PELVIS_SPAWN_Z,
    DEFAULT_APERTURE,
    DEFAULT_FOCAL_LENGTH,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    vertical_fov_deg,
    anchor_rot_for_spawn,
)


# --- XR anchor ---
# The anchor is one rigid transform for the eye and the controllers. --eye_height puts the eye on the
# robot's camera mount.

LEGACY_ANCHOR_Z = -0.95
"""The default ``anchor_pos.z`` [m]."""

DEFAULT_EYE_HEIGHT = 0.0
"""Off. Set ``--eye_height`` or ``COHUB_EYE_HEIGHT`` to your own eye height
[m] to move the view onto the robot's camera mount."""

NOMINAL_HAND_HEIGHT = 0.85
"""Hand height [m] used to report the controller height an anchor asks for."""


def anchor_rot_for(robot: str, scene_cfg) -> tuple[float, float, float, float]:
    """``anchor_rot`` that starts an operator facing the way their robot does.

    ``robot`` is the scene key (``"robot_a"``); ``scene_cfg`` the running task's ``env_cfg.scene``.
    """
    try:
        spawn = getattr(scene_cfg, robot)
    except AttributeError:
        raise ValueError(
            f"scene cfg has no {robot!r}; the platform contract names the robots robot_a and robot_b"
        ) from None
    if spawn is None:
        raise ValueError(f"scene cfg entry {robot!r} is unset")

    return anchor_rot_for_spawn(spawn.init_state.rot)


def default_eye_height() -> float:
    """``COHUB_EYE_HEIGHT`` if set, else :data:`DEFAULT_EYE_HEIGHT`."""
    raw = os.environ.get("COHUB_EYE_HEIGHT")
    if not raw:
        return DEFAULT_EYE_HEIGHT
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_EYE_HEIGHT


def anchor_z_for_eye_height(eye_height: float) -> float:
    """``anchor_pos.z`` that lands the operator's eye on the camera mount."""
    return EGO_HEIGHT_ABOVE_PELVIS - eye_height


def configured_pelvis_spawn_z(env_cfg) -> float:
    """Robot A's task-specific initial pelvis height, with the platform default as fallback."""
    try:
        return float(env_cfg.scene.robot_a.init_state.pos[2])
    except (AttributeError, IndexError, TypeError):
        return PELVIS_SPAWN_Z


def controller_height_for(
    anchor_z: float,
    scene_height: float = NOMINAL_HAND_HEIGHT,
    pelvis_spawn_z: float = PELVIS_SPAWN_Z,
) -> float:
    """Real-world controller height that reaches ``scene_height`` [m]."""
    return scene_height - (pelvis_spawn_z + anchor_z)


def calibrate_xr_view(env_cfg, eye_height: float | None = None) -> float:
    """Point the XR anchor so the eye lands on the camera mount.

    Returns the ``anchor_pos.z`` that was applied. Passing ``eye_height <= 0``
    leaves the task's own anchor alone.
    """
    if eye_height is None:
        eye_height = default_eye_height()
    if getattr(env_cfg, "xr", None) is None:
        return 0.0
    if eye_height <= 0.0:
        return env_cfg.xr.anchor_pos[2]

    anchor_z = anchor_z_for_eye_height(eye_height)
    _set_anchor_z(env_cfg, anchor_z)
    return anchor_z


def _set_anchor_z(env_cfg, anchor_z: float) -> None:
    """Write ``anchor_pos.z`` on both copies of the XR config.

    XY is left alone: ``anchor_pos`` is added in world coordinates, so a
    forward offset would not turn with the robot.
    """
    # Python floats only: ``Vec3d`` rejects numpy scalars.
    for cfg in (
        getattr(env_cfg, "xr", None),
        getattr(getattr(env_cfg, "isaac_teleop", None), "xr_cfg", None),
    ):
        if cfg is not None:
            x, y, _ = cfg.anchor_pos
            cfg.anchor_pos = (float(x), float(y), float(anchor_z))


_BASE_ANCHOR_Z: float | None = None
"""The standing anchor height, latched on the first :func:`set_anchor_dz`."""


def set_anchor_dz(env_cfg, dz: float) -> None:
    """Move the operator's viewpoint by ``dz`` [m] to ride the robot's crouch.

    Lowering the anchor by the crouch depth keeps the pelvis-relative wrist
    targets unchanged. Follows the commanded height, not the measured pelvis.
    """
    _set_anchor_z(env_cfg, _base_anchor_z(env_cfg) + dz)


def mount_rise_over_pelvis(asset) -> float:
    """How far the ego camera mount sits above the pelvis, right now [m].

    Standing this is :data:`EGO_HEIGHT_ABOVE_PELVIS`. It grows slightly in a
    crouch because the torso pitches back.
    """
    import numpy as np

    data = asset.data
    index = list(data.body_names).index(EGO_MOUNT_LINK)
    torso = data.body_pos_w.torch[0, index].cpu().numpy()
    quat = data.body_quat_w.torch[0, index].cpu().numpy()
    x, y, z, w = quat
    # World z of R @ offset uses the bottom row of R.
    offset = np.asarray(EGO_OFFSET_POS, dtype=np.float64)
    bottom_row = np.array(
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]
    )
    mount_z = float(torso[2]) + float(bottom_row @ offset)
    # Python float: see :func:`_set_anchor_z`.
    return float(mount_z - float(data.root_pos_w.torch[0, 2].cpu()))


def _base_anchor_z(env_cfg) -> float:
    """The standing anchor height, latched the first time it is asked for."""
    global _BASE_ANCHOR_Z
    if _BASE_ANCHOR_Z is None:
        _BASE_ANCHOR_Z = env_cfg.xr.anchor_pos[2] if getattr(env_cfg, "xr", None) else 0.0
    return _BASE_ANCHOR_Z


def describe_xr_view(
    eye_height: float | None = None, pelvis_spawn_z: float = PELVIS_SPAWN_Z
) -> str:
    """A startup banner describing the anchor and controller heights."""
    if eye_height is None:
        eye_height = default_eye_height()
    if eye_height <= 0.0:
        return (
            f"[xr view] anchor left at {LEGACY_ANCHOR_Z:+.2f}: eye near "
            f"{pelvis_spawn_z + LEGACY_ANCHOR_Z + 1.70:.2f} m, controllers near "
            f"{controller_height_for(LEGACY_ANCHOR_Z, pelvis_spawn_z=pelvis_spawn_z):.2f} m\n"
            f"          the robot's head is hidden so it cannot block the view; "
            f"--no-hide_own_head keeps it\n"
            f"          --eye_height <your eye height> instead puts the view on the "
            f"camera mount, matching the recorded images (lower vantage point)"
        )

    anchor_z = anchor_z_for_eye_height(eye_height)
    return (
        f"[xr view] operator eye height {eye_height:.2f} m\n"
        f"          anchor_pos.z {anchor_z:+.3f} -> eye at world z "
        f"{pelvis_spawn_z + anchor_z + eye_height:.3f} m, on the d435 camera mount\n"
        f"          hold the controllers near "
        f"{controller_height_for(anchor_z, pelvis_spawn_z=pelvis_spawn_z):.2f} m to put the "
        f"robot's hands at {NOMINAL_HAND_HEIGHT:.2f} m\n"
        f"          the robot's hand is exactly where your controller is -- nothing is offset\n"
        f"          recorded ego camera: {horizontal_fov_deg():.0f} deg horizontal, "
        f"{EGO_PITCH_DOWN_DEG:g} deg down, same origin\n"
        f"          this vantage point is lower than the default -- try it seated, "
        f"or --eye_height 0 to go back"
    )


# --- Ego windows on the monitor ---
# A Kit viewport on the robot's ego mount, with the recorded sensor's lens.

logger = logging.getLogger(__name__)

DEFAULT_APERTURE_XY = (DEFAULT_APERTURE, DEFAULT_APERTURE * DEFAULT_HEIGHT / DEFAULT_WIDTH)
"""The window's film back (horizontal, vertical) [mm], matching the recorded
sensor."""


def camera_basis_rows(pitch_down_deg: float = EGO_PITCH_DOWN_DEG):
    """Camera axes expressed in mount-link coordinates, as rows.

    * camera ``+X`` (image right) -> link ``-Y``
    * camera ``+Y`` (image up)    -> the pitched up axis
    * camera ``+Z``               -> minus the pitched forward axis
    """
    t = math.radians(pitch_down_deg)
    return (
        (0.0, -1.0, 0.0),
        (math.sin(t), 0.0, math.cos(t)),
        (-math.cos(t), 0.0, math.sin(t)),
    )


def camera_transform(
    offset: tuple[float, float, float] = EGO_OFFSET_POS,
    pitch_down_deg: float = EGO_PITCH_DOWN_DEG,
):
    """The camera's local transform under the mount link, as a ``Gf.Matrix4d``."""
    from pxr import Gf

    r = camera_basis_rows(pitch_down_deg)
    return Gf.Matrix4d(
        r[0][0], r[0][1], r[0][2], 0.0,
        r[1][0], r[1][1], r[1][2], 0.0,
        r[2][0], r[2][1], r[2][2], 0.0,
        offset[0], offset[1], offset[2], 1.0,
    )


def create_ego_camera(
    link_prim_path: str,
    name: str = "EgoCam",
    offset: tuple[float, float, float] = EGO_OFFSET_POS,
    focal_length: float = DEFAULT_FOCAL_LENGTH,
    pitch_down_deg: float = EGO_PITCH_DOWN_DEG,
    aperture: tuple[float, float] = DEFAULT_APERTURE_XY,
) -> str:
    """Define a camera prim under ``link_prim_path`` and return its path.

    ``offset`` is the eye point in link-local metres; ``focal_length`` and
    ``aperture`` are in mm and default to the recorded sensor's.
    """
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    path = f"{link_prim_path}/{name}"
    camera = UsdGeom.Camera.Define(stage, path)
    UsdGeom.Xformable(camera).AddTransformOp().Set(camera_transform(offset, pitch_down_deg))
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(aperture[0])
    camera.CreateVerticalApertureAttr(aperture[1])
    # Same near plane as the recorded sensor.
    camera.CreateClippingRangeAttr().Set((0.05, 1000.0))
    return path


def open_ego_view(
    robot_prim_path: str,
    title: str,
    width: int = 640,
    height: int = 480,
    position: tuple[int, int] = (60, 60),
    offset: tuple[float, float, float] = EGO_OFFSET_POS,
    focal_length: float = DEFAULT_FOCAL_LENGTH,
    pitch_down_deg: float = EGO_PITCH_DOWN_DEG,
    camera_path: str | None = None,
    resolution: tuple[int, int] | None = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
):
    """Show a robot ego camera in its own window. Returns it, or ``None`` without a UI.

    With ``camera_path`` the viewport reuses that camera; otherwise a monitor-only
    one is created. ``resolution`` is what it renders, ``width``/``height`` the window.
    """
    if camera_path is None:
        camera_path = create_ego_camera(
            f"{robot_prim_path}/{EGO_MOUNT_LINK}",
            offset=offset,
            focal_length=focal_length,
            pitch_down_deg=pitch_down_deg,
        )
    try:
        from omni.kit.viewport.utility import create_viewport_window
    except ImportError:
        logger.info("viewport extension unavailable (headless?); ego view %s not shown", title)
        return None

    window = create_viewport_window(
        name=title,
        camera_path=camera_path,
        width=width,
        height=height,
        position_x=position[0],
        position_y=position[1],
    )
    if window is None:
        logger.warning("could not create ego view window %s", title)
        return window

    if resolution is not None:
        api = getattr(window, "viewport_api", None)
        try:
            api.fill_frame = False
            api.resolution = resolution
        except AttributeError:
            logger.warning("viewport %s has no settable resolution; framing may not match the dataset", title)
        else:
            print(f"[ego view] {title} renders {resolution[0]}x{resolution[1]} -- the recorded shape")
    print(f"[ego view] {title} -> {camera_path}")
    return window


# --- Ego frame in the headset ---
# Four bars on the plane `distance` metres down the ego camera's axis, inner edges on the recorded
# frustum. Scene geometry: the robot's own ego camera cannot see it, other cameras can.

# omni.ui packs colours as 0xAABBGGRR.
FRAME_COLOURS = {
    "red": 0xFF0000FF,
    "white": 0xFFFFFFFF,
    "green": 0xFF00FF00,
    "yellow": 0xFF00FFFF,
    "cyan": 0xFFFFFF00,
    "magenta": 0xFFFF00FF,
}
DEFAULT_FRAME_COLOUR = "red"

DEFAULT_FRAME_DISTANCE = 1.0
"""Metres down the camera axis to draw the frame."""

DEFAULT_THICKNESS_FRAC = 0.02
"""Bar thickness as a fraction of ``distance``."""

DEFAULT_MARGIN_PX = 1.0
"""How far outside the recorded image the bars start, in recorded pixels."""

RESOLUTION_SCALE = 300
"""UI pixels per metre for the bar textures."""


def frame_extent(
    distance: float = DEFAULT_FRAME_DISTANCE,
    focal_length: float = DEFAULT_FOCAL_LENGTH,
    aperture: float = DEFAULT_APERTURE,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> tuple[float, float]:
    """Full width and height [m] of the recorded image ``distance`` down the axis."""
    half_w = distance * (0.5 * aperture) / focal_length
    half_h = half_w * (height / width)
    return 2.0 * half_w, 2.0 * half_h


def bar_layout(
    frame_w: float,
    frame_h: float,
    thickness: float,
    margin: float = 0.0,
) -> list[tuple[float, float, float, float]]:
    """Four bars around a ``frame_w`` x ``frame_h`` rectangle, as ``(cx, cy, w, h)``.

    In the plane's own coordinates, +x right and +y up, origin at the frame's
    centre. Every bar lies outside the rectangle grown by ``margin``.
    """
    inner_x = 0.5 * frame_w + margin
    inner_y = 0.5 * frame_h + margin
    span_x = 2.0 * inner_x + 2.0 * thickness
    span_y = 2.0 * inner_y
    return [
        (0.0, inner_y + 0.5 * thickness, span_x, thickness),  # top
        (0.0, -(inner_y + 0.5 * thickness), span_x, thickness),  # bottom
        (-(inner_x + 0.5 * thickness), 0.0, thickness, span_y),  # left
        (inner_x + 0.5 * thickness, 0.0, thickness, span_y),  # right
    ]


def describe_ego_frame(
    robot: str,
    distance: float = DEFAULT_FRAME_DISTANCE,
    colour: str = DEFAULT_FRAME_COLOUR,
    recording: bool = False,
) -> str:
    """A startup line saying what the frame is, and what it costs here."""
    w, h = frame_extent(distance)
    text = (
        f"[ego frame] {robot}: {colour} frame {w:.2f} x {h:.2f} m at {distance:.2f} m, "
        f"the recorded {DEFAULT_WIDTH}x{DEFAULT_HEIGHT} "
        f"({horizontal_fov_deg():.0f} x {vertical_fov_deg():.0f} deg, {EGO_PITCH_DOWN_DEG:g} deg down)\n"
        f"            exact for things at that distance; nearer and further it is a guide"
    )
    if recording:
        text += (
            "\n            it is scene geometry: this robot's own ego camera cannot see it, but the\n"
            "            partner's ego camera and the scene camera can -- like the hidden head,\n"
            "            render_dataset_images.py rebuilds those views; --no-ego_frame turns it off"
        )
    return text


class EgoFrame:
    """The recorded frame's boundary, drawn into the headset on the robot's camera.

    ``distance`` is metres down the camera axis; ``colour`` a key of
    :data:`FRAME_COLOURS` or a packed ``0xAABBGGRR`` int. The lens arguments default to
    the recorded sensor's.
    """

    def __init__(
        self,
        robot_prim_path: str,
        enabled: bool = True,
        distance: float = DEFAULT_FRAME_DISTANCE,
        colour: str | int = DEFAULT_FRAME_COLOUR,
        thickness_frac: float = DEFAULT_THICKNESS_FRAC,
        margin_px: float = DEFAULT_MARGIN_PX,
        offset: tuple[float, float, float] = EGO_OFFSET_POS,
        pitch_down_deg: float = EGO_PITCH_DOWN_DEG,
        focal_length: float = DEFAULT_FOCAL_LENGTH,
        aperture: float = DEFAULT_APERTURE,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ):
        if distance <= 0.0:
            raise ValueError(f"ego frame distance must be positive, got {distance}")
        self.robot_prim_path = robot_prim_path
        self.enabled = bool(enabled)
        self.distance = float(distance)
        self.colour = FRAME_COLOURS[colour] if isinstance(colour, str) else int(colour)
        self.thickness = float(thickness_frac) * self.distance
        self._offset = tuple(float(v) for v in offset)
        self._pitch_down_deg = float(pitch_down_deg)
        self._lens = (float(focal_length), float(aperture), int(width), int(height))
        # One recorded pixel, in metres on the frame's plane.
        w, h = self.extent
        self.margin = float(margin_px) * (w / width)
        self._container = None
        self._components: list = []

    @property
    def extent(self) -> tuple[float, float]:
        """Full width and height of the recorded image at ``distance`` [m]."""
        return frame_extent(self.distance, *self._lens)

    @property
    def bars(self) -> list[tuple[float, float, float, float]]:
        """The four bars as ``(cx, cy, w, h)`` in the frame plane's coordinates."""
        w, h = self.extent
        return bar_layout(w, h, self.thickness, self.margin)

    @property
    def mount_prim_path(self) -> str:
        return f"{self.robot_prim_path}/{EGO_MOUNT_LINK}"

    @property
    def shown(self) -> bool:
        return self._container is not None

    def show(self) -> bool:
        """Put the frame in the headset. Returns True if it is up."""
        if not self.enabled:
            return False
        if self._container is not None:
            self._container.visible = True
            return True
        try:
            self._container = self._build()
        except Exception as exc:  # noqa: BLE001
            logger.info("ego frame unavailable, disabling: %s", exc)
            print(f"[ego frame] OFF - {exc}")
            self.enabled = False
            return False
        return True

    def hide(self) -> None:
        """Take the frame down. :meth:`show` brings it back without rebuilding."""
        if self._container is not None:
            self._container.visible = False

    def clear(self) -> None:
        """Destroy the frame's widgets. The next :meth:`show` rebuilds them."""
        container, self._container = self._container, None
        self._components = []
        if container is None:
            return
        try:
            container.root.clear()
        except Exception as exc:  # noqa: BLE001
            logger.info("ego frame clear failed: %s", exc)

    def _build(self):
        """Create the scene-view container and its four bars."""
        import omni.ui as ui
        from omni.kit.scene_view.xr import XRSceneView
        from omni.kit.scene_view.xr_utils import SpatialSource, UiContainer, WidgetComponent
        from omni.ui import scene as sc
        from pxr import Gf

        # Spatial sources compose in order: prim -> camera -> plane.
        space_stack = [
            SpatialSource.new_prim_path_source(self.mount_prim_path),
            SpatialSource.new_transform_matrix_source(camera_transform(self._offset, self._pitch_down_deg)),
            SpatialSource.new_translation_source(Gf.Vec3d(0.0, 0.0, -self.distance)),
        ]
        container = UiContainer(XRSceneView, space_stack=space_stack)

        components = []
        for cx, cy, w, h in self.bars:
            component = WidgetComponent(
                ui.Rectangle,
                width=w,
                height=h,
                resolution_scale=RESOLUTION_SCALE,
                widget_kwargs={"style": {"background_color": self.colour}},
            )
            container.manipulator.add_component(component, offset=sc.Matrix44.get_translation_matrix(cx, cy, 0.0))
            components.append(component)
        # Keep references so the widgets stay alive.
        self._components = components
        return container


# --- Hiding the operator's own head ---
# The head shell is made invisible in USD, so it is hidden for every camera in the process.

HEAD_PRIM = "torso_link/visuals/head_link"
"""Where the head shell lives under a G1 prim."""


def _make_authorable(stage, path: str) -> bool:
    """Clear the instanceable flag on whatever ancestor is providing it.

    Returns True when ``path`` can be authored on afterwards.
    """
    from pxr import Sdf

    prim = stage.GetPrimAtPath(path)
    if prim.IsValid() and not prim.IsInstanceProxy():
        return True

    # The instance root is the nearest ancestor with the flag set. There can
    # be more than one on the way.
    cleared = False
    current = Sdf.Path(path)
    ancestors = []
    while current != Sdf.Path.absoluteRootPath:
        ancestors.append(current)
        current = current.GetParentPath()
    for ancestor in reversed(ancestors):
        node = stage.GetPrimAtPath(ancestor)
        if node.IsValid() and node.IsInstanceable():
            node.SetInstanceable(False)
            cleared = True

    prim = stage.GetPrimAtPath(path)
    return bool(prim.IsValid() and not prim.IsInstanceProxy()) and cleared


def hide_head(robot_prim_path: str) -> bool:
    """Hide one robot's head shell. Returns True if it is now invisible."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    path = f"{robot_prim_path}/{HEAD_PRIM}"

    if not stage.GetPrimAtPath(path).IsValid():
        logger.warning("no head shell at %s; nothing hidden", path)
        return False

    if not _make_authorable(stage, path):
        logger.warning("could not clear instancing for %s; head stays visible", path)
        return False

    imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
    imageable.MakeInvisible()
    visible = imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
    if visible:
        logger.warning("%s still computes as visible", path)
        return False

    print(f"[xr view] hid {path} so it cannot block the headset operator's view")
    return True


# --- Status text ---
# Head-locked text drawn by Kit's XR scene view. Each process draws its own copy.

# omni.ui packs colours as 0xAABBGGRR.
WHITE = 0xFFFFFFFF
GREEN = 0xFF40E040
RED = 0xFF4040FF

SLOT = "/cohub_session_outcome"
"""Prim path the widget is copied to. One slot, so a new message replaces the
previous one. Must be a root path: intermediate prims are not created."""


class SessionHud:
    """Session outcomes as head-locked text. A no-op without a headset or the XR widget extension.

    ``distance`` and ``height`` place the text [m]; ``duration`` is how long it stays [s].
    """

    def __init__(
        self,
        enabled: bool = True,
        distance: float = 2.0,
        height: float = 0.45,
        duration: float = 4.0,
    ):
        self.enabled = bool(enabled)
        self._distance = float(distance)
        self._height = float(height)
        self._duration = float(duration)
        self._last: str | None = None
        self._last_at = 0.0

    def show(self, text: str, colour: int = WHITE, now: float | None = None) -> None:
        """Put ``text`` in front of the operator. Cheap to call every frame.

        The same text repeated is suppressed until the previous copy has timed
        out.
        """
        if not self.enabled:
            return
        now = time.monotonic() if now is None else now
        if text == self._last and now - self._last_at < self._duration:
            return
        self._last, self._last_at = text, now
        try:
            self._draw(text, colour)
        except Exception as exc:  # noqa: BLE001
            logger.info("session HUD unavailable, disabling: %s", exc)
            print(f"[hud] OFF - {exc}")
            self.enabled = False

    def _draw(self, text: str, colour: int) -> None:
        """Put the widget in the headset."""
        from isaaclab.ui.xr_widgets import show_instruction
        from pxr import Gf

        show_instruction(
            text=text,
            # -Z is forward from the XR camera, +Y up.
            prim_path_source="/_xr/stage/xrCamera",
            translation=Gf.Vec3f(0.0, self._height, -self._distance),
            display_duration=self._duration,
            font_size=0.13,
            text_color=colour,
            target_prim_path=SLOT,
        )

    def clear(self) -> None:
        """Take the text down now."""
        self._last = None
        if not self.enabled:
            return
        try:
            from isaaclab.ui.xr_widgets import hide_instruction

            hide_instruction(SLOT)
        except Exception as exc:  # noqa: BLE001
            logger.info("session HUD hide failed: %s", exc)
