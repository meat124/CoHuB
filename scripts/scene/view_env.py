"""Open a task's scene in a Kit window, with no headset, recorder or terminations.

    python scripts/scene/view_env.py --task Isaac-Handover-G1x2-v0
    python scripts/scene/view_env.py --paused                       # physics stopped
    python scripts/scene/view_env.py --ego_view both                # + ego windows

Keys, after clicking the viewport:

    SPACE   pause / resume physics
    R       reset the episode
    P       print every asset's world pose
    S       print the pose of the selected prim, or of the pink probe
    1       free camera
    2 3 5   Robot A / B / C's eye
    4       the recorded third-person view
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Open a task's scene in the Kit viewport and idle.")
parser.add_argument("--task", type=str, default="Isaac-CoCarry-G1x2-v0", help="Name of the task.")
parser.add_argument("--paused", action="store_true", help="Start with physics stopped (layout mode).")
parser.add_argument(
    "--eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Initial viewport camera position (m)."
)
parser.add_argument(
    "--lookat", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Initial viewport camera target (m)."
)
# Same flag names as record_demos.py.
def _ego_robot(value: str) -> str:
    """Accept ``a`` for ``robot_a``."""
    short = value.lower()
    return f"robot_{short}" if short in ("a", "b", "c") else short


parser.add_argument(
    "--ego_view",
    type=_ego_robot,
    choices=("none", "robot_a", "robot_b", "robot_c", "both", "all"),
    default="none",
    help=(
        "Open a window per robot ego view ('both' = A and B, 'all' = every robot). Number keys work without it."
    ),
)
parser.add_argument("--ego_view_size", type=str, default="640x480", help="WxH of each ego window.")
parser.add_argument(
    "--ego_view_focal",
    type=float,
    default=None,
    help="Override the ego lens [mm]. Default is the recorded sensor's own, so the window frames what the dataset does.",
)
# Same arm-IK flags as record_demos.py.
parser.add_argument("--elbow_clearance", type=float, default=0.24, metavar="M",
                    help="Minimum distance of each elbow from the torso's midplane [m] (a soft IK task). 0 restores "
                    "upstream.")
parser.add_argument("--elbow_clearance_cost", type=float, default=10.0,
                    help="Weight of the elbow clearance task.")
AppLauncher.add_app_launcher_args(parser)
# This Isaac Lab runs headless unless a visualizer is named. Pink IK is CPU-only.
parser.set_defaults(device="cpu", visualizer=["kit"])
args_cli = parser.parse_args()

from cohub.task_registry import gym_id  # noqa: E402

args_cli.task = gym_id(args_cli.task)

if args_cli.headless or "kit" not in (args_cli.visualizer or []):
    parser.error("this script *is* the window: --headless, and a --viz without 'kit', leave nothing to look at.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import weakref

import carb
import gymnasium as gym
import numpy as np
import omni
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import cohub.tasks  # noqa: F401
from cohub.teleop.follower import ConstantFollower, follower_action
from cohub.runtime import configure_cpu_runtime, configure_render
from cohub.g1.geometry import DEFAULT_HEIGHT, DEFAULT_WIDTH, EGO_MOUNT_LINK
from cohub.keys import ROBOTS3
from cohub.recording.recorder import robots_present
from cohub.teleop.view import create_ego_camera, open_ego_view

# With a Korean IME active Kit delivers the Hangul character of the key.
_KEY_ALIASES = {" ": "SPACE", "ㄱ": "R", "ㅔ": "P", "ㄴ": "S"}

_BOUND_KEYS = ("SPACE", "R", "P", "S", "1", "2", "3", "4", "5")


class ViewerKeys:
    """Key presses from the Kit window, as one-shot flags the loop drains."""

    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_key(event, *args),
        )
        self._hits: set[str] = set()

    def _on_key(self, event, *args) -> bool:
        raw = event.input
        name = (raw if isinstance(raw, str) else raw.name).upper()
        # Digit row and numpad keys are treated alike.
        name = name.removeprefix("KEY_").removeprefix("NUMPAD_")
        name = _KEY_ALIASES.get(name, name)
        etype = event.type
        is_press = etype == carb.input.KeyboardEventType.KEY_PRESS or str(etype).endswith("KEY_PRESS")
        # Kit delivers a press twice, so hits are a set.
        if is_press and name in _BOUND_KEYS:
            self._hits.add(name)
        return True

    def take(self, name: str) -> bool:
        if name in self._hits:
            self._hits.discard(name)
            return True
        return False

    def close(self) -> None:
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None


def _row(value) -> np.ndarray:
    """First env's row of a scene quantity, whether or not it is a ProxyArray."""
    tensor = value.torch if hasattr(value, "torch") else value
    return tensor[0].cpu().numpy()


def print_poses(env) -> None:
    """Print every articulation and rigid body's world pose. Quaternions are xyzw."""
    print(f"[view] world poses -- {args_cli.task}")
    groups = (("articulation", env.scene.articulations), ("rigid", env.scene.rigid_objects))
    for kind, group in groups:
        for name, asset in sorted(group.items()):
            pos = _row(asset.data.root_pos_w)
            x, y, z, w = _row(asset.data.root_quat_w)
            yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
            print(
                f"  {kind:<12} {name:<18} pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})"
                f"  quat_xyzw=({x:+.3f}, {y:+.3f}, {z:+.3f}, {w:+.3f})  yaw={yaw:+7.2f} deg"
            )


_PROBE_PATH = "/World/envs/env_0/ViewProbe"

_ROOM_PATH = "/World/envs/env_0/Room"
"""Prim path of a downloaded room. Absent when the task has none."""


def make_probe() -> str:
    """Define a pink pin at the origin, to drag onto a spot and read with ``S``.

    Visual only: no rigid body, no collider. The pin's base is the point.
    """
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = UsdGeom.Xform.Define(stage, _PROBE_PATH)
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    pink = [Gf.Vec3f(1.0, 0.15, 0.75)]
    stem = UsdGeom.Cylinder.Define(stage, _PROBE_PATH + "/Stem")
    stem.CreateRadiusAttr(0.004)
    stem.CreateHeightAttr(0.30)
    stem.CreateAxisAttr("Z")
    stem.CreateExtentAttr([(-0.004, -0.004, -0.15), (0.004, 0.004, 0.15)])
    stem.CreateDisplayColorAttr(pink)
    UsdGeom.Xformable(stem).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.15))

    ball = UsdGeom.Sphere.Define(stage, _PROBE_PATH + "/Ball")
    ball.CreateRadiusAttr(0.02)
    ball.CreateExtentAttr([(-0.02, -0.02, -0.02), (0.02, 0.02, 0.02)])
    ball.CreateDisplayColorAttr(pink)
    UsdGeom.Xformable(ball).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.30))

    return _PROBE_PATH


def _selected_paths() -> list[str]:
    """What is selected in the viewport or the Stage tree, in stage order."""
    import omni.usd

    return list(omni.usd.get_context().get_selection().get_selected_prim_paths())


def print_selection(env) -> None:
    """Print the selected prims' poses, read from USD, in the task frame.

    With a downloaded room the position is also printed in the room's own
    frame. Reports the probe when nothing is selected.
    """
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    paths = _selected_paths()
    if not paths:
        paths = [_PROBE_PATH]
        print("[view] nothing selected -- reporting the probe")

    origin = _row(env.scene.env_origins)

    room = stage.GetPrimAtPath(_ROOM_PATH)
    room_inv = None
    if room.IsValid():
        room_inv = (
            UsdGeom.Xformable(room)
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            .GetInverse()
        )

    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"[view] {path}: no such prim")
            continue
        xform = UsdGeom.Xformable(prim)
        if not xform:
            print(f"[view] {path}: not transformable")
            continue
        matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world = matrix.ExtractTranslation()
        # Remove scale and shear before extracting the rotation.
        quat = matrix.RemoveScaleShear().ExtractRotationQuat()
        w = quat.GetReal()
        x, y, z = quat.GetImaginary()
        yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        task = [world[i] - origin[i] for i in range(3)]
        print(f"[view] {path}")
        print(
            f"       task frame  pos=({task[0]:+.3f}, {task[1]:+.3f}, {task[2]:+.3f})"
            f"  quat_xyzw=({x:+.3f}, {y:+.3f}, {z:+.3f}, {w:+.3f})  yaw={yaw:+7.2f} deg"
        )
        if room_inv is not None:
            in_room = room_inv.Transform(world)
            print(
                f"       room frame  pos=({in_room[0]:+.3f}, {in_room[1]:+.3f},"
                f" {in_room[2]:+.3f})"
            )


_ROBOT_PRIMS = {name: f"Robot_{name.rsplit('_', 1)[1].upper()}" for name in ROBOTS3}
"""Scene entity name -> prim name, for every robot the platform knows."""


# Camera name -> (width, height) the dataset stores. The viewport renders
# recorded views at this shape.
_RECORDED_SHAPE: dict[str, tuple[int, int]] = {}


def _class_default(cls, name):
    """A configclass field's class default, whatever the instance holds."""
    import dataclasses

    for field in dataclasses.fields(cls):
        if field.name != name:
            continue
        if field.default_factory is not dataclasses.MISSING:
            return field.default_factory()
        return None if field.default is dataclasses.MISSING else field.default
    return None


def read_scene_camera(env_cfg):
    """The recorded third-person camera's pose and lens, or ``None``.

    Must be read before the camera sensors are stripped from the cfg.
    """
    cam = getattr(env_cfg.scene, "scene_cam", None)
    if cam is None:
        # Stripped when launched without --enable_cameras. The class default
        # still holds the pose.
        cam = _class_default(type(env_cfg.scene), "scene_cam")
    if cam is None:
        return None
    spawn = getattr(cam, "spawn", None)
    return {
        "pos": tuple(cam.offset.pos),
        "rot": tuple(cam.offset.rot),
        "convention": cam.offset.convention,
        "focal_length": getattr(spawn, "focal_length", 12.0),
        "horizontal_aperture": getattr(spawn, "horizontal_aperture", 20.955),
        "shape": (cam.width, cam.height),
    }


def _basis(rot) -> tuple[np.ndarray, np.ndarray]:
    """Forward and up of a ``convention="world"`` camera pose.

    ``+X`` is the view direction and ``+Z`` is up. Quaternions are xyzw.
    """
    from isaaclab.utils.math import quat_apply

    # The quaternion must be normalised before quat_apply.
    q = torch.tensor([rot], dtype=torch.float32)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    axis = lambda v: quat_apply(q, torch.tensor([v], dtype=torch.float32))[0].numpy()
    return axis((1.0, 0.0, 0.0)), axis((0.0, 0.0, 1.0))


def scene_camera_eye_lookat(spec) -> tuple[tuple, tuple]:
    """Where to stand the free camera so it starts framed like the dataset."""
    forward, _ = _basis(spec["rot"])
    eye = np.asarray(spec["pos"], dtype=float)
    return tuple(float(v) for v in eye), tuple(float(v) for v in eye + 3.0 * forward)


def make_scene_camera(spec) -> str | None:
    """Define a USD camera on the recorded third-person pose and lens.

    A USD camera looks down its own -Z with +Y up. The sensor's pose is
    +X forward and +Z up.
    """
    import omni.usd
    from pxr import Gf, UsdGeom

    forward, up = _basis(spec["rot"])
    right = np.cross(forward, up)
    stage = omni.usd.get_context().get_stage()
    path = "/World/envs/env_0/ViewSceneCam"
    camera = UsdGeom.Camera.Define(stage, path)
    rows = (right, up, -forward)
    UsdGeom.Xformable(camera).AddTransformOp().Set(
        Gf.Matrix4d(
            float(rows[0][0]), float(rows[0][1]), float(rows[0][2]), 0.0,
            float(rows[1][0]), float(rows[1][1]), float(rows[1][2]), 0.0,
            float(rows[2][0]), float(rows[2][1]), float(rows[2][2]), 0.0,
            float(spec["pos"][0]), float(spec["pos"][1]), float(spec["pos"][2]), 1.0,
        )
    )
    camera.CreateFocalLengthAttr(spec["focal_length"])
    width, height = spec["shape"]
    camera.CreateHorizontalApertureAttr(spec["horizontal_aperture"])
    camera.CreateVerticalApertureAttr(spec["horizontal_aperture"] * height / width)
    camera.CreateClippingRangeAttr().Set((0.05, 1000.0))
    _RECORDED_SHAPE["scene"] = (width, height)
    return path


def make_ego_cameras(robots) -> dict[str, str]:
    """Define one USD camera per robot, where the recorded ego sensor sits."""
    for robot in robots:
        _RECORDED_SHAPE[robot] = (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    return {
        robot: create_ego_camera(
            f"/World/envs/env_0/{_ROBOT_PRIMS[robot]}/{EGO_MOUNT_LINK}", name="ViewEgoCam"
        )
        for robot in robots
    }


def open_ego_windows(cameras: dict[str, str], robots) -> list:
    """The ``--ego_view`` windows, one per requested robot."""
    if args_cli.ego_view == "none":
        return []
    try:
        width, height = (int(v) for v in args_cli.ego_view_size.lower().split("x"))
    except ValueError:
        print(f"[view] bad --ego_view_size {args_cli.ego_view_size!r}; using 640x480")
        width, height = 640, 480

    if args_cli.ego_view == "all":
        wanted = robots
    elif args_cli.ego_view == "both":
        wanted = tuple(robot for robot in ("robot_a", "robot_b") if robot in robots)
    else:
        wanted = (args_cli.ego_view,)
    missing = [robot for robot in wanted if robot not in robots]
    if missing:
        print(f"[view] {args_cli.task} has no {', '.join(missing)}; skipping those ego windows")
        wanted = tuple(robot for robot in wanted if robot in robots)
    windows = []
    for index, robot in enumerate(wanted):
        window = open_ego_view(
            robot_prim_path=f"/World/envs/env_0/{_ROBOT_PRIMS[robot]}",
            title=f"Ego view - {robot}",
            width=width,
            height=height,
            position=(60 + index * (width + 20), 60),
            camera_path=cameras[robot],
            **({} if args_cli.ego_view_focal is None else {"focal_length": args_cli.ego_view_focal}),
        )
        if window is not None:
            windows.append(window)
    return windows


_VIEW_KEYS = {"robot_a": "2", "robot_b": "3", "robot_c": "5"}


def view_keys(robots) -> tuple[tuple[str, str], ...]:
    """``(key, robot)`` for the robots this task has, in layout order."""
    return tuple((_VIEW_KEYS[robot], robot) for robot in robots if robot in _VIEW_KEYS)


class MainViewport:
    """The main viewport, re-pointed by the number keys.

    Recorded views render at the shape the dataset stores and are letterboxed
    into the window. The free camera fills the window.
    """

    def __init__(self, cameras: dict[str, str]):
        self._cameras = cameras
        self._viewport = None
        self._free = None
        try:
            from omni.kit.viewport.utility import get_active_viewport

            self._viewport = get_active_viewport()
        except ImportError:
            return
        if self._viewport is not None:
            self._free = str(self._viewport.camera_path)

    @property
    def available(self) -> bool:
        return self._viewport is not None

    def _pin(self, shape: tuple[int, int] | None) -> str:
        """Render at ``shape``, or hand the frame back to Kit when ``None``."""
        try:
            if shape is None:
                self._viewport.fill_frame = True
                return "fills the window"
            self._viewport.fill_frame = False
            self._viewport.resolution = shape
            return f"renders {shape[0]}x{shape[1]} -- the recorded shape"
        except AttributeError:
            return "framing not pinned: this viewport has no settable resolution"

    def show(self, which: str) -> None:
        """``"free"``, ``"scene"``, or one of the ``robot_*`` names."""
        if self._viewport is None:
            return
        path = self._free if which == "free" else self._cameras.get(which)
        if path is None:
            print(f"[view] no {which} camera in this task")
            return
        self._viewport.camera_path = path
        note = self._pin(None if which == "free" else _RECORDED_SHAPE.get(which))
        print(f"[view] main viewport -> {which} ({path}); {note}")


def main():
    if configure_cpu_runtime():
        print("[runtime] CUDA unavailable: using ordinary CPU memory for Warp staging buffers")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task
    env_cfg.isaac_teleop = None
    env_cfg.recorders = None
    for name in ("upper_body_a", "upper_body_b", "upper_body_c"):
        term = getattr(env_cfg.actions, name, None)
        if term is None:
            continue
        term.elbow_clearance = args_cli.elbow_clearance
        term.elbow_clearance_cost = args_cli.elbow_clearance_cost
    if args_cli.elbow_clearance:
        print(f"[ik] elbow clearance {1000 * args_cli.elbow_clearance:.0f} mm from the torso midplane, cost {args_cli.elbow_clearance_cost:g}")
    else:
        print("[ik] elbow clearance OFF")
    # The scene camera's pose must be read before the sensors are removed.
    scene_spec = read_scene_camera(env_cfg)
    env_cfg = remove_camera_configs(env_cfg)
    configure_render(env_cfg.sim.render)

    # Terminations are off so the scene does not reset while it is inspected.
    for term in ("success", "time_out"):
        if getattr(env_cfg.terminations, term, None) is not None:
            setattr(env_cfg.terminations, term, None)

    # The free camera starts on the recorded third-person pose.
    if scene_spec is not None:
        env_cfg.viewer.eye, env_cfg.viewer.lookat = scene_camera_eye_lookat(scene_spec)
        fmt = lambda v: "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"
        print(
            f"[view] free camera starts on the recorded scene view:"
            f" eye {fmt(env_cfg.viewer.eye)} looking at {fmt(env_cfg.viewer.lookat)}"
        )
    if args_cli.eye is not None:
        env_cfg.viewer.eye = tuple(args_cli.eye)
    if args_cli.lookat is not None:
        env_cfg.viewer.lookat = tuple(args_cli.lookat)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    briefing = getattr(env_cfg, "operator_briefing", "")
    if briefing:
        print(briefing)
    print_poses(env)

    robots = robots_present(env)
    cameras = make_ego_cameras(robots)
    if scene_spec is not None:
        cameras["scene"] = make_scene_camera(scene_spec)
        shape = _RECORDED_SHAPE["scene"]
        print(f"[view] recorded third-person camera rebuilt at {cameras['scene']} ({shape[0]}x{shape[1]})")
    ego_windows = open_ego_windows(cameras, robots)
    main_viewport = MainViewport(cameras)
    make_probe()

    keys = ViewerKeys()
    paused = args_cli.paused
    print(
        f"[view] {args_cli.task} up, physics {'PAUSED' if paused else 'running'}"
        " -- click the viewport, then SPACE pause/resume, R reset, P print poses"
    )
    print(
        "[view] S prints the selected prim's pose -- or the pink probe's, which is"
        " there to be dragged onto a bare spot you want a number for"
    )
    if main_viewport.available:
        legend = [("1", "free camera"), ("4", "the recorded third-person view")]
        legend += [(key, f"Robot {robot[-1].upper()}'s eye") for key, robot in view_keys(robots)]
        shown = ", ".join(f"{key} {what}" for key, what in sorted(legend))
        print(f"[view] {shown} -- in the main viewport")
    if ego_windows:
        print(f"[view] {len(ego_windows)} ego window(s) open; each is its own render pass")

    follower = ConstantFollower()
    with torch.inference_mode():
        while simulation_app.is_running():
            if keys.take("SPACE"):
                paused = not paused
                print(f"[view] physics {'PAUSED' if paused else 'running'}")
            if keys.take("P"):
                print_poses(env)
            if keys.take("S"):
                print_selection(env)
            for key, which in (("1", "free"), ("4", "scene"), *view_keys(robots)):
                if keys.take(key):
                    main_viewport.show(which)
            if keys.take("R"):
                env.reset()
                print("[view] reset")
                print_poses(env)

            if paused:
                # render() pumps the Kit app loop, so the window stays live.
                env.sim.render()
                continue

            slices = [follower_action(env, name, follower) for name in robots]
            action = torch.as_tensor(np.concatenate(slices), dtype=torch.float32, device=env.device).unsqueeze(0)
            env.step(action)

    keys.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
