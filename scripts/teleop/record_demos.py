# Adapted from IsaacLab scripts/tools/record_demos.py (BSD-3-Clause,
# Isaac Lab Project Developers) for CoHuB's two-operator tasks.
"""Record multi-humanoid teleop demonstrations.

This process owns physics, the grasp term and the dataset. Each robot's 32D
action comes from a headset, a mirror process (``xr_mirror.py``) or a local
follower. ``scripts/teleop/collect.sh`` starts a whole session; by hand:

    python scripts/teleop/record_demos.py --xr --visualizer kit
    python scripts/teleop/xr_mirror.py --xr

Start both before a headset connects, with the same ``--task``.
"""

import argparse
import contextlib
import pathlib as _pathlib_early
import sys as _sys_early

from isaaclab.app import AppLauncher

# source/ must be on sys.path before the cohub imports below.
_sys_early.path.insert(0, str(_pathlib_early.Path(__file__).resolve().parents[2] / "source"))
from cohub.federation import ports_for_robot  # noqa: E402
from cohub.federation.link import TELEOP_ACTION_DIM as ROBOT_SLICE  # noqa: E402
from cohub.teleop.controls import ROT_TIME_CONSTANT as LEAN_COMP_DEFAULT_TAU  # noqa: E402
from cohub.teleop.view import (  # noqa: E402
    FRAME_COLOURS,
    DEFAULT_FRAME_COLOUR,
    DEFAULT_FRAME_DISTANCE,
)

_B_STATE_PORT, _B_ACTION_PORT = ports_for_robot("robot_b")
_A_STATE_PORT, _A_ACTION_PORT = ports_for_robot("robot_a")
_C_STATE_PORT, _C_ACTION_PORT = ports_for_robot("robot_c")

parser = argparse.ArgumentParser(description="Record multi-humanoid teleop demonstrations.")
parser.add_argument("--task", type=str, default="Isaac-CoCarry-G1x2-v0", help="Name of the task.")
parser.add_argument(
    "--robot_b_source",
    type=str,
    default="federated",
    choices=["federated", "follower"],
    help="Robot B's action source: 'federated' = a second headset (xr_mirror.py); 'follower' = a local device.",
)
parser.add_argument(
    "--follower",
    type=str,
    default="constant",
    choices=["constant", "keyboard"],
    help="Robot B's device with --robot_b_source follower; under 'federated' it only supplies session keys.",
)
parser.add_argument(
    "--robot_a_source",
    type=str,
    default="local",
    choices=["local", "federated"],
    help="Robot A's action source: 'local' = this process's headset; 'federated' = its own xr_mirror.py.",
)
parser.add_argument("--fed_peer", type=str, default="127.0.0.1", help="Host running the secondary process.")
parser.add_argument("--fed_state_port", type=int, default=_B_STATE_PORT, help="UDP port the secondary listens on.")
parser.add_argument("--fed_action_port", type=int, default=_B_ACTION_PORT, help="UDP port this process listens on.")
parser.add_argument(
    "--fed_a_peer", type=str, default="127.0.0.1", help="Host running operator 1's mirror, with --robot_a_source federated."
)
parser.add_argument(
    "--fed_a_state_port", type=int, default=_A_STATE_PORT, help="UDP port operator 1's mirror listens on."
)
parser.add_argument(
    "--fed_a_action_port", type=int, default=_A_ACTION_PORT, help="UDP port this process listens on for Robot A's slice."
)
parser.add_argument(
    "--robot_c_source",
    type=str,
    default="federated",
    choices=["federated", "follower"],
    help="Robot C's action source (three-robot tasks): 'federated' = a third headset; 'follower' = a hold pose.",
)
parser.add_argument(
    "--fed_c_peer", type=str, default="127.0.0.1", help="Host running operator 3's mirror."
)
parser.add_argument(
    "--fed_c_state_port", type=int, default=_C_STATE_PORT, help="UDP port operator 3's mirror listens on."
)
parser.add_argument(
    "--fed_c_action_port", type=int, default=_C_ACTION_PORT, help="UDP port this process listens on for Robot C's slice."
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default=None,
    help="Where to write demos. Default: datasets/<task>/<task>.hdf5.",
)
parser.add_argument(
    "--save",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="--no-save is a practice session: nothing is recorded, and episodes.txt logs the attempts as unsaved.",
)
parser.add_argument(
    "--resume_dataset",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Append to an existing --dataset_file, continuing its demo numbering. Off truncates it on startup.",
)
parser.add_argument(
    "--shard_demos",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Write one file per demonstration into <dataset dir>/data/ instead of a single HDF5.",
)
parser.add_argument(
    "--async_export",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Write each shard on a background thread (sharded sessions only). Off writes inside the loop.",
)
parser.add_argument(
    "--controller_session_buttons",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Start/pause on the right controller's A, reset on B. Replaces upstream's anchor-rotation toggle.",
)
parser.add_argument(
    "--export_mode",
    type=str,
    default="success",
    choices=["success", "all"],
    help="'success' saves only successful episodes; 'all' saves every episode with a success flag.",
)
parser.add_argument(
    "--vr_status",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show each demonstration's outcome as text in the headsets (an XR overlay; not in the recorded images).",
)
parser.add_argument(
    "--render_every",
    type=int,
    default=1,
    help="Control steps per rendered frame. Above 1 raises the control rate; keep 1 when recording images.",
)
parser.add_argument(
    "--antialiasing",
    type=str,
    default="auto",
    choices=["auto", "Off", "FXAA", "TAA", "DLSS", "DLAA"],
    help="RTX anti-aliasing. 'auto' = DLAA with --enable_cameras, otherwise Isaac Lab's DLSS.",
)
parser.add_argument("--num_success_steps", type=int, default=10)
parser.add_argument("--operators", type=str, default=str(_pathlib_early.Path(__file__).resolve().parents[2] / "source" / "cohub" / "recording" / "operators.yaml"),
                    help="Operator metadata YAML (copied into the dataset as configs/operators.yaml).")
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default="cloudxrjs",
    help="CloudXR .env path or shorthand ('cloudxrjs', 'avp', 'none').",
)
parser.add_argument("--auto_launch_cloudxr", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--wait_for_port",
    type=float,
    default=600.0,
    help="Seconds to wait when another runtime holds the CloudXR startup port (47998/udp). 0 fails immediately.",
)
parser.add_argument("--link_port", type=int, default=8080, help="Port for the headset landing page (0 disables).")
parser.add_argument("--stream_bitrate", type=int, default=50, help="Client streaming bitrate cap in Mbps (max 100).")
parser.add_argument("--stream_per_eye", type=int, default=1024, help="Client per-eye render size in pixels.")
parser.add_argument("--stream_fps", type=int, default=72, help="Client target frame rate.")
parser.add_argument(
    "--rate_report", type=int, default=250, help="Steps between control-rate reports (0 disables)."
)
parser.add_argument(
    "--ego_view",
    type=str,
    default=None,
    choices=["none", "robot_a", "robot_b", "robot_c", "both", "all"],
    help="Ego views to show in monitor windows ('both' = A and B). Default: robot_b on a follower, else none.",
)
parser.add_argument("--ego_view_size", type=str, default="640x480", help="Ego view window size, WxH.")
parser.add_argument(
    "--haptics",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Vibrate each Quest controller in proportion to that wrist's measured load.",
)
parser.add_argument(
    "--yaw_shaping",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Remap yaw commands onto the band where the locomotion policy actually turns.",
)
parser.add_argument(
    "--max_speed",
    type=float,
    default=None,
    help="Full-stick walk speed [m/s]. Default: loco_calibration.DEFAULT_MAX_SPEED (0.5). 0 disables walking.",
)
parser.add_argument(
    "--hide_own_head",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Hide the headset operator's own robot head, which blocks their view (--xr only). Recordings lack it.",
)
parser.add_argument(
    "--ego_frame",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Draw the recorded ego image's boundary in the headset (--xr only). Default: only without cameras.",
)
parser.add_argument(
    "--ego_frame_distance",
    type=float,
    default=DEFAULT_FRAME_DISTANCE,
    help="Distance along the camera axis at which the frame is drawn [m]. 1.0 is where the hands work.",
)
parser.add_argument(
    "--ego_frame_color",
    choices=sorted(FRAME_COLOURS),
    default=DEFAULT_FRAME_COLOUR,
    help="Colour of the frame.",
)
parser.add_argument(
    "--eye_height",
    type=float,
    default=0.0,
    help="Your eye height [m]: starts the view at the robot's camera mount. 0 keeps the default anchor.",
)
parser.add_argument(
    "--ego_view_focal", type=float, default=None, help="Ego camera focal length in mm (smaller is wider)."
)
parser.add_argument(
    "--crouch",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Left controller X lowers Robot A's hips, Y raises them. Walking is disabled while crouched.",
)
parser.add_argument(
    "--crouch_rate",
    type=float,
    default=0.25,
    help="Metres per second the hips move while X or Y is held.",
)
parser.add_argument(
    "--crouch_min",
    type=float,
    default=0.60,
    help="Lowest hip height the crouch allows [m]. Below 0.60 the robot topples when the arms reach.",
)
parser.add_argument(
    "--anchor_follows_crouch",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Lower the operator's viewpoint with the robot's hips, so the arms keep their pelvis-relative pose.",
)
parser.add_argument(
    "--ik_time_constant",
    type=float,
    default=0.0,
    help="Time the arms take to close 63%% of a pose error [simulated s]. 0 keeps the default; lower is faster.",
)
parser.add_argument(
    "--constant_time_constant",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Derive the IK gain from the control period and the solve count instead of upstream's fixed 0.075.",
)
parser.add_argument(
    "--retargeting",
    type=str,
    default="pipelined",
    choices=["pipelined", "sync"],
    help="'pipelined' (upstream) returns the previous frame's result; 'sync' computes this frame's in the loop.",
)
parser.add_argument(
    "--waist",
    type=str,
    default="all",
    choices=["all", "yaw", "none"],
    help="Waist joints the arm IK may use: 'all' (upstream), 'yaw' (turn, no lean) or 'none' (rigid).",
)
parser.add_argument(
    "--elbow_clearance",
    type=float,
    default=0.24,
    metavar="M",
    help="Minimum distance of each elbow from the torso's midplane [m] (a soft IK task). 0 restores upstream.",
)
parser.add_argument(
    "--elbow_clearance_cost",
    type=float,
    default=10.0,
    help="Weight of the elbow clearance task.",
)
parser.add_argument(
    "--max_fps",
    type=float,
    default=50.0,
    help="Cap on this process's loop rate [Hz]. 50 is the control rate (real time). 0 removes the cap.",
)
parser.add_argument(
    "--lean_comp",
    type=float,
    default=LEAN_COMP_DEFAULT_TAU,
    help="Tie the wrist targets to the torso's lean; time constant with the hand still [s]. inf = full, 0 = off.",
)
parser.add_argument(
    "--reach_feedback",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Print to the terminal when a wrist falls behind its controller (a target out of the arm's reach).",
)
AppLauncher.add_app_launcher_args(parser)
# Pink IK is CPU-only.
parser.set_defaults(device="cpu")
# Cameras are off by default. --enable_cameras turns them on.
parser.set_defaults(enable_cameras=False)
args_cli = parser.parse_args()

from cohub.task_registry import gym_id  # noqa: E402

args_cli.task = gym_id(args_cli.task)

def _task_slug(task: str) -> str:
    """``Isaac-CoCarry-G1x2-v0`` -> ``cocarry``, the dataset directory name."""
    name = task
    for prefix in ("Isaac-",):
        name = name[len(prefix):] if name.startswith(prefix) else name
    for suffix in ("-v0", "-v1", "-G1x2", "-G1x3"):
        while name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("-", "_").lower() or "demos"


if args_cli.dataset_file is None:
    _slug = _task_slug(args_cli.task)
    args_cli.dataset_file = f"./datasets/{_slug}/{_slug}.hdf5"

_FEDERATED = args_cli.robot_b_source == "federated"
_FED_A = args_cli.robot_a_source == "federated"
# Robot C's source is settled after gym.make, once the scene is known.
_FED_C = args_cli.robot_c_source == "federated"
if _FED_A and args_cli.xr:
    # A federated Robot A means this process has no headset operator.
    parser.error(
        "--robot_a_source federated moves operator 1 into their own mirror process; "
        "run this one without --xr and start scripts/teleop/xr_mirror.py --robot robot_a "
        f"--state_port {args_cli.fed_a_state_port} --action_port {args_cli.fed_a_action_port}"
    )
if args_cli.ego_view is None:
    # The ego window is for a Robot B operator without a headset.
    args_cli.ego_view = "none" if _FEDERATED else "robot_b"
if _FED_A and args_cli.auto_launch_cloudxr:
    # The mirrors own the CloudXR runtimes in this layout.
    args_cli.auto_launch_cloudxr = False
    print("[cloudxr] not launching a runtime: Robot A is federated, the mirrors own theirs")
if (_FEDERATED or _FED_A) and args_cli.follower == "constant" and not args_cli.headless:
    # With a remote operator the follower's keys are the local way to start,
    # pause or reset the session.
    args_cli.follower = "keyboard"

# The CloudXR profile must be applied before Kit starts.
import sys as _sys  # noqa: E402

from cohub.teleop.cloudxr import apply_profile as _apply_profile  # noqa: E402

CLOUDXR_PROFILE = _apply_profile("primary")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Skip the "Start AR" click, so the XR session starts when the headset connects.
if args_cli.xr:
    import carb.settings

    carb.settings.get_settings().set("/isaaclab/xr/auto_start", True)

import logging
import os
import signal
import time

import gymnasium as gym
import torch
import yaml

from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

from cohub.runtime import StepSplit
from cohub.subtasks import make_tracker

import cohub.tasks  # noqa: F401  (registers the CoHuB tasks)
from cohub.teleop.follower import ConstantFollower
from cohub.federation import PrimaryLink, scene_state
from cohub.recording.recorder import DetachedManagerTerm
from cohub.keys import ROBOTS3
from cohub.teleop.cloudxr import start_link_server, force_cleanup, wait_for_startup_port


def release_cloudxr() -> list[int]:
    """Clean up the CloudXR runtime this process owns, if it owns one.

    With ``--robot_a_source federated`` it owns none: the mirrors own theirs.
    """
    if _FED_A:
        return []
    return force_cleanup(install_dir=CLOUDXR_PROFILE.install_dir)


from cohub.teleop.controls import (
    CrouchControl,
    ControllerHaptics,
    object_interaction_forces,
    DEFAULT_MAX_SPEED,
    shape_walk,
    shape_yaw,
)
from cohub.teleop.view import EgoFrame, describe_ego_frame, open_ego_view

if args_cli.max_speed is None:
    args_cli.max_speed = DEFAULT_MAX_SPEED


def shape_locomotion(slice_tensor) -> None:
    """Walk-speed rescale and yaw shaping, in place, on one robot's 32D slice.

    Indices 28:31 are (vx, vy, wz). Every ingestion path goes through here.
    """
    vx, vy = shape_walk(float(slice_tensor[28]), float(slice_tensor[29]), args_cli.max_speed)
    slice_tensor[28] = vx
    slice_tensor[29] = vy
    if args_cli.yaw_shaping:
        slice_tensor[30] = shape_yaw(float(slice_tensor[30]))
from cohub.teleop.controls import (
    OperatorInputCapture,
    SessionButtons,
    release_upstream_a_binding,
    guard_xr_session,
    LeanCompensator,
)
from cohub.runtime import CpuLoad, describe_resources, configure_cpu_runtime, configure_render
from cohub.teleop.view import (
    hide_head,
    GREEN,
    RED,
    SessionHud,
    calibrate_xr_view,
    configured_pelvis_spawn_z,
    describe_xr_view,
    mount_rise_over_pelvis,
    set_anchor_dz,
)
from cohub.recording.dataset_writer import EpisodeLog, ResumableHDF5DatasetFileHandler, ShardedHDF5DatasetFileHandler
from cohub.recording.demo_source import stamp_env_args
from cohub.recording.recorder import SYNC_STATE
from cohub.g1.geometry import EGO_HEIGHT_ABOVE_PELVIS

logger = logging.getLogger(__name__)

_FOLLOWER_DEVICE_CODES = {"constant": 0, "keyboard": 1, "gamepad": 2, "quest": 3}

_CROUCH_CONTROLS = """
    X (left, hold)    crouch  -- hold until the hips are where you want them
    Y (left, hold)    stand back up -- stops at standing, never higher
    X + Y together    back to standing height (0.72 m)
                      walking is locked while crouched: walk up to the
                      object, stop, crouch, pick it up, then stand"""

_CONTROLS = """
============================================================
  Session ready -- {task} (follower={follower})

  Operator 1 -- Robot A, Quest
    thumbstick        walk / turn
    trigger (>60%)    close that hand -- this is the grasp
    release trigger   let go{crouch}

  Operator 2 -- Robot B, keyboard (focus the Isaac Sim window)
    I / K             walk forward / back
    J / L             strafe left / right
    U / O             turn left / right
    Up / Down         raise / lower both wrists
    Left / Right      wrists back / forward
    SPACE             toggle grasp (watch for "[follower] grasp CLOSED")
    R                 reset wrist offsets

  Either operator, from the PC
    P                 start / pause recording
    N                 reset the episode

  Robot B's ego view is the separate window ("Ego view - robot_b");
  --ego_view none turns it off, --ego_view both shows Robot A's too.
============================================================
"""

_CONTROLS_FEDERATED = """
============================================================
  Session ready -- {task} (Robot B federated from {peer})

  Both operators are in a headset, one per process:
    thumbstick        walk / turn
    trigger (>60%)    close that hand -- this is the grasp
    release trigger   let go{crouch}

  This process owns physics and the dataset. Robot B's slice arrives over
  UDP from scripts/teleop/xr_mirror.py; if that link goes quiet, Robot B holds
  its arms and stops walking, and recording continues.

  Session control -- start/pause and reset -- reaches this process from
  three places, and they all do the same thing:

    Quest A in-VR panel     directly
    Quest B in-VR panel     over the link (its start/pause asks for a state,
                            so pressing Start on an already-running session
                            does nothing rather than pausing it)
    this PC's keyboard      P start / pause, N reset  (focus the Isaac Sim
                            window; the only one that still works when both
                            headsets are unreachable)

  Robot B's own keys ({follower}) still move its stand-in pose, which is what
  it holds until the second headset connects.
============================================================
"""


WAIST_BANNER = {
    "all": "free -- yaw, pitch and roll all in the solve",
    "yaw": "yaw only -- the torso turns with the ego camera, but does not lean",
    "none": "locked upright -- steadiest, and the ego camera cannot turn",
}

UNREACHABLE_WARN_M = 0.06
"""Wrist tracking error that counts as the arm being out of reach [m]."""

RESET_ECHO_WINDOW_S = 1.0
"""Window after a reset in which a reset from the pipeline is its echo [s]."""

SESSION_SETTLE_S = 2.0
"""Window after operator 1's session comes up in which resets are ignored [s]."""

UNREACHABLE_REPORT_S = 2.0
"""Interval between out-of-reach reports while the condition lasts [s]."""


def wrist_tracking_error(env, term_name: str) -> tuple[float, float]:
    """(left, right) distance from the commanded wrist pose [m].

    Zero if the action term does not publish it.
    """
    try:
        error = env.action_manager.get_term(term_name).tracking_error
    except (AttributeError, KeyError, ValueError):
        return 0.0, 0.0
    return float(error[0, 0]), float(error[0, 1])


def _retarget_step_info(teleop):
    """The retargeter's step info for the frame just advanced, or ``None``.

    Carries ``returned_age_frames`` and ``frame_deadline_miss``.
    """
    lifecycle = getattr(teleop, "_session_lifecycle", None)
    session = getattr(lifecycle, "_session", None)
    return getattr(session, "last_step_info", None)


class _RateMeter:
    """Reports the achieved control rate of the recording loop."""

    def __init__(self, report_every: int):
        self.report_every = report_every
        self.count = 0
        self.steps: list[float] = []
        self.sim: list[float] = []
        self.physics: list[float] = []
        self.render: list[float] = []
        self.age: list[int] = []
        self.miss = 0
        self._last = time.perf_counter()
        self._cpu = CpuLoad()

    def tick(
        self,
        sim_s: float = 0.0,
        physics_s: float = 0.0,
        render_s: float = 0.0,
        age_frames: "int | None" = None,
        deadline_miss: bool = False,
    ) -> None:
        now = time.perf_counter()
        self.steps.append(now - self._last)
        self.sim.append(sim_s)
        self.physics.append(physics_s)
        self.render.append(render_s)
        if age_frames is not None:
            self.age.append(age_frames)
            if deadline_miss:
                self.miss += 1
        self._last = now
        self.count += 1
        if self.report_every and self.count % self.report_every == 0:
            steps = sorted(self.steps)
            n = len(steps)
            # The gate reads the median: a reset costs seconds and drags the mean.
            median_hz = 1.0 / steps[n // 2] if steps[n // 2] > 0 else 0.0
            mean_hz = n / sum(steps) if sum(steps) > 0 else 0.0
            verdict = "PASS" if median_hz >= 30 else ("SIMPLIFY" if median_hz >= 15 else "REDESIGN")
            resources = describe_resources(self._cpu)
            # env.step is split into PhysX, render and the Python between them.
            sim = sorted(self.sim)
            sim_median = sim[len(sim) // 2] if sim else 0.0
            physics = sorted(self.physics)
            physics_median = physics[len(physics) // 2] if physics else 0.0
            render = sorted(self.render)
            render_median = render[len(render) // 2] if render else 0.0
            other_median = max(sim_median - physics_median - render_median, 0.0)
            retarget = ""
            if self.age:
                ages = sorted(self.age)
                retarget = (
                    f"  [retarget age p50 {ages[len(ages) // 2]} frame(s),"
                    f" miss {100.0 * self.miss / len(self.age):.0f}%]"
                )
            print(
                f"[rate] median {median_hz:.1f} Hz, mean {mean_hz:.1f} Hz over {n} steps"
                f"  (p90 {steps[int(n * 0.9)] * 1e3:.0f} ms, worst {steps[-1] * 1e3:.0f} ms)"
                f"  gate: {verdict}"
                + (
                    f"  [env.step {sim_median * 1e3:.0f} ms of {steps[n // 2] * 1e3:.0f} ms:"
                    f" physx {physics_median * 1e3:.0f}, render {render_median * 1e3:.0f},"
                    f" other {other_median * 1e3:.0f};"
                    f" teleop {max(steps[n // 2] - sim_median, 0.0) * 1e3:.0f} ms]"
                    if sim_median > 0
                    else ""
                )
                + retarget
                + (f"  [{resources}]" if resources else "")
            )
            self.steps.clear()
            self.sim.clear()
            self.physics.clear()
            self.render.clear()
            self.age.clear()
            self.miss = 0

# Set by SIGINT/SIGTERM so the main loop can unwind through its context managers.
_shutdown_requested = False

# The dataset handler, for the exit paths that skip env.close().
_dataset_handler = None


def _flush_dataset(timeout: float = 120.0) -> None:
    """Write queued demonstrations to disk before this process exits.

    ``os._exit`` skips ``env.close()`` and with it the shard writer's drain.
    Called from both hard-exit paths. Never raises.
    """
    handler = _dataset_handler
    if handler is None:
        return
    try:
        pending = getattr(getattr(handler, "_write_queue", None), "qsize", lambda: 0)()
        if pending:
            print(f"\n[save] {pending} demonstration(s) still queued; writing them out...")
        # Only the sharded handler's close takes a timeout.
        try:
            handler.close(timeout=timeout)
        except TypeError:
            handler.close()
        written = getattr(handler, "written", None)
        if written:
            print(f"[save] {len(written)} shard(s) are on disk")
    except Exception as exc:  # noqa: BLE001
        print(f"[save] could not flush the dataset: {exc!r}")


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        global _shutdown_requested
        if _shutdown_requested:
            # Second Ctrl+C: flush the dataset, clean up CloudXR and leave.
            print("\nForced shutdown - cleaning up CloudXR runtime...")
            _flush_dataset()
            release_cloudxr()
            os._exit(1)
        _shutdown_requested = True
        print("\nShutdown requested; finishing current step (Ctrl+C again to force)...")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)


from cohub.teleop.cloudxr import resolve_cloudxr_env  # noqa: E402


def _make_follower():
    if args_cli.follower == "keyboard":
        try:
            from cohub.teleop.follower import KeyboardFollower

            return KeyboardFollower()
        except Exception as exc:  # noqa: BLE001
            # A headless run has no app window keyboard.
            print(f"[follower] no keyboard available ({exc}); using the constant pose instead")
            return ConstantFollower()
    return ConstantFollower()


def _load_episode_meta(has_robot_c: bool = False) -> None:
    """Fill SYNC_STATE.episode_meta from the operators YAML."""
    meta = SYNC_STATE.episode_meta
    meta["operator_device_b"] = (
        _FOLLOWER_DEVICE_CODES["quest"] if _FEDERATED else _FOLLOWER_DEVICE_CODES[args_cli.follower]
    )
    if has_robot_c:
        # Written only on a task that has a third robot.
        meta["operator_id_c"] = 0.0
        meta["operator_device_c"] = float(
            _FOLLOWER_DEVICE_CODES["quest"] if _FED_C else _FOLLOWER_DEVICE_CODES["constant"]
        )
    if os.path.exists(args_cli.operators):
        with open(args_cli.operators) as f:
            data = yaml.safe_load(f) or {}
        for key in (
            "partner_id",
            "operator_id_a",
            "operator_id_b",
            "operator_id_c",
            "terrain_id",
            "grasp_abstraction",
            "role_a_leader",
        ):
            if key in data and key in meta:
                meta[key] = float(data[key])


def _copy_operators_config(dataset_root: str) -> None:
    """Copy the operators YAML into ``<dataset root>/configs/``.

    ``meta/*`` records integer codes and the YAML says what they mean.
    """
    if not os.path.exists(args_cli.operators):
        return
    target = os.path.join(dataset_root, "configs", os.path.basename(args_cli.operators))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(args_cli.operators, "rb") as source:
        content = source.read()
    if os.path.exists(target):
        with open(target, "rb") as existing:
            if existing.read() == content:
                return
        print(f"[dataset] {target} differs from {args_cli.operators}; overwriting with this session's")
    with open(target, "wb") as out:
        out.write(content)


def calibrate_upper_ik(env_cfg) -> None:
    """Apply the arm IK flags to the robots' IK terms and print the result."""
    import math

    from cohub.g1.actions import IK_TIME_CONSTANT, normalised_task_gain, upper_ik_action_cfg

    tau = args_cli.ik_time_constant or IK_TIME_CONSTANT
    for name in ("upper_body_a", "upper_body_b"):
        term = getattr(env_cfg.actions, name, None)
        if term is None:
            continue
        term.ik_time_constant = tau
        term.constant_time_constant = args_cli.constant_time_constant
        term.elbow_clearance = args_cli.elbow_clearance
        term.elbow_clearance_cost = args_cli.elbow_clearance_cost
        if args_cli.waist != "all":
            # Rebuilt by the factory so the joint lists stay consistent.
            setattr(env_cfg.actions, name, upper_ik_action_cfg(term.asset_name, waist=args_cli.waist))
            rebuilt = getattr(env_cfg.actions, name)
            rebuilt.ik_time_constant = tau
            rebuilt.constant_time_constant = args_cli.constant_time_constant
            rebuilt.elbow_clearance = args_cli.elbow_clearance
            rebuilt.elbow_clearance_cost = args_cli.elbow_clearance_cost

    solves = int(getattr(env_cfg, "decimation", 4))
    if args_cli.constant_time_constant:
        gain = f"{normalised_task_gain(env_cfg.sim.dt * solves, tau, solves):.4f}"
        source = f"derived, {1000 * tau:.0f} ms time constant over {solves} solves/step"
    else:
        gain = "0.0750"
        implied = -env_cfg.sim.dt * solves / math.log(0.925**solves)
        source = f"upstream's fixed value, which at {solves} solves/step means {1000 * implied:.0f} ms"
    print(f"[ik] arm gain {gain} -- {source}")
    print(f"[ik] waist {WAIST_BANNER[args_cli.waist]}")
    if not args_cli.elbow_clearance:
        print("[ik] elbow clearance OFF -- a supinated reach may fold the elbow into the chest")
    else:
        print(f"[ik] elbow clearance {1000 * args_cli.elbow_clearance:.0f} mm from the torso midplane, cost {args_cli.elbow_clearance_cost:g}")
    print(
        f"[ik] out-of-reach reporting {'on' if args_cli.reach_feedback else 'OFF'}"
        f" -- says so past {1000 * UNREACHABLE_WARN_M:.0f} mm of wrist error"
    )


def create_environment_config(output_dir: str, output_file_name: str):
    if configure_cpu_runtime():
        print("[runtime] CUDA unavailable: using ordinary CPU memory for Warp staging buffers")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task

    # 0.0 means "not given". None lets the helpers read COHUB_EYE_HEIGHT.
    eye_height = args_cli.eye_height or None
    calibrate_xr_view(env_cfg, eye_height)
    print(describe_xr_view(eye_height, configured_pelvis_spawn_z(env_cfg)))
    if args_cli.crouch:
        print(
            f"[crouch] left X/Y move the hips at {args_cli.crouch_rate:.2f} m/s, "
            f"down to {args_cli.crouch_min:.2f} m (standing is 0.72 m)\n"
            f"         viewpoint {'follows' if args_cli.anchor_follows_crouch else 'stays put'}"
            " -- --no-anchor_follows_crouch to compare\n"
            "         walking is locked out while crouched; stand up to move"
        )

    calibrate_upper_ik(env_cfg)

    if args_cli.retargeting == "sync" and env_cfg.isaac_teleop is not None:
        from isaacteleop.teleop_session_manager import RetargetingExecutionConfig

        env_cfg.isaac_teleop.retargeting_execution = RetargetingExecutionConfig(mode="sync")
        print("[teleop] retargeting sync: poses computed in-frame instead of one frame late")

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    if not args_cli.enable_cameras:
        env_cfg = remove_camera_configs(env_cfg)
    configure_render(env_cfg.sim.render)

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    if not args_cli.save:
        # A practice session attaches no recorder terms, so nothing is buffered.
        env_cfg.recorders = None
        return env_cfg, success_term

    entry = gym.spec(args_cli.task).kwargs.get("demo_recorder_cfg_entry_point")
    if entry is None:
        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

        env_cfg.recorders = ActionStateRecorderManagerCfg()
    else:
        env_cfg.recorders = load_cfg_from_registry(args_cli.task, "demo_recorder_cfg_entry_point")
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = (
        DatasetExportMode.EXPORT_ALL if args_cli.export_mode == "all" else DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    )
    if args_cli.shard_demos:
        # One file per demo. A shard directory always appends.
        env_cfg.recorders.dataset_file_handler_class_type = ShardedHDF5DatasetFileHandler
        # Class attribute: RecorderManager constructs the handler with no arguments.
        ShardedHDF5DatasetFileHandler.async_write = bool(args_cli.async_export)
        if not args_cli.resume_dataset:
            print("[dataset] --no-resume_dataset has no effect while sharding; a shard directory always appends")
    elif args_cli.resume_dataset:
        # Isaac Lab truncates the dataset on startup. This handler appends.
        env_cfg.recorders.dataset_file_handler_class_type = ResumableHDF5DatasetFileHandler

    return env_cfg, success_term


def _exported_demo_index(env) -> int:
    """How many demos the dataset holds, including those from earlier sessions."""
    handler = getattr(env.recorder_manager, "_dataset_file_handler", None)
    count = getattr(handler, "demo_count", None)
    return int(count) if count is not None else 0


def auto_termination_reason(env) -> str | None:
    """The failure terms that just ended the episode, comma-separated, or ``None``.

    ``success`` is never among them: this script polls it itself.
    """
    if not bool(env.reset_buf[0]):
        return None
    fired = sorted(
        key.split("/", 1)[1]
        for key, value in (env.extras.get("log") or {}).items()
        if key.startswith("Episode_Termination/") and float(value) > 0.0
    )
    return ", ".join(fired) if fired else "a task limit"


def process_success_condition(
    env, success_evaluator, success_step_count: int, subtask_tracker=None, step: int = 0
) -> tuple[int, bool, tuple[str, ...]]:
    """Poll the success term, and export the episode when it holds.

    Returns ``(streak, ended, missing)``. ``missing`` names the subtask
    milestones the episode never reached. An episode that reaches the goal
    with milestones missing is not saved as a success.
    """
    if success_evaluator is None:
        return success_step_count, False, ()
    if bool(success_evaluator(env)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            missing: tuple[str, ...] = ()
            if subtask_tracker is not None:
                reached, _ = subtask_tracker.result(True, step)
                missing = tuple(name for name in subtask_tracker.names if not reached[name])
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            if not missing:
                env.recorder_manager.set_success_to_episodes(
                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                )
                env.recorder_manager.export_episodes([0])
                print("Success condition met! Recording completed.")
            elif args_cli.export_mode == "all":
                # Export mode "all" keeps it, saved as a failure.
                env.recorder_manager.export_episodes([0])
                print(f"Goal reached, but {', '.join(missing)} never was. Saved as a failure.")
            else:
                print(f"Goal reached, but {', '.join(missing)} never was. Not saved.")
            return success_step_count, True, missing
    else:
        success_step_count = 0
    return success_step_count, False, ()


def setup_ego_views(robots: list[str]) -> list:
    """Open an ego-view window per requested robot the task has."""
    if args_cli.ego_view == "none":
        return []
    try:
        width, height = (int(v) for v in args_cli.ego_view_size.lower().split("x"))
    except ValueError:
        print(f"[ego view] bad --ego_view_size {args_cli.ego_view_size!r}; using 640x480")
        width, height = 640, 480

    if args_cli.ego_view == "both":
        wanted = ("robot_a", "robot_b")
    elif args_cli.ego_view == "all":
        wanted = tuple(ROBOTS3)
    else:
        wanted = (args_cli.ego_view,)
    prim_names = {name: f"Robot_{name.rsplit('_', 1)[1].upper()}" for name in ROBOTS3}
    skipped = [name for name in wanted if name not in robots]
    if skipped:
        print(f"[ego view] {args_cli.task} has no {', '.join(skipped)}")
    wanted = tuple(name for name in wanted if name in robots)
    windows = []
    for index, robot in enumerate(wanted):
        window = open_ego_view(
            robot_prim_path=f"/World/envs/env_0/{prim_names[robot]}",
            title=f"Ego view - {robot}",
            width=width,
            height=height,
            position=(60 + index * (width + 20), 60),
            # None keeps the recorded sensor's focal length.
            **({} if args_cli.ego_view_focal is None else {"focal_length": args_cli.ego_view_focal}),
        )
        if window is not None:
            windows.append(window)
    return windows


def main() -> None:
    _install_signal_handlers()

    # A runtime orphaned by a previous hard exit would block this one.
    killed = release_cloudxr()
    if killed:
        print(f"Cleaned up orphaned CloudXR runtime(s): {killed}")

    if args_cli.xr and args_cli.link_port:
        urls = start_link_server(
            args_cli.link_port, args_cli.stream_bitrate, args_cli.stream_per_eye, args_cli.stream_fps
        )
        print("\n" + "=" * 60)
        print("  On the Quest browser, open:")
        for url in urls:
            print(f"      {url}")
        print("  (tap 'Accept certificate', then 'Open teleop client')")
        print("=" * 60 + "\n")

    # --dataset_file is a .hdf5 path, whose stem is the shard prefix, or the
    # dataset root directory.
    if args_cli.dataset_file.endswith(".hdf5"):
        dataset_root = os.path.dirname(args_cli.dataset_file) or "."
        output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    else:
        dataset_root = args_cli.dataset_file.rstrip("/") or "."
        output_file_name = os.path.basename(dataset_root)
    output_dir = os.path.join(dataset_root, "data") if args_cli.shard_demos else dataset_root
    # A practice session creates only the dataset root, for episodes.txt.
    os.makedirs(dataset_root if not args_cli.save else output_dir, exist_ok=True)

    _copy_operators_config(dataset_root)
    episode_log = EpisodeLog(os.path.join(dataset_root, "episodes.txt"), args_cli.task)

    env_cfg, success_term = create_environment_config(output_dir, output_file_name)

    try:
        from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

        XRVisualization.assign_manager(TeleopVisualizationManager)
    except ModuleNotFoundError:
        # A Kit profile without omni.ui has no XR widgets to draw.
        pass

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    success_evaluator = DetachedManagerTerm(env, success_term) if success_term is not None else None
    # The subtask chain the eval client scores. None or an empty chain means
    # no extra gate.
    subtask_tracker = make_tracker(args_cli.task)
    if subtask_tracker is not None and not subtask_tracker.names:
        subtask_tracker = None
    if subtask_tracker is not None:
        print(f"[subtasks] {args_cli.task}: {' -> '.join(subtask_tracker.names)}")
    # The grasp model is stamped from the environment. 0 is physical contact.
    if "grasp" in env.action_manager.active_terms:
        SYNC_STATE.episode_meta["grasp_abstraction"] = 0.0
        print("[grasp] holding model: physical contact")
    # Provenance is stamped before the first episode. hidden_heads is empty
    # when this process has no --xr operator.
    hidden_heads = ["Robot_A"] if (args_cli.xr and args_cli.hide_own_head) else []
    global _dataset_handler
    _dataset_handler = getattr(env.recorder_manager, "_dataset_file_handler", None)
    stamp_env_args(
        getattr(env.recorder_manager, "_dataset_file_handler", None),
        env,
        "record_demos.py",
        hidden_heads,
    )
    # The action vector is 32D per robot, in this order.
    robot_names = [name for name in ROBOTS3 if name in env.scene.articulations]
    has_robot_c = "robot_c" in robot_names
    if not has_robot_c and args_cli.robot_c_source != parser.get_default("robot_c_source"):
        print(f"[task] {args_cli.task} has no robot_c: --robot_c_source is ignored")
    _load_episode_meta(has_robot_c)
    lock_locomotion = bool(getattr(env_cfg, "lock_locomotion", False))
    if lock_locomotion:
        print(f"[task] locomotion locked: all {len(robot_names)} robots stay put")
    ego_windows = setup_ego_views(robot_names)  # noqa: F841

    for robot in hidden_heads:
        hide_head(f"/World/envs/env_0/{robot}")

    from isaaclab_teleop import create_isaac_teleop_device, poll_control_events

    episode_log.open_session(
        env,
        success_term,
        dataset=args_cli.dataset_file,
        export_mode=(
            "nothing (--no-save practice session)"
            if not args_cli.save
            else "successful episodes only" if args_cli.export_mode != "all" else "every episode"
        ),
        num_success_steps=args_cli.num_success_steps,
    )
    print(f"[log] every episode outcome is appended to {episode_log.path}")
    if not args_cli.save:
        print("[dataset] NOT SAVING (--no-save): practice session, no demos will be written")

    recorded = 0
    success_step_count = 0
    should_reset = False
    running = False
    # Whether this episode's ending has already been reported.
    outcome_claimed = False

    def request_reset():
        nonlocal should_reset
        should_reset = True

    def set_running(value: bool):
        nonlocal running
        running = value

    callbacks = {
        "R": request_reset,
        "RESET": request_reset,
        "START": lambda: set_running(True),
        "STOP": lambda: set_running(False),
    }

    if args_cli.auto_launch_cloudxr and args_cli.wait_for_port > 0:
        # Another runtime may hold the shared startup port. Wait for it.
        if not wait_for_startup_port(args_cli.wait_for_port, should_stop=lambda: _shutdown_requested):
            return

    teleop = create_isaac_teleop_device(
        env_cfg.isaac_teleop,
        sim_device=args_cli.device,
        callbacks=callbacks,
        cloudxr_env_file=resolve_cloudxr_env(args_cli.cloudxr_env),
        auto_launch_cloudxr=args_cli.auto_launch_cloudxr,
    )
    xr_guard = guard_xr_session(teleop)
    # The follower drives Robot B unless it is federated. Its keys are the
    # PC-side session control either way.
    follower = _make_follower()
    link = (
        PrimaryLink(
            peer_host=args_cli.fed_peer,
            state_port=args_cli.fed_state_port,
            action_port=args_cli.fed_action_port,
        )
        if _FEDERATED
        else None
    )
    # Operator 3's mirror, on a task with a robot_c.
    link_c = (
        PrimaryLink(
            peer_host=args_cli.fed_c_peer,
            state_port=args_cli.fed_c_state_port,
            action_port=args_cli.fed_c_action_port,
        )
        if (has_robot_c and _FED_C)
        else None
    )
    # Operator 1's mirror, when Robot A is federated. Each link has its own
    # port pair.
    link_a = (
        PrimaryLink(
            peer_host=args_cli.fed_a_peer,
            state_port=args_cli.fed_a_state_port,
            action_port=args_cli.fed_a_action_port,
        )
        if _FED_A
        else None
    )
    hud = SessionHud(enabled=args_cli.vr_status and args_cli.xr)
    if hud.enabled:
        print("[hud] demo outcomes will show in the headset")
    # The ego frame is scene geometry that other cameras can see, so by
    # default it is drawn only when this process records no images.
    records_images = bool(args_cli.enable_cameras)
    if args_cli.ego_frame is None:
        frame_wanted = args_cli.xr and not records_images
        if args_cli.xr and records_images:
            print(
                "[ego frame] off: this process records images, and the frame is geometry\n"
                "            the partner's ego camera and the scene camera can see.\n"
                "            --ego_frame forces it (render_dataset_images.py afterwards), or\n"
                "            --robot_a_source federated + xr_mirror.py --robot robot_a\n"
                "            gives you the frame in a mirror that records nothing."
            )
    else:
        frame_wanted = args_cli.ego_frame and args_cli.xr
    ego_frame = EgoFrame(
        "/World/envs/env_0/Robot_A",
        enabled=frame_wanted,
        distance=args_cli.ego_frame_distance,
        colour=args_cli.ego_frame_color,
    )
    if ego_frame.show():
        print(describe_ego_frame(
            "robot_a", args_cli.ego_frame_distance, args_cli.ego_frame_color,
            recording=bool(args_cli.enable_cameras),
        ))

    unreachable_since: float | None = None
    unreachable_reported = 0.0

    def report_unreachable(miss: tuple[float, float], now: float) -> None:
        """Report on the terminal when the arms stop following the hands."""
        nonlocal unreachable_since, unreachable_reported
        worst = max(miss)
        if worst < UNREACHABLE_WARN_M:
            if unreachable_since is not None:
                print(f"[reach] arms following again after {now - unreachable_since:.1f} s")
            unreachable_since = None
            return
        if unreachable_since is None:
            unreachable_since = now
            unreachable_reported = 0.0
        if now - unreachable_reported >= UNREACHABLE_REPORT_S:
            unreachable_reported = now
            print(
                f"[reach] out of arm: left {1000 * miss[0]:.0f} mm, right {1000 * miss[1]:.0f} mm"
                f" behind the controllers"
            )

    def announce(outcome: str, message: str, colour: int) -> None:
        """Report how a demonstration ended, to the headsets and the terminal."""
        nonlocal outcome_claimed
        outcome_claimed = True
        hud.show(message, colour)
        for peer in (link, link_a, link_c):
            if peer is not None:
                peer.announce_outcome(outcome)
        print(f"[session] {message}")
    buttons = SessionButtons() if (args_cli.xr and args_cli.controller_session_buttons) else None
    if buttons is not None:
        buttons.bind(teleop)
        if release_upstream_a_binding(teleop):
            print("[buttons] A is start/pause here; upstream's anchor-rotation toggle is unbound")
    # Session control from every source is OR'd.
    control_sources = [source for source in (link, link_a, link_c, follower, buttons) if source is not None]

    with contextlib.ExitStack() as stack:
        stack.enter_context(teleop)
        for _peer in (link, link_a, link_c):
            if _peer is not None:
                stack.enter_context(_peer)
        env.sim.reset()
        env.reset()
        if success_evaluator is not None:
            success_evaluator.reset()
        teleop.reset()
        follower.reset()
        crouch_help = _CROUCH_CONTROLS if args_cli.crouch else ""
        if _FEDERATED:
            print(
                _CONTROLS_FEDERATED.format(
                    task=args_cli.task,
                    peer=args_cli.fed_peer,
                    follower=args_cli.follower,
                    crouch=crouch_help,
                )
            )
        else:
            print(_CONTROLS.format(task=args_cli.task, follower=args_cli.follower, crouch=crouch_help))
        if _FED_A:
            print(
                f"[federation] Robot A is remote: operator 1 connects through\n"
                f"    python scripts/teleop/xr_mirror.py --xr --robot robot_a"
                f" --state_port {args_cli.fed_a_state_port} --action_port {args_cli.fed_a_action_port}"
                f"  (peer {args_cli.fed_a_peer})\n"
                f"  This process has no headset operator, so no head is hidden and"
                f" every camera records the robots whole."
            )
        if link_c is not None:
            print(
                f"[federation] Robot C is remote: operator 3 connects through\n"
                f"    python scripts/teleop/xr_mirror.py --xr --robot robot_c"
                f" --state_port {args_cli.fed_c_state_port} --action_port {args_cli.fed_c_action_port}"
                f"  (peer {args_cli.fed_c_peer})"
            )
        elif has_robot_c:
            print("[federation] Robot C holds a constant pose (--robot_c_source follower)")
        print(
            f"[loco] full stick walks at {args_cli.max_speed:.2f} m/s (--max_speed)"
        )
        # The briefing is printed verbatim, never .format()ed.
        briefing = getattr(env_cfg, "operator_briefing", "")
        if briefing:
            print(briefing)

        # Steps recorded into the episode in progress.
        recorded_steps = 0
        # How this episode ended. Written by the reset block below.
        pending_outcome = None
        last_action_a = None
        teleop_online = False
        # Operator 1's panel state in the previous frame. None until reported.
        panel_active = None
        secondary_online = False
        episode_id = 0

        class _MirrorRobot:
            """One robot driven over a federation link, or by a local hold pose."""

            def __init__(self, name: str, operator: str, peer, stamp: str, raw: str):
                self.name = name
                self.operator = operator
                self.link = peer
                self.stamp = stamp  # SYNC_STATE field for the sample time
                self.raw = raw  # SYNC_STATE field for the raw input
                self.hand_sensor = f"hand_contact_{name.rsplit('_', 1)[1]}"
                self.index = robot_names.index(name)
                self.online = False
                self.last_action = None
                self.follower = follower if name == "robot_b" else ConstantFollower()

            def poll(self) -> bool:
                """Drain this link and keep the newest slice. True if one arrived."""
                if self.link is None:
                    return False
                packet = self.link.poll_action()
                if packet is not None:
                    self.last_action = torch.as_tensor(
                        packet.action, dtype=torch.float32, device=env.device
                    )
                    shape_locomotion(self.last_action)
                    # Both processes share this PC's clock.
                    setattr(SYNC_STATE, self.stamp, packet.send_time)
                    setattr(SYNC_STATE, self.raw, packet.operator_raw.tolist())
                if self.link.secondary_online != self.online:
                    self.online = self.link.secondary_online
                    label = self.name.rsplit("_", 1)[1].upper()
                    print(
                        f"[federation] Robot {label} online"
                        if self.online
                        else f"[federation] Robot {label} OFFLINE - holding its pose; the others are still controllable"
                    )
                return packet is not None

            def slice(self, dtype, device):
                """This robot's 32D command for this step."""
                if self.link is not None and self.last_action is not None:
                    action = self.last_action.to(dtype=dtype)
                    if not self.link.secondary_online:
                        # Link down: keep the arm targets, stop the walking.
                        action = action.clone()
                        action[28:31] = 0.0
                    return action, True
                # The hold pose is pelvis-relative and the action is in the
                # env frame, so it needs this robot's current pelvis.
                data = env.scene[self.name].data
                values = self.follower.advance(
                    data.root_pos_w.torch[0].cpu().numpy(),
                    data.root_quat_w.torch[0].cpu().numpy(),
                )
                SYNC_STATE.follower_sample_time = time.time()
                return torch.as_tensor(values, dtype=dtype, device=device), False

        mirror_robots = [
            _MirrorRobot("robot_b", "2", link, "secondary_sample_time", "operator_raw_b"),
        ]
        if has_robot_c:
            mirror_robots.append(
                _MirrorRobot("robot_c", "3", link_c, "tertiary_sample_time", "operator_raw_c")
            )
        # See RESET_ECHO_WINDOW_S and SESSION_SETTLE_S.
        reset_echo_until = 0.0
        reset_settle_until = time.monotonic() + SESSION_SETTLE_S
        rate = _RateMeter(report_every=args_cli.rate_report)
        step_split = StepSplit(env.sim)
        # Local controller haptics, only when operator 1 is local.
        haptics = ControllerHaptics() if (args_cli.haptics and not _FED_A) else None
        operator_capture = OperatorInputCapture()
        crouch = (
            CrouchControl(rate=args_cli.crouch_rate, minimum=args_cli.crouch_min)
            if args_cli.crouch
            else None
        )
        # Standing reference for the camera-mount rise. Re-learned slowly
        # while the robot stands, held while crouched.
        mount_rise_standing = mount_rise_over_pelvis(env.scene["robot_a"])
        if args_cli.eye_height and args_cli.eye_height > 0.0:
            # Account for the standing torso pose once PhysX has initialized.
            set_anchor_dz(env_cfg, mount_rise_standing - EGO_HEIGHT_ABOVE_PELVIS)
        MOUNT_REF_ALPHA = 0.02

        # One lean compensator per robot.
        lean_comp_a = None
        lean_comp: dict[str, LeanCompensator] = {}
        if args_cli.lean_comp > 0.0:
            lean_comp = {name: LeanCompensator(rot_time_constant=args_cli.lean_comp) for name in robot_names}
            lean_comp_a = lean_comp["robot_a"]
            print(
                f"[lean] wrist targets ride the torso's fast lean (tau {args_cli.lean_comp:g} s)"
                " -- headset-sourced slices only"
            )

        def _shape_slice(action_t, robot: str, lean):
            """One robot's slice, re-attached to its own measured pelvis."""
            data = env.scene[robot].data
            arr = lean.apply(
                action_t.cpu().numpy(),
                data.root_pos_w.torch[0].cpu().numpy(),
                data.root_quat_w.torch[0].cpu().numpy(),
                float(env.step_dt),
            )
            return torch.as_tensor(arr, dtype=action_t.dtype, device=action_t.device)

        def settle_after_episode() -> None:
            """Everything an episode boundary needs beyond resetting the env.

            Shared by every way an episode can end.
            """
            nonlocal recorded_steps, outcome_claimed, running, episode_id, success_step_count
            nonlocal reset_echo_until
            # Set before teleop.reset(): its reset echoes on the next advance().
            reset_echo_until = time.monotonic() + RESET_ECHO_WINDOW_S
            if success_evaluator is not None:
                success_evaluator.reset([0])
            if subtask_tracker is not None:
                subtask_tracker.reset()
            teleop.reset()
            follower.reset()
            recorded_steps = 0
            outcome_claimed = False
            # A reset pauses. Press A or P to begin the next demo.
            if running:
                running = False
                print("[session] paused - press A on the controller (or P) to start the next demo")
            for compensator in lean_comp.values():
                compensator.reset()
            if crouch is not None:
                # The robot respawns standing.
                crouch.reset()
                if args_cli.anchor_follows_crouch:
                    set_anchor_dz(env_cfg, 0.0)
                if last_action_a is not None:
                    last_action_a[31] = crouch.height
            # The mirrors watch this counter to reset on the same boundary.
            episode_id += 1
            success_step_count = 0

        def publish_state() -> None:
            """Send the scene state to every mirror.

            Called on every path through the loop, including paused frames.
            """
            peers = [(link_a, "robot_a", "hand_contact_a")]
            peers += [(mirror.link, mirror.name, mirror.hand_sensor) for mirror in mirror_robots]
            for peer, robot, sensor in peers:
                if peer is None:
                    continue
                # Haptic loads are measured here and sent per link.
                haptic = None
                if args_cli.haptics:
                    left, right = object_interaction_forces(env, robot, sensor)
                    haptic = (left, right)
                scene_state.publish(
                    peer,
                    env.scene,
                    episode_id=episode_id,
                    running=running,
                    haptic=haptic,
                    # Lets a mirror refuse a session on another task.
                    task=args_cli.task,
                )

        # Real-time pacing with a deadline accumulator. A loop that falls
        # behind re-bases instead of catching up.
        _pace_period = 1.0 / args_cli.max_fps if args_cli.max_fps > 0 else 0.0
        _pace_next = time.perf_counter()
        if _pace_period:
            print(
                f"[rate] capped at {args_cli.max_fps:.0f} Hz"
                f" ({args_cli.max_fps * float(env.step_dt):.2f}x real time) -- --max_fps 0 to unlock"
            )

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running() and not _shutdown_requested:
                if _pace_period:
                    _pace_now = time.perf_counter()
                    if _pace_now < _pace_next:
                        time.sleep(_pace_next - _pace_now)
                        _pace_next += _pace_period
                    else:
                        _pace_next = _pace_now + _pace_period
                live_action_a = teleop.advance()
                now = time.time()

                # Polled every iteration to keep the sockets drained.
                for mirror in mirror_robots:
                    if mirror.poll() and mirror.link is link and link is not None:
                        # Robot B's link is the recorded round trip.
                        SYNC_STATE.link_rtt = link.rtt
                secondary_online = mirror_robots[0].online

                if _FED_A:
                    # Operator 1's slice, from their mirror.
                    packet = link_a.poll_action()
                    if packet is not None:
                        last_action_a = (
                            torch.as_tensor(packet.action, dtype=torch.float32, device=env.device)
                            .flatten()
                            .clone()
                        )
                        shape_locomotion(last_action_a)
                        SYNC_STATE.teleop_sample_time = packet.send_time
                        SYNC_STATE.operator_raw_a = packet.operator_raw.tolist()
                        if link is None:
                            SYNC_STATE.link_rtt = link_a.rtt
                    if link_a.secondary_online != teleop_online:
                        teleop_online = link_a.secondary_online
                        print(
                            "[federation] Robot A online"
                            if teleop_online
                            else "[federation] Robot A OFFLINE - holding its pose; Robot B still controllable"
                        )

                ctrl = poll_control_events(teleop)
                # Act on changes of the panel state only, so other sources
                # can set ``running`` too. The first reported state is
                # recorded without acting on it.
                if ctrl.is_active is not None and panel_active is not None and ctrl.is_active != panel_active:
                    running = ctrl.is_active
                    print(f"[panel] recording {'started' if running else 'paused'}")
                if ctrl.is_active is not None:
                    panel_active = ctrl.is_active
                if ctrl.should_reset:
                    if time.monotonic() < reset_settle_until:
                        pass  # the session coming up
                    elif time.monotonic() < reset_echo_until:
                        # The echo of this process's own teleop.reset().
                        pass
                    else:
                        should_reset = True

                # Every source is drained every frame.
                toggle_run = False
                reset_req = False
                for source in control_sources:
                    source_toggle, source_reset = source.consume_control_flags()
                    toggle_run = toggle_run or source_toggle
                    reset_req = reset_req or source_reset
                if toggle_run:
                    running = not running
                    print(f"[PC] recording {'started' if running else 'paused'}")
                if reset_req:
                    should_reset = True
                    print("[PC] reset requested")

                # The local operator-1 path. Skipped when Robot A is federated.
                if _FED_A:
                    pass
                elif live_action_a is not None:
                    if not teleop_online:
                        print("[teleop] online")
                        teleop_online = True
                        # A session coming up emits a reset of its own.
                        reset_settle_until = time.monotonic() + SESSION_SETTLE_S
                    last_action_a = live_action_a.flatten().clone()
                    SYNC_STATE.operator_raw = operator_capture.capture(teleop)
                    shape_locomotion(last_action_a)
                    if crouch is not None:
                        # Index 31 is the hip height in metres. The crouch
                        # buttons own it.
                        last_action_a[31] = crouch.update(teleop, now)
                        if args_cli.anchor_follows_crouch:
                            # Track the achieved camera mount, not the
                            # commanded pelvis.
                            rise = mount_rise_over_pelvis(env.scene["robot_a"])
                            if crouch.anchor_dz == 0.0:
                                mount_rise_standing += MOUNT_REF_ALPHA * (rise - mount_rise_standing)
                            set_anchor_dz(env_cfg, crouch.anchor_dz + rise - mount_rise_standing)
                    SYNC_STATE.teleop_sample_time = now
                elif teleop_online:
                    print("[teleop] OFFLINE - holding Robot A pose; Robot B still controllable")
                    teleop_online = False

                if last_action_a is None:
                    # No teleop action yet. Render and publish state only.
                    env.sim.render()
                    publish_state()
                    continue

                action_a = last_action_a
                if not teleop_online:
                    # Link down: stop walking, keep arm targets and hip height.
                    action_a = action_a.clone()
                    action_a[28:31] = 0.0

                slices = [action_a]
                for mirror in mirror_robots:
                    mirror_action, from_headset = mirror.slice(action_a.dtype, action_a.device)
                    if lean_comp_a is not None and from_headset:
                        # Only headset slices: the follower already anchors
                        # to the full pelvis rotation.
                        mirror_action = _shape_slice(mirror_action, mirror.name, lean_comp[mirror.name])
                    slices.append(mirror_action)
                if lean_comp_a is not None:
                    # Re-applied every step with the current pelvis.
                    slices[0] = _shape_slice(action_a, "robot_a", lean_comp_a)
                action = torch.cat(slices).unsqueeze(0)
                if lock_locomotion:
                    # Zero every robot's velocity commands, keep hip height.
                    for _i in range(len(slices)):
                        action[0, _i * ROBOT_SLICE + 28 : _i * ROBOT_SLICE + 31] = 0.0
                SYNC_STATE.step_wall_time = time.time()

                auto_ended = None
                if running:
                    _sim_started = time.perf_counter()
                    _split_before = step_split.snapshot()
                    env.step(action.repeat(env.num_envs, 1))
                    _sim_elapsed = time.perf_counter() - _sim_started
                    _physics_s, _render_s = step_split.since(_split_before)
                    recorded_steps += 1
                    # Must be read before anything else moves reset_buf.
                    auto_ended = auto_termination_reason(env)
                    if subtask_tracker is not None:
                        subtask_tracker.update(
                            env, step=recorded_steps, terminal=auto_ended is not None
                        )
                    _retarget = _retarget_step_info(teleop)
                    rate.tick(
                        _sim_elapsed,
                        _physics_s,
                        _render_s,
                        getattr(_retarget, "returned_age_frames", None),
                        bool(getattr(_retarget, "frame_deadline_miss", False)),
                    )
                    reach_miss = (
                        wrist_tracking_error(env, "upper_body_a")
                        if args_cli.reach_feedback
                        else (0.0, 0.0)
                    )
                    report_unreachable(reach_miss, time.time())
                    if haptics is not None and haptics.bind(teleop):
                        haptics.send(*object_interaction_forces(env, "robot_a", "hand_contact_a"))
                else:
                    env.sim.render()

                publish_state()

                if auto_ended is not None:
                    # The env already reset itself inside the step.
                    kept = args_cli.save and args_cli.export_mode == "all"
                    announce(
                        "discarded",
                        f"ENDED   {auto_ended} - {'saved as a failure' if kept else 'not saved'}",
                        RED,
                    )
                    # Before settle_after_episode(), which zeroes the step count.
                    episode_log.record(
                        env,
                        episode_id=episode_id,
                        outcome="RESET",
                        note=f"{auto_ended} - {'saved as a failure' if kept else 'not saved'}",
                        steps=recorded_steps,
                        fired=[t for t in auto_ended.split(", ") if t in env.termination_manager.active_terms],
                        success_term=success_term,
                    )
                    pending_outcome = None
                    settle_after_episode()
                    should_reset = False
                elif running:
                    # Success is polled only for executed control steps.
                    success_step_count, success_reset, missing = process_success_condition(
                        env, success_evaluator, success_step_count,
                        subtask_tracker, recorded_steps,
                    )
                    if success_reset and missing:
                        should_reset = True
                        kept = args_cli.save and args_cli.export_mode == "all"
                        note = f"{', '.join(missing)} not reached"
                        pending_outcome = (
                            "REJECTED",
                            f"{note} - {'saved as a failure' if kept else 'not saved'}",
                        )
                        announce("discarded", f"REJECTED   {note}", RED)
                    elif success_reset:
                        should_reset = True
                        if args_cli.save:
                            demo = f"demo_{max(_exported_demo_index(env) - 1, 0)}"
                            pending_outcome = ("SUCCESS", f"{demo} saved")
                            announce("success", f"SUCCESS   {demo} saved", GREEN)
                        else:
                            pending_outcome = ("SUCCESS", "not saved (--no-save)")
                            announce("success", "SUCCESS   not saved (--no-save)", GREEN)

                if env.recorder_manager.exported_successful_episode_count > recorded:
                    recorded = env.recorder_manager.exported_successful_episode_count
                    print(f"Recorded {recorded} successful demonstrations.")

                if should_reset:
                    print("Resetting environment...")
                    # An unclaimed reset is the operators abandoning the attempt.
                    if recorded_steps and not outcome_claimed:
                        kept = args_cli.save and args_cli.export_mode == "all"
                        announce(
                            "discarded",
                            "RESET   saved as a failure" if kept else "RESET   not saved",
                            RED,
                        )
                    # Written before the export, so an entry exists even if
                    # the export raises.
                    if recorded_steps or pending_outcome is not None:
                        outcome, note = pending_outcome or (
                            "ABANDONED",
                            "saved as a failure" if args_cli.export_mode == "all" else "not saved",
                        )
                        episode_log.record(
                            env,
                            episode_id=episode_id,
                            outcome=outcome,
                            note=note,
                            steps=recorded_steps,
                            success_term=success_term,
                        )
                    pending_outcome = None
                    # Offer the episode to the exporter before dropping it.
                    # Skipped when nothing was stepped.
                    if recorded_steps:
                        env.recorder_manager.record_pre_reset([0])
                    env.sim.reset()
                    env.recorder_manager.reset()
                    env.reset()
                    settle_after_episode()
                    should_reset = False

                if env.sim.is_stopped():
                    break

    if haptics is not None:
        haptics.stop()
    env.close()
    if xr_guard is not None and (summary := xr_guard.summary()):
        print(summary)
    # Compare what reached the disk with what was announced.
    landed = getattr(_dataset_handler, "written", None)
    if landed is not None and args_cli.shard_demos and len(landed) != recorded:
        print(
            f"WARNING: {recorded} demonstration(s) were announced but {len(landed)} reached the disk."
            " The missing ones were still queued when this process ended."
        )
    episode_log.close(recorded)
    if not args_cli.save:
        print("Practice session completed (--no-save): nothing was written")
        print(f"Episode-by-episode outcomes: {episode_log.path}")
        return
    print(f"Recording session completed with {recorded} successful demonstrations")
    if landed is not None and args_cli.shard_demos:
        print(f"  {len(landed)} shard(s) written this session")
    print(f"Episode-by-episode outcomes: {episode_log.path}")
    print(f"Demonstrations saved to: {output_dir}{'/' if args_cli.shard_demos else ''}")

    if recorded and not args_cli.enable_cameras:
        print(
            "\n[cameras] This session recorded state and no pixels, which is the default.\n"
            "          Rebuild them before anything reads the images:\n"
            f"            python scripts/dataset/render_dataset_images.py {output_dir}{'/' if args_cli.shard_demos else ''} \\\n"
            "                --headless --enable_cameras --device cpu\n"
            "          That is also what puts them on the renderer evaluation uses\n"
            "          Videos and the export read the rebuilt copy."
        )

    if recorded and hidden_heads and args_cli.enable_cameras:
        print(
            "\n[cameras] Robot A's head was hidden for the headset, and USD visibility is\n"
            "          per stage rather than per camera -- so ego_b and scene recorded it\n"
            "          missing. Rebuild those views before using the images:\n"
            f"            python scripts/dataset/render_dataset_images.py {output_dir}{'/' if args_cli.shard_demos else ''} \\\n"
            "                --headless --enable_cameras --device cpu\n"
            "          (--no-hide_own_head records them correct at the cost of the\n"
            "           operator looking through their own head.)"
        )
    if recorded and ego_frame.shown and args_cli.enable_cameras:
        print(
            "\n[cameras] the ego frame drawn for the headset is scene geometry, and Robot A's\n"
            "          own ego camera is the only one built not to see it -- ego_b and scene\n"
            "          may show its bars in front of Robot A. Same repair as the hidden head:\n"
            f"            python scripts/dataset/render_dataset_images.py {output_dir}{'/' if args_cli.shard_demos else ''} \\\n"
            "                --headless --enable_cameras --device cpu\n"
            "          (--no-ego_frame records without it.)"
        )


def entrypoint() -> None:
    """Run the collector without letting Kit hide Python startup failures."""
    try:
        main()
    except BaseException:
        # SimulationApp.close() can force status 0 and hide the exception.
        # Print it, flush the dataset, clean up CloudXR, then leave through
        # os._exit.
        import traceback

        traceback.print_exc()
        _flush_dataset()
        release_cloudxr()
        _sys.stdout.flush()
        _sys.stderr.flush()
        os._exit(1)

    # Kit can skip atexit, so CloudXR is released explicitly.
    release_cloudxr()
    with contextlib.suppress(Exception):
        simulation_app.update()
        simulation_app.close()


if __name__ == "__main__":
    entrypoint()
