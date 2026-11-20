"""Mirror process: one headset operator driving one robot (``--robot``).

Holds a copy of the scene, re-poses it from the state the recording process
sends over UDP, and sends back its robot's 32D action. No physics, sensors or
recorders. Start the recording process first:

    python scripts/teleop/record_demos.py --xr --visualizer kit
    python scripts/teleop/xr_mirror.py --xr
"""

import argparse
import contextlib
import pathlib as _pathlib_early
import sys as _sys_early

from isaaclab.app import AppLauncher

# source/ must be on sys.path before the cohub imports below.
_sys_early.path.insert(0, str(_pathlib_early.Path(__file__).resolve().parents[2] / "source"))
from cohub.teleop.view import (  # noqa: E402
    FRAME_COLOURS,
    DEFAULT_FRAME_COLOUR,
    DEFAULT_FRAME_DISTANCE,
)

parser = argparse.ArgumentParser(description="Second-headset process: operator 2, mirrored from the primary.")
parser.add_argument(
    "--robot",
    type=str,
    default="robot_b",
    choices=["robot_a", "robot_b", "robot_c"],
    help="Which robot this operator drives. robot_c only exists on a three-robot task.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-CoCarry-G1x2-v0",
    help="Task whose scene to mirror. Must match the primary's --task.",
)
parser.add_argument("--peer", type=str, default="127.0.0.1", help="Host running the primary process.")
parser.add_argument(
    "--state_port",
    type=int,
    default=None,
    help="UDP port this process listens on. Default: 47701 (robot_b), 47711 (robot_a), 47721 (robot_c).",
)
parser.add_argument(
    "--action_port",
    type=int,
    default=None,
    help="UDP port the primary listens on. Defaults by --robot: 47702 for "
    "robot_b, 47712 for robot_a, 47722 for robot_c.",
)
parser.add_argument(
    "--cloudxr_profile",
    type=str,
    default=None,
    help="CloudXR profile. Default: 'primary' for robot_a (page 8080), 'secondary' for robot_b (8081).",
)
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default="cloudxrjs",
    help="CloudXR .env path or shorthand ('cloudxrjs', 'avp', 'none').",
)
parser.add_argument(
    "--auto_launch_cloudxr",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Launch this profile's CloudXR runtime with the session. Off to run it yourself (cloudxr_runtime.py).",
)
parser.add_argument("--link_port", type=int, default=0, help="Landing-page port (0 = use the profile's).")
parser.add_argument("--stream_bitrate", type=int, default=50, help="Client streaming bitrate cap in Mbps (max 100).")
parser.add_argument("--stream_per_eye", type=int, default=1024, help="Client per-eye render size in pixels.")
parser.add_argument("--stream_fps", type=int, default=72, help="Client target frame rate.")
parser.add_argument(
    "--retargeting",
    type=str,
    default="sync",
    choices=["pipelined", "sync"],
    help="Retargeter mode (see record_demos.py). Default 'sync'; 'pipelined' adds one frame of pose age.",
)
parser.add_argument(
    "--max_fps",
    type=float,
    default=36.0,
    help="Cap on this process's loop rate [Hz]. 0 follows --stream_fps, negative runs uncapped.",
)
parser.add_argument(
    "--haptics",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Vibrate this operator's controllers from the load the primary measures on their robot's hands.",
)
parser.add_argument(
    "--hide_own_head",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Hide this operator's own robot head in this process's render only; the primary's recordings keep it.",
)
parser.add_argument(
    "--ego_frame",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Draw the recorded ego image's boundary in the headset (--xr only). Free here: a mirror records nothing.",
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
parser.add_argument("--eye_height", type=float, default=0.0, help="Your eye height in m; 0 keeps the roomier anchor.")
parser.add_argument(
    "--crouch",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Left controller X lowers this robot's hips, Y raises them, and the viewpoint follows.",
)
parser.add_argument("--crouch_rate", type=float, default=0.25, help="Metres per second the hips move while held.")
parser.add_argument(
    "--crouch_min",
    type=float,
    default=0.60,
    help="Lowest hip height to allow in m; matches the primary's default.",
)
parser.add_argument(
    "--anchor_follows_crouch",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Lower this operator's viewpoint with their robot's hips, so the arms keep tracking during a crouch.",
)
parser.add_argument(
    "--wait_for_port",
    type=float,
    default=600.0,
    help="Seconds to wait when another runtime holds the CloudXR startup port (47998/udp). 0 fails immediately.",
)
parser.add_argument(
    "--controller_session_buttons",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Start/pause on this operator's right-controller A and reset on B, forwarded to the primary.",
)
parser.add_argument(
    "--vr_status",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show each demonstration's outcome as text in this headset; the primary decides it and sends it over.",
)
parser.add_argument("--rate_report", type=int, default=250, help="Frames between rate reports (0 disables).")
parser.add_argument(
    "--mirror_check",
    type=int,
    default=0,
    help="Exit after this many state packets, reporting how faithfully they were mirrored. 0 runs normally.",
)
AppLauncher.add_app_launcher_args(parser)
# Pink IK is CPU-only.
parser.set_defaults(device="cpu")
# rendering_mode stays on the AppLauncher default. "performance" renders the XR
# world black.
args_cli = parser.parse_args()

from cohub.task_registry import gym_id  # noqa: E402

args_cli.task = gym_id(args_cli.task)

# Needed before Kit starts, for the CloudXR profile check below.
_OPERATOR = {"robot_a": "1", "robot_b": "2", "robot_c": "3"}
_PRIM_NAMES = {"robot_a": "Robot_A", "robot_b": "Robot_B", "robot_c": "Robot_C"}


# The CloudXR profile must be applied before anything Kit-related is imported.
from cohub.teleop.cloudxr import apply_profile, pin_cloudxr_launcher, profile_for_robot  # noqa: E402

from cohub.runtime import exit_gate  # noqa: E402
from cohub.federation import ports_for_robot  # noqa: E402

# Each robot's link has its own port pair.
_state_port, _action_port = ports_for_robot(args_cli.robot)
if args_cli.state_port is None:
    args_cli.state_port = _state_port
if args_cli.action_port is None:
    args_cli.action_port = _action_port

PROFILE = apply_profile(args_cli.cloudxr_profile or profile_for_robot(args_cli.robot).name)

# Launching a runtime into a profile that already has a live one kills it, so
# refuse. --no-auto_launch_cloudxr skips the check.
if args_cli.xr and args_cli.auto_launch_cloudxr:
    from cohub.teleop.cloudxr import _runtime_pids  # noqa: E402

    _owners = _runtime_pids(PROFILE.install_dir)
    if _owners:
        parser.error(
            f"CloudXR profile {PROFILE.name!r} (operator "
            f"{_OPERATOR[args_cli.robot]}, {PROFILE.install_dir}) already has a "
            f"live runtime (pid {_owners[0]}). Launching another there would kill it.\n"
            f"  If another session owns it, that session is the one to stop.\n"
            f"  If it is left over from a crash: ./scripts/teleop/collect.sh cleanup {PROFILE.name}"
        )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if args_cli.xr:
    import carb.settings

    # Skip the "Start AR" click.
    carb.settings.get_settings().set("/isaaclab/xr/auto_start", True)

import logging  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab_teleop import IsaacTeleopCfg, XrAnchorRotationMode, XrCfg  # noqa: E402

from cohub.teleop.follower import ConstantFollower  # noqa: E402
from cohub.runtime import configure_render, CpuLoad, describe_resources  # noqa: E402
from cohub.federation import ResetRelay, SecondaryLink, scene_state  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402  (registry side effects)
from isaaclab.sensors import SensorBaseCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import cohub.tasks  # noqa: F401, E402  (registers the CoHuB tasks)
from cohub.teleop.cloudxr import start_link_server, force_cleanup, wait_for_startup_port  # noqa: E402
from cohub.teleop.controls import (  # noqa: E402
    ControllerHaptics,
    OperatorInputCapture,
    SessionButtons,
    release_upstream_a_binding,
    guard_xr_session,
    CrouchControl,
)
from cohub.teleop.view import (  # noqa: E402
    hide_head,
    EgoFrame,
    describe_ego_frame,
    GREEN,
    RED,
    SessionHud,
    LEGACY_ANCHOR_Z,
    anchor_rot_for,
    anchor_z_for_eye_height,
    default_eye_height,
    mount_rise_over_pelvis,
    set_anchor_dz,
)

logger = logging.getLogger(__name__)

ANCHOR_TARGET = None
"""Stand-in for an env config, so the crouch can move this process's anchor."""

_shutdown_requested = False


def cleanup_own_runtime() -> list[int]:
    """Clean up this profile's CloudXR runtime, only if this process started it.

    With ``--no-auto_launch_cloudxr`` the runtime belongs to another terminal
    and is left alone.
    """
    if not args_cli.auto_launch_cloudxr:
        return []
    return force_cleanup(install_dir=PROFILE.install_dir)


def cloudxr_failure_hint() -> str | None:
    """Why this profile's runtime refused to start, read from its log, or ``None``."""
    logs = sorted((PROFILE.install_dir / "logs").glob("cxr_server.*.log"))
    if not logs:
        return None
    try:
        tail = logs[-1].read_text(errors="replace")
    except OSError:
        return None
    if "STREAMSDK_PORT_UNAVAILABLE" not in tail:
        return None
    port = next(
        (line.split("Port ")[1].split()[0] for line in tail.splitlines() if "is already in use" in line),
        "47998",
    )
    return (
        f"This profile's CloudXR runtime could not start: port {port} was already in use\n"
        f"  ({logs[-1]}).\n\n"
        "  That port belongs to the other operator's runtime, and it is held only while\n"
        "  their headset is actually streaming -- so this is almost certainly the session\n"
        "  order: operator 1 connected before this process started. Nothing here can move\n"
        "  the port (the runtime's own advice, NV_CXR_SERVER_PORT, moves a different one).\n\n"
        "  Start both processes first, then put both headsets on:\n"
        "      ./scripts/teleop/collect.sh dual"
    )


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        global _shutdown_requested
        if _shutdown_requested:
            print("\nForced shutdown...")
            cleanup_own_runtime()
            os._exit(1)
        _shutdown_requested = True
        print("\nShutdown requested; finishing current frame (Ctrl+C again to force)...")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)


from cohub.teleop.cloudxr import resolve_cloudxr_env  # noqa: E402


def build_env_cfg():
    """The named task's env cfg, with every sensor stripped from the scene."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    configure_render(env_cfg.sim.render)
    stripped = []
    for name, value in list(env_cfg.scene.__dict__.items()):
        if isinstance(value, SensorBaseCfg):
            setattr(env_cfg.scene, name, None)
            stripped.append(name)
    kept = sorted(
        name for name, value in env_cfg.scene.__dict__.items()
        if value is not None and hasattr(value, "prim_path")
    )
    print(f"[scene] {args_cli.task}: mirroring {', '.join(kept)}"
          + (f"  (stripped {len(stripped)} sensors)" if stripped else ""))
    return env_cfg


def build_teleop_cfg(scene_cfg) -> IsaacTeleopCfg:
    """The upstream teleop pipeline, anchored to this process's robot."""
    from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
        _build_g1_locomanipulation_pipeline,
    )

    eye_height = args_cli.eye_height or default_eye_height()
    anchor_z = anchor_z_for_eye_height(eye_height) if eye_height > 0.0 else LEGACY_ANCHOR_Z

    # The anchor rotation sets which way the operator faces on connecting.
    xr = XrCfg(anchor_pos=(0.0, 0.0, anchor_z), anchor_rot=anchor_rot_for(args_cli.robot, scene_cfg))
    xr.anchor_prim_path = f"/World/envs/env_0/{_PRIM_NAMES[args_cli.robot]}/pelvis"
    xr.fixed_anchor_height = True
    xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
    xr.anchor_rotation_smoothing_time = 0.3
    cfg = IsaacTeleopCfg(
        pipeline_builder=_build_g1_locomanipulation_pipeline,
        sim_device=args_cli.device,
        xr_cfg=xr,
    )
    if args_cli.retargeting == "sync":
        from isaacteleop.teleop_session_manager import RetargetingExecutionConfig

        cfg.retargeting_execution = RetargetingExecutionConfig(mode="sync")
        print("[teleop] retargeting sync: poses computed in-frame instead of one frame late")
    # set_anchor_dz expects an env config, so it gets an object shaped like one.
    global ANCHOR_TARGET
    ANCHOR_TARGET = SimpleNamespace(xr=cfg.xr_cfg, isaac_teleop=cfg)
    return cfg


class _RateMeter:
    """Frames per second of this process's own loop."""

    def __init__(self, report_every: int):
        self.report_every = report_every
        self.count = 0
        self.steps: list[float] = []
        self._last = time.perf_counter()
        self._cpu = CpuLoad()

    def tick(self, link_age: float | None) -> None:
        now = time.perf_counter()
        self.steps.append(now - self._last)
        self._last = now
        self.count += 1
        if self.report_every and self.count % self.report_every == 0:
            steps = sorted(self.steps)
            n = len(steps)
            median_hz = 1.0 / steps[n // 2] if steps[n // 2] > 0 else 0.0
            age = "no state yet" if link_age is None else f"newest state {link_age * 1e3:.0f} ms old"
            resources = describe_resources(self._cpu)
            print(
                f"[rate] mirror {median_hz:.1f} Hz median over {n} frames"
                f"  (p90 {steps[int(n * 0.9)] * 1e3:.0f} ms); {age}"
                + (f"  [{resources}]" if resources else "")
            )
            self.steps.clear()


def print_banner(landing_urls) -> None:
    operator = _OPERATOR[args_cli.robot]
    print("\n" + "=" * 62)
    print(f"  Mirror process -- operator {operator} in {args_cli.robot}")
    print(f"  CloudXR profile '{PROFILE.name}' (operator {operator}'s): {PROFILE.install_dir}")
    print(
        f"    wss {PROFILE.proxy_port}   backend {PROFILE.backend_port}"
        f"   stream {PROFILE.media_port}/udp   client {PROFILE.ui_port}"
    )
    if args_cli.xr and args_cli.auto_launch_cloudxr:
        print("\n  This process owns its CloudXR runtime, pinned to the profile above")
        print("  (Isaac Lab would otherwise point it at ~/.cloudxr and tear down the")
        print("  other operator's). It starts with the session and stops with it.")
    elif args_cli.xr:
        print("\n  --no-auto_launch_cloudxr: start this profile's runtime yourself:")
        print("      " + " ".join(PROFILE.runtime_command()))
        print("  Note it shuts itself down ~150 s after the last client leaves, so")
        print("  connect the headset promptly or let this process own it instead.")
        if not PROFILE.is_installed():
            # Without the manifest the OpenXR loader uses the default runtime.
            print(f"\n  !! {PROFILE.runtime_json} does not exist yet.")
            print("     Run that command once before this one, or this headset will")
            print("     attach to the wrong runtime.")
    if landing_urls:
        print(f"\n  On operator {operator}'s Quest, open:")
        for url in landing_urls:
            print(f"      {url}")
    print("\n  Waiting for scene state from the primary process on "
          f"{args_cli.peer}:{args_cli.state_port}.")
    print("  Robot poses come from there; nothing here is simulated.")
    print("=" * 62 + "\n")


def main() -> int:
    _install_signal_handlers()

    killed = cleanup_own_runtime()
    if killed:
        print(f"Cleaned up orphaned CloudXR runtime(s) for '{PROFILE.name}': {killed}")

    landing_urls = []
    if args_cli.xr:
        port = args_cli.link_port or PROFILE.ui_port
        if port:
            landing_urls = start_link_server(
                port,
                args_cli.stream_bitrate,
                args_cli.stream_per_eye,
                args_cli.stream_fps,
                wss_port=PROFILE.proxy_port,
            )

    env_cfg = build_env_cfg()
    sim = SimulationContext(
        SimulationCfg(
            device=args_cli.device,
            dt=env_cfg.sim.dt,
            render_interval=env_cfg.sim.render_interval,
            # State is dictated by the recording process, so gravity is off.
            gravity=(0.0, 0.0, 0.0),
        )
    )
    scene = InteractiveScene(env_cfg.scene)
    # scene_state.apply() needs these entities.
    for required in ("robot_a", "robot_b", "object", args_cli.robot):
        if required not in scene.keys():
            raise SystemExit(f"task {args_cli.task} has no scene entity {required!r}; the mirror cannot follow it")
    sim.reset()

    if args_cli.hide_own_head:
        hide_head(f"/World/envs/env_0/{_PRIM_NAMES[args_cli.robot]}")

    from isaaclab_teleop import create_isaac_teleop_device, poll_control_events

    if args_cli.auto_launch_cloudxr:
        # Must run before the device is created, which launches the runtime.
        pin_cloudxr_launcher(PROFILE)
        if args_cli.wait_for_port > 0 and not wait_for_startup_port(
            args_cli.wait_for_port, should_stop=lambda: _shutdown_requested
        ):
            return 1

    link = SecondaryLink(
        peer_host=args_cli.peer,
        state_port=args_cli.state_port,
        action_port=args_cli.action_port,
    )

    reset_relay = ResetRelay(link)

    teleop = create_isaac_teleop_device(
        build_teleop_cfg(env_cfg.scene),
        sim_device=args_cli.device,
        callbacks={
            # This operator's start/stop/reset is forwarded to the recording
            # process. Reset goes through the relay.
            "START": lambda: link.request_run_state(True),
            "STOP": lambda: link.request_run_state(False),
            "RESET": reset_relay.request,
            "R": reset_relay.request,
        },
        cloudxr_env_file=resolve_cloudxr_env(args_cli.cloudxr_env),
        auto_launch_cloudxr=args_cli.auto_launch_cloudxr,
    )
    xr_guard = guard_xr_session(teleop)

    print_banner(landing_urls)

    # 0 follows the client's frame rate. Negative is uncapped.
    target_fps = args_cli.stream_fps if args_cli.max_fps == 0.0 else args_cli.max_fps
    frame_budget = 1.0 / target_fps if target_fps > 0 else 0.0
    next_deadline = None
    print(
        f"  frame cap: {target_fps:g} Hz"
        if frame_budget > 0
        else "  frame cap: none (this competes with the recording process for the GPU)"
    )

    buttons = SessionButtons() if (args_cli.xr and args_cli.controller_session_buttons) else None
    if buttons is not None:
        buttons.bind(teleop)
        if release_upstream_a_binding(teleop):
            print("[buttons] A is start/pause here; upstream's anchor-rotation toggle is unbound")
    hud = SessionHud(enabled=args_cli.vr_status and args_cli.xr)
    ego_frame = EgoFrame(
        f"/World/envs/env_0/{_PRIM_NAMES[args_cli.robot]}",
        enabled=args_cli.ego_frame and args_cli.xr,
        distance=args_cli.ego_frame_distance,
        colour=args_cli.ego_frame_color,
    )
    if ego_frame.show():
        print(describe_ego_frame(args_cli.robot, args_cli.ego_frame_distance, args_cli.ego_frame_color))
    _OUTCOMES = {
        "success": ("SUCCESS   demonstration saved", GREEN),
        "discarded": ("RESET   not saved", RED),
    }
    hold = ConstantFollower()
    capture = OperatorInputCapture()
    haptics = ControllerHaptics() if args_cli.haptics else None
    rate = _RateMeter(report_every=args_cli.rate_report)
    crouch = CrouchControl(rate=args_cli.crouch_rate, minimum=args_cli.crouch_min) if args_cli.crouch else None
    # Standing reference for the mount's rise over the pelvis.
    mount_rise_standing = mount_rise_over_pelvis(scene[args_cli.robot])
    MOUNT_REF_ALPHA = 0.02

    last_action = None
    teleop_online = False
    primary_seen = False
    last_episode = None
    states_applied = 0
    worst_joint_error = 0.0
    worst_root_error = 0.0

    with teleop, link, contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and not _shutdown_requested:
            state = link.poll_state()
            if state is not None and state.task and state.task != args_cli.task:
                # The mirror and the recording process must run the same task.
                print("\n" + "=" * 62)
                print("  Wrong task: this mirror and the recording process are in different scenes.")
                print(f"    this process : {args_cli.task}")
                print(f"    the primary  : {state.task}")
                print("\n  Nothing is wrong with the link -- restart this process with the")
                print("  primary's task and both operators will be in the same room:")
                print(f"      python scripts/teleop/xr_mirror.py --xr --task {state.task}")
                print("=" * 62 + "\n")
                return 1
            if state is not None:
                if not primary_seen:
                    primary_seen = True
                    print(
                        f"[federation] primary online ({state.num_joints} joints per robot"
                        + (f", {len(scene_state.extra_keys(scene))} scene prop(s)" if scene_state.extra_keys(scene) else "")
                        + ")"
                    )
                if last_episode is not None and state.episode_id != last_episode:
                    # New episode: reset this operator's retargeters too.
                    reset_relay.follow_primary(teleop)
                    if crouch is not None:
                        crouch.reset()
                        if args_cli.anchor_follows_crouch and ANCHOR_TARGET is not None:
                            set_anchor_dz(ANCHOR_TARGET, 0.0)
                        if last_action is not None:
                            last_action[31] = crouch.height
                    print(f"[federation] episode {state.episode_id}")
                last_episode = state.episode_id
                # The outcome is held for several frames. The HUD drops repeats.
                if state.outcome is not None:
                    hud.show(*_OUTCOMES[state.outcome])
                scene_state.apply(scene, state, args_cli.device)
                states_applied += 1
            elif primary_seen and link.stale:
                primary_seen = False
                print("[federation] primary OFFLINE - the scene is frozen at its last state")

            if state is not None:
                # Step only when new state arrived, to push it through to the
                # renderer.
                scene.write_data_to_sim()
                sim.step(render=True)
                scene.update(sim.get_physics_dt())
            else:
                sim.render()

            if state is not None:
                joint_error, root_error = scene_state.pose_error(scene, state)
                worst_joint_error = max(worst_joint_error, joint_error)
                worst_root_error = max(worst_root_error, root_error)

            live = teleop.advance()
            if live is not None:
                if not teleop_online:
                    print("[teleop] online")
                    teleop_online = True
                    # A session coming up emits a reset of its own. Do not forward it.
                    reset_relay.settle()
                last_action = live.flatten().cpu().numpy().astype(np.float32)
                if crouch is not None:
                    # Index 31 is the hip height in metres. The crouch buttons own it.
                    last_action[31] = crouch.update(teleop, time.time())
                    if args_cli.anchor_follows_crouch and ANCHOR_TARGET is not None:
                        # Track the achieved mount, not the commanded pelvis.
                        rise = mount_rise_over_pelvis(scene[args_cli.robot])
                        if crouch.anchor_dz == 0.0:
                            mount_rise_standing += MOUNT_REF_ALPHA * (rise - mount_rise_standing)
                        set_anchor_dz(ANCHOR_TARGET, crouch.anchor_dz + rise - mount_rise_standing)
            elif teleop_online:
                print("[teleop] OFFLINE - holding this robot's last pose")
                teleop_online = False

            if buttons is not None:
                pressed_run, pressed_reset = buttons.consume_control_flags()
                if pressed_run:
                    link.request_toggle_run()
                if pressed_reset:
                    reset_relay.request()

            # advance() already fired the callbacks. This only reads the state.
            poll_control_events(teleop)

            if last_action is None:
                # No headset sample yet: hold the nominal pose at this robot's pelvis.
                if state is not None:
                    root = scene_state.robot_root_from_state(scene, state, args_cli.robot)
                    root_pos, root_quat = root[:3], root[3:]
                else:
                    root_pos, root_quat = None, None
                outgoing = hold.advance(root_pos, root_quat)
            else:
                outgoing = last_action

            link.send_action(
                outgoing,
                teleop_online=teleop_online,
                operator_raw=capture.capture(teleop) if teleop_online else None,
            )

            if haptics is not None and state is not None and haptics.bind(teleop):
                # The haptic slot carries the load on this operator's robot.
                haptics.send(float(state.haptic[0]), float(state.haptic[1]))

            rate.tick(link.seconds_since_last_packet if primary_seen else None)

            if frame_budget > 0.0:
                # Fixed-deadline pacing. A late frame still sleeps briefly and
                # the deadline re-anchors.
                now = time.perf_counter()
                if next_deadline is None:
                    next_deadline = now + frame_budget
                elif now < next_deadline:
                    time.sleep(next_deadline - now)
                    next_deadline += frame_budget
                else:
                    time.sleep(0.002)
                    next_deadline = time.perf_counter() + frame_budget

            if args_cli.mirror_check and states_applied >= args_cli.mirror_check:
                break

    if haptics is not None:
        haptics.stop()

    if xr_guard is not None and (guard_summary := xr_guard.summary()):
        print(guard_summary)

    code = 0
    if args_cli.mirror_check:
        print(f"\nMirror check over {states_applied} state packets:")
        print(f"  worst joint disagreement: {worst_joint_error:.2e} rad")
        print(f"  worst root disagreement:  {worst_root_error:.2e} m")
        # Too few packets is a failure.
        enough = states_applied >= args_cli.mirror_check
        if not enough:
            print(f"  only {states_applied} of {args_cli.mirror_check} packets arrived -- nothing was verified")
        ok = enough and worst_joint_error < 1e-3 and worst_root_error < 1e-3
        print("MIRROR CHECK PASS" if ok else "MIRROR CHECK FAIL")
        code = 0 if ok else 1
    elif states_applied == 0:
        flag = f"--{args_cli.robot}_source federated"
        print(f"\nNo state ever arrived from the primary -- is it running with "
              f"{flag}, and are the ports ({args_cli.state_port}/"
              f"{args_cli.action_port}) the same on both sides?")
    else:
        print(f"\nApplied {states_applied} state packets.")
    return code


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        # Print the diagnosis before the traceback.
        hint = cloudxr_failure_hint()
        if hint is None:
            raise
        print("\n" + "=" * 62 + f"\n  {hint}\n" + "=" * 62 + "\n")
        code = 1
    finally:
        cleanup_own_runtime()
        with contextlib.suppress(Exception):
            simulation_app.update()
    # simulation_app.close() never returns, so exit_gate carries the status.
    exit_gate(code)
