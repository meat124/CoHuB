"""One CloudXR runtime per headset: its ports and install directory, the landing page, and cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from urllib.parse import urlencode


# --- Profiles ---
# Each profile has its own install directory and ports, set through environment variables before the
# Kit app starts: that is when the OpenXR loader reads XR_RUNTIME_JSON.

@dataclass(frozen=True)
class CloudXRProfile:
    """One process's share of the CloudXR ports and state directory."""

    name: str
    install_dir: Path
    proxy_port: int
    """WSS signalling, what the headset's browser connects to."""
    backend_port: int
    """The runtime's plain-WS backend, behind the proxy. TCP. Set by
    ``NV_CXR_SERVER_PORT``."""
    media_port: int
    """This profile's CloudXR media socket, set by ``NV_CXR_MEDIA_PORT``. UDP.

    Every runtime also tests :data:`STARTUP_CHECK_PORT` at startup, so start
    all processes before any headset connects.
    """
    ui_port: int
    """Static web client / landing page."""
    turn_port: int
    """coturn, only used in USB-local mode."""

    def env(self) -> dict[str, str]:
        return {
            "CXR_INSTALL_DIR": str(self.install_dir),
            "CXR_HOST_VOLUME_PATH": str(self.install_dir),
            "PROXY_PORT": str(self.proxy_port),
            "NV_CXR_SERVER_PORT": str(self.backend_port),
            "NV_CXR_MEDIA_PORT": str(self.media_port),
            "USB_BACKEND_PORT": str(self.backend_port),
            "USB_UI_PORT": str(self.ui_port),
            "USB_TURN_PORT": str(self.turn_port),
            # Read by the OpenXR loader to pick a runtime.
            "XR_RUNTIME_JSON": str(self.install_dir / "openxr_cloudxr.json"),
            "NV_CXR_RUNTIME_DIR": str(self.install_dir / "run"),
            "NV_CXR_OUTPUT_DIR": str(self.install_dir / "logs"),
        }

    @property
    def runtime_json(self) -> Path:
        return self.install_dir / "openxr_cloudxr.json"

    def is_installed(self) -> bool:
        """Whether this profile's runtime has been started at least once.

        The OpenXR manifest is written into the install directory on first
        launch.
        """
        return self.runtime_json.is_file()

    def runtime_command(self) -> list[str]:
        """The command that starts this profile's CloudXR runtime standalone."""
        return [".venv/bin/python", "scripts/teleop/cloudxr_runtime.py", "--profile", self.name]


PRIMARY = CloudXRProfile(
    name="primary",
    install_dir=Path.home() / ".cloudxr",
    proxy_port=48322,
    backend_port=49100,
    media_port=47998,
    ui_port=8080,
    turn_port=3478,
)
"""Operator 1's profile, the stock layout."""

SECONDARY = CloudXRProfile(
    name="secondary",
    install_dir=Path.home() / ".cloudxr-b",
    proxy_port=48332,
    backend_port=49110,
    media_port=48998,
    ui_port=8081,
    turn_port=3488,
)
"""Operator 2's profile."""

TERTIARY = CloudXRProfile(
    name="tertiary",
    install_dir=Path.home() / ".cloudxr-c",
    proxy_port=48342,
    backend_port=49120,
    media_port=49998,
    ui_port=8082,
    turn_port=3498,
)
"""Operator 3's profile."""

PROFILES = {p.name: p for p in (PRIMARY, SECONDARY, TERTIARY)}

_PROFILE_BY_ROBOT = {"robot_a": PRIMARY, "robot_b": SECONDARY, "robot_c": TERTIARY}


def profile_for_robot(robot: str) -> CloudXRProfile:
    """The profile belonging to whoever drives ``robot``.

    Robot A maps to :data:`PRIMARY`, Robot B to :data:`SECONDARY`, Robot C to
    :data:`TERTIARY`, in every layout.
    """
    try:
        return _PROFILE_BY_ROBOT[robot]
    except KeyError:
        raise ValueError(
            f"no CloudXR profile for {robot!r}; the platform contract names the robots "
            f"{' and '.join(sorted(_PROFILE_BY_ROBOT))}"
        ) from None

STARTUP_CHECK_PORT = 47998
"""UDP port every runtime tests at startup, whatever it is configured with.

Bound only while a headset is streaming.
"""


def startup_port_busy(port: int = STARTUP_CHECK_PORT) -> bool:
    """Whether a runtime starting right now would fail its port check."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return True
    return False


def wait_for_startup_port(timeout: float, should_stop=None) -> bool:
    """Wait until a CloudXR runtime can start (:data:`STARTUP_CHECK_PORT` free). Returns whether it can.

    ``should_stop`` is polled while waiting.
    """
    import time

    if not startup_port_busy():
        return True

    print("\n" + "=" * 62)
    print(f"  Waiting: UDP {STARTUP_CHECK_PORT} is busy, so a CloudXR runtime cannot start.")
    print("\n  The other operator's runtime holds that port while their headset is")
    print("  streaming, and no setting moves it. Ask them to close their client tab")
    print("  for a moment; they can reconnect as soon as this says the port is free.")
    print("  Neither process needs restarting.")
    print(f"\n  Giving up after {timeout:g}s. Ctrl+C to stop waiting.")
    print("=" * 62 + "\n")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if should_stop is not None and should_stop():
            return False
        time.sleep(1.0)
        if not startup_port_busy():
            print("[cloudxr] port free -- starting this profile's runtime\n")
            return True
    print(f"[cloudxr] UDP {STARTUP_CHECK_PORT} never came free; not starting a runtime that would die on it")
    return False


def get_profile(name: str) -> CloudXRProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown CloudXR profile {name!r}; expected one of {sorted(PROFILES)}") from None


def apply_profile(name: str) -> CloudXRProfile:
    """Put a profile into ``os.environ``. Call before the Kit app is launched.

    Existing values are left alone, so the shell can override any port.
    """
    profile = get_profile(name)
    profile.install_dir.mkdir(parents=True, exist_ok=True)
    for key, value in profile.env().items():
        os.environ.setdefault(key, value)
    return profile


def active_install_dir() -> Path:
    """The install directory this process is configured to use."""
    return Path(os.environ.get("CXR_INSTALL_DIR", str(PRIMARY.install_dir))).expanduser()


def pin_wss_backend_port(port: int) -> None:
    """Make ``CloudXRLauncher``'s WSS proxy forward to the runtime on ``port``.

    Wraps ``wss.run`` to fill in ``backend_port``. Raises if ``wss.run`` does
    not take that parameter.
    """
    import inspect

    from isaacteleop.cloudxr import wss

    if "backend_port" not in inspect.signature(wss.run).parameters:
        raise RuntimeError(
            "isaacteleop's wss.run no longer takes 'backend_port'. Re-check how the proxy "
            "chooses its backend before running two CloudXR runtimes: if it still defaults "
            "to one profile's port, the second headset will join the first operator's "
            "stream instead of failing."
        )

    original = wss.run

    async def run_with_pinned_backend(*args, **kwargs):
        kwargs.setdefault("backend_port", port)
        return await original(*args, **kwargs)

    wss.run = run_with_pinned_backend


def pin_cloudxr_launcher(profile: CloudXRProfile) -> None:
    """Make Isaac Lab's auto-launched CloudXR runtime use this profile.

    Replaces ``isaacteleop.cloudxr.CloudXRLauncher`` with a subclass that
    forces this profile's install directory and backend port.
    """
    from isaacteleop import cloudxr

    original = cloudxr.CloudXRLauncher
    if getattr(original, "_cohub_pinned_profile", None) == profile.name:
        return

    class PinnedCloudXRLauncher(original):
        _cohub_pinned_profile = profile.name

        def __init__(self, install_dir=None, **kwargs):  # noqa: ARG002
            pin_wss_backend_port(profile.backend_port)
            super().__init__(install_dir=str(profile.install_dir), **kwargs)

    cloudxr.CloudXRLauncher = PinnedCloudXRLauncher


def resolve_cloudxr_env(value: str | None) -> str | None:
    """The CloudXR env file a ``--cloudxr_env`` argument names: a keyword
    (``cloudxrjs``, ``avp``), a path as given, or None for none."""
    if value is None or value.strip() == "" or value.lower() == "none":
        return None
    from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV

    return {"cloudxrjs": CLOUDXR_JS_ENV, "avp": CLOUDXR_AVP_ENV}.get(value.lower(), value)


# --- Landing page ---
# Two links: accept the self-signed certificate, then open the client with the streaming settings
# filled in.

CLIENT_BASE = "https://nvidia.github.io/IsaacTeleop/client/"
WSS_PROXY_PORT = 48322

_PAGE = """<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoHuB teleop</title>
<style>
 body {{ font-family: system-ui, sans-serif; background:#111; color:#eee;
        margin:0; padding:6vh 6vw; font-size:2.2vh; }}
 h1 {{ font-size:3.4vh; margin:0 0 3vh; }}
 a.step {{ display:block; background:#2a6; color:#fff; text-decoration:none;
           padding:3vh; margin:2vh 0; border-radius:1.2vh; font-size:3vh; }}
 a.step.alt {{ background:#36c; }}
 .n {{ opacity:.75; font-size:2vh; margin-top:1vh; }}
 code {{ background:#222; padding:.2em .4em; border-radius:.3em; }}
</style>
<h1>CoHuB &mdash; Quest teleop</h1>
<a class="step" href="https://{host}:{wss}/">1. Accept certificate</a>
<div class="n">Tap <b>Advanced</b> &rarr; <b>Proceed to {host} (unsafe)</b>, then come back.</div>
<a class="step alt" href="{client}">2. Open teleop client</a>
<div class="n">Streaming: {bitrate} Mbps, {per_eye}px per eye, {fps} FPS &mdash; server <code>{host}</code></div>
"""


class _Handler(BaseHTTPRequestHandler):
    params: dict = {}
    wss_port: int = WSS_PROXY_PORT

    def do_GET(self):  # noqa: N802
        host = self.headers.get("Host", "").split(":")[0] or "localhost"
        query = dict(self.params)
        query["serverIP"] = host
        client_url = CLIENT_BASE + "?" + urlencode(query)
        body = _PAGE.format(
            host=host,
            wss=self.wss_port,
            client=client_url,
            bitrate=query.get("maxStreamingBitrateMbps", "-"),
            per_eye=query.get("perEyeWidth", "-"),
            fps=query.get("deviceFrameRate", "-"),
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def local_ips() -> list[str]:
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3)
        return [ip for ip in out.stdout.split() if ":" not in ip]
    except (OSError, subprocess.SubprocessError):
        return []


def start_link_server(
    port: int, bitrate_mbps: int, per_eye: int, fps: int, wss_port: int = WSS_PROXY_PORT
) -> list[str]:
    """Serve the landing page in a daemon thread; return the URLs to open.

    ``wss_port`` is the CloudXR WSS proxy this session owns.
    """
    _Handler.params = {
        "maxStreamingBitrateMbps": bitrate_mbps,
        "perEyeWidth": per_eye,
        "perEyeHeight": per_eye,
        "deviceFrameRate": fps,
    }
    if wss_port != WSS_PROXY_PORT:
        _Handler.params["port"] = wss_port
    _Handler.wss_port = wss_port
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return [f"http://{ip}:{port}" for ip in local_ips()]


# --- Orphaned runtime cleanup ---
# Scoped by install directory (read from /proc/<pid>/environ), so one profile's cleanup leaves the
# others alone.

_RUNTIME_MARKER = "isaacteleop.cloudxr.runtime"
_STALE_FILES = ("ipc_cloudxr", "runtime_started", "cloudxr.pid", "monado.pid")


def _resolve(path: str | os.PathLike) -> Path:
    return Path(path).expanduser().resolve()


def _runtime_install_dir(pid: int) -> Path | None:
    """Which install directory a runtime process belongs to.

    Returns ``None`` when the environment cannot be read.
    """
    try:
        raw = (Path("/proc") / str(pid) / "environ").read_bytes()
    except (OSError, PermissionError):
        return None
    for item in raw.split(b"\0"):
        key, _, value = item.partition(b"=")
        if key == b"CXR_INSTALL_DIR" and value:
            return _resolve(value.decode("utf-8", "replace"))
    # A runtime started without the variable uses the stock location.
    return _resolve(PRIMARY.install_dir)


def _runtime_pids(install_dir: Path | None = None) -> list[int]:
    """PIDs of live CloudXR runtime workers, optionally only one profile's."""
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except (OSError, PermissionError):
            continue
        if _RUNTIME_MARKER not in cmdline:
            continue
        pid = int(entry.name)
        if install_dir is not None and _runtime_install_dir(pid) != install_dir:
            continue
        pids.append(pid)
    return pids


def _kill_group(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def force_cleanup(timeout: float = 4.0, install_dir: str | os.PathLike | None = None, all_profiles: bool = False):
    """Terminate orphaned CloudXR runtimes and clear stale state files. Returns the PIDs killed.

    Only the runtime owning ``install_dir`` (default: this process's) is touched, unless ``all_profiles``.
    """
    scope = None if all_profiles else _resolve(install_dir or active_install_dir())
    pids = [pid for pid in _runtime_pids(scope) if pid != os.getpid()]

    for pid in pids:
        _kill_group(pid, signal.SIGTERM)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _runtime_pids(scope):
            break
        time.sleep(0.2)

    for pid in _runtime_pids(scope):
        _kill_group(pid, signal.SIGKILL)

    for run_dir in _run_dirs(scope):
        for name in _STALE_FILES:
            try:
                (run_dir / name).unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    return pids


def running_runtimes() -> dict[str, list[int]]:
    """Which runtimes are alive, by profile name. Kills nothing."""

    by_dir = {_resolve(p.install_dir): p.name for p in PROFILES.values()}
    found: dict[str, list[int]] = {}
    for pid in _runtime_pids():
        install = _runtime_install_dir(pid)
        name = by_dir.get(install, str(install) if install else "unknown")
        found.setdefault(name, []).append(pid)
    return found


def _run_dirs(scope: Path | None) -> list[Path]:
    if scope is not None:
        return [scope / "run"]

    return [p.install_dir / "run" for p in PROFILES.values()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kill orphaned CloudXR runtimes.")
    parser.add_argument(
        "profile",
        nargs="*",
        default=[],
        help="Which session(s) to clean: 'primary' (the recording process), "
        "'secondary' (the second headset), 'tertiary' (the third). Several may "
        "be named at once, which is how a layout cleans up exactly the "
        "profiles it started. Defaults to this shell's CXR_INSTALL_DIR, i.e. "
        "the primary unless you set it.",
    )
    parser.add_argument("--all", action="store_true", help="Clean every profile's runtime.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Report which runtimes are alive and exit, killing nothing. What "
        "a status check wants; --all is for clearing the decks by hand.",
    )
    cli = parser.parse_args()

    if cli.list:
        alive = running_runtimes()
        if not alive:
            print("CloudXR runtimes: none running")
        else:
            for name, pids in sorted(alive.items()):
                print(f"CloudXR runtime '{name}': pid {', '.join(str(p) for p in pids)}")
        raise SystemExit(0)

    def report(where: str, killed: list[int]) -> None:
        print(f"CloudXR cleanup ({where}): killed {killed}" if killed else f"CloudXR cleanup ({where}): nothing running")

    if cli.all:
        report("every profile", force_cleanup(all_profiles=True))
        raise SystemExit(0)

    # One pass per named profile, so other sessions' runtimes are left alone.
    for target in [get_profile(name).install_dir for name in cli.profile] or [None]:
        report(str(_resolve(target or active_install_dir())), force_cleanup(install_dir=target))
