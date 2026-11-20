"""Start one profile's CloudXR runtime on its own ports.

    python scripts/teleop/cloudxr_runtime.py --profile secondary

The first run against a new install directory prompts for the CloudXR EULA.
"""

import argparse
import signal
import sys
import time


from cohub.teleop.cloudxr import PROFILES, apply_profile, pin_wss_backend_port  # noqa: E402

parser = argparse.ArgumentParser(description="Start one profile's CloudXR runtime.")
parser.add_argument("--profile", type=str, default="secondary", choices=sorted(PROFILES), help="Which profile.")
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default="cloudxrjs",
    help="CloudXR .env path or shorthand ('cloudxrjs', 'avp', 'none').",
)
parser.add_argument(
    "--accept_eula",
    action="store_true",
    help="Accept the NVIDIA CloudXR EULA without the prompt (otherwise asked once per install directory).",
)
args = parser.parse_args()

PROFILE = apply_profile(args.profile)


def _resolve_cloudxr_env(value: str) -> str | None:
    if not value or value.lower() == "none":
        return None
    from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV

    return {"cloudxrjs": CLOUDXR_JS_ENV, "avp": CLOUDXR_AVP_ENV}.get(value.lower(), value)


def main() -> int:
    from isaacteleop.cloudxr import CloudXRLauncher

    pin_wss_backend_port(PROFILE.backend_port)

    print(f"\nCloudXR runtime -- profile '{PROFILE.name}'")
    print(f"  install dir : {PROFILE.install_dir}")
    print(f"  runtime      : NV_CXR_SERVER_PORT={PROFILE.backend_port}  (tcp, signalling)")
    print(f"                 NV_CXR_MEDIA_PORT={PROFILE.media_port}  (udp, stream)")
    print(f"  wss proxy    : {PROFILE.proxy_port} -> localhost:{PROFILE.backend_port}")
    print(f"  landing page : {PROFILE.ui_port} (served by the teleop script, not this one)\n")

    launcher = CloudXRLauncher(
        install_dir=str(PROFILE.install_dir),
        env_config=_resolve_cloudxr_env(args.cloudxr_env),
        accept_eula=args.accept_eula,
    )
    print("Runtime ready. Ctrl+C to stop (this leaves the other profile's runtime alone).\n")

    stopping = False

    def handle(signum, _frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle)

    try:
        while not stopping:
            time.sleep(0.5)
            launcher.health_check()
    except RuntimeError as exc:
        print(f"\nRuntime stopped unexpectedly: {exc}")
        return 1
    finally:
        print("Stopping CloudXR runtime...")
        launcher.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
