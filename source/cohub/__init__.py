"""CoHuB: a multi-humanoid collaborative manipulation benchmark.

Importing :mod:`cohub.tasks` registers the benchmark environments with
gymnasium (same lazy-entry-point pattern as ``isaaclab_tasks``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.1.0"  # keep in sync with pyproject.toml


_PACKAGE_MARKER = Path("source") / "cohub" / "__init__.py"


def _checkout_of(start: Path) -> Path | None:
    """The cohub checkout ``start`` lives in, or None if it is outside one."""
    for candidate in (start, *start.parents):
        if (candidate / _PACKAGE_MARKER).is_file():
            return candidate
    return None


def _caller_checkout() -> Path | None:
    """The checkout the entry point belongs to: by script path, then by cwd."""
    script = sys.argv[0] if sys.argv else ""
    if script and not script.startswith("-"):
        try:
            resolved = Path(script).resolve()
        except OSError:
            resolved = None
        if resolved is not None:
            found = _checkout_of(resolved.parent)
            if found is not None:
                return found
    try:
        return _checkout_of(Path.cwd())
    except OSError:
        return None


def _warn_if_foreign_checkout() -> None:
    """Warn on stderr when this package was loaded from a different checkout.

    Set ``COHUB_CHECKOUT_WARNING=0`` to silence it.
    """
    if os.environ.get("COHUB_CHECKOUT_WARNING") == "0":
        return
    caller = _caller_checkout()
    if caller is None:
        return
    own = Path(__file__).resolve().parents[2]
    if caller.resolve() == own:
        return
    print(
        f"[cohub] warning: imported {own}/source while running from"
        f" {caller}.\n"
        f"[cohub]   The shared .venv pins the editable install to that"
        f" checkout, so a bare\n"
        f"[cohub]   interpreter loads its code. Set"
        f" PYTHONPATH={caller}/source.",
        file=sys.stderr,
    )


_warn_if_foreign_checkout()
