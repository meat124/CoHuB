"""Reading a recorded dataset offline, and the provenance stamped in it."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import pathlib
import re
import subprocess


# --- Dataset readers ---

_SHARD_NAME = re.compile(r"^.+\.demo_(?P<index>\d+)\.hdf5$")
_DEMO_NAME = re.compile(r"^demo_(\d+)$")

DATA_DIR = "data"
"""Shard directory under the dataset root."""

DEFAULT_FPS = 30.0
"""Frame rate to assume when a dataset says nothing about its own rate."""

VIDEO_DIR = "videos"
"""Video directory of a sharded dataset. Single-file datasets use the root. See :func:`video_dir`."""


def _shard_dir(path: pathlib.Path) -> pathlib.Path | None:
    """The shard directory for ``path`` (itself or its ``data/`` child), or ``None``."""
    if path.is_file():
        return None
    if path.is_dir():
        if any(_SHARD_NAME.match(child.name) for child in path.iterdir()):
            return path
        nested = path / DATA_DIR
        if nested.is_dir() and any(_SHARD_NAME.match(child.name) for child in nested.iterdir()):
            return nested
    return None


def shard_files(dataset_path) -> list[pathlib.Path]:
    """The shards of a sharded dataset in demo order, or ``[]`` for a file."""
    directory = _shard_dir(pathlib.Path(dataset_path).expanduser())
    if directory is None:
        return []
    shards = [child for child in directory.iterdir() if _SHARD_NAME.match(child.name)]
    return sorted(shards, key=lambda p: int(_SHARD_NAME.match(p.name).group("index")))


def dataset_root(dataset_path) -> pathlib.Path:
    """The dataset root directory.

    ``datasets/cocarry`` for both ``datasets/cocarry/cocarry.hdf5`` and
    ``datasets/cocarry/data/cocarry.demo_0.hdf5``.
    """
    path = pathlib.Path(dataset_path).expanduser()
    directory = _shard_dir(path)
    if directory is not None:
        return directory.parent if directory.name == DATA_DIR else directory
    if path.is_file() and _SHARD_NAME.match(path.name):
        return path.parent.parent if path.parent.name == DATA_DIR else path.parent
    return path.parent if path.is_file() else path


def video_dir(dataset_path, demo_name: str) -> pathlib.Path:
    """Where one demonstration's MP4s go.

    ``<root>/videos/demo_N`` for a sharded dataset, ``<root>/demo_N`` for a
    single-file one.
    """
    root = dataset_root(dataset_path)
    path = pathlib.Path(dataset_path).expanduser()
    sharded = _shard_dir(path) is not None or bool(_SHARD_NAME.match(path.name))
    return (root / VIDEO_DIR / demo_name) if sharded else (root / demo_name)


class DemoSource:
    """Context manager over the demonstrations of a single-file or sharded dataset::

        with DemoSource(path) as demos:
            for name in demos.names:
                group = demos[name]

    ``names`` is in demo-index order. ``demos[name]`` is an ``h5py.Group``.
    """

    def __init__(self, dataset_path):
        self.path = pathlib.Path(dataset_path).expanduser()
        self._streams: list = []
        self._groups: dict = {}
        self.attrs: dict = {}
        """The ``data`` group attributes, e.g. ``env_args``."""
        self.problems: list[str] = []
        """Shard naming faults found while opening. See :meth:`_check_shard_name`."""

    def __enter__(self) -> DemoSource:
        # Close the files opened so far if opening fails part-way.
        try:
            return self._open()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _open(self) -> DemoSource:
        import h5py

        shards = shard_files(self.path)
        if shards:
            # env_args comes from the lowest-indexed shard. The rest must agree.
            names: list[str] = []
            for shard in shards:
                stream = h5py.File(shard, "r")
                self._streams.append(stream)
                data = stream["data"]
                if not self.attrs:
                    self.attrs = dict(data.attrs)
                elif _env_name(dict(data.attrs)) != _env_name(self.attrs):
                    print(
                        f"WARNING: {shard.name} was recorded with "
                        f"{_env_name(dict(data.attrs))!r}, not {_env_name(self.attrs)!r}"
                    )
                for name in data:
                    if _DEMO_NAME.match(name):
                        self._check_shard_name(shard, name)
                        self._groups[name] = data[name]
                        names.append(name)
            self.names = sorted(names, key=lambda n: int(n.split("_")[1]))
            for problem in self.problems:
                print(f"WARNING: {problem}")
            return self

        if not self.path.is_file():
            raise SystemExit(f"{self.path} is neither a dataset file nor a directory of shards")
        stream = h5py.File(self.path, "r")
        self._streams.append(stream)
        data = stream.get("data")
        if data is None:
            raise SystemExit(f"{self.path} has no 'data' group")
        self.attrs = dict(data.attrs)
        self._groups = {name: data[name] for name in data if _DEMO_NAME.match(name)}
        self.names = sorted(self._groups, key=lambda n: int(n.split("_")[1]))
        return self

    def _check_shard_name(self, shard: pathlib.Path, name: str) -> None:
        """Record a shard whose file index differs from its group name, or a duplicate group."""
        if name in self._groups:
            self.problems.append(
                f"{shard.name} holds {name}, which another shard in this directory also holds -- "
                "one of the two is shadowed and cannot be read"
            )
        match = _SHARD_NAME.match(shard.name)
        if match and name != f"demo_{int(match.group('index'))}":
            self.problems.append(
                f"{shard.name} holds a group named {name} -- renaming a shard does not rename the "
                "group inside it. To drop a demonstration, delete its shard and leave the gap."
            )

    def __exit__(self, *exc) -> None:
        for stream in self._streams:
            stream.close()
        self._streams.clear()
        self._groups.clear()

    def __getitem__(self, name: str):
        return self._groups[name]

    def __contains__(self, name: str) -> bool:
        return name in self._groups

    def __len__(self) -> int:
        return len(self._groups)

    def __iter__(self):
        return iter(self.names)


def _env_name(attrs: dict) -> str:
    """The gym env name inside a ``data`` group's attributes, or ``""``."""
    env_args = attrs.get("env_args")
    if env_args is None:
        return ""
    try:
        return str(json.loads(env_args).get("env_name") or "")
    except (TypeError, ValueError):
        return ""


def dataset_env_name(dataset_path) -> str:
    """The gym env name a dataset (an ``.hdf5`` or a shard directory) was recorded with.

    Exits if the dataset carries no ``env_args``.
    """
    with DemoSource(dataset_path) as demos:
        attrs = demos.attrs
    if "env_args" not in attrs:
        raise SystemExit(
            f"{dataset_path} does not say which task recorded it (no env_args); pass --task explicitly."
        )
    name = _env_name(attrs)
    if not name:
        raise SystemExit(f"{dataset_path}: env_args carries no env_name; pass --task explicitly.")
    return name


def _median_rate(demo_group, path: str) -> float:
    """Steps per second from the median step of a recorded clock, or 0.0 if it has none."""
    import numpy as np

    stamps = demo_group.get(path) if demo_group is not None else None
    if stamps is None or len(stamps) < 2:
        return 0.0
    steps = np.diff(np.asarray(stamps[()], dtype=float).reshape(-1))
    steps = steps[steps > 0]
    return 1.0 / float(np.median(steps)) if len(steps) else 0.0


def demo_fps(demo_group, attrs: dict, clock: str = "sim") -> tuple[float, str]:
    """The playback rate of one demonstration and where it came from: ``(fps, source)``.

    ``clock="sim"`` reads ``sync/sim_time``, ``"wall"`` reads ``sync/step_wall_time``;
    the fallbacks are ``env_args`` and :data:`DEFAULT_FPS`.
    """
    if clock == "wall":
        rate = _median_rate(demo_group, "sync/step_wall_time")
        if rate:
            return rate, "sync/step_wall_time -- the session's own pace"
        # No wall stamps: fall back to sim time.

    rate = _median_rate(demo_group, "sync/sim_time")
    if rate:
        return rate, "sync/sim_time"

    try:
        sim_args = json.loads(attrs.get("env_args") or "{}").get("sim_args") or {}
    except (TypeError, ValueError):
        sim_args = {}
    dt = float(sim_args.get("dt") or 0.0)
    decimation = int(sim_args.get("decimation") or 1)
    if dt > 0.0 and decimation > 0:
        return 1.0 / (dt * decimation), "env_args sim_args"

    return DEFAULT_FPS, "guessed -- the dataset states no rate"


def parse_selection(text: str) -> set[int] | None:
    """``"3,7-9"`` -> ``{3, 7, 8, 9}``. Empty or blank returns ``None`` (everything)."""
    if not text or not text.strip():
        return None
    chosen: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            chosen.update(range(int(start), int(end) + 1))
        else:
            chosen.add(int(part))
    return chosen


def human(size: float) -> str:
    """Format a byte count, e.g. 12.3 MB."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable: the loop returns at GB")


# --- Provenance ---
# Written into data.attrs["env_args"]["provenance"] by every tool that creates or rewrites a dataset.

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the checkout


def _git(*args: str) -> str:
    """A git command's output, or "" if git or the repository is unavailable. Never raises."""
    try:
        return subprocess.run(
            ["git", "-C", _REPO, *args],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def scene_assets(env) -> list[str]:
    """Every asset and sensor name in the environment's scene, sorted."""
    scene = getattr(env, "scene", None)
    if scene is None:
        return []
    names: set[str] = set()
    for group in ("articulations", "rigid_objects", "sensors", "extras"):
        names.update(getattr(scene, group, {}) or {})
    cfg = getattr(scene, "cfg", None)
    if cfg is not None:
        names.update(
            key for key, value in vars(cfg).items() if not key.startswith("_") and value is not None
        )
    return sorted(names)


def render_settings(env=None) -> dict:
    """The render state behind these pixels, as far as it can be read here.

    Records the ``COHUB_*`` render overrides, the render cfg's modes and the
    ``rtx`` keys carb reports (``runtime.RENDER_SETTING_KEYS``). Best-effort:
    works without ``env`` and without carb.
    """
    import os

    block = {
        key: os.environ[key]
        for key in ("COHUB_AA", "COHUB_RENDER_MODE", "COHUB_RENDER_SPP", "COHUB_EVAL_SPP")
        if key in os.environ
    }
    render_cfg = getattr(getattr(getattr(env, "cfg", None), "sim", None), "render", None)
    if render_cfg is not None:
        for field in ("antialiasing_mode", "rendering_mode", "ambient_light_intensity"):
            value = getattr(render_cfg, field, None)
            if value is not None:
                block[field] = value
    try:
        from cohub.runtime import render_settings as carb_render_settings

        readback = carb_render_settings()
        if any(value is not None for value in readback.values()):
            block["carb"] = readback
    except Exception:  # noqa: BLE001
        pass
    return block


def provenance(env=None, written_by: str = "", hidden_heads: list[str] | None = None) -> dict:
    """The provenance block, ready to hand to ``add_env_args``.

    ``written_by`` names the script. ``hidden_heads`` lists robots whose head was
    invisible while the frames were rendered; it is written even when empty.
    """
    commit = _git("rev-parse", "HEAD")
    block = {
        "written_by": written_by,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        # A dirty tree means the commit does not describe the code that ran.
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")) if commit else False,
    }
    if hidden_heads is not None:
        block["hidden_heads"] = list(hidden_heads)
    if env is not None:
        block["scene_assets"] = scene_assets(env)
    render = render_settings(env)
    if render:
        block["render"] = render
    return block


def hidden_heads_in(attrs: dict) -> list[str] | None:
    """Robots recorded with their head hidden, or ``None`` if the file is silent.

    ``[]`` means nothing was hidden. ``None`` means the file does not carry
    the field.
    """
    import json

    env_args = attrs.get("env_args")
    if env_args is None:
        return None
    try:
        block = json.loads(env_args).get("provenance") or {}
    except (TypeError, ValueError):
        return None
    heads = block.get("hidden_heads")
    return list(heads) if heads is not None else None


def restamp(env_args: dict, env=None, written_by: str = "", hidden_heads: list[str] | None = None) -> dict:
    """Re-stamp a rewritten dataset without losing who collected it. Returns a new dict.

    The collection session's stamp is kept under ``provenance/collected``;
    ``provenance`` describes the last tool to write the file.
    """
    stamped = dict(env_args)
    previous = stamped.get("provenance") or {}
    fresh = provenance(env, written_by, hidden_heads=hidden_heads)
    # An existing ``collected`` block wins over the previous tool's own stamp.
    collected = previous.get("collected") or (previous or None)
    if collected is not None:
        # Do not nest the chain.
        collected = {k: v for k, v in collected.items() if k != "collected"}
        fresh["collected"] = collected
    stamped["provenance"] = fresh
    return stamped


def collected_stamp(attrs: dict) -> dict:
    """The provenance of the collection session, or ``{}`` if the file has none."""
    try:
        block = json.loads(attrs.get("env_args") or "{}").get("provenance") or {}
    except (TypeError, ValueError):
        return {}
    return block.get("collected") or block


def articulation_index(env) -> dict:
    """Joint and body names per articulation, so the joint vectors can be read."""
    scene = getattr(env, "scene", None)
    if scene is None:
        return {}
    index = {}
    for name, asset in (getattr(scene, "articulations", {}) or {}).items():
        data = getattr(asset, "data", None)
        entry = {}
        for field in ("joint_names", "body_names"):
            names = getattr(asset, field, None) or getattr(data, field, None)
            if names is not None:
                entry[field] = [str(n) for n in names]
        if entry:
            index[name] = entry
    return index


def stamp_env_args(handler, env=None, written_by: str = "", hidden_heads: list[str] | None = None) -> None:
    """Stamp provenance and the articulation index onto an open dataset handler.

    Does nothing if the handler has no ``add_env_args``. Never raises.
    """
    add = getattr(handler, "add_env_args", None)
    if add is None:
        return
    try:
        add(
            {
                "provenance": provenance(env, written_by, hidden_heads),
                "articulation": articulation_index(env),
            }
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[dataset] could not stamp provenance: {exc}")
