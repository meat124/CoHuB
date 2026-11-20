"""Writing a session to disk: the HDF5 file handlers and the episode log."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import queue
import re
import threading

import h5py
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab.utils.datasets.hdf5_dataset_file_handler import DATASET_FORMAT_VERSION

from cohub.subtasks import read_checks, read_metrics


# --- One file, appended across restarts ---
# Existing demos stay, the next episode goes after the highest demo_N, and the file's env_args are
# preserved.

_DEMO_NAME = re.compile(r"^demo_(\d+)$")
_CORRUPT_HDF5_MARKERS = (
    "truncated file",
    "file signature not found",
    "bad object header",
    "bad byte number in an address",
    "address overflow",
    "addr overflow",
)


def _quarantine_path(file_path: str) -> str:
    """Return a unique, adjacent name for a damaged dataset."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = f"{file_path}.corrupt-{stamp}"
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{file_path}.corrupt-{stamp}-{suffix}"
        suffix += 1
    return candidate


def is_empty_episode(episode) -> bool:
    """Whether an episode holds no recorded step. Decided by its ``actions``.

    An episode that carries only metadata counts as empty.
    """
    if episode is None or getattr(episode, "is_empty", lambda: False)():
        return True
    actions = (getattr(episode, "data", None) or {}).get("actions")
    return actions is None or len(actions) == 0


def next_demo_index(data_group) -> int:
    """One past the highest ``demo_N`` in an open HDF5 ``data`` group."""
    highest = -1
    for name in data_group:
        match = _DEMO_NAME.match(name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


class ResumableHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    """Appends to an existing dataset instead of overwriting it.

    A drop-in for ``RecorderManagerBaseCfg.dataset_file_handler_class_type``.
    Behaves like its parent on a path that does not exist yet.
    """

    def create(self, file_path: str, env_name: str | None = None):
        # The caller passes a stem. Add the suffix as the parent does, so the
        # existence check tests the path that will be opened.
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"

        if not os.path.isfile(file_path):
            return super().create(file_path, env_name)

        # Opened directly: the parent's open() requires the "data" group.
        import h5py

        try:
            self._hdf5_file_stream = h5py.File(file_path, "a")
        except OSError as exc:
            # Move a corrupt file aside and start a new dataset. Lock and
            # permission failures stay hard errors.
            message = str(exc).lower()
            if not any(marker in message for marker in _CORRUPT_HDF5_MARKERS):
                raise
            quarantined = _quarantine_path(file_path)
            os.replace(file_path, quarantined)
            print(f"[dataset] damaged HDF5 preserved as {quarantined}; starting a new dataset")
            return super().create(file_path, env_name)

        if "data" not in self._hdf5_file_stream:
            self._hdf5_file_stream.attrs["format_version"] = DATASET_FORMAT_VERSION
            self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
            self._hdf5_data_group.attrs["total"] = 0
        else:
            self._hdf5_data_group = self._hdf5_file_stream["data"]
        self._restore_env_args(env_name)
        self._demo_count = next_demo_index(self._hdf5_data_group)
        print(
            f"[dataset] appending to {file_path}: {len(self._hdf5_data_group)} demo(s) already there,"
            f" next is demo_{self._demo_count}"
        )
        return None

    def _restore_env_args(self, env_name: str | None) -> None:
        """Carry the reopened file's ``env_args`` into this handler's copy.

        The parent rewrites the whole attribute from its instance dict on every
        ``add_env_args()``, so the dict is seeded from the file first.
        """
        stored = self._hdf5_data_group.attrs.get("env_args")
        if stored is not None:
            try:
                self._env_args.update(json.loads(stored))
            except (TypeError, ValueError):
                pass
        # The caller's name wins; the file's stands in when the caller passed None.
        self.add_env_args({"env_name": env_name or self._env_args.get("env_name", ""), "type": 2})

    def write_episode(self, episode, *args, **kwargs):
        """Write the episode, skipping empty ones (see :func:`is_empty_episode`)."""
        if is_empty_episode(episode):
            return None
        return super().write_episode(episode, *args, **kwargs)


# --- One file per demonstration ---
# A shard is built under .hdf5.partial, moved into place with os.replace, then verified. With
# async_write the write runs on a background thread.

PARTIAL_SUFFIX = ".partial"

_SHARD_NAME = re.compile(r"^.+\.demo_(?P<index>\d+)\.hdf5$")
"""``cocarry.demo_12.hdf5`` -> 12, for any prefix. The index is unique per directory."""


def shard_name(prefix: str, index: int) -> str:
    """The file name for one demonstration."""
    return f"{prefix}.demo_{index}.hdf5"


def next_shard_index(directory: str) -> int:
    """One past the highest ``*.demo_N.hdf5`` in ``directory``. Skips ``.partial`` files."""
    highest = -1
    if os.path.isdir(directory):
        for name in os.listdir(directory):
            match = _SHARD_NAME.match(name)
            if match:
                highest = max(highest, int(match.group("index")))
    return highest + 1


class ShardedHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    """Writes each demonstration to its own file.

    A drop-in for ``dataset_file_handler_class_type``. The path
    ``RecorderManager`` builds is read as directory + file-name prefix.
    """

    verify_shards = True
    """Reopen each finished shard read-only and check the demonstration is in it."""

    fsync_shards = True
    """Fsync each shard before the rename."""

    async_write = True
    """Write shards on a background thread instead of inside the recording loop."""

    def __init__(self):
        # RecorderManager constructs handlers with no arguments, so the
        # options above are class attributes.
        super().__init__()
        self._shard_dir: str = ""
        self._prefix: str = ""
        self._written: list[str] = []
        self._closed = False
        # Background writer state. ``_queued`` holds indices handed out but
        # not yet on disk.
        self._write_queue: queue.Queue | None = None
        self._writer: threading.Thread | None = None
        self._writer_error: BaseException | None = None
        self._queued: set[int] = set()
        # Guards ``_queued``, ``_writer_error`` and ``_written``, which the
        # writer thread shares with the record loop.
        self._state_lock = threading.RLock()

    # -- Operations --

    def create(self, file_path: str, env_name: str | None = None):
        """Take the directory and prefix. Open nothing."""
        # The path may arrive with or without the extension.
        if file_path.endswith(".hdf5"):
            file_path = file_path[: -len(".hdf5")]
        self._shard_dir, self._prefix = os.path.split(file_path)
        self._shard_dir = self._shard_dir or "."
        os.makedirs(self._shard_dir, exist_ok=True)

        self._restore_env_args(env_name)
        self._demo_count = next_shard_index(self._shard_dir)
        existing = len([n for n in os.listdir(self._shard_dir) if _SHARD_NAME.match(n)])
        print(
            f"[dataset] sharding into {self._shard_dir}/: {existing} shard(s) already there,"
            f" next is demo_{self._demo_count}"
        )
        return None

    def _restore_env_args(self, env_name: str | None) -> None:
        """Seed this handler's ``env_args`` from the newest shard already there."""
        newest = next_shard_index(self._shard_dir) - 1
        if newest >= 0:
            path = os.path.join(self._shard_dir, shard_name(self._prefix, newest))
            # The newest index may belong to another prefix; then there is
            # nothing to read.
            try:
                with h5py.File(path, "r") as handle:
                    stored = handle["data"].attrs.get("env_args")
                if stored is not None:
                    self._env_args.update(json.loads(stored))
            except (OSError, KeyError, TypeError, ValueError):
                pass
        self.add_env_args({"env_name": env_name or self._env_args.get("env_name", ""), "type": 2})

    def add_env_args(self, env_args: dict):
        """Accumulate, and write through only when a shard is open.

        The args are stamped onto each shard as it is created.
        """
        self._env_args.update(env_args)
        if self._hdf5_data_group is not None:
            self._hdf5_data_group.attrs["env_args"] = json.dumps(self._env_args)

    def get_env_name(self) -> str | None:
        """The task name, from the buffer rather than from an open file."""
        return self._env_args.get("env_name") or None

    def write_episode(self, episode, demo_id=None, dataset_compression: bool = True):
        """Hand one demonstration to the writer; return before the bytes land.

        With ``async_write = False`` the shard is written here, synchronously.
        Returns ``None``; the path appears in :attr:`written` once it is on disk.
        """
        # A failed background write stops the session at the next export.
        with self._state_lock:
            error, self._writer_error = self._writer_error, None
        if error is not None:
            raise RuntimeError("a background shard write failed; not recording over it") from error

        # Checked first, so an empty episode creates no file.
        if is_empty_episode(episode):
            return None

        index = self._demo_count if demo_id is None else int(demo_id)
        target = os.path.join(self._shard_dir, shard_name(self._prefix, index))
        # One snapshot for the whole scan.
        with self._state_lock:
            queued = frozenset(self._queued)
        # A queued index is as taken as one on disk.
        if os.path.exists(target) or index in queued:
            if demo_id is not None:
                # An explicit id must not collide.
                raise RuntimeError(f"{target} already exists")
            # Otherwise take the next free index and report it.
            index = next_shard_index(self._shard_dir)
            if queued:
                index = max(index, max(queued) + 1)
            target = os.path.join(self._shard_dir, shard_name(self._prefix, index))
            print(f"[dataset] demo_{self._demo_count} was taken; writing demo_{index}")

        # The record scripts read the count right after export.
        self._demo_count = max(self._demo_count, index + 1)
        # Snapshot: add_env_args may mutate the dict before the writer runs.
        env_args_json = json.dumps(self._env_args)

        if not self.async_write:
            self._write_shard(episode, index, target, env_args_json, dataset_compression)
            return None

        if self._writer is None:
            self._write_queue = queue.Queue()
            self._writer = threading.Thread(target=self._writer_loop, name="shard-writer", daemon=True)
            self._writer.start()
        # Added before the put, so the next scan sees it.
        with self._state_lock:
            self._queued.add(index)
        self._write_queue.put((episode, index, target, env_args_json, dataset_compression))
        print(f"[dataset] demo_{index} queued for background write")
        return None

    def _writer_loop(self) -> None:
        while True:
            job = self._write_queue.get()
            if job is None:
                break
            episode, index, target, env_args_json, dataset_compression = job
            try:
                self._write_shard(episode, index, target, env_args_json, dataset_compression)
            except BaseException as error:  # noqa: BLE001
                with self._state_lock:
                    # The first failure wins.
                    if self._writer_error is None:
                        self._writer_error = error
                print(f"[dataset] WRITE FAILED demo_{index}: {error!r} -- the next export will refuse")
            finally:
                with self._state_lock:
                    self._queued.discard(index)
                self._write_queue.task_done()

    def _write_shard(self, episode, index: int, target: str, env_args_json: str, dataset_compression: bool) -> None:
        """Write one shard to disk, on whichever thread calls it.

        Uses local h5py handles only and duplicates the parent's write loop:
        the inherited ``_hdf5_file_stream`` belongs to the main thread.
        """
        partial = target + PARTIAL_SUFFIX
        stream = h5py.File(partial, "w")
        try:
            stream.attrs["format_version"] = DATASET_FORMAT_VERSION
            data_group = stream.create_group("data")
            data_group.attrs["total"] = 0
            data_group.attrs["env_args"] = env_args_json

            episode_group = data_group.create_group(f"demo_{index}")
            samples = len(episode.data["actions"]) if "actions" in episode.data else 0
            episode_group.attrs["num_samples"] = samples
            if episode.seed is not None:
                episode_group.attrs["seed"] = episode.seed
            if episode.success is not None:
                episode_group.attrs["success"] = episode.success

            def create_dataset_helper(group, key, value):
                if isinstance(value, dict):
                    key_group = group.create_group(key)
                    for sub_key, sub_value in value.items():
                        create_dataset_helper(key_group, sub_key, sub_value)
                elif dataset_compression:
                    group.create_dataset(key, data=value.cpu().numpy(), compression="gzip", compression_opts=2)
                else:
                    group.create_dataset(key, data=value.cpu().numpy())

            for key, value in episode.data.items():
                create_dataset_helper(episode_group, key, value)
            data_group.attrs["total"] = int(data_group.attrs["total"]) + samples

            stream.flush()
            if self.fsync_shards:
                try:
                    os.fsync(stream.id.get_vfd_handle())
                except (OSError, AttributeError, ValueError):
                    # Only the default (sec2) driver exposes a file descriptor.
                    pass
        finally:
            stream.close()

        os.replace(partial, target)
        self._sync_directory()
        with self._state_lock:
            self._written.append(target)

        size = os.path.getsize(target) / (1024 * 1024)
        print(f"[dataset] demo_{index} -> {os.path.basename(target)}  ({samples} samples, {size:.1f} MB)")
        if self.verify_shards:
            self._verify(target, index, samples)

    def _sync_directory(self) -> None:
        """Fsync the directory so the rename itself is durable."""
        if not self.fsync_shards:
            return
        try:
            fd = os.open(self._shard_dir, os.O_RDONLY)
        except OSError:
            return  # not every filesystem lets a directory be opened
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _verify(self, path: str, index: int, samples: int) -> None:
        """Reopen a finished shard and check the demonstration is in it."""
        try:
            with h5py.File(path, "r") as handle:
                demo = handle["data"][f"demo_{index}"]
                stored = int(demo.attrs["num_samples"])
            if stored != samples:
                print(f"[dataset] WARNING {os.path.basename(path)}: {stored} samples on disk, expected {samples}")
        except (OSError, KeyError) as error:
            print(f"[dataset] WARNING {os.path.basename(path)} did not reopen cleanly: {error}")

    def flush(self):
        """Flush the open shard, if any."""
        if self._hdf5_file_stream is not None:
            self._hdf5_file_stream.flush()

    def close(self, timeout: float = 600.0):
        """End the session: drain the background writer; an interrupted shard stays ``.partial``.

        Safe to call more than once and from ``__del__``.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        writer = getattr(self, "_writer", None)
        if writer is not None and writer.is_alive():
            pending = self._write_queue.qsize()
            if pending:
                print(f"[save] finishing {pending} pending demonstration(s)...")
            self._write_queue.put(None)
            writer.join(timeout=timeout)
            if writer.is_alive():
                print(f"[dataset] WARNING: shard writer still running after {timeout:.0f} s; a .partial may remain")
        error = getattr(self, "_writer_error", None)
        if error is not None:
            # close() must not raise, so the failure is printed.
            print(f"[dataset] WARNING: a background shard write failed this session: {error!r}")
        stream = getattr(self, "_hdf5_file_stream", None)
        if stream is not None:
            # Only reachable if a write was interrupted. The .partial file is kept.
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._hdf5_file_stream = None
            self._hdf5_data_group = None
        written = getattr(self, "_written", None)
        if written:
            print(f"[dataset] {len(written)} shard(s) written to {self._shard_dir}/")

    # -- Properties --

    @property
    def written(self) -> list[str]:
        """Paths of the shards this session wrote, in the order written. Returns a copy."""
        with self._state_lock:
            return list(self._written)

    # -- Reading back --

    def open(self, file_path: str, mode: str = "r"):
        """Open one shard. Raises ``ValueError`` for a directory.

        To read a whole directory use :class:`cohub.recording.demo_source.DemoSource`.
        """
        if os.path.isdir(file_path):
            raise ValueError(
                f"{file_path} is a directory of shards; open one shard, or use cohub.recording.demo_source.DemoSource"
            )
        return super().open(file_path, mode)

    def load_episode(self, episode_name: str, device: str, *args, **kwargs):
        """One demonstration by name, from whichever shard holds it.

        ``_demo_count`` is restored afterwards: it means the next index to write.
        """
        match = re.match(r"^demo_(\d+)$", episode_name)
        if match is None:
            return None
        path = os.path.join(self._shard_dir, shard_name(self._prefix, int(match.group(1))))
        if not os.path.isfile(path):
            return None
        count = self._demo_count
        try:
            super().open(path, "r")
            return super().load_episode(episode_name, device, *args, **kwargs)
        finally:
            if self._hdf5_file_stream is not None:
                self._hdf5_file_stream.close()
            self._hdf5_file_stream = None
            self._hdf5_data_group = None
            self._demo_count = count


# --- episodes.txt ---
# One entry per episode, flushed as it happens: which predicate ended it and which of its published
# checks held.

_RULE = "=" * 78


def check_key(term_cfg) -> str:
    """The :mod:`cohub.subtasks` key a term's predicate publishes under.

    The function's name, qualified by the asset when the term names one.
    """
    func = getattr(term_cfg, "func", None)
    if func is None:
        return ""
    key = getattr(func, "checks_key", func.__name__)
    params = getattr(term_cfg, "params", None) or {}
    asset = getattr(params.get("asset_cfg"), "name", None)
    return f"{key}[{asset}]" if asset else key


def _source(term_cfg) -> str:
    func = getattr(term_cfg, "func", None)
    if func is None:
        return "unknown"
    return f"{getattr(func, '__module__', '?')}:{getattr(func, '__name__', '?')}"


def _headline(term_cfg) -> str:
    """The predicate's first docstring line, as its plain-language reason."""
    doc = (getattr(getattr(term_cfg, "func", None), "__doc__", "") or "").strip()
    return doc.splitlines()[0].strip() if doc else "(no description on the predicate)"


def _params(term_cfg) -> str:
    """The term's configured thresholds, as the operator would want them read."""
    params = getattr(term_cfg, "params", None) or {}
    parts = []
    for name, value in params.items():
        shown = getattr(value, "name", None) if hasattr(value, "name") else None
        parts.append(f"{name}={shown if shown is not None else value!r}")
    return "  ".join(parts)


def _checks_line(breakdown: dict[str, bool]) -> str:
    if not breakdown:
        return "(this predicate publishes no breakdown)"
    return "  ".join(f"{name}={'yes' if ok else 'NO'}" for name, ok in breakdown.items())


def _metrics_line(metrics: dict[str, float]) -> str:
    if not metrics:
        return ""
    return "  ".join(f"{name}={value:+.4f}" for name, value in metrics.items())


class EpisodeLog:
    """Appends one entry per episode to a text file next to the dataset."""

    def __init__(self, path: str, task: str):
        self.path = os.path.abspath(path)
        self._task = task
        self._counts: dict[str, int] = {}

    # -- writing ---------------------------------------------------------
    def _write(self, text: str) -> None:
        """Append and flush, per entry."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as out:
            out.write(text)

    def open_session(self, env, success_term, *, dataset: str, export_mode: str, num_success_steps: int) -> None:
        """Write the session header, including what this task ends on.

        The conditions are read off the live managers.
        """
        lines = [
            "",
            _RULE,
            f"session   {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}",
            f"task      {self._task}",
            f"dataset   {dataset}",
            f"export    {export_mode}",
            _RULE,
            "",
            "How this task ends",
            "",
        ]
        if success_term is not None:
            lines += [
                f"  SUCCESS  {_source(success_term)}",
                f"      {_headline(success_term)}",
                f"      held for {num_success_steps} consecutive steps before the demo is exported",
            ]
            if _params(success_term):
                lines.append(f"      {_params(success_term)}")
            lines.append("")
        manager = getattr(env, "termination_manager", None)
        for name in getattr(manager, "active_terms", []):
            term_cfg = manager.get_term_cfg(name)
            lines += [
                f"  RESET    {name}  ({_source(term_cfg)})",
                f"      {_headline(term_cfg)}",
            ]
            if _params(term_cfg):
                lines.append(f"      {_params(term_cfg)}")
            lines.append("")
        lines += [
            "  RESET    manual",
            "      An operator asked for a reset, or the session ended mid-episode.",
            "",
            _RULE,
            "",
        ]
        self._write("\n".join(lines))

    def record(
        self,
        env,
        *,
        episode_id: int,
        outcome: str,
        note: str,
        steps: int,
        fired: list[str] | tuple[str, ...] = (),
        success_term=None,
    ) -> None:
        """Write one episode's ending.

        ``outcome`` is ``SUCCESS``, ``RESET`` or ``ABANDONED``; ``note`` the demo name
        when exported; ``fired`` the termination terms that were true.
        """
        self._counts[outcome] = self._counts.get(outcome, 0) + 1
        seconds = steps * float(getattr(env, "step_dt", 0.0))
        lines = [f"episode {episode_id}  {outcome}  {note}  ({steps} steps, {seconds:.1f} s)"]

        manager = getattr(env, "termination_manager", None)
        for name in fired:
            term_cfg = manager.get_term_cfg(name)
            lines.append(f"    ended by  {name}  ({_source(term_cfg)})")
            lines.append(f"              {_headline(term_cfg)}")
            if _params(term_cfg):
                lines.append(f"              {_params(term_cfg)}")
            breakdown = read_checks(env, check_key(term_cfg))
            if breakdown:
                lines.append(f"              {_checks_line(breakdown)}")
            metrics = read_metrics(env, check_key(term_cfg))
            if metrics:
                lines.append(f"              measured: {_metrics_line(metrics)}")

        if success_term is not None:
            breakdown = read_checks(env, check_key(success_term))
            # On a failure this shows how far the attempt had got.
            label = "    passed    " if outcome == "SUCCESS" else "    progress  "
            lines.append(f"{label}{success_term.func.__name__}: {_checks_line(breakdown)}")
            metrics = read_metrics(env, check_key(success_term))
            if metrics:
                lines.append(f"              measured: {_metrics_line(metrics)}")
            if _params(success_term):
                lines.append(f"              limits:   {_params(success_term)}")

        self._write("\n".join(lines) + "\n\n")

    def close(self, demos: int) -> None:
        tally = "  ".join(f"{name} {count}" for name, count in sorted(self._counts.items()))
        self._write(f"session ended: {demos} demo(s) exported   [{tally or 'no episodes'}]\n{_RULE}\n")
