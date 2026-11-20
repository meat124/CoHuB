"""MP4 encoding of a recorded dataset's camera frames, after the session.

This module only encodes. It touches no simulator or camera.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..keys import DEMO_VIDEOS

logger = logging.getLogger(__name__)


def video_stem(key: str) -> str:
    """The file stem camera ``key`` is written under: ``ego_a`` -> ``ego_robot_a``."""
    return Path(dict(DEMO_VIDEOS).get(key, f"{key}.mp4")).stem


def demo_images(demo_group):
    """The camera group of one demo, or ``None`` if it recorded none.

    Reads ``obs/images``, else the top-level ``images`` group.
    """
    images = demo_group.get("obs/images")
    return images if images is not None else demo_group.get("images")


def open_writer(path: Path, fps: float, quality: int = 6, scale: int = 1, preset: str | None = None):
    """An open MP4 writer at ``path``, with this benchmark's encoder settings.

    Callers must ``close()`` it. ``scale`` upsamples the file by an integer
    factor. ``preset`` is x264's speed/size preset (``None`` = ``medium``).
    """
    import imageio.v2 as imageio

    scale = int(scale)
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    output_params: list[str] = []
    if scale > 1:
        # Lanczos keeps the upscaled picture sharp.
        output_params += ["-vf", f"scale=iw*{scale}:ih*{scale}:flags=lanczos"]
    if preset:
        output_params += ["-preset", str(preset)]

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(path),
        fps=max(float(fps), 1.0),
        codec="libx264",
        quality=quality,
        # Do not resize frames to a macro-block multiple.
        macro_block_size=None,
        output_params=output_params,
    )


DEPTH_UNITS_PER_METRE = 1000.0
"""Depth units per metre: millimetres."""

DEPTH_INVALID = 0
"""The encoded value for a pixel with no return."""

DEPTH_MAX_METRES = 65535.0 / DEPTH_UNITS_PER_METRE


def open_lossless_writer(path: Path, fps: float):
    """An open lossless MP4 writer (``libx264rgb -qp 0``).

    RGB stays RGB, so a frame carrying packed numbers survives bit for bit.
    """
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(path),
        fps=max(float(fps), 1.0),
        codec="libx264rgb",
        pixelformat="rgb24",
        output_params=["-qp", "0"],
        macro_block_size=None,
    )


def pack_depth(frames, units_per_metre: float = DEPTH_UNITS_PER_METRE):
    """Metres -> the RGB bytes a lossless video carries them in::

        R = value >> 8,  G = value & 0xFF,  B = 0

    :func:`unpack_depth` reverses it exactly. Raises ``ValueError`` if a
    value is past :data:`DEPTH_MAX_METRES`.
    """
    import numpy as np

    metres = np.asarray(frames, dtype=np.float32)
    # Drop the recorder's trailing channel axis.
    if metres.ndim >= 3 and metres.shape[-1] == 1:
        metres = metres[..., 0]
    seen = np.isfinite(metres) & (metres > 0.0)
    if seen.any() and float(metres[seen].max()) > DEPTH_MAX_METRES:
        raise ValueError(
            f"depth reaches {float(metres[seen].max()):.1f} m, past the "
            f"{DEPTH_MAX_METRES:.1f} m that {units_per_metre:g} units/metre can encode"
        )

    value = np.where(seen, np.clip(np.rint(metres * units_per_metre), 1, 65535), DEPTH_INVALID).astype(np.uint16)
    return np.stack(
        [(value >> 8).astype(np.uint8), (value & 0xFF).astype(np.uint8), np.zeros(value.shape, np.uint8)],
        axis=-1,
    )


def unpack_depth(frames, units_per_metre: float = DEPTH_UNITS_PER_METRE, invalid: float = 0.0):
    """The RGB bytes back to metres. The inverse of :func:`pack_depth`.

    ``invalid`` is the value for pixels with no return (``float("inf")`` matches the HDF5).
    """
    import numpy as np

    rgb = np.asarray(frames)
    value = (rgb[..., 0].astype(np.uint16) << 8) | rgb[..., 1].astype(np.uint16)
    metres = value.astype(np.float32) / float(units_per_metre)
    return np.where(value == DEPTH_INVALID, np.float32(invalid), metres)


def encode(frames, path: Path, fps: float, quality: int = 6) -> int:
    """Write ``frames`` (``T x H x W x 3`` uint8) to ``path``. Returns the count."""
    import numpy as np

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] < 3 or len(frames) == 0:
        return 0

    writer = open_writer(path, fps=fps, quality=quality)
    try:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame[..., :3]).astype(np.uint8))
    finally:
        writer.close()
    return int(len(frames))


def encode_depth(frames, path: Path, fps: float, quality: int = 6, near=None, far=None) -> int:
    """Write a depth stream (``T x H x W`` metres) to ``path`` as grey video.

    For viewing only: the scale is the demo's own 1st-99th percentile range
    unless ``near``/``far`` are given, so two files are not comparable. Near is
    bright, far is dark, and pixels with no return are black. For metres out of
    a video use :func:`pack_depth`.
    """
    import numpy as np

    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim == 4 and frames.shape[-1] == 1:
        frames = frames[..., 0]
    if frames.ndim != 3 or len(frames) == 0:
        return 0

    seen = np.isfinite(frames) & (frames > 0.0)
    if not seen.any():
        return 0
    values = frames[seen]
    low = float(np.percentile(values, 1.0)) if near is None else float(near)
    high = float(np.percentile(values, 99.0)) if far is None else float(far)
    if high <= low:
        high = low + 1e-3

    writer = open_writer(path, fps=fps, quality=quality)
    try:
        for frame, valid in zip(frames, seen):
            grey = (1.0 - np.clip((frame - low) / (high - low), 0.0, 1.0)) * 255.0
            grey = grey.astype(np.uint8)
            grey[~valid] = 0
            writer.append_data(np.repeat(grey[..., None], 3, axis=-1))
    finally:
        writer.close()
    return int(len(frames))


def output_names(demo_group, depth: bool = False, suffix: str = "") -> list[str]:
    """The files :func:`render_demo` would write for this demo."""
    images = demo_images(demo_group)
    if images is None:
        return []
    names: list[str] = []
    for key, filename in DEMO_VIDEOS:
        camera = images.get(key)
        if camera is None:
            continue
        stem = Path(filename).stem
        if "rgb" in camera:
            names.append(f"{stem}{suffix}.mp4")
        if depth and "depth" in camera:
            names.append(f"{stem}_depth{suffix}.mp4")
    return names


def render_demo(
    demo_group, out_dir: Path, fps: float, quality: int = 6, depth: bool = False, suffix: str = ""
) -> dict[str, int]:
    """Encode every camera in one HDF5 demo group into ``out_dir``. Returns file name -> frame count.

    ``depth`` also encodes the depth stream as grey video; ``suffix`` is appended to every file name.
    """
    images = demo_images(demo_group)
    if images is None:
        return {}

    written: dict[str, int] = {}
    for key, filename in DEMO_VIDEOS:
        camera = images.get(key)
        if camera is None:
            continue
        stem = Path(filename).stem
        if "rgb" in camera:
            name = f"{stem}{suffix}.mp4"
            count = encode(camera["rgb"][()], out_dir / name, fps=fps, quality=quality)
            if count:
                written[name] = count
        if depth and "depth" in camera:
            name = f"{stem}_depth{suffix}.mp4"
            count = encode_depth(camera["depth"][()], out_dir / name, fps=fps, quality=quality)
            if count:
                written[name] = count
    return written


# -- What a trainer reads back out of these files --

def export_crf(quality: int = 6) -> int:
    """The libx264 CRF for imageio ``quality`` (20 for the benchmark's 6)."""
    return int((1 - quality / 10.0) * 51)


def as_decoded(frame, quality: int = 6, backend: str = "pyav", context: int = 3):
    """``frame`` (H, W, 3) uint8 RGB after an export-like round trip through
    libx264 at :func:`export_crf` and yuv420p.

    Approximates what a policy trained on the exported MP4s saw. The frame is
    encoded ``context`` times and the last copy returned. ``backend`` is
    ``"pyav"`` (in-process, fast) or ``"imageio"`` (the export's own ffmpeg).
    """
    import numpy as np

    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint8 RGB, got {frame.shape} {frame.dtype}")
    crf = export_crf(quality)
    if backend == "pyav":
        import io

        import av

        buf = io.BytesIO()
        with av.open(buf, "w", format="mp4") as out:
            stream = out.add_stream("libx264", rate=50)
            stream.width, stream.height = int(frame.shape[1]), int(frame.shape[0])
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf), "threads": "1"}
            src = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for _ in range(max(1, int(context))):
                for packet in stream.encode(src):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
        buf.seek(0)
        with av.open(buf) as inp:
            decoded = [f.to_ndarray(format="rgb24") for f in inp.decode(video=0)]
        return decoded[-1]
    if backend == "imageio":
        import tempfile

        import imageio.v2 as imageio

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.mp4"
            writer = open_writer(path, fps=50.0, quality=quality)
            for _ in range(max(1, int(context))):
                writer.append_data(frame)
            writer.close()
            reader = imageio.get_reader(str(path))
            try:
                decoded = np.asarray(reader.get_data(max(1, int(context)) - 1))
            finally:
                reader.close()
        return decoded
    raise ValueError(f"unknown backend {backend!r} (pyav, imageio)")
