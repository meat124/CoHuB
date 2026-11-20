"""Process-level helpers for an entry point: render settings, exit status, rate reports, scene state writes."""

from __future__ import annotations

from collections.abc import Sequence
import os
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# --- Runtime and render settings ---

def configure_cpu_runtime() -> bool:
    """Make IsaacLab/Warp CPU runs work when no CUDA driver is installed.

    Returns:
        ``True`` when the CPU fallback was enabled, otherwise ``False``.
    """

    try:
        import warp as wp
    except ImportError:
        return False

    if wp.is_cuda_available():
        return False

    cpu = wp.get_device("cpu")
    cpu.pinned_allocator = cpu.default_allocator
    return True


RENDER_ANTIALIASING = "DLAA"
"""Anti-aliasing for a live session. DLAA keeps the cameras' native resolution."""


def configure_render(render_cfg) -> None:
    """Set up the RTX renderer on ``render_cfg`` in place. Call before ``gym.make``.

    ``COHUB_AA`` (DLSS|DLAA|TAA|FXAA|Off) and ``COHUB_RENDER_MODE``
    (performance|balanced|quality) override it; an AppLauncher ``--rendering_mode`` wins.
    """
    import os

    render_cfg.antialiasing_mode = os.environ.get("COHUB_AA", RENDER_ANTIALIASING)
    render_cfg.ambient_light_intensity = 0.0
    render_cfg.enable_translucency = True
    render_cfg.enable_reflections = True
    # These two carb settings do not reach the cameras; pin_render_products() does that.
    render_cfg.carb_settings = {
        **(render_cfg.carb_settings or {}),
        "rtx.rendermode": "RaytracedLighting",
        "rtx.directLighting.sampledLighting.enabled": False,
    }
    mode = os.environ.get("COHUB_RENDER_MODE")
    if mode:
        render_cfg.rendering_mode = mode
    print(f"[render] antialiasing={render_cfg.antialiasing_mode} mode={render_cfg.rendering_mode}")


RENDER_SPP = 16
"""Path-traced samples per pixel for every camera this benchmark records.
The OptiX denoiser only loads if ``/usr/share/nvidia/nvoptix.bin`` is present."""

RENDER_SETTING_KEYS = (
    "/rtx/rendermode",
    "/rtx/pathtracing/totalSpp",
    "/rtx/pathtracing/denoiser/enabled",
    "/rtx/directLighting/sampledLighting/enabled",
    "/rtx/sceneDb/ambientLightIntensity",
    "/rtx/post/aa/op",
    "/rtx/translucency/enabled",
    "/rtx/reflections/enabled",
)
"""The carb keys a frame's look depends on, read back by :func:`render_settings`."""


def pin_render_products(spp: int | None = None) -> int:
    """Path-trace the camera sensors. Call right after ``gym.make``. Returns the spp pinned.

    ``spp`` defaults to ``COHUB_RENDER_SPP``, else :data:`RENDER_SPP`; zero or less skips the pin.
    """
    import os

    if spp is None:
        spp = int(os.environ.get("COHUB_RENDER_SPP", RENDER_SPP))
    if spp <= 0:
        print("[render] render-product pin SKIPPED -- frames are on the Kit profile's own mode")
        return 0

    import omni.replicator.core as rep

    rep.settings.set_render_pathtraced(samples_per_pixel=spp)
    print(f"[render] render products: path-traced, {spp} samples/pixel")
    return spp


def render_settings() -> dict:
    """What the renderer is set to right now, read from carb."""
    try:
        import carb
    except ImportError:
        return {}
    settings = carb.settings.get_settings()
    return {key: settings.get(key) for key in RENDER_SETTING_KEYS}


# --- Exit status ---
# SimulationApp.close() never returns and always exits 0, so a check that launched Kit ends through
# exit_gate.

def exit_gate(failures) -> "None":
    """Exit 1 if ``failures`` (count, list or bool) is truthy, else 0."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failures else 0)


def fail_gate_on_exception() -> "None":
    """Make an unhandled exception exit 1 through :func:`exit_gate`.

    Call once, right after importing this module.
    """
    import traceback

    def hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
        print(f"\nFAILURES:\n  - the check did not finish: {exc_type.__name__}: {exc}")
        exit_gate(True)

    sys.excepthook = hook


# --- Step time split ---
# Assumes sim.step(render=False) plus explicit sim.render() calls.

class StepSplit:
    """Accumulating physics/render timers over a ``SimulationContext``.

    Usage::

        split = StepSplit(env.sim)
        before = split.snapshot()
        env.step(action)
        physics_s, render_s = split.since(before)
    """

    def __init__(self, sim):
        self.physics = 0.0
        self.render = 0.0
        orig_step = sim.step
        orig_render = sim.render

        def timed_step(*args, **kwargs):
            started = time.perf_counter()
            try:
                return orig_step(*args, **kwargs)
            finally:
                self.physics += time.perf_counter() - started

        def timed_render(*args, **kwargs):
            started = time.perf_counter()
            try:
                return orig_render(*args, **kwargs)
            finally:
                self.render += time.perf_counter() - started

        sim.step = timed_step
        sim.render = timed_render

    def snapshot(self) -> tuple[float, float]:
        """Accumulated ``(physics_s, render_s)`` so far."""
        return self.physics, self.render

    def since(self, before: tuple[float, float]) -> tuple[float, float]:
        """``(physics_s, render_s)`` spent since ``before = snapshot()``."""
        return self.physics - before[0], self.render - before[1]


# --- GPU and CPU load ---

def _query_nvidia_smi() -> tuple[int, int] | None:
    """One synchronous ``nvidia-smi`` read; ``None`` if it cannot be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        used, total = (int(v.strip()) for v in out.stdout.splitlines()[0].split(","))
    except (ValueError, IndexError):
        return None
    return used, total


class _GpuMemoryProbe:
    """The last completed VRAM reading, refreshed on a background thread.

    ``sample()`` returns the cached reading and starts a refresh when none is
    in flight. The first call returns ``None``.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._value: tuple[int, int] | None = None
        self._inflight = False

    def sample(self) -> tuple[int, int] | None:
        with self._lock:
            value = self._value
            start = not self._inflight
            if start:
                self._inflight = True
        if start:
            threading.Thread(target=self._refresh, name="gpu-memory-probe", daemon=True).start()
        return value

    def _refresh(self) -> None:
        value = _query_nvidia_smi()
        with self._lock:
            if value is not None:
                self._value = value
            self._inflight = False


_GPU_MEMORY = _GpuMemoryProbe()


def gpu_memory_mb() -> tuple[int, int] | None:
    """``(used, total)`` VRAM in MiB for the whole card, or ``None``.

    The reading is the last one the background probe completed.
    """
    return _GPU_MEMORY.sample()


class CpuLoad:
    """How many cores' worth of CPU this process is using.

    ``time.process_time()`` over elapsed wall time. A value near 1.0 at a low
    rate points to a single-threaded stage.
    """

    def __init__(self):
        self._cpu = time.process_time()
        self._wall = time.perf_counter()

    def sample(self) -> float | None:
        """Average cores used since the last call, or ``None`` on the first."""
        cpu, wall = time.process_time(), time.perf_counter()
        elapsed = wall - self._wall
        used = cpu - self._cpu
        self._cpu, self._wall = cpu, wall
        if elapsed <= 0:
            return None
        return used / elapsed


def describe_resources(cpu: CpuLoad | None = None) -> str:
    """One compact line for a rate report."""
    parts = []
    memory = gpu_memory_mb()
    if memory is not None:
        used, total = memory
        parts.append(f"vram {used}/{total} MiB")
    if cpu is not None:
        cores = cpu.sample()
        if cores is not None:
            parts.append(f"cpu {cores:.1f} cores")
    return "  ".join(parts)


# --- Scene state writes ---
# PhysX rejects a velocity write on a kinematic body.

def is_kinematic(asset) -> bool:
    """Whether ``asset`` was spawned with ``kinematic_enabled``."""
    rigid_props = getattr(getattr(getattr(asset, "cfg", None), "spawn", None), "rigid_props", None)
    return bool(getattr(rigid_props, "kinematic_enabled", False))


def reset_scene_to(
    scene,
    state: dict[str, dict[str, dict[str, torch.Tensor]]],
    env_ids: Sequence[int] | None = None,
    is_relative: bool = False,
) -> None:
    """``InteractiveScene.reset_to``, minus velocity writes on kinematic bodies.

    ``state`` is in ``scene.get_state()`` form; ``env_ids`` None writes every environment.
    A kinematic body ``state`` has no entry for goes to its default pose.
    """
    if env_ids is None:
        env_ids = scene._ALL_INDICES

    for asset_name, articulation in scene.articulations.items():
        asset_state = state["articulation"][asset_name]
        root_pose = asset_state["root_pose"].clone().to(scene.device)
        if is_relative:
            root_pose[:, :3] += scene.env_origins[env_ids]
        root_velocity = asset_state["root_velocity"].clone().to(scene.device)
        articulation.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        articulation.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        joint_position = asset_state["joint_position"].clone().to(scene.device)
        joint_velocity = asset_state["joint_velocity"].clone().to(scene.device)
        articulation.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
        articulation.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)
        # Assumes PD control over the joints.
        articulation.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
        articulation.set_joint_velocity_target_index(target=joint_velocity, env_ids=env_ids)

    for asset_name, deformable_object in scene.deformable_objects.items():
        asset_state = state["deformable_object"][asset_name]
        nodal_position = asset_state["nodal_position"].clone().to(scene.device)
        if is_relative:
            # nodal_position is (envs, nodes, 3): offset every node.
            nodal_position += scene.env_origins[env_ids].unsqueeze(1)
        nodal_velocity = asset_state["nodal_velocity"].clone().to(scene.device)
        deformable_object.write_nodal_pos_to_sim(nodal_position, env_ids=env_ids)
        deformable_object.write_nodal_velocity_to_sim(nodal_velocity, env_ids=env_ids)

    for asset_name, rigid_object in scene.rigid_objects.items():
        if asset_name not in state["rigid_object"] and is_kinematic(rigid_object):
            # Recorded while this prop was fixed: it stood at its default pose.
            root_pose = rigid_object.data.default_root_pose.torch[env_ids].clone()
            root_pose[:, :3] += scene.env_origins[env_ids]
            rigid_object.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            continue
        asset_state = state["rigid_object"][asset_name]
        root_pose = asset_state["root_pose"].clone().to(scene.device)
        if is_relative:
            root_pose[:, :3] += scene.env_origins[env_ids]
        rigid_object.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        # Skip the velocity write on kinematic bodies.
        if is_kinematic(rigid_object):
            continue
        root_velocity = asset_state["root_velocity"].clone().to(scene.device)
        rigid_object.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    for asset_name, surface_gripper in scene.surface_grippers.items():
        surface_gripper.set_grippers_command(state["gripper"][asset_name])

    # Propagate the joint targets written above.
    scene.write_data_to_sim()
