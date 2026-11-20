"""Isaac Sim env client for XPolicyLab's client-server eval protocol.

`CoHuBTaskEnv` must be constructed after Isaac Sim's `AppLauncher`. The action
is one 35D block per robot in `cohub.keys.ROBOTS3` order:
`[joint targets(31), base_vel(3), height(1)]`. `get_obs()` poses are world-frame wxyz.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


SETTLE_RENDERS_MAX = 30
"""Maximum renders after a reset while the view settles. COHUB_SETTLE_RENDERS overrides."""


def load_env_cfg_type(env_cfg_type: str) -> dict:
    """Load `baselines/env_cfg/<env_cfg_type>.yml`."""
    path = _REPO_ROOT / "baselines" / "env_cfg" / f"{env_cfg_type}.yml"
    if not path.is_file():
        raise FileNotFoundError(
            f"no baselines/env_cfg/{env_cfg_type}.yml at {path} -- register this env_cfg_type first "
            "(config.robot/config.sim for XPolicyLab, plus this module's own `gym_id` key)"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def _xyzw_to_wxyz(pos_quat_xyzw: np.ndarray) -> np.ndarray:
    pos, q = pos_quat_xyzw[:3], pos_quat_xyzw[3:7]
    return np.concatenate([pos, q[[3, 0, 1, 2]]]).astype(np.float32)


def fallen_term_names(robots) -> tuple[str, ...]:
    """The per-robot fall termination term names."""
    return tuple(f"{robot}_fallen" for robot in robots)


def action_term_name(robot: str) -> str:
    """`"robot_c" -> "upper_body_c"`: the name of a robot's upper-body action term."""
    return f"upper_body_{robot.rsplit('_', 1)[1]}"

_TIMING_BUCKETS = ("env_step", "obs_render", "codec", "video", "tracker", "settle")
"""Wall-clock buckets reported per episode in `episode_result()["timing"]`."""


def _new_timing() -> dict[str, float]:
    return dict.fromkeys(_TIMING_BUCKETS, 0.0)


def obs_codec_setting(value: str | None = None) -> str | None:
    """The decoder `get_obs` runs frames through, from COHUB_OBS_CODEC.

    `pyav` (default), `imageio`, or `none` / `0` / `off` for the raw render buffer.
    """
    value = (os.environ.get("COHUB_OBS_CODEC", "pyav") if value is None else value).strip().lower()
    if value in ("", "none", "0", "off", "raw", "false"):
        return None
    if value not in ("pyav", "imageio"):
        raise ValueError(f"COHUB_OBS_CODEC must be pyav, imageio or none, got {value!r}")
    return value


def observation_is_current(episode_step: int, cache_step: int, obs_stride: int) -> bool:
    """Whether the frame rendered at `cache_step` can be reused at `episode_step`.

    `cache_step < 0` means nothing is rendered yet.
    """
    return cache_step >= 0 and (episode_step - cache_step) < obs_stride


VIDEO_ONLY_CAMERAS = ("scene_2",)
"""Dataset cameras that can be recorded into an eval video but not served to a policy."""

SCENE_CAMERA_MODES = ("fixed", "follow")
"""`fixed` keeps the task's authored scene camera pose. `follow` re-poses the recorded camera to track the robots."""


def validate_camera_plan(
    video_cameras: Sequence[str], obs_cameras: Sequence[str], scene_camera: str,
    smooth_video: bool = False,
) -> None:
    """Check the camera arguments. Raises ValueError on a plan that cannot be served.

    The recorder never reads a policy's camera. With `smooth_video=True` it
    reads a record twin. With `smooth_video=False` it holds the policy's last
    frame, and `scene_camera="follow"` is refused when the policy reads `scene`.
    """
    video_only = [c for c in obs_cameras if c in VIDEO_ONLY_CAMERAS]
    if video_only:
        raise ValueError(
            f"obs_cameras {video_only} have no policy vision slot; they can be recorded "
            f"(video_cameras) but not served. See VIDEO_ONLY_CAMERAS."
        )
    if scene_camera not in SCENE_CAMERA_MODES:
        raise ValueError(f"scene_camera must be one of {SCENE_CAMERA_MODES}, got {scene_camera!r}")
    if scene_camera == "follow" and "scene" in obs_cameras and not smooth_video:
        raise ValueError(
            "scene_camera='follow' moves the scene camera, but 'scene' is in obs_cameras, "
            "i.e. the policy would observe a view it was not trained on. Either serve the "
            "policy with scene_camera='fixed' (its hook: SCENE_CAM=fixed), or record from a "
            "twin (smooth_video=True / EVAL_SMOOTH_VIDEO=1), which leaves the policy's camera "
            "on the authored pose and moves the twin instead."
        )


def follow_camera_pose(
    home: Sequence[float],
    robots: Sequence[Sequence[float]],
    hfov_deg: float,
    distance: float,
    max_dolly: float = 1.5,
    margin: float = 1.0,
    lookat_dz: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Where the following camera stands and looks: `(eye, target)`.

    The eye sits on the line from `home` to the robots' centroid, `distance` from
    it, backing off to keep every robot in view but never past `home` or more
    than `max_dolly` from it.
    """
    robots = np.asarray(robots, dtype=np.float64).reshape(-1, 3)
    home = np.asarray(home, dtype=np.float64).reshape(3)
    target = robots.mean(axis=0)
    target[2] += lookat_dz
    half_extent = margin
    for i in range(len(robots)):
        for j in range(i + 1, len(robots)):
            half_extent = max(half_extent, np.linalg.norm(robots[i, :2] - robots[j, :2]) / 2.0 + margin)
    need = half_extent / max(np.tan(np.radians(hfov_deg) / 2.0), 1e-6)
    line = target - home
    length = float(np.linalg.norm(line))
    # Distance from the target: the largest of the three bounds.
    dist = max(float(distance), float(need), length - float(max_dolly))
    eye = home if length <= dist or length < 1e-6 else target - line / length * dist
    return np.asarray(eye, dtype=np.float32), np.asarray(target, dtype=np.float32)


class _AsyncWriter:
    """Feeds an imageio writer from a worker thread, so the control loop only enqueues.

    The queue is bounded. An encoder failure is raised at `close()`.
    """

    def __init__(self, writer, maxsize: int = 64):
        import queue
        import threading

        self._writer = writer
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._drain, name="cohub-video-writer", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            if self._error is None:
                try:
                    self._writer.append_data(frame)
                except BaseException as exc:  # noqa: BLE001 - reported at close()
                    self._error = exc

    def append_data(self, frame) -> None:
        self._queue.put(frame)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        self._writer.close()
        if self._error is not None:
            raise RuntimeError("video encoder failed") from self._error


class CoHuBTaskEnv:
    """Drives a `cohub.tasks` gym env behind XPolicyLab's client-server protocol.

    Construct it after `isaaclab.app.AppLauncher`. Every robot's upper-body action
    term is swapped for joint targets. `video_dir` gets one mp4 per episode per camera.
    """

    def __init__(
        self,
        gym_id: str,
        device: str = "cpu",
        num_envs: int = 1,
        episode_step_limit: int = 300,
        video_dir: str | None = None,
        video_cameras: Sequence[str] = ("scene",),
        obs_cameras: Sequence[str] = ("scene", "ego_a", "ego_b"),
        obs_stride: int = 1,
        video_stride: int = 1,
        ego_depth: bool = True,
        drop_unused_cameras: bool = False,
        scene_camera: str = "fixed",
        follow_distance: float = 2.0,
        follow_max_dolly: float = 1.5,
        video_scale: int = 1,
        smooth_video: bool = False,
    ):
        """`obs_cameras`: the views `get_obs()` reads; each costs a render.
        `obs_stride`: control steps one observation is reused for.
        `video_stride`, `video_scale`: record every Nth step; integer upsampling.
        `ego_depth=False`: drop the ego cameras' depth annotator.
        `drop_unused_cameras`: remove the sensors nothing reads.
        `scene_camera`: `"fixed"` or `"follow"` (`follow_distance`, `follow_max_dolly`).
        `smooth_video`: record policy-input cameras from a twin sensor.
        """
        import gymnasium as gym

        import isaaclab_tasks  # noqa: F401
        from isaaclab.managers import SceneEntityCfg
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        import cohub.tasks  # noqa: F401
        from cohub.g1.actions import joint_target_action_cfg
        from cohub.keys import ROBOTS3, action_order, urdf_order, CAMERA_SENSORS
        from cohub.g1.mdp import body_pos_w, body_quat_w
        from cohub.recording.demo_video import as_decoded, export_crf, open_writer
        from cohub.recording.recorder import camera_frame, robots_in_cfg
        from cohub.runtime import configure_render

        self._body_pos_w = body_pos_w
        self._body_quat_w = body_quat_w
        self._camera_frame = camera_frame
        self._camera_keys = dict(CAMERA_SENSORS)  # {"ego_a": "ego_cam_a", "ego_b": "ego_cam_b", "scene": "scene_cam"}
        self._scene_entity_cfg = SceneEntityCfg
        self._open_video_writer = open_writer
        # `get_obs` serves frames decoded like the LeRobot export's H.264 (see `_decoded_frame`).
        self._as_decoded = as_decoded
        self._export_crf = export_crf
        self._obs_codec = obs_codec_setting()
        self._decoded_memo: dict[str, tuple[tuple[int, int], np.ndarray]] = {}
        self._obs_codec_logged = False

        self.gym_id = gym_id
        self.episode_step_limit = episode_step_limit
        self._num_joint_targets = len(action_order())
        self._video_dir = Path(video_dir) if video_dir else None
        # Video camera names are checked after the env cfg is parsed.
        unknown = [c for c in obs_cameras if c not in self._camera_keys]
        if unknown:
            raise ValueError(
                f"unknown obs cameras {unknown}; this scene has {sorted(self._camera_keys)}"
            )
        self._video_cameras = list(video_cameras)
        self._obs_cameras = list(obs_cameras)
        if obs_stride < 1 or video_stride < 1:
            raise ValueError(f"obs_stride/video_stride must be >= 1, got {obs_stride}/{video_stride}")
        self._obs_stride = int(obs_stride)
        self._video_stride = int(video_stride)
        self._ego_depth = bool(ego_depth)
        self._drop_unused_cameras = bool(drop_unused_cameras)
        self._smooth_video = bool(smooth_video)
        validate_camera_plan(
            self._video_cameras, self._obs_cameras, scene_camera, self._smooth_video
        )
        self._scene_camera = scene_camera
        # Camera key -> the sensor the video reads it from (a twin for a policy-input camera).
        self._record_sensor: dict[str, str] = {}
        self._follow_distance = float(follow_distance)
        if self._follow_distance <= 0:
            raise ValueError(f"follow_distance must be > 0, got {follow_distance}")
        self._follow_max_dolly = float(follow_max_dolly)
        if self._follow_max_dolly < 0:
            raise ValueError(f"follow_max_dolly must be >= 0, got {follow_max_dolly}")
        self._video_scale = int(video_scale)
        if self._video_scale < 1:
            raise ValueError(f"video_scale must be >= 1, got {video_scale}")
        # Smoothed (eye, target) per following camera. Missing means snap on the next update.
        self._follow_states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cut_latched = False
        # One camera read per sensor per step. `_latest_obs_frame` is the last frame handed to the policy.
        self._frame_memo: dict[str, tuple[int, np.ndarray]] = {}
        self._latest_obs_frame: dict[str, np.ndarray] = {}
        self._obs_cache: dict | None = None
        self._obs_cache_step = -1
        # Renders done and observations handed out.
        self._obs_render_count = 0
        self._obs_call_count = 0
        self._timing = _new_timing()
        self._video_writers: dict[str, Any] = {}
        self._video_paths: dict[str, Path] = {}
        self._episode_index = -1
        self._settle_renders = 0

        env_cfg = parse_env_cfg(gym_id, device=device, num_envs=num_envs)
        # The render settings the dataset was rendered with.
        configure_render(env_cfg.sim.render)
        # The task's cut view (cohub.g1.cameras). Must be read before the camera budget.
        self._video_cut = getattr(env_cfg, "video_cut", None)
        cut_name = getattr(self._video_cut, "name", None)
        unknown = [c for c in self._video_cameras if c not in self._camera_keys and c != cut_name]
        if unknown:
            known = sorted(self._camera_keys) + ([cut_name] if cut_name else [])
            raise ValueError(f"unknown video cameras {unknown}; this task offers {known}")
        if cut_name in self._video_cameras:
            for source in (self._video_cut.before, self._video_cut.after):
                if source not in self._camera_keys:
                    raise ValueError(
                        f"{self.gym_id}'s video_cut names camera {source!r}, which is not a dataset "
                        f"camera ({sorted(self._camera_keys)})"
                    )
            self._cut_sources = (self._video_cut.before, self._video_cut.after)
            by_check = bool(self._video_cut.checks_key and self._video_cut.flag)
            if by_check == bool(self._video_cut.entity):
                raise ValueError(
                    f"{self.gym_id}'s video_cut needs exactly one condition: checks_key+flag, "
                    f"or entity+axis+past"
                )
        else:
            self._video_cut, self._cut_sources = None, ()
        self._apply_camera_budget(env_cfg)
        self._add_record_twins(env_cfg)
        # Authored pose and lens of each fixed view, used by `follow_camera_pose`.
        from cohub.g1.geometry import horizontal_fov_deg

        self._scene_homes: dict[str, tuple[np.ndarray, float]] = {}
        for camera in ("scene", *VIDEO_ONLY_CAMERAS):
            cam_cfg = getattr(env_cfg.scene, self._camera_keys.get(camera, ""), None)
            if cam_cfg is None:
                continue
            spawn = cam_cfg.spawn
            self._scene_homes[camera] = (
                np.asarray(cam_cfg.offset.pos, dtype=np.float32),
                horizontal_fov_deg(spawn.focal_length, spawn.horizontal_aperture),
            )
        filmed = set(self._video_cameras) | set(self._cut_sources)
        homeless = [c for c in ("scene", *VIDEO_ONLY_CAMERAS) if c in filmed and c not in self._scene_homes]
        if self._scene_camera == "follow" and homeless:
            raise ValueError(f"scene_camera='follow' but this scene has no {homeless} camera to move")
        env_cfg.isaac_teleop = None
        env_cfg.recorders = None
        # The robot set must be read before `gym.make`: the swap below depends on it.
        self.robots = robots_in_cfg(env_cfg.scene)
        if not self.robots:
            raise ValueError(
                f"{gym_id} declares none of {ROBOTS3} in its scene -- this client "
                "drives CoHuB cooperative-G1 tasks, whose robots carry those names"
            )
        # Swap each robot's upper-body term for joint targets.
        for robot in self.robots:
            term = action_term_name(robot)
            if not hasattr(env_cfg.actions, term):
                raise AttributeError(
                    f"{gym_id} has {robot} in its scene but no `actions.{term}` to swap for "
                    f"joint targets; name the term `{term}` (cohub.tasks._base.env_cfg)"
                )
            setattr(env_cfg.actions, term, joint_target_action_cfg(robot))
        self.env = gym.make(gym_id, cfg=env_cfg).unwrapped
        self.device = self.env.device
        self.num_envs = self.env.num_envs

        # Permutation from each robot's PhysX joint order to observation.state's URDF order.
        state_names = urdf_order()
        self._joint_state_perm = {}
        for robot in self.robots:
            lookup = {name: i for i, name in enumerate(self.env.scene[robot].data.joint_names)}
            self._joint_state_perm[robot] = np.asarray([lookup[name] for name in state_names], dtype=np.int64)

        self._pin_render_sampling()
        self._control_hz = round(1.0 / float(self.env.step_dt))

        self.episode_step = 0
        self._stop_check = None
        # The done reason must be recorded at the step it fires: env.step auto-resets
        # the scene, and env.reset() does not clear the termination buffers.
        self._done = False
        self._done_reason: str | None = None

        # Subtask milestones (cohub.subtasks).
        from cohub.subtasks import make_tracker
        from cohub.task_registry import sentences

        self._tracker = make_tracker(gym_id)
        # Instructions per agent: index 0 is the shared sentence, then one per robot in `ROBOTS3` order.
        said = sentences(gym_id) or ("", "", "")
        self._instructions = {"duo": said[0]}
        for index, sentence in enumerate(said[1:]):
            self._instructions[ROBOTS3[index]] = sentence
        # The success predicate's breakdown, frozen at the step the episode ended.
        self._final_checks: dict[str, bool] | None = None
        self._final_metrics: dict[str, float] | None = None
        self._fallen_terms = fallen_term_names(self.robots)

    def _apply_camera_budget(self, env_cfg) -> None:
        """Remove unneeded camera work from the env cfg. Must run before `gym.make`.

        `ego_depth=False` drops the ego cameras' depth annotator.
        `drop_unused_cameras=True` removes the sensors in neither `obs_cameras`
        nor `video_cameras`. An unrequested video-only camera is always removed.
        """
        from cohub.g1.geometry import DEPTH

        needed = set(self._obs_cameras) | set(self._video_cameras) | set(self._cut_sources)
        dropped, undepthed = [], []
        for key, sensor_name in self._camera_keys.items():
            sensor_cfg = getattr(env_cfg.scene, sensor_name, None)
            if sensor_cfg is None:
                continue
            if key not in needed:
                if self._drop_unused_cameras or key in VIDEO_ONLY_CAMERAS:
                    setattr(env_cfg.scene, sensor_name, None)
                    dropped.append(sensor_name)
                continue
            if not self._ego_depth and DEPTH in (sensor_cfg.data_types or ()):
                sensor_cfg.data_types = [t for t in sensor_cfg.data_types if t != DEPTH]
                undepthed.append(sensor_name)
        if dropped or undepthed:
            print(
                f"[CoHuBTaskEnv] camera budget: obs={self._obs_cameras} "
                f"video={self._video_cameras} dropped={dropped} depth_off={undepthed}"
            )

    def _add_record_twins(self, env_cfg) -> None:
        """Add a record twin for every camera that is both recorded and read by the policy.

        A twin is a copy of the sensor on the same mount with the same lens, RGB
        only, at prim path `<original>Rec`. Only with `smooth_video`.
        """
        if not self._smooth_video:
            return
        import copy

        from cohub.g1.geometry import RGB

        for camera in self._video_cameras:
            if camera not in self._obs_cameras:
                continue  # recorded from the sensor itself
            sensor_name = self._camera_keys[camera]
            sensor_cfg = getattr(env_cfg.scene, sensor_name, None)
            if sensor_cfg is None:
                continue
            twin_name = f"{sensor_name}_rec"
            twin = copy.deepcopy(sensor_cfg)
            twin.prim_path = f"{sensor_cfg.prim_path}Rec"
            twin.data_types = [RGB]
            setattr(env_cfg.scene, twin_name, twin)
            self._record_sensor[camera] = twin_name
        if self._record_sensor:
            print(f"[CoHuBTaskEnv] record twins: {self._record_sensor}")

    def _pin_render_sampling(self) -> None:
        """Pin the render sampling to the dataset's (`cohub.runtime.pin_render_products`).

        `COHUB_EVAL_SPP` overrides the sample count for this process only.
        """
        import os

        from cohub.runtime import pin_render_products

        override = os.environ.get("COHUB_EVAL_SPP")
        if override is not None:
            print(
                f"[CoHuBTaskEnv] COHUB_EVAL_SPP={override} overrides the shared "
                "render pin -- this eval no longer matches the dataset's render. "
                "For a matched run, unset it (or set COHUB_RENDER_SPP, which moves both)."
            )
        pin_render_products(int(override) if override is not None else None)

    def set_stop_check(self, stop_check) -> None:
        self._stop_check = stop_check

    # Observations

    def _camera_rgb(self, sensor_name: str, bucket: str = "obs_render") -> np.ndarray:
        """One camera's current RGB frame, read at most once per step.

        Reading `sensor.data.output` forces the render. `bucket` is the timing
        bucket the read is charged to.
        """
        memo = self._frame_memo.get(sensor_name)
        if memo is not None and memo[0] == self.episode_step:
            return memo[1]
        t0 = time.perf_counter()
        sensor = self.env.scene[sensor_name]
        frame = self._camera_frame(sensor.data.output)
        rgb = frame["rgb"][0].cpu().numpy()  # (H, W, 3) uint8, env 0
        self._timing[bucket] += time.perf_counter() - t0
        self._frame_memo[sensor_name] = (self.episode_step, rgb)
        return rgb

    def _wrist_pose_wxyz(self, robot: str, side: str) -> np.ndarray:
        body_name = f"{side}_wrist_yaw_link"
        asset_cfg = self._scene_entity_cfg(robot)
        pos = self._body_pos_w(self.env, asset_cfg, body_name)[0].cpu().numpy()
        quat_xyzw = self._body_quat_w(self.env, asset_cfg, body_name)[0].cpu().numpy()
        return _xyzw_to_wxyz(np.concatenate([pos, quat_xyzw]))

    def _pelvis_pose_wxyz(self, robot: str) -> np.ndarray:
        data = self.env.scene[robot].data
        pos = data.root_pos_w.torch[0].cpu().numpy()
        quat_xyzw = data.root_quat_w.torch[0].cpu().numpy()
        return _xyzw_to_wxyz(np.concatenate([pos, quat_xyzw]))

    def _joint_pos_urdf_order(self, robot: str) -> np.ndarray:
        """This robot's 43 joint angles [rad] in `keys.urdf_order()` order."""
        joint_pos = self.env.scene[robot].data.joint_pos.torch[0].cpu().numpy()
        return joint_pos[self._joint_state_perm[robot]].astype(np.float32)

    _VISION_SLOT = {
        "scene": "cam_head",
        "ego_a": "cam_left_wrist",
        "ego_b": "cam_right_wrist",
        "ego_c": "cam_third_view",
    }
    """Camera key -> XPolicyLab's generic obs slot. Adapters map the slots back through their `COHUB_CAMERA_SLOT`."""

    def get_obs(self, env_idx: int = 0) -> dict:
        """This step's observation, or the cached one while it is still current.

        `obs_stride` must divide the policy's action chunk. At
        `obs_stride == n_action_steps` every observation a current-frame policy
        consumes is rendered at its own step.
        """
        self._obs_call_count += 1
        if self._obs_cache is not None and observation_is_current(
            self.episode_step, self._obs_cache_step, self._obs_stride
        ):
            cached = self._obs_cache
            return cached if cached["env_idx"] == env_idx else {**cached, "env_idx": env_idx}

        vision = {}
        for camera in self._obs_cameras:
            rgb = self._camera_rgb(self._camera_keys[camera])
            if self._obs_codec:
                rgb = self._decoded_frame(camera, rgb)
            vision[self._VISION_SLOT[camera]] = {"color": rgb}
            self._latest_obs_frame[camera] = rgb

        # One group per robot in the scene.
        cohub_state = {
            robot: {
                "pelvis_pose": self._pelvis_pose_wxyz(robot),
                "left_eef_pose": self._wrist_pose_wxyz(robot, "left"),
                "right_eef_pose": self._wrist_pose_wxyz(robot, "right"),
                "joint_pos": self._joint_pos_urdf_order(robot),
            }
            for robot in self.robots
        }
        lead = cohub_state[self.robots[0]]

        obs = {
            "vision": vision,
            "state": {
                # Standard XPolicyLab slots: the first robot only.
                "left_ee_pose": lead["left_eef_pose"],
                "right_ee_pose": lead["right_eef_pose"],
                "mobile": {"base_pose": lead["pelvis_pose"]},
            },
            # Every robot's full state.
            "cohub_state": cohub_state,
            # `instruction` is the shared sentence. `cohub_instruction` has one per agent.
            "instruction": self._instructions["duo"],
            "cohub_instruction": dict(self._instructions),
            "additional_info": {"frequency": 30},
            "data_format_version": "v1.0",
            "env_idx": env_idx,
        }
        self._obs_cache = obs
        self._obs_cache_step = self.episode_step
        self._obs_render_count += 1
        return obs

    def _decoded_frame(self, camera: str, rgb: np.ndarray) -> np.ndarray:
        """`rgb` after the LeRobot export's H.264 round trip, as training frames are decoded.

        Memoised per step. `COHUB_OBS_CODEC=none` serves the raw render instead.
        """
        key = (self._episode_index, self.episode_step)
        memo = self._decoded_memo.get(camera)
        if memo is not None and memo[0] == key:
            return memo[1]
        t0 = time.perf_counter()
        decoded = self._as_decoded(rgb, backend=self._obs_codec)
        elapsed = time.perf_counter() - t0
        self._timing["codec"] += elapsed
        self._decoded_memo[camera] = (key, decoded)
        if not self._obs_codec_logged:
            self._obs_codec_logged = True
            print(
                f"[env] obs codec: libx264 crf {self._export_crf()} yuv420p via {self._obs_codec} "
                f"-- frames as the LeRobot export decodes them ({elapsed * 1000:.0f} ms/frame; "
                "COHUB_OBS_CODEC=none for the raw render)",
                flush=True,
            )
        return decoded

    def get_obs_batch(self, env_idx_list: list[int]) -> list[dict]:
        return [self.get_obs(env_idx) for env_idx in env_idx_list]

    # Actions

    def _robot_action_35d(self, robot_action: dict, robot: str | None = None) -> np.ndarray:
        """One robot's `cohub_raw_action` entry -> its 35D env action slice.

        Input: `{joint_targets(31, keys.action_order() order), base_vel(3), height(1)}`.
        Output: `[joint_targets(31) base_vel(3) height(1)]`.
        """
        joint_targets = np.asarray(robot_action["joint_targets"], dtype=np.float32)
        if joint_targets.shape != (self._num_joint_targets,):
            raise ValueError(
                f"joint_targets must be shape ({self._num_joint_targets},), got {joint_targets.shape} -- "
                "see cohub.keys.action_order() for the expected order"
            )
        base_vel = np.asarray(robot_action["base_vel"], dtype=np.float32)
        height = np.asarray(robot_action["height"], dtype=np.float32)
        return np.concatenate([joint_targets, base_vel, height]).astype(np.float32)

    def _action_tensor(self, action: dict) -> torch.Tensor:
        raw = action.get("cohub_raw_action")
        if raw is None:
            raise KeyError(
                "action has no 'cohub_raw_action' -- CoHuBTaskEnv currently requires it "
                "(the standard XPolicyLab ee-pose keys alone can't drive every robot's "
                "wrists/hands/locomotion; see model.py's _pack_dual_arm_action docstring)"
            )
        try:
            blocks = [self._robot_action_35d(raw[robot], robot) for robot in self.robots]
        except KeyError as exc:
            raise KeyError(
                f"{exc} missing from cohub_raw_action. The env drives the articulation with "
                "joint targets {joint_targets(31), base_vel(3), height(1)}"
                f". This task has {len(self.robots)} robot(s) -- "
                f"{', '.join(self.robots)} -- and every one of them needs an entry; a policy "
                f"whose action head was trained on a different robot count cannot drive it."
            ) from exc
        # `ROBOTS3` order: robot i owns block i of the env action.
        full = np.concatenate(blocks).astype(np.float32)
        return torch.as_tensor(full, dtype=torch.float32, device=self.device).unsqueeze(0)

    def take_action(self, action: dict) -> None:
        action_tensor = self._action_tensor(action)
        # The follow camera must be posed before `env.step` renders the recorded frame.
        if self._follow_active() and self.episode_step % self._video_stride == 0:
            self._update_follow_camera()
        # no_grad, not inference_mode: tensors created under inference_mode cannot be
        # updated in place by a later env.reset().
        t0 = time.perf_counter()
        with torch.no_grad():
            _, _, terminated, truncated, _ = self.env.step(action_tensor)
        self._timing["env_step"] += time.perf_counter() - t0
        self.episode_step += 1
        self._record_video_frame()
        done_now = bool(terminated[0]) or bool(truncated[0])
        if not self._done:
            t0 = time.perf_counter()
            if self._tracker is not None:
                # The scene is already auto-reset on the done step: the tracker must use the checks store.
                self._tracker.update(self.env, step=self.episode_step, terminal=done_now)
            self._timing["tracker"] += time.perf_counter() - t0
            if done_now:
                # The scene is already auto-reset: record the reason and the final checks now.
                self._done = True
                self._done_reason = self._classify_done(bool(truncated[0]))
                self._snapshot_final_state()

    def _classify_done(self, truncated: bool) -> str:
        """Why the episode ended, read from the term buffers of the step that fired."""
        if self._success():
            return "success"
        for name in self._fallen_terms:
            try:
                if bool(self.env.termination_manager.get_term(name)[0]):
                    return name
            except KeyError:
                continue
        for name in getattr(self.env.termination_manager, "active_terms", []):
            if name in ("success", *self._fallen_terms):
                continue
            try:
                if bool(self.env.termination_manager.get_term(name)[0]):
                    return name
            except KeyError:
                continue
        return "time_out" if truncated else "terminated"

    def _record_video_frame(self) -> None:
        """Append one frame per recorded camera every `video_stride` executed steps."""
        if not self._video_writers:
            return
        if self._video_stride > 1 and (self.episode_step - 1) % self._video_stride:
            return
        for camera, writer in self._video_writers.items():
            if camera == getattr(self._video_cut, "name", None):
                before, after = self._cut_sources
                frame = self._video_frame(after if self._cut_done() else before)
                if frame is None:
                    continue
                writer.append_data(frame)
                continue
            frame = self._video_frame(camera)
            if frame is None:
                continue
            writer.append_data(frame)

    def _cut_done(self) -> bool:
        """Whether the cut has happened this episode (latched once true)."""
        if not self._cut_latched:
            cut = self._video_cut
            if cut.checks_key:
                from cohub.subtasks import read_checks

                self._cut_latched = bool(read_checks(self.env, cut.checks_key).get(cut.flag, False))
            else:
                root = self.env.scene[cut.entity].data.root_pos_w.torch[0] - self.env.scene.env_origins[0]
                self._cut_latched = float(root[cut.axis]) > cut.past
        return self._cut_latched

    def _video_frame(self, camera: str) -> np.ndarray | None:
        """This step's frame for one camera.

        From its record twin if it has one. Else, for a policy-input camera, the
        policy's last frame (None before the first observation). Else the camera itself.
        """
        twin = self._record_sensor.get(camera)
        if twin is not None:
            return self._camera_rgb(twin, bucket="video")
        if camera in self._obs_cameras:
            return self._latest_obs_frame.get(camera)
        return self._camera_rgb(self._camera_keys[camera], bucket="video")

    def _followed_cameras(self) -> list[str]:
        """Fixed views filmed this run (the cut's sources included) that can follow."""
        if self._scene_camera != "follow":
            return []
        filmed = set(self._video_writers) | set(self._cut_sources)
        return [c for c in self._camera_keys if c in filmed and c in self._scene_homes]

    def _follow_active(self) -> bool:
        return bool(self._followed_cameras())

    def _update_follow_camera(self) -> None:
        """Re-pose every followed fixed view onto the robots (`follow_camera_pose`).

        Smoothed with a fixed gain across recorded frames. `reset` clears the
        state so the first frame of an episode snaps.
        """
        robots = [
            self.env.scene[robot].data.root_pos_w.torch[0].cpu().numpy()
            for robot in self.robots
        ]
        for camera in self._followed_cameras():
            home, hfov = self._scene_homes[camera]
            eye, target = follow_camera_pose(
                home, robots, hfov, self._follow_distance, self._follow_max_dolly
            )
            previous = self._follow_states.get(camera)
            if previous is not None:
                gain = 0.3
                eye = previous[0] + gain * (eye - previous[0])
                target = previous[1] + gain * (target - previous[1])
            self._follow_states[camera] = (eye, target)
            sensor = self._record_sensor.get(camera, self._camera_keys[camera])
            # The twin moves when the policy reads that view.
            sensor = self._record_sensor.get(camera, self._camera_keys[camera])
            self.env.scene[sensor].set_world_poses_from_view(eye[None], target[None])

    def take_action_batch(self, action_list: list[dict], env_idx_list: list[int]) -> None:
        assert len(action_list) == len(env_idx_list), (
            f"action num != env num: {len(action_list)} != {len(env_idx_list)}"
        )
        # Batched multi-env stepping is not implemented (num_envs=1).
        for action in action_list:
            self.take_action(action)

    # Episode lifecycle

    def _success(self) -> bool:
        try:
            return bool(self.env.termination_manager.get_term("success")[0])
        except KeyError:
            return False

    def _fallen(self) -> bool:
        for name in self._fallen_terms:
            try:
                if bool(self.env.termination_manager.get_term(name)[0]):
                    return True
            except KeyError:
                continue
        return False

    def is_episode_end(self) -> bool:
        if self._stop_check is not None and self._stop_check():
            return True
        if self._done:
            return True
        if self.episode_step == 0:
            # env.reset() does not clear the termination buffers. A just-reset episode is not over.
            return False
        return self.episode_step >= self.episode_step_limit or self._success() or self._fallen()

    def _snapshot_final_state(self) -> None:
        """Freeze the predicate's published breakdown at episode end."""
        if self._tracker is None:
            return
        self._final_checks, self._final_metrics = self._tracker.snapshot(self.env)

    def episode_result(self) -> dict:
        """How the last episode ended. Call once `is_episode_end()` is True.

        `success` is the last subtask of the chain, not the bare `success`
        DoneTerm. The DoneTerm's answer is `env_success`. When it is true but a
        subtask was not observed, `reason` is `success_missing_<subtask>`. A task
        with no chain scores the DoneTerm. Idempotent.
        """
        reason = self._done_reason
        if reason is None:
            if self._success():
                reason = "success"
            elif self._fallen():
                reason = next(
                    (n for n in self._fallen_terms
                     if bool(self.env.termination_manager.get_term(n)[0])),
                    "fallen",
                )
            else:
                reason = "step_limit"
        env_success = reason == "success"
        result = {"success": env_success, "reason": reason, "steps": self.episode_step}
        if self._tracker is not None:
            if self._final_checks is None:
                # No DoneTerm fired, so the scene was not reset: snapshot now.
                self._snapshot_final_state()
            subtasks, subtask_steps = self._tracker.result(
                final_success=env_success, final_step=self.episode_step
            )
            if subtasks:
                scored = subtasks[self._tracker.names[-1]]
                if env_success and not scored:
                    missing = next(
                        (name for name, done in subtasks.items() if not done), "chain"
                    )
                    reason = f"success_missing_{missing}"
                result.update(success=scored, reason=reason, env_success=env_success)
            result.update(subtasks=subtasks, subtask_steps=subtask_steps)
            if self._tracker.details:
                details, detail_steps = self._tracker.detail_result()
                result.update(subtask_details=details, subtask_detail_steps=detail_steps)
            result.update(final_checks=self._final_checks, final_metrics=self._final_metrics)
        result["timing"] = {name: round(seconds, 3) for name, seconds in self._timing.items()}
        result["obs_renders"] = self._obs_render_count
        result["obs_calls"] = self._obs_call_count
        result["obs_codec"] = self._obs_codec or "none"
        result["settle_renders"] = self._settle_renders
        return result

    def _settle_render(self, max_renders: int = SETTLE_RENDERS_MAX, tol: float = 0.5) -> None:
        """Render after a reset until the ego view stops changing.

        Stops after `max_renders`, or when the mean frame difference drops below
        `tol`. `COHUB_SETTLE_RENDERS=0` disables it.
        """
        max_renders = int(os.environ.get("COHUB_SETTLE_RENDERS", max_renders))
        cameras = [self._camera_keys[c] for c in self._obs_cameras if c != "scene"] or [
            self._camera_keys[c] for c in self._obs_cameras
        ]
        self._settle_renders = 0
        if max_renders <= 0 or not cameras:
            return
        t0 = time.perf_counter()
        prev = None
        renders = 0
        for renders in range(1, max_renders + 1):
            self.env.sim.render()
            frames = []
            for name in cameras:
                sensor = self.env.scene[name]
                sensor.update(0.0, force_recompute=True)
                frames.append(sensor.data.output["rgb"][0, ..., :3].float())
            cur = torch.cat(frames)
            if prev is not None and float((cur - prev).abs().mean()) < tol:
                break
            prev = cur
        self._settle_renders = renders
        self._timing["settle"] = self._timing.get("settle", 0.0) + time.perf_counter() - t0
        if self._episode_index < 0:
            print(f"[env] settle: {renders} renders, {time.perf_counter() - t0:.2f}s", flush=True)

    def finish_episode(self) -> None:
        self._close_video_writer()

    def get_running_env_idx_list(self) -> list[int]:
        return list(range(self.num_envs))

    def reset(self, seed: int | None = None, settle: bool = True, video_stem: str | None = None) -> None:
        """`settle=False` is for a caller about to overwrite the scene, which
        settles the renderer itself. `video_stem` names the episode's video
        files when the seed does not tell episodes apart."""
        # sim.reset() before env.reset(), as record_demos.py does.
        self.env.sim.reset()
        self.env.reset(seed=seed)
        self._close_video_writer()
        self.episode_step = 0
        self._done = False
        self._done_reason = None
        # A cached observation belongs to the episode that rendered it.
        self._obs_cache = None
        self._obs_cache_step = -1
        self._frame_memo = {}
        self._latest_obs_frame = {}
        self._follow_states.clear()
        self._cut_latched = False
        self._obs_render_count = 0
        self._obs_call_count = 0
        self._timing = _new_timing()
        # Must run after the timing reset.
        if settle:
            self._settle_render()
        if self._tracker is not None:
            self._tracker.reset()
        self._final_checks = None
        self._final_metrics = None
        self._episode_index += 1
        if self._video_dir is not None:
            stem = video_stem or (f"seed{seed:03d}" if seed is not None else f"episode{self._episode_index:03d}")
            for camera in self._video_cameras:
                # One directory per camera, the same file name in each.
                path = self._video_dir / camera / f"{stem}_RUNNING.mp4"
                self._video_paths[camera] = path
                # fps is the control rate divided by `video_stride`, so playback is real time.
                self._video_writers[camera] = _AsyncWriter(
                    self._open_video_writer(
                        path,
                        fps=self._control_hz / self._video_stride,
                        scale=self._video_scale,
                        preset="veryfast",
                    )
                )
            if self._follow_active():
                self._update_follow_camera()

    def _close_video_writer(self) -> None:
        """Close the writers and rename each file from `_RUNNING` to the episode's verdict."""
        if not self._video_writers:
            return
        for writer in self._video_writers.values():
            writer.close()
        self._video_writers = {}
        if not self._video_paths:
            return
        result = self.episode_result()
        verdict = "SUCCESS" if result["success"] else f"FAIL-{result['reason']}"
        for path in self._video_paths.values():
            final = path.with_name(
                f"{path.name[: -len('_RUNNING.mp4')]}_{verdict}_{result['steps']}steps.mp4"
            )
            path.replace(final)
        self._video_paths = {}

    def close(self) -> None:
        self._close_video_writer()
        self.env.close()
