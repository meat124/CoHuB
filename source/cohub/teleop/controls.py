"""What the operator's controllers do: walking and lean shaping, session buttons, crouch, haptics, raw input."""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np


# --- Walk and turn shaping ---
# The policy barely turns below about 0.8 rad/s, so yaw input is mapped onto the band where it
# responds. The shaped value is what is recorded.

YAW_DEADZONE = 0.15
"""Raw yaw commands below this are treated as zero."""

YAW_MIN = 0.85
"""Bottom of the responsive band [rad/s]."""

YAW_MAX = 1.2
"""Yaw rate [rad/s] at full stick deflection."""


def shape_yaw(command: float) -> float:
    """Map a raw yaw command onto the band where the policy turns."""
    magnitude = abs(command)
    if magnitude < YAW_DEADZONE:
        return 0.0
    span = max(1.0 - YAW_DEADZONE, 1e-6)
    scaled = min((magnitude - YAW_DEADZONE) / span, 1.0)
    shaped = YAW_MIN + (YAW_MAX - YAW_MIN) * scaled
    return shaped if command > 0 else -shaped


WALK_FULL_STICK = 0.5
"""What a fully deflected thumbstick arrives as [m/s]."""

WALK_DEADZONE = 0.03
"""Walk commands below this [m/s] are treated as zero."""

DEFAULT_MAX_SPEED = 0.5
"""Full-stick walk speed [m/s] after re-scaling. A command ceiling, not an
achieved speed. ``record_demos.py --max_speed`` overrides it per session."""


def shape_walk(vx: float, vy: float, max_speed: float = DEFAULT_MAX_SPEED) -> tuple[float, float]:
    """Re-scale a raw walk command so a full stick means ``max_speed`` [m/s].

    A scale, not a clamp, so the whole deflection range stays live.
    """
    magnitude = (vx * vx + vy * vy) ** 0.5
    if magnitude < WALK_DEADZONE:
        return 0.0, 0.0
    scale = max_speed / WALK_FULL_STICK
    return vx * scale, vy * scale


# --- Lean compensation ---
# The XR anchor follows only the pelvis's translation and yaw. This rotates each wrist target by the
# pelvis's pitch and roll, minus the slow part, so a held pose stays put. Only for slices that came
# through an XR anchor.

ROT_TIME_CONSTANT = float("inf")
"""Reference time constant while the hand is still [sim s]. ``inf`` freezes
the reference, which is full attachment."""

TRACK_TIME_CONSTANT = 0.3
"""Reference time constant while the hand is moving [sim s]."""

HOLD_RADIUS = 0.02
HOLD_SPAN = 0.06
"""How far the hand must leave the held pose before the reference unfreezes
[m], and over how much further the release ramps to full [m]."""

HOLD_FOLLOW_TIME_CONSTANT = 0.5
"""How fast the held pose follows the hand once it has been released [sim s]."""

HOLD_SMOOTH_TIME_CONSTANT = 0.4
"""Low-pass on the wrist targets while the hand is holding still [sim s]."""

Z_TIME_CONSTANT = float("inf")
"""Reference time constant for the height term while the hand is still [sim s]."""

_WRIST_BLOCKS = ((0, 3), (7, 10))
_WRIST_QUATS = ((3, 7), (10, 14))


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def _conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _yaw_only(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=np.float64)


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # float64 version of follower_base.quat_apply.
    x, y, z, w = q
    t = 2.0 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)


class LeanCompensator:
    """Torso attachment for one robot's 32D slice.

    One instance per robot. Call :meth:`reset` on every episode boundary.
    """

    def __init__(
        self,
        rot_time_constant: float = ROT_TIME_CONSTANT,
        z_time_constant: float = Z_TIME_CONSTANT,
        track_time_constant: float = TRACK_TIME_CONSTANT,
        hold_smooth_time_constant: float = HOLD_SMOOTH_TIME_CONSTANT,
        hold_radius: float = HOLD_RADIUS,
        hold_span: float = HOLD_SPAN,
    ) -> None:
        self.rot_time_constant = float(rot_time_constant)
        self.z_time_constant = float(z_time_constant)
        self.track_time_constant = float(track_time_constant)
        self.hold_smooth_time_constant = float(hold_smooth_time_constant)
        self.hold_radius = float(hold_radius)
        self.hold_span = float(hold_span)
        self.reset()

    HOLD_FOLLOW_TIME_CONSTANT = HOLD_FOLLOW_TIME_CONSTANT

    def reset(self) -> None:
        """Reseed on the next sample; the first compensated slice is unchanged."""
        self._lp_delta: np.ndarray | None = None
        self._lp_qcorr: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
        self._z_ref: float | None = None
        self._hold_ref: np.ndarray | None = None
        self._hip_ref: float | None = None
        self._lp_rel: np.ndarray | None = None

    def apply(
        self,
        slice32: np.ndarray,
        pelvis_pos,
        pelvis_quat_xyzw,
        dt: float,
    ) -> np.ndarray:
        """Return a compensated copy of ``slice32`` (wrist poses in columns 0:14, env frame).

        ``dt`` is simulated seconds since the previous call.
        """
        pos = np.asarray(pelvis_pos, dtype=np.float64)
        quat = np.asarray(pelvis_quat_xyzw, dtype=np.float64)
        quat = quat / max(float(np.linalg.norm(quat)), 1e-9)

        q_flat = _yaw_only(quat)
        pr = quat_mul(_conj(q_flat), quat).astype(np.float64)
        z = float(pos[2])

        if self._z_ref is None:
            # First sample: sets the height reference, no correction.
            self._z_ref = z
            self._lp_delta = None
            self._lp_qcorr = np.array([0.0, 0.0, 0.0, 1.0])
            return np.array(slice32, copy=True)

        # Wrist positions in the anchor's frame: yaw-only, height pinned.
        origin = np.array([pos[0], pos[1], self._z_ref], dtype=np.float64)
        rel = np.concatenate([
            _rotate(_conj(q_flat), np.asarray(slice32[a:b], dtype=np.float64) - origin)
            for a, b in _WRIST_BLOCKS
        ])
        hip = float(slice32[31])
        if self._hold_ref is None:
            self._hold_ref = rel.copy()
            self._hip_ref = hip
        # Distance from the held pose. A commanded crouch counts too.
        excursion = float(
            np.linalg.norm((rel - self._hold_ref).reshape(len(_WRIST_BLOCKS), 3), axis=1).max()
        )
        excursion = max(excursion, abs(hip - self._hip_ref))

        a_still = 1.0 - np.exp(-dt / self.rot_time_constant) if self.rot_time_constant > 0 else 1.0
        a_track = 1.0 - np.exp(-dt / self.track_time_constant) if self.track_time_constant > 0 else 1.0
        blend = float(np.clip((excursion - self.hold_radius) / max(self.hold_span, 1e-9), 0.0, 1.0))
        # The held pose follows only while the hand is released.
        a_follow = blend * (1.0 - np.exp(-dt / self.HOLD_FOLLOW_TIME_CONSTANT))
        self._hold_ref += a_follow * (rel - self._hold_ref)
        self._hip_ref += a_follow * (hip - self._hip_ref)
        a_rot = a_still + blend * (a_track - a_still)
        # Smooth the wrist targets while the hand holds still.
        smoothed = None
        if self.hold_smooth_time_constant > 0.0:
            if self._lp_rel is None:
                self._lp_rel = rel.copy()
            a_hold = 1.0 - np.exp(-dt / self.hold_smooth_time_constant)
            a_lp = blend + (1.0 - blend) * a_hold
            self._lp_rel = self._lp_rel + a_lp * (rel - self._lp_rel)
            smoothed = self._lp_rel

        # Full rotation between the anchor's yaw-only frame and the pelvis.
        q_corr = quat_mul(quat, _conj(q_flat))
        q_corr /= max(float(np.linalg.norm(q_corr)), 1e-9)
        dz_full = z - self._z_ref

        # The slow part of the correction is subtracted as a translation.
        out = np.array(slice32, dtype=np.float64, copy=True)
        origin = np.array([pos[0], pos[1], self._z_ref], dtype=np.float64)
        if self._lp_delta is None:
            self._lp_delta = np.zeros((len(_WRIST_BLOCKS), 3), dtype=np.float64)
        for k, (start, end) in enumerate(_WRIST_BLOCKS):
            p_in = np.asarray(out[start:end], dtype=np.float64)
            if smoothed is not None:
                p_in = origin + _rotate(q_flat, smoothed[3 * k:3 * k + 3])
            p_full = origin + _rotate(q_corr, p_in - origin)
            p_full[2] += dz_full
            # The slow part is stored and subtracted in the pelvis frame.
            delta_body = _rotate(_conj(quat), p_full - p_in)
            self._lp_delta[k] += a_rot * (delta_body - self._lp_delta[k])
            out[start:end] = p_full - _rotate(quat, self._lp_delta[k])

        # Same for the wrist orientations.
        if float(np.dot(self._lp_qcorr, q_corr)) < 0.0:
            self._lp_qcorr = -self._lp_qcorr
        self._lp_qcorr = self._lp_qcorr + a_rot * (q_corr - self._lp_qcorr)
        self._lp_qcorr /= max(float(np.linalg.norm(self._lp_qcorr)), 1e-9)
        q_apply = quat_mul(_conj(self._lp_qcorr), q_corr)
        for start, end in _WRIST_QUATS:
            q_in = out[start:end]
            norm = float(np.linalg.norm(q_in))
            if norm > 1e-6:
                out[start:end] = quat_mul(q_apply, q_in / norm)
        return out.astype(np.asarray(slice32).dtype, copy=False)


# --- Haptics ---
# The controller vibrates with the grip force while a hand holds the object. It reads private
# attributes of the teleop device; if they change, haptics turn off and recording continues.

logger = logging.getLogger(__name__)

# Re-issued every control step: a new pulse supersedes the one in flight.
PULSE_FREQUENCY_HZ = 0.0
"""0 asks the backend for its default frequency."""

PULSE_DURATION_S = 0.05
"""Pulse length [s]. Slightly longer than one control step, so the pulse
train is continuous."""

_MAX_DROPS = 20
"""Consecutive failed sends before haptics turn off. Reset by any successful
send."""

_BIND_WARN_AFTER = 200
"""Steps without a session before a warning is printed."""


def resolve_teleop_session(teleop):
    """The live ``TeleopSession`` behind an Isaac Lab teleop device, or None."""
    for holder in (getattr(teleop, "_session_lifecycle", None), teleop):
        if holder is None:
            continue
        session = getattr(holder, "_session", None)
        if session is not None:
            return session
    return None


def resolve_controller_tracker(teleop, capability: str = "get_left_controller"):
    """The live controller tracker offering ``capability`` and its DeviceIO session: ``(tracker, session)``.

    The pipeline's own tracker is reused; constructing a second one would fail.
    """
    session = resolve_teleop_session(teleop)
    if session is None:
        return None, None
    deviceio_session = getattr(session, "deviceio_session", None)
    sources = getattr(session, "_sources", None)
    if deviceio_session is None or not sources:
        return None, None
    for source in sources:
        get_tracker = getattr(source, "get_tracker", None)
        if get_tracker is None:
            continue
        tracker = get_tracker()
        if tracker is not None and hasattr(tracker, capability):
            return tracker, deviceio_session
    return None, deviceio_session


class ControllerHaptics:
    """Vibrates each Quest controller in proportion to that hand's load.

    Args:
        force_floor: Interaction forces below this are ignored [N].
        force_ceiling: Force mapped to full amplitude [N].
        min_amplitude: Amplitude to use once past the floor.
    """

    def __init__(
        self,
        force_floor: float = 1.0,
        force_ceiling: float = 100.0,
        min_amplitude: float = 0.08,
    ):
        self.force_floor = force_floor
        self.force_ceiling = force_ceiling
        self.min_amplitude = min_amplitude
        self._tracker = None
        self._session = None
        self._last_session = None
        self._failed = False
        self._attempts = 0
        self._drops = 0
        self._rebinds = 0
        self._last_sent: dict[str, float] = {"left": -1.0, "right": -1.0}

    def bind(self, teleop) -> bool:
        """Find the live controller tracker behind an Isaac Lab teleop device.

        Call this every frame. The DeviceIO session can be replaced while the
        tracker object stays the same, so the binding is keyed on the session.
        """
        if self._failed:
            return False

        session = resolve_teleop_session(teleop)
        if session is None:
            if self._tracker is not None:
                self._release("teleop session went away")
            self._attempts += 1
            if self._attempts == _BIND_WARN_AFTER:
                print(
                    "[haptics] no live teleop session after "
                    f"{_BIND_WARN_AFTER} steps - haptics are OFF. "
                    "Is the headset connected and 'Start Teleop' pressed?"
                )
            return False  # XR session not started yet

        tracker, deviceio_session = resolve_controller_tracker(teleop, "apply_left_haptic_feedback")
        if deviceio_session is None:
            return False

        if self._tracker is not None and deviceio_session is self._session:
            return True
        if self._tracker is not None:
            self._release("session was replaced")

        if tracker is None or not hasattr(tracker, "apply_left_haptic_feedback"):
            self._failed = True
            print("[haptics] OFF - no controller tracker exposes haptic feedback")
            return False

        # Only a new session resets the drop count.
        fresh = deviceio_session is not self._last_session

        self._tracker = tracker
        self._session = deviceio_session
        self._last_session = deviceio_session
        self._last_sent = {"left": -1.0, "right": -1.0}
        if fresh:
            self._drops = 0
            self._rebinds += 1
            if self._rebinds == 1:
                # Confirmation pulse on both controllers.
                print("[haptics] bound to the live controller tracker"
                      " - sending confirmation pulse to both controllers")
                self.send(self.force_ceiling, self.force_ceiling)
            else:
                print(f"[haptics] re-bound after XR session restart (#{self._rebinds - 1})")
        return True

    def _release(self, why: str) -> None:
        """Forget a handle that is no longer live, keeping haptics enabled."""
        self._tracker = None
        self._session = None
        self._last_sent = {"left": -1.0, "right": -1.0}
        logger.info("haptics released its binding: %s", why)

    def amplitude_for(self, force_magnitude: float) -> float:
        """Map a wrist force magnitude [N] to a vibration amplitude in [0, 1]."""
        if force_magnitude <= self.force_floor:
            return 0.0
        span = max(self.force_ceiling - self.force_floor, 1e-6)
        fraction = min((force_magnitude - self.force_floor) / span, 1.0)
        return self.min_amplitude + (1.0 - self.min_amplitude) * fraction

    def send(self, left_force: float, right_force: float) -> None:
        """Update both controllers from this step's force magnitudes [N]."""
        if self._tracker is None:
            return
        for endpoint, force in (("left", left_force), ("right", right_force)):
            amplitude = self.amplitude_for(force)
            # Skip an unchanged zero.
            if amplitude == 0.0 and self._last_sent[endpoint] == 0.0:
                continue
            self._last_sent[endpoint] = amplitude
            method = (
                self._tracker.apply_left_haptic_feedback
                if endpoint == "left"
                else self._tracker.apply_right_haptic_feedback
            )
            try:
                method(self._session, amplitude, PULSE_FREQUENCY_HZ, PULSE_DURATION_S)
            except Exception as exc:  # noqa: BLE001
                # A stale handle is expected after a session rebuild: re-bind.
                self._drops += 1
                self._release(f"send failed: {exc}")
                if self._drops >= _MAX_DROPS:
                    self._failed = True
                    print(f"[haptics] OFF after {_MAX_DROPS} consecutive failed sends: {exc}")
                return
            self._drops = 0

    def stop(self) -> None:
        """Silence both actuators, e.g. when a session ends."""
        if self._tracker is not None:
            self._last_sent = {"left": -1.0, "right": -1.0}
            self.send(0.0, 0.0)


def object_interaction_forces(env, robot: str = "robot_a", contact_sensor: str = "hand_contact_a"):
    """Left and right interaction force with the object [N], 0 when not engaged.

    A hand that holds the object reports its grip force, the value recorded
    as ``states/grasp/force``. Any wrist contact force is taken on top.
    """
    forces = [0.0, 0.0]
    try:
        term = env.action_manager.get_term("grasp")
        offset = list(term.cfg.robot_names).index(robot) * term._hands_per_robot
        for side in (0, 1):
            hand = offset + side
            if bool(term.latched[0, hand]):
                forces[side] = float(term.grasp_force[0, hand].norm())
    except (KeyError, ValueError, AttributeError, IndexError):
        pass

    sensor = env.scene.sensors.get(contact_sensor)
    if sensor is not None and sensor.data.net_forces_w is not None:
        contact = sensor.data.net_forces_w.torch[0]
        for side in (0, 1):
            if side < contact.shape[0]:
                forces[side] = max(forces[side], float(contact[side].norm()))

    return forces[0], forces[1]


# --- Session buttons ---
# Right controller: A start/pause, B reset. Upstream binds A to the anchor-rotation toggle;
# release_upstream_a_binding frees it.

class SessionButtons:
    """Rising edges on the right controller's A and B.

    Same shape as
    :meth:`cohub.teleop.follower.FollowerDevice.consume_control_flags`.
    """

    def __init__(self, hold_off: float = 0.4):
        self.bound = False
        self.hold_off = hold_off
        """Seconds a press blocks a repeat."""
        self._prev = (False, False)
        self._last_edge = {"a": 0.0, "b": 0.0}
        self._failed = False

    def consume_control_flags(self, now: float | None = None) -> tuple[bool, bool]:
        """Return ``(toggle_run, reset)`` pressed since the last call."""
        import time

        if now is None:
            now = time.monotonic()

        a, b = self._read(self._teleop)
        prev_a, prev_b = self._prev
        self._prev = (a, b)

        toggle = self._edge("a", a and not prev_a, now)
        reset = self._edge("b", b and not prev_b, now)
        return toggle, reset

    def bind(self, teleop) -> None:
        """Remember the device to read from."""
        self._teleop = teleop

    _teleop = None

    def _edge(self, key: str, pressed: bool, now: float) -> bool:
        if not pressed:
            return False
        if now - self._last_edge[key] < self.hold_off:
            return False
        self._last_edge[key] = now
        return True

    def _read(self, teleop) -> tuple[bool, bool]:
        """``(A pressed, B pressed)``, or ``(False, False)`` if unreadable."""
        if self._failed or teleop is None:
            return False, False
        try:
            # Resolve every call: the session can be replaced on a link hiccup.
            tracker, session = resolve_controller_tracker(teleop, "get_right_controller")
            if tracker is None or session is None:
                self.bound = False
                return False, False
            snapshot = tracker.get_right_controller(session)
            data = getattr(snapshot, "data", None)
            if data is None:
                return False, False  # controller asleep or out of the play space
            inputs = data.inputs
            if not self.bound:
                self.bound = True
                print("[buttons] right controller A/B are live - A start/pause, B reset")
            return bool(inputs.primary_click), bool(inputs.secondary_click)
        except Exception as exc:  # noqa: BLE001
            logger.info("session buttons unavailable, disabling: %s", exc)
            print(f"[buttons] OFF - could not read the right controller: {exc}")
            self._failed = True
            return False, False


def release_upstream_a_binding(teleop) -> bool:
    """Stop A from also toggling anchor rotation. Returns whether it worked.

    Applied to the manager instance, so other devices in the process are
    unaffected.
    """
    manager = getattr(teleop, "_anchor_manager", None)
    if manager is None or not hasattr(manager, "toggle_anchor_rotation"):
        logger.info("no anchor manager to unbind A from; upstream layout may have changed")
        return False
    manager.toggle_anchor_rotation = lambda: None
    return True


# --- Crouch ---
# Left controller: hold X to sink, Y to rise, both to stand. The height goes into action index 31
# every step.

DEFAULT_HIP_HEIGHT = 0.72
"""Standing pelvis height [m]."""

HIP_MIN = 0.40
"""Lowest hip height the retargeter clamps to [m]."""

HIP_MAX = 1.00
"""Highest hip height the retargeter clamps to [m]."""

CROUCH_MAX = DEFAULT_HIP_HEIGHT
"""Highest the crouch buttons will go [m]: standing."""

DEFAULT_RATE = 0.25
"""Metres per second at which a held button moves the pelvis."""

MAX_DT = 0.1
"""Longest step this will integrate over [s]."""


class CrouchControl:
    """Owns Robot A's hip height, driven by the left controller's X and Y.

    Poll :meth:`update` once per control step and write :attr:`height` into action
    index 31. ``rate`` is metres per second while a button is held.
    """

    def __init__(
        self,
        rate: float = DEFAULT_RATE,
        minimum: float = HIP_MIN,
        maximum: float = CROUCH_MAX,
    ):
        self.rate = rate
        self.minimum = min(max(minimum, HIP_MIN), HIP_MAX)
        self.maximum = min(max(maximum, HIP_MIN), HIP_MAX)
        if self.minimum > self.maximum:
            self.minimum, self.maximum = self.maximum, self.minimum
        self._height = DEFAULT_HIP_HEIGHT
        self._last_time: float | None = None
        self._failed = False
        self.bound = False

    @property
    def height(self) -> float:
        """Commanded hip height [m], action index 31 for Robot A."""
        return self._height

    @property
    def anchor_dz(self) -> float:
        """How far the XR anchor should drop to ride this crouch [m].

        Negative while crouched. See :func:`cohub.teleop.view.set_anchor_dz`.
        """
        return self._height - DEFAULT_HIP_HEIGHT

    def reset(self) -> None:
        """Back to standing, e.g. on an episode reset."""
        self._height = DEFAULT_HIP_HEIGHT
        self._last_time = None

    def update(self, teleop, now: float) -> float:
        """Advance the height by one step and return it.

        ``now`` is a wall clock reading (``time.time()``); the step is the gap
        since the previous call, capped at :data:`MAX_DT`.
        """
        down, up = self._buttons(teleop)

        dt = 0.0 if self._last_time is None else min(max(now - self._last_time, 0.0), MAX_DT)
        self._last_time = now

        if down and up:
            # Both buttons: return to standing.
            self._height = DEFAULT_HIP_HEIGHT
        elif down:
            self._height -= self.rate * dt
        elif up:
            self._height += self.rate * dt

        self._height = min(max(self._height, self.minimum), self.maximum)
        return self._height

    def _buttons(self, teleop) -> tuple[bool, bool]:
        """``(X pressed, Y pressed)`` on the left controller, ``(False, False)``
        when the controller cannot be read."""
        if self._failed:
            return False, False
        try:
            # Resolve every call: the session can be replaced on a link hiccup.
            tracker, session = resolve_controller_tracker(teleop)
            if tracker is None or session is None:
                self.bound = False
                return False, False
            snapshot = tracker.get_left_controller(session)
            data = getattr(snapshot, "data", None)
            if data is None:
                # Controller asleep or out of the play space.
                return False, False
            inputs = data.inputs
            if not self.bound:
                self.bound = True
                print("[crouch] left controller X/Y are live - hold X to sink, Y to rise")
            return bool(inputs.primary_click), bool(inputs.secondary_click)
        except Exception as exc:  # noqa: BLE001
            logger.info("crouch buttons unavailable, disabling: %s", exc)
            print(f"[crouch] OFF - could not read the left controller: {exc}")
            self._failed = True
            return False, False


# --- Raw operator input ---
# Controller and head poses off the live TeleopSession result, before retargeting.

# ControllerInput layout (isaacteleop.retargeting_engine.tensor_types):
#   0 grip position, 1 grip orientation, 2 grip valid, 3 aim position,
#   4 aim orientation, 5 aim valid, 6 primary click, 7 secondary click,
#   8 thumbstick x, 9 thumbstick y, 10 thumbstick click, 11 menu click,
#   12 squeeze value, 13 trigger value
_SCALAR_FIELDS = (
    ("primary_click", 6),
    ("secondary_click", 7),
    ("thumbstick_x", 8),
    ("thumbstick_y", 9),
    ("thumbstick_click", 10),
    ("menu_click", 11),
    ("squeeze", 12),
    ("trigger", 13),
)
_GRIP_POSITION = 0
_GRIP_ORIENTATION = 1
_GRIP_VALID = 2


def _as_list(value) -> list[float]:
    try:
        return [float(v) for v in value]
    except TypeError:
        return [float(value)]


def _controller_frame(group) -> dict[str, list[float]] | None:
    """One controller's raw state as flat float vectors."""
    if group is None or getattr(group, "is_none", True):
        return None
    frame = {
        "grip_pose": _as_list(group[_GRIP_POSITION]) + _as_list(group[_GRIP_ORIENTATION]),
        "grip_valid": [float(group[_GRIP_VALID])],
    }
    for name, index in _SCALAR_FIELDS:
        frame[name] = [float(group[index])]
    return frame


class OperatorInputCapture:
    """Pulls raw controller and head poses out of the live teleop session."""

    def __init__(self):
        self._failed = False

    def capture(self, teleop) -> dict[str, list[float]]:
        """Return a flat ``{name: vector}`` snapshot, empty if unavailable."""
        if self._failed:
            return {}
        try:
            session = resolve_teleop_session(teleop)
            if session is None:
                return {}
            result = getattr(session, "last_result", None)
            snapshot: dict[str, list[float]] = {}

            groups = {}
            if isinstance(result, dict):
                for key, value in result.items():
                    lowered = str(key).lower()
                    if "controller" in lowered and "left" in lowered:
                        groups["left"] = value
                    elif "controller" in lowered and "right" in lowered:
                        groups["right"] = value
                    elif "head" in lowered:
                        groups["head"] = value
            lifecycle = getattr(teleop, "_session_lifecycle", teleop)
            cached_right = getattr(lifecycle, "last_right_controller", None)
            if "right" not in groups and cached_right is not None:
                groups["right"] = cached_right

            for side in ("left", "right"):
                frame = _controller_frame(groups.get(side))
                if frame:
                    for name, values in frame.items():
                        snapshot[f"controller_{side}_{name}"] = values

            head = groups.get("head")
            if head is not None and not getattr(head, "is_none", True):
                snapshot["head_pose"] = _as_list(head[0]) + _as_list(head[1])

            return snapshot
        except Exception as exc:  # noqa: BLE001
            logger.info("operator input capture unavailable, disabling: %s", exc)
            self._failed = True
            return {}


# --- Session rebuild guard ---
# Upstream rebuilds the teleop session on every pipeline exception, and an untracked controller raises
# every frame. This allows one rebuild per backoff interval.

# Seconds.
DEFAULT_BACKOFF_S = 1.0
DEFAULT_MAX_BACKOFF_S = 5.0


class XrSessionGuard:
    """Rate-limits IsaacTeleop session rebuilds after a failed step.

    Install with :func:`guard_xr_session`. It replaces three bound methods on the
    lifecycle instance and leaves the class untouched.
    """

    def __init__(
        self,
        lifecycle,
        backoff: float = DEFAULT_BACKOFF_S,
        max_backoff: float = DEFAULT_MAX_BACKOFF_S,
        report: Callable[[str], None] = print,
    ):
        self._lifecycle = lifecycle
        self._min_backoff = backoff
        self._max_backoff = max_backoff
        self._backoff = backoff
        self._blocked_until = 0.0
        self._report = report

        # rebuilds: sessions rebuilt. suppressed: rebuilds the guard blocked.
        self.rebuilds = 0
        self.suppressed = 0
        self._storm_open = False

        self._orig_teardown = lifecycle._teardown_dead_session
        self._orig_try_start = lifecycle._try_start_session
        self._orig_step = lifecycle.step

    # Installed hooks
    def _teardown(self) -> None:
        self._orig_teardown()
        self.rebuilds += 1
        self._blocked_until = time.monotonic() + self._backoff
        if not self._storm_open:
            self._storm_open = True
            self._report(
                "[xr] teleop session dropped -- most often a controller or headset that "
                "stopped reporting a pose.\n"
                "     Retrying about once a second instead of every frame; the sim keeps "
                "running and Robot B stays controllable.\n"
                "     Pick the controllers back up, or re-enter the client tab, and it "
                "reconnects on its own."
            )
        # Doubles while failures keep arriving.
        self._backoff = min(self._backoff * 2.0, self._max_backoff)

    def _try_start(self) -> bool:
        # A live session always reaches the original.
        if self._lifecycle._session is not None:
            return self._orig_try_start()
        if time.monotonic() < self._blocked_until:
            self.suppressed += 1
            return False
        return self._orig_try_start()

    def _step(self, *args, **kwargs):
        action = self._orig_step(*args, **kwargs)
        if action is not None and self._storm_open:
            self._storm_open = False
            self._backoff = self._min_backoff
            self._report(f"[xr] teleop session back after {self.rebuilds} reconnect(s)")
        return action

    def summary(self) -> str | None:
        """One line for the session teardown, or ``None`` if nothing failed."""
        if not self.rebuilds:
            return None
        return (
            f"[xr] teleop session was rebuilt {self.rebuilds} time(s); "
            f"{self.suppressed} same-frame rebuild(s) suppressed"
        )


def guard_xr_session(
    device,
    backoff: float = DEFAULT_BACKOFF_S,
    max_backoff: float = DEFAULT_MAX_BACKOFF_S,
    report: Callable[[str], None] = print,
) -> XrSessionGuard | None:
    """Guard ``device``'s session rebuilds, if it has a lifecycle to guard.

    Returns ``None`` for a device that does not carry a
    ``TeleopSessionLifecycle``, so callers can install unconditionally.
    """
    lifecycle = getattr(device, "_session_lifecycle", None)
    if lifecycle is None:
        return None
    if not all(
        hasattr(lifecycle, name)
        for name in ("_teardown_dead_session", "_try_start_session", "step", "_session")
    ):
        # Upstream API changed: report and continue without the guard.
        report("[xr] session-rebuild guard not installed: upstream lifecycle API changed")
        return None

    guard = XrSessionGuard(lifecycle, backoff=backoff, max_backoff=max_backoff, report=report)
    lifecycle._teardown_dead_session = guard._teardown
    lifecycle._try_start_session = guard._try_start
    lifecycle.step = guard._step
    return guard
