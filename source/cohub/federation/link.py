"""The wire between the recording process and a mirror: two UDP datagram formats and the endpoints that exchange them."""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
import logging
import socket
import struct
import time

import numpy as np

from cohub.keys import TELEOP_ACTION_DIM


# --- Wire format ---
# primary -> mirror: StatePacket (where every body is). mirror -> primary: ActionPacket. Fixed-layout
# little-endian UDP datagrams; quaternions are xyzw and poses are in the env frame.

MAGIC = b"MHF1"
"""Protocol magic. Bump the trailing digit on any incompatible layout change."""

VERSION = 2
"""Wire version. A peer on another version is rejected."""

KIND_STATE = 1
KIND_ACTION = 2


OPERATOR_RAW_DIM = 48
"""Length of the zero-padded raw operator input vector."""

POSE_DIM = 7
"""Position (3) + quaternion xyzw (4)."""

TASK_FIELD = 32
"""Bytes reserved for the task id in the header, zero-padded. Longer ids are
truncated."""


# State flags (primary -> secondary)
STATE_RUNNING = 1 << 0
"""The primary is stepping physics and recording, not idling."""

STATE_SUCCESS = 1 << 1
"""The episode that just ended met the task's success condition."""

STATE_DISCARDED = 1 << 2
"""The episode that just ended did not, and was reset by hand.

Both outcome bits are levels: the primary holds one set for a few seconds.
"""


# Action flags (secondary -> primary)
ACTION_TELEOP_ONLINE = 1 << 0
"""The secondary's own XR session is live; without this the action is stale."""


class ProtocolError(ValueError):
    """A datagram could not be decoded as the packet it claimed to be."""


# magic, version, kind, two layout dimensions that depend on kind, seq, ack,
# flags, the sender's wall clock, two edge counters, the sender's task id.
# Start/pause and reset are counters so a request survives a dropped datagram.
_HEADER = struct.Struct(f"<4sHHHHIIIdHH{TASK_FIELD}s")
_HEADER_SIZE = _HEADER.size  # 68 bytes


def _pack_task(task: str) -> bytes:
    """Task id as the fixed-width field, truncated on a byte boundary."""
    return task.encode("utf-8", "replace")[:TASK_FIELD]


def _unpack_task(raw: bytes) -> str:
    return raw.rstrip(b"\0").decode("utf-8", "replace")


def _unpack_header(payload: bytes, expected_kind: int) -> tuple[int, int, int, int, int, float, int, int, str]:
    if len(payload) < _HEADER_SIZE:
        raise ProtocolError(f"datagram too short: {len(payload)} < {_HEADER_SIZE} bytes")
    magic, version, kind, dim_a, dim_b, seq, aux, flags, send_time, count_a, count_b, task = _HEADER.unpack_from(
        payload
    )
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic!r}")
    if version != VERSION:
        raise ProtocolError(f"protocol version {version} != {VERSION}")
    if kind != expected_kind:
        raise ProtocolError(f"kind {kind} != {expected_kind}")
    return dim_a, dim_b, seq, aux, flags, send_time, count_a, count_b, _unpack_task(task)


def _floats(payload: bytes, count: int) -> np.ndarray:
    """Decode the float32 body. Rejects any length but the exact one."""
    expected = _HEADER_SIZE + count * 4
    if len(payload) != expected:
        raise ProtocolError(f"body length {len(payload) - _HEADER_SIZE} != {count * 4} bytes")
    return np.frombuffer(payload, dtype="<f4", count=count, offset=_HEADER_SIZE)


@dataclass
class StatePacket:
    """Everything the secondary needs to re-pose its copy of the scene.

    Poses are in the env frame, position [m] + xyzw. ``seq`` is monotonic (older
    packets are dropped), ``episode_id`` increments on reset, ``flags`` is a
    ``STATE_*`` bitfield and ``extras`` holds every other dynamic body
    (:func:`cohub.federation.scene_state.extra_state`).
    """

    seq: int
    episode_id: int
    send_time: float
    root_a: np.ndarray
    joint_q_a: np.ndarray
    root_b: np.ndarray
    joint_q_b: np.ndarray
    object_pose: np.ndarray
    haptic: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    extras: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    task: str = ""
    flags: int = 0

    @property
    def running(self) -> bool:
        return bool(self.flags & STATE_RUNNING)

    @property
    def outcome(self) -> str | None:
        """``"success"``, ``"discarded"`` or ``None`` if no episode just ended."""
        if self.flags & STATE_SUCCESS:
            return "success"
        if self.flags & STATE_DISCARDED:
            return "discarded"
        return None

    @property
    def num_joints(self) -> int:
        return int(self.joint_q_a.shape[0])

    def encode(self) -> bytes:
        n = self.num_joints
        if self.joint_q_b.shape[0] != n:
            raise ProtocolError(f"robot joint counts differ: {n} vs {self.joint_q_b.shape[0]}")
        extras = np.asarray(self.extras, dtype="<f4").reshape(-1)
        if extras.shape[0] > 0xFFFF:
            raise ProtocolError(f"extras is {extras.shape[0]} floats, past the {0xFFFF} the header can carry")
        header = _HEADER.pack(
            MAGIC,
            VERSION,
            KIND_STATE,
            n,
            extras.shape[0],
            self.seq & 0xFFFFFFFF,
            self.episode_id,
            self.flags,
            self.send_time,
            0,
            0,
            _pack_task(self.task),
        )
        body = np.concatenate(
            (
                np.asarray(self.root_a, dtype="<f4").reshape(-1),
                np.asarray(self.joint_q_a, dtype="<f4").reshape(-1),
                np.asarray(self.root_b, dtype="<f4").reshape(-1),
                np.asarray(self.joint_q_b, dtype="<f4").reshape(-1),
                np.asarray(self.object_pose, dtype="<f4").reshape(-1),
                np.asarray(self.haptic, dtype="<f4").reshape(-1),
                extras,
            )
        )
        expected = 2 * (POSE_DIM + n) + POSE_DIM + 2 + extras.shape[0]
        if body.shape[0] != expected:
            raise ProtocolError(f"state body is {body.shape[0]} floats, expected {expected}")
        return header + body.astype("<f4").tobytes()

    @classmethod
    def decode(cls, payload: bytes) -> StatePacket:
        n, num_extras, seq, episode_id, flags, send_time, _, _, task = _unpack_header(payload, KIND_STATE)
        values = _floats(payload, 2 * (POSE_DIM + n) + POSE_DIM + 2 + num_extras)
        cursor = 0

        def take(count: int) -> np.ndarray:
            nonlocal cursor
            chunk = np.array(values[cursor : cursor + count], dtype=np.float32)
            cursor += count
            return chunk

        return cls(
            seq=seq,
            episode_id=episode_id,
            send_time=send_time,
            flags=flags,
            task=task,
            root_a=take(POSE_DIM),
            joint_q_a=take(n),
            root_b=take(POSE_DIM),
            joint_q_b=take(n),
            object_pose=take(POSE_DIM),
            haptic=take(2),
            extras=take(num_extras),
        )


@dataclass
class ActionPacket:
    """Operator 2's contribution, as produced by the secondary's XR pipeline.

    ``action`` is the robot's 32D slice. ``ack_state_seq`` is the newest state the
    sender saw, for the round-trip measurement. ``toggle_run_count`` and
    ``reset_count`` wrap at 2**16; the receiver reacts to increments.
    """

    seq: int
    ack_state_seq: int
    send_time: float
    action: np.ndarray
    operator_raw: np.ndarray = field(default_factory=lambda: np.zeros(OPERATOR_RAW_DIM, dtype=np.float32))
    flags: int = 0
    toggle_run_count: int = 0
    reset_count: int = 0

    @property
    def teleop_online(self) -> bool:
        return bool(self.flags & ACTION_TELEOP_ONLINE)

    def encode(self) -> bytes:
        action = np.asarray(self.action, dtype="<f4").reshape(-1)
        if action.shape[0] != TELEOP_ACTION_DIM:
            raise ProtocolError(f"action is {action.shape[0]}D, expected {TELEOP_ACTION_DIM}D")
        raw = np.asarray(self.operator_raw, dtype="<f4").reshape(-1)
        if raw.shape[0] != OPERATOR_RAW_DIM:
            raise ProtocolError(f"operator_raw is {raw.shape[0]}D, expected {OPERATOR_RAW_DIM}D")
        header = _HEADER.pack(
            MAGIC,
            VERSION,
            KIND_ACTION,
            TELEOP_ACTION_DIM,
            OPERATOR_RAW_DIM,
            self.seq & 0xFFFFFFFF,
            self.ack_state_seq & 0xFFFFFFFF,
            self.flags,
            self.send_time,
            self.toggle_run_count & 0xFFFF,
            self.reset_count & 0xFFFF,
            # The task id travels on the state packet only.
            b"",
        )
        return header + np.concatenate((action, raw)).astype("<f4").tobytes()

    @classmethod
    def decode(cls, payload: bytes) -> ActionPacket:
        action_dim, raw_dim, seq, ack, flags, send_time, toggle_run, reset, _ = _unpack_header(payload, KIND_ACTION)
        if action_dim != TELEOP_ACTION_DIM or raw_dim != OPERATOR_RAW_DIM:
            raise ProtocolError(f"layout ({action_dim}, {raw_dim}) != ({TELEOP_ACTION_DIM}, {OPERATOR_RAW_DIM})")
        values = _floats(payload, action_dim + raw_dim)
        return cls(
            seq=seq,
            ack_state_seq=ack,
            send_time=send_time,
            flags=flags,
            toggle_run_count=toggle_run,
            reset_count=reset,
            action=np.array(values[:action_dim], dtype=np.float32),
            operator_raw=np.array(values[action_dim:], dtype=np.float32),
        )


def pack_operator_raw(groups: dict[str, list[float]]) -> np.ndarray:
    """Flatten ``OperatorInputCapture`` groups into the fixed-width vector.

    Keys are sorted. The result is truncated or zero-padded to
    :data:`OPERATOR_RAW_DIM`.
    """
    flat: list[float] = []
    for key in sorted(groups):
        flat.extend(float(v) for v in groups[key])
    out = np.zeros(OPERATOR_RAW_DIM, dtype=np.float32)
    take = min(len(flat), OPERATOR_RAW_DIM)
    out[:take] = flat[:take]
    return out


# --- Endpoints ---
# Receives never block: each poll drains the socket and keeps the newest packet. A quiet peer yields
# the last packet plus a `stale` flag.

logger = logging.getLogger(__name__)

DEFAULT_STATE_PORT = 47701
"""Where the secondary listens for the primary's scene state."""

DEFAULT_ACTION_PORT = 47702
"""Where the primary listens for the secondary's action slice."""

ROBOT_A_STATE_PORT = 47711
ROBOT_A_ACTION_PORT = 47712
"""Robot A's mirror. Each mirror needs its own port pair."""

ROBOT_C_STATE_PORT = 47721
ROBOT_C_ACTION_PORT = 47722
"""Robot C's mirror."""

_PORTS_BY_ROBOT = {
    "robot_a": (ROBOT_A_STATE_PORT, ROBOT_A_ACTION_PORT),
    "robot_b": (DEFAULT_STATE_PORT, DEFAULT_ACTION_PORT),
    "robot_c": (ROBOT_C_STATE_PORT, ROBOT_C_ACTION_PORT),
}


def ports_for_robot(robot: str) -> tuple[int, int]:
    """Return ``(state_port, action_port)`` for the robot a mirror drives.

    Args:
        robot: ``"robot_a"``, ``"robot_b"`` or ``"robot_c"``.
    """
    try:
        return _PORTS_BY_ROBOT[robot]
    except KeyError:
        raise ValueError(
            f"robot must be one of {', '.join(sorted(_PORTS_BY_ROBOT))}, got {robot!r}"
        ) from None

DEFAULT_STALE_AFTER = 0.5
"""Seconds without a packet before the peer counts as offline."""

_RECV_BUFFER = 65536
_SOCKET_RCVBUF = 1 << 20
"""Kernel receive buffer, 1 MiB."""


def _is_newer(candidate: int, current: int) -> bool:
    """Wraparound-safe ``candidate > current`` for 32-bit sequence numbers."""
    return ((candidate - current) & 0xFFFFFFFF) < 0x80000000 and candidate != current


class _Endpoint:
    """Shared socket plumbing: one bound receive port, one send target."""

    def __init__(self, bind_port: int, peer_host: str, peer_port: int, stale_after: float, role: str = "process"):
        self._peer = (peer_host, peer_port)
        self.stale_after = stale_after

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SOCKET_RCVBUF)
        try:
            self._sock.bind(("0.0.0.0", bind_port))
        except OSError as exc:
            self._sock.close()
            if exc.errno != errno.EADDRINUSE:
                raise
            raise OSError(
                f"UDP port {bind_port} is already in use, so this {role} cannot receive. "
                f"Another {role} is probably still running -- check with "
                f"`ss -lunp | grep {bind_port}` and stop it, or pass different ports "
                f"to both processes."
            ) from exc
        self._sock.setblocking(False)

        self._tx_seq = 0
        self._rx_seq = 0
        self._last_rx_monotonic = 0.0
        self._decode_errors = 0

    def _next_seq(self) -> int:
        self._tx_seq = (self._tx_seq + 1) & 0xFFFFFFFF
        return self._tx_seq

    def _send(self, payload: bytes) -> None:
        try:
            self._sock.sendto(payload, self._peer)
        except OSError as exc:
            # ECONNREFUSED is normal before the peer starts.
            logger.debug(f"federation send failed: {exc}")

    def _drain(self, decode) -> object | None:
        """Read everything queued, return the newest decodable packet."""
        newest = None
        while True:
            try:
                payload = self._sock.recv(_RECV_BUFFER)
            except BlockingIOError:
                break
            except OSError as exc:
                logger.debug(f"federation recv failed: {exc}")
                break
            try:
                packet = decode(payload)
            except ProtocolError as exc:
                self._decode_errors += 1
                if self._decode_errors <= 5:
                    logger.warning(f"federation dropped a malformed packet: {exc}")
                continue
            if not _is_newer(packet.seq, self._rx_seq):
                continue  # reordered or replayed
            if newest is None or _is_newer(packet.seq, newest.seq):
                newest = packet
        if newest is not None:
            self._rx_seq = newest.seq
            self._last_rx_monotonic = time.monotonic()
        return newest

    @property
    def stale(self) -> bool:
        """True when nothing has arrived recently (including: ever)."""
        if self._last_rx_monotonic == 0.0:
            return True
        return (time.monotonic() - self._last_rx_monotonic) > self.stale_after

    @property
    def seconds_since_last_packet(self) -> float:
        if self._last_rx_monotonic == 0.0:
            return float("inf")
        return time.monotonic() - self._last_rx_monotonic

    def close(self) -> None:
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class ResetRelay:
    """Forwards operator 2's reset requests to the primary.

    Drops two kinds of request that no person made: the echo raised by
    following the primary's reset, and the resets the teleop pipeline raises
    while a session comes up (see :meth:`settle`).
    """

    ECHO_WINDOW = 1.0
    """Seconds after following the primary in which a request is our own echo."""

    SETTLE_WINDOW = 2.0
    """Seconds after a session comes up in which resets are the pipeline's."""

    def __init__(self, link: SecondaryLink):
        self._link = link
        self._echo_until = 0.0
        # The first session comes up before the loop can observe it.
        self._settle_until = time.monotonic() + self.SETTLE_WINDOW

    def settle(self) -> None:
        """Note that this operator's session just came up (or came back)."""
        self._settle_until = time.monotonic() + self.SETTLE_WINDOW

    def follow_primary(self, teleop) -> None:
        """Reset ``teleop`` for the primary's new episode, silently.

        Args:
            teleop: The device to reset; anything with a ``reset()``.
        """
        self._echo_until = time.monotonic() + self.ECHO_WINDOW
        teleop.reset()

    def request(self) -> None:
        """The ``"RESET"`` callback: operator 2's panel, or something else."""
        now = time.monotonic()
        if now < self._settle_until:
            return  # the pipeline's own reset
        if now < self._echo_until:
            self._echo_until = 0.0  # one follow, one echo
            return
        self._link.request_reset()


class PrimaryLink(_Endpoint):
    """The simulation owner's end: publishes state, consumes Robot B's action.

        with PrimaryLink() as link:
            link.send_state(...)
            packet = link.poll_action()  # None if nothing new
    """

    def __init__(
        self,
        peer_host: str = "127.0.0.1",
        state_port: int = DEFAULT_STATE_PORT,
        action_port: int = DEFAULT_ACTION_PORT,
        stale_after: float = DEFAULT_STALE_AFTER,
    ):
        super().__init__(
            bind_port=action_port,
            peer_host=peer_host,
            peer_port=state_port,
            stale_after=stale_after,
            role="primary (recording) process",
        )
        self.last_action: ActionPacket | None = None
        self.rtt: float = 0.0
        """Round-trip time of the newest acknowledged state packet [s]."""

        self._send_times: dict[int, float] = {}
        self._toggle_run_count = 0
        self._reset_count = 0
        self._seen_counts = False
        self._outcome_flag = 0
        self._outcome_until = 0.0

    OUTCOME_WINDOW = 2.5
    """How long an outcome bit stays set on outgoing state packets [s]."""

    def announce_outcome(self, outcome: str | None) -> None:
        """Tell the secondary how the episode that just ended went.

        Args:
            outcome: ``"success"``, ``"discarded"``, or ``None`` to clear.
        """
        self._outcome_flag = {"success": STATE_SUCCESS, "discarded": STATE_DISCARDED}.get(outcome or "", 0)
        self._outcome_until = time.monotonic() + self.OUTCOME_WINDOW if self._outcome_flag else 0.0

    def _outcome_bits(self) -> int:
        if self._outcome_flag and time.monotonic() >= self._outcome_until:
            self._outcome_flag = 0
        return self._outcome_flag

    def send_state(
        self,
        *,
        root_a: np.ndarray,
        joint_q_a: np.ndarray,
        root_b: np.ndarray,
        joint_q_b: np.ndarray,
        object_pose: np.ndarray,
        haptic: np.ndarray | None = None,
        extras: np.ndarray | None = None,
        episode_id: int = 0,
        running: bool = False,
        task: str = "",
    ) -> int:
        """Publish one frame of scene state. Returns the sequence number sent.

        ``haptic`` is the load on the hands of this link's robot.
        """
        seq = self._next_seq()
        packet = StatePacket(
            seq=seq,
            episode_id=episode_id,
            send_time=time.time(),
            flags=(STATE_RUNNING if running else 0) | self._outcome_bits(),
            root_a=root_a,
            joint_q_a=joint_q_a,
            root_b=root_b,
            joint_q_b=joint_q_b,
            object_pose=object_pose,
            haptic=np.zeros(2, dtype=np.float32) if haptic is None else haptic,
            extras=np.zeros(0, dtype=np.float32) if extras is None else extras,
            task=task,
        )
        # Local send times give the round trip without clock agreement.
        self._send_times[seq] = time.monotonic()
        while len(self._send_times) > 512:
            # Dicts keep insertion order: the first key is the oldest send.
            del self._send_times[next(iter(self._send_times))]
        self._send(packet.encode())
        return seq

    def poll_action(self) -> ActionPacket | None:
        """Newest action received since the last call, or ``None``."""
        packet = self._drain(ActionPacket.decode)
        if packet is None:
            return None
        self.last_action = packet

        sent_at = self._send_times.pop(packet.ack_state_seq, None)
        if sent_at is not None:
            self.rtt = time.monotonic() - sent_at

        # The first packet sets the baseline, so a reconnecting secondary
        # does not trigger a reset.
        if not self._seen_counts:
            self._toggle_run_count = packet.toggle_run_count
            self._reset_count = packet.reset_count
            self._seen_counts = True
        return packet

    def consume_control_flags(self) -> tuple[bool, bool]:
        """Return ``(toggle_run, reset)`` requested by operator 2 since the last call.

        Same signature as
        :meth:`cohub.teleop.follower.FollowerDevice.consume_control_flags`.
        """
        packet = self.last_action
        if packet is None:
            return False, False
        toggle = packet.toggle_run_count != self._toggle_run_count
        reset = packet.reset_count != self._reset_count
        self._toggle_run_count = packet.toggle_run_count
        self._reset_count = packet.reset_count
        return toggle, reset

    @property
    def secondary_online(self) -> bool:
        """The peer is sending *and* its own XR session is live."""
        return not self.stale and self.last_action is not None and self.last_action.teleop_online


class SecondaryLink(_Endpoint):
    """The mirror process's end: consumes state, publishes Robot B's action."""

    def __init__(
        self,
        peer_host: str = "127.0.0.1",
        state_port: int = DEFAULT_STATE_PORT,
        action_port: int = DEFAULT_ACTION_PORT,
        stale_after: float = DEFAULT_STALE_AFTER,
    ):
        super().__init__(
            bind_port=state_port,
            peer_host=peer_host,
            peer_port=action_port,
            stale_after=stale_after,
            role="secondary (second headset) process",
        )
        self.last_state: StatePacket | None = None
        self._toggle_run_count = 0
        self._reset_count = 0

    def poll_state(self) -> StatePacket | None:
        """Newest scene state received since the last call, or ``None``."""
        packet = self._drain(StatePacket.decode)
        if packet is not None:
            self.last_state = packet
        return packet

    def request_toggle_run(self) -> None:
        self._toggle_run_count = (self._toggle_run_count + 1) & 0xFFFF

    def request_run_state(self, active: bool) -> None:
        """Ask the primary to be running or paused.

        Sends a toggle only when the primary's reported state differs.
        """
        state = self.last_state
        if state is not None and state.running == active:
            return
        self.request_toggle_run()

    def request_reset(self) -> None:
        self._reset_count = (self._reset_count + 1) & 0xFFFF

    def send_action(
        self,
        action: np.ndarray,
        *,
        teleop_online: bool,
        operator_raw: dict[str, list[float]] | np.ndarray | None = None,
    ) -> int:
        """Publish operator 2's 32D slice. Returns the sequence number sent."""
        if operator_raw is None:
            raw = pack_operator_raw({})
        elif isinstance(operator_raw, dict):
            raw = pack_operator_raw(operator_raw)
        else:
            raw = np.asarray(operator_raw, dtype=np.float32)

        seq = self._next_seq()
        packet = ActionPacket(
            seq=seq,
            # Newest state seen, so the primary can measure the round trip.
            ack_state_seq=self.last_state.seq if self.last_state is not None else 0,
            send_time=time.time(),
            action=action,
            operator_raw=raw,
            flags=ACTION_TELEOP_ONLINE if teleop_online else 0,
            toggle_run_count=self._toggle_run_count,
            reset_count=self._reset_count,
        )
        self._send(packet.encode())
        return seq
