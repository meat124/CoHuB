#!/usr/bin/env bash
# Start a collection session: every mirror process plus the recording process.
# The recording process stays in this terminal. The mirrors stop when it exits.
#
#   ./scripts/teleop/collect.sh solo               # you drive Robot A, a keyboard follower drives B
#                                           #   (Robot C holds still on a G1x3 task)
#   ./scripts/teleop/collect.sh dual               # two headsets, three processes
#   ./scripts/teleop/collect.sh triple             # three headsets, four processes (a G1x3 task)
#   ./scripts/teleop/collect.sh --task cocarry     # as many headsets as the task has humanoids
#   ./scripts/teleop/collect.sh --task cocarry --agents 1    # or a number of agents: 1, 2 or 3
#
#   ./scripts/teleop/collect.sh <layout> --print   # each process's command, to start by hand; starts nothing
#   ./scripts/teleop/collect.sh check              # what is holding the teleop ports; stops nothing
#   ./scripts/teleop/collect.sh cleanup [primary|secondary|tertiary ...|--all]
#                                           # kill an orphaned CloudXR runtime (default: primary)
#
# Options (everything else is passed to record_demos.py):
#   --task <name>      task for every process (default cocarry, or movehouse
#                      in the triple layout)
#   --agents <n>     the number of agents, one headset each: 1, 2 or 3, in place of solo, dual or triple
#   --no-save          practice session, nothing is written
#   --enable_cameras   also record the session's live camera frames
#   --no-ego_frame     do not draw the ego camera frame in the headsets
#   --ego_frame_distance <m>, --ego_frame_color <name>
#                      where the frame is drawn and its colour (1.0, red)
#
# A session records state, not pixels. Rebuild the images afterwards:
#
#   python scripts/dataset/render_dataset_images.py datasets/<task>/data \
#       --headless --enable_cameras --device cpu
#
# Headset pages:
#
#   Quest A (operator 1)  -> http://<this PC>:8080
#   Quest B (operator 2)  -> http://<this PC>:8081
#   Quest C (operator 3)  -> http://<this PC>:8082    (triple only)
#
# Connect the headsets after this script reports that every process is up.
# The triple layout needs about 13 GiB of VRAM.
# Each mirror logs to logs/<mirror>/latest-xr_mirror.log.
set -e
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES} PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/source${PYTHONPATH:+:$PYTHONPATH}"

# Kill a profile's orphaned CloudXR runtime and clear its stale state; --list only reports.
cloudxr_cleanup() {
  .venv/bin/python -m cohub.teleop.cloudxr "$@"
}

check_ports() {
  echo "UDP federation ports (47711/47712 Robot A's link, 47701/47702 Robot B's, 47721/47722 Robot C's):"
  ss -lunp 2>/dev/null | grep -E "4770[12]|4771[12]|4772[12]" || echo "  free"
  echo
  echo "CloudXR TCP ports (primary 48322/49100/8080, secondary 48332/49110/8081, tertiary 48342/49120/8082):"
  ss -ltnp 2>/dev/null | grep -E "48322|49100|48332|49110|48342|49120|:8080|:8081|:8082" || echo "  free"
  echo
  # UDP, bound only while a headset is streaming.
  echo "CloudXR streaming ports (primary 47998/udp, secondary 48998/udp, tertiary 49998/udp):"
  ss -lunp 2>/dev/null | grep -E "47998|48998|49998" || echo "  free"
  echo
  echo "CloudXR runtimes:"
  # --list only: --all stops the runtimes.
  cloudxr_cleanup --list 2>/dev/null || true
  echo
  echo "(nothing above was stopped. To actually clear one: ./scripts/teleop/collect.sh cleanup primary|secondary|tertiary)"
}

usage() {
  # Print the header above as usage.
  sed -n '2,/^set -e/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 1
}
MODE="${1:-}"
case "$MODE" in
  solo|dual|triple) shift ;;
  check) check_ports; exit 0 ;;
  cleanup) shift; cloudxr_cleanup "$@"; exit ;;
  --*) MODE="" ;;
  *) usage ;;
esac
case "$MODE" in ""|-h|--help) [ $# -gt 0 ] || usage ;; esac

TASK=""
AGENTS=""
EXTRA=()
# Ego frame options go to every mirror, not to the recording process.
FRAME_ARGS=()
PRINT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --agents) AGENTS="$2"; shift 2 ;;
    -h|--help) usage ;;
    --print) PRINT=1; shift ;;
    --no-ego_frame|--ego_frame) FRAME_ARGS+=("$1"); shift ;;
    --ego_frame_distance|--ego_frame_color) FRAME_ARGS+=("$1" "$2"); shift 2 ;;
    --ego_frame_distance=*|--ego_frame_color=*) FRAME_ARGS+=("$1"); shift ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

# The layout: named first, or from --agents, or from the task's humanoids.
if [ -z "$MODE" ]; then
  [ -n "$AGENTS" ] || AGENTS=$(python3 -c "import sys; from cohub.task_registry import TASKS, gym_id; print(TASKS[gym_id(sys.argv[1])]['robots'])" "${TASK:-cocarry}") || exit 2
  case "$AGENTS" in
    1) MODE=solo ;;
    2) MODE=dual ;;
    3) MODE=triple ;;
    *) echo "[collect] --agents $AGENTS: expected 1, 2 or 3" >&2; exit 2 ;;
  esac
fi
if [ -z "$TASK" ]; then
  if [ "$MODE" = triple ]; then TASK=movehouse; else TASK=cocarry; fi
fi

MIRRORS=()
MIRROR_NAMES=()
SESSION_PROFILES=()

# Must match profile_for_robot() and the profile ports in
# source/cohub/teleop/cloudxr.py.
profile_for_robot() {
  case "$1" in
    robot_a) echo primary ;;
    robot_b) echo secondary ;;
    robot_c) echo tertiary ;;
  esac
}

profile_ports() {
  case "$1" in
    primary)   echo "49100 8080" ;;
    secondary) echo "49110 8081" ;;
    tertiary)  echo "49120 8082" ;;
  esac
}

spawn_mirror() {
  local name="$1"; shift
  # Record the mirror's profile for cleanup(). --robot defaults to robot_b.
  local robot=robot_b prev=
  local arg
  for arg in "$@"; do
    if [ "$prev" = "--robot" ]; then robot="$arg"; fi
    prev="$arg"
  done
  local cmd=(.venv/bin/python scripts/teleop/xr_mirror.py --xr --task "$TASK" "${FRAME_ARGS[@]}" "$@")
  if [ "$PRINT" = 1 ]; then
    echo "# $name ($(profile_for_robot "$robot") profile)"
    printf '%q ' "${cmd[@]}"; echo
    return
  fi
  mkdir -p "logs/$name"
  # setsid: Ctrl+C reaches only the recording process. cleanup() stops the
  # mirrors by process group.
  setsid "${cmd[@]}" > "logs/$name/latest-xr_mirror.log" 2>&1 &
  MIRRORS+=("$!")
  MIRROR_NAMES+=("$name")
  SESSION_PROFILES+=("$(profile_for_robot "$robot")")
  echo "  $name up (pid $!) -- transcript: logs/$name/latest-xr_mirror.log"
}

# True once no mirror process group has any member left.
mirrors_gone() {
  local pid
  for pid in "${MIRRORS[@]}"; do
    if kill -0 -- "-$pid" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

wait_for_mirrors() {
  local deadline=$((SECONDS + $1))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if mirrors_gone; then return 0; fi
    sleep 0.5
  done
  return 1
}

cleanup() {
  [ "${#MIRRORS[@]}" -eq 0 ] && return
  echo "[collect] stopping ${#MIRRORS[@]} mirror(s)..."

  # SIGKILL follows if SIGTERM is ignored.
  local stage sig grace pid
  for stage in TERM:30 KILL:10; do
    sig="${stage%%:*}"; grace="${stage#*:}"
    if mirrors_gone; then break; fi
    echo "[collect]   SIG$sig (up to ${grace}s)..."
    for pid in "${MIRRORS[@]}"; do kill -"$sig" -- "-$pid" 2>/dev/null || true; done
    wait_for_mirrors "$grace" || true
  done

  # Release the CloudXR runtimes of this session's profiles only.
  if [ "${#SESSION_PROFILES[@]}" -gt 0 ]; then
    cloudxr_cleanup "${SESSION_PROFILES[@]}" || true
  fi

  # Warn if a teleop port is still held.
  local ports=() prof held
  for prof in "${SESSION_PROFILES[@]}"; do ports+=($(profile_ports "$prof")); done
  if [ "${#ports[@]}" -gt 0 ]; then
    local re; re="$(IFS='|'; echo "${ports[*]}")"
    held="$(ss -ltnp 2>/dev/null | grep -E ":(${re})[[:space:]]" || true)"
    if [ -n "$held" ]; then
      echo "[collect] warning: a teleop port is still held after cleanup --"
      printf '%s\n' "$held" | sed 's/^/    /'
      echo "[collect] the next session will refuse to start; clear it with:"
      echo "[collect]     ./scripts/teleop/collect.sh cleanup --all"
    fi
  fi

  if mirrors_gone; then
    echo "[collect] mirrors stopped, runtimes released."
  else
    echo "[collect] warning: a mirror survived SIGKILL; check: ps -ef | grep xr_mirro[r]"
  fi
}
trap cleanup EXIT
# Survive Ctrl+C so that cleanup runs. The recording process handles the signal itself.
trap ':' INT

case "$MODE" in
  solo)
    spawn_mirror mirror_a --robot robot_a
    # Robots without a mirror hold a local pose.
    PRIMARY_ARGS=(--robot_a_source federated --robot_b_source follower --robot_c_source follower)
    echo "  your Quest opens http://<this PC>:8080 once the mirror's banner is in its transcript"
    ;;
  dual)
    # Robot A's mirror must start last.
    spawn_mirror mirror_b
    spawn_mirror mirror_a --robot robot_a
    PRIMARY_ARGS=(--robot_a_source federated --robot_c_source follower)
    echo "  Quest A -> http://<this PC>:8080    Quest B -> http://<this PC>:8081"
    ;;
  triple)
    # Robot A's mirror must start last.
    spawn_mirror mirror_c --robot robot_c
    spawn_mirror mirror_b
    spawn_mirror mirror_a --robot robot_a
    PRIMARY_ARGS=(--robot_a_source federated)
    echo "  Quest A -> http://<this PC>:8080    Quest B -> http://<this PC>:8081"
    echo "  Quest C -> http://<this PC>:8082"
    echo "  Four processes, ~13 GiB of VRAM. Connect the headsets one at a time"
    echo "  once every banner is in its transcript."
    ;;
esac

RECORD=(.venv/bin/python scripts/teleop/record_demos.py --task "$TASK" "${PRIMARY_ARGS[@]}" "${EXTRA[@]}")
if [ "$PRINT" = 1 ]; then
  echo "# the recording process"
  printf '%q ' "${RECORD[@]}"; echo
  echo "# One terminal each, in this order; the headsets after every banner is up."
  exit 0
fi

# Stop now if a mirror died during startup.
sleep 8
for i in "${!MIRRORS[@]}"; do
  if ! kill -0 "${MIRRORS[$i]}" 2>/dev/null; then
    echo "[collect] ${MIRROR_NAMES[$i]} died during startup:"
    tail -5 "logs/${MIRROR_NAMES[$i]}/latest-xr_mirror.log" 2>/dev/null | sed 's/^/    /'
    exit 1
  fi
done

echo "[collect] starting the recording process (this terminal)..."
"${RECORD[@]}"
