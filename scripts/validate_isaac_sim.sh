#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# validate_isaac_sim.sh — optional physical smoke test of the Isaac backend.
#
#     ./scripts/validate_isaac_sim.sh              # full: READY + one cylinder
#     ./scripts/validate_isaac_sim.sh --ready-only # startup and READY only
#     WISEPACK_ISAAC_ROBOT=panda ./scripts/validate_isaac_sim.sh
#
# WHICH ROBOT. Whatever `WISEPACK_ISAAC_ROBOT` selects, or the configured
# default from config/isaac_robots.yaml. The validator does not choose one — a
# second place that decided the robot would be a second thing to keep in step
# with the registry — it reports the one that answered.
#
# WHAT IT PROVES, in order:
#     1. Isaac Sim 6.0.1 is installed and its bundled Python starts;
#     2. the procedural scene builds and the SELECTED robot loads and validates
#        against its profile;
#     3. the ROS 2 bridge comes up and READY reaches the wire;
#     4. at least one cylinder is picked, carried, RELEASED and SETTLED;
#     5. the process exits cleanly.
#
# EXIT CODES
#     0   passed
#     1   failed — a stage that should have worked did not
#    77   SKIPPED: Isaac Sim, an NVIDIA GPU or a usable environment is genuinely
#         absent. Distinct from failure on purpose: the standard test suite and
#         CI must never require a simulator, and a skipped optional dependency
#         must not be reported as a broken implementation.
#
# This is NOT part of `pytest tests`. Nothing in the normal suite needs Isaac.
# ---------------------------------------------------------------------------
set -u

REPO="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
LOG="[isaac-validate]"
READY_ONLY=0
TIMEOUT_S="${WISEPACK_ISAAC_VALIDATE_TIMEOUT:-900}"

for arg in "$@"; do
    case "$arg" in
        --ready-only) READY_ONLY=1 ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "$LOG unknown option: $arg" >&2; exit 2 ;;
    esac
done

skip() {
    echo "$LOG SKIPPED: $1"
    echo "$LOG   The Isaac backend is optional. Everything else in WISEPACK,"
    echo "$LOG   including the full test suite, runs without it."
    exit 77
}

# ---- preconditions ---------------------------------------------------------
if [ -n "${ISAAC_SIM_ROOT:-}" ]; then
    ROOT="$ISAAC_SIM_ROOT"
else
    ROOT="$(ls -d /data/isaac-sim/isaac-sim-6.* /opt/isaac-sim "$HOME/isaacsim" \
            2>/dev/null | sort -Vr | head -1)"
fi
[ -n "${ROOT:-}" ] && [ -x "$ROOT/python.sh" ] \
    || skip "no Isaac Sim installation with a bundled python.sh (set ISAAC_SIM_ROOT)"

VERSION="unknown"
[ -f "$ROOT/VERSION" ] && VERSION="$(cat "$ROOT/VERSION")"
case "$VERSION" in
    6.*) ;;
    *) skip "Isaac Sim at $ROOT reports version '$VERSION'; this backend needs 6.x" ;;
esac

command -v nvidia-smi >/dev/null 2>&1 \
    || skip "nvidia-smi not found — Isaac Sim needs an NVIDIA GPU"
nvidia-smi -L >/dev/null 2>&1 \
    || skip "nvidia-smi found no usable GPU"

echo "$LOG Isaac Sim $VERSION at $ROOT"
echo "$LOG GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Headless unless a display exists: this script is expected to run over SSH and
# in the acceptance demonstration, neither of which has a window.
if [ -z "${WISEPACK_ISAAC_HEADLESS:-}" ]; then
    if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
        export WISEPACK_ISAAC_HEADLESS=0
    else
        export WISEPACK_ISAAC_HEADLESS=1
    fi
fi
echo "$LOG headless=$WISEPACK_ISAAC_HEADLESS timeout=${TIMEOUT_S}s"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export WISEPACK_PRESET="${WISEPACK_PRESET:-isaac_cylinders_smoke}"
export WISEPACK_SEED="${WISEPACK_SEED:-42}"
ISAAC_LOG="$(mktemp -t wisepack-isaac-validate-XXXXXX.log)"
echo "$LOG log: $ISAAC_LOG"

FAILURES=0
note_fail() { echo "$LOG FAIL: $1" >&2; FAILURES=$((FAILURES + 1)); }
note_pass() { echo "$LOG ok: $1"; }

# ---- stage 1: startup, scene, READY ----------------------------------------
# --self-test builds the scene, loads and VALIDATES the selected robot,
# publishes READY and exits. It
# is the cheapest thing that still proves the GPU, the renderer, the asset
# download and the ROS bridge all work.
echo "$LOG [1/2] building the scene and waiting for READY ..."
timeout "$TIMEOUT_S" "$REPO/scripts/run_wisepack_isaac.sh" --self-test \
    > "$ISAAC_LOG" 2>&1
RC=$?

if [ "$RC" -eq 124 ]; then
    note_fail "Isaac Sim did not finish startup within ${TIMEOUT_S}s"
elif [ "$RC" -ne 0 ]; then
    note_fail "Isaac Sim exited with status $RC during the self-test"
fi

grep -q '\[isaac-scene\] scene ready' "$ISAAC_LOG" \
    && note_pass "procedural scene built (table, container, cylinders)" \
    || note_fail "the scene was never reported ready"

grep -q '\[isaac-app\] READY' "$ISAAC_LOG" \
    && note_pass "simulator reported READY" \
    || note_fail "no READY was reported"

grep -q '\[isaac-bridge\] -> READY' "$ISAAC_LOG" \
    && note_pass "READY published on the ROS 2 feedback topic" \
    || note_fail "READY never reached the ROS 2 bridge"

if [ "$FAILURES" -ne 0 ]; then
    echo "$LOG last 40 lines of $ISAAC_LOG:" >&2
    tail -40 "$ISAAC_LOG" | sed 's/^/    /' >&2
    exit 1
fi

if [ "$READY_ONLY" -eq 1 ]; then
    echo "$LOG --ready-only: startup validated, physical execution not exercised"
    exit 0
fi

# ---- stage 2: one physical pick, release and settle ------------------------
# --smoke-test plans with the SAME optimizer the stack uses, then PUBLISHES real
# commands on /wisepack/isaac/command which the simulator receives back through
# its own subscription over DDS. So this exercises the serialization, the QoS and
# the run gating for real, while still needing no Docker, no Orion-LD and no
# operator. The orchestrator half of the contract is covered by `pytest tests`.
echo "$LOG [2/2] executing one cylinder end to end ..."
PHYS_LOG="$(mktemp -t wisepack-isaac-physical-XXXXXX.log)"
echo "$LOG log: $PHYS_LOG"

ISAAC_SIM_ROOT="$ROOT" timeout "$TIMEOUT_S" \
    "$REPO/scripts/run_wisepack_isaac.sh" --smoke-test 1 \
    > "$PHYS_LOG" 2>&1
RC=$?

if [ "$RC" -eq 124 ]; then
    note_fail "the physical run did not finish within ${TIMEOUT_S}s"
elif [ "$RC" -ne 0 ]; then
    note_fail "the physical run exited with status $RC"
fi

for marker in \
    "GRASPING:grasp reported" \
    "RELEASING:gripper opened and the item released" \
    "SETTLING:PhysX settling observed" \
    "ITEM_COMPLETED:at least one cylinder completed physically"
do
    state="${marker%%:*}"; label="${marker#*:}"
    grep -q "SMOKE-STATE $state" "$PHYS_LOG" \
        && note_pass "$label" \
        || note_fail "$label — no $state feedback was received"
done

grep -q 'SMOKE-RESULT PASS' "$PHYS_LOG" \
    && note_pass "the settled item was verified inside the container" \
    || note_fail "the settled item was not verified inside the container"

echo ""
if [ "$FAILURES" -eq 0 ]; then
    grep -E '^SMOKE-(STATE|RESULT|POSE)' "$PHYS_LOG" | sed 's/^/    /'
    echo "$LOG PASSED — physical pick, release and settle validated"
    exit 0
fi
echo "$LOG last 60 lines of $PHYS_LOG:" >&2
tail -60 "$PHYS_LOG" | sed 's/^/    /' >&2
echo "$LOG FAILED — $FAILURES check(s)" >&2
exit 1
