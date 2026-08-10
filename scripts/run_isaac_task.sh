#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_isaac_task.sh — run one Isaac Sim script and REPORT WHAT HAPPENED.
#
#     ./scripts/run_isaac_task.sh <log-file> <timeout-seconds> <script> [args...]
#
# WHY THIS EXISTS
# ---------------
# Isaac Sim is slow to start, writes hundreds of lines of extension chatter, and
# installs its own shutdown handling — so a script that fails inside the render
# loop can exit with NO traceback in the log at all. Waiting on a completion
# marker in that log then waits forever on a process that is already dead.
#
# That is not hypothetical: it happened, and cost a ten-minute wait on a
# generator that had exited after twelve seconds.
#
# So this runner never waits on log CONTENT. It waits on the PROCESS:
#
#   * the child's PID is captured explicitly;
#   * the wait ends the moment that PID exits, whatever the log says;
#   * a bounded overall timeout terminates a genuinely hung run;
#   * on any failure the exit code and the log tail are printed, so the reason
#     is in front of you rather than in a file you have to remember to open.
#
# The completion MARKER is used only to distinguish "finished successfully" from
# "exited without doing the work" — never to decide when to stop waiting.
# ---------------------------------------------------------------------------

set -u

MARKER="${ISAAC_TASK_MARKER:-WROTE}"
TAIL_LINES="${ISAAC_TASK_TAIL:-40}"

if [ $# -lt 3 ]; then
    sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

LOG="$1"; shift
LIMIT="$1"; shift
SCRIPT="$1"; shift

ISAAC_ROOT="${ISAAC_SIM_ROOT:-$(ls -d /data/isaac-sim/isaac-sim-6.* 2>/dev/null | sort -Vr | head -1)}"
if [ -z "${ISAAC_ROOT:-}" ] || [ ! -x "$ISAAC_ROOT/python.sh" ]; then
    echo "run_isaac_task: no Isaac Sim 6.x with a bundled python.sh" >&2
    echo "                set ISAAC_SIM_ROOT to point at one" >&2
    exit 1
fi

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

echo "[isaac] $ISAAC_ROOT/python.sh $SCRIPT $*"
echo "[isaac] log: $LOG  (timeout ${LIMIT}s)"

"$ISAAC_ROOT/python.sh" "$SCRIPT" "$@" >"$LOG" 2>&1 &
PID=$!
echo "[isaac] pid: $PID"

WAITED=0
while kill -0 "$PID" 2>/dev/null; do
    if [ "$WAITED" -ge "$LIMIT" ]; then
        echo "[isaac] TIMEOUT after ${LIMIT}s — terminating pid $PID" >&2
        kill -TERM "$PID" 2>/dev/null
        sleep 5
        kill -KILL "$PID" 2>/dev/null
        echo "[isaac] --- last $TAIL_LINES log lines ---" >&2
        tail -n "$TAIL_LINES" "$LOG" >&2
        exit 124
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

# THE CHILD HAS EXITED. Its status is the authority on what happened; the log is
# only evidence about why.
wait "$PID"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo "[isaac] FAILED: exit code $STATUS after ${WAITED}s" >&2
    echo "[isaac] --- last $TAIL_LINES log lines ---" >&2
    tail -n "$TAIL_LINES" "$LOG" >&2
    exit "$STATUS"
fi

if ! grep -q "$MARKER" "$LOG"; then
    # EXITED CLEANLY WITHOUT DOING THE WORK. Isaac reports success on shutdown
    # even when the script never reached its output, so a zero exit code alone
    # is not evidence that anything was produced.
    echo "[isaac] exited 0 after ${WAITED}s but never printed '$MARKER' —" >&2
    echo "        it shut down without completing the task." >&2
    echo "[isaac] --- last $TAIL_LINES log lines ---" >&2
    tail -n "$TAIL_LINES" "$LOG" >&2
    exit 3
fi

echo "[isaac] completed in ${WAITED}s"
grep -E "^(WROTE|  )" "$LOG" | head -20
exit 0
