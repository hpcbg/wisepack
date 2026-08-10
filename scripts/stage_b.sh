#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stage_b.sh — Stage A acquisition -> the real FoundationPose worker -> pose.
#
#     ./scripts/stage_b.sh            # acquire a fresh frame, then estimate
#     ./scripts/stage_b.sh --reuse    # estimate from the last acquisition
#
# Sends the workcell's RGB, depth, intrinsics and exact instance mask through
# the SAME WISEPACK-managed FoundationPose Docker worker the bolt regression
# uses, and evaluates the result against Isaac ground truth.
#
# Isaac ground truth is DIAGNOSTICS ONLY: it is used after the estimate exists,
# never as an input to it.
# ---------------------------------------------------------------------------
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/.cache-perception/stage-b"
REUSE=0
if [ "${1:-}" = "--reuse" ]; then
    REUSE=1
    # CONSUMED HERE. Anything left in "$@" is forwarded to the Python stage,
    # which does not know this flag.
    shift
fi

if [ "$REUSE" = "0" ]; then
    "$REPO/scripts/stage_a.sh" /tmp/wisepack-stage-a.log >/dev/null || {
        echo "[stage-b] the Stage A acquisition failed; nothing to estimate from" >&2
        exit 1
    }
    echo "[stage-b] acquired a fresh frame from the workcell"
else
    echo "[stage-b] reusing the last Stage A acquisition"
fi

# THE WORKER MUST BE RUNNING. It is opt-in and is not started by the launcher.
if ! curl -sf "http://127.0.0.1:${WISEPACK_FP_PORT:-22201}/health" >/dev/null; then
    echo "[stage-b] the FoundationPose worker is not answering." >&2
    echo "          Start it:  ./scripts/setup_foundationpose.sh --no-build --run" >&2
    exit 1
fi

python3 "$REPO/scripts/stage_b_foundationpose.py" "$@" || exit $?

cat <<EOF

[stage-b] images to inspect:
    $OUT/overlay_estimate.png   FoundationPose CAD (green) vs the exact mask (white)
    $OUT/overlay_gt.png         the same for Isaac ground truth, for comparison
    $OUT/stage_b.json           the observation and every measured number
    $REPO/.cache-perception/stage-a/d435_rgb.png     the RGB it estimated from
    $REPO/.cache-perception/stage-a/d435_depth.png   the depth it estimated from
EOF
