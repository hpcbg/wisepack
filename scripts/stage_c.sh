#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stage_c.sh — express the FoundationPose estimate in the WISEPACK workarea.
#
#     ./scripts/stage_c.sh            # fresh: acquire, estimate, transform
#     ./scripts/stage_c.sh --reuse    # transform the last Stage B result
#
# Takes the camera-frame PhysicalObservation from Stage B and applies the SE(3)
# transform derived from the Isaac scene, through the same generic
# RigidTransform the physical camera will use with a measured extrinsic.
#
# Isaac ground truth is diagnostics only. The workarea observation is built from
# the FoundationPose estimate alone.
#
# --reuse is much faster: it starts neither Isaac nor a new inference.
# ---------------------------------------------------------------------------
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/.cache-perception/stage-c"
REUSE=0
if [ "${1:-}" = "--reuse" ]; then REUSE=1; shift; fi

if [ "$REUSE" = "0" ]; then
    "$REPO/scripts/stage_b.sh" >/dev/null || {
        echo "[stage-c] Stage B failed; there is no estimate to transform" >&2
        exit 1
    }
    echo "[stage-c] fresh acquisition and estimate complete"
else
    echo "[stage-c] reusing the last Stage B estimate"
fi

python3 "$REPO/scripts/stage_c_workarea.py" "$@" || exit $?

cat <<EOF

[stage-c] results to inspect:
    $OUT/stage_c.json           camera pose, transform, workarea pose, metrics
    $REPO/.cache-perception/stage-b/stage_b.json    the Stage B result (unchanged)
    $REPO/.cache-perception/stage-b/overlay_estimate.png   the pose it came from
EOF
