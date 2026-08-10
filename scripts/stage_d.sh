#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stage_d.sh — perceived Cylinder5 -> plan -> Digital Twin -> approval pending.
#
#     ./scripts/stage_d.sh            # fresh: acquire, estimate, transform, plan
#     ./scripts/stage_d.sh --reuse    # plan from the last Stage C result
#
# Runs the Stage C observation through the ORDINARY WISEPACK workflow: the same
# apply_observation_batch, the same optimizer, the same Digital Twin validation
# and the same approval gate a planar camera batch uses.
#
# IT DOES NOT MOVE THE ROBOT. Stage D stops at "approval pending".
# ---------------------------------------------------------------------------
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/.cache-perception/stage-d"
REUSE=0
if [ "${1:-}" = "--reuse" ]; then REUSE=1; shift; fi

if [ "$REUSE" = "0" ]; then
    "$REPO/scripts/stage_c.sh" >/dev/null || {
        echo "[stage-d] Stage C failed; there is no observation to plan from" >&2
        exit 1
    }
    echo "[stage-d] fresh acquisition, estimate and transform complete"
else
    echo "[stage-d] reusing the last Stage C observation"
fi

python3 "$REPO/scripts/stage_d_plan.py" "$@" || exit $?

cat <<EOF

[stage-d] results to inspect:
    $OUT/stage_d.json    run id, revision, plan, twin, approval, and the
                         proof that planning used the physical centre
    $REPO/.cache-perception/stage-c/stage_c.json   the observation it planned from
    $REPO/.cache-perception/stage-b/overlay_estimate.png   the pose it came from
EOF
