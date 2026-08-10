#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stage_a.sh — acquire one RGB-D frame from the WISEPACK workcell in Isaac.
#
#     ./scripts/stage_a.sh
#
# Launches the EXISTING WISEPACK workcell (table, containers, layout) with the
# `cad_cylinder5_single` scenario, loads the actual Cylinder5.stl, runs the
# D435-compatible simulated RGB-D camera, and captures RGB + depth + the exact
# synthetic instance mask.
#
# Takes about a minute: most of it is Isaac Sim starting.
#
# No robot is loaded — WisepackScene builds the cell, the robot adapter loads
# the arm, and Stage A is about the scene and the sensing.
# ---------------------------------------------------------------------------
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${1:-/tmp/wisepack-stage-a.log}"
OUT="$REPO/.cache-perception/stage-a"

echo "[stage-a] building the WISEPACK workcell with Cylinder5 CAD..."
"$REPO/scripts/run_isaac_task.sh" "$LOG" 900 \
    "$REPO/simulators/isaac/stage_a_check.py" >/dev/null 2>&1
STATUS=$?

grep -E "^  \[(PASS|FAIL)\]" "$LOG" || true
echo
grep -E "^STAGE-A [0-9]+/" "$LOG" || echo "[stage-a] no summary — see $LOG"

if [ "$STATUS" -ne 0 ]; then
    echo "[stage-a] FAILED (exit $STATUS). Full log: $LOG" >&2
    exit "$STATUS"
fi

cat <<EOF

[stage-a] images to inspect:
    $OUT/workcell.png                 the whole cell: table, container, tube
    $OUT/d435_rgb.png                 what the simulated D435 sees
    $OUT/d435_depth.png               colourised depth
    $OUT/cylinder5_mask.png           the exact synthetic instance mask
    $OUT/d435_rgb_mask_overlay.png    RGB with the mask outlined
    $OUT/stage_a.json                 every check, with its measured value

[stage-a] the frame was also exported for Stage B:
    $REPO/.cache-perception/isaac-reference/stage_a_workcell/
EOF
