#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# physical_c5.sh — the PHYSICAL D435 and one real Cylinder5, end to end.
#
#     ./scripts/physical_c5.sh                  # capture, segment, estimate
#     ./scripts/physical_c5.sh --segment-only   # stop after the mask
#     ./scripts/physical_c5.sh --model cylinder3
#     ./scripts/physical_c5.sh --frames 10      # more frames -> better spread
#     ./scripts/physical_c5.sh --dataset <name> # re-run on an earlier capture
#     ./scripts/physical_c5.sh --roi 255,70,445,719
#
# THE ROI SAYS WHERE TO LOOK. NOT WHAT IS THERE.
# ----------------------------------------------
# `--roi x0,y0,x1,y1` is a rectangle in colour-image pixels, given by whoever
# set the bench up. It exists because the single-object rules below are the
# RIGHT rules and were being applied to the whole room: a bench holding one
# intended tube and a dozen unrelated objects fails the connected-component
# limit on the unrelated ones. Inside the window, "one object on a surface" is
# a true statement again.
#
# It cannot and does not identify the object. Which CAD part is in the window
# is `--model`, stated by an operator, exactly as it is without an ROI — and no
# ROI is ever derived from a model's dimensions, which would be recognising the
# part by its shape and then measuring its pose with that same shape.
#
# The physical counterpart of stage_b.sh: same FoundationPose worker, same
# provider, same PhysicalObservation. What differs is that the pixels come from
# the camera on the desk and that NO GROUND TRUTH EXISTS — so this reports
# repeatability and plausibility, and never an accuracy.
#
# WHAT AN OPERATOR MUST DO FIRST, because nothing here can check it:
#
#   * put ONE of the named CAD part in view, ON a flat surface, with the rest of
#     the bench clear. `depth_plane_foreground` fits the dominant plane and
#     keeps what stands on it; a cluttered table yields one component holding
#     several objects, and a CAD model registered against that produces a
#     confident pose on the wrong thing.
#   * know WHICH part it is. `--model` is an input and is never inferred:
#     several WISEPACK tubes share an outer diameter and differ only in length,
#     which a single view cannot resolve.
#
# THE MASK IS SHOWN BEFORE ANY POSE IS COMPUTED, and `--segment-only` stops
# there. If the mask is wrong the pose is meaningless, and looking at it after
# the fact is how a wrong-object pose gets believed.
#
# NO ROBOT. No plan, no IK, no motion, no gripper.
# ---------------------------------------------------------------------------
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/.cache-perception/physical-c5"
PORT="${WISEPACK_FP_PORT:-22201}"

# THE WORKER MUST BE RUNNING. It is opt-in and the launcher does not start it.
if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "[physical-c5] the FoundationPose worker is not answering on ${PORT}." >&2
    echo "              Start it: ./scripts/setup_foundationpose.sh --no-build --run" >&2
    exit 2
fi

python3 "$REPO/scripts/physical_c5.py" "$@"
STATUS=$?

cat <<EOF

[physical-c5] images to inspect:
    $OUT/rgb.jpg             the real colour frame it estimated from
    $OUT/depth_aligned.jpg   the aligned depth, same instant
    $OUT/mask.jpg            the generated mask
    $OUT/mask_overlay.jpg    THAT MASK ON THE PHOTOGRAPH — look at this one
    $OUT/pose_overlay.jpg    the CAD reprojected at the estimated pose
    $OUT/physical_c5.json    every measured number, including repeatability
EOF
exit $STATUS
