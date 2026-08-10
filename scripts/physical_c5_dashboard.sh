#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# physical_c5_dashboard.sh — the physical D435 result, in the normal dashboard.
#
#     ./scripts/physical_c5_dashboard.sh                     # live camera
#     ./scripts/physical_c5_dashboard.sh --dataset cylinder5-20260810-133408
#     ./scripts/physical_c5_dashboard.sh --no-run            # show the last result
#
# TWO STEPS, NEITHER OF THEM NEW. It runs ./scripts/physical_c5.sh, which
# writes .cache-perception/physical-c5/physical_c5.json and its four images,
# and then starts the ORDINARY dashboard, which reads that artefact and renders
# the "Physical RGB-D 6-DoF — Intel RealSense D435" panel. There is no second
# inference path and no separate demo page.
#
# WHAT THE PANEL WILL AND WILL NOT CLAIM
# --------------------------------------
# The pose stays in `camera_color_optical_frame` from the sensor to the screen.
# The physical camera has never been calibrated to the work area, so
# `workarea_pose_available` is false and the panel says, in words, that
# work-area calibration is required before planning or execution. Nothing here
# runs the packing optimizer, places anything in `wisepack_workarea`, or drives
# the Digital Twin.
#
# THE SIMULATED END-TO-END DEMONSTRATION IS UNCHANGED. `./scripts/stage_e.sh`
# remains the perception-to-planning run, and this script does not touch it.
#
# LIVE AND REPLAYED ARE LABELLED DIFFERENTLY. Both are real D435 data; a
# recorded capture is not evidence that the camera worked just now, so the
# badge reads LIVE PHYSICAL D435 or RECORDED PHYSICAL D435 DATA accordingly.
# ---------------------------------------------------------------------------
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8080}"
MODE="${WISEPACK_DASHBOARD_MODE:-sim}"

RUN=1
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-run) RUN=0 ;;
        *) ARGS+=("$arg") ;;
    esac
done

if [ "$RUN" = "1" ]; then
    # DEFAULTS THAT MATCH THE VALIDATED BENCH SETUP, overridable by passing the
    # same flags through. The ROI is an OPERATOR statement about where to look
    # and is not derived from the CAD model; it is a default here only because
    # this repository's demonstration bench has not moved.
    case " ${ARGS[*]-} " in
        *" --model "*) ;;
        *) ARGS+=(--model cylinder5) ;;
    esac
    case " ${ARGS[*]-} " in
        *" --roi "*) ;;
        *) ARGS+=(--roi 255,70,445,719) ;;
    esac
    case " ${ARGS[*]-} " in
        *" --frames "*) ;;
        *) ARGS+=(--frames 5) ;;
    esac
    echo "[physical-c5-dashboard] running the physical perception step"
    "$REPO/scripts/physical_c5.sh" "${ARGS[@]}" || {
        status=$?
        echo "[physical-c5-dashboard] the physical run did not produce a pose." >&2
        echo "                        Nothing is published: a dashboard panel is" >&2
        echo "                        not a place to show a result that does not" >&2
        echo "                        exist. Fix the run first, or start the" >&2
        echo "                        dashboard with --no-run to show the last" >&2
        echo "                        successful one." >&2
        exit $status
    }
else
    echo "[physical-c5-dashboard] --no-run: showing the last physical result"
fi

RESULT="$REPO/.cache-perception/physical-c5/physical_c5.json"
if [ ! -f "$RESULT" ]; then
    echo "[physical-c5-dashboard] no physical result at $RESULT" >&2
    echo "                        Run without --no-run first." >&2
    exit 3
fi

python3 - "$RESULT" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    document = json.load(handle)
pose = (document.get("observation") or {}).get("pose") or {}
task = (document.get("observation") or {}).get("task") or {}
centre = [round(v, 1) for v in (task.get("object_center_mm") or [])]
print(f"[physical-c5-dashboard] {document.get('run_label', '?')}")
print(f"[physical-c5-dashboard]   object   {document.get('model_id')}  "
      f"centre {centre} mm in {(document.get('observation') or {}).get('frame_id')}")
print(f"[physical-c5-dashboard]   pose_valid {pose.get('valid')}   "
      f"workarea_pose_available {pose.get('workarea_pose_available')}")
PY

echo "[physical-c5-dashboard] starting the dashboard on port $PORT"
echo "[physical-c5-dashboard] the panel is 'Physical RGB-D 6-DoF — Intel RealSense D435'"
echo "[physical-c5-dashboard] planning stays disabled: no camera-to-work-area extrinsic exists"
PORT="$PORT" exec "$REPO/run_wisepack_dashboard.sh" "$MODE"
