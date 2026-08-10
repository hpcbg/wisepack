#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stage_e.sh — launch WISEPACK showing the simulated-RGB-D Cylinder5 run.
#
#     ./scripts/stage_e.sh            # acquire a fresh frame first, then launch
#     ./scripts/stage_e.sh --reuse    # launch using the last Stage C result
#
# Starts the ORDINARY WISEPACK dashboard and acquires the simulated RGB-D batch
# through the normal workflow, so the perception panel, the plan, the Digital
# Twin and the approval gate all describe one run at one revision.
#
# IT DOES NOT MOVE THE ROBOT. Approving here records the decision; execution is
# a separate step and Stage E does not take it.
# ---------------------------------------------------------------------------
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${WISEPACK_PORT:-8200}"
REUSE=0
if [ "${1:-}" = "--reuse" ]; then REUSE=1; shift; fi

if [ "$REUSE" = "0" ]; then
    echo "[stage-e] acquiring a fresh simulated RGB-D observation..."
    "$REPO/scripts/stage_c.sh" >/dev/null || {
        echo "[stage-e] the acquisition chain failed; nothing to show" >&2
        exit 1
    }
else
    echo "[stage-e] reusing the last Stage C observation"
fi

if [ ! -f "$REPO/.cache-perception/stage-c/stage_c.json" ]; then
    echo "[stage-e] no observation found. Run ./scripts/stage_c.sh first." >&2
    exit 1
fi

cd "$REPO/web"
echo "[stage-e] starting the dashboard on http://127.0.0.1:$PORT"
"$REPO/.venv-perception/bin/python" app.py --source sim --port "$PORT" \
    --preset cad_cylinder5_single &
DASH=$!
trap 'kill $DASH 2>/dev/null' EXIT INT TERM

for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null && break
    sleep 1
done

echo "[stage-e] acquiring the simulated RGB-D batch through the normal workflow..."
curl -s -X POST "http://127.0.0.1:$PORT/api/command" \
     -H 'Content-Type: application/json' \
     -d '{"command":"acquire_simulated_rgbd","args":{}}' | sed 's/^/[stage-e] /'
echo

cat <<EOF

[stage-e] open  http://127.0.0.1:$PORT

  Physical Perception panel   backend, camera, mask source, estimated centre,
                              tube axis, position/axis error vs ground truth,
                              and the RGB / depth / mask / pose-overlay images
  Digital Twin                the plan for this revision
  Approval                    pending — approving does NOT move the robot

Press Ctrl-C to stop the dashboard.
EOF
wait $DASH
