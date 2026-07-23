#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# measure_dds_fiware_latency.sh — DDS -> FIWARE latency for the audit trail.
#
# Runs INSIDE the Vulcanexus image:
#     ./run_vulcanexus_wisepack.sh measure_dds_fiware_latency.sh
#
# Wrapper around scripts/measure_dds_fiware_latency.py, adapted from HARMONY's
# measure_dds_fiware_latency.sh. It starts the Orion-LD DDS broker if it is not
# already up, then measures.
#
# Environment overrides:
#     DDS_LATENCY_WARMUP        warmup samples          (default 3)
#     DDS_LATENCY_SAMPLES       timed samples           (default 20)
#     DDS_LATENCY_TIMEOUT_SEC   per-sample timeout      (default 10)
#     ORION                     broker URL              (default http://localhost:1026)
# ---------------------------------------------------------------------------
set -u

REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=scripts/lib_validate.sh
source "$REPO/scripts/lib_validate.sh"

WS="$REPO/wisepack_ws"
DDS_DIR="$WS/src/wisepack_fiware/dds"
ORION="${ORION:-http://localhost:1026}"
export WISEPACK_RESULTS_DIR="${WISEPACK_RESULTS_DIR:-$REPO/results}"

head_ "WISEPACK DDS/FIWARE latency measurement"
wisepack_source_ros "$WS"
wisepack_report_env
mkdir -p "$WISEPACK_RESULTS_DIR"

VER="$(curl -s --max-time 3 "$ORION/version" 2>/dev/null)"
if printf '%s' "$VER" | grep -qi 'orionld'; then
    pass "Orion-LD already running on $ORION"
elif printf '%s' "$VER" | grep -qi 'orion'; then
    fail "port 1026 is served by a NON-LD Orion. Stop it first."
    exit 1
else
    warn "broker not reachable — starting the Orion-LD DDS stack"
    ( cd "$DDS_DIR" && python3 generate_config.py --domain "$ROS_DOMAIN_ID" >/dev/null \
        && docker compose -f docker-compose.dds.yml up -d ) >/dev/null 2>&1
    if await_orion "$ORION" 60; then
        pass "Orion-LD DDS broker up"
    else
        fail "Orion-LD did not come up within 60 s"
        exit 1
    fi
fi

export ORION
exec python3 "$REPO/scripts/measure_dds_fiware_latency.py"
