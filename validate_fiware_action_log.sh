#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# validate_fiware_action_log.sh — prove the audit trail reaches Orion-LD.
#
# Runs INSIDE the Vulcanexus image:
#     ./run_vulcanexus_wisepack.sh validate_fiware_action_log.sh
#
# The required default live path, and the ONLY path this script accepts:
#
#     WISEPACK node -> ROS 2 topic -> DDS -> Orion-LD DDS bridge -> NGSI-LD
#
# There is deliberately no direct-HTTP fallback for the core audit trail. If the
# DDS path is broken this script fails rather than quietly proving something
# else works.
#
# Steps:
#   1. start (or adopt) the DDS-enabled Orion-LD stack
#   2. start the WISEPACK ROS workflow
#   3. inject a KNOWN sequence of actions through the operator command path
#   4. query Orion-LD
#   5. verify the action sequence and payload
#   6. check the KPI entity was updated
#   7. write a timestamped validation report
# ---------------------------------------------------------------------------
set -u

REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=scripts/lib_validate.sh
source "$REPO/scripts/lib_validate.sh"

WS="$REPO/wisepack_ws"
DDS_DIR="$WS/src/wisepack_fiware/dds"
RESULTS_DIR="${WISEPACK_RESULTS_DIR:-$REPO/results}"
ORION="${ORION:-http://localhost:1026}"
PRESET="${WISEPACK_PRESET:-mixed_pipes_small}"
SEED="${WISEPACK_SEED:-42}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG=/tmp/wisepack_fiware.log
REPORT="$RESULTS_DIR/wisepack-fiware-validation-$STAMP.md"

mkdir -p "$RESULTS_DIR"
head_ "WISEPACK FIWARE action-log validation"
wisepack_source_ros "$WS"
wisepack_report_env
info "Orion-LD      : $ORION"

# The exit code is the failure COUNT, so it must survive teardown. rclpy nodes
# under Fast DDS not uncommonly segfault while shutting down (exit 139); without
# pinning the code here that teardown crash would be reported as a validation
# failure even when every check passed.
FINAL_RC=0
cleanup() {
    kill "${LAUNCH_PID:-0}" 2>/dev/null || true
    pkill -f wisepack_ 2>/dev/null || true
    wait "${LAUNCH_PID:-0}" 2>/dev/null || true
    exit "$FINAL_RC"
}
trap cleanup EXIT INT TERM

# ---- 1. broker --------------------------------------------------------------
head_ "1. Orion-LD DDS broker"
VER="$(curl -s --max-time 3 "$ORION/version" 2>/dev/null)"
if printf '%s' "$VER" | grep -qi 'orionld'; then
    pass "Orion-LD already running on $ORION"
elif printf '%s' "$VER" | grep -qi 'orion'; then
    fail "port 1026 is served by a NON-LD Orion (NGSI-v2). Stop it first."
    exit "$FAILURES"
else
    warn "broker not reachable — generating the mapping and starting it"
    ( cd "$DDS_DIR" && python3 generate_config.py --domain "$ROS_DOMAIN_ID" ) \
        | sed 's/^/      /'
    ( cd "$DDS_DIR" && docker compose -f docker-compose.dds.yml up -d ) \
        >/dev/null 2>&1
    if await_orion "$ORION" 60; then
        pass "Orion-LD DDS broker up"
    else
        fail "Orion-LD did not come up on $ORION within 60 s"
        exit "$FAILURES"
    fi
fi

# The mapping must match the running ROS_DOMAIN_ID or nothing bridges at all.
CONFIGURED_DOMAIN="$(python3 -c "
import json;print(json.load(open('$DDS_DIR/context_broker_config.json'))['dds']['ddsmodule']['dds']['domain'])
" 2>/dev/null)"
if [ "$CONFIGURED_DOMAIN" = "$ROS_DOMAIN_ID" ]; then
    pass "DDS domain $CONFIGURED_DOMAIN matches ROS_DOMAIN_ID"
else
    fail "mapping domain=$CONFIGURED_DOMAIN but ROS_DOMAIN_ID=$ROS_DOMAIN_ID — nothing will bridge"
fi

# ---- 2. the workflow --------------------------------------------------------
head_ "2. WISEPACK ROS workflow"
ros2 launch wisepack_bringup demo.launch.py \
    preset:="$PRESET" seed:="$SEED" dynamic_events:=false \
    results_dir:="$RESULTS_DIR" > "$LOG" 2>&1 &
LAUNCH_PID=$!
if await_log "orchestrator up" "$LOG" 60; then
    pass "workflow started (preset $PRESET, seed $SEED)"
else
    fail "workflow did not start"; exit "$FAILURES"
fi
sleep 8

# ---- 3. inject a KNOWN action sequence --------------------------------------
head_ "3. Injecting a known action sequence"
approve() {
    ros2 topic pub --once /wisepack/operator/approval std_msgs/msg/String \
        'data: APPROVE' --qos-reliability reliable --qos-durability transient_local \
        >/dev/null 2>&1
}
send() {
    ros2 topic pub --once /wisepack/operator/command std_msgs/msg/String \
        "data: '$1'" --qos-reliability reliable --qos-durability transient_local \
        >/dev/null 2>&1
}
info "grasp_failure -> approve -> run to completion"
send '{"command":"grasp_failure","args":{}}'
sleep 2
approve
if await_log "artefacts written" "$LOG" 180; then
    pass "cycle completed"
else
    warn "cycle did not complete in 180 s — continuing with what was logged"
fi

ROS_SEQ="$(echo_state /wisepack/action/sequence 8 | tail -1 | sed 's/[^0-9]//g')"
[ -z "$ROS_SEQ" ] && ROS_SEQ=0
info "action sequence on ROS 2: $ROS_SEQ"

# ---- 4-6. query and verify Orion-LD ----------------------------------------
head_ "4-6. Verifying the trail in Orion-LD"
# Give the bridge a moment to settle the final samples.
sleep 5
export ORION
python3 -m wisepack_fiware.verify \
    --expect-sequence "$ROS_SEQ" \
    --json-out "$RESULTS_DIR/wisepack-fiware-verify-$STAMP.json"
VERIFY_RC=$?
if [ "$VERIFY_RC" -eq 0 ]; then
    pass "all FIWARE entity/attribute checks passed"
else
    fail "$VERIFY_RC FIWARE check(s) failed"
fi

# ---- 7. report --------------------------------------------------------------
head_ "7. Report"
python3 - "$RESULTS_DIR/wisepack-fiware-verify-$STAMP.json" "$REPORT" \
         "$ROS_SEQ" "$ORION" "$PRESET" "$SEED" <<'PY'
import json, sys, datetime
verify_path, report_path, ros_seq, orion, preset, seed = sys.argv[1:7]
try:
    doc = json.load(open(verify_path))
except Exception:
    doc = {"entities": {}, "failures": 1, "broker_detail": "verify output missing"}

L = []
A = L.append
A(f"# WISEPACK FIWARE action-log validation — {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
A("")
A("## Data path under test")
A("")
A("```")
A("WISEPACK node -> ROS 2 topic -> DDS -> Orion-LD DDS bridge -> NGSI-LD entity")
A("```")
A("")
A("There is no direct-HTTP path for the core audit trail. Every value below")
A("reached FIWARE by crossing DDS.")
A("")
A(f"- Broker: `{orion}` — {doc.get('broker_detail', 'unknown')}")
A(f"- Scenario: `{preset}` seed `{seed}`")
A(f"- Action sequence observed on ROS 2: **{ros_seq}**")
seq = doc.get("action_sequence", {})
A(f"- Action sequence observed in FIWARE: **{seq.get('value')}** ({seq.get('state')})")
A("")
A("## Entities and attributes")
A("")
A("| Entity | Attribute | State | Value |")
A("|---|---|---|---|")
for short, block in sorted(doc.get("entities", {}).items()):
    for attr, info in sorted(block.get("attributes", {}).items()):
        val = str(info.get("value"))
        if len(val) > 70:
            val = val[:67] + "..."
        val = val.replace("|", "\\|")
        A(f"| `{block['urn']}` | `{attr}` | {info['state']} | {val if info['state']=='REAL' else ''} |")
A("")
latest = doc.get("latest_action")
if latest:
    A("## Latest action event, as read back from Orion-LD")
    A("")
    A("```json")
    A(json.dumps(latest, indent=2)[:2000])
    A("```")
    A("")
A("## Behaviour of this mapping — stated, not assumed")
A("")
A("The Orion-LD DDS bridge is **state-oriented**: each attribute holds the")
A("LATEST value written to it. `WISEPACKActionStream.actionJson` therefore")
A("contains the most recent action event, not an append-only log, and")
A("`.sequence` contains the highest sequence number reached. This demonstrator")
A("does NOT claim append-only semantics from the broker. The append-only record")
A("is the timestamped `results/wisepack-actions-*.jsonl` artefact, and the")
A("FIWARE entity is the live, queryable state of that stream. Historical")
A("retention would come from QuantumLeap subscribing to these entities; that is")
A("documented as a limitation and an extension point, not implied here.")
A("")
A(f"**Result: {'PASS' if doc.get('failures', 1) == 0 else 'FAIL'}** "
  f"({doc.get('failures', '?')} failure(s), "
  f"{doc.get('uninitialised', 0)} uninitialised attribute(s))")
A("")
open(report_path, "w").write("\n".join(L))
print(f"  wrote {report_path}")
PY

head_ "Summary"
if [ "$FAILURES" -eq 0 ]; then
    pass "FIWARE action-log validation PASSED"
else
    fail "$FAILURES check(s) failed"
fi
info "report: $REPORT"
FINAL_RC="$FAILURES"
exit "$FAILURES"
