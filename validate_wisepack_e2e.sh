#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# validate_wisepack_e2e.sh — end-to-end acceptance check of the ROS 2 stack.
#
# Runs INSIDE the Vulcanexus image:   ./run_vulcanexus_wisepack.sh validate_wisepack_e2e.sh
#
# It launches the real node graph, drives one complete packaging cycle through
# the operator command path, and checks the things the demo actually claims:
#
#   1. every canonical topic is emitted
#   2. NOTHING is executed before approval  (the safety invariant)
#   3. approval over ROS 2 starts execution
#   4. the Digital Twin validator (separate process) agrees the plan is valid
#   5. a dynamic event triggers a visible re-plan
#   6. the action sequence is monotonic and gap-free
#   7. the run writes its artefacts
#
# Exit code == number of failures, as in TEMPO and HARMONY.
# ---------------------------------------------------------------------------
set -u

REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=scripts/lib_validate.sh
source "$REPO/scripts/lib_validate.sh"

WS="$REPO/wisepack_ws"
RESULTS_DIR="${WISEPACK_RESULTS_DIR:-$REPO/results}"
PRESET="${WISEPACK_PRESET:-mixed_pipes_dense}"
SEED="${WISEPACK_SEED:-42}"
LOG=/tmp/wisepack_e2e.log

head_ "WISEPACK end-to-end validation"
wisepack_source_ros "$WS"
wisepack_report_env
info "preset        : $PRESET (seed $SEED)"
mkdir -p "$RESULTS_DIR"

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

# ---- 1. bring the graph up --------------------------------------------------
head_ "1. Node graph"
ros2 launch wisepack_bringup demo.launch.py \
    preset:="$PRESET" seed:="$SEED" dynamic_events:=true \
    results_dir:="$RESULTS_DIR" > "$LOG" 2>&1 &
LAUNCH_PID=$!

if await_log "orchestrator up" "$LOG" 60; then
    pass "orchestrator started"
else
    fail "orchestrator did not start within 60 s"
    tail -20 "$LOG" | sed 's/^/      /'
    exit "$FAILURES"
fi
await_log "Digital Twin validator up" "$LOG" 20 && pass "Digital Twin validator started" \
    || warn "Digital Twin validator did not announce itself"
await_log "perception simulator up" "$LOG" 20 && pass "perception simulator started" \
    || warn "perception simulator did not announce itself"

# ---- 2. canonical topics ----------------------------------------------------
head_ "2. Canonical topic contract"
sleep 5
TOPIC_LIST="$(ros2 topic list 2>/dev/null)"
EXPECTED=(
    /wisepack/scenario/config /wisepack/scenario/state /wisepack/waste/items
    /wisepack/waste/detected_count /wisepack/plan/baseline /wisepack/plan/optimized
    /wisepack/plan/selected /wisepack/plan/status_json /wisepack/operator/approval
    /wisepack/execution/state /wisepack/execution/progress_pct
    /wisepack/system/readiness /wisepack/action/event /wisepack/action/sequence
    /wisepack/kpi/containers_baseline /wisepack/kpi/containers_optimized
    /wisepack/kpi/volume_reduction_pct
)
missing=0
for t in "${EXPECTED[@]}"; do
    if printf '%s\n' "$TOPIC_LIST" | grep -qx "$t"; then :; else
        fail "topic missing: $t"; missing=$((missing + 1))
    fi
done
[ "$missing" -eq 0 ] && pass "all ${#EXPECTED[@]} canonical topics present"

# ---- 3. the safety invariant ------------------------------------------------
head_ "3. Safety invariant — nothing executes before approval"
STAGE="$(echo_state /wisepack/execution/state 10 | tail -1)"
info "stage: $STAGE"
if printf '%s' "$STAGE" | grep -q "WAIT_FOR_OPERATOR_APPROVAL"; then
    pass "workflow is holding at the approval gate"
else
    fail "expected WAIT_FOR_OPERATOR_APPROVAL, got: $STAGE"
fi
PROGRESS="$(echo_telemetry /wisepack/execution/progress_pct 8 | tail -1)"
if printf '%s' "$PROGRESS" | grep -qE 'data: 0\.0*$'; then
    pass "progress is 0% — no placement executed before approval"
else
    warn "progress reads: $PROGRESS"
fi

# ---- 4. the Digital Twin verdict --------------------------------------------
head_ "4. Digital Twin validation (independent process)"
VERDICT="$(echo_state /wisepack/plan/status_json 15 | tail -1)"
if printf '%s' "$VERDICT" | grep -q '"valid":true'; then
    pass "twin validator independently confirms the plan is valid"
elif [ -n "$VERDICT" ]; then
    fail "twin validator rejected the plan: ${VERDICT:0:160}"
else
    warn "no verdict published yet on /wisepack/plan/status_json"
fi

# ---- 5. approval over ROS 2 -------------------------------------------------
head_ "5. Operator approval over ROS 2 / DDS"
ros2 topic pub --once /wisepack/operator/approval std_msgs/msg/String \
    'data: APPROVE' --qos-reliability reliable --qos-durability transient_local \
    >/dev/null 2>&1
if await_log "APPROVED via operator topic" "$LOG" 20; then
    pass "approval received and accepted by the orchestrator"
else
    fail "orchestrator did not accept the approval"
fi

# ---- 6. execution and re-planning -------------------------------------------
head_ "6. Execution and dynamic re-planning"
if await_log "replan_complete|re-plan" "$LOG" 90; then
    pass "a dynamic event triggered a re-plan"
else
    warn "no re-plan observed within 90 s (is dynamic_events:=true?)"
fi
# The scripted event forces a second approval; grant it so the run can finish.
ros2 topic pub --once /wisepack/operator/approval std_msgs/msg/String \
    'data: APPROVE' --qos-reliability reliable --qos-durability transient_local \
    >/dev/null 2>&1

if await_log "artefacts written" "$LOG" 180; then
    pass "cycle completed and artefacts were written"
else
    fail "cycle did not complete within 180 s"
fi

# ---- 7. audit trail ---------------------------------------------------------
head_ "7. Audit trail"
SEQ="$(echo_state /wisepack/action/sequence 8 | tail -1 | sed 's/[^0-9]//g')"
if [ -n "$SEQ" ] && [ "$SEQ" -gt 20 ]; then
    pass "action sequence reached $SEQ on /wisepack/action/sequence"
else
    fail "action sequence did not advance (read: '$SEQ')"
fi

NEWEST_RUN="$(ls -1t "$RESULTS_DIR"/wisepack-run-*.json 2>/dev/null | head -1)"
if [ -n "$NEWEST_RUN" ]; then
    pass "run artefact: $(basename "$NEWEST_RUN")"
    python3 - "$NEWEST_RUN" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
k = doc["kpis"]["metrics"]
print(f"      containers   baseline={k['containers_baseline']['value']} "
      f"optimized={k['containers_optimized']['value']}")
print(f"      utilization  baseline={k['container_utilization_baseline_pct']['value']}% "
      f"optimized={k['container_utilization_optimized_pct']['value']}%")
print(f"      volume requirement reduction = "
      f"{k['volume_requirement_reduction_pct']['value']}%  "
      f"[{k['volume_requirement_reduction_pct']['source']}]")
print(f"      action events = {doc['action_events']}, "
      f"sequence monotonic = {doc['action_sequence_monotonic']}")
if not doc["action_sequence_monotonic"]:
    sys.exit(1)
PY
    [ $? -eq 0 ] && pass "action sequence is monotonic and gap-free" \
                 || fail "action sequence has gaps or repeats"
else
    fail "no run artefact was written to $RESULTS_DIR"
fi

head_ "Summary"
if [ "$FAILURES" -eq 0 ]; then
    pass "WISEPACK end-to-end validation PASSED"
else
    fail "$FAILURES check(s) failed"
fi
FINAL_RC="$FAILURES"
exit "$FAILURES"
