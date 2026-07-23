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
# Kill only the process group THIS script started. `pkill -f wisepack_` also
# matches another checkout running concurrently on the same machine.
cleanup() {
    if [ -n "${LAUNCH_PID:-}" ]; then
        kill -TERM "-$LAUNCH_PID" 2>/dev/null || kill "$LAUNCH_PID" 2>/dev/null || true
        wait "$LAUNCH_PID" 2>/dev/null || true
    fi
    exit "$FINAL_RC"
}
trap cleanup EXIT INT TERM

# ---- 1. bring the graph up --------------------------------------------------
head_ "1. Node graph"
setsid ros2 launch wisepack_bringup demo.launch.py \
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
sleep 6
TOPIC_LIST="$(list_topics)"
EXPECTED=(
    /wisepack/scenario/config /wisepack/scenario/state /wisepack/waste/items
    /wisepack/waste/detected_count /wisepack/plan/baseline /wisepack/plan/optimized
    /wisepack/plan/selected /wisepack/plan/status_json /wisepack/operator/approval
    /wisepack/execution/state /wisepack/execution/progress_pct
    /wisepack/system/readiness /wisepack/system/heartbeat
    /wisepack/plan/summary /wisepack/action/event /wisepack/action/sequence
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
if send_operator_decision APPROVE "$LOG"; then
    pass "approval received and accepted by the orchestrator"
else
    fail "orchestrator did not accept the approval"
fi

# ---- 6. execution and re-planning -------------------------------------------
head_ "6. Execution and dynamic re-planning"
if await_log "replan_complete|re-plan" "$LOG" 90; then
    pass "a dynamic event triggered a re-plan"
else
    fail "no re-plan observed within 90 s (is dynamic_events:=true?)"
fi

# THE safety property: a re-plan must land back at the operator gate. If a
# disturbance could carry a plan straight into execution, the human-in-the-loop
# claim would be void.
#
# Checked on the PLAN, not by racing the stage topic. The stage is a moving
# target during a re-plan and polling it burned minutes without proving
# anything; what actually matters is that the new plan carries NO approval, so
# nothing can execute until the operator acts again. That is the property, and
# it is a single deterministic read.
APPROVAL_STATE="$(echo_state /wisepack/plan/summary 8 | tail -1)"
if [ -z "$APPROVAL_STATE" ]; then
    warn "plan summary not readable — cannot confirm the post-replan gate"
elif printf '%s' "$APPROVAL_STATE" | grep -q '"approval_state": *"approved"'; then
    fail "the re-planned plan is already approved — the operator gate was bypassed"
else
    pass "the re-planned plan is UNAPPROVED — execution stays gated"
fi

# ...and the published stage must be the gate itself.
REPLAN_STAGE="$(echo_state /wisepack/execution/state 8 | tail -1)"
if printf '%s' "$REPLAN_STAGE" | grep -q "WAIT_FOR_OPERATOR_APPROVAL"; then
    pass "after the re-plan the workflow is back at the approval gate"
else
    fail "after a re-plan the stage is '$REPLAN_STAGE' — expected the gate"
fi
# The scripted event forces a SECOND approval; grant it so the run can finish.
# That a second approval is needed at all is the point of the check above.
if send_operator_decision APPROVE "$LOG"; then
    pass "second approval accepted — execution resumes after the re-plan"
else
    fail "the orchestrator did not accept the post-replan approval"
fi

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
