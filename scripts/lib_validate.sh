#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# lib_validate.sh — shared plumbing for the WISEPACK validation scripts.
#
# Same visual grammar as TEMPO's scripts/lib_validate.sh and HARMONY's
# validate_dds_*.sh (head/pass/warn/fail/info, failure count as the exit code),
# factored out because WISEPACK has four of these.
#
# Source it, don't run it.
# ---------------------------------------------------------------------------

if [ -t 1 ]; then
    BOLD=$'\033[1m'; RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'
    CYN=$'\033[36m'; RST=$'\033[0m'
else
    BOLD=''; RED=''; GRN=''; YEL=''; CYN=''; RST=''
fi

FAILURES=0

hr()    { printf '%s\n' "------------------------------------------------------------"; }
head_() { printf '\n%s%s%s\n' "$BOLD$CYN" "$1" "$RST"; hr; }
pass()  { printf '  %s+%s %s\n' "$GRN" "$RST" "$1"; }
warn()  { printf '  %s!%s %s\n' "$YEL" "$RST" "$1"; }
fail()  { printf '  %sx%s %s\n' "$RED" "$RST" "$1"; FAILURES=$((FAILURES + 1)); }
info()  { printf '    %s\n' "$1"; }

# ROS 2 environment: Vulcanexus first, plain Jazzy fallback. The DDS behaviour
# WISEPACK's FIWARE path relies on is Vulcanexus's Fast DDS, but a plain Jazzy
# box should still run the logic and SAY that its FIWARE values will not fill.
wisepack_source_ros() {
    local ws="$1"
    ROS_SETUP=""
    set +u
    for s in /opt/vulcanexus/jazzy/setup.bash /opt/ros/jazzy/setup.bash; do
        if [ -f "$s" ]; then
            # shellcheck disable=SC1090
            source "$s"; ROS_SETUP="$s"; break
        fi
    done
    if [ -z "$ROS_SETUP" ]; then
        echo "ERROR: no ROS 2 environment found (looked for Vulcanexus Jazzy and ROS 2 Jazzy)."
        echo "       Run this inside the image: ./run_vulcanexus_wisepack.sh <script>"
        exit 1
    fi
    if [ -f "$ws/install/setup.bash" ]; then
        # shellcheck disable=SC1090
        source "$ws/install/setup.bash"
    else
        echo "ERROR: $ws is not built. Run ./run_vulcanexus_wisepack.sh, which builds it."
        exit 1
    fi
    set -u

    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
    case "$ROS_SETUP" in
        */vulcanexus/*) VULCANEXUS=1 ;;
        *)              VULCANEXUS=0 ;;
    esac
}

wisepack_report_env() {
    info "ROS env       : $ROS_SETUP"
    info "ROS_DOMAIN_ID : $ROS_DOMAIN_ID"
    if [ "$VULCANEXUS" -eq 1 ]; then
        pass "Vulcanexus detected — DDS TypeObject propagation as validated."
    else
        warn "Plain ROS 2 Jazzy (no Vulcanexus): topics will publish, but Orion-LD"
        info "values may stay \"uninitialized\" — a documented environment"
        info "requirement of the FIWARE DDS Enabler, not a WISEPACK defect."
    fi
}

# Wait for evidence rather than sleeping and hoping.
# await_log PATTERN LOGFILE TIMEOUT_S
await_log() {
    local pattern="$1" logfile="$2" timeout="${3:-30}" waited=0
    while [ "$waited" -lt "$timeout" ]; do
        if grep -qE "$pattern" "$logfile" 2>/dev/null; then return 0; fi
        sleep 1; waited=$((waited + 1))
    done
    return 1
}

# await_orion URL TIMEOUT_S — wait for an Orion-LD banner specifically.
await_orion() {
    local url="${1:-http://localhost:1026}" timeout="${2:-40}" waited=0
    while [ "$waited" -lt "$timeout" ]; do
        if curl -s --max-time 2 "$url/version" 2>/dev/null | grep -qi 'orionld'; then
            return 0
        fi
        sleep 1; waited=$((waited + 1))
    done
    return 1
}

# Topic discovery WITHOUT the ROS 2 daemon.
#
# `ros2 topic list` talks to a long-lived daemon that survives between runs and
# outlives the containers that started it. On a host-networked machine that has
# run many stacks, a stale daemon returns an EMPTY list and, with stderr
# discarded, the validation reports every topic as missing while the stack is
# demonstrably healthy. Reproduced here: 17/17 topics "missing" while the twin
# validator and the approval path were both working.
#
# --no-daemon does direct DDS discovery. It is a little slower and completely
# immune to daemon state, which is the right trade for a validation script.
list_topics() {
    local out
    out="$(ros2 topic list --no-daemon 2>&1)"
    if [ -z "$out" ] || printf '%s' "$out" | grep -qi 'error'; then
        sleep 3
        out="$(ros2 topic list --no-daemon 2>&1)"
    fi
    printf '%s\n' "$out"
}

# `ros2 topic echo` subscribes RELIABLE/VOLATILE by default, which matches
# NEITHER WISEPACK's BEST_EFFORT telemetry nor its TRANSIENT_LOCAL state topics —
# it would just sit there printing nothing. Ask for the QoS the publisher offers.
# Publish an operator decision and CONFIRM the orchestrator took it.
#
# `ros2 topic pub --once` fires as soon as it has one matching subscription,
# which on a busy host can be Orion-LD's bridge rather than the orchestrator.
# One unconfirmed publish is not evidence of an approval, so this retries until
# the orchestrator says so in its own log.
#   send_operator_decision APPROVE|REJECT LOGFILE [attempts]
send_operator_decision() {
    local decision="$1" logfile="$2" attempts="${3:-4}" n=0
    local marker="APPROVED via operator topic"
    [ "$decision" = "REJECT" ] && marker="REJECTED via operator topic"
    local before
    before="$(grep -c "$marker" "$logfile" 2>/dev/null || true)"
    before="${before:-0}"
    while [ "$n" -lt "$attempts" ]; do
        n=$((n + 1))
        timeout 15 ros2 topic pub --once /wisepack/operator/approval \
            std_msgs/msg/String "data: $decision" \
            --qos-reliability reliable --qos-durability transient_local \
            >/dev/null 2>&1
        local after
        for _ in 1 2 3 4 5 6; do
            after="$(grep -c "$marker" "$logfile" 2>/dev/null || true)"
            after="${after:-0}"
            [ "$after" -gt "$before" ] && return 0
            sleep 1
        done
    done
    return 1
}

# Latched-state read, with one retry: DDS discovery can need a moment on a busy
# host, and a single empty read is not evidence that a topic has no value.
echo_state() {
    local out
    out="$(timeout "${2:-8}" ros2 topic echo "$1" --once \
        --qos-reliability reliable --qos-durability transient_local 2>/dev/null \
        | grep '^data:')"
    if [ -z "$out" ]; then
        sleep 2
        out="$(timeout "${2:-8}" ros2 topic echo "$1" --once \
            --qos-reliability reliable --qos-durability transient_local 2>/dev/null \
            | grep '^data:')"
    fi
    printf '%s\n' "$out"
}
echo_telemetry() {
    timeout "${2:-8}" ros2 topic echo "$1" --once --qos-reliability best_effort \
        2>/dev/null | grep '^data:' || true
}
