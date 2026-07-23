#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_wisepack_demo.sh — the ONE COMMAND acceptance demonstration.
#
#     ./run_wisepack_demo.sh                 # full: Docker + ROS 2 + FIWARE
#     ./run_wisepack_demo.sh --no-fiware     # ROS 2 only (no Orion-LD)
#     ./run_wisepack_demo.sh --core-only     # pure Python, no Docker at all
#
# A fresh machine with Docker can run the default form. Nothing else is needed:
# no host ROS 2, no host Python packages, no manual configuration.
#
# WHAT IT DOES, in order:
#    1. build the Docker image
#    2. build the ROS 2 workspace
#    3. run the test suite
#    4. start the DDS-enabled Orion-LD stack and generate its mapping
#    5. start the WISEPACK ROS 2 nodes
#    6. show baseline planning, then optimized planning
#    7. request and grant operator approval over ROS 2 / DDS
#    8. execute the simulated packaging
#    9. inject a dynamic event and re-plan
#   10. verify the action trail reached FIWARE
#   11. generate the final result artefacts and figures
#
# It is deterministic: the same seed produces the same containers, the same
# placements and the same container counts on any machine. Timings differ.
#
# Exit code == number of failed stages.
# ---------------------------------------------------------------------------
set -u

REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=scripts/lib_validate.sh
source "$REPO/scripts/lib_validate.sh"

IMAGE="${WISEPACK_IMAGE:-wisepack:jazzy}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ORION="${ORION:-http://localhost:1026}"
export WISEPACK_RESULTS_DIR="${WISEPACK_RESULTS_DIR:-$REPO/results}"
PRESET="${WISEPACK_PRESET:-mixed_pipes_dense}"
SEED="${WISEPACK_SEED:-42}"
DDS_DIR="$REPO/wisepack_ws/src/wisepack_fiware/dds"

WITH_FIWARE=1
CORE_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --no-fiware) WITH_FIWARE=0 ;;
        --core-only) CORE_ONLY=1; WITH_FIWARE=0 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$WISEPACK_RESULTS_DIR"

cat <<'BANNER'

  WISEPACK — Intelligent Robotic Sorting and Volume-Optimized Packaging
             of Nuclear Waste with Human-in-the-Loop AI
             acceptance demonstration

BANNER
info "preset       : $PRESET (seed $SEED)"
info "results      : $WISEPACK_RESULTS_DIR"
info "FIWARE       : $([ "$WITH_FIWARE" -eq 1 ] && echo "enabled ($ORION)" || echo disabled)"

# ---- Stage 0: core-only shortcut -------------------------------------------
if [ "$CORE_ONLY" -eq 1 ]; then
    head_ "Core-only mode — pure Python, no Docker, no ROS, no FIWARE"
    info "The SAME domain model, generator, optimizer, validator and KPIs the"
    info "full stack runs. Only the transport is missing."
    if python3 -m pytest "$REPO/tests" -q; then
        pass "test suite passed"
    else
        fail "test suite failed"
    fi
    python3 "$REPO/scripts/generate_demo_artifacts.py" || fail "artefact generation failed"
    head_ "Summary"
    [ "$FAILURES" -eq 0 ] && pass "core demonstration PASSED" || fail "$FAILURES stage(s) failed"
    exit "$FAILURES"
fi

# ---- Stage 1: image ---------------------------------------------------------
head_ "1. Docker image"
if ! command -v docker >/dev/null 2>&1; then
    fail "docker is not installed. Use ./run_wisepack_demo.sh --core-only instead."
    exit 1
fi
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    pass "image $IMAGE present"
else
    info "building $IMAGE (Vulcanexus Jazzy + py_trees + fastapi) — first run takes a while"
    if docker build -f "$REPO/Dockerfile.wisepack" -t "$IMAGE" "$REPO"; then
        pass "image built"
    else
        fail "image build failed"; exit "$FAILURES"
    fi
fi

"$REPO/scripts/clean_dds_shm.sh" || true

# ---- Stage 2: FIWARE --------------------------------------------------------
if [ "$WITH_FIWARE" -eq 1 ]; then
    head_ "2. Orion-LD DDS stack"
    VER="$(curl -s --max-time 3 "$ORION/version" 2>/dev/null)"
    if printf '%s' "$VER" | grep -qi 'orionld'; then
        pass "Orion-LD already running on $ORION"
    elif printf '%s' "$VER" | grep -qi 'orion'; then
        fail "port 1026 is served by a NON-LD Orion (NGSI-v2). Stop it, or use --no-fiware."
        WITH_FIWARE=0
    else
        info "generating the DDS -> NGSI-LD mapping from bridge_config.yaml"
        if ( cd "$DDS_DIR" && python3 generate_config.py --domain "$ROS_DOMAIN_ID" ) \
                | sed 's/^/      /'; then
            pass "context_broker_config.json generated"
        else
            fail "mapping generation failed"
        fi
        info "starting the Orion-LD DDS broker"
        ( cd "$DDS_DIR" && docker compose -f docker-compose.dds.yml up -d ) >/dev/null 2>&1
        if await_orion "$ORION" 90; then
            pass "Orion-LD DDS broker up on $ORION"
        else
            fail "Orion-LD did not come up within 90 s — continuing without FIWARE"
            WITH_FIWARE=0
        fi
    fi
fi

# ---- Stages 3-9: everything that needs ROS, inside the container ------------
head_ "3-9. Build, test and run the WISEPACK stack"
docker run --rm -i $([ -t 1 ] && echo -t) \
    --name wisepack-demo \
    --net=host --ipc=host --privileged \
    -e "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" \
    -e "WISEPACK_RESULTS_DIR=$WISEPACK_RESULTS_DIR" \
    -e "WISEPACK_PRESET=$PRESET" \
    -e "WISEPACK_SEED=$SEED" \
    -e "ORION=$ORION" \
    -v "$REPO:$REPO" -w "$REPO" \
    "$IMAGE" \
    bash -lc '
        set -u
        source /opt/vulcanexus/jazzy/setup.bash

        echo ""
        echo "  [3] building the ROS 2 workspace"
        if [ ! -f wisepack_ws/install/setup.bash ]; then
            ( cd wisepack_ws && colcon build --symlink-install ) || exit 10
        else
            echo "      already built"
        fi
        source wisepack_ws/install/setup.bash

        echo ""
        echo "  [4] test suite"
        python3 -m pytest tests -q || exit 11

        echo ""
        echo "  [5-9] end-to-end validation (approval gate, execution, re-plan)"
        ./validate_wisepack_e2e.sh
    '
E2E_RC=$?
if [ "$E2E_RC" -eq 0 ]; then
    pass "build, tests and end-to-end validation passed"
elif [ "$E2E_RC" -eq 10 ]; then
    fail "colcon build failed"
elif [ "$E2E_RC" -eq 11 ]; then
    fail "test suite failed"
else
    fail "end-to-end validation reported $E2E_RC failure(s)"
fi

# ---- Stage 10: FIWARE verification -----------------------------------------
if [ "$WITH_FIWARE" -eq 1 ]; then
    head_ "10. FIWARE audit-trail verification"
    "$REPO/run_vulcanexus_wisepack.sh" validate_fiware_action_log.sh
    if [ $? -eq 0 ]; then
        pass "the action trail is observable in Orion-LD"
    else
        fail "FIWARE verification reported failures"
    fi
fi

# ---- Stage 11: artefacts ----------------------------------------------------
head_ "11. Result artefacts and figures"
if python3 "$REPO/scripts/generate_demo_artifacts.py" > /tmp/wisepack_artifacts.log 2>&1; then
    pass "artefacts and figures generated"
    grep -E '^\s+(images|results)/' /tmp/wisepack_artifacts.log | head -20 | sed 's/^/    /'
else
    fail "artefact generation failed"
    tail -10 /tmp/wisepack_artifacts.log | sed 's/^/      /'
fi

# ---- Summary ----------------------------------------------------------------
head_ "Summary"
NEWEST="$(ls -1t "$WISEPACK_RESULTS_DIR"/wisepack-validation-*.md 2>/dev/null | head -1)"
if [ -n "$NEWEST" ]; then
    info "validation report: $(basename "$NEWEST")"
    sed -n '/^## Measured packing result/,/^## Proposal KPI/p' "$NEWEST" \
        | head -22 | sed 's/^/    /'
fi
echo ""
if [ "$FAILURES" -eq 0 ]; then
    pass "WISEPACK acceptance demonstration PASSED"
    info "dashboard:  ./run_wisepack_dashboard.sh          (live)"
    info "            ./run_wisepack_dashboard.sh sim      (no ROS, no FIWARE)"
else
    fail "$FAILURES stage(s) failed"
fi
exit "$FAILURES"
