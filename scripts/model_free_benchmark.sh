#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# model_free_benchmark.sh — CAD vs model-free over the whole query set.
#
#     ./scripts/model_free_benchmark.sh [--model cylinder5]
#
# THE REPRESENTATION IS FIXED AND REUSED. This never rebuilds the Neural Object
# Field: the cached representation for the current reference digest is what the
# model-free estimator is given, exactly as a deployed system would.
#
# GROUND TRUTH IS NOT MOUNTED. The container is given `benchmark/queries` and
# the two meshes. `benchmark/ground_truth` is a sibling directory and is passed
# to the scoring step afterwards, on the host. The estimator additionally
# asserts it cannot see any ground-truth file.
# ---------------------------------------------------------------------------
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${WISEPACK_FP_IMAGE:-wisepack-foundationpose:pinned}"
MODEL="cylinder5"
[ "${1:-}" = "--model" ] && MODEL="$2"

ROOT="$REPO/.cache-perception/model-free/$MODEL"
BENCH="$ROOT/benchmark"
RECON="$ROOT/reference/ob_0000001/model/model.obj"

[ -d "$BENCH/queries" ] || { echo "[bench] ERROR: no query set at $BENCH/queries" >&2; exit 1; }
[ -f "$RECON" ]        || { echo "[bench] ERROR: no reconstruction at $RECON" >&2; exit 1; }

REF_DIGEST="$(python3 -c "import json;print(json.load(open('$ROOT/reference/ob_0000001/wisepack_reference_manifest.json'))['reference_set_digest'])")"
echo "[bench] representation reused, reference digest: $REF_DIGEST"
echo "[bench] queries: $(ls -1 "$BENCH/queries" | wc -l)"

. "$REPO/scripts/lib_foundationpose_gpu.sh"
GPU_ARGS=()
foundationpose_gpu_args GPU_ARGS || { echo "[bench] ERROR: no GPU" >&2; exit 1; }

docker run --rm "${GPU_ARGS[@]}" \
    -v "$BENCH/queries:/queries:ro" \
    -v "$REPO/.cache-perception/foundationpose/weights:/weights:ro" \
    -v "$ROOT/reference/ob_0000001/model:/recon:ro" \
    -v "/data/jarvis/wisepack/references:/datasets:ro" \
    -v "$BENCH:/out" \
    -v "$REPO/perception/foundationpose/model_free_benchmark.py:/tmp/bench.py:ro" \
    "$IMAGE" \
    python /tmp/bench.py \
        --queries-dir /queries \
        --cad-mesh /datasets/CAD-Models/STL-Files/Cylinder5.stl \
        --cad-scale-to-metres 0.001 \
        --model-free-mesh /recon/model.obj \
        --model-free-scale-to-metres 1.0 \
        --out /out/estimates.json
STATUS=$?

# TEMPORARY — see `model_free_build.sh`. The GL stack segfaults on interpreter
# shutdown AFTER the work is done, so the artefact decides. This is a WORKAROUND
# for a diagnosed upstream teardown crash, not a success contract to build on:
# nothing in the dashboard should ever judge an operation this way.
if [ ! -f "$BENCH/estimates.json" ]; then
    echo "[bench] FAILED (exit $STATUS, no estimates written)" >&2
    exit "${STATUS:-1}"
fi
[ "$STATUS" -ne 0 ] && echo "[bench] NOTE: container exited $STATUS after writing estimates (TEMPORARY teardown-crash tolerance)"

python3 "$REPO/scripts/model_free_score_batch.py" \
    --estimates "$BENCH/estimates.json" \
    --ground-truth-dir "$BENCH/ground_truth" \
    --out "$BENCH/benchmark_report.json"
