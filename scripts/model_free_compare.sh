#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# model_free_compare.sh — one query frame, two estimators, scored afterwards.
#
#     ./scripts/model_free_compare.sh [--model cylinder5]
#
# THE CONTROL. Both estimators receive the SAME query — same RGB, same depth,
# same intrinsics, same instance mask, same camera pose, same object pose, same
# rendered material — and differ in exactly one input:
#
#     CAD         the exact Cylinder5 mesh
#     model-free  ONLY the mesh reconstructed by the Neural Object Field from
#                 the rendered reference views
#
# GROUND TRUTH CANNOT REACH EITHER ESTIMATOR, and that is enforced by mounts
# rather than by discipline. The estimator container is given the query frame's
# IMAGES and the two meshes; `ground_truth.json` lives in the same directory as
# the images, so it is copied to a scratch query directory WITHOUT it. Scoring
# runs afterwards, on the host, from the untouched original.
# ---------------------------------------------------------------------------
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${WISEPACK_FP_IMAGE:-wisepack-foundationpose:pinned}"
MODEL="cylinder5"
[ "${1:-}" = "--model" ] && MODEL="$2"

SOURCE="$REPO/.cache-perception/isaac-reference/stage_a_workcell"
ROOT="$REPO/.cache-perception/model-free/$MODEL"
RECON="$ROOT/reference/ob_0000001/model/model.obj"
OUT="$ROOT/comparison"
QUERY="$OUT/query"

[ -f "$RECON" ] || { echo "[compare] ERROR: no reconstruction at $RECON" >&2; exit 1; }
[ -f "$SOURCE/ground_truth.json" ] || { echo "[compare] ERROR: no query at $SOURCE" >&2; exit 1; }

# THE QUERY, WITHOUT THE ANSWER. Images and intrinsics only — `ground_truth.json`
# is deliberately not copied, so the container physically cannot read it.
rm -rf "$QUERY"; mkdir -p "$QUERY"
cp -r "$SOURCE/rgb" "$SOURCE/depth" "$SOURCE/masks" "$SOURCE/cam_K.txt" "$QUERY/"
if [ -e "$QUERY/ground_truth.json" ]; then
    echo "[compare] ERROR: ground truth leaked into the query directory" >&2; exit 1
fi
echo "[compare] query prepared without ground truth: $QUERY"

. "$REPO/scripts/lib_foundationpose_gpu.sh"
GPU_ARGS=()
foundationpose_gpu_args GPU_ARGS || { echo "[compare] ERROR: no GPU" >&2; exit 1; }

mkdir -p "$OUT"
docker run --rm "${GPU_ARGS[@]}" \
    -v "$QUERY:/query:ro" \
    -v "$REPO/.cache-perception/foundationpose/weights:/weights:ro" \
    -v "$ROOT:/model-free:ro" \
    -v "/data/jarvis/wisepack/references:/datasets:ro" \
    -v "$OUT:/out" \
    -v "$REPO/perception/foundationpose/model_free_query.py:/tmp/query.py:ro" \
    "$IMAGE" \
    python /tmp/query.py \
        --query-dir /query \
        --cad-mesh /datasets/CAD-Models/STL-Files/Cylinder5.stl \
        --cad-scale-to-metres 0.001 \
        --model-free-mesh /model-free/reference/ob_0000001/model/model.obj \
        --model-free-scale-to-metres 1.0 \
        --out /out/estimates.json
STATUS=$?

# THE SAME RULE AS THE BUILD: the GL stack segfaults on teardown after the work
# is done, so the artefact decides, and a crash before it still fails.
if [ ! -f "$OUT/estimates.json" ]; then
    echo "[compare] FAILED (exit $STATUS, no estimates written)" >&2
    exit "${STATUS:-1}"
fi
[ "$STATUS" -ne 0 ] && echo "[compare] NOTE: container exited $STATUS after writing estimates (teardown crash)"

echo "[compare] estimates: $OUT/estimates.json"
echo "[compare] scoring against ground truth NOW, on the host, after both estimates exist"
python3 "$REPO/scripts/model_free_score.py" \
    --estimates "$OUT/estimates.json" \
    --ground-truth "$SOURCE/ground_truth.json" \
    --out "$OUT/comparison.json"
