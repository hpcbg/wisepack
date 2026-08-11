#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# physical_model_free.sh — SIM-REFERENCE -> PHYSICAL D435 model-free test.
#
#     ./scripts/physical_model_free.sh --roi 255,70,445,719 --frames 12
#     ./scripts/physical_model_free.sh --roi ... --frames 5 --label "pose 2"
#     ./scripts/physical_model_free.sh --dataset cylinder5-2026... --roi ...
#
# WHAT IS BEING TESTED: whether a Neural Object Field built ONLY from RENDERED
# reference views can locate the real object in a real camera's frame. The
# representation is the cached one, reused untouched — it has never seen a
# photograph, and nothing here retrains it. That is the whole experiment: if it
# were rebuilt from physical images this would measure something else.
#
# THE OPERATOR MUST HAVE PLACED THE PART. Which part is in the ROI is stated
# with --model-id and can never be inferred; several WISEPACK tubes share an
# outer diameter and differ only in length.
#
# WHAT THIS CANNOT REPORT: accuracy. No independently measured physical pose
# for this object exists, so there is no error to compute. It reports how well
# each method agrees with ITSELF across frames (repeatability) and how closely
# the two agree with EACH OTHER (agreement). Two estimators sharing a camera,
# a frame and a mask can be wrong together, so agreement is not accuracy.
#
# CAD NEVER REACHES THE MODEL-FREE ESTIMATOR. The container is given the STL
# and the reconstruction as two separate meshes, and each estimator is built
# from exactly one of them. The shared mask is produced from depth and
# intrinsics alone, with no model of any kind.
# ---------------------------------------------------------------------------
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${WISEPACK_FP_IMAGE:-wisepack-foundationpose:pinned}"
MODEL="cylinder5"
FRAMES=12
ROI=""
DATASET=""
LABEL="pose 1"
RUN="run1"

while [ $# -gt 0 ]; do
    case "$1" in
        --model-id) MODEL="$2"; shift 2;;
        --frames)   FRAMES="$2"; shift 2;;
        --roi)      ROI="$2"; shift 2;;
        --dataset)  DATASET="$2"; shift 2;;
        --label)    LABEL="$2"; shift 2;;
        --run)      RUN="$2"; shift 2;;
        *) echo "[physical-mf] unknown argument: $1" >&2; exit 2;;
    esac
done

ROOT="$REPO/.cache-perception/model-free/$MODEL"
RECON="$ROOT/reference/ob_0000001/model/model.obj"
MANIFEST="$ROOT/reference/ob_0000001/wisepack_reference_manifest.json"
OUT="$ROOT/physical/$RUN"

[ -f "$RECON" ] || { echo "[physical-mf] ERROR: no reconstruction at $RECON" >&2; exit 1; }

# THE REPRESENTATION IS REUSED, NEVER REBUILT, and its digest is printed so the
# report can name the exact thing that produced the model-free estimates.
REF_DIGEST="$(python3 -c "import json;print(json.load(open('$MANIFEST'))['reference_set_digest'])")"
CAD_TO_MF="$(python3 -c "import json;print(json.load(open('$MANIFEST')).get('cad_supplied_to_estimator'))")"
echo "[physical-mf] model-free representation: digest $REF_DIGEST (simulated reference views only)"
echo "[physical-mf] cad_supplied_to_estimator: $CAD_TO_MF"
if [ "$CAD_TO_MF" != "False" ]; then
    echo "[physical-mf] ERROR: the representation records CAD exposure; it is not a model-free representation" >&2
    exit 1
fi

# ------------------------------------------------------------------ prepare
PREP_ARGS=(--model-id "$MODEL" --frames "$FRAMES" --out "$OUT/frames")
[ -n "$ROI" ]     && PREP_ARGS+=(--roi "$ROI")
[ -n "$DATASET" ] && PREP_ARGS+=(--dataset "$DATASET")
python3 "$REPO/scripts/physical_model_free_prepare.py" "${PREP_ARGS[@]}" || exit 1

CAPTURE_ROOT="$(python3 -c "import json;print(json.load(open('$OUT/frames/prepare_manifest.json'))['capture_root'])")"
# THE REFERENCE POINT COMES FROM THE REGISTRY, so the physical numbers measure
# the same point on the object that the simulated benchmark measured.
CENTRE="$(PYTHONPATH="$REPO/wisepack_ws/src/wisepack_core" python3 -c \
    "from wisepack_core.rgbd import load_object_registry
print(','.join(str(float(v)) for v in load_object_registry(repo_root='$REPO').models['$MODEL'].model_center_mm))")"
if [ -z "$CENTRE" ]; then
    echo "[physical-mf] ERROR: could not read model_center_mm for $MODEL from the registry" >&2
    exit 1
fi
echo "[physical-mf] reference point (mesh frame, mm): $CENTRE"

# ----------------------------------------------------------------- estimate
. "$REPO/scripts/lib_foundationpose_gpu.sh"
GPU_ARGS=()
foundationpose_gpu_args GPU_ARGS || { echo "[physical-mf] ERROR: no GPU" >&2; exit 1; }

docker run --rm "${GPU_ARGS[@]}" \
    -v "$OUT/frames:/frames:ro" \
    -v "$REPO/.cache-perception/foundationpose/weights:/weights:ro" \
    -v "$ROOT/reference/ob_0000001/model:/recon:ro" \
    -v "/data/jarvis/wisepack/references:/datasets:ro" \
    -v "$OUT:/out" \
    -v "$REPO/perception/foundationpose/model_free_benchmark.py:/tmp/run.py:ro" \
    "$IMAGE" \
    python /tmp/run.py \
        --queries-dir /frames \
        --cad-mesh /datasets/CAD-Models/STL-Files/Cylinder5.stl \
        --cad-scale-to-metres 0.001 \
        --model-free-mesh /recon/model.obj \
        --model-free-scale-to-metres 1.0 \
        --model-id "$MODEL" \
        `# --opt=value, because the reference point's first component is` \
        `# negative and argparse would read a bare -130.0,... as a flag` \
        --model-center-mm="$CENTRE" \
        --reference-digest "$REF_DIGEST" \
        --capture-root "$CAPTURE_ROOT" \
        --label "$LABEL" \
        --out /out/estimates.json
STATUS=$?

# TEMPORARY, and the same rule as the simulated runs: the GL stack segfaults on
# interpreter shutdown AFTER the work is done, so the artefact decides. This is
# a workaround for a diagnosed upstream teardown crash — see the known-issues
# note — and must never become a dashboard success contract.
if [ ! -f "$OUT/estimates.json" ]; then
    echo "[physical-mf] FAILED (exit $STATUS, no estimates written)" >&2
    exit "${STATUS:-1}"
fi
[ "$STATUS" -ne 0 ] && echo "[physical-mf] NOTE: container exited $STATUS after writing estimates (TEMPORARY teardown-crash tolerance)"

# -------------------------------------------------------------------- score
python3 "$REPO/scripts/physical_repeatability_score.py" \
    --estimates "$OUT/estimates.json" \
    --label "$LABEL" \
    --out "$OUT/physical_report.json"
