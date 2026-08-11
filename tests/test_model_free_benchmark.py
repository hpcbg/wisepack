"""The model-free benchmark's integrity, tested without Isaac, GPU or images.

WHY THESE EXIST
---------------
The first run of this benchmark produced ten queries, ten CAD estimates, ten
model-free estimates, and a full set of aggregate statistics — all of them
wrong, because the ground truth was read several simulation steps before the
frame was rendered and the workpiece was still moving. Nothing failed. The
numbers were plausible, internally consistent, and reported CAD at 15 mm on a
path already measured at 3.3 mm.

The two properties that would have caught it are the two properties tested
here: that the ground truth is read AFTER the frame exists with physics
stopped, and that every query is checked by reprojecting the CAD onto its own
mask before it is allowed into the set. Both are source-level checks, because
the code they guard needs Isaac to run and this machine has no Isaac in CI.
A source-level check is weaker than an execution check and is not a substitute
for one — it is here to make a silent removal loud.

The third group is about GROUND-TRUTH SEPARATION, which is the claim the whole
comparison rests on: the estimator must not be able to read the answer. That is
enforced by mounts, and mounts are text in a shell script, so text is the right
thing to assert about.
"""

from __future__ import annotations

import importlib.util
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR = os.path.join(REPO, "simulators", "isaac",
                         "generate_model_free_queries.py")
BENCH_SH = os.path.join(REPO, "scripts", "model_free_benchmark.sh")
BENCH_PY = os.path.join(REPO, "perception", "foundationpose",
                        "model_free_benchmark.py")
SCORER = os.path.join(REPO, "scripts", "model_free_score_batch.py")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _load_generator():
    """Import the generator module WITHOUT Isaac.

    It imports `isaacsim` inside `main()` rather than at module scope, so the
    pure planning code can be exercised anywhere. If that ever stops being true
    this import fails, which is the correct outcome: the plan would then be
    untestable off a simulator host.
    """
    spec = importlib.util.spec_from_file_location(
        "wisepack_model_free_queries", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# 1. The query plan
# --------------------------------------------------------------------------- #


def test_the_query_plan_is_deterministic():
    """No randomness: the same benchmark must be reproducible from the file."""
    module = _load_generator()
    assert module._query_plan() == module._query_plan()


def test_every_query_attribute_is_defined_for_every_candidate():
    """A SHORT LIST WOULD SILENTLY MISALIGN the poses. The plan is four
    hand-written parallel lists, and a candidate that took its yaw from one
    index and its camera from another would still render, still score, and be
    wrong in a way no output would show."""
    plan = _load_generator()._query_plan()
    for entry in plan:
        assert set(entry) == {"index", "offset_m", "yaw_deg", "tilt_deg",
                              "camera"}
        assert len(entry["offset_m"]) == 2
        assert len(entry["camera"]) == 3
    assert [e["index"] for e in plan] == list(range(len(plan)))


def test_there_are_more_candidates_than_the_requested_query_count():
    """QUERIES ARE REJECTED, so candidates must outnumber them. Otherwise the
    only way to reach the requested count is to weaken a gate — which is the
    one thing that must not happen to reach a number."""
    assert len(_load_generator()._query_plan()) > 12


# --------------------------------------------------------------------------- #
# 2. When the ground-truth pose is read (the defect that shipped)
# --------------------------------------------------------------------------- #


def _generator_body() -> str:
    return "\n".join(line for line in _read(GENERATOR).splitlines()
                     if not line.lstrip().startswith("#"))


def test_physics_is_stopped_before_the_ground_truth_pose_is_read():
    """`rep.orchestrator.step` advances the simulation, and both `warmup` and
    `capture` call it. With the timeline playing, the body moved 14-39 mm
    between the pose read and the render."""
    body = _generator_body()
    assert "app_utils.pause()" in body
    assert body.index("app_utils.pause()") < body.index("camera.capture()")


def test_the_pose_is_read_again_after_the_frame_and_must_agree():
    """The pose and the pixels have to describe the same instant, and the only
    way to know they do is to look twice."""
    body = _generator_body()
    after_capture = body[body.index("camera.capture()"):]
    assert "object_pose_now()" in after_capture
    assert "moved_during_capture" in after_capture


def test_a_query_is_rejected_when_its_ground_truth_does_not_reproject():
    """THE GATE. Independent of the arithmetic that produced the pose: mask
    from the renderer, mesh from the registry, K from the camera profile."""
    body = _generator_body()
    assert "def reprojection_iou" in body
    assert "ground_truth_reprojection" in body
    threshold = re.search(r"if iou < (0\.\d+):", body)
    assert threshold, "the reprojection gate no longer compares against a number"
    assert float(threshold.group(1)) >= 0.85


def test_the_reprojection_check_uses_the_registry_mesh_not_the_scene_offsets():
    """A check that reused the numbers under test could not fail. The mesh is
    re-read through the registry so the two paths are independent."""
    body = _generator_body()
    assert "load_object_registry" in body
    assert "check_vertices" in body


def test_the_generated_ground_truth_records_its_own_verification():
    """The artefact carries the evidence, so a stale ground-truth file written
    before this check existed is distinguishable from one written after."""
    body = _generator_body()
    assert "reprojection_iou_against_own_mask" in body


# --------------------------------------------------------------------------- #
# 3. Ground-truth separation
# --------------------------------------------------------------------------- #


def test_the_benchmark_container_is_given_queries_but_never_ground_truth():
    """THE CLAIM THE COMPARISON RESTS ON, enforced by mounts rather than by
    remembering. `ground_truth` is a sibling of `queries` precisely so that
    mounting one cannot drag in the other."""
    script = "\n".join(line for line in _read(BENCH_SH).splitlines()
                       if not line.lstrip().startswith("#"))
    assert "$BENCH/queries:/queries:ro" in script
    for line in script.splitlines():
        if line.strip().startswith("-v ") and "ground_truth" in line:
            raise AssertionError(f"ground truth is mounted: {line.strip()}")


def test_the_estimator_asserts_it_cannot_see_the_answer():
    """Belt as well as braces: if the mount boundary were ever broken, this
    fails loudly instead of quietly producing a flattered result."""
    body = _read(BENCH_PY)
    assert "ground truth is visible to the estimator" in body
    assert "os.walk(args.queries_dir)" in body


def test_the_estimator_never_writes_a_ground_truth_derived_field():
    """Nothing the estimator emits may be a function of the answer."""
    document = _read(BENCH_PY)
    assert '"ground_truth_read": False' in document


def test_scoring_happens_on_the_host_after_the_estimates_exist():
    """Ordering is the point: the scorer reads `estimates.json`, which the
    container has already finished writing."""
    script = _read(BENCH_SH)
    assert script.index("estimates.json") < script.index("model_free_score_batch.py")
    assert "--ground-truth-dir" in script


# --------------------------------------------------------------------------- #
# 4. What the scorer reports
# --------------------------------------------------------------------------- #


def test_the_aggregate_reports_spread_and_not_only_a_mean():
    """Ten queries is a small sample. A mean alone would hide the difference
    between consistently close and occasionally very wrong."""
    body = _read(SCORER)
    for key in ("mean", "median", "std", "min", "max", "p90", "n"):
        assert f'"{key}"' in body


def test_the_axis_metric_is_undirected():
    """A straight tube reversed end for end is the same object in the same
    place; a directed angle would report 180 deg of error for no error."""
    assert "abs(float(a @ b))" in _read(SCORER)


def test_both_methods_are_scored_against_the_same_ground_truth_file():
    """One truth per query, read once, used for both — so the comparison
    cannot be biased by which file each method was scored against."""
    body = _read(SCORER)
    assert body.count("T_camera_object") == 1
    assert 'for tag in ("cad", "model_free")' in body
