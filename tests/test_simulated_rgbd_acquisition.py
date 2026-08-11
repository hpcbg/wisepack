"""Dashboard-native simulated RGB-D acquisition, and the separations it rests on.

THE FEATURE. `./run_wisepack_dashboard.sh` can now acquire a simulated
D435-compatible RGB-D frame, estimate a 6-DoF pose from it with the real
FoundationPose worker, and drive the ordinary workflow — without
`./scripts/stage_e.sh`, which used to start a second dashboard on another port
to show a result the first one could not produce.

WHAT MUST STAY TRUE, and what each check is for:

    ONE IMPLEMENTATION. The dashboard and the Stage B/C CLIs call the same
    `perception/simulated_rgbd_pipeline.py`. A second copy in `web/app.py` would
    make the demonstration and the evidence two different measurements wearing
    one name.

    GROUND TRUTH IS EVALUATION ONLY. Isaac knows where it put the object. That
    number may be READ AFTER an estimate exists, to score it. It may never enter
    the estimator, the observation, the batch, the planner or a placement.

    THE ANGULAR METRIC IS THE TUBE-AXIS LINE. A straight tube reversed end for
    end is the same object in the same place; scoring that as ~180 deg of error
    would report a correct pose as wrong.

    A DELAYED RESULT CANNOT OVERWRITE A NEWER RUN. An Isaac render plus
    inference is a minute of wall clock.

Mostly source-level for the dashboard parts: `web/app.py` imports FastAPI, which
this host deliberately does not have. The pipeline's own arithmetic is exercised
directly, with no worker, no GPU and no Isaac.
"""

from __future__ import annotations

import math
import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in (os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"),
             os.path.join(REPO, "perception")):
    if _pkg not in sys.path:
        sys.path.insert(0, _pkg)

from wisepack_core.acquisition import (                            # noqa: E402
    ACQUISITION_ISAAC, acquisition_provenance)
from wisepack_core.pose import (Orientation, axis_line_angle_deg)  # noqa: E402

import simulated_rgbd_pipeline as PIPE                             # noqa: E402

APP = os.path.join(REPO, "web", "app.py")
PROVIDER = os.path.join(REPO, "perception", "providers",
                        "foundationpose_rgbd.py")
STAGE_B = os.path.join(REPO, "scripts", "stage_b_foundationpose.py")
STAGE_C = os.path.join(REPO, "scripts", "stage_c_workarea.py")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _code_only(source: str) -> str:
    """Python with docstrings and comments removed.

    These modules EXPLAIN what they must not do and have to NAME those things to
    explain them — "stage_e.sh", "ground truth" and "FoundationPose" all appear
    in prose saying they are not reimplemented. A check that could not tell the
    explanation from the behaviour would forbid writing the reason down.
    """
    source = re.sub(r'"""(?:.|\n)*?"""', "", source)
    return "\n".join(line.split("#")[0] for line in source.splitlines())


# --------------------------------------------------------------------------- #
# Provenance: simulated, and never dressed as physical
# --------------------------------------------------------------------------- #


def test_the_acquisition_names_itself_and_its_provenance():
    assert acquisition_provenance(ACQUISITION_ISAAC) == "simulated"
    assert PIPE.SIMULATED_NOTE
    assert "RENDERED" in PIPE.SIMULATED_NOTE


def test_the_provider_has_a_simulated_acquisition_of_its_own():
    """NOT `acquire_reference`, which stamps "must not be planned against"."""
    source = _read(PROVIDER)
    assert "def acquire_simulated" in source
    body = source[source.index("def acquire_simulated"):
                  source.index("def _estimate")]
    assert "ACQUISITION_ISAAC" in body
    assert "ACQUISITION_REFERENCE" not in _code_only(body), (
        "a simulated workcell run DOES plan, validate and reach the approval "
        "gate; stamping it `reference` tells the operator the opposite")


def test_the_frame_provenance_says_simulated_not_a_real_d435():
    scene = {"acquisition_backend": "isaac_sim",
             "camera_profile": "d435_compatible_simulated",
             "mask_source": "isaac_instance_gt", "mask_provenance": "synthetic"}
    document = PIPE.frame_provenance(scene)
    assert document["acquisition"] == ACQUISITION_ISAAC
    assert document["provenance"] == "simulated"
    assert document["camera_profile"] == "d435_compatible_simulated"
    # Never a claim of sensor fidelity that has not been validated.
    assert "d435_compatible" in document["camera_profile"]


# --------------------------------------------------------------------------- #
# Ground truth is EVALUATION ONLY
# --------------------------------------------------------------------------- #


def test_the_estimator_is_never_given_the_scene():
    """THE STRUCTURAL GUARD. `estimate()` cannot read a ground-truth pose,
    because it never receives one — this is a signature, not a convention."""
    import inspect
    parameters = set(inspect.signature(PIPE.estimate).parameters)
    assert parameters == {"model_id", "refine_iterations", "batch_id",
                          "provider"}, parameters
    for forbidden in ("scene", "truth", "ground_truth", "settled"):
        assert forbidden not in parameters


def test_estimate_reads_no_ground_truth_file():
    source = _code_only(_read(PIPE.__file__))
    body = source[source.index("def estimate("):source.index("def workarea_transform")]
    for forbidden in ("load_scene", "T_camera_object", "settled_workarea",
                      "ground_truth"):
        assert forbidden not in body, (
            f"estimate() touches {forbidden!r}; ground truth must not exist on "
            "the perception path at all")


def test_the_evaluation_functions_require_an_estimate_first():
    """Ordering enforced by the call signature: the observation is an argument."""
    import inspect
    for function in (PIPE.evaluate_camera_frame, PIPE.evaluate_workarea):
        parameters = list(inspect.signature(function).parameters)
        assert parameters[0] == "observation", (
            f"{function.__name__} must take the ESTIMATE as its first argument, "
            "so it cannot run before one exists")


def test_the_workflow_batch_carries_no_ground_truth():
    """What crosses into planning is an estimate and CAD geometry. Nothing else."""
    source = _code_only(_read(PIPE.__file__))
    body = source[source.index("def workarea_batch("):
                  source.index("def _task_axis")]
    for forbidden in ("evaluation", "settled", "truth", "error_mm", "scene"):
        assert forbidden not in body, (
            f"the workflow batch mentions {forbidden!r}; ground truth must not "
            "reach the planner in any form")


def test_the_dashboard_does_not_feed_evaluation_back_into_anything():
    source = _code_only(_read(APP))
    body = source[source.index("def _acquire_simulated_rgbd"):
                  source.index("@app.post(\"/api/perception/simulated/acquire\")")]
    # The evaluation may be READ into the reply, and must not be handed to the
    # batch, the engine or the planner.
    assert "_apply_physical_batch(result.batch, token)" in body, (
        "the batch — not the document — is what enters the workflow")
    assert "evaluation" in body, "the comparison is reported for display"
    assert "batch.observations" not in body, (
        "the dashboard must not reach into the batch to alter an observation")


def test_the_optimizer_has_no_isaac_or_ground_truth_special_case():
    packing = _code_only(_read(os.path.join(
        REPO, "wisepack_ws", "src", "wisepack_core", "wisepack_core",
        "packing.py")))
    for forbidden in ("isaac", "ground_truth", "settled", "simulated_rgbd",
                      "foundationpose", "acquisition"):
        assert forbidden not in packing.lower(), (
            f"packing.py branches on {forbidden!r}; the optimizer must see "
            "generic items and nothing about where they were measured")


# --------------------------------------------------------------------------- #
# The ground-truth metric itself
# --------------------------------------------------------------------------- #


def _about_axis(axis, degrees):
    half = math.radians(degrees) / 2.0
    norm = math.sqrt(sum(v * v for v in axis))
    unit = [v / norm for v in axis]
    s = math.sin(half)
    return Orientation(x=unit[0] * s, y=unit[1] * s, z=unit[2] * s,
                       w=math.cos(half))


def test_position_error_is_the_distance_between_two_points():
    """Plain arithmetic, asserted so a refactor cannot quietly change it."""
    delta = [3.0, 4.0, 12.0]
    assert math.isclose(math.sqrt(sum(v * v for v in delta)), 13.0)


def test_an_end_for_end_reversal_is_zero_axis_error():
    """THE HEADLINE. A straight tube turned end for end is the same object in
    the same place, and the tube-axis LINE is undirected."""
    identity = Orientation(x=0.0, y=0.0, z=0.0, w=1.0)
    flipped = _about_axis((1.0, 0.0, 0.0), 180.0)   # z axis -> -z
    assert axis_line_angle_deg(identity, flipped, "z") == pytest.approx(0.0,
                                                                       abs=1e-6)


def test_arbitrary_spin_about_the_tube_axis_is_not_an_error():
    """A round tube's spin about its own axis is not a task quantity."""
    identity = Orientation(x=0.0, y=0.0, z=0.0, w=1.0)
    for degrees in (17.0, 90.0, 231.0):
        spun = _about_axis((0.0, 0.0, 1.0), degrees)
        assert axis_line_angle_deg(identity, spun, "z") == pytest.approx(
            0.0, abs=1e-6), f"{degrees} deg of spin was scored as error"


def test_a_genuine_tilt_is_reported():
    """The metric must still MEASURE something: it is not identically zero."""
    identity = Orientation(x=0.0, y=0.0, z=0.0, w=1.0)
    tilted = _about_axis((1.0, 0.0, 0.0), 5.0)
    assert axis_line_angle_deg(identity, tilted, "z") == pytest.approx(5.0,
                                                                      abs=1e-6)


def test_the_metric_is_the_shared_one_not_a_local_copy():
    source = _code_only(_read(PIPE.__file__))
    assert "axis_line_angle_deg" in source
    assert "def axis_line_angle_deg" not in source, (
        "the angular metric must be the validated shared implementation")


def test_no_benchmark_number_is_hard_coded():
    """§6: display the values calculated from the CURRENT run."""
    for path in (PIPE.__file__, APP, os.path.join(REPO, "web", "index.html")):
        text = _read(path)
        for headline in ("4.0 mm", "0.83", "0.926", "5.005"):
            assert headline not in text, (
                f"{headline} is hard-coded in {os.path.basename(path)}; the "
                "panel must show this run's own numbers")


# --------------------------------------------------------------------------- #
# Frames: simulated may transform, physical may not
# --------------------------------------------------------------------------- #


def test_the_transform_is_read_from_the_scene_and_is_never_invented():
    source = _code_only(_read(PIPE.__file__))
    body = source[source.index("def workarea_transform("):
                  source.index("def to_workarea(")]
    assert "T_workarea_camera" in body
    assert "RigidTransform" in body, (
        "the generic transform the physical camera will use with a measured "
        "extrinsic — only the source of the numbers differs")
    assert "return None" in body, (
        "a frame with no exported transform must yield NO transform, not a "
        "default one")


def test_a_missing_transform_leaves_the_pose_in_the_camera_frame():
    assert PIPE.workarea_transform({}) is None
    assert PIPE.workarea_transform({"workarea": {}}) is None


def test_the_physical_path_still_has_no_workarea_pose():
    """The simulated path's known transform must not leak into the physical one."""
    source = _read(APP)
    body = source[source.index("def api_perception_physical_acquire"):
                  source.index("def _acquire_simulated_rgbd")]
    assert '"workarea_pose_available": False' in body, (
        "the physical D435 has no measured extrinsic and must keep saying so")


def test_the_physical_result_carries_no_simulator_ground_truth():
    physical = _read(os.path.join(REPO, "perception", "physical_pipeline.py"))
    code = _code_only(physical)
    for forbidden in ("ground_truth", "T_camera_object", "settled_workarea",
                      "isaac"):
        assert forbidden not in code.lower(), (
            f"the physical pipeline references {forbidden!r}; no simulator "
            "ground truth exists for a physical object and none may appear")


# --------------------------------------------------------------------------- #
# Stale results
# --------------------------------------------------------------------------- #


def test_the_stale_guard_uses_the_existing_identifiers():
    source = _code_only(_read(APP))
    body = source[source.index("def run_token("):
                  source.index("def _apply_physical_batch")]
    assert "run_id" in body and "scenario_revision" in body, (
        "the guard must reuse the run and revision the workflow already "
        "stamps, not a second scheme")


def test_a_superseded_result_is_refused_before_the_workflow_sees_it():
    source = _read(APP)
    body = source[source.index("def _apply_physical_batch"):
                  source.index("@app.get(\"/api/perception/physical/models\")")]
    guard = body[:body.index("with STATE.lock")]
    assert "superseded_reason(token)" in guard, (
        "the check must happen BEFORE the engine or the orchestrator is "
        "touched, or a late result has already replaced the newer run")


def test_both_slow_acquisitions_capture_a_token_before_the_slow_part():
    source = _read(APP)
    for name, following in (
            ("def api_perception_physical_acquire", "run_physical("),
            ("def _acquire_simulated_rgbd", "run_simulated(")):
        body = source[source.index(name):]
        body = body[:body.index(following)]
        assert "token = run_token()" in body, (
            f"{name} must capture the run token before the slow call, or the "
            "token describes the run the result would overwrite")


# --------------------------------------------------------------------------- #
# Architectural guards
# --------------------------------------------------------------------------- #


def test_the_dashboard_contains_no_second_foundationpose_implementation():
    """No estimation, no transform arithmetic and no error metric in the server.

    `acquire_reference(` is deliberately NOT forbidden: the offline self-test
    under Diagnostics runs the saved tutorial dataset through the provider, which
    is a regression of the provider itself and never touches a run. What must not
    appear is the SIMULATED acquisition path, which belongs to the shared
    pipeline the CLIs also call.
    """
    code = _code_only(_read(APP))
    for forbidden in ("acquire_simulated(", "T_workarea_camera",
                      "apply_to_orientation", "apply_to_position",
                      "axis_line_angle_deg", "settled_workarea",
                      "T_camera_object"):
        assert forbidden not in code, (
            f"web/app.py performs {forbidden!r}; estimation, transforms and "
            "evaluation belong to the shared pipeline")


def test_the_dashboard_never_shells_out_to_a_stage_script():
    code = _code_only(_read(APP))
    for forbidden in ("stage_e", "stage_c.sh", "stage_b.sh", "subprocess",
                      "os.system", "popen"):
        assert forbidden not in code.lower(), (
            f"web/app.py references {forbidden!r}; the dashboard calls the "
            "library, never a script, and never parses a script's output")


def test_the_operator_command_is_generic_not_model_specific():
    source = _read(APP)
    commands = re.findall(r'if command == "([a-z_]+)"', source)
    for command in commands:
        assert "cylinder" not in command, f"{command} names a specific part"
        assert "c5" not in command.split("_"), f"{command} names a specific part"
    # And the batch that crosses to the orchestrator is the generic one.
    assert '"submit_observation_batch"' in source


def test_both_stage_clis_are_thin_wrappers_over_the_shared_library():
    for path in (STAGE_B, STAGE_C):
        code = _code_only(_read(path))
        assert "simulated_rgbd_pipeline" in code, (
            f"{os.path.basename(path)} must call the shared library")
        # The algorithms are no longer here.
        for forbidden in ("provider.acquire_", "RigidTransform("):
            assert forbidden not in code, (
                f"{os.path.basename(path)} still performs {forbidden!r} itself; "
                "the extraction has not actually removed the duplicate")


def test_stage_e_remains_available_as_a_regression_helper():
    """It is no longer REQUIRED, and it must not be broken either."""
    script = _read(os.path.join(REPO, "scripts", "stage_e.sh"))
    assert "acquire_simulated_rgbd" in script
    assert "run_wisepack_dashboard.sh" in script, (
        "the script should point at the ordinary path it is no longer needed "
        "for")


def test_the_command_alias_delegates_rather_than_reimplementing():
    source = _read(APP)
    handler = source[source.index('if command == "acquire_simulated_rgbd"'):
                     source.index('if command == "detect_physical_objects"')]
    assert "_acquire_simulated_rgbd(" in handler
    code = _code_only(handler)
    for forbidden in ("ObservationBatch(", "generate_plans", "start_run("):
        assert forbidden not in code, (
            f"the command alias performs {forbidden!r}; it must delegate to the "
            "one implementation the endpoint uses")
