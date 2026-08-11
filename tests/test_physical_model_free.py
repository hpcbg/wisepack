"""The physical sim-reference -> D435 experiment's integrity, without hardware.

THE TWO CLAIMS THIS EXPERIMENT MAKES, and therefore the two things worth
testing, are both about what did NOT happen:

  1. The model-free estimator never saw CAD — not the mesh, and not indirectly
     through the mask it shares with the CAD estimator.
  2. No accuracy was reported, because no independently measured physical pose
     for this object exists. Repeatability and inter-method agreement are not
     accuracy, and the difference is the whole reason the distinction is drawn.

A source-level check is weaker than an execution check. These exist so that
removing either property becomes loud rather than silent, on a machine with no
camera and no GPU.
"""

from __future__ import annotations

import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SH = os.path.join(REPO, "scripts", "physical_model_free.sh")
PREPARE = os.path.join(REPO, "scripts", "physical_model_free_prepare.py")
SCORER = os.path.join(REPO, "scripts", "physical_repeatability_score.py")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _executable_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))


def _load_scorer():
    spec = importlib.util.spec_from_file_location(
        "wisepack_physical_score", SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# 1. CAD must not reach the model-free estimator
# --------------------------------------------------------------------------- #


def test_the_two_meshes_are_separate_and_the_stl_is_only_the_cad_mesh():
    """The STL and the reconstruction are passed as two distinct arguments, so
    each estimator is built from exactly one of them."""
    script = _executable_lines(_read(RUN_SH))
    assert "--cad-mesh /datasets/CAD-Models/STL-Files/Cylinder5.stl" in script
    assert "--model-free-mesh /recon/model.obj" in script
    # The reconstruction mount is read-only and is the ONLY thing behind /recon.
    assert "/model:/recon:ro" in script


def test_the_run_refuses_a_representation_that_records_cad_exposure():
    """The manifest asserts the representation was built without CAD. If that
    ever stops being true the run must stop, not quietly continue."""
    script = _executable_lines(_read(RUN_SH))
    assert "cad_supplied_to_estimator" in script
    assert 'if [ "$CAD_TO_MF" != "False" ]' in script


def test_the_representation_is_reused_and_never_rebuilt_here():
    """SIM-TO-REAL IS THE EXPERIMENT. Rebuilding the representation from
    physical images would measure something else entirely."""
    script = _executable_lines(_read(RUN_SH))
    assert "model_free_build.sh" not in script
    assert "reference_set_digest" in script


def test_the_shared_mask_is_produced_without_any_model():
    """The mask is the one input both estimators share, so it is the only way
    CAD could reach the model-free path indirectly. `depth_plane_foreground`
    takes depth and intrinsics — no mesh, no model_id, no dimensions."""
    body = _executable_lines(_read(PREPARE))
    assert 'segment("depth_plane_foreground", depth_mm, intrinsics' in body
    assert '"mask_is_cad_free": True' in body
    # The model id is carried for provenance only; it must not be handed to the
    # segmenter as an input it could select on.
    assert "segment(\"depth_plane_foreground\", depth_mm, intrinsics, args.model_id" not in body


def test_one_mask_per_frame_is_written_once_and_shared():
    """If each estimator segmented independently, a pose difference could come
    from the masks rather than from the geometry each was given."""
    body = _executable_lines(_read(PREPARE))
    assert body.count("cv2.imwrite(f\"{fdir}/masks/000000.png\"") == 1


def test_no_mask_is_fabricated_when_validation_fails():
    """A frame with an unusable mask is excluded and recorded, never repaired."""
    body = _executable_lines(_read(PREPARE))
    assert "if not result.valid:" in body
    assert "MASK REJECTED" in body
    assert "dropped.append" in body


# --------------------------------------------------------------------------- #
# 2. No accuracy may be reported
# --------------------------------------------------------------------------- #


def test_the_report_states_that_no_physical_ground_truth_exists():
    body = _read(SCORER)
    assert '"physical_ground_truth_available": False' in body
    assert '"accuracy_reported": False' in body


def test_the_scorer_computes_no_error_against_any_truth():
    """There is no truth to compute an error against, so no ground-truth file
    is opened and no `*_error_*` quantity is produced."""
    body = _read(SCORER)
    assert "ground_truth" not in body.replace(
        "physical_ground_truth_available", "").replace(
        "ground_truth_note", "")
    for forbidden in ("position_error", "orientation_error", "accuracy_mm"):
        assert forbidden not in body


def test_repeatability_and_agreement_are_named_as_what_they_are():
    body = _read(SCORER)
    assert "REPEATABILITY" in body and "AGREEMENT" in body
    assert "not accuracy" in body.lower()


# --------------------------------------------------------------------------- #
# 3. The axis statistics are undirected
# --------------------------------------------------------------------------- #


def test_the_axis_angle_is_undirected():
    module = _load_scorer()
    assert module._line_angle((1, 0, 0), (-1, 0, 0)) < 1e-9
    assert abs(module._line_angle((1, 0, 0), (0, 1, 0)) - 90.0) < 1e-9


def test_the_mean_axis_is_invariant_to_sign_flips():
    """A SET OF UNDIRECTED AXES CANNOT BE AVERAGED AS VECTORS: `a` and `-a` are
    the same axis, and averaging them would cancel to nothing and report an
    enormous spread for a perfectly repeatable estimator."""
    import numpy as np
    module = _load_scorer()
    axes = [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0])]
    mean = module._mean_axis(axes)
    assert abs(abs(float(mean @ np.array([1.0, 0.0, 0.0]))) - 1.0) < 1e-9
    for axis in axes:
        assert module._line_angle(axis, mean) < 1e-6


def test_the_mean_axis_tracks_a_genuine_direction():
    import numpy as np
    module = _load_scorer()
    tilted = np.array([np.cos(np.radians(10.0)), np.sin(np.radians(10.0)), 0.0])
    mean = module._mean_axis([np.array([1.0, 0.0, 0.0]), tilted])
    assert 4.0 < module._line_angle(mean, np.array([1.0, 0.0, 0.0])) < 6.0
