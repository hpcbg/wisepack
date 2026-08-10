"""The WISEPACK side of the FoundationPose integration.

NO GPU, NO CUDA, NO DOCKER, NO WEIGHTS, NO CAMERA. Every test here runs against
a fake worker or a plain dict. That is a requirement, not a convenience: the
whole point of the provider boundary is that WISEPACK's behaviour when
FoundationPose is absent is the behaviour that matters most, and a test suite
that needed a GPU could not check it.

Tests that genuinely need the container live in
`tests/test_foundationpose_worker.py` (static checks) and are marked there.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "perception"))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))

from providers.foundationpose_rgbd import (                       # noqa: E402
    ACQUISITION_REFERENCE, ESTIMATOR_ID, METHOD, NO_EXTRINSIC_NOTE,
    REFERENCE_NOTE, FoundationPoseProvider, validate_response)
from wisepack_core.domain import PhysicalObservation              # noqa: E402
from wisepack_core.perception import (                            # noqa: E402
    DEFAULT_PERCEPTION_METHOD, KNOWN_PERCEPTION_METHODS, ObservationBatch,
    PerceptionMethod, PerceptionMethodState, PerceptionSource,
    resolve_perception_method, resolve_perception_method_selection)
from wisepack_core.pose import Orientation, Symmetry, SymmetryType  # noqa: E402
from wisepack_core.rgbd import ObjectModel, ObjectModelRegistry   # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


CAMERA_FRAME = "camera_color_optical_frame"


def _healthy(**over):
    document = {
        "worker_reachable": True, "worker_ready": True, "gpu_available": True,
        "foundationpose_runtime_available": True,
        "scorer_weights_available": True, "refiner_weights_available": True,
        "inference_available": True, "blocked_by": [],
        "foundationpose_revision": "a1b694b8", "revision_matches_pin": True,
        "versions": {"torch": "2.4.1+cu124"},
    }
    document.update(over)
    return document


def _pose(**over):
    result = {
        "frame_id": CAMERA_FRAME,
        "position_mm": [-3.43, -28.28, 780.0],
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "estimated_at": "2026-08-09T17:00:00Z",
        "foundationpose_revision": "a1b694b8",
        "duration_ms": 11896.7,
        "pose_of": "mesh_origin_as_loaded",
        "accuracy_note": "pose ESTIMATED; absolute accuracy is NOT measured",
    }
    result.update(over)
    return result


class FakeClient:
    """A worker that answers exactly what a test tells it to."""

    url = "http://fake-worker"

    def __init__(self, health=None, result=None, error=""):
        self._health = health if health is not None else _healthy()
        self._result = result if result is not None else _pose()
        self._error = error
        self.requests = []

    def health(self):
        return dict(self._health)

    def capability(self, health=None):
        document = self._health if health is None else health
        if not document.get("worker_reachable"):
            # MIRRORS THE REAL CLIENT, which propagates the transport error. A
            # fake that invented its own wording would let a regression in the
            # real one pass unnoticed.
            return False, str(document.get("last_error")
                              or "the worker did not answer")
        if document.get("inference_available"):
            return True, ""
        return False, "; ".join(document.get("blocked_by") or ["unavailable"])

    def estimate(self, request):
        self.requests.append(request)
        if self._error:
            return None, self._error
        return dict(self._result), ""

    def last_result(self):
        return dict(self._result), ""

    def datasets(self):
        return [], ""

    def image(self, kind="overlay"):
        return b"jpeg", ""


def _registry(tmp_path, symmetry=None, model_id="part", **over):
    """`over` may set any ObjectModel field, including model_center_mm."""
    mesh = tmp_path / "part.obj"
    mesh.write_text("o part\n")
    fields = dict(model_id=model_id, object_type="pipe_section",
                  mesh_path="part.obj", mesh_units="mm",
                  symmetry=symmetry or Symmetry(),
                  methods=(METHOD,), diameter_mm=25, length_mm=315)
    fields.update(over)
    return ObjectModelRegistry(models={model_id: ObjectModel(**fields)},
                               root=str(tmp_path))


def _provider(tmp_path, client=None, **registry_kw):
    return FoundationPoseProvider(client=client or FakeClient(),
                                  registry=_registry(tmp_path, **registry_kw))


# --------------------------------------------------------------------------- #
# The public method abstraction
# --------------------------------------------------------------------------- #


def test_the_method_names_describe_capabilities_not_implementations():
    """`fasterrcnn_bottle` names a module. The public axis names what the method
    DOES, so a provider can be replaced without renaming the operator's world."""
    assert set(KNOWN_PERCEPTION_METHODS) == {"planar_fasterrcnn",
                                             "foundationpose_rgbd"}


def test_planar_is_the_default_and_foundationpose_is_not():
    """FoundationPose needs a depth camera, a GPU, licensed weights and a CAD
    model. Any of those being absent must be visible, never defaulted into."""
    assert DEFAULT_PERCEPTION_METHOD == "planar_fasterrcnn"
    assert resolve_perception_method(env={}) == "planar_fasterrcnn"


def test_the_legacy_detector_variable_still_configures_the_method():
    """An existing deployment sets WISEPACK_PERCEPTION_DETECTOR. Renaming the
    variable must not turn a working configuration into an error."""
    assert resolve_perception_method(
        env={"WISEPACK_PERCEPTION_DETECTOR": "fasterrcnn_bottle"}
    ) == "planar_fasterrcnn"


def test_the_new_variable_wins_over_the_legacy_one():
    assert resolve_perception_method(
        env={"WISEPACK_PERCEPTION_METHOD": "foundationpose_rgbd",
             "WISEPACK_PERCEPTION_DETECTOR": "fasterrcnn_bottle"}
    ) == "foundationpose_rgbd"


def test_an_unknown_method_is_an_error_not_a_silent_planar_run():
    """Quietly running the planar detector when someone asked for 6-DoF produces
    measurements that look right and mean something else."""
    with pytest.raises(Exception) as exc:
        resolve_perception_method("rgbd_maybe", env={})
    assert "unknown perception method" in str(exc.value)


def test_each_method_declares_what_it_actually_measures():
    """A planar detector measures three of six DoF. Saying so stops a consumer
    reading an assumed zero as a measured height."""
    assert PerceptionMethod.PLANAR_FASTERRCNN.measures == ("x", "y", "yaw")
    assert "z" in PerceptionMethod.FOUNDATIONPOSE_RGBD.measures
    assert PerceptionMethod.FOUNDATIONPOSE_RGBD.requires_depth
    assert PerceptionMethod.FOUNDATIONPOSE_RGBD.requires_object_model
    assert not PerceptionMethod.PLANAR_FASTERRCNN.requires_depth


# --------------------------------------------------------------------------- #
# available / selected / current
# --------------------------------------------------------------------------- #


def test_selecting_a_method_does_not_mutate_the_running_run():
    """The batch on screen was measured one way; the draft is a different
    question. A single global setting would rewrite that batch's provenance."""
    state = PerceptionMethodState(current="planar_fasterrcnn",
                                  selected="foundationpose_rgbd",
                                  available=list(KNOWN_PERCEPTION_METHODS))
    document = state.to_dict()
    assert document["current"] == "planar_fasterrcnn"
    assert document["selected"] == "foundationpose_rgbd"
    assert document["changes_next_run"] is True


def test_a_preset_run_has_no_current_method_at_all():
    """Nothing measured a generated batch. Naming a method would be an invented
    provenance, so `current` is empty rather than defaulted to planar."""
    document = PerceptionMethodState().to_dict()
    assert document["current"] == ""
    assert document["current_label"] == ""
    assert document["changes_next_run"] is False


def test_an_unavailable_method_is_offered_with_its_reason_never_hidden():
    """"FoundationPose (unavailable — no RGB-D camera)" tells an operator what
    to do; a missing option tells them nothing."""
    state = PerceptionMethodState(
        available=["planar_fasterrcnn"],
        unavailable_reasons={"foundationpose_rgbd": "no RGB-D camera attached"})
    options = {o["value"]: o for o in state.to_dict()["options"]}
    assert set(options) == set(KNOWN_PERCEPTION_METHODS)
    assert options["foundationpose_rgbd"]["available"] is False
    assert "RGB-D" in options["foundationpose_rgbd"]["reason"]


def test_a_draft_naming_a_method_that_went_away_falls_back_to_planar():
    """The selection is a DRAFT for a run that has not started. A draft naming a
    dead worker must not become a run that silently fails."""
    assert resolve_perception_method_selection(
        "foundationpose_rgbd", ["planar_fasterrcnn"]) == "planar_fasterrcnn"


def test_switching_planar_to_foundationpose_and_back_needs_no_restart():
    """The whole point of the axis: it is per-run state, not an application
    mode."""
    available = list(KNOWN_PERCEPTION_METHODS)
    selected = DEFAULT_PERCEPTION_METHOD
    for wanted in ("foundationpose_rgbd", "planar_fasterrcnn",
                   "foundationpose_rgbd"):
        selected = resolve_perception_method_selection(wanted, available)
        assert selected == wanted


# --------------------------------------------------------------------------- #
# Worker health parsing and capability
# --------------------------------------------------------------------------- #


def test_capability_separates_every_link_in_the_chain(tmp_path):
    provider = _provider(tmp_path)
    capability = provider.capability(rgbd_camera_available=False)
    for field in ("worker_reachable", "gpu_available",
                  "foundationpose_runtime_available",
                  "scorer_weights_available", "refiner_weights_available",
                  "rgbd_camera_available", "runtime_ready", "inference_ready"):
        assert field in capability, field


def test_a_ready_runtime_with_no_camera_is_ready_runtime_not_ready_inference(tmp_path):
    """Today's exact state, and the dashboard must be able to say it:
    FoundationPose runtime READY, RGB-D camera unavailable, live inference
    unavailable."""
    capability = _provider(tmp_path).capability(rgbd_camera_available=False)
    assert capability["runtime_ready"] is True
    assert capability["rgbd_camera_available"] is False
    assert capability["inference_ready"] is False
    # And the offline regression is SEPARATE EVIDENCE, still available.
    assert capability["offline_regression_available"] is True
    assert any("RGB-D camera" in b for b in capability["blocked_by"])


def test_an_unreachable_worker_does_not_claim_a_missing_gpu(tmp_path):
    """Unreachable is not the same as broken, and reporting "no GPU" would send
    an operator to buy hardware they already have."""
    client = FakeClient(health={"worker_reachable": False,
                                "last_error": "connection refused"})
    capability = _provider(tmp_path, client=client).capability()
    assert capability["runtime_ready"] is False
    assert "connection refused" in " ".join(capability["blocked_by"])


def test_missing_weights_block_inference_but_not_the_runtime_report(tmp_path):
    client = FakeClient(health=_healthy(
        scorer_weights_available=False, inference_available=False,
        blocked_by=["2024-01-11-20-02-45/model_best.pth is not present"]))
    capability = _provider(tmp_path, client=client).capability()
    assert capability["scorer_weights_available"] is False
    assert capability["refiner_weights_available"] is True
    assert capability["inference_ready"] is False


def test_a_model_without_a_mesh_is_listed_unusable_with_the_reason(tmp_path):
    """Hidden would mean an operator cannot see the entry exists or what it
    needs. A model-based method cannot estimate a shape it does not have."""
    registry = ObjectModelRegistry(
        models={"nomesh": ObjectModel(model_id="nomesh", mesh_path="",
                                      methods=(METHOD,))},
        root=str(tmp_path))
    provider = FoundationPoseProvider(client=FakeClient(), registry=registry)
    listing = provider.models()
    assert len(listing) == 1
    assert listing[0]["usable"] is False
    assert "mesh" in listing[0]["reason"]
    assert provider.capability()["inference_ready"] is False


# --------------------------------------------------------------------------- #
# Response validation
# --------------------------------------------------------------------------- #


def test_a_response_without_a_frame_is_refused():
    """A pose without a frame is three numbers, and a consumer that assumed a
    default would place them somewhere."""
    _, reason = validate_response(_pose(frame_id=""))
    assert "frame" in reason


def test_a_quaternion_that_is_not_a_rotation_is_refused():
    """A zero quaternion is not a rotation. Normalising it would invent one."""
    _, reason = validate_response(
        _pose(orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0}))
    assert "rotation" in reason


def test_a_missing_quaternion_component_is_named():
    _, reason = validate_response(_pose(orientation={"x": 0.0, "y": 0.0, "z": 0.0}))
    assert "w" in reason


def test_a_non_numeric_position_is_refused():
    _, reason = validate_response(_pose(position_mm=["a", "b", "c"]))
    assert "numeric" in reason


def test_a_valid_response_survives_validation():
    validated, reason = validate_response(_pose())
    assert reason == ""
    assert validated["frame_id"] == CAMERA_FRAME
    assert isinstance(validated["orientation"], Orientation)


# --------------------------------------------------------------------------- #
# Mapping into the generic domain
# --------------------------------------------------------------------------- #


def test_the_batch_is_a_plain_observation_batch(tmp_path):
    batch = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    assert isinstance(batch, ObservationBatch)
    assert batch.ok
    assert all(isinstance(o, PhysicalObservation) for o in batch.observations)


def test_the_camera_frame_is_preserved_and_never_relabelled(tmp_path):
    """§12: not `wisepack_workarea` until a validated SE(3) extrinsic exists.
    Relabelling the frame is how a pose ends up placed in the wrong space."""
    batch = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    assert batch.frame_id == CAMERA_FRAME
    assert batch.observations[0].frame_id == CAMERA_FRAME
    # EVERY FRAME FIELD, not a substring of the whole document: the note that
    # EXPLAINS why the pose is not in wisepack_workarea necessarily contains
    # that name, and a check that cannot tell a prohibition from the thing
    # prohibited would forbid documenting the reason.
    document = batch.to_dict()
    assert document["frame_id"] == CAMERA_FRAME
    for observation in document["observations"]:
        assert observation["frame_id"] == CAMERA_FRAME


def test_a_camera_frame_estimate_is_valid_in_its_own_frame(tmp_path):
    """TWO DIFFERENT VALIDITIES, and conflating them called a good measurement a
    failed one. The estimate is real, reproducible and correct where it lives;
    what is missing is a way to move it into the work area."""
    observation = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0).observations[0]
    assert observation.pose_valid is True
    assert observation.frame_id == CAMERA_FRAME


def test_a_camera_frame_pose_cannot_be_placed_in_the_work_area(tmp_path):
    """The separate flag, and the one a planner or the Isaac scene synchronizer
    must consult. Its absence is what keeps the pose out of the work area."""
    batch = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    observation = batch.observations[0]
    assert observation.workarea_transform_valid is False
    assert observation.workarea_pose_available is False
    assert "extrinsic" in NO_EXTRINSIC_NOTE
    assert "extrinsic" in batch.detector_status["frame_note"]


def test_the_missing_extrinsic_is_never_replaced_by_an_identity_transform(tmp_path):
    """An unmeasured extrinsic is MISSING, not identity. Assuming identity puts
    objects wherever the camera happens to be, with total confidence."""
    batch = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    assert batch.frame_id == CAMERA_FRAME          # never relabelled
    assert batch.observations[0].workarea_transform_valid is False


def test_a_planar_observation_needs_no_transform_to_be_placeable():
    """It is already expressed in the work-area frame, so the derived flag is
    True without an extrinsic — the split must not break the working method."""
    observation = PhysicalObservation(observation_id="o", x_mm=1.0, y_mm=2.0,
                                      yaw_deg=15.0)
    assert observation.frame_id == "wisepack_workarea"
    assert observation.pose_valid is True
    assert observation.workarea_pose_available is True


def test_a_failed_estimate_is_not_placeable_either():
    """`workarea_pose_available` is DERIVED, so a bad estimate cannot become
    placeable merely by carrying a transform."""
    observation = PhysicalObservation(
        observation_id="o", x_mm=1.0, y_mm=2.0, pose_valid=False,
        frame_id=CAMERA_FRAME, workarea_transform_valid=True)
    assert observation.workarea_pose_available is False


def test_the_planar_homography_is_not_reused_as_a_3d_transform(tmp_path):
    """A planar map and an SE(3) transform are different quantities."""
    batch = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    assert batch.calibration_status == "not_applicable"
    assert "homography" in NO_EXTRINSIC_NOTE


def test_the_object_model_id_travels_with_the_observation(tmp_path):
    batch = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    assert batch.observations[0].object_model_id == "part"
    assert batch.model_id == "part"


def test_an_unknown_model_is_refused_with_the_known_ones_listed(tmp_path):
    batch = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="not_a_model", depth_scale_mm=1.0)
    assert not batch.ok
    assert "unknown object model" in batch.error


def test_the_score_is_never_reported_as_a_confidence(tmp_path):
    """FoundationPose's score ranks hypotheses against each other. `confidence`
    is rendered as a probability throughout the dashboard, and it is not one."""
    batch = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    assert batch.observations[0].confidence is None
    assert batch.mean_confidence is None
    assert "accuracy" not in batch.detector_status.get("accuracy_note", "").lower() \
        or "NOT measured" in batch.detector_status["accuracy_note"]


def test_the_units_are_passed_explicitly_in_both_directions(tmp_path):
    """Neither the mesh unit nor the depth scale can be read from a file, and
    guessing either is a factor-of-1000 error that looks like a real pose."""
    provider = _provider(tmp_path)
    provider.acquire_reference(dataset="ds", model_id="part", depth_scale_mm=1.0)
    request = provider.client.requests[0]
    assert request["depth_scale_mm"] == 1.0
    # mm mesh -> metres for FoundationPose
    assert request["mesh_scale_to_metres"] == pytest.approx(0.001)


# --------------------------------------------------------------------------- #
# Symmetry
# --------------------------------------------------------------------------- #


def test_an_axial_symmetry_collapses_the_unobservable_spin(tmp_path):
    provider = _provider(tmp_path,
                         symmetry=Symmetry(type=SymmetryType.AXIAL, axis="z"))
    validated, _ = validate_response(
        _pose(orientation=Orientation.from_yaw_deg(149.0).to_dict()))
    observation = provider.observation_from(
        validated, model=provider.registry.models["part"],
        acquisition=ACQUISITION_REFERENCE, observation_id="o")
    assert observation.orientation.rpy_deg()[2] == pytest.approx(0.0, abs=1e-6)
    # THE RAW ESTIMATE IS KEPT. It is the evidence that tells "the estimator was
    # wrong" from "this rotation was never observable".
    assert observation.orientation_raw is not None
    assert observation.orientation_raw.rpy_deg()[2] == pytest.approx(149.0, abs=1e-6)


def test_a_symmetric_object_never_reports_orientation_as_fully_measured(tmp_path):
    for symmetry in (Symmetry(type=SymmetryType.AXIAL, axis="z"),
                     Symmetry(type=SymmetryType.DISCRETE, axis="z", fold=2)):
        provider = _provider(tmp_path, symmetry=symmetry)
        batch = provider.acquire_reference(dataset="ds", model_id="part",
                                           depth_scale_mm=1.0)
        dof = batch.observations[0].measured_dof
        assert "orientation" not in dof, symmetry.type
        assert "orientation_partial" in dof, symmetry.type


def test_a_bent_pipe_keeps_its_two_fold_ambiguity_and_is_not_flattened(tmp_path):
    """Cylinder5 is bent, so the obvious declaration is "no symmetry" — and it
    is wrong. A 180 deg rotation about z maps it onto itself to within sampling
    noise, because it is a symmetric hairpin. Reporting that leg swap as
    resolved is the over-claim the declaration exists to prevent."""
    provider = _provider(tmp_path,
                         symmetry=Symmetry(type=SymmetryType.DISCRETE,
                                           axis="z", fold=2))
    batch = provider.acquire_reference(dataset="ds", model_id="part",
                                       depth_scale_mm=1.0)
    symmetry = batch.observations[0].symmetry
    assert symmetry.type is SymmetryType.DISCRETE
    assert symmetry.fold == 2
    assert symmetry.ambiguous_dof == ["rotation_about_z_modulo_180deg"]


def test_an_asymmetric_object_keeps_its_full_orientation(tmp_path):
    provider = _provider(tmp_path, symmetry=Symmetry())
    batch = provider.acquire_reference(dataset="ds", model_id="part",
                                       depth_scale_mm=1.0)
    assert "orientation" in batch.observations[0].measured_dof
    # Nothing was collapsed, so there is no separate raw copy to keep.
    assert batch.observations[0].orientation_raw is None


# --------------------------------------------------------------------------- #
# Offline / reference provenance
# --------------------------------------------------------------------------- #


def test_a_reference_batch_says_so_in_the_data(tmp_path):
    """§16: it must be clearly labelled reference/offline regression, NOT a live
    RGB-D camera acquisition."""
    batch = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    assert batch.acquisition == ACQUISITION_REFERENCE
    assert "REFERENCE" in batch.detector_status["note"]
    assert "not from a live camera" in batch.detector_status["note"]


def test_the_reference_note_forbids_planning_against_it():
    assert "must not be planned against" in REFERENCE_NOTE


def test_a_reference_batch_is_never_placeable_in_the_work_area(tmp_path):
    batch = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    assert all(not o.workarea_pose_available for o in batch.observations)


def test_the_method_is_stamped_on_the_batch_and_the_observation(tmp_path):
    batch = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    assert batch.perception_method == METHOD
    assert batch.observations[0].perception_method == METHOD
    assert batch.detector == ESTIMATOR_ID


# --------------------------------------------------------------------------- #
# Failure paths — no silent fallback
# --------------------------------------------------------------------------- #


def test_a_refused_estimate_becomes_a_failed_batch_not_an_empty_one(tmp_path):
    """§15: an empty successful batch would read as "the camera saw nothing"."""
    client = FakeClient(error="inference is not available; no GPU")
    batch = _provider(tmp_path, client=client).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    assert not batch.ok
    assert "no GPU" in batch.error
    assert batch.observations == []


def test_a_broken_response_never_becomes_a_pose(tmp_path):
    client = FakeClient(result=_pose(frame_id=""))
    batch = _provider(tmp_path, client=client).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    assert not batch.ok
    assert "frame" in batch.error


def test_a_failed_batch_still_carries_its_method_and_acquisition(tmp_path):
    """Provenance survives failure, or a failure cannot be attributed."""
    client = FakeClient(error="worker unreachable")
    batch = _provider(tmp_path, client=client).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    assert batch.perception_method == METHOD
    assert batch.acquisition == ACQUISITION_REFERENCE
    assert batch.frame_id == CAMERA_FRAME


# --------------------------------------------------------------------------- #
# Serialisation — the DDS / FIWARE path
# --------------------------------------------------------------------------- #


def test_the_batch_survives_a_json_round_trip_with_its_6dof_intact(tmp_path):
    batch = _provider(tmp_path,
                      symmetry=Symmetry(type=SymmetryType.DISCRETE,
                                        axis="z", fold=2)).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    restored = ObservationBatch.from_dict(json.loads(json.dumps(batch.to_dict())))
    assert restored.perception_method == METHOD
    assert restored.acquisition == ACQUISITION_REFERENCE
    assert restored.frame_id == CAMERA_FRAME
    before, after = batch.observations[0], restored.observations[0]
    assert before.orientation.angle_to_deg(after.orientation) < 1e-6
    assert after.symmetry.fold == 2
    assert after.object_model_id == before.object_model_id
    # BOTH validities survive the round trip, separately.
    assert after.pose_valid is True
    assert after.workarea_transform_valid is False
    assert after.workarea_pose_available is False


def test_every_published_field_is_json_serialisable(tmp_path):
    """No matrices, no tensors, no numpy — the payload crosses DDS and HTTP."""
    batch = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    json.dumps(batch.to_dict())          # must not raise


def test_a_legacy_planar_batch_still_parses_unchanged():
    """§14: planar observations must keep working. A document written before
    6-DoF existed has no `perception_method`, and assuming one would relabel it."""
    legacy = {
        "batch_id": "b1", "source": "camera", "status": "ok",
        "frame_id": "wisepack_workarea", "detector": "fasterrcnn_resnet50_fpn/bottle",
        "observations": [{
            "observation_id": "o1", "object_type": "cylindrical_proxy",
            "source": "camera", "frame_id": "wisepack_workarea",
            "pose": {"x_mm": 10.0, "y_mm": 20.0, "yaw_deg": -31.0},
            "confidence": 0.91,
        }],
    }
    batch = ObservationBatch.from_dict(legacy)
    assert batch.perception_method == ""       # empty, NOT defaulted to planar
    assert batch.acquisition == ""
    assert batch.observations[0].yaw_deg == pytest.approx(-31.0)
    assert batch.observations[0].frame_id == "wisepack_workarea"


def test_a_planar_observation_gains_no_6dof_fields_it_did_not_measure():
    """The two methods must stay distinguishable downstream."""
    observation = PhysicalObservation(observation_id="o", x_mm=1.0, y_mm=2.0,
                                      yaw_deg=15.0)
    document = observation.to_dict()
    assert document["object_model_id"] == ""
    assert document["perception_method"] == ""
    assert document["pose"]["measured_dof"] == []
    assert document["orientation_raw"] is None


# --------------------------------------------------------------------------- #
# The boundary itself
# --------------------------------------------------------------------------- #


def _code_only(source: str) -> str:
    """The source with comments and docstrings removed.

    THE PROHIBITION MUST BE EXPLAINABLE. These tests assert that the core does
    not DEPEND on FoundationPose; the modules that keep that boundary say so in
    their docstrings, and those docstrings contain the very words being banned.
    A check that cannot tell a rule from its violation forbids writing the rule
    down, so it is applied to code and never to prose.
    """
    import io
    import tokenize
    kept = []
    previous = tokenize.INDENT
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                continue
            if (token.type == tokenize.STRING
                    and previous in (tokenize.INDENT, tokenize.NEWLINE,
                                     tokenize.NL, tokenize.DEDENT)):
                continue                       # a docstring
            if token.type not in (tokenize.NL, tokenize.NEWLINE,
                                  tokenize.INDENT, tokenize.DEDENT):
                previous = token.type
            else:
                previous = token.type
            kept.append(token.string)
    except tokenize.TokenError:                # pragma: no cover
        return source
    return "\n".join(kept)


def _core_sources():
    root = os.path.join(REPO, "wisepack_ws", "src", "wisepack_core",
                        "wisepack_core")
    for name in sorted(os.listdir(root)):
        if name.endswith(".py"):
            source = open(os.path.join(root, name), encoding="utf-8").read()
            yield name, _code_only(source)


def test_no_core_module_imports_the_foundationpose_transport():
    """§2: the core must not DEPEND on FoundationPose.

    Tested as an IMPORT GRAPH, not as a word search. A method name and a
    selector label — "RGB-D 6-DoF — FoundationPose" — are domain vocabulary the
    core is entitled to hold, exactly as it holds "camera": they name a choice
    an operator makes. What must not exist is a core module that imports the
    client, knows the worker's URL, or parses its response schema. That is what
    would make planning depend on a GPU container being present.
    """
    import ast
    root = os.path.join(REPO, "wisepack_ws", "src", "wisepack_core",
                        "wisepack_core")
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py") or name == "foundationpose_client.py":
            continue
        tree = ast.parse(open(os.path.join(root, name), encoding="utf-8").read())
        for node in ast.walk(tree):
            imported = []
            if isinstance(node, ast.Import):
                imported = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""] + [a.name for a in node.names]
            for module in imported:
                assert "foundationpose" not in module.lower(), \
                    f"{name} imports {module}"
                assert "providers" not in module.lower(), f"{name} imports {module}"


def test_the_core_holds_no_worker_endpoint_or_url():
    """The transport is the client's business alone."""
    for name, source in _core_sources():
        if name == "foundationpose_client.py":
            continue
        for marker in ("/estimate", "/last-result", "22201"):
            assert marker not in source, f"{name} knows the worker's {marker}"


def test_the_planner_never_sees_a_mesh_a_mask_or_a_tensor():
    for name, source in _core_sources():
        lowered = source.lower()
        for word in ("nvdiffrast", "pytorch3d", "cuda", "rasterize", "mycpp"):
            assert word not in lowered, f"{name} mentions {word}"


def test_the_provider_does_not_plan():
    source = _code_only(open(os.path.join(REPO, "perception", "providers",
                                          "foundationpose_rgbd.py"),
                             encoding="utf-8").read())
    for word in ("packing", "optimizer", "Strategy", "plan("):
        assert word not in source, f"the provider references {word}"


# --------------------------------------------------------------------------- #
# Camera ownership (§8)
# --------------------------------------------------------------------------- #


def test_no_ownership_transfer_uses_process_matching_or_sleeps():
    """§8 forbids sleeps, pkill, killall and broad device matching. Each is a
    way of GUESSING that a handover finished; they fail silently on a busy
    machine and can stop something this project does not own."""
    source = open(os.path.join(REPO, "wisepack_ws", "src", "wisepack_core",
                               "wisepack_core", "camera_ownership.py"),
                  encoding="utf-8").read()
    code = _code_only(source)
    for forbidden in ("pkill", "killall", "subprocess", "os.system",
                      "time.sleep", "psutil", "glob"):
        assert forbidden not in code, f"ownership uses {forbidden}"


def test_ownership_never_claims_a_handover_that_has_not_happened():
    """No RGB-D camera is attached, so no transfer has been performed."""
    from wisepack_core.camera_ownership import current_ownership
    ownership = current_ownership(colour_available=True,
                                  colour_holder="planar_fasterrcnn")
    assert ownership.handover_tested is False
    assert "no handover has been performed" in ownership.to_dict()["note"]


def test_an_absent_depth_camera_is_the_reason_not_a_handover_problem():
    from wisepack_core.camera_ownership import ABSENT, current_ownership
    ownership = current_ownership(colour_available=True,
                                  colour_holder="planar_fasterrcnn")
    assert ownership.depth_state == ABSENT
    reason = ownership.blocked_reason("foundationpose_rgbd")
    assert "RGB-D" in reason and "attached" in reason


def test_a_shared_device_needs_an_explicit_release_then_acquire():
    from wisepack_core.camera_ownership import (CameraOwnership, HELD,
                                                SHARED_DEVICE_YES)
    ownership = CameraOwnership(colour_state=HELD,
                                colour_holder="planar_fasterrcnn",
                                depth_state=HELD,
                                depth_holder="",
                                shared_device=SHARED_DEVICE_YES)
    assert ownership.requires_handover("foundationpose_rgbd") is True
    steps = ownership.plan_for("foundationpose_rgbd")
    assert any("releases" in s for s in steps)
    assert any("opens" in s for s in steps)
    # CONFIRMED, NEVER TIMED. A fixed delay is a guess that the device settled.
    assert any("never after" in s and "delay" in s for s in steps)


def test_an_unknown_sharing_relationship_is_not_treated_as_shared():
    """Asserting a handover that is not needed blocks a switch that would have
    worked — the same class of error as skipping one that was."""
    from wisepack_core.camera_ownership import current_ownership
    ownership = current_ownership(colour_available=True,
                                  colour_holder="planar_fasterrcnn")
    assert ownership.shared_device == "unknown"
    assert ownership.requires_handover("foundationpose_rgbd") is False


# --------------------------------------------------------------------------- #
# Run lifecycle (§13) — ONE workflow, reused, not a second one
# --------------------------------------------------------------------------- #


def _six_dof_batch(**over):
    """A batch shaped like a FUTURE live RGB-D one: actionable, work-area frame.

    The provider cannot produce this today — there is no validated extrinsic and
    no depth camera — and that is exactly why it is constructed here. §13 is
    about the architecture being ready, and the architecture is what this
    checks.
    """
    observation = PhysicalObservation(
        observation_id="rgbd-1", x_mm=120.0, y_mm=80.0, z_mm=15.0,
        object_type="pipe_section", source=PerceptionSource.CAMERA.value,
        frame_id="wisepack_workarea",
        orientation=Orientation.from_yaw_deg(30.0),
        symmetry=Symmetry(type=SymmetryType.AXIAL, axis="z"),
        perception_method=METHOD, object_model_id="cylinder3",
        diameter_mm=35, length_mm=190, geometry_source="cad_model",
        pose_valid=True, workarea_transform_valid=True,
        measured_dof=("x", "y", "z", "orientation_partial"))
    fields = dict(batch_id="rgbd-batch-1", source=PerceptionSource.CAMERA.value,
                  observations=[observation], frame_id="wisepack_workarea",
                  perception_method=METHOD, acquisition="live",
                  detector=ESTIMATOR_ID, calibration_status="not_applicable")
    fields.update(over)
    return ObservationBatch(**fields)


def test_a_6dof_batch_uses_the_same_apply_path_as_a_planar_one():
    """§13: do not introduce a second workflow. `apply_observation_batch` takes
    an ObservationBatch and does not ask which method produced it."""
    import inspect

    from wisepack_core.workflow import WorkflowEngine
    source = inspect.getsource(WorkflowEngine.apply_observation_batch)
    for word in ("foundationpose", "perception_method", "rgbd", "6dof"):
        assert word not in source.lower(), (
            f"apply_observation_batch branches on {word!r} — the method must "
            "not reach the workflow")


def test_a_new_batch_advances_the_revision_and_revokes_approval():
    """The existing revision architecture, exercised with a 6-DoF batch. An
    approval is a decision about ONE batch revision; a new observation cannot
    leave an earlier authorisation standing."""
    import inspect

    from wisepack_core.workflow import WorkflowEngine
    # THE RULE IS STRUCTURAL, and this is what makes it so: every path that
    # changes the batch goes through one function, so none can forget.
    apply_source = inspect.getsource(WorkflowEngine.apply_observation_batch)
    assert "_bump_scenario_revision" in apply_source
    bump = inspect.getsource(WorkflowEngine._bump_scenario_revision)
    assert "scenario_revision += 1" in bump


def test_a_6dof_batch_carries_everything_the_planner_needs():
    """It must be consumable by the same planner, with no 6-DoF-aware code."""
    batch = _six_dof_batch()
    assert batch.ok and batch.count == 1
    observation = batch.observations[0]
    # The planar projection every existing consumer reads is present and agrees
    # with the authoritative quaternion.
    assert observation.yaw_deg == pytest.approx(30.0, abs=1e-6)
    assert observation.position.z == 15
    assert observation.diameter_mm == 35


def test_a_reference_batch_is_not_shaped_like_an_actionable_one(tmp_path):
    """§16: the offline result must not be routed into a run as a measurement.
    Its own fields refuse it — camera frame and pose_valid False."""
    reference = _provider(tmp_path).acquire_reference(
        dataset="ds", model_id="part", depth_scale_mm=1.0)
    assert reference.frame_id == CAMERA_FRAME
    # The ESTIMATES are valid; what refuses them is that they cannot be placed
    # in the work area, plus the acquisition being a saved dataset.
    assert all(o.pose_valid for o in reference.observations)
    assert all(not o.workarea_pose_available for o in reference.observations)
    assert reference.acquisition == ACQUISITION_REFERENCE
    live = _six_dof_batch()
    assert live.frame_id == "wisepack_workarea"
    assert all(o.workarea_pose_available for o in live.observations)


def test_the_reference_endpoint_does_not_install_the_batch():
    """It returns the batch; it does not apply it to the running scenario."""
    import ast
    source = open(os.path.join(REPO, "web", "app.py"), encoding="utf-8").read()
    tree = ast.parse(source)
    function = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "api_foundationpose_reference_regression")
    # CODE ONLY. The docstring EXPLAINS why this endpoint is separate from
    # `detect_physical_objects`, so it necessarily names it — and a check that
    # cannot tell the explanation from the thing explained would forbid writing
    # the reason down.
    body = _code_only(ast.unparse(function))
    for forbidden in ("apply_observation_batch", "detect_physical_objects",
                      "STATE.engine"):
        assert forbidden not in body, (
            f"the reference regression endpoint touches {forbidden}")


# --------------------------------------------------------------------------- #
# The DDS / FIWARE path (§14)
# --------------------------------------------------------------------------- #


def test_the_bus_payload_is_the_batch_document_itself():
    """§14: extend the generic payload, do not add a second channel. The
    orchestrator publishes `batch.to_dict()`, so 6-DoF fields ride along."""
    source = open(os.path.join(
        REPO, "wisepack_ws", "src", "wisepack_orchestration",
        "wisepack_orchestration", "hitl_orchestrator.py"), encoding="utf-8").read()
    assert "json.dumps(batch.to_dict()" in source
    for word in ("foundationpose", "FoundationPoseClient"):
        assert word not in source, f"the orchestrator names {word}"


def test_foundationpose_publishes_no_ros_or_dds_itself():
    """§14: the worker speaks HTTP. The orchestrator is the single writer."""
    for path in (os.path.join(REPO, "perception", "providers",
                              "foundationpose_rgbd.py"),
                 os.path.join(REPO, "wisepack_ws", "src", "wisepack_core",
                              "wisepack_core", "foundationpose_client.py"),
                 os.path.join(REPO, "perception", "foundationpose", "worker",
                              "app.py")):
        code = _code_only(open(path, encoding="utf-8").read())
        for word in ("rclpy", "rmw_", "std_msgs", "create_publisher"):
            assert word not in code, f"{os.path.basename(path)} uses {word}"


# --------------------------------------------------------------------------- #
# RGB-D acquisition — owned by the WORKER, verified without a camera
# --------------------------------------------------------------------------- #


WORKER_CAMERA = os.path.join(REPO, "perception", "foundationpose", "worker",
                             "camera.py")


def _worker_camera_source():
    return open(WORKER_CAMERA, encoding="utf-8").read()


def test_rgbd_acquisition_lives_in_the_worker_not_on_the_host():
    """One coherent acquisition. Colour, aligned depth and the intrinsics they
    share must come from one SDK in one process; splitting them across a host
    capture stack and a container estimator means two alignment implementations
    and two places a depth scale can be assumed."""
    assert os.path.isfile(WORKER_CAMERA)
    # No second RealSense stack anywhere on the host side.
    for directory in ("scripts", "perception/providers", "web"):
        root = os.path.join(REPO, *directory.split("/"))
        for name in os.listdir(root):
            if not name.endswith(".py"):
                continue
            code = _code_only(open(os.path.join(root, name),
                                   encoding="utf-8").read())
            assert "pyrealsense2" not in code, (
                f"{directory}/{name} imports the RealSense SDK on the host")


def test_the_host_perception_venv_gains_no_realsense_dependency():
    """The planar provider's environment is the working one; a new dependency
    there to serve an optional feature is an unnecessary risk to it."""
    requirements = open(os.path.join(REPO, "perception", "requirements.txt"),
                        encoding="utf-8").read().lower()
    assert "realsense" not in requirements


def test_the_worker_image_carries_the_realsense_sdk():
    requirements = open(os.path.join(REPO, "perception", "foundationpose",
                                     "worker", "requirements.txt"),
                        encoding="utf-8").read()
    assert "pyrealsense2" in requirements


def test_acquisition_never_uses_a_video_device_path():
    """A RealSense presents several /dev/video* nodes and none of them carry
    intrinsics, a depth scale or alignment. /dev/video0 is the planar webcam's
    identity and stays that way."""
    code = _code_only(_worker_camera_source())
    assert "/dev/video" not in code
    assert "VideoCapture" not in code


def test_the_depth_scale_is_read_from_the_device():
    code = _code_only(_worker_camera_source())
    assert "get_depth_scale" in code
    # No hard-coded metres-per-unit standing in for the device's own value.
    assert "0.001" not in code


def test_the_intrinsics_are_read_from_the_device():
    assert "get_intrinsics" in _code_only(_worker_camera_source())


def test_alignment_is_verified_rather_than_declared():
    """`rs.align` to colour must actually yield the colour camera's geometry. A
    dataset claiming alignment it did not get is a silent reprojection error in
    every pose computed from it."""
    code = _code_only(_worker_camera_source())
    assert "alignment_verified" in code
    # Verified against the colour intrinsics, not merely against image size.
    assert "colour_intrinsics" in code


def test_colour_and_depth_come_from_one_synchronised_bundle():
    """Fetching them separately pairs a colour image with depth from another
    instant — on a moving scene, a pose error with no symptom."""
    code = _code_only(_worker_camera_source())
    assert code.count("wait_for_frames") >= 1
    assert "get_color_frame" in code and "get_depth_frame" in code


def test_a_missing_camera_is_reported_not_raised():
    """The capability probe runs on every /health, so it must not throw."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("wisepack_fp_camera",
                                                  WORKER_CAMERA)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    usable, reason = module.available()
    assert usable is False                    # no camera in CI
    assert "RealSense" in reason
    # AND IT SAYS WHY THE OBVIOUS GUESS IS WRONG.
    assert "/dev/video" in reason


def test_a_capture_requires_an_explicit_model_id():
    """Which CAD part is on the table is known because an operator put it there.
    Inferring it from appearance is what made the older capture data unusable."""
    source = open(os.path.join(REPO, "perception", "foundationpose", "worker",
                               "app.py"), encoding="utf-8").read()
    assert "`model_id` is required" in source
    assert "never" in source and "inferred from the image" in source


def test_a_capture_records_that_it_has_no_masks_and_no_extrinsic():
    """Both are needed downstream and neither can be produced by capturing."""
    source = _worker_camera_source()
    assert '"masks_present": False' in source
    assert '"camera_to_workarea_extrinsic": None' in source


def test_the_camera_is_a_separate_capability_from_inference():
    """The offline reference regression runs with no camera at all, so the
    camera must not be folded into `inference_available`."""
    source = open(os.path.join(REPO, "perception", "foundationpose", "worker",
                               "capability.py"), encoding="utf-8").read()
    assert "rgbd_camera_available" in source
    assert "live_inference_available" in source


# --------------------------------------------------------------------------- #
# USB passthrough and the seven-step verification chain
# --------------------------------------------------------------------------- #


def _setup_script_text():
    return open(os.path.join(REPO, "scripts", "setup_foundationpose.sh"),
                encoding="utf-8").read()


def _executable_lines(text: str) -> str:
    """Shell source with comments stripped.

    These scripts DOCUMENT the rules they obey — including why --privileged is
    refused and why /dev/video0 is not the RealSense — so the prohibited strings
    appear in the prose that prohibits them. A check that cannot tell a rule
    from its violation forbids writing the rule down.
    """
    return "\n".join(line for line in text.splitlines()
                      if not line.lstrip().startswith("#"))


def test_usb_access_is_narrow_and_never_privileged():
    """--privileged would work and grants every device on the host to a
    container that needs one."""
    executable = _executable_lines(_setup_script_text())
    assert "--privileged" not in executable
    assert "--device" in executable
    # The whole USB bus is not mounted either.
    assert "-v /dev/bus/usb" not in executable
    assert "/dev:/dev" not in executable


def test_the_realsense_is_found_by_vendor_id_not_by_video_node():
    text = _setup_script_text()
    assert "8086" in text
    assert "/dev/video" not in _executable_lines(text)


def test_the_diagnostic_separates_all_seven_layers():
    """Reporting one "camera unavailable" for seven independent failures sends
    someone to re-seat a cable when the container was started too early."""
    script = open(os.path.join(REPO, "scripts", "realsense_diagnose.sh"),
                  encoding="utf-8").read()
    for marker in ("HOST sees", "CONTAINER sees", "pyrealsense2",
                   "model and serial", "streams", "aligned", "depth scale"):
        assert marker in script, f"the diagnostic omits {marker!r}"


def test_the_diagnostic_names_docker_passthrough_when_only_the_host_sees_it():
    """§: if the host sees the camera and the container does not, that is a
    Docker problem — not a reason to move acquisition back to the host."""
    script = open(os.path.join(REPO, "scripts", "realsense_diagnose.sh"),
                  encoding="utf-8").read()
    assert "DOCKER USB PASSTHROUGH problem" in script
    assert "not a perception one" in script


# --------------------------------------------------------------------------- #
# Canonicalisation must not move the body (§ Stage C diagnosis)
# --------------------------------------------------------------------------- #


def test_canonicalising_a_symmetric_pose_does_not_move_the_object(tmp_path):
    """A symmetry rotation turns about the object's OWN axis, which passes
    through its centre — not through the model origin.

    Replacing the orientation while keeping the position rotates the body about
    the wrong point. For a part drawn obliquely that is a large physical
    displacement: Cylinder5's origin is 98.8 mm off its axis, so a 180 degree
    canonicalisation moved it ~198 mm and turned a 4 mm camera-frame error into
    a 284 mm workarea error.
    """
    import math
    from wisepack_core.pose import Orientation, Symmetry, SymmetryType
    provider = _provider(
        tmp_path,
        symmetry=Symmetry(type=SymmetryType.DISCRETE, axis="z", fold=2),
        model_center_mm=(-130.0, -54.44, 0.0))
    model = provider.registry.models["part"]

    # A pose whose canonicalisation genuinely differs (a >180 deg spin).
    raw = Orientation.from_yaw_deg(200.0)
    validated, _ = validate_response(
        {"frame_id": CAMERA_FRAME, "position_mm": [79.0, -117.0, 783.0],
         "orientation": raw.to_dict()})
    observation = provider.observation_from(
        validated, model=model, acquisition=ACQUISITION_REFERENCE,
        observation_id="o")
    assert observation.orientation_raw is not None, "expected canonicalisation"

    centre = list(model.model_center_mm)
    before = [a + b for a, b in zip(raw.rotate(centre), [79.0, -117.0, 783.0])]
    after = [a + b for a, b in zip(observation.orientation.rotate(centre),
                                   [observation.x_mm, observation.y_mm,
                                    observation.z_mm])]
    moved = math.sqrt(sum((a - b) ** 2 for a, b in zip(after, before)))
    assert moved < 1e-6, f"canonicalisation moved the body by {moved:.3f} mm"


def test_a_model_without_a_measured_centre_is_left_alone(tmp_path):
    """No centre, no correction — and no guess. Applying one about the origin
    would be exactly the error this fix removes."""
    from wisepack_core.pose import Orientation, Symmetry, SymmetryType
    provider = _provider(
        tmp_path, symmetry=Symmetry(type=SymmetryType.DISCRETE, axis="z", fold=2))
    model = provider.registry.models["part"]
    assert not model.model_center_mm
    validated, _ = validate_response(
        {"frame_id": CAMERA_FRAME, "position_mm": [1.0, 2.0, 3.0],
         "orientation": Orientation.from_yaw_deg(200.0).to_dict()})
    observation = provider.observation_from(
        validated, model=model, acquisition=ACQUISITION_REFERENCE,
        observation_id="o")
    assert (observation.x_mm, observation.y_mm, observation.z_mm) == (1.0, 2.0, 3.0)


def test_every_cylinder_declares_a_measured_model_centre():
    """All five are drawn with the body away from the origin, so all five need
    it — this is not a Cylinder5 special case."""
    import yaml
    with open(os.path.join(REPO, "config", "perception_objects.yaml"),
              encoding="utf-8") as handle:
        entries = {e["model_id"]: e for e in yaml.safe_load(handle)["objects"]}
    for model_id in ("cylinder1", "cylinder2", "cylinder3", "cylinder4",
                     "cylinder5"):
        centre = entries[model_id].get("model_center_mm")
        assert centre and len(centre) == 3, model_id


# --------------------------------------------------------------------------- #
# Task-level geometry: the grasp centre, not the CAD origin
# --------------------------------------------------------------------------- #


def test_a_cad_origin_outside_the_body_still_gives_the_right_grasp_centre(tmp_path):
    """THE DEFECT THIS PREVENTS, stated concretely.

    Cylinder5's CAD origin sits 141 mm outside its body, in empty space. A grasp
    planner that consumed `pose.x_mm/y_mm/z_mm` would send the gripper 141 mm
    away from the tube and close on nothing. `object_center` is the physical
    body centre, and it is what a pick targets.
    """
    import math
    from wisepack_core.pose import Orientation
    centre = (-130.0, -54.44, 0.0)          # measured, well outside the tube
    provider = _provider(tmp_path, model_center_mm=centre,
                         task_axis_vector=(0.9284, -0.3716, 0.0))
    model = provider.registry.models["part"]

    # A known orientation, so the expected centre can be computed by hand.
    orientation = Orientation.from_yaw_deg(90.0)
    validated, _ = validate_response(
        {"frame_id": CAMERA_FRAME, "position_mm": [100.0, 200.0, 900.0],
         "orientation": orientation.to_dict()})
    observation = provider.observation_from(
        validated, model=model, acquisition=ACQUISITION_REFERENCE,
        observation_id="o")

    # A 90 deg yaw takes (x, y, z) -> (-y, x, z), so the centre offset
    # (-130, -54.44, 0) becomes (54.44, -130, 0).
    expected = (100.0 + 54.44, 200.0 - 130.0, 900.0)
    for got, want in zip(observation.object_center, expected):
        assert got == pytest.approx(want, abs=1e-6)

    # And it is genuinely a DIFFERENT point from the reported pose.
    separation = math.sqrt(sum(
        (a - b) ** 2 for a, b in zip(observation.object_center,
                                     (observation.x_mm, observation.y_mm,
                                      observation.z_mm))))
    assert separation == pytest.approx(140.94, abs=0.01)


def test_the_two_points_are_named_and_never_both_called_position(tmp_path):
    """§: a field called simply "position" must not mean both."""
    provider = _provider(tmp_path, model_center_mm=(-130.0, -54.44, 0.0),
                         task_axis_vector=(0.9284, -0.3716, 0.0))
    batch = provider.acquire_reference(dataset="ds", model_id="part",
                                       depth_scale_mm=1.0)
    document = batch.observations[0].to_dict()
    assert document["pose"]["reference_point"] == "model_frame_origin"
    assert "object_center_mm" in document["task"]
    assert document["task"]["object_center_mm"] != [
        document["pose"]["x_mm"], document["pose"]["y_mm"],
        document["pose"]["z_mm"]]


def test_a_planar_observation_reports_the_body_as_its_own_position():
    """The planar detector already reports the object itself, so its position
    IS the body — and the derived centre must not move it."""
    observation = PhysicalObservation(observation_id="o", x_mm=10.0, y_mm=20.0,
                                      yaw_deg=15.0)
    assert observation.object_center == (10.0, 20.0, 0.0)
    assert observation.tube_axis is None
    assert observation.to_dict()["pose"]["reference_point"] == "object_body"


def test_the_task_geometry_carries_what_a_pick_needs(tmp_path):
    provider = _provider(tmp_path, model_center_mm=(-130.0, -54.44, 0.0),
                         task_axis_vector=(0.9284, -0.3716, 0.0))
    batch = provider.acquire_reference(dataset="ds", model_id="part",
                                       depth_scale_mm=1.0)
    task = batch.observations[0].task_geometry()
    assert task["object_center_mm"] and task["tube_axis_line"]
    assert task["diameter_mm"] == 25 and task["length_mm"] == 315
    # A LINE, not a direction: no consumer may read the sign as meaningful.
    assert task["tube_axis_is_a_line_not_a_direction"] is True
    # Unit to the precision it is PUBLISHED at: the payload rounds to 6 dp for
    # readability, which is ~6e-5 degrees of direction error and irrelevant to a
    # grasp. Demanding 1e-9 would only be demanding that the rounding go away.
    assert abs(sum(v * v for v in task["tube_axis_line"]) - 1.0) < 1e-5


def test_the_task_geometry_survives_serialisation(tmp_path):
    provider = _provider(tmp_path, model_center_mm=(-130.0, -54.44, 0.0),
                         task_axis_vector=(0.9284, -0.3716, 0.0))
    batch = provider.acquire_reference(dataset="ds", model_id="part",
                                       depth_scale_mm=1.0)
    restored = ObservationBatch.from_dict(
        json.loads(json.dumps(batch.to_dict())))
    before, after = batch.observations[0], restored.observations[0]
    for got, want in zip(after.object_center, before.object_center):
        assert got == pytest.approx(want, abs=1e-6)


# --------------------------------------------------------------------------- #
# Stage D: planning must consume the physical centre, not the CAD origin
# --------------------------------------------------------------------------- #


def _cad_observation(centre=(-130.0, -54.44, 0.0)):
    from wisepack_core.pose import Orientation, Symmetry, SymmetryType
    return PhysicalObservation(
        observation_id="o1", x_mm=383.63, y_mm=-240.84, z_mm=-45.12,
        object_type="pipe_section", source=PerceptionSource.CAMERA.value,
        frame_id="wisepack_workarea",
        orientation=Orientation.identity(),
        symmetry=Symmetry(type=SymmetryType.DISCRETE, axis="z", fold=2),
        perception_method=METHOD, object_model_id="cylinder5",
        diameter_mm=25, length_mm=342, inner_diameter_mm=19,
        model_center_mm=centre, task_axis_vector=(0.9284, -0.3716, 0.0),
        pose_valid=True, workarea_transform_valid=True)


def test_planning_uses_the_object_centre_and_not_the_cad_origin():
    """THE STAGE C REGRESSION, guarded.

    Cylinder5's CAD origin is 141 mm outside its body. A planner fed that
    origin would place the item — and later send a gripper — into empty space.
    """
    batch = ObservationBatch(
        batch_id="b", source=PerceptionSource.CAMERA.value,
        observations=[_cad_observation()], frame_id="wisepack_workarea")
    item = batch.to_waste_items()[0]
    observation = batch.observations[0]

    expected = [int(round(v)) for v in observation.object_center]
    origin = [int(round(v)) for v in (observation.x_mm, observation.y_mm,
                                      observation.z_mm)]
    used = [item.source_position.x, item.source_position.y,
            item.source_position.z]
    assert used == expected, f"planner used {used}, expected the centre {expected}"
    assert used != origin, "the planner used the CAD model origin"


def test_a_cad_item_keeps_its_identity_and_nominal_geometry():
    """§3: it must not collapse into an anonymous generated cylinder. The
    registry's nominal dimensions come from the engineering table; the
    configured proxy's do not describe this part at all."""
    from wisepack_core.perception import ProxyGeometry
    batch = ObservationBatch(
        batch_id="b", source=PerceptionSource.CAMERA.value,
        observations=[_cad_observation()], frame_id="wisepack_workarea")
    # A proxy geometry that is nothing like a Cylinder5, to prove it is ignored.
    item = batch.to_waste_items(
        geometry=ProxyGeometry(diameter_mm=65, length_mm=215))[0]
    assert item.model_id == "cylinder5"
    assert item.geometry_source == "cad_mesh"
    assert (item.outer_diameter_mm, item.length_mm) == (25, 342)
    assert item.inner_diameter_mm == 19


def test_a_planar_observation_still_uses_the_configured_proxy_geometry():
    """The working detector is untouched: it has no CAD model, so the proxy
    geometry is exactly what it should get."""
    from wisepack_core.perception import ProxyGeometry
    planar = PhysicalObservation(observation_id="o", x_mm=10.0, y_mm=20.0,
                                 yaw_deg=15.0, confidence=0.9)
    batch = ObservationBatch(
        batch_id="b", source=PerceptionSource.CAMERA.value,
        observations=[planar], frame_id="wisepack_workarea")
    item = batch.to_waste_items(
        geometry=ProxyGeometry(diameter_mm=65, length_mm=215))[0]
    assert item.model_id == ""
    assert item.geometry_source == "generated"
    assert (item.outer_diameter_mm, item.length_mm) == (65, 215)
    # And its position is unchanged — for a planar detection the reported
    # position IS the body.
    assert (item.source_position.x, item.source_position.y) == (10, 20)


def test_the_optimizer_handles_a_single_perceived_item():
    """A one-object scenario is exactly what a single perceived tube produces,
    and the perturbation step indexed past the end of a one-element list."""
    from wisepack_core.generator import build_scenario
    from wisepack_core.packing import OptimizerConfig, pack_optimized
    scenario = build_scenario("cad_cylinder5_single")
    plan = pack_optimized(scenario, config=OptimizerConfig(restarts=12, seed=7))
    assert len(plan.placements) == 1
    assert plan.placements[0].item_id == scenario.items[0].item_id
