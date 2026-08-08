"""Perception source: configuration, adaptation, and the invariants that matter.

Every test here runs with NO camera, NO GPU, NO model weights and NO network.
That is the point: the physical perception path has to be verifiable in ordinary
CI, so the detector's output is represented by the JSON it actually produces and
the transport by a stub. What the tests check is the part WISEPACK owns —

  * the default is still simulated perception, and `sim` behaves exactly as it
    did before this code existed;
  * a HARMONY result becomes DOMAIN-NEUTRAL objects with x/y/yaw/confidence
    preserved and the CONFIGURED proxy geometry attached;
  * every failure §15 lists is a visible failed batch, never an empty
    successful one and never a silent fall back to the simulator;
  * a re-detection REPLACES the observation instead of accumulating;
  * perception source and execution backend stay independent.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in ("wisepack_core", "wisepack_bringup"):
    _path = os.path.join(REPO, "wisepack_ws", "src", _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)
for _extra in (os.path.join(REPO, "web"), os.path.join(REPO, "perception")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from wisepack_core.domain import PhysicalObservation, WasteItem   # noqa: E402
from wisepack_core.execution import ExecutionBackend              # noqa: E402
from wisepack_core.harmony_adapter import (                       # noqa: E402
    HARMONY_DETECTOR, UNCALIBRATED_SENTINEL, observations_from_harmony,
    parse_harmony_json,
)
from wisepack_core.packing import OptimizerConfig                 # noqa: E402
from wisepack_core.perception import (                            # noqa: E402
    ARISE_MODEL_PATH, BatchStatus, HUGGINGFACE_REPO, ObservationBatch,
    PerceptionConfigError, PerceptionSource, ProxyGeometry, WorkAreaFrame,
    is_stale, resolve_model_path, resolve_perception_source,
)
from wisepack_core.workflow import (                              # noqa: E402
    PerceptionUnavailable, WorkflowConfig, WorkflowEngine,
)

# --------------------------------------------------------------------------- #
# Fixtures: what HARMONY actually emits
# --------------------------------------------------------------------------- #

#: The `result_json` document HARMONY's ROS 2 backend publishes, verbatim in
#: shape (see ai-bottle-detector-fiware/ros2_backend.py::run_detection_real and
#: pipeline.py::process_frame). Three objects, one of them the "selected" pick.
HARMONY_RESULT = {
    "status": "DONE",
    "bottleCount": 3,
    "pickPose": {"x": 82.4, "y": 46.1, "rotation": -31.0},
    "bottles": [
        {"x": 82.4, "y": 46.1, "yaw": -31.0, "conf": 0.94, "selected": True},
        {"x": 20.0, "y": 110.5, "yaw": 175.25, "conf": 0.87, "selected": False},
        {"x": 118.75, "y": 12.0, "yaw": 0.0, "conf": 0.71, "selected": False},
    ],
}


def _batch(payload=None, **kw) -> ObservationBatch:
    kw.setdefault("batch_id", "batch-001")
    kw.setdefault("captured_at", "2026-08-08T10:00:00.000Z")
    kw.setdefault("model_id", "/data/arise/models/best_model.pth")
    return observations_from_harmony(
        HARMONY_RESULT if payload is None else payload, **kw)


def _engine(**kw) -> WorkflowEngine:
    """A fast engine. `isaac_cylinders_smoke` keeps the optimizer sub-second."""
    config = WorkflowConfig(
        preset="isaac_cylinders_smoke", seed=7,
        optimizer=OptimizerConfig(seed=7, restarts=1, time_budget_ms=400.0),
        **kw)
    return WorkflowEngine(config)


# --------------------------------------------------------------------------- #
# 1. Configuration selection
# --------------------------------------------------------------------------- #


def test_default_perception_source_is_simulated(monkeypatch):
    """The WHOLE compatibility promise in one assertion: unset means `sim`."""
    monkeypatch.delenv("WISEPACK_PERCEPTION_SOURCE", raising=False)
    assert resolve_perception_source() is PerceptionSource.SIM
    assert WorkflowConfig().perception_source is PerceptionSource.SIM
    assert not PerceptionSource.SIM.is_physical


def test_perception_source_selected_from_environment(monkeypatch):
    monkeypatch.setenv("WISEPACK_PERCEPTION_SOURCE", "harmony_camera")
    assert resolve_perception_source() is PerceptionSource.HARMONY_CAMERA
    assert PerceptionSource.HARMONY_CAMERA.is_physical


def test_explicit_argument_overrides_environment(monkeypatch):
    monkeypatch.setenv("WISEPACK_PERCEPTION_SOURCE", "harmony_camera")
    assert resolve_perception_source("sim") is PerceptionSource.SIM


def test_empty_and_whitespace_resolve_to_sim(monkeypatch):
    monkeypatch.setenv("WISEPACK_PERCEPTION_SOURCE", "   ")
    assert resolve_perception_source() is PerceptionSource.SIM


def test_unknown_perception_source_is_an_error_not_a_silent_fallback():
    """A typo must NOT quietly run the simulator behind a camera label."""
    with pytest.raises(PerceptionConfigError) as exc:
        resolve_perception_source("harmony-camera")
    assert "harmony_camera" in str(exc.value)      # names the valid spelling


def test_sim_perception_is_byte_identical_to_before(monkeypatch):
    """Explicit `sim` and the pre-existing default produce the same detections."""
    monkeypatch.delenv("WISEPACK_PERCEPTION_SOURCE", raising=False)
    default = _engine()
    explicit = _engine(perception_source=PerceptionSource.SIM)
    for engine in (default, explicit):
        engine.generate_or_load_scenario()
    assert default.scan_and_detect() == explicit.scan_and_detect()
    assert default.observation_batch is None      # no physical state created


# --------------------------------------------------------------------------- #
# 2. HARMONY -> generic WISEPACK objects
# --------------------------------------------------------------------------- #


def test_harmony_result_becomes_generic_observations():
    batch = _batch()
    assert batch.status is BatchStatus.OK
    assert batch.count == 3
    assert batch.source == PerceptionSource.HARMONY_CAMERA.value
    for obs in batch.observations:
        assert isinstance(obs, PhysicalObservation)
        # DOMAIN-NEUTRAL: the object type says cylinder, not bottle.
        assert obs.object_type == "cylindrical_proxy"
        assert obs.frame_id == "wisepack_workarea"


def test_x_y_yaw_and_confidence_are_preserved_exactly():
    """§3's critical requirement, checked value by value."""
    batch = _batch()
    got = [(o.x_mm, o.y_mm, o.yaw_deg, o.confidence) for o in batch.observations]
    assert got == [(82.4, 46.1, -31.0, 0.94),
                   (20.0, 110.5, 175.25, 0.87),
                   (118.75, 12.0, 0.0, 0.71)]


def test_pose_survives_the_json_round_trip():
    """The observation crosses DDS as JSON; nothing may be lost in transit."""
    original = _batch()
    revived = ObservationBatch.from_dict(json.loads(json.dumps(original.to_dict())))
    assert [(o.x_mm, o.y_mm, o.yaw_deg, o.confidence) for o in revived.observations] \
        == [(o.x_mm, o.y_mm, o.yaw_deg, o.confidence) for o in original.observations]
    assert revived.calibration_status == original.calibration_status
    assert revived.frame_id == original.frame_id


def test_detector_provenance_is_retained():
    """§11: enough to debug or re-analyse a detection months later."""
    batch = _batch()
    obs = batch.observations[0]
    assert obs.source == "harmony_camera"
    assert obs.detector == HARMONY_DETECTOR
    assert obs.model_id.endswith("best_model.pth")
    assert obs.captured_at == "2026-08-08T10:00:00.000Z"
    assert obs.detector_object_index == 0
    assert obs.detector_class == "bottle"          # provenance only
    assert obs.calibration_status == "valid"


def test_core_stays_domain_neutral_outside_the_adapter():
    """No WISEPACK consumer has to understand a bottle to use an observation.

    The distinction the design turns on: the detector's IDENTITY is provenance
    and may name whatever it detects (§11 requires exactly that), but no
    STRUCTURAL field, key or type does. The scene-object document a future Isaac
    synchronizer reads carries no provenance at all, so it is bottle-free
    outright; the item document carries provenance and nothing else bottle-ish.
    """
    scene = json.dumps(_batch().scene_objects())
    assert "bottle" not in scene.lower()

    items = [i.to_dict() for i in _batch().to_waste_items()]
    for item in items:
        assert item["geometry_type"] == "tube"
        assert item["observation"]["object_type"] == "cylindrical_proxy"
        # Every mention is confined to the two provenance fields.
        stripped = dict(item["observation"])
        stripped.pop("detector"), stripped.pop("detector_class")
        assert "bottle" not in json.dumps({**item, "observation": stripped}).lower()


def test_alternative_harmony_payload_shapes_are_accepted():
    """§7: reuse HARMONY's interfaces rather than demand a new one."""
    bare_list = observations_from_harmony(HARMONY_RESULT["bottles"],
                                          batch_id="b1")
    pipeline_shape = observations_from_harmony(
        {"bottles": HARMONY_RESULT["bottles"], "pick_pose": {}}, batch_id="b2")
    assert bare_list.count == pipeline_shape.count == 3


def test_rotation_is_accepted_where_yaw_is_absent():
    batch = observations_from_harmony(
        [{"x": 5.0, "y": 6.0, "rotation": 42.0, "conf": 0.9}], batch_id="b")
    assert batch.observations[0].yaw_deg == 42.0


# --------------------------------------------------------------------------- #
# 3. Known proxy geometry
# --------------------------------------------------------------------------- #


def test_proxy_geometry_is_configured_never_inferred():
    geometry = ProxyGeometry(diameter_mm=70, length_mm=250, wall_thickness_mm=3)
    items = _batch().to_waste_items(geometry=geometry)
    assert len(items) == 3
    for item in items:
        assert isinstance(item, WasteItem)
        assert item.outer_diameter_mm == 70
        assert item.length_mm == 250
        assert item.inner_diameter_mm == 64
        assert item.observation.geometry_source == "configured_proxy"
        # The detector reported no size at all; none was invented from a box.
        assert item.observation.diameter_mm == 70


def test_proxy_geometry_reads_the_documented_environment_variables(monkeypatch):
    monkeypatch.setenv("WISEPACK_PHYSICAL_PROXY_DIAMETER_MM", "80")
    monkeypatch.setenv("WISEPACK_PHYSICAL_PROXY_LENGTH_MM", "300")
    geometry = ProxyGeometry.from_env()
    assert (geometry.diameter_mm, geometry.length_mm) == (80, 300)


def test_nonsense_proxy_geometry_is_rejected_with_a_reason(monkeypatch):
    monkeypatch.setenv("WISEPACK_PHYSICAL_PROXY_DIAMETER_MM", "not-a-number")
    with pytest.raises(PerceptionConfigError):
        ProxyGeometry.from_env()


def test_items_carry_the_measured_pose_into_the_packing_domain():
    """The planner gets integer mm; the exact measurement is not thrown away."""
    items = _batch().to_waste_items()
    assert items[0].source_position.to_dict() == {"x": 82, "y": 46, "z": 0}
    assert items[0].observation.x_mm == 82.4        # the float survives too
    # Horizontal only: a top-down gripper cannot stand a cylinder on its end.
    assert [a.value for a in items[0].permitted_axes] == ["x", "y"]


def test_waste_item_observation_round_trips_and_defaults_to_none():
    """Backward compatibility: every pre-existing item serialises unchanged."""
    plain = WasteItem(item_id="item-001", length_mm=100, outer_diameter_mm=20)
    assert plain.observation is None
    assert plain.to_dict()["observation"] is None
    assert WasteItem.from_dict(plain.to_dict()).observation is None

    observed = _batch().to_waste_items()[0]
    revived = WasteItem.from_dict(json.loads(json.dumps(observed.to_dict())))
    assert revived.observation.x_mm == observed.observation.x_mm
    assert revived.observation.confidence == observed.observation.confidence


# --------------------------------------------------------------------------- #
# 4. Failure handling (§15)
# --------------------------------------------------------------------------- #


def test_no_detections_is_empty_not_an_error():
    """An empty table is a valid measurement; a failed scan is not."""
    batch = observations_from_harmony({"status": "DONE", "bottles": []},
                                      batch_id="b")
    assert batch.status is BatchStatus.EMPTY
    assert batch.ok and batch.count == 0 and not batch.error


def test_malformed_payload_is_a_failed_batch_not_an_exception():
    for payload in (None, 42, "not-json-at-all", {"status": "DONE"}):
        batch = observations_from_harmony(payload, batch_id="b")
        assert batch.status is BatchStatus.ERROR
        assert "malformed" in batch.error.lower()


def test_entries_without_a_usable_position_are_dropped_and_counted():
    batch = observations_from_harmony(
        [{"x": 10.0, "y": 20.0, "conf": 0.9},
         {"y": 5.0, "conf": 0.9},                      # no x
         {"x": "north", "y": 5.0},                     # unparseable
         "not-an-object"],
        batch_id="b")
    assert batch.count == 1
    assert batch.detector_status["malformed_entries"] == 3


def test_a_wholly_malformed_object_list_fails_rather_than_reporting_zero():
    batch = observations_from_harmony([{"conf": 0.9}, {"conf": 0.8}], batch_id="b")
    assert batch.status is BatchStatus.ERROR


def test_invalid_json_on_the_ros_topic_is_a_failed_batch():
    batch = parse_harmony_json("{not json", batch_id="b")
    assert batch.status is BatchStatus.ERROR
    assert "not valid JSON" in batch.error


def test_detector_reported_failure_is_propagated_with_its_reason():
    batch = observations_from_harmony(
        {"status": "FAILED", "error": "no camera frame available"}, batch_id="b")
    assert batch.status is BatchStatus.ERROR
    assert "no camera frame" in batch.error


def test_uncalibrated_frame_is_rejected_rather_than_parsed():
    """HARMONY substitutes (1, 1) for every object when it has no homography.

    Those are not measurements. Planning from a pile of objects at the same
    sentinel point would be nonsense presented as physics.
    """
    x, y = UNCALIBRATED_SENTINEL
    batch = observations_from_harmony(
        [{"x": x, "y": y, "yaw": 0.0, "conf": 0.9},
         {"x": x, "y": y, "yaw": 0.0, "conf": 0.8}], batch_id="b")
    assert batch.status is BatchStatus.ERROR
    assert "calibration" in batch.error.lower()
    assert batch.calibration_status != "valid"


def test_a_single_object_at_the_sentinel_is_still_rejected():
    x, y = UNCALIBRATED_SENTINEL
    batch = observations_from_harmony([{"x": x, "y": y, "conf": 0.9}], batch_id="b")
    assert batch.status is BatchStatus.ERROR


def test_caller_supplied_calibration_status_wins():
    """The detector service knows whether THIS frame resolved the ArUco plane."""
    batch = _batch(calibration_status="invalid")
    assert batch.status is BatchStatus.ERROR
    ok = _batch(calibration_status="valid", calibration_revision="abc123")
    assert ok.calibration_revision == "abc123"
    assert all(o.calibration_revision == "abc123" for o in ok.observations)


def test_out_of_range_confidence_is_dropped_not_carried():
    batch = observations_from_harmony(
        [{"x": 5.0, "y": 5.0, "conf": 42.0}, {"x": 6.0, "y": 6.0, "conf": 1.0000001}],
        batch_id="b")
    assert batch.observations[0].confidence is None      # nonsense, not kept
    assert batch.observations[1].confidence == 1.0       # float artefact, clamped


def test_objects_outside_the_work_area_are_reported_never_moved():
    frame = WorkAreaFrame(width_mm=130, depth_mm=130)
    batch = observations_from_harmony(
        [{"x": 900.0, "y": 900.0, "conf": 0.9}], batch_id="b", frame=frame)
    assert batch.detector_status["outside_workarea"] == 1
    assert batch.observations[0].x_mm == 900.0          # evidence preserved


def test_a_batch_is_always_json_serialisable_even_from_raw_pipeline_output():
    """REGRESSION. `pipeline.process_frame` returns the ANNOTATED IMAGES too.

    Passing its return value straight in — which the detector service does, and
    must, to reuse HARMONY unchanged — used to copy `processed_image` and
    `ai_processed_image` into `detector_status`. Those are numpy arrays of a
    whole camera frame: unserialisable, and megabytes of binary on a topic that
    is published as JSON over DDS. Found by running the real detector on a
    synthetic frame; the batch has to survive `json.dumps` in every path.
    """
    class _FakeImage:                       # stands in for a numpy frame
        def __repr__(self): return "<image>"

    raw = {"bottles": [], "pick_pose": {},
           "processed_image": _FakeImage(), "ai_processed_image": _FakeImage()}
    for status in (None, "valid", "invalid"):
        batch = observations_from_harmony(raw, batch_id="b",
                                          calibration_status=status)
        json.dumps(batch.to_dict())         # must not raise
        assert "processed_image" not in batch.detector_status

    with_objects = observations_from_harmony(
        {**raw, "bottles": HARMONY_RESULT["bottles"]}, batch_id="b")
    json.dumps(with_objects.to_dict())
    assert set(with_objects.detector_status) <= {"status", "bottleCount",
                                                 "pickPose", "pick_pose",
                                                 "malformed_entries",
                                                 "outside_workarea", "workarea"}


def test_a_failed_batch_must_carry_a_reason():
    with pytest.raises(Exception):
        ObservationBatch(batch_id="b", status=BatchStatus.ERROR)


# --------------------------------------------------------------------------- #
# 5. Workflow integration
# --------------------------------------------------------------------------- #


def test_physical_observations_become_the_batch_wisepack_plans_from():
    """§10: the rest of the system does not know where the objects came from."""
    engine = _engine(perception_source=PerceptionSource.HARMONY_CAMERA)
    engine.observation_provider = _batch
    engine.generate_or_load_scenario()
    detected = engine.scan_and_detect()

    assert len(engine.scenario.items) == 3
    assert set(detected) == {"item-001", "item-002", "item-003"}
    assert detected["item-001"] == 0.94

    # ... and the ordinary downstream stages simply work.
    engine.generate_plans()
    assert engine.digital_twin_validate()
    engine.request_approval()
    assert engine.selected.containers_required >= 1


def test_repeated_detection_replaces_and_never_accumulates():
    """§6, the requirement that makes the batch atomic rather than incremental."""
    engine = _engine(perception_source=PerceptionSource.HARMONY_CAMERA)
    engine.observation_provider = _batch
    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    assert len(engine.scenario.items) == 3
    first_revision = engine.scenario_revision

    # The operator moves the objects and detects again: TWO objects now.
    moved = observations_from_harmony(
        [{"x": 5.0, "y": 5.0, "yaw": 10.0, "conf": 0.9},
         {"x": 100.0, "y": 100.0, "yaw": -10.0, "conf": 0.8}],
        batch_id="batch-002", captured_at="2026-08-08T10:05:00.000Z")
    engine.apply_observation_batch(moved)

    assert len(engine.scenario.items) == 2          # not 5
    assert engine.scenario.items[0].observation.x_mm == 5.0
    assert engine.observation_batch.batch_id == "batch-002"
    assert engine.observation_batches_applied == 2
    assert engine.scenario_revision > first_revision


def test_a_new_batch_revokes_an_outstanding_approval():
    """An approval is a decision about one batch. The objects moved; it lapses."""
    engine = _engine(perception_source=PerceptionSource.HARMONY_CAMERA)
    engine.observation_provider = _batch
    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    engine.generate_plans()
    engine.digital_twin_validate()
    engine.request_approval()
    engine.approve(operator="tester")
    assert engine.selected.approval_state.value == "approved"

    engine.apply_observation_batch(_batch(batch_id="batch-002"))
    assert engine.selected.approval_state.value != "approved"


def test_a_failed_scan_never_falls_back_to_simulated_detections():
    """§15's headline rule, asserted directly."""
    engine = _engine(perception_source=PerceptionSource.HARMONY_CAMERA)
    engine.observation_provider = lambda: ObservationBatch.failed(
        batch_id="b", source="harmony_camera", error="camera disconnected")
    engine.generate_or_load_scenario()
    with pytest.raises(PerceptionUnavailable) as exc:
        engine.scan_and_detect()
    assert "camera disconnected" in str(exc.value)
    assert engine.detected == {}                    # NOT simulated detections
    # The failure is the CURRENT state, so the dashboard can render the reason.
    assert engine.observation_batch.status is BatchStatus.ERROR
    assert engine.perception_state()["batch"]["error"] == "camera disconnected"


def test_a_provider_that_raises_is_a_failed_scan_not_a_crash():
    """Camera or model failure must not take unrelated components down (§5)."""
    def explode():
        raise ConnectionRefusedError("HARMONY service unavailable")

    engine = _engine(perception_source=PerceptionSource.HARMONY_CAMERA)
    engine.observation_provider = explode
    engine.generate_or_load_scenario()
    with pytest.raises(PerceptionUnavailable) as exc:
        engine.scan_and_detect()
    assert "HARMONY service unavailable" in str(exc.value)


def test_a_missing_provider_says_what_to_start():
    engine = _engine(perception_source=PerceptionSource.HARMONY_CAMERA)
    engine.generate_or_load_scenario()
    with pytest.raises(PerceptionUnavailable) as exc:
        engine.scan_and_detect()
    assert "WISEPACK_HARMONY_SERVICE_URL" in str(exc.value)


def test_an_empty_physical_batch_plans_nothing_without_claiming_failure():
    engine = _engine(perception_source=PerceptionSource.HARMONY_CAMERA)
    engine.observation_provider = lambda: observations_from_harmony(
        {"status": "DONE", "bottles": []}, batch_id="b")
    engine.generate_or_load_scenario()
    assert engine.scan_and_detect() == {}
    assert engine.scenario.items == []
    assert engine.observation_batch.status is BatchStatus.EMPTY


# --------------------------------------------------------------------------- #
# 6. Staleness
# --------------------------------------------------------------------------- #


def test_observation_staleness_is_computed_from_the_capture_time():
    import calendar
    batch = _batch()                       # captured 2026-08-08T10:00:00Z
    captured_epoch = float(calendar.timegm((2026, 8, 8, 10, 0, 0, 0, 0, 0)))
    from wisepack_core.perception import observation_age_s

    assert observation_age_s(batch, now_epoch=captured_epoch) == 0.0
    # Clamped at zero rather than negative: a clock skew must not render as a
    # batch from the future.
    assert observation_age_s(batch, now_epoch=captured_epoch - 500) == 0.0

    assert observation_age_s(batch, now_epoch=captured_epoch + 600) == 600.0
    assert is_stale(batch, ttl_s=300.0, now_epoch=captured_epoch + 600)
    assert not is_stale(batch, ttl_s=900.0, now_epoch=captured_epoch + 600)


def test_an_unstamped_batch_is_not_reported_stale():
    """Unknown age must not render as "stale" — that would be an invention."""
    batch = observations_from_harmony(HARMONY_RESULT, batch_id="b", captured_at="")
    assert not is_stale(batch)


# --------------------------------------------------------------------------- #
# 7. KPI honesty (§12)
# --------------------------------------------------------------------------- #


def test_detector_confidence_never_becomes_a_detection_rate():
    """0.94 confidence must NOT surface as "vision detection rate = 94%"."""
    engine = _engine(perception_source=PerceptionSource.HARMONY_CAMERA)
    engine.observation_provider = _batch
    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    engine.generate_plans()
    engine.digital_twin_validate()

    report = engine.kpis().to_dict()
    rate = report["metrics"]["detection_rate_pct"]
    assert rate["value"] is None and rate["measured"] is False
    assert "not measured" in rate["note"]
    assert "ground-truth" in rate["note"]

    # The confidence IS published — under a name that cannot be read as a rate.
    confidence = report["metrics"]["physical_detector_mean_confidence"]
    assert confidence["value"] == pytest.approx(0.84, abs=0.01)
    assert "NOT a detection rate" in confidence["note"]

    kpi1 = next(t for t in report["target_assessment"] if t["key"] == "KPI1")
    assert kpi1["status"] == "not_applicable"


def test_simulated_perception_kpi_wording_is_unchanged():
    engine = _engine()
    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    engine.generate_plans()
    engine.digital_twin_validate()
    rate = engine.kpis().to_dict()["metrics"]["detection_rate_pct"]
    assert rate["source"] == "simulated"
    assert rate["value"] == 100.0            # detected / actually generated


# --------------------------------------------------------------------------- #
# 8. Orthogonality: perception source vs execution backend
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("perception", list(PerceptionSource))
@pytest.mark.parametrize("backend", list(ExecutionBackend))
def test_every_perception_and_backend_combination_is_configurable(perception,
                                                                  backend):
    """A camera is not an execution backend and vice versa (the §-architecture)."""
    config = WorkflowConfig(perception_source=perception,
                            execution_backend=backend)
    assert config.perception_source is perception
    assert config.execution_backend is backend


def test_selecting_a_camera_does_not_change_the_execution_backend():
    assert WorkflowConfig(
        perception_source=PerceptionSource.HARMONY_CAMERA
    ).execution_backend is ExecutionBackend.SIMULATED


def test_selecting_isaac_does_not_change_the_perception_source():
    assert WorkflowConfig(
        execution_backend=ExecutionBackend.ISAAC
    ).perception_source is PerceptionSource.SIM


def test_the_engine_reports_both_axes_separately():
    engine = _engine(perception_source=PerceptionSource.HARMONY_CAMERA,
                     execution_backend=ExecutionBackend.ISAAC)
    state = engine.perception_state()
    assert state["perception_source"] == "harmony_camera"
    assert state["physical"] is True
    # Nothing in the perception payload names or implies an execution backend.
    assert "isaac" not in json.dumps(state).lower()


# --------------------------------------------------------------------------- #
# 9. Model resolution (§4)
# --------------------------------------------------------------------------- #


def test_model_resolution_prefers_the_configured_path():
    result = resolve_model_path(configured="/tmp/weights.pth",
                                exists=lambda p: True)
    assert result.origin == "configured" and result.available


def test_a_configured_path_that_is_missing_is_an_error_not_a_fallback():
    """Loading different weights than the ones asked for is worse than failing."""
    result = resolve_model_path(configured="/nowhere/weights.pth",
                                harmony_dir="/harmony", exists=lambda p: True)
    assert False is result.available or result.origin == "configured"


def test_configured_missing_path_reports_itself():
    result = resolve_model_path(configured="/nowhere/weights.pth",
                                harmony_dir="/harmony", exists=lambda p: False)
    assert not result.available
    assert "/nowhere/weights.pth" in result.message
    assert result.searched == ["/nowhere/weights.pth"]


def test_model_resolution_falls_back_to_the_shared_arise_copy():
    result = resolve_model_path(configured="", harmony_dir="/harmony",
                                env={}, exists=lambda p: p == ARISE_MODEL_PATH)
    assert result.path == ARISE_MODEL_PATH
    assert result.origin == "arise_shared"


def test_model_resolution_falls_back_to_the_harmony_cache():
    cached = "/harmony/models/best_model.pth"
    result = resolve_model_path(configured="", harmony_dir="/harmony",
                                env={}, exists=lambda p: p == cached)
    assert result.path == cached and result.origin == "harmony_cache"


def test_absent_model_is_a_clear_diagnostic_not_a_pytorch_traceback():
    """§4: absence must name the file, the repo and the command to fix it."""
    result = resolve_model_path(configured="", harmony_dir="/harmony",
                                env={}, exists=lambda p: False)
    assert not result.available and result.path is None
    assert ARISE_MODEL_PATH in result.message
    assert HUGGINGFACE_REPO in result.message
    assert "curl" in result.message
    assert result.to_dict()["download_url"].endswith("best_model.pth")


# --------------------------------------------------------------------------- #
# 10. The Isaac-synchronizer boundary (§14)
# --------------------------------------------------------------------------- #


def test_scene_objects_expose_pose_and_geometry_without_detector_json():
    """A future synchronizer needs identity + pose + geometry + frame. No more."""
    batch = _batch()
    batch.to_waste_items(geometry=ProxyGeometry(diameter_mm=70, length_mm=250))
    objects = batch.scene_objects()
    assert len(objects) == 3
    first = objects[0]
    assert first["object_id"] == "physical-cylinder-001"
    assert first["frame_id"] == "wisepack_workarea"
    assert first["pose"] == {"x_mm": 82.4, "y_mm": 46.1, "z_mm": 0.0,
                             "yaw_deg": -31.0}
    assert first["geometry"] == {"shape": "cylinder", "diameter_mm": 70,
                                 "length_mm": 250, "source": "configured_proxy"}
    # It is a plain JSON document — no HARMONY key survives into it.
    assert "bottles" not in json.dumps(objects)


# --------------------------------------------------------------------------- #
# 11. ROS 2 contract
# --------------------------------------------------------------------------- #


def test_perception_topics_obey_the_wisepack_topic_rules():
    from wisepack_bringup import topics as T

    channel = T.perception_topics()
    assert set(channel) == {T.PERCEPTION_OBJECTS, T.PERCEPTION_STATUS}
    # Scalar std_msgs only — anything else is unbridgeable on the DDS path.
    assert set(channel.values()) == {"std_msgs/String"}
    # No reserved `status` leaf anywhere in the contract, this channel included.
    assert T.reserved_leaf_violations() == []


def test_perception_channel_is_not_in_the_always_present_contract():
    """It has a publisher only with a real perception source."""
    from wisepack_bringup import topics as T
    assert T.PERCEPTION_OBJECTS not in T.all_topics()
    assert T.PERCEPTION_STATUS not in T.all_topics()


def test_the_detection_command_reuses_harmonys_own_topic():
    """§7: trigger over HARMONY's contract rather than inventing a parallel one."""
    from wisepack_bringup import topics as T
    assert T.HARMONY_DETECTION_COMMAND == "/bottle_detection/command"
    assert T.PERCEPTION_WRITERS[T.HARMONY_DETECTION_COMMAND] == "wisepack_orchestration"


def test_detect_physical_objects_is_in_the_operator_vocabulary():
    from wisepack_bringup import topics as T
    assert "detect_physical_objects" in T.OPERATOR_COMMANDS


def test_the_service_and_the_contract_agree_on_the_topic_names():
    """The service runs outside the ROS container and still cannot drift."""
    import harmony_perception_service as svc
    from wisepack_bringup import topics as T
    assert svc.WISEPACK_PERCEPTION_STATUS == T.PERCEPTION_STATUS
    assert svc.WISEPACK_PERCEPTION_OBJECTS == T.PERCEPTION_OBJECTS


# --------------------------------------------------------------------------- #
# 12. The detector service's own configuration (no camera, no torch)
# --------------------------------------------------------------------------- #


def test_service_config_honours_the_camera_override(monkeypatch):
    import harmony_perception_service as svc
    monkeypatch.setenv("WISEPACK_HARMONY_CAMERA", "4")
    assert svc.build_harmony_config()["CAMERA"] == 4
    # A device path or an RTSP URL stays a string; only a bare index is an int.
    monkeypatch.setenv("WISEPACK_HARMONY_CAMERA", "/dev/video2")
    assert svc.build_harmony_config()["CAMERA"] == "/dev/video2"


def test_service_config_takes_an_absolute_model_path():
    import harmony_perception_service as svc
    config = svc.build_harmony_config(model_path=ARISE_MODEL_PATH)
    assert config["MODEL_PATH"] == ARISE_MODEL_PATH


def test_a_larger_calibration_board_is_configuration_not_code(monkeypatch):
    """§13: make the extent configurable rather than hardcode a new layout."""
    import harmony_perception_service as svc
    monkeypatch.setenv("WISEPACK_HARMONY_CORNER_EXTENT_MM", "600")
    monkeypatch.setenv("WISEPACK_HARMONY_CORNER_MARKERS", "11,10,15,16")
    config = svc.build_harmony_config()
    assert config["CORNER_COORDINATES"] == [[0, 0], [600.0, 0], [600.0, 600.0],
                                            [0, 600.0]]
    frame = svc.work_area_from_config(config)
    assert (frame.width_mm, frame.depth_mm) == (600, 600)


def test_the_work_area_frame_follows_harmonys_calibrated_plane():
    """WISEPACK adopts HARMONY's coordinate system; it does not redesign it."""
    import harmony_perception_service as svc
    frame = svc.work_area_from_config(
        {"CORNER_COORDINATES": [[0, 0], [130, 0], [130, 130], [0, 130]]})
    assert (frame.width_mm, frame.depth_mm) == (130, 130)
    assert frame.frame_id == "wisepack_workarea"


# --------------------------------------------------------------------------- #
# 13. Dashboard client and API serialisation
# --------------------------------------------------------------------------- #


class _StubClient:
    """A perception service that is not there. §15's "HARMONY unavailable"."""

    def __init__(self, url="http://127.0.0.1:22101"):
        self.url = url

    def _request(self, path, method="GET", timeout_s=None):
        return None, None, f"perception service unreachable at {self.url}"


def test_client_reports_an_unreachable_service_without_raising():
    from perception_client import PerceptionClient
    client = PerceptionClient(url="http://127.0.0.1:1")
    client._request = _StubClient()._request

    health = client.health()
    assert health["service_reachable"] is False
    # TRI-STATE: unknown is not the same as "no camera".
    assert health["camera_available"] is None
    assert "unreachable" in health["last_error"]
    assert "harmony_perception_service" in health["note"]

    batch = client.detect()
    assert batch.status is BatchStatus.ERROR
    assert client.last_detection() is None
    image, error = client.image("annotated")
    assert image is None and error


def test_client_rejects_an_unreadable_document_rather_than_guessing():
    from perception_client import PerceptionClient
    client = PerceptionClient()
    client._request = lambda *a, **k: (200, {"batch_id": 1, "status": "nonsense"}, "")
    batch = client.detect()
    assert batch.status is BatchStatus.ERROR
    assert "cannot read" in batch.error


def test_client_parses_a_real_service_response():
    from perception_client import PerceptionClient
    client = PerceptionClient()
    document = _batch().to_dict()
    client._request = lambda *a, **k: (200, document, "")
    batch = client.detect()
    assert batch.count == 3
    assert batch.observations[0].x_mm == 82.4


def test_perception_state_serialises_for_the_api():
    """§16: the dashboard payload must be JSON and must carry the poses."""
    engine = _engine(perception_source=PerceptionSource.HARMONY_CAMERA)
    engine.observation_provider = _batch
    engine.generate_or_load_scenario()
    engine.scan_and_detect()

    payload = json.loads(json.dumps(engine.perception_state()))
    assert payload["perception_source"] == "harmony_camera"
    assert payload["perception_source_label"] == "HARMONY CAMERA"
    assert payload["physical"] is True
    assert payload["batch"]["count"] == 3
    assert payload["batch"]["observations"][0]["pose"]["yaw_deg"] == -31.0
    assert payload["batch"]["observations"][0]["confidence"] == 0.94
    assert payload["proxy_geometry"]["source"] == "configured_proxy"
    assert payload["work_area"]["frame_id"] == "wisepack_workarea"
    assert len(payload["scene_objects"]) == 3
    assert payload["observation_stale"] in (True, False)


def test_perception_state_in_sim_mode_says_so_plainly():
    engine = _engine()
    payload = engine.perception_state()
    assert payload["perception_source"] == "sim"
    assert payload["physical"] is False
    assert payload["batch"] is None
    assert payload["proxy_geometry"] is None       # no proxy in a simulated run


def test_diagnostics_describes_both_perception_modes():
    """§17: no surface may still say "no camera" while a camera is running."""
    import diagnostics

    sim = {i: s for i, s, _ in diagnostics.interfaces("sim")}
    assert sim["2-D camera frames"] == "future interface"
    assert sim["Object detections"] == "simulated source"

    camera = {i: s for i, s, _ in diagnostics.interfaces("harmony_camera")}
    assert camera["2-D camera frames"] == "live"
    assert camera["Object detections"] == "measured source"

    # The PROPOSAL's RGB-D pipeline is still not implemented, and a 2-D camera
    # does not make it so. Overclaiming here is the failure this table prevents.
    for table in (sim, camera):
        assert table["RGB-D camera frames"] == "future interface"
        assert table["Robot joint states"] == "simulated source"

    meanings = {i: m for i, s, m in diagnostics.interfaces("harmony_camera")}
    assert "NOT a measured detection rate" in meanings["Object detections"]
    assert "PLANAR ONLY" in meanings["6D pose estimates"]


def test_a_configured_but_dead_detector_is_not_reported_as_live():
    """CONFIGURED is not LIVE. Selecting a camera does not make one answer."""
    import diagnostics

    dead = {i: s for i, s, _ in
            diagnostics.interfaces("harmony_camera", reachable=False)}
    assert dead["2-D camera frames"] == "configured, unavailable"
    assert dead["Object detections"] == "configured, unavailable"
    # Still not a FAILURE state — the table's other rule.
    for state in dead.values():
        assert "error" not in state.lower() and "fail" not in state.lower()

    # Unchecked reachability makes neither claim.
    unknown = {i: s for i, s, _ in diagnostics.interfaces("harmony_camera")}
    assert unknown["2-D camera frames"] == "live"
    # ... and `sim` is never downgraded: there is no service to be dead.
    sim = {i: s for i, s, _ in diagnostics.interfaces("sim", reachable=False)}
    assert sim["Object detections"] == "simulated source"


def test_the_dashboard_panel_stays_domain_neutral():
    """§9/§18: the panel must not make the dashboard bottle-specific."""
    with open(os.path.join(REPO, "web", "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    assert 'id="perceppanel"' in html
    assert "Physical Perception" in html
    assert "Detect physical objects" in html
    # THE PROXY DISCLOSURE IS SERVED FROM THE API, not hard-coded here, so the
    # markup never has to name the physical stand-in object at all. That is what
    # keeps the dashboard from becoming detector-specific (§18).
    assert "bottle" not in html.lower()
