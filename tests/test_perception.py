"""Perception source: configuration, adaptation, and the invariants that matter.

Every test here runs with NO camera, NO GPU, NO model weights and NO network.
That is the point: the physical perception path has to be verifiable in ordinary
CI, so the detector's output is represented by the JSON it actually produces and
the transport by a stub. What the tests check is the part WISEPACK owns —

  * the default is still simulated perception, and `sim` behaves exactly as it
    did before this code existed;
  * a detector result becomes DOMAIN-NEUTRAL objects with x/y/yaw/confidence
    preserved and the CONFIGURED proxy geometry attached;
  * every failure §15 lists is a visible failed batch, never an empty
    successful one and never a silent fall back to the simulator;
  * a re-detection REPLACES the observation instead of accumulating;
  * perception source and execution backend stay independent.
"""

from __future__ import annotations

import json
import os
import pathlib
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
# THE PROVIDER, imported from `perception/providers/` — the only detector-aware
# code in the system. `wisepack_core` deliberately no longer contains it.
from providers.fasterrcnn_bottle import (                          # noqa: E402
    DETECTOR_ID, UNCALIBRATED_SENTINEL, observations_from_detections,
    parse_detection_json,
)
from wisepack_core.packing import OptimizerConfig                 # noqa: E402
from wisepack_core.perception import (                            # noqa: E402
    ARISE_MODEL_PATH, BatchStatus, DEFAULT_DETECTOR, HUGGINGFACE_REPO,
    KNOWN_DETECTORS, ObservationBatch, PerceptionConfigError, PerceptionSource,
    ProxyGeometry, WorkAreaFrame, is_stale, resolve_detector, resolve_model_path,
    resolve_perception_source,
)
from wisepack_core.workflow import (                              # noqa: E402
    PerceptionUnavailable, WorkflowConfig, WorkflowEngine,
)

# --------------------------------------------------------------------------- #
# Fixtures: what the WISEPACK detector actually emits
# --------------------------------------------------------------------------- #

#: The scalar part of what `providers/fasterrcnn_bottle.BottleDetector.process_frame`
#: returns — verified against the real detector by
#: `test_the_real_detector_reproduces_the_reference_measurement` below, which
#: runs the actual network when torch and the weights are present. Three
#: objects, one of them the "selected" pick.
DETECTOR_RESULT = {
    "status": "DONE",
    "object_count": 3,
    "pick_pose": {"x": 82.4, "y": 46.1, "rotation": -31.0},
    "objects": [
        {"x": 82.4, "y": 46.1, "yaw": -31.0, "conf": 0.94, "selected": True},
        {"x": 20.0, "y": 110.5, "yaw": 175.25, "conf": 0.87, "selected": False},
        {"x": 118.75, "y": 12.0, "yaw": 0.0, "conf": 0.71, "selected": False},
    ],
}


def _batch(payload=None, **kw) -> ObservationBatch:
    kw.setdefault("batch_id", "batch-001")
    kw.setdefault("captured_at", "2026-08-08T10:00:00.000Z")
    kw.setdefault("model_id", "/data/arise/models/best_model.pth")
    return observations_from_detections(
        DETECTOR_RESULT if payload is None else payload, **kw)


def _code_without_prose(text: str) -> str:
    """Source with comment lines AND docstrings removed.

    Docstrings are located with `ast` rather than by matching quotes, so a
    triple quote inside an ordinary string cannot confuse it. Needed because
    these modules DOCUMENT the rule they obey — `domain.py` says in prose that
    nothing there names a bottle or a detector — and a substring check that
    could not tell a prohibition from the thing prohibited would forbid writing
    the rule down.
    """
    import ast
    lines = text.splitlines()
    drop = set()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            doc = body[0].value
            drop.update(range(doc.lineno - 1, (doc.end_lineno or doc.lineno)))
    return "\n".join(line for i, line in enumerate(lines)
                      if i not in drop and not line.lstrip().startswith("#"))


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
    monkeypatch.setenv("WISEPACK_PERCEPTION_SOURCE", "camera")
    assert resolve_perception_source() is PerceptionSource.CAMERA
    assert PerceptionSource.CAMERA.is_physical


def test_explicit_argument_overrides_environment(monkeypatch):
    monkeypatch.setenv("WISEPACK_PERCEPTION_SOURCE", "camera")
    assert resolve_perception_source("sim") is PerceptionSource.SIM


def test_empty_and_whitespace_resolve_to_sim(monkeypatch):
    monkeypatch.setenv("WISEPACK_PERCEPTION_SOURCE", "   ")
    assert resolve_perception_source() is PerceptionSource.SIM


def test_unknown_perception_source_is_an_error_not_a_silent_fallback():
    """A typo must NOT quietly run the simulator behind a camera label."""
    with pytest.raises(PerceptionConfigError) as exc:
        resolve_perception_source("harmony-camera")
    assert "camera" in str(exc.value)      # names the valid spelling


def test_the_dashboard_never_downgrades_an_invalid_source_to_sim():
    """REGRESSION. `web/app.py` used to swallow PerceptionConfigError.

    It caught the exception, set the source to `sim`, and put the reason in a
    `config_error` field whose only renderer sits behind an early return that a
    non-physical source never passes. Net effect:
    an unrecognised `WISEPACK_PERCEPTION_SOURCE` started
    a dashboard producing SIMULATED detections for an operator who had asked for
    a camera, with nothing on screen to say so.

    A source assertion rather than a mock, for the same reason
    `tests/test_isaac_robots.py` asserts on index.html: this is a rule about
    what the module is allowed to contain, and it has to hold whether or not
    FastAPI is installed in the environment running the suite.
    """
    source = (pathlib.Path(REPO) / "web" / "app.py").read_text()

    resolution = [ln for ln in source.splitlines()
                  if "PERCEPTION_SOURCE = resolve_perception_source()" in ln]
    assert resolution, "web/app.py must resolve the perception source at import"
    # Resolved at module scope, unguarded — not inside a try/except that could
    # substitute a different source.
    assert resolution[0].startswith("PERCEPTION_SOURCE ="), \
        f"the resolution is indented, so it is inside a guard: {resolution[0]!r}"

    assert "except PerceptionConfigError" not in source, (
        "web/app.py must not catch PerceptionConfigError — an unrecognised "
        "perception source has to stop the process, exactly as the ROS "
        "orchestrator does, instead of silently selecting `sim`")
    assert "PERCEPTION_CONFIG_ERROR" not in source, (
        "the config_error carrier is dead once the fallback is gone; leaving it "
        "invites the fallback back")


def test_dashboard_and_orchestrator_resolve_the_source_identically():
    """The two entry points must not disagree about what is a valid config.

    A value that kills the ROS stack must not quietly run in the dashboard, and
    vice versa. Both call the same core resolver and neither guards it.
    """
    web = (pathlib.Path(REPO) / "web" / "app.py").read_text()
    orch = (pathlib.Path(REPO) / "wisepack_ws" / "src" / "wisepack_orchestration"
            / "wisepack_orchestration" / "hitl_orchestrator.py").read_text()
    for name, text in (("web/app.py", web), ("hitl_orchestrator.py", orch)):
        assert "resolve_perception_source(" in text, name
        assert "except PerceptionConfigError" not in text, (
            f"{name} swallows an invalid perception source")


@pytest.mark.parametrize("value,expected", [
    ("sim", PerceptionSource.SIM),
    ("camera", PerceptionSource.CAMERA),
])
def test_valid_sources_are_accepted_from_the_environment(monkeypatch, value,
                                                         expected):
    """Both supported modes still resolve — the fix must not break either."""
    monkeypatch.setenv("WISEPACK_PERCEPTION_SOURCE", value)
    assert resolve_perception_source() is expected
    assert WorkflowConfig(perception_source=resolve_perception_source()
                          ).perception_source is expected


@pytest.mark.parametrize("bad", [
    "harmony_camera",       # the value this branch used before the refactor
    "harmony-camera", "cam", "kamera", "physical", "real", "true", "1",
])
def test_an_invalid_source_never_resolves_to_sim(monkeypatch, bad):
    """The core contract the dashboard now relies on: raise, never downgrade."""
    monkeypatch.setenv("WISEPACK_PERCEPTION_SOURCE", bad)
    with pytest.raises(PerceptionConfigError) as exc:
        resolve_perception_source()
    message = str(exc.value)
    assert bad.lower() in message.lower()       # names what was wrong
    assert "known: sim, camera" in message      # names the valid spellings


def test_an_invalid_source_stops_the_dashboard_process(monkeypatch):
    """End-to-end: the server must not come up and must not run the simulator.

    Skipped where FastAPI is absent (the dashboard cannot start at all there);
    the source assertions above hold unconditionally.
    """
    pytest.importorskip("fastapi",
                        reason="the dashboard needs FastAPI to start")
    import subprocess
    env = {**os.environ, "WISEPACK_PERCEPTION_SOURCE": "harmony_camera"}
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "web", "app.py"),
         "--source", "sim", "--port", "0"],
        capture_output=True, text=True, timeout=120, env=env, cwd=REPO)
    assert proc.returncode != 0, "the dashboard started despite an invalid source"
    combined = proc.stdout + proc.stderr
    assert "harmony_camera" in combined and "known: sim, camera" in combined, (
        "the failure must name both the bad value and the valid spellings")
    assert "SIMULATED PERCEPTION" not in combined


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
    assert batch.source == PerceptionSource.CAMERA.value
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
    assert obs.source == "camera"
    assert obs.detector == DETECTOR_ID
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


def test_alternative_detector_payload_shapes_are_accepted():
    """Absorbing the variety is the PROVIDER's job, not the domain's.

    `bottles` is the key the implementation this provider was adapted from used;
    it is still accepted so a recorded result or an older transport document
    keeps parsing, while the current detector emits `objects`.
    """
    bare_list = observations_from_detections(DETECTOR_RESULT["objects"],
                                             batch_id="b1")
    detector_shape = observations_from_detections(
        {"objects": DETECTOR_RESULT["objects"], "pick_pose": {}}, batch_id="b2")
    legacy_shape = observations_from_detections(
        {"bottles": DETECTOR_RESULT["objects"], "pick_pose": {}}, batch_id="b3")
    assert bare_list.count == detector_shape.count == legacy_shape.count == 3


def test_rotation_is_accepted_where_yaw_is_absent():
    batch = observations_from_detections(
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
    batch = observations_from_detections({"status": "DONE", "bottles": []},
                                      batch_id="b")
    assert batch.status is BatchStatus.EMPTY
    assert batch.ok and batch.count == 0 and not batch.error


def test_malformed_payload_is_a_failed_batch_not_an_exception():
    for payload in (None, 42, "not-json-at-all", {"status": "DONE"}):
        batch = observations_from_detections(payload, batch_id="b")
        assert batch.status is BatchStatus.ERROR
        assert "malformed" in batch.error.lower()


def test_entries_without_a_usable_position_are_dropped_and_counted():
    batch = observations_from_detections(
        [{"x": 10.0, "y": 20.0, "conf": 0.9},
         {"y": 5.0, "conf": 0.9},                      # no x
         {"x": "north", "y": 5.0},                     # unparseable
         "not-an-object"],
        batch_id="b")
    assert batch.count == 1
    assert batch.detector_status["malformed_entries"] == 3


def test_a_wholly_malformed_object_list_fails_rather_than_reporting_zero():
    batch = observations_from_detections([{"conf": 0.9}, {"conf": 0.8}], batch_id="b")
    assert batch.status is BatchStatus.ERROR


def test_invalid_json_on_the_ros_topic_is_a_failed_batch():
    batch = parse_detection_json("{not json", batch_id="b")
    assert batch.status is BatchStatus.ERROR
    assert "not valid JSON" in batch.error


def test_detector_reported_failure_is_propagated_with_its_reason():
    batch = observations_from_detections(
        {"status": "FAILED", "error": "no camera frame available"}, batch_id="b")
    assert batch.status is BatchStatus.ERROR
    assert "no camera frame" in batch.error


def test_uncalibrated_frame_is_rejected_rather_than_parsed():
    """The calibration substitutes (1, 1) for every object when it has no homography.

    Those are not measurements. Planning from a pile of objects at the same
    sentinel point would be nonsense presented as physics.
    """
    x, y = UNCALIBRATED_SENTINEL
    batch = observations_from_detections(
        [{"x": x, "y": y, "yaw": 0.0, "conf": 0.9},
         {"x": x, "y": y, "yaw": 0.0, "conf": 0.8}], batch_id="b")
    assert batch.status is BatchStatus.ERROR
    assert "calibration" in batch.error.lower()
    assert batch.calibration_status != "valid"


def test_a_single_object_at_the_sentinel_is_still_rejected():
    x, y = UNCALIBRATED_SENTINEL
    batch = observations_from_detections([{"x": x, "y": y, "conf": 0.9}], batch_id="b")
    assert batch.status is BatchStatus.ERROR


def test_caller_supplied_calibration_status_wins():
    """The detector service knows whether THIS frame resolved the ArUco plane."""
    batch = _batch(calibration_status="invalid")
    assert batch.status is BatchStatus.ERROR
    ok = _batch(calibration_status="valid", calibration_revision="abc123")
    assert ok.calibration_revision == "abc123"
    assert all(o.calibration_revision == "abc123" for o in ok.observations)


def test_out_of_range_confidence_is_dropped_not_carried():
    batch = observations_from_detections(
        [{"x": 5.0, "y": 5.0, "conf": 42.0}, {"x": 6.0, "y": 6.0, "conf": 1.0000001}],
        batch_id="b")
    assert batch.observations[0].confidence is None      # nonsense, not kept
    assert batch.observations[1].confidence == 1.0       # float artefact, clamped


def test_objects_outside_the_work_area_are_reported_never_moved():
    frame = WorkAreaFrame(width_mm=130, depth_mm=130)
    batch = observations_from_detections(
        [{"x": 900.0, "y": 900.0, "conf": 0.9}], batch_id="b", frame=frame)
    assert batch.detector_status["outside_workarea"] == 1
    assert batch.observations[0].x_mm == 900.0          # evidence preserved


def test_a_batch_is_always_json_serialisable_even_from_raw_pipeline_output():
    """REGRESSION. `process_frame` returns the ANNOTATED IMAGES too.

    Passing its return value straight in — which the detector service does —
    used to copy the annotated frames into `detector_status`. Those are numpy
    arrays of a whole camera frame: unserialisable, and megabytes of binary on a
    topic that is published as JSON over DDS. Found by running the real detector
    on a synthetic frame; the batch has to survive `json.dumps` in every path.
    """
    class _FakeImage:                       # stands in for a numpy frame
        def __repr__(self): return "<image>"

    raw = {"objects": [], "pick_pose": {},
           "annotated_image": _FakeImage(), "detections_image": _FakeImage(),
           # the key names the earlier implementation used, for good measure
           "processed_image": _FakeImage(), "ai_processed_image": _FakeImage()}
    for status in (None, "valid", "invalid"):
        batch = observations_from_detections(raw, batch_id="b",
                                             calibration_status=status)
        json.dumps(batch.to_dict())         # must not raise
        assert "annotated_image" not in batch.detector_status
        assert "processed_image" not in batch.detector_status

    with_objects = observations_from_detections(
        {**raw, "objects": DETECTOR_RESULT["objects"]}, batch_id="b")
    json.dumps(with_objects.to_dict())
    assert set(with_objects.detector_status) <= {"status", "object_count",
                                                 "objects_without_orientation",
                                                 "caps_detected", "bottleCount",
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
    engine = _engine(perception_source=PerceptionSource.CAMERA)
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
    engine = _engine(perception_source=PerceptionSource.CAMERA)
    engine.observation_provider = _batch
    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    assert len(engine.scenario.items) == 3
    first_revision = engine.scenario_revision

    # The operator moves the objects and detects again: TWO objects now.
    moved = observations_from_detections(
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
    engine = _engine(perception_source=PerceptionSource.CAMERA)
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
    engine = _engine(perception_source=PerceptionSource.CAMERA)
    engine.observation_provider = lambda: ObservationBatch.failed(
        batch_id="b", source="camera", error="camera disconnected")
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
        raise ConnectionRefusedError("perception service unavailable")

    engine = _engine(perception_source=PerceptionSource.CAMERA)
    engine.observation_provider = explode
    engine.generate_or_load_scenario()
    with pytest.raises(PerceptionUnavailable) as exc:
        engine.scan_and_detect()
    assert "perception service unavailable" in str(exc.value)


def test_a_missing_provider_says_what_to_start():
    engine = _engine(perception_source=PerceptionSource.CAMERA)
    engine.generate_or_load_scenario()
    with pytest.raises(PerceptionUnavailable) as exc:
        engine.scan_and_detect()
    assert "WISEPACK_PERCEPTION_SERVICE_URL" in str(exc.value)


def test_an_empty_physical_batch_plans_nothing_without_claiming_failure():
    engine = _engine(perception_source=PerceptionSource.CAMERA)
    engine.observation_provider = lambda: observations_from_detections(
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


# --------------------------------------------------------------------------- #
# 6a. Timestamp semantics
# --------------------------------------------------------------------------- #


class _StubFrame:
    """A camera frame. Only `.copy()` is used by the detector runtime."""

    def copy(self):
        return self


class _StubDetector:
    """Stands in for the provider: `process_frame` and nothing else.

    That IS the provider contract, which is why the whole service is testable
    without torch: if a stub this small satisfies the runtime, no detector
    knowledge has leaked out of `perception/providers/`.
    """

    def __init__(self, *, calibrated=True, inference_error=None, objects=None):
        self.calibrated = calibrated
        self.inference_error = inference_error
        self.objects = objects

    def process_frame(self, _frame):
        if self.inference_error:
            raise RuntimeError(self.inference_error)
        return {
            "objects": self.objects if self.objects is not None else
                       [{"x": 10.0, "y": 20.0, "yaw": 5.0, "conf": 0.9}],
            "pick_pose": {},
            "calibration": {
                "status": "valid" if self.calibrated else "invalid",
                "revision": "abc123def456" if self.calibrated else "",
                "markers_in_frame": self.calibrated,
            },
        }


def _runtime(monkeypatch, clock, *, frame=_StubFrame(), calibrated=True,
             pipeline_error=None, inference_error=None, objects=None):
    """A DetectorRuntime with the camera, provider and clock replaced.

    No torch, no cv2, no camera and no model file: everything heavy is behind
    the three seams stubbed here, which is what makes the timestamp semantics
    testable in ordinary CI.
    """
    import perception_service as svc
    monkeypatch.setattr(svc, "utc_now_iso", lambda: next(clock))
    rt = svc.DetectorRuntime()

    def ensure_detector():
        if pipeline_error:
            raise RuntimeError(pipeline_error)
        rt._detector = _StubDetector(calibrated=calibrated,
                                     inference_error=inference_error,
                                     objects=objects)
        rt.model_loaded = True

    monkeypatch.setattr(rt, "_ensure_detector", ensure_detector)
    monkeypatch.setattr(rt, "frame", lambda timeout_s=3.0: frame)
    monkeypatch.setattr(rt, "jpeg", lambda _f: b"")
    return rt


#: A clock whose ticks are far enough apart to model a cold start: the request
#: at T0, the model load taking 30 s, the frame arriving at T0+30.
def _clock(*stamps):
    return iter(stamps)


def test_captured_at_is_the_frame_time_not_the_request_time(monkeypatch):
    """REGRESSION. `captured_at` used to be stamped before the model load.

    On a cold start the model load is ~30 s, so every first batch claimed to
    describe a scene half a minute older than the frame it actually analysed —
    and `captured_at` is exactly the instant staleness and the future Isaac
    synchronizer treat as "when the world looked like this".
    """
    rt = _runtime(monkeypatch, _clock("2026-08-08T10:00:00.000Z",   # requested
                                      "2026-08-08T10:00:30.000Z"))  # frame in hand
    batch = rt.detect()
    assert batch.ok and batch.count == 1
    assert batch.captured_at == "2026-08-08T10:00:30.000Z"   # the FRAME
    assert batch.requested_at == "2026-08-08T10:00:00.000Z"  # the REQUEST
    assert batch.captured_at != batch.requested_at
    # Every observation carries the frame time too, not the request time.
    assert all(o.captured_at == "2026-08-08T10:00:30.000Z"
               for o in batch.observations)
    # `last_inference_at` (the health field) follows the frame as well.
    assert rt.last_inference_at == "2026-08-08T10:00:30.000Z"


def test_captured_at_survives_the_json_round_trip(monkeypatch):
    """The batch reaches /api/perception as JSON; both stamps must arrive."""
    rt = _runtime(monkeypatch, _clock("2026-08-08T11:00:00.000Z",
                                      "2026-08-08T11:00:02.000Z"))
    original = rt.detect()
    revived = ObservationBatch.from_dict(
        json.loads(json.dumps(original.to_dict())))
    assert revived.captured_at == "2026-08-08T11:00:02.000Z"
    assert revived.requested_at == "2026-08-08T11:00:00.000Z"


def test_a_batch_that_never_reached_the_camera_has_no_capture_time(monkeypatch):
    """No frame acquired means NO capture time. Never the request time.

    Stamping the request time here would assert a measurement that never
    happened — and would make a failed scan look like a fresh observation to a
    staleness check.
    """
    no_model = _runtime(monkeypatch, _clock("2026-08-08T12:00:00.000Z"),
                        pipeline_error="weights are not available")
    b = no_model.detect()
    assert b.status is BatchStatus.ERROR
    assert b.captured_at == ""
    assert b.requested_at == "2026-08-08T12:00:00.000Z"

    no_frame = _runtime(monkeypatch, _clock("2026-08-08T12:05:00.000Z"),
                        frame=None)
    b2 = no_frame.detect()
    assert b2.status is BatchStatus.ERROR
    assert b2.captured_at == ""
    assert b2.requested_at == "2026-08-08T12:05:00.000Z"

    # An unstamped batch is never reported stale — it is reported FAILED, which
    # is the accurate thing to say about it.
    for failed in (b, b2):
        assert is_stale(failed) is False


def test_a_failure_after_capture_keeps_the_capture_time(monkeypatch):
    """Inference failed, but a frame WAS acquired — that instant is real."""
    rt = _runtime(monkeypatch, _clock("2026-08-08T13:00:00.000Z",
                                      "2026-08-08T13:00:01.000Z"),
                  inference_error="CUDA out of memory")
    b = rt.detect()
    assert b.status is BatchStatus.ERROR
    assert "CUDA out of memory" in b.error
    assert b.captured_at == "2026-08-08T13:00:01.000Z"
    assert b.requested_at == "2026-08-08T13:00:00.000Z"


def test_staleness_is_measured_from_the_frame_not_the_request(monkeypatch):
    """The point of the fix: a 30 s cold start must not age the observation.

    With the old behaviour this batch was 330 s old against a 300 s TTL and
    rendered STALE the instant it was produced.
    """
    import calendar
    rt = _runtime(monkeypatch, _clock("2026-08-08T14:00:00.000Z",    # requested
                                      "2026-08-08T14:00:30.000Z"))   # frame
    batch = rt.detect()
    frame_epoch = float(calendar.timegm((2026, 8, 8, 14, 0, 30, 0, 0, 0)))
    from wisepack_core.perception import observation_age_s
    assert observation_age_s(batch, now_epoch=frame_epoch) == 0.0
    # 300 s after the FRAME is the TTL boundary; 300 s after the REQUEST is not.
    assert not is_stale(batch, ttl_s=300.0, now_epoch=frame_epoch + 299)
    assert is_stale(batch, ttl_s=300.0, now_epoch=frame_epoch + 301)


def test_requested_at_defaults_empty_and_is_optional():
    """Producers that have no request time (a bare transport document) still work."""
    batch = parse_detection_json(
        json.dumps({"bottles": [{"x": 1.0, "y": 2.0, "conf": 0.9}]}),
        batch_id="b", captured_at="2026-08-08T15:00:00.000Z")
    assert batch.requested_at == ""
    assert batch.captured_at == "2026-08-08T15:00:00.000Z"
    assert ObservationBatch(batch_id="b").requested_at == ""


def test_an_unstamped_batch_is_not_reported_stale():
    """Unknown age must not render as "stale" — that would be an invention."""
    batch = observations_from_detections(DETECTOR_RESULT, batch_id="b", captured_at="")
    assert not is_stale(batch)


# --------------------------------------------------------------------------- #
# 7. KPI honesty (§12)
# --------------------------------------------------------------------------- #


def test_detector_confidence_never_becomes_a_detection_rate():
    """0.94 confidence must NOT surface as "vision detection rate = 94%"."""
    engine = _engine(perception_source=PerceptionSource.CAMERA)
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
        perception_source=PerceptionSource.CAMERA
    ).execution_backend is ExecutionBackend.SIMULATED


def test_selecting_isaac_does_not_change_the_perception_source():
    assert WorkflowConfig(
        execution_backend=ExecutionBackend.ISAAC
    ).perception_source is PerceptionSource.SIM


def test_the_engine_reports_both_axes_separately():
    engine = _engine(perception_source=PerceptionSource.CAMERA,
                     execution_backend=ExecutionBackend.ISAAC)
    state = engine.perception_state()
    assert state["perception_source"] == "camera"
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
                                cache_dir="/cache", exists=lambda p: True)
    assert False is result.available or result.origin == "configured"


def test_configured_missing_path_reports_itself():
    result = resolve_model_path(configured="/nowhere/weights.pth",
                                cache_dir="/cache", exists=lambda p: False)
    assert not result.available
    assert "/nowhere/weights.pth" in result.message
    assert result.searched == ["/nowhere/weights.pth"]


def test_model_resolution_falls_back_to_the_shared_arise_copy():
    result = resolve_model_path(configured="", cache_dir="/cache",
                                env={}, exists=lambda p: p == ARISE_MODEL_PATH)
    assert result.path == ARISE_MODEL_PATH
    assert result.origin == "arise_shared"


def test_model_resolution_falls_back_to_the_wisepack_cache():
    """The cache is WISEPACK's OWN — no foreign checkout is ever searched."""
    cached = "/cache/best_model.pth"
    result = resolve_model_path(configured="", cache_dir="/cache",
                                env={}, exists=lambda p: p == cached)
    assert result.path == cached and result.origin == "wisepack_cache"


def test_the_wisepack_cache_lives_inside_the_repository_by_default():
    """Disposable, git-ignored, and next to the perception environment."""
    import model_store
    cache = model_store.default_cache_dir(env={})
    assert cache == os.path.join(REPO, ".cache-perception", "models")
    gitignore = (pathlib.Path(REPO) / ".gitignore").read_text()
    assert ".cache-perception/" in gitignore
    assert ".venv-perception/" in gitignore


def test_absent_model_is_a_clear_diagnostic_not_a_pytorch_traceback():
    """§4: absence must name the file, the repo and the command to fix it."""
    result = resolve_model_path(configured="", cache_dir="/cache",
                                env={}, exists=lambda p: False)
    assert not result.available and result.path is None
    assert ARISE_MODEL_PATH in result.message
    assert HUGGINGFACE_REPO in result.message
    assert "curl" in result.message
    assert "setup_perception.sh" in result.message
    assert result.to_dict()["download_url"].endswith("best_model.pth")


def test_an_explicit_model_path_is_never_replaced_by_a_download(tmp_path,
                                                                monkeypatch):
    """A named file that is absent is a miss to report, not a reason to fetch."""
    import model_store
    downloads = []
    monkeypatch.setattr(model_store, "download_model",
                        lambda *a, **k: downloads.append(a) or "")
    result = model_store.ensure_model(
        configured=str(tmp_path / "nope.pth"),
        cache_dir=str(tmp_path / "cache"),
        env={"WISEPACK_PERCEPTION_MODEL_PATH": str(tmp_path / "nope.pth")},
        allow_download=True)
    assert not result.available
    assert downloads == [], "an explicit path was silently replaced by a download"


def test_the_download_can_be_switched_off_for_an_air_gapped_host(tmp_path):
    import model_store
    assert model_store.download_enabled(env={}) is True
    assert model_store.download_enabled(
        env={"WISEPACK_PERCEPTION_MODEL_DOWNLOAD": "0"}) is False
    result = model_store.ensure_model(
        configured="", cache_dir=str(tmp_path / "cache"),
        env={}, allow_download=False)
    assert not result.available or result.origin != "downloaded"


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
    # It is a plain JSON document — no detector-specific key survives into it.
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


def test_the_orchestrator_is_the_single_dds_writer_for_perception():
    """The host service publishes NO DDS; the container does.

    The perception service speaks HTTP only, because WISEPACK's validated
    middleware is the CONTAINERIZED Vulcanexus / Fast DDS runtime. Publishing
    from the host would have needed a second, unvalidated middleware install
    there. So the orchestrator — inside the container, on Vulcanexus — fetches
    over HTTP and owns both perception topics.
    """
    from wisepack_bringup import topics as T
    assert set(T.PERCEPTION_WRITERS) == {T.PERCEPTION_STATUS, T.PERCEPTION_OBJECTS}
    assert set(T.PERCEPTION_WRITERS.values()) == {"wisepack_orchestration"}
    assert not hasattr(T, "HARMONY_DETECTION_COMMAND"), (
        "the WISEPACK contract must not name a provider's own topic")


def test_detect_physical_objects_is_in_the_operator_vocabulary():
    from wisepack_bringup import topics as T
    assert "detect_physical_objects" in T.OPERATOR_COMMANDS


def test_the_host_service_contains_no_middleware():
    """No rclpy, no DDS, no ROS sourcing — see §10/§11.

    An earlier revision started a ROS 2 node in this host process and sourced a
    ROS environment for it. That required a host middleware installation and
    created a second, unvalidated path to the same NGSI-LD attributes. The
    service is HTTP-only now, which is also what lets `sim` + camera run with no
    ROS anywhere.
    """
    import ast
    service = pathlib.Path(REPO) / "perception" / "perception_service.py"
    tree = ast.parse(service.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    # AST, not a text scan: the file EXPLAINS at length why it has no
    # middleware, and a substring check could not tell the prohibition from the
    # thing prohibited.
    for forbidden in ("rclpy", "std_msgs", "rosidl_runtime_py", "ros2_backend"):
        assert forbidden not in imported, (
            f"the host perception service imports {forbidden!r} — it must not "
            "require or duplicate middleware on the host")

    # The launcher must not source a middleware environment for it either.
    lib = pathlib.Path(REPO) / "scripts" / "lib_perception_service.sh"
    code = "\n".join(line for line in lib.read_text().splitlines()
                      if not line.lstrip().startswith("#"))
    for forbidden in ("/opt/vulcanexus", "/opt/ros", "setup.bash"):
        assert forbidden not in code, (
            f"the perception launcher sources {forbidden!r} — the host must "
            "not need a middleware installation to run the camera")


# --------------------------------------------------------------------------- #
# 12. The detector service's own configuration (no camera, no torch)
# --------------------------------------------------------------------------- #


def test_service_config_honours_the_camera_override(monkeypatch):
    from perception_config import PerceptionConfig
    monkeypatch.setenv("WISEPACK_PERCEPTION_CAMERA", "4")
    assert PerceptionConfig.from_env().camera == 4
    # A device path or an RTSP URL stays a string; only a bare index is an int.
    monkeypatch.setenv("WISEPACK_PERCEPTION_CAMERA", "/dev/video2")
    assert PerceptionConfig.from_env().camera == "/dev/video2"
    monkeypatch.setenv("WISEPACK_PERCEPTION_CAMERA", "rtsp://cam.local/stream")
    assert PerceptionConfig.from_env().camera == "rtsp://cam.local/stream"


def test_service_config_takes_the_resolved_model_path():
    from perception_config import PerceptionConfig
    config = PerceptionConfig.from_env(model_path=ARISE_MODEL_PATH)
    assert config.model_path == ARISE_MODEL_PATH


def test_a_bad_perception_setting_is_reported_never_silently_defaulted(monkeypatch):
    """A typo that quietly restored the default would change what is measured."""
    from perception_config import PerceptionConfig, PerceptionConfigurationError
    for name, value in (("WISEPACK_PERCEPTION_WIDTH", "wide"),
                        ("WISEPACK_PERCEPTION_CONFIDENCE", "very"),
                        ("WISEPACK_PERCEPTION_CONFIDENCE", "5"),
                        ("WISEPACK_PERCEPTION_SET_RESOLUTION", "maybe"),
                        ("WISEPACK_PERCEPTION_CALIBRATION_EXTENT_MM", "big"),
                        ("WISEPACK_PERCEPTION_CALIBRATION_MARKERS", "11,ten")):
        monkeypatch.setenv(name, value)
        with pytest.raises(PerceptionConfigurationError):
            PerceptionConfig.from_env()
        monkeypatch.delenv(name)


def test_a_larger_calibration_board_is_configuration_not_code(monkeypatch):
    """§13: make the extent configurable rather than hardcode a new layout."""
    from perception_config import PerceptionConfig
    monkeypatch.setenv("WISEPACK_PERCEPTION_CALIBRATION_EXTENT_MM", "600")
    monkeypatch.setenv("WISEPACK_PERCEPTION_CALIBRATION_MARKERS", "11,10,15,16")
    monkeypatch.delenv("WISEPACK_PHYSICAL_WORKAREA_WIDTH_MM", raising=False)
    monkeypatch.delenv("WISEPACK_PHYSICAL_WORKAREA_DEPTH_MM", raising=False)
    config = PerceptionConfig.from_env()
    assert config.board.corners_mm == ((0.0, 0.0), (600.0, 0.0),
                                       (600.0, 600.0), (0.0, 600.0))
    assert config.board.marker_ids == (11, 10, 15, 16)
    frame = config.work_area()
    assert (frame.width_mm, frame.depth_mm) == (600, 600)


def test_the_work_area_frame_follows_the_calibrated_plane(monkeypatch):
    """The measured plane IS the work-area frame — no transform of our own."""
    from perception_config import PerceptionConfig
    for name in ("WISEPACK_PERCEPTION_CALIBRATION_EXTENT_MM",
                 "WISEPACK_PHYSICAL_WORKAREA_WIDTH_MM",
                 "WISEPACK_PHYSICAL_WORKAREA_DEPTH_MM"):
        monkeypatch.delenv(name, raising=False)
    frame = PerceptionConfig.from_env().work_area()
    assert (frame.width_mm, frame.depth_mm) == (130, 130)
    assert frame.frame_id == "wisepack_workarea"


def test_a_calibration_board_must_pair_every_marker_with_a_coordinate():
    from perception_config import CalibrationBoard, PerceptionConfigurationError
    with pytest.raises(PerceptionConfigurationError):
        CalibrationBoard(marker_ids=(1, 2, 3), corners_mm=((0, 0), (1, 0),
                                                           (1, 1), (0, 1)))
    with pytest.raises(PerceptionConfigurationError):
        CalibrationBoard.square(130, (11, 11, 15, 16))     # duplicate ids


# --------------------------------------------------------------------------- #
# 13. Dashboard client and API serialisation
# --------------------------------------------------------------------------- #


class _StubClient:
    """A perception service that is not there. §15's "detector unavailable"."""

    def __init__(self, url="http://127.0.0.1:22101"):
        self.url = url

    def _request(self, path, method="GET", timeout_s=None):
        return None, None, f"perception service unreachable at {self.url}"


def test_client_reports_an_unreachable_service_without_raising():
    from wisepack_core.perception_client import PerceptionClient
    client = PerceptionClient(url="http://127.0.0.1:1")
    client._request = _StubClient()._request

    health = client.health()
    assert health["service_reachable"] is False
    # TRI-STATE: unknown is not the same as "no camera".
    assert health["camera_available"] is None
    assert "unreachable" in health["last_error"]
    assert "perception/perception_service.py" in health["note"]

    batch = client.detect()
    assert batch.status is BatchStatus.ERROR
    assert client.last_detection() is None
    image, error = client.image("annotated")
    assert image is None and error


def test_client_rejects_an_unreadable_document_rather_than_guessing():
    from wisepack_core.perception_client import PerceptionClient
    client = PerceptionClient()
    client._request = lambda *a, **k: (200, {"batch_id": 1, "status": "nonsense"}, "")
    batch = client.detect()
    assert batch.status is BatchStatus.ERROR
    assert "cannot read" in batch.error


def test_client_parses_a_real_service_response():
    from wisepack_core.perception_client import PerceptionClient
    client = PerceptionClient()
    document = _batch().to_dict()
    client._request = lambda *a, **k: (200, document, "")
    batch = client.detect()
    assert batch.count == 3
    assert batch.observations[0].x_mm == 82.4


def test_perception_state_serialises_for_the_api():
    """§16: the dashboard payload must be JSON and must carry the poses."""
    engine = _engine(perception_source=PerceptionSource.CAMERA)
    engine.observation_provider = _batch
    engine.generate_or_load_scenario()
    engine.scan_and_detect()

    payload = json.loads(json.dumps(engine.perception_state()))
    assert payload["perception_source"] == "camera"
    assert payload["perception_source_label"] == "PHYSICAL CAMERA"
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

    camera = {i: s for i, s, _ in diagnostics.interfaces("camera")}
    assert camera["2-D camera frames"] == "live"
    assert camera["Object detections"] == "measured source"

    # The PROPOSAL's RGB-D pipeline is still not implemented, and a 2-D camera
    # does not make it so. Overclaiming here is the failure this table prevents.
    for table in (sim, camera):
        assert table["RGB-D camera frames"] == "future interface"
        assert table["Robot joint states"] == "simulated source"

    meanings = {i: m for i, s, m in diagnostics.interfaces("camera")}
    assert "NOT a measured detection rate" in meanings["Object detections"]
    assert "PLANAR ONLY" in meanings["6D pose estimates"]


def test_a_configured_but_dead_detector_is_not_reported_as_live():
    """CONFIGURED is not LIVE. Selecting a camera does not make one answer."""
    import diagnostics

    dead = {i: s for i, s, _ in
            diagnostics.interfaces("camera", reachable=False)}
    assert dead["2-D camera frames"] == "configured, unavailable"
    assert dead["Object detections"] == "configured, unavailable"
    # Still not a FAILURE state — the table's other rule.
    for state in dead.values():
        assert "error" not in state.lower() and "fail" not in state.lower()

    # Unchecked reachability makes neither claim.
    unknown = {i: s for i, s, _ in diagnostics.interfaces("camera")}
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
    # The acquisition control. Named for the ACTION, never for the object: it
    # says "Detect & plan", not "Detect bottles".
    assert 'id="c-detect-physical"' in html
    assert "Detect &amp; plan" in html
    # THE PROXY DISCLOSURE IS SERVED FROM THE API, not hard-coded here, so the
    # markup never has to name the physical stand-in object at all. That is what
    # keeps the dashboard from becoming detector-specific (§18).
    assert "bottle" not in html.lower()


# --------------------------------------------------------------------------- #
# 14. WISEPACK owns perception; a provider is an implementation detail
# --------------------------------------------------------------------------- #


def test_the_public_perception_sources_are_sim_and_camera():
    """The source answers WHERE observations come from — never which AI ran."""
    assert [s.value for s in PerceptionSource] == ["sim", "camera"]
    assert PerceptionSource.CAMERA.label == "PHYSICAL CAMERA"
    # No provider, vendor or model name in the public vocabulary.
    for source in PerceptionSource:
        blob = f"{source.value} {source.label} {source.detail}".lower()
        for forbidden in ("harmony", "bottle", "faster", "rcnn", "yolo"):
            assert forbidden not in blob, (
                f"{source.value!r} exposes {forbidden!r} as architecture")


def test_provider_selection_is_a_separate_axis_from_the_source():
    """§5: adding a second camera method must not need a second source."""
    assert DEFAULT_DETECTOR == "fasterrcnn_bottle"
    assert DEFAULT_DETECTOR in KNOWN_DETECTORS
    assert resolve_detector() == DEFAULT_DETECTOR
    # The two are resolved independently, from different variables.
    from wisepack_core.perception import (PERCEPTION_DETECTOR_ENV,
                                          PERCEPTION_SOURCE_ENV)
    assert PERCEPTION_DETECTOR_ENV != PERCEPTION_SOURCE_ENV
    with pytest.raises(PerceptionConfigError):
        resolve_detector("no_such_provider")


def test_the_core_contains_no_detector_specific_module():
    """§3: detector-specific parsing lives in the provider, not in the domain."""
    core = pathlib.Path(REPO) / "wisepack_ws" / "src" / "wisepack_core" / "wisepack_core"
    assert not (core / "harmony_adapter.py").exists(), (
        "detector-specific parsing must not live in the domain package")
    provider = pathlib.Path(REPO) / "perception" / "providers" / "fasterrcnn_bottle.py"
    assert provider.exists()
    assert "UNCALIBRATED_SENTINEL" in provider.read_text(), (
        "the provider must own the detector's quirks")


@pytest.mark.parametrize("module", [
    "packing.py", "workflow.py", "validator.py", "kpi.py", "execution.py",
    "domain.py", "isaac_contract.py", "isaac_transform.py", "whole_process.py",
])
def test_no_planning_or_execution_module_mentions_a_detector(module):
    """§14: no bottle- or provider-specific dependency downstream.

    EXECUTABLE CODE ONLY — comments and docstrings are stripped. `domain.py`
    states in prose that nothing there names a bottle or a detector, and a check
    that could not tell a prohibition from the thing prohibited would forbid
    writing the rule down.
    """
    path = (pathlib.Path(REPO) / "wisepack_ws" / "src" / "wisepack_core"
            / "wisepack_core" / module)
    code = _code_without_prose(path.read_text())
    for forbidden in ("harmony", "bottle", "fasterrcnn", "faster_rcnn"):
        assert forbidden not in code.lower(), (
            f"{module} references {forbidden!r} — planning, workflow and "
            "execution must stay domain-neutral")


def test_the_operator_facing_ui_shows_no_provider_architecture_label():
    """§13: a provider name must not be reachable as a source label."""
    html = (pathlib.Path(REPO) / "web" / "index.html").read_text()
    assert "harmony" not in html.lower()
    # The panel names the METHOD, from the service's health document.
    assert "detector_display_name" in html
    assert "Physical Perception" in html


def test_provenance_is_available_but_is_not_the_architecture():
    """§13: implementation origin belongs in diagnostics, not in the label."""
    from providers import fasterrcnn_bottle as provider
    assert provider.IMPLEMENTATION_ORIGIN == "HARMONY"   # provenance, kept
    assert provider.PROVIDER_NAME == "fasterrcnn_bottle"  # architecture, generic
    assert provider.DISPLAY_NAME == "Faster R-CNN"        # method, operator-facing
    # The observation's own provenance names the METHOD, not the vendor.
    assert "harmony" not in DETECTOR_ID.lower()
    assert _batch().observations[0].detector == DETECTOR_ID


def test_camera_perception_is_independent_of_the_dashboard_data_source():
    """§17.4: --source and the perception source are unrelated axes."""
    text = (pathlib.Path(REPO) / "web" / "app.py").read_text()
    # The dashboard resolves them from different variables, neither derived
    # from the other.
    assert 'SOURCE = os.environ.get("WISEPACK_SOURCE", "sim")' in text
    assert "PERCEPTION_SOURCE = resolve_perception_source()" in text
    assert "--perception-source" in text and "--source" in text
    # Neither is derived from the other: no line both tests the data source and
    # assigns the perception source, or vice versa.
    for line in text.splitlines():
        if "PERCEPTION_SOURCE" in line and "resolve_perception_source" not in line:
            assert not line.lstrip().startswith("SOURCE ="), line
        if line.lstrip().startswith("PERCEPTION_SOURCE ="):
            assert "WISEPACK_SOURCE" not in line, line


def test_the_containerized_vulcanexus_stack_is_untouched_by_perception():
    """§10/§22: perception must not weaken or replace the validated DDS path."""
    from wisepack_bringup import topics as T
    # The always-present contract is unchanged: perception is an optional
    # channel and adds nothing to the topics every run must publish.
    assert T.PERCEPTION_OBJECTS not in T.all_topics()
    assert T.PERCEPTION_STATUS not in T.all_topics()
    # Both perception topics remain bridgeable scalar std_msgs.
    assert set(T.perception_topics().values()) == {"std_msgs/String"}
    assert T.reserved_leaf_violations() == []

    launcher = (pathlib.Path(REPO) / "run_wisepack_dashboard.sh").read_text()
    # The container still sources the CONTAINERIZED Vulcanexus runtime.
    assert "source /opt/vulcanexus/jazzy/setup.bash" in launcher
    # ...and the host-side perception block installs nothing.
    assert "apt install" not in launcher.lower()
