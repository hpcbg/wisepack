"""Stale FIWARE state, and the scene handshake — both manual-validation defects.

DEFECT 1, in `./run_wisepack_dashboard.sh fiware`. Approval showed correctly for
a few seconds, then the UI returned to WAIT_FOR_OPERATOR_APPROVAL; the Digital
Twin showed mixed_pipes_dense with 3 baseline and 2 optimized containers while
the KPI cards read 1 and 1. Orion-LD holds CURRENT STATE, not a log: every
attribute keeps its last value until something overwrites it, so KPI scalars
left by an earlier isaac_cylinders_smoke run were being read back and merged
into the running scenario's view, and a stale `stage` attribute was rewinding the
workflow on screen.

DEFECT 3, in `./run_wisepack_dashboard.sh isaac-fiware`. Isaac was up with a
visibly correct scene — Panda, four cylinders, empty containers — and the
dashboard said "the physical scene has not been rebuilt for this scenario yet"
with Approve disabled. Two causes, both closed here: the initial scene was
trusted inside `open_run()`, which only runs after approval while approval waits
on the scene gate; and the run gate discarded the simulator's READY before any
run existed, so `simulator_ready` never became true.

Nothing here needs Orion-LD, ROS, Isaac Sim, a GPU or a browser.
"""

from __future__ import annotations

import os

import pytest

from wisepack_core.correlation import (
    CORRELATION_FACETS, RunCorrelation, describe_mismatch,
)
from wisepack_core.generator import build_scenario
from wisepack_core.isaac_contract import (
    IsaacCommandType, IsaacFeedback, IsaacState, SceneAcknowledgement,
)
from wisepack_core.isaac_transform import scene_fingerprint

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# The correlation payload
# --------------------------------------------------------------------------- #

ACTIVE = RunCorrelation(run_id="run-dense", scenario_id="mixed_pipes_dense-s42",
                        scenario_revision=2, plan_id="plan-optimized-dense",
                        plan_revision=2, approval_revision=2, sequence=10)


def test_a_matching_stamp_matches():
    same = RunCorrelation.from_dict(ACTIVE.to_dict())
    assert same is not None and same.matches(ACTIVE)
    assert same.mismatches(ACTIVE) == {}


def test_the_observed_stale_run_is_rejected():
    """isaac_cylinders_smoke projections beside a mixed_pipes_dense run."""
    stale = RunCorrelation(run_id="run-smoke",
                           scenario_id="isaac_cylinders_smoke-s42",
                           scenario_revision=1)
    mismatches = stale.mismatches(ACTIVE)
    assert set(mismatches) == {"run_id", "scenario_id", "scenario_revision"}
    text = describe_mismatch(mismatches)
    assert "run-smoke" in text and "run-dense" in text


def test_an_absent_facet_is_unknown_not_matching():
    """A projection that makes no claim about a facet is not asserting a match.

    Skipping it is the conservative choice for a rolling upgrade — an older
    publisher that stamps nothing is unknown, not wrong — and the caller decides
    what an entirely unstamped projection means.
    """
    partial = RunCorrelation(run_id="run-dense")
    assert partial.matches(ACTIVE)
    assert not partial.is_unstamped
    assert RunCorrelation().is_unstamped


def test_a_non_correlation_document_parses_to_none():
    for junk in (None, "", 42, [], {"hello": "world"},
                 {"schema_version": "something-else/1.0"}):
        assert RunCorrelation.from_dict(junk) is None


def test_every_declared_facet_round_trips():
    parsed = RunCorrelation.from_dict(ACTIVE.to_dict())
    for facet in CORRELATION_FACETS:
        assert getattr(parsed, facet) == getattr(ACTIVE, facet)


# --------------------------------------------------------------------------- #
# The snapshot merge
# --------------------------------------------------------------------------- #

class _FakeState:
    """Just enough of the dashboard state object for the ROS base snapshot."""

    def __init__(self, mirror):
        import threading
        self.lock = threading.Lock()
        self.ros_mirror = mirror
        self.events = []
        self.notice = ""
        self.fiware_connected = True
        self.fiware_last_error = ""


def _ros_mirror(run_id="run-dense", revision=2,
                scenario_id="mixed_pipes_dense-s42"):
    """A canonical ROS view of the dense scenario: 3 baseline, 2 optimized."""
    stamp = {"run_id": run_id, "scenario_revision": revision,
             "scenario_id": scenario_id}
    return {
        "stage": "WAIT_FOR_OPERATOR_APPROVAL",
        "run_id": run_id,
        "scenario": {"scenario_id": scenario_id, "items": [], "totals": {}},
        "selected": {"plan_id": "plan-optimized-dense",
                     "approval_state": "pending", "is_valid": True},
        "plan_summary": {"selected_plan_id": "plan-optimized-dense",
                         "scenario_revision": revision,
                         "approval_revision": revision,
                         "approval_plan_id": "plan-optimized-dense"},
        "kpi": {"containers_baseline": 3, "containers_optimized": 2},
        "stamps": {"scenario": stamp, "selected": stamp, "plan_summary": stamp},
    }


def _provider(entities, mirror=None):
    from web.snapshot import FiwareSnapshotProvider
    return FiwareSnapshotProvider(_FakeState(mirror or _ros_mirror()),
                                  lambda: entities)


def _stamp(run_id="run-dense", revision=2,
           scenario_id="mixed_pipes_dense-s42", sequence=10):
    return RunCorrelation(run_id=run_id, scenario_id=scenario_id,
                          scenario_revision=revision,
                          sequence=sequence).to_dict()


def test_a_current_run_fiware_view_is_applied():
    snap = _provider({
        "system": {"stage": "PICK_ITEM", "runCorrelation": _stamp()},
        "kpi": {"containersBaseline": 3, "containersOptimized": 2,
                "runCorrelation": _stamp()},
    }).snapshot()
    assert snap.fiware_sync_status == "synchronized"
    assert snap.stage == "PICK_ITEM"
    assert snap.rejected_stale_fields == []


def test_a_stale_fiware_system_state_cannot_rewind_the_workflow():
    """current ROS: approved/executing; stale FIWARE: WAIT_FOR_OPERATOR_APPROVAL."""
    mirror = _ros_mirror()
    mirror["stage"] = "PICK_ITEM"
    mirror["selected"]["approval_state"] = "approved"
    snap = _provider({
        "system": {"stage": "WAIT_FOR_OPERATOR_APPROVAL",
                   "runCorrelation": _stamp(run_id="run-smoke", revision=1,
                                            scenario_id="isaac_cylinders_smoke-s42")},
    }, mirror).snapshot()
    assert snap.stage == "PICK_ITEM", "FIWARE must not rewind the mission workflow"
    assert snap.fiware_stage == "WAIT_FOR_OPERATOR_APPROVAL"
    assert snap.fiware_sync_status == "stale"
    assert any(r["entity"] == "system" for r in snap.rejected_stale_fields)


def test_the_observed_stale_kpi_entity_cannot_populate_the_dense_scenario():
    """ROS says 3 baseline / 2 optimized; a smoke-run KPI entity says 1 / 1."""
    snap = _provider({
        "kpi": {"containersBaseline": 1, "containersOptimized": 1,
                "runCorrelation": _stamp(run_id="run-smoke", revision=1,
                                         scenario_id="isaac_cylinders_smoke-s42")},
    }).snapshot()
    assert snap.kpis["containers_baseline"]["value"] == 3
    assert snap.kpis["containers_optimized"]["value"] == 2
    assert "another run" in snap.panel_sources["kpis"]
    assert any(r["entity"] == "kpi" for r in snap.rejected_stale_fields)


def test_stale_fiware_never_reaches_the_operator_controls():
    """Approve, Reject, Resume and Step are enabled from ROS/DDS and nothing else."""
    snap = _provider({
        "system": {"stage": "COMPLETE", "readiness": False,
                   "runCorrelation": _stamp(run_id="run-old", revision=0)},
    }).snapshot()
    control = snap.control_state()
    assert control["stage"] == "WAIT_FOR_OPERATOR_APPROVAL"
    assert control["source"] == "ros"
    assert control["approval_state"] == "pending"
    assert snap.can_approve is True


def test_an_out_of_order_update_is_rejected():
    """NGSI-LD gives no ordering guarantee, and a delayed DDS sample can land
    after a newer one. Re-applying it is how a panel flickers backwards."""
    mirror = _ros_mirror()
    mirror["stage"] = "NEXT_ITEM"          # distinguishable from either sample
    entities = {"system": {"stage": "PLACE_ITEM",
                           "runCorrelation": _stamp(sequence=10)}}
    provider = _provider(entities, mirror)
    assert provider.snapshot().stage == "PLACE_ITEM"

    # A delayed sample from earlier in the SAME run arrives afterwards.
    entities["system"] = {"stage": "WAIT_FOR_OPERATOR_APPROVAL",
                          "runCorrelation": _stamp(sequence=4)}
    snap = provider.snapshot()
    assert snap.stage == "NEXT_ITEM", (
        "the late sample must be dropped, leaving the canonical ROS stage")
    assert any("out-of-order" in r["reason"] for r in snap.rejected_stale_fields)


def test_partial_synchronization_is_its_own_state():
    """Some entities current, others not — neither 'connected' nor 'stale'."""
    snap = _provider({
        "system": {"stage": "PICK_ITEM", "runCorrelation": _stamp()},
        "kpi": {"containersBaseline": 1, "runCorrelation": _stamp(
            run_id="run-smoke", revision=1)},
    }).snapshot()
    assert snap.fiware_sync_status == "partial"
    assert "awaiting current-run synchronization" in snap.fiware_sync_detail
    assert snap.stage == "PICK_ITEM"
    assert snap.kpis["containers_baseline"]["value"] == 3


def test_a_reachable_but_wholly_stale_broker_says_so():
    snap = _provider({
        "system": {"stage": "COMPLETE", "runCorrelation": _stamp(run_id="old")},
        "kpi": {"containersBaseline": 1, "runCorrelation": _stamp(run_id="old")},
    }).snapshot()
    assert snap.fiware_connected is True
    assert snap.fiware_sync_status == "stale"
    assert "FIWARE reachable — awaiting current-run synchronization" \
        in snap.fiware_sync_detail


def test_an_unstamped_entity_is_applied_and_reported_as_unverified():
    """A pre-correlation orchestrator must not blank every panel."""
    snap = _provider({
        "system": {"stage": "PICK_ITEM"},
        "kpi": {"containersBaseline": 3, "containersOptimized": 2},
    }).snapshot()
    assert snap.stage == "PICK_ITEM"
    assert snap.rejected_stale_fields == []
    assert snap.fiware_sync_status == "synchronized"


def test_no_mixed_run_snapshot_is_ever_produced():
    """The union of what is displayed must describe exactly one run."""
    snap = _provider({
        "system": {"stage": "PICK_ITEM", "runCorrelation": _stamp()},
        "kpi": {"containersBaseline": 1, "containersOptimized": 1,
                "runCorrelation": _stamp(run_id="run-smoke", revision=1)},
        "plan": {"summary": {"selected_plan_id": "plan-from-smoke"},
                 "runCorrelation": _stamp(run_id="run-smoke", revision=1)},
    }).snapshot()
    # Nothing from run-smoke reached the view.
    assert snap.selected_plan_id == "plan-optimized-dense"
    assert snap.kpis["containers_baseline"]["value"] == 3
    assert snap.kpis["containers_optimized"]["value"] == 2


def test_an_unreachable_broker_is_distinguished_from_a_stale_one():
    def boom():
        raise OSError("connection refused")
    from web.snapshot import FiwareSnapshotProvider
    snap = FiwareSnapshotProvider(_FakeState(_ros_mirror()), boom).snapshot()
    assert snap.fiware_connected is False
    assert snap.fiware_sync_status == "disconnected"


def test_the_diagnostics_expose_both_sides_of_the_correlation():
    snap = _provider({
        "system": {"stage": "COMPLETE",
                   "runCorrelation": _stamp(run_id="run-smoke", revision=1)},
    }).snapshot()
    payload = snap.to_state()["fiware"]
    for key in ("sync_status", "sync_detail", "run_id", "scenario_revision",
                "stage", "rejected_stale_fields"):
        assert key in payload, f"diagnostics need {key}"
    assert payload["run_id"] == "run-smoke"
    assert snap.run_id == "run-dense"


def test_the_diagnostics_page_reports_the_required_correlation_rows():
    src = _read(os.path.join(REPO, "web", "diagnostics.py"))
    for row in ("canonical_run_id", "fiware_run_id",
                "canonical_scenario_revision", "fiware_scenario_revision",
                "canonical_stage", "fiware_stage", "fiware_sync_status",
                "rejected_stale_fields"):
        assert row in src, f"the Diagnostics page must show {row}"


def test_every_stamped_entity_is_published_by_the_orchestrator():
    """A projection cannot gain state without gaining a stamp."""
    topics = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_bringup",
                                "wisepack_bringup", "topics.py"))
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "hitl_orchestrator.py"))
    yaml_src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_fiware",
                                  "config", "bridge_config.yaml"))
    for name in ("system", "scenario", "plan", "kpi", "robot", "actions",
                 "anomaly", "inventory", "cutting"):
        assert f'"{name}": CORRELATION_' in topics, \
            f"{name} has no correlation topic"
        assert f"/wisepack/correlation/{name}" in yaml_src, \
            f"the {name} correlation is not bridged into FIWARE"
        assert f'"{name}"' in src, f"nothing publishes the {name} correlation"
    assert yaml_src.count("runCorrelation") == 9


def test_a_projection_is_stamped_only_with_facets_it_claims():
    """Caught by LIVE validation, and it was a false positive generator.

    The scenario entity was stamped with `approval_revision`, which it has no
    relationship to. It is published before the approval gate is entered, so it
    carried revision 0; the moment `request_approval()` advanced the approval to
    1 the whole scenario projection was judged stale and withheld — while its
    contents were exactly current.
    """
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "hitl_orchestrator.py"))
    table = src[src.index("_CORRELATION_FACETS = {"):]
    table = table[:table.index("}", table.index("\"kpi\""))]
    # Only the plan digest and the system stage describe an approval.
    assert '"plan": ("plan_id", "plan_revision", "approval_revision")' in table
    assert '"system": ("approval_revision",)' in table
    assert '"kpi": ("plan_id", "plan_revision")' in table
    assert '"scenario"' not in table, (
        "the scenario projection makes no claim about a plan or an approval")
    for entity in ("robot", "actions", "anomaly", "inventory", "cutting"):
        assert f'"{entity}"' not in table

    body = src[src.index("    def publish_correlation(self"):]
    body = body[:body.index("\n    def ", 10)]
    assert "doc[facet] = None" in body, (
        "facets outside the subset must be cleared, not merely undeclared")
    # run_id / scenario_id / scenario_revision are never cleared: they are what
    # distinguishes one run from another.
    for facet in ("run_id", "scenario_id", "scenario_revision"):
        assert f'"{facet}"' not in body.split("for facet in (")[1].split(")")[0]


def test_the_stamp_is_published_after_the_values_it_stamps():
    """The ordering IS the guarantee — see wisepack_core/correlation.py.

    Correlation last: a reader polling mid-update sees the OLD stamp beside some
    new values and withholds the entity. Correlation first: it sees the NEW
    stamp beside some OLD values and trusts them — the mixed-run dashboard,
    reintroduced by the fix meant to prevent it.
    """
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "hitl_orchestrator.py"))
    for method, value_call in (("publish_kpis", "self.p_kpi[topic_key].publish"),
                               ("publish_state", "self.p_backend.publish"),
                               ("publish_scenario", "self.p_items.publish")):
        body = src[src.index(f"def {method}(self)"):]
        body = body[:body.index("\n    def ", 10)]
        assert body.index(value_call) < body.index("publish_correlation"), (
            f"{method} stamps before it publishes — a reader polling in between "
            "would trust stale values")


# --------------------------------------------------------------------------- #
# The scene handshake
# --------------------------------------------------------------------------- #

def test_the_fingerprint_is_deterministic_and_scenario_specific():
    a = build_scenario("isaac_cylinders_smoke", seed=42)
    b = build_scenario("isaac_cylinders_smoke", seed=42)
    c = build_scenario("isaac_cylinders_smoke", seed=43)
    assert scene_fingerprint(a) == scene_fingerprint(b)
    assert scene_fingerprint(a) != scene_fingerprint(c)


def test_the_fingerprint_covers_geometry_not_only_ids():
    """Ids can survive a change the plan is written against."""
    scenario = build_scenario("isaac_cylinders_smoke", seed=42)
    before = scene_fingerprint(scenario)
    scenario.items[0].outer_diameter_mm += 5.0
    assert scene_fingerprint(scenario) != before


def test_simulator_ready_and_scene_ready_are_distinct_states():
    assert IsaacState.SIMULATOR_READY.value == "SIMULATOR_READY"
    assert IsaacState.SCENE_READY.value == "SCENE_READY"
    from wisepack_core.execution import stage_for_isaac_state
    # Neither authorises anything by moving the workflow on.
    assert stage_for_isaac_state(IsaacState.SIMULATOR_READY) is None
    assert stage_for_isaac_state(IsaacState.SCENE_READY) is None


def test_the_sync_command_exists_and_is_distinct_from_a_reset():
    assert IsaacCommandType.SYNC_SCENE.value == "SYNC_SCENE"
    assert IsaacCommandType.SYNC_SCENE is not IsaacCommandType.RESET_SCENE


def _acknowledgement(**overrides):
    scenario = build_scenario("isaac_cylinders_smoke", seed=42)
    base = dict(run_id="run-1", scenario_id=scenario.scenario_id,
                scenario_revision=2, preset="isaac_cylinders_smoke", seed=42,
                scene_fingerprint=scene_fingerprint(scenario),
                object_ids=[i.item_id for i in scenario.items],
                object_count=len(scenario.items),
                robot_home_verified=True, container_empty_verified=True)
    base.update(overrides)
    return SceneAcknowledgement(**base), scenario


def _expected(scenario, **overrides):
    base = dict(run_id="run-1", scenario_id=scenario.scenario_id, revision=2,
                preset="isaac_cylinders_smoke", seed=42,
                fingerprint=scene_fingerprint(scenario),
                object_count=len(scenario.items))
    base.update(overrides)
    return base


def test_a_correlated_acknowledgement_is_accepted():
    ack, scenario = _acknowledgement()
    assert ack.mismatches(**_expected(scenario)) == []


def test_the_acknowledgement_survives_the_wire():
    ack, scenario = _acknowledgement()
    feedback = IsaacFeedback.from_json(IsaacFeedback(
        state=IsaacState.SCENE_READY, run_id="run-1", scenario_revision=2,
        scene=ack).to_json())
    assert feedback.scene is not None
    assert feedback.scene.scene_fingerprint == ack.scene_fingerprint
    assert feedback.scene.object_count == len(scenario.items)
    assert feedback.scene.mismatches(**_expected(scenario)) == []


@pytest.mark.parametrize("overrides,expected_word", [
    ({"scenario_revision": 1}, "revision"),
    ({"run_id": "run-previous"}, "run"),
    ({"scenario_id": "mixed_pipes_dense-s42"}, "scenario"),
    ({"preset": "mixed_pipes_dense"}, "preset"),
    ({"seed": 7}, "seed"),
    ({"scene_fingerprint": "0" * 64}, "fingerprint"),
    ({"object_count": 3}, "object"),
    ({"robot_home_verified": False}, "home"),
    ({"container_empty_verified": False}, "container"),
])
def test_every_way_an_acknowledgement_can_be_wrong_is_named(overrides,
                                                            expected_word):
    """Each is reported by name. "scene not ready" is not actionable."""
    ack, scenario = _acknowledgement(**overrides)
    reasons = ack.mismatches(**_expected(scenario))
    assert reasons, f"{overrides} must be rejected"
    assert any(expected_word in r for r in reasons), reasons


def test_a_stale_scene_ready_from_an_old_revision_is_rejected():
    ack, scenario = _acknowledgement(scenario_revision=1)
    reasons = ack.mismatches(**_expected(scenario, revision=2))
    assert any("revision" in r for r in reasons)


def test_the_initial_handshake_runs_before_approval_not_after():
    """THE DEADLOCK. `open_run` executes inside the execution loop, which runs
    after approval — while approval waits on the scene gate. Requesting the
    scene from the scenario-setup path is what breaks the cycle."""
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "hitl_orchestrator.py"))
    behaviour = src[src.index("class GenerateOrLoadScenario"):
                    src.index("class ScanAndDetect")]
    assert "sync_physical_scene()" in behaviour, (
        "the scene handshake must start when the scenario and run_id exist, "
        "not after approval")
    assert "def sync_physical_scene" in src


def test_the_initial_scene_is_no_longer_trusted():
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "isaac_bridge.py"))
    open_run = src[src.index("    def open_run(self, engine)"):
                   src.index("    def abort_run")]
    assert "self.scene_revision = self.required_revision" not in open_run, (
        "opening a run must not declare the scene ready — only a correlated "
        "SCENE_READY may do that")


def test_simulator_readiness_is_not_discarded_before_a_run_exists():
    """The other half of the deadlock: READY was dropped by the run gate."""
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "isaac_bridge.py"))
    handler = src[src.index("    def _on_feedback"):src.index("    def _apply")]
    assert "lifecycle" in handler
    assert "if not lifecycle:" in handler, (
        "readiness and scene lifecycle must bypass the execution run gate")


def test_readiness_triggers_the_handshake():
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "isaac_bridge.py"))
    assert "_sync_scene_if_needed" in src
    body = src[src.index("    def _sync_scene_if_needed"):]
    body = body[:body.index("\n    def ", 10)]
    # Idempotent per (run_id, revision), or a repeated latched READY restarts
    # the handshake and resets its timeout clock forever.
    assert "self.scene_requested_for_run == engine.run_id" in body


def test_a_generic_simulator_ready_does_not_open_the_scene_gate():
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "isaac_bridge.py"))
    gate = src[src.index("    def scene_ready(self)"):
               src.index("    def scene_block_reason")]
    assert "simulator_ready" not in gate, (
        "the scene gate must not be satisfiable by simulator readiness")
    assert "self.scene_revision == self.required_revision" in gate


def test_a_correct_existing_scene_is_verified_rather_than_rebuilt():
    src = _read(os.path.join(REPO, "simulators", "isaac", "wisepack_isaac.py"))
    body = src[src.index("    def _sync_scene(self"):]
    body = body[:body.index("\n    def ", 10)]
    assert "_scene_matches(command)" in body
    assert "not rebuilding" in body
    assert "verified_without_rebuild=True" in body
    # But an acknowledgement is ALWAYS published, freshly correlated.
    assert "IsaacState.SCENE_READY" in body
    assert "_verify_scene_usable()" in body


def test_a_scene_holding_previous_items_is_never_reused():
    src = _read(os.path.join(REPO, "simulators", "isaac", "wisepack_isaac.py"))
    body = src[src.index("    def _scene_matches(self"):]
    body = body[:body.index("\n    def ", 10)]
    assert "_is_in_container" in body, (
        "a scene whose fingerprint matches but which still holds the previous "
        "run's objects is not reusable")
    assert "_robot_is_home" in body


def test_the_acknowledgement_is_measured_not_echoed():
    """Echoing the request back would check the orchestrator's numbers against
    themselves, which is exactly no check at all."""
    src = _read(os.path.join(REPO, "simulators", "isaac", "wisepack_isaac.py"))
    body = src[src.index("    def _scene_acknowledgement(self"):]
    body = body[:body.index("\n    def ", 10)]
    assert "scene_fingerprint(self.scenario" in body
    assert "sorted(self.scene.items)" in body
    assert "len(self.scene.items)" in body
    assert "self._robot_is_home()" in body


def test_the_reset_builds_the_scenario_before_requesting_the_rebuild():
    """It used to request first, stamping the PREVIOUS revision and no
    fingerprint — asking for a scenario that did not exist yet."""
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "hitl_orchestrator.py"))
    reset = src[src.index("def _reset_run"):src.index("def _write_artifacts")]
    assert reset.index("generate_or_load_scenario()") < \
        reset.index("request_scene_reset"), \
        "the scenario must exist before the rebuild is requested"


def test_the_operator_message_matches_what_is_on_screen():
    """Claiming the scene "has not been rebuilt" while the operator looks at a
    correct Panda, four cylinders and an empty container reads as a broken
    dashboard. The accurate statement is that the acknowledgement is missing."""
    for path in (("web", "snapshot.py"),
                 ("wisepack_ws", "src", "wisepack_orchestration",
                  "wisepack_orchestration", "isaac_bridge.py")):
        text = " ".join(_read(os.path.join(REPO, *path)).split())
        text = text.replace('" "', "")
        assert "Isaac is ready, but the scene acknowledgement for the " \
               "current run has not been received" in text, path
        assert "has not been rebuilt for this scenario yet" not in text, (
            f"{path[-1]} still claims the scene was never rebuilt, which is "
            "false when a correct scene is visibly present")


def test_the_scene_diagnostics_report_both_readiness_levels():
    src = _read(os.path.join(REPO, "web", "diagnostics.py"))
    for row in ("simulator_process", "ros_bridge", "requested_scene_revision",
                "acknowledged_scene_revision", "expected_object_count",
                "actual_object_count", "expected_fingerprint",
                "acknowledged_fingerprint", "scene_status"):
        assert row in src, f"the Diagnostics page must show {row}"


def test_scene_status_names_every_outcome():
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "isaac_bridge.py"))
    body = src[src.index("    def scene_status(self)"):]
    body = body[:body.index("\n    def ", 10)]
    for status in ("building", "ready", "mismatch", "failed"):
        assert f'"{status}"' in body


def test_approval_is_blocked_until_the_scene_matches_this_run():
    from web.snapshot import DashboardSnapshot
    snap = DashboardSnapshot(
        mode="ros", control_stage="WAIT_FOR_OPERATOR_APPROVAL",
        control_approval_state="pending", control_plan_id="plan-x",
        control_plan_valid=True, control_scene_ready=False,
        control_scene_reason="Isaac is ready, but the scene acknowledgement "
                             "for the current run has not been received")
    assert snap.can_approve is False
    assert "acknowledgement" in snap.approval_block_reason
