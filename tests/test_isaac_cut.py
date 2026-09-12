"""The Isaac cut-and-place skill: contract, planner, workflow and bridge.

What these pin, without Isaac Sim:

* EXECUTE_CUT is a first-class command: it carries the planner's cut geometry,
  round-trips through JSON, and refuses to be commanded without it; the two
  new physical states map onto the cut stages the workflow already has;
* the bench-scale `isaac_cut_demo` preset makes the cut-aware planner
  RECOMMEND a cut — because the whole tube fits the bin in no orientation —
  and only the arm that carries the combined tool may run it;
* the whole-process layer registers the MEASURED segments the simulator
  reports, waits for the retained segment to be placed, freezes that
  placement and re-plans the rest with packing approval required again — and
  cut approval still authorises no ordinary pick;
* the orchestrator bridge dispatches exactly the planner's decision (cut
  plane, segment ids and lengths, retained segment, its validated target),
  outside the placement queue, and turns the simulator's CUT_COMPLETED and
  ITEM_COMPLETED into the workflow steps above, remembering where the
  remainder lies so the next EXECUTE_ITEM picks it from there.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"),
             os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration"),
             os.path.join(REPO, "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)

from wisepack_core.cutting import CutState                          # noqa: E402
from wisepack_core.domain import ApprovalState, ItemStatus, Vec3   # noqa: E402
from wisepack_core.events import Stage                              # noqa: E402
from wisepack_core.execution import (ExecutionBackend,              # noqa: E402
                                     preset_physical_compatibility,
                                     robot_state_for_isaac_state,
                                     stage_for_isaac_state)
from wisepack_core.generator import build_scenario                  # noqa: E402
from wisepack_core.isaac_contract import (ContractError, Dimensions,  # noqa: E402
                                          IsaacCommand, IsaacCommandType,
                                          IsaacFeedback, IsaacState,
                                          ITEM_PROGRESS_ORDER, Pose,
                                          SCHEMA_VERSION)
from wisepack_core.packing import OptimizerConfig                   # noqa: E402
from wisepack_core.robots import BASE_SKILLS, KNOWN_SKILLS, load_registry  # noqa: E402
from wisepack_core.workflow import (ApprovalRequired, WorkflowConfig,  # noqa: E402
                                    WorkflowEngine, WorkflowError)

PRESET = "isaac_cut_demo"


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #

def _cut_geometry(retained="tube-long-s2"):
    return {"proposal_id": "cut-x", "request_id": "cutreq-1", "cut_offset_mm": 151.5,
            "kerf_mm": 3, "segment_ids": ["tube-long-s1", "tube-long-s2"],
            "segment_lengths_mm": [150, 267], "retained_segment_id": retained}


def _cut_command(**overrides):
    fields = dict(
        command=IsaacCommandType.EXECUTE_CUT, run_id="run-1", sequence_index=-1,
        item_id="tube-long", dimensions=Dimensions(420, 40, 34),
        source_pose=Pose(480.0, -320.0, 20.0, "x", "table"),
        target_pose=Pose(133.5, 20.0, 20.0, "x", "container:CNT-01"),
        container_id="CNT-01", container_inner_mm={"x": 300, "y": 220, "z": 150},
        scenario_revision=1, cut=_cut_geometry())
    fields.update(overrides)
    return IsaacCommand(**fields)


def test_the_schema_bumped_a_minor_for_the_additive_cut_vocabulary():
    assert SCHEMA_VERSION == "wisepack-isaac/1.2"


def test_execute_cut_round_trips_with_its_geometry():
    command = _cut_command()
    back = IsaacCommand.from_json(command.to_json())
    assert back.command is IsaacCommandType.EXECUTE_CUT
    assert back.cut == command.cut
    assert back.sequence_index == -1
    assert back.target_pose == command.target_pose


def test_execute_cut_refuses_to_be_commanded_without_the_planner_decision():
    with pytest.raises(ContractError):
        _cut_command(cut=None)
    with pytest.raises(ContractError):
        _cut_command(cut={k: v for k, v in _cut_geometry().items() if k != "segment_ids"})
    with pytest.raises(ContractError):
        _cut_command(cut={**_cut_geometry(), "retained_segment_id": "somebody-else"})
    with pytest.raises(ContractError):
        _cut_command(cut={**_cut_geometry(), "segment_ids": ["a", "b", "c"],
                          "segment_lengths_mm": [1, 2, 3]})


def test_an_ordinary_pick_cannot_smuggle_cut_geometry():
    with pytest.raises(ContractError):
        _cut_command(command=IsaacCommandType.EXECUTE_ITEM, sequence_index=0)


def test_the_cut_states_map_onto_the_cut_stages_and_the_progress_order():
    assert stage_for_isaac_state(IsaacState.CUTTING) is Stage.CUT_IN_PROGRESS
    assert stage_for_isaac_state(IsaacState.CUT_COMPLETED) is Stage.CUT_COMPLETED
    assert robot_state_for_isaac_state(IsaacState.CUTTING) == "cutting"
    order = list(ITEM_PROGRESS_ORDER)
    assert order.index(IsaacState.GRASPING) < order.index(IsaacState.CUTTING) \
        < order.index(IsaacState.CUT_COMPLETED) < order.index(IsaacState.LIFTING)


# --------------------------------------------------------------------------- #
# The demo preset and the planner
# --------------------------------------------------------------------------- #

def test_the_demo_preset_is_bench_scale_and_only_the_cutter_arm_runs_it():
    scenario = build_scenario(PRESET, seed=7)
    assert scenario.max_containers == 1
    assert scenario.container_template.inner_size == Vec3(300, 220, 150)
    long_tube = scenario.item("tube-long")
    assert long_tube.is_cuttable and long_tube.length_mm == 420
    assert max(i.outer_diameter_mm for i in scenario.items) <= 78
    registry = load_registry()
    verdicts = {rid: preset_physical_compatibility(PRESET, p)[0]
                for rid, p in registry.profiles.items()}
    assert verdicts == {"panda": True, "xarm7": False}
    panda = registry.profiles["panda"]
    assert panda.has_cutter and "CUT" in panda.supported_skills
    assert "CUT" in KNOWN_SKILLS and "CUT" not in BASE_SKILLS
    assert panda.end_effector_tool["cut_offset_m"][0] > 0.0


def _engine(preset=PRESET, seed=7, backend=ExecutionBackend.SIMULATED):
    eng = WorkflowEngine(WorkflowConfig(
        preset=preset, seed=seed, execution_backend=backend,
        optimizer=OptimizerConfig(seed=seed, restarts=4, time_budget_ms=2500)))
    eng.generate_or_load_scenario(build_scenario(preset, seed))
    eng.scan_and_detect()
    eng.generate_plans()
    eng.digital_twin_validate()
    return eng


def test_the_planner_recommends_cutting_the_tube_that_fits_nowhere():
    eng = _engine()
    assert "tube-long" in eng.selected.unplaced_item_ids
    cmp = eng.wp.generate_cut_alternatives()
    assert cmp.recommend_cut, cmp.reason
    best = cmp.recommended
    assert best.is_cut and not best.plan.unplaced_item_ids
    assert best.containers == 1
    prop = best.proposals[0]
    assert prop.n_cuts == 1 and prop.is_validated
    assert sum(prop.segment_lengths_mm) + prop.kerf_mm == 420
    assert set(prop.derived_item_ids_for()) <= {p.item_id for p in best.plan.placements}


# --------------------------------------------------------------------------- #
# Whole process: the Isaac skill's path through the workflow
# --------------------------------------------------------------------------- #

def _approved_cut_engine():
    eng = _engine(backend=ExecutionBackend.ISAAC)
    cmp = eng.wp.generate_cut_alternatives()
    eng.wp.select_alternative(cmp.recommended_label)
    eng.wp.approve_cut("test")
    request = eng.wp.build_cut_request()
    assert request is not None
    return eng, request


def _segments_payload(eng, retained=None):
    alt = eng.wp._selected_alternative()
    prop = alt.proposals[0]
    ids = prop.derived_item_ids_for()
    order = [p.item_id for p in alt.plan.ordered_placements]
    retained = retained or next(c for c in order if c in ids)
    return {
        "proposal_id": prop.proposal_id, "request_id": "cutreq-1",
        "source_item_id": prop.source_item_id, "kerf_mm": prop.kerf_mm,
        "retained_segment_id": retained,
        "segments": [
            {"item_id": ids[0], "length_mm": prop.segment_lengths_mm[0],
             "pose": Pose(405.0, -320.0, 20.0, "x", "table").to_dict(),
             "retained": ids[0] == retained},
            {"item_id": ids[1], "length_mm": prop.segment_lengths_mm[1],
             "pose": Pose(613.5, -320.0, 20.0, "x", "table").to_dict(),
             "retained": ids[1] == retained},
        ],
    }, retained


def test_the_isaac_cut_registers_measured_segments_and_waits_for_the_placement():
    eng, request = _approved_cut_engine()
    revision = eng.scenario_revision
    eng.wp.isaac_cut_begun(request["request_id"])
    assert eng.wp.cut_skill_state is CutState.IN_PROGRESS
    assert eng.stage is Stage.CUT_IN_PROGRESS
    payload, retained = _segments_payload(eng)
    result = eng.wp.complete_isaac_cut(payload)
    assert result.succeeded and eng.wp.latest_cut_result["validation"]["valid"]
    ids = {i.item_id for i in eng.scenario.items}
    assert "tube-long" not in ids and set(result.resulting_child_ids) <= ids
    assert eng.scenario_revision == revision + 1
    remainder = next(c for c in result.resulting_child_ids if c != retained)
    assert eng.scenario.item(remainder).source_position.x in (405, 614)
    assert eng.scenario.item(remainder).cut_history[-1]["source_pose"]["frame"] == "table"
    # NO RE-PLAN YET: the retained segment is in the fingers, on its way.
    assert eng.wp.cut_awaiting_placement and eng.wp.retained_segment_id == retained
    assert eng.stage is not Stage.WAIT_FOR_OPERATOR_APPROVAL


def test_the_placed_segment_is_frozen_and_the_rest_needs_packing_approval():
    eng, request = _approved_cut_engine()
    eng.wp.isaac_cut_begun(request["request_id"])
    payload, retained = _segments_payload(eng)
    eng.wp.complete_isaac_cut(payload)
    eng.wp.retained_segment_placed(retained, {"position_error_mm": 12.0})
    assert eng.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert eng.selected.approval_state is ApprovalState.PENDING
    placed = eng.selected.placement_for_item(retained)
    assert placed is not None and placed.executed
    assert eng.scenario.item(retained).status is ItemStatus.PLACED
    pending = [p.item_id for p in eng.selected.placements if not p.executed]
    assert retained not in pending and len(pending) == 3
    assert not eng.wp.cut_awaiting_placement
    # Cut approval authorised the cut-and-place; it authorises NO other pick.
    with pytest.raises((ApprovalRequired, WorkflowError)):
        eng.step_execution()


def test_a_cut_whose_measured_lengths_deviate_is_replanned_not_reused():
    """The proposal's plan is reused ONLY when the actual segments match it."""
    eng, request = _approved_cut_engine()
    eng.wp.isaac_cut_begun(request["request_id"])
    payload, retained = _segments_payload(eng)
    proposed = eng.wp._selected_alternative().plan
    # The simulator measured a different split (same total, kerf included).
    payload["segments"][0]["length_mm"] -= 10
    payload["segments"][1]["length_mm"] += 10
    result = eng.wp.complete_isaac_cut(payload)
    assert result.actual_segment_lengths_mm != list(
        eng.wp._selected_alternative().proposals[0].segment_lengths_mm)
    lengths = {i.item_id: i.length_mm for i in eng.scenario.items}
    assert lengths[result.resulting_child_ids[0]] == payload["segments"][0]["length_mm"]
    eng.wp.retained_segment_placed(retained, {"position_error_mm": 3.0})
    trail = json.dumps([e.to_dict() for e in eng.log.events()])
    assert "differ from the proposal" in trail
    # A fresh plan, not the proposal's, and it still needs packing approval.
    assert eng.selected is not proposed
    assert eng.selected.approval_state is ApprovalState.PENDING
    assert eng.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    with pytest.raises((ApprovalRequired, WorkflowError)):
        eng.step_execution()


def test_matching_segments_reuse_the_validated_plan_but_validate_it_again():
    eng, request = _approved_cut_engine()
    eng.wp.isaac_cut_begun(request["request_id"])
    payload, retained = _segments_payload(eng)
    proposed = eng.wp._selected_alternative().plan
    eng.wp.complete_isaac_cut(payload)
    eng.wp.retained_segment_placed(retained, {"position_error_mm": 2.0})
    assert eng.selected is proposed
    events = [e.to_dict() for e in eng.log.events()]
    replan = [e for e in events if e.get("action") == "REPLAN_AFTER_CUT"]
    assert replan and replan[-1]["details"]["placements_valid"] == len(eng.selected.placements)
    assert set(replan[-1]["details"]["pending"]) == {
        p.item_id for p in eng.selected.placements if not p.executed}


def test_a_cut_that_fails_before_the_cut_leaves_the_pipe_whole():
    eng, request = _approved_cut_engine()
    eng.wp.isaac_cut_begun(request["request_id"])
    eng.wp.isaac_cut_failed("blades jammed (simulated)")
    assert eng.wp.cut_skill_state is CutState.FAILED
    assert eng.scenario.item("tube-long") is not None
    assert eng.wp.selected_cut_label == "no_cut"
    assert eng.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL


def test_a_carry_that_fails_after_the_cut_replans_both_segments():
    eng, request = _approved_cut_engine()
    eng.wp.isaac_cut_begun(request["request_id"])
    payload, retained = _segments_payload(eng)
    eng.wp.complete_isaac_cut(payload)
    eng.wp.isaac_cut_failed("dropped the segment on the way")
    ids = {p.item_id for p in eng.selected.placements}
    assert retained in ids
    assert eng.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert not any(p.executed for p in eng.selected.placements)


# --------------------------------------------------------------------------- #
# The bridge: dispatch and feedback
# --------------------------------------------------------------------------- #

def _bridge():
    from test_scene_sync import _FakeNode, _bridge_module          # noqa: PLC0415
    module = _bridge_module()
    node = _FakeNode()
    bridge = module.IsaacExecutionBridge(node)
    bridge.simulator_ready = True
    return bridge, node


def _feedback(bridge, state, command, **kw):
    return IsaacFeedback.from_json(IsaacFeedback(
        state=state, run_id=bridge.gate.run_id, item_id=kw.pop("item_id", command.item_id),
        sequence_index=-1, container_id=command.container_id,
        target_pose=command.target_pose, **kw).to_json())


def _ready(bridge, eng):
    """The simulator's READY for this run, as it answers RUN_BEGIN."""
    bridge._apply(eng, IsaacFeedback.from_json(IsaacFeedback(
        state=IsaacState.READY, run_id=eng.run_id).to_json()))


def test_the_bridge_dispatches_exactly_the_planners_cut():
    bridge, node = _bridge()
    eng, request = _approved_cut_engine()
    assert bridge.request_cut(eng, request)
    # NOT YET: RUN_BEGIN went out and the cut waits for the simulator's READY,
    # because the latched keep-last topic would otherwise drop RUN_BEGIN.
    assert not [m for m in node.published if m.get("command") == "EXECUTE_CUT"]
    _ready(bridge, eng)
    commands = [m for m in node.published if m.get("command") == "EXECUTE_CUT"]
    assert len(commands) == 1
    command = IsaacCommand.from_dict(commands[0])
    alt = eng.wp._selected_alternative()
    prop = alt.proposals[0]
    assert command.item_id == "tube-long" and command.sequence_index == -1
    assert command.cut["segment_ids"] == prop.derived_item_ids_for()
    assert command.cut["segment_lengths_mm"] == list(prop.segment_lengths_mm)
    assert command.cut["cut_offset_mm"] == pytest.approx(
        prop.cut_positions_mm[0] + prop.kerf_mm / 2.0)
    retained = command.cut["retained_segment_id"]
    placement = alt.plan.placement_for_item(retained)
    assert command.target_pose.frame == f"container:{placement.container_id}"
    assert command.container_inner_mm == {"x": 300, "y": 220, "z": 150}
    # RUN_BEGIN went out first: the cut is the first physical act of the run.
    kinds = [m.get("command") for m in node.published]
    assert kinds.index("RUN_BEGIN") < kinds.index("EXECUTE_CUT")
    assert bridge._cut_in_flight is not None
    assert not bridge.request_cut(eng, request)          # one at a time


def test_cut_feedback_drives_the_workflow_and_remembers_where_the_remainder_lies():
    bridge, node = _bridge()
    eng, request = _approved_cut_engine()
    bridge.request_cut(eng, request)
    _ready(bridge, eng)
    command = IsaacCommand.from_dict(
        next(m for m in node.published if m.get("command") == "EXECUTE_CUT"))
    revision = eng.scenario_revision
    for state in (IsaacState.MOVING_TO_PICK, IsaacState.GRASPING, IsaacState.CUTTING):
        bridge._apply(eng, _feedback(bridge, state, command))
    assert eng.wp.cut_skill_state is CutState.IN_PROGRESS
    payload, retained = _segments_payload(eng)
    bridge._apply(eng, _feedback(bridge, IsaacState.CUT_COMPLETED, command,
                                 detail={"cut": payload}))
    assert eng.scenario_revision == revision + 1
    assert bridge.required_revision == bridge.scene_revision == eng.scenario_revision
    remainder = next(c for c in payload["segments"] if not c["retained"])["item_id"]
    assert remainder in bridge.derived_poses
    for state in (IsaacState.LIFTING, IsaacState.MOVING_TO_CONTAINER,
                  IsaacState.RELEASING, IsaacState.SETTLING):
        bridge._apply(eng, _feedback(bridge, state, command, item_id=retained))
    actual = Pose(140.0, 22.0, 20.0, "x", "container:CNT-01")
    bridge._apply(eng, _feedback(bridge, IsaacState.ITEM_COMPLETED, command,
                                 item_id=retained, actual_pose=actual,
                                 position_error_mm=6.5))
    assert bridge._cut_in_flight is None
    assert eng.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert eng.selected.placement_for_item(retained).executed
    assert bridge.results and bridge.results[-1]["item_id"] == retained
    # The next pick of the remainder starts where the cut left it.
    eng.approve()
    nxt = eng.next_physical_placement()
    if nxt is not None and nxt[1].item_id == remainder:
        built = bridge._build_command(eng, nxt[0], nxt[1], nxt[2], 0)
        assert built.source_pose == bridge.derived_poses[remainder]
    trail = json.dumps([e.to_dict() for e in eng.log.events()])
    assert "isaac_cut_dispatched" in trail and "CUT_COMPLETED" in trail
    assert "isaac_retained_segment_placed" in trail


def test_the_simulated_backend_still_cuts_through_the_operator_command():
    """The logical backend keeps `simulate_cut`; only Isaac executes physically."""
    eng = _engine()
    cmp = eng.wp.generate_cut_alternatives()
    eng.wp.select_alternative(cmp.recommended_label)
    eng.wp.approve_cut("test")
    eng.wp.simulate_cut()
    assert eng.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert "tube-long" not in {i.item_id for i in eng.scenario.items}


def test_a_cut_does_not_shift_the_expected_pose_of_untouched_objects():
    """The generated row slot is the slot the body was BUILT in. Removing the
    parent from the scenario and appending its segments must leave every other
    object's expected source pose exactly where its body still lies."""
    from wisepack_core.isaac_transform import table_pose_for_index
    bridge, node = _bridge()
    eng, request = _approved_cut_engine()
    bridge.request_scene_sync(eng, eng.scenario_revision)      # the scene as built
    built = {i.item_id: n for n, i in enumerate(eng.scenario.items)}
    expected = {iid: table_pose_for_index(n, eng.scenario.item(iid), bridge.layout)
                for iid, n in built.items() if iid != "tube-long"}
    bridge.request_cut(eng, request)
    _ready(bridge, eng)
    command = IsaacCommand.from_dict(
        next(m for m in node.published if m.get("command") == "EXECUTE_CUT"))
    payload, retained = _segments_payload(eng)
    bridge._apply(eng, _feedback(bridge, IsaacState.CUT_COMPLETED, command,
                                 detail={"cut": payload}))
    bridge._apply(eng, _feedback(bridge, IsaacState.ITEM_COMPLETED, command,
                                 item_id=retained,
                                 actual_pose=Pose(140.0, 22.0, 20.0, "x", "container:CNT-01"),
                                 position_error_mm=6.5))
    eng.approve()
    # Every untouched object keeps the slot its body was built in, whatever its
    # index in the re-registered scenario; the remainder picks from its
    # measured cut pose.
    order = [i.item_id for i in eng.scenario.items]
    assert order.index("tube-short-a") != built["tube-short-a"]      # it DID shift
    for placement in eng.selected.placements:
        if placement.executed:
            continue
        item = eng.scenario.item(placement.item_id)
        container = eng.selected.container(placement.container_id)
        built_cmd = bridge._build_command(eng, placement, item, container, 0)
        if item.item_id in expected:
            assert built_cmd.source_pose == expected[item.item_id]
        else:
            assert built_cmd.source_pose == bridge.derived_poses[item.item_id]
