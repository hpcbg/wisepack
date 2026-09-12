"""The fixed-pipe dismantling scenario: an INSTALLED component is not waste
until a dismantling cut releases a section of it.

What these tests pin, against the code that runs the live demonstration:
the installed pipe is excluded from packing before the cut; the cut planner
proposes the predefined dismantling cut through the SAME comparison /
approval / request path as a loose cut; the bridge dispatches the fixed
segment; after CUT_COMPLETE the fixed remainder is still INSTALLED and never a
packing candidate, the released segment is a derived waste item with its
provenance, and the ordinary packing workflow places it. The loose-pipe cut
demo keeps passing unchanged (tests/test_isaac_cut.py).
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration"))

from wisepack_core.cut_optimizer import dismantling_segments, plan_cut_aware  # noqa: E402
from wisepack_core.domain import (ApprovalState, DomainError, ItemStatus,   # noqa: E402
                                  Scenario, WasteItem)
from wisepack_core.execution import ExecutionBackend                    # noqa: E402
from wisepack_core.generator import (FIXED_PIPE_INSTALLATION, build_scenario,  # noqa: E402
                                     preset_config)
from wisepack_core.isaac_contract import (ContractError, Dimensions,  # noqa: E402
                                          IsaacCommand, IsaacCommandType, Pose)
from wisepack_core.isaac_transform import source_pose_for               # noqa: E402
from wisepack_core.packing import OptimizerConfig, pack_baseline, pack_optimized  # noqa: E402
from wisepack_core.workflow import (ApprovalRequired, WorkflowConfig,   # noqa: E402
                                    WorkflowEngine, WorkflowError)

PRESET = "isaac_fixed_pipe_dismantling"


def _scenario():
    return build_scenario(PRESET, seed=7)


def _engine(backend=ExecutionBackend.ISAAC):
    eng = WorkflowEngine(WorkflowConfig(
        preset=PRESET, seed=7, execution_backend=backend,
        optimizer=OptimizerConfig(seed=7, restarts=4, time_budget_ms=2500)))
    eng.generate_or_load_scenario(_scenario())
    eng.scan_and_detect()
    eng.generate_plans()
    eng.digital_twin_validate()
    return eng


def _approved(eng):
    cmp = eng.wp.generate_cut_alternatives()
    eng.wp.select_alternative(cmp.recommended_label)
    eng.wp.approve_cut("test")
    request = eng.wp.build_cut_request()
    assert request is not None
    return cmp, request


def _cut_payload(eng, request):
    alt = eng.wp._selected_alternative()
    prop = alt.proposals[0]
    ids = prop.derived_item_ids_for()
    fixed, released = ids[0], ids[1]          # fixed_end -z: the +z segment is released
    return {
        "proposal_id": prop.proposal_id, "request_id": request["request_id"],
        "source_item_id": prop.source_item_id, "kerf_mm": prop.kerf_mm,
        "retained_segment_id": released, "fixed_segment_id": fixed, "installed": True,
        "segments": [
            {"item_id": fixed, "length_mm": prop.segment_lengths_mm[0],
             "pose": Pose(23.5, -520.0, 250.0, "x", "table").to_dict(),
             "retained": False, "fixed": True},
            {"item_id": released, "length_mm": prop.segment_lengths_mm[1],
             "pose": Pose(225.0, -520.0, 250.0, "x", "table").to_dict(), "retained": True}],
    }, released, fixed


# --------------------------------------------------------------------------- #
# The installed component
# --------------------------------------------------------------------------- #

def test_the_preset_holds_one_installed_pipe_run_beside_loose_tubes():
    scenario = _scenario()
    installed = scenario.installed_components
    assert [i.item_id for i in installed] == ["pipe-run-1"]
    pipe = installed[0]
    assert pipe.status is ItemStatus.INSTALLED and pipe.is_installed
    assert pipe.installation["fixed_end"] == "-z"
    assert pipe.installation["removable_length_mm"] == 150
    assert pipe.installation["elevation_mm"] == 250 == pipe.source_position.z
    assert pipe.length_mm == 400 and pipe.is_cuttable
    assert preset_config(PRESET).item_count == 4
    assert "isaac_fixed_pipe_dismantling" in build_scenario(PRESET, 3).preset


def test_an_installation_forces_the_installed_status_and_validates_its_fixed_end():
    item = WasteItem(item_id="x", length_mm=300, outer_diameter_mm=40,
                     installation={"removable_length_mm": 100})
    assert item.status is ItemStatus.INSTALLED
    assert item.installation["fixed_end"] == "+z"
    assert item.installation["component_id"] == "x"
    with pytest.raises(DomainError):
        WasteItem(item_id="y", length_mm=300, outer_diameter_mm=40,
                  installation={"fixed_end": "sideways"})
    doc = item.to_dict()
    assert doc["is_installed"] and doc["installation"]["removable_length_mm"] == 100
    assert WasteItem.from_dict(doc).is_installed


def test_the_installed_component_is_excluded_from_packing_before_the_cut():
    scenario = _scenario()
    assert "pipe-run-1" not in [i.item_id for i in scenario.packable_items]
    for plan in (pack_baseline(scenario), pack_optimized(scenario, config=OptimizerConfig(seed=7))):
        placed = {p.item_id for p in plan.placements}
        assert "pipe-run-1" not in placed
        assert "pipe-run-1" not in plan.unplaced_item_ids, \
            "an installed component is neither placed nor 'unplaced' — it is not waste"
        assert placed == {"tube-short-a", "tube-short-b", "tube-mid"}


def test_the_installed_pose_is_its_declared_position_never_a_row_slot():
    scenario = _scenario()
    pipe = scenario.item("pipe-run-1")
    pose = source_pose_for(None, 3, pipe)
    assert (pose.x_mm, pose.y_mm, pose.z_mm, pose.axis, pose.frame) == (100.0, -520.0, 250.0, "x", "table")
    loose = source_pose_for(None, 0, scenario.item("tube-short-a"))
    assert loose.z_mm == 20.0 and loose.x_mm != pose.x_mm


# --------------------------------------------------------------------------- #
# The dismantling cut through the cut-aware workflow
# --------------------------------------------------------------------------- #

def test_the_planner_proposes_the_predefined_dismantling_cut():
    cmp = plan_cut_aware(_scenario())
    assert cmp.recommend_cut and cmp.recommended_label.startswith("dismantle:pipe-run-1:247-150")
    best = cmp.recommended
    prop = best.proposals[0]
    assert prop.segment_lengths_mm == [247, 150] and prop.kerf_mm == 3
    assert prop.is_validated
    assert dismantling_segments(_scenario().item("pipe-run-1"), 3) == ([247, 150], 1, 0)
    placed = {p.item_id for p in best.plan.placements}
    assert "pipe-run-1-s2" in placed, "the released section is packed"
    assert "pipe-run-1-s1" not in placed, "the fixed remainder is never a packing candidate"
    assert "Dismantling pipe-run-1" in cmp.reason
    assert cmp.no_cut.containers == 1 == best.containers


def test_the_bridge_dispatches_the_fixed_segment_and_retains_the_released_one():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_scene_sync import _FakeNode, _bridge_module          # noqa: PLC0415
    module = _bridge_module()
    node = _FakeNode()
    bridge = module.IsaacExecutionBridge(node)
    bridge.simulator_ready = True
    eng = _engine()
    cmp, request = _approved(eng)
    assert bridge.request_cut(eng, request)
    command = bridge._cut_in_flight["command"]
    assert command.command is IsaacCommandType.EXECUTE_CUT
    assert command.cut["installed"] and command.cut["fixed_segment_id"] == "pipe-run-1-s1"
    assert command.cut["retained_segment_id"] == "pipe-run-1-s2"
    assert command.cut["component_id"] == "pipe-run-1"
    assert command.cut["cut_offset_mm"] == 248.5
    assert (command.source_pose.x_mm, command.source_pose.z_mm) == (100.0, 250.0)
    assert command.target_pose.frame == "container:CNT-01"


def test_the_contract_refuses_to_retain_the_installed_segment():
    cut = {"proposal_id": "p", "request_id": "r", "cut_offset_mm": 151.5, "kerf_mm": 3,
           "segment_ids": ["pipe-run-1-s1", "pipe-run-1-s2"], "segment_lengths_mm": [150, 247],
           "retained_segment_id": "pipe-run-1-s2", "fixed_segment_id": "pipe-run-1-s2"}
    fields = dict(command=IsaacCommandType.EXECUTE_CUT, run_id="run-1", sequence_index=-1,
                  item_id="pipe-run-1", dimensions=Dimensions(400, 40, 34),
                  source_pose=Pose(-80.0, -160.0, 250.0, "x", "table"),
                  target_pose=Pose(75.0, 60.0, 20.0, "x", "container:CNT-01"),
                  container_id="CNT-01", container_inner_mm={"x": 300, "y": 220, "z": 150},
                  scenario_revision=1, cut=cut)
    with pytest.raises(ContractError):
        IsaacCommand(**fields)
    cut["retained_segment_id"] = "pipe-run-1-s1"
    command = IsaacCommand(**fields)
    assert IsaacCommand.from_json(command.to_json()).cut["fixed_segment_id"] == "pipe-run-1-s2"
    cut["fixed_segment_id"] = "elsewhere"
    with pytest.raises(ContractError):
        IsaacCommand(**fields)


def test_after_the_cut_the_remainder_stays_installed_and_the_released_segment_is_waste():
    eng = _engine()
    cmp, request = _approved(eng)
    eng.wp.isaac_cut_begun(request["request_id"])
    payload, released, fixed = _cut_payload(eng, request)
    result = eng.wp.complete_isaac_cut(payload)
    assert result.succeeded and eng.wp.latest_cut_result["validation"]["valid"]
    ids = {i.item_id: i for i in eng.scenario.items}
    assert "pipe-run-1" not in ids
    remainder, waste = ids[fixed], ids[released]
    # The fixed side: still an INSTALLED component, provenance carried over,
    # never packable.
    assert remainder.status is ItemStatus.INSTALLED and remainder.is_installed
    assert remainder.installation["fixed_end"] == "-z"
    assert remainder.installation["dismantled_from"] == "pipe-run-1"
    assert remainder.installation["cut_operation_id"] == request["request_id"]
    assert remainder.length_mm == 247 and not remainder.is_cuttable
    assert remainder not in eng.scenario.packable_items
    assert remainder.source_position.z == 250
    # The released side: a derived waste item with dismantling provenance.
    assert waste.status is not ItemStatus.INSTALLED and waste.installation is None
    assert waste.parent_item_id == "pipe-run-1" and waste.generation == 1
    assert waste.length_mm == 150 and waste.outer_diameter_mm == 40
    lineage = waste.cut_history[-1]
    assert lineage["dismantling"] and lineage["dismantled_from"] == "pipe-run-1"
    assert lineage["cut_operation_id"] == request["request_id"]
    assert lineage["status"] == "available_for_packing"
    assert lineage["source_pose"]["position_mm"]["z"] == 250.0
    assert waste in eng.scenario.packable_items
    assert eng.wp.retained_segment_id == released and eng.wp.cut_awaiting_placement


def test_the_released_segment_goes_through_the_normal_packing_workflow():
    eng = _engine()
    cmp, request = _approved(eng)
    eng.wp.isaac_cut_begun(request["request_id"])
    payload, released, fixed = _cut_payload(eng, request)
    eng.wp.complete_isaac_cut(payload)
    eng.wp.retained_segment_placed(released, {"position_error_mm": 4.0})
    # Placed straight from the cut; the plan stands; packing approval again.
    assert eng.scenario.item(released).status is ItemStatus.PLACED
    assert eng.selected.placement_for_item(released).executed
    assert eng.selected.placement_for_item(fixed) is None
    assert eng.selected.approval_state is ApprovalState.PENDING
    with pytest.raises((ApprovalRequired, WorkflowError)):
        eng.step_execution()
    pending = [p.item_id for p in eng.selected.placements if not p.executed]
    assert set(pending) == {"tube-short-a", "tube-short-b", "tube-mid"}
    eng.approve()
    nxt = eng.next_physical_placement()
    assert nxt is not None and nxt[1].item_id in pending
    # And a fresh re-plan on the derived scenario would still never touch the remainder.
    replanned = pack_optimized(eng.scenario, config=OptimizerConfig(seed=7))
    assert fixed not in {p.item_id for p in replanned.placements}
    assert fixed not in replanned.unplaced_item_ids


def test_the_dashboard_snapshot_tells_the_dismantling_story_compactly():
    eng = _engine()
    cmp, request = _approved(eng)
    snap = eng.wp.dismantling_snapshot()
    assert snap["component_id"] == "pipe-run-1"
    assert snap["installed"]["length_mm"] == 400 and snap["installed"]["elevation_mm"] == 250
    assert snap["proposal"]["segments_mm"] == [247, 150]
    assert snap["proposal"]["cut_from_free_end_mm"] == 150
    assert snap["released_segment"] == {"item_id": "pipe-run-1-s2", "length_mm": 150}
    assert snap["fixed_remainder"]["item_id"] == "pipe-run-1-s1"
    assert snap["cut_approval_state"] == "approved"
    assert snap["registered_as_waste"] is None
    assert snap["packing_target"]["container_id"] == "CNT-01"
    assert "cut" in snap["execution_status"] or "proposed" in snap["execution_status"]
    eng.wp.isaac_cut_begun(request["request_id"])
    payload, released, fixed = _cut_payload(eng, request)
    eng.wp.complete_isaac_cut(payload)
    eng.wp.retained_segment_placed(released, {})
    snap = eng.wp.dismantling_snapshot()
    assert snap["registered_as_waste"]["item_id"] == released
    assert snap["registered_as_waste"]["cut_operation_id"] == request["request_id"]
    assert snap["packing_target"]["executed"]
    assert snap["execution_status"] == "released segment placed in the container"
    assert eng.wp.cut_snapshot()["dismantling"]["fixed_remainder"]["status"] == "installed"


def test_the_loose_pipe_cut_demo_sees_no_dismantling_view():
    eng = WorkflowEngine(WorkflowConfig(
        preset="isaac_cut_demo", seed=7, execution_backend=ExecutionBackend.ISAAC,
        optimizer=OptimizerConfig(seed=7, restarts=4, time_budget_ms=2500)))
    eng.generate_or_load_scenario(build_scenario("isaac_cut_demo", 7))
    eng.scan_and_detect(); eng.generate_plans(); eng.digital_twin_validate()
    cmp = eng.wp.generate_cut_alternatives()
    assert cmp.recommended_label.startswith("cut:tube-long:150-267")
    assert eng.wp.dismantling_snapshot() is None
    assert eng.wp.cut_snapshot()["dismantling"] is None
    assert not any(a.label.startswith("dismantle:") for a in cmp.alternatives)


def test_a_scenario_without_installed_components_is_unchanged():
    scenario = build_scenario("isaac_cylinders_smoke", 42)
    assert scenario.installed_components == [] and scenario.packable_items == scenario.items
    assert FIXED_PIPE_INSTALLATION["component_id"] == "pipe-run-1"
    assert isinstance(scenario, Scenario)
