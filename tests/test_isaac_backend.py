"""The Isaac execution backend, tested WITHOUT Isaac Sim and without a GPU.

That constraint is the point of the file, not a limitation of it. Isaac is an
optional backend: normal CI and `pytest tests` must never need a simulator, a
GPU or a display. So everything here exercises the parts that decide whether a
physical run is correct — the contract, the coordinate conversion, the run
gating, the state mapping and the workflow integration — using plain Python.

What is deliberately NOT covered here, because it genuinely needs the simulator:
the scene building, the IK convergence and the settling physics. Those are
covered by `scripts/validate_isaac_sim.sh`, which skips with exit code 77 when
Isaac is absent.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess

import pytest

from wisepack_core.domain import (
    Axis, Container, ContainerStatus, ItemStatus, Placement, Vec3,
)
from wisepack_core.events import Actor, Stage
from wisepack_core.execution import (
    ExecutionBackend, ISAAC_STATE_STAGE, parse_backend,
    robot_state_for_isaac_state, stage_for_isaac_state,
)
from wisepack_core.generator import ISAAC_SMOKE_PRESET, PRESETS, build_scenario
from wisepack_core.isaac_contract import (
    SCHEMA_VERSION, ContractError, Dimensions, IsaacCommand, IsaacCommandType,
    IsaacFeedback, IsaacState, Pose, RunGate,
)
from wisepack_core.isaac_transform import (
    DEFAULT_LAYOUT, SceneLayout, axis_deviation_deg, axis_from_quaternion,
    check_containment, container_slot, dimensions_for, mm_to_m, placement_pose,
    pose_to_world, quaternion_for_axis, table_pose_for_index, world_to_pose,
)
from wisepack_core.packing import OptimizerConfig, pack_optimized
from wisepack_core.validator import PlacementValidator
from wisepack_core.workflow import (
    ApprovalRequired, WorkflowConfig, WorkflowEngine,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Execution-backend selection
# --------------------------------------------------------------------------- #

def test_backend_names_parse():
    assert parse_backend("simulated") is ExecutionBackend.SIMULATED
    assert parse_backend("isaac") is ExecutionBackend.ISAAC
    assert parse_backend("  ISAAC  ") is ExecutionBackend.ISAAC


def test_absent_backend_means_simulated():
    """The default must preserve the existing behaviour exactly."""
    assert parse_backend(None) is ExecutionBackend.SIMULATED
    assert parse_backend("") is ExecutionBackend.SIMULATED
    assert WorkflowConfig().execution_backend is ExecutionBackend.SIMULATED


def test_unknown_backend_is_rejected_not_defaulted():
    """A typo must not quietly select the simulated backend.

    Falling back would produce a run that reports physical execution it never
    performed — the single worst failure mode this feature has.
    """
    with pytest.raises(ValueError, match="unknown execution_backend"):
        parse_backend("issac")
    with pytest.raises(ValueError, match="unknown execution_backend"):
        parse_backend("real_robot")


def test_only_isaac_claims_to_be_physical():
    assert ExecutionBackend.ISAAC.is_physical is True
    assert ExecutionBackend.SIMULATED.is_physical is False
    assert "ISAAC" in ExecutionBackend.ISAAC.label
    assert "PHYSIC" in ExecutionBackend.ISAAC.label.upper()
    # The simulated label must never suggest physics.
    assert "PHYSIC" not in ExecutionBackend.SIMULATED.label.upper()


# --------------------------------------------------------------------------- #
# Contract serialisation
# --------------------------------------------------------------------------- #

def _command() -> IsaacCommand:
    return IsaacCommand(
        command=IsaacCommandType.EXECUTE_ITEM,
        run_id="run-abc123", sequence_index=3, item_id="item-002",
        dimensions=Dimensions(200, 50, 38),
        source_pose=Pose(480.0, -210.0, 25.0, "x", "table"),
        target_pose=Pose(100.0, 25.0, 25.0, "y", "container:CNT-01"),
        container_id="CNT-01",
        container_inner_mm={"x": 300, "y": 220, "z": 150},
        preset=ISAAC_SMOKE_PRESET, seed=42, plan_id="plan-1", total_items=4)


def test_command_round_trips_through_json():
    original = _command()
    restored = IsaacCommand.from_json(original.to_json())
    assert restored.to_dict() == original.to_dict()
    assert restored.source_pose.frame == "table"
    assert restored.target_pose.frame == "container:CNT-01"
    assert restored.target_pose.axis == "y"


def test_feedback_round_trips_through_json():
    original = IsaacFeedback(
        state=IsaacState.ITEM_COMPLETED, run_id="run-abc123",
        sequence_index=3, item_id="item-002", container_id="CNT-01",
        dimensions=Dimensions(200, 50, 38),
        target_pose=Pose(100.0, 25.0, 25.0, "y", "container:CNT-01"),
        actual_pose=Pose(108.5, 31.0, 24.0, "y", "container:CNT-01"),
        position_error_mm=10.4, message="settled",
        detail={"settle_elapsed_s": 1.2})
    restored = IsaacFeedback.from_json(original.to_json())
    assert restored.to_dict() == original.to_dict()
    assert restored.is_item_terminal is True
    assert restored.position_error_mm == pytest.approx(10.4)


def test_every_required_contract_field_is_present():
    """The fields the integration contract mandates, on both message types."""
    required = {"schema_version", "run_id", "item_id", "sequence_index",
                "source_pose", "target_pose", "dimensions", "timestamp"}
    command = json.loads(_command().to_json())
    assert required <= set(command)
    assert "command" in command

    feedback = json.loads(IsaacFeedback(
        state=IsaacState.SETTLING, run_id="r", item_id="item-001",
        sequence_index=0).to_json())
    assert required <= set(feedback)
    assert "state" in feedback


def test_incompatible_schema_major_is_refused():
    """Refused, not best-effort parsed: a mis-read pose moves a robot."""
    doc = json.loads(_command().to_json())
    doc["schema_version"] = "wisepack-isaac/2.0"
    with pytest.raises(ContractError, match="incompatible"):
        IsaacCommand.from_dict(doc)


def test_a_compatible_minor_bump_is_accepted():
    doc = json.loads(_command().to_json())
    major = SCHEMA_VERSION.rsplit(".", 1)[0]
    doc["schema_version"] = f"{major}.9"
    assert IsaacCommand.from_dict(doc).item_id == "item-002"


def test_malformed_payloads_raise_contract_errors():
    for blob in ("", "not json", "[]", '{"schema_version": "wisepack-isaac/1.0"}'):
        with pytest.raises(ContractError):
            IsaacCommand.from_json(blob)


def test_execute_item_without_a_pose_pair_is_refused():
    """A physical pick cannot be commanded without knowing where and where to."""
    with pytest.raises(ContractError, match="missing"):
        IsaacCommand(command=IsaacCommandType.EXECUTE_ITEM, run_id="r",
                     sequence_index=0, item_id="item-001",
                     dimensions=Dimensions(200, 50, 38))


def test_execute_item_requires_a_sequence_index():
    with pytest.raises(ContractError, match="sequence_index"):
        IsaacCommand(
            command=IsaacCommandType.EXECUTE_ITEM, run_id="r", item_id="i-1",
            dimensions=Dimensions(200, 50, 38),
            source_pose=Pose(0, 0, 0), target_pose=Pose(0, 0, 0))


def test_lifecycle_commands_need_no_item():
    """RUN_BEGIN / RUN_END / RUN_ABORT are about the run, not about an item."""
    for kind in (IsaacCommandType.RUN_BEGIN, IsaacCommandType.RUN_END,
                 IsaacCommandType.RUN_ABORT):
        assert IsaacCommand(command=kind, run_id="r").item_id is None


def test_a_terminal_feedback_must_name_its_item():
    for state in (IsaacState.ITEM_COMPLETED, IsaacState.ITEM_FAILED):
        with pytest.raises(ContractError, match="must name the item"):
            IsaacFeedback(state=state, run_id="r")


# --------------------------------------------------------------------------- #
# Run gating: duplicate and stale rejection
# --------------------------------------------------------------------------- #

def test_a_command_before_run_begin_is_rejected():
    gate = RunGate()
    assert "no run has been opened" in gate.reject_reason("run-1", 0)


def test_a_foreign_run_id_is_rejected():
    """An Isaac process left over from a previous invocation must not report
    into a run it knows nothing about."""
    gate = RunGate()
    gate.adopt("run-1")
    assert gate.reject_reason("run-1", 0) == ""
    assert "not the current run" in gate.reject_reason("run-2", 0)
    assert gate.accepts_run("run-1") and not gate.accepts_run("run-2")


def test_a_replayed_sequence_index_is_rejected():
    """THE latched-topic trap: TRANSIENT_LOCAL redelivers on every re-subscribe,
    and acting twice means picking an item already in the container."""
    gate = RunGate()
    gate.adopt("run-1")
    assert gate.reject_reason("run-1", 0) == ""
    gate.mark_done(0)
    assert "already executed" in gate.reject_reason("run-1", 0)
    assert gate.reject_reason("run-1", 1) == ""
    assert gate.completed_count == 1


def test_a_genuine_retry_is_not_mistaken_for_a_replay():
    """Regression, found end to end on real hardware.

    A physical attempt that fails is legitimately re-commanded for the SAME
    placement. Keying the duplicate guard on ``sequence_index`` alone made a
    retry indistinguishable from a latched redelivery, and the simulator
    silently discarded every retry — the arm failed a pre-grasp, the
    orchestrator dispatched attempt 1, and nothing happened.
    """
    gate = RunGate()
    gate.adopt("run-1")
    gate.mark_done(0, attempt=0)
    assert "already executed" in gate.reject_reason("run-1", 0, attempt=0)
    assert gate.reject_reason("run-1", 0, attempt=1) == "", \
        "a retry of a failed placement must be executed, not dropped"
    gate.mark_done(0, attempt=1)
    # Retries are attempts, not progress: one placement resolved, not two.
    assert gate.completed_count == 1


def test_the_attempt_number_travels_on_the_wire():
    command = IsaacCommand.from_json(IsaacCommand(
        command=IsaacCommandType.EXECUTE_ITEM, run_id="r", sequence_index=2,
        attempt=1, item_id="item-003", dimensions=Dimensions(200, 50, 38),
        source_pose=Pose(0, 0, 0), target_pose=Pose(0, 0, 0)).to_json())
    assert command.attempt == 1
    # Absent on an older payload means attempt 0 — a compatible addition.
    doc = json.loads(_command().to_json())
    doc.pop("attempt")
    assert IsaacCommand.from_dict(doc).attempt == 0


def test_adopting_a_new_run_clears_the_item_history():
    gate = RunGate()
    gate.adopt("run-1")
    gate.mark_done(0)
    gate.adopt("run-2")
    assert gate.reject_reason("run-2", 0) == ""
    assert gate.completed_count == 0


def test_an_empty_run_id_cannot_be_adopted():
    with pytest.raises(ContractError):
        RunGate().adopt("")


# --------------------------------------------------------------------------- #
# Coordinate and unit conversion
# --------------------------------------------------------------------------- #

def test_units_are_millimetres_in_and_metres_out():
    assert mm_to_m(1500) == pytest.approx(1.5)
    position, _ = pose_to_world(Pose(0.0, 0.0, 0.0, "x", "table"), DEFAULT_LAYOUT)
    assert position[2] == pytest.approx(DEFAULT_LAYOUT.table_top_z_m)


def test_axis_quaternions_map_the_cylinder_axis_onto_the_world_axis():
    """A USD cylinder points along its own +Z; these rotations lay it down."""
    for axis in (Axis.X, Axis.Y, Axis.Z):
        quaternion = quaternion_for_axis(axis)
        assert axis_from_quaternion(quaternion) is axis
        assert axis_deviation_deg(quaternion, axis) == pytest.approx(0.0, abs=1e-6)
    assert quaternion_for_axis(Axis.Z) == (1.0, 0.0, 0.0, 0.0)


def test_axis_deviation_folds_end_for_end_symmetry():
    """A cylinder lying along +X and along -X is the same physical placement."""
    flipped = (0.0, 0.0, 1.0, 0.0)          # 180 deg about Y: +Z -> -Z
    assert axis_deviation_deg(flipped, Axis.Z) == pytest.approx(0.0, abs=1e-6)


def test_placement_converts_from_min_corner_to_centre():
    """domain.Placement carries the MIN CORNER; the contract carries the CENTRE.

    A robot is commanded to a centre, never to a corner.
    """
    placement = Placement(item_id="item-001", container_id="CNT-01",
                          position=Vec3(10, 20, 30), axis=Axis.X,
                          size=Vec3(200, 50, 50))
    pose = placement_pose(placement)
    assert (pose.x_mm, pose.y_mm, pose.z_mm) == (110.0, 45.0, 55.0)
    assert pose.frame == "container:CNT-01"
    assert pose.axis == "x"


def test_world_conversion_round_trips_exactly():
    """The inverse must be exact, or a measured pose cannot be compared to a plan."""
    for frame in ("table", "container:CNT-01", "container:CNT-02"):
        original = Pose(123.0, -45.5, 67.25, "y", frame)
        position, quaternion = pose_to_world(original, DEFAULT_LAYOUT)
        restored = world_to_pose(position, quaternion, frame, DEFAULT_LAYOUT)
        assert restored.x_mm == pytest.approx(original.x_mm, abs=1e-6)
        assert restored.y_mm == pytest.approx(original.y_mm, abs=1e-6)
        assert restored.z_mm == pytest.approx(original.z_mm, abs=1e-6)
        assert restored.axis == original.axis
        assert restored.frame == frame


def test_containers_are_laid_out_in_distinct_places():
    first = DEFAULT_LAYOUT.container_origin_for("CNT-01")
    second = DEFAULT_LAYOUT.container_origin_for("CNT-02")
    assert first != second
    assert container_slot("CNT-01") == 0
    assert container_slot("CNT-02") == 1
    # An unparseable id is a naming change, not a reason to refuse to draw.
    assert container_slot("CNT") == 0


def test_the_inner_origin_sits_inside_the_outer_shell():
    outer = DEFAULT_LAYOUT.container_outer_origin_for("CNT-01")
    inner = DEFAULT_LAYOUT.container_origin_for("CNT-01")
    thickness = DEFAULT_LAYOUT.container_wall_thickness_m
    for n in range(3):
        assert inner[n] == pytest.approx(outer[n] + thickness)


def test_an_unknown_frame_is_refused():
    """Guessing would place a physical object somewhere nobody specified."""
    with pytest.raises(ValueError, match="unknown pose frame"):
        pose_to_world(Pose(0, 0, 0, "x", "bin"), DEFAULT_LAYOUT)
    with pytest.raises(ValueError, match="unknown pose frame"):
        world_to_pose((0, 0, 0), (1, 0, 0, 0), "elsewhere", DEFAULT_LAYOUT)


def test_pick_poses_rest_the_item_on_the_table():
    scenario = build_scenario(ISAAC_SMOKE_PRESET, seed=42)
    for index, item in enumerate(scenario.items):
        pose = table_pose_for_index(index, item, DEFAULT_LAYOUT)
        # Centre one radius above the surface, and lying along world X so a
        # top-down gripper closes ACROSS it.
        assert pose.z_mm == pytest.approx(item.outer_diameter_mm / 2.0)
        assert pose.axis == "x"
        assert pose.frame == "table"


def test_pick_slots_do_not_overlap():
    scenario = build_scenario(ISAAC_SMOKE_PRESET, seed=42)
    poses = [table_pose_for_index(n, i, DEFAULT_LAYOUT)
             for n, i in enumerate(scenario.items)]
    widest = max(i.outer_diameter_mm for i in scenario.items)
    ys = sorted(p.y_mm for p in poses)
    for a, b in zip(ys, ys[1:]):
        assert b - a > widest, "two items would be spawned intersecting"


def test_dimensions_come_from_the_domain_item():
    scenario = build_scenario(ISAAC_SMOKE_PRESET, seed=42)
    item = scenario.items[0]
    dims = dimensions_for(item)
    assert dims.length_mm == item.length_mm
    assert dims.outer_diameter_mm == item.outer_diameter_mm
    assert dims.inner_diameter_mm == item.inner_diameter_mm


# --------------------------------------------------------------------------- #
# Reachability
# --------------------------------------------------------------------------- #

def test_the_default_layout_is_reachable_for_the_smoke_scenario():
    """An unreachable goal does not fail loudly in Isaac — the IK converges
    short and the gripper closes on air. So it is checked here instead."""
    scenario = build_scenario(ISAAC_SMOKE_PRESET, seed=42)
    DEFAULT_LAYOUT.validate(scenario.container_template.inner_size,
                            len(scenario.items))


def test_an_out_of_reach_container_is_rejected():
    far = SceneLayout(container_outer_xy_m=(1.4, 1.4))
    with pytest.raises(ValueError, match="not reachable"):
        far.validate(Vec3(300, 220, 150), 4)


def test_a_container_folded_under_the_arm_is_rejected():
    """Both ends of the shell matter: too close has no comfortable IK solution."""
    near = SceneLayout(container_outer_xy_m=(0.0, 0.0))
    with pytest.raises(ValueError, match="not reachable"):
        near.validate(Vec3(300, 220, 150), 4)


# --------------------------------------------------------------------------- #
# Containment verdicts
# --------------------------------------------------------------------------- #

INNER = Vec3(300, 220, 150)


def test_an_item_at_its_planned_pose_is_contained():
    verdict = check_containment(
        Pose(150.0, 110.0, 25.0, "x", "container:CNT-01"), INNER, 200, 50)
    assert verdict.ok


def test_an_item_outside_the_footprint_is_not_contained():
    verdict = check_containment(
        Pose(600.0, 110.0, 25.0, "x", "container:CNT-01"), INNER, 200, 50)
    assert not verdict.ok and not verdict.inside_footprint
    assert "footprint" in verdict.detail


def test_an_item_resting_on_the_rim_is_not_contained():
    verdict = check_containment(
        Pose(150.0, 110.0, 900.0, "x", "container:CNT-01"), INNER, 200, 50)
    assert not verdict.ok and not verdict.below_rim_overflow
    assert "resting on top" in verdict.detail


def test_an_item_below_the_floor_is_not_contained():
    verdict = check_containment(
        Pose(150.0, 110.0, -500.0, "x", "container:CNT-01"), INNER, 200, 50)
    assert not verdict.ok and not verdict.above_floor


def test_an_item_that_left_the_scene_is_not_contained():
    verdict = check_containment(
        Pose(1e9, 1e9, 1e9, "x", "container:CNT-01"), INNER, 200, 50)
    assert not verdict.ok


def test_containment_tolerates_a_settled_item_leaning_on_a_wall():
    """A drop is not a placement to the millimetre; leaning on a wall is in."""
    verdict = check_containment(
        Pose(-12.0, 110.0, 25.0, "x", "container:CNT-01"), INNER, 200, 50)
    assert verdict.ok


def test_a_release_clear_of_the_walls_is_not_moved():
    """The rule only rescues a release that would happen against a wall."""
    from wisepack_core.domain import Vec3
    from wisepack_core.isaac_contract import Dimensions, Pose
    from wisepack_core.isaac_transform import ReleaseClearance, safe_release_pose

    inner = Vec3(300, 220, 150)
    dims = Dimensions(length_mm=150, outer_diameter_mm=50, inner_diameter_mm=40)
    middle = Pose(x_mm=150.0, y_mm=110.0, z_mm=25.0, axis="x", frame="c")
    moved_pose, moved = safe_release_pose(middle, dims, inner)
    assert moved == pytest.approx(0.0, abs=1e-6)
    assert (moved_pose.x_mm, moved_pose.y_mm) == (150.0, 110.0)


def test_a_flush_release_is_pulled_into_the_interior():
    """A dense plan puts items against the wall; releasing there clips the rim."""
    from wisepack_core.domain import Vec3
    from wisepack_core.isaac_contract import Dimensions, Pose
    from wisepack_core.isaac_transform import ReleaseClearance, safe_release_pose

    inner = Vec3(300, 220, 150)
    dims = Dimensions(length_mm=150, outer_diameter_mm=50, inner_diameter_mm=40)
    clearance = ReleaseClearance(wall_mm=10.0, object_mm=8.0)
    flush = Pose(x_mm=75.0, y_mm=25.0, z_mm=25.0, axis="x", frame="c")
    pose, moved = safe_release_pose(flush, dims, inner, clearance=clearance)
    assert moved > 0.0
    # Footprint clears both walls: half-length 75 in x, radius 25 in y.
    assert pose.x_mm >= 75.0 + 10.0 - 1e-6
    assert pose.y_mm >= 25.0 + 10.0 - 1e-6
    assert pose.x_mm <= 300 - 75.0 - 10.0 + 1e-6
    # The plan is untouched: only z and axis carry through unchanged.
    assert pose.z_mm == flush.z_mm and pose.axis == flush.axis


def test_releases_are_kept_apart_from_what_is_already_down():
    from wisepack_core.domain import Vec3
    from wisepack_core.isaac_contract import Dimensions, Pose
    from wisepack_core.isaac_transform import ReleaseClearance, safe_release_pose

    inner = Vec3(300, 220, 150)
    dims = Dimensions(length_mm=100, outer_diameter_mm=50, inner_diameter_mm=40)
    clearance = ReleaseClearance(wall_mm=10.0, object_mm=8.0)
    planned = Pose(x_mm=150.0, y_mm=110.0, z_mm=25.0, axis="x", frame="c")
    pose, _ = safe_release_pose(planned, dims, inner, clearance=clearance,
                                occupied=[(150.0, 110.0)])
    assert (pose.x_mm, pose.y_mm) != (150.0, 110.0), \
        "a second object must not be released onto the first"


def test_an_object_too_long_to_clear_both_walls_is_centred():
    """Better centred than clamped hard against one wall."""
    from wisepack_core.domain import Vec3
    from wisepack_core.isaac_contract import Dimensions, Pose
    from wisepack_core.isaac_transform import safe_release_pose

    inner = Vec3(300, 220, 150)
    huge = Dimensions(length_mm=295, outer_diameter_mm=50, inner_diameter_mm=40)
    planned = Pose(x_mm=10.0, y_mm=110.0, z_mm=25.0, axis="x", frame="c")
    pose, _ = safe_release_pose(planned, huge, inner)
    assert pose.x_mm == pytest.approx(150.0)


def test_the_release_point_never_changes_the_reported_target():
    """The error must stay measured against the PLAN, not the release point."""
    src = _read(os.path.join(REPO, "simulators", "isaac", "robot.py"))
    finish = src[src.index("    def _finish_item("):]
    end = finish.find("\n    def ", 10)
    finish = finish if end < 0 else finish[:end]
    assert "target=command.target_pose" in finish, (
        "the settled pose must be compared with the planned pose; comparing it "
        "with the adjusted release point would flatter the metric")
    assert "_release_pose" not in finish


# --------------------------------------------------------------------------- #
# Physical state -> workflow stage
# --------------------------------------------------------------------------- #

def test_every_isaac_state_maps_onto_the_existing_workflow():
    """Exhaustive. A state added to the contract and forgotten in the map would
    render as whatever the previous stage was — a timeline that stops moving."""
    for state in IsaacState:
        assert state in ISAAC_STATE_STAGE, f"{state.value} has no stage mapping"
    mapped = {s for s in ISAAC_STATE_STAGE.values() if s is not None}
    assert mapped <= set(Stage), "a mapping invented a stage the workflow lacks"


def test_ready_does_not_move_the_workflow_stage():
    """READY is about the SIMULATOR. The plan may still be awaiting approval,
    and advancing the stage there would show a pick before authorisation."""
    assert stage_for_isaac_state(IsaacState.READY) is None


def test_physical_states_map_to_the_stage_that_means_the_same_thing():
    assert stage_for_isaac_state(IsaacState.GRASPING) is Stage.PICK_ITEM
    assert stage_for_isaac_state(IsaacState.LIFTING) is Stage.VERIFY_PICK
    assert stage_for_isaac_state(IsaacState.RELEASING) is Stage.PLACE_ITEM
    # Settling is where physics decides the placement, not the plan.
    assert stage_for_isaac_state(IsaacState.SETTLING) is Stage.VERIFY_PLACEMENT
    assert stage_for_isaac_state(IsaacState.ITEM_COMPLETED) is Stage.UPDATE_CONTAINER_STATE
    assert stage_for_isaac_state(IsaacState.RUN_COMPLETED) is Stage.COMPLETE
    assert stage_for_isaac_state(IsaacState.RUN_FAILED) is Stage.DEGRADED


def test_robot_state_uses_the_existing_vocabulary():
    assert robot_state_for_isaac_state(IsaacState.GRASPING) == "picking"
    assert robot_state_for_isaac_state(IsaacState.RELEASING) == "placing"
    assert robot_state_for_isaac_state(IsaacState.READY) == "idle"


# --------------------------------------------------------------------------- #
# Driving the workflow with a FAKE Isaac
# --------------------------------------------------------------------------- #

def _approved_engine(backend=ExecutionBackend.ISAAC) -> WorkflowEngine:
    engine = WorkflowEngine(WorkflowConfig(
        preset=ISAAC_SMOKE_PRESET, seed=42, execution_backend=backend))
    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    engine.generate_plans()
    engine.digital_twin_validate()
    engine.request_approval()
    engine.approve(auto=True)
    return engine


class FakeIsaac:
    """A simulator that always succeeds, driven through the real contract.

    Every message it produces is a genuine ``IsaacFeedback`` that has been
    serialised and parsed, so this exercises the wire format as well as the
    workflow integration.
    """

    def __init__(self, engine: WorkflowEngine) -> None:
        self.engine = engine
        self.gate = RunGate()
        self.gate.adopt(engine.run_id)
        self.states_seen = []

    def _feed(self, state: IsaacState, command: IsaacCommand, **kw):
        message = IsaacFeedback.from_json(IsaacFeedback(
            state=state, run_id=self.gate.run_id, item_id=command.item_id,
            sequence_index=command.sequence_index,
            container_id=command.container_id,
            target_pose=command.target_pose, **kw).to_json())
        self.states_seen.append(message.state)
        return message

    def execute(self, command: IsaacCommand, *, error_mm: float = 12.0):
        placement = self.engine.selected.placement_for_item(command.item_id)
        self.engine.begin_physical_item(placement)
        for state in (IsaacState.MOVING_TO_PICK, IsaacState.GRASPING,
                      IsaacState.LIFTING, IsaacState.MOVING_TO_CONTAINER,
                      IsaacState.RELEASING, IsaacState.SETTLING):
            feedback = self._feed(state, command)
            self.engine.note_physical_progress(
                stage_for_isaac_state(feedback.state), f"isaac_{state.value.lower()}",
                feedback.item_id, feedback.container_id, feedback.message,
                robot_state=robot_state_for_isaac_state(feedback.state))
        # A MEASURED landing, offset from the plan — as a real drop is.
        target = command.target_pose
        actual = Pose(target.x_mm + error_mm, target.y_mm, target.z_mm,
                      target.axis, target.frame)
        feedback = self._feed(IsaacState.ITEM_COMPLETED, command,
                              actual_pose=actual,
                              position_error_mm=error_mm)
        self.gate.mark_done(command.sequence_index)
        return self.engine.complete_physical_item(
            placement, details={"position_error_mm": feedback.position_error_mm,
                                "actual_pose": feedback.actual_pose.to_dict()})


def _command_for(engine, placement, index) -> IsaacCommand:
    item = engine.scenario.item(placement.item_id)
    container = engine.selected.container(placement.container_id)
    return IsaacCommand.from_json(IsaacCommand(
        command=IsaacCommandType.EXECUTE_ITEM, run_id=engine.run_id,
        sequence_index=index, item_id=item.item_id,
        dimensions=dimensions_for(item),
        source_pose=table_pose_for_index(
            [i.item_id for i in engine.scenario.items].index(item.item_id), item),
        target_pose=placement_pose(placement),
        container_id=container.container_id,
        container_inner_mm=container.inner_size.to_dict()).to_json())


def test_a_full_physical_run_completes_the_plan():
    engine = _approved_engine()
    isaac = FakeIsaac(engine)
    guard = 0
    while not engine.finished and guard < 50:
        guard += 1
        nxt = engine.next_physical_placement()
        if nxt is None:
            break
        placement, _, _ = nxt
        isaac.execute(_command_for(engine, placement, engine.cursor.index))

    assert engine.finished
    assert engine.stage is Stage.COMPLETE
    assert engine.progress_pct == pytest.approx(100.0)
    assert all(p.executed for p in engine.selected.placements)
    assert engine.stats.cycles_completed == len(engine.selected.placements)


def test_physical_execution_produces_the_same_shape_of_audit_trail():
    """The backend changes WHO moved the item, not what the record looks like."""
    engine = _approved_engine()
    isaac = FakeIsaac(engine)
    nxt = engine.next_physical_placement()
    isaac.execute(_command_for(engine, nxt[0], 0))

    ok, why = engine.log.sequence_is_monotonic()
    assert ok, why
    actors = {e.actor for e in engine.log.events()}
    assert Actor.ISAAC_SIM in actors, "physical events must be attributed to Isaac"
    # ...and distinctly from the simulated robot, whose outcomes are coin flips.
    assert Actor.ROBOT_SIM not in actors

    stages = [e.stage for e in engine.log.events()]
    for stage in (Stage.PICK_ITEM, Stage.VERIFY_PICK, Stage.PLACE_ITEM,
                  Stage.VERIFY_PLACEMENT, Stage.UPDATE_CONTAINER_STATE):
        assert stage in stages, f"{stage.value} missing from the physical trail"


def test_the_measured_pose_and_its_error_reach_the_audit_trail():
    """Never claim a dropped item reproduced the plan exactly."""
    engine = _approved_engine()
    isaac = FakeIsaac(engine)
    nxt = engine.next_physical_placement()
    isaac.execute(_command_for(engine, nxt[0], 0), error_mm=17.5)

    settled = [e for e in engine.log.events() if e.action == "isaac_item_settled"]
    assert settled, "no settle event was recorded"
    assert settled[-1].details["position_error_mm"] == pytest.approx(17.5)
    assert settled[-1].details["actual_pose"]["position_mm"]["x"] != \
        settled[-1].details.get("target_x")


def test_a_physical_failure_retries_then_abandons_like_the_simulated_backend():
    engine = _approved_engine()
    placement = engine.next_physical_placement()[0]
    budget = engine.config.robot.max_pick_retries

    for _ in range(budget):
        engine.begin_physical_item(placement)
        engine.fail_physical_item(placement, "gripper closed on air")
        assert not placement.executed, "an item must not be abandoned early"
    engine.begin_physical_item(placement)
    engine.fail_physical_item(placement, "gripper closed on air")
    # Same retry budget as the simulated backend, so KPI2/KPI3 stay comparable.
    assert placement.executed, "the item should be abandoned after the budget"
    assert engine.stats.cycles_attempted >= 1
    assert engine.stats.cycles_completed == 0


def test_physical_execution_is_refused_before_approval():
    """The safety invariant is backend-independent and is CHECKED, not trusted."""
    engine = WorkflowEngine(WorkflowConfig(
        preset=ISAAC_SMOKE_PRESET, seed=42,
        execution_backend=ExecutionBackend.ISAAC))
    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    engine.generate_plans()
    engine.digital_twin_validate()
    engine.request_approval()

    with pytest.raises(ApprovalRequired):
        engine.next_physical_placement()
    placement = engine.selected.ordered_placements[0]
    with pytest.raises(ApprovalRequired):
        engine.begin_physical_item(placement)
    with pytest.raises(ApprovalRequired):
        engine.complete_physical_item(placement)


def test_the_container_closes_when_its_last_item_settles():
    engine = _approved_engine()
    isaac = FakeIsaac(engine)
    guard = 0
    while not engine.finished and guard < 50:
        guard += 1
        nxt = engine.next_physical_placement()
        if nxt is None:
            break
        isaac.execute(_command_for(engine, nxt[0], engine.cursor.index))
    used = engine.selected.containers_used
    assert used
    assert all(c.status is ContainerStatus.COMPLETE for c in used)
    placed = [i for i in engine.scenario.items if i.status is ItemStatus.PLACED]
    assert len(placed) == len(engine.selected.placements)


# --------------------------------------------------------------------------- #
# The simulated backend must be untouched
# --------------------------------------------------------------------------- #

def test_the_simulated_backend_still_runs_unchanged():
    """The whole feature is additive. This is the regression guard for that."""
    from wisepack_core.workflow import run_headless
    engine = run_headless(WorkflowConfig(preset="mixed_pipes_small", seed=42))
    assert engine.finished
    assert engine.stage is Stage.COMPLETE
    actors = {e.actor for e in engine.log.events()}
    assert Actor.ROBOT_SIM in actors
    assert Actor.ISAAC_SIM not in actors, \
        "a simulated run must never be attributed to the physical backend"


def test_the_simulated_backend_is_deterministic_across_the_change():
    a = build_scenario("mixed_pipes_dense", seed=42)
    b = build_scenario("mixed_pipes_dense", seed=42)
    assert json.dumps(a.to_dict(), sort_keys=True) == \
           json.dumps(b.to_dict(), sort_keys=True)


def test_the_smoke_preset_does_not_disturb_the_benchmark_presets():
    """The Isaac scenario must not change the measured optimization result."""
    for preset in ("mixed_pipes_dense", "curated_volume_reduction"):
        scenario = build_scenario(preset, seed=42)
        plan = pack_optimized(scenario, config=OptimizerConfig(seed=42, restarts=6))
        PlacementValidator().validate_plan(plan, scenario)
        assert plan.is_valid
        # Full axis freedom is retained everywhere except the Isaac preset.
        for item in scenario.items:
            assert set(item.permitted_axes) == {Axis.X, Axis.Y, Axis.Z}


# --------------------------------------------------------------------------- #
# The smoke scenario
# --------------------------------------------------------------------------- #

def test_the_smoke_preset_is_registered_and_small():
    assert ISAAC_SMOKE_PRESET in PRESETS
    scenario = build_scenario(ISAAC_SMOKE_PRESET, seed=42)
    assert 3 <= len(scenario.items) <= 5, \
        "a first robotic integration must not run a 40-item benchmark"


def test_the_smoke_items_fit_a_panda_gripper():
    """80 mm jaw opening. A larger pipe cannot be grasped at all, and the run
    would fail for a reason that has nothing to do with WISEPACK."""
    scenario = build_scenario(ISAAC_SMOKE_PRESET, seed=42)
    for item in scenario.items:
        assert item.outer_diameter_mm <= 70


def test_the_smoke_items_avoid_the_unreachable_vertical_axis():
    """A top-down parallel gripper cannot stand a pipe on its end."""
    scenario = build_scenario(ISAAC_SMOKE_PRESET, seed=42)
    for item in scenario.items:
        assert Axis.Z not in item.permitted_axes
        assert set(item.permitted_axes) == {Axis.X, Axis.Y}


@pytest.mark.parametrize("seed", [7, 42, 1234])
def test_the_smoke_scenario_plans_into_one_container(seed):
    scenario = build_scenario(ISAAC_SMOKE_PRESET, seed=seed)
    plan = pack_optimized(scenario, config=OptimizerConfig(seed=seed, restarts=6))
    PlacementValidator().validate_plan(plan, scenario)
    assert plan.is_valid
    assert not plan.unplaced_item_ids
    assert plan.containers_required == 1


@pytest.mark.parametrize("seed", [7, 42, 1234])
def test_every_smoke_placement_converts_to_a_reachable_world_pose(seed):
    scenario = build_scenario(ISAAC_SMOKE_PRESET, seed=seed)
    plan = pack_optimized(scenario, config=OptimizerConfig(seed=seed, restarts=6))
    PlacementValidator().validate_plan(plan, scenario)
    for placement in plan.ordered_placements:
        position, _ = pose_to_world(placement_pose(placement), DEFAULT_LAYOUT)
        reach = math.hypot(position[0], position[1])
        assert DEFAULT_LAYOUT.robot_min_reach_m <= reach <= DEFAULT_LAYOUT.robot_max_reach_m


# --------------------------------------------------------------------------- #
# Topic contract
# --------------------------------------------------------------------------- #

def _topics():
    spec = importlib.util.spec_from_file_location(
        "wp_topics_isaac",
        os.path.join(REPO, "wisepack_ws", "src", "wisepack_bringup",
                     "wisepack_bringup", "topics.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_isaac_channel_uses_bridgeable_standard_messages():
    """Standard scalar std_msgs only — the rule the whole contract rests on."""
    topics = _topics()
    for topic, msg_type in topics.isaac_topics().items():
        assert msg_type == "std_msgs/String", (
            f"{topic} uses {msg_type}; a custom message would have to be built "
            "and then imported by Isaac's bundled interpreter")
        assert topic.startswith("/wisepack/")


def test_no_topic_uses_the_reserved_status_leaf():
    assert _topics().reserved_leaf_violations() == []


def test_the_execution_backend_topic_is_part_of_the_always_present_contract():
    topics = _topics()
    assert topics.EXECUTION_BACKEND in topics.all_topics()
    assert topics.all_topics()[topics.EXECUTION_BACKEND] == "std_msgs/String"


def test_the_isaac_transport_is_not_in_the_always_present_contract():
    """It has no publisher in an ordinary simulated run, and the live QoS test
    asserts every contract topic HAS one."""
    topics = _topics()
    for topic in topics.isaac_topics():
        assert topic not in topics.all_topics()


def test_each_isaac_topic_has_exactly_one_declared_writer():
    topics = _topics()
    assert set(topics.ISAAC_WRITERS) == set(topics.isaac_topics())
    assert topics.ISAAC_WRITERS[topics.ISAAC_COMMAND] == "wisepack_orchestration"
    assert topics.ISAAC_WRITERS[topics.ISAAC_FEEDBACK] == "isaac_sim"


def test_the_backend_attribute_is_mapped_into_fiware():
    """Recording a pick without recording WHICH backend produced it cannot
    support either claim."""
    yaml = pytest.importorskip("yaml")
    path = os.path.join(REPO, "wisepack_ws", "src", "wisepack_fiware", "config",
                        "bridge_config.yaml")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    mapped = {m["ros_topic"]: m for m in cfg["ros_to_fiware"]}
    topics = _topics()
    assert topics.EXECUTION_BACKEND in mapped
    assert mapped[topics.EXECUTION_BACKEND]["fiware_attribute"] == "executionBackend"
    # The raw transport is deliberately NOT bridged: the audit-relevant facts
    # already travel as ActionEvents on /wisepack/action/event.
    for topic in topics.isaac_topics():
        assert topic not in mapped


# --------------------------------------------------------------------------- #
# Adapter isolation
# --------------------------------------------------------------------------- #

#: Test files that legitimately contain the marker strings because they are the
#: ones doing the searching.
_IMPORT_SCANNERS = {"test_isaac_backend.py", "test_simulator_view.py"}


def test_simulator_imports_are_confined_to_the_adapter():
    """Isaac Sim is one implementation behind the contract, not a dependency of
    it. If `isaacsim` leaks outside simulators/isaac/, the ordinary test suite
    and the orchestrator start needing a GPU."""
    offenders = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs
                   if d not in (".git", "build", "install", "log", "__pycache__")]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            relative = os.path.relpath(path, REPO)
            if relative.startswith(os.path.join("simulators", "isaac")):
                continue
            # The scanner tests necessarily NAME the markers they search for.
            # An explicit allowlist rather than "skip tests/", so a real leak
            # into an ordinary test is still caught.
            if os.path.basename(relative) in _IMPORT_SCANNERS:
                continue
            with open(path, encoding="utf-8") as fh:
                source = fh.read()
            for marker in ("import isaacsim", "from isaacsim",
                           "from pxr import", "import omni.", "import carb"):
                if marker in source:
                    offenders.append(f"{relative}: {marker}")
    assert not offenders, (
        "simulator-specific imports outside the Isaac adapter: " + str(offenders))


def test_the_orchestration_bridge_has_no_simulator_imports():
    """A real robot cell answering the same topics must be a drop-in swap."""
    path = os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                        "wisepack_orchestration", "isaac_bridge.py")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    for marker in ("isaacsim", "pxr", "omni.", "carb"):
        assert marker not in source, (
            f"the orchestrator-side bridge imports {marker}; it must speak only "
            "the ROS contract")


# --------------------------------------------------------------------------- #
# Isaac adapter modules that carry no Isaac dependency
# --------------------------------------------------------------------------- #

# Imported as a PACKAGE, not by file path. `simulators/isaac/__init__.py`
# imports nothing, so this reaches config and result without pulling isaacsim
# into a test run that must work on a machine with no GPU.
from simulators.isaac import config as isaac_config          # noqa: E402
from simulators.isaac import result as isaac_result          # noqa: E402


def test_motion_configuration_defaults_are_self_consistent():
    isaac_config.MotionConfig().validate()


def test_a_clearance_below_the_drop_height_is_rejected():
    bad = isaac_config.MotionConfig(container_clearance=0.01, drop_height=0.06)
    with pytest.raises(ValueError, match="container_clearance"):
        bad.validate()


def test_releasing_at_the_exact_target_is_rejected():
    with pytest.raises(ValueError, match="drop_height"):
        isaac_config.MotionConfig(drop_height=0.0).validate()


def test_a_lift_that_never_clears_the_pre_grasp_height_is_rejected():
    with pytest.raises(ValueError, match="lift_height"):
        isaac_config.MotionConfig(lift_height=0.05, pre_grasp_height=0.12).validate()


def test_the_adapter_config_module_needs_no_simulator():
    """Proof, not assumption: this module was imported above with no Isaac."""
    assert isaac_config.MotionConfig().settle_timeout > 0
    assert "isaacsim" not in str(isaac_config.__dict__.keys())


def test_settling_requires_a_stable_interval_not_one_quiet_sample():
    """A cylinder is instantaneously motionless at the top of every bounce."""
    result = isaac_result
    monitor = result.SettleMonitor(linear_threshold=0.01, angular_threshold=0.1,
                                   stable_time=0.5, timeout=6.0)
    monitor.start(0.0)
    assert monitor.update(0.1, [0, 0, 0], [0, 0, 0]) == (False, False)
    # An excursion restarts the clock.
    assert monitor.update(0.3, [0, 0, 1.0], [0, 0, 0]) == (False, False)
    assert monitor.update(0.5, [0, 0, 0], [0, 0, 0]) == (False, False)
    assert monitor.update(1.1, [0, 0, 0], [0, 0, 0]) == (True, False)


def test_settling_times_out_on_an_item_that_never_rests():
    result = isaac_result
    monitor = result.SettleMonitor(0.01, 0.1, 0.35, 2.0)
    monitor.start(0.0)
    settled, timed_out = monitor.update(2.5, [0, 0, 5.0], [0, 0, 0])
    assert not settled and timed_out


def test_a_settled_contained_item_passes_evaluation():
    result = isaac_result
    target = Pose(150.0, 110.0, 25.0, "x", "container:CNT-01")
    actual = Pose(162.0, 110.0, 25.0, "x", "container:CNT-01")
    outcome = result.evaluate_placement(
        actual=actual, target=target, actual_quaternion=quaternion_for_axis(Axis.X),
        container_inner=INNER, length_mm=200, diameter_mm=50,
        settled=True, timed_out=False, settle_detail={})
    assert outcome.ok
    assert outcome.position_error_mm == pytest.approx(12.0)
    assert outcome.axis_error_deg == pytest.approx(0.0, abs=1e-6)


def test_an_item_that_left_the_bin_fails_evaluation():
    result = isaac_result
    outcome = result.evaluate_placement(
        actual=Pose(900.0, 110.0, 25.0, "x", "container:CNT-01"),
        target=Pose(150.0, 110.0, 25.0, "x", "container:CNT-01"),
        actual_quaternion=quaternion_for_axis(Axis.X), container_inner=INNER,
        length_mm=200, diameter_mm=50, settled=True, timed_out=False,
        settle_detail={})
    assert not outcome.ok and outcome.reasons


def test_a_missing_item_is_a_distinct_failure_from_a_bad_landing():
    result = isaac_result
    outcome = result.evaluate_placement(
        actual=None, target=Pose(150.0, 110.0, 25.0, "x", "container:CNT-01"),
        actual_quaternion=None, container_inner=INNER, length_mm=200,
        diameter_mm=50, settled=False, timed_out=False, settle_detail={})
    assert not outcome.ok
    assert "no longer present" in outcome.message


def test_a_settle_timeout_on_a_contained_item_is_a_note_not_a_failure():
    """An item wedged against a wall keeps a residual velocity and is, in every
    sense that matters, in the container. It is recorded, not failed."""
    result = isaac_result
    outcome = result.evaluate_placement(
        actual=Pose(150.0, 110.0, 25.0, "x", "container:CNT-01"),
        target=Pose(150.0, 110.0, 25.0, "x", "container:CNT-01"),
        actual_quaternion=quaternion_for_axis(Axis.X), container_inner=INNER,
        length_mm=200, diameter_mm=50, settled=False, timed_out=True,
        settle_detail={})
    assert outcome.ok
    assert not outcome.reasons
    assert any("settle_timeout" in note for note in outcome.notes)
    assert outcome.detail["settle_timed_out"] is True


# --------------------------------------------------------------------------- #
# Dashboard read model
# --------------------------------------------------------------------------- #

def _snapshot_module():
    """web/snapshot.py imports no ROS and no FastAPI, so it tests directly.

    Registered in ``sys.modules`` BEFORE it is executed. ``@dataclass`` resolves
    string annotations by looking its own module up there, and a module loaded
    from a file spec without registration is absent — which fails inside
    dataclasses with an unrelated-looking AttributeError.
    """
    import sys
    name = "wp_snapshot"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, "web", "snapshot.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeState:
    def __init__(self, mirror=None, engine=None):
        import threading
        self.lock = threading.RLock()
        self.engine = engine
        self.events = []
        self.notice = ""
        self.auto_step = False
        self.ros_mirror = mirror
        self.fiware_connected = None
        self.fiware_last_error = ""


def test_the_dashboard_does_not_claim_a_backend_it_has_not_heard_from():
    """An unknown backend and a known-simulated one are different states.

    Defaulting to "SIMULATED" would be a claim about a run the dashboard has not
    heard from — and the same defaulting would mislabel a physical run.
    """
    snapshot = _snapshot_module()
    state = _FakeState(mirror={"stage": "IDLE"})
    badge = snapshot.RosSnapshotProvider(state).snapshot().backend_badge()
    assert badge["known"] is False
    assert badge["physical"] is False


def test_the_dashboard_reports_isaac_when_the_orchestrator_says_so():
    snapshot = _snapshot_module()
    state = _FakeState(mirror={
        "stage": "PLACE_ITEM",
        "execution_backend": {
            "backend": "isaac", "label": "ISAAC SIM / PHYSICS",
            "detail": "physical", "physical": True,
            "isaac": {"simulator_ready": True, "last_state": "SETTLING"}},
        "isaac_results": [{"item_id": "item-001", "state": "ITEM_COMPLETED",
                           "position_error_mm": 103.9}],
    })
    snap = snapshot.RosSnapshotProvider(state).snapshot()
    badge = snap.backend_badge()
    assert badge["known"] is True and badge["physical"] is True
    assert badge["label"] == "ISAAC SIM / PHYSICS"
    assert snap.isaac["simulator_ready"] is True
    assert snap.isaac_results[0]["position_error_mm"] == pytest.approx(103.9)
    # The execution badge travels alongside the SOURCE badge, never instead of
    # it: they answer different questions.
    payload = snap.to_state()
    assert "execution" in payload and "badge" in payload
    assert payload["badge"]["source"] == "ros"


def test_sim_mode_never_claims_physical_execution():
    """The presentation mode has no ROS, no FIWARE and no simulator."""
    snapshot = _snapshot_module()
    engine = WorkflowEngine(WorkflowConfig(preset="mixed_pipes_small", seed=1))
    engine.generate_or_load_scenario()
    badge = snapshot.SimSnapshotProvider(
        _FakeState(engine=engine)).snapshot().backend_badge()
    assert badge["backend"] == "simulated"
    assert badge["physical"] is False


def test_the_frontend_shows_a_separate_execution_badge():
    with open(os.path.join(REPO, "web", "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    assert 'id="b-execution"' in html
    assert 'id="b-source"' in html, "the source badge must remain"
    # The physics badge must not simply reuse the "live data" styling.
    assert ".badge.phys{" in html


def test_the_physical_diagnostics_panel_is_hidden_without_a_physical_backend():
    """An empty 'Physical execution' table in a simulated run would imply a
    physical run that produced nothing."""
    with open(os.path.join(REPO, "web", "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    assert 'id="physpanel" style="display:none"' in html
    assert "if (!doc || !doc.physical) { box.style.display = \"none\"; return; }" in html
    # Target and measured are always presented as a pair.
    assert "planned pose" in html and "measured pose" in html


# --------------------------------------------------------------------------- #
# Launcher option parsing
# --------------------------------------------------------------------------- #

DASHBOARD = os.path.join(REPO, "run_wisepack_dashboard.sh")
DEMO = os.path.join(REPO, "run_wisepack_demo.sh")
ISAAC_LAUNCHER = os.path.join(REPO, "scripts", "run_wisepack_isaac.sh")
VALIDATOR = os.path.join(REPO, "scripts", "validate_isaac_sim.sh")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _code(path: str) -> str:
    """Script text with comment lines removed.

    These launchers deliberately QUOTE the dangerous forms they avoid — a
    `pkill -f isaac` pattern, sourcing the host ROS environment — so the next
    reader understands what the code is protecting against. That documentation
    must not fail its own regression test. Same technique the existing
    test_launchers.py uses for the same reason.
    """
    return "\n".join(line for line in _read(path).splitlines()
                     if not line.lstrip().startswith("#"))


@pytest.mark.parametrize("script", [DASHBOARD, DEMO, ISAAC_LAUNCHER, VALIDATOR])
def test_the_new_scripts_exist_and_are_executable(script):
    assert os.path.isfile(script), f"{script} is missing"
    assert os.access(script, os.X_OK), f"{script} is not executable"


@pytest.mark.parametrize("script", [ISAAC_LAUNCHER, VALIDATOR])
def test_the_new_scripts_are_syntactically_valid(script):
    result = subprocess.run(["bash", "-n", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_dashboard_rejects_an_unknown_mode():
    result = subprocess.run([DASHBOARD, "nonsense"], capture_output=True,
                            text=True, cwd=REPO, timeout=60)
    assert result.returncode == 2
    assert "unknown mode" in result.stderr
    assert "isaac" in result.stderr


def test_the_dashboard_help_documents_every_mode():
    result = subprocess.run([DASHBOARD, "--help"], capture_output=True,
                            text=True, cwd=REPO, timeout=60)
    assert result.returncode == 0
    for mode in ("isaac", "isaac-fiware", "sim", "fiware"):
        assert mode in result.stdout, f"--help does not document {mode}"
    assert "WISEPACK_ISAAC_HEADLESS" in result.stdout
    assert "ISAAC_SIM_ROOT" in result.stdout


def test_the_demo_help_documents_the_isaac_option():
    result = subprocess.run([DEMO, "--help"], capture_output=True, text=True,
                            cwd=REPO, timeout=60)
    assert result.returncode == 0
    assert "--isaac-sim" in result.stdout
    assert "--isaac-sim --no-fiware" in result.stdout


def test_the_demo_rejects_an_unknown_option():
    result = subprocess.run([DEMO, "--bogus"], capture_output=True, text=True,
                            cwd=REPO, timeout=60)
    assert result.returncode == 2
    assert "--isaac-sim" in result.stderr


def test_core_only_and_isaac_are_mutually_exclusive():
    """--core-only promises no Docker and no ROS; Isaac execution needs both."""
    result = subprocess.run([DEMO, "--core-only", "--isaac-sim"],
                            capture_output=True, text=True, cwd=REPO, timeout=60)
    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr


def test_isaac_modes_select_the_isaac_backend():
    text = _read(DASHBOARD)
    assert 'EXECUTION_BACKEND="isaac"' in text
    assert "isaac|isaac-fiware) EXECUTION_BACKEND=" in text
    assert 'execution_backend:="${WISEPACK_EXECUTION_BACKEND}"' in text


def test_sim_mode_is_not_repurposed():
    """`sim` must remain presentation-only: no ROS, no FIWARE, no simulator."""
    text = _read(DASHBOARD)
    sim_block = text[text.index('if [ "$MODE" = "sim" ]'):
                     text.index("# ── Refuse to run alongside")]
    assert "isaac" not in sim_block.lower()
    assert "--source sim" in sim_block


def test_isaac_modes_keep_ros_as_the_data_source():
    """Isaac is an execution backend, not a fourth data source."""
    text = _read(DASHBOARD)
    assert 'if [ "$MODE" = "fiware" ] || [ "$MODE" = "isaac-fiware" ]' in text
    # `isaac` alone must NOT switch the source away from ros.
    assert 'MODE" = "isaac" ]; then\n    SOURCE="fiware"' not in text


def test_the_isaac_smoke_preset_is_the_physical_default():
    text = _read(DASHBOARD)
    assert 'PRESET="${WISEPACK_PRESET:-isaac_cylinders_smoke}"' in text, \
        "a physical run must not default to the 40-item packing benchmark"
    assert 'PRESET="${WISEPACK_PRESET:-mixed_pipes_dense}"' in text, \
        "the non-physical modes must keep the benchmark preset"


def test_the_launcher_passes_the_required_environment_through():
    text = _read(DASHBOARD)
    for variable in ("ROS_DOMAIN_ID", "WISEPACK_PRESET", "WISEPACK_SEED",
                     "WISEPACK_RESULTS_DIR"):
        assert variable in text, f"{variable} is not passed to Isaac"


def test_the_launcher_verifies_the_bundled_python_before_starting():
    text = _read(ISAAC_LAUNCHER)
    assert '-x "$path/python.sh"' in text
    assert '! -x "$ISAAC_ROOT/python.sh"' in text


def test_the_launcher_prefers_6_0_1_and_never_silently_downgrades():
    text = _read(ISAAC_LAUNCHER)
    assert "isaac-sim-6.0.1" in text
    assert "sort -Vr" in text, "candidates must be searched newest first"
    assert "this backend is written against Isaac Sim 6.0.1" in text


def test_terminality_is_a_property_of_the_message_not_the_enum():
    """Regression, found end to end and invisible to every static check.

    ``is_item_terminal`` / ``is_run_terminal`` belong to IsaacFeedback. The
    orchestrator bridge read them off the bare IsaacState enum, which raises
    AttributeError — and since that branch sits ahead of the progress handler,
    EVERY non-READY report failed to apply. The orchestrator watched the
    simulator come up, heard nothing further, and timed out items the arm had
    actually completed.
    """
    feedback = IsaacFeedback(state=IsaacState.ITEM_COMPLETED, run_id="r",
                             item_id="item-001")
    assert feedback.is_item_terminal is True
    assert feedback.is_run_terminal is False
    for name in ("is_item_terminal", "is_run_terminal"):
        assert not hasattr(IsaacState.ITEM_COMPLETED, name), (
            f"IsaacState must NOT expose {name}; reading it off the enum is the "
            "bug this test pins")

    source = _read(os.path.join(
        REPO, "wisepack_ws", "src", "wisepack_orchestration",
        "wisepack_orchestration", "isaac_bridge.py"))
    assert "feedback.is_run_terminal" in source
    assert "feedback.is_item_terminal" in source
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "state.is_run_terminal" not in code
    assert "state.is_item_terminal" not in code


def test_the_orchestrator_sends_the_retry_counter_as_the_attempt():
    """Otherwise the simulator's duplicate guard eats every retry."""
    path = os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                        "wisepack_orchestration", "isaac_bridge.py")
    source = _read(path)
    assert "attempt=engine.cursor.retries" in source
    assert "self.gate.mark_done(index, attempt)" in source


def test_the_launcher_runs_isaac_unbuffered():
    """The READY gate and the smoke markers are read as a STREAM.

    Python block-buffers stdout to a pipe, so without this both callers wait out
    their full timeout on a simulator that is working perfectly.
    """
    assert "PYTHONUNBUFFERED=1" in _code(ISAAC_LAUNCHER)


def test_the_launcher_supports_both_headless_settings():
    text = _read(ISAAC_LAUNCHER)
    assert "WISEPACK_ISAAC_HEADLESS" in text
    assert "--headless" in text
    assert "no DISPLAY" in text, "headless must be usable over SSH"


def test_the_launcher_isolates_isaac_from_the_host_ros_environment():
    code = _code(ISAAC_LAUNCHER)
    for variable in ("PYTHONPATH", "AMENT_PREFIX_PATH", "ROS_DISTRO",
                     "LD_LIBRARY_PATH"):
        assert f"unset {variable}" in code, f"{variable} must be scrubbed"
    # ROS_DOMAIN_ID is the one that must survive: it is what puts Isaac on the
    # same DDS domain as WISEPACK.
    assert 'export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"' in code
    # Isaac's OWN ros env script restores its internal libraries afterwards.
    # Without it the bridge cannot load librmw_implementation.so and `import
    # rclpy` fails — measured on this machine while bringing the backend up.
    assert 'source "$ISAAC_ROOT/setup_ros_env.sh"' in code
    assert "/opt/ros" not in code, \
        "the host ROS environment must never be sourced into Isaac's interpreter"


@pytest.mark.parametrize("script", [DASHBOARD, ISAAC_LAUNCHER, VALIDATOR])
def test_isaac_cleanup_never_uses_a_broad_pattern(script):
    """A pattern kill would take out another project's simulator on a shared
    machine — the same rule the container reaping already follows."""
    code = _code(script)
    for reckless in ("pkill -f isaac", "pkill -9 -f isaac", "pkill isaac",
                     "killall", "pkill -f python", "pkill -f ros2"):
        assert reckless not in code, f"{os.path.basename(script)} uses {reckless}"


def test_the_dashboard_stops_only_the_isaac_process_it_started():
    text = _read(DASHBOARD)
    assert 'kill -TERM "-$ISAAC_PID"' in text
    assert "isaac_cleanup" in text
    assert "trap 'isaac_cleanup' EXIT INT TERM" in text


def test_the_dashboard_does_not_block_on_isaac_but_still_fails_loudly():
    """It WATCHES Isaac; it no longer waits for it before starting the stack.

    The old contract was "block until READY, then start the container". It made
    port 8080 unavailable for as long as Isaac took to compile shaders — minutes
    on a cold cache — and an operator watching a dead port cannot tell a slow
    simulator from a broken launcher. The DIAGNOSTIC half of that contract is
    unchanged and is asserted here: a timeout and a death are both still named,
    with the log tail, and the readiness gates upstream are untouched.
    """
    text = _read(DASHBOARD)
    assert "ISAAC_READY_TIMEOUT" in text
    assert "did not report READY" in text
    assert "tail -25" in text, "a failure must print a useful diagnostic"
    assert "ISAAC_WATCHER_PID" in text, "readiness is watched, not waited on"

    # THE ORDER IS THE POINT: `docker run` must be reached without an
    # intervening wait on Isaac.
    watcher = text.index("ISAAC_WATCHER_PID=$!")
    docker = text.index("DOCKER_RUN=(docker run")
    assert watcher < docker
    between = text[watcher:docker]
    assert "for _ in $(seq 1 \"$ISAAC_READY_TIMEOUT\")" not in between, \
        "nothing may block between starting Isaac and starting the stack"
    assert "[isaac-launch] not blocking on Isaac" in text


def test_a_dead_isaac_is_reported_without_taking_the_dashboard_down():
    """Reported and DEGRADED — never restarted, never silently absorbed."""
    text = _read(DASHBOARD)
    watcher = text[text.index("(\n        # Bounded, quiet"):
                   text.index("ISAAC_WATCHER_PID=$!")]
    assert 'kill -0 "$ISAAC_PID"' in watcher
    assert "startup_status.py\" degrade" in watcher
    assert "ROBOT_MODEL_INVALID" in watcher
    # No restart loop. Checked against the CODE rather than the comment that
    # says so — the comment reads "it never restarts anything" and would
    # otherwise satisfy a naive substring scan.
    code = "\n".join(line for line in watcher.splitlines()
                     if not line.strip().startswith("#"))
    for reckless in ("restart", "respawn", "setsid", "run_wisepack_isaac.sh"):
        assert reckless not in code, f"the watcher must not {reckless}"


def test_the_container_protection_is_preserved():
    text = _read(DASHBOARD)
    assert "wisepack_reap_stale" in text
    assert "WISEPACK_REAP_OTHERS" in text


def test_the_validator_skips_rather_than_fails_without_isaac():
    text = _read(VALIDATOR)
    assert "exit 77" in text
    assert "nvidia-smi" in text
    assert "SKIPPED" in text
    assert "The Isaac backend is optional" in text


def test_the_validator_checks_the_whole_physical_sequence():
    text = _read(VALIDATOR)
    for state in ("GRASPING", "RELEASING", "SETTLING", "ITEM_COMPLETED"):
        assert state in text, f"the validator does not verify {state}"
    assert "SMOKE-RESULT PASS" in text


def test_the_demo_treats_a_missing_simulator_as_a_skip():
    text = _read(DEMO)
    assert "-eq 77" in text
    assert "SKIPPED" in text
    assert "--isaac-sim" in text
