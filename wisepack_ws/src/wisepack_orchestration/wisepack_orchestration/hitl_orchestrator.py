"""HitL orchestrator — the py_trees behaviour tree over the WISEPACK workflow.

Every behaviour here is a THIN ADAPTER. It calls exactly one method on
``wisepack_core.WorkflowEngine`` and translates the result into a py_trees
Status. None of them re-implements planning, validation, execution or the
approval rule, which is what keeps ROS mode and the no-ROS dashboard mode
provably identical — there is only one implementation to be right or wrong.

Tree shape (a Sequence with a guarded execution loop):

    WISEPACK
    +-- Prepare              (Sequence, runs once)
    |   +-- GenerateOrLoadScenario
    |   +-- ScanAndDetect
    |   +-- GeneratePlans
    |   +-- DigitalTwinValidate
    +-- AwaitApproval        (RUNNING until the operator decides)
    +-- ExecuteLoop          (one placement per tick until COMPLETE)

The approval gate is a real gate: AwaitApproval returns RUNNING forever until an
APPROVE arrives on the operator topic, and ExecuteLoop sits behind it in a
Sequence, so the tree structurally cannot reach a pick before approval. The
engine asserts the same invariant independently — belt and braces, because this
is the safety-relevant one.

py_trees is the same behaviour-tree engine HARMONY's task_pack_bottle uses
(ros-jazzy-py-trees); the tree shape follows that pattern.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String

import py_trees

from wisepack_bringup import topics as T
from wisepack_bringup.qos import ORCHESTRATOR_PERIOD_S, qos_for
from wisepack_core.artifacts import (
    latest_latency_p50_ms, write_run_artifacts, write_validation_report,
)
from wisepack_core.domain import Strategy
from wisepack_core.events import DynamicEvent, DynamicEventType, Stage
from wisepack_core.packing import OptimizerConfig
from wisepack_core.workflow import (
    ApprovalRequired, RobotSimConfig, WorkflowConfig, WorkflowEngine,
)


# --------------------------------------------------------------------------- #
# Behaviours
# --------------------------------------------------------------------------- #


class _EngineBehaviour(py_trees.behaviour.Behaviour):
    """Base: holds the orchestrator node so behaviours can reach the engine."""

    def __init__(self, name: str, owner: "HitLOrchestrator") -> None:
        super().__init__(name)
        self.owner = owner

    @property
    def engine(self) -> WorkflowEngine:
        return self.owner.engine


class GenerateOrLoadScenario(_EngineBehaviour):
    def update(self):
        if self.engine.scenario is not None:
            return py_trees.common.Status.SUCCESS
        self.engine.generate_or_load_scenario()
        self.owner.publish_scenario()
        return py_trees.common.Status.SUCCESS


class ScanAndDetect(_EngineBehaviour):
    def update(self):
        if self.engine.detected:
            return py_trees.common.Status.SUCCESS
        self.engine.scan_and_detect()
        self.owner.publish_detection()
        return py_trees.common.Status.SUCCESS


class GeneratePlans(_EngineBehaviour):
    def update(self):
        if self.engine.optimized is not None:
            return py_trees.common.Status.SUCCESS
        self.engine.generate_plans()
        self.owner.publish_plans()
        return py_trees.common.Status.SUCCESS


class DigitalTwinValidate(_EngineBehaviour):
    """Runs the validator locally AND publishes the plan for the twin node.

    The separate wisepack_optimization/twin_validator node re-validates the same
    plan in its own process. Both must agree; the orchestrator publishes its own
    verdict on the plan-status topic and the twin node publishes over DDS, so a
    disagreement is visible rather than silent.
    """

    def update(self):
        if self.engine.selected is not None:
            return py_trees.common.Status.SUCCESS
        ok = self.engine.digital_twin_validate()
        self.owner.publish_plans()
        if not ok:
            self.owner.get_logger().error(
                "selected plan failed Digital Twin validation — holding")
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class AwaitApproval(_EngineBehaviour):
    """The gate. RUNNING until an operator decision arrives. Never times out.

    A timeout here would mean "proceed because nobody answered", which is exactly
    the autonomous continuation this demonstrator refuses to simulate.
    """

    def initialise(self):
        if self.engine.stage is not Stage.WAIT_FOR_OPERATOR_APPROVAL:
            self.engine.request_approval()
            self.owner.publish_state()

    def update(self):
        state = (self.engine.selected.approval_state.value
                 if self.engine.selected else "pending")
        if state == "approved":
            return py_trees.common.Status.SUCCESS
        if state == "rejected":
            # reject() already queued a re-plan; go round again.
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING


class ExecuteLoop(_EngineBehaviour):
    """One placement per tick: pick, verify, place, verify, update."""

    def update(self):
        if self.engine.finished:
            return py_trees.common.Status.SUCCESS
        try:
            more = self.engine.step_execution()
        except ApprovalRequired as exc:
            # Structurally unreachable behind AwaitApproval; if it ever fires,
            # something re-planned underneath us and the tree must go back.
            self.owner.get_logger().warn(f"execution gated: {exc}")
            return py_trees.common.Status.FAILURE
        self.owner.publish_execution()
        if self.engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL:
            return py_trees.common.Status.FAILURE       # a re-plan needs approval
        return (py_trees.common.Status.RUNNING if more
                else py_trees.common.Status.SUCCESS)


def build_tree(owner: "HitLOrchestrator") -> py_trees.behaviour.Behaviour:
    prepare = py_trees.composites.Sequence(name="Prepare", memory=True)
    prepare.add_children([
        GenerateOrLoadScenario("GENERATE_OR_LOAD_SCENARIO", owner),
        ScanAndDetect("SCAN_AND_DETECT", owner),
        GeneratePlans("GENERATE_PLANS", owner),
        DigitalTwinValidate("DIGITAL_TWIN_VALIDATE", owner),
    ])
    root = py_trees.composites.Sequence(name="WISEPACK", memory=True)
    root.add_children([
        prepare,
        AwaitApproval("WAIT_FOR_OPERATOR_APPROVAL", owner),
        ExecuteLoop("EXECUTE", owner),
    ])
    return root


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #


class HitLOrchestrator(Node):
    """Owns the WorkflowEngine, ticks the tree, and publishes the contract."""

    def __init__(self) -> None:
        super().__init__("wisepack_hitl_orchestrator")

        self.declare_parameter("preset", "mixed_pipes_dense")
        self.declare_parameter("seed", 42)
        self.declare_parameter("strategy", "max_density")
        self.declare_parameter("auto_approve", False)
        self.declare_parameter("dynamic_events", True)
        self.declare_parameter("pick_failure_probability", 0.08)
        self.declare_parameter("results_dir", "results")
        self.declare_parameter("tick_period_s", 0.7)

        preset = self.get_parameter("preset").value
        seed = int(self.get_parameter("seed").value)
        self.results_dir = self.get_parameter("results_dir").value
        self.auto_approve = bool(self.get_parameter("auto_approve").value)

        events = []
        if bool(self.get_parameter("dynamic_events").value):
            events = [DynamicEvent(
                event_type=DynamicEventType.ITEM_INJECT,
                trigger="placement:4",
                label="High-priority ILW component arrives late",
                payload={"item": {"length_mm": 1200, "outer_diameter_mm": 220,
                                  "inner_diameter_mm": 186,
                                  "material": "stainless_316L",
                                  "priority": 9, "dose_class": "ILW"}})]

        self.engine = WorkflowEngine(WorkflowConfig(
            preset=preset, seed=seed,
            strategy=Strategy(self.get_parameter("strategy").value),
            optimizer=OptimizerConfig(seed=seed, restarts=6),
            robot=RobotSimConfig(
                pick_failure_probability=float(
                    self.get_parameter("pick_failure_probability").value),
                seed=seed),
            dynamic_events=events,
            auto_approve=self.auto_approve))

        # -- publishers: exactly one writer per topic -------------------------
        def pub(topic, msg_type):
            return self.create_publisher(msg_type, topic, qos_for(topic))

        self.p_scenario_cfg = pub(T.SCENARIO_CONFIG, String)
        self.p_scenario_state = pub(T.SCENARIO_STATE, String)
        self.p_items = pub(T.WASTE_ITEMS, String)
        self.p_detected = pub(T.WASTE_DETECTED_COUNT, Int32)
        self.p_baseline = pub(T.PLAN_BASELINE, String)
        self.p_optimized = pub(T.PLAN_OPTIMIZED, String)
        self.p_selected = pub(T.PLAN_SELECTED, String)
        self.p_geometry = pub(T.PLAN_GEOMETRY, String)
        self.p_state = pub(T.EXECUTION_STATE, String)
        self.p_item = pub(T.EXECUTION_CURRENT_ITEM, String)
        self.p_container = pub(T.EXECUTION_CURRENT_CONTAINER, String)
        self.p_progress = pub(T.EXECUTION_PROGRESS_PCT, Float32)
        self.p_ready = pub(T.SYSTEM_READINESS, Bool)
        self.p_event = pub(T.ACTION_EVENT, String)
        self.p_sequence = pub(T.ACTION_SEQUENCE, Int32)
        self.p_dynamic = pub(T.DYNAMIC_EVENT, String)
        self.p_kpi = {
            "containers_baseline": pub(T.KPI_CONTAINERS_BASELINE, Int32),
            "containers_optimized": pub(T.KPI_CONTAINERS_OPTIMIZED, Int32),
            "utilization_baseline_pct": pub(T.KPI_UTILIZATION_BASELINE_PCT, Float32),
            "utilization_optimized_pct": pub(T.KPI_UTILIZATION_OPTIMIZED_PCT, Float32),
            "volume_reduction_pct": pub(T.KPI_VOLUME_REDUCTION_PCT, Float32),
            "optimization_ms": pub(T.KPI_OPTIMIZATION_MS, Float32),
            "pick_success_pct": pub(T.KPI_PICK_SUCCESS_PCT, Float32),
            "end_to_end_success_pct": pub(T.KPI_END_TO_END_SUCCESS_PCT, Float32),
        }

        # -- subscriptions: the operator command path ------------------------
        # These are the ONLY inbound topics. Orion-LD writes them when a
        # dashboard PATCHes the mapped NGSI-LD attribute, so an external HMI and
        # the bundled dashboard drive the workflow through the identical path.
        self.create_subscription(String, T.OPERATOR_APPROVAL, self._on_approval,
                                 qos_for(T.OPERATOR_APPROVAL))
        self.create_subscription(String, T.OPERATOR_COMMAND, self._on_command,
                                 qos_for(T.OPERATOR_COMMAND))

        # Every action event goes out on DDS the moment it is recorded. This is
        # the audit path; nothing batches it and nothing bypasses it.
        self.engine.log.add_sink(self._publish_event)

        self.tree = build_tree(self)
        self.tree.setup_with_descendants()
        self._artifacts_written = False

        period = float(self.get_parameter("tick_period_s").value)
        self.create_timer(period, self._tick)
        # Heartbeat: republishes state even when unchanged, so the Deadline on
        # the execution-state topic is satisfied while the loop is healthy and
        # missed the moment this node dies.
        self.create_timer(ORCHESTRATOR_PERIOD_S, self.publish_state)

        self.get_logger().info(
            f"WISEPACK orchestrator up — preset={preset} seed={seed} "
            f"auto_approve={self.auto_approve}")

    # -- tick ------------------------------------------------------------- #

    def _tick(self) -> None:
        try:
            self.tree.tick_once()
        except Exception as exc:                        # noqa: BLE001
            self.get_logger().error(f"behaviour tree error: {exc}")
            self.engine.enter_degraded(f"behaviour tree error: {exc}")
        if self.engine.finished and not self._artifacts_written:
            self._write_artifacts()

    # -- publishing ------------------------------------------------------- #

    def _publish_event(self, event) -> None:
        self.p_event.publish(String(data=event.to_json()))
        self.p_sequence.publish(Int32(data=event.sequence))
        if event.action.startswith("dynamic_event"):
            self.p_dynamic.publish(String(data=json.dumps(event.details)))
        # Surface the decision-relevant transitions on the node log too. The
        # audit trail is the topic, but a human (and validate_wisepack_e2e.sh)
        # reads the console, and a re-plan that is invisible there looks like a
        # re-plan that did not happen.
        if event.stage is Stage.REPLAN or event.action.startswith("dynamic_event"):
            self.get_logger().info(f"[{event.action}] {event.message}")

    def publish_scenario(self) -> None:
        scenario = self.engine.scenario
        if scenario is None:
            return
        self.p_scenario_cfg.publish(String(data=json.dumps({
            "preset": scenario.preset, "seed": scenario.seed,
            "scenario_id": scenario.scenario_id,
            "container_template": scenario.container_template.to_dict()
            if scenario.container_template else None})))
        self.p_scenario_state.publish(String(data=json.dumps(
            {"scenario_id": scenario.scenario_id, **scenario.to_dict()["totals"]})))
        self.p_items.publish(String(data=json.dumps(
            [i.to_dict() for i in scenario.items])))

    def publish_detection(self) -> None:
        self.p_detected.publish(Int32(data=len(self.engine.detected)))

    def publish_plans(self) -> None:
        engine = self.engine
        if engine.baseline:
            self.p_baseline.publish(String(data=json.dumps(engine.baseline.summary())))
        if engine.optimized:
            self.p_optimized.publish(String(data=json.dumps(engine.optimized.summary())))
        if engine.selected:
            self.p_selected.publish(String(data=json.dumps(engine.selected.summary())))
            # Full geometry for the twin validator and the dashboard.
            self.p_geometry.publish(String(data=json.dumps({
                "plan_id": engine.selected.plan_id,
                "scenario_id": engine.selected.scenario_id,
                "containers": [c.to_dict() for c in engine.selected.containers],
                "placements": [p.to_dict() for p in engine.selected.placements],
            })))
        self.publish_kpis()

    def publish_state(self) -> None:
        engine = self.engine
        self.p_state.publish(String(data=engine.stage.value))
        self.p_ready.publish(Bool(data=not engine.finished
                                  and engine.stage is not Stage.DEGRADED))
        self.p_progress.publish(Float32(data=float(engine.progress_pct)))

    def publish_execution(self) -> None:
        engine = self.engine
        self.p_item.publish(String(data=engine.current_item_id or ""))
        self.p_container.publish(String(data=engine.current_container_id or ""))
        self.publish_state()
        self.publish_kpis()

    def publish_kpis(self) -> None:
        engine = self.engine
        if engine.baseline is None or engine.optimized is None:
            return
        try:
            report = engine.kpis(latest_latency_p50_ms(self.results_dir))
        except Exception:                               # noqa: BLE001
            return
        ints = {"containers_baseline": "containers_baseline",
                "containers_optimized": "containers_optimized"}
        floats = {
            "utilization_baseline_pct": "container_utilization_baseline_pct",
            "utilization_optimized_pct": "container_utilization_optimized_pct",
            "volume_reduction_pct": "volume_requirement_reduction_pct",
            "optimization_ms": "optimization_time_ms",
            "pick_success_pct": "simulated_pick_success_rate_pct",
            "end_to_end_success_pct": "simulated_end_to_end_success_rate_pct",
        }
        for topic_key, metric_key in ints.items():
            value = report.value(metric_key)
            if value is not None:
                self.p_kpi[topic_key].publish(Int32(data=int(value)))
        for topic_key, metric_key in floats.items():
            value = report.value(metric_key)
            # An unmeasured KPI is NOT published as 0.0 — a consumer would read
            # that as a measured zero. It is simply absent until it exists.
            if value is not None:
                self.p_kpi[topic_key].publish(Float32(data=float(value)))

    # -- inbound ----------------------------------------------------------- #

    def _on_approval(self, msg: String) -> None:
        decision = (msg.data or "").strip().upper()
        if decision == T.APPROVE:
            try:
                self.engine.approve(operator="fiware/dashboard")
                self.get_logger().info("plan APPROVED via operator topic")
            except Exception as exc:                    # noqa: BLE001
                self.get_logger().warn(f"approval refused: {exc}")
        elif decision == T.REJECT:
            self.engine.reject(reason="rejected via operator topic")
            self.get_logger().info("plan REJECTED via operator topic — re-planning")
        elif decision:
            self.get_logger().warn(f"unknown approval value {decision!r}")
        self.publish_state()

    def _on_command(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data or "{}")
        except ValueError:
            self.get_logger().warn(f"malformed operator command: {msg.data!r}")
            return
        command = payload.get("command", "")
        args = payload.get("args", {}) or {}
        if command not in T.OPERATOR_COMMANDS:
            self.get_logger().warn(f"unknown operator command {command!r}")
            return
        try:
            self._apply_command(command, args)
        except Exception as exc:                        # noqa: BLE001
            self.get_logger().warn(f"command {command} failed: {exc}")
        self.publish_state()

    def _apply_command(self, command: str, args: Dict[str, Any]) -> None:
        engine = self.engine
        if command == "approve":
            engine.approve(operator=args.get("operator", "fiware/dashboard"))
        elif command == "reject":
            engine.reject(reason=args.get("reason", "rejected via command topic"))
        elif command == "alternative_strategy":
            strategy = Strategy(args.get("strategy", "retrievability"))
            engine.config.strategy = strategy
            engine.stats.operator_interventions += 1
            engine.generate_plans(strategy)
            engine.digital_twin_validate()
            engine.request_approval()
            self.publish_plans()
        elif command == "inject_item":
            engine.apply_dynamic_event(DynamicEvent(
                event_type=DynamicEventType.ITEM_INJECT,
                trigger=f"placement:{engine.cursor.index}",
                label=args.get("label", "Operator-injected component"),
                payload={"item": args.get("item", {
                    "length_mm": 1100, "outer_diameter_mm": 200,
                    "inner_diameter_mm": 170, "priority": 9,
                    "dose_class": "ILW"})}))
            self.publish_scenario()
            self.publish_plans()
        elif command == "container_unavailable":
            container_id = args.get("container_id") or (
                engine.selected.containers_used[0].container_id
                if engine.selected and engine.selected.containers_used else "")
            engine.apply_dynamic_event(DynamicEvent(
                event_type=DynamicEventType.CONTAINER_UNAVAILABLE,
                trigger=f"placement:{engine.cursor.index}",
                label=f"{container_id} out of service",
                payload={"container_id": container_id}))
            self.publish_plans()
        elif command == "grasp_failure":
            engine.apply_dynamic_event(DynamicEvent(
                event_type=DynamicEventType.GRASP_FAILURE,
                trigger=f"placement:{engine.cursor.index}",
                label="Operator-injected grasp failure"))
        elif command == "step":
            engine.step_execution()
            self.publish_execution()

    # -- artefacts --------------------------------------------------------- #

    def _write_artifacts(self) -> None:
        self._artifacts_written = True
        engine = self.engine
        if engine.selected is None:
            return
        try:
            kpis = engine.kpis(latest_latency_p50_ms(self.results_dir))
            artifacts = write_run_artifacts(
                engine.scenario, engine.baseline, engine.optimized,
                engine.selected, kpis, engine.log, self.results_dir)
            write_validation_report(
                engine.scenario, engine.baseline, engine.optimized,
                engine.selected, kpis, engine.log, artifacts, self.results_dir)
            self.get_logger().info(f"artefacts written: {artifacts.stamp}")
        except Exception as exc:                        # noqa: BLE001
            self.get_logger().error(f"artefact write failed: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HitLOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
