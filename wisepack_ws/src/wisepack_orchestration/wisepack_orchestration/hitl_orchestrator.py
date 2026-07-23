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
from wisepack_bringup.qos import ORCHESTRATOR_PERIOD_S, heartbeat_qos, qos_for
from wisepack_core.artifacts import (
    latest_latency_p50_ms, write_run_artifacts, write_validation_report,
)
from wisepack_core.domain import ApprovalState, Source, Strategy
from wisepack_core.events import Actor, DynamicEvent, DynamicEventType, Stage
from wisepack_core.packing import OptimizerConfig
from wisepack_core.workflow import (
    ApprovalRequired, RobotSimConfig, WorkflowConfig, WorkflowEngine, WorkflowError,
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

        # ORDER MATTERS, and it is safety-relevant.
        #
        # The approval check comes FIRST, before the pause gate. A re-plan sets
        # the plan back to pending AND clears auto_step; when the pause gate was
        # checked first, this behaviour returned RUNNING forever and the tree
        # never went back to AwaitApproval — the stage stayed at NEXT_ITEM after
        # a re-plan, i.e. execution continued on a plan nobody had approved.
        # Measured on the live stack before this was corrected.
        plan = self.engine.selected
        if plan is None or plan.approval_state is not ApprovalState.APPROVED:
            return py_trees.common.Status.FAILURE       # back to the gate

        if not getattr(self.owner, "auto_step", True):
            # Paused with an APPROVED plan: hold here so `resume` continues
            # rather than asking for approval a second time.
            return py_trees.common.Status.RUNNING
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
        self.p_plan_summary = pub(T.PLAN_SUMMARY, String)
        self.p_state = pub(T.EXECUTION_STATE, String)
        # Published with the OFFERING profile: Deadline + Liveliness are
        # offered so a strict external consumer can use them. Subscribers in
        # this repository deliberately do not request them (see qos.py).
        self.p_heartbeat = self.create_publisher(
            Int32, T.SYSTEM_HEARTBEAT, heartbeat_qos())
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
        # Gates the ExecuteLoop behaviour. Approving sets it; `pause` clears it
        # WITHOUT touching approval, so `resume` needs no second approval.
        self.auto_step = True

        self._heartbeat = 0
        period = float(self.get_parameter("tick_period_s").value)
        self.create_timer(period, self._tick)
        # State is republished so late joiners and the dashboard stay current;
        # the WATCHDOG lives on its own topic. Keeping them separate is what
        # stopped the dashboard's state subscription from having to request a
        # liveliness lease it could not match against Orion-LD's DDS bridge.
        self.create_timer(ORCHESTRATOR_PERIOD_S, self.publish_state)
        self.create_timer(ORCHESTRATOR_PERIOD_S, self._publish_heartbeat)

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

        # A re-plan REPLACES the plan and resets approval, so the plan topics
        # must be republished or the dashboard keeps rendering the superseded
        # plan with a stale "approved" badge — which is exactly the state a
        # human must never be shown while deciding whether to authorise a pick.
        # A re-plan can be triggered from deep inside step_execution(), so this
        # hooks the event stream rather than any one call site.
        if event.action in ("replan_complete", "scenario_ready"):
            self.auto_step = False
            try:
                self.publish_scenario()
                self.publish_plans()
                self.publish_state()
            except Exception as exc:                    # noqa: BLE001
                self.get_logger().warn(f"post-replan publish failed: {exc}")

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
        """Publish the COMPLETE plans, plus a compact digest for FIWARE.

        Full `PackingPlan.to_dict()` on each plan topic — the dashboard's Digital
        Twin and the twin validator both need real placement geometry, and no
        amount of summary can be turned back into a container drawing.
        """
        engine = self.engine
        if engine.baseline:
            self.p_baseline.publish(
                String(data=json.dumps(engine.baseline.to_dict(), default=str)))
        if engine.optimized:
            self.p_optimized.publish(
                String(data=json.dumps(engine.optimized.to_dict(), default=str)))
        if engine.selected:
            self.p_selected.publish(
                String(data=json.dumps(engine.selected.to_dict(), default=str)))
        # ~1 kB digest: this is what the FIWARE bridge maps.
        self.p_plan_summary.publish(String(data=json.dumps({
            "baseline": engine.baseline.summary() if engine.baseline else None,
            "optimized": engine.optimized.summary() if engine.optimized else None,
            "selected": engine.selected.summary() if engine.selected else None,
            "selected_plan_id": engine.selected.plan_id if engine.selected else None,
            "selection_reason": engine.selection_reason,
        }, default=str)))
        self.publish_kpis()

    def _publish_heartbeat(self) -> None:
        """The watchdog tick. Its silence is what means "the orchestrator died"."""
        self._heartbeat += 1
        self.p_heartbeat.publish(Int32(data=self._heartbeat))

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
                if (self.engine.selected
                        and self.engine.selected.approval_state
                        is ApprovalState.APPROVED):
                    self.get_logger().info("already approved — ignoring")
                    return
                self.engine.approve(operator="fiware/dashboard")
                self.auto_step = True
                self.publish_plans()
                self.get_logger().info("plan APPROVED via operator topic")
            except Exception as exc:                    # noqa: BLE001
                self.get_logger().warn(f"approval refused: {exc}")
        elif decision == T.REJECT:
            self.engine.reject(reason="rejected via operator topic")
            self.auto_step = False
            self.publish_plans()
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
        except (ApprovalRequired, WorkflowError, ValueError) as exc:
            # A command that is not legal in the current stage is an operator
            # mistake, not a node fault. Log it and carry on serving.
            self.get_logger().warn(f"command {command} refused: {exc}")
        except Exception as exc:                        # noqa: BLE001
            self.get_logger().error(f"command {command} failed unexpectedly: {exc}")
        self.publish_state()

    def _apply_command(self, command: str, args: Dict[str, Any]) -> None:
        """Every advertised operator command, implemented here.

        The dashboard's button list and `T.OPERATOR_COMMANDS` are the same
        contract; a command that reaches here and is not handled is a bug, and
        `test_operator_command_parity` fails on it rather than leaving a dead
        button in the UI.
        """
        engine = self.engine

        if command == "approve":
            engine.approve(operator=args.get("operator", "fiware/dashboard"))
            self.auto_step = True
            self.publish_plans()

        elif command == "reject":
            engine.reject(reason=args.get("reason", "rejected via command topic"))
            self.auto_step = False
            self.publish_scenario()
            self.publish_plans()

        elif command == "alternative_strategy":
            # Deterministic rotation when none is named, so repeatedly pressing
            # the button walks the three strategies instead of sticking.
            order = [s.value for s in Strategy]
            requested = args.get("strategy")
            if requested:
                strategy = Strategy(requested)
            else:
                current = engine.config.strategy.value
                strategy = Strategy(order[(order.index(current) + 1) % len(order)])
            engine.config.strategy = strategy
            engine.stats.operator_interventions += 1
            engine.generate_plans(strategy)
            engine.digital_twin_validate()
            engine.request_approval()
            self.auto_step = False
            self.publish_plans()
            self.get_logger().info(f"strategy -> {strategy.value}; re-approval required")

        elif command == "inject_item":
            engine.apply_dynamic_event(DynamicEvent(
                event_type=DynamicEventType.ITEM_INJECT,
                trigger=f"placement:{engine.cursor.index}",
                label=args.get("label", "Operator-injected component"),
                payload={"item": args.get("item", {
                    "length_mm": 1100, "outer_diameter_mm": 200,
                    "inner_diameter_mm": 170, "priority": 9,
                    "dose_class": "ILW"})}))
            self.auto_step = False
            self.publish_scenario()
            self.publish_plans()

        elif command == "container_unavailable":
            container_id = args.get("container_id") or (
                engine.selected.containers_used[0].container_id
                if engine.selected and engine.selected.containers_used else "")
            if not container_id:
                raise ValueError("no container available to mark unavailable")
            engine.apply_dynamic_event(DynamicEvent(
                event_type=DynamicEventType.CONTAINER_UNAVAILABLE,
                trigger=f"placement:{engine.cursor.index}",
                label=f"{container_id} out of service",
                payload={"container_id": container_id}))
            self.auto_step = False
            self.publish_plans()

        elif command == "grasp_failure":
            engine.apply_dynamic_event(DynamicEvent(
                event_type=DynamicEventType.GRASP_FAILURE,
                trigger=f"placement:{engine.cursor.index}",
                label="Operator-injected grasp failure"))

        elif command == "pause":
            # Pausing does NOT invalidate the plan — it stays approved, so
            # `resume` needs no second approval.
            self.auto_step = False
            engine.log.emit(Stage(engine.stage), "pause_execution", Actor.OPERATOR,
                            source=Source.OPERATOR,
                            message="automatic execution paused by operator")

        elif command == "resume":
            if not (engine.selected
                    and engine.selected.approval_state is ApprovalState.APPROVED):
                raise ValueError("cannot resume: the current plan is not approved")
            self.auto_step = True
            engine.log.emit(Stage(engine.stage), "resume_execution", Actor.OPERATOR,
                            source=Source.OPERATOR,
                            message="automatic execution resumed by operator")

        elif command == "step":
            engine.step_execution()
            self.publish_execution()

        elif command == "reset":
            self._reset_run(args)

        elif command == "write_artifacts":
            paths = self._write_artifacts(force=True)
            engine.log.emit(Stage(engine.stage), "write_artifacts", Actor.OPERATOR,
                            source=Source.OPERATOR,
                            message=f"artefacts written ({len(paths)} files)",
                            details={"files": paths})

    def _reset_run(self, args: Dict[str, Any]) -> None:
        """Build a brand-new run from the operator's scenario settings.

        This is the dashboard's "Generate & plan". It replaces the engine
        entirely rather than mutating the old one: a half-reset engine carrying
        the previous run's counters and frozen placements is exactly the kind of
        state that makes a demo behave differently the second time it is shown.
        """
        preset = args.get("preset", self.engine.config.preset)
        seed = int(args.get("seed", self.engine.config.seed))
        strategy = Strategy(args.get("strategy", self.engine.config.strategy.value))

        overrides: Dict[str, Any] = {}
        if preset != "curated_volume_reduction":
            for key in ("item_count", "length_range_mm", "diameter_range_mm",
                        "container_spec"):
                value = args.get(key)
                if value:
                    overrides[key] = (tuple(value) if isinstance(value, list)
                                      else value)

        events = []
        if args.get("dynamic_events_enabled", True):
            events = [DynamicEvent(
                event_type=DynamicEventType.ITEM_INJECT,
                trigger="placement:4",
                label="High-priority ILW component arrives late",
                payload={"item": {"length_mm": 1200, "outer_diameter_mm": 220,
                                  "inner_diameter_mm": 186,
                                  "material": "stainless_316L",
                                  "priority": 9, "dose_class": "ILW"}})]

        self.engine = WorkflowEngine(WorkflowConfig(
            preset=preset, seed=seed, strategy=strategy,
            optimizer=OptimizerConfig(seed=seed, restarts=6),
            robot=RobotSimConfig(
                pick_failure_probability=float(
                    args.get("pick_failure_probability",
                             self.engine.config.robot.pick_failure_probability)),
                seed=seed),
            dynamic_events=events,
            generator_overrides=overrides,
            auto_approve=self.auto_approve))
        self.engine.log.add_sink(self._publish_event)

        # Drive straight to the approval gate — never past it.
        self.engine.generate_or_load_scenario()
        self.engine.scan_and_detect()
        self.engine.generate_plans()
        self.engine.digital_twin_validate()
        self.engine.request_approval()

        self._artifacts_written = False
        self.auto_step = False
        self.tree = build_tree(self)
        self.tree.setup_with_descendants()

        self.publish_scenario()
        self.publish_detection()
        self.publish_plans()
        self.publish_state()
        self.get_logger().info(
            f"reset -> preset={preset} seed={seed} strategy={strategy.value}; "
            f"awaiting approval")

    # -- artefacts --------------------------------------------------------- #

    def _write_artifacts(self, force: bool = False) -> Dict[str, str]:
        """Write the run artefacts. Returns {kind: path}.

        `force` allows an operator to capture evidence mid-run, which is the
        point of the dashboard's "Write artefacts" button — waiting for
        completion is no use when the interesting state is the one on screen.
        """
        self._artifacts_written = True
        engine = self.engine
        if engine.selected is None or engine.baseline is None:
            self.get_logger().warn("nothing to write yet — no plan exists")
            return {}
        try:
            kpis = engine.kpis(latest_latency_p50_ms(self.results_dir))
            artifacts = write_run_artifacts(
                engine.scenario, engine.baseline, engine.optimized,
                engine.selected, kpis, engine.log, self.results_dir)
            report = write_validation_report(
                engine.scenario, engine.baseline, engine.optimized,
                engine.selected, kpis, engine.log, artifacts, self.results_dir)
            paths = dict(artifacts.paths)
            paths["validation_report"] = report
            self.get_logger().info(f"artefacts written: {artifacts.stamp}")
            return paths
        except Exception as exc:                        # noqa: BLE001
            self.get_logger().error(f"artefact write failed: {exc}")
            return {}


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
