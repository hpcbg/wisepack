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
from wisepack_core.execution import ExecutionBackend, parse_backend
from wisepack_core.packing import OptimizerConfig
from wisepack_core.anomaly import AnomalyEvent
from wisepack_core.workflow import (
    AnomalyHold, ApprovalRequired, RobotSimConfig, WorkflowConfig, WorkflowEngine,
    WorkflowError,
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
    """One placement per tick: pick, verify, place, verify, update.

    BACKEND-AGNOSTIC BY CONSTRUCTION. The gating logic below — approval first,
    then the pause gate, then a re-plan check afterwards — is identical for both
    execution backends and is the safety-relevant part. Only the single line that
    actually advances a placement differs, and the two alternatives share a
    return contract (True while more remains, False when the run is done) so
    there is no second control flow to get wrong.
    """

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
        # An anomaly hold blocks execution deterministically, before approval.
        # A warning holds an approved plan (resume after acknowledgement); a
        # critical also revoked approval, so it drops through to the gate below.
        if self.engine.anomaly_hold:
            return py_trees.common.Status.RUNNING

        plan = self.engine.selected
        if plan is None or plan.approval_state is not ApprovalState.APPROVED:
            # A physical backend may have an item in flight when approval is
            # withdrawn. Tell it to stop before returning to the gate, or the
            # arm carries on placing an item from a superseded plan.
            if self.owner.isaac is not None:
                self.owner.isaac.abort_run(
                    self.engine, "plan is no longer approved")
            return py_trees.common.Status.FAILURE       # back to the gate

        if not getattr(self.owner, "auto_step", True):
            # Paused with an APPROVED plan: hold here so `resume` continues
            # rather than asking for approval a second time.
            return py_trees.common.Status.RUNNING
        try:
            # BOTH backends go through the same gate. The dispatch differs
            # (Isaac executes physically, the simulated model resolves a seeded
            # coin flip); the exception handling does not, and must not — an
            # anomaly hold stops a PHYSICAL run exactly as it stops a simulated
            # one, and returning to the tree is the only safe response to either.
            if self.owner.isaac is not None:
                more = self.owner.isaac.tick(self.engine)
            else:
                more = self.engine.step_execution()
        except (ApprovalRequired, AnomalyHold) as exc:
            # Structurally unreachable behind AwaitApproval / the anomaly gate;
            # if it fires, something changed underneath us and the tree must
            # re-evaluate rather than execute.
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
        # WHO EXECUTES, not where the dashboard reads from. See
        # wisepack_core.execution for why those are different concepts.
        self.declare_parameter("execution_backend",
                               ExecutionBackend.SIMULATED.value)
        self.declare_parameter("isaac_ready_timeout_s", 240.0)
        self.declare_parameter("isaac_item_timeout_s", 180.0)

        preset = self.get_parameter("preset").value
        seed = int(self.get_parameter("seed").value)
        self.results_dir = self.get_parameter("results_dir").value
        self.auto_approve = bool(self.get_parameter("auto_approve").value)
        # An unknown backend name raises here, before any publisher exists. A
        # typo that quietly selected `simulated` would produce a run that
        # reported physical execution it never performed.
        self.execution_backend = parse_backend(
            self.get_parameter("execution_backend").value)

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
            auto_approve=self.auto_approve,
            execution_backend=self.execution_backend))

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
        self.p_comparison = pub(T.PLAN_STRATEGY_COMPARISON, String)
        # Anomaly monitoring. The orchestrator is the single writer of
        # the recorded stream and the state; it INGESTS from the external seam.
        self.p_anomaly = pub(T.ANOMALY_EVENT, String)
        self.p_anomaly_state = pub(T.ANOMALY_STATE, String)
        # Whole-process layer: SIMULATED cutting, container inventory, SIMULATED
        # logistics. The orchestrator is the single writer of every one.
        self.p_cut_proposal = pub(T.CUTTING_PROPOSAL, String)
        self.p_cut_state = pub(T.CUTTING_STATE, String)
        self.p_cut_result = pub(T.CUTTING_RESULT, String)
        self.p_cut_request = pub(T.CUTTING_REQUEST, String)
        self.p_inv_state = pub(T.INVENTORY_CONTAINER_STATE, String)
        self.p_inv_summary = pub(T.INVENTORY_SUMMARY, String)
        self.p_inv_event = pub(T.INVENTORY_CONTAINER_EVENT, String)
        self.p_inv_reservation = pub(T.INVENTORY_RESERVATION, String)
        self.p_log_task = pub(T.LOGISTICS_CONTAINER_TASK, String)
        self.p_log_task_state = pub(T.LOGISTICS_CONTAINER_TASK_STATE, String)
        self.p_log_robot = pub(T.LOGISTICS_MOBILE_ROBOT_STATE, String)
        self.p_state = pub(T.EXECUTION_STATE, String)
        # Published with the OFFERING profile: Deadline + Liveliness are
        # offered so a strict external consumer can use them. Subscribers in
        # this repository deliberately do not request them (see qos.py).
        self.p_heartbeat = self.create_publisher(
            Int32, T.SYSTEM_HEARTBEAT, heartbeat_qos())
        self.p_item = pub(T.EXECUTION_CURRENT_ITEM, String)
        self.p_container = pub(T.EXECUTION_CURRENT_CONTAINER, String)
        self.p_progress = pub(T.EXECUTION_PROGRESS_PCT, Float32)
        self.p_backend = pub(T.EXECUTION_BACKEND, String)
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
        # Ingest seam: a future external Topic #2 detector (or the bundled
        # simulator) publishes structured events here.
        self.create_subscription(String, T.ANOMALY_EXTERNAL, self._on_anomaly,
                                 qos_for(T.ANOMALY_EXTERNAL))
        # Separate cut-approval channel and inventory-request channel — both
        # written by Orion-LD from a mapped NGSI-LD attribute PATCH (brief §6/§13).
        self.create_subscription(String, T.CUTTING_APPROVAL, self._on_cut_approval,
                                 qos_for(T.CUTTING_APPROVAL))
        self.create_subscription(String, T.INVENTORY_REQUEST, self._on_inventory_request,
                                 qos_for(T.INVENTORY_REQUEST))

        # Every action event goes out on DDS the moment it is recorded. This is
        # the audit path; nothing batches it and nothing bypasses it.
        self.engine.log.add_sink(self._publish_event)

        # THE EXECUTION BACKEND. When this is None the simulated robot model in
        # WorkflowEngine.step_execution runs, exactly as it always has. When it
        # is an IsaacExecutionBridge, step_execution is never called at all — see
        # ExecuteLoop — so the two backends can never both claim a placement.
        self.isaac = None
        if self.execution_backend is ExecutionBackend.ISAAC:
            from .isaac_bridge import IsaacExecutionBridge      # noqa: PLC0415
            self.isaac = IsaacExecutionBridge(
                self,
                ready_timeout_s=float(
                    self.get_parameter("isaac_ready_timeout_s").value),
                item_timeout_s=float(
                    self.get_parameter("isaac_item_timeout_s").value))
            self.get_logger().info(
                "execution backend: ISAAC SIM — the simulated robot model is "
                "disabled for this run; placements are executed physically")

        self.tree = build_tree(self)
        self.tree.setup_with_descendants()
        self._artifacts_written = False
        # Gates the ExecuteLoop behaviour. Approving sets it; `pause` clears it
        # WITHOUT touching approval, so `resume` needs no second approval.
        self.auto_step = True
        # (proposal_id, approval_revision) of the last CUTTING_REQUEST emitted —
        # the dedup authority lives in WholeProcess.build_cut_request(); this is
        # kept for observability and a belt-and-braces guard.
        self._last_cut_request_key = None

        self._heartbeat = 0
        period = float(self.get_parameter("tick_period_s").value)
        self.create_timer(period, self._tick)
        # State is republished so late joiners and the dashboard stay current;
        # the WATCHDOG lives on its own topic. Keeping them separate is what
        # stopped the dashboard's state subscription from having to request a
        # liveliness lease it could not match against Orion-LD's DDS bridge.
        self.create_timer(ORCHESTRATOR_PERIOD_S, self.publish_state)
        self.create_timer(ORCHESTRATOR_PERIOD_S, self._publish_heartbeat)

        self._publish_anomaly_state()          # latch an initial "no anomaly" state
        self.publish_whole_process()           # latch initial inventory/logistics
        self.get_logger().info(
            f"WISEPACK orchestrator up — preset={preset} seed={seed} "
            f"auto_approve={self.auto_approve} "
            f"execution_backend={self.execution_backend.value}")

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
                # The batch just changed: overwrite the latched comparison topic
                # with a "cleared" marker stamped with the NEW revision, so a
                # late subscriber never renders a comparison from the old batch.
                self.p_comparison.publish(String(data=json.dumps({
                    "schema_version": "1.0", "status": "cleared",
                    "scenario_revision": self.engine.scenario_revision,
                    "scenario_id": (self.engine.scenario.scenario_id
                                    if self.engine.scenario else None),
                    "results": []})))
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
            "scenario_revision": engine.scenario_revision,
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
        # Latched, so a dashboard attaching mid-run learns which backend is
        # authoritative rather than defaulting to "simulated" and mislabelling a
        # physical run. The payload carries the live simulator status too, so
        # "isaac selected" and "isaac actually up" stay distinguishable.
        backend = {"backend": self.execution_backend.value,
                   "label": self.execution_backend.label,
                   "detail": self.execution_backend.detail,
                   "physical": self.execution_backend.is_physical}
        if self.isaac is not None:
            backend["isaac"] = self.isaac.status()
            # Backend-neutral: the dashboard reads `visualization` without
            # knowing which backend produced it, so a future real cell needs no
            # dashboard change.
            backend["visualization"] = self.isaac.visualization
        self.p_backend.publish(String(data=json.dumps(backend, default=str)))

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
        except (ApprovalRequired, AnomalyHold, WorkflowError, ValueError) as exc:
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
            # MUST route through the active backend. Calling step_execution()
            # here unconditionally would run the SIMULATED robot model for one
            # placement in the middle of a physical run — a fabricated outcome
            # for an item the arm never touched, recorded in the audit trail as
            # though it had been.
            if self.isaac is not None:
                self.isaac.tick(engine)
            else:
                engine.step_execution()
            self.publish_execution()

        elif command == "compare_strategies":
            self._compare_strategies(args)

        elif command == "inject_anomaly":
            # Operator-injected SIMULATED anomaly. Same deterministic path as an
            # event arriving on the ingest seam.
            cls = args.get("anomaly_class", "camera_view_lost")
            event = AnomalyEvent.simulate(
                cls, severity=args.get("severity"),
                confidence=args.get("confidence"))
            self._ingest_anomaly(event)

        elif command == "acknowledge_anomaly":
            engine.acknowledge_anomaly(args.get("operator", "dashboard operator"))
            self._publish_anomaly_state()

        elif command == "reset":
            self._reset_run(args)

        elif command == "write_artifacts":
            paths = self._write_artifacts(force=True)
            engine.log.emit(Stage(engine.stage), "write_artifacts", Actor.OPERATOR,
                            source=Source.OPERATOR,
                            message=f"artefacts written ({len(paths)} files)",
                            details={"files": paths})

        # -- cut-aware HITL controls (brief §6) --
        elif command == "compare_cut_aware":
            engine.wp.generate_cut_alternatives()
            self.auto_step = False
            self.publish_whole_process()

        elif command == "select_cut_alternative":
            engine.wp.select_alternative(args.get("label", "no_cut"))
            self.publish_whole_process()

        elif command == "limit_cuts":
            engine.wp.limit_cuts(int(args.get("max_cuts", 1)))
            self.publish_whole_process()

        elif command == "set_min_segment":
            engine.wp.set_minimum_segment_mm(int(args.get("mm", 400)))
            self.publish_whole_process()

        elif command == "prefer_no_cut":
            engine.wp.set_prefer_no_cut(bool(args.get("prefer", True)))
            self.publish_whole_process()

        elif command == "approve_cut":
            engine.wp.approve_cut(args.get("operator", "fiware/dashboard"))
            self.publish_whole_process()

        elif command == "reject_cut":
            engine.wp.reject_cut(args.get("reason", "operator preferred no cutting"))
            self.publish_whole_process()

        elif command == "simulate_cut":
            engine.wp.simulate_cut(deviation_mm=int(args.get("deviation_mm", 0)))
            self.publish_plans()
            self.publish_whole_process()

        elif command == "simulate_cut_failure":
            engine.wp.simulate_cut_failure(args.get("reason", "blade jam (simulated)"))
            self.publish_whole_process()

        # -- inventory + logistics controls (brief §13) --
        elif command == "init_inventory":
            engine.wp.initialise_simulated_inventory(int(args.get("count", 4)))
            self.publish_whole_process()

        elif command == "check_containers":
            engine.wp.check_container_availability()
            engine.wp.run_logistics_to_quiescence()
            self.publish_whole_process()

        elif command in ("reserve_container", "release_container",
                         "request_delivery", "mark_container_unavailable",
                         "restore_container", "mark_container_full"):
            op = {
                "reserve_container": "reserve",
                "release_container": "release_reservation",
                "request_delivery": "request_delivery",
                "mark_container_unavailable": "mark_unavailable",
                "restore_container": "restore",
                "mark_container_full": "mark_full",
            }[command]
            cid = args.get("container_id")
            if not cid:
                raise ValueError(f"{command} requires container_id")
            engine.wp.inventory_operation(
                op, cid, actor=args.get("operator", "dashboard"),
                reason=args.get("reason", ""),
                holder=args.get("holder", engine.selected.plan_id
                                if engine.selected else "operator"))
            engine.wp.run_logistics_to_quiescence()
            self.publish_whole_process()

        elif command == "collect_full_containers":
            engine.wp.collect_full_containers()
            self.publish_whole_process()

    def _on_anomaly(self, msg: String) -> None:
        """Ingest an anomaly from the external seam and react deterministically."""
        import time as _time
        received = _time.monotonic()
        try:
            event = AnomalyEvent.from_dict(json.loads(msg.data))
        except (ValueError, KeyError) as exc:
            self.get_logger().warn(f"rejected malformed anomaly: {exc}")
            return
        self._ingest_anomaly(event, received)

    def _ingest_anomaly(self, event: AnomalyEvent,
                        received_monotonic: Optional[float] = None) -> None:
        """React to an anomaly and publish the recorded stream + state.

        Safety-critical response is LOCAL and deterministic (anomaly -> engine
        -> pause/hold), and does NOT wait for FIWARE. The FIWARE path is an
        ADDITIONAL analytics/traceability route, not the stopping mechanism.
        """
        record = self.engine.apply_anomaly(event, received_monotonic)
        # Record the anomaly on the canonical stream (this is what maps to FIWARE).
        self.p_anomaly.publish(String(data=json.dumps(record, default=str)))
        self._publish_anomaly_state()
        # A critical anomaly revoked approval; republish plans/state so the
        # dashboard shows the workflow back at the gate.
        if record.get("reaction") == "hold":
            self.auto_step = False
            self.publish_plans()
        self.publish_state()

    def _publish_anomaly_state(self) -> None:
        self.p_anomaly_state.publish(
            String(data=json.dumps(self.engine.anomaly_snapshot(), default=str)))

    # -- whole-process (cutting / inventory / logistics) ------------------- #

    def _on_cut_approval(self, msg: String) -> None:
        """A PATCH of WISEPACKSystem.cutApproval — approves cutting ONLY."""
        decision = (msg.data or "").strip().upper()
        try:
            if decision in ("APPROVE_CUT", "APPROVE"):
                self.engine.wp.approve_cut("fiware/dashboard")
            elif decision in ("REJECT_CUT", "REJECT"):
                self.engine.wp.reject_cut("rejected via cut-approval channel")
            else:
                self.get_logger().warn(f"unknown cut approval {msg.data!r}")
                return
        except Exception as exc:                            # noqa: BLE001
            self.get_logger().warn(f"cut approval refused: {exc}")
            return
        self.publish_whole_process()
        self.publish_state()

    def _on_inventory_request(self, msg: String) -> None:
        """A PATCH of WISEPACKSystem.inventoryRequest — a JSON {command,args}."""
        try:
            payload = json.loads(msg.data or "{}")
        except ValueError:
            self.get_logger().warn(f"malformed inventory request: {msg.data!r}")
            return
        command = payload.get("command", "")
        args = payload.get("args", {}) or {}
        if command not in T.OPERATOR_COMMANDS:
            self.get_logger().warn(f"unknown inventory command {command!r}")
            return
        try:
            self._apply_command(command, args)
        except Exception as exc:                            # noqa: BLE001
            self.get_logger().warn(f"inventory command {command} refused: {exc}")
        self.publish_state()

    def publish_whole_process(self) -> None:
        """Publish the cutting / inventory / logistics state on their topics."""
        wp = self.engine.wp
        cut = wp.cut_snapshot()
        if cut is not None:
            self.p_cut_proposal.publish(String(data=json.dumps(cut, default=str)))
            self.p_cut_state.publish(String(data=json.dumps(
                {"stage": self.engine.stage.value,
                 "cut_approval_state": cut["cut_approval_state"],
                 "selected_label": cut["selected_label"]}, default=str)))
            # Emit the request to the EXTERNAL FUTURE cutting skill EXACTLY ONCE
            # per approved proposal revision. build_cut_request() returns a new
            # request only on the first call after an approval and None on every
            # periodic republication, so this cannot create duplicates.
            request = wp.build_cut_request()
            if request is not None:
                self._last_cut_request_key = (request["proposal_id"],
                                              request["approval_revision"])
                self.p_cut_request.publish(String(data=json.dumps(
                    request, default=str)))
            if cut.get("latest_cut_result"):
                self.p_cut_result.publish(String(data=json.dumps(
                    cut["latest_cut_result"], default=str)))
        inv = wp.inventory
        self.p_inv_state.publish(String(data=json.dumps(
            inv.semantic_states(), default=str)))
        self.p_inv_summary.publish(String(data=json.dumps(
            inv.summary(), default=str)))
        log = wp.logistics
        self.p_log_robot.publish(String(data=json.dumps(
            log.robot.to_dict(), default=str)))
        self.p_log_task.publish(String(data=json.dumps(
            log.facility_map(), default=str)))
        active = log.tasks.get(log.robot.current_task_id or "")
        if active is not None:
            self.p_log_task_state.publish(String(data=json.dumps(
                active.to_dict(), default=str)))

    def _compare_strategies(self, args: Dict[str, Any]) -> None:
        """Run + validate every strategy and publish a structured comparison.

        Authoritative in ROS/FIWARE modes: the dashboard has no engine, so this
        is the ONLY place a live comparison is produced. It is decision support
        and mutates no plan (see WorkflowEngine.build_strategy_comparison), so
        the rejection rules below guard state consistency, not safety.
        """
        engine = self.engine
        if engine.scenario is None:
            raise WorkflowError("no scenario — generate a plan first")
        # Revision guard: if the dashboard names a revision, it must be current.
        req_rev = args.get("scenario_revision")
        if req_rev is not None and int(req_rev) != engine.scenario_revision:
            raise WorkflowError(
                f"scenario revision {req_rev} is stale "
                f"(current is {engine.scenario_revision})")
        # One comparison at a time.
        if getattr(self, "_comparing", False):
            raise WorkflowError("a strategy comparison is already in progress")
        # Not while a plan is being (re)generated.
        if engine.stage in (Stage.GENERATE_BASELINE_PLAN,
                            Stage.GENERATE_OPTIMIZED_PLAN, Stage.REPLAN):
            raise WorkflowError("planning in progress — try again shortly")

        self._comparing = True
        try:
            strategies = args.get("strategies") or None
            comparison = engine.build_strategy_comparison(strategies)
        finally:
            self._comparing = False

        self.p_comparison.publish(
            String(data=json.dumps(comparison, default=str)))
        self.get_logger().info(
            f"strategy comparison {comparison['comparison_id']} published "
            f"(revision {comparison['scenario_revision']})")

    def _reset_run(self, args: Dict[str, Any]) -> None:
        """Build a brand-new run from the operator's scenario settings.

        This is the dashboard's "Generate & plan". It replaces the engine
        entirely rather than mutating the old one: a half-reset engine carrying
        the previous run's counters and frozen placements is exactly the kind of
        state that makes a demo behave differently the second time it is shown.
        """
        previous_engine = self.engine
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
            auto_approve=self.auto_approve,
            execution_backend=self.execution_backend))
        self.engine.log.add_sink(self._publish_event)

        # A reset is a NEW run with a new run_id, so the physical scene has to be
        # rebuilt around the new scenario. Abort whatever the simulator was doing
        # before the old engine goes out of scope; the bridge re-opens on its
        # next tick when it notices the run_id changed.
        if self.isaac is not None:
            self.isaac.abort_run(previous_engine,
                                 "operator reset — starting a new run")
            self.isaac.run_open = False

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
