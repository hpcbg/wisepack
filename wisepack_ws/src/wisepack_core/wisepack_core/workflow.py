"""The WISEPACK workflow engine — ROS-free, and the single source of behaviour.

This is the module that makes "the same domain logic in both modes" true rather
than aspirational. It owns the whole cycle:

    IDLE -> GENERATE_OR_LOAD_SCENARIO -> SCAN_SOURCE_BIN -> DETECT_ITEMS
         -> GENERATE_BASELINE_PLAN -> GENERATE_OPTIMIZED_PLAN
         -> DIGITAL_TWIN_VALIDATE -> WAIT_FOR_OPERATOR_APPROVAL
         -> (PICK_ITEM -> VERIFY_PICK -> PLACE_ITEM -> VERIFY_PLACEMENT
             -> UPDATE_CONTAINER_STATE -> NEXT_ITEM|REPLAN)* -> COMPLETE

The py_trees behaviour tree in wisepack_orchestration does NOT re-implement any
of this: each of its behaviours calls one method here and translates the return
value into a py_trees Status. The dashboard's sim mode calls the same methods.
That is why a bug fixed in one mode is fixed in both.

SAFETY INVARIANT, enforced not documented: no physical action may occur before
the operator approves the plan. ``step_execution`` raises ApprovalRequired if
called while the selected plan is unapproved, and the invariant is asserted
again inside the pick itself. It is a programming error to reach it, which is
exactly why it is checked rather than trusted.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .domain import (
    ApprovalState, Container, ContainerStatus, ItemStatus, PackingPlan,
    Placement, Scenario, Source, Strategy, ValidationStatus,
)
from .events import (
    Actor, ActionLog, DynamicEvent, DynamicEventType, PRE_APPROVAL_STAGES,
    Result, Stage, Stopwatch,
)
from .generator import build_scenario, inject_item
from .kpi import ExecutionStats, KPIReport, compute_kpis
from .packing import (
    OptimizerConfig, STRATEGY_WEIGHTS, pack_baseline, pack_optimized, select_plan,
)
from .validator import DEFAULT_VALIDATION, PlacementValidator, ValidationConfig


class ApprovalRequired(RuntimeError):
    """Raised when execution is attempted on a plan the operator has not approved."""


class WorkflowError(RuntimeError):
    """Raised when the workflow is driven out of order."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class RobotSimConfig:
    """Simulated robot behaviour. Every outcome is seeded and reproducible.

    ``pick_failure_probability`` is the ONLY source of pick failure. There is no
    kinematics, no grasp model and no physics — a "failed pick" here is a
    seeded coin flip, and the KPI module labels the resulting rate ``simulated``
    for precisely that reason.
    """

    pick_failure_probability: float = 0.08
    max_pick_retries: int = 2
    #: Simulated motion durations, milliseconds. Used for the timeline and the
    #: action-duration analytics; they are not measurements of anything.
    pick_duration_ms: float = 900.0
    place_duration_ms: float = 1100.0
    travel_duration_ms: float = 600.0
    #: Perception confidence floor for the simulated detector.
    detection_confidence_min: float = 0.82
    detection_confidence_max: float = 0.99
    #: Probability the simulated detector misses an item entirely.
    detection_miss_probability: float = 0.0
    seed: int = 42


@dataclass
class WorkflowConfig:
    preset: str = "mixed_pipes_small"
    seed: int = 42
    strategy: Strategy = Strategy.MAX_DENSITY
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    robot: RobotSimConfig = field(default_factory=RobotSimConfig)
    validation: ValidationConfig = field(default=DEFAULT_VALIDATION)
    dynamic_events: List[DynamicEvent] = field(default_factory=list)
    #: Auto-approve without a human. Only for headless acceptance runs, which
    #: say so in the audit trail (the approval event's actor stays `system`).
    auto_approve: bool = False
    generator_overrides: Dict[str, Any] = field(default_factory=dict)
    max_replans: int = 5

    def __post_init__(self) -> None:
        # Same coercion as OptimizerConfig: strategies reach here as strings from
        # YAML and from the dashboard.
        self.strategy = Strategy(self.strategy)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


# A "cycle" is one ITEM carried through pick -> verify -> place -> verify ->
# update. It is deliberately NOT one grasp attempt: an item that needs three
# grasps and then succeeds is one completed cycle with three pick attempts, so
# KPI2 (pick success) and KPI3 (end-to-end success) measure different things and
# cannot collapse onto the same number.


@dataclass
class ExecutionCursor:
    """Where execution has got to in the selected plan."""

    index: int = 0
    retries: int = 0

    def reset(self) -> None:
        self.index = 0
        self.retries = 0


class WorkflowEngine:
    """One packaging run. Not thread-safe by itself; callers hold the lock."""

    def __init__(self, config: Optional[WorkflowConfig] = None,
                 action_log: Optional[ActionLog] = None) -> None:
        self.config = config or WorkflowConfig()
        self.run_id = f"run-{uuid.uuid4().hex[:8]}"
        self.cycle_id = f"cycle-{uuid.uuid4().hex[:6]}"
        self.log = action_log or ActionLog()
        self.log.set_context(cycle_id=self.cycle_id)

        self.stage: Stage = Stage.IDLE
        self.scenario: Optional[Scenario] = None
        # THE COMPARISON PAIR: both algorithms over the FULL current scenario.
        # These back KPI4 and the dashboard's baseline-vs-optimized table, and
        # they must always be apples-to-apples — same items, same container
        # spec, neither carrying execution state the other lacks.
        self.baseline: Optional[PackingPlan] = None
        self.optimized: Optional[PackingPlan] = None
        # THE EXECUTION PLAN. After a re-plan this also carries the frozen,
        # already-executed placements and the containers holding them, so it is
        # NOT comparable with `baseline` and is never used as a KPI operand.
        self.selected: Optional[PackingPlan] = None
        self.selection_reason: str = ""
        self.cursor = ExecutionCursor()
        self.stats = ExecutionStats()
        self.validator = PlacementValidator(self.config.validation)
        self.dynamic_events: List[DynamicEvent] = list(self.config.dynamic_events)
        self.detected: Dict[str, float] = {}       # item_id -> simulated confidence
        self.robot_state = "idle"
        self.current_item_id: Optional[str] = None
        self.current_container_id: Optional[str] = None
        self.degraded_reason: str = ""
        self.finished = False
        self._rng = random.Random(self.config.robot.seed)
        self._t_exec_start: Optional[float] = None
        self._strategy_plans: Dict[str, PackingPlan] = {}
        # Container ids are issued from a run-global counter. A re-plan must not
        # re-issue the id of a retired (unavailable) container, or the
        # retirement is silently undone and the packer fills a box that is out
        # of service. See replan().
        self._comparison_scenario: Optional[Scenario] = None
        self._container_index_offset = 0
        self._retired_containers: Dict[str, Container] = {}
        # Set by the grasp_failure dynamic event. An explicit flag rather than
        # re-seeding the RNG and hoping its next draw lands below the failure
        # threshold — that was not actually deterministic.
        self._force_pick_failure = False

    # -- helpers ------------------------------------------------------------ #

    def _emit(self, stage: Stage, action: str, actor: Actor,
              result: Result = Result.OK, **kw: Any):
        return self.log.emit(stage, action, actor, result, **kw)

    def _set_stage(self, stage: Stage) -> None:
        self.stage = stage

    @property
    def progress_pct(self) -> float:
        if not self.selected or not self.selected.placements:
            return 0.0
        done = sum(1 for p in self.selected.placements if p.executed)
        return 100.0 * done / len(self.selected.placements)

    @property
    def containers(self) -> List[Container]:
        return self.selected.containers if self.selected else []

    # ------------------------------------------------------------------ #
    # Stage 1 — scenario
    # ------------------------------------------------------------------ #

    def generate_or_load_scenario(self, scenario: Optional[Scenario] = None) -> Scenario:
        watch = Stopwatch()
        self._set_stage(Stage.GENERATE_OR_LOAD_SCENARIO)
        if scenario is None:
            scenario = build_scenario(
                self.config.preset, self.config.seed,
                **self.config.generator_overrides)
        self.scenario = scenario
        self.log.set_context(scenario_id=scenario.scenario_id)
        if scenario.dynamic_events and not self.dynamic_events:
            self.dynamic_events = [DynamicEvent.from_dict(d)
                                   for d in scenario.dynamic_events]
        self._emit(Stage.GENERATE_OR_LOAD_SCENARIO, "scenario_ready",
                   Actor.TASK_GENERATOR, duration_ms=watch.ms,
                   message=f"{len(scenario.items)} items, preset "
                           f"{scenario.preset}, seed {scenario.seed}",
                   details={
                       "preset": scenario.preset,
                       "seed": scenario.seed,
                       "items": len(scenario.items),
                       "curated": scenario.curated,
                       "segregation_groups": scenario.segregation_groups,
                       "occupied_volume_m3": round(
                           scenario.total_occupied_volume_mm3 / 1e9, 4),
                   })
        return scenario

    # ------------------------------------------------------------------ #
    # Stage 2/3 — simulated perception
    # ------------------------------------------------------------------ #

    def scan_and_detect(self) -> Dict[str, float]:
        """SIMULATED perception. Publishes ground truth with a confidence label.

        There is no image, no model and no sensor. Each item is "detected" with
        a seeded confidence drawn from the configured range, and optionally
        missed with a configured probability. The action events are marked
        ``simulated`` so nothing downstream can mistake this for perception
        performance.
        """
        if self.scenario is None:
            raise WorkflowError("scan_and_detect before a scenario was loaded")
        watch = Stopwatch()
        self._set_stage(Stage.SCAN_SOURCE_BIN)
        self._emit(Stage.SCAN_SOURCE_BIN, "scan_bin", Actor.PERCEPTION_SIM,
                   duration_ms=watch.lap(), source=Source.SIMULATED,
                   message="simulated bin scan (no RGB-D sensor in this demo)",
                   details={"items_present": len(self.scenario.items)})

        self._set_stage(Stage.DETECT_ITEMS)
        cfg = self.config.robot
        detected: Dict[str, float] = {}
        missed: List[str] = []
        for item in self.scenario.items:
            if self._rng.random() < cfg.detection_miss_probability:
                missed.append(item.item_id)
                continue
            detected[item.item_id] = round(self._rng.uniform(
                cfg.detection_confidence_min, cfg.detection_confidence_max), 3)
        self.detected = detected
        self.stats.detectable_items = len(self.scenario.items)
        self.stats.detected_items = len(detected)

        self._emit(Stage.DETECT_ITEMS, "detect_items", Actor.PERCEPTION_SIM,
                   duration_ms=watch.ms, source=Source.SIMULATED,
                   message=f"{len(detected)}/{len(self.scenario.items)} items "
                           "detected (simulated)",
                   details={"detected": len(detected), "missed": missed,
                            "confidence_range": [cfg.detection_confidence_min,
                                                 cfg.detection_confidence_max],
                            "note": "simulated perception — no vision model exists"})
        return detected

    # ------------------------------------------------------------------ #
    # Stage 4/5/6 — planning and Digital Twin validation
    # ------------------------------------------------------------------ #

    def generate_plans(self, strategy: Optional[Strategy] = None
                       ) -> Tuple[PackingPlan, PackingPlan]:
        if self.scenario is None:
            raise WorkflowError("generate_plans before a scenario was loaded")
        strategy = strategy or self.config.strategy

        watch = Stopwatch()
        self._set_stage(Stage.GENERATE_BASELINE_PLAN)
        self.baseline = pack_baseline(self.scenario, validation=self.config.validation)
        self._emit(Stage.GENERATE_BASELINE_PLAN, "baseline_planned",
                   Actor.OPTIMIZER, duration_ms=watch.lap(),
                   message=f"baseline: {self.baseline.containers_required} "
                           f"containers, {self.baseline.utilization_pct:.1f}% used",
                   details=self.baseline.summary())

        self._set_stage(Stage.GENERATE_OPTIMIZED_PLAN)
        cfg = OptimizerConfig(
            strategy=strategy,
            restarts=self.config.optimizer.restarts,
            time_budget_ms=self.config.optimizer.time_budget_ms,
            improve=self.config.optimizer.improve,
            improve_rounds=self.config.optimizer.improve_rounds,
            seed=self.config.seed,
            validation=self.config.validation)
        self.optimized = pack_optimized(self.scenario, config=cfg)
        self._strategy_plans[strategy.value] = self.optimized
        self._comparison_scenario = self.scenario
        self._emit(Stage.GENERATE_OPTIMIZED_PLAN, "optimized_planned",
                   Actor.OPTIMIZER, duration_ms=watch.lap(),
                   message=f"optimized ({strategy.value}): "
                           f"{self.optimized.containers_required} containers, "
                           f"{self.optimized.utilization_pct:.1f}% used",
                   details=self.optimized.summary())
        return self.baseline, self.optimized

    def digital_twin_validate(self) -> bool:
        """Validate BOTH plans independently, then select one.

        Both are validated, not just the one about to run: the comparison shown
        to the operator is only meaningful if the baseline it is compared against
        was itself checked.
        """
        if self.scenario is None or self.baseline is None or self.optimized is None:
            raise WorkflowError("digital_twin_validate before planning")
        watch = Stopwatch()
        self._set_stage(Stage.DIGITAL_TWIN_VALIDATE)

        base_report = self.validator.validate_plan(self.baseline, self.scenario)
        opt_report = self.validator.validate_plan(self.optimized, self.scenario)

        self.selected, self.selection_reason = select_plan(
            self.baseline, self.optimized, self.scenario,
            STRATEGY_WEIGHTS[self.optimized.strategy])
        self.stats.placements_validated = (base_report.placements_valid
                                           + opt_report.placements_valid)

        for item in self.scenario.items:
            if self.selected.placement_for_item(item.item_id):
                item.status = ItemStatus.PLANNED
            elif item.item_id in self.selected.unplaced_item_ids:
                item.status = ItemStatus.UNPLACED

        ok = self.selected.is_valid
        self._emit(Stage.DIGITAL_TWIN_VALIDATE, "validate_placements",
                   Actor.DIGITAL_TWIN, Result.OK if ok else Result.FAILED,
                   duration_ms=watch.ms,
                   message=(f"selected {self.selected.algorithm}: "
                            f"{self.selection_reason}"),
                   details={
                       "baseline": base_report.to_dict(),
                       "optimized": opt_report.to_dict(),
                       "selected_plan_id": self.selected.plan_id,
                       "selected_algorithm": self.selected.algorithm,
                       "selection_reason": self.selection_reason,
                   })
        return ok

    def compare_strategies(self) -> Dict[str, PackingPlan]:
        """Run all three operator-selectable strategies for side-by-side review."""
        if self.scenario is None:
            raise WorkflowError("compare_strategies before a scenario was loaded")
        watch = Stopwatch()
        for strategy in Strategy:
            if strategy.value in self._strategy_plans:
                continue
            cfg = OptimizerConfig(
                strategy=strategy, restarts=self.config.optimizer.restarts,
                time_budget_ms=self.config.optimizer.time_budget_ms,
                improve=self.config.optimizer.improve, seed=self.config.seed,
                validation=self.config.validation)
            self._strategy_plans[strategy.value] = pack_optimized(
                self.scenario, config=cfg,
                plan_id=f"plan-{strategy.value}-{self.scenario.scenario_id}")
        self._emit(Stage.GENERATE_OPTIMIZED_PLAN, "compare_strategies",
                   Actor.OPTIMIZER, duration_ms=watch.ms,
                   message="all packing strategies evaluated",
                   details={k: v.summary() for k, v in
                            sorted(self._strategy_plans.items())})
        return dict(self._strategy_plans)

    @property
    def strategy_plans(self) -> Dict[str, PackingPlan]:
        return dict(self._strategy_plans)

    # ------------------------------------------------------------------ #
    # Stage 7 — Human in the Loop
    # ------------------------------------------------------------------ #

    def request_approval(self) -> None:
        if self.selected is None:
            raise WorkflowError("request_approval before a plan was selected")
        self._set_stage(Stage.WAIT_FOR_OPERATOR_APPROVAL)
        self.selected.approval_state = ApprovalState.PENDING
        self._emit(Stage.WAIT_FOR_OPERATOR_APPROVAL, "await_approval",
                   Actor.ORCHESTRATOR, Result.PENDING,
                   message="plan awaiting operator decision — no physical action "
                           "is authorised",
                   details=self.selected.summary())

    def approve(self, operator: str = "operator", auto: bool = False) -> None:
        if self.selected is None:
            raise WorkflowError("approve before a plan was selected")
        if self.stage is not Stage.WAIT_FOR_OPERATOR_APPROVAL:
            raise WorkflowError(
                f"approve called in stage {self.stage.value}, expected "
                f"{Stage.WAIT_FOR_OPERATOR_APPROVAL.value}")
        self.selected.approval_state = ApprovalState.APPROVED
        if not auto:
            self.stats.operator_interventions += 1
        self._emit(Stage.WAIT_FOR_OPERATOR_APPROVAL, "approve_plan",
                   Actor.SYSTEM if auto else Actor.OPERATOR,
                   source=Source.MEASURED if auto else Source.OPERATOR,
                   message=("auto-approved (headless acceptance run)" if auto
                            else f"plan approved by {operator}"),
                   details={"plan_id": self.selected.plan_id, "auto": auto})
        self._set_stage(Stage.PICK_ITEM)
        self.cursor.reset()
        self._t_exec_start = time.monotonic()

    def reject(self, reason: str = "operator rejected the plan",
               operator: str = "operator") -> None:
        """Reject the plan. Triggers a re-plan, never an execution."""
        if self.selected is None:
            raise WorkflowError("reject before a plan was selected")
        self.selected.approval_state = ApprovalState.REJECTED
        self.stats.operator_interventions += 1
        self._emit(Stage.WAIT_FOR_OPERATOR_APPROVAL, "reject_plan",
                   Actor.OPERATOR, Result.REJECTED, source=Source.OPERATOR,
                   message=reason,
                   details={"plan_id": self.selected.plan_id,
                            "operator": operator})
        self.replan(f"operator rejection: {reason}")

    # ------------------------------------------------------------------ #
    # Stage 8-13 — simulated execution
    # ------------------------------------------------------------------ #

    def _assert_approved(self) -> None:
        """The safety invariant. Checked, not trusted."""
        if self.selected is None:
            raise ApprovalRequired("no plan selected")
        if self.selected.approval_state is not ApprovalState.APPROVED:
            raise ApprovalRequired(
                f"plan {self.selected.plan_id} is "
                f"{self.selected.approval_state.value}; physical action requires "
                "an approved plan")
        if self.stage in PRE_APPROVAL_STAGES:
            raise ApprovalRequired(
                f"stage {self.stage.value} precedes approval; refusing to act")

    def step_execution(self) -> bool:
        """Execute one placement. Returns False when nothing remains to do.

        One call == one full pick/verify/place/verify/update sequence for one
        item, which is what makes the behaviour tree's tick and the dashboard's
        animation frame the same unit of work.
        """
        self._assert_approved()
        assert self.selected is not None                # narrowed by _assert_approved

        # A dynamic event scheduled for this point fires before the pick.
        self._fire_due_events(placement_index=self.cursor.index)
        if self.stage is Stage.REPLAN:
            return True                                  # re-plan already queued

        placements = self.selected.ordered_placements
        pending = [p for p in placements if not p.executed]
        if not pending:
            return self._complete()

        placement = pending[0]
        item = self.scenario.item(placement.item_id) if self.scenario else None
        container = self.selected.container(placement.container_id)
        if item is None or container is None:            # pragma: no cover
            raise WorkflowError(f"placement {placement.item_id} lost its item "
                                "or container")

        self.current_item_id = item.item_id
        self.current_container_id = container.container_id

        # -- PICK ---------------------------------------------------------- #
        self._set_stage(Stage.PICK_ITEM)
        self.robot_state = "picking"
        watch = Stopwatch()
        self.stats.pick_attempts += 1
        if self._force_pick_failure:
            grasp_ok = False
            self._force_pick_failure = False          # one forced failure only
        else:
            grasp_ok = (self._rng.random()
                        >= self.config.robot.pick_failure_probability)
        self._emit(Stage.PICK_ITEM, "pick_item", Actor.ROBOT_SIM,
                   Result.OK if grasp_ok else Result.FAILED,
                   item_id=item.item_id, container_id=container.container_id,
                   duration_ms=self.config.robot.pick_duration_ms,
                   source=Source.SIMULATED,
                   message=f"simulated pick of {item.item_id}",
                   details={"attempt": self.cursor.retries + 1,
                            "failure_probability":
                                self.config.robot.pick_failure_probability})

        # -- VERIFY PICK ---------------------------------------------------- #
        self._set_stage(Stage.VERIFY_PICK)
        if not grasp_ok:
            return self._handle_pick_failure(placement, item.item_id,
                                             container.container_id)

        self.stats.pick_successes += 1
        self.cursor.retries = 0
        item.status = ItemStatus.PICKED
        self._emit(Stage.VERIFY_PICK, "verify_pick", Actor.ROBOT_SIM,
                   item_id=item.item_id, container_id=container.container_id,
                   duration_ms=watch.lap(), source=Source.SIMULATED,
                   message="grasp confirmed (simulated)")

        # -- PLACE ---------------------------------------------------------- #
        self._set_stage(Stage.PLACE_ITEM)
        self.robot_state = "placing"
        self._emit(Stage.PLACE_ITEM, "place_item", Actor.ROBOT_SIM,
                   item_id=item.item_id, container_id=container.container_id,
                   duration_ms=self.config.robot.place_duration_ms,
                   source=Source.SIMULATED,
                   message=f"simulated place into {container.container_id}",
                   details={"position": placement.position.to_dict(),
                            "axis": placement.axis.value})

        # -- VERIFY PLACEMENT ----------------------------------------------- #
        # Re-checks the ACTUAL geometry against the plan rather than assuming the
        # place succeeded because it was planned. In the real system this is the
        # Digital Twin comparing observed pose to intended pose.
        self._set_stage(Stage.VERIFY_PLACEMENT)
        geometry_ok = (placement.validation_status is ValidationStatus.VALID
                       and placement.box.within(container.inner_size))
        self._emit(Stage.VERIFY_PLACEMENT, "verify_placement", Actor.DIGITAL_TWIN,
                   Result.OK if geometry_ok else Result.FAILED,
                   item_id=item.item_id, container_id=container.container_id,
                   duration_ms=watch.lap(),
                   message="placement re-validated against container geometry",
                   details={"validation_status": placement.validation_status.value})
        if not geometry_ok:
            self.replan(f"placement verification failed for {item.item_id}")
            return True

        # -- UPDATE CONTAINER STATE ----------------------------------------- #
        self._set_stage(Stage.UPDATE_CONTAINER_STATE)
        placement.executed = True
        item.status = ItemStatus.PLACED
        container.status = ContainerStatus.FILLING
        remaining = [p for p in self.selected.placements_for(container.container_id)
                     if not p.executed]
        if not remaining:
            container.status = ContainerStatus.COMPLETE
        self.robot_state = "idle"
        self._emit(Stage.UPDATE_CONTAINER_STATE, "update_container",
                   Actor.ORCHESTRATOR, item_id=item.item_id,
                   container_id=container.container_id, duration_ms=watch.lap(),
                   message=f"{container.container_id} now "
                           f"{container.status.value}",
                   details={"executed": sum(1 for p in self.selected.placements
                                            if p.executed),
                            "total": len(self.selected.placements),
                            "progress_pct": round(self.progress_pct, 1)})

        self.stats.cycles_attempted += 1
        self.stats.cycles_completed += 1
        self.cursor.index += 1
        self._set_stage(Stage.NEXT_ITEM)
        self._fire_due_events(elapsed_s=self._elapsed_s())
        return True

    def _handle_pick_failure(self, placement: Placement, item_id: str,
                             container_id: str) -> bool:
        """Retry a failed grasp; give up on the item after the retry budget."""
        self.cursor.retries += 1
        if self.cursor.retries <= self.config.robot.max_pick_retries:
            self._emit(Stage.VERIFY_PICK, "retry_pick", Actor.ROBOT_SIM,
                       Result.RETRY, item_id=item_id, container_id=container_id,
                       source=Source.SIMULATED,
                       message=f"grasp failed — retry "
                               f"{self.cursor.retries}/"
                               f"{self.config.robot.max_pick_retries}")
            return True
        self._emit(Stage.VERIFY_PICK, "abandon_item", Actor.ROBOT_SIM,
                   Result.FAILED, item_id=item_id, container_id=container_id,
                   source=Source.SIMULATED,
                   message=f"grasp failed {self.cursor.retries}x — item "
                           "abandoned, cycle counted as failed")
        placement.executed = True                      # removed from the queue
        placement.validation_status = ValidationStatus.INVALID
        self.stats.cycles_attempted += 1               # attempted, not completed
        if self.scenario:
            item = self.scenario.item(item_id)
            if item:
                item.status = ItemStatus.UNPLACED
        self.cursor.retries = 0
        self.cursor.index += 1
        return True

    def _complete(self) -> bool:
        self._set_stage(Stage.COMPLETE)
        self.finished = True
        self.robot_state = "idle"
        self.current_item_id = None
        executed = sum(1 for p in (self.selected.placements if self.selected else [])
                       if p.executed)
        self._emit(Stage.COMPLETE, "cycle_complete", Actor.ORCHESTRATOR,
                   message=f"packaging complete — {executed} placements executed",
                   details={
                       "executed": executed,
                       "containers": self.selected.containers_required if self.selected else 0,
                       "replans": self.stats.replans,
                       "pick_attempts": self.stats.pick_attempts,
                       "pick_successes": self.stats.pick_successes,
                   })
        return False

    def _elapsed_s(self) -> float:
        return 0.0 if self._t_exec_start is None else time.monotonic() - self._t_exec_start

    # ------------------------------------------------------------------ #
    # Re-planning
    # ------------------------------------------------------------------ #

    def replan(self, cause: str) -> None:
        """Pause, re-plan around the current physical state, and re-request approval.

        Already-executed placements are FROZEN: the items are physically in the
        containers and re-planning cannot move them. Only the remainder is
        re-optimized. Getting this wrong would produce a plan that instructs the
        robot to place an item into a space another item already occupies.
        """
        if self.scenario is None:
            raise WorkflowError("replan before a scenario was loaded")
        if self.stats.replans >= self.config.max_replans:
            self.enter_degraded(f"re-plan limit ({self.config.max_replans}) "
                                f"reached; last cause: {cause}")
            return

        watch = Stopwatch()
        self._set_stage(Stage.REPLAN)
        self.stats.replans += 1
        self.stats.replan_causes.append(cause)
        if self.selected and self.selected.approval_state is ApprovalState.PENDING:
            self.selected.approval_state = ApprovalState.SUPERSEDED

        executed = [p for p in (self.selected.placements if self.selected else [])
                    if p.executed]
        frozen_ids = {p.item_id for p in executed}

        # Remember every container this plan touched, and retire the unavailable
        # ones permanently. Advancing the index offset past the highest id in
        # use guarantees the next plan cannot re-issue a retired id.
        for container in (self.selected.containers if self.selected else []):
            if container.status is ContainerStatus.UNAVAILABLE:
                self._retired_containers[container.container_id] = container
            suffix = container.container_id.rsplit("-", 1)[-1]
            if suffix.isdigit():
                self._container_index_offset = max(
                    self._container_index_offset, int(suffix))

        self._emit(Stage.REPLAN, "replan_start", Actor.ORCHESTRATOR,
                   duration_ms=watch.lap(), message=f"re-planning: {cause}",
                   details={"cause": cause, "replan_number": self.stats.replans,
                            "frozen_placements": len(executed)})

        # Re-plan over the not-yet-executed items only.
        remaining = Scenario(
            scenario_id=self.scenario.scenario_id,
            preset=self.scenario.preset,
            seed=self.scenario.seed,
            items=[i for i in self.scenario.items
                   if i.item_id not in frozen_ids
                   and i.status is not ItemStatus.REMOVED],
            container_template=self.scenario.container_template,
            max_containers=self.scenario.max_containers,
            curated=self.scenario.curated)

        exec_baseline = pack_baseline(
            remaining, validation=self.config.validation,
            container_index_offset=self._container_index_offset)
        cfg = OptimizerConfig(
            strategy=self.config.strategy,
            restarts=self.config.optimizer.restarts,
            time_budget_ms=self.config.optimizer.time_budget_ms,
            improve=self.config.optimizer.improve,
            seed=self.config.seed + self.stats.replans,
            container_index_offset=self._container_index_offset,
            validation=self.config.validation)
        exec_optimized = pack_optimized(
            remaining, config=cfg,
            plan_id=f"plan-optimized-{remaining.scenario_id}-r{self.stats.replans}")

        base_ok = self.validator.validate_plan(exec_baseline, remaining)
        opt_ok = self.validator.validate_plan(exec_optimized, remaining)
        self.selected, self.selection_reason = select_plan(
            exec_baseline, exec_optimized, remaining,
            STRATEGY_WEIGHTS[exec_optimized.strategy])
        self.stats.placements_validated += (base_ok.placements_valid
                                            + opt_ok.placements_valid)

        # Re-attach the frozen placements AND the containers holding them, so
        # the new plan still describes the whole physical state — including any
        # container that has been retired mid-run.
        known = {c.container_id for c in self.selected.containers}
        for p in executed:
            if not self.selected.placement_for_item(p.item_id):
                self.selected.placements.append(p)
            if p.container_id not in known:
                carried = self._retired_containers.get(p.container_id)
                if carried is None:
                    carried = self.scenario.container_template.respec(p.container_id)
                    carried.status = ContainerStatus.FILLING
                self.selected.containers.append(carried)
                known.add(p.container_id)
        for n, p in enumerate(sorted(self.selected.placements,
                                     key=lambda q: (not q.executed,
                                                    q.container_id,
                                                    q.position.z, q.item_id)), 1):
            p.placement_order = n

        # Refresh the comparison pair over the WHOLE current scenario. Without
        # this, KPI4 would compare a partial re-plan against a full-batch
        # baseline and report a meaningless (often negative) "reduction".
        self._refresh_comparison()

        self._emit(Stage.REPLAN, "replan_complete", Actor.OPTIMIZER,
                   duration_ms=watch.ms,
                   message=f"re-plan {self.stats.replans}: "
                           f"{self.selected.containers_required} containers, "
                           f"{self.selection_reason}",
                   details=self.selected.summary())

        self.cursor.reset()
        self.request_approval()
        if self.config.auto_approve:
            self.approve(auto=True)

    def _refresh_comparison(self) -> None:
        """Re-run BOTH algorithms over the full current scenario, for reporting.

        Executed placements are irrelevant here: the question KPI4 answers is
        "how many containers would each algorithm need for this batch", and that
        is a property of the batch, not of how far execution has got. Removed
        items are excluded because they are no longer part of the batch.
        """
        if self.scenario is None:
            return
        full = Scenario(
            scenario_id=self.scenario.scenario_id,
            preset=self.scenario.preset,
            seed=self.scenario.seed,
            items=[i for i in self.scenario.items
                   if i.status is not ItemStatus.REMOVED],
            container_template=self.scenario.container_template,
            max_containers=self.scenario.max_containers,
            curated=self.scenario.curated)
        self.baseline = pack_baseline(full, validation=self.config.validation)
        self.optimized = pack_optimized(full, config=OptimizerConfig(
            strategy=self.config.strategy,
            restarts=self.config.optimizer.restarts,
            time_budget_ms=self.config.optimizer.time_budget_ms,
            improve=self.config.optimizer.improve,
            seed=self.config.seed,
            validation=self.config.validation))
        self.validator.validate_plan(self.baseline, full)
        self.validator.validate_plan(self.optimized, full)
        self._comparison_scenario = full

    def enter_degraded(self, reason: str) -> None:
        """Stop and hold. The demo never simulates unsafe autonomous continuation."""
        self._set_stage(Stage.DEGRADED)
        self.degraded_reason = reason
        self.robot_state = "held"
        self.finished = True
        self._emit(Stage.DEGRADED, "enter_degraded", Actor.ORCHESTRATOR,
                   Result.FAILED, message=f"DEGRADED — held: {reason}",
                   details={"reason": reason})

    # ------------------------------------------------------------------ #
    # Dynamic events
    # ------------------------------------------------------------------ #

    def _fire_due_events(self, *, placement_index: Optional[int] = None,
                         elapsed_s: Optional[float] = None) -> None:
        for event in self.dynamic_events:
            if event.matches(stage=self.stage, placement_index=placement_index,
                             elapsed_s=elapsed_s):
                self.apply_dynamic_event(event)

    def apply_dynamic_event(self, event: DynamicEvent) -> None:
        """Apply one scripted disturbance and re-plan if it invalidates the plan."""
        if self.scenario is None:
            raise WorkflowError("dynamic event before a scenario was loaded")
        event.fired = True
        watch = Stopwatch()
        kind = event.event_type
        payload = event.payload
        detail: Dict[str, Any] = {"event_type": kind.value, "trigger": event.trigger}

        if kind is DynamicEventType.ITEM_INJECT:
            item = inject_item(self.scenario, payload.get("item", payload))
            detail["item_id"] = item.item_id
            detail["length_mm"] = item.length_mm
            detail["outer_diameter_mm"] = item.outer_diameter_mm

        elif kind is DynamicEventType.ITEM_REMOVED:
            item = self.scenario.item(payload.get("item_id", ""))
            if item:
                item.status = ItemStatus.REMOVED
                detail["item_id"] = item.item_id

        elif kind is DynamicEventType.ITEM_RECLASSIFIED:
            item = self.scenario.item(payload.get("item_id", ""))
            if item:
                old = item.segregation_group
                item.segregation_group = payload.get("segregation_group", old)
                item.dose_class = payload.get("dose_class", item.dose_class)
                detail.update({"item_id": item.item_id, "from": old,
                               "to": item.segregation_group})

        elif kind is DynamicEventType.CONTAINER_UNAVAILABLE:
            cid = payload.get("container_id", "")
            for container in self.containers:
                if container.container_id == cid:
                    container.status = ContainerStatus.UNAVAILABLE
                    detail["container_id"] = cid

        elif kind is DynamicEventType.CONTAINER_RESTORED:
            cid = payload.get("container_id", "")
            for container in self.containers:
                if container.container_id == cid:
                    container.status = ContainerStatus.AVAILABLE
                    detail["container_id"] = cid

        elif kind is DynamicEventType.GRASP_FAILURE:
            # Force exactly the next pick to fail, through the same code path a
            # random failure takes. A flag, not an RNG re-seed: re-seeding only
            # made a failure *likely*, which is not what "inject a failure"
            # means and made the behaviour untestable.
            self._force_pick_failure = True
            detail["note"] = "next simulated grasp forced to fail"

        elif kind is DynamicEventType.SEGREGATION_RULE_CHANGE:
            groups = tuple(payload.get("allowed_segregation_groups", ()))
            for container in self.containers:
                container.allowed_segregation_groups = groups
            detail["allowed_segregation_groups"] = list(groups)

        elif kind is DynamicEventType.OPTIMIZER_TIMEOUT:
            self.config.optimizer.time_budget_ms = float(
                payload.get("time_budget_ms", 50.0))
            detail["time_budget_ms"] = self.config.optimizer.time_budget_ms

        elif kind is DynamicEventType.OPERATOR_REJECT:
            self.stats.operator_interventions += 1
            detail["reason"] = payload.get("reason", "operator rejected the plan")

        self._emit(Stage.REPLAN if event.forces_replan else self.stage,
                   f"dynamic_event:{kind.value}", Actor.EVENT_INJECTOR,
                   duration_ms=watch.ms, source=Source.SIMULATED,
                   message=event.label, details=detail)

        if event.forces_replan:
            self.replan(f"dynamic event {kind.value}: {event.label}")

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def kpis(self, latency_p50_ms: Optional[float] = None) -> KPIReport:
        if self.scenario is None or self.baseline is None \
                or self.optimized is None or self.selected is None:
            raise WorkflowError("kpis() before planning completed")
        self.stats.fiware_events_logged = self.log.count
        # The reduction operands are the comparison pair, never `selected`:
        # after a re-plan `selected` carries frozen placements the baseline has
        # no counterpart for, so comparing them is not apples-to-apples.
        return compute_kpis(self._comparison_scenario or self.scenario,
                            self.baseline, self.optimized,
                            self.selected, self.stats, self.log,
                            latency_p50_ms, self.run_id)

    def snapshot(self) -> Dict[str, Any]:
        """Complete state for the dashboard's REST initial render."""
        return {
            "run_id": self.run_id,
            "cycle_id": self.cycle_id,
            "stage": self.stage.value,
            "finished": self.finished,
            "degraded_reason": self.degraded_reason,
            "robot_state": self.robot_state,
            "current_item_id": self.current_item_id,
            "current_container_id": self.current_container_id,
            "progress_pct": round(self.progress_pct, 1),
            "scenario": self.scenario.to_dict() if self.scenario else None,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "optimized": self.optimized.to_dict() if self.optimized else None,
            "selected": self.selected.to_dict() if self.selected else None,
            "selected_plan_id": self.selected.plan_id if self.selected else None,
            "selection_reason": self.selection_reason,
            "approval_state": (self.selected.approval_state.value
                               if self.selected else "pending"),
            "detected": self.detected,
            "dynamic_events": [e.to_dict() for e in self.dynamic_events],
            "stats": self.stats.to_dict(),
            "action_count": self.log.count,
        }


# --------------------------------------------------------------------------- #
# Headless run
# --------------------------------------------------------------------------- #


def run_headless(config: WorkflowConfig,
                 on_event: Optional[Callable[[Any], None]] = None,
                 max_steps: int = 5000) -> WorkflowEngine:
    """Drive one complete cycle with no UI. Used by the acceptance demo and tests."""
    engine = WorkflowEngine(config)
    if on_event:
        engine.log.add_sink(on_event)

    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    engine.generate_plans()
    engine.digital_twin_validate()
    engine.request_approval()
    engine.approve(auto=True)

    steps = 0
    while not engine.finished and steps < max_steps:
        if not engine.step_execution():
            break
        steps += 1
    if steps >= max_steps:
        engine.enter_degraded(f"exceeded {max_steps} execution steps")
    return engine


__all__ = [
    "ApprovalRequired", "WorkflowError", "RobotSimConfig", "WorkflowConfig",
    "WorkflowEngine", "run_headless",
]
