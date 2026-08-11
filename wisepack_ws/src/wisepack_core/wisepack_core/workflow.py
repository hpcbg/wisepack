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
    Placement, Scenario, Source, Strategy, ValidationStatus, WasteItem,
)
from .anomaly import AnomalyEvent, Reaction
from .events import (
    Actor, ActionLog, DynamicEvent, DynamicEventType, PRE_APPROVAL_STAGES,
    Result, Stage, Stopwatch, utc_now_iso,
)
from .execution import ExecutionBackend
from .generator import build_scenario, inject_item
from .perception import (
    ObservationBatch, PerceptionSource, ProxyGeometry, WorkAreaFrame,
    is_stale, observation_age_s,
)
from .kpi import ExecutionStats, KPIReport, compute_kpis
from .packing import (
    OptimizerConfig, STRATEGY_WEIGHTS, pack_baseline, pack_optimized, select_plan,
)
from .validator import DEFAULT_VALIDATION, PlacementValidator, ValidationConfig


class ApprovalRequired(RuntimeError):
    """Raised when execution is attempted on a plan the operator has not approved."""


class AnomalyHold(RuntimeError):
    """Raised when execution is attempted while an anomaly holds the workflow."""


class PerceptionUnavailable(RuntimeError):
    """Raised when a REAL perception source could not deliver observations.

    Deliberately NOT caught-and-substituted anywhere. §15 forbids a failed
    camera scan from silently falling back to simulated detections while the
    interface still claims a real camera, so a physical perception failure
    stops the run and shows itself. Switching back to the simulator is an
    explicit configuration change by the operator, never an automatic one.
    """


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
    #: Who performs the approved placements. SIMULATED keeps the existing seeded
    #: robot model; ISAAC hands each placement to Isaac Sim and waits for the
    #: physical outcome. Planning, validation and the approval gate are identical
    #: either way — see wisepack_core.execution.
    execution_backend: ExecutionBackend = ExecutionBackend.SIMULATED
    #: WHERE THE OBJECT OBSERVATIONS COME FROM. Orthogonal to
    #: `execution_backend` above and to the dashboard's data source: a camera is
    #: not a way of executing and not a way of reading state. SIM is the default
    #: and its behaviour is byte-identical to before this field existed.
    perception_source: PerceptionSource = PerceptionSource.SIM
    #: The KNOWN dimensions of the physical proxy cylinders. Used only when
    #: `perception_source` is physical — a detector measures position, not size,
    #: so the geometry the packer needs is declared rather than invented.
    proxy_geometry: ProxyGeometry = field(default_factory=ProxyGeometry)
    #: How the detector's calibrated plane maps onto the WISEPACK work area.
    work_area: WorkAreaFrame = field(default_factory=WorkAreaFrame)
    #: WHICH ROBOT executes, by id from config/isaac_robots.yaml. Meaningful
    #: only for the Isaac backend — a simulated run has no robot, and the
    #: dashboard shows "Logical workflow simulator" rather than a robot name.
    #:
    #: Recorded on the ACTIVE run so its artefacts, its FIWARE projection and
    #: its scene handshake all name the arm that actually moved. Changing the
    #: dashboard's draft does not touch this: only a reset (a new run) does, and
    #: that is the whole reason the two are separate fields.
    robot_id: str = ""

    def __post_init__(self) -> None:
        # Same coercion as OptimizerConfig: strategies reach here as strings from
        # YAML and from the dashboard.
        self.strategy = Strategy(self.strategy)
        self.execution_backend = ExecutionBackend(self.execution_backend)
        self.perception_source = PerceptionSource(self.perception_source)
        self.robot_id = str(self.robot_id or "").strip().lower()


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
        # -- anomaly monitoring & workflow response state -------------------- #
        self.anomaly_hold = False                # execution blocked by an anomaly
        self.anomaly_ack_required = False        # operator must acknowledge
        self.latest_anomaly: Optional[Dict[str, Any]] = None
        self.anomaly_history: List[Dict[str, Any]] = []
        self._anomaly_seq = 0
        self._rng = random.Random(self.config.robot.seed)
        self._t_exec_start: Optional[float] = None
        self._strategy_plans: Dict[str, PackingPlan] = {}
        # Container ids are issued from a run-global counter. A re-plan must not
        # re-issue the id of a retired (unavailable) container, or the
        # retirement is silently undone and the packer fills a box that is out
        # of service. See replan().
        self._comparison_scenario: Optional[Scenario] = None
        # A monotonically increasing revision of the item/container problem.
        # It bumps whenever the batch the optimizer sees changes (new scenario,
        # injected/removed/reclassified item, container retired, re-plan). A
        # strategy comparison is stamped with the revision it was computed
        # against, so a comparison from a superseded batch is never rendered.
        self.scenario_revision = 0
        #: WHICH (revision, plan) the current approval decision refers to.
        #: An approval is not a boolean, it is a decision about one specific
        #: plan of one specific batch revision. Without the stamp, an approval
        #: granted for the previous scenario silently authorises the next one.
        self.approval_revision = 0
        self.approval_plan_id: Optional[str] = None
        #: revision -> structured strategy-comparison dict (see build_...).
        self._strategy_comparison: Optional[Dict[str, Any]] = None
        self._container_index_offset = 0
        self._retired_containers: Dict[str, Container] = {}
        # Set by the grasp_failure dynamic event. An explicit flag rather than
        # re-seeding the RNG and hoping its next draw lands below the failure
        # threshold — that was not actually deterministic.
        self._force_pick_failure = False
        # -- physical perception ------------------------------------------- #
        #: HOW A PHYSICAL BATCH IS FETCHED. Injected rather than imported: the
        #: engine is ROS-free and transport-free by design, so it must not know
        #: whether the observations arrive over HTTP from the detector service
        #: or over DDS from the adapter node. Called with no arguments and must
        #: return an ObservationBatch (a FAILED one, not an exception, when the
        #: detector is unhappy). None while the perception source is `sim`.
        self.observation_provider: Optional[Callable[[], ObservationBatch]] = None
        #: THE CURRENT physical observation. Singular on purpose: a re-detection
        #: REPLACES it, so stale objects cannot accumulate (§6).
        self.observation_batch: Optional[ObservationBatch] = None
        self.observation_batches_applied = 0

        # The whole-process layer (cut-aware planning + container inventory +
        # simulated logistics). Held here but self-contained in whole_process.py
        # so the core packing workflow above is untouched.
        from .whole_process import WholeProcess    # noqa: PLC0415 (avoid cycle)
        self.wp = WholeProcess(self)

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
        self._bump_scenario_revision()
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

    @property
    def objects_detected(self) -> bool:
        """Have the objects to plan from been observed?

        NOT `bool(self.detected)`, AND THE DIFFERENCE IS A REAL DEFECT THIS
        REPLACED. `detected` maps item id -> CONFIDENCE, and it is filled only
        for observations that carry one. A FoundationPose observation carries
        none on purpose — the estimator's score ranks pose hypotheses against
        each other and is never rendered as a probability — so a perfectly good
        RGB-D batch of one Cylinder5 left `detected` empty. The behaviour tree
        read that as "nothing detected", re-entered ScanAndDetect and parked the
        workflow at DETECT_ITEMS forever, with the item sitting in the scenario.

        The question the tree is really asking is whether objects were observed.
        Whether each came with a confidence is a different question, and one
        detector legitimately answers it "no".
        """
        if self.detected:
            return True
        batch = self.observation_batch
        return bool(batch is not None and batch.ok
                    and self.scenario is not None and self.scenario.items)

    def scan_and_detect(self) -> Dict[str, float]:
        """Acquire the object observations this run will plan from.

        ONE SEAM, TWO SOURCES, selected by ``config.perception_source``:

          sim     the existing simulated detector — unchanged, still the
                  default, still labelled `simulated` everywhere.
          camera  a real camera frame, measured by whichever perception
                  provider is configured, as domain-neutral observations.

        Everything after this method is identical in both cases: the planner,
        the validator, the approval gate and the execution backend cannot tell
        which one ran, which is exactly the property §10 asks for.
        """
        if self.scenario is None:
            raise WorkflowError("scan_and_detect before a scenario was loaded")
        if self.config.perception_source.is_physical:
            return self._scan_physical()
        return self._scan_simulated()

    def _scan_simulated(self) -> Dict[str, float]:
        """SIMULATED perception. Publishes ground truth with a confidence label.

        There is no image, no model and no sensor. Each item is "detected" with
        a seeded confidence drawn from the configured range, and optionally
        missed with a configured probability. The action events are marked
        ``simulated`` so nothing downstream can mistake this for perception
        performance.
        """
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
    # Stage 2/3 — physical perception
    # ------------------------------------------------------------------ #

    def _scan_physical(self) -> Dict[str, float]:
        """REAL perception: request one detection and adopt what it returns.

        DETERMINISTIC ONE-SHOT (§6). Inference runs once per request, not
        continuously: a Faster R-CNN pass is expensive, and a plan built from a
        frame that keeps changing under it is not reproducible. The operator
        asks; the detector answers; that answer is the batch.
        """
        watch = Stopwatch()
        self._set_stage(Stage.SCAN_SOURCE_BIN)
        source = self.config.perception_source
        self._emit(Stage.SCAN_SOURCE_BIN, "scan_bin", Actor.PERCEPTION_CAMERA,
                   duration_ms=watch.lap(), source=Source.MEASURED,
                   message=f"physical work-area scan requested "
                           f"({source.value})",
                   details={"perception_source": source.value,
                            "frame_id": self.config.work_area.frame_id})

        if self.observation_provider is None:
            batch = ObservationBatch.failed(
                batch_id=f"batch-{self.observation_batches_applied + 1:03d}",
                source=source.value,
                error=("no perception provider is connected — "
                       f"{source.value} was selected but nothing is wired to "
                       "fetch detections. Start the WISEPACK perception service "
                       "and check WISEPACK_PERCEPTION_SERVICE_URL."))
        else:
            try:
                batch = self.observation_provider()
            except Exception as exc:                        # noqa: BLE001
                # A provider that raises is a failed scan, not a crashed
                # workflow: §5 requires camera or model failure not to take
                # unrelated WISEPACK components down with it.
                batch = ObservationBatch.failed(
                    batch_id=f"batch-{self.observation_batches_applied + 1:03d}",
                    source=source.value,
                    error=f"perception provider failed: {exc}")
        return self.apply_observation_batch(batch, _watch=watch)

    def apply_observation_batch(self, batch: ObservationBatch,
                                _watch: Optional[Stopwatch] = None
                                ) -> Dict[str, float]:
        """Adopt one physical observation batch as the batch WISEPACK plans from.

        REPLACEMENT, NOT ACCUMULATION (§6). The scenario's item list is rebuilt
        wholesale from this batch, so detecting again after moving the objects
        yields exactly the objects now on the table — never those plus the ones
        that used to be there. The container specification and `max_containers`
        still come from the preset: those describe the packaging target, which
        no camera observes.

        A FAILED batch is recorded and raised. It never becomes an empty
        successful scan and it never falls back to the simulator.
        """
        if self.scenario is None:
            raise WorkflowError("observation batch before a scenario was loaded")
        watch = _watch or Stopwatch()
        self._set_stage(Stage.DETECT_ITEMS)
        # Recorded BEFORE the failure check: the dashboard has to be able to
        # render why the last scan failed, which means the failed batch must be
        # the current state rather than something that was thrown away.
        self.observation_batch = batch

        if not batch.ok:
            self._emit(Stage.DETECT_ITEMS, "detect_items", Actor.PERCEPTION_CAMERA,
                       result=Result.FAILED, duration_ms=watch.ms,
                       source=Source.MEASURED,
                       message=f"physical detection failed: {batch.error}",
                       details={"perception_source": batch.source,
                                "error": batch.error,
                                "calibration_status": batch.calibration_status,
                                "detector": batch.detector,
                                "note": ("no fallback to simulated detections — "
                                         "the perception source is unchanged "
                                         "and reports failure")})
            raise PerceptionUnavailable(batch.error)

        items = batch.to_waste_items(
            geometry=self.config.proxy_geometry,
            permitted_axes=("x", "y"))
        self.scenario.items = items
        self.observation_batches_applied += 1
        self._bump_scenario_revision()

        detected: Dict[str, float] = {}
        for item in items:
            confidence = item.observation.confidence if item.observation else None
            # An observation without a confidence is still an observation. It is
            # recorded as detected with no confidence figure rather than being
            # assigned an invented one.
            if confidence is not None:
                detected[item.item_id] = round(confidence, 3)
        self.detected = detected
        # DELIBERATELY NOT `detectable_items = len(items)`. The denominator of a
        # detection rate is how many objects were ACTUALLY THERE, and no camera
        # knows that — only a ground-truth trial does. Leaving it at 0 is what
        # makes kpi.py report KPI1 as "not measured" instead of a fabricated
        # 100%. See §12.
        self.stats.detectable_items = 0
        self.stats.detected_items = len(items)

        self._emit(Stage.DETECT_ITEMS, "detect_items", Actor.PERCEPTION_CAMERA,
                   duration_ms=watch.ms, source=Source.MEASURED,
                   message=f"{len(items)} physical objects detected "
                           f"({batch.source})",
                   details={
                       "perception_source": batch.source,
                       "batch_id": batch.batch_id,
                       "detected": len(items),
                       "status": batch.status.value,
                       "detector": batch.detector,
                       "model_id": batch.model_id,
                       "frame_id": batch.frame_id,
                       "calibration_status": batch.calibration_status,
                       "captured_at": batch.captured_at,
                       "scenario_revision": self.scenario_revision,
                       "proxy_geometry": self.config.proxy_geometry.to_dict(),
                       # Mean of the detector's own confidences. NOT a detection
                       # rate and never reported as one.
                       "mean_confidence": batch.mean_confidence,
                       "note": ("real detector output; the observed objects are "
                                "physical proxies for cylindrical workpieces. "
                                "Detector confidence is not a measured "
                                "detection rate."),
                   })
        return detected

    def perception_state(self) -> Dict[str, Any]:
        """The current perception state, for the dashboard and diagnostics.

        Always answers, in every mode. In `sim` it says so plainly rather than
        rendering an empty physical panel.
        """
        source = self.config.perception_source
        batch = self.observation_batch
        payload: Dict[str, Any] = {
            "perception_source": source.value,
            "perception_source_label": source.label,
            "perception_source_detail": source.detail,
            "physical": source.is_physical,
            "batch": batch.to_dict() if batch else None,
            "batches_applied": self.observation_batches_applied,
            "scene_objects": batch.scene_objects() if batch and batch.ok else [],
            "proxy_geometry": (self.config.proxy_geometry.to_dict()
                               if source.is_physical else None),
            "work_area": (self.config.work_area.to_dict()
                          if source.is_physical else None),
            "provider_connected": self.observation_provider is not None,
        }
        age = observation_age_s(batch)
        payload["observation_age_s"] = None if age is None else round(age, 1)
        payload["observation_stale"] = is_stale(batch)
        return payload

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

    def build_strategy_comparison(
            self, strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compute a structured, independently-validated strategy comparison.

        This is decision support. It runs the optimizer once per strategy over
        the CURRENT batch and validates each result with the same independent
        validator the workflow uses — and it mutates NOTHING: not the selected
        plan, not the approved plan, not the execution plan, not the approval
        state. A comparison must never be a second source of planning truth.

        The result is stamped with ``scenario_revision`` so a consumer can tell
        whether it still describes the batch on screen. It is cached per
        revision; the cache is dropped by ``_bump_scenario_revision`` whenever
        the batch changes.
        """
        if self.scenario is None:
            raise WorkflowError("compare_strategies before a scenario was loaded")

        rev = self.scenario_revision
        if self._strategy_comparison and self._strategy_comparison.get(
                "scenario_revision") == rev and strategies is None:
            return self._strategy_comparison

        requested = strategies or [s.value for s in Strategy]
        scenario = self._comparison_scenario or self.scenario
        validator = self.validator
        watch = Stopwatch()

        results: List[Dict[str, Any]] = []
        for name in requested:
            strategy = Strategy(name)                    # rejects an unknown name
            cfg = OptimizerConfig(
                strategy=strategy,
                restarts=self.config.optimizer.restarts,
                time_budget_ms=self.config.optimizer.time_budget_ms,
                improve=self.config.optimizer.improve,
                seed=self.config.seed,
                validation=self.config.validation)
            plan = pack_optimized(
                scenario, config=cfg,
                plan_id=f"plan-{name}-{scenario.scenario_id}-rev{rev}")
            # Independent re-validation. pack_optimized validates internally, but
            # the comparison must be able to SHOW the verdict per strategy.
            report = validator.validate_plan(plan, scenario)
            results.append({
                "strategy": name,
                "plan_id": plan.plan_id,
                "containers": plan.containers_required,
                "utilization_pct": round(plan.utilization_pct, 2),
                "required_capacity_m3": round(plan.required_capacity_mm3 / 1e9, 4),
                "empty_capacity_m3": round(plan.unused_capacity_mm3 / 1e9, 4),
                "unplaced_items": len(plan.unplaced_item_ids),
                "optimization_time_ms": round(plan.computation_time_ms, 1),
                "score": round(plan.objective_score, 4),
                "valid": report.valid,
                "violations": [str(v) for v in report.violations[:10]],
            })

        comparison = {
            "schema_version": "1.0",
            "comparison_id": f"cmp-{uuid.uuid4().hex[:10]}",
            "revision": rev,
            "scenario_id": scenario.scenario_id,
            "scenario_revision": rev,
            "run_id": self.run_id,
            "generated_at": utc_now_iso(),
            "status": "completed",
            "selected_strategy": self.config.strategy.value,
            "results": results,
        }
        if strategies is None:
            self._strategy_comparison = comparison
        self._emit(Stage(self.stage), "strategy_comparison", Actor.OPTIMIZER,
                   duration_ms=watch.ms,
                   message=f"compared {len(results)} strategies over revision {rev}",
                   details={"comparison_id": comparison["comparison_id"],
                            "scenario_revision": rev,
                            "containers": {r["strategy"]: r["containers"]
                                           for r in results}})
        return comparison

    @property
    def strategy_comparison(self) -> Optional[Dict[str, Any]]:
        return self._strategy_comparison

    # ------------------------------------------------------------------ #
    # Stage 7 — Human in the Loop
    # ------------------------------------------------------------------ #

    def request_approval(self) -> None:
        """Enter the approval gate — which ALWAYS means a pending decision.

        THE INVARIANT: ``WAIT_FOR_OPERATOR_APPROVAL`` and an approval state of
        anything other than ``pending`` are contradictory. The dashboard cannot
        render that honestly — it has to say "decision required" while offering
        no decision to make — and the operator cannot act on it. So the gate is
        only ever entered through here, and here always sets pending, stamped
        with the revision and plan the decision is about.
        """
        if self.selected is None:
            raise WorkflowError("request_approval before a plan was selected")
        self._set_stage(Stage.WAIT_FOR_OPERATOR_APPROVAL)
        self.selected.approval_state = ApprovalState.PENDING
        self.approval_revision = self.scenario_revision
        self.approval_plan_id = self.selected.plan_id
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
        # An approval authorises ONE plan of ONE batch revision. A decision made
        # against a plan that has since been superseded — the operator clicked
        # while a re-plan or a scenario reset was in flight — is not a decision
        # about what is on screen now, and must not authorise it.
        if self.approval_plan_id and self.approval_plan_id != self.selected.plan_id:
            raise WorkflowError(
                f"approval targets plan {self.approval_plan_id} but the "
                f"selected plan is now {self.selected.plan_id}; a fresh "
                "decision is required")
        if self.approval_revision != self.scenario_revision:
            raise WorkflowError(
                f"approval targets scenario revision {self.approval_revision} "
                f"but the current revision is {self.scenario_revision}; a fresh "
                "decision is required")
        if self.selected.approval_state is not ApprovalState.PENDING:
            raise WorkflowError(
                f"approve called with the plan already "
                f"{self.selected.approval_state.value}")
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

    def revoke_approval(self, reason: str,
                        actor: Actor = Actor.ORCHESTRATOR) -> None:
        """Withdraw execution authorisation and return to a PENDING gate.

        The one way any path — critical anomaly, re-plan, scene reset, plan
        revision — asks for renewed approval. It exists so that "we need a new
        decision" cannot be expressed as "set the stage back and hope", which is
        how the contradictory state (`WAIT_FOR_OPERATOR_APPROVAL` +
        `approved`) was reachable: the stage moved and the approval did not.

        Unconditional by design. An earlier version revoked only when the plan
        was currently APPROVED, so a revocation arriving against a plan in any
        other state left the stamp pointing at a decision nobody had made for
        this revision.
        """
        if self.selected is None:
            return
        previous = self.selected.approval_state
        self.selected.approval_state = ApprovalState.PENDING
        self.approval_revision = self.scenario_revision
        self.approval_plan_id = self.selected.plan_id
        self._set_stage(Stage.WAIT_FOR_OPERATOR_APPROVAL)
        self._emit(Stage.WAIT_FOR_OPERATOR_APPROVAL, "revoke_approval",
                   actor, Result.PENDING,
                   message=f"execution authorisation revoked: {reason}",
                   details={"plan_id": self.selected.plan_id,
                            "previous_approval_state": previous.value,
                            "scenario_revision": self.scenario_revision,
                            "reason": reason})

    def resume_execution_stage(self) -> None:
        """Leave the approval gate once the plan is authorised.

        Sitting at ``WAIT_FOR_OPERATOR_APPROVAL`` with an approved plan is the
        contradictory state itself: the dashboard has to say "decision required"
        while every control is correctly disabled, because the decision has
        already been made. An approved plan belongs in an execution stage even
        if something downstream (an anomaly hold, a paused run, a physical scene
        that is still rebuilding) immediately stops it there — those states have
        their own honest wording.

        Unlike ``approve()`` this does NOT reset the cursor: it is used to
        resume a run whose earlier placements are physically done.
        """
        if self.selected is None:
            return
        if self.selected.approval_state is not ApprovalState.APPROVED:
            return
        if self.stage is not Stage.WAIT_FOR_OPERATOR_APPROVAL:
            return
        self._set_stage(Stage.NEXT_ITEM if self.cursor.index else Stage.PICK_ITEM)

    def approval_inconsistency(self) -> str:
        """"" when approval state is coherent, else why it is not.

        Checked rather than assumed. The states below are not merely odd to look
        at — each one means the dashboard would have to present a control whose
        effect it cannot describe.
        """
        state = (self.selected.approval_state if self.selected else None)
        if self.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL:
            if state is ApprovalState.APPROVED:
                return ("the workflow is at the approval gate but the plan is "
                        "already approved — there is no decision to make")
            if self.selected is not None:
                if self.approval_plan_id != self.selected.plan_id:
                    return (f"the pending decision targets plan "
                            f"{self.approval_plan_id} but the selected plan is "
                            f"{self.selected.plan_id}")
                if self.approval_revision != self.scenario_revision:
                    return (f"the pending decision targets scenario revision "
                            f"{self.approval_revision} but the current revision "
                            f"is {self.scenario_revision}")
        if (state is ApprovalState.APPROVED
                and self.approval_revision != self.scenario_revision):
            return (f"the plan is approved for scenario revision "
                    f"{self.approval_revision} but the current revision is "
                    f"{self.scenario_revision}")
        return ""

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
        # An anomaly hold blocks execution before the approval check even runs.
        # A safety-critical response must not depend on FIWARE or on the operator
        # having noticed — it is deterministic and local.
        if self.anomaly_hold:
            raise AnomalyHold(
                f"execution held by anomaly {self._latest_anomaly_class()} — "
                "operator acknowledgement required")

        self._assert_approved()
        assert self.selected is not None                # narrowed by _assert_approved

        # A dynamic event scheduled for this point fires before the pick.
        self._fire_due_events(placement_index=self.cursor.index)
        # If that event triggered a re-plan, STOP HERE.
        #
        # Guarding only on Stage.REPLAN was wrong and safety-relevant: replan()
        # finishes by calling request_approval(), which leaves the stage at
        # WAIT_FOR_OPERATOR_APPROVAL, not REPLAN. So the guard never fired and
        # this method went on to pick and place one more item using the
        # SUPERSEDED plan — after the operator's approval had been revoked.
        # Observed on the live stack as stage NEXT_ITEM immediately after a
        # re-plan. The authoritative test is the approval state itself.
        if self.stage in (Stage.REPLAN, Stage.WAIT_FOR_OPERATOR_APPROVAL):
            return True
        if (self.selected is None
                or self.selected.approval_state is not ApprovalState.APPROVED):
            return True

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

    # ------------------------------------------------------------------ #
    # Stage 8-13 — PHYSICAL execution (execution_backend = isaac)
    # ------------------------------------------------------------------ #
    #
    # These methods are the same workflow, driven from outside instead of from
    # a coin flip. They deliberately reuse the identical stages, the identical
    # counters and the identical completion path as ``step_execution``, so the
    # audit trail, the KPIs, the progress percentage and the dashboard timeline
    # do not change shape when the backend does. What changes is only WHERE the
    # pick/place outcome comes from.
    #
    # The safety invariant is unchanged and is asserted in exactly the same
    # place: nothing below may run on an unapproved plan.

    def next_physical_placement(self
                                ) -> Optional[Tuple[Placement, WasteItem, Container]]:
        """The next placement to hand to a physical backend, or None.

        Returns None when there is nothing to do *right now* — the plan is
        finished, or a dynamic event has just forced a re-plan and the operator
        has to authorise it again. The caller must not treat None as "run
        complete"; ``finished`` says that.

        Dynamic events fire here, before the item is dispatched, exactly as they
        do in ``step_execution``. That ordering is what makes a late arrival
        interrupt the run *between* items rather than while the gripper is shut.
        """
        self._assert_approved()
        assert self.selected is not None                # narrowed above

        self._fire_due_events(placement_index=self.cursor.index)
        # Same guard as step_execution, and for the same measured reason: a
        # re-plan leaves the stage at WAIT_FOR_OPERATOR_APPROVAL, not REPLAN, so
        # the approval state is the authoritative test.
        if self.stage in (Stage.REPLAN, Stage.WAIT_FOR_OPERATOR_APPROVAL):
            return None
        if (self.selected is None
                or self.selected.approval_state is not ApprovalState.APPROVED):
            return None

        pending = [p for p in self.selected.ordered_placements if not p.executed]
        if not pending:
            return None

        placement = pending[0]
        item = self.scenario.item(placement.item_id) if self.scenario else None
        container = self.selected.container(placement.container_id)
        if item is None or container is None:            # pragma: no cover
            raise WorkflowError(
                f"placement {placement.item_id} lost its item or container")
        return placement, item, container

    def begin_physical_item(self, placement: Placement) -> None:
        """Record that a physical pick of ``placement`` has been commanded."""
        self._assert_approved()
        self._set_stage(Stage.PICK_ITEM)
        self.robot_state = "picking"
        self.current_item_id = placement.item_id
        self.current_container_id = placement.container_id
        self.stats.pick_attempts += 1
        self._emit(Stage.PICK_ITEM, "isaac_pick_commanded", Actor.ISAAC_SIM,
                   Result.PENDING, item_id=placement.item_id,
                   container_id=placement.container_id,
                   source=Source.SIMULATED,
                   message=f"Isaac Sim commanded to pick {placement.item_id}",
                   details={"attempt": self.cursor.retries + 1,
                            "sequence_index": self.cursor.index,
                            "axis": placement.axis.value,
                            "backend": ExecutionBackend.ISAAC.value})

    def note_physical_progress(self, stage: Optional[Stage], action: str,
                               item_id: Optional[str], container_id: Optional[str],
                               message: str, robot_state: Optional[str] = None,
                               details: Optional[Dict[str, Any]] = None) -> None:
        """Record one intermediate physical state on the existing audit trail.

        ``stage`` is the WISEPACK stage the physical state maps to (see
        ``execution.stage_for_isaac_state``); None leaves the stage alone, which
        is what a simulator-lifecycle report such as READY must do.
        """
        if stage is not None:
            self._set_stage(stage)
        if robot_state is not None:
            self.robot_state = robot_state
        self._emit(stage or self.stage, action, Actor.ISAAC_SIM,
                   item_id=item_id, container_id=container_id,
                   source=Source.SIMULATED, message=message,
                   details=dict(details or {}))

    def complete_physical_item(self, placement: Placement,
                               details: Optional[Dict[str, Any]] = None) -> bool:
        """Commit a physically executed placement. Returns False when the run ends.

        ``details`` carries the MEASURED outcome — the settled pose and its
        distance from the planned one. It is recorded verbatim; nothing here
        overwrites a measured pose with the target it was aiming at.

        Note what is NOT asserted: that the item landed where the optimizer
        planned. A released cylinder settles under gravity and contact, and the
        first iteration reports the resulting error rather than pretending it is
        zero. The placement is marked executed because the item is physically in
        the container, which is what ``executed`` has always meant.
        """
        self._assert_approved()
        assert self.selected is not None
        item = self.scenario.item(placement.item_id) if self.scenario else None
        container = self.selected.container(placement.container_id)
        if item is None or container is None:            # pragma: no cover
            raise WorkflowError(
                f"cannot complete {placement.item_id}: its item or container is gone")

        detail = dict(details or {})
        self.stats.pick_successes += 1
        self.cursor.retries = 0
        item.status = ItemStatus.PLACED

        self._set_stage(Stage.UPDATE_CONTAINER_STATE)
        placement.executed = True
        container.status = ContainerStatus.FILLING
        remaining = [p for p in self.selected.placements_for(container.container_id)
                     if not p.executed]
        if not remaining:
            container.status = ContainerStatus.COMPLETE
        self.robot_state = "idle"

        self._emit(Stage.UPDATE_CONTAINER_STATE, "isaac_item_settled",
                   Actor.ISAAC_SIM, item_id=item.item_id,
                   container_id=container.container_id,
                   source=Source.SIMULATED,
                   message=f"{item.item_id} physically settled in "
                           f"{container.container_id}",
                   details={**detail,
                            "container_status": container.status.value,
                            "executed": sum(1 for p in self.selected.placements
                                            if p.executed),
                            "total": len(self.selected.placements),
                            "progress_pct": round(self.progress_pct, 1),
                            "backend": ExecutionBackend.ISAAC.value})

        self.stats.cycles_attempted += 1
        self.stats.cycles_completed += 1
        self.cursor.index += 1
        self._set_stage(Stage.NEXT_ITEM)
        self._fire_due_events(elapsed_s=self._elapsed_s())

        if not [p for p in self.selected.placements if not p.executed]:
            return self._complete()
        return True

    def fail_physical_item(self, placement: Placement, reason: str,
                           details: Optional[Dict[str, Any]] = None) -> bool:
        """A physical attempt failed. Retries, then abandons, exactly as simulated.

        The retry budget is ``RobotSimConfig.max_pick_retries`` — the same one the
        simulated backend uses — so the two backends give up after the same
        number of attempts and KPI2/KPI3 remain comparable between them.
        """
        self._set_stage(Stage.VERIFY_PICK)
        self.robot_state = "idle"
        self._emit(Stage.VERIFY_PICK, "isaac_item_failed", Actor.ISAAC_SIM,
                   Result.FAILED, item_id=placement.item_id,
                   container_id=placement.container_id,
                   source=Source.SIMULATED,
                   message=f"physical execution of {placement.item_id} failed: "
                           f"{reason}",
                   details={**dict(details or {}), "reason": reason,
                            "backend": ExecutionBackend.ISAAC.value})
        more = self._handle_pick_failure(placement, placement.item_id,
                                         placement.container_id)
        if self.selected and not [p for p in self.selected.placements
                                  if not p.executed]:
            return self._complete()
        return more

    def complete_physical_run(self) -> bool:
        """End the run once the backend reports RUN_COMPLETED."""
        if self.finished:
            return False
        return self._complete()

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
        # SUPERSEDE WHATEVER STATE IT WAS IN. Restricting this to PENDING left
        # an APPROVED plan approved across a re-plan, so the authorisation the
        # operator gave for the old plan survived into the new one.
        if self.selected and self.selected.approval_state in (
                ApprovalState.PENDING, ApprovalState.APPROVED):
            self.selected.approval_state = ApprovalState.SUPERSEDED
            self.approval_plan_id = None

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
        # A re-plan is a new batch: any prior strategy comparison is stale.
        # (Revision was already bumped by the batch-changing event; if the
        # re-plan had another cause, bump defensively so the comparison clears.)
        self._strategy_comparison = None

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

    #: Event types that change the batch the optimizer sees, and therefore
    #: invalidate any strategy comparison computed against the old batch.
    _BATCH_CHANGING = frozenset({
        DynamicEventType.ITEM_INJECT, DynamicEventType.ITEM_REMOVED,
        DynamicEventType.ITEM_RECLASSIFIED, DynamicEventType.CONTAINER_UNAVAILABLE,
        DynamicEventType.SEGREGATION_RULE_CHANGE,
    })

    def _bump_scenario_revision(self) -> None:
        """Advance the batch revision, discarding what no longer applies to it.

        An approval is a decision about ONE batch revision, so an outstanding
        authorisation cannot survive the batch changing underneath it. Revoking
        here rather than in each caller is what makes the rule structural: every
        path that changes the batch — a new scenario, an injected or removed
        item, a retired container, a segregation-rule change — goes through this
        one function, and none of them can forget.
        """
        self.scenario_revision += 1
        self._strategy_comparison = None
        if (self.selected is not None
                and self.selected.approval_state is ApprovalState.APPROVED):
            self.revoke_approval(
                f"the batch changed (scenario revision {self.scenario_revision})")

    # ------------------------------------------------------------------ #
    # Anomaly integration (EDF Topic #2 demonstration)
    # ------------------------------------------------------------------ #

    def _latest_anomaly_class(self) -> str:
        return (self.latest_anomaly or {}).get("anomaly_class", "unknown")

    def apply_anomaly(self, event: AnomalyEvent,
                      received_monotonic: Optional[float] = None) -> Dict[str, Any]:
        """React deterministically to an anomaly. SIMULATED integration only.

        info     -> record and continue.
        warning  -> PAUSE execution; operator acknowledgement required to resume.
        critical -> HOLD: revoke execution authorisation, preserve completed
                    placements, require an operator decision + new approval.

        The reaction is local and deterministic: it does not wait for FIWARE and
        does not depend on the operator having seen a dashboard. That is the
        point of routing safety-critical response through the ROS topic straight
        to the orchestrator, with FIWARE as an ADDITIONAL analytics path.
        """
        if self.scenario is not None:
            event.scenario_id = self.scenario.scenario_id
            event.scenario_revision = self.scenario_revision
        event.cycle_id = self.cycle_id
        self._anomaly_seq += 1
        event.sequence = self._anomaly_seq

        reaction = event.reaction
        # Latency from when the reaction node received the event to acting on it.
        latency_ms = 0.0
        if received_monotonic is not None:
            latency_ms = max(0.0, (time.monotonic() - received_monotonic) * 1000.0)

        record = event.to_dict()
        record["reaction_latency_ms"] = round(latency_ms, 2)
        record["acknowledged"] = event.is_ok        # OK needs no acknowledgement
        self.latest_anomaly = record
        self.anomaly_history.append(record)

        result = Result.OK
        if reaction is Reaction.CONTINUE:
            message = f"anomaly {event.anomaly_class.value} (info) — recorded"
        elif reaction is Reaction.PAUSE:
            self.anomaly_hold = True
            self.anomaly_ack_required = True
            result = Result.RETRY
            message = (f"anomaly {event.anomaly_class.value} (warning) — execution "
                       "PAUSED, operator acknowledgement required")
        else:  # HOLD
            self.anomaly_hold = True
            self.anomaly_ack_required = True
            result = Result.FAILED
            # Revoke execution authorisation. Completed placements stay put.
            #
            # Unconditional, and it used to be conditional on the plan being
            # APPROVED. That guard is what let the gate be entered without the
            # approval being withdrawn: any state other than APPROVED fell
            # through with the stamp untouched, and acknowledging the anomaly
            # then left "at the approval gate, already approved" — a decision
            # the operator is asked for and cannot give.
            self.revoke_approval(
                f"critical anomaly {event.anomaly_class.value}",
                actor=Actor.EVENT_INJECTOR)
            message = (f"anomaly {event.anomaly_class.value} (critical) — execution "
                       "HELD, authorisation revoked, operator decision required")

        self._emit(self.stage, f"anomaly:{event.anomaly_class.value}",
                   Actor.EVENT_INJECTOR, result, source=Source.SIMULATED,
                   message=message, details={**record, "reaction": reaction.value})
        return record

    def acknowledge_anomaly(self, operator: str = "operator") -> None:
        """Operator acknowledgement of the held anomaly.

        For a WARNING this clears the hold so execution can resume (still an
        approved plan). For a CRITICAL the hold clears but authorisation was
        REVOKED, so the operator must re-approve before anything executes —
        acknowledgement is not authorisation.
        """
        if not self.anomaly_ack_required:
            raise WorkflowError("no anomaly awaiting acknowledgement")
        self.anomaly_ack_required = False
        self.anomaly_hold = False
        self.stats.operator_interventions += 1
        if self.latest_anomaly:
            self.latest_anomaly["acknowledged"] = True
        cls = self._latest_anomaly_class()
        self._emit(self.stage, "anomaly_acknowledged", Actor.OPERATOR,
                   source=Source.OPERATOR,
                   message=f"anomaly {cls} acknowledged by {operator}",
                   details={"anomaly_class": cls})

    def anomaly_snapshot(self) -> Dict[str, Any]:
        """Everything the anomaly panel and analytics need."""
        history = self.anomaly_history
        by_class: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        ok = nok = 0
        latencies = []
        for a in history:
            by_class[a["anomaly_class"]] = by_class.get(a["anomaly_class"], 0) + 1
            by_severity[a["severity"]] = by_severity.get(a["severity"], 0) + 1
            if a["status"] == "OK":
                ok += 1
            else:
                nok += 1
            if a.get("reaction_latency_ms"):
                latencies.append(a["reaction_latency_ms"])
        pauses = sum(1 for a in history if a.get("reaction") == "pause")
        holds = sum(1 for a in history if a.get("reaction") == "hold")
        return {
            "label": "Simulated anomaly event — not a validated "
                     "anomaly detector",
            "hold": self.anomaly_hold,
            "ack_required": self.anomaly_ack_required,
            "latest": self.latest_anomaly,
            "history": history[-30:],
            "count": len(history),
            "by_class": by_class,
            "by_severity": by_severity,
            "ok_count": ok,
            "nok_count": nok,
            "pauses": pauses,
            "holds": holds,
            "anomaly_to_hold_latency_ms": (round(max(latencies), 2)
                                           if latencies else None),
        }

    def apply_dynamic_event(self, event: DynamicEvent) -> None:
        """Apply one scripted disturbance and re-plan if it invalidates the plan."""
        if self.scenario is None:
            raise WorkflowError("dynamic event before a scenario was loaded")
        event.fired = True
        if event.event_type in self._BATCH_CHANGING:
            self._bump_scenario_revision()
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
                            latency_p50_ms, self.run_id,
                            perception_source=self.config.perception_source.value,
                            perception_mean_confidence=(
                                self.observation_batch.mean_confidence
                                if self.observation_batch else None))

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
            "scenario_revision": self.scenario_revision,
            "strategy_comparison": self._strategy_comparison,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "optimized": self.optimized.to_dict() if self.optimized else None,
            "selected": self.selected.to_dict() if self.selected else None,
            "selected_plan_id": self.selected.plan_id if self.selected else None,
            "selection_reason": self.selection_reason,
            "approval_state": (self.selected.approval_state.value
                               if self.selected else "pending"),
            # WHICH decision, about WHICH batch. A consumer that merges an
            # approval state with a plan from a different revision is merging
            # two different runs; these are what let it notice.
            "approval_revision": self.approval_revision,
            "approval_plan_id": self.approval_plan_id,
            "approval_inconsistency": self.approval_inconsistency(),
            "detected": self.detected,
            "dynamic_events": [e.to_dict() for e in self.dynamic_events],
            "stats": self.stats.to_dict(),
            "action_count": self.log.count,
            "anomaly": self.anomaly_snapshot(),
            "whole_process": self.wp.snapshot(),
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
    "ApprovalRequired", "AnomalyHold", "WorkflowError", "RobotSimConfig",
    "WorkflowConfig", "WorkflowEngine", "run_headless",
]
