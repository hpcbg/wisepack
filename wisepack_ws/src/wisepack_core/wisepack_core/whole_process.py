"""Whole-process manager — the cut + inventory + logistics workflow layer.

The core packing workflow in workflow.py is left intact (brief §2); this is a
cohesive add-on that a :class:`WorkflowEngine` composes as ``engine.wp``. It owns:

  * the Human-in-the-Loop CUTTING workflow (brief §6): generate alternatives,
    validate the cut plan, wait for a SEPARATE cut approval, drive the simulated
    external cutting skill, register the ACTUAL derived segments, then re-plan and
    require packing approval AGAIN — cut approval never approves the packing plan;
  * the operational container INVENTORY and its audited operations (brief §13),
    including making the optimizer inventory-aware (reservations, delivery
    requests, ``WAIT_FOR_CONTAINER`` and shortage);
  * the simulated container LOGISTICS (brief §15) tasks and robot.

Every state change emits an ActionEvent through the engine's log using the
extended timeline actions of brief §18, so the audit trail is one stream.

Nothing here imports ROS or FastAPI. The orchestrator node and the dashboard both
call these methods; the numbers therefore match across sim, ROS and FIWARE modes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

from .cut_optimizer import CutPlannerConfig, WholeProcessComparison, plan_cut_aware
from .cut_validator import validate_result
from .cutting import (
    CutApprovalState, CutResult, CutState, derive_segments,
)
from .domain import Container, Scenario, Source, Strategy, WasteItem
from .events import Actor, Result, Stage, utc_now_iso
from .inventory import (
    ContainerInventory, ContainerLifecycleState, LOCATION_CELL,
)
from .logistics import LogisticsSimulator, TransportTaskType


class WholeProcessError(RuntimeError):
    """Raised when the cut/inventory workflow is driven out of order."""


class WholeProcess:
    """Cut-aware + inventory + logistics state for one workflow run."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.inventory = ContainerInventory(simulated=True)
        self.logistics = LogisticsSimulator(inventory=self.inventory)
        self.cut_planner_config = CutPlannerConfig()

        self.comparison: Optional[WholeProcessComparison] = None
        self.selected_cut_label: Optional[str] = None
        self.cut_approval_state = CutApprovalState.PENDING
        self.cut_results: List[CutResult] = []
        self.latest_cut_result: Optional[Dict[str, Any]] = None
        self.cut_events: List[Dict[str, Any]] = []
        self.prefer_no_cut = False
        # Cut-request idempotency + skill state. A CUTTING_REQUEST is emitted
        # EXACTLY ONCE per approved proposal revision. plan_revision bumps every
        # time a new comparison is generated (a new proposal revision);
        # approval_revision bumps on each operator cut approval. A request is keyed
        # on (approval_revision) and gated on the skill being PROPOSED, so periodic
        # state republication can never re-trigger it.
        self.plan_revision = 0
        self.approval_revision = 0
        self._approved_plan_revision = -1
        self._requested_approval_revision = -1
        self._request_seq = 0
        self.cut_skill_state = CutState.PROPOSED
        self.cut_request: Optional[Dict[str, Any]] = None

        self.plan_container_status = "ok"        # ok | waiting_for_container
        self.planning_result: Dict[str, Any] = {}
        self._inventory_initialised = False

    def reset(self) -> None:
        self.__init__(self.engine)

    # -- emit helper ------------------------------------------------------ #

    def _emit(self, stage: Stage, action: str, actor: Actor,
              result: Result = Result.OK, **kw: Any) -> None:
        self.engine._emit(stage, action, actor, result, **kw)

    # =================================================================== #
    # CUT-AWARE HUMAN-IN-THE-LOOP WORKFLOW (brief §6)
    # =================================================================== #

    def generate_cut_alternatives(self) -> WholeProcessComparison:
        """Run the whole-process cut comparison over the current scenario."""
        eng = self.engine
        if eng.scenario is None:
            raise WholeProcessError("generate_cut_alternatives before a scenario")
        eng._set_stage(Stage.GENERATE_CUT_ALTERNATIVES)
        cmp = plan_cut_aware(eng.scenario, optimizer=eng.config.optimizer,
                             config=self.cut_planner_config)
        self.comparison = cmp
        # A new comparison is a new proposal revision: reset the approval and the
        # cutting skill, so any earlier (now stale) approval cannot request a cut.
        self.cut_approval_state = CutApprovalState.PENDING
        self.plan_revision += 1
        self.cut_skill_state = CutState.PROPOSED
        self.cut_request = None

        for alt in cmp.alternatives:
            for prop in alt.proposals:
                self._emit(Stage.GENERATE_CUT_ALTERNATIVES,
                           "CUT_PROPOSAL_GENERATED", Actor.OPTIMIZER,
                           source=Source.SIMULATED,
                           message=f"proposal {prop.proposal_id}: "
                                   f"{prop.segment_lengths_mm} mm",
                           details=prop.to_dict())
            break  # one representative emission set; full data in the snapshot

        eng._set_stage(Stage.DIGITAL_TWIN_VALIDATE_CUT_PLAN)
        recommended = cmp.recommended
        validated = all(p.is_validated for p in recommended.proposals)
        self._emit(Stage.DIGITAL_TWIN_VALIDATE_CUT_PLAN, "CUT_PLAN_VALIDATED",
                   Actor.DIGITAL_TWIN,
                   Result.OK if (validated or not recommended.is_cut) else Result.FAILED,
                   message=cmp.reason,
                   details={"recommended": recommended.summary(),
                            "recommend_cut": cmp.recommend_cut})

        if cmp.recommend_cut and not self.prefer_no_cut:
            self.selected_cut_label = cmp.recommended_label
            eng._set_stage(Stage.WAIT_FOR_CUT_APPROVAL)
            self._emit(Stage.WAIT_FOR_CUT_APPROVAL, "WAIT_FOR_CUT_APPROVAL",
                       Actor.ORCHESTRATOR, Result.PENDING,
                       message="cut plan awaiting a SEPARATE operator cut "
                               "approval — packing approval is still required "
                               "afterwards",
                       details={"selected": self.selected_cut_label})
        else:
            self.selected_cut_label = "no_cut"
        return cmp

    def select_alternative(self, label: str) -> None:
        if self.comparison is None:
            raise WholeProcessError("select_alternative before generating")
        labels = {a.label for a in self.comparison.alternatives} | {"no_cut"}
        if label not in labels:
            raise WholeProcessError(f"unknown cut alternative {label!r}")
        self.selected_cut_label = label
        self.cut_approval_state = CutApprovalState.PENDING

    def limit_cuts(self, max_cuts: int) -> WholeProcessComparison:
        self.cut_planner_config = replace(self.cut_planner_config,
                                          max_cuts_per_plan=max(1, int(max_cuts)))
        return self.generate_cut_alternatives()

    def set_minimum_segment_mm(self, mm: int) -> WholeProcessComparison:
        """Raise the minimum segment on every cuttable pipe, then re-plan."""
        eng = self.engine
        if eng.scenario is not None:
            for it in eng.scenario.items:
                if it.is_cuttable:
                    it.minimum_segment_length_mm = max(1, int(mm))
        return self.generate_cut_alternatives()

    def set_prefer_no_cut(self, prefer: bool) -> WholeProcessComparison:
        self.prefer_no_cut = bool(prefer)
        return self.generate_cut_alternatives()

    # -- the separate cut approval --------------------------------------- #

    def _selected_alternative(self):
        if self.comparison is None or self.selected_cut_label in (None, "no_cut"):
            return None
        for a in self.comparison.alternatives:
            if a.label == self.selected_cut_label:
                return a
        return None

    def approve_cut(self, operator: str = "operator") -> None:
        alt = self._selected_alternative()
        if alt is None:
            raise WholeProcessError("no cut alternative selected to approve")
        if not alt.valid:
            raise WholeProcessError("cannot approve an invalid cut alternative")
        self.cut_approval_state = CutApprovalState.APPROVED
        # A distinct approval revision, bound to the exact proposal revision the
        # operator saw. A later comparison bumps plan_revision, which strands this
        # approval (see build_cut_request's plan_revision guard).
        self.approval_revision += 1
        self._approved_plan_revision = self.plan_revision
        self.cut_skill_state = CutState.PROPOSED
        self.engine.stats.operator_interventions += 1
        self.engine._set_stage(Stage.CUT_REQUESTED)
        self._emit(Stage.CUT_REQUESTED, "CUT_APPROVED", Actor.OPERATOR,
                   source=Source.OPERATOR,
                   message=f"cutting approved by {operator} — "
                           "packing approval is still required after re-plan",
                   details={"label": alt.label, "approval_revision":
                            self.approval_revision,
                            "proposals": [p.proposal_id for p in alt.proposals]})

    def reject_cut(self, reason: str = "operator preferred no cutting") -> None:
        self.cut_approval_state = CutApprovalState.REJECTED
        self.selected_cut_label = "no_cut"
        self.cut_skill_state = CutState.PROPOSED
        self.engine.stats.operator_interventions += 1
        self._emit(Stage.WAIT_FOR_CUT_APPROVAL, "CUT_REJECTED", Actor.OPERATOR,
                   Result.REJECTED, source=Source.OPERATOR, message=reason)

    # -- the request to the external cutting skill (idempotent) ----------- #

    def build_cut_request(self) -> Optional[Dict[str, Any]]:
        """Produce the CUTTING_REQUEST for an approved cut — EXACTLY ONCE.

        Returns a stable request-identity dict the first time it is called after
        an approval, then ``None`` on every subsequent call for that same
        approval revision. This is what makes periodic state republication safe:
        the orchestrator publishes only when this returns non-None. A request is
        emitted only when the proposal is independently validated, the operator
        has approved THAT proposal revision, no request has yet gone out for this
        approval revision, and the skill is still in PROPOSED (ready for
        APPROVED -> REQUESTED). On success the skill transitions to REQUESTED.
        """
        alt = self._selected_alternative()
        if alt is None:
            return None
        if self.cut_approval_state is not CutApprovalState.APPROVED:
            return None
        if not alt.valid or not all(p.is_validated for p in alt.proposals):
            return None
        # A stale approval (for an older proposal revision) must not request a
        # cut for a newer proposal.
        if self._approved_plan_revision != self.plan_revision:
            return None
        if self._requested_approval_revision == self.approval_revision:
            return None
        if self.cut_skill_state is not CutState.PROPOSED:
            return None

        eng = self.engine
        self._request_seq += 1
        prop = alt.proposals[0]
        request = {
            "request_id": f"cutreq-{eng.run_id}-{self._request_seq:04d}",
            "cut_plan_id": alt.label,
            "proposal_id": prop.proposal_id,
            "scenario_id": eng.scenario.scenario_id if eng.scenario else None,
            "scenario_revision": eng.scenario_revision,
            "plan_revision": self.plan_revision,
            "approval_revision": self.approval_revision,
            "requested_at": utc_now_iso(),
            "source": Source.SIMULATED.value,
            "selected_proposal": prop.to_dict(),
            "proposals": [p.to_dict() for p in alt.proposals],
            "label": "SIMULATED CUT REQUEST",
        }
        # Explicit APPROVED -> REQUESTED skill transition so republication cannot
        # retrigger, and record the approval revision this request covers.
        self.cut_skill_state = CutState.REQUESTED
        self._requested_approval_revision = self.approval_revision
        self.cut_request = request
        return request

    # -- simulated external cutting skill -------------------------------- #

    def simulate_cut(self, *, deviation_mm: int = 0) -> CutResult:
        """Drive the simulated cutting skill to COMPLETED, then re-plan.

        ``deviation_mm`` shifts material between the two outermost segments (still
        conserving length) to exercise the ``cut_result_deviation`` path where the
        ACTUAL result differs from the proposal.
        """
        alt = self._selected_alternative()
        if alt is None or self.cut_approval_state is not CutApprovalState.APPROVED:
            raise WholeProcessError("simulate_cut before an approved cut")
        eng = self.engine

        # Emit the request exactly once (idempotent): if the orchestrator already
        # published it on approval, this returns None and the skill is REQUESTED.
        self.build_cut_request()
        eng._set_stage(Stage.CUT_IN_PROGRESS)
        self.cut_skill_state = CutState.IN_PROGRESS
        self._emit(Stage.CUT_IN_PROGRESS, "CUT_REQUESTED", Actor.ORCHESTRATOR,
                   source=Source.SIMULATED,
                   message="SIMULATED external cutting skill: REQUESTED -> READY "
                           "-> IN_PROGRESS")

        results: List[CutResult] = []
        for prop in alt.proposals:
            actual = list(prop.segment_lengths_mm)
            if deviation_mm and len(actual) >= 2:
                actual[0] += deviation_mm
                actual[-1] -= deviation_mm
            result = CutResult(
                proposal_id=prop.proposal_id, source_item_id=prop.source_item_id,
                actual_segment_lengths_mm=actual,
                resulting_child_ids=prop.derived_item_ids_for(),
                actual_kerf_mm=prop.kerf_mm, completion_status=CutState.COMPLETED,
                quality_check_state="passed")
            results.append(result)

        self.cut_results = results
        eng._set_stage(Stage.CUT_COMPLETED)
        self.cut_skill_state = CutState.COMPLETED
        for result in results:
            parent = eng.scenario.item(result.source_item_id)
            verdict = validate_result(result, parent) if parent else {"valid": False}
            self.latest_cut_result = {**result.to_dict(), "validation": verdict}
            self._emit(Stage.CUT_COMPLETED, "CUT_COMPLETED", Actor.ROBOT_SIM,
                       Result.OK if verdict.get("valid") else Result.FAILED,
                       source=Source.SIMULATED,
                       message=f"cut {result.source_item_id} -> "
                               f"{result.actual_segment_lengths_mm} mm "
                               f"(deviation {deviation_mm} mm)",
                       details=self.latest_cut_result)
        self._register_derived_items(results)
        self._replan_after_cut()
        return results[0] if results else None

    def simulate_cut_failure(self, reason: str = "blade jam (simulated)") -> CutResult:
        """Fail the cut: no derived items, revert to the no-cut plan, re-approve."""
        alt = self._selected_alternative()
        if alt is None or self.cut_approval_state is not CutApprovalState.APPROVED:
            raise WholeProcessError("simulate_cut_failure before an approved cut")
        eng = self.engine
        prop = alt.proposals[0]
        result = CutResult(
            proposal_id=prop.proposal_id, source_item_id=prop.source_item_id,
            actual_segment_lengths_mm=[], resulting_child_ids=[],
            actual_kerf_mm=prop.kerf_mm, completion_status=CutState.FAILED,
            failure_reason=reason)
        self.latest_cut_result = result.to_dict()
        self.build_cut_request()                 # the request went out before failure
        self.cut_skill_state = CutState.FAILED
        self.cut_approval_state = CutApprovalState.PENDING
        self.selected_cut_label = "no_cut"
        eng._set_stage(Stage.CUT_COMPLETED)
        self._emit(Stage.CUT_COMPLETED, "CUT_FAILED", Actor.ROBOT_SIM,
                   Result.FAILED, source=Source.SIMULATED, message=reason,
                   details=self.latest_cut_result)
        # The pipe stays whole; the existing no-cut plan must be re-approved.
        if eng.selected is not None:
            eng.request_approval()
        return result

    def _register_derived_items(self, results: List[CutResult]) -> None:
        """Replace each cut parent with its ACTUAL derived segments (brief §6)."""
        eng = self.engine
        scenario = eng.scenario
        remove: set = set()
        add: List[WasteItem] = []
        for result in results:
            parent = scenario.item(result.source_item_id)
            if parent is None:
                continue
            children = derive_segments(
                parent, result.actual_segment_lengths_mm,
                kerf_mm=result.actual_kerf_mm,
                child_ids=result.resulting_child_ids or None)
            parent.derived_item_ids = [c.item_id for c in children]
            remove.add(parent.item_id)
            add.extend(children)
        scenario.items = [i for i in scenario.items if i.item_id not in remove] + add
        eng._bump_scenario_revision()
        eng._set_stage(Stage.REGISTER_DERIVED_ITEMS)
        self._emit(Stage.REGISTER_DERIVED_ITEMS, "DERIVED_ITEMS_REGISTERED",
                   Actor.ORCHESTRATOR, source=Source.SIMULATED,
                   message=f"registered {len(add)} derived segment(s) from "
                           f"{len(remove)} cut pipe(s)",
                   details={"removed": sorted(remove),
                            "added": [c.item_id for c in add],
                            "scenario_revision": eng.scenario_revision})

    def _replan_after_cut(self) -> None:
        """Re-run planning on the derived scenario; require packing approval."""
        eng = self.engine
        eng._set_stage(Stage.REPLAN_AFTER_CUT)
        eng.generate_plans()
        eng.digital_twin_validate()
        self._emit(Stage.REPLAN_AFTER_CUT, "REPLAN_AFTER_CUT", Actor.OPTIMIZER,
                   message="re-planned on derived items; packing approval "
                           "required again (cut approval did NOT approve packing)",
                   details=eng.selected.summary() if eng.selected else {})
        eng.request_approval()

    # =================================================================== #
    # CONTAINER INVENTORY (brief §9-§14)
    # =================================================================== #

    def initialise_simulated_inventory(self, count: int = 4, *,
                                       spec: Optional[Container] = None) -> None:
        """Deterministic simulated inventory, clearly labelled simulated."""
        eng = self.engine
        template = spec or (eng.scenario.container_template
                            if eng.scenario else None)
        if template is None:
            raise WholeProcessError("no container template for inventory")
        for i in range(count):
            cid = f"INV-{i:02d}"
            if cid in self.inventory:
                continue
            self.inventory.register(template.respec(cid))
            self.inventory.mark_available(cid)
            self._emit(Stage.CHECK_CONTAINER_AVAILABILITY, "CONTAINER_REGISTERED",
                       Actor.ORCHESTRATOR, source=Source.SIMULATED,
                       message=f"registered simulated container {cid}",
                       details=self.inventory.get(cid).semantic_state())
        self._inventory_initialised = True

    def check_container_availability(self) -> Dict[str, Any]:
        """Make the plan inventory-aware: reserve containers or wait (brief §14)."""
        eng = self.engine
        if eng.selected is None:
            raise WholeProcessError("check_container_availability before a plan")
        if not self._inventory_initialised:
            self.initialise_simulated_inventory()
        eng._set_stage(Stage.CHECK_CONTAINER_AVAILABILITY)
        needed = eng.selected.containers_required
        group = (eng.scenario.segregation_groups[0]
                 if eng.scenario and eng.scenario.segregation_groups else "A")

        selectable = self.inventory.selectable_for(group)
        reserved: List[str] = []
        for ic in selectable[:needed]:
            self.inventory.reserve(ic.container_id, holder=eng.selected.plan_id,
                                   segregation_group=group)
            reserved.append(ic.container_id)
            self._emit(Stage.RESERVE_CONTAINER, "CONTAINER_RESERVED",
                       Actor.ORCHESTRATOR, source=Source.SIMULATED,
                       message=f"reserved {ic.container_id} for "
                               f"{eng.selected.plan_id}",
                       details=self.inventory.get(ic.container_id).semantic_state())

        shortfall = needed - len(reserved)
        delivery_tasks: List[str] = []
        shortage = False
        if shortfall > 0:
            shortage = True
            self.plan_container_status = "waiting_for_container"
            ev = self.inventory.record_shortage(
                group, needed_mm3=eng.selected.required_capacity_mm3)
            eng._set_stage(Stage.WAIT_FOR_CONTAINER)
            self._emit(Stage.WAIT_FOR_CONTAINER, "INVENTORY_SHORTAGE_DETECTED",
                       Actor.ORCHESTRATOR, Result.PENDING, source=Source.SIMULATED,
                       message=f"{shortfall} more container(s) needed for group "
                               f"{group}; delivery/replenishment requested",
                       details=ev)
        else:
            self.plan_container_status = "ok"

        # Request delivery of the reserved containers to the cell.
        for cid in reserved:
            self.inventory.request_delivery(cid, workstation=LOCATION_CELL)
            task = self.logistics.request(
                cid, TransportTaskType.DELIVER_EMPTY_CONTAINER,
                scenario=eng.scenario.scenario_id if eng.scenario else None,
                plan=eng.selected.plan_id)
            delivery_tasks.append(task.task_id)
            self._emit(Stage.RESERVE_CONTAINER, "CONTAINER_DELIVERY_REQUESTED",
                       Actor.ORCHESTRATOR, source=Source.SIMULATED,
                       message=f"delivery task {task.task_id} for {cid}",
                       details=task.to_dict())

        self.planning_result = {
            "inventory_containers_selected": reserved,
            "reservations_created": len(reserved),
            "additional_containers_required": max(0, shortfall),
            "delivery_tasks_required": delivery_tasks,
            "collection_tasks_expected": needed,
            "inventory_shortage": shortage,
            "plan_status": ("WAITING_FOR_CONTAINER" if shortage else "ok"),
            "segregation_group": group,
            "source": Source.SIMULATED.value,
        }
        # When inventory is satisfied, return to the packing-approval gate so the
        # operator can approve; on a shortage the run stays in WAIT_FOR_CONTAINER
        # and packing approval is blocked until containers are replenished.
        if not shortage:
            eng.request_approval()
        return self.planning_result

    def run_logistics_to_quiescence(self, max_ticks: int = 500) -> int:
        return self.logistics.run_to_quiescence(max_ticks)

    # -- audited operator inventory operations (brief §13) ---------------- #

    _OP_ACTION = {
        "reserve": "CONTAINER_RESERVED",
        "release_reservation": "CONTAINER_RELEASED",
        "request_delivery": "CONTAINER_DELIVERY_REQUESTED",
        "mark_at_cell": "CONTAINER_ARRIVED",
        "mark_filling": "CONTAINER_FILLING_STARTED",
        "mark_full": "CONTAINER_FULL",
        "mark_sealed": "CONTAINER_FULL",
        "request_collection": "CONTAINER_COLLECTION_REQUESTED",
        "mark_dispatched": "CONTAINER_DISPATCHED",
        "mark_available": "CONTAINER_REGISTERED",
        "mark_unavailable": "CONTAINER_RELEASED",
    }

    def inventory_operation(self, op: str, container_id: str,
                            actor: str = "operator", reason: str = "",
                            **kw: Any) -> Dict[str, Any]:
        """Run an audited inventory op and emit its ActionEvent + logistics."""
        method = getattr(self.inventory, op, None)
        if method is None or op.startswith("_"):
            raise WholeProcessError(f"unknown inventory operation {op!r}")
        extra = {}
        if op == "reserve":
            extra["holder"] = kw.get("holder", "operator")
        ev = method(container_id, actor=actor,
                    reason=reason or op.replace("_", " "), **extra)
        action = self._OP_ACTION.get(op, "CONTAINER_REGISTERED")
        self._emit(Stage.CHECK_CONTAINER_AVAILABILITY, action, Actor.OPERATOR,
                   Result.OK if ev.applied else Result.REJECTED,
                   source=Source.OPERATOR if actor != "orchestrator"
                   else Source.SIMULATED,
                   message=f"{op} {container_id} by {actor}",
                   details=ev.to_dict())
        # Delivery requests spawn a transport task.
        if op == "request_delivery":
            task = self.logistics.request(
                container_id, TransportTaskType.DELIVER_EMPTY_CONTAINER,
                requested_by=actor)
            self._emit(Stage.CHECK_CONTAINER_AVAILABILITY,
                       "TRANSPORT_TASK_REQUESTED", Actor.ORCHESTRATOR,
                       source=Source.SIMULATED, details=task.to_dict())
        return ev.to_dict()

    def collect_full_containers(self) -> List[str]:
        """Request collection + dispatch for every FULL/SEALED container."""
        eng = self.engine
        collected: List[str] = []
        for ic in self.inventory.all():
            if ic.state in (ContainerLifecycleState.FULL,
                            ContainerLifecycleState.SEALED):
                if ic.state is ContainerLifecycleState.FULL:
                    self.inventory.mark_sealed(ic.container_id)
                self.inventory.request_collection(ic.container_id)
                self.inventory.mark_in_transit_from_cell(ic.container_id)
                eng._set_stage(Stage.COLLECT_FULL_CONTAINER)
                task = self.logistics.request(
                    ic.container_id, TransportTaskType.REMOVE_FULL_CONTAINER)
                self._emit(Stage.COLLECT_FULL_CONTAINER,
                           "CONTAINER_COLLECTION_REQUESTED", Actor.ORCHESTRATOR,
                           source=Source.SIMULATED,
                           message=f"collection task {task.task_id} for "
                                   f"{ic.container_id}", details=task.to_dict())
                collected.append(ic.container_id)
        self.logistics.run_to_quiescence()
        return collected

    # =================================================================== #
    # Snapshot / analytics
    # =================================================================== #

    def cut_snapshot(self) -> Optional[Dict[str, Any]]:
        if self.comparison is None:
            return None
        return {
            **self.comparison.to_dict(),
            "selected_label": self.selected_cut_label,
            "cut_approval_state": self.cut_approval_state.value,
            "cut_skill_state": self.cut_skill_state.value,
            "plan_revision": self.plan_revision,
            "approval_revision": self.approval_revision,
            "cut_request": self.cut_request,
            "latest_cut_result": self.latest_cut_result,
            "cut_results": [r.to_dict() for r in self.cut_results],
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "cut": self.cut_snapshot(),
            "inventory": self.inventory.to_dict(),
            "logistics": self.logistics.to_dict(),
            "planning_result": self.planning_result,
            "plan_container_status": self.plan_container_status,
            "analytics": self.analytics(),
        }

    def analytics(self) -> Dict[str, Any]:
        """Whole-process analytics with provenance (brief §17)."""
        cut: Dict[str, Any] = {"provenance": "simulated_cutting_measured_packing"}
        if self.comparison is not None:
            c = self.comparison
            cut.update({
                "pipes_evaluated": len(c.pipes_considered),
                "cut_candidates": c.candidates_evaluated,
                "recommend_cut": c.recommend_cut,
                "containers_no_cut": c.no_cut.containers,
                "containers_recommended": c.recommended.containers,
                "containers_avoided": c.no_cut.containers - c.recommended.containers,
                "cuts_recommended": c.recommended.n_cuts,
                "cutting_time_s": round(c.recommended.cutting_time_s, 1),
                "extra_handling_time_s": round(c.recommended.handling_time_s, 1),
                "kerf_loss_cm3": round(c.recommended.kerf_loss_cm3, 2),
            })
        cut["cuts_executed"] = len([r for r in self.cut_results
                                    if r.succeeded])
        cut["resulting_segments"] = sum(len(r.actual_segment_lengths_mm)
                                        for r in self.cut_results if r.succeeded)
        return {
            "cutting": cut,
            "inventory": {**self.inventory.summary(),
                          "provenance": "software_state"},
            "logistics": self.logistics.analytics(),
        }


__all__ = ["WholeProcess", "WholeProcessError"]
