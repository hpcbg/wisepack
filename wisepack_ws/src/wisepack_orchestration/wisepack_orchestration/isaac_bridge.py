"""The Isaac Sim execution backend, as seen from the WISEPACK orchestrator.

WHAT THIS IS NOT
----------------
It is not a second workflow. There is one WorkflowEngine, one behaviour tree,
one approval gate and one audit trail, and this module plugs into all of them.
It replaces exactly one thing: the body of the execution step. Where the
simulated backend resolves a placement with a seeded coin flip inside
``WorkflowEngine.step_execution``, this bridge hands the placement to a real
PhysX scene and waits to be told what physically happened.

Everything else is untouched — scenario generation, both packing algorithms, the
Digital Twin validation, re-planning, the operator command vocabulary, every
canonical topic and every KPI definition. The optimizer does not know a robot
exists, and it still does not.

SINGLE AUTHORITY, ENFORCED
--------------------------
When ``execution_backend=isaac`` the orchestrator never calls
``step_execution()``. That is the whole guarantee: there is no moment at which a
simulated outcome and a physical outcome both claim the same placement, because
only one of the two code paths is ever reached. The orchestrator remains the sole
publisher of every execution topic; Isaac publishes on the feedback topic and
nothing else.

WHY THE ORCHESTRATOR STILL OWNS THE STATE
-----------------------------------------
Isaac could publish ``/wisepack/execution/state`` directly and save a hop. It
must not. Two writers on a latched state topic means a reader sees whichever
wrote last, which is the exact failure the single-writer rule in
``wisepack_bringup.topics`` exists to prevent — and it was observed on this
project before that rule was adopted. So Isaac reports PHYSICS, the orchestrator
decides WORKFLOW STATE, and the mapping between them is declared once in
``wisepack_core.execution``.

THREE FAILURE MODES THIS HANDLES EXPLICITLY
-------------------------------------------
  * Isaac never starts. The command topic is latched, so a simulator that comes
    up late still receives RUN_BEGIN — but one that never comes up at all must
    not leave the workflow hanging silently. ``ready_timeout_s`` bounds it and
    the run enters DEGRADED with a diagnostic naming the topic to check.
  * A stale simulator from a previous invocation is still running. It would
    happily report ITEM_COMPLETED into a run it knows nothing about, and the
    engine would mark a placement executed that no robot touched. Every inbound
    message is gated on ``run_id`` by ``RunGate``.
  * A latched command is redelivered on re-subscribe. Acting twice on it would
    pick an item that is already in the container, so the gate also de-duplicates
    on ``sequence_index``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from std_msgs.msg import String

from wisepack_bringup import topics as T
from wisepack_bringup.qos import qos_for
from wisepack_core.domain import Container, Placement, WasteItem
from wisepack_core.events import Actor, Result, Stage
from wisepack_core.execution import (
    ExecutionBackend, robot_state_for_isaac_state, stage_for_isaac_state,
)
from wisepack_core.isaac_contract import (
    ContractError, IsaacCommand, IsaacCommandType, IsaacFeedback, IsaacState,
    RunGate,
)
from wisepack_core.isaac_transform import (
    DEFAULT_LAYOUT, SceneLayout, dimensions_for, placement_pose,
    table_pose_for_index,
)

LOG = "[isaac-bridge]"

#: Isaac physical state -> the audit-trail action name it is recorded under.
#: These names are what a FIWARE consumer sees in the NGSI-LD action stream, so
#: they are part of the external contract and are declared once, here.
#:
#: The two terminal item states are absent on purpose: WorkflowEngine emits those
#: itself (``isaac_item_settled`` / ``isaac_item_failed``) as part of committing
#: the placement, so recording them here as well would double-count the event.
_PROGRESS_ACTION = {
    IsaacState.READY: "isaac_simulator_ready",
    IsaacState.MOVING_TO_PICK: "isaac_moving_to_pick",
    IsaacState.GRASPING: "isaac_item_grasped",
    IsaacState.LIFTING: "isaac_item_lifted",
    IsaacState.MOVING_TO_CONTAINER: "isaac_moving_to_container",
    IsaacState.RELEASING: "isaac_item_released",
    IsaacState.SETTLING: "isaac_settling",
    IsaacState.RUN_COMPLETED: "isaac_run_completed",
    IsaacState.RUN_FAILED: "isaac_run_failed",
}


class IsaacExecutionBridge:
    """Drives one WISEPACK run through Isaac Sim, one placement at a time.

    Lives inside the orchestrator process and is ticked by the ExecuteLoop
    behaviour at the tree's tick rate. All callbacks arrive on the same rclpy
    executor thread as the tick, so the state below needs no locking — stated
    explicitly because it would stop being true under a MultiThreadedExecutor.
    """

    def __init__(self, node, *, layout: Optional[SceneLayout] = None,
                 ready_timeout_s: float = 240.0,
                 item_timeout_s: float = 180.0,
                 command_resend_s: float = 5.0) -> None:
        self.node = node
        self.layout = layout or DEFAULT_LAYOUT
        self.ready_timeout_s = float(ready_timeout_s)
        self.item_timeout_s = float(item_timeout_s)
        self.command_resend_s = float(command_resend_s)

        self.publisher = node.create_publisher(
            String, T.ISAAC_COMMAND, qos_for(T.ISAAC_COMMAND))
        node.create_subscription(String, T.ISAAC_FEEDBACK, self._on_feedback,
                                 qos_for(T.ISAAC_FEEDBACK))

        self.gate = RunGate()
        self.simulator_ready = False
        self.run_open = False
        self.run_finished = False
        #: (placement, sequence_index) currently with the simulator, if any.
        self._in_flight: Optional[Tuple[Placement, int]] = None
        self._in_flight_command: Optional[IsaacCommand] = None
        self._dispatched_at: float = 0.0
        self._last_resend_at: float = 0.0
        self._run_opened_at: float = 0.0
        self._last_state: Optional[IsaacState] = None
        #: Latest per-item physical outcome, newest last. Bounded: this is a
        #: diagnostic surface, not a second audit trail.
        self.results: list = []
        self._degraded = False
        #: The backend-neutral visualization descriptor the simulator reported
        #: with READY. Passed through verbatim — the orchestrator does not know
        #: what a WebRTC port is and must not learn.
        self.visualization: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Introspection (the dashboard diagnostics read this)
    # ------------------------------------------------------------------ #

    def status(self) -> Dict[str, Any]:
        return {
            "backend": ExecutionBackend.ISAAC.value,
            "run_id": self.gate.run_id,
            "simulator_ready": self.simulator_ready,
            "run_open": self.run_open,
            "run_finished": self.run_finished,
            "last_state": self._last_state.value if self._last_state else None,
            "in_flight_item": (self._in_flight[0].item_id
                               if self._in_flight else None),
            "items_completed": self.gate.completed_count,
            "results": list(self.results[-12:]),
            "visualization": self.visualization,
        }

    # ------------------------------------------------------------------ #
    # Outbound
    # ------------------------------------------------------------------ #

    def _publish(self, command: IsaacCommand) -> None:
        self.publisher.publish(String(data=command.to_json()))
        self.node.get_logger().info(
            f"{LOG} -> {command.command.value}"
            + (f" {command.item_id} #{command.sequence_index}"
               if command.item_id else ""))

    def open_run(self, engine) -> None:
        """Publish RUN_BEGIN for ``engine``'s run and start waiting for READY.

        Idempotent per run_id: calling it again for the same run does nothing,
        so a tick that arrives before the simulator answers cannot restart the
        handshake and reset the timeout clock forever.
        """
        if self.run_open and self.gate.run_id == engine.run_id:
            return
        self.gate.adopt(engine.run_id)
        self.simulator_ready = False
        self.run_open = True
        self.run_finished = False
        self._degraded = False
        self._in_flight = None
        self._in_flight_command = None
        self._run_opened_at = time.monotonic()
        self.results = []

        scenario = engine.scenario
        self._publish(IsaacCommand(
            command=IsaacCommandType.RUN_BEGIN,
            run_id=engine.run_id,
            plan_id=engine.selected.plan_id if engine.selected else "",
            preset=scenario.preset if scenario else engine.config.preset,
            seed=int(scenario.seed if scenario else engine.config.seed),
            total_items=len(scenario.items) if scenario else 0,
        ))
        engine.note_physical_progress(
            None, "isaac_run_begin", None, None,
            "Isaac Sim asked to build the scene and stand by",
            details={"preset": scenario.preset if scenario else "",
                     "seed": int(scenario.seed if scenario else 0),
                     "run_id": engine.run_id,
                     "backend": ExecutionBackend.ISAAC.value})

    def abort_run(self, engine, reason: str) -> None:
        """Tell Isaac to stop. Used when approval is withdrawn or a re-plan lands."""
        if not self.run_open:
            return
        self._publish(IsaacCommand(
            command=IsaacCommandType.RUN_ABORT, run_id=self.gate.run_id))
        if self._in_flight is not None:
            placement, _ = self._in_flight
            self.node.get_logger().warn(
                f"{LOG} aborting in-flight {placement.item_id}: {reason}")
        self._in_flight = None
        self._in_flight_command = None
        engine.note_physical_progress(
            None, "isaac_run_abort", None, None,
            f"Isaac Sim execution aborted: {reason}",
            robot_state="idle", details={"reason": reason})

    def close_run(self, engine) -> None:
        if not self.run_open or self.run_finished:
            return
        self._publish(IsaacCommand(
            command=IsaacCommandType.RUN_END, run_id=self.gate.run_id))
        self.run_finished = True

    # ------------------------------------------------------------------ #
    # The tick — called by the ExecuteLoop behaviour
    # ------------------------------------------------------------------ #

    def tick(self, engine) -> bool:
        """Advance physical execution by at most one dispatch.

        Returns True while the run should continue, False once it is finished.
        Mirrors the return contract of ``WorkflowEngine.step_execution`` so the
        ExecuteLoop behaviour reads identically for both backends.
        """
        if engine.finished:
            return False

        # A `reset` command replaces the engine, and with it the run_id. The new
        # run needs its own scene, so re-open rather than continuing to talk
        # about the previous one.
        if self.gate.run_id != engine.run_id:
            self.open_run(engine)
            return True

        if not self.run_open:
            self.open_run(engine)
            return True

        if not self.simulator_ready:
            return self._await_simulator(engine)

        if self._in_flight is not None:
            self._check_item_timeout(engine)
            return True

        return self._dispatch_next(engine)

    def _await_simulator(self, engine) -> bool:
        waited = time.monotonic() - self._run_opened_at
        if waited < self.ready_timeout_s:
            return True
        if self._degraded:
            return True
        self._degraded = True
        engine.enter_degraded(
            f"Isaac Sim did not report READY within {self.ready_timeout_s:.0f}s. "
            f"Check that the simulator is running on the host with the same "
            f"ROS_DOMAIN_ID, and that {T.ISAAC_FEEDBACK} has a publisher "
            f"(`ros2 topic info {T.ISAAC_FEEDBACK}`).")
        return False

    def _check_item_timeout(self, engine) -> None:
        assert self._in_flight is not None
        placement, index = self._in_flight
        elapsed = time.monotonic() - self._dispatched_at
        if elapsed > self.item_timeout_s:
            self.node.get_logger().error(
                f"{LOG} {placement.item_id} (#{index}) produced no terminal "
                f"feedback in {elapsed:.0f}s — failing it")
            self._in_flight = None
            self._in_flight_command = None
            engine.fail_physical_item(
                placement,
                f"no terminal feedback from Isaac Sim within "
                f"{self.item_timeout_s:.0f}s",
                details={"sequence_index": index, "elapsed_s": round(elapsed, 1)})
            self.node.publish_execution()
            return
        # Re-publish periodically until the simulator acknowledges with any
        # progress state. The command topic is latched, so this only matters when
        # the simulator restarted mid-run and missed the original.
        if (self._last_state is None
                and self._in_flight_command is not None
                and time.monotonic() - self._last_resend_at > self.command_resend_s):
            self._last_resend_at = time.monotonic()
            self._publish(self._in_flight_command)

    def _dispatch_next(self, engine) -> bool:
        nxt = engine.next_physical_placement()
        if nxt is None:
            # Either a re-plan is pending approval (the workflow handles that) or
            # every placement is executed. Only the latter ends the run.
            if engine.selected and not [p for p in engine.selected.placements
                                        if not p.executed]:
                self.close_run(engine)
                return engine.complete_physical_run()
            return True

        placement, item, container = nxt
        index = engine.cursor.index
        # The retry counter IS the attempt number. Sending it is what lets the
        # simulator tell a genuine retry from a latched replay — see
        # isaac_contract.RunGate.
        command = self._build_command(engine, placement, item, container, index,
                                      attempt=engine.cursor.retries)
        self._in_flight = (placement, index)
        self._in_flight_command = command
        self._dispatched_at = time.monotonic()
        self._last_resend_at = self._dispatched_at
        self._last_state = None

        engine.begin_physical_item(placement)
        self._publish(command)
        self.node.publish_execution()
        return True

    def _build_command(self, engine, placement: Placement, item: WasteItem,
                       container: Container, index: int,
                       attempt: int = 0) -> IsaacCommand:
        """Turn one accepted placement into a physical instruction.

        Every coordinate here comes from ``wisepack_core.isaac_transform``. There
        is no arithmetic in this module on purpose — see that module's docstring
        for why the conversion is concentrated in one testable place.
        """
        scenario = engine.scenario
        try:
            spawn_index = [i.item_id for i in scenario.items].index(item.item_id)
        except (AttributeError, ValueError):
            # An item injected by a dynamic event after the scene was built has
            # no spawned body. Send it anyway with a best-effort slot: Isaac
            # answers ITEM_FAILED "unknown item", which is the honest outcome and
            # is visible in the trail, rather than a silent skip here.
            spawn_index = index
            self.node.get_logger().warn(
                f"{LOG} {item.item_id} is not in the scene Isaac built "
                "(injected after RUN_BEGIN); expecting ITEM_FAILED")

        return IsaacCommand(
            command=IsaacCommandType.EXECUTE_ITEM,
            run_id=engine.run_id,
            sequence_index=index,
            attempt=attempt,
            item_id=item.item_id,
            dimensions=dimensions_for(item),
            source_pose=table_pose_for_index(spawn_index, item, self.layout),
            target_pose=placement_pose(placement),
            container_id=container.container_id,
            container_inner_mm=container.inner_size.to_dict(),
            plan_id=engine.selected.plan_id if engine.selected else "",
            preset=scenario.preset if scenario else "",
            seed=int(scenario.seed if scenario else 0),
            total_items=len(scenario.items) if scenario else 0,
        )

    # ------------------------------------------------------------------ #
    # Inbound
    # ------------------------------------------------------------------ #

    def _on_feedback(self, msg: String) -> None:
        try:
            feedback = IsaacFeedback.from_json(msg.data or "")
        except ContractError as exc:
            # Named loudly rather than swallowed: a malformed physical report is
            # either a version skew or a different simulator on the domain, and
            # both need a human.
            self.node.get_logger().error(
                f"{LOG} rejected malformed feedback: {exc}")
            return

        engine = self.node.engine
        self.node.get_logger().info(
            f"{LOG} <- {feedback.state.value} "
            f"{feedback.item_id or '-'} run={feedback.run_id}")
        reason = self.gate.reject_reason(feedback.run_id, -1)
        if reason:
            self.node.get_logger().warn(
                f"{LOG} ignoring {feedback.state.value} — {reason}")
            return

        try:
            self._apply(engine, feedback)
        except Exception as exc:                        # noqa: BLE001
            # The item and state are in the message; include both, because
            # "bridge error" alone is unactionable at 40 items in.
            self.node.get_logger().error(
                f"{LOG} failed to apply {feedback.state.value} for "
                f"{feedback.item_id or '-'}: {exc!r}")

    def _apply(self, engine, feedback: IsaacFeedback) -> None:
        state = feedback.state
        self._last_state = state

        if state is IsaacState.READY:
            # Refreshed on every READY, not only the first: a simulator that
            # restarts with streaming newly enabled must be able to correct a
            # previously-published "unavailable".
            self.visualization = feedback.detail.get("visualization")
            if not self.simulator_ready:
                self.simulator_ready = True
                self.node.get_logger().info(
                    f"{LOG} Isaac Sim reported READY for run {feedback.run_id}")
                engine.note_physical_progress(
                    None, _PROGRESS_ACTION[state], None, None,
                    "Isaac Sim scene built; physical execution may begin",
                    details={**feedback.detail,
                             "backend": ExecutionBackend.ISAAC.value})
            return

        # NOTE the receiver: these are properties of the FEEDBACK MESSAGE, not of
        # the state enum. Calling them on `state` raises AttributeError, and
        # because this branch sits ahead of the progress handler it made EVERY
        # non-READY report fail to apply — the orchestrator saw the simulator
        # come up and then heard nothing more, and eventually timed the item out
        # while the arm had in fact completed it. Measured end to end.
        if feedback.is_run_terminal:
            self._on_run_terminal(engine, feedback)
            return

        if feedback.is_item_terminal:
            self._on_item_terminal(engine, feedback)
            return

        # An intermediate physical state. It moves the WISEPACK stage through the
        # existing vocabulary rather than displaying a parallel one.
        engine.note_physical_progress(
            stage_for_isaac_state(state),
            _PROGRESS_ACTION.get(state, f"isaac_{state.value.lower()}"),
            feedback.item_id, feedback.container_id,
            feedback.message or f"Isaac Sim: {state.value}",
            robot_state=robot_state_for_isaac_state(state),
            details={**feedback.detail, "isaac_state": state.value})
        self.node.publish_execution()

    def _on_item_terminal(self, engine, feedback: IsaacFeedback) -> None:
        if self._in_flight is None:
            self.node.get_logger().warn(
                f"{LOG} {feedback.state.value} for {feedback.item_id} arrived "
                "with nothing in flight — ignoring")
            return
        placement, index = self._in_flight
        if feedback.item_id != placement.item_id:
            self.node.get_logger().warn(
                f"{LOG} {feedback.state.value} names {feedback.item_id} but "
                f"{placement.item_id} is in flight — ignoring")
            return

        attempt = (self._in_flight_command.attempt
                   if self._in_flight_command is not None else 0)
        self._in_flight = None
        self._in_flight_command = None
        self.gate.mark_done(index, attempt)

        outcome = {
            "item_id": feedback.item_id,
            "sequence_index": index,
            "state": feedback.state.value,
            "target_pose": (feedback.target_pose.to_dict()
                            if feedback.target_pose else None),
            "actual_pose": (feedback.actual_pose.to_dict()
                            if feedback.actual_pose else None),
            "position_error_mm": feedback.position_error_mm,
            "message": feedback.message,
            **feedback.detail,
        }
        self.results.append(outcome)
        if len(self.results) > 64:
            del self.results[:len(self.results) - 64]

        if feedback.state is IsaacState.ITEM_COMPLETED:
            engine.complete_physical_item(placement, details=outcome)
        else:
            engine.fail_physical_item(
                placement, feedback.message or "physical execution failed",
                details=outcome)
        self.node.publish_execution()

    def _on_run_terminal(self, engine, feedback: IsaacFeedback) -> None:
        if feedback.state is IsaacState.RUN_COMPLETED:
            self.run_finished = True
            engine.note_physical_progress(
                None, _PROGRESS_ACTION[feedback.state], None, None,
                feedback.message or "Isaac Sim finished the physical run",
                robot_state="idle",
                details={**feedback.detail,
                         "items_completed": self.gate.completed_count})
            return

        # RUN_FAILED. WISEPACK never simulates unsafe autonomous continuation, so
        # the honest response is to hold rather than to fall back to the
        # simulated backend and finish the run "successfully".
        self.run_finished = True
        self._in_flight = None
        engine.log.emit(
            Stage.DEGRADED, _PROGRESS_ACTION[feedback.state], Actor.ISAAC_SIM,
            Result.FAILED,
            message=feedback.message or "Isaac Sim reported RUN_FAILED",
            details=dict(feedback.detail))
        if not engine.finished:
            engine.enter_degraded(
                f"Isaac Sim reported RUN_FAILED: "
                f"{feedback.message or 'no detail supplied'}")
        self.node.publish_execution()


__all__ = ["IsaacExecutionBridge"]
