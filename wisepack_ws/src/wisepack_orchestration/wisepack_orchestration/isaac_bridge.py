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
    RunGate, SceneAcknowledgement,
)
from wisepack_core.isaac_transform import (
    DEFAULT_LAYOUT, SceneLayout, dimensions_for, placement_pose,
    layout_for_robot, scene_fingerprint, table_pose_for_index,
)

LOG = "[isaac-bridge]"

#: Feedback about the SCENE rather than about an item or a run's execution.
#: Correlated by its own fields, not by the run gate — see `_on_feedback`.
_SCENE_LIFECYCLE = (IsaacState.RESET_REQUESTED, IsaacState.RESETTING,
                    IsaacState.SCENE_READY, IsaacState.RESET_FAILED)

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
                 command_resend_s: float = 5.0,
                 robot=None) -> None:
        self.node = node
        # THE ROBOT DECIDES THE WORKCELL. The layout is derived from the
        # selected robot's profile — a shorter arm needs the bin nearer its base
        # — and the SIMULATOR derives the same layout from the same profile. If
        # this end used the shared default while the simulator used the robot's,
        # every plan pose would be converted against a bin that is not where the
        # objects are. `scene_fingerprint` hashes the robot id together with the
        # layout precisely so that disagreement is caught before a pick, not
        # after one.
        self.robot = robot
        self.layout = layout or layout_for_robot(robot)
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
        #: SCENE GATE. The scenario revision the PHYSICAL scene is known to
        #: correspond to, and the one currently required. Until they match,
        #: nothing may be approved and nothing may be picked — the objects from
        #: the previous run are still lying in the container while the new plan
        #: assumes they are back at their source poses.
        self.scene_revision: int = -1
        self.required_revision: int = 0
        self.reset_in_progress: bool = False
        self.reset_failed_reason: str = ""
        self._reset_requested_at: float = 0.0
        self.reset_timeout_s: float = 180.0
        #: WHAT was asked for and WHAT came back, kept apart so a mismatch can be
        #: described instead of merely reported as "not ready".
        self.requested_fingerprint: str = ""
        self.requested_object_count: int = 0
        self.acknowledged: Optional[SceneAcknowledgement] = None
        self.scene_mismatch: str = ""
        #: True once the scene handshake has been sent for the current run. The
        #: handshake used to happen only on an explicit reset, so a first run sat
        #: behind a gate nothing was ever going to open.
        self.scene_requested_for_run: str = ""
        #: The backend-neutral visualization descriptor the simulator reported
        #: with READY. Passed through verbatim — the orchestrator does not know
        #: what a WebRTC port is and must not learn.
        self.visualization: Optional[Dict[str, Any]] = None
        self.simulator_version: Optional[str] = None
        #: What the simulator reported about its robot, verbatim. The
        #: orchestrator does not interpret it — it does not know what an
        #: articulation is and must not learn — it forwards it to Diagnostics.
        self.robot_status: Dict[str, Any] = {}
        #: Non-empty once the simulator has reported ROBOT_MODEL_INVALID. While
        #: it holds a reason nothing may be approved and nothing may be picked.
        self.robot_model_error: str = ""

    # ------------------------------------------------------------------ #
    # Introspection (the dashboard diagnostics read this)
    # ------------------------------------------------------------------ #

    def status(self) -> Dict[str, Any]:
        return {
            "backend": ExecutionBackend.ISAAC.value,
            "run_id": self.gate.run_id,
            # WHICH ARM this run selected, and what the simulator says it is
            # actually running. Kept apart: they are the same until they are
            # not, and the case where they differ is the one worth seeing.
            "robot_id": self.robot_id,
            "robot_display_name": (self.robot.display_name if self.robot
                                   else ""),
            "robot_profile_revision": (self.robot.revision if self.robot else ""),
            "robot_status": dict(self.robot_status),
            "acknowledged_robot": (self.acknowledged.robot_id
                                   if self.acknowledged else ""),
            "robot_model_error": self.robot_model_error,
            "simulator_ready": self.simulator_ready,
            "run_open": self.run_open,
            "run_finished": self.run_finished,
            "last_state": self._last_state.value if self._last_state else None,
            "in_flight_item": (self._in_flight[0].item_id
                               if self._in_flight else None),
            "items_completed": self.gate.completed_count,
            "results": list(self.results[-12:]),
            "visualization": self.visualization,
            "simulator_version": self.simulator_version,
            "scene_revision": self.scene_revision,
            "required_revision": self.required_revision,
            "scene_ready": self.scene_ready,
            "reset_in_progress": self.reset_in_progress,
            "reset_failed_reason": self.reset_failed_reason,
            # TWO READINESS LEVELS, never collapsed. `simulator_ready` is the
            # process, its ROS bridge and the physics app; `scene_ready` is one
            # exact run's world, verified. Only the second authorises a pick.
            "ros_bridge_ready": self.simulator_ready,
            "scene_status": self.scene_status,
            "scene_mismatch": self.scene_mismatch,
            "requested_fingerprint": self.requested_fingerprint,
            "acknowledged_fingerprint": (self.acknowledged.scene_fingerprint
                                         if self.acknowledged else ""),
            "expected_object_count": self.requested_object_count,
            "actual_object_count": (self.acknowledged.object_count
                                    if self.acknowledged else None),
            "acknowledged_scene": (self.acknowledged.to_dict()
                                   if self.acknowledged else None),
        }

    @property
    def in_flight_item(self) -> Optional[str]:
        """The item currently with the simulator, or None."""
        return self._in_flight[0].item_id if self._in_flight else None

    @property
    def robot_id(self) -> str:
        return getattr(self.robot, "robot_id", "") or ""

    @property
    def robot_profile_revision(self) -> str:
        return getattr(self.robot, "revision", "") or ""

    @property
    def scene_status(self) -> str:
        """building | ready | mismatch | failed | awaiting-acknowledgement."""
        if self.reset_failed_reason:
            return "failed"
        if self.scene_mismatch:
            return "mismatch"
        if self.reset_in_progress:
            return "building"
        if self.scene_ready:
            return "ready"
        return "awaiting-acknowledgement"

    @property
    def scene_ready(self) -> bool:
        """Is the PHYSICAL scene rebuilt for the scenario now being executed?

        Exact equality, not `>=`: a SCENE_READY for an older revision must never
        authorise a newer scenario, and a stale simulator replaying an old
        SCENE_READY must not satisfy the gate either.
        """
        return (not self.reset_in_progress
                and not self.reset_failed_reason
                and not self.scene_mismatch
                # A robot whose model did not validate has an unknown
                # relationship between what is commanded and what moves. No
                # scene is "ready" for it.
                and not self.robot_model_error
                and self.scene_revision == self.required_revision)

    def scene_block_reason(self) -> str:
        """Why physical execution is not authorised, in the operator's words."""
        if self.robot_model_error:
            return (f"the simulator could not stand up the selected robot: "
                    f"{self.robot_model_error}")
        if self.reset_failed_reason:
            return f"the simulator could not rebuild the scene: {self.reset_failed_reason}"
        if self.reset_in_progress:
            return "the simulator is rebuilding the physical scene"
        if self.scene_mismatch:
            return f"the simulator's scene does not match this run: {self.scene_mismatch}"
        if self.scene_revision != self.required_revision:
            if self.simulator_ready:
                # ACCURACY MATTERS HERE. Claiming the scene "has not been
                # rebuilt" while the operator is looking at a correct-looking
                # Panda, four cylinders and an empty container reads as a bug in
                # the dashboard. What is actually missing is the correlated
                # acknowledgement for THIS run.
                return ("Isaac is ready, but the scene acknowledgement for the "
                        "current run has not been received")
            return ("the physical scene has not been built and acknowledged for "
                    "this scenario yet — the previous run's objects may still "
                    "be in the container")
        return ""

    def rebind_robot(self, profile) -> None:
        """Adopt a NEW robot, and forget everything that described the old one.

        Called only from a reset, because a reset is the only moment a robot may
        change: there is no in-flight item, the engine is being replaced and the
        physical scene is about to be rebuilt.

        The layout is re-derived rather than kept. Every pose this bridge sends
        is converted through it, and the two arms do not share a workcell — the
        xArm 7 works a bin 80 mm nearer its base — so carrying the old layout
        forward would send the new arm to the old arm's coordinates with the
        right run id, the right revision and no way to notice.
        """
        self.robot = profile
        self.layout = layout_for_robot(profile)
        # A previous robot's acknowledgement says nothing about this one.
        self.acknowledged = None
        self.scene_revision = -1
        self.scene_mismatch = ""
        self.requested_fingerprint = ""
        self.scene_requested_for_run = ""
        self.robot_status = {}
        self.robot_model_error = ""
        self.simulator_ready = False

    def request_scene_reset(self, engine, revision: int) -> None:
        """Ask the backend to REBUILD its scene for `revision`, and gate on it.

        Called whenever a NEW scenario is generated. Generating a new software
        scenario is NOT sufficient to reset a physical backend, and treating it
        as sufficient is what let the arm be sent after objects that were
        already sitting in the container.
        """
        self._request_scene(engine, revision, rebuild=True)

    def request_scene_sync(self, engine, revision: Optional[int] = None) -> None:
        """Ask the backend to VERIFY OR BUILD its scene for the current run.

        Sent on EVERY run, including the first one — which is the fix for a real
        deadlock. The initial scene used to be trusted on the grounds that the
        launcher passed Isaac the same (preset, seed) the run was planned from,
        and that trust was applied inside `open_run()`. But `open_run()` runs
        inside the execution loop, which only runs after approval, and approval
        waits for the scene gate. So on a first `isaac-fiware` launch Isaac was
        up with a correct four-cylinder scene, the dashboard said the scene had
        not been rebuilt, and Approve stayed disabled with no way out but
        "Reset run & generate".

        Unlike a reset this permits Isaac to answer "already correct" after
        verifying, rather than destroying and recreating a scene that matches.
        It must still send a fresh SCENE_READY correlated with THIS run: an
        acknowledgement from the previous run is not evidence about this one.
        """
        if revision is None:
            revision = int(getattr(engine, "scenario_revision", 0))
        self._request_scene(engine, revision, rebuild=False)

    def _sync_scene_if_needed(self, engine) -> None:
        """Request a scene acknowledgement for this run, exactly once.

        Idempotent per (run_id, revision): a repeated READY, a re-published
        latched sample or a second tick must not restart the handshake and reset
        its timeout clock forever.
        """
        if not self.simulator_ready or engine.scenario is None:
            return
        revision = int(getattr(engine, "scenario_revision", 0))
        if (self.scene_requested_for_run == engine.run_id
                and self.required_revision == revision):
            return
        if self.scene_ready and self.scene_revision == revision:
            return
        self.request_scene_sync(engine, revision)

    def _request_scene(self, engine, revision: int, *, rebuild: bool) -> None:
        scenario = engine.scenario
        self.required_revision = int(revision)
        self.reset_in_progress = True
        self.reset_failed_reason = ""
        self.scene_mismatch = ""
        self.acknowledged = None
        self._reset_requested_at = time.monotonic()
        self._in_flight = None
        self._in_flight_command = None
        self.scene_requested_for_run = engine.run_id
        # Computed from the scenario THIS run planned against, by the same
        # function Isaac will use on the scene it actually has. That turns
        # "is the world the right one?" into a string comparison.
        self.requested_fingerprint = (
            scene_fingerprint(scenario, self.layout, self.robot_id)
            if scenario is not None else "")
        self.requested_object_count = len(scenario.items) if scenario else 0
        command = (IsaacCommandType.RESET_SCENE if rebuild
                   else IsaacCommandType.SYNC_SCENE)
        self._publish(IsaacCommand(
            command=command,
            run_id=engine.run_id,
            preset=scenario.preset if scenario else engine.config.preset,
            seed=int(scenario.seed if scenario else engine.config.seed),
            robot_id=self.robot_id,
            total_items=self.requested_object_count,
            scenario_revision=self.required_revision))
        engine.note_physical_progress(
            None, ("isaac_scene_reset_requested" if rebuild
                   else "isaac_scene_sync_requested"), None, None,
            (f"physical scene {'rebuild' if rebuild else 'verification'} "
             f"requested for scenario revision {self.required_revision}"),
            details={"scenario_revision": self.required_revision,
                     "preset": scenario.preset if scenario else "",
                     "robot_id": self.robot_id,
                     "scene_fingerprint": self.requested_fingerprint,
                     "expected_object_count": self.requested_object_count})

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
        # THE INITIAL SCENE IS NOT TRUSTED, and used to be.
        #
        # The old shortcut set `scene_revision = required_revision` here on the
        # first run, reasoning that the launcher gave Isaac the same (preset,
        # seed) the run was planned from. Two things were wrong with it. Built
        # from the right preset is not built FOR THIS RUN — nothing tied the
        # scene to a run_id or a revision. And this code path runs inside the
        # execution loop, which is only reached after approval, while approval
        # itself waits on the scene gate: on a first launch the gate could never
        # open. The handshake is now requested from the readiness path instead,
        # where it happens before the operator is asked to decide.
        scenario = engine.scenario
        self._publish(IsaacCommand(
            command=IsaacCommandType.RUN_BEGIN,
            run_id=engine.run_id,
            plan_id=engine.selected.plan_id if engine.selected else "",
            preset=scenario.preset if scenario else engine.config.preset,
            seed=int(scenario.seed if scenario else engine.config.seed),
            robot_id=self.robot_id,
            total_items=len(scenario.items) if scenario else 0,
        ))
        engine.note_physical_progress(
            None, "isaac_run_begin", None, None,
            "Isaac Sim asked to build the scene and stand by",
            details={"preset": scenario.preset if scenario else "",
                     "seed": int(scenario.seed if scenario else 0),
                     "run_id": engine.run_id,
                     "robot_id": self.robot_id,
                     "backend": ExecutionBackend.ISAAC.value})

    def abort_run(self, engine, reason: str) -> None:
        """Tell Isaac to stop. Used when approval is withdrawn or a re-plan lands."""
        if not self.run_open:
            return
        self._publish(IsaacCommand(
            command=IsaacCommandType.RUN_ABORT, run_id=self.gate.run_id,
            robot_id=self.robot_id))
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
            command=IsaacCommandType.RUN_END, run_id=self.gate.run_id,
            robot_id=self.robot_id))
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

        # THE SCENE GATE. Nothing may be picked until the physical scene has
        # been rebuilt for THIS scenario revision.
        if not self.scene_ready:
            return self._await_scene(engine)

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

    def _await_scene(self, engine) -> bool:
        """Wait for SCENE_READY, bounded. On timeout, HOLD — never proceed.

        Proceeding with stale scene contents is the failure this whole handshake
        exists to prevent, so a timeout degrades the run rather than continuing.
        """
        if self.reset_failed_reason:
            if not self._degraded:
                self._degraded = True
                engine.enter_degraded(
                    f"Isaac Sim could not rebuild the physical scene: "
                    f"{self.reset_failed_reason}. Execution is HELD — the "
                    "previous run's objects may still be in the container.")
            return False
        waited = time.monotonic() - self._reset_requested_at
        if waited < self.reset_timeout_s:
            return True
        if not self._degraded:
            self._degraded = True
            self.reset_failed_reason = (
                f"no SCENE_READY for revision {self.required_revision} within "
                f"{self.reset_timeout_s:.0f}s")
            engine.enter_degraded(
                f"Isaac Sim did not report SCENE_READY for scenario revision "
                f"{self.required_revision} within {self.reset_timeout_s:.0f}s. "
                "Execution is HELD rather than run against a stale scene. "
                "Restart the simulator, or check its log.")
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
            scenario_revision=self.required_revision,
            robot_id=self.robot_id,
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
            f"{feedback.item_id or '-'} run={feedback.run_id}"
            + (f" robot={feedback.robot_id}" if feedback.robot_id else ""))

        # STALE FEEDBACK FROM ANOTHER ROBOT IS DROPPED, whatever it says.
        #
        # Both processes share one DDS domain, and a simulator started for a
        # previous run with a different arm may simply never have been shut
        # down. Its READY would set `simulator_ready`, its SCENE_READY would be
        # measured against this run's fingerprint, and its ITEM_COMPLETED would
        # mark a placement executed that this run's robot never touched. The run
        # gate cannot see any of that: the ids and revisions may match perfectly
        # and still describe a different machine in a differently-arranged cell.
        #
        # An EMPTY robot_id is not a mismatch — an older simulator does not send
        # one, and refusing it would break a rolling upgrade for no safety gain.
        if (self.robot_id and feedback.robot_id
                and feedback.robot_id != self.robot_id):
            self.node.get_logger().warn(
                f"{LOG} ignoring {feedback.state.value} from robot "
                f"{feedback.robot_id!r}: this run selected {self.robot_id!r}")
            return

        # THE RUN GATE APPLIES TO EXECUTION, NOT TO LIFECYCLE.
        #
        # It used to apply to everything, and its first rule is "no run has been
        # opened yet (RUN_BEGIN not received)". A run is opened from the
        # execution loop, which runs after approval — so on a first launch the
        # simulator's READY was discarded, `simulator_ready` never became True,
        # the scene handshake never started, and approval waited on a gate that
        # nothing could open. The operator saw a correct scene, a disabled
        # Approve button, and no way forward except a reset.
        #
        # Readiness and the scene lifecycle are not claims about an item, and
        # they carry their own correlation: a run_id, a scenario revision and a
        # full scene acknowledgement, every field of which is checked in
        # `_on_reset_state`. They are therefore admitted before a run exists and
        # validated on their own terms.
        lifecycle = (feedback.state in (IsaacState.READY,
                                        IsaacState.SIMULATOR_READY,
                                        IsaacState.ROBOT_MODEL_INVALID)
                     or feedback.state in _SCENE_LIFECYCLE)
        if not lifecycle:
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

        if state is IsaacState.ROBOT_MODEL_INVALID:
            self._on_robot_model_invalid(engine, feedback)
            return

        if state in (IsaacState.READY, IsaacState.SIMULATOR_READY):
            # SIMULATOR-LEVEL READINESS ONLY. The process is up, its ROS bridge
            # is up and the physics application is running. It says nothing
            # about which run the scene corresponds to, and it authorises
            # nothing — that is what the scene handshake below is for.
            #
            # Refreshed on every report, not only the first: a simulator that
            # restarts with streaming newly enabled must be able to correct a
            # previously-published "unavailable".
            self.visualization = feedback.detail.get("visualization")
            self.simulator_version = feedback.detail.get("simulator_version")
            robot = feedback.detail.get("robot")
            if isinstance(robot, dict):
                self.robot_status = dict(robot)
                # A simulator that came back healthy clears a previous model
                # failure — otherwise a restart that fixed the configuration
                # would leave the run held for a reason no longer true.
                if robot.get("model_valid"):
                    self.robot_model_error = ""
            if not self.simulator_ready:
                self.simulator_ready = True
                self.node.get_logger().info(
                    f"{LOG} Isaac Sim reported {state.value} for run "
                    f"{feedback.run_id} — simulator up; scene not yet "
                    "acknowledged for this run")
                engine.note_physical_progress(
                    None, _PROGRESS_ACTION[IsaacState.READY], None, None,
                    "Isaac Sim process and ROS bridge ready — the scene is not "
                    "yet acknowledged for this run",
                    details={**feedback.detail,
                             "backend": ExecutionBackend.ISAAC.value})
            # THE INITIAL HANDSHAKE. Ask for the scene as soon as the simulator
            # can answer, and do it here rather than in the execution loop: the
            # loop only runs after approval, and approval waits for the scene.
            # Requesting it from the readiness path is what breaks that
            # deadlock, and it makes the first run take exactly the same
            # correlated path as a reset.
            self._sync_scene_if_needed(engine)
            return

        # NOTE the receiver: these are properties of the FEEDBACK MESSAGE, not of
        # the state enum. Calling them on `state` raises AttributeError, and
        # because this branch sits ahead of the progress handler it made EVERY
        # non-READY report fail to apply — the orchestrator saw the simulator
        # come up and then heard nothing more, and eventually timed the item out
        # while the arm had in fact completed it. Measured end to end.
        if state in _SCENE_LIFECYCLE:
            self._on_reset_state(engine, feedback)
            return

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
            # EVERY physical action event names the arm that performed it. The
            # audit trail outlives the run, and "the robot picked item-003" is
            # not a usable record when two robots are selectable.
            details={**feedback.detail, "isaac_state": state.value,
                     "robot_id": feedback.robot_id or self.robot_id})
        self.node.publish_execution()

    def _on_robot_model_invalid(self, engine, feedback: IsaacFeedback) -> None:
        """The simulator could not stand up its robot. HOLD — never proceed.

        Distinct from RUN_FAILED because it is a configuration fault rather than
        a physical one, and the remedy is different: nothing is retried, the
        backend goes DEGRADED, and approval stays disabled until a simulator
        reports a robot that validates. A partially loaded robot must never
        reach execution.
        """
        self.robot_model_error = (feedback.message
                                  or "the robot model did not validate")
        robot = feedback.detail.get("robot")
        if isinstance(robot, dict):
            self.robot_status = dict(robot)
        self.reset_in_progress = False
        self._in_flight = None
        self._in_flight_command = None
        self.node.get_logger().error(
            f"{LOG} ROBOT_MODEL_INVALID ({feedback.robot_id or '?'}): "
            f"{self.robot_model_error}")
        engine.note_physical_progress(
            None, "isaac_robot_model_invalid", None, None,
            f"the Isaac backend could not stand up "
            f"{feedback.robot_id or 'the selected robot'}: "
            f"{self.robot_model_error}",
            robot_state="idle",
            details={**feedback.detail, "robot_id": feedback.robot_id,
                     "isaac_state": IsaacState.ROBOT_MODEL_INVALID.value})
        if not self._degraded:
            self._degraded = True
            engine.enter_degraded(
                f"Isaac Sim reported ROBOT_MODEL_INVALID for "
                f"{feedback.robot_id or 'the selected robot'}: "
                f"{self.robot_model_error}. Execution is HELD and approval is "
                "disabled — a partially loaded robot is not run.")
        self.node.publish_execution()

    def _on_reset_state(self, engine, feedback: IsaacFeedback) -> None:
        """Track the scene-rebuild lifecycle and open the gate on SCENE_READY."""
        state = feedback.state
        if state is IsaacState.SCENE_READY:
            # ACCEPTED ONLY WHEN EVERY CORRELATION FIELD MATCHES. A SCENE_READY
            # is a claim about one exact run's world, and the ways it can be
            # wrong are not interchangeable: an old revision, another run's id, a
            # different preset or seed, the right ids with different geometry, a
            # missing object, an unverified home pose. Each is named rather than
            # collapsed into "not ready", because "not ready" is not actionable
            # and the operator can see a scene sitting right there.
            scenario = engine.scenario
            reasons = []
            acknowledgement = feedback.scene
            if acknowledgement is None:
                # An older simulator that answers with no acknowledgement can
                # still be correlated on the revision it carries — but nothing
                # beyond that is verified, and it is recorded as such.
                if feedback.scenario_revision != self.required_revision:
                    reasons.append(
                        f"acknowledged scenario revision "
                        f"{feedback.scenario_revision} but this run is at "
                        f"{self.required_revision}")
            else:
                reasons = acknowledgement.mismatches(
                    run_id=engine.run_id,
                    scenario_id=(scenario.scenario_id if scenario else ""),
                    revision=self.required_revision,
                    preset=(scenario.preset if scenario else ""),
                    seed=int(scenario.seed) if scenario else 0,
                    fingerprint=self.requested_fingerprint,
                    object_count=self.requested_object_count,
                    robot_id=self.robot_id,
                    robot_profile_revision=self.robot_profile_revision)
            if reasons:
                self.reset_in_progress = False
                self.scene_mismatch = "; ".join(reasons)
                self.acknowledged = acknowledgement
                self.node.get_logger().warn(
                    f"{LOG} rejecting SCENE_READY: {self.scene_mismatch}")
                engine.note_physical_progress(
                    None, "isaac_scene_rejected", None, None,
                    f"scene acknowledgement rejected: {self.scene_mismatch}",
                    details={"scenario_revision": feedback.scenario_revision,
                             "required_revision": self.required_revision,
                             "reasons": reasons})
                self.node.publish_execution()
                return
            self.scene_revision = self.required_revision
            self.acknowledged = acknowledgement
            self.scene_mismatch = ""
            self.reset_in_progress = False
            self.reset_failed_reason = ""
            self.gate.adopt(engine.run_id)     # a verified scene is a fresh run
            verified_only = bool(acknowledgement
                                 and acknowledgement.verified_without_rebuild)
            self.node.get_logger().info(
                f"{LOG} scene {'verified' if verified_only else 'rebuilt'} for "
                f"revision {self.scene_revision}")
            engine.note_physical_progress(
                None, "isaac_scene_ready", None, None,
                (f"physical scene {'verified' if verified_only else 'rebuilt'} "
                 f"for scenario revision {self.scene_revision}"),
                details={**feedback.detail,
                         "scenario_revision": self.scene_revision,
                         "scene": (acknowledgement.to_dict()
                                   if acknowledgement else None)})
        elif state is IsaacState.RESET_FAILED:
            self.reset_in_progress = False
            self.reset_failed_reason = feedback.message or "reset failed"
            engine.note_physical_progress(
                None, "isaac_scene_reset_failed", None, None,
                f"physical scene rebuild FAILED: {self.reset_failed_reason}",
                details=dict(feedback.detail))
        else:
            self.reset_in_progress = True
            engine.note_physical_progress(
                None, f"isaac_scene_{state.value.lower()}", None, None,
                feedback.message or f"Isaac Sim: {state.value}",
                details=dict(feedback.detail))
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
            "robot_id": feedback.robot_id or self.robot_id,
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
