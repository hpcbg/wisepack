"""WISEPACK physical execution backend — NVIDIA Isaac Sim 6.0.1 standalone app.

    ISAAC_SIM_ROOT/python.sh simulators/isaac/wisepack_isaac.py \
        --preset isaac_cylinders_smoke --seed 42 [--headless]

Normally started by ``scripts/run_wisepack_isaac.sh``, which handles simulator
discovery, the ROS environment isolation and process ownership.

WHAT THIS PROCESS DOES
----------------------
1. builds the procedural scene — ground, table, open-top container, cylinders,
   lighting, camera — from the SAME (preset, seed) the WISEPACK stack planned
   from, so the object ids, dimensions and masses match by construction;
2. loads the Franka Emika Panda from the Isaac Sim asset root;
3. reports READY on /wisepack/isaac/feedback;
4. executes each accepted placement it is sent, one at a time;
5. reports the MEASURED physical outcome of every item.

WHAT IT DOES NOT DO
-------------------
Perceive. There is no camera pipeline, no detector and no pose estimator in this
iteration: item poses are ground truth, because this process spawned the items
itself. That is a stated limitation, not an oversight — see the README. The
extension point is ``wisepack_core.isaac_transform.table_pose_for_index``, and
replacing it with real perception changes nothing else.

Plan. It receives accepted placements and executes them. It never chooses an
item order, never re-plans, and never decides that a placement is a good idea.
The approval gate is upstream and stays upstream.

ORDER OF IMPORTS MATTERS AND IS NOT STYLE
-----------------------------------------
``SimulationApp`` must be constructed before ANY omniverse-level import, because
those modules are provided by the extension runtime and do not exist as
importable packages until the app has loaded them. Likewise ``import rclpy``
resolves to Isaac's internally-built ROS 2 only after the ros2.bridge extension
is enabled. Hence the deferred imports below, which are not an accident.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# --------------------------------------------------------------------------- #
# 1. Arguments and the SimulationApp
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# wisepack_core and wisepack_bringup are imported from the workspace SOURCE, not
# from a colcon install. Isaac's bundled interpreter cannot import a built ROS
# package anyway, and both modules are pure Python — wisepack_core is stdlib
# only, and wisepack_bringup.qos needs nothing but rclpy, which Isaac provides
# its own build of.
for _pkg in ("wisepack_core", "wisepack_bringup"):
    _path = os.path.join(REPO, "wisepack_ws", "src", _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)
# The repository root, so the sibling modules are reached as
# `simulators.isaac.<name>` rather than by bare name. Bare names are NOT safe
# here: Isaac's interpreter pre-bundles a large site-packages tree, and
# `import config` resolved to OpenCV's cv2/config.py before this was a package.
# See simulators/__init__.py for the measured traceback.
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="WISEPACK Isaac Sim physical execution backend")
    parser.add_argument("--preset",
                        default=os.environ.get("WISEPACK_PRESET",
                                               "isaac_cylinders_smoke"),
                        help="WISEPACK scenario preset to build the scene from")
    parser.add_argument("--seed", type=int,
                        default=int(os.environ.get("WISEPACK_SEED", "42")),
                        help="Deterministic seed, matching the WISEPACK stack")
    parser.add_argument("--headless", action="store_true",
                        default=os.environ.get("WISEPACK_ISAAC_HEADLESS") == "1",
                        help="Run with no window (required over SSH)")
    parser.add_argument("--exit-on-complete", action="store_true",
                        help="Close once the run finishes (used by the validator)")
    parser.add_argument("--max-runtime", type=float, default=0.0,
                        help="Wall-clock ceiling in seconds; 0 disables it")
    parser.add_argument("--self-test", action="store_true",
                        help=("Build the scene, report READY, then exit without "
                              "waiting for commands. Proves the simulator, the "
                              "GPU and the ROS bridge work."))
    parser.add_argument("--reset-test", action="store_true",
                        help=("Bounded live validation of the scene-reset "
                              "handshake: execute one item, rebuild the scene "
                              "for a NEW scenario, verify the container is "
                              "empty and every source object respawned, then "
                              "execute the first item of the new run."))
    parser.add_argument("--reset-seed", type=int, default=1234,
                        help="Seed for the second scenario in --reset-test")
    parser.add_argument("--smoke-test", type=int, default=0, metavar="N",
                        help=("Self-driving physical validation: plan locally "
                              "with the SAME optimizer the stack uses, then "
                              "publish N real commands onto the ROS command "
                              "topic and execute them. Exercises the whole "
                              "contract over DDS without needing the WISEPACK "
                              "stack, Docker or an operator."))
    return parser.parse_known_args()[0]


ARGS = _parse_args()

# Streaming is decided BEFORE the app is constructed. The livestream extension
# captures the application framebuffer, and its size plus headless/hide_ui are
# launch configuration rather than runtime settings — enabling it afterwards
# would stream a window that had already been created the wrong shape.
sys.path.insert(0, os.path.join(REPO, "simulators"))
from simulators.isaac.streaming import StreamingConfig               # noqa: E402

STREAM = StreamingConfig.from_env()

_LAUNCH = {
    "headless": bool(ARGS.headless),
    # The scene is five boxes, four cylinders and one arm. The heavier renderers
    # buy nothing here and cost startup time on every run.
    "renderer": "RaytracedLighting",
}
if STREAM.enabled:
    # hide_ui False so the streamed frame carries the viewport, per the shipped
    # standalone_examples/api/isaacsim.simulation_app/livestream.py.
    _LAUNCH.update({
        "headless": True, "hide_ui": False,
        "width": STREAM.width, "height": STREAM.height,
        "window_width": STREAM.width, "window_height": STREAM.height,
    })

from isaacsim import SimulationApp                                  # noqa: E402

simulation_app = SimulationApp(_LAUNCH)

# --------------------------------------------------------------------------- #
# 2. Omniverse-level imports (only valid after SimulationApp exists)
# --------------------------------------------------------------------------- #

import carb                                                          # noqa: E402
import numpy as np                                                   # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils             # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager       # noqa: E402

# The Franka helper and the ROS 2 bridge both live in extensions that are not
# enabled by default in a standalone app.
app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")
app_utils.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

STREAM_STATUS = None            # set by _start_streaming(); None = not attempted


def _start_streaming():
    """Enable Isaac's WebRTC livestream, or explain why it is not available.

    Returns (status, message). Never raises: a stream is an optional convenience
    and must not be able to stop a physical run that is otherwise working.
    """
    # Local imports: this runs BEFORE the module-level import block below, so
    # nothing from it is bound yet — including LOG_APP, whose absence turned a
    # working stream into a NameError traceback on the success path.
    from simulators.isaac.config import LOG_APP as _LOG            # noqa: PLC0415
    from simulators.isaac.streaming import (                        # noqa: PLC0415
        PRIMARY_STREAM_SETTING, missing_extensions, port_is_free,
    )
    from wisepack_core.visualization import VisualizationStatus     # noqa: PLC0415

    manager = omni.kit.app.get_app().get_extension_manager()
    absent = missing_extensions(
        lambda name: bool(manager.get_extension_id_by_module(name))
        or manager.is_extension_enabled(name)
        or any(e.get("name") == name for e in manager.get_extensions()))
    if absent:
        return (VisualizationStatus.ERROR,
                f"required livestream extension(s) not available: {absent}")

    # Refuse to start a SECOND stream server on an occupied port. Kit silently
    # falls back to "an unoccupied port" in that case, so the URL WISEPACK
    # publishes would point at a different, older stream — which shows a
    # picture, of the wrong thing.
    for label, port in (("signal", STREAM.signal_port),
                        ("stream", STREAM.stream_port)):
        if not port_is_free(port):
            return (VisualizationStatus.ERROR,
                    f"{label} port {port} is already in use — another Isaac "
                    "stream is probably running. Stop it, or set "
                    "WISEPACK_ISAAC_SIGNAL_PORT / WISEPACK_ISAAC_STREAM_PORT.")

    simulation_app.set_setting(f"{PRIMARY_STREAM_SETTING}/signalPort",
                               STREAM.signal_port)
    simulation_app.set_setting(f"{PRIMARY_STREAM_SETTING}/streamPort",
                               STREAM.stream_port)
    simulation_app.set_setting(f"{PRIMARY_STREAM_SETTING}/streamType", "webrtc")
    simulation_app.set_setting(f"{PRIMARY_STREAM_SETTING}/targetFps",
                               STREAM.target_fps)
    simulation_app.set_setting("/app/window/drawMouse", True)
    app_utils.enable_extension("omni.kit.livestream.app")
    simulation_app.update()
    print(f"{_LOG} WebRTC livestream enabled on signal "
          f"{STREAM.host}:{STREAM.signal_port} (media UDP {STREAM.stream_port})")
    return (VisualizationStatus.READY, "")


if STREAM.enabled:
    import omni.kit.app                                             # noqa: E402
    STREAM_STATUS = _start_streaming()

import rclpy                                                         # noqa: E402
from isaacsim.robot.experimental.manipulators.examples.franka.franka import (  # noqa: E402
    Franka,
)

from wisepack_core.generator import build_scenario                   # noqa: E402
from wisepack_core.isaac_contract import (                           # noqa: E402
    IsaacCommand, IsaacCommandType, IsaacState,
)
from wisepack_core.isaac_transform import (                          # noqa: E402
    DEFAULT_LAYOUT, dimensions_for, placement_pose, table_pose_for_index,
)

from simulators.isaac.config import (                                # noqa: E402
    LOG_APP, AppConfig, MotionConfig, PhysicsConfig,
)
from simulators.isaac.bridge import (                                # noqa: E402
    IsaacRosBridge, init_ros, shutdown_ros,
)
from simulators.isaac.robot import PandaSequence                     # noqa: E402
from simulators.isaac.scene import ROBOT_PATH, WisepackScene         # noqa: E402


# --------------------------------------------------------------------------- #
# 3. The application
# --------------------------------------------------------------------------- #


class WisepackIsaacApp:
    """Owns the scene, the robot sequence and the ROS transport.

    ONE COMMAND AT A TIME. WISEPACK dispatches the next placement only after the
    previous one reported a terminal state, so there is never a queue here. A
    command arriving while an item is in flight is a protocol violation and is
    rejected loudly rather than buffered — buffering it would hide the fact that
    the two ends disagree about who is waiting for whom.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.layout = DEFAULT_LAYOUT
        self.scene = WisepackScene(self.layout, config.physics)
        self.bridge: IsaacRosBridge = None                  # type: ignore[assignment]
        self.sequence: PandaSequence = None                 # type: ignore[assignment]
        self.robot = None
        self.run_id = ""
        self.run_active = False
        self.pending: IsaacCommand = None                   # type: ignore[assignment]
        self.items_completed = 0
        self.items_failed = 0
        self.scenario = None
        self._started_at = time.monotonic()
        self._ready_announced = False
        #: The command currently being executed, kept so the terminal feedback
        #: can echo the identifiers WISEPACK sent (sequence_index, container_id,
        #: dimensions). The sequence clears its own copy when it finishes.
        self._last_command: IsaacCommand = None         # type: ignore[assignment]
        #: >0 in --smoke-test mode: how many items to self-drive. Also switches
        #: on the SMOKE-* marker lines the validator greps for.
        #: Scenario revision the CURRENT scene corresponds to. None until a
        #: reset has happened; the startup build is revision-agnostic and is
        #: trusted by the orchestrator (see isaac_bridge.open_run).
        self._scene_revision = None
        self.smoke_items = 0
        self._smoke_queue: list = []
        self._smoke_publisher = None
        self._smoke_dispatched = 0
        #: Set once the run has reached a terminal state, so `exit_on_complete`
        #: can distinguish "finished" from "has not started yet". Counting items
        #: was wrong: a run that ended having executed nothing left the loop
        #: spinning forever.
        self._run_ended = False

    # -- setup ---------------------------------------------------------- #

    def build(self) -> None:
        cfg = self.config
        print(f"{LOG_APP} preset={cfg.preset} seed={cfg.seed} "
              f"headless={cfg.headless} device={cfg.physics.device}")
        self.scenario = build_scenario(cfg.preset, seed=cfg.seed)
        print(f"{LOG_APP} scenario {self.scenario.scenario_id}: "
              f"{len(self.scenario.items)} items")

        inner = (self.scenario.container_template.inner_size
                 if self.scenario.container_template else None)
        if inner is not None:
            # Fail here, before a robot moves, rather than discovering mid-run
            # that a placement is outside the arm's envelope. An unreachable IK
            # goal does not raise — it converges short and the gripper closes on
            # air, which reads as a grasp problem and is not one.
            self.layout.validate(inner, len(self.scenario.items),
                                 clearance_m=cfg.motion.container_clearance)

        SimulationManager.set_physics_sim_device(cfg.physics.device)
        simulation_app.update()

        # Containers the plan may use. The scenario names one template; a plan
        # that opens a second gets one, so a two-container run still renders.
        container_ids = [f"CNT-{n:02d}" for n in (1, 2)]
        self.scene.build(self.scenario, container_ids)

        print(f"{LOG_APP} loading the Franka Panda from the Isaac asset root")
        self.robot = Franka(robot_path=ROBOT_PATH, create_robot=True)
        # Mounted ON the table top, which is the datum every pose is measured
        # against. Set before physics starts, so it is a placement rather than a
        # teleport.
        self.robot.set_world_poses(
            positions=np.array([[0.0, 0.0, self.layout.table_top_z_m]]),
            orientations=np.array([[1.0, 0.0, 0.0, 0.0]]))

        if STREAM.enabled:
            self._pin_spectator_camera()

        self.sequence = PandaSequence(
            scene=self.scene, layout=self.layout, motion=cfg.motion,
            physics_dt=cfg.physics.physics_dt,
            on_state=self._on_sequence_state, on_done=self._on_item_done)
        self.sequence.attach_robot(self.robot)

        init_ros()
        self.bridge = IsaacRosBridge(on_command=self._on_command)

    def _pin_spectator_camera(self) -> None:
        """Point the streamed viewport at the workcell, once, at startup.

        Without this the stream carries whatever the default development
        viewport frames — on a freshly created stage that is a perspective view
        pointed away from the table, so an operator opening the Simulator View
        sees an empty grid and reasonably concludes the run is broken. The demo
        camera is placed by the scene builder to frame the table, the Panda, the
        pick row and the container together.
        """
        from simulators.isaac.streaming import SPECTATOR_CAMERA     # noqa: PLC0415
        try:
            from omni.kit.viewport.utility import get_active_viewport  # noqa: PLC0415
            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("no active viewport")
            viewport.camera_path = SPECTATOR_CAMERA
            print(f"{LOG_APP} stream camera pinned to {SPECTATOR_CAMERA}")
        except Exception as exc:                        # noqa: BLE001
            # Named, not swallowed. The run is still valid — this only affects
            # what a spectator sees — so it must not abort execution, but a
            # silent failure here is exactly the "empty grid" confusion above.
            carb.log_warn(f"{LOG_APP} could not pin the stream camera to "
                          f"{SPECTATOR_CAMERA}: {exc!r}. The stream will show "
                          "the default viewport.")

    def visualization(self) -> dict:
        """The backend-neutral visualization descriptor for this run.

        Isaac-specific knowledge stops here: what leaves this method is the same
        shape a real robot cell would publish.
        """
        from simulators.isaac.streaming import (                    # noqa: PLC0415
            describe, desktop_descriptor,
        )
        from wisepack_core.visualization import (                   # noqa: PLC0415
            VisualizationStatus, unavailable,
        )
        if STREAM.enabled:
            status, message = (STREAM_STATUS
                               or (VisualizationStatus.ERROR,
                                   "streaming was requested but never started"))
            return describe(STREAM, status=status, message=message,
                            extra={"preset": self.config.preset}).to_dict()
        display = os.environ.get("DISPLAY", "")
        if not self.config.headless and display:
            # A GUI run on a real desktop IS watchable — just not by us. Say so
            # rather than reporting "unavailable" to an operator who is already
            # looking at it through NoMachine or Sunshine.
            return desktop_descriptor(display).to_dict()
        return unavailable(
            "isaac",
            "Isaac Sim is running headless with WebRTC streaming disabled "
            "(set WISEPACK_ISAAC_STREAMING=1 to enable it)").to_dict()

    # -- ROS in ---------------------------------------------------------- #

    def _on_command(self, command: IsaacCommand) -> None:
        if command.command is IsaacCommandType.RUN_BEGIN:
            self._begin_run(command)
            return

        if command.command is IsaacCommandType.RESET_SCENE:
            # Deliberately NOT run-gated: a reset is how a new run becomes
            # legitimate, so gating it on the previous run's id would make the
            # scene un-resettable exactly when it most needs resetting.
            self._reset_scene(command)
            return

        reason = self.bridge.gate.reject_reason(
            command.run_id, command.sequence_index, command.attempt)
        if reason:
            print(f"{LOG_APP} ignoring {command.command.value}: {reason}")
            return

        if command.command is IsaacCommandType.EXECUTE_ITEM:
            if self.sequence.busy or self.pending is not None:
                print(f"{LOG_APP} REJECTED {command.item_id}: "
                      f"{self.sequence.command.item_id if self.sequence.command else 'an item'} "
                      "is still in flight. WISEPACK must wait for a terminal "
                      "state before dispatching the next placement.")
                return
            self.pending = command
        elif command.command is IsaacCommandType.RUN_END:
            self._end_run("WISEPACK reported the plan complete")
        elif command.command is IsaacCommandType.RUN_ABORT:
            self.sequence.abort("WISEPACK sent RUN_ABORT")
            self.pending = None

    def _begin_run(self, command: IsaacCommand) -> None:
        """Adopt a new run and announce READY.

        The scene is NOT rebuilt: it was built from (preset, seed) at startup and
        RUN_BEGIN carries the same pair. If they disagree, that is a launcher
        misconfiguration and the run would be executing a plan for objects that
        are not in the scene — so it is reported rather than papered over.
        """
        if command.preset and command.preset != self.config.preset:
            print(f"{LOG_APP} WARNING: WISEPACK planned preset "
                  f"{command.preset!r} but this simulator built "
                  f"{self.config.preset!r}. Item ids and dimensions will not "
                  "match. Restart Isaac with the same --preset.")
        if command.seed and int(command.seed) != int(self.config.seed):
            print(f"{LOG_APP} WARNING: WISEPACK planned seed {command.seed} but "
                  f"this simulator built seed {self.config.seed}.")

        self.bridge.gate.adopt(command.run_id)
        self.run_id = command.run_id
        self.run_active = True
        self.items_completed = 0
        self.items_failed = 0
        self.sequence.abort("new run opened")
        self.pending = None
        self._announce_ready()

    def _announce_ready(self) -> None:
        self.bridge.publish(
            IsaacState.READY, self.run_id or "isaac-standby",
            message="Isaac Sim scene built; physical execution may begin",
            detail={
                "preset": self.config.preset,
                "seed": self.config.seed,
                "items": sorted(self.scene.items),
                "containers": {k: v.to_dict()
                               for k, v in self.scene.containers.items()},
                "motion": self.config.motion.to_dict(),
                "grasp": "temporary fixed joint (secure-grasp approximation)",
                "perception": "none — item poses are ground truth",
                "headless": self.config.headless,
                # How to WATCH this run, in the backend-neutral shape. Metadata
                # only: rendered frames never travel on this path.
                "visualization": self.visualization(),
                "simulator_version": _isaac_version(),
            })
        self._ready_announced = True

    # -- sequence callbacks ---------------------------------------------- #

    def _on_sequence_state(self, state: IsaacState, message: str,
                           detail: dict) -> None:
        command = self.sequence.command
        if self.smoke_items:
            print(f"SMOKE-STATE {state.value} "
                  f"{command.item_id if command else '-'}")
        self.bridge.publish(
            state, self.run_id,
            item_id=command.item_id if command else None,
            sequence_index=command.sequence_index if command else -1,
            container_id=command.container_id if command else None,
            dimensions=command.dimensions if command else None,
            source_pose=command.source_pose if command else None,
            target_pose=command.target_pose if command else None,
            message=message, detail=detail)

    def _on_item_done(self, item_id: str, outcome) -> None:
        """Report the MEASURED result. Never the target dressed up as an outcome."""
        command = self._last_command
        if command is not None and command.item_id != item_id:
            command = None
        if outcome.ok:
            self.items_completed += 1
            state = IsaacState.ITEM_COMPLETED
        else:
            self.items_failed += 1
            state = IsaacState.ITEM_FAILED

        if self.smoke_items:
            # Machine-readable lines for scripts/validate_isaac_sim.sh. Printed
            # from the MEASURED outcome, so a validator cannot pass on a run
            # that reported a target as though it were a result.
            #
            # The terminal state gets a SMOKE-STATE line too. Without it the
            # marker vocabulary is inconsistent — every intermediate state is
            # greppable and the one that says whether the item actually made it
            # is not — and the validator reported "no ITEM_COMPLETED feedback"
            # for a run that had completed the item.
            print(f"SMOKE-STATE {state.value} {item_id}")
            print(f"SMOKE-RESULT {'PASS' if outcome.ok else 'FAIL'} {item_id} "
                  f"{outcome.message}")
            if outcome.actual_pose is not None:
                print(f"SMOKE-POSE {item_id} target="
                      f"{outcome.target_pose.to_dict() if outcome.target_pose else None} "
                      f"actual={outcome.actual_pose.to_dict()} "
                      f"error_mm={outcome.position_error_mm}")

        detail = dict(outcome.detail)
        detail.update({
            "settled": outcome.settled,
            "settle_timed_out": outcome.timed_out,
            "axis_error_deg": (None if outcome.axis_error_deg is None
                               else round(outcome.axis_error_deg, 2)),
            "notes": list(outcome.notes),
        })
        self.bridge.publish(
            state, self.run_id, item_id=item_id,
            sequence_index=command.sequence_index if command else -1,
            container_id=command.container_id if command else None,
            dimensions=command.dimensions if command else None,
            source_pose=command.source_pose if command else None,
            target_pose=outcome.target_pose,
            actual_pose=outcome.actual_pose,
            position_error_mm=outcome.position_error_mm,
            message=outcome.message, detail=detail)
        if command is not None:
            self.bridge.gate.mark_done(command.sequence_index, command.attempt)
        self._last_command = None

    def _reset_scene(self, command: IsaacCommand) -> None:
        """Rebuild the physical scene for a NEW scenario, then report SCENE_READY.

        THE SAFETY OPERATION. Everything that could carry state from the previous
        run is undone here, in an order that matters: stop the arm before
        touching what it might be holding, drop the weld before deleting the
        body it welds to, and only then rebuild.
        """
        revision = command.scenario_revision
        preset = command.preset or self.config.preset
        seed = int(command.seed or self.config.seed)
        self.bridge.publish(
            IsaacState.RESETTING, command.run_id,
            scenario_revision=revision,
            message=f"rebuilding the scene for {preset} seed {seed} "
                    f"(revision {revision})")
        print(f"{LOG_APP} RESET_SCENE -> {preset} seed {seed} rev {revision}")

        try:
            # 1. Stop the arm and drop anything it is holding. Order matters:
            #    abort() detaches the grasp joint BEFORE any body is deleted.
            self.sequence.abort("scene reset requested")
            self.pending = None
            self._last_command = None

            # 2. Cancel the previous run outright. Late feedback for it is
            #    rejected from here on because the gate no longer knows that id.
            self.run_active = False
            self._run_ended = False
            self.items_completed = 0
            self.items_failed = 0
            self.bridge.gate.adopt(command.run_id)

            # 3. Home the robot with an open gripper, and let it get there
            #    before the world changes underneath it.
            self.robot.open_gripper()
            self.robot.reset_to_default_pose()
            for _ in range(60):
                simulation_app.update()

            # 4. STOP THE TIMELINE BEFORE TOUCHING THE STAGE.
            #
            #    Not optional, and not a tidiness measure. Deleting a rigid-body
            #    prim while physics is playing invalidates the PhysX *tensor
            #    simulation view* — and that view is shared by every prim in the
            #    stage, the Franka articulation included. Measured on a live
            #    run: the items were removed and respawned correctly, the reset
            #    reported success, and the very next read of the arm's joints
            #    raised
            #
            #        "Simulation view object is invalidated and cannot be used
            #         again to call getDofPositions"
            #
            #    i.e. the scene looked ready while the robot was unusable. That
            #    is precisely the failure this whole handshake exists to
            #    prevent, so the stage is mutated only while stopped, and the
            #    views are rebuilt by the subsequent play().
            app_utils.stop()
            for _ in range(5):
                simulation_app.update()

            # 5. Rebuild the objects from the NEW preset and seed. This removes
            #    every previous cylinder — including the ones lying in the
            #    container — so the container is empty by construction.
            self.scenario = build_scenario(preset, seed=seed)
            self.config.preset, self.config.seed = preset, seed
            self.scene.reset_items(self.scenario)

            # 6. Start physics again. Prims created while stopped acquire their
            #    views here, which is why step 5 runs between stop and play.
            app_utils.play()
            simulation_app.update()

            # 7. Re-command the home pose. stop() restores the AUTHORED joint
            #    values, which is not the ready pose the sequence assumes, and
            #    the drives would otherwise drift there over the first seconds
            #    of the new run — the same undefined starting configuration that
            #    startup already guards against.
            self.robot.open_gripper()
            self.robot.reset_to_default_pose()

            # 8. Zero velocities and let contacts settle before anything is
            #    reported ready.
            def _update(frames):
                for _ in range(frames):
                    simulation_app.update()
            self.scene.settle_items(_update, frames=120)

            # 9. PROVE the rebuilt world is usable before claiming it is. A
            #    scene that reports SCENE_READY with a dead articulation view or
            #    an unreadable item is worse than one that reports RESET_FAILED,
            #    because the orchestrator would authorise motion against it.
            self._verify_scene_usable()

            summary = {
                "preset": preset, "seed": seed,
                "items": sorted(self.scene.items),
                "containers": {k: v.to_dict()
                               for k, v in self.scene.containers.items()},
                "scenario_id": self.scenario.scenario_id,
            }
            self._scene_revision = revision
            self.bridge.publish(
                IsaacState.SCENE_READY, command.run_id,
                scenario_revision=revision,
                message=f"scene rebuilt for {self.scenario.scenario_id} "
                        f"({len(self.scene.items)} items)",
                detail=summary)
            print(f"{LOG_APP} SCENE_READY revision {revision}: "
                  f"{len(self.scene.items)} items, container empty")
        except Exception as exc:                        # noqa: BLE001
            # Reported, never swallowed: the orchestrator HOLDS on RESET_FAILED
            # rather than running against a stale scene.
            carb.log_error(f"{LOG_APP} scene reset failed: {exc!r}")
            self.bridge.publish(
                IsaacState.RESET_FAILED, command.run_id,
                scenario_revision=revision,
                message=f"{type(exc).__name__}: {exc}")

    def _verify_scene_usable(self) -> None:
        """Raise unless the rebuilt world can actually be commanded and read.

        Called between rebuilding and announcing SCENE_READY. Every check here
        corresponds to something that has been observed to survive a rebuild
        looking healthy:

          * the articulation's physics view can be invalidated by a stage
            mutation and still answer ``is_valid``-style questions — reading the
            joints is what actually proves it;
          * a freshly created RigidPrim whose view failed to initialise returns
            no pose, and a plan built against that item would send the arm to a
            pose nobody computed;
          * a grasp joint left attached means the gripper still owns a body that
            no longer exists.
        """
        import numpy as _np                              # noqa: PLC0415

        dof = self.robot.get_dof_positions().numpy()[0]
        if not _np.all(_np.isfinite(dof)):
            raise RuntimeError(
                "the arm reported non-finite joint positions after the rebuild")

        unreadable = [item for item in self.scene.items
                      if self.scene.item_world_pose(item) is None]
        if unreadable:
            raise RuntimeError(
                f"{len(unreadable)} item(s) have no readable pose after the "
                f"rebuild: {', '.join(sorted(unreadable))}")

        if self.sequence.grasp.is_attached:
            raise RuntimeError(
                f"the grasp joint for {self.sequence.grasp.attached_item} "
                "survived the reset")

    def _pre_pick_refusal(self, command: IsaacCommand) -> str:
        """Physical sanity checks before ANY motion. "" when it is safe to pick.

        Every one of these is a way the arm could otherwise be sent somewhere
        the plan does not describe. When one fails the correct response is to
        stop and report — NOT to move toward an object at an unexpected place,
        which is how a mis-synchronised scene becomes uncontrolled motion.
        """
        item = command.item_id or ""
        if command.scenario_revision and self._scene_revision is not None \
                and command.scenario_revision != self._scene_revision:
            return (f"command is for scenario revision "
                    f"{command.scenario_revision} but the scene was built for "
                    f"{self._scene_revision}")
        if not self.scene.has_item(item):
            return f"{item} does not exist in the current scene"
        if self.sequence.grasp.is_attached:
            return (f"a grasp joint for {self.sequence.grasp.attached_item} is "
                    "still attached — the previous item was never released")
        pose = self.scene.item_world_pose(item)
        if pose is None:
            return f"{item} has no readable pose"

        import numpy as _np                              # noqa: PLC0415
        from wisepack_core.isaac_transform import pose_to_world as _to_world  # noqa: PLC0415
        actual = _np.asarray(pose[0], dtype=float)

        # Already in the destination container? Then the plan and the world
        # disagree about where this item is, and picking is meaningless.
        if command.target_pose is not None:
            origin = self.layout.container_origin_for(
                command.container_id or "CNT-01")
            inner = command.container_inner_mm or {}
            from wisepack_core.isaac_transform import mm_to_m as _mm  # noqa: PLC0415
            inside = (origin[0] <= actual[0] <= origin[0] + _mm(inner.get("x", 0))
                      and origin[1] <= actual[1] <= origin[1] + _mm(inner.get("y", 0))
                      and actual[2] >= origin[2] - 0.02)
            if inside:
                return (f"{item} is ALREADY inside {command.container_id} — the "
                        "physical scene does not match the plan (was the scene "
                        "reset skipped?)")

        # Near the pose the plan was built against?
        if command.source_pose is not None:
            expected, _ = _to_world(command.source_pose, self.layout)
            drift = float(_np.linalg.norm(actual - _np.asarray(expected)))
            if drift > self.config.motion.max_source_drift_m:
                return (f"{item} is {drift*1000:.0f} mm from its expected source "
                        f"pose (limit {self.config.motion.max_source_drift_m*1000:.0f} "
                        "mm) — the scene does not match the plan")
        return ""

    # -- self-driving smoke mode ----------------------------------------- #

    def prepare_smoke_run(self, count: int) -> None:
        """Plan locally and queue real commands for the ROS command topic.

        Uses the SAME optimizer and the SAME transform layer the WISEPACK stack
        uses, so what gets validated is the real contract rather than a
        hand-written approximation of it. The commands are PUBLISHED, not
        injected: they leave this process on /wisepack/isaac/command and come
        back through its own subscription over DDS, so the smoke test exercises
        the serialization, the QoS and the run gating end to end.

        This exists so the physical validation can run with no Docker, no
        Orion-LD and no operator. The orchestrator half of the contract is
        covered by the ordinary test suite.
        """
        from std_msgs.msg import String                        # noqa: PLC0415
        from wisepack_bringup import topics as WT              # noqa: PLC0415
        from wisepack_bringup.qos import qos_for               # noqa: PLC0415
        from wisepack_core.packing import OptimizerConfig, pack_optimized  # noqa: PLC0415
        from wisepack_core.validator import PlacementValidator  # noqa: PLC0415

        plan = pack_optimized(self.scenario, config=OptimizerConfig(
            seed=self.config.seed, restarts=6))
        PlacementValidator().validate_plan(plan, self.scenario)
        print(f"{LOG_APP} smoke plan: {plan.containers_required} container(s), "
              f"{len(plan.placements)} placements, valid={plan.is_valid}")

        run_id = f"smoke-{self.config.preset}-{self.config.seed}"
        index_of = {i.item_id: n for n, i in enumerate(self.scenario.items)}
        self.smoke_items = max(1, int(count))
        # RESET THE DISPATCH COUNTER, or a second run after a scene reset never
        # starts. The pump releases an EXECUTE_ITEM only when the resolved count
        # has caught up with the dispatched count, and a scene reset zeroes the
        # resolved counters — so a stale dispatched count from the previous run
        # holds the queue closed forever. Observed exactly that way: the rebuilt
        # scene reached SCENE_READY, RUN_BEGIN round-tripped, READY was
        # published, and then nothing was ever sent.
        self._smoke_dispatched = 0
        self._smoke_publisher = self.bridge.create_publisher(
            String, WT.ISAAC_COMMAND, qos_for(WT.ISAAC_COMMAND))

        self._smoke_queue = [IsaacCommand(
            command=IsaacCommandType.RUN_BEGIN, run_id=run_id,
            preset=self.config.preset, seed=self.config.seed,
            plan_id=plan.plan_id, total_items=len(self.scenario.items))]

        for index, placement in enumerate(plan.ordered_placements[:self.smoke_items]):
            item = self.scenario.item(placement.item_id)
            container = plan.container(placement.container_id)
            self._smoke_queue.append(IsaacCommand(
                command=IsaacCommandType.EXECUTE_ITEM,
                run_id=run_id, sequence_index=index, item_id=item.item_id,
                dimensions=dimensions_for(item),
                source_pose=table_pose_for_index(
                    index_of[item.item_id], item, self.layout),
                target_pose=placement_pose(placement),
                container_id=container.container_id,
                container_inner_mm=container.inner_size.to_dict(),
                plan_id=plan.plan_id, preset=self.config.preset,
                seed=self.config.seed,
                scenario_revision=int(self._scene_revision or 0)))
        self._smoke_queue.append(IsaacCommand(
            command=IsaacCommandType.RUN_END, run_id=run_id))
        print(f"{LOG_APP} smoke queue: {len(self._smoke_queue)} commands")

    def _pump_smoke_queue(self) -> None:
        """Publish the next queued command once the previous one has RESOLVED.

        "Free" is not the same as "finished", and conflating them was a real bug
        here: a command published on one frame has not yet been delivered back
        over DDS on the next, so `sequence.busy` is still False and
        `self.pending` is still None. The pump therefore raced ahead and
        published RUN_END while the item was still in flight, and the run ended
        with "0 completed, 0 failed" having executed nothing.
        (Measured: smoke queue of 3 commands, all published within three frames.)

        The gate is therefore the ACKNOWLEDGED count — how many items have
        reported a terminal state — not the simulator's momentary idleness.
        """
        if not self._smoke_queue or self._smoke_publisher is None:
            return
        if self.sequence.busy or self.pending is not None:
            return

        resolved = self.items_completed + self.items_failed
        head = self._smoke_queue[0]
        if head.command is IsaacCommandType.EXECUTE_ITEM:
            if not self.run_active:
                return          # RUN_BEGIN has not been round-tripped yet
            if resolved < self._smoke_dispatched:
                return          # the previous item is still being executed
        elif head.command is IsaacCommandType.RUN_END:
            if resolved < self._smoke_dispatched:
                return          # never end the run out from under an item

        from std_msgs.msg import String                        # noqa: PLC0415
        self._smoke_queue.pop(0)
        if head.command is IsaacCommandType.EXECUTE_ITEM:
            self._smoke_dispatched += 1
        self._smoke_publisher.publish(String(data=head.to_json()))

    def _run_reset_validation(self) -> int:
        """Execute an item, rebuild for a NEW scenario, verify, execute again.

        Prints RESET-VALIDATE markers so a shell script can assert on them
        without parsing Isaac's own log.
        """
        import numpy as _np                                      # noqa: PLC0415
        from wisepack_core.isaac_transform import (               # noqa: PLC0415
            pose_to_world as _to_world, table_pose_for_index as _table,
        )

        def pump(frames=1):
            for _ in range(frames):
                simulation_app.update()
                self.bridge.spin_once()
                self._pump_smoke_queue()
                if self.pending is not None and not self.sequence.busy:
                    command = self.pending
                    self.pending = None
                    self._last_command = command
                    refusal = self._pre_pick_refusal(command)
                    if refusal:
                        print(f"RESET-VALIDATE REFUSED {command.item_id} {refusal}")
                        self._on_item_done(command.item_id or "",
                                           _refused_outcome(command, refusal))
                    elif not self.sequence.begin(command):
                        self._on_item_done(command.item_id or "",
                                           _missing_item_outcome(command))
                self.sequence.step()

        # --- phase 1: one item of the ORIGINAL scenario --------------------
        first_preset, first_seed = self.config.preset, self.config.seed
        self.prepare_smoke_run(1)
        deadline = time.monotonic() + 240
        while (self.items_completed + self.items_failed) < 1 \
                and time.monotonic() < deadline:
            pump()
        print(f"RESET-VALIDATE PHASE1 completed={self.items_completed} "
              f"failed={self.items_failed}")

        placed = [i for i in self.scene.items
                  if self._is_in_container(i)]
        print(f"RESET-VALIDATE PHASE1-IN-CONTAINER {len(placed)}")

        # --- phase 2: rebuild for a NEW scenario ---------------------------
        new_seed = int(getattr(ARGS, "reset_seed", 1234))
        self._reset_scene(IsaacCommand(
            command=IsaacCommandType.RESET_SCENE, run_id="reset-validate-2",
            preset=first_preset, seed=new_seed, scenario_revision=2))
        pump(30)

        if self._scene_revision != 2:
            print("RESET-VALIDATE RESULT FAIL scene was not rebuilt")
            return 2

        # --- phase 3: verify the rebuilt world -----------------------------
        still_in_container = [i for i in self.scene.items
                              if self._is_in_container(i)]
        print(f"RESET-VALIDATE CONTAINER-CLEARED "
              f"{'yes' if not still_in_container else 'no ' + str(still_in_container)}")

        expected = self.scenario.items
        respawned, at_source = len(self.scene.items), 0
        for index, item in enumerate(expected):
            pose = self.scene.item_world_pose(item.item_id)
            if pose is None:
                continue
            want, _ = _to_world(_table(index, item, self.layout), self.layout)
            drift = float(_np.linalg.norm(
                _np.asarray(pose[0]) - _np.asarray(want)))
            if drift <= self.config.motion.max_source_drift_m:
                at_source += 1
        print(f"RESET-VALIDATE RESPAWNED {respawned}/{len(expected)} "
              f"at-source={at_source}/{len(expected)}")

        dof = self.robot.get_dof_positions().numpy()[0]
        home = _np.array([0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741])
        home_err = float(_np.max(_np.abs(dof[:7] - home)))
        print(f"RESET-VALIDATE ROBOT-HOME max_joint_error={home_err:.3f} rad")
        print(f"RESET-VALIDATE GRASP-JOINT "
              f"{'attached' if self.sequence.grasp.is_attached else 'released'}")

        # --- phase 4: first item of the NEW run ----------------------------
        self.items_completed = self.items_failed = 0
        self.prepare_smoke_run(1)
        deadline = time.monotonic() + 240
        while (self.items_completed + self.items_failed) < 1 \
                and time.monotonic() < deadline:
            pump()
        print(f"RESET-VALIDATE PHASE2 completed={self.items_completed} "
              f"failed={self.items_failed}")

        ok = (not still_in_container and respawned == len(expected)
              and at_source == len(expected) and home_err < 0.35
              and not self.sequence.grasp.is_attached
              and self.items_completed == 1)
        print(f"RESET-VALIDATE RESULT {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 2

    def _is_in_container(self, item_id: str) -> bool:
        """True when this item is sitting inside container CNT-01."""
        import numpy as _np                                      # noqa: PLC0415
        from wisepack_core.isaac_transform import mm_to_m as _mm  # noqa: PLC0415
        pose = self.scene.item_world_pose(item_id)
        if pose is None:
            return False
        origin = self.layout.container_origin_for("CNT-01")
        inner = self.scene.containers.get("CNT-01")
        if inner is None:
            return False
        p = _np.asarray(pose[0], dtype=float)
        return bool(origin[0] <= p[0] <= origin[0] + _mm(inner.x)
                    and origin[1] <= p[1] <= origin[1] + _mm(inner.y)
                    and p[2] >= origin[2] - 0.02)

    def _end_run(self, reason: str) -> None:
        if not self.run_active:
            return
        self.run_active = False
        self._run_ended = True
        self.bridge.publish(
            IsaacState.RUN_COMPLETED, self.run_id, message=reason,
            detail={"items_completed": self.items_completed,
                    "items_failed": self.items_failed})
        print(f"{LOG_APP} run finished: {self.items_completed} completed, "
              f"{self.items_failed} failed")

    def fail_run(self, reason: str) -> None:
        self.run_active = False
        self._run_ended = True
        self.bridge.publish(
            IsaacState.RUN_FAILED, self.run_id or "isaac-standby",
            message=reason,
            detail={"items_completed": self.items_completed,
                    "items_failed": self.items_failed})
        print(f"{LOG_APP} RUN FAILED: {reason}")

    # -- main loop -------------------------------------------------------- #

    def run(self) -> int:
        """Returns a process exit code."""
        app_utils.play()
        simulation_app.update()

        # PUT THE ARM IN A KNOWN POSE AND HOLD IT THERE. This is not cosmetic.
        #
        # `Franka.__init__` RECORDS a default state but never commands it, so at
        # play() the joint drives hold whatever the USD authored, and the arm
        # drifts to that configuration over the following seconds. The first
        # EXECUTE_ITEM may arrive one second later (the self-driving smoke test)
        # or a minute later (a real run, waiting for an operator to approve) —
        # and the arm is in a DIFFERENT place in each case.
        #
        # Measured exactly that way: the identical pre-grasp goal converged
        # comfortably when the command arrived immediately, and timed out at
        # both 360 and 900 frames when it arrived after the approval gate. That
        # reads as an unreachable target and is really an undefined starting
        # configuration. Commanding the canonical Panda ready pose once, before
        # anything can be dispatched, makes the sequence independent of when the
        # operator decides.
        self.robot.reset_to_default_pose()

        # Then let physics settle: the spawned items resolve small penetrations
        # with the table over the first frames, and grasping during that is how
        # an item gets flicked across the surface.
        for _ in range(120):
            simulation_app.update()

        if not self._ready_announced:
            self._announce_ready()
        print(f"{LOG_APP} READY — waiting for WISEPACK commands on the "
              "Isaac command topic")

        if getattr(ARGS, "self_test", False):
            # Spin briefly so the READY publication actually reaches the wire
            # before the process exits. Closing immediately after publishing on
            # a RELIABLE topic with no matched reader discards it, and the
            # validator would report "no READY" for a simulator that worked.
            for _ in range(120):
                simulation_app.update()
                self.bridge.spin_once()
            print(f"{LOG_APP} self-test complete — scene, GPU and ROS bridge OK")
            return 0

        if self.smoke_items:
            self.prepare_smoke_run(self.smoke_items)
        if getattr(ARGS, "reset_test", False):
            return self._run_reset_validation()

        while simulation_app.is_running():
            simulation_app.update()
            self.bridge.spin_once()
            self._pump_smoke_queue()

            if self.pending is not None and not self.sequence.busy:
                command = self.pending
                self.pending = None
                self._last_command = command
                # PHYSICAL SANITY BEFORE MOTION. Refuse rather than approach an
                # object that is not where the plan says it is.
                refusal = self._pre_pick_refusal(command)
                if refusal:
                    print(f"{LOG_APP} REFUSING to pick {command.item_id}: {refusal}")
                    self._on_item_done(command.item_id or "",
                                       _refused_outcome(command, refusal))
                elif not self.sequence.begin(command):
                    self._on_item_done(
                        command.item_id or "",
                        _missing_item_outcome(command))

            self.sequence.step()

            if (self.config.max_runtime_s > 0
                    and time.monotonic() - self._started_at
                    > self.config.max_runtime_s):
                self.fail_run(
                    f"exceeded the {self.config.max_runtime_s:.0f}s runtime "
                    "ceiling")
                return 3
            if self.config.exit_on_complete and self._run_ended:
                return 0 if self.items_failed == 0 else 2
        return 0


def _isaac_version() -> str:
    """The INSTALLED Isaac Sim version, read from the installation itself.

    The adapter knows which installation it launched, so reporting "—" in the
    Simulator View was withholding information it already had.
    """
    root = os.environ.get("ISAAC_SIM_ROOT") or os.environ.get("ISAAC_PATH", "")
    try:
        with open(os.path.join(root, "VERSION"), encoding="utf-8") as fh:
            return f"Isaac Sim {fh.read().strip()}"
    except OSError:
        return "Isaac Sim (version file not readable)"


def _refused_outcome(command: IsaacCommand, reason: str):
    """A pre-pick refusal, reported as ITEM_FAILED with the exact reason."""
    from simulators.isaac.result import PlacementOutcome                # noqa: PLC0415
    return PlacementOutcome(
        ok=False, reasons=[f"pre-pick check failed: {reason}"], notes=[],
        actual_pose=None, target_pose=command.target_pose,
        position_error_mm=None, axis_error_deg=None, settled=False,
        timed_out=False, detail={"pre_pick_refusal": reason})


def _missing_item_outcome(command: IsaacCommand):
    from simulators.isaac.result import PlacementOutcome           # noqa: PLC0415
    reason = (f"{command.item_id} is not in the Isaac scene; it was not part of "
              f"the preset/seed the scene was built from")
    return PlacementOutcome(
        ok=False, reasons=[reason], notes=[], actual_pose=None,
        target_pose=command.target_pose, position_error_mm=None,
        axis_error_deg=None, settled=False, timed_out=False,
        detail={"reason": reason})


def main() -> int:
    motion = MotionConfig.from_env()
    config = AppConfig(
        preset=ARGS.preset, seed=ARGS.seed, headless=bool(ARGS.headless),
        exit_on_complete=bool(ARGS.exit_on_complete or ARGS.self_test
                              or ARGS.smoke_test or ARGS.reset_test),
        max_runtime_s=float(ARGS.max_runtime),
        motion=motion, physics=PhysicsConfig.from_env())

    app = WisepackIsaacApp(config)
    app.smoke_items = int(getattr(ARGS, "smoke_test", 0) or 0)
    if getattr(ARGS, "reset_test", False):
        app.smoke_items = 1        # enables the SMOKE-* markers too
    try:
        app.build()
        return app.run()
    except KeyboardInterrupt:
        print(f"{LOG_APP} interrupted")
        return 130
    except Exception as exc:                            # noqa: BLE001
        # Reported, then re-raised. WISEPACK must learn that the physical backend
        # died — otherwise the orchestrator sits out its whole timeout waiting
        # for a process that is already gone — but the traceback must still reach
        # the operator's terminal rather than being swallowed into a tidy exit
        # code. Failing silently here is how "Isaac did nothing" becomes a
        # twenty-minute debugging session.
        carb.log_error(f"{LOG_APP} fatal: {exc!r}")
        if app.bridge is not None:
            app.fail_run(f"{type(exc).__name__}: {exc}")
            for _ in range(30):
                simulation_app.update()
                app.bridge.spin_once()
        raise
    finally:
        shutdown_ros(app.bridge)
        simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
