"""The physical scene-reset handshake — the safety defect, pinned.

THE DEFECT. Generating a new software scenario did not reset the physical
backend. The previous run's cylinders were still lying in the container while
the new plan assumed every one of them was back at its source pose, so the arm
was dispatched after objects that were not there. That is uncontrolled motion,
not a failed pick.

Nothing here needs Isaac Sim, a GPU or a browser: the handshake is a contract
plus a gate, and both are testable in plain Python. The physics of the rebuild
itself is exercised by the bounded live validation described in the README.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading

import pytest

from wisepack_core.execution import (
    PHYSICAL_MAX_DIAMETER_MM, PHYSICAL_MAX_ITEMS, physical_presets,
    preset_physical_compatibility,
)
from wisepack_core.isaac_contract import (
    RESET_STATES, IsaacCommand, IsaacCommandType, IsaacFeedback, IsaacState,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #

def test_the_reset_command_and_states_exist():
    assert IsaacCommandType.RESET_SCENE.value == "RESET_SCENE"
    for name in ("RESET_REQUESTED", "RESETTING", "SCENE_READY", "RESET_FAILED"):
        assert hasattr(IsaacState, name), f"missing state {name}"
    assert RESET_STATES == {IsaacState.RESET_REQUESTED, IsaacState.RESETTING,
                            IsaacState.SCENE_READY, IsaacState.RESET_FAILED}


def test_reset_states_do_not_move_the_workflow_stage():
    """They are simulator lifecycle, like READY — the gate is separate."""
    from wisepack_core.execution import stage_for_isaac_state
    for state in RESET_STATES:
        assert stage_for_isaac_state(state) is None


def test_the_scenario_revision_travels_on_both_messages():
    cmd = IsaacCommand.from_json(IsaacCommand(
        command=IsaacCommandType.RESET_SCENE, run_id="r",
        scenario_revision=7).to_json())
    assert cmd.scenario_revision == 7
    fb = IsaacFeedback.from_json(IsaacFeedback(
        state=IsaacState.SCENE_READY, run_id="r", scenario_revision=7).to_json())
    assert fb.scenario_revision == 7


def test_a_reset_command_needs_no_item():
    """It is about the SCENE, not about a placement."""
    cmd = IsaacCommand(command=IsaacCommandType.RESET_SCENE, run_id="r")
    assert cmd.item_id is None


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #

class _FakeNode:
    def __init__(self):
        self.published = []
        self._logger = self
    def create_publisher(self, *a, **k):
        node = self
        class P:
            def publish(self, msg): node.published.append(msg.data)
        return P()
    def create_subscription(self, *a, **k): return None
    def get_logger(self): return self
    def info(self, *a): pass
    def warn(self, *a): pass
    def error(self, *a): pass
    def publish_execution(self): pass


class _FakeEngine:
    def __init__(self):
        self.run_id = "run-1"
        self.scenario_revision = 1
        self.scenario = None
        self.selected = None
        self.finished = False
        self.degraded = ""
        self.notes = []
        class C:
            preset, seed = "isaac_cylinders_smoke", 42
        self.config = C()
    def note_physical_progress(self, *a, **k): self.notes.append(a)
    def enter_degraded(self, reason): self.degraded = reason; self.finished = True


def _bridge():
    """Import the orchestrator bridge with a stub std_msgs/rclpy layer."""
    import types
    for name, attrs in (("std_msgs", {}), ("std_msgs.msg", {"String": type(
            "String", (), {"__init__": lambda s, data="": setattr(s, "data", data)})})):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod
    src = os.path.join(REPO, "wisepack_ws", "src")
    for pkg in ("wisepack_bringup", "wisepack_orchestration"):
        path = os.path.join(src, pkg)
        if path not in sys.path:
            sys.path.insert(0, path)
    # wisepack_bringup.qos needs rclpy; the bridge only uses qos_for, so stub it.
    if "rclpy" not in sys.modules:
        pytest.skip("rclpy is not importable here")
    spec = importlib.util.spec_from_file_location(
        "wp_isaac_bridge",
        os.path.join(src, "wisepack_orchestration", "wisepack_orchestration",
                     "isaac_bridge.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["wp_isaac_bridge"] = module
    spec.loader.exec_module(module)
    return module


def test_the_bridge_source_gates_every_dispatch_on_the_scene():
    """Asserted on the source: the gate must sit BEFORE any dispatch."""
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "isaac_bridge.py"))
    tick = src[src.index("    def tick(self, engine)"):src.index("    def _await_simulator")]
    assert "if not self.scene_ready:" in tick
    assert tick.index("if not self.scene_ready:") < tick.index("_dispatch_next"), \
        "the scene gate must precede dispatch"


def test_scene_ready_requires_an_exact_revision_match():
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "isaac_bridge.py"))
    assert "self.scene_revision == self.required_revision" in src, \
        "a >= comparison would let an older SCENE_READY authorise a newer scenario"
    # The acknowledgement is checked field by field and REJECTED by name; the
    # rejection sets a mismatch the gate reads, so an unmatched SCENE_READY can
    # never fall through to opening it.
    handler = src[src.index("def _on_reset_state"):]
    handler = handler[:handler.index("\n    def ", 10)]
    assert "rejecting SCENE_READY" in handler
    assert handler.index("self.scene_mismatch =") < handler.index(
        "self.scene_revision = self.required_revision")
    assert "not self.scene_mismatch" in src, \
        "a rejected acknowledgement must keep the gate closed"


def test_a_new_scenario_requests_a_physical_rebuild():
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "hitl_orchestrator.py"))
    reset = src[src.index("def _reset_run"):src.index("def _write_artifacts")]
    assert "request_scene_reset" in reset, \
        "generating a new scenario must ask the backend to rebuild"
    assert "abort_run" in reset


def test_approval_is_refused_until_the_scene_is_ready():
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "hitl_orchestrator.py"))
    assert "_scene_refusal" in src
    # Both the topic path and the command path.
    assert src.count("_scene_refusal()") >= 2
    assert "cannot approve:" in src


def test_the_reset_timeout_holds_rather_than_continuing():
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                             "wisepack_orchestration", "isaac_bridge.py"))
    wait = src[src.index("def _await_scene"):src.index("def _check_item_timeout")]
    assert "reset_timeout_s" in wait
    assert "enter_degraded" in wait
    assert "rather than run against a stale scene" in wait


def test_the_dashboard_blocks_approval_on_the_scene_gate():
    spec = importlib.util.spec_from_file_location(
        "wp_snap_reset", os.path.join(REPO, "web", "snapshot.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["wp_snap_reset"] = m
    spec.loader.exec_module(m)

    snap = m.DashboardSnapshot(mode="ros")
    snap.control_stage = "WAIT_FOR_OPERATOR_APPROVAL"
    snap.control_plan_id = "plan-1"
    snap.control_approval_state = "pending"
    snap.control_plan_valid = True
    assert snap.can_approve is True                      # baseline

    snap.control_scene_ready = False
    snap.control_scene_reason = "the physical scene has not been rebuilt"
    assert snap.can_approve is False
    assert "rebuilt" in snap.approval_block_reason

    # The simulated backend has no scene concept and must never be blocked.
    snap.control_scene_ready = None
    snap.control_scene_reason = ""
    assert snap.can_approve is True


# --------------------------------------------------------------------------- #
# Isaac adapter: reset + pre-pick checks (source-level; no Isaac import)
# --------------------------------------------------------------------------- #

ISAAC_APP = os.path.join(REPO, "simulators", "isaac", "wisepack_isaac.py")


def test_the_reset_clears_everything_that_could_carry_over():
    src = _read(ISAAC_APP)
    body = src[src.index("def _reset_scene"):src.index("def _pre_pick_refusal")]
    for step, why in [
        ('self.sequence.abort(', "must stop active robot motion"),
        ("open_gripper()", "must release the gripper"),
        ("reset_to_default_pose()", "must return the Panda to a safe home pose"),
        ("gate.adopt(", "must cancel the previous run so late feedback is rejected"),
        ("reset_items(", "must remove the previous cylinders"),
        ("settle_items(", "must clear velocities and let contacts settle"),
        ("build_scenario(", "must recreate objects from the NEW preset and seed"),
        ("SCENE_READY", "must report readiness for the new revision"),
    ]:
        assert step in body, f"reset {why}"


def test_the_reset_drops_the_grasp_joint_before_deleting_bodies():
    """Order matters: abort() detaches the weld before any body is removed."""
    src = _read(ISAAC_APP)
    body = src[src.index("def _reset_scene"):src.index("def _pre_pick_refusal")]
    assert body.index("self.sequence.abort(") < body.index("reset_items(")


def test_removing_the_items_is_what_empties_the_container():
    scene = _read(os.path.join(REPO, "simulators", "isaac", "scene.py"))
    reset = scene[scene.index("def reset_items"):scene.index("def settle_items")]
    assert "RemovePrim" in reset, "previous items must be REMOVED, not repositioned"
    assert "container" in reset.lower()


def test_reset_failure_is_reported_not_swallowed():
    src = _read(ISAAC_APP)
    body = src[src.index("def _reset_scene"):src.index("def _pre_pick_refusal")]
    assert "RESET_FAILED" in body
    assert "carb.log_error" in body


@pytest.mark.parametrize("check,why", [
    ("does not exist in the current scene", "requested object must exist"),
    ("scenario revision", "object must belong to the current run"),
    ("expected source", "object must be near its expected source pose"),
    ("ALREADY inside", "object must not already be in the destination container"),
    ("grasp joint", "no stale fixed joint may remain"),
])
def test_every_required_pre_pick_check_is_present(check, why):
    src = _read(ISAAC_APP)
    body = src[src.index("def _pre_pick_refusal"):src.index("# -- self-driving")]
    assert check in body, f"missing pre-pick check: {why}"


def test_a_failed_pre_pick_check_stops_instead_of_approaching():
    src = _read(ISAAC_APP)
    assert "_pre_pick_refusal(command)" in src
    assert "REFUSING to pick" in src
    assert "_refused_outcome(" in src
    # The refusal path must NOT start the motion sequence.
    dispatch = src[src.index("refusal = self._pre_pick_refusal"):
                   src.index("self.sequence.step()")]
    assert dispatch.index("_refused_outcome(") < dispatch.index("sequence.begin(")


def test_a_reset_is_accepted_even_when_the_run_gate_would_reject_it():
    """A reset is how a NEW run becomes legitimate, so gating it on the OLD
    run's id would make the scene un-resettable exactly when it must be reset."""
    src = _read(ISAAC_APP)
    handler = src[src.index("def _on_command"):src.index("def _begin_run")]
    assert handler.index("RESET_SCENE") < handler.index("reject_reason")


def _reset_scene_source() -> str:
    src = _read(os.path.join(REPO, "simulators", "isaac", "wisepack_isaac.py"))
    start = src.index("    def _reset_scene(")
    return src[start:src.index("\n    def ", start + 10)]


def test_the_stage_is_mutated_only_while_the_timeline_is_stopped():
    """Caught by the LIVE run, and it is THE safety failure of this feature.

    Deleting a rigid body while physics is playing invalidates the PhysX tensor
    simulation view for the WHOLE stage — the Franka articulation included. The
    first live attempt rebuilt the items perfectly, published SCENE_READY, and
    then could not read the arm's joints at all:

        "Simulation view object is invalidated and cannot be used again to
         call getDofPositions"

    A scene that reports ready while the robot is unusable is exactly what the
    handshake exists to prevent, so the ordering below is load-bearing:
    stop -> mutate -> play.
    """
    body = _reset_scene_source()
    stop, mutate, play = (body.index("app_utils.stop()"),
                          body.index("self.scene.reset_items("),
                          body.index("app_utils.play()"))
    assert stop < mutate, "the stage is mutated before the timeline is stopped"
    assert mutate < play, "physics restarts before the new items exist"


def test_scene_ready_is_published_only_after_the_world_is_proven_usable():
    body = _reset_scene_source()
    assert (body.index("self._verify_scene_usable()")
            < body.index("IsaacState.SCENE_READY"))


def test_the_usability_check_reads_the_arm_and_every_item():
    src = _read(os.path.join(REPO, "simulators", "isaac", "wisepack_isaac.py"))
    start = src.index("    def _verify_scene_usable(")
    body = src[start:src.index("\n    def ", start + 10)]
    # Reading the joints is the only thing that proves the articulation view
    # survived; asking the wrapper whether it is valid does not.
    assert "get_dof_positions()" in body
    assert "item_world_pose(" in body
    assert "grasp.is_attached" in body
    assert body.count("raise") >= 3


def test_the_self_driving_dispatch_counter_is_reset_for_a_new_run():
    """Caught by the LIVE run: the rebuilt scene reached SCENE_READY, RUN_BEGIN
    round-tripped, READY was published — and no item was ever dispatched.

    The pump releases an EXECUTE_ITEM only once the resolved count catches up
    with the dispatched count, and a reset zeroes the resolved counters. A
    stale dispatched count therefore holds the queue shut forever.
    """
    src = _read(os.path.join(REPO, "simulators", "isaac", "wisepack_isaac.py"))
    start = src.index("    def prepare_smoke_run(")
    body = src[start:src.index("\n    def ", start + 10)]
    assert "self._smoke_dispatched = 0" in body


def test_the_isaac_bridge_publish_accepts_every_feedback_field_used():
    """Caught by the LIVE run: publish() had no scenario_revision parameter, so
    the first RESET_SCENE raised TypeError and failed the whole run.

    Checks the adapter's publisher against the fields the reset path passes.
    """
    src = _read(os.path.join(REPO, "simulators", "isaac", "bridge.py"))
    signature = src[src.index("    def publish("):src.index("\"\"\"Publish one")]
    for field in ("state", "run_id", "item_id", "sequence_index",
                  "container_id", "scenario_revision", "dimensions",
                  "source_pose", "target_pose", "actual_pose",
                  "position_error_mm", "message", "detail"):
        assert field in signature, f"publish() cannot carry {field!r}"


# --------------------------------------------------------------------------- #
# Preset compatibility
# --------------------------------------------------------------------------- #

def test_the_isaac_smoke_preset_is_the_only_physically_compatible_one():
    compat = physical_presets()
    assert compat["isaac_cylinders_smoke"] == ""


@pytest.mark.parametrize("preset", ["mixed_pipes_dense", "mixed_pipes_small",
                                    "segregated_materials", "mixed_geometries"])
def test_large_benchmarks_are_unavailable_for_physical_execution(preset):
    ok, reason = preset_physical_compatibility(preset)
    assert ok is False
    assert reason, "an unavailable preset must say why"


def test_the_reason_names_the_actual_limit():
    ok, reason = preset_physical_compatibility("mixed_pipes_dense")
    assert str(PHYSICAL_MAX_ITEMS) in reason or str(PHYSICAL_MAX_DIAMETER_MM) in reason


def test_the_dashboard_marks_incompatible_presets_only_in_isaac_mode():
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert '"preset_compatibility": (physical_presets()' in app
    assert 'if snap.execution_backend == "isaac" else {}' in app


def test_the_frontend_disables_them_with_the_reason_rather_than_hiding_them():
    html = _read(os.path.join(REPO, "web", "index.html"))
    block = html[html.index("const compat = s.preset_compatibility"):
                 html.index("const active = (s.active_scenario")]
    assert "opt.disabled = Boolean(why)" in block
    assert "unavailable" in block
    assert "opt.title = why" in block, "the reason must be discoverable"


# --------------------------------------------------------------------------- #
# Viewing modes and Simulator View wording
# --------------------------------------------------------------------------- #

LAUNCHER = os.path.join(REPO, "scripts", "run_wisepack_isaac.sh")


def _code(path: str) -> str:
    return "\n".join(l for l in _read(path).splitlines()
                     if not l.lstrip().startswith("#"))


@pytest.mark.parametrize("mode", ["desktop", "webrtc", "none"])
def test_every_view_mode_is_handled(mode):
    assert f"    {mode})" in _code(LAUNCHER)


def test_desktop_mode_refuses_without_a_display():
    code = _code(LAUNCHER)
    assert "view mode 'desktop' needs a display" in code
    assert "exit 8" in code


def test_webrtc_mode_implies_headless_and_streaming():
    code = _code(LAUNCHER)
    block = code[code.index("    webrtc)"):code.index("    none)")]
    assert "HEADLESS=1" in block
    assert "WISEPACK_ISAAC_STREAMING=1" in block


def test_the_launcher_prints_the_resolved_configuration():
    code = _code(LAUNCHER)
    for field in ("view mode", "headless", "DISPLAY", "streaming",
                  "signalling", "media", "watch it"):
        assert field in code, f"the startup summary omits {field!r}"


def test_the_startup_summary_never_prints_the_ssh_port():
    code = _code(LAUNCHER)
    summary = code[code.index('echo "$LOG  view mode'):code.index('exec "$ISAAC_ROOT')]
    assert "WISEPACK_SSH_PORT" not in summary


def test_the_readme_documents_all_four_viewing_choices():
    readme = _read(os.path.join(REPO, "README.md"))
    assert "Choose how to watch Isaac" in readme
    for mode in ("WISEPACK_ISAAC_VIEW_MODE=desktop",
                 "WISEPACK_ISAAC_VIEW_MODE=webrtc",
                 "WISEPACK_ISAAC_VIEW_MODE=none"):
        assert mode in readme
    # C: GUI + WebRTC — must give a definite answer either way.
    assert "Not supported here" in readme
    assert "SSH TCP tunnel alone" in readme
    assert "cannot carry it" in readme


SIM = os.path.join(REPO, "web", "simulator.html")


def test_the_webrtc_view_does_not_promise_browser_video():
    html = _read(SIM)
    assert "NVIDIA Isaac Sim WebRTC Streaming Client" in html
    assert "This browser cannot display the stream" in html
    # The contradictory instruction is gone from the non-embeddable path.
    assert 'd.transport === "webrtc" && !d.embeddable' in html


def test_unusable_controls_are_hidden_rather_than_offered():
    html = _read(SIM)
    assert "connect.hidden = !browserCanShow" in html
    assert "full.hidden = !browserCanShow" in html
    assert "copy.hidden = !d.viewer_url" in html, "Copy endpoint must be retained"


def test_the_ports_are_shown_separately():
    html = _read(SIM)
    assert "signalling port" in html and "media port" in html
    assert "/TCP" in html and "/UDP" in html


def test_service_ready_client_connected_and_frames_are_distinguished():
    html = _read(SIM)
    for row in ("service ready", "native client connected",
                "rendered frames verified"):
        assert row in html, f"missing the {row!r} row"
    # A listening port must never be labelled "Connected".
    assert "d.client_connected === true" in html


def test_the_desktop_view_explains_where_to_look():
    html = _read(SIM)
    assert "Isaac GUI is running on the host desktop" in html
    assert "NoMachine or Sunshine/Moonlight" in html


def test_the_simulator_version_is_reported_rather_than_dashed():
    src = _read(ISAAC_APP)
    assert "def _isaac_version" in src
    assert '"simulator_version": _isaac_version()' in src
    bridge = _read(os.path.join(REPO, "wisepack_ws", "src",
                                "wisepack_orchestration", "wisepack_orchestration",
                                "isaac_bridge.py"))
    assert "simulator_version" in bridge
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert '"simulator_version"' in app


def test_the_descriptor_reports_ports_and_connection_facts():
    from wisepack_core.visualization import VisualizationDescriptor
    from simulators.isaac.streaming import StreamingConfig, describe
    d = describe(StreamingConfig(enabled=True))
    assert d.signal_port == 49100 and d.media_port == 47998
    assert d.client_connected is None, "must be unknown, not a false claim"
    assert d.frames_verified is False
    assert VisualizationDescriptor.from_dict(d.to_dict()).media_port == 47998
