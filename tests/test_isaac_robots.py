"""Robot selection: the registry, the public API, and the run correlation.

NO ISAAC, NO GPU, NO SIMULATOR. Everything here runs in the ordinary suite,
which is the point: the robot registry is the one place a robot is described, so
it has to be falsifiable without the thing it describes.

The tests are grouped by the claim they defend:

    1. the registry parses, validates and refuses
    2. exactly ONE robot list exists — not one in Python and another in HTML
    3. selection precedence, and what a draft may and may not change
    4. the robot is part of run correlation and of the scene fingerprint
    5. an acknowledgement from one robot cannot authorise another's run
"""

from __future__ import annotations

import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in ("wisepack_core", "wisepack_bringup"):
    _path = os.path.join(REPO, "wisepack_ws", "src", _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)
sys.path.insert(0, os.path.join(REPO, "web"))

from wisepack_core.correlation import CORRELATION_FACETS, RunCorrelation
from wisepack_core.execution import (
    ExecutionBackend, physical_presets, preset_physical_compatibility,
)
from wisepack_core.generator import ISAAC_SMOKE_PRESET, build_scenario
from wisepack_core.isaac_contract import (
    IsaacCommand, IsaacCommandType, IsaacFeedback, IsaacState,
    SceneAcknowledgement,
)
from wisepack_core.isaac_transform import (
    DEFAULT_LAYOUT, layout_for_robot, scene_fingerprint,
)
from wisepack_core.robots import (
    KNOWN_KINEMATICS, KNOWN_SKILLS, ROBOT_ENV_VAR, RobotConfigError,
    load_registry, parse_registry, registry_path,
)

REGISTRY = os.path.join(REPO, "config", "isaac_robots.yaml")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture()
def registry():
    return load_registry(REGISTRY, reload=True)


@pytest.fixture()
def doc():
    """The registry as a plain dict, for mutation in the refusal tests."""
    import yaml
    with open(REGISTRY, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# 1. The registry
# --------------------------------------------------------------------------- #


def test_the_tracked_registry_exists_and_is_where_the_loader_looks():
    assert os.path.exists(REGISTRY), \
        "config/isaac_robots.yaml is the single tracked robot definition"
    assert os.path.abspath(registry_path()) == os.path.abspath(REGISTRY)


def test_both_supported_robots_are_configured(registry):
    assert set(registry.profiles) == {"panda", "xarm7"}
    for profile in registry.ordered:
        assert profile.enabled
        assert profile.implementation_status in ("validated", "experimental",
                                                 "planned")


def test_every_profile_is_completely_and_consistently_described(registry):
    for profile in registry.ordered:
        assert profile.display_name and profile.manufacturer and profile.model
        assert profile.arm_joint_names, "an arm needs joints"
        assert profile.gripper_joint_names, "a gripper needs at least one drive"
        assert len(profile.home_joint_positions) == len(profile.arm_joint_names)
        assert profile.end_effector_link and profile.end_effector_prim
        assert profile.tool_centre_point_m > 0
        assert profile.nominal_reach_m > 0
        assert profile.asset_path_candidates
        assert profile.kinematics in KNOWN_KINEMATICS
        assert set(profile.supported_skills) <= set(KNOWN_SKILLS)
        assert profile.dof == (len(profile.arm_joint_names)
                               + len(profile.gripper_joint_names)
                               + len(profile.gripper_mimic_joint_names))


def test_the_whole_first_iteration_skill_set_is_claimed_by_both(registry):
    """HOME .. VERIFY. A robot that cannot do all of it is not selectable."""
    for profile in registry.ordered:
        assert list(profile.supported_skills) == list(KNOWN_SKILLS)


def test_exactly_one_default_and_it_is_runnable(registry):
    assert registry.default_robot_id in registry.profiles
    assert registry.default().enabled
    assert registry.default_robot_id == registry.default().robot_id


def test_a_second_default_key_is_impossible_by_construction(doc):
    """YAML gives one `default_robot`; the point is that it is REQUIRED."""
    doc.pop("default_robot")
    with pytest.raises(RobotConfigError, match="default_robot"):
        parse_registry(doc, source_path="<test>")


def test_duplicate_robot_ids_are_rejected_not_last_wins(doc):
    doc["robots"].append(copy.deepcopy(doc["robots"][0]))
    with pytest.raises(RobotConfigError, match="duplicate robot id"):
        parse_registry(doc, source_path="<test>")


def test_an_unknown_selected_robot_is_rejected(registry):
    with pytest.raises(RobotConfigError, match="unknown robot"):
        registry.get("ur10e")


def test_a_disabled_robot_cannot_be_selected(doc):
    # A NON-DEFAULT robot: disabling the default is its own refusal, tested
    # separately, and would mask this one.
    victim = next(r for r in doc["robots"] if r["id"] != doc["default_robot"])
    victim["enabled"] = False
    reg = parse_registry(doc, source_path="<test>")
    assert victim["id"] not in reg.enabled_ids
    with pytest.raises(RobotConfigError, match="disabled"):
        reg.get(victim["id"])
    # ...and it is still LISTED, greyed rather than hidden, so an operator who
    # wonders where an arm went gets the answer.
    assert victim["id"] in reg.profiles
    assert any(r["id"] == victim["id"] and not r["enabled"]
               for r in reg.to_public_dict()["robots"])


def test_a_disabled_default_is_rejected_at_load(doc):
    doc["robots"] = [r for r in doc["robots"]]
    for entry in doc["robots"]:
        if entry["id"] == doc["default_robot"]:
            entry["enabled"] = False
    with pytest.raises(RobotConfigError, match="default"):
        parse_registry(doc, source_path="<test>")


def test_an_unknown_key_in_a_profile_is_rejected_not_ignored(doc):
    doc["robots"][0]["payload_kg"] = 3.5
    with pytest.raises(RobotConfigError, match="unknown keys"):
        parse_registry(doc, source_path="<test>")


def test_a_kinematics_implementation_that_does_not_exist_is_refused(doc):
    doc["robots"][0]["kinematics"] = "collision-aware-motion-planning"
    with pytest.raises(RobotConfigError, match="not implemented"):
        parse_registry(doc, source_path="<test>")


def test_no_profile_may_claim_motion_planning(registry):
    """A differential IK controller is not a motion planner. Stated, not implied."""
    for profile in registry.ordered:
        assert profile.is_motion_planner is False
        assert profile.to_public_dict()["motion_planning"] is False


def test_a_home_vector_of_the_wrong_length_is_refused(doc):
    doc["robots"][0]["home_joint_positions"] = [0.0, 0.0]
    with pytest.raises(RobotConfigError, match="home_joint_positions"):
        parse_registry(doc, source_path="<test>")


def test_a_gripper_that_never_moves_is_refused(doc):
    doc["robots"][0]["closed_gripper_positions"] = list(
        doc["robots"][0]["open_gripper_positions"])
    with pytest.raises(RobotConfigError, match="never visibly move"):
        parse_registry(doc, source_path="<test>")


def test_the_profile_revision_changes_when_the_profile_does(doc):
    before = parse_registry(doc, source_path="<test>").get("xarm7").revision
    doc_2 = copy.deepcopy(doc)
    for entry in doc_2["robots"]:
        if entry["id"] == "xarm7":
            entry["tool_centre_point_m"] = 0.20
    after = parse_registry(doc_2, source_path="<test>").get("xarm7").revision
    assert before != after, \
        "a changed tool-centre-point is a different machine to a validated plan"


# --------------------------------------------------------------------------- #
# 2. Exactly one robot list
# --------------------------------------------------------------------------- #


def test_the_public_api_payload_is_generated_from_the_registry(registry):
    public = registry.to_public_dict()
    assert public["default_robot"] == registry.default_robot_id
    assert [r["id"] for r in public["robots"]] == \
        [p.robot_id for p in registry.ordered]
    for entry, profile in zip(public["robots"], registry.ordered):
        assert entry["display_name"] == profile.display_name
        assert entry["dof"] == profile.dof
        assert entry["compatible_presets"] == list(profile.supported_presets)


def test_the_public_payload_leaks_no_asset_paths_or_prim_paths(registry):
    """Identity and capability only. Not this host's filesystem."""
    import json
    blob = json.dumps(registry.to_public_dict())
    for profile in registry.ordered:
        for candidate in profile.asset_path_candidates:
            assert candidate not in blob
        assert profile.root_prim_path not in blob
        assert profile.end_effector_prim not in blob
        for joint in profile.arm_joint_names:
            assert f'"{joint}"' not in blob
    assert "omniverse-content" not in blob and "/World/" not in blob


def test_no_robot_list_is_duplicated_in_the_html_or_the_javascript(registry):
    """THE RULE THIS WHOLE MODULE EXISTS FOR.

    Two lists that agree today disagree after the next edit, and the one the
    operator sees would be the stale one. The selector is built from
    /api/config/robots at runtime; nothing in the frontend names a robot.
    """
    for name in ("index.html", "simulator.html", "diagnostics.html",
                 "inventory.html"):
        html = _read(os.path.join(REPO, "web", name))
        for profile in registry.ordered:
            assert profile.display_name not in html, (
                f"{name} names {profile.display_name!r}; the selector must be "
                "built from GET /api/config/robots, not from markup")
            assert f'"{profile.robot_id}"' not in html
            assert f"'{profile.robot_id}'" not in html


def test_the_selector_markup_is_empty_and_filled_from_the_backend():
    html = _read(os.path.join(REPO, "web", "index.html"))
    assert '<select id="s-robot"></select>' in html, \
        "the Robot select must ship with no options"
    assert "function fillRobots(s)" in html
    assert "s.robots" in html


def test_the_config_endpoint_is_read_only_and_registry_backed():
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert '@app.get("/api/config/robots")' in app
    assert "load_registry().to_public_dict()" in app
    # No POST/PUT/PATCH/DELETE anywhere on the robot config path.
    for verb in ("post", "put", "patch", "delete"):
        assert f'@app.{verb}("/api/config/robots")' not in app


# --------------------------------------------------------------------------- #
# 3. Selection precedence and the draft/active split
# --------------------------------------------------------------------------- #


def test_an_explicit_value_beats_everything(registry):
    env = {ROBOT_ENV_VAR: "panda"}
    assert registry.resolve(explicit="xarm7", draft="panda",
                            env=env).robot_id == "xarm7"


def test_the_environment_override_beats_the_draft(registry):
    """The override exists for automation and must not be overruled by a draft."""
    env = {ROBOT_ENV_VAR: "xarm7"}
    assert registry.resolve(draft="panda", env=env).robot_id == "xarm7"


def test_the_draft_beats_the_configured_default(registry):
    other = next(p.robot_id for p in registry.ordered
                 if p.robot_id != registry.default_robot_id)
    assert registry.resolve(draft=other, env={}).robot_id == other


def test_nothing_selected_falls_back_to_the_configured_default(registry):
    assert registry.resolve(env={}).robot_id == registry.default_robot_id


def test_an_unknown_value_raises_rather_than_falling_through(registry):
    """A typo must not quietly select another arm."""
    with pytest.raises(RobotConfigError):
        registry.resolve(explicit="franka2", env={})
    with pytest.raises(RobotConfigError):
        registry.resolve(env={ROBOT_ENV_VAR: "nosuchrobot"})


def test_the_draft_is_a_separate_field_from_the_active_run():
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert '"robot_id": None,' in app, "the draft robot lives on STATE.settings"
    assert '"active_robot_id": active_robot_id,' in app
    assert '"draft_robot_id": draft_robot_id,' in app


def test_dashboard_polling_cannot_overwrite_a_chosen_robot():
    """The regression that made the preset dropdown feel broken, for robots."""
    app = _read(os.path.join(REPO, "web", "app.py"))
    block = app[app.index("physical = bool(execution.get"):
                app.index('draft_robot_id = settings.get("robot_id")')]
    assert "if not STATE.settings_touched" in block
    assert 'settings.get("robot_id") is None' in block, \
        "the robot draft is seeded only while it has not been chosen"

    html = _read(os.path.join(REPO, "web", "index.html"))
    seeding = html[html.index("if (!DRAFT_TOUCHED) {"):
                   html.index("// ROBOT SELECTION IS ONLY MEANINGFUL")]
    assert "rb.value = wantRobot" in seeding, \
        "the robot is seeded inside the untouched branch, like the preset"


def test_the_active_robot_changes_only_on_a_reset():
    orch = _read(os.path.join(
        REPO, "wisepack_ws", "src", "wisepack_orchestration",
        "wisepack_orchestration", "hitl_orchestrator.py"))
    reset = orch[orch.index("def _reset_run("):orch.index("def _write_artifacts(")]
    assert "self._resolve_robot(" in reset
    assert "robot_id=robot_id" in reset
    # ...and nowhere else. Only the constructor and the reset build a config.
    assert orch.count("robot_id=robot_id") == 1
    assert orch.count("WorkflowConfig(") == 2


def test_a_robot_cannot_change_while_an_item_is_being_carried():
    orch = _read(os.path.join(
        REPO, "wisepack_ws", "src", "wisepack_orchestration",
        "wisepack_orchestration", "hitl_orchestrator.py"))
    reset = orch[orch.index("def _reset_run("):orch.index("def _write_artifacts(")]
    assert "self.isaac.in_flight_item" in reset
    assert "cannot switch to" in reset


def test_the_logical_modes_expose_no_isaac_robot_selector():
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert '"robot_selector": physical,' in app
    assert '"execution_source_label": ("" if physical' in app
    assert "logical workflow simulator" in app.lower()

    html = _read(os.path.join(REPO, "web", "index.html"))
    assert "s.robot_selector" in html
    assert "Logical workflow simulator" in html
    # HIDDEN, not merely disabled. A greyed-out arm still asserts that an arm is
    # involved, which is the contradiction the operator reported.
    assert 'rb.style.display = "none"' in html
    assert 'rlabel.textContent = "Execution source"' in html
    assert 'rfixed.style.display = ""' in html


def test_the_logical_modes_report_no_robot_at_all():
    """`robot_id` absent rather than defaulted. A logical run has no robot."""
    app = _read(os.path.join(REPO, "web", "app.py"))
    block = app[app.index("physical = bool(execution.get"):
                app.index('draft_robot_id = settings.get("robot_id")')]
    assert 'settings = {**settings, "robot_id": None}' in block
    assert "active_robot, active_robot_id = None, None" in block

    html = _read(os.path.join(REPO, "web", "index.html"))
    # ...and the draft carried into a reset is null too, so no run is ever
    # stamped with the name of an arm that never moved.
    assert "STATE && STATE.robot_selector && $(\"#s-robot\")" in html


def test_changing_the_robot_requires_confirmation_before_a_reset():
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert '"robot_change_requires_reset"' in app
    html = _read(os.path.join(REPO, "web", "index.html"))
    block = html[html.index('$("#c-reset").onclick'):
                 html.index('$("#c-strategies").onclick')]
    assert "STATE.robot_change_requires_reset" in block
    assert "window.confirm" in block


# --------------------------------------------------------------------------- #
# 4. Preset compatibility
# --------------------------------------------------------------------------- #


def test_a_robot_that_does_not_support_the_preset_is_blocked(registry, doc):
    xarm = registry.get("xarm7")
    ok, reason = preset_physical_compatibility(ISAAC_SMOKE_PRESET, xarm)
    assert ok and reason == ""

    # Restrict the arm to something else and the same preset is refused. Done by
    # editing the profile rather than by naming a preset that happens to fail
    # the BACKEND bounds too — that would prove nothing about the robot bound,
    # which is the one under test.
    for entry in doc["robots"]:
        if entry["id"] == "xarm7":
            entry["supported_presets"] = ["some_other_preset"]
    restricted = parse_registry(doc, source_path="<test>").get("xarm7")
    blocked, why = preset_physical_compatibility(ISAAC_SMOKE_PRESET, restricted)
    assert blocked is False
    assert restricted.display_name in why and "not among them" in why


def test_the_incompatibility_reason_names_the_robot_not_just_the_preset(doc):
    """"Unavailable" alone sends the operator to look at the wrong control."""
    for entry in doc["robots"]:
        if entry["id"] == "xarm7":
            entry["supported_presets"] = ["some_other_preset"]
    restricted = parse_registry(doc, source_path="<test>").get("xarm7")
    reasons = physical_presets(restricted)
    assert restricted.display_name in reasons[ISAAC_SMOKE_PRESET]
    # A backend-level refusal is still phrased as one, without the robot: the
    # two causes need different remedies and must read differently.
    assert restricted.display_name not in reasons["mixed_pipes_dense"]


def test_the_backend_level_bounds_still_apply_without_a_robot():
    ok, reason = preset_physical_compatibility("mixed_pipes_dense")
    assert ok is False and reason


def test_the_profile_refusal_is_empty_for_a_supported_pair(registry):
    for profile in registry.ordered:
        assert profile.preset_refusal(ISAAC_SMOKE_PRESET) == ""
        assert profile.supports_preset(ISAAC_SMOKE_PRESET)


# --------------------------------------------------------------------------- #
# 5. Run correlation and the scene fingerprint
# --------------------------------------------------------------------------- #


def test_robot_id_is_a_run_correlation_facet():
    assert "robot_id" in CORRELATION_FACETS


def test_a_projection_from_another_robot_does_not_match_this_run():
    active = RunCorrelation(run_id="r1", robot_id="xarm7")
    stale = RunCorrelation(run_id="r1", robot_id="panda")
    assert not stale.matches(active)
    assert stale.mismatches(active)["robot_id"] == {
        "expected": "xarm7", "found": "panda"}


def test_a_projection_that_names_no_robot_is_unknown_not_wrong():
    """A simulated run makes no claim about a robot, and is not a mismatch."""
    active = RunCorrelation(run_id="r1", robot_id="xarm7")
    assert RunCorrelation(run_id="r1").matches(active)


def test_the_robot_is_part_of_the_scene_fingerprint():
    scenario = build_scenario(ISAAC_SMOKE_PRESET, seed=42)
    panda = scene_fingerprint(scenario, DEFAULT_LAYOUT, "panda")
    xarm = scene_fingerprint(scenario, DEFAULT_LAYOUT, "xarm7")
    assert panda != xarm, \
        "the same objects in front of a different arm is a different scene"


def test_the_fingerprint_defaults_to_the_layouts_own_robot():
    """A caller that already built the layout cannot forget to pass the id."""
    scenario = build_scenario(ISAAC_SMOKE_PRESET, seed=42)
    registry = load_registry(REGISTRY, reload=True)
    layout = layout_for_robot(registry.get("xarm7"))
    assert scene_fingerprint(scenario, layout) == \
        scene_fingerprint(scenario, layout, "xarm7")


def test_the_workcell_moves_for_a_shorter_arm(registry):
    """The xArm 7 cannot reach the Panda's bin corner. MEASURED, see the README."""
    xarm = layout_for_robot(registry.get("xarm7"))
    panda = layout_for_robot(registry.get("panda"))
    assert panda.container_outer_xy_m == DEFAULT_LAYOUT.container_outer_xy_m, \
        "the default layout IS the Panda layout; it declares no overrides"
    assert xarm.container_outer_xy_m != panda.container_outer_xy_m
    assert xarm.robot_max_reach_m < panda.robot_max_reach_m
    assert xarm.robot_id == "xarm7"


def test_both_robots_layouts_pass_their_own_reachability_check(registry):
    """The check that stops an unreachable goal being discovered by the arm."""
    scenario = build_scenario(ISAAC_SMOKE_PRESET, seed=42)
    inner = scenario.container_template.inner_size
    for profile in registry.ordered:
        layout = layout_for_robot(profile)
        layout.validate(inner, len(scenario.items), clearance_m=0.10)


def test_the_panda_layout_would_fail_the_xarm_envelope(registry):
    """Proof the override is load-bearing rather than decorative."""
    import dataclasses
    scenario = build_scenario(ISAAC_SMOKE_PRESET, seed=42)
    inner = scenario.container_template.inner_size
    forced = dataclasses.replace(
        DEFAULT_LAYOUT,
        robot_max_reach_m=registry.get("xarm7").workcell.robot_max_reach_m)
    with pytest.raises(ValueError, match="not reachable"):
        forced.validate(inner, len(scenario.items), clearance_m=0.10)


def test_the_camera_follows_the_selected_robots_workcell(registry):
    xarm = layout_for_robot(registry.get("xarm7"))
    panda = layout_for_robot(registry.get("panda"))
    assert xarm.camera_position_m != panda.camera_position_m
    scene = _read(os.path.join(REPO, "simulators", "isaac", "scene.py"))
    assert "self.layout.camera_position_m" in scene
    assert "1.55" not in scene, "the camera must not be hard-coded in the scene"


# --------------------------------------------------------------------------- #
# 6. One robot's acknowledgement cannot authorise another's run
# --------------------------------------------------------------------------- #


def test_a_panda_scene_ready_is_rejected_for_an_xarm_run():
    ack = SceneAcknowledgement(
        run_id="r1", scenario_id="s1", scenario_revision=2, preset="p", seed=42,
        robot_id="panda", robot_profile_revision="aaaa", scene_fingerprint="fp",
        object_ids=["item-001"], object_count=1,
        robot_home_verified=True, container_empty_verified=True)
    reasons = ack.mismatches(
        run_id="r1", scenario_id="s1", revision=2, preset="p", seed=42,
        fingerprint="fp", object_count=1,
        robot_id="xarm7", robot_profile_revision="bbbb")
    assert any("robot panda" in r for r in reasons)
    assert any("profile revision" in r for r in reasons)


def test_an_xarm_scene_ready_is_rejected_for_a_panda_run():
    ack = SceneAcknowledgement(run_id="r1", robot_id="xarm7",
                               robot_home_verified=True,
                               container_empty_verified=True)
    reasons = ack.mismatches(run_id="r1", scenario_id="", revision=0,
                             preset="", seed=0, fingerprint="",
                             object_count=0, robot_id="panda")
    assert any("robot xarm7" in r for r in reasons)


def test_a_matching_robot_produces_no_robot_reason():
    ack = SceneAcknowledgement(run_id="r1", robot_id="xarm7",
                               robot_profile_revision="abc",
                               robot_home_verified=True,
                               container_empty_verified=True)
    reasons = ack.mismatches(run_id="r1", scenario_id="", revision=0,
                             preset="", seed=0, fingerprint="", object_count=0,
                             robot_id="xarm7", robot_profile_revision="abc")
    assert reasons == []


def test_an_acknowledgement_with_no_robot_is_tolerated():
    """An older simulator does not send one; a rolling upgrade must still work."""
    ack = SceneAcknowledgement(run_id="r1", robot_home_verified=True,
                               container_empty_verified=True)
    reasons = ack.mismatches(run_id="r1", scenario_id="", revision=0,
                             preset="", seed=0, fingerprint="", object_count=0,
                             robot_id="xarm7", robot_profile_revision="abc")
    assert reasons == []


def test_the_robot_survives_a_command_round_trip():
    command = IsaacCommand(command=IsaacCommandType.RUN_BEGIN, run_id="r1",
                           robot_id="xarm7")
    assert IsaacCommand.from_json(command.to_json()).robot_id == "xarm7"


def test_the_robot_survives_a_feedback_round_trip():
    feedback = IsaacFeedback(state=IsaacState.READY, run_id="r1",
                             robot_id="xarm7")
    assert IsaacFeedback.from_json(feedback.to_json()).robot_id == "xarm7"


def test_stale_feedback_from_another_robot_is_dropped_by_the_bridge():
    bridge = _read(os.path.join(
        REPO, "wisepack_ws", "src", "wisepack_orchestration",
        "wisepack_orchestration", "isaac_bridge.py"))
    guard = bridge[bridge.index("STALE FEEDBACK FROM ANOTHER ROBOT"):
                   bridge.index("if state is IsaacState.ROBOT_MODEL_INVALID")]
    assert "feedback.robot_id != self.robot_id" in guard
    assert "return" in guard
    # ...and an EMPTY id is not a mismatch.
    assert "and feedback.robot_id" in guard


def test_the_simulator_refuses_a_command_addressed_to_another_robot():
    src = _read(os.path.join(REPO, "simulators", "isaac", "wisepack_isaac.py"))
    body = src[src.index("def _pre_pick_refusal("):src.index("def prepare_smoke_run(")]
    assert "command.robot_id != self.profile.robot_id" in body
    assert "model_valid" in body, \
        "a robot whose model did not validate may not pick either"


# --------------------------------------------------------------------------- #
# 7. Robot diagnostics and the degraded gate
# --------------------------------------------------------------------------- #


def test_a_model_validation_failure_degrades_the_backend_and_blocks_approval():
    bridge = _read(os.path.join(
        REPO, "wisepack_ws", "src", "wisepack_orchestration",
        "wisepack_orchestration", "isaac_bridge.py"))
    assert "def _on_robot_model_invalid(" in bridge
    assert "enter_degraded(" in bridge[bridge.index("_on_robot_model_invalid"):]
    # The scene gate closes on it, and the gate is what the approval check reads.
    ready = bridge[bridge.index("def scene_ready("):bridge.index("def scene_block_reason(")]
    assert "self.robot_model_error" in ready
    reason = bridge[bridge.index("def scene_block_reason("):
                    bridge.index("def rebind_robot(")]
    assert "robot_model_error" in reason


def test_the_simulator_reports_robot_model_invalid_and_exits_nonzero():
    src = _read(os.path.join(REPO, "simulators", "isaac", "wisepack_isaac.py"))
    body = src[src.index("def _robot_model_invalid("):src.index("def robot_status(")]
    assert "IsaacState.ROBOT_MODEL_INVALID" in body
    assert "return 5" in body
    assert "approval" in body


def test_the_state_is_mapped_onto_a_workflow_stage():
    """An unmapped state renders as whatever the previous stage happened to be."""
    from wisepack_core.events import Stage
    from wisepack_core.execution import ISAAC_STATE_STAGE, stage_for_isaac_state
    assert set(ISAAC_STATE_STAGE) == set(IsaacState)
    assert stage_for_isaac_state(IsaacState.ROBOT_MODEL_INVALID) is Stage.DEGRADED


def test_diagnostics_reports_expected_against_discovered_joints():
    diag = _read(os.path.join(REPO, "web", "diagnostics.py"))
    for row in ("configured_robots", "selected_robot", "active_robot",
                "robot_profile_revision", "robot_asset_resolved",
                "robot_articulation_valid", "robot_expected_arm_joints",
                "robot_discovered_arm_joints", "robot_end_effector_resolved",
                "robot_gripper_ready", "robot_home_verified",
                "robot_kinematics_ready", "robot_scene_ready",
                "last_robot_error"):
        assert f'"{row}"' in diag, f"Diagnostics must report {row}"


def test_the_adapter_reports_discovered_names_not_configured_ones():
    base = _read(os.path.join(REPO, "simulators", "isaac", "adapters", "base.py"))
    body = base[base.index("def get_diagnostics("):]
    assert "self._discovered_dof_names" in body
    assert "self._discovered_link_names" in body
    assert "never raises" in body.lower() or "Never raises" in base


# --------------------------------------------------------------------------- #
# 8. The Panda backend is unchanged where it must be
# --------------------------------------------------------------------------- #


def test_the_panda_profile_still_describes_the_arm_the_code_used_to_hardcode(registry):
    panda = registry.get("panda")
    assert panda.arm_joint_names == [f"panda_joint{n}" for n in range(1, 8)]
    assert panda.gripper_joint_names == ["panda_finger_joint1",
                                         "panda_finger_joint2"]
    assert panda.end_effector_link == "panda_hand"
    assert panda.tool_centre_point_m == pytest.approx(0.103)
    assert panda.home_joint_positions == pytest.approx(
        [0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741])
    assert panda.open_gripper_positions == [0.04, 0.04]
    assert panda.closed_gripper_positions == [0.0, 0.0]


def test_the_xarm_profile_matches_the_measured_asset(registry):
    xarm = registry.get("xarm7")
    assert xarm.arm_joint_names == [f"joint{n}" for n in range(1, 8)]
    # ONE driven joint; the other five are mimics and must not be commanded.
    assert xarm.gripper_joint_names == ["drive_joint"]
    assert len(xarm.gripper_mimic_joint_names) == 5
    assert xarm.end_effector_link == "xarm_gripper_base_link"
    assert xarm.tool_centre_point_m == pytest.approx(0.162, abs=0.002)
    assert xarm.articulation_root != xarm.root_prim_path, \
        "this asset's articulation root is <root>/root_joint, not the Xform"


def test_no_franka_helper_class_is_imported_any_more():
    """One kinematics implementation, not NVIDIA's for one arm and ours for the other."""
    for name in ("wisepack_isaac.py", "robot.py", "scene.py", "grasp.py"):
        src = _read(os.path.join(REPO, "simulators", "isaac", name))
        assert "examples.franka" not in src
        assert "import Franka" not in src


def _executable_source(path: str) -> str:
    """The file's CODE, with comments and docstrings removed.

    Prose may name a robot — the comments explaining why the tool-centre-point
    differs between the two arms are the most useful lines in those files. What
    must not name one is the CODE: a robot name in an expression is a branch on
    robot identity, and that is how two arms become two state machines.

    Done by re-emitting the parsed AST with every docstring dropped. Comments
    never survive parsing, so what is left is exactly the executable text.
    """
    import ast
    tree = ast.parse(_read(path))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


def test_the_sequence_holds_no_robot_specific_branch():
    """A robot-specific condition in a state machine is how two arms become two."""
    code = _executable_source(
        os.path.join(REPO, "simulators", "isaac", "robot.py")).lower()
    for token in ("panda", "franka", "xarm", "ufactory"):
        assert token not in code, \
            f"the placement sequence must not branch on {token!r}"


def test_the_scene_builder_no_longer_names_a_robot():
    path = os.path.join(REPO, "simulators", "isaac", "scene.py")
    assert "ROBOT_PATH" not in _read(path), \
        "the robot's prim path belongs to its profile, not to the stage layout"
    code = _executable_source(path).lower()
    for token in ("panda", "franka", "xarm", "ufactory"):
        assert token not in code


def test_the_grasp_welder_takes_the_hand_prim_rather_than_naming_one():
    code = _executable_source(
        os.path.join(REPO, "simulators", "isaac", "grasp.py")).lower()
    for token in ("panda", "franka", "xarm"):
        assert token not in code
    assert "hand_path" in code, "the adapter passes the resolved link in"


def test_the_execution_backend_badge_names_the_robot():
    assert ExecutionBackend.ISAAC.badge("UFACTORY xArm 7") == \
        "ISAAC SIM / UFACTORY XARM 7"
    assert ExecutionBackend.ISAAC.badge("") == "ISAAC SIM / PHYSICS"
    # The simulated backend has no robot and its badge never gains one.
    assert ExecutionBackend.SIMULATED.badge("UFACTORY xArm 7") == \
        "SIMULATED EXECUTION"


def test_the_backend_detail_no_longer_claims_a_specific_arm():
    detail = ExecutionBackend.ISAAC.detail.lower()
    assert "panda" not in detail and "franka" not in detail
