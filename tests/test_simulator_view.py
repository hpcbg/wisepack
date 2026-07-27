"""Simulator View, backend-neutral visualization, navigation and host settings.

NONE of this needs Isaac Sim, a GPU, NVENC, WebRTC, a browser, NoMachine or
Sunshine. That is the point: visualization is an optional capability of an
optional backend, and the dashboard's behaviour when it is absent is exactly
what most needs testing — an operator must never be shown an empty player or a
permanent spinner.

The live parts that genuinely need hardware (an actual WebRTC client receiving
rendered frames) are NOT asserted here and are not claimed anywhere else either.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys

import pytest

from wisepack_core.visualization import (
    STATUS_LABEL, VisualizationDescriptor, VisualizationStatus,
    VisualizationTransport, unavailable,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD = os.path.join(REPO, "run_wisepack_dashboard.sh")
ISAAC_LAUNCHER = os.path.join(REPO, "scripts", "run_wisepack_isaac.sh")
LOCAL_ENV_LIB = os.path.join(REPO, "scripts", "lib_local_env.sh")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _code(path: str) -> str:
    """Script text without comment lines — these scripts quote what they avoid."""
    return "\n".join(line for line in _read(path).splitlines()
                     if not line.lstrip().startswith("#"))


# --------------------------------------------------------------------------- #
# The descriptor
# --------------------------------------------------------------------------- #

def test_every_status_has_operator_wording():
    """No status may render as a bare enum value or an empty box."""
    for status in VisualizationStatus:
        assert status in STATUS_LABEL
        assert STATUS_LABEL[status].strip()


def test_the_documented_states_are_all_present():
    labels = set(STATUS_LABEL.values())
    for wording in ("Stream unavailable", "Stream starting", "Ready to connect",
                    "Connected", "Disconnected"):
        assert wording in labels, f"missing the {wording!r} state"


def test_the_three_required_transports_exist():
    values = {t.value for t in VisualizationTransport}
    assert {"desktop", "webrtc", "none"} <= values


def test_a_none_transport_can_never_report_itself_available():
    """Contradictory input is resolved, not passed through to a connect button
    that cannot possibly work."""
    d = VisualizationDescriptor(backend="isaac", available=True,
                                transport=VisualizationTransport.NONE,
                                status=VisualizationStatus.READY)
    assert d.available is False
    assert d.status is VisualizationStatus.UNAVAILABLE


def test_unavailable_always_carries_a_reason():
    d = unavailable("simulated", "the simulated backend has no renderer")
    assert d.available is False
    assert d.transport is VisualizationTransport.NONE
    assert d.message
    assert d.status_label == "Stream unavailable"


def test_descriptor_round_trips():
    original = VisualizationDescriptor(
        backend="isaac", available=True,
        transport=VisualizationTransport.WEBRTC,
        status=VisualizationStatus.READY,
        viewer_url="http://127.0.0.1:49100", stream_id="s1",
        camera_name="/World/DemoCamera", interactive=True, embeddable=False,
        message="up", client_hint="use the native client")
    assert VisualizationDescriptor.from_dict(original.to_dict()).to_dict() \
        == original.to_dict()


@pytest.mark.parametrize("garbage", [None, "nonsense", 42, [], {"transport": "smoke"}])
def test_unparseable_descriptors_degrade_instead_of_raising(garbage):
    """A malformed descriptor must not take out the dashboard poll loop."""
    d = VisualizationDescriptor.from_dict(garbage)
    assert d.available is False
    assert d.status_label == "Stream unavailable"
    assert d.message


def test_an_unknown_status_becomes_an_explicit_error():
    d = VisualizationDescriptor.from_dict(
        {"backend": "isaac", "transport": "webrtc", "status": "banana"})
    assert d.status is VisualizationStatus.ERROR


# --------------------------------------------------------------------------- #
# The Isaac adapter's descriptors (no Isaac import required)
# --------------------------------------------------------------------------- #

from simulators.isaac import streaming as isaac_streaming      # noqa: E402


def test_streaming_disabled_reports_unavailable_with_a_reason():
    d = isaac_streaming.describe(isaac_streaming.StreamingConfig(enabled=False))
    assert d.available is False
    assert d.transport is VisualizationTransport.NONE
    assert "WISEPACK_ISAAC_STREAMING" in d.message


def test_streaming_enabled_reports_webrtc_and_a_loopback_endpoint():
    d = isaac_streaming.describe(isaac_streaming.StreamingConfig(enabled=True))
    assert d.transport is VisualizationTransport.WEBRTC
    assert d.available is True
    assert d.viewer_url.startswith("http://127.0.0.1:"), \
        "the default endpoint must be loopback — the stream is unauthenticated"
    assert d.camera_name == isaac_streaming.SPECTATOR_CAMERA


def test_the_stream_is_not_advertised_as_embeddable():
    """Isaac Sim 6.0.1 ships no in-browser client, so an iframe renders blank."""
    d = isaac_streaming.describe(isaac_streaming.StreamingConfig(enabled=True))
    assert d.embeddable is False
    assert d.client_hint, "an operator must be told which client to use"


def test_the_desktop_transport_is_offered_for_externally_managed_viewers():
    d = isaac_streaming.desktop_descriptor(":0")
    assert d.transport is VisualizationTransport.DESKTOP
    assert d.available is True
    # WISEPACK must never claim to manage those services.
    assert "does not manage" in d.client_hint


def test_an_explicit_stream_url_overrides_the_derived_one():
    cfg = isaac_streaming.StreamingConfig(
        enabled=True, viewer_url="https://proxy.example/isaac")
    assert cfg.resolved_viewer_url() == "https://proxy.example/isaac"


def test_streaming_config_rejects_impossible_ports():
    with pytest.raises(ValueError):
        isaac_streaming.StreamingConfig(signal_port=0).validate()
    with pytest.raises(ValueError, match="must differ"):
        isaac_streaming.StreamingConfig(signal_port=5000,
                                        stream_port=5000).validate()


def test_the_required_extensions_are_the_installed_6_0_1_names():
    """Older releases used omni.services.livestream.webrtc, which is NOT in
    this install; writing against it would fail at runtime, not at import."""
    assert isaac_streaming.REQUIRED_EXTENSIONS == (
        "omni.kit.livestream.app", "omni.kit.livestream.webrtc")
    assert isaac_streaming.PRIMARY_STREAM_SETTING == \
        "/exts/omni.kit.livestream.app/primaryStream"


def test_missing_extension_detection():
    assert isaac_streaming.missing_extensions(lambda n: True) == []
    assert isaac_streaming.missing_extensions(lambda n: False) == \
        list(isaac_streaming.REQUIRED_EXTENSIONS)


def test_port_probe_detects_an_occupied_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        taken = s.getsockname()[1]
        assert isaac_streaming.port_is_free(taken, host="127.0.0.1") is False


# --------------------------------------------------------------------------- #
# Dashboard read model
# --------------------------------------------------------------------------- #

def _snapshot_module():
    name = "wp_snapshot_vis"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, "web", "snapshot.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _snap(**kw):
    snapshot = _snapshot_module()
    return snapshot.DashboardSnapshot(mode=kw.pop("mode", "ros"), **kw)


def test_the_simulated_backend_reports_no_stream_with_a_reason():
    snap = _snap(execution_backend="simulated")
    doc = snap.to_visualization()
    assert doc["transport"] == "none"
    assert doc["available"] is False
    assert "no renderer" in doc["message"]


def test_isaac_without_a_descriptor_yet_says_it_is_waiting():
    """Not an error, and not a blank panel: the simulator has not reported."""
    doc = _snap(execution_backend="isaac").to_visualization()
    assert doc["available"] is False
    assert "waiting" in doc["message"]


def test_a_published_descriptor_is_passed_through():
    published = isaac_streaming.describe(
        isaac_streaming.StreamingConfig(enabled=True)).to_dict()
    doc = _snap(execution_backend="isaac", visualization=published).to_visualization()
    assert doc["transport"] == "webrtc"
    assert doc["status_label"] == "Ready to connect"
    assert doc["viewer_url"] == published["viewer_url"]


def test_the_ros_provider_picks_the_descriptor_off_the_backend_topic():
    snapshot = _snapshot_module()
    import threading

    class S:
        def __init__(self):
            self.lock = threading.RLock(); self.engine = None; self.events = []
            self.notice = ""; self.auto_step = False
            self.fiware_connected = None; self.fiware_last_error = ""
            self.ros_mirror = {
                "stage": "PICK_ITEM",
                "execution_backend": {
                    "backend": "isaac", "label": "ISAAC SIM / PHYSICS",
                    "physical": True,
                    "visualization": {"backend": "isaac", "available": True,
                                      "transport": "webrtc", "status": "ready",
                                      "viewer_url": "http://127.0.0.1:49100"}},
            }
    snap = snapshot.RosSnapshotProvider(S()).snapshot()
    assert snap.visualization["transport"] == "webrtc"
    assert snap.to_visualization()["viewer_url"] == "http://127.0.0.1:49100"


# --------------------------------------------------------------------------- #
# FIWARE source honesty
# --------------------------------------------------------------------------- #

def test_fiware_mode_with_no_fiware_data_says_degraded_not_ros():
    """The screen must never assert `source=fiware` in the terminal, `ROS 2 /
    DDS` in the header and `FIWARE unreachable` in the pill all at once."""
    snap = _snap(mode="fiware")
    snap.panel_sources = {p: "ros" for p in _snapshot_module().PANELS}
    badge = snap.badge()
    assert badge["source"] == "fiware-degraded"
    assert "DEGRADED" in badge["label"]
    assert "NOT being read back from FIWARE" in badge["detail"]


def test_plain_ros_mode_is_unaffected():
    snap = _snap(mode="ros")
    snap.panel_sources = {p: "ros" for p in _snapshot_module().PANELS}
    assert snap.badge()["source"] == "ros"


def test_the_launcher_verifies_fiware_health_and_checks_the_result():
    code = _code(DASHBOARD)
    assert "FIWARE_OK=1" in code
    assert 'if [ "$FIWARE_OK" -eq 1 ]' in code, \
        "the health loop's RESULT must be checked, not merely waited out"
    assert "exit 7" in code, "a failed FIWARE start must fail the launch"
    assert "WISEPACK_FIWARE_DEGRADED" in code, \
        "there must be an EXPLICIT degraded opt-in rather than a silent one"


def test_isaac_and_isaac_fiware_select_different_sources():
    code = _code(DASHBOARD)
    assert 'if [ "$MODE" = "fiware" ] || [ "$MODE" = "isaac-fiware" ]' in code
    # `isaac` alone keeps the ROS source; only the -fiware variant switches it.
    assert 'SOURCE="ros"' in code


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #

PAGES = ("index", "inventory", "diagnostics", "simulator")


@pytest.mark.parametrize("page", PAGES)
def test_every_page_carries_the_same_primary_navigation(page):
    html = _read(os.path.join(REPO, "web", f"{page}.html"))
    assert 'class="wpnav"' in html
    for label in ("Dashboard", "Container Inventory", "Diagnostics",
                  "Simulator View"):
        assert f">{label}</a>" in html, f"{page}.html is missing {label}"


@pytest.mark.parametrize("page,href", [
    ("index", "/"), ("inventory", "/inventory"),
    ("diagnostics", "/diagnostics"), ("simulator", "/simulator"),
])
def test_each_page_marks_itself_as_the_active_navigation_item(page, href):
    html = _read(os.path.join(REPO, "web", f"{page}.html"))
    assert f'<a href="{href}" aria-current="page">' in html, \
        f"{page}.html does not mark {href} as the current page"


@pytest.mark.parametrize("page", PAGES)
def test_navigation_has_active_styling_and_responsive_rules(page):
    html = _read(os.path.join(REPO, "web", f"{page}.html"))
    assert '.wpnav a[aria-current="page"]{' in html, "no active-state styling"
    assert "@media (max-width:820px)" in html, "no responsive rule"


@pytest.mark.parametrize("page", ("index", "inventory", "diagnostics"))
def test_simulator_view_is_hidden_until_a_backend_offers_one(page):
    html = _read(os.path.join(REPO, "web", f"{page}.html"))
    assert 'id="nav-simulator" hidden' in html, \
        "Simulator View must start hidden and be revealed by the descriptor"
    assert "refreshSimulatorNav" in html


def test_the_container_inventory_and_diagnostics_routes_exist():
    app = _read(os.path.join(REPO, "web", "app.py"))
    for route in ('@app.get("/inventory"', '@app.get("/diagnostics"',
                  '@app.get("/simulator"', '@app.get("/api/inventory")',
                  '@app.get("/api/diagnostics")', '@app.get("/api/logistics")',
                  '@app.get("/api/visualization")'):
        assert route in app, f"missing route {route}"


def test_the_analytics_pages_and_backends_are_present():
    for name in ("inventory.html", "diagnostics.html", "diagnostics.py"):
        assert os.path.isfile(os.path.join(REPO, "web", name)), f"missing {name}"
    core = os.path.join(REPO, "wisepack_ws", "src", "wisepack_core",
                        "wisepack_core")
    for name in ("inventory.py", "logistics.py", "anomaly.py",
                 "whole_process.py", "cut_optimizer.py"):
        assert os.path.isfile(os.path.join(core, name)), f"missing core/{name}"


def test_the_anomaly_panel_keeps_its_domain_neutral_title():
    html = _read(os.path.join(REPO, "web", "index.html"))
    assert "Anomaly Monitoring &amp; Workflow Response" in html \
        or "Anomaly Monitoring & Workflow Response" in html
    assert "EDF Topic #2 Integration Demo" not in html


def test_no_page_reintroduces_the_obsolete_demo_title():
    for page in PAGES:
        html = _read(os.path.join(REPO, "web", f"{page}.html"))
        assert "EDF Topic #2 Integration Demo" not in html


# --------------------------------------------------------------------------- #
# Simulator View behaviour
# --------------------------------------------------------------------------- #

def test_the_simulator_page_renders_every_required_field():
    html = _read(os.path.join(REPO, "web", "simulator.html"))
    for field in ("connection", "execution backend", "simulator version",
                  "current item", "physical state", "endpoint"):
        assert field in html, f"the Simulator View does not show {field!r}"
    for control in ("Connect", "Open full screen"):
        assert control in html, f"missing the {control} control"


def test_the_simulator_page_never_shows_an_empty_or_endless_loader():
    html = _read(os.path.join(REPO, "web", "simulator.html"))
    # Every render path ends in a named state with wording attached.
    assert "d.status_label" in html
    assert "No visual stream is offered" in html
    assert "spinner" not in html.lower()


def test_the_simulator_page_warns_about_the_unauthenticated_stream():
    html = _read(os.path.join(REPO, "web", "simulator.html"))
    assert "unauthenticated" in html
    assert "SSH port forwarding" in html or "port forwarding" in html


def test_frames_never_travel_over_ros_or_fiware():
    """The transport split, asserted rather than merely intended."""
    html = _read(os.path.join(REPO, "web", "simulator.html"))
    assert "never travel over ROS 2 or FIWARE" in html
    topics = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_bringup",
                                "wisepack_bringup", "topics.py"))
    for banned in ("frame", "jpeg", "video", "image"):
        assert f"/wisepack/{banned}" not in topics


# --------------------------------------------------------------------------- #
# Active scenario synchronisation
# --------------------------------------------------------------------------- #

def test_the_active_run_preset_overrides_the_local_form_state():
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert 'active_preset = scenario.get("preset")' in app
    assert '"preset": active_preset' in app
    assert "settings_locked" in app


def test_the_controls_lock_while_a_run_is_active_or_awaiting_a_decision():
    app = _read(os.path.join(REPO, "web", "app.py"))
    block = app[app.index("locked_stages = ("):app.index("payload.update({", app.index("locked_stages = ("))]
    for stage in ("WAIT_FOR_OPERATOR_APPROVAL", "PICK_ITEM", "PLACE_ITEM"):
        assert stage in block, f"{stage} should lock the scenario controls"


def test_the_frontend_syncs_and_disables_the_controls():
    html = _read(os.path.join(REPO, "web", "index.html"))
    assert "function syncScenarioControls(" in html
    assert "node.disabled = locked" in html


# --------------------------------------------------------------------------- #
# Launcher: streaming options and process ownership
# --------------------------------------------------------------------------- #

def test_the_launcher_parses_every_documented_streaming_variable():
    code = _code(ISAAC_LAUNCHER)
    for var in ("WISEPACK_ISAAC_STREAMING", "WISEPACK_ISAAC_STREAM_HOST",
                "WISEPACK_ISAAC_SIGNAL_PORT", "WISEPACK_ISAAC_STREAM_PORT",
                "WISEPACK_ISAAC_VIEWER_PORT", "WISEPACK_ISAAC_STREAM_URL"):
        assert var in code, f"{var} is not handled by the launcher"


def test_streaming_is_opt_in_and_defaults_to_loopback():
    code = _code(ISAAC_LAUNCHER)
    assert 'STREAMING="${WISEPACK_ISAAC_STREAMING:-0}"' in code
    assert 'STREAM_HOST="${WISEPACK_ISAAC_STREAM_HOST:-127.0.0.1}"' in code


def test_the_launcher_verifies_the_extensions_and_the_port():
    code = _code(ISAAC_LAUNCHER)
    assert "omni.kit.livestream.app" in code
    assert "omni.kit.livestream.webrtc" in code
    assert "already in use" in code, "no guard against a second stream server"
    assert "exit 6" in code


def test_the_launcher_never_probes_an_external_ip_service():
    """Auto-discovering a public IP would leak the host and is not needed."""
    code = _code(ISAAC_LAUNCHER) + _code(DASHBOARD)
    for probe in ("ifconfig.me", "ipify", "icanhazip", "checkip",
                  "api.ipify.org", "curl -s https://ip"):
        assert probe not in code, f"{probe} must not be contacted"


def test_cleanup_owns_a_process_group_and_verifies_it():
    code = _code(DASHBOARD)
    assert "isaac_group_size" in code, "ownership must be counted, not assumed"
    assert 'kill -TERM "-$ISAAC_PID"' in code
    assert "did not exit on TERM" in code, "no bounded escalation to KILL"
    assert "process group $ISAAC_PID is gone" in code, "cleanup does not verify"


def test_the_group_leader_reports_its_own_pid():
    """`setsid` forks, so `$!` is a process that exits immediately — a cleanup
    keyed on it finds nothing to kill and leaves Isaac holding the GPU."""
    code = _code(DASHBOARD)
    assert 'echo $$ > "$1"' in code
    assert "ISAAC_PIDFILE" in code


@pytest.mark.parametrize("script", [DASHBOARD, ISAAC_LAUNCHER])
def test_cleanup_never_touches_unrelated_services(script):
    code = _code(script)
    for reckless in ("pkill", "killall", "systemctl stop", "service ",
                     "sunshine", "nxserver", "moonlight"):
        assert reckless not in code.lower(), \
            f"{os.path.basename(script)} references {reckless!r}"


# --------------------------------------------------------------------------- #
# Host-local SSH port configuration
# --------------------------------------------------------------------------- #

def test_the_template_is_tracked_and_carries_only_a_placeholder():
    example = os.path.join(REPO, "config", "local.env.example")
    assert os.path.isfile(example)
    text = _read(example)
    assert "WISEPACK_SSH_PORT=YOUR_SSH_PORT" in text
    # A real port must never reach the template.
    assert not re.search(r"WISEPACK_SSH_PORT=\d+", text)


def test_the_local_file_is_git_ignored():
    result = subprocess.run(["git", "check-ignore", "-v", "config/local.env"],
                            capture_output=True, text=True, cwd=REPO, timeout=60)
    assert result.returncode == 0, \
        "config/local.env is NOT ignored — a host port could be committed"
    assert "config/local.env" in result.stdout


def test_the_template_itself_is_not_ignored():
    result = subprocess.run(
        ["git", "check-ignore", "-q", "config/local.env.example"],
        capture_output=True, text=True, cwd=REPO, timeout=60)
    assert result.returncode != 0, "the tracked template must not be ignored"


def test_no_tracked_file_contains_a_concrete_ssh_port():
    """Scans TRACKED content only — config/local.env is untracked by design."""
    listing = subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True, cwd=REPO, timeout=120).stdout.split()
    offenders = []
    pattern = re.compile(r"WISEPACK_SSH_PORT\s*[=:]\s*(\d+)")
    for rel in listing:
        path = os.path.join(REPO, rel)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                body = fh.read()
        except OSError:                                  # pragma: no cover
            continue
        if pattern.search(body):
            offenders.append(rel)
    assert not offenders, f"a concrete SSH port is present in: {offenders}"


def test_the_resolver_never_assumes_port_22():
    code = _code(LOCAL_ENV_LIB)
    assert "WISEPACK_SSH_PORT_PLACEHOLDER" in code
    assert ":-22}" not in code and "=22" not in code


def test_the_resolver_precedence_and_placeholder():
    """Exported value wins; otherwise SSH_CONNECTION's 4th field; else a
    placeholder that cannot be mistaken for a port."""
    script = f'''
        set -u
        source "{LOCAL_ENV_LIB}"
        export WISEPACK_SSH_PORT=12345
        wisepack_resolve_ssh_port && echo "explicit=$WISEPACK_SSH_PORT"
        unset WISEPACK_SSH_PORT
        export SSH_CONNECTION="10.0.0.1 5555 10.0.0.2 2222"
        wisepack_resolve_ssh_port && echo "from_ssh=$WISEPACK_SSH_PORT"
        unset WISEPACK_SSH_PORT SSH_CONNECTION
        wisepack_resolve_ssh_port || echo "unresolved=$WISEPACK_SSH_PORT"
    '''
    out = subprocess.run(["bash", "-c", script], capture_output=True,
                         text=True, timeout=60).stdout
    assert "explicit=12345" in out
    assert "from_ssh=2222" in out, "must take the SERVER-side (4th) field"
    assert "unresolved=<ssh-port>" in out


def test_the_readme_uses_a_variable_not_a_literal_port():
    readme = _read(os.path.join(REPO, "README.md"))
    if "ssh -p" in readme:
        assert '${WISEPACK_SSH_PORT}' in readme, \
            "README SSH examples must use the variable, never a literal port"
        assert not re.search(r"ssh -p\s+\"?\d+", readme)


# --------------------------------------------------------------------------- #
# Isolation
# --------------------------------------------------------------------------- #

def test_generic_dashboard_and_core_modules_import_no_isaac():
    """Visualization is backend-neutral; a real cell must be a drop-in swap."""
    generic = [
        os.path.join(REPO, "web", "app.py"),
        os.path.join(REPO, "web", "snapshot.py"),
        os.path.join(REPO, "web", "ros_observer.py"),
        os.path.join(REPO, "web", "diagnostics.py"),
        os.path.join(REPO, "wisepack_ws", "src", "wisepack_core",
                     "wisepack_core", "visualization.py"),
    ]
    for path in generic:
        body = _read(path)
        for marker in ("import isaacsim", "from isaacsim", "from pxr import",
                       "import omni.", "import carb"):
            assert marker not in body, f"{os.path.basename(path)} imports {marker}"


def test_the_visualization_module_names_no_simulator():
    """It may mention Isaac in prose, but must not depend on it in code."""
    path = os.path.join(REPO, "wisepack_ws", "src", "wisepack_core",
                        "wisepack_core", "visualization.py")
    code = "\n".join(l for l in _read(path).splitlines()
                     if not l.lstrip().startswith("#"))
    body = code.split('"""', 2)[-1]          # past the module docstring
    for marker in ("omni", "isaacsim", "webrtc_port", "kit."):
        assert marker not in body, f"visualization.py leaks {marker!r} into code"


def test_the_simulator_page_consumes_only_the_generic_descriptor():
    html = _read(os.path.join(REPO, "web", "simulator.html"))
    assert "/api/visualization" in html
    # It must not reach for Isaac-specific endpoints or settings.
    for marker in ("omni.kit", "isaacsim", "primaryStream"):
        assert marker not in html, f"the Simulator View hard-codes {marker!r}"
