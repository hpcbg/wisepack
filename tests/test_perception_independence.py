"""WISEPACK owns its perception runtime. This file proves it.

THE PROPERTY UNDER TEST
-----------------------
Camera perception must work on a machine where the HARMONY repository does not
exist. Every executable line the perception service needs lives in THIS
repository; HARMONY is a source of adapted code and an attribution, not a runtime
dependency.

That is easy to state and easy to lose: one `sys.path.insert` pointing at a
checkout, one `<harmony>/torch_venv/bin/python` in a launcher, one config file
read from another project's directory, and the feature silently starts depending
on a clone that a reviewer may not have. Each of those is exactly what this file
fails on.

WHAT IS AND IS NOT FORBIDDEN
----------------------------
PROVENANCE IS NOT A DEPENDENCY. Naming HARMONY in a docstring, a comment, NOTICE,
the README or a `model_origin` field is REQUIRED — the code was adapted from it
and the attribution has to be visible. So these checks run over EXECUTABLE LINES
ONLY: Python docstrings and comments are stripped with `ast`, shell comment lines
by prefix. A check that could not tell a prohibition from the thing prohibited
would forbid documenting the rule.

What is forbidden is a RUNTIME reference: a path under /data/arise/harmony, a
foreign virtualenv, or a `WISEPACK_HARMONY_*` setting whose purpose is to find
one.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The checkout this feature used to reach into. Absent on a fresh machine, and
#: irrelevant on this one.
HARMONY_CHECKOUT = "/data/arise/harmony"

#: Runtime references that would reintroduce the dependency. Each is a literal
#: that can only appear in executable code for one reason.
FORBIDDEN_RUNTIME_REFERENCES = (
    HARMONY_CHECKOUT,
    "ai-bottle-detector-fiware",
    "torch_venv",
    "WISEPACK_HARMONY_PATH",
    "WISEPACK_HARMONY_RUNTIME_DIR",
    "WISEPACK_HARMONY_CORNER_MARKERS",
    "WISEPACK_HARMONY_CORNER_EXTENT_MM",
    "WISEPACK_PERCEPTION_PYTHON",
    "harmony_camera",
)

#: Production Python: everything that runs in a deployment. Tests are excluded
#: on purpose — THIS file names the forbidden strings in order to forbid them.
PRODUCTION_PYTHON_ROOTS = ("perception", "web", "scripts",
                           os.path.join("wisepack_ws", "src"), "simulators")

#: Production shell.
PRODUCTION_SHELL = ("run_wisepack_dashboard.sh", "run_wisepack_demo.sh",
                    "run_vulcanexus_wisepack.sh", "build_wisepack_ws.sh",
                    os.path.join("scripts", "lib_perception_service.sh"),
                    os.path.join("scripts", "lib_host_processes.sh"),
                    os.path.join("scripts", "lib_local_env.sh"),
                    os.path.join("scripts", "setup_perception.sh"))


def _python_code_without_prose(text: str) -> str:
    """Source with comment lines AND docstrings removed."""
    lines = text.splitlines()
    drop = set()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            doc = body[0].value
            drop.update(range(doc.lineno - 1, (doc.end_lineno or doc.lineno)))
    return "\n".join(line for i, line in enumerate(lines)
                     if i not in drop and not line.lstrip().startswith("#"))


def _shell_code_without_comments(text: str) -> str:
    """Shell has no block comments, so a leading `#` line is documentation."""
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))


def _production_python_files():
    for root in PRODUCTION_PYTHON_ROOTS:
        base = pathlib.Path(REPO) / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts or "/test" in str(path):
                continue
            yield path


# --------------------------------------------------------------------------- #
# 1. No runtime reference survives anywhere in production code
# --------------------------------------------------------------------------- #


def test_no_production_python_references_a_harmony_runtime_path():
    """Docstrings and comments may name HARMONY. Executable lines may not."""
    offenders = []
    for path in _production_python_files():
        code = _python_code_without_prose(path.read_text(encoding="utf-8"))
        for forbidden in FORBIDDEN_RUNTIME_REFERENCES:
            if forbidden in code:
                offenders.append(f"{path.relative_to(REPO)}: {forbidden}")
    assert not offenders, (
        "production code carries a runtime reference to another project's "
        "checkout, environment or settings:\n  " + "\n  ".join(offenders))


def test_no_production_shell_resolves_a_foreign_interpreter():
    offenders = []
    for name in PRODUCTION_SHELL:
        path = pathlib.Path(REPO) / name
        if not path.exists():
            continue
        code = _shell_code_without_comments(path.read_text(encoding="utf-8"))
        for forbidden in FORBIDDEN_RUNTIME_REFERENCES:
            if forbidden in code:
                offenders.append(f"{name}: {forbidden}")
    assert not offenders, (
        "a launcher would execute code from, or look for an interpreter in, "
        "another project:\n  " + "\n  ".join(offenders))


def test_the_configuration_template_offers_no_foreign_setting():
    """`config/local.env.example` is what an operator copies. It must not
    suggest pointing WISEPACK at another repository."""
    text = (pathlib.Path(REPO) / "config" / "local.env.example").read_text()
    for forbidden in FORBIDDEN_RUNTIME_REFERENCES:
        assert forbidden not in text, (
            f"the config template still offers {forbidden!r}")


def test_the_local_env_allowlist_no_longer_honours_foreign_settings():
    """An allowlisted key is honoured; a removed one must be inert.

    Driven, not read: the parser is run with a file that sets every removed
    variable, and none of them may reach the environment.
    """
    library = os.path.join(REPO, "scripts", "lib_local_env.sh")
    script = textwrap.dedent(f"""\
        set -u
        . "{library}"
        mkdir -p "$TMPDIR_TEST/config"
        cat > "$TMPDIR_TEST/config/local.env" <<'EOF'
        WISEPACK_HARMONY_PATH=/somewhere/harmony
        WISEPACK_HARMONY_RUNTIME_DIR=/somewhere/runtime
        WISEPACK_HARMONY_CORNER_MARKERS=1,2,3,4
        WISEPACK_HARMONY_CORNER_EXTENT_MM=999
        WISEPACK_PERCEPTION_PYTHON=/somewhere/python
        WISEPACK_PERCEPTION_CAMERA=7
        EOF
        wisepack_load_local_env "$TMPDIR_TEST"
        echo "CAMERA=${{WISEPACK_PERCEPTION_CAMERA:-}}"
        echo "HARMONY_PATH=${{WISEPACK_HARMONY_PATH:-}}"
        echo "HARMONY_RUNTIME=${{WISEPACK_HARMONY_RUNTIME_DIR:-}}"
        echo "HARMONY_MARKERS=${{WISEPACK_HARMONY_CORNER_MARKERS:-}}"
        echo "HARMONY_EXTENT=${{WISEPACK_HARMONY_CORNER_EXTENT_MM:-}}"
        echo "PERCEPTION_PYTHON=${{WISEPACK_PERCEPTION_PYTHON:-}}"
    """)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        environment = {k: v for k, v in os.environ.items()
                       if not k.startswith(("WISEPACK_HARMONY",
                                            "WISEPACK_PERCEPTION"))}
        environment["TMPDIR_TEST"] = tmp
        result = subprocess.run(["bash", "-c", script], capture_output=True,
                                text=True, env=environment, timeout=30)
    out = result.stdout
    # The WISEPACK-owned key still works...
    assert "CAMERA=7" in out, out
    # ...and every removed one is inert.
    for line in ("HARMONY_PATH=", "HARMONY_RUNTIME=", "HARMONY_MARKERS=",
                 "HARMONY_EXTENT=", "PERCEPTION_PYTHON="):
        assert f"{line}\n" in out + "\n", (
            f"{line} was honoured by the allowlist parser:\n{out}")


# --------------------------------------------------------------------------- #
# 2. The runtime resolves entirely inside WISEPACK
# --------------------------------------------------------------------------- #

#: Imported and exercised in a SUBPROCESS, with a hostile environment: a
#: `WISEPACK_HARMONY_PATH` that would break everything if it were still read, and
#: a `sys.path` audit afterwards. Anything loaded from the forbidden checkout
#: fails the assertion inside the child, so this is a behavioural check rather
#: than a text scan.
_IMPORT_PROBE = r'''
import json, os, sys
REPO = sys.argv[1]
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
sys.path.insert(0, os.path.join(REPO, "perception"))

import perception_config, calibration, model_store
from providers import fasterrcnn_bottle as provider

config = perception_config.PerceptionConfig.from_env(model_path="/tmp/x.pth")
resolution = model_store.ensure_model(configured="", cache_dir="/nonexistent",
                                      env={}, allow_download=False)

loaded = sorted({getattr(m, "__file__", "") or ""
                 for m in sys.modules.values()})
foreign = [f for f in loaded if f.startswith("/data/arise/harmony")]
print(json.dumps({
    "camera": config.camera,
    "markers": list(config.board.marker_ids),
    "detector_id": provider.DETECTOR_ID,
    "implementation_origin": provider.IMPLEMENTATION_ORIGIN,
    "sentinel": list(calibration.UNCALIBRATED_SENTINEL),
    "model_origin": resolution.origin,
    "foreign_modules": foreign,
    "foreign_syspath": [p for p in sys.path if "harmony" in p.lower()],
}))
'''


def _probe(extra_env=None):
    environment = {**os.environ,
                   # Hostile: if any of these were still consulted the probe
                   # would either fail or report a foreign path.
                   "WISEPACK_HARMONY_PATH": "/nonexistent/harmony",
                   "WISEPACK_PERCEPTION_PYTHON": "/nonexistent/python",
                   "WISEPACK_PERCEPTION_MODEL_DOWNLOAD": "0"}
    for key in list(environment):
        if key.startswith("WISEPACK_PERCEPTION_CALIBRATION"):
            del environment[key]
    environment.pop("WISEPACK_PERCEPTION_CAMERA", None)
    environment.update(extra_env or {})
    result = subprocess.run([sys.executable, "-c", _IMPORT_PROBE, REPO],
                            capture_output=True, text=True, env=environment,
                            timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    import json
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_perception_runtime_imports_nothing_from_a_harmony_checkout():
    """THE HEADLINE PROPERTY: delete the clone and nothing changes."""
    probe = _probe()
    assert probe["foreign_modules"] == [], (
        "a perception module was loaded from another project's checkout: "
        f"{probe['foreign_modules']}")
    assert probe["foreign_syspath"] == [], (
        "the perception runtime put another project's directory on sys.path: "
        f"{probe['foreign_syspath']}")


def test_perception_configuration_resolves_with_no_harmony_present():
    """Configuration comes from WISEPACK's own environment variables."""
    probe = _probe({"WISEPACK_PERCEPTION_CAMERA": "3",
                    "WISEPACK_PERCEPTION_CALIBRATION_MARKERS": "20,21,22,23"})
    assert probe["camera"] == 3
    assert probe["markers"] == [20, 21, 22, 23]
    assert probe["sentinel"] == [1.0, 1.0]
    # Provenance survives — it is an attribution, not a dependency.
    assert probe["implementation_origin"] == "HARMONY"
    assert "harmony" not in probe["detector_id"].lower()


#: The strongest form of the property, and the only one that cannot be fooled by
#: a dynamic import or a file read that never goes through `import`:
#: `sys.addaudithook` fires on `open`, `import`, `exec`, `os.listdir` and more,
#: so a hook that RAISES on any event mentioning the checkout turns "we believe
#: nothing reads from there" into "nothing can read from there without failing".
#:
#: It runs a REAL detection — network construction, checkpoint load, ArUco
#: calibration, inference, annotation — because the interesting failure is a lazy
#: import on the inference path, not one at module scope.
_AUDIT_PROBE = r'''
import json, os, sys

FORBIDDEN = "/data/arise/harmony"
touched = []

def hook(event, args):
    def check(value):
        if isinstance(value, str) and FORBIDDEN in value:
            touched.append([event, value])
            raise RuntimeError("touched " + FORBIDDEN + ": " + event + " " + value)
    for arg in args:
        check(arg)
        if isinstance(arg, (list, tuple)):
            for item in arg:
                check(item)

sys.addaudithook(hook)

REPO = sys.argv[1]
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
sys.path.insert(0, os.path.join(REPO, "perception"))

import cv2
from perception_config import PerceptionConfig
from providers import fasterrcnn_bottle as provider

config = PerceptionConfig.from_env(model_path=sys.argv[2])
detector = provider.build_detector(config)
frame = cv2.imread(os.path.join(REPO, "tests", "data", "perception",
                                "calibrated-scene.jpg"))
result = detector.process_frame(frame.copy())
print(json.dumps({
    "objects": [[round(o["x"], 3), round(o["y"], 3), round(o["yaw"], 3)]
                for o in result["objects"]],
    "calibration": result["calibration"]["status"],
    "touched": touched,
}))
'''


def test_a_real_detection_never_reads_from_the_harmony_checkout():
    """§13, enforced rather than believed — and with the network actually run.

    SKIPPED where torch or the weights are absent: this needs the perception
    environment. Where it can run, it is the decisive check.
    """
    try:
        import cv2                                           # noqa: F401,PLC0415
        import torch                                         # noqa: F401,PLC0415
        import torchvision                                   # noqa: F401,PLC0415
    except ImportError as exc:
        pytest.skip(f"the perception environment is not active ({exc})")

    sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
    sys.path.insert(0, os.path.join(REPO, "perception"))
    from wisepack_core.perception import resolve_model_path   # noqa: PLC0415
    import model_store                                        # noqa: PLC0415

    resolution = resolve_model_path(cache_dir=model_store.default_cache_dir())
    if not resolution.available:
        pytest.skip("no detector weights are available on this host")

    result = subprocess.run(
        [sys.executable, "-c", _AUDIT_PROBE, REPO, resolution.path],
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (
        "a real detection pass tried to read from " + HARMONY_CHECKOUT + ":\n"
        + result.stdout + result.stderr)

    import json
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    assert probe["touched"] == []
    # And it really did detect: an audit hook over a no-op proves nothing.
    assert probe["calibration"] == "valid"
    assert len(probe["objects"]) == 2


def test_every_module_the_service_needs_is_inside_this_repository():
    """The whole perception package, enumerated. A missing file is a broken
    promise, not a lazily discovered import error at demo time."""
    perception = pathlib.Path(REPO) / "perception"
    for name in ("perception_service.py", "perception_config.py", "camera.py",
                 "calibration.py", "model_store.py", "requirements.txt",
                 os.path.join("providers", "__init__.py"),
                 os.path.join("providers", "fasterrcnn_bottle.py")):
        assert (perception / name).exists(), f"perception/{name} is missing"
    assert (pathlib.Path(REPO) / "scripts" / "setup_perception.sh").exists()
    assert (pathlib.Path(REPO) / "scripts"
            / "generate_calibration_sheet.py").exists()


def test_the_launcher_names_a_wisepack_owned_environment():
    """`.venv-perception/` inside the working directory, created by our script."""
    library = (pathlib.Path(REPO) / "scripts"
               / "lib_perception_service.sh").read_text()
    code = _shell_code_without_comments(library)
    assert ".venv-perception" in code
    assert "setup_perception.sh" in code
    # And never activated into the launcher shell.
    assert "bin/activate" not in code


# --------------------------------------------------------------------------- #
# 3. The generic core stays free of the detector
# --------------------------------------------------------------------------- #


def test_the_core_package_imports_nothing_detector_specific():
    """§13: no torch, no cv2, no provider, anywhere in `wisepack_core`.

    The domain package is imported by every ROS node and by the containerised
    dashboard, neither of which has any of that installed. An import added here
    would not fail in a unit test — it would fail inside the container, at
    launch, in front of an audience.
    """
    core = (pathlib.Path(REPO) / "wisepack_ws" / "src" / "wisepack_core"
            / "wisepack_core")
    forbidden = {"torch", "torchvision", "cv2", "numpy", "PIL",
                 "providers", "fasterrcnn_bottle", "pipeline", "camera",
                 "calibration", "perception_config", "model_store"}
    offenders = []
    for path in sorted(core.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name in forbidden:
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, (
        "the domain package imports detector-specific machinery:\n  "
        + "\n  ".join(offenders))


def test_the_core_carries_no_detector_module():
    core = (pathlib.Path(REPO) / "wisepack_ws" / "src" / "wisepack_core"
            / "wisepack_core")
    for forbidden in ("harmony_adapter.py", "pipeline.py", "fasterrcnn.py",
                      "detector.py"):
        assert not (core / forbidden).exists(), (
            f"{forbidden} must live in perception/providers/, not in the domain")


#: The ONE detector-flavoured string the domain is allowed to contain: the
#: PUBLIC IDENTIFIER of the released weights on Hugging Face. It is an
#: attribution and a download address — data, not an architecture and not a path
#: into anybody's working tree. Exempted explicitly rather than by loosening the
#: rule, so a second such string cannot slip in unnoticed.
PUBLISHED_MODEL_IDENTIFIER = "hpcbg/harmony-bottle-detector"


#: The provider's own name. `perception.py` is the REGISTRY — it has to know
#: which providers exist in order to let `WISEPACK_PERCEPTION_DETECTOR` choose
#: between them — so the name appears there as a selectable value.
PROVIDER_REGISTRY_NAME = "fasterrcnn_bottle"

#: The PUBLIC PERCEPTION METHOD names, which the registry also has to hold for
#: the same reason: they are the values `WISEPACK_PERCEPTION_METHOD` selects
#: between, and the labels the operator picks from.
#:
#: `planar_fasterrcnn` NAMES AN ARCHITECTURE, and that is worth stating plainly
#: rather than hiding behind an exemption. The rule this file enforces is that
#: the domain must not DEPEND on a detector; a selectable value is data, not a
#: dependency, and `perception.py` imports nothing from the provider. But the
#: name is still less neutral than "planar_rgb" would have been. It is the
#: name that was specified for the public setting, so it is used, and it is
#: exempted BY NAME so a third such string cannot slip in unnoticed.
PERCEPTION_METHOD_NAMES = ("planar_fasterrcnn", "foundationpose_rgbd")


@pytest.mark.parametrize("module", [
    "packing.py", "workflow.py", "validator.py", "kpi.py", "execution.py",
    "domain.py", "perception_client.py",
])
def test_no_core_module_carries_a_detector_name_in_its_code(module):
    """Executable lines only — these files DOCUMENT the rule they obey.

    `perception.py` is deliberately absent from this list: it is the provider
    registry and the model resolver, and what it is allowed to contain is
    pinned exactly by the next test rather than loosely by this one.
    """
    path = (pathlib.Path(REPO) / "wisepack_ws" / "src" / "wisepack_core"
            / "wisepack_core" / module)
    code = _python_code_without_prose(path.read_text(encoding="utf-8")).lower()
    for forbidden in ("harmony", "bottle", "fasterrcnn", "faster_rcnn",
                      "aruco", "torch"):
        assert forbidden not in code, (
            f"{module} names {forbidden!r} in executable code — the domain must "
            "stay detector-neutral")


def test_the_registry_names_a_provider_and_a_model_and_nothing_else():
    """`perception.py` may name a SHORT, FIXED LIST of things, all of them data.

    The provider's SELECTABLE NAME (so a second provider is a configuration
    change) and the PUBLIC IDENTIFIER of the released weights (an attribution
    and a download address). Neither is a path into anybody's working tree.
    Exempted by name rather than by loosening the rule, so a third such string
    cannot slip in unnoticed.
    """
    path = (pathlib.Path(REPO) / "wisepack_ws" / "src" / "wisepack_core"
            / "wisepack_core" / "perception.py")
    code = _python_code_without_prose(path.read_text(encoding="utf-8")).lower()
    code = code.replace(PUBLISHED_MODEL_IDENTIFIER, "<published-model>")
    code = code.replace(PROVIDER_REGISTRY_NAME, "<provider-name>")
    for method in PERCEPTION_METHOD_NAMES:
        code = code.replace(method, "<method-name>")
    for forbidden in ("harmony", "bottle", "fasterrcnn", "faster_rcnn",
                      "aruco", "torch", "cv2"):
        assert forbidden not in code, (
            f"perception.py names {forbidden!r} beyond the provider name and "
            "the published model identifier")


def test_the_method_names_are_selectable_values_not_imports():
    """The exemption is only defensible if the names really are inert data."""
    import ast
    path = (pathlib.Path(REPO) / "wisepack_ws" / "src" / "wisepack_core"
            / "wisepack_core" / "perception.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        for module in modules:
            lowered = module.lower()
            for forbidden in ("provider", "fasterrcnn", "foundationpose",
                              "torch", "cv2"):
                assert forbidden not in lowered, (
                    f"perception.py imports {module} — a selectable name became "
                    "a dependency")


def test_the_two_exemptions_are_data_not_dependencies():
    from wisepack_core.perception import (DEFAULT_DETECTOR,
                                          HUGGINGFACE_MODEL_URL,
                                          HUGGINGFACE_REPO)
    assert DEFAULT_DETECTOR == PROVIDER_REGISTRY_NAME
    assert HUGGINGFACE_REPO == PUBLISHED_MODEL_IDENTIFIER
    # A URL, not a directory anyone has to have checked out.
    assert HUGGINGFACE_MODEL_URL.startswith("https://huggingface.co/")
    assert HARMONY_CHECKOUT not in HUGGINGFACE_MODEL_URL
