"""The FoundationPose worker's contract, tested without GPU, image or weights.

WHY THESE RUN IN ORDINARY CI (§34)
----------------------------------
The worker's most important property is what it does when things are MISSING: it
must start, answer, and say precisely which of five prerequisites is absent. That
behaviour is exactly what a machine with no GPU and no weights can exercise —
and this machine is one, so the tests run here rather than only in the container.

The parts that genuinely need CUDA (building the estimator, running
`register()`) are not simulated. They are covered by the container smoke test
and by the bolt regression, and neither is claimed to have passed until it has.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO, "perception", "foundationpose", "worker")
if WORKER not in sys.path:
    sys.path.insert(0, WORKER)

import capability as cap                                        # noqa: E402


@pytest.fixture
def weights_dir(tmp_path, monkeypatch):
    """Point the worker's probes at a throwaway weights mount."""
    monkeypatch.setattr(cap, "WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setattr(cap, "DATASETS_DIR", str(tmp_path / "datasets"))
    return tmp_path


def _write_checkpoint(root, folder, size=cap.MIN_PLAUSIBLE_CHECKPOINT_BYTES + 1):
    path = root / "weights" / folder
    path.mkdir(parents=True, exist_ok=True)
    (path / cap.CHECKPOINT_FILE).write_bytes(b"\0" * size)
    return path / cap.CHECKPOINT_FILE


# --------------------------------------------------------------------------- #
# 1. The worker starts and diagnoses itself when everything is missing
# --------------------------------------------------------------------------- #


def test_a_snapshot_is_produced_with_nothing_installed(weights_dir):
    """THE HEADLINE PROPERTY (§2). No GPU, no runtime, no weights — still an
    answer, and `worker_ready` is about the HTTP service alone."""
    snapshot = cap.Capabilities().snapshot()
    assert snapshot["worker_ready"] is True
    assert snapshot["inference_available"] is False
    assert snapshot["blocked_by"], "an unavailable worker must say why"


def test_every_prerequisite_is_a_separate_field(weights_dir):
    """Five independent things fail independently; one boolean would hide four."""
    snapshot = cap.Capabilities().snapshot()
    for field in ("worker_ready", "gpu_available",
                  "foundationpose_runtime_available",
                  "scorer_weights_available", "refiner_weights_available",
                  "inference_available", "last_error"):
        assert field in snapshot, field


def test_inference_is_the_conjunction_of_the_others(weights_dir, monkeypatch):
    """`inference_available` must never be true while a prerequisite is false."""
    capabilities = cap.Capabilities()
    monkeypatch.setattr(capabilities, "gpu",
                        lambda: cap.Probe("gpu", True))
    monkeypatch.setattr(capabilities, "runtime",
                        lambda: cap.Probe("foundationpose_runtime", True))
    # Weights still missing -> inference must stay false.
    assert capabilities.snapshot()["inference_available"] is False

    _write_checkpoint(weights_dir, cap.REFINER_DIR)
    assert capabilities.snapshot()["inference_available"] is False, \
        "one checkpoint is not both"
    _write_checkpoint(weights_dir, cap.SCORER_DIR)
    assert capabilities.snapshot()["inference_available"] is True


# --------------------------------------------------------------------------- #
# 2. Weight probes — the failure modes that look like success
# --------------------------------------------------------------------------- #


def test_a_missing_weights_mount_says_so_and_names_the_fix(weights_dir):
    probe = cap.Capabilities().refiner_weights()
    assert not probe.available
    assert "not mounted" in probe.reason
    assert "setup_foundationpose.sh" in probe.reason


def test_an_absent_checkpoint_explains_that_weights_are_not_shipped(weights_dir):
    (weights_dir / "weights").mkdir(parents=True)
    probe = cap.Capabilities().scorer_weights()
    assert not probe.available
    assert cap.SCORER_DIR in probe.reason
    assert "not shipped with WISEPACK" in probe.reason


def test_a_rate_limited_html_page_is_not_mistaken_for_a_checkpoint(weights_dir):
    """The exact failure the official source produces when quota-limited: an
    HTML page saved under the right filename."""
    _write_checkpoint(weights_dir, cap.REFINER_DIR, size=4096)
    probe = cap.Capabilities().refiner_weights()
    assert not probe.available
    assert "not a checkpoint" in probe.reason
    assert probe.detail["size_bytes"] == 4096


def test_a_present_checkpoint_reports_its_path_and_size(weights_dir):
    path = _write_checkpoint(weights_dir, cap.REFINER_DIR)
    probe = cap.Capabilities().refiner_weights()
    assert probe.available and probe.reason == ""
    assert probe.detail["path"] == str(path)


def test_the_refiner_and_scorer_are_probed_independently(weights_dir):
    """They are separate downloads and either can arrive alone."""
    _write_checkpoint(weights_dir, cap.REFINER_DIR)
    capabilities = cap.Capabilities()
    assert capabilities.refiner_weights().available
    assert not capabilities.scorer_weights().available


def test_weights_appearing_later_are_noticed_without_a_restart(weights_dir):
    """Cheap probes are re-read every time, so a download during a run shows up."""
    capabilities = cap.Capabilities()
    assert not capabilities.refiner_weights().available
    _write_checkpoint(weights_dir, cap.REFINER_DIR)
    assert capabilities.refiner_weights().available


# --------------------------------------------------------------------------- #
# 3. Runtime probe — the reason must name the right project
# --------------------------------------------------------------------------- #


def test_a_missing_dependency_is_named_individually(weights_dir):
    """"No module named 'pytorch3d'" must not be reported as a FoundationPose
    failure — it sends someone to the wrong repository."""
    probe = cap.Capabilities().runtime()
    if probe.available:
        pytest.skip("this machine has the full FoundationPose runtime")
    assert not probe.available
    assert ("required modules are unavailable" in probe.reason
            or "estimator could not be imported" in probe.reason)


def test_the_gpu_probe_explains_a_missing_container_runtime(weights_dir):
    probe = cap.Capabilities().gpu()
    if probe.available:
        assert "compute_capability" in probe.detail
        return
    assert ("torch is not importable" in probe.reason
            or "--gpus all" in probe.reason
            or "CUDA probe failed" in probe.reason)


# --------------------------------------------------------------------------- #
# 4. Provenance — a result must name the code that produced it
# --------------------------------------------------------------------------- #


def test_the_snapshot_carries_the_revision_and_whether_it_matches_the_pin(
        weights_dir):
    snapshot = cap.Capabilities().snapshot()
    assert "foundationpose_revision" in snapshot
    assert "foundationpose_expected_revision" in snapshot
    assert "revision_matches_pin" in snapshot
    # Outside the container neither is known, and claiming a match would be a
    # lie about which code ran.
    assert snapshot["revision_matches_pin"] is False


def test_the_snapshot_states_the_third_party_licence(weights_dir):
    """§7: the licence distinction must be visible at the runtime boundary,
    not only in a document nobody opens."""
    note = cap.Capabilities().snapshot()["licence_note"]
    assert "NVIDIA Source Code License" in note
    assert "non-commercial" in note.lower()
    assert "ships no weights" in note


def test_the_snapshot_is_json_serialisable(weights_dir):
    """It travels over HTTP to the dashboard."""
    json.dumps(cap.Capabilities().snapshot())


# --------------------------------------------------------------------------- #
# 5. The weights resolver — official source only
# --------------------------------------------------------------------------- #


def _weights_script():
    return os.path.join(REPO, "scripts", "foundationpose_weights.py")


def test_the_weights_resolver_reports_incompleteness_without_downloading(
        tmp_path):
    result = subprocess.run(
        [sys.executable, _weights_script(), "--dir", str(tmp_path), "--check"],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 1, "incomplete weights must be a non-zero exit"
    assert "INCOMPLETE" in result.stdout
    assert "drive.google.com" in result.stdout, "it must name the official source"


def test_the_weights_resolver_records_size_and_sha256(tmp_path):
    """§5: filenames, exact sizes and SHA-256, once the weights exist."""
    for folder in (cap.REFINER_DIR, cap.SCORER_DIR):
        directory = tmp_path / folder
        directory.mkdir(parents=True)
        (directory / cap.CHECKPOINT_FILE).write_bytes(
            b"wisepack-test" * 100_000)
    result = subprocess.run(
        [sys.executable, _weights_script(), "--dir", str(tmp_path), "--check"],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COMPLETE" in result.stdout
    assert "sha256" in result.stdout

    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["source"] == "official"
    assert "drive.google.com" in provenance["source_url"]
    for role in ("refiner", "scorer"):
        entry = provenance["checkpoints"][role]
        assert entry["present"] and entry["size_bytes"] == 1_300_000
        assert len(entry["sha256"]) == 64
    # The hashes are OBSERVED, not verified against upstream, and the record
    # must not overstate what they prove.
    assert "not verified against an upstream reference" in provenance["hash_note"]


def test_no_unofficial_mirror_appears_anywhere_in_the_resolver():
    """Community re-uploads of these checkpoints exist. Using one would make
    WISEPACK depend on weights nobody can attest to."""
    source = open(_weights_script(), encoding="utf-8").read()
    for forbidden in ("huggingface.co", "hf.co", "civitai", "modelscope"):
        assert forbidden not in source, f"the resolver references {forbidden}"
    assert source.count("drive.google.com") >= 1


# --------------------------------------------------------------------------- #
# 6. Container ownership and safety (§8 of the brief)
# --------------------------------------------------------------------------- #


def _setup_script():
    return open(os.path.join(REPO, "scripts", "setup_foundationpose.sh"),
                encoding="utf-8").read()


def _executable_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))


def test_the_worker_container_is_owned_by_exact_name_and_label():
    """Ownership is by name AND a label WISEPACK applied — so a container this
    project did not create can never be stopped by it."""
    code = _executable_lines(_setup_script())
    assert 'name=^/${CONTAINER}$' in code
    assert "label=wisepack.owned=true" in code
    assert "--label wisepack.owned=true" in code


def test_no_broad_container_matching_is_used():
    code = _executable_lines(_setup_script())
    for reckless in ("docker kill $(", "docker rm $(", "pkill", "killall",
                     "docker stop $(", "--filter status=", "grep foundation"):
        assert reckless not in code, f"the helper uses `{reckless}`"


def test_the_docker_socket_is_never_mounted_and_privileged_is_never_used():
    """§13/§8: neither is acceptable for convenience."""
    code = _executable_lines(_setup_script())
    assert "docker.sock" not in code
    assert "--privileged" not in code


def test_the_weights_are_mounted_read_only_and_never_baked():
    code = _executable_lines(_setup_script())
    assert ":/weights:ro" in code
    dockerfile = open(
        os.path.join(REPO, "perception", "foundationpose", "Dockerfile"),
        encoding="utf-8").read()
    body = _executable_lines(dockerfile)
    for forbidden in ("model_best.pth", "drive.google.com", "gdown"):
        assert forbidden not in body, (
            f"the image build references {forbidden} — weights must not be "
            "baked into the image")


def test_the_reference_datasets_are_mounted_not_copied():
    """§8: `references/` already lives beside the repo; duplicating 183 MB to
    satisfy a container layout would be waste and a second thing to drift."""
    code = _executable_lines(_setup_script())
    assert ":/datasets:ro" in code
    for forbidden in ("cp -r", "rsync", "tar -c"):
        assert forbidden not in code


def test_the_image_pins_a_revision_and_an_ada_compatible_cuda():
    dockerfile = open(
        os.path.join(REPO, "perception", "foundationpose", "Dockerfile"),
        encoding="utf-8").read()
    # A pinned 40-character revision, not a branch.
    assert "a1b694b83e633c2cb6115b9063d940a687759392" in dockerfile
    assert "git checkout ${FOUNDATIONPOSE_REF}" in dockerfile
    # SM 8.9 is Ada. Building without it would silently PTX-JIT or fail.
    assert 'TORCH_CUDA_ARCH_LIST="8.9"' in dockerfile
    assert "cuda:12.4" in dockerfile
    # CUDA 11.3 is upstream's pin and predates this GPU. Checked over
    # EXECUTABLE lines only: the Dockerfile explains at length why it does not
    # use that base, and a check that cannot tell a prohibition from the thing
    # prohibited would forbid writing the reason down.
    assert "cudagl:11.3" not in _executable_lines(dockerfile)


def test_the_image_does_not_vendor_foundationpose_source():
    """§7: WISEPACK is MIT and must ship a RECIPE, not the licensed code."""
    root = os.path.join(REPO, "perception", "foundationpose")
    for directory, _dirs, files in os.walk(root):
        for name in files:
            assert name not in ("estimater.py", "datareader.py", "Utils.py",
                                "run_demo.py"), (
                f"{os.path.join(directory, name)} looks like vendored "
                "FoundationPose source")


def test_the_worker_speaks_http_only_and_publishes_no_middleware():
    """§31: the host must not gain a second ROS/DDS stack."""
    import ast
    source = open(os.path.join(WORKER, "app.py"), encoding="utf-8").read()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("rclpy", "std_msgs", "rosidl_runtime_py"):
        assert forbidden not in imported


def test_the_worker_reports_the_camera_frame_and_never_relabels_it():
    """§20: a model-based estimate is in the CAMERA frame until a measured
    extrinsic says otherwise."""
    source = open(os.path.join(WORKER, "app.py"), encoding="utf-8").read()
    assert 'CAMERA_FRAME = "camera_color_optical_frame"' in source
    assert "wisepack_workarea" not in source, (
        "the worker must never label its own output with the work-area frame")


def test_the_worker_states_which_mesh_frame_the_pose_refers_to():
    """The Isaac-ROS-versus-NVLabs comparison is only meaningful if each side
    says which frame it reports."""
    source = open(os.path.join(WORKER, "app.py"), encoding="utf-8").read()
    assert '"pose_of": "mesh_origin_as_loaded"' in source


def test_the_worker_never_calls_a_score_an_accuracy():
    """§39: no ground truth exists for either dataset."""
    source = open(os.path.join(WORKER, "app.py"), encoding="utf-8").read()
    assert "absolute pose accuracy is NOT measured" in source
    assert "repeatability" in source


# ---------------------------------------------------------------------------
# Dataset discovery — the reference tree is nested, and is NOT copied
# ---------------------------------------------------------------------------


def _app_module():
    """`app.py` is written for the container; import it by path, with the
    worker directory on sys.path so its sibling `capability` import resolves."""
    import importlib.util
    import sys
    if WORKER not in sys.path:
        sys.path.insert(0, WORKER)
    spec = importlib.util.spec_from_file_location(
        "wisepack_fp_app", os.path.join(WORKER, "app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_dataset(root, frames=2, mask=True, intrinsics=True):
    os.makedirs(os.path.join(root, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(root, "depth"), exist_ok=True)
    for i in range(frames):
        for kind in ("rgb", "depth"):
            open(os.path.join(root, kind, f"{i:07d}.png"), "wb").close()
    if mask:
        os.makedirs(os.path.join(root, "masks"), exist_ok=True)
        open(os.path.join(root, "masks", "0000001.png"), "wb").close()
    if intrinsics:
        with open(os.path.join(root, "cam_K.txt"), "w", encoding="utf-8") as fh:
            fh.write("1 0 1\n0 1 1\n0 0 1\n")


def test_a_dataset_nested_deep_in_the_reference_tree_is_found(tmp_path):
    """The tutorial's demo set is four directories down, beside unrelated ROS
    and Isaac assets. Requiring datasets at the mount root would have meant
    copying the tree into a WISEPACK-shaped layout."""
    app = _app_module()
    nested = tmp_path / "Tutorial" / "isaac_bin_picking" / "FoundationPose_related" / "bolt"
    _make_dataset(str(nested))
    found = app.discover_datasets(str(tmp_path))
    assert len(found) == 1
    described = found[0].describe()
    assert described["complete"] is True
    # Named RELATIVE to the mount, so a request never carries a host path.
    assert described["name"] == os.path.join(
        "Tutorial", "isaac_bin_picking", "FoundationPose_related", "bolt")


def test_an_incomplete_dataset_is_still_reported_with_its_problems(tmp_path):
    """A set missing its mask is exactly what an operator needs told about —
    reporting only usable datasets hides the reason the usable one is absent."""
    app = _app_module()
    _make_dataset(str(tmp_path / "half"), mask=False)
    described = app.discover_datasets(str(tmp_path))[0].describe()
    assert described["complete"] is False
    assert any("mask" in p for p in described["problems"])


def test_discovery_does_not_descend_into_a_datasets_own_frame_directories(tmp_path):
    """`rgb/` inside a dataset is not another dataset."""
    app = _app_module()
    _make_dataset(str(tmp_path / "set"))
    assert len(app.discover_datasets(str(tmp_path))) == 1


def test_a_physical_capture_is_a_dataset_root(tmp_path, monkeypatch):
    """A frame the worker CAPTURED must be estimable against.

    The two original roots hold third-party reference material and generated
    Isaac cases; a physical capture is neither, so nothing could name one. The
    worker could acquire from the D435 and then be unable to estimate from what
    it had just written — the one combination that makes the camera useless.
    """
    monkeypatch.setattr(cap, "CAPTURES_DIR", str(tmp_path / "captures"))
    monkeypatch.setattr(cap, "DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setattr(cap, "ISAAC_DATASETS_DIR", str(tmp_path / "isaac"))
    for name in ("captures", "datasets", "isaac"):
        (tmp_path / name).mkdir()
    roots = cap.dataset_roots()
    assert str(tmp_path / "captures") in roots
    # SEARCHED LAST. The reference tree and the generated cases are the stable
    # material; a capture directory grows with every acquisition, and a name
    # collision must not shadow the datasets a regression depends on.
    assert roots.index(str(tmp_path / "captures")) == len(roots) - 1


def test_a_capture_root_that_is_not_mounted_is_simply_absent(tmp_path,
                                                             monkeypatch):
    """No captures directory is an ordinary state, not an error: a worker
    started without the capture mount still serves the reference datasets."""
    monkeypatch.setattr(cap, "CAPTURES_DIR", str(tmp_path / "nowhere"))
    monkeypatch.setattr(cap, "DATASETS_DIR", str(tmp_path / "datasets"))
    (tmp_path / "datasets").mkdir()
    assert str(tmp_path / "nowhere") not in cap.dataset_roots()


def test_a_dataset_name_cannot_escape_the_mount(tmp_path):
    """Names are caller-supplied relative paths now, so containment is checked
    rather than assumed. A read-only mount still must not become a way to read
    any file the caller names."""
    app = _app_module()
    inside = tmp_path / "set"
    _make_dataset(str(inside))
    assert app.resolve_dataset("set", str(tmp_path)) == str(inside)
    for escape in ("../elsewhere", "set/../../etc", "/etc"):
        with pytest.raises(app.DatasetError):
            app.resolve_dataset(escape, str(tmp_path))


def test_the_reference_tree_is_mounted_and_never_copied():
    """§8: references/ already lives beside the repository. Nothing may
    duplicate it merely to satisfy a container layout."""
    text = _setup_script()
    assert "-v \"$DATASETS_DIR:/datasets:ro\"" in _executable_lines(text)
    for copier in ("cp -r", "cp -a", "rsync"):
        assert copier not in _executable_lines(text)


# ---------------------------------------------------------------------------
# The unit and extension traps
# ---------------------------------------------------------------------------


def test_the_depth_scale_has_no_default():
    """A uint16 millimetre image and a float32 metre image are both ordinary and
    are indistinguishable from the pixels. Defaulting either way is a
    factor-of-1000 error that yields a confident, wrong pose."""
    source = open(os.path.join(WORKER, "app.py"), encoding="utf-8").read()
    assert 'if "depth_scale_mm" not in request:' in source
    assert 'request.get("depth_scale_mm", 1000.0)' not in source


def test_the_native_extensions_are_probed_by_name():
    """`import estimater` succeeds without them: upstream's Utils.py wraps both
    in try/except and leaves them None. The failure then surfaces inside
    register(), at mycpp.cluster_poses. A probe that reports the runtime
    available and then crashes mid-estimate is worse than no probe."""
    source = open(os.path.join(WORKER, "capability.py"), encoding="utf-8").read()
    assert "mycpp" in source
    assert "bundlesdf.mycuda.common" in source


def test_the_image_builds_the_mycpp_extension():
    dockerfile = open(
        os.path.join(REPO, "perception", "foundationpose", "Dockerfile"),
        encoding="utf-8").read()
    assert "mycpp" in dockerfile
    assert "mycpp/build" in dockerfile


def test_the_weights_are_still_not_baked_into_the_image():
    """§4: the checkpoints stay on the host under their own licence. The
    expected in-image path is a SYMLINK to the read-only mount, not a copy."""
    dockerfile = open(
        os.path.join(REPO, "perception", "foundationpose", "Dockerfile"),
        encoding="utf-8").read()
    assert "ln -sfn /weights" in dockerfile
    lines = _executable_lines(dockerfile)
    assert "COPY weights" not in lines
    assert "model_best.pth" not in lines


# ---------------------------------------------------------------------------
# GPU access
# ---------------------------------------------------------------------------


def test_gpu_access_is_verified_by_running_it_not_by_reading_config():
    """daemon.json named a runtime binary that was not installed. Trusting that
    listing produced a --gpus all that stopped the worker starting at all —
    the opposite of the rule this worker is built around."""
    text = _executable_lines(_setup_script())
    assert "docker run --rm --gpus all" in text
    assert "torch.cuda.is_available()" in text


def test_a_host_without_gpu_access_still_starts_the_worker():
    """A missing capability must be REPORTED by a running worker, never turned
    into a container that cannot start and so cannot say why."""
    text = _setup_script()
    assert "starting the worker" in text or "WITHOUT a GPU" in text
    assert "GPU_ARGS=()" in text


def test_the_driver_libraries_are_mounted_read_only():
    text = _executable_lines(_setup_script())
    assert ":ro\"" in text
    assert "libcuda.so.1" in text


# ---------------------------------------------------------------------------
# Symmetry is measured, and the registry records the measurement
# ---------------------------------------------------------------------------


def _registry():
    import yaml
    with open(os.path.join(REPO, "config", "perception_objects.yaml"),
              encoding="utf-8") as handle:
        return {o["model_id"]: o for o in yaml.safe_load(handle)["objects"]}


def test_every_pipe_section_declares_a_geometric_symmetry_and_a_task_equivalence():
    """Two DIFFERENT questions, both answered per part.

    GEOMETRIC is what the CAD is: Cylinder1-3 have square ends and are axially
    symmetric; Cylinder4 and Cylinder5 have intentional saddle-cut ends, so
    their spin is geometrically observable and they are `discrete` instead.

    TASK is what picking needs: position and the tube-axis LINE, for all five.
    Neither may overwrite the other."""
    registry = _registry()
    axial = ("cylinder1", "cylinder2", "cylinder3")
    saddled = ("cylinder4", "cylinder5")
    for model_id in axial:
        assert registry[model_id]["symmetry"]["type"] == "axial", model_id
    for model_id in saddled:
        assert registry[model_id]["symmetry"]["type"] == "discrete", model_id
    for model_id in axial + saddled:
        assert registry[model_id]["task_pose_equivalence"] == "axis_line", model_id


def test_the_bent_pipe_records_its_measured_two_fold_ambiguity():
    """Cylinder5 is not axially symmetric, which is why it is the good early
    6-DoF test — but a 180 deg rotation about z maps it onto itself to within
    sampling noise, because it is a symmetric hairpin. Declaring `none` here
    would report a leg-swap as a resolved measurement."""
    symmetry = _registry()["cylinder5"]["symmetry"]
    assert symmetry["type"] == "discrete"
    assert symmetry["fold"] == 2
    assert symmetry["axis"] == "z"


def test_every_object_declares_its_mesh_units():
    """Neither STL nor OBJ records a unit. mm consumed as m puts a 190 mm pipe
    190 metres away, with total confidence."""
    for model_id, entry in _registry().items():
        assert entry.get("mesh_units") in ("mm", "m"), model_id


def test_the_registry_points_at_no_import_script_that_does_not_exist():
    """The registry used to name scripts/import_perception_assets.sh, which was
    never written. Documentation that names a missing tool is a defect."""
    text = open(os.path.join(REPO, "config", "perception_objects.yaml"),
                encoding="utf-8").read()
    for match in re.findall(r"scripts/[\w./-]+", text):
        assert os.path.exists(os.path.join(REPO, match)), match


# --------------------------------------------------------------------------- #
# Segmentation as its own step
# --------------------------------------------------------------------------- #


def _app_source():
    with open(os.path.join(WORKER, "app.py"), encoding="utf-8") as handle:
        return handle.read()


def test_a_mask_can_be_produced_without_estimating_a_pose():
    """LOOK AT THE MASK FIRST. A mask is an input to a pose measurement, and on
    a physical scene it is the input most likely to be wrong: plane-based
    foreground selection assumes one object on a dominant surface, and a
    cluttered table yields one component holding several objects. /estimate can
    segment for itself, but only after committing to inference — too late for
    an operator to judge whether the region is the object they meant."""
    source = _app_source()
    assert '@app.post("/segment")' in source
    assert "def segment_frame" in source


def test_segmentation_does_not_require_the_estimator():
    """"The mask is wrong" and "the estimator is missing" are different
    problems. Gating /segment on inference_available would present them as one
    on every machine without a GPU."""
    source = _app_source()
    body = source[source.index('@app.post("/segment")'):
                  source.index('@app.get("/camera")')]
    assert "inference_available" not in body
    assert "estimator.register" not in body


def test_the_segment_route_states_the_depth_scale_is_required():
    """A wrong depth scale moves the fitted plane, which silently changes which
    pixels are foreground. It cannot be read from the image."""
    source = _app_source()
    body = source[source.index('@app.post("/segment")'):
                  source.index('@app.get("/camera")')]
    assert "`depth_scale_mm` is required" in body


def test_the_mask_is_drawn_on_the_photograph_not_only_on_black():
    """A white blob on black tells an operator its area; the same blob outlined
    on the RGB tells them whether it is the object they meant — the only
    question the physical segmentation step is asking."""
    source = _app_source()
    assert "def _render_segmentation" in source
    body = source[source.index("def _render_segmentation"):
                  source.index("def _render_overlay")]
    assert 'images["segmentation"]' in body
    assert "drawContours" in body
