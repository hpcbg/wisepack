"""What this worker can actually do, probed rather than assumed.

THE RULE THIS FILE EXISTS TO ENFORCE
------------------------------------
A container that builds is not a container that can estimate a pose. Between
"the image exists" and "inference is ready" sit at least five independent
things, each of which fails on its own:

    a GPU is visible to the container      (nvidia runtime, --gpus)
    torch can use it                       (driver/CUDA/torch agreement)
    the FoundationPose runtime imports      (CUDA extensions, pytorch3d, nvdiffrast)
    the scorer weights are present          (mounted, licensed, downloaded)
    the refiner weights are present         (same, separately)

So each is a SEPARATE FIELD with its own reason, and `inference_available` is
their conjunction. Collapsing them into one boolean is what produces the
dashboard that says "unavailable" while an operator has no idea whether to plug
in a GPU, rebuild an image, or wait for a download.

EVERY PROBE IS NON-FATAL. Import errors, missing directories and CUDA failures
are captured and reported. The worker's job when something is missing is to keep
answering and say what is missing — a process that exits takes the diagnosis
with it.

CACHED, BECAUSE THE EXPENSIVE ONES ARE EXPENSIVE. Importing the FoundationPose
runtime pulls in torch, pytorch3d and nvdiffrast and costs seconds; the health
endpoint is polled. The heavy probes run once and are reused; the cheap ones
(files on disk) are re-read every time, because weights can appear while the
container is running and the dashboard must notice without a restart.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Where the read-only mounts land inside the container.
WEIGHTS_DIR = os.environ.get("WISEPACK_FP_WEIGHTS_DIR", "/weights")
DATASETS_DIR = os.environ.get("WISEPACK_FP_DATASETS_DIR", "/datasets")
#: A SECOND dataset root, for generated reference cases. Separate because it
#: cannot be nested inside the read-only reference mount — Docker cannot create
#: a mountpoint inside a read-only mount — and because generated cases and
#: third-party reference material have different lifetimes anyway.
ISAAC_DATASETS_DIR = os.environ.get("WISEPACK_FP_ISAAC_DATASETS_DIR",
                                    "/isaac-reference")


def dataset_roots():
    """Every root a dataset may live under, in search order."""
    return [d for d in (DATASETS_DIR, ISAAC_DATASETS_DIR) if os.path.isdir(d)]

#: The two checkpoint directories FoundationPose requires, exactly as upstream
#: names them. Hard-coded because they are upstream's identifiers, not a
#: WISEPACK choice — and because a typo here would look like a missing download.
REFINER_DIR = "2023-10-28-18-33-37"
SCORER_DIR = "2024-01-11-20-02-45"
CHECKPOINT_FILE = "model_best.pth"

#: Below this a "checkpoint" is not a checkpoint — most likely an HTML error
#: page saved with a 200, which is exactly what a rate-limited Google Drive
#: returns. Catching it here beats a confusing failure inside torch.load.
MIN_PLAUSIBLE_CHECKPOINT_BYTES = 1_000_000


@dataclass
class Probe:
    """One capability: whether it is there, and — when not — precisely why."""

    name: str
    available: bool = False
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"available": self.available, "reason": self.reason,
                **({"detail": self.detail} if self.detail else {})}


def _weight_probe(name: str, directory: str) -> Probe:
    """A checkpoint directory on the read-only mount. Cheap; never cached."""
    path = os.path.join(WEIGHTS_DIR, directory, CHECKPOINT_FILE)
    if not os.path.isdir(WEIGHTS_DIR):
        return Probe(name, False,
                     f"the weights directory {WEIGHTS_DIR} is not mounted. "
                     "Run ./scripts/setup_foundationpose.sh --weights on the "
                     "host, then restart the worker.")
    if not os.path.isfile(path):
        return Probe(name, False,
                     f"{directory}/{CHECKPOINT_FILE} is not present under "
                     f"{WEIGHTS_DIR}. FoundationPose weights are distributed "
                     "under the NVIDIA Source Code License and are not shipped "
                     "with WISEPACK; fetch them from the official source.")
    size = os.path.getsize(path)
    if size < MIN_PLAUSIBLE_CHECKPOINT_BYTES:
        return Probe(name, False,
                     f"{path} is only {size} bytes — that is not a checkpoint. "
                     "A rate-limited download often saves an HTML page under "
                     "the right filename; delete it and fetch again.",
                     {"path": path, "size_bytes": size})
    return Probe(name, True, "", {"path": path, "size_bytes": size})


class Capabilities:
    """The worker's self-knowledge. One instance per process."""

    def __init__(self) -> None:
        self._gpu: Optional[Probe] = None
        self._runtime: Optional[Probe] = None
        self._versions: Dict[str, Any] = {}
        self.last_error: str = ""

    # -- expensive probes, run once ---------------------------------------- #

    def gpu(self) -> Probe:
        if self._gpu is not None:
            return self._gpu
        try:
            import torch                                     # noqa: PLC0415
        except Exception as exc:                             # noqa: BLE001
            self._gpu = Probe("gpu", False, f"torch is not importable: {exc}")
            return self._gpu

        self._versions["torch"] = getattr(torch, "__version__", "unknown")
        self._versions["torch_cuda"] = getattr(torch.version, "cuda", None)
        try:
            if not torch.cuda.is_available():
                # SPECIFIC, because the three causes need different fixes.
                self._gpu = Probe(
                    "gpu", False,
                    "torch.cuda.is_available() is False — the container has no "
                    "GPU. Check that the container was started with `--gpus all` "
                    "and that the NVIDIA container runtime is installed.")
                return self._gpu
            index = torch.cuda.current_device()
            name = torch.cuda.get_device_name(index)
            major, minor = torch.cuda.get_device_capability(index)
            detail = {
                "device": name,
                "compute_capability": f"{major}.{minor}",
                "device_count": torch.cuda.device_count(),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(index).total_memory / 1e9, 1),
            }
            self._versions["cuda_runtime"] = getattr(torch.version, "cuda", None)
            self._gpu = Probe("gpu", True, "", detail)
        except Exception as exc:                             # noqa: BLE001
            self._gpu = Probe("gpu", False, f"CUDA probe failed: {exc}")
        return self._gpu

    def runtime(self) -> Probe:
        """Can the FoundationPose estimator actually be constructed?

        Imports are attempted INDIVIDUALLY so the reason names the one that
        failed. `import estimater` alone would report "No module named
        'pytorch3d'" as a FoundationPose failure and send someone to the wrong
        repository.
        """
        if self._runtime is not None:
            return self._runtime

        missing: List[str] = []
        for module, why in (
                ("torch", "the deep-learning runtime"),
                ("pytorch3d", "mesh operations used by the refiner"),
                ("nvdiffrast", "the differentiable rasteriser"),
                ("trimesh", "mesh loading")):
            try:
                __import__(module)
            except Exception as exc:                         # noqa: BLE001
                missing.append(f"{module} ({why}): {exc}")

        if missing:
            self._runtime = Probe(
                "foundationpose_runtime", False,
                "required modules are unavailable: " + "; ".join(missing))
            return self._runtime

        try:
            # The estimator module itself. This is what pulls in the compiled
            # CUDA extensions, so it is the real test of the image build.
            import estimater                                 # noqa: F401,PLC0415
        except Exception as exc:                             # noqa: BLE001
            self._runtime = Probe(
                "foundationpose_runtime", False,
                f"the FoundationPose estimator could not be imported: {exc}. "
                "This usually means the CUDA extensions did not build — see the "
                "image build log.")
            return self._runtime

        # THE NATIVE EXTENSIONS, BY NAME — because `import estimater` does not
        # test them. Upstream's `Utils.py` wraps both in try/except and leaves
        # them None, so the import above succeeds in a container where they are
        # missing and the failure surfaces much later: `mycpp.cluster_poses` is
        # called at estimater.py:120 while building the rotation grid, i.e.
        # inside `register()`. A capability probe that reports "runtime
        # available" and then crashes mid-estimate is worse than no probe, since
        # it moves the diagnosis from startup to the middle of a run.
        broken: List[str] = []
        for module, why in (
                ("mycpp", "pose-hypothesis clustering, used by register()"),
                ("bundlesdf.mycuda.common", "CUDA kernels used by the refiner")):
            try:
                __import__(module)
            except Exception as exc:                         # noqa: BLE001
                broken.append(f"{module} ({why}): {exc}")
        if broken:
            self._runtime = Probe(
                "foundationpose_runtime", False,
                "the FoundationPose native extensions are missing, so "
                "register() would fail part-way through: " + "; ".join(broken)
                + ". Rebuild the image and check the extension build output.")
            return self._runtime

        self._runtime = Probe("foundationpose_runtime", True, "",
                              {"revision": self.source_revision()})
        return self._runtime

    # -- cheap, re-read every time ----------------------------------------- #

    def refiner_weights(self) -> Probe:
        return _weight_probe("refiner_weights", REFINER_DIR)

    def scorer_weights(self) -> Probe:
        return _weight_probe("scorer_weights", SCORER_DIR)

    def rgbd_camera(self) -> Probe:
        """Is an RGB-D camera usable BY THIS WORKER? Cheap; never cached.

        Not cached because a camera can be plugged in while the container runs
        and the dashboard must notice without a restart — the same rule the
        weight probes follow.
        """
        try:
            from camera import available                     # noqa: PLC0415
        except Exception as exc:                             # noqa: BLE001
            return Probe("rgbd_camera", False,
                         f"the RGB-D acquisition module is unavailable: {exc}")
        usable, reason = available()
        return Probe("rgbd_camera", usable, "" if usable else reason)

    def datasets(self) -> Probe:
        if not os.path.isdir(DATASETS_DIR):
            return Probe("datasets", False,
                         f"{DATASETS_DIR} is not mounted")
        entries = sorted(e for e in os.listdir(DATASETS_DIR)
                         if os.path.isdir(os.path.join(DATASETS_DIR, e)))
        return Probe("datasets", bool(entries),
                     "" if entries else f"{DATASETS_DIR} is mounted but empty",
                     {"available": entries})

    # -- identity ----------------------------------------------------------- #

    def source_revision(self) -> str:
        """The FoundationPose revision this image was built from.

        Read from the file the Dockerfile wrote, falling back to git. A result
        that cannot name the code that produced it is not reproducible.
        """
        recorded = "/opt/foundationpose.revision"
        if os.path.isfile(recorded):
            try:
                with open(recorded, encoding="utf-8") as handle:
                    return handle.read().strip()
            except OSError:
                pass
        directory = os.environ.get("FOUNDATIONPOSE_DIR", "/opt/foundationpose")
        try:
            return subprocess.run(["git", "-C", directory, "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=5
                                  ).stdout.strip()
        except Exception:                                    # noqa: BLE001
            return ""

    @property
    def expected_revision(self) -> str:
        return os.environ.get("FOUNDATIONPOSE_REF", "")

    # -- the whole picture -------------------------------------------------- #

    def snapshot(self) -> Dict[str, Any]:
        gpu = self.gpu()
        runtime = self.runtime()
        refiner = self.refiner_weights()
        scorer = self.scorer_weights()
        datasets = self.datasets()
        camera = self.rgbd_camera()

        # INFERENCE READINESS IS NOT CAMERA READINESS. The offline reference
        # regression runs with no camera at all, so the camera is reported as
        # its own capability and is NOT folded into `inference_available` —
        # which answers "can this worker estimate a pose from data it is given".
        inference = (gpu.available and runtime.available
                     and refiner.available and scorer.available)
        blockers = [p.reason for p in (gpu, runtime, refiner, scorer)
                    if not p.available and p.reason]

        revision = self.source_revision()
        expected = self.expected_revision
        return {
            # WORKER_READY IS ABOUT THE HTTP SERVICE, nothing else. It is True
            # whenever this answer is being produced — which is the point: the
            # container starts and diagnoses itself even with no weights and no
            # GPU.
            "worker_ready": True,
            "gpu_available": gpu.available,
            "foundationpose_runtime_available": runtime.available,
            "scorer_weights_available": scorer.available,
            "refiner_weights_available": refiner.available,
            "inference_available": inference,
            # THE CAMERA, SEPARATELY. Live acquisition needs both.
            "rgbd_camera_available": camera.available,
            "live_inference_available": bool(inference and camera.available),
            "last_error": self.last_error,
            # WHY, per capability. A single "unavailable" tells an operator
            # nothing about which of five different fixes to apply.
            "blocked_by": blockers,
            "probes": {
                "gpu": gpu.to_dict(),
                "foundationpose_runtime": runtime.to_dict(),
                "refiner_weights": refiner.to_dict(),
                "scorer_weights": scorer.to_dict(),
                "datasets": datasets.to_dict(),
                "rgbd_camera": camera.to_dict(),
            },
            "versions": dict(self._versions),
            "foundationpose_revision": revision,
            "foundationpose_expected_revision": expected,
            # A build from an unexpected revision is not an error, but it must
            # be visible: results are only comparable within one revision.
            "revision_matches_pin": bool(revision and expected
                                         and revision == expected),
            "weights_dir": WEIGHTS_DIR,
            "datasets_dir": DATASETS_DIR,
            "licence_note": (
                "FoundationPose is third-party software under the NVIDIA Source "
                "Code License (non-commercial research use). WISEPACK vendors "
                "none of it and ships no weights."),
        }


__all__ = ["Capabilities", "Probe", "WEIGHTS_DIR", "DATASETS_DIR",
           "ISAAC_DATASETS_DIR", "dataset_roots",
           "REFINER_DIR", "SCORER_DIR", "CHECKPOINT_FILE"]
