#!/usr/bin/env python3
"""WISEPACK FoundationPose worker — the HTTP surface WISEPACK talks to.

WHAT THIS IS
------------
A thin WISEPACK-owned wrapper around a pinned third-party FoundationPose
checkout. It owns no policy: it loads a mesh, takes an RGB-D frame plus a mask,
runs `register()`, and returns a pose in the CAMERA frame with enough provenance
for WISEPACK to build a `PhysicalObservation`.

    GET  /health                  capability, per-part, always answers
    GET  /datasets                which reference datasets are mounted
    POST /estimate                one registration -> pose in the camera frame
    GET  /last-result             the previous result, unchanged
    GET  /image/{kind}            rgb | depth | mask | overlay  (diagnostics)

IT STARTS WITHOUT WEIGHTS, ON PURPOSE
-------------------------------------
Nothing heavy is imported at module scope and no capability is required to
serve. A worker with no GPU and no weights still starts and still answers
`/health` with exactly which of the five prerequisites is missing. A process
that refused to start would take the diagnosis with it, and the dashboard would
have nothing to show but a connection error.

WHAT IT DOES NOT DO
-------------------
* No ROS, no DDS. HTTP only, like the planar service — WISEPACK's validated
  middleware is the containerised Vulcanexus runtime and this must not become a
  second one.
* No robot. It returns a pose; it commands nothing.
* No frame conversion. The pose is reported in the camera optical frame and
  labelled as such. Converting it to the work area needs a measured extrinsic,
  which is WISEPACK's business and not this worker's.
* No symmetry canonicalisation. The raw estimate is returned; WISEPACK applies
  the declared symmetry, because the object registry lives there.

LICENCE. FoundationPose is third-party, under the NVIDIA Source Code License
(non-commercial research use). This file is WISEPACK's (MIT) and vendors none of
it.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from capability import (Capabilities, CHECKPOINT_FILE, DATASETS_DIR,
                        ISAAC_DATASETS_DIR, REFINER_DIR, SCORER_DIR,
                        WEIGHTS_DIR, dataset_roots)

#: The frame every pose from this worker is expressed in. Named explicitly and
#: never silently renamed: a model-based estimator reports relative to the
#: camera, and calling that a world frame puts objects inside the lens.
CAMERA_FRAME = "camera_color_optical_frame"

DEFAULT_PORT = 22201


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------- #
# Dataset reading — the reference-directory layout
# --------------------------------------------------------------------------- #


class DatasetError(RuntimeError):
    """A dataset that cannot be used, with the reason a human needs."""


#: How deep to look for a demo directory. The tutorial's `bolt` set is four
#: levels down; beyond that a match is more likely to be a coincidence than a
#: dataset, and the walk stops paying for itself.
DATASET_SEARCH_DEPTH = 5


def discover_datasets(root: str) -> List["ReferenceDataset"]:
    """Directories that look like a FoundationPose demo set, anywhere under `root`.

    "Looks like" means it has intrinsics, or an rgb/ and a depth/ directory —
    upstream's own layout. Reported whether or not it is COMPLETE, because a
    dataset missing its mask is exactly the thing an operator needs told.
    """
    found: List["ReferenceDataset"] = []
    if not os.path.isdir(root):
        return found
    base_depth = root.rstrip("/").count(os.sep)
    for directory, subdirs, files in os.walk(root):
        if directory.count(os.sep) - base_depth >= DATASET_SEARCH_DEPTH:
            subdirs[:] = []
            continue
        looks_like = ("cam_K.txt" in files
                      or ("rgb" in subdirs and "depth" in subdirs))
        if looks_like:
            try:
                found.append(ReferenceDataset(directory, base=root))
            except DatasetError:
                continue
            # Do not descend into a dataset's own rgb/depth/masks folders.
            subdirs[:] = [d for d in subdirs
                          if d not in ("rgb", "depth", "masks", "mesh")]
    return found


def resolve_any_dataset_file(name: str) -> str:
    """Same as `resolve_any_dataset` but for a FILE — a mesh, typically."""
    for root in dataset_roots():
        try:
            resolved = resolve_dataset(name, root)
        except DatasetError:
            continue
        if os.path.isfile(resolved):
            return resolved
    raise DatasetError(f"no file {name!r} under "
                       + " or ".join(dataset_roots() or ["(none)"]))


def resolve_any_dataset(name: str) -> str:
    """Resolve a dataset name against EVERY root, in order.

    Two roots exist because a generated reference case cannot be mounted inside
    the read-only reference tree. A name is still unambiguous: the roots hold
    different material, and the first match wins.
    """
    problems = []
    for root in dataset_roots():
        try:
            resolved = resolve_dataset(name, root)
        except DatasetError as exc:
            problems.append(str(exc))
            continue
        if os.path.isdir(resolved):
            return resolved
    raise DatasetError(
        f"no dataset {name!r} under " + " or ".join(dataset_roots() or ["(none)"])
        + ("; " + "; ".join(problems) if problems else ""))


def resolve_dataset(name: str, root: str = DATASETS_DIR) -> str:
    """Turn a caller-supplied dataset name into a path INSIDE the mount.

    Names are now relative paths several segments long, so `..` is a plausible
    typo as well as a plausible probe. Resolved and checked rather than trusted:
    the mount is read-only, but a worker that will read any absolute path an
    HTTP caller names is still a worker that leaks the filesystem.
    """
    if os.path.isabs(name):
        raise DatasetError("name a dataset relative to the mount, not by "
                           "absolute path")
    resolved = os.path.realpath(os.path.join(root, name))
    base = os.path.realpath(root)
    if resolved != base and not resolved.startswith(base + os.sep):
        raise DatasetError(f"{name} resolves outside {root}")
    return resolved


class ReferenceDataset:
    """A FoundationPose demo directory: `cam_K.txt`, `rgb/`, `depth/`, `masks/`.

    This is upstream's own layout (`run_demo.py` + `YcbineoatReader`), reused
    verbatim rather than reinvented — the reference data already IS this shape,
    and inventing a WISEPACK format would mean a conversion step that could
    introduce exactly the unit and alignment errors the exercise is meant to
    detect.
    """

    def __init__(self, root: str, base: str = "") -> None:
        self.root = root
        #: The path a caller names in /estimate — relative to the mount, so a
        #: request never contains an absolute host path.
        self.name = (os.path.relpath(root, base) if base else
                     os.path.basename(root.rstrip("/")))
        if not os.path.isdir(root):
            raise DatasetError(f"{root} is not a directory")
        self.rgb_dir = os.path.join(root, "rgb")
        self.depth_dir = os.path.join(root, "depth")
        self.mask_dir = os.path.join(root, "masks")
        self.k_path = os.path.join(root, "cam_K.txt")

    # -- inventory --------------------------------------------------------- #

    def _listing(self, directory: str) -> List[str]:
        if not os.path.isdir(directory):
            return []
        return sorted(f for f in os.listdir(directory)
                      if f.lower().endswith((".png", ".jpg", ".jpeg")))

    @property
    def rgb_files(self) -> List[str]:
        return self._listing(self.rgb_dir)

    @property
    def depth_files(self) -> List[str]:
        return self._listing(self.depth_dir)

    @property
    def mask_files(self) -> List[str]:
        return self._listing(self.mask_dir)

    def intrinsics(self):
        """`K` as a 3x3 list. RAISES when absent — there is no default.

        Guessing intrinsics produces a pose that is wrong by a scale factor and
        looks entirely plausible, which is the worst kind of wrong.
        """
        if not os.path.isfile(self.k_path):
            raise DatasetError(
                f"{self.k_path} is missing. A pose cannot be expressed in "
                "metres without intrinsics, and they must not be guessed.")
        import numpy as np                                   # noqa: PLC0415
        return np.loadtxt(self.k_path).reshape(3, 3)

    def describe(self) -> Dict[str, Any]:
        rgb, depth, masks = self.rgb_files, self.depth_files, self.mask_files
        problems: List[str] = []
        if not rgb:
            problems.append("no rgb/ images")
        if not depth:
            problems.append("no depth/ images")
        if not masks:
            problems.append("no masks/ image (register() needs one)")
        if not os.path.isfile(self.k_path):
            problems.append("no cam_K.txt (intrinsics)")
        if rgb and depth and len(rgb) != len(depth):
            problems.append(f"{len(rgb)} rgb vs {len(depth)} depth images")
        return {
            "name": self.name,
            "root": self.root,
            "rgb_frames": len(rgb),
            "depth_frames": len(depth),
            "masks": len(masks),
            "has_intrinsics": os.path.isfile(self.k_path),
            "complete": not problems,
            "problems": problems,
        }

    # -- reading ------------------------------------------------------------ #

    def load(self, index: int, depth_scale_mm: float = 1000.0,
             mask_name: Optional[str] = None):
        """(rgb, depth_metres, mask_bool) for one frame.

        `depth_scale_mm` is how many MILLIMETRES one raw depth unit represents;
        FoundationPose works in metres, so the conversion happens here, once,
        explicitly. It is a PARAMETER because it is a property of the dataset,
        not of the code — a 16-bit PNG in millimetres and a float32 image in
        metres both occur, and assuming either silently is a factor-of-1000
        error.
        """
        import cv2                                           # noqa: PLC0415
        import numpy as np                                   # noqa: PLC0415

        rgb_files, depth_files = self.rgb_files, self.depth_files
        if not rgb_files or not depth_files:
            raise DatasetError(f"{self.root} has no rgb/depth frames")
        if not 0 <= index < min(len(rgb_files), len(depth_files)):
            raise DatasetError(
                f"frame {index} is out of range (0..{min(len(rgb_files), len(depth_files)) - 1})")

        colour = cv2.imread(os.path.join(self.rgb_dir, rgb_files[index]))
        if colour is None:
            raise DatasetError(f"could not decode rgb/{rgb_files[index]}")
        colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)

        raw = cv2.imread(os.path.join(self.depth_dir, depth_files[index]),
                         cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise DatasetError(f"could not decode depth/{depth_files[index]}")
        depth = raw.astype(np.float32) * (float(depth_scale_mm) / 1000.0)
        # ZERO IS "NO MEASUREMENT", not "at the lens". Left as zero, which is
        # what the estimator treats as invalid.
        depth[raw == 0] = 0.0

        mask = None
        names = self.mask_files
        if names:
            chosen = mask_name or names[0]
            mask_image = cv2.imread(os.path.join(self.mask_dir, chosen),
                                    cv2.IMREAD_UNCHANGED)
            if mask_image is None:
                raise DatasetError(f"could not decode masks/{chosen}")
            if mask_image.ndim == 3:
                mask_image = mask_image[..., 0]
            mask = mask_image.astype(bool)
            if not mask.any():
                raise DatasetError(
                    f"masks/{chosen} selects no pixels — register() has no "
                    "object region to work from")
            if mask.shape != depth.shape:
                raise DatasetError(
                    f"mask {mask.shape} does not match the image "
                    f"{depth.shape}; a mask from another resolution selects the "
                    "wrong pixels")
        return colour, depth, mask


# --------------------------------------------------------------------------- #
# The estimator, built lazily and kept
# --------------------------------------------------------------------------- #


class Estimator:
    """Owns the FoundationPose estimator for ONE mesh. Built on first use.

    Rebuilt when the mesh changes: FoundationPose binds the model at
    construction, and swapping meshes underneath a live estimator is not a
    supported operation upstream.
    """

    def __init__(self, capabilities: Capabilities) -> None:
        self._lock = threading.RLock()
        self._capabilities = capabilities
        self._estimator = None
        self._mesh = None
        self._mesh_key: Optional[Tuple[str, float]] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_images: Dict[str, bytes] = {}
        #: Why the pose overlay is missing, when it is. Kept separate from
        #: `last_error`, which is about estimation: a failed drawing must never
        #: be reported as a failed measurement.
        self.last_overlay_error: str = ""

    def _require_ready(self) -> None:
        snapshot = self._capabilities.snapshot()
        if not snapshot["inference_available"]:
            raise RuntimeError(
                "FoundationPose inference is not available: "
                + "; ".join(snapshot["blocked_by"] or ["unknown reason"]))

    def _load_mesh(self, mesh_path: str, scale_to_metres: float):
        """The mesh, scaled to METRES — the unit FoundationPose works in.

        THE SCALE IS SUPPLIED, NEVER SNIFFED. Neither OBJ nor STL records a
        unit; a millimetre mesh consumed as metres fits nothing and a metre mesh
        consumed as millimetres fits nothing, both without an error.
        """
        import trimesh                                       # noqa: PLC0415
        if not os.path.isfile(mesh_path):
            raise DatasetError(f"mesh {mesh_path} does not exist")
        mesh = trimesh.load(mesh_path, force="mesh")
        if scale_to_metres != 1.0:
            mesh.apply_scale(float(scale_to_metres))
        return mesh

    def estimator_for(self, mesh_path: str, scale_to_metres: float):
        with self._lock:
            key = (os.path.abspath(mesh_path), float(scale_to_metres))
            if self._estimator is not None and self._mesh_key == key:
                return self._estimator, self._mesh

            self._require_ready()
            mesh = self._load_mesh(mesh_path, scale_to_metres)

            # Imported HERE, not at module scope: this is what drags in torch,
            # pytorch3d, nvdiffrast and the CUDA extensions, and the worker must
            # start without them.
            import nvdiffrast.torch as dr                    # noqa: PLC0415
            from estimater import (FoundationPose,           # noqa: PLC0415
                                   PoseRefinePredictor, ScorePredictor)

            estimator = FoundationPose(
                model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                mesh=mesh, scorer=ScorePredictor(),
                refiner=PoseRefinePredictor(),
                glctx=dr.RasterizeCudaContext())
            self._estimator, self._mesh, self._mesh_key = estimator, mesh, key
            return estimator, mesh

    def register(self, mesh_path: str, scale_to_metres: float,
                 colour, depth, mask, intrinsics, refine_iterations: int = 5):
        """One registration. Returns the 4x4 `ob_in_cam`, in metres."""
        estimator, mesh = self.estimator_for(mesh_path, scale_to_metres)
        with self._lock:
            pose = estimator.register(K=intrinsics, rgb=colour, depth=depth,
                                      ob_mask=mask, iteration=refine_iterations)
        return pose, mesh


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _pose_payload(pose, mesh) -> Dict[str, Any]:
    """A 4x4 in metres -> the document WISEPACK consumes.

    MILLIMETRES OUT. WISEPACK's domain is millimetres everywhere; converting at
    this boundary means exactly one place does it.

    THE QUATERNION IS COMPUTED HERE and travels beside the matrix, so WISEPACK
    never has to re-derive a rotation and the two cannot drift.
    """
    import numpy as np                                       # noqa: PLC0415

    matrix = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    rotation = matrix[:3, :3]
    translation_m = matrix[:3, 3]

    # Shepperd's method, matching wisepack_core.pose.Orientation.from_matrix so
    # the two agree bit for bit on the same input.
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        w, x = 0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s
        y, z = (rotation[0, 2] - rotation[2, 0]) / s, (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = (1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) ** 0.5 * 2.0
        w, x = (rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s
        y, z = (rotation[0, 1] + rotation[1, 0]) / s, (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = (1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) ** 0.5 * 2.0
        w, x = (rotation[0, 2] - rotation[2, 0]) / s, (rotation[0, 1] + rotation[1, 0]) / s
        y, z = 0.25 * s, (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = (1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) ** 0.5 * 2.0
        w, x = (rotation[1, 0] - rotation[0, 1]) / s, (rotation[0, 2] + rotation[2, 0]) / s
        y, z = (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s
    norm = (x * x + y * y + z * z + w * w) ** 0.5

    extents = [float(v) for v in mesh.extents] if mesh is not None else None
    return {
        "frame_id": CAMERA_FRAME,
        "position_mm": [float(v) * 1000.0 for v in translation_m],
        "orientation": {"x": x / norm, "y": y / norm, "z": z / norm,
                        "w": w / norm},
        # The raw transform, for anyone who needs it. Metres, as the estimator
        # produced it — NOT converted, so nothing is lost to a round trip.
        "matrix_m": [[float(v) for v in row] for row in matrix],
        "mesh_extents_m": extents,
        # WHICH FRAME OF THE MESH the pose refers to. Upstream's `register()`
        # returns the pose of the mesh AS LOADED, not of its bounding-box
        # centre; stating it stops a comparison against another implementation
        # silently comparing two different frames.
        "pose_of": "mesh_origin_as_loaded",
        "estimated_at": utc_now(),
    }


def create_app(capabilities: Optional[Capabilities] = None):
    from fastapi import FastAPI, HTTPException               # noqa: PLC0415
    from fastapi.responses import Response                   # noqa: PLC0415

    capabilities = capabilities or Capabilities()
    estimator = Estimator(capabilities)
    app = FastAPI(title="WISEPACK FoundationPose worker")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """ALWAYS 200. What is missing is a field, never a transport error."""
        snapshot = capabilities.snapshot()
        snapshot["last_result_at"] = (estimator.last_result or {}).get(
            "estimated_at", "")
        return snapshot

    @app.get("/datasets")
    def datasets() -> Dict[str, Any]:
        """Every FoundationPose-shaped directory under the mount.

        SEARCHED, NOT ASSUMED FLAT. The reference material is a tree — the
        tutorial's demo set sits several levels down beside its ROS code and its
        Isaac assets — and requiring datasets at the top level would have meant
        copying them into a WISEPACK-shaped layout. `references/` is already
        beside the repository; duplicating 183 MB to satisfy a directory
        convention would be waste and a second copy to drift.
        """
        found = []
        for root in dataset_roots():
            found.extend(discover_datasets(root))
        recognised = {d.name.split(os.sep)[0] for d in found}
        # WHAT WAS SEARCHED AND NOT MATCHED, so nothing disappears quietly. A
        # recursive search that reports only its hits looks identical to a
        # search that ran on an empty mount.
        unrecognised = sorted(
            e for e in (os.listdir(DATASETS_DIR)
                        if os.path.isdir(DATASETS_DIR) else [])
            if os.path.isdir(os.path.join(DATASETS_DIR, e))
            and e not in recognised)
        return {"root": DATASETS_DIR,
                "roots": dataset_roots(),
                "search_depth": DATASET_SEARCH_DEPTH,
                "datasets": [d.describe() for d in found],
                "not_a_foundationpose_dataset": unrecognised}

    @app.get("/segmentation")
    def segmentation_methods() -> Dict[str, Any]:
        """Which mask sources exist, and which are only planned.

        Listed so the dashboard can say what does NOT exist yet, rather than
        implying `depth_plane_foreground` is the answer for a cluttered scene.
        """
        from segmentation import (DEFAULTS, METHODS,            # noqa: PLC0415
                                  PLANNED_METHODS, VALIDATION)
        return {
            "available": sorted(METHODS),
            "planned": PLANNED_METHODS,
            "supplied_with_dataset": "dataset",
            "defaults": DEFAULTS,
            "validation": VALIDATION,
        }

    @app.get("/camera")
    def camera_info(serial: str = ""):
        """The RGB-D device: model, serial, firmware, streams, depth scale.

        ANSWERS EVEN WITH NO CAMERA, with the reason — the same rule as
        /health. A 404 here would be indistinguishable from a broken worker.
        """
        from camera import CameraUnavailable, available, describe  # noqa: PLC0415
        usable, reason = available()
        if not usable:
            return {"available": False, "reason": reason, "device": None}
        try:
            return {"available": True, "reason": "", "device": describe(serial)}
        except CameraUnavailable as exc:
            return {"available": False, "reason": str(exc), "device": None}

    @app.post("/camera/capture")
    def camera_capture(request: Dict[str, Any]):
        """Capture a controlled RGB-D dataset for ONE known CAD part.

        `model_id` is REQUIRED and is never inferred from the image: which part
        is on the table is known because an operator put it there.
        """
        from camera import CameraUnavailable, capture_dataset  # noqa: PLC0415
        model_id = str(request.get("model_id", "")).strip()
        if not model_id:
            raise HTTPException(
                400, "`model_id` is required: which CAD part is in view. It is "
                     "known because you placed it there, and it is never "
                     "inferred from the image.")
        try:
            return capture_dataset(
                model_id=model_id,
                frames=int(request.get("frames", 30)),
                name=str(request.get("name", "")),
                serial=str(request.get("serial", "")),
                width=int(request.get("width", 0)),
                height=int(request.get("height", 0)),
                fps=int(request.get("fps", 0)),
                align=bool(request.get("align", True)))
        except CameraUnavailable as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/estimate")
    def estimate(request: Dict[str, Any]) -> Dict[str, Any]:
        """One registration against a mounted dataset.

        REFUSED WITH A REASON when anything is missing — never approximated,
        and never answered with a pose the worker could not actually compute.
        """
        snapshot = capabilities.snapshot()
        if not snapshot["inference_available"]:
            raise HTTPException(
                status_code=409,
                detail={"error": "inference is not available",
                        "blocked_by": snapshot["blocked_by"],
                        "health": {k: snapshot[k] for k in (
                            "gpu_available", "foundationpose_runtime_available",
                            "scorer_weights_available",
                            "refiner_weights_available")}})

        dataset_name = str(request.get("dataset", "")).strip()
        mesh_path = str(request.get("mesh_path", "")).strip()
        if not dataset_name:
            raise HTTPException(400, "a `dataset` is required")

        try:
            root = resolve_any_dataset(dataset_name)
            dataset = ReferenceDataset(root, base=DATASETS_DIR)
            index = int(request.get("frame", 0))
            # NO DEFAULT, DELIBERATELY. A uint16 image in millimetres and a
            # float32 image in metres are both ordinary, they are
            # indistinguishable from the pixels alone, and picking either
            # silently is a factor-of-1000 error that yields a confidently wrong
            # pose. The caller knows; the caller states it.
            if "depth_scale_mm" not in request:
                raise DatasetError(
                    "`depth_scale_mm` is required: how many millimetres one raw "
                    "depth unit represents (1.0 for a uint16 millimetre image, "
                    "1000.0 for a float32 metre image). It cannot be inferred "
                    "from the image and guessing it scales the pose.")
            depth_scale_mm = float(request["depth_scale_mm"])
            if depth_scale_mm <= 0:
                raise DatasetError("`depth_scale_mm` must be positive")
            colour, depth, mask = dataset.load(
                index, depth_scale_mm=depth_scale_mm,
                mask_name=request.get("mask"))
            # WHERE THE MASK COMES FROM, chosen explicitly and never guessed.
            #
            #   "dataset"                the mask supplied WITH the data. The
            #                            tutorial bolt regression uses this and
            #                            keeps using it: it is the known
            #                            reference input, and replacing it would
            #                            stop the regression testing what it was
            #                            built to test.
            #   "depth_plane_foreground" measured from the depth: fit the work
            #                            surface, keep what stands on it.
            #
            # NO SILENT FALLBACK between them. A mask whose provenance nobody
            # can state is the one thing `mask_source` exists to prevent.
            mask_source = str(request.get("mask_source", "dataset")).strip()
            segmentation_document: Dict[str, Any] = {"mask_source": "dataset"}
            if mask_source != "dataset":
                from segmentation import (SegmentationError,  # noqa: PLC0415
                                          segment)
                import numpy as np                        # noqa: PLC0415
                # `dataset.load()` has already converted depth to METRES;
                # segmentation works in millimetres, so the conversion happens
                # here, once, explicitly. Zero stays zero — it means "no
                # measurement", not "at the lens".
                depth_mm = (depth * 1000.0).astype(np.uint16)
                try:
                    segmented = segment(
                        mask_source, depth_mm, dataset.intrinsics(),
                        dict(request.get("segmentation") or {}))
                except SegmentationError as exc:
                    raise DatasetError(str(exc)) from exc
                segmentation_document = segmented.to_dict()
                if not segmented.valid:
                    raise DatasetError(
                        f"segmentation produced no usable mask: "
                        f"{segmented.reason}")
                mask = segmented.mask
            if mask is None:
                raise DatasetError(
                    f"{dataset_name} has no registration mask; FoundationPose "
                    "needs an object region for register()")
            intrinsics = dataset.intrinsics()
            if not mesh_path:
                raise DatasetError("a `mesh_path` is required")
            if os.path.isabs(mesh_path):
                raise DatasetError("name the mesh relative to the mount, not "
                                   "by absolute path")
            # RELATIVE TO THE MOUNT, not to the dataset. A mesh does not have to
            # live inside the frames it is registered against — WISEPACK's own
            # CAD models sit in a separate directory of the same reference tree,
            # and resolving against the dataset could not name them at all.
            mesh_path = resolve_any_dataset_file(mesh_path)
            scale = float(request.get("mesh_scale_to_metres", 1.0))
            iterations = int(request.get("refine_iterations", 5))
        except (DatasetError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

        started = time.monotonic()
        try:
            pose, mesh = estimator.register(
                mesh_path, scale, colour, depth, mask, intrinsics, iterations)
        except Exception as exc:                             # noqa: BLE001
            capabilities.last_error = f"inference failed: {exc}"
            raise HTTPException(500, capabilities.last_error) from exc

        payload = _pose_payload(pose, mesh)
        payload.update({
            "dataset": dataset_name,
            "frame_index": index,
            "frame_file": dataset.rgb_files[index],
            "mesh_path": mesh_path,
            "mesh_scale_to_metres": scale,
            "segmentation": segmentation_document,
            "refine_iterations": iterations,
            "depth_scale_mm": depth_scale_mm,
            "intrinsics": [[float(v) for v in row] for row in intrinsics],
            "duration_ms": round((time.monotonic() - started) * 1000.0, 1),
            "foundationpose_revision": capabilities.source_revision(),
            # NO SCORE IS REPORTED AS AN ACCURACY. The estimator's internal
            # score is a ranking statistic over pose hypotheses, not a distance
            # to a true pose, and there is no ground truth here.
            "accuracy_note": (
                "pose ESTIMATED. No ground truth exists for this dataset, so "
                "absolute pose accuracy is NOT measured; repeatability across "
                "frames is the only quantity available."),
        })
        estimator.last_result = payload
        _render_overlay(estimator, colour, depth, mask, pose, mesh, intrinsics)
        capabilities.last_error = ""
        return payload

    @app.get("/last-result")
    def last_result() -> Dict[str, Any]:
        if estimator.last_result is None:
            return {"status": "none",
                    "message": "no estimation has been requested yet"}
        return estimator.last_result

    @app.get("/image/{kind}")
    def image(kind: str):
        data = estimator.last_images.get(kind)
        if data is None:
            detail = f"no {kind!r} image yet — run an estimation first"
            if kind == "overlay" and estimator.last_overlay_error:
                detail = ("the pose overlay could not be drawn: "
                          f"{estimator.last_overlay_error}. The estimate itself "
                          "is unaffected; see /last-result.")
            raise HTTPException(404, detail)
        return Response(data, media_type="image/jpeg")

    return app


def _render_overlay(estimator: Estimator, colour, depth, mask, pose, mesh,
                    intrinsics) -> None:
    """Diagnostic images. FAILURE HERE MUST NOT FAIL AN ESTIMATION.

    The overlay is how an operator sees whether a pose is plausible, but it is a
    convenience: losing the picture must not lose the measurement.
    """
    try:
        import cv2                                           # noqa: PLC0415
        import numpy as np                                   # noqa: PLC0415

        images: Dict[str, bytes] = {}
        bgr = cv2.cvtColor(colour, cv2.COLOR_RGB2BGR)
        images["rgb"] = cv2.imencode(".jpg", bgr)[1].tobytes()

        finite = depth[depth > 0]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            scaled = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
            visual = cv2.applyColorMap((scaled * 255).astype(np.uint8),
                                       cv2.COLORMAP_TURBO)
            visual[depth <= 0] = 0
            images["depth"] = cv2.imencode(".jpg", visual)[1].tobytes()

        if mask is not None:
            images["mask"] = cv2.imencode(
                ".jpg", (mask.astype(np.uint8) * 255))[1].tobytes()

        # The estimated pose drawn on the frame: upstream's own helpers, so the
        # picture matches what the reference implementation would show.
        try:
            import trimesh                                   # noqa: PLC0415
            from Utils import draw_posed_3d_box, draw_xyz_axis  # noqa: PLC0415
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            centre = np.asarray(pose) @ np.linalg.inv(to_origin)
            overlay = draw_posed_3d_box(intrinsics, img=bgr.copy(),
                                        ob_in_cam=centre, bbox=bbox)
            # THE AXIS IS SCALED TO THE OBJECT, not to a fixed 5 cm. A triad
            # longer than the part it describes hides the part; on a 49 mm bolt
            # at 780 mm the difference is the whole picture.
            axis_m = float(max(extents)) * 0.75
            overlay = draw_xyz_axis(overlay, ob_in_cam=centre, scale=axis_m,
                                    K=intrinsics, thickness=2, transparency=0,
                                    is_input_rgb=False)
            # THE MASK OUTLINE, because the box alone cannot show WHICH object
            # was registered. In a bin of four near-identical bolts, a pose on
            # the wrong one looks exactly as convincing as a pose on the right
            # one until the mask is drawn beside it.
            if mask is not None:
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
            images["overlay"] = cv2.imencode(".jpg", overlay)[1].tobytes()
        except Exception as exc:                             # noqa: BLE001
            # NOT SILENT. The estimate still stands — but an operator staring at
            # a missing overlay deserves the reason rather than a blank panel.
            estimator.last_overlay_error = f"{type(exc).__name__}: {exc}"

        estimator.last_images = images
    except Exception:                                        # noqa: BLE001
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("WISEPACK_FP_PORT",
                                                   DEFAULT_PORT)))
    parser.add_argument("--check", action="store_true",
                        help="print the capability snapshot and exit")
    args = parser.parse_args()

    capabilities = Capabilities()
    if args.check:
        print(json.dumps(capabilities.snapshot(), indent=2))
        return

    # THE SNAPSHOT IS PRINTED AT START-UP, always, so `docker logs` answers
    # "why can it not infer" without anyone having to curl anything.
    snapshot = capabilities.snapshot()
    print("[fp-worker] " + json.dumps(
        {k: snapshot[k] for k in (
            "worker_ready", "gpu_available",
            "foundationpose_runtime_available", "scorer_weights_available",
            "refiner_weights_available", "inference_available")}))
    for reason in snapshot["blocked_by"]:
        print(f"[fp-worker] blocked: {reason}")

    import uvicorn                                           # noqa: PLC0415
    uvicorn.run(create_app(capabilities), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
