"""Instance masks for FoundationPose — one binary mask, however it was produced.

THE ABSTRACTION IS THE POINT
----------------------------
`register()` needs one thing from segmentation: a binary mask of the object. It
must not care how that mask was obtained, because the way it is obtained changes
as the scene does:

    depth_plane_foreground   ONE known object on a stable surface. Geometric:
                             fit the work surface, keep what stands on it.
    yolo_instance            (not implemented) a cluttered scene of touching
                             pipes, where plane removal alone yields one
                             connected blob containing several objects.

So a method returns a `SegmentationResult` and nothing downstream branches on
which method ran. Adding the second one is a new entry in `METHODS`.

WHAT `depth_plane_foreground` IS, AND WHAT IT IS NOT
---------------------------------------------------
It IS a measurement derived from RGB-D geometry: deproject the depth with the
camera's own intrinsics, fit the dominant plane, and keep what stands above it
inside the configured work area.

It is NOT a ground-truth mask, and it is NOT clutter segmentation. On a table of
touching pipes it will happily return one component containing several objects —
which is exactly why it is named after its mechanism rather than its purpose.

NO ABSOLUTE DEPTH THRESHOLD ANYWHERE. `depth < 0.7 m` encodes where the table
happened to be when someone wrote it; move the camera and it silently selects
the wrong thing. Every threshold here is relative to the MEASURED plane.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

#: Fit parameters. Deliberately few, and every one is a distance from a measured
#: surface rather than a distance from the camera.
DEFAULTS: Dict[str, Any] = {
    #: A point within this of the fitted plane IS the plane. Roughly the depth
    #: noise of a D4xx at table range; too small and the table survives as
    #: speckle, too large and a flat-lying pipe is absorbed into it.
    "plane_tolerance_mm": 6.0,
    #: Minimum height above the plane for a point to be object. Separate from
    #: the tolerance: it is the clearance a real object must show before its
    #: pixels are trusted.
    "min_height_mm": 8.0,
    #: Ceiling for the work volume — an operator's hand, or a wall behind the
    #: table, is not the object.
    "max_height_mm": 400.0,
    #: Optional lateral bound around the plane centroid, in millimetres. None
    #: means "the whole plane".
    "roi_radius_mm": None,
    #: OPTIONAL OPERATOR ROI, in COLOUR-IMAGE PIXELS: [x0, y0, x1, y1], or None
    #: for the whole frame.
    #:
    #: WHAT IT IS: a statement of WHERE TO LOOK, given by whoever set the scene
    #: up. On a bench that holds one intended object and a dozen unrelated ones,
    #: the single-object rules below are the right rules — they were simply
    #: being applied to the whole room instead of to the part of it the operator
    #: meant.
    #:
    #: WHAT IT IS NOT, and this is the line that matters: it does NOT identify
    #: the object. Which CAD part is in the window is `model_id`, stated by the
    #: operator, exactly as it is without an ROI. A window says "the thing I
    #: mean is in here"; it cannot say what that thing is, and nothing here
    #: derives one from the other. In particular no ROI is ever computed from a
    #: CAD model's dimensions — that would be identifying the object by its
    #: shape and then measuring its pose with the same shape.
    "roi_px": None,
    #: RANSAC effort. Fixed iterations and a fixed seed, so the same frame gives
    #: the same plane every time — a segmentation that wanders between runs
    #: would make pose repeatability meaningless.
    "ransac_iterations": 200,
    "ransac_seed": 0,
    #: A plane must account for at least this share of valid depth pixels, or
    #: there is no dominant surface and the scene is not what this method
    #: assumes.
    "min_plane_inlier_fraction": 0.25,
    #: Component selection for the controlled single-object test.
    "component": "largest",           # "largest" | "centre"
    #: Morphology: close holes only. A large kernel would fuse neighbouring
    #: objects, which is precisely the failure this method must not hide.
    "close_kernel_px": 5,
    "min_component_area_px": 200,
}

#: Mask validation limits. A mask that passes these is not necessarily right; a
#: mask that fails one is definitely unusable, and saying which is the point.
VALIDATION: Dict[str, Any] = {
    "min_area_fraction": 0.0005,      # negligible: ~300 px in a 1280x720 frame
    "max_area_fraction": 0.60,        # nearly the whole image
    "min_valid_depth_fraction": 0.5,  # depth must exist inside the mask
    "max_components": 3,              # disconnected/noisy beyond this
    "min_separation_mm": 8.0,         # object must stand off the plane
}


class SegmentationError(RuntimeError):
    """Segmentation that cannot produce a usable mask, with the reason."""


class SegmentationResult:
    """One binary mask and everything needed to judge it.

    `valid` is FALSE with a reason rather than an exception, because a bad mask
    is a state the operator has to see — and because the diagnostics are most
    useful precisely when the mask failed.
    """

    def __init__(self, mask: Any, method: str, diagnostics: Dict[str, Any],
                 valid: bool = True, reason: str = "") -> None:
        self.mask = mask
        self.method = method
        self.diagnostics = diagnostics
        self.valid = valid
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            # NAMED AFTER ITS MECHANISM. Nothing downstream may read this as
            # "ground truth"; it is a measurement with a stated provenance.
            "mask_source": self.method,
            "mask_valid": self.valid,
            "reason": self.reason,
            **self.diagnostics,
        }


# --------------------------------------------------------------------------- #
# Plane fitting
# --------------------------------------------------------------------------- #


def deproject(depth_mm: Any, intrinsics: Any) -> Tuple[Any, Any]:
    """Valid depth pixels -> (points Nx3 in millimetres, their flat indices).

    Uses the camera's OWN intrinsics. A pinhole deprojection with guessed fx/cx
    produces a point cloud that is wrong by a scale and a shear, and a plane fit
    to it looks perfectly healthy.
    """
    import numpy as np                                       # noqa: PLC0415

    k = np.asarray(intrinsics, dtype=np.float64)
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    height, width = depth_mm.shape[:2]
    vs, us = np.mgrid[0:height, 0:width]
    z = depth_mm.astype(np.float64)
    # ZERO IS "NO MEASUREMENT", not "at the lens".
    valid = z > 0
    flat = np.flatnonzero(valid)
    zz = z[valid]
    xx = (us[valid] - cx) * zz / fx
    yy = (vs[valid] - cy) * zz / fy
    return np.stack([xx, yy, zz], axis=1), flat


def fit_plane(points: Any, tolerance_mm: float, iterations: int,
              seed: int) -> Tuple[Optional[Any], Any, float]:
    """Robust dominant-plane fit. Returns (plane, inlier mask, rms residual).

    RANSAC, then a least-squares refit on the inliers: RANSAC alone gives a
    plane defined by three points and inherits their noise, and the refit is
    what makes the residual meaningful as a quality figure.

    DETERMINISTIC — fixed iterations, fixed seed. A plane that wanders between
    runs would make pose repeatability a measurement of the segmentation.
    """
    import numpy as np                                       # noqa: PLC0415

    if points.shape[0] < 3:
        return None, np.zeros(points.shape[0], dtype=bool), 0.0

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(points.shape[0], dtype=bool)
    best_count = 0
    for _ in range(max(1, iterations)):
        sample = rng.choice(points.shape[0], size=3, replace=False)
        a, b, c = points[sample]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue                       # collinear sample: no plane
        normal = normal / length
        distances = np.abs((points - a) @ normal)
        inliers = distances <= tolerance_mm
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_count < 3:
        return None, best_inliers, 0.0

    # Least-squares refit on the inliers: the plane through their centroid whose
    # normal is the smallest singular vector.
    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    offset = float(normal @ centroid)
    signed = points @ normal - offset
    refined = np.abs(signed) <= tolerance_mm
    residual = float(np.sqrt(np.mean((points[refined] @ normal - offset) ** 2))
                     ) if refined.any() else 0.0
    return (normal, offset), refined, residual


# --------------------------------------------------------------------------- #
# depth_plane_foreground
# --------------------------------------------------------------------------- #


METHOD_DEPTH_PLANE = "depth_plane_foreground"


def _extent_mm(mask: Any, depth_mm: Any, intrinsics: Any) -> Dict[str, Any]:
    """How big the selected region actually is, in millimetres.

    MEASURED FROM THE DEPTH, NEVER FROM A CAD MODEL. It exists so an operator
    can see that the region they are about to hand to a pose estimator is the
    size of the thing they meant — a 200 mm blob where a 342 mm tube was
    expected is the cheapest possible warning that the wrong object was
    selected.

    NOTHING CONSUMES THIS. It is a diagnostic printed for a human; using it to
    pick a component would be identifying an object by its dimensions, which is
    exactly what the registry forbids and what `model_id` exists to state.

    The extents are the principal axes of the masked pixels scaled to the
    region's median range — an approximation good enough to tell 200 mm from
    340 mm, and not a measurement of the object.
    """
    import numpy as np                                        # noqa: PLC0415

    rows, columns = np.nonzero(mask)
    if rows.size < 50:
        return {}
    ranges = depth_mm[rows, columns].astype(np.float64)
    ranges = ranges[ranges > 0]
    if ranges.size < 50:
        return {}
    median = float(np.median(ranges))
    pixels = np.stack([columns, rows], axis=1).astype(np.float64)
    pixels -= pixels.mean(axis=0)
    singular = np.linalg.svd(pixels, compute_uv=False)
    # 2*sqrt(3)*sigma is the full width of a uniform bar of that spread.
    spread = 2.0 * math.sqrt(3.0) * singular / math.sqrt(pixels.shape[0])
    fx = float(intrinsics[0][0])
    fy = float(intrinsics[1][1])
    return {
        "mask_median_range_mm": round(median, 1),
        "mask_extent_long_mm": round(float(spread[0]) * median / fx, 1),
        "mask_extent_across_mm": round(float(spread[1]) * median / fy, 1),
        "extent_note": ("measured from depth, NOT from any CAD model, and not "
                        "used to select anything — a diagnostic for a human"),
    }


def depth_plane_foreground(depth_mm: Any, intrinsics: Any,
                           options: Optional[Dict[str, Any]] = None,
                           colour_shape: Optional[Tuple[int, int]] = None
                           ) -> SegmentationResult:
    """Segment ONE object standing on the dominant work surface.

    The stages, in order, each of which can fail with its own reason:

        deproject valid depth with the real intrinsics
        fit the dominant plane robustly, on the WHOLE frame
        drop plane points
        keep foreground ABOVE the plane and inside the work volume
        restrict to the operator's ROI, if one was given
        project back into the aligned colour image
        select one connected component
        close small holes only

    THE ROI DOES NOT IDENTIFY THE OBJECT. It restricts WHERE this method looks,
    so the single-object rules apply to the region an operator pointed at rather
    than to every unrelated thing on the bench. Which part is in that region is
    `model_id`, and it is stated, never inferred — see `roi_px` in DEFAULTS.
    """
    import cv2                                               # noqa: PLC0415
    import numpy as np                                       # noqa: PLC0415

    settings = dict(DEFAULTS)
    settings.update(options or {})
    started = time.monotonic()
    height, width = depth_mm.shape[:2]
    diagnostics: Dict[str, Any] = {
        "method": METHOD_DEPTH_PLANE,
        "image_size": [int(width), int(height)],
        "settings": {k: settings[k] for k in
                     ("plane_tolerance_mm", "min_height_mm", "max_height_mm",
                      "roi_radius_mm", "roi_px", "component")},
        "plane_detected": False,
        "plane_residual_mm": None,
        "plane_inlier_fraction": None,
        "foreground_points": 0,
        "mask_pixels": 0,
        "components": 0,
        "selected_component": None,
    }
    empty = np.zeros((height, width), dtype=bool)

    points, flat_index = deproject(depth_mm, intrinsics)
    diagnostics["valid_depth_pixels"] = int(points.shape[0])
    if points.shape[0] < 100:
        return SegmentationResult(
            empty, METHOD_DEPTH_PLANE, diagnostics, False,
            "the depth image has almost no valid pixels; there is nothing to "
            "segment. Check that the depth stream is running and that the "
            "scene is within the sensor's range.")

    plane, inliers, residual = fit_plane(
        points, float(settings["plane_tolerance_mm"]),
        int(settings["ransac_iterations"]), int(settings["ransac_seed"]))
    if plane is None:
        return SegmentationResult(
            empty, METHOD_DEPTH_PLANE, diagnostics, False,
            "no plane could be fitted to the depth data. This method assumes a "
            "dominant flat work surface; it is not a general segmenter.")

    normal, offset = plane
    fraction = float(inliers.mean())
    diagnostics.update({
        "plane_detected": True,
        "plane_residual_mm": round(residual, 4),
        "plane_inlier_fraction": round(fraction, 4),
        "plane_normal": [round(float(v), 6) for v in normal],
    })
    if fraction < float(settings["min_plane_inlier_fraction"]):
        return SegmentationResult(
            empty, METHOD_DEPTH_PLANE, diagnostics, False,
            f"the best plane explains only {fraction:.1%} of valid depth "
            "pixels, so there is no dominant work surface. Point the camera at "
            "the tabletop, or use a segmentation method that does not assume "
            "one.")

    # HEIGHT ABOVE THE PLANE, SIGNED SO "UP" IS TOWARDS THE CAMERA.
    #
    # The normal's direction out of the SVD is arbitrary, so it is oriented
    # GEOMETRICALLY: the camera is at the origin, so the normal points towards
    # it exactly when `offset` is negative. With that convention a point closer
    # to the camera than the surface — an object standing on it — has a
    # positive signed distance, always.
    #
    # An earlier version compared the MEDIAN signed distance against zero, which
    # cannot work: in a scene dominated by the surface itself that median is
    # ~0 by construction, so the test was reading noise and the objects were
    # discarded as being below the table.
    if offset > 0:
        normal, offset = -normal, -offset
        diagnostics["plane_normal"] = [round(float(v), 6) for v in normal]
    signed = points @ normal - offset

    above = (signed >= float(settings["min_height_mm"])) & \
            (signed <= float(settings["max_height_mm"]))

    radius = settings.get("roi_radius_mm")
    if radius:
        # A WORK-AREA BOUND, measured on the plane rather than in camera
        # coordinates: distance from the plane centroid, projected onto it.
        centroid = points[inliers].mean(axis=0)
        offsets = points - centroid
        lateral = offsets - np.outer(offsets @ normal, normal)
        above &= np.linalg.norm(lateral, axis=1) <= float(radius)
        diagnostics["roi_radius_mm"] = float(radius)

    # THE OPERATOR'S WINDOW, applied AFTER the plane fit and BEFORE anything is
    # counted. Two consequences, both deliberate:
    #
    #   the plane stays a FULL-FRAME measurement. Fitting it inside a narrow
    #   window would estimate the work surface from a sliver of tabletop, and a
    #   worse plane moves every height threshold that follows.
    #
    #   the single-object rules then apply INSIDE the window. The component
    #   count, the selection and every validity limit see only what the operator
    #   pointed at, which is what makes "one object on a surface" a true
    #   statement about a bench that also holds a dozen other things.
    box = settings.get("roi_px")
    if box:
        try:
            x0, y0, x1, y1 = (int(round(float(v))) for v in box)
        except (TypeError, ValueError) as exc:
            raise SegmentationError(
                f"roi_px must be [x0, y0, x1, y1] in colour-image pixels, got "
                f"{box!r} ({exc})") from exc
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise SegmentationError(
                f"roi_px [{x0}, {y0}, {x1}, {y1}] is empty after clamping to "
                f"the {width}x{height} image; nothing could be segmented in it")
        rows, columns = np.divmod(flat_index, width)
        inside = (columns >= x0) & (columns < x1) & (rows >= y0) & (rows < y1)
        above &= inside
        diagnostics["roi_px"] = [x0, y0, x1, y1]
        diagnostics["roi_area_px"] = int((x1 - x0) * (y1 - y0))
        # WHAT WAS EXCLUDED, stated as a number. An ROI that quietly discarded
        # most of the object would otherwise look like a clean segmentation.
        diagnostics["foreground_points_full_frame"] = int(
            ((signed >= float(settings["min_height_mm"]))
             & (signed <= float(settings["max_height_mm"]))).sum())

    diagnostics["foreground_points"] = int(above.sum())
    if not above.any():
        where = (f" inside the ROI {diagnostics['roi_px']}" if box else "")
        elsewhere = (
            f" {diagnostics['foreground_points_full_frame']} such points exist "
            "in the FULL frame, so the ROI is pointing somewhere else."
            if box and diagnostics.get("foreground_points_full_frame") else "")
        return SegmentationResult(
            empty, METHOD_DEPTH_PLANE, diagnostics, False,
            f"no points{where} stand between {settings['min_height_mm']} mm and "
            f"{settings['max_height_mm']} mm above the fitted surface. Either "
            "no object is present, or it is lying flatter than the minimum "
            f"height.{elsewhere}")

    # Project the surviving points back into the ALIGNED colour image. This is
    # a scatter, not a reprojection: depth is already aligned to colour, so a
    # depth pixel and a colour pixel share an index.
    foreground = np.zeros(height * width, dtype=np.uint8)
    foreground[flat_index[above]] = 255
    foreground = foreground.reshape(height, width)

    # CLOSE SMALL HOLES ONLY. A larger kernel would bridge neighbouring objects,
    # which is exactly the failure this method must not paper over.
    kernel_size = int(settings["close_kernel_px"])
    if kernel_size >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground, connectivity=8)
    areas = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)
             if stats[i, cv2.CC_STAT_AREA] >= int(settings["min_component_area_px"])]
    diagnostics["components"] = len(areas)
    if not areas:
        return SegmentationResult(
            empty, METHOD_DEPTH_PLANE, diagnostics, False,
            "no connected foreground component is large enough to be an "
            f"object (minimum {settings['min_component_area_px']} px).")

    if settings["component"] == "centre":
        target = np.array([width / 2.0, height / 2.0])
        _, chosen = min((float(np.linalg.norm(centroids[i] - target)), i)
                        for _area, i in areas)
    else:
        _, chosen = max(areas)

    mask = labels == chosen
    diagnostics.update({
        "selected_component": int(chosen),
        "component_areas_px": sorted((a for a, _ in areas), reverse=True),
        "mask_pixels": int(mask.sum()),
        "mask_centroid_px": [round(float(v), 2) for v in centroids[chosen]],
        "duration_ms": round((time.monotonic() - started) * 1000.0, 2),
    })
    diagnostics.update(_extent_mm(mask, depth_mm, intrinsics))

    valid, reason = validate_mask(mask, depth_mm, points, flat_index, above,
                                  signed, diagnostics, settings)
    return SegmentationResult(mask, METHOD_DEPTH_PLANE, diagnostics, valid,
                              reason)


def validate_mask(mask: Any, depth_mm: Any, points: Any, flat_index: Any,
                  above: Any, signed: Any, diagnostics: Dict[str, Any],
                  settings: Dict[str, Any]) -> Tuple[bool, str]:
    """Reject masks that are obviously unusable, and say which check failed.

    Run BEFORE FoundationPose, because `register()` given a mask of the tabletop
    returns a pose rather than an error — a confident answer about the wrong
    pixels.
    """
    import numpy as np                                       # noqa: PLC0415

    limits = dict(VALIDATION)
    limits.update({k: settings[k] for k in VALIDATION if k in settings})
    height, width = mask.shape[:2]
    area = int(mask.sum())
    fraction = area / float(height * width)
    diagnostics["mask_area_fraction"] = round(fraction, 6)

    if area == 0:
        return False, "the mask is empty"
    if fraction < float(limits["min_area_fraction"]):
        return False, (f"the mask covers {fraction:.4%} of the image, which is "
                       "negligible — too small to be the object")
    if fraction > float(limits["max_area_fraction"]):
        return False, (f"the mask covers {fraction:.1%} of the image, which is "
                       "nearly all of it — the work surface was probably not "
                       "removed")

    inside = depth_mm[mask]
    valid_depth = float((inside > 0).mean()) if inside.size else 0.0
    diagnostics["valid_depth_fraction_in_mask"] = round(valid_depth, 4)
    if valid_depth < float(limits["min_valid_depth_fraction"]):
        return False, (f"only {valid_depth:.1%} of the masked pixels have valid "
                       "depth; FoundationPose cannot register against mostly "
                       "missing geometry")

    if diagnostics.get("components", 0) > int(limits["max_components"]):
        return False, (f"the foreground broke into {diagnostics['components']} "
                       "components, which is more disconnected than the "
                       "single-object assumption allows")

    # SEPARATION FROM THE PLANE, on the selected component only: a component can
    # pass every pixel-count test and still be a sliver of tabletop noise.
    selected = np.zeros(height * width, dtype=bool)
    selected[flat_index[above]] = True
    selected = selected.reshape(height, width) & mask
    heights = signed[above][selected.reshape(-1)[flat_index[above]]] \
        if selected.any() else np.array([])
    if heights.size:
        median_height = float(np.median(heights))
        diagnostics["median_height_above_plane_mm"] = round(median_height, 2)
        if median_height < float(limits["min_separation_mm"]):
            return False, (f"the selected component sits only {median_height:.1f} "
                           "mm above the fitted surface, which is not enough to "
                           "distinguish it from the surface itself")
    return True, ""


#: The provider registry. FoundationPose consumes the mask and never reads this.
METHODS = {METHOD_DEPTH_PLANE: depth_plane_foreground}

#: Named but NOT implemented. Listed so the abstraction is visible and so the
#: dashboard can say what does not exist yet, rather than implying that
#: `depth_plane_foreground` is the final answer for a cluttered scene.
PLANNED_METHODS = {
    "yolo_instance": (
        "instance segmentation for the cluttered multi-pipe scene, where plane "
        "removal alone yields one connected component containing several "
        "objects. Needs a model trained on the WISEPACK pipe geometry; see "
        "perception/foundationpose/SEGMENTATION.md."),
}


def segment(method: str, depth_mm: Any, intrinsics: Any,
            options: Optional[Dict[str, Any]] = None) -> SegmentationResult:
    """Run one named segmentation method. NO SILENT FALLBACK.

    An unknown or unimplemented method is an error. Falling back to another one
    would produce a mask whose provenance nobody could state, and the whole
    reason `mask_source` is reported is that the provenance matters.
    """
    if method in PLANNED_METHODS:
        raise SegmentationError(
            f"the {method!r} segmentation method is not implemented: "
            + PLANNED_METHODS[method])
    if method not in METHODS:
        raise SegmentationError(
            f"unknown segmentation method {method!r}; available: "
            + ", ".join(sorted(METHODS)))
    return METHODS[method](depth_mm, intrinsics, options)


__all__ = ["SegmentationResult", "SegmentationError", "segment", "METHODS",
           "PLANNED_METHODS", "METHOD_DEPTH_PLANE", "depth_plane_foreground",
           "fit_plane", "deproject", "validate_mask", "DEFAULTS", "VALIDATION"]
