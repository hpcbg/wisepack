"""Whole-scene instance segmentation on the fitted work plane, from RGB-D.

WHAT IT MEASURES. One aligned RGB-D frame of the bench, seen from above. The
work plane is fitted to the depth image (RANSAC + least-squares refit); every
connected region that either stands above that plane or has the bluish-grey
colour of bare steel against the bench is one INSTANCE. For each instance the
mask pixels are intersected with the plane (a ray-plane intersection per pixel,
which does not depend on the depth noise of a small part) and the resulting
plane points give the footprint: centroid, principal axis, length along it,
width across it. Height above the plane is the 95th percentile of the depth
residuals inside the mask.

WHAT IT DOES NOT DO. It does not recognise anything and it does not estimate a
full 6-DoF pose: an object is a footprint on a plane plus a heading. Objects
that lean, stack or overlap are measured as one region and reported as such.
The whole module is NumPy + OpenCV so it runs on the dashboard host without the
GPU worker; the worker's single-object `depth_plane_foreground` is untouched.

NOTHING IS FILLED IN. A region that cannot be measured is returned with a
reason and no numbers; the caller decides what to do with it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                                         # pragma: no cover
    import cv2
except ImportError:                                          # pragma: no cover
    cv2 = None  # type: ignore[assignment]

#: Defaults, all overridable through `options`. Millimetres and pixels.
DEFAULTS: Dict[str, Any] = {
    # Plane fit.
    "plane_tolerance_mm": 6.0,
    "ransac_iterations": 300,
    "ransac_samples": 60000,
    "ransac_seed": 0,
    "min_plane_inlier_fraction": 0.25,
    # Height band that counts as "standing on the plane".
    "min_height_mm": 5.0,
    "max_height_mm": 300.0,
    # Colour rule: bare steel on this bench reflects the sky-lit room and reads
    # BLUE-GREY (OpenCV hue 90-105) against the orange cork (hue 25-60). Hue is
    # what survives the D435's auto white balance drifting between captures —
    # a blue-minus-red threshold did not. A pixel in the metal hue band with a
    # little saturation is foreground even where the depth cannot see a 5 mm
    # bolt. Disable with `colour_foreground: false` on a different bench.
    "colour_foreground": True,
    "metal_hue_range": [80, 125],
    "metal_min_saturation": 20,
    "metal_min_value": 40,
    # Morphology and component filtering.
    "open_kernel_px": 3,
    "close_kernel_px": 5,
    "min_component_area_px": 300,
    "max_component_area_fraction": 0.20,
    # Optional operator ROI [x0, y0, x1, y1] in colour pixels: only components
    # whose centroid lies inside are kept.
    "roi_px": None,
    # Regions to ignore, [{"rect": [x0, y0, x1, y1], "note": "..."}].
    "ignore_regions_px": [],
}


class SceneSegmentationError(RuntimeError):
    """The frame could not be segmented at all (no plane, no depth)."""


@dataclass
class Plane:
    """`n . p + d = 0` in the camera frame, `n` pointing TOWARD the camera."""

    normal: Tuple[float, float, float]
    d: float
    residual_mm: float
    inlier_fraction: float

    def height(self, points: np.ndarray) -> np.ndarray:
        """Signed distance above the plane, toward the camera, millimetres."""
        return points @ np.asarray(self.normal) + self.d

    def to_dict(self) -> Dict[str, Any]:
        return {"normal": [round(float(v), 5) for v in self.normal],
                "d_mm": round(float(self.d), 2),
                "residual_mm": round(float(self.residual_mm), 2),
                "inlier_fraction": round(float(self.inlier_fraction), 3)}


@dataclass
class Instance:
    """One connected region on the plane and its measured footprint."""

    index: int
    area_px: int
    centroid_px: Tuple[float, float]
    bbox_px: Tuple[int, int, int, int]
    #: Footprint, in the camera frame, on the plane.
    centre_mm: Tuple[float, float, float]
    axis_line: Tuple[float, float, float]
    length_mm: float
    width_mm: float
    height_mm: float
    height_median_mm: float
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    #: Set when the region was set aside, with the reason.
    ignored: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "area_px": self.area_px,
            "centroid_px": [round(v, 1) for v in self.centroid_px],
            "bbox_px": list(self.bbox_px),
            "centre_mm": [round(v, 2) for v in self.centre_mm],
            "axis_line": [round(v, 5) for v in self.axis_line],
            "length_mm": round(self.length_mm, 1),
            "width_mm": round(self.width_mm, 1),
            "height_mm": round(self.height_mm, 1),
            "height_median_mm": round(self.height_median_mm, 1),
            "ignored": self.ignored,
        }


@dataclass
class SceneSegmentation:
    plane: Plane
    instances: List[Instance]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"plane": self.plane.to_dict(),
                "instances": [i.to_dict() for i in self.instances],
                **self.diagnostics}


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #


def load_capture(root: str, frame: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bgr uint8 HxWx3, depth_mm uint16 HxW, K 3x3) from a capture directory.

    The layout is the one the FoundationPose worker's `capture_dataset` writes:
    `rgb/000000.png`, `depth/000000.png` (uint16 millimetres) and `cam_K.txt`.
    """
    if cv2 is None:
        raise SceneSegmentationError("OpenCV (cv2) is required to read the capture")
    rgb_path = os.path.join(root, "rgb", f"{frame:06d}.png")
    depth_path = os.path.join(root, "depth", f"{frame:06d}.png")
    k_path = os.path.join(root, "cam_K.txt")
    for path in (rgb_path, depth_path, k_path):
        if not os.path.isfile(path):
            raise SceneSegmentationError(f"capture is incomplete: {path} is missing")
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        raise SceneSegmentationError(f"capture frames under {root} could not be decoded")
    if depth.ndim != 2:
        raise SceneSegmentationError("the depth image is not single-channel")
    if depth.shape != bgr.shape[:2]:
        raise SceneSegmentationError(
            f"depth {depth.shape} and colour {bgr.shape[:2]} differ in size; the "
            "capture is not aligned")
    K = np.loadtxt(k_path, dtype=float).reshape(3, 3)
    return bgr, depth.astype(np.uint16), K


# --------------------------------------------------------------------------- #
# Plane
# --------------------------------------------------------------------------- #


def deproject(depth_mm: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Camera-frame points (HxWx3, mm) and the valid-depth mask."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth_mm.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth_mm.astype(np.float64)
    valid = z > 0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1), valid


def fit_plane(points: np.ndarray, valid: np.ndarray, settings: Dict[str, Any]) -> Plane:
    """RANSAC on a subsample, least-squares refit on the inliers."""
    candidates = points[valid]
    if len(candidates) < 100:
        raise SceneSegmentationError(
            f"only {len(candidates)} valid depth pixels; no plane can be fitted")
    rng = np.random.default_rng(int(settings["ransac_seed"]))
    count = min(len(candidates), int(settings["ransac_samples"]))
    sample = candidates[rng.choice(len(candidates), count, replace=False)]
    tolerance = float(settings["plane_tolerance_mm"])
    best: Optional[Tuple[int, np.ndarray, float]] = None
    for _ in range(int(settings["ransac_iterations"])):
        tri = sample[rng.choice(count, 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -float(n @ tri[0])
        inliers = int((np.abs(sample @ n + d) < tolerance).sum())
        if best is None or inliers > best[0]:
            best = (inliers, n, d)
    if best is None:
        raise SceneSegmentationError("no plane hypothesis could be formed")
    _, n, d = best
    inlier_mask = np.abs(sample @ n + d) < tolerance
    fraction = float(inlier_mask.mean())
    if fraction < float(settings["min_plane_inlier_fraction"]):
        raise SceneSegmentationError(
            f"the best plane explains only {fraction:.0%} of the depth; the "
            "frame is not a view of a work surface")
    inliers = sample[inlier_mask]
    centre = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centre, full_matrices=False)
    n = vt[2]
    # The normal points TOWARD the camera: the optical axis is +Z, the table is
    # in front of it, so "up" from the table is -Z in this frame.
    if n[2] > 0:
        n = -n
    d = -float(n @ centre)
    residual = float(np.abs(inliers @ n + d).mean())
    return Plane(tuple(float(v) for v in n), d, residual, fraction)


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #


def _plane_basis(normal: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    seed = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def _inside(rect: Sequence[float], point: Sequence[float]) -> bool:
    x0, y0, x1, y1 = (float(v) for v in rect)
    return x0 <= float(point[0]) <= x1 and y0 <= float(point[1]) <= y1


def segment_scene(bgr: np.ndarray, depth_mm: np.ndarray, K: np.ndarray,
                  options: Optional[Dict[str, Any]] = None) -> SceneSegmentation:
    """Every object standing on the work plane, with its measured footprint."""
    if cv2 is None:
        raise SceneSegmentationError("OpenCV (cv2) is required for segmentation")
    settings = dict(DEFAULTS)
    settings.update({k: v for k, v in (options or {}).items() if v is not None})

    points, valid = deproject(depth_mm, K)
    if settings.get("colour_foreground", True):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo, hi = (int(v) for v in settings["metal_hue_range"])
        fg_colour = ((hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)
                     & (hsv[..., 1] >= int(settings["metal_min_saturation"]))
                     & (hsv[..., 2] >= int(settings["metal_min_value"])))
    else:
        fg_colour = np.zeros(depth_mm.shape, dtype=bool)
    # THE PLANE IS FITTED TO THE BENCH, NOT TO THE PARTS: metal-coloured pixels
    # are left out of the fit, so a bench crowded with tubes does not pull the
    # plane up and read every plate as flat.
    plane = fit_plane(points, valid & ~fg_colour, settings)
    height = plane.height(points)
    height[~valid] = np.nan

    fg_depth = (height > float(settings["min_height_mm"])) & \
               (height < float(settings["max_height_mm"]))
    fg_depth &= valid
    # A metal-coloured pixel far BELOW the plane is not a part on the bench —
    # it is the floor or a chair seen past the table's edge — and one far above
    # it is not either. Colour only counts where the depth agrees or is absent.
    with np.errstate(invalid="ignore"):
        plausible = (~valid) | ((height > -float(settings["plane_tolerance_mm"]) * 3)
                                & (height < float(settings["max_height_mm"])))
    fg_colour = fg_colour & plausible
    foreground = (fg_depth | fg_colour).astype(np.uint8)
    roi = settings.get("roi_px")
    if roi:
        # THE ROI IS A CROP, NOT JUST A FILTER: pixels outside it are bench
        # background, so a part at the edge of the work area is not welded to
        # whatever lies beyond the bench (the desk edge, a chair) by one
        # connected component.
        x0, y0, x1, y1 = (int(v) for v in roi)
        cropped = np.zeros_like(foreground)
        cropped[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = \
            foreground[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        foreground = cropped

    k_open = int(settings["open_kernel_px"])
    k_close = int(settings["close_kernel_px"])
    if k_open > 1:
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
                                      np.ones((k_open, k_open), np.uint8))
    if k_close > 1:
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE,
                                      np.ones((k_close, k_close), np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground, connectivity=8)
    h, w = depth_mm.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v_grid, u_grid = np.mgrid[0:h, 0:w]
    e1, e2 = _plane_basis(plane.normal)
    n = np.asarray(plane.normal)
    min_area = int(settings["min_component_area_px"])
    max_area = float(settings["max_component_area_fraction"]) * h * w
    ignore_regions = list(settings.get("ignore_regions_px") or [])

    instances: List[Instance] = []
    dropped = {"too_small": 0, "too_large": 0, "outside_roi": 0}
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            dropped["too_small"] += 1
            continue
        if area > max_area:
            dropped["too_large"] += 1
            continue
        centroid = (float(centroids[label][0]), float(centroids[label][1]))
        if roi and not _inside(roi, centroid):
            dropped["outside_roi"] += 1
            continue
        mask = labels == label
        x0, y0, bw, bh = (int(stats[label, i]) for i in range(4))

        # Ray-plane intersection for every mask pixel: the footprint is measured
        # ON the plane, so a part whose depth is noisy (a bolt) or partly
        # missing (a shiny tube edge) still gets a footprint from its outline.
        uu = u_grid[mask].astype(np.float64)
        vv = v_grid[mask].astype(np.float64)
        rays = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(len(uu))], axis=-1)
        t = -plane.d / (rays @ n)
        on_plane = rays * t[:, None]
        centre = on_plane.mean(axis=0)
        local = np.stack([(on_plane - centre) @ e1, (on_plane - centre) @ e2], axis=-1)
        if len(local) >= 3:
            cov = np.cov(local.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            major = eigenvectors[:, int(np.argmax(eigenvalues))]
            minor = eigenvectors[:, int(np.argmin(eigenvalues))]
        else:
            major, minor = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        along = local @ major
        across = local @ minor
        # The extent of the outline, with only the outermost 0.1 % of pixels
        # trimmed as stray: a 1-99 percentile span would read every part 2 %
        # short by construction.
        length = float(np.percentile(along, 99.9) - np.percentile(along, 0.1))
        width = float(np.percentile(across, 99.9) - np.percentile(across, 0.1))
        axis3 = major[0] * e1 + major[1] * e2
        axis3 = axis3 / np.linalg.norm(axis3)
        heights = height[mask]
        finite = heights[np.isfinite(heights)]
        h95 = float(np.percentile(finite, 95)) if len(finite) else 0.0
        hmed = float(np.median(finite)) if len(finite) else 0.0

        instance = Instance(
            index=len(instances) + 1, area_px=area, centroid_px=centroid,
            bbox_px=(x0, y0, x0 + bw, y0 + bh),
            centre_mm=tuple(float(c) for c in centre),
            axis_line=tuple(float(a) for a in axis3),
            length_mm=length, width_mm=width, height_mm=h95,
            height_median_mm=hmed, mask=mask)
        for region in ignore_regions:
            rect = region.get("rect") if isinstance(region, dict) else region
            if rect and _inside(rect, centroid):
                note = region.get("note", "") if isinstance(region, dict) else ""
                instance.ignored = ("inside a configured ignore region"
                                    + (f": {note}" if note else ""))
                break
        instances.append(instance)

    diagnostics = {
        "method": "depth_plane_instances",
        "image_size": [int(w), int(h)],
        "settings": {k: v for k, v in settings.items() if k != "ignore_regions_px"},
        "ignore_regions_px": ignore_regions,
        "valid_depth_fraction": round(float(valid.mean()), 4),
        "foreground_pixels": int(foreground.sum()),
        "components": int(count - 1),
        "dropped": dropped,
        "instances_measured": len(instances),
        "note": ("footprints are measured on the fitted plane from the mask "
                 "outline; heights are the 95th percentile of depth above the "
                 "plane inside the mask. Sizes, not identities."),
    }
    return SceneSegmentation(plane=plane, instances=instances, diagnostics=diagnostics)


# --------------------------------------------------------------------------- #
# Pictures, for the panel and the README
# --------------------------------------------------------------------------- #


def render_overlay(bgr: np.ndarray, instances: Sequence[Instance],
                   labels: Optional[Dict[int, str]] = None,
                   colours: Optional[Dict[int, Tuple[int, int, int]]] = None
                   ) -> np.ndarray:
    """Boxes, headings and labels drawn on the colour frame (BGR)."""
    if cv2 is None:
        raise SceneSegmentationError("OpenCV (cv2) is required to render")
    out = bgr.copy()
    for inst in instances:
        colour = (colours or {}).get(inst.index)
        if colour is None:
            colour = (0, 0, 220) if inst.ignored else (0, 200, 0)
        x0, y0, x1, y1 = inst.bbox_px
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, 2)
        text = (labels or {}).get(inst.index) or f"#{inst.index}"
        cv2.putText(out, text, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 2, cv2.LINE_AA)
    return out


def render_mask_overlay(bgr: np.ndarray, instances: Sequence[Instance]) -> np.ndarray:
    """Every instance mask tinted on the colour frame, ignored ones in red."""
    if cv2 is None:
        raise SceneSegmentationError("OpenCV (cv2) is required to render")
    out = bgr.copy()
    for inst in instances:
        if inst.mask is None:
            continue
        tint = np.array((40, 40, 230) if inst.ignored else (60, 220, 60), dtype=np.uint8)
        region = out[inst.mask]
        out[inst.mask] = (0.55 * region + 0.45 * tint).astype(np.uint8)
    return out


def render_depth(depth_mm: np.ndarray) -> np.ndarray:
    """A colour-mapped depth picture for looking at, never for measuring."""
    if cv2 is None:
        raise SceneSegmentationError("OpenCV (cv2) is required to render")
    valid = depth_mm > 0
    if not valid.any():
        return np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth_mm[valid], [2, 98])
    scaled = np.clip((depth_mm.astype(np.float64) - lo) / max(1.0, hi - lo), 0, 1)
    image = cv2.applyColorMap((255 - scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    image[~valid] = 0
    return image


__all__ = ["DEFAULTS", "Instance", "Plane", "SceneSegmentation",
           "SceneSegmentationError", "load_capture", "deproject", "fit_plane",
           "segment_scene", "render_overlay", "render_mask_overlay", "render_depth"]
