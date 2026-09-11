"""The DEMO workcell frame assumptions — loaded from configuration, never typed in.

WHAT THIS IS
------------
The physical D435 reports a 6-DoF pose in `camera_color_optical_frame`. Isaac
Sim needs that object in its world. Between the two sit exactly two rigid
transforms:

    camera_color_optical_frame -> wisepack_workarea -> table (Isaac workcell, mm)

and `config/isaac_workcell.yaml` is the ONE place they are written down. This
module reads that file into `RigidTransform`s with their provenance attached and
hands them to `wisepack_core.scene_sync`, which is the ONE place they are
applied. Nothing in the simulator, the bridge or the dashboard does its own axis
swap.

WHAT IT IS NOT
--------------
A calibration. Every transform in the demo file carries `method:
configured_demo`, and everything downstream repeats that word: the scene
acknowledgement, the audit trail and the dashboard say "configured demo
transform". A measured calibration is a different provenance value —
`measured_calibration` — recorded as a separate transform when one exists.
Conflating the two would let an assumption be quoted as a measurement.

NO IDENTITY FALLBACK. A missing file, a missing transform or a transform with
no method is reported as unavailable, and `scene_sync` refuses to place
anything. An unmeasured extrinsic is missing, not identity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .pose import CAMERA_OPTICAL_FRAME, WORKAREA_FRAME, PoseError, RigidTransform

#: Where the demo assumptions live, relative to the repository root.
WORKCELL_CONFIG_RELPATH = os.path.join("config", "isaac_workcell.yaml")
#: Override for tests and unusual hosts. Never needed for the demonstrator.
WORKCELL_CONFIG_ENV = "WISEPACK_ISAAC_WORKCELL"

SCHEMA = "wisepack-isaac-workcell/1.0"

#: The Isaac workcell frame in millimetres — `table` in
#: wisepack_core.isaac_transform: origin at the robot base on the table top.
TABLE_FRAME = "table"

#: Provenance vocabulary. `configured_demo` is what the tracked file says;
#: `measured_calibration` is what a future calibration procedure will say.
#: Anything else is carried verbatim and labelled as itself.
PROVENANCE_CONFIGURED_DEMO = "configured_demo"
PROVENANCE_MEASURED_CALIBRATION = "measured_calibration"

#: The operator-facing wording for each provenance. The demo one deliberately
#: contains neither "calibrated" nor "measured".
PROVENANCE_LABELS = {
    PROVENANCE_CONFIGURED_DEMO: "Configured demo transform",
    PROVENANCE_MEASURED_CALIBRATION: "Measured calibration",
}


def provenance_label(source: str) -> str:
    """Human wording for a transform provenance, never claiming more than it is."""
    if not source:
        return "No transform"
    return PROVENANCE_LABELS.get(source, f"Transform ({source})")


class WorkcellConfigError(ValueError):
    """The workcell configuration cannot be used, with the reason."""


@dataclass(frozen=True)
class WorkareaBounds:
    """Where a synchronized source object is allowed to be, in the work area."""

    x_mm: Tuple[float, float] = (-320.0, 320.0)
    y_mm: Tuple[float, float] = (-260.0, 260.0)
    source_plane_z_mm: float = 0.0
    plane_tolerance_below_mm: float = 30.0
    plane_tolerance_above_mm: float = 80.0

    def refusal(self, x_mm: float, y_mm: float, z_mm: float,
                radius_mm: float = 0.0) -> str:
        """Why a centre at (x, y, z) is outside the demo work area, or ""."""
        if not self.x_mm[0] <= x_mm <= self.x_mm[1]:
            return (f"x = {x_mm:.1f} mm is outside the configured work-area "
                    f"bounds [{self.x_mm[0]:.0f}, {self.x_mm[1]:.0f}] mm")
        if not self.y_mm[0] <= y_mm <= self.y_mm[1]:
            return (f"y = {y_mm:.1f} mm is outside the configured work-area "
                    f"bounds [{self.y_mm[0]:.0f}, {self.y_mm[1]:.0f}] mm")
        expected = self.source_plane_z_mm + radius_mm
        if z_mm < expected - self.plane_tolerance_below_mm:
            return (f"z = {z_mm:.1f} mm puts the object centre "
                    f"{expected - z_mm:.1f} mm below where it would rest on the "
                    f"source plane (z = {self.source_plane_z_mm:.0f} mm + radius "
                    f"{radius_mm:.1f} mm); limit {self.plane_tolerance_below_mm:.0f} mm")
        if z_mm > expected + self.plane_tolerance_above_mm:
            return (f"z = {z_mm:.1f} mm floats the object centre "
                    f"{z_mm - expected:.1f} mm above the source plane; limit "
                    f"{self.plane_tolerance_above_mm:.0f} mm")
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {"x_mm": list(self.x_mm), "y_mm": list(self.y_mm),
                "source_plane_z_mm": self.source_plane_z_mm,
                "plane_tolerance_below_mm": self.plane_tolerance_below_mm,
                "plane_tolerance_above_mm": self.plane_tolerance_above_mm}


@dataclass
class WorkcellFrames:
    """The configured transforms, with the provenance of the whole set.

    `available` is False when either link of the chain is missing or carries no
    method. The chain is then reported as unavailable — it is never patched with
    an identity.
    """

    camera_to_workarea: Optional[RigidTransform] = None
    workarea_to_table: Optional[RigidTransform] = None
    provenance: str = ""
    bounds: WorkareaBounds = field(default_factory=WorkareaBounds)
    camera: Dict[str, Any] = field(default_factory=dict)
    path: str = ""
    error: str = ""

    @property
    def available(self) -> bool:
        return (self.camera_to_workarea is not None
                and self.camera_to_workarea.valid
                and self.workarea_to_table is not None
                and self.workarea_to_table.valid
                and not self.error)

    @property
    def unavailable_reason(self) -> str:
        if self.error:
            return self.error
        if self.camera_to_workarea is None:
            return (f"no {CAMERA_OPTICAL_FRAME} -> {WORKAREA_FRAME} transform is "
                    f"configured in {self.path or WORKCELL_CONFIG_RELPATH}")
        if not self.camera_to_workarea.valid:
            return (f"the {CAMERA_OPTICAL_FRAME} -> {WORKAREA_FRAME} transform "
                    "declares no method; an unlabelled transform is not usable "
                    "for a physical action")
        if self.workarea_to_table is None:
            return (f"no {WORKAREA_FRAME} -> {TABLE_FRAME} transform is "
                    f"configured in {self.path or WORKCELL_CONFIG_RELPATH}")
        if not self.workarea_to_table.valid:
            return (f"the {WORKAREA_FRAME} -> {TABLE_FRAME} transform declares "
                    "no method")
        return ""

    @property
    def label(self) -> str:
        """`Configured demo transform` / `Measured calibration` / ..."""
        return provenance_label(self.provenance) if self.available else "No transform"

    @property
    def frame_chain(self) -> List[str]:
        return [CAMERA_OPTICAL_FRAME, WORKAREA_FRAME, TABLE_FRAME, "world"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "provenance": self.provenance,
            "label": self.label,
            "frame_chain": self.frame_chain,
            "camera_to_workarea": (self.camera_to_workarea.to_dict()
                                   if self.camera_to_workarea else None),
            "workarea_to_table": (self.workarea_to_table.to_dict()
                                  if self.workarea_to_table else None),
            "bounds": self.bounds.to_dict(),
            "camera": dict(self.camera),
            "path": self.path,
        }


def _transform_from(entry: Dict[str, Any], where: str) -> RigidTransform:
    if not isinstance(entry, dict):
        raise WorkcellConfigError(f"{where}: each transform must be a mapping")
    try:
        transform = RigidTransform.from_dict(entry)
    except PoseError as exc:
        raise WorkcellConfigError(f"{where}: {exc}") from exc
    if not transform.method:
        raise WorkcellConfigError(
            f"{where}: transform {transform.child_frame} -> "
            f"{transform.parent_frame} declares no method. A transform with no "
            "provenance cannot be used for a physical action; write "
            f"`method: {PROVENANCE_CONFIGURED_DEMO}` if that is what it is.")
    return transform


def workcell_from_dict(document: Dict[str, Any], path: str = "") -> WorkcellFrames:
    """Parse a workcell document. Refuses rather than repairs."""
    if not isinstance(document, dict):
        raise WorkcellConfigError(f"{path or 'workcell'}: not a mapping")
    schema = str(document.get("schema", ""))
    if schema.rsplit(".", 1)[0] != SCHEMA.rsplit(".", 1)[0]:
        raise WorkcellConfigError(
            f"{path or 'workcell'}: schema {schema!r} is not {SCHEMA}")
    provenance = str(document.get("provenance", "") or "")
    if not provenance:
        raise WorkcellConfigError(
            f"{path or 'workcell'}: `provenance` is required — "
            f"{PROVENANCE_CONFIGURED_DEMO} or {PROVENANCE_MEASURED_CALIBRATION}")

    camera_to_workarea = None
    workarea_to_table = None
    for index, entry in enumerate(document.get("transforms") or []):
        where = f"{path or 'workcell'}: transforms[{index}]"
        transform = _transform_from(entry, where)
        pair = (transform.parent_frame, transform.child_frame)
        if pair == (WORKAREA_FRAME, CAMERA_OPTICAL_FRAME):
            camera_to_workarea = transform
        elif pair == (TABLE_FRAME, WORKAREA_FRAME):
            workarea_to_table = transform
        else:
            raise WorkcellConfigError(
                f"{where}: unexpected frame pair {transform.child_frame} -> "
                f"{transform.parent_frame}; the chain is {CAMERA_OPTICAL_FRAME} "
                f"-> {WORKAREA_FRAME} -> {TABLE_FRAME}")
        if transform.method != provenance:
            raise WorkcellConfigError(
                f"{where}: method {transform.method!r} disagrees with the "
                f"file's provenance {provenance!r}; one file, one provenance")

    area = document.get("workarea") or {}
    bounds_doc = area.get("bounds_mm") or {}

    def _pair(name: str, default: Tuple[float, float]) -> Tuple[float, float]:
        raw = bounds_doc.get(name)
        if raw is None:
            return default
        try:
            low, high = (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise WorkcellConfigError(
                f"workarea.bounds_mm.{name} must be [low, high]") from exc
        if low >= high:
            raise WorkcellConfigError(
                f"workarea.bounds_mm.{name}: low {low} is not below high {high}")
        return (low, high)

    bounds = WorkareaBounds(
        x_mm=_pair("x", WorkareaBounds.x_mm),
        y_mm=_pair("y", WorkareaBounds.y_mm),
        source_plane_z_mm=float(area.get("source_plane_z_mm", 0.0) or 0.0),
        plane_tolerance_below_mm=float(
            area.get("plane_tolerance_below_mm", 30.0) or 0.0),
        plane_tolerance_above_mm=float(
            area.get("plane_tolerance_above_mm", 80.0) or 0.0))

    return WorkcellFrames(
        camera_to_workarea=camera_to_workarea,
        workarea_to_table=workarea_to_table,
        provenance=provenance,
        bounds=bounds,
        camera=dict(document.get("camera") or {}),
        path=path)


def workcell_config_path(repo_root: str = "") -> str:
    configured = os.environ.get(WORKCELL_CONFIG_ENV, "").strip()
    if configured:
        return configured
    root = repo_root or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    return os.path.join(root, WORKCELL_CONFIG_RELPATH)


def load_workcell(path: Optional[str] = None, repo_root: str = "") -> WorkcellFrames:
    """Load the workcell frames. A BROKEN file is reported, never silently empty.

    Returns a `WorkcellFrames` whose `available` is False — with the reason —
    rather than raising, so a bridge can start, report "transform unavailable"
    and refuse synchronization, instead of crashing the orchestrator over a
    configuration file the operator can see and fix.
    """
    resolved = path or workcell_config_path(repo_root)
    if not os.path.isfile(resolved):
        return WorkcellFrames(
            path=resolved,
            error=f"workcell configuration not found at {resolved}")
    try:
        import yaml                                            # noqa: PLC0415
        with open(resolved, encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        return workcell_from_dict(document, path=resolved)
    except Exception as exc:                                   # noqa: BLE001
        return WorkcellFrames(path=resolved, error=f"{resolved}: {exc}")


__all__ = [
    "WorkcellFrames", "WorkareaBounds", "WorkcellConfigError",
    "load_workcell", "workcell_from_dict", "workcell_config_path",
    "provenance_label", "PROVENANCE_CONFIGURED_DEMO",
    "PROVENANCE_MEASURED_CALIBRATION", "PROVENANCE_LABELS", "TABLE_FRAME",
    "WORKCELL_CONFIG_RELPATH", "WORKCELL_CONFIG_ENV", "SCHEMA",
]
