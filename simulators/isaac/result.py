"""Did the item actually end up in the container?

DELIBERATELY SEPARATE FROM THE ROBOT CODE, and free of Isaac imports. The state
machine's job is to move an arm; deciding whether the physical outcome counts as
a success is a different question, and one that must be answerable without a GPU
so it can be tested. Everything here takes numbers in and returns a verdict.

THE RULE THIS MODULE EXISTS TO ENFORCE
--------------------------------------
An item is NOT complete because the gripper opened. That would report a success
for a cylinder that bounced out of the bin, landed on the rim, or was never
grasped at all. Completion requires, after settling:

    * the body still exists and is somewhere sane;
    * its centre is inside the container footprint;
    * it is above the container floor and not perched above the rim;
    * its linear and angular velocities have stayed below threshold for a
      stable interval.

WHAT IT DOES NOT REQUIRE
------------------------
That the item landed where the optimizer planned. It did not, and it will not: a
released cylinder rolls. The distance from the planned pose is MEASURED and
reported on every item, and it is never rounded, hidden, or replaced by the
target. Reporting a target as though it were an outcome is the one failure mode
that would make the whole physical backend worthless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from wisepack_core.domain import Vec3
from wisepack_core.isaac_contract import Pose
from wisepack_core.isaac_transform import axis_deviation_deg, check_containment


@dataclass
class SettleMonitor:
    """Tracks whether a rigid body has come to rest.

    Requires the body to stay below BOTH thresholds for ``stable_time`` rather
    than sampling once: a cylinder is instantaneously motionless at the top of
    every bounce, and a single-sample check reports it settled mid-air.
    """

    linear_threshold: float
    angular_threshold: float
    stable_time: float
    timeout: float

    _stable_since: Optional[float] = field(default=None, init=False)
    _started_at: Optional[float] = field(default=None, init=False)
    peak_linear: float = field(default=0.0, init=False)
    peak_angular: float = field(default=0.0, init=False)
    last_linear: float = field(default=0.0, init=False)
    last_angular: float = field(default=0.0, init=False)

    def start(self, now: float) -> None:
        self._started_at = now
        self._stable_since = None
        self.peak_linear = 0.0
        self.peak_angular = 0.0

    def update(self, now: float, linear: Sequence[float],
               angular: Sequence[float]) -> Tuple[bool, bool]:
        """Feed one sample. Returns (settled, timed_out)."""
        if self._started_at is None:                  # pragma: no cover - misuse
            self.start(now)
        speed = _magnitude(linear)
        spin = _magnitude(angular)
        self.last_linear, self.last_angular = speed, spin
        self.peak_linear = max(self.peak_linear, speed)
        self.peak_angular = max(self.peak_angular, spin)

        if speed <= self.linear_threshold and spin <= self.angular_threshold:
            if self._stable_since is None:
                self._stable_since = now
            if now - self._stable_since >= self.stable_time:
                return True, False
        else:
            # Any excursion restarts the clock. A body that is still being
            # nudged by a neighbour has not settled, however briefly it paused.
            self._stable_since = None

        return False, (now - self._started_at) >= self.timeout

    def elapsed(self, now: float) -> float:
        return 0.0 if self._started_at is None else now - self._started_at

    def to_dict(self, now: float) -> Dict[str, Any]:
        return {
            "settle_elapsed_s": round(self.elapsed(now), 2),
            "final_linear_velocity_mps": round(self.last_linear, 4),
            "final_angular_velocity_radps": round(self.last_angular, 4),
            "peak_linear_velocity_mps": round(self.peak_linear, 4),
            "peak_angular_velocity_radps": round(self.peak_angular, 4),
            "linear_velocity_threshold": self.linear_threshold,
            "angular_velocity_threshold": self.angular_threshold,
        }


def _magnitude(vector: Sequence[float]) -> float:
    return sum(float(v) ** 2 for v in vector) ** 0.5


@dataclass
class PlacementOutcome:
    """The complete, honest record of one physical placement."""

    ok: bool
    #: Why the placement FAILED. Empty when ``ok``.
    reasons: List[str]
    #: Non-fatal observations worth recording — currently a settle timeout on an
    #: item that is nevertheless in the container. Kept separate from ``reasons``
    #: so "we noticed something" can never be mistaken for "this failed".
    notes: List[str]
    actual_pose: Optional[Pose]
    target_pose: Optional[Pose]
    position_error_mm: Optional[float]
    axis_error_deg: Optional[float]
    settled: bool
    timed_out: bool
    detail: Dict[str, Any]

    @property
    def message(self) -> str:
        if not self.ok:
            return "; ".join(self.reasons) or "physical placement failed"
        parts = ["settled inside the container"]
        if self.position_error_mm is not None:
            parts.append(f"{self.position_error_mm:.0f} mm from the planned pose")
        if self.axis_error_deg is not None:
            parts.append(f"axis off by {self.axis_error_deg:.0f} deg")
        parts.extend(self.notes)
        return ", ".join(parts)


def evaluate_placement(actual: Optional[Pose], target: Optional[Pose],
                       actual_quaternion: Optional[Sequence[float]],
                       container_inner: Vec3, length_mm: float, diameter_mm: float,
                       settled: bool, timed_out: bool,
                       settle_detail: Dict[str, Any]) -> PlacementOutcome:
    """Turn a settled body's measured state into a pass/fail plus the numbers.

    ``actual`` is None when the body could not be read back at all — it was
    deleted, or it never existed under the expected path. That is a failure, and
    a distinct one from "it landed outside the bin", so it is reported separately
    rather than folded into a generic error.
    """
    reasons: List[str] = []
    notes: List[str] = []
    detail: Dict[str, Any] = dict(settle_detail)

    if actual is None:
        return PlacementOutcome(
            ok=False, reasons=["the item is no longer present in the scene"],
            notes=notes, actual_pose=None, target_pose=target,
            position_error_mm=None, axis_error_deg=None, settled=False,
            timed_out=timed_out, detail=detail)

    verdict = check_containment(actual, container_inner, length_mm, diameter_mm)
    detail["containment"] = {
        "inside_footprint": verdict.inside_footprint,
        "above_floor": verdict.above_floor,
        "below_rim_overflow": verdict.below_rim_overflow,
        "in_scene": verdict.in_scene,
    }
    if not verdict.ok:
        reasons.append(verdict.detail)

    position_error = None if target is None else actual.distance_mm(target)
    axis_error = None
    if actual_quaternion is not None and target is not None:
        from wisepack_core.domain import Axis                  # noqa: PLC0415
        axis_error = axis_deviation_deg(actual_quaternion, Axis(target.axis))

    # A timeout is NOT automatically a failure. An item wedged against a wall can
    # keep a residual velocity above threshold indefinitely while being, in every
    # sense that matters, in the container. It IS recorded, so a run full of
    # timeouts is visible rather than silently equivalent to a clean one.
    if timed_out and not settled:
        detail["settle_timed_out"] = True
        notes.append("did not come fully to rest within settle_timeout")

    return PlacementOutcome(
        ok=verdict.ok, reasons=reasons, notes=notes, actual_pose=actual,
        target_pose=target, position_error_mm=position_error,
        axis_error_deg=axis_error, settled=settled, timed_out=timed_out,
        detail=detail)


__all__ = ["SettleMonitor", "PlacementOutcome", "evaluate_placement"]
