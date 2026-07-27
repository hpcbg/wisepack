"""Cut-aware planning domain — proposals, results and segment derivation.

A "cut" here means segmenting a *straight pipe* along its axis so the resulting
pieces fit residual cavities that the whole pipe cannot. This module is ROS-free
and FastAPI-free, exactly like domain.py, and it holds only the typed objects and
the pure arithmetic. The three concerns are deliberately split:

  * cutting.py        — the model (this file): CutProposal, CutResult, and the
                        function that turns an approved proposal into the derived
                        WasteItem segments. It is trusted to build a *consistent*
                        proposal, never trusted to certify one.
  * cut_validator.py  — an INDEPENDENT check of conservation and lineage. The
                        planner must not be its own examiner (see §3 of the
                        brief), so nothing in this file calls the validator.
  * cut_optimizer.py  — bounded candidate generation and the whole-process
                        objective that decides whether cutting is worth it.

Kerf model
----------
Cutting a pipe at ``n`` positions yields ``n + 1`` segments and consumes ``n``
saw kerfs. Length is conserved as::

    original_length = sum(segment_lengths) + n * kerf_mm

Material *volume* is conserved minus the kerf swept volume: each kerf removes one
tube cross-section (pi/4 * (OD^2 - ID^2)) times ``kerf_mm`` of metal as swarf.
Both identities are re-derived from scratch by the validator.

Everything a cut produces is provenance-labelled SIMULATED: there is no physical
cutting controller in this demonstrator, only an external-skill seam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .domain import (
    DomainError, GeometryType, Source, WasteItem, _require_id,
)

#: Length tolerance for the conservation identity, in mm. Segment lengths and
#: kerf are whole millimetres, so exact integer conservation is achievable and
#: this tolerance only absorbs a deviated *actual* cut result, never a planning
#: rounding error. Documented because the validator quotes it.
CONSERVATION_TOLERANCE_MM = 1


class CutState(str, Enum):
    """Lifecycle of a single cut, mirrored by the simulated cutting skill."""

    PROPOSED = "proposed"
    REQUESTED = "requested"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    QUALITY_CHECK_REQUIRED = "quality_check_required"


class CutApprovalState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class CutConfig:
    """Deterministic cost model for the cutting skill.

    None of these are measurements — they are declared, documented process
    constants used to price a cut against the container it might save. They are
    surfaced with SIMULATED provenance wherever they drive a reported figure.
    """

    kerf_mm: int = 3
    #: Fixed setup per cutting job (fixturing the pipe), seconds.
    setup_time_s: float = 20.0
    #: Time to make one cut, seconds.
    time_per_cut_s: float = 12.0
    #: Extra material-handling time charged per resulting segment, seconds.
    handling_time_per_segment_s: float = 6.0

    def __post_init__(self) -> None:
        if self.kerf_mm < 0:
            raise DomainError("kerf_mm must be >= 0")
        for name in ("setup_time_s", "time_per_cut_s",
                     "handling_time_per_segment_s"):
            if getattr(self, name) < 0:
                raise DomainError(f"{name} must be >= 0")

    def cutting_time_s(self, n_cuts: int) -> float:
        return self.setup_time_s + self.time_per_cut_s * max(0, n_cuts)

    def handling_time_s(self, n_segments: int) -> float:
        return self.handling_time_per_segment_s * max(0, n_segments)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kerf_mm": self.kerf_mm,
            "setup_time_s": self.setup_time_s,
            "time_per_cut_s": self.time_per_cut_s,
            "handling_time_per_segment_s": self.handling_time_per_segment_s,
            "source": Source.SIMULATED.value,
        }


# --------------------------------------------------------------------------- #
# Geometry helpers (pure)
# --------------------------------------------------------------------------- #


def tube_cross_section_mm2(outer_diameter_mm: int,
                           inner_diameter_mm: Optional[int]) -> float:
    """Metal cross-sectional area of a tube (solid rod when inner is None/0)."""
    inner = inner_diameter_mm or 0
    return (math.pi / 4.0) * (outer_diameter_mm ** 2 - inner ** 2)


def segment_lengths_from_positions(original_length_mm: int,
                                   cut_positions_mm: List[int],
                                   kerf_mm: int) -> List[int]:
    """Resulting segment lengths for kerf slots at ``cut_positions_mm``.

    Each cut position is the near edge of a kerf slot of width ``kerf_mm`` along
    the original pipe coordinate. Positions must be strictly increasing and leave
    room for their own kerf. Returns ``n + 1`` segment lengths.
    """
    positions = sorted(int(p) for p in cut_positions_mm)
    segments: List[int] = []
    prev_end = 0
    for p in positions:
        seg = p - prev_end
        if seg <= 0:
            raise DomainError(
                f"cut position {p} does not leave a positive segment "
                f"(previous kerf ended at {prev_end})")
        segments.append(seg)
        prev_end = p + kerf_mm
    tail = original_length_mm - prev_end
    if tail <= 0:
        raise DomainError(
            f"cut positions consume the whole pipe (tail={tail} mm)")
    segments.append(tail)
    return segments


def positions_from_segment_lengths(segment_lengths: List[int],
                                   kerf_mm: int) -> List[int]:
    """Inverse of :func:`segment_lengths_from_positions` (kerf near-edges)."""
    positions: List[int] = []
    cursor = 0
    for seg in segment_lengths[:-1]:
        cursor += int(seg)
        positions.append(cursor)
        cursor += kerf_mm
    return positions


# --------------------------------------------------------------------------- #
# CutProposal
# --------------------------------------------------------------------------- #


@dataclass
class CutProposal:
    """A proposed segmentation of one pipe, with its whole-process economics.

    A proposal is *consistent by construction* when built through
    :meth:`for_segments`, but it is only *valid* once cut_validator.py signs off
    (``validator_result``). ``approval_state`` is a human decision distinct from
    validity: a validated proposal still needs an operator to approve cutting.
    """

    proposal_id: str
    source_item_id: str
    original_length_mm: int
    outer_diameter_mm: int
    inner_diameter_mm: Optional[int]
    cut_positions_mm: List[int]
    segment_lengths_mm: List[int]
    kerf_mm: int
    total_kerf_mm: int
    estimated_cutting_time_s: float
    estimated_handling_time_s: float
    expected_containers_before: int
    expected_containers_after: int
    expected_utilization_before_pct: float
    expected_utilization_after_pct: float
    expected_capacity_saving_mm3: int
    objective_benefit: float
    reason: str
    validator_result: Optional[Dict[str, Any]] = None
    approval_state: CutApprovalState = CutApprovalState.PENDING
    source: Source = Source.SIMULATED
    derived_item_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_id("proposal_id", self.proposal_id)
        _require_id("source_item_id", self.source_item_id)
        self.approval_state = CutApprovalState(self.approval_state)
        self.source = Source(self.source)

    @property
    def n_cuts(self) -> int:
        return len(self.cut_positions_mm)

    @property
    def n_segments(self) -> int:
        return len(self.segment_lengths_mm)

    @property
    def containers_saved(self) -> int:
        return self.expected_containers_before - self.expected_containers_after

    @property
    def is_validated(self) -> bool:
        return bool(self.validator_result and self.validator_result.get("valid"))

    # -- construction ------------------------------------------------------ #

    @staticmethod
    def for_segments(proposal_id: str, item: WasteItem,
                     segment_lengths_mm: List[int], *,
                     config: CutConfig,
                     reason: str = "",
                     containers_before: int = 0, containers_after: int = 0,
                     utilization_before_pct: float = 0.0,
                     utilization_after_pct: float = 0.0,
                     objective_benefit: float = 0.0) -> "CutProposal":
        """Build a consistent proposal for cutting ``item`` into given segments.

        Cut positions, total kerf and the process-time estimates are all derived
        here so the proposal cannot be internally contradictory; the economics
        (container counts, utilization, benefit) are supplied by the optimizer,
        which alone has run the packer on the before/after scenarios.
        """
        if item.geometry_type is not GeometryType.TUBE:
            raise DomainError("only tubes can be cut")
        segs = [int(s) for s in segment_lengths_mm]
        if len(segs) < 2:
            raise DomainError("a cut produces at least two segments")
        n_cuts = len(segs) - 1
        positions = positions_from_segment_lengths(segs, config.kerf_mm)
        total_kerf = n_cuts * config.kerf_mm
        # expected_capacity_saving is a *capacity* figure the optimizer fills in
        # (it alone has packed the before/after scenarios); for_segments leaves
        # it 0 so the constructor is usable in unit tests without a packer.
        return CutProposal(
            proposal_id=proposal_id,
            source_item_id=item.item_id,
            original_length_mm=item.length_mm,
            outer_diameter_mm=item.outer_diameter_mm,
            inner_diameter_mm=item.inner_diameter_mm,
            cut_positions_mm=positions,
            segment_lengths_mm=segs,
            kerf_mm=config.kerf_mm,
            total_kerf_mm=total_kerf,
            estimated_cutting_time_s=config.cutting_time_s(n_cuts),
            estimated_handling_time_s=config.handling_time_s(len(segs)),
            expected_containers_before=containers_before,
            expected_containers_after=containers_after,
            expected_utilization_before_pct=round(utilization_before_pct, 3),
            expected_utilization_after_pct=round(utilization_after_pct, 3),
            expected_capacity_saving_mm3=0,
            objective_benefit=round(objective_benefit, 6),
            reason=reason,
        )

    @property
    def kerf_material_loss_mm3(self) -> float:
        cross = tube_cross_section_mm2(self.outer_diameter_mm,
                                       self.inner_diameter_mm)
        return cross * self.total_kerf_mm

    def derived_item_ids_for(self) -> List[str]:
        return [f"{self.source_item_id}-s{i + 1}" for i in range(self.n_segments)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source_item_id": self.source_item_id,
            "original_length_mm": self.original_length_mm,
            "outer_diameter_mm": self.outer_diameter_mm,
            "inner_diameter_mm": self.inner_diameter_mm,
            "cut_positions_mm": list(self.cut_positions_mm),
            "segment_lengths_mm": list(self.segment_lengths_mm),
            "n_cuts": self.n_cuts,
            "n_segments": self.n_segments,
            "kerf_mm": self.kerf_mm,
            "total_kerf_mm": self.total_kerf_mm,
            "kerf_material_loss_mm3": round(self.kerf_material_loss_mm3, 1),
            "estimated_cutting_time_s": round(self.estimated_cutting_time_s, 2),
            "estimated_handling_time_s": round(self.estimated_handling_time_s, 2),
            "expected_containers_before": self.expected_containers_before,
            "expected_containers_after": self.expected_containers_after,
            "containers_saved": self.containers_saved,
            "expected_utilization_before_pct":
                round(self.expected_utilization_before_pct, 2),
            "expected_utilization_after_pct":
                round(self.expected_utilization_after_pct, 2),
            "expected_capacity_saving_mm3": self.expected_capacity_saving_mm3,
            "objective_benefit": round(self.objective_benefit, 6),
            "reason": self.reason,
            "validator_result": self.validator_result,
            "is_validated": self.is_validated,
            "approval_state": self.approval_state.value,
            "derived_item_ids": self.derived_item_ids_for(),
            "source": self.source.value,
            "label": "SIMULATED CUT PROPOSAL",
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CutProposal":
        return CutProposal(
            proposal_id=d["proposal_id"],
            source_item_id=d["source_item_id"],
            original_length_mm=int(d["original_length_mm"]),
            outer_diameter_mm=int(d["outer_diameter_mm"]),
            inner_diameter_mm=d.get("inner_diameter_mm"),
            cut_positions_mm=list(d.get("cut_positions_mm", [])),
            segment_lengths_mm=list(d.get("segment_lengths_mm", [])),
            kerf_mm=int(d.get("kerf_mm", 0)),
            total_kerf_mm=int(d.get("total_kerf_mm", 0)),
            estimated_cutting_time_s=float(d.get("estimated_cutting_time_s", 0.0)),
            estimated_handling_time_s=float(d.get("estimated_handling_time_s", 0.0)),
            expected_containers_before=int(d.get("expected_containers_before", 0)),
            expected_containers_after=int(d.get("expected_containers_after", 0)),
            expected_utilization_before_pct=float(
                d.get("expected_utilization_before_pct", 0.0)),
            expected_utilization_after_pct=float(
                d.get("expected_utilization_after_pct", 0.0)),
            expected_capacity_saving_mm3=int(d.get("expected_capacity_saving_mm3", 0)),
            objective_benefit=float(d.get("objective_benefit", 0.0)),
            reason=d.get("reason", ""),
            validator_result=d.get("validator_result"),
            approval_state=CutApprovalState(d.get("approval_state", "pending")),
            source=Source(d.get("source", "simulated")),
        )


# --------------------------------------------------------------------------- #
# CutResult
# --------------------------------------------------------------------------- #


@dataclass
class CutResult:
    """The outcome of executing (simulating) an approved cut.

    ``actual_segment_lengths_mm`` may deviate from the proposal — that is the
    whole point of the ``cut_result_deviation`` scenario — which is why the
    downstream flow re-registers derived items from the ACTUAL dimensions and
    forces a fresh packing approval rather than trusting the plan built on the
    proposal.
    """

    proposal_id: str
    source_item_id: str
    actual_segment_lengths_mm: List[int]
    resulting_child_ids: List[str]
    actual_kerf_mm: int
    completion_status: CutState = CutState.COMPLETED
    quality_check_state: str = "not_required"
    source: Source = Source.SIMULATED
    failure_reason: str = ""

    def __post_init__(self) -> None:
        _require_id("source_item_id", self.source_item_id)
        self.completion_status = CutState(self.completion_status)
        self.source = Source(self.source)

    @property
    def succeeded(self) -> bool:
        return self.completion_status is CutState.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source_item_id": self.source_item_id,
            "actual_segment_lengths_mm": list(self.actual_segment_lengths_mm),
            "resulting_child_ids": list(self.resulting_child_ids),
            "actual_kerf_mm": self.actual_kerf_mm,
            "completion_status": self.completion_status.value,
            "quality_check_state": self.quality_check_state,
            "failure_reason": self.failure_reason,
            "succeeded": self.succeeded,
            "source": self.source.value,
            "label": "SIMULATED CUT RESULT",
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CutResult":
        return CutResult(
            proposal_id=d["proposal_id"],
            source_item_id=d["source_item_id"],
            actual_segment_lengths_mm=list(d.get("actual_segment_lengths_mm", [])),
            resulting_child_ids=list(d.get("resulting_child_ids", [])),
            actual_kerf_mm=int(d.get("actual_kerf_mm", 0)),
            completion_status=CutState(d.get("completion_status", "completed")),
            quality_check_state=d.get("quality_check_state", "not_required"),
            source=Source(d.get("source", "simulated")),
            failure_reason=d.get("failure_reason", ""),
        )


# --------------------------------------------------------------------------- #
# Segment derivation (proposal / result -> WasteItems)
# --------------------------------------------------------------------------- #


def _child_item(parent: WasteItem, child_id: str, length_mm: int,
                index: int, kerf_mm: int, generation: int) -> WasteItem:
    """One derived segment: inherits everything material, keeps lineage.

    Derived segments are NOT themselves cuttable (``cut_allowed=False``). One
    generation of cutting is modelled in the demonstrator, which also guarantees
    the candidate generation terminates.
    """
    history = list(parent.cut_history) + [{
        "parent_item_id": parent.item_id,
        "segment_index": index,
        "kerf_mm": kerf_mm,
        "source": Source.SIMULATED.value,
    }]
    return WasteItem(
        item_id=child_id,
        length_mm=int(length_mm),
        outer_diameter_mm=parent.outer_diameter_mm,
        geometry_type=GeometryType.TUBE,
        inner_diameter_mm=parent.inner_diameter_mm,
        material=parent.material,
        segregation_group=parent.segregation_group,   # inherited (validated)
        weight_kg=_segment_weight(parent, length_mm),
        priority=parent.priority,
        dose_class=parent.dose_class,
        permitted_axes=parent.permitted_axes,
        injected=parent.injected,
        profile_fill_ratio=parent.profile_fill_ratio,
        cut_allowed=False,
        parent_item_id=parent.item_id,
        generation=generation,
        cut_history=history,
    )


def _segment_weight(parent: WasteItem, length_mm: int) -> float:
    """Mass of a segment, pro-rated by length (mass is conserved minus kerf)."""
    if parent.length_mm <= 0:
        return 0.0
    return round(parent.weight_kg * (float(length_mm) / parent.length_mm), 4)


def derive_segments(parent: WasteItem, segment_lengths_mm: List[int], *,
                    kerf_mm: int,
                    child_ids: Optional[List[str]] = None) -> List[WasteItem]:
    """Turn a set of segment lengths into derived WasteItems (lineage set).

    Used for BOTH the proposal (planning) and the actual result (registration);
    the caller passes whichever segment lengths apply. Child ids default to
    ``<parent>-s<k>`` and must be unique — enforced here, not just downstream.
    """
    lengths = [int(s) for s in segment_lengths_mm]
    if len(lengths) < 2:
        raise DomainError("cutting produces at least two segments")
    ids = child_ids or [f"{parent.item_id}-s{i + 1}" for i in range(len(lengths))]
    if len(ids) != len(lengths):
        raise DomainError("child_ids count must match segment count")
    if len(set(ids)) != len(ids):
        raise DomainError(f"derived child ids are not unique: {ids}")
    generation = parent.generation + 1
    return [_child_item(parent, cid, length, i, kerf_mm, generation)
            for i, (cid, length) in enumerate(zip(ids, lengths))]


__all__ = [
    "CONSERVATION_TOLERANCE_MM", "CutState", "CutApprovalState", "CutConfig",
    "CutProposal", "CutResult", "tube_cross_section_mm2",
    "segment_lengths_from_positions", "positions_from_segment_lengths",
    "derive_segments",
]
