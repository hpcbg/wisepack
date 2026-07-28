"""WHICH RUN a piece of state belongs to — the stamp that makes FIWARE safe to read.

THE PROBLEM THIS SOLVES
-----------------------
Orion-LD holds CURRENT STATE, not a log. Every attribute keeps whatever value was
last written to it, for as long as nobody writes again — across process
restarts, across scenario changes, across runs. A KPI attribute written by
``isaac_cylinders_smoke`` (1 baseline container, 1 optimized) is still sitting
there when ``mixed_pipes_dense`` (3 and 2) is running, and reading it back
produces a dashboard showing one run's Digital Twin beside another run's KPI
cards. Measured exactly that way.

Nothing in the transport can detect this. The DDS→NGSI-LD bridge maps one topic
to one attribute and has no notion of a run; an HTTP 200 proves the broker is up,
not that what it returned describes what is happening now; and an entity
existing proves only that something wrote it once. Freshness has to be carried
IN the data, so this is the payload that carries it.

THE ORDERING RULE, WHICH IS THE WHOLE DESIGN
--------------------------------------------
The correlation attribute for an entity is published **after** the values it
stamps, never before. That direction is deliberate:

  * publish correlation LAST  -> a reader that polls mid-update sees the OLD
    correlation with some new values, judges the entity stale, and withholds it.
    Nothing wrong is displayed; the next poll is clean.
  * publish correlation FIRST -> a reader that polls mid-update sees the NEW
    correlation with some OLD values and trusts them. That is the mixed-run
    dashboard, reintroduced by the fix meant to prevent it.

So "the correlation matches the active run" carries a real guarantee: every
value in that entity was written for this run, before this stamp.

``sequence`` is monotonic per publishing process and exists to reject
out-of-order arrival — NGSI-LD gives no ordering guarantee across attributes,
and a delayed DDS sample can land after a newer one. A reader keeps the highest
sequence it has seen per entity and ignores anything below it.

Pure stdlib on purpose: imported by the orchestrator under Vulcanexus, by the
dashboard, and by the tests, all from this one file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

SCHEMA_VERSION = "wisepack-correlation/1.0"

#: Facets compared when deciding whether a projection describes the active run.
#: Order matters only for the message a human reads.
CORRELATION_FACETS = ("run_id", "scenario_id", "scenario_revision",
                      "plan_id", "plan_revision", "approval_revision")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class RunCorrelation:
    """The identity of one run, as carried alongside a projection of its state.

    Every field is optional because not every entity has every facet: the KPI
    entity has no plan revision of its own, the action stream has no approval.
    A facet that is ``None`` is NOT compared — absence means "this projection
    makes no claim about that", not "it matches anything". A publisher that
    omits a facet it does have would defeat the check, so the orchestrator fills
    in everything it knows and the per-entity subset is chosen there.
    """

    run_id: Optional[str] = None
    scenario_id: Optional[str] = None
    scenario_revision: Optional[int] = None
    plan_id: Optional[str] = None
    plan_revision: Optional[int] = None
    approval_revision: Optional[int] = None
    #: Monotonic per publisher. Used only to reject out-of-order arrival.
    sequence: int = 0
    published_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "scenario_revision": self.scenario_revision,
            "plan_id": self.plan_id,
            "plan_revision": self.plan_revision,
            "approval_revision": self.approval_revision,
            "sequence": int(self.sequence),
            "published_at": self.published_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @staticmethod
    def from_dict(doc: Any) -> Optional["RunCorrelation"]:
        """Parse, returning None for anything that is not a correlation stamp.

        Never raises. This is called on data fetched from an external broker in
        a dashboard poll loop, and an exception there takes out the whole page
        for a check whose job is to make the page MORE reliable.
        """
        if not isinstance(doc, dict):
            return None
        if str(doc.get("schema_version", "")).split("/", 1)[0] not in (
                "wisepack-correlation",):
            return None

        def _int(key):
            value = doc.get(key)
            try:
                return None if value is None else int(value)
            except (TypeError, ValueError):
                return None

        return RunCorrelation(
            run_id=doc.get("run_id") or None,
            scenario_id=doc.get("scenario_id") or None,
            scenario_revision=_int("scenario_revision"),
            plan_id=doc.get("plan_id") or None,
            plan_revision=_int("plan_revision"),
            approval_revision=_int("approval_revision"),
            sequence=_int("sequence") or 0,
            published_at=str(doc.get("published_at", "")),
        )

    # -- comparison --------------------------------------------------------- #

    def mismatches(self, active: "RunCorrelation") -> Dict[str, Any]:
        """{facet: {"expected": ..., "found": ...}} for every facet that differs.

        A facet absent from EITHER side is skipped: the projection makes no
        claim, or the active run has nothing to compare against. Skipping is the
        conservative choice for a rolling upgrade — an older publisher that
        stamps nothing is unknown, not wrong — and the caller decides what to do
        with a projection carrying no stamp at all (see ``is_unstamped``).
        """
        out: Dict[str, Any] = {}
        for facet in CORRELATION_FACETS:
            mine, theirs = getattr(self, facet), getattr(active, facet)
            if mine is None or theirs is None:
                continue
            if mine != theirs:
                out[facet] = {"expected": theirs, "found": mine}
        return out

    def matches(self, active: "RunCorrelation") -> bool:
        return not self.mismatches(active)

    @property
    def is_unstamped(self) -> bool:
        """True when this carries no identity at all — unknown, not matching."""
        return all(getattr(self, f) is None for f in CORRELATION_FACETS)

    def describe(self) -> str:
        parts = [f"run {self.run_id or '—'}"]
        if self.scenario_id:
            parts.append(self.scenario_id)
        if self.scenario_revision is not None:
            parts.append(f"revision {self.scenario_revision}")
        return ", ".join(parts)


def describe_mismatch(mismatches: Dict[str, Any]) -> str:
    """One operator-readable sentence for a mismatch dict."""
    if not mismatches:
        return ""
    bits = [f"{facet} {detail['found']!r} (current run has "
            f"{detail['expected']!r})" for facet, detail in mismatches.items()]
    return "; ".join(bits)


__all__ = [
    "SCHEMA_VERSION", "CORRELATION_FACETS", "RunCorrelation",
    "describe_mismatch", "utc_now_iso",
]
