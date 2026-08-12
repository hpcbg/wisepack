"""Learned object representations — what a model-free estimator registers against.

WHY THIS IS NOT THE OBJECT REGISTRY. `rgbd.ObjectModel` answers "what shape is
this part": a CAD mesh, its declared units, its task axis, its bore. That is
ENGINEERING GEOMETRY, and packing depends on it being exact.

A representation answers a different question: "what has been LEARNED about
this object's appearance, well enough to find it in a frame". It is built from
reference views, it is an artefact of a particular estimator at a particular
revision, and it is not a measurement of the part. The Cylinder5 representation
has no bore. That is fine for pose and wrong for volume.

Keeping them apart is the point. If one type carried both, some later change
would route a reconstruction into the packing arithmetic and nothing would
complain — the numbers would simply be wrong, and plausibly so.

READINESS IS A LIVE QUESTION. A representation is registered in configuration
but its mesh is a build artefact that may not exist on this machine. `ready`
therefore checks the file, every time it is asked, and reports WHY when the
answer is no. Nothing here builds anything: training a Neural Object Field
takes minutes and belongs in an explicit offline step, never in a request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: The registry file, relative to the repository root.
REPRESENTATION_REGISTRY_PATH = os.path.join("config",
                                            "object_representations.yaml")

#: Where built representations live, relative to the repository root. A cache
#: directory, deliberately: these are large generated artefacts, they are not
#: tracked, and their absence is a normal state the dashboard must describe.
REPRESENTATION_STORE_PATH = os.path.join(".cache-perception", "model-free")

#: Environment override for the store, mirroring how the object assets root is
#: resolved. Used by the worker, where the store is a container mount.
REPRESENTATION_STORE_ENV = "WISEPACK_REPRESENTATION_STORE"

#: What a representation's validation may claim. `experimental` is the honest
#: default for anything whose physical validation is incomplete.
VALIDATION_EXPERIMENTAL = "experimental"
VALIDATION_VALIDATED = "validated"


class RepresentationError(Exception):
    """A representation was requested that cannot be used, with the reason."""


@dataclass
class ObjectRepresentation:
    """One learned representation of one object, for one perception method."""

    representation_id: str
    model_id: str
    method: str
    #: WHAT AN OPERATOR READS. The id and the digest identify the artefact and
    #: belong in provenance; neither should be the primary thing on screen.
    display_name: str = ""
    reference_digest: str = ""
    reference_source: str = ""
    reference_view_count: int = 0
    foundationpose_revision: str = ""
    mesh: str = ""
    mesh_scale_to_mm: float = 1000.0
    validation_status: str = VALIDATION_EXPERIMENTAL
    #: The short operator-facing clause. `validation_note` is the full
    #: qualification and belongs in documentation and provenance details.
    validation_summary: str = ""
    validation_note: str = ""
    validation_evidence: List[str] = field(default_factory=list)
    validation_gate: str = ""
    physical_ground_truth_available: bool = False
    #: PERCEPTION ONLY UNLESS SAID OTHERWISE. The default is the safe answer:
    #: a representation is not engineering geometry, and a registry entry that
    #: forgot to say so must not thereby become authoritative for packing.
    usable_for_packing_geometry: bool = False
    geometry_note: str = ""

    @classmethod
    def from_dict(cls, document: Dict[str, Any]) -> "ObjectRepresentation":
        return cls(
            representation_id=str(document.get("id", "")).strip(),
            model_id=str(document.get("model_id", "")).strip(),
            method=str(document.get("method", "")).strip(),
            display_name=str(document.get("display_name", "")).strip(),
            reference_digest=str(document.get("reference_digest", "")).strip(),
            reference_source=str(document.get("reference_source", "")).strip(),
            reference_view_count=int(document.get("reference_view_count", 0) or 0),
            foundationpose_revision=str(
                document.get("foundationpose_revision", "")).strip(),
            mesh=str(document.get("mesh", "")).strip(),
            mesh_scale_to_mm=float(document.get("mesh_scale_to_mm", 1000.0)),
            validation_status=str(document.get("validation_status")
                                  or VALIDATION_EXPERIMENTAL).strip(),
            validation_summary=" ".join(
                str(document.get("validation_summary", "")).split()),
            validation_note=" ".join(
                str(document.get("validation_note", "")).split()),
            validation_evidence=[" ".join(str(e).split())
                                 for e in (document.get("validation_evidence")
                                           or [])],
            validation_gate=" ".join(
                str(document.get("validation_gate", "")).split()),
            physical_ground_truth_available=bool(
                document.get("physical_ground_truth_available", False)),
            usable_for_packing_geometry=bool(
                document.get("usable_for_packing_geometry", False)),
            geometry_note=" ".join(str(document.get("geometry_note", "")).split()),
        )

    @property
    def is_experimental(self) -> bool:
        return self.validation_status != VALIDATION_VALIDATED

    @property
    def label(self) -> str:
        """What to show an operator. Falls back to the id, never to nothing."""
        return self.display_name or self.model_id or self.representation_id

    def mesh_path(self, store_root: str) -> str:
        return os.path.join(store_root, self.mesh) if self.mesh else ""

    def mesh_exists(self, store_root: str) -> bool:
        path = self.mesh_path(store_root)
        return bool(path) and os.path.isfile(path)

    def readiness(self, store_root: str) -> "Readiness":
        """READY, or NOT READY WITH THE REASON. Never a bare boolean.

        "Not ready" and "not ready because the representation has never been
        built on this machine" send an operator to different places, and the
        second is the only one they can act on.
        """
        if not self.mesh:
            return Readiness(False, f"{self.representation_id} declares no mesh")
        path = self.mesh_path(store_root)
        if not os.path.isfile(path):
            return Readiness(
                False,
                f"the representation {self.representation_id} has not been "
                f"built on this machine: no mesh at {path}. Build it offline "
                f"with ./scripts/model_free_build.sh — it is never built from "
                f"the dashboard.")
        return Readiness(True, "")

    def to_dict(self, store_root: str = "") -> Dict[str, Any]:
        document: Dict[str, Any] = {
            "id": self.representation_id,
            "label": self.label,
            "model_id": self.model_id,
            "method": self.method,
            "reference_digest": self.reference_digest,
            #: The first 8 characters, for provenance details. Enough to tell
            #: two representations apart at a glance; the full digest stays in
            #: the payload for anything that needs to match exactly.
            "reference_digest_short": self.reference_digest[:8],
            "reference_source": self.reference_source,
            "reference_view_count": self.reference_view_count,
            "foundationpose_revision": self.foundationpose_revision,
            "validation_status": self.validation_status,
            "validation_summary": self.validation_summary,
            "validation_note": self.validation_note,
            "validation_evidence": list(self.validation_evidence),
            "validation_gate": self.validation_gate,
            "experimental": self.is_experimental,
            # SAID OUT LOUD IN THE PAYLOAD. The dashboard must never render a
            # physical accuracy for this method, and the reason travels with
            # the data rather than living in a comment someone has to find.
            "physical_ground_truth_available":
                self.physical_ground_truth_available,
            "usable_for_packing_geometry": self.usable_for_packing_geometry,
            "geometry_note": self.geometry_note,
        }
        if store_root:
            readiness = self.readiness(store_root)
            document["ready"] = readiness.ready
            document["reason"] = readiness.reason
        return document


@dataclass
class Readiness:
    ready: bool
    reason: str


@dataclass
class RepresentationRegistry:
    """Every registered representation, and where the built ones live."""

    representations: Dict[str, ObjectRepresentation] = field(default_factory=dict)
    store_root: str = ""
    #: A BROKEN REGISTRY IS NOT AN EMPTY ONE. "Nothing is configured" and "the
    #: configuration cannot be read" send an operator to different files.
    error: str = ""

    @classmethod
    def from_dict(cls, document: Dict[str, Any], store_root: str
                  ) -> "RepresentationRegistry":
        entries = {}
        for raw in document.get("representations") or []:
            representation = ObjectRepresentation.from_dict(raw)
            if representation.representation_id:
                entries[representation.representation_id] = representation
        return cls(representations=entries, store_root=store_root)

    def for_model(self, model_id: str, method: str
                  ) -> Optional[ObjectRepresentation]:
        """The representation this object uses for this method, or None.

        NONE IS AN ANSWER, not an error to paper over. A model with no
        registered representation cannot run model-free, and the caller must
        say that rather than reaching for the CAD mesh.
        """
        for representation in self.representations.values():
            if (representation.model_id == model_id
                    and representation.method == method):
                return representation
        return None

    def require(self, model_id: str, method: str) -> ObjectRepresentation:
        """The representation, or a refusal naming what is missing.

        NEVER FALLS BACK TO CAD. Model-free means the estimator is not given a
        CAD model; quietly substituting one would produce a confident pose from
        a method the operator did not select, labelled as the method they did.
        """
        representation = self.for_model(model_id, method)
        if representation is None:
            raise RepresentationError(
                f"no {method} representation is registered for {model_id!r}. "
                "A learned representation must be prepared offline before this "
                "method can run; CAD is NOT substituted.")
        readiness = representation.readiness(self.store_root)
        if not readiness.ready:
            raise RepresentationError(readiness.reason)
        return representation

    def ready_for(self, model_id: str, method: str) -> Readiness:
        representation = self.for_model(model_id, method)
        if representation is None:
            return Readiness(
                False,
                f"no {method} representation is registered for {model_id!r}")
        return representation.readiness(self.store_root)

    def any_ready(self, method: str) -> bool:
        """Whether ANY registered representation for this method is built.

        What the METHOD selector needs: the method is offerable when something
        can be estimated with it, and which object is a separate choice.
        """
        return any(r.readiness(self.store_root).ready
                   for r in self.representations.values()
                   if r.method == method)

    def unavailable_reason(self, method: str) -> str:
        """Why this method cannot run here, in the operator's terms."""
        if self.error:
            return self.error
        candidates = [r for r in self.representations.values()
                      if r.method == method]
        if not candidates:
            return (f"no learned representation is registered for {method}; "
                    "one must be prepared offline")
        reasons = [c.readiness(self.store_root).reason for c in candidates]
        return "; ".join(r for r in reasons if r) or ""


def representation_store_root(repo_root: str = "") -> str:
    root = repo_root or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    override = os.environ.get(REPRESENTATION_STORE_ENV, "").strip()
    return override or os.path.join(root, REPRESENTATION_STORE_PATH)


def load_representation_registry(path: Optional[str] = None,
                                 repo_root: str = "",
                                 store_root: str = ""
                                 ) -> RepresentationRegistry:
    """Load the registry. A BROKEN registry is reported, never silently empty."""
    root = repo_root or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    resolved = path or os.path.join(root, REPRESENTATION_REGISTRY_PATH)
    store = store_root or representation_store_root(root)
    if not os.path.isfile(resolved):
        return RepresentationRegistry(representations={}, store_root=store,
                                      error="")
    try:
        import yaml                                            # noqa: PLC0415
        with open(resolved, encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        return RepresentationRegistry.from_dict(document, store_root=store)
    except Exception as exc:                                   # noqa: BLE001
        return RepresentationRegistry(representations={}, store_root=store,
                                      error=f"{resolved}: {exc}")


__all__ = [
    "RepresentationError", "ObjectRepresentation", "Readiness",
    "RepresentationRegistry", "load_representation_registry",
    "representation_store_root", "REPRESENTATION_REGISTRY_PATH",
    "REPRESENTATION_STORE_PATH", "REPRESENTATION_STORE_ENV",
    "VALIDATION_EXPERIMENTAL", "VALIDATION_VALIDATED",
]
