"""Model-free as a PERCEPTION METHOD in the dashboard, and what must not leak.

THE THREE CLAIMS THIS INTEGRATION MAKES
---------------------------------------
1. Model-free is a METHOD on the existing axis — not a new object source, not
   an execution backend, not a second workflow. It reads the same frames from
   the same devices and produces the same ObservationBatch.
2. NO CAD REACHES THE MODEL-FREE ESTIMATOR. Not the mesh, not a path to it, and
   not indirectly when a representation is missing — that case refuses.
3. The reconstruction is NOT engineering geometry. WISEPACK keeps packing
   against exact CAD, and the representation records that it must not be used
   for that.

Each is tested here against the real modules, with no camera, no GPU and no
worker: the code that decides these things is pure and can be exercised
directly, which is the only way these stay true rather than becoming true by
convention.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"),
              os.path.join(REPO, "perception")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from wisepack_core.acquisition import (                            # noqa: E402
    ACQUISITION_ISAAC, ACQUISITION_PLANAR, ACQUISITION_REALSENSE,
    METHOD_ACQUISITIONS)
from wisepack_core.perception import PerceptionMethod              # noqa: E402
from wisepack_core.representation import (                         # noqa: E402
    RepresentationError, load_representation_registry)

CAD = PerceptionMethod.FOUNDATIONPOSE_RGBD.value
MODEL_FREE = PerceptionMethod.FOUNDATIONPOSE_RGBD_MODEL_FREE.value
PLANAR = PerceptionMethod.PLANAR_FASTERRCNN.value


def _registry(store=""):
    return load_representation_registry(repo_root=REPO, store_root=store)


class _SpyClient:
    """A worker that never answers, but remembers exactly what it was asked."""

    def __init__(self):
        self.requests = []

    def estimate(self, request):
        self.requests.append(dict(request))
        return None, "spy client: no worker in tests"


def _provider(store=""):
    from providers.foundationpose_rgbd import FoundationPoseProvider
    provider = FoundationPoseProvider(representations=_registry(store))
    provider.client = _SpyClient()
    return provider


# --------------------------------------------------------------------------- #
# 1. It is a METHOD, on the axis that already exists
# --------------------------------------------------------------------------- #


def test_model_free_is_a_perception_method_not_a_source_or_backend():
    assert MODEL_FREE in {m.value for m in PerceptionMethod}
    from wisepack_core.acquisition import ACQUISITION_SOURCES
    assert MODEL_FREE not in ACQUISITION_SOURCES
    from wisepack_core.perception import PerceptionSource
    assert MODEL_FREE not in {s.value for s in PerceptionSource}


def test_both_foundationpose_methods_share_one_provider():
    """They differ in the geometry the estimator is handed, not in how a frame
    is read. A second provider module would be the same code twice."""
    assert (PerceptionMethod(CAD).provider_module
            == PerceptionMethod(MODEL_FREE).provider_module)


def test_the_source_to_method_matrix_is_exactly_the_intended_one():
    """THE WHOLE COMPATIBILITY CLAIM, in one assertion."""
    assert METHOD_ACQUISITIONS[PLANAR] == (ACQUISITION_PLANAR,)
    for method in (CAD, MODEL_FREE):
        assert set(METHOD_ACQUISITIONS[method]) == {ACQUISITION_REALSENSE,
                                                    ACQUISITION_ISAAC}


@pytest.mark.parametrize("device", [ACQUISITION_REALSENSE, ACQUISITION_ISAAC])
def test_both_rgbd_methods_are_offered_for_both_rgbd_devices(device):
    offered = {m for m, sources in METHOD_ACQUISITIONS.items()
               if device in sources}
    assert offered == {CAD, MODEL_FREE}


def test_a_compatible_rgbd_method_survives_switching_physical_and_simulated():
    """SWITCHING DEVICE MUST NOT SILENTLY RESTYLE THE RUN. Both RGB-D devices
    accept both RGB-D methods, so an operator who chose model-free and moved
    from simulated to physical keeps model-free."""
    for source, target in ((ACQUISITION_ISAAC, ACQUISITION_REALSENSE),
                           (ACQUISITION_REALSENSE, ACQUISITION_ISAAC)):
        for method in (CAD, MODEL_FREE):
            assert source in METHOD_ACQUISITIONS[method]
            # The method stays compatible after the switch, so nothing forces
            # a change — which is what the dashboard's `set_acquisition` keys
            # its "method_changed_to" decision on.
            assert target in METHOD_ACQUISITIONS[method]


def test_planar_is_never_offered_for_an_rgbd_device():
    for device in (ACQUISITION_REALSENSE, ACQUISITION_ISAAC):
        assert device not in METHOD_ACQUISITIONS[PLANAR]


def test_the_method_declares_what_it_needs_and_what_it_does_not():
    cad, free = PerceptionMethod(CAD), PerceptionMethod(MODEL_FREE)
    assert cad.requires_object_model and not cad.requires_representation
    # THE PROPERTY THE WHOLE SEPARATION TURNS ON.
    assert not free.requires_object_model and free.requires_representation
    assert free.requires_depth
    assert free.measures == cad.measures
    assert free.estimator_geometry == "learned_representation"
    assert cad.estimator_geometry == "cad"


def test_the_operator_facing_labels_distinguish_the_two():
    assert PerceptionMethod(CAD).selector_label.endswith("(CAD)")
    assert PerceptionMethod(MODEL_FREE).selector_label.endswith("(model-free)")
    # SAYS WHAT IS NOT SUPPLIED, because "model-free" alone invites the reading
    # that WISEPACK packs without CAD, which it does not.
    detail = PerceptionMethod(MODEL_FREE).selector_detail
    assert "no CAD mesh is supplied" in detail


# --------------------------------------------------------------------------- #
# 2. No CAD reaches the model-free estimator
# --------------------------------------------------------------------------- #


def test_the_model_free_request_carries_the_representation_and_no_cad():
    provider = _provider()
    provider.acquire_physical(dataset="d", model_id="cylinder5",
                              depth_scale_mm=1.0, method=MODEL_FREE)
    assert len(provider.client.requests) == 1
    mesh = provider.client.requests[0]["mesh_path"]
    assert mesh.endswith("model.obj")
    # NOT THE STL, and not anything under the CAD tree.
    assert ".stl" not in mesh.lower()
    assert "CAD-Models" not in mesh


def test_the_cad_request_still_carries_the_cad_mesh():
    """The control. Model-free must not have changed what CAD mode does."""
    provider = _provider()
    provider.acquire_physical(dataset="d", model_id="cylinder5",
                              depth_scale_mm=1.0, method=CAD)
    mesh = provider.client.requests[0]["mesh_path"]
    assert mesh.lower().endswith(".stl")


def test_the_two_methods_send_their_own_declared_units():
    """A metre-scale reconstruction sent as millimetres would be a pose 1000x
    away, so each geometry's unit travels with it rather than being assumed."""
    cad, free = _provider(), _provider()
    cad.acquire_physical(dataset="d", model_id="cylinder5",
                         depth_scale_mm=1.0, method=CAD)
    free.acquire_physical(dataset="d", model_id="cylinder5",
                          depth_scale_mm=1.0, method=MODEL_FREE)
    assert cad.client.requests[0]["mesh_scale_to_metres"] == pytest.approx(0.001)
    assert free.client.requests[0]["mesh_scale_to_metres"] == pytest.approx(1.0)


def test_a_missing_representation_refuses_and_never_calls_the_estimator():
    """NOT READY IS A REFUSAL. Not a CAD run, not a substitution, and not a
    request the worker could answer with the wrong geometry."""
    provider = _provider(store="/nonexistent-representation-store")
    batch = provider.acquire_physical(dataset="d", model_id="cylinder5",
                                      depth_scale_mm=1.0, method=MODEL_FREE)
    assert not batch.ok
    # THE ESTIMATOR WAS NEVER ASKED ANYTHING. No mesh of any kind was sent.
    assert provider.client.requests == []
    # The failed batch still says which method was attempted, so the refusal
    # cannot be read as a CAD failure.
    assert batch.perception_method == MODEL_FREE
    assert "has not been built" in batch.error


def test_require_refuses_rather_than_returning_cad():
    registry = _registry(store="/nonexistent-representation-store")
    with pytest.raises(RepresentationError) as excinfo:
        registry.require("cylinder5", MODEL_FREE)
    assert "not been built" in str(excinfo.value)


def test_an_unregistered_object_refuses_and_names_the_gap():
    registry = _registry()
    with pytest.raises(RepresentationError) as excinfo:
        registry.require("no_such_part", MODEL_FREE)
    message = str(excinfo.value)
    assert "no_such_part" in message
    # SAYS IT WILL NOT SUBSTITUTE, because that is the question a reader has.
    assert "CAD is NOT substituted" in message


def test_no_cad_dimension_is_used_to_choose_an_object_or_a_region():
    """IDENTIFYING THE PART BY ITS SIZE and then measuring its pose with that
    same shape is circular. The provider selects geometry by `model_id`, which
    an operator states, and the physical mask comes from depth alone."""
    from providers import foundationpose_rgbd as module
    source = open(module.__file__, encoding="utf-8").read()
    body = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    for forbidden in ("length_mm ==", "diameter_mm ==", "closest_model",
                      "best_match", "match_dimensions"):
        assert forbidden not in body


# --------------------------------------------------------------------------- #
# 3. Perception representation is NOT engineering geometry
# --------------------------------------------------------------------------- #


def test_the_registered_representation_is_not_authoritative_for_packing():
    registry = _registry()
    representation = registry.for_model("cylinder5", MODEL_FREE)
    assert representation is not None
    assert representation.usable_for_packing_geometry is False
    # WHY, recorded with it rather than left to be rediscovered.
    assert "bore" in representation.geometry_note


def test_a_representation_defaults_to_not_being_packing_geometry():
    """The SAFE default: an entry that forgot to say must not thereby become
    authoritative."""
    from wisepack_core.representation import ObjectRepresentation
    bare = ObjectRepresentation.from_dict({"id": "x", "model_id": "y",
                                           "method": MODEL_FREE})
    assert bare.usable_for_packing_geometry is False


def test_the_observation_geometry_stays_cad_for_both_methods():
    """MODEL-FREE CHANGES THE ESTIMATOR'S INPUT, NOT WISEPACK'S GEOMETRY. The
    dimensions on an observation are the part's, from the object registry, for
    both methods — so nothing routes a bore-less reconstruction into packing."""
    from providers import foundationpose_rgbd as module
    source = open(module.__file__, encoding="utf-8").read()
    assert 'geometry_source="cad_model"' in source
    # And there is exactly one of them: no branch sets a different source for
    # model-free, which would be the reconstruction entering the domain.
    assert source.count("geometry_source=") == 1


def test_the_reconstruction_is_never_the_source_of_dimensions():
    from providers import foundationpose_rgbd as module
    source = open(module.__file__, encoding="utf-8").read()
    body = source[source.index("def observation_from"):]
    for field in ("diameter_mm=model.diameter_mm",
                  "length_mm=model.length_mm",
                  "inner_diameter_mm=model.inner_diameter_mm"):
        assert field in body, field
    # NOTHING READS A DIMENSION OFF THE REPRESENTATION.
    for forbidden in ("representation.diameter", "representation.length",
                      "representation.volume", "rep.diameter"):
        assert forbidden not in body


# --------------------------------------------------------------------------- #
# 4. Validation status is data, not UI copy
# --------------------------------------------------------------------------- #


def test_the_validation_status_lives_in_configuration():
    registry = _registry()
    representation = registry.for_model("cylinder5", MODEL_FREE)
    assert representation.validation_summary
    assert representation.validation_note
    # THE GATE IS RECORDED, so lifting the qualification is a decision against
    # a stated condition rather than a remembered intention.
    assert representation.validation_gate
    assert representation.is_experimental


def test_no_validation_wording_is_hard_coded_in_the_frontend():
    """The status changes when validation proceeds. A phrase living in
    JavaScript is one that goes stale silently."""
    page = open(os.path.join(REPO, "web", "index.html"), encoding="utf-8").read()
    for phrase in ("one physical pose", "12 frames", "12/12",
                   "multi-pose physical validation"):
        assert phrase not in page, phrase


def test_no_object_name_or_digest_is_hard_coded_in_the_frontend():
    """A second registered representation must appear without editing the UI."""
    page = open(os.path.join(REPO, "web", "index.html"), encoding="utf-8").read()
    assert "9e59f85169e2d07f" not in page
    assert "cylinder5-simref-15v" not in page


def test_the_representation_reports_that_it_has_no_physical_ground_truth():
    """NOTHING MAY RENDER A PHYSICAL ACCURACY for this method, and the reason
    travels with the data rather than living in a comment."""
    registry = _registry()
    document = registry.for_model("cylinder5", MODEL_FREE).to_dict(
        registry.store_root)
    assert document["physical_ground_truth_available"] is False


def test_readiness_reports_a_reason_and_never_offers_to_build():
    """A DASHBOARD MUST NOT TRAIN A NEURAL OBJECT FIELD. The reason points at
    the offline script rather than at a button."""
    registry = _registry(store="/nonexistent-representation-store")
    readiness = registry.ready_for("cylinder5", MODEL_FREE)
    assert not readiness.ready
    assert "model_free_build.sh" in readiness.reason
