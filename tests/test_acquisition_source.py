"""The simulated RGB-D camera is its own acquisition source.

WHAT WENT WRONG WITHOUT IT. "Camera" hid a fourth axis — WHICH camera — so a
run acquired in simulation reported "no perception service is answering" about
a webcam it never used, a missing RealSense blocked an inference that needed
none, and `object_source: camera` sat outside `available: ['sim']`.

Each source now answers for itself.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
sys.path.insert(0, os.path.join(REPO, "perception"))

from wisepack_core.acquisition import (                            # noqa: E402
    ACQUISITION_ISAAC, ACQUISITION_PLANAR, ACQUISITION_REALSENSE,
    ACQUISITION_SOURCES, AcquisitionState, acquisition_provenance,
    acquisitions_for)


def test_the_simulated_camera_is_a_source_in_its_own_right():
    assert ACQUISITION_ISAAC in ACQUISITION_SOURCES
    assert ACQUISITION_ISAAC != ACQUISITION_REALSENSE
    assert ACQUISITION_ISAAC != ACQUISITION_PLANAR


def test_only_the_simulated_source_is_simulated():
    """Provenance is carried, never inferred at the call site.

    ONE WORD PER SIDE. `simulated` rather than `synthetic`, so the acquisition
    axis, the dashboard panel and the written reports all use the same term for
    the same fact — `synthetic` still describes the CONTENT of a rendered mask,
    which is a different claim from where the frame came from.
    """
    assert acquisition_provenance(ACQUISITION_ISAAC) == "simulated"
    assert acquisition_provenance(ACQUISITION_REALSENSE) == "measured"
    assert acquisition_provenance(ACQUISITION_PLANAR) == "measured"


def test_foundationpose_can_be_fed_by_either_rgbd_source():
    """A simulated camera and a physical D435 are interchangeable to the
    estimator — that is the whole sim-to-real point."""
    sources = acquisitions_for("foundationpose_rgbd")
    assert ACQUISITION_ISAAC in sources and ACQUISITION_REALSENSE in sources
    # The planar detector cannot use depth.
    assert acquisitions_for("planar_fasterrcnn") == (ACQUISITION_PLANAR,)


def test_one_unavailable_source_does_not_veto_the_others():
    """THE HEADLINE. A missing webcam and a missing D435 must leave a working
    simulated acquisition usable."""
    state = AcquisitionState(
        current=ACQUISITION_ISAAC, available=[ACQUISITION_ISAAC],
        unavailable_reasons={ACQUISITION_PLANAR: "no perception service",
                             ACQUISITION_REALSENSE: "no D435 attached"})
    assert state.any_camera_available
    assert state.is_available(ACQUISITION_ISAAC)
    assert not state.is_available(ACQUISITION_REALSENSE)


def test_an_unavailable_source_carries_its_reason():
    state = AcquisitionState(
        available=[ACQUISITION_ISAAC],
        unavailable_reasons={ACQUISITION_REALSENSE: "no D435 attached"})
    options = {o["value"]: o for o in state.to_dict()["options"]}
    assert set(options) == set(ACQUISITION_SOURCES)
    assert options[ACQUISITION_REALSENSE]["reason"] == "no D435 attached"
    assert options[ACQUISITION_ISAAC]["available"] is True


def test_no_current_source_when_nothing_was_acquired():
    """A preset run acquired nothing; naming a camera would invent provenance."""
    document = AcquisitionState().to_dict()
    assert document["current"] == ""
    assert document["current_label"] == ""
    assert document["current_provenance"] == ""


# --------------------------------------------------------------------------- #
# The capability follows the acquisition
# --------------------------------------------------------------------------- #


def _provider(tmp_path):
    from providers.foundationpose_rgbd import FoundationPoseProvider
    from wisepack_core.rgbd import ObjectModel, ObjectModelRegistry
    mesh = tmp_path / "part.obj"
    mesh.write_text("o part\n")

    class Client:
        url = "http://fake"

        def health(self):
            return {"worker_reachable": True, "worker_ready": True,
                    "gpu_available": True,
                    "foundationpose_runtime_available": True,
                    "scorer_weights_available": True,
                    "refiner_weights_available": True,
                    "inference_available": True, "blocked_by": [],
                    # NO physical camera, which is this machine's real state.
                    "rgbd_camera_available": False,
                    "probes": {"rgbd_camera": {
                        "reason": "no RealSense device is connected"}}}

        def capability(self, health=None):
            return True, ""

    registry = ObjectModelRegistry(
        models={"part": ObjectModel(model_id="part", mesh_path="part.obj",
                                    methods=("foundationpose_rgbd",),
                                    diameter_mm=25, length_mm=342)},
        root=str(tmp_path))
    return FoundationPoseProvider(client=Client(), registry=registry)


def test_a_simulated_run_does_not_require_a_physical_realsense(tmp_path):
    """§: it must not require a physical RealSense."""
    capability = _provider(tmp_path).capability(
        acquisition=ACQUISITION_ISAAC, simulated_frames_available=True)
    assert capability["rgbd_camera_available"] is False   # honestly reported
    assert capability["rgbd_frames_available"] is True    # but not required
    assert capability["inference_ready"] is True
    assert capability["blocked_by"] == []


def test_a_realsense_run_still_requires_the_device(tmp_path):
    """The guard is not weakened for the physical path."""
    capability = _provider(tmp_path).capability(
        acquisition=ACQUISITION_REALSENSE, simulated_frames_available=True)
    assert capability["inference_ready"] is False
    assert any("RealSense" in b for b in capability["blocked_by"])


def test_a_simulated_run_with_no_frame_is_refused(tmp_path):
    """Availability is not "the concept exists" — a frame must be there."""
    capability = _provider(tmp_path).capability(
        acquisition=ACQUISITION_ISAAC, simulated_frames_available=False)
    assert capability["inference_ready"] is False
    assert any("simulated" in b for b in capability["blocked_by"])
    # And it does NOT complain about a RealSense it never needed.
    assert not any("RealSense" in b for b in capability["blocked_by"])


def test_the_capability_states_which_acquisition_it_is_about(tmp_path):
    """Two runs on one deployment have different answers; a capability with no
    stated source cannot be read correctly by either."""
    provider = _provider(tmp_path)
    assert provider.capability(acquisition=ACQUISITION_ISAAC,
                               simulated_frames_available=True)["acquisition"] \
        == ACQUISITION_ISAAC
    assert provider.capability(acquisition=ACQUISITION_REALSENSE
                               )["acquisition"] == ACQUISITION_REALSENSE
