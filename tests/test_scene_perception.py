"""The whole-scene RGB-D method: segmentation, classification, observations.

What these pin, without a camera and without the GPU worker:

* the plane fit and instance segmentation on a SYNTHETIC depth image whose
  objects have known footprints — sizes come back within a few millimetres
  and the ignore region takes an instance out with its reason;
* classification by footprint against the tracked catalogue, including the
  height rule that tells a plate from a short tube and the bolt proxy;
* the observation conventions the scene synchronizer relies on: the body
  centre is `object_center`, the tube axis is the measured heading, the plate's
  thickness points along the plane normal, and the CAD identity travels;
* the batch is multi-object and every non-observed instance is reported;
* measured-footprint proxies keep their measured size through `to_waste_items`;
* the demo preset packs into ONE container, both arms support it, the tracked
  workcell puts the work area 500 mm ahead of the base, and the method is
  registered wherever a method must be.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (os.path.join(REPO, "perception"),
             os.path.join(REPO, "wisepack_ws", "src", "wisepack_core")):
    if path not in sys.path:
        sys.path.insert(0, path)

cv2 = pytest.importorskip("cv2")

from providers.scene_depth_plane import (                          # noqa: E402
    GEOMETRY_MEASURED_FOOTPRINT, METHOD, build_batch, load_catalogue,
    orientation_on_plane)
from scene_segmentation import (Instance, load_capture,            # noqa: E402
                                segment_scene)
from wisepack_core.acquisition import (ACQUISITION_REALSENSE,      # noqa: E402
                                       METHOD_ACQUISITIONS)
from wisepack_core.domain import GEOMETRY_SOURCE_CAD_MESH          # noqa: E402
from wisepack_core.execution import preset_physical_compatibility  # noqa: E402
from wisepack_core.generator import PRESETS, build_scenario        # noqa: E402
from wisepack_core.isaac_transform import DEFAULT_LAYOUT, pick_yaw_deg  # noqa: E402
from wisepack_core.packing import pack_optimized                   # noqa: E402
from wisepack_core.perception import (KNOWN_PERCEPTION_METHODS,    # noqa: E402
                                      PERCEPTION_METHOD_ALIASES,
                                      PerceptionMethod)
from wisepack_core.pose import CAMERA_OPTICAL_FRAME                # noqa: E402
from wisepack_core.robots import load_registry                     # noqa: E402
from wisepack_core.scene_sync import synchronize_scene             # noqa: E402
from wisepack_core.workcell import load_workcell                   # noqa: E402

# --------------------------------------------------------------------------- #
# A synthetic bench: a plane at 530 mm, seen by a D435-like camera
# --------------------------------------------------------------------------- #

W, H = 1280, 720
K = np.array([[913.0, 0.0, 648.0], [0.0, 912.0, 375.0], [0.0, 0.0, 1.0]])
PLANE_Z = 530.0


def _bench(objects):
    """(bgr, depth_mm) with `objects` = [(cx_mm, cy_mm, length, width, height, yaw_deg)].

    Cork-coloured plane; each object a raised, blue-grey rotated rectangle.
    """
    depth = np.full((H, W), PLANE_Z, dtype=np.float64)
    bgr = np.zeros((H, W, 3), dtype=np.uint8)
    bgr[...] = (95, 135, 165)                                   # cork: red > blue
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) * PLANE_Z / K[0, 0]
    y = (v - K[1, 2]) * PLANE_Z / K[1, 1]
    for cx, cy, length, width, height, yaw in objects:
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        along = (x - cx) * c + (y - cy) * s
        across = -(x - cx) * s + (y - cy) * c
        inside = (np.abs(along) <= length / 2.0) & (np.abs(across) <= width / 2.0)
        depth[inside] = PLANE_Z - height
        bgr[inside] = (150, 120, 100)                           # steel: blue > red
    depth += np.random.default_rng(1).normal(0.0, 0.8, depth.shape)
    return bgr, depth.astype(np.uint16)


BENCH = [
    (-150.0, 0.0, 342.0, 25.0, 25.0, 85.0),      # long tube (cylinder5-sized)
    (60.0, -90.0, 190.0, 35.0, 35.0, 20.0),      # medium tube
    (60.0, 80.0, 70.0, 35.0, 35.0, -40.0),       # short tube
    (180.0, 40.0, 60.0, 40.0, 8.0, 10.0),        # plate
    (150.0, -130.0, 40.0, 20.0, 20.0, 0.0),      # small cylinder
    (240.0, 120.0, 45.0, 6.0, 6.0, 30.0),        # bolt
    (-330.0, 150.0, 78.0, 30.0, 15.0, 70.0),     # the marker at the lower left
]


@pytest.fixture(scope="module")
def scene():
    bgr, depth = _bench(BENCH)
    catalogue = load_catalogue()
    seg = segment_scene(bgr, depth, K, {"ignore_regions_px": catalogue.ignore_regions_px})
    return bgr, depth, seg, catalogue


def test_the_plane_is_found_with_the_normal_toward_the_camera(scene):
    _, _, seg, _ = scene
    assert seg.plane.residual_mm < 2.0
    assert seg.plane.inlier_fraction > 0.6
    assert seg.plane.normal[2] < 0                       # -Z: toward the camera
    assert seg.plane.height(np.array([[0.0, 0.0, PLANE_Z - 20.0]]))[0] == pytest.approx(20.0, abs=1.5)


def test_every_object_is_one_instance_with_its_footprint(scene):
    _, _, seg, _ = scene
    live = [i for i in seg.instances if not i.ignored]
    assert len(live) == len(BENCH) - 1
    measured = sorted((i.length_mm, i.width_mm, i.height_mm) for i in live)
    expected = sorted((o[2], o[3], o[4]) for o in BENCH[:-1])
    for (ml, mw, mh), (el_, ew, eh) in zip(measured, expected):
        assert ml == pytest.approx(el_, abs=6.0)
        assert mw == pytest.approx(ew, abs=6.0)
        assert mh == pytest.approx(eh, abs=3.0)


def test_the_heading_and_centre_are_measured_on_the_plane(scene):
    _, _, seg, _ = scene
    tube = max(seg.instances, key=lambda i: i.length_mm)
    cx, cy, _, _, _, yaw = BENCH[0]
    assert tube.centre_mm[0] == pytest.approx(cx, abs=3.0)
    assert tube.centre_mm[1] == pytest.approx(cy, abs=3.0)
    assert tube.centre_mm[2] == pytest.approx(PLANE_Z, abs=3.0)
    heading = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0)
    dot = abs(sum(a * b for a, b in zip(tube.axis_line, heading)))
    assert dot == pytest.approx(1.0, abs=0.01)


def test_the_configured_ignore_region_takes_the_marker_out_with_a_reason(scene):
    _, _, seg, catalogue = scene
    assert catalogue.ignore_regions_px, "the tracked catalogue declares no ignore region"
    ignored = [i for i in seg.instances if i.ignored]
    assert len(ignored) == 1
    assert ignored[0].centroid_px[0] < 150
    assert "ignore region" in ignored[0].ignored


def test_classification_by_footprint_names_the_registry_models(scene):
    _, _, seg, catalogue = scene
    classes = set()
    for inst in seg.instances:
        if inst.ignored:
            continue
        cls = catalogue.classify(inst)
        assert cls is not None, inst.to_dict()
        classes.add((cls.demo_type, cls.model_id))
    assert classes == {("long_tube", "cylinder5"), ("medium_tube", "cylinder3"),
                       ("short_tube", "cylinder2"), ("plate", "plate2"),
                       ("small_cylinder", "cylinder1"), ("bolt", "")}


def test_a_flat_footprint_is_a_plate_and_a_tall_one_a_short_tube():
    catalogue = load_catalogue()
    flat = Instance(1, 1000, (0, 0), (0, 0, 1, 1), (0, 0, 0), (1, 0, 0), 62.0, 39.0, 7.0, 6.0)
    tall = Instance(2, 1000, (0, 0), (0, 0, 1, 1), (0, 0, 0), (1, 0, 0), 62.0, 39.0, 28.0, 26.0)
    assert catalogue.classify(flat).demo_type == "plate"
    assert catalogue.classify(tall).demo_type == "short_tube"


def test_a_footprint_that_matches_nothing_is_unclassified_not_guessed():
    catalogue = load_catalogue()
    desk = Instance(3, 40000, (0, 0), (0, 0, 1, 1), (0, 0, 0), (1, 0, 0), 404.0, 73.0, 10.0, 8.0)
    assert catalogue.classify(desk) is None


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


def _batch(scene):
    _, _, seg, catalogue = scene
    return build_batch(seg.instances, seg.plane, catalogue=catalogue,
                       batch_id="physical-scene-1", dataset="synthetic",
                       repo_root=REPO)


def test_the_batch_is_multi_object_and_reports_everything_measured(scene):
    batch, report = _batch(scene)
    assert batch.ok
    assert batch.count == len(BENCH) - 1
    assert batch.perception_method == METHOD
    assert batch.acquisition == ACQUISITION_REALSENSE
    assert batch.frame_id == CAMERA_OPTICAL_FRAME
    statuses = [c.status for c in report]
    assert statuses.count("observed") == len(BENCH) - 1
    assert statuses.count("ignored") == 1
    counts = batch.detector_status["counts"]
    assert counts["measured"] == len(report)
    assert batch.detector_status["identity_note"].startswith("object class assigned by FOOTPRINT")


def test_the_body_centre_is_the_object_centre_whatever_the_model_origin(scene):
    batch, _ = _batch(scene)
    by_model = {o.object_model_id: o for o in batch.observations}
    c5 = by_model["cylinder5"]
    # Cylinder5's CAD origin is 141 mm from its body; the OBSERVATION places the
    # body at the measured footprint plus one radius toward the camera.
    centre = c5.object_center
    cx, cy = BENCH[0][0], BENCH[0][1]
    assert centre[0] == pytest.approx(cx, abs=3.0)
    assert centre[1] == pytest.approx(cy, abs=3.0)
    assert centre[2] == pytest.approx(PLANE_Z - 12.5, abs=3.0)
    assert math.dist((c5.x_mm, c5.y_mm, c5.z_mm), centre) > 100.0
    assert (c5.length_mm, c5.diameter_mm, c5.inner_diameter_mm) == (342, 25, 19)
    assert c5.geometry_source == "cad_model"


def test_the_tube_axis_is_the_measured_heading(scene):
    batch, _ = _batch(scene)
    c5 = next(o for o in batch.observations if o.object_model_id == "cylinder5")
    yaw = BENCH[0][5]
    heading = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0)
    dot = abs(sum(a * b for a, b in zip(c5.tube_axis, heading)))
    assert dot == pytest.approx(1.0, abs=0.02)


def test_the_plate_lies_flat_with_its_thickness_along_the_plane_normal(scene):
    batch, _ = _batch(scene)
    plate = next(o for o in batch.observations if o.object_model_id == "plate2")
    thin = plate.orientation.axis("z")                     # the mesh's 8 mm axis
    up = (0.0, 0.0, -1.0)                                  # toward the camera
    assert abs(sum(a * b for a, b in zip(thin, up))) == pytest.approx(1.0, abs=0.02)
    assert plate.object_center[2] == pytest.approx(PLANE_Z - 4.0, abs=3.0)


def test_orientation_on_plane_maps_the_task_axis_and_the_thin_axis():
    q = orientation_on_plane(heading=(0.0, 1.0, 0.0), plane_normal_up=(0.0, 0.0, -1.0),
                             task_axis=(0.9284, -0.3716, 0.0),
                             extents_mm=(315.5, 148.0, 25.0))
    assert q.rotate((0.9284, -0.3716, 0.0)) == pytest.approx((0.0, 1.0, 0.0), abs=1e-4)
    assert q.rotate((0.0, 0.0, 1.0)) == pytest.approx((0.0, 0.0, -1.0), abs=1e-4)


def test_a_bolt_is_a_measured_proxy_that_keeps_its_size(scene):
    batch, _ = _batch(scene)
    bolt = next(o for o in batch.observations if not o.object_model_id)
    assert bolt.geometry_source == GEOMETRY_MEASURED_FOOTPRINT
    assert bolt.diameter_mm == 6
    assert bolt.length_mm == pytest.approx(45, abs=6)
    items = batch.to_waste_items(permitted_axes=("x", "y"))
    proxy = next(i for i in items if not i.model_id)
    assert (proxy.outer_diameter_mm, proxy.length_mm) == (bolt.diameter_mm, bolt.length_mm)
    assert proxy.length_mm < 100                            # not the 215 mm proxy
    cad = next(i for i in items if i.model_id == "cylinder5")
    assert cad.geometry_source == GEOMETRY_SOURCE_CAD_MESH


# --------------------------------------------------------------------------- #
# Into the workcell: one container, every object synchronized
# --------------------------------------------------------------------------- #


def test_the_whole_scene_synchronizes_and_packs_into_one_container(scene):
    batch, _ = _batch(scene)
    scenario = build_scenario("isaac_scene_physical", seed=42)
    items = batch.to_waste_items(permitted_axes=("x", "y"))
    scenario.items = items
    frames = load_workcell()
    spec = synchronize_scene(items, batch, frames, run_id="run-t",
                             scenario_revision=2, layout=DEFAULT_LAYOUT)
    assert len(spec.objects) == len(items)
    assert {o.item_id for o in spec.objects} == {i.item_id for i in items}
    for obj in spec.objects:
        reach = math.hypot(obj.source_pose.x_mm, obj.source_pose.y_mm) / 1000.0
        assert DEFAULT_LAYOUT.robot_min_reach_m <= reach <= DEFAULT_LAYOUT.robot_max_reach_m
    plan = pack_optimized(scenario)
    assert not plan.unplaced_item_ids
    assert len(plan.containers_used) == 1
    yaws = [pick_yaw_deg(o.source_pose) for o in spec.objects]
    assert all(-90.0 < y <= 90.0 for y in yaws)


def test_the_demo_preset_uses_one_container_and_both_arms_support_it():
    assert PRESETS["isaac_scene_physical"]["max_containers"] == 1
    assert PRESETS["isaac_cylinder5_physical"]["max_containers"] == 1
    scenario = build_scenario("isaac_scene_physical", seed=42)
    assert scenario.max_containers == 1
    for profile in load_registry().profiles.values():
        ok, reason = preset_physical_compatibility("isaac_scene_physical", profile)
        assert ok, reason


def test_the_tracked_workcell_puts_the_work_area_500_mm_ahead_of_the_base():
    frames = load_workcell()
    assert frames.available, frames.unavailable_reason
    assert frames.workarea_to_table.translation_mm == pytest.approx((500.0, 0.0, 0.0))
    assert frames.camera_to_workarea.revision == "demo-2026-09-12"


def test_the_method_is_registered_everywhere_a_method_must_be():
    method = PerceptionMethod.RGBD_SCENE_DEPTH_PLANE
    assert method.value in KNOWN_PERCEPTION_METHODS
    assert METHOD_ACQUISITIONS[method.value] == (ACQUISITION_REALSENSE,)
    assert PERCEPTION_METHOD_ALIASES["scene"] == method.value
    assert method.is_multi_object and method.requires_depth
    assert not method.requires_object_model and not method.requires_representation
    assert not method.is_foundationpose
    assert method.measures == ("x", "y", "z", "yaw")
    assert "footprint" in method.selector_detail


def test_the_isaac_scene_builds_as_many_containers_as_the_scenario_allows():
    src = open(os.path.join(REPO, "simulators", "isaac", "wisepack_isaac.py"),
               encoding="utf-8").read()
    assert 'for n in (1, 2)' not in src
    assert 'getattr(self.scenario, "max_containers"' in src


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(REPO, ".cache-perception", "rgbd-captures",
                                   "cylinder5-20260912-045336")),
    reason="the recorded whole-bench capture is not on this machine")
def test_the_recorded_bench_capture_yields_the_documented_scene():
    root = os.path.join(REPO, ".cache-perception", "rgbd-captures", "cylinder5-20260912-045336")
    bgr, depth, K_ = load_capture(root)
    catalogue = load_catalogue()
    seg = segment_scene(bgr, depth, K_, {"ignore_regions_px": catalogue.ignore_regions_px,
                                         "roi_px": catalogue.scene_roi_px})
    batch, report = build_batch(seg.instances, seg.plane, catalogue=catalogue, repo_root=REPO)
    by_class = {}
    for c in report:
        if c.status == "observed":
            by_class[c.demo_class.demo_type] = by_class.get(c.demo_class.demo_type, 0) + 1
    assert by_class == {"long_tube": 2, "medium_tube": 2, "short_tube": 3,
                        "plate": 3, "small_cylinder": 3, "bolt": 5}
    assert sum(1 for c in report if c.status == "ignored") == 1
