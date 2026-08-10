"""`depth_plane_foreground` — a mask measured from RGB-D geometry.

NO CAMERA, NO GPU, NO DOCKER. The scenes are synthesised: a tilted plane at a
known distance with a known box standing on it, rendered as a depth image
through real intrinsics. That makes the expected answer checkable by hand, and
it means the failure modes can be produced deliberately — which is the part that
matters, because a mask that fails must fail with the right reason.

WHAT THIS METHOD IS. One known object on a stable surface, segmented by fitting
the surface and keeping what stands on it. It is NOT clutter segmentation and it
is NOT a ground-truth mask; the name says the mechanism for that reason.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "perception", "foundationpose", "worker"))

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from segmentation import (                                       # noqa: E402
    DEFAULTS, METHOD_DEPTH_PLANE, PLANNED_METHODS, SegmentationError,
    depth_plane_foreground, fit_plane, deproject, segment)

#: A plausible RealSense colour intrinsic at 640x480. The exact numbers do not
#: matter; that the SAME ones are used to deproject and to check does.
INTRINSICS = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
WIDTH, HEIGHT = 640, 480


def _plane_depth(offset_mm, normal):
    """The depth image of a REAL 3-D plane, seen through INTRINSICS.

    A plane is `n . p = d`. Substituting the pinhole deprojection
    `x = (u-cx)z/fx`, `y = (v-cy)z/fy` gives

        z(u, v) = d / ( nx (u-cx)/fx + ny (v-cy)/fy + nz )

    which is 1/linear in the pixel coordinates, NOT linear. An earlier version
    of this helper made depth linear in v and called it a tilted plane; it is
    not one, and the fitter was right to reject 9% of it. Generating the scene
    from the same geometry the fitter assumes is what makes the test meaningful.
    """
    fx, fy = INTRINSICS[0][0], INTRINSICS[1][1]
    cx, cy = INTRINSICS[0][2], INTRINSICS[1][2]
    us = np.arange(WIDTH)[None, :].astype(np.float64)
    vs = np.arange(HEIGHT)[:, None].astype(np.float64)
    nx, ny, nz = normal
    denominator = nx * (us - cx) / fx + ny * (vs - cy) / fy + nz
    return float(offset_mm) / denominator


def _scene(plane_mm=800.0, tilt=0.0, box=None, noise_mm=0.0, seed=0,
           invalid_fraction=0.0):
    """A depth image: a tilted plane, optionally with boxes standing on it.

    `box` is (u0, u1, v0, v1, height_mm) or a list of them. An object standing
    `h` above the surface is the SAME plane with its offset reduced by `h` —
    i.e. parallel and closer to the camera along the normal, which is what
    "standing on the table" means geometrically.
    """
    rng = np.random.default_rng(seed)
    normal = np.array([0.0, float(tilt), 1.0])
    normal = normal / np.linalg.norm(normal)
    depth = _plane_depth(plane_mm, normal)

    for entry in ([] if box is None else (box if isinstance(box, list) else [box])):
        u0, u1, v0, v1, height_mm = entry
        raised = _plane_depth(plane_mm - float(height_mm), normal)
        depth[v0:v1, u0:u1] = raised[v0:v1, u0:u1]

    if noise_mm:
        depth = depth + rng.normal(0.0, noise_mm, depth.shape)
    if invalid_fraction:
        holes = rng.random(depth.shape) < invalid_fraction
        depth[holes] = 0.0
    return np.clip(depth, 0, None).astype(np.uint16)


def _run(depth, **options):
    return depth_plane_foreground(depth, INTRINSICS, options)


def _segmentation_code() -> str:
    """The module's executable source, with comments and docstrings removed."""
    import io
    import tokenize
    source = open(os.path.join(REPO, "perception", "foundationpose", "worker",
                               "segmentation.py"), encoding="utf-8").read()
    kept, previous = [], tokenize.INDENT
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if (token.type == tokenize.STRING
                and previous in (tokenize.INDENT, tokenize.NEWLINE,
                                 tokenize.NL, tokenize.DEDENT)):
            previous = token.type
            continue
        previous = token.type
        kept.append(token.string)
    return "\n".join(kept)


# --------------------------------------------------------------------------- #
# The geometry
# --------------------------------------------------------------------------- #


def test_a_plane_is_fitted_to_a_tilted_surface():
    points, _ = deproject(_scene(tilt=0.4), INTRINSICS)
    plane, inliers, residual = fit_plane(points, 6.0, 200, 0)
    assert plane is not None
    assert inliers.mean() > 0.95, "a pure plane should be almost all inliers"
    assert residual < 2.0


def test_the_plane_fit_is_deterministic():
    """A plane that wandered between runs would make pose repeatability a
    measurement of the segmentation instead of the estimator."""
    depth = _scene(tilt=0.3, box=(250, 390, 180, 300, 40), noise_mm=1.5)
    first = _run(depth).diagnostics
    second = _run(depth).diagnostics
    assert first["mask_pixels"] == second["mask_pixels"]
    assert first["plane_normal"] == second["plane_normal"]


def test_an_object_standing_on_the_plane_is_segmented():
    depth = _scene(tilt=0.3, box=(250, 390, 180, 300, 40))
    result = _run(depth)
    assert result.valid, result.reason
    assert result.method == METHOD_DEPTH_PLANE
    assert result.diagnostics["plane_detected"] is True
    # The mask should be the box and essentially nothing else.
    expected = (390 - 250) * (300 - 180)
    assert result.diagnostics["mask_pixels"] == pytest.approx(expected, rel=0.1)
    assert result.mask[240, 320]           # inside the box
    assert not result.mask[10, 10]         # bare table


def test_no_absolute_depth_threshold_is_used():
    """`depth < 0.7 m` encodes where the table happened to be. The same scene
    at a different range must segment identically."""
    near = _run(_scene(plane_mm=600.0, tilt=0.3, box=(250, 390, 180, 300, 40)))
    far = _run(_scene(plane_mm=1400.0, tilt=0.3, box=(250, 390, 180, 300, 40)))
    assert near.valid and far.valid
    assert near.diagnostics["mask_pixels"] == pytest.approx(
        far.diagnostics["mask_pixels"], rel=0.15)
    # CODE ONLY. The module docstring explains why `depth < 0.7 m` is wrong, so
    # it necessarily contains that number; a check that cannot tell the
    # prohibition from the thing prohibited would forbid writing it down.
    assert "0.7" not in _segmentation_code()


def test_the_diagnostics_report_every_required_field():
    result = _run(_scene(tilt=0.3, box=(250, 390, 180, 300, 40)))
    document = result.to_dict()
    for field in ("plane_detected", "plane_residual_mm", "foreground_points",
                  "mask_pixels", "mask_valid", "selected_component",
                  "mask_source"):
        assert field in document, f"diagnostics omit {field}"
    # NAMED AFTER ITS MECHANISM, never after its authority. ("foreground"
    # contains "ground"; what must not appear is a claim of truth.)
    assert document["mask_source"] == METHOD_DEPTH_PLANE
    assert "truth" not in document["mask_source"]


def test_the_object_component_is_selected_among_several():
    """Two objects on the table: the controlled single-object test selects one,
    and says which."""
    depth = _scene(tilt=0.3, box=[(120, 200, 150, 230, 40),
                                  (300, 460, 180, 330, 40)])
    result = _run(depth)
    assert result.diagnostics["components"] == 2
    assert result.diagnostics["selected_component"] is not None
    # "largest" by default.
    areas = result.diagnostics["component_areas_px"]
    assert result.diagnostics["mask_pixels"] == pytest.approx(areas[0], rel=0.05)


def test_the_centre_component_can_be_selected_instead():
    depth = _scene(tilt=0.3, box=[(20, 130, 40, 150, 40),
                                  (300, 380, 210, 280, 40)])
    largest = _run(depth, component="largest")
    centre = _run(depth, component="centre")
    assert largest.diagnostics["selected_component"] != \
        centre.diagnostics["selected_component"]


def test_a_work_area_radius_excludes_what_is_outside_it():
    depth = _scene(tilt=0.3, box=[(300, 380, 210, 280, 40),
                                  (10, 90, 20, 90, 40)])
    unbounded = _run(depth)
    bounded = _run(depth, roi_radius_mm=120.0)
    assert unbounded.diagnostics["components"] == 2
    assert bounded.diagnostics["components"] == 1


# --------------------------------------------------------------------------- #
# Failing clearly
# --------------------------------------------------------------------------- #


def test_an_empty_table_yields_no_mask_and_says_why():
    result = _run(_scene(tilt=0.3))
    assert not result.valid
    assert "no points stand" in result.reason
    assert result.diagnostics["plane_detected"] is True


def test_a_scene_with_no_dominant_plane_is_refused():
    """This method ASSUMES a work surface. Without one it must say so rather
    than fit a plane to whatever is there."""
    rng = np.random.default_rng(1)
    depth = (rng.random((HEIGHT, WIDTH)) * 1500 + 300).astype(np.uint16)
    result = _run(depth)
    assert not result.valid
    assert "dominant" in result.reason or "no plane" in result.reason


def test_a_depth_image_with_almost_no_valid_pixels_is_refused():
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    result = _run(depth)
    assert not result.valid
    assert "valid pixels" in result.reason


def test_a_raised_surface_larger_than_the_table_becomes_the_plane():
    """Worth pinning, because it is why the "nearly the whole image" guard can
    never fire from THIS method: RANSAC fits the DOMINANT surface. Raise more
    than half the frame and the raised part is the dominant one, so it is the
    plane and nothing stands above it. The method reports "no object" rather
    than returning a mask of everything, which is the right answer."""
    result = _run(_scene(tilt=0.3, box=(0, WIDTH, 0, 300, 40)))
    assert not result.valid
    assert result.diagnostics["plane_inlier_fraction"] > 0.6
    assert "no points stand" in result.reason


def test_the_whole_image_guard_is_wired_and_names_itself():
    """The guard exists for masks that did NOT come from a plane fit — a YOLO
    mask, or a debug rectangle — where "the work surface was not removed" is a
    real failure. Exercised by tightening the limit below what this scene
    produces, since this method structurally cannot exceed the default."""
    depth = _scene(tilt=0.3, box=(180, 460, 120, 360, 40))
    assert _run(depth).valid
    result = _run(depth, max_area_fraction=0.05)
    assert not result.valid
    assert "nearly all of it" in result.reason


def test_a_negligible_mask_is_rejected():
    depth = _scene(tilt=0.3, box=(318, 326, 238, 246, 40))
    result = _run(depth, min_component_area_px=1)
    assert not result.valid
    assert "negligible" in result.reason or "large enough" in result.reason


def test_an_object_barely_above_the_surface_is_rejected():
    """Not enough clearance to distinguish it from the surface itself."""
    depth = _scene(tilt=0.3, box=(250, 390, 180, 300, 9))
    result = _run(depth, min_height_mm=2.0, min_separation_mm=25.0)
    assert not result.valid
    assert "above the fitted surface" in result.reason


def test_this_method_always_produces_a_mask_backed_by_valid_depth():
    """STRUCTURAL, and worth stating: the mask is derived FROM valid depth
    pixels, so it cannot contain a region the estimator has no geometry for.
    Punching holes in the depth removes those pixels from the mask rather than
    leaving them inside it."""
    depth = _scene(tilt=0.3, box=(250, 390, 180, 300, 40))
    holed = depth.copy()
    holed[200:260, 280:360] = 0
    result = _run(holed)
    assert result.valid, result.reason
    # Not exactly 1.0: closing small holes can pull a few no-depth pixels back
    # in at the boundary, which is the trade the morphology is there to make.
    assert result.diagnostics["valid_depth_fraction_in_mask"] > 0.99


def test_the_valid_depth_guard_is_wired_and_names_itself():
    """The guard matters for masks from OTHER providers — a YOLO mask can
    happily cover a region the depth sensor never returned. Exercised here by
    demanding more valid depth than any mask can have."""
    depth = _scene(tilt=0.3, box=(250, 390, 180, 300, 40))
    result = _run(depth, min_valid_depth_fraction=1.1)
    assert not result.valid
    assert "valid depth" in result.reason


def test_too_many_components_are_rejected_as_noisy():
    boxes = [(40 + 90 * i, 90 + 90 * i, 100, 170, 40) for i in range(5)]
    result = _run(_scene(tilt=0.3, box=boxes))
    assert not result.valid
    assert "components" in result.reason


# --------------------------------------------------------------------------- #
# The abstraction
# --------------------------------------------------------------------------- #


def test_an_unknown_method_is_an_error_with_no_fallback():
    """A silent fallback would produce a mask whose provenance nobody could
    state — and provenance is the entire reason `mask_source` is reported."""
    with pytest.raises(SegmentationError) as exc:
        segment("magic", _scene(), INTRINSICS)
    assert "unknown segmentation method" in str(exc.value)


def test_a_planned_but_unimplemented_method_says_so_rather_than_substituting():
    assert "yolo_instance" in PLANNED_METHODS
    with pytest.raises(SegmentationError) as exc:
        segment("yolo_instance", _scene(), INTRINSICS)
    assert "not implemented" in str(exc.value)
    # And it names what it would take, rather than just failing.
    assert "cluttered" in str(exc.value)


def test_the_method_is_not_described_as_ground_truth_anywhere():
    """The name states the mechanism. Nothing in the CODE may call it truth —
    the prose says the opposite, at length, and is allowed to."""
    code = _segmentation_code()
    assert "ground_truth" not in code
    assert "ground truth" not in code.lower()


def test_no_bounding_box_mask_is_manufactured():
    """A rectangle from a detection box is not a segmentation. Permitted only as
    an explicit debug experiment, and there is no such code path here."""
    code = _segmentation_code()
    assert "rectangle" not in code
    assert "bounding_box_mask" not in code


def test_the_defaults_are_relative_to_the_measured_plane():
    """Every threshold is a distance from a MEASURED surface, never from the
    camera."""
    for key in ("plane_tolerance_mm", "min_height_mm", "max_height_mm"):
        assert key in DEFAULTS
    assert "max_depth_mm" not in DEFAULTS
    assert "depth_threshold_mm" not in DEFAULTS


# --------------------------------------------------------------------------- #
# The reference bolt keeps its supplied mask
# --------------------------------------------------------------------------- #


def test_the_bolt_regression_defaults_to_the_supplied_mask():
    """It exists to test the KNOWN reference FoundationPose inputs. A
    regression that recomputes its own inputs cannot detect a change in the
    estimator."""
    source = open(os.path.join(REPO, "perception", "foundationpose", "worker",
                               "app.py"), encoding="utf-8").read()
    assert 'request.get("mask_source", "dataset")' in source


def test_the_provider_never_requests_a_computed_mask_for_the_reference_run():
    provider = open(os.path.join(REPO, "perception", "providers",
                                 "foundationpose_rgbd.py"),
                    encoding="utf-8").read()
    assert "depth_plane_foreground" not in provider


def test_segmentation_is_a_provider_registry_not_a_hard_coded_call():
    """FoundationPose consumes a binary mask and must not care which provider
    made it; adding the cluttered-scene method must be a new entry, not a
    change to the estimator."""
    from segmentation import METHODS, PLANNED_METHODS
    assert METHOD_DEPTH_PLANE in METHODS
    assert callable(METHODS[METHOD_DEPTH_PLANE])
    assert "yolo_instance" in PLANNED_METHODS


def test_the_planned_method_points_at_the_documented_path():
    from segmentation import PLANNED_METHODS
    assert "SEGMENTATION.md" in PLANNED_METHODS["yolo_instance"]
    assert os.path.isfile(os.path.join(REPO, "perception", "foundationpose",
                                       "SEGMENTATION.md"))


def test_the_documentation_records_what_labelled_data_actually_exists():
    """The Stage 2 decision rests on facts that were checked: the tutorial's
    checkpoint is bolt-only and COCO has no pipe class, so no off-the-shelf
    model recognises the WISEPACK parts."""
    document = open(os.path.join(REPO, "perception", "foundationpose",
                                 "SEGMENTATION.md"), encoding="utf-8").read()
    assert "18 train / 6 val" in document
    assert "bolt-only" in document or "`0: bolt`" in document
    assert "COCO" in document
    # And the CAD-driven alternative, rather than inventing labels.
    assert "Isaac Sim" in document


# --------------------------------------------------------------------------- #
# Symmetry semantics for straight tubes (measured, not assumed)
# --------------------------------------------------------------------------- #


def test_an_axial_tube_axis_is_a_line_not_an_arrow():
    """For a straight tube with identical ends, a 180 deg rotation about a
    transverse axis exchanges the two ends and changes nothing. So an end swap
    must never be counted as pose error."""
    sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
    from wisepack_core.pose import (Orientation, Symmetry, SymmetryType,
                                    symmetry_aware_angle_deg)
    axial = Symmetry(type=SymmetryType.AXIAL, axis="z")
    upright = Orientation.identity()
    # Flipped end for end: +z becomes -z.
    flipped = Orientation.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert symmetry_aware_angle_deg(upright, flipped, axial) == pytest.approx(0.0, abs=1e-9)


def test_axial_spin_is_never_counted_as_pose_error():
    sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
    from wisepack_core.pose import (Orientation, Symmetry, SymmetryType,
                                    symmetry_aware_angle_deg)
    axial = Symmetry(type=SymmetryType.AXIAL, axis="z")
    for spin in (17.0, 90.0, 143.0, 359.0):
        assert symmetry_aware_angle_deg(
            Orientation.identity(), Orientation.from_yaw_deg(spin), axial
        ) == pytest.approx(0.0, abs=1e-9)


def test_a_real_axis_tilt_IS_counted_as_error():
    """The metric must not discard everything — only the unobservable part."""
    sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
    from wisepack_core.pose import (Orientation, Symmetry, SymmetryType,
                                    symmetry_aware_angle_deg)
    axial = Symmetry(type=SymmetryType.AXIAL, axis="z")
    tilted = Orientation.from_matrix(
        [[1, 0, 0], [0, 0.7071068, -0.7071068], [0, 0.7071068, 0.7071068]])
    assert symmetry_aware_angle_deg(
        Orientation.identity(), tilted, axial) == pytest.approx(45.0, abs=1e-3)


def test_every_cylinder_is_declared_straight_and_none_is_bent():
    """The engineering table gives D x L x T for five STRAIGHT round tubes. An
    earlier revision declared Cylinder5 a bent hairpin on the strength of a
    coordinate-axis-only symmetry test; that conclusion is gone."""
    import yaml
    with open(os.path.join(REPO, "config", "perception_objects.yaml"),
              encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    entries = {e["model_id"]: e for e in document["objects"]}
    for model_id in ("cylinder1", "cylinder2", "cylinder3", "cylinder4",
                     "cylinder5"):
        entry = entries[model_id]
        assert entry["object_type"] == "pipe_section", model_id
        assert "hairpin" not in str(entry).lower(), model_id
        assert "bent" not in str(entry.get("description", "")).lower(), model_id


def test_the_straight_square_ended_tubes_are_axial_and_cylinder5_is_not():
    """Measured: C1-C4 spin unobservable; C5's saddle ends make spin
    observable, leaving only the end swap ambiguous."""
    import yaml
    with open(os.path.join(REPO, "config", "perception_objects.yaml"),
              encoding="utf-8") as handle:
        entries = {e["model_id"]: e for e in yaml.safe_load(handle)["objects"]}
    for model_id in ("cylinder1", "cylinder2", "cylinder3"):
        assert entries[model_id]["symmetry"]["type"] == "axial", model_id
    # C4 and C5 both carry intentional saddle-cut ends, so spin is
    # geometrically observable and the end swap is what remains ambiguous.
    for model_id in ("cylinder4", "cylinder5"):
        assert entries[model_id]["symmetry"]["type"] == "discrete", model_id
        assert entries[model_id]["symmetry"]["fold"] == 2, model_id
    # ... and for PICKING, every one of them is an axis line.
    for model_id in ("cylinder1", "cylinder2", "cylinder3", "cylinder4",
                     "cylinder5"):
        assert entries[model_id]["task_pose_equivalence"] == "axis_line"


def test_the_symmetry_tool_tests_the_meshs_own_axis():
    """The coordinate-axis-only test is what produced the wrong answer."""
    source = open(os.path.join(REPO, "scripts", "measure_mesh_symmetry.py"),
                  encoding="utf-8").read()
    assert "principal" in source
    assert "centroid" in source.lower()
