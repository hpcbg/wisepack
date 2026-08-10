"""A simulated RGB-D camera in the WISEPACK workcell.

WHY THIS IS BUILT RATHER THAN LOADED
------------------------------------
NVIDIA ships RealSense USD assets for the **D455, D457 and D555**. It ships
**no D435**, which is the sensor WISEPACK will actually use. Loading a D455 and
calling it a D435 would put a different sensor behind the label: the D455 has a
~95 deg depth field of view and a 95 mm stereo baseline against the D435's
~87 deg and 50 mm, so a scene framed in simulation would not be the scene the
real camera sees.

So the camera is CONSTRUCTED from the D435's published geometry
(`config/rgbd_sensors.yaml`) using Isaac's own camera APIs. A pinhole projection
is fully determined by field of view and resolution, which is exactly what that
profile declares.

WHAT THIS IS NOT
----------------
* NOT a physical RealSense, and never described as one. The profile's
  `describe_backend()` supplies the wording the dashboard shows.
* NOT librealsense. `pyrealsense2` belongs to the physical acquisition path
  alone; nothing here imports it, and a simulated frame must never travel
  through a code path that implies a device was read.
* NOT a noise model. Synthetic depth here is exact within the rendered scene: no
  stereo matching, no IR pattern, no holes on low-texture metal. The profile
  lists those limitations and they are reported with the frame.

WHAT IT PRODUCES
----------------
The same generic content the physical camera produces — colour, depth aligned to
that colour by construction, and the intrinsics both share — plus two things
only a simulator can give, both explicitly labelled as synthetic ground truth:
exact instance masks, and the exact camera transform.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import omni.replicator.core as rep
from pxr import Gf, UsdGeom

from wisepack_core.rgbd_sensors import BACKEND_ISAAC, SensorProfile

from .config import LOG_SCENE

#: Frames stepped before the first read. Path-traced output is noisy for the
#: first several samples and the annotators are not populated until the render
#: pipeline has run.
WARMUP_STEPS = 24

#: The mask source this camera provides. NAMED AS GROUND TRUTH so no consumer
#: can mistake it for a segmentation result: it comes from the renderer, which
#: knows exactly which pixels belong to which prim.
MASK_SOURCE_ISAAC_GT = "isaac_instance_gt"

#: USD cameras look down -Z with +Y up; every intrinsic matrix, FoundationPose
#: and OpenCV use +Z forward with +Y down. One flip, applied in one place.
USD_TO_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0])


class SimulatedRGBDCamera:
    """An Isaac camera configured to a declared sensor profile."""

    def __init__(self, stage: Any, prim_path: str, profile: SensorProfile,
                 position_m: Tuple[float, float, float],
                 look_at_m: Tuple[float, float, float]) -> None:
        if profile.colour is None:
            raise ValueError(
                f"sensor profile {profile.sensor_id!r} declares no colour "
                "stream; there is nothing to build a camera from")
        self.profile = profile
        self.prim_path = prim_path
        self.width = profile.colour.width
        self.height = profile.colour.height

        # THE PROJECTION IS DERIVED FROM THE PROFILE, not chosen here. USD
        # expresses a camera by focal length and aperture rather than by focal
        # length in pixels, so the profile's fx/fy are converted once: with the
        # aperture fixed at an arbitrary but conventional 20.955 mm, the focal
        # length that reproduces fx is f = fx * aperture / width.
        fx, fy = profile.colour.focal_lengths_px()
        horizontal_aperture = 20.955
        focal_length = fx * horizontal_aperture / self.width
        vertical_aperture = fy and (self.height * focal_length / fy)

        camera = UsdGeom.Camera.Define(stage, prim_path)
        camera.CreateFocalLengthAttr(float(focal_length))
        camera.CreateHorizontalApertureAttr(float(horizontal_aperture))
        camera.CreateVerticalApertureAttr(float(vertical_aperture))
        # THE NEAR PLANE MATTERS. USD's default clipping range starts at 1.0 m,
        # so a part on a table 0.9 m away is clipped away entirely and the scene
        # renders as pure background — which reads as a broken sensor.
        near = float(profile.min_range_m or 0.05)
        far = float(profile.max_range_m or 20.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(max(near, 0.01), far))

        self._place(camera, position_m, look_at_m)
        self.camera = camera

        self._render_product = rep.create.render_product(
            prim_path, (self.width, self.height))
        self._rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self._depth = rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane")
        # SEMANTIC, not instance: with Isaac 6's label schema the instance
        # annotator returns objects under an empty label while the semantic one
        # resolves them to {'class': '<model_id>'} — which is what pairs a mask
        # with the right CAD model.
        self._mask = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
        for annotator in (self._rgb, self._depth, self._mask):
            annotator.attach(self._render_product)
        print(f"{LOG_SCENE} simulated {profile.model} at {prim_path}: "
              f"{self.width}x{self.height}, fx={fx:.1f} fy={fy:.1f}")

    # -- placement ---------------------------------------------------------- #

    def _place(self, camera: Any, position_m: Tuple[float, float, float],
               look_at_m: Tuple[float, float, float],
               world_up: Tuple[float, float, float] = (0.0, 0.0, 1.0)) -> None:
        """Point the camera at a target, in the stage's own frame.

        Built as an explicit basis rather than by guessing euler angles: the
        camera's -Z must point at the target, which is a USD convention and not
        an arbitrary choice.
        """
        eye = np.asarray(position_m, dtype=np.float64)
        target = np.asarray(look_at_m, dtype=np.float64)
        forward = target - eye
        norm = np.linalg.norm(forward)
        if norm < 1e-9:
            raise ValueError("the camera cannot look at its own position")
        forward = forward / norm
        # Z UP, because the WISEPACK stage is Z up — `WisepackScene.build`
        # states it and every layout coordinate assumes it. An earlier version
        # of this file hard-coded Y up, which would have rolled the camera 90
        # degrees and framed the cell sideways.
        up_hint = np.asarray(world_up, dtype=np.float64)
        if abs(float(forward @ up_hint)) > 0.999:
            # Looking straight along the up axis: any perpendicular will do, and
            # world X is the stable choice for a top-down view of the pick row.
            up_hint = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, up_hint)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        # USD camera basis: +X right, +Y up, -Z forward.
        basis = np.eye(4)
        basis[:3, 0] = right
        basis[:3, 1] = up
        basis[:3, 2] = -forward
        basis[:3, 3] = eye

        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(
            Gf.Matrix4d(*[float(v) for v in basis.T.flatten()]))
        self._world_from_usd_camera = basis

    # -- acquisition -------------------------------------------------------- #

    def warmup(self, steps: int = WARMUP_STEPS) -> None:
        for _ in range(max(0, steps)):
            rep.orchestrator.step(rt_subframes=4)

    def intrinsics_matrix(self) -> List[List[float]]:
        """The camera's own K. Nominal, from the profile it was built to."""
        return self.profile.colour.intrinsics_matrix()

    def camera_from_world(self) -> Any:
        """The 4x4 world -> CAMERA OPTICAL frame transform.

        Exact, because the camera is part of the scene. This is what the
        camera->workarea transform is derived from, and it is the mechanism the
        physical camera will later use with a measured extrinsic — the same code
        path, a different source for the numbers.
        """
        world_from_optical = self._world_from_usd_camera @ USD_TO_OPTICAL
        return np.linalg.inv(world_from_optical)

    def capture(self) -> Dict[str, Any]:
        """One synchronised colour + depth + instance-mask observation.

        Depth is ALIGNED TO COLOUR by construction: both come from one camera
        with one projection, so there is no alignment step and nothing to verify.
        That is a genuine difference from the physical D435, where alignment is
        performed and must be checked — and it is recorded here rather than
        glossed, because a synthetic frame that looks perfect is not evidence
        about a real one.
        """
        rep.orchestrator.step(rt_subframes=8)
        rgb = np.asarray(self._rgb.get_data())[..., :3]
        depth_m = np.asarray(self._depth.get_data(), dtype=np.float64)
        depth_m[~np.isfinite(depth_m)] = 0.0
        # uint16 millimetres, with 0 meaning "no measurement" — the encoding a
        # real depth sensor produces and the one the FoundationPose reader
        # expects.
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        depth_mm[depth_m <= 0] = 0

        mask_data = self._mask.get_data()
        labels = mask_data["info"]["idToLabels"]
        ids = np.asarray(mask_data["data"])

        return {
            "rgb": rgb.astype(np.uint8),
            "depth_mm": depth_mm,
            "instance_ids": ids,
            "id_to_labels": labels,
            "intrinsics": self.intrinsics_matrix(),
            "width": self.width, "height": self.height,
            "depth_scale_mm_per_unit": 1.0,
            "aligned": True,
            "alignment_method": (
                "none needed — colour and depth are rendered through one "
                "camera with one projection"),
            "camera_from_world": self.camera_from_world().tolist(),
            "mask_source": MASK_SOURCE_ISAAC_GT,
            **self.profile.describe_backend(BACKEND_ISAAC),
            "synthetic_limitations": list(self.profile.limitations),
        }

    def mask_for(self, capture: Dict[str, Any], model_id: str) -> Any:
        """The exact instance mask for one CAD model. GROUND TRUTH, labelled.

        Matched on the model_id the scene labelled the prim with, so a mask can
        never be paired with the wrong mesh.
        """
        matching = [int(key) for key, value in capture["id_to_labels"].items()
                    if model_id in str(value)]
        if not matching:
            raise ValueError(
                f"{model_id} does not appear in the rendered instance "
                f"segmentation; labels were {capture['id_to_labels']}")
        return np.isin(capture["instance_ids"], matching)


__all__ = ["SimulatedRGBDCamera", "MASK_SOURCE_ISAAC_GT", "WARMUP_STEPS",
           "USD_TO_OPTICAL"]
