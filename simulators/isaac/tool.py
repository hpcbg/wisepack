"""The combined gripper + cutter end effector: a VISUAL, kinematically animated
shear on the Panda hand.

WHAT IT IS. One end effector with two independently actuated mechanisms:

    grasp_frame   the parallel-jaw gripper the robot already has — the point
                  between its fingertips, `tool_centre_point_m` along the hand's
                  approach axis. This is the FUNCTIONAL grasp: the fingers close
                  on the tube and the grasp joint holds it (see grasp.py);
    cut_frame     a shear mounted on the SAME end effector: two blades on a
                  bracket bolted to the hand, offset a fixed distance from the
                  grasp frame ALONG the direction a gripped tube lies, closing
                  across the tube exactly as the fingers do.

The transform between the two frames is FIXED and DECLARED (`ToolSpec`), so a
sequence that puts the cut frame on a planner-selected cut plane knows where the
fingers are — on the segment that will be retained — by construction.

WHAT IT IS NOT. It is not a tool changer, it is not a cutting-physics model, and
IT DOES NOT TAKE PART IN PHYSX CONTACT AT ALL. The bracket and the blades are
plain USD geometry parented under the hand link — no rigid body, no collision
API, no joints, no mass — and the blades are ANIMATED by writing their local
translation each frame. The cut itself is a DISCRETE EVENT handled by the scene
(`WisepackScene.split_item`) when the blades have visibly met. Steel is not
fractured by contact forces here; the demonstrator says so wherever the cut is
reported.

WHY VISUAL-ONLY, measured. The first version of this tool authored the blades
as driven rigid bodies with `physics:collisionEnabled = false`. Isaac Sim 6.0.1
still resolved contacts against them: in a controlled check a free tube whose
end overlapped a blade by 1 mm was pushed 54 mm, and a tube under the bracket
49 mm (`tool_check.py` keeps that check, with the expected answer now 0 mm). On
the live cut the opening blades rolled the remainder segment 6 cm off its
cut-side pose, and a later release popped a held segment out of the container.
Geometry that carries no physics schema cannot do either.

BUILT BEFORE PLAY, under the hand prim, so the renderer moves it with the hand
and PhysX never sees it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import XformPrim
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from .config import LOG_ROBOT
from .grasp import _quat_multiply, _quat_rotate

#: The tool's prim name under the hand link (its full path is hand + this).
TOOL_PRIM_NAME = "wisepack_gripper_cutter"
TOOL_ROOT = TOOL_PRIM_NAME


@dataclass(frozen=True)
class ToolSpec:
    """The declared geometry of the combined tool, in the HAND frame (metres).

    `cut_offset_m` is the vector from the grasp frame to the cut frame. Its
    first component is along the hand's X axis — the direction a gripped tube
    lies when the fingers (which close along Y) hold it across — so the cut
    plane sits that far from the fingertips along the tube.
    """

    cut_offset_m: Tuple[float, float, float] = (0.06, 0.0, 0.0)
    #: Fingertip depth of the grasp frame along the hand's approach axis (Z).
    tcp_m: float = 0.103
    #: Blade travel: fully open half-gap, and the closed half-gap (blades meet).
    blade_open_half_gap_m: float = 0.035
    blade_closed_half_gap_m: float = 0.0
    #: Blade geometry (x thickness along the tube, y width, z height). THINNER
    #: THAN THE KERF (3 mm): closed blades sit inside the gap the cut leaves.
    blade_size_m: Tuple[float, float, float] = (0.0025, 0.014, 0.05)
    bracket_size_m: Tuple[float, float, float] = (0.024, 0.09, 0.03)
    #: Blade animation speed, metres of half-gap per simulation frame
    #: (35 mm of travel in ~24 frames at 60 Hz: visibly a stroke, not a jump).
    blade_speed_m_per_frame: float = 0.0015

    def to_dict(self) -> Dict[str, Any]:
        return {"cut_offset_m": list(self.cut_offset_m), "tcp_m": self.tcp_m,
                "blade_open_half_gap_m": self.blade_open_half_gap_m,
                "blade_closed_half_gap_m": self.blade_closed_half_gap_m,
                "blade_size_m": list(self.blade_size_m),
                "bracket_size_m": list(self.bracket_size_m),
                "blade_speed_m_per_frame": self.blade_speed_m_per_frame,
                "physics": "none (visual, kinematically animated)"}

    @staticmethod
    def from_dict(doc: Optional[Dict[str, Any]], tcp_m: float) -> "ToolSpec":
        doc = dict(doc or {})
        offset = doc.get("cut_offset_m") or (0.06, 0.0, 0.0)
        return ToolSpec(
            cut_offset_m=tuple(float(v) for v in offset),
            tcp_m=float(tcp_m),
            blade_open_half_gap_m=float(doc.get("blade_open_half_gap_m", 0.035)),
            blade_closed_half_gap_m=float(doc.get("blade_closed_half_gap_m", 0.0)),
            blade_speed_m_per_frame=float(doc.get("blade_speed_m_per_frame", 0.0015)))


class GripperCutterTool:
    """Authors the shear under the hand and animates its blades."""

    def __init__(self, spec: ToolSpec) -> None:
        self.spec = spec
        self.hand_path = ""
        self.root = ""
        self.bracket_path = ""
        self.blade_paths: Tuple[str, str] = ("", "")
        self._blade_translate_attrs: list = []
        self.built = False
        self.closed_commanded = False
        # Animation state: the CURRENT half-gap and where it is heading.
        self._half_gap_m = float(spec.blade_open_half_gap_m)
        self._target_half_gap_m = float(spec.blade_open_half_gap_m)

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #

    @property
    def cut_frame_in_hand_m(self) -> np.ndarray:
        """The cut frame origin in the hand frame: the TCP plus the declared offset."""
        return np.array([self.spec.cut_offset_m[0], self.spec.cut_offset_m[1],
                         self.spec.tcp_m + self.spec.cut_offset_m[2]], dtype=float)

    @property
    def grasp_to_cut_m(self) -> np.ndarray:
        """The FIXED transform grasp_frame -> cut_frame (translation, hand frame)."""
        return np.array(self.spec.cut_offset_m, dtype=float)

    def bracket_in_hand_m(self) -> np.ndarray:
        cut = self.cut_frame_in_hand_m
        # The bracket sits above the blades, towards the hand.
        return np.array([cut[0], 0.0, cut[2] - self.spec.blade_size_m[2] / 2.0
                         - self.spec.bracket_size_m[2] / 2.0], dtype=float)

    def blade_rest_in_hand_m(self, sign: float) -> np.ndarray:
        return self._blade_in_hand_m(sign, self.spec.blade_open_half_gap_m)

    def _blade_in_hand_m(self, sign: float, half_gap_m: float) -> np.ndarray:
        cut = self.cut_frame_in_hand_m
        return np.array([cut[0], sign * float(half_gap_m), cut[2]], dtype=float)

    # ------------------------------------------------------------------ #
    # Authoring (before play)
    # ------------------------------------------------------------------ #

    def build(self, hand_path: str, hand_position: Optional[Sequence[float]] = None,
              hand_orientation: Optional[Sequence[float]] = None) -> None:
        """Author the bracket and the two blades as children of `hand_path`.

        Every part is expressed in the HAND'S LOCAL FRAME, so it rides with the
        hand through USD composition alone. `hand_position`/`hand_orientation`
        are accepted for call compatibility and not needed: a child prim needs
        no world pose. NO PHYSICS SCHEMA IS APPLIED TO ANY PART — that is the
        whole point (see the module docstring), and `_assert_physics_free`
        checks it right after authoring.
        """
        stage = stage_utils.get_current_stage(backend="usd")
        hand = stage.GetPrimAtPath(hand_path)
        if not hand or not hand.IsValid():
            raise ValueError(f"cannot mount the gripper+cutter tool: {hand_path} is not on the stage")
        self.hand_path = hand_path
        self.root = f"{hand_path}/{TOOL_PRIM_NAME}"
        self.bracket_path = f"{self.root}/bracket"
        self.blade_paths = (f"{self.root}/blade_pos", f"{self.root}/blade_neg")
        if stage.GetPrimAtPath(self.root):
            stage.RemovePrim(self.root)
        UsdGeom.Xform.Define(stage, self.root)

        def cube(path: str, local: np.ndarray, size: Sequence[float],
                 colour: Sequence[float]) -> UsdGeom.Cube:
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.0)
            geom.CreateDisplayColorAttr([Gf.Vec3f(*(float(c) for c in colour))])
            xf = UsdGeom.Xformable(geom.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in local)))
            xf.AddScaleOp().Set(Gf.Vec3f(*(float(v) for v in size)))
            return geom

        # Bracket: a dark block under the hand, beside the fingers.
        cube(self.bracket_path, self.bracket_in_hand_m(), self.spec.bracket_size_m,
             (0.18, 0.2, 0.24))
        self._blade_translate_attrs = []
        for path, sign in zip(self.blade_paths, (1.0, -1.0)):
            geom = cube(path, self._blade_in_hand_m(sign, self._half_gap_m),
                        self.spec.blade_size_m, (0.82, 0.84, 0.88))
            self._blade_translate_attrs.append(
                geom.GetPrim().GetAttribute("xformOp:translate"))
        self._assert_physics_free(stage)
        self.built = True
        print(f"{LOG_ROBOT} gripper+cutter tool built under {hand_path.rsplit('/', 1)[-1]} "
              f"as visual geometry (no rigid body, no collider): cut frame "
              f"{tuple(round(float(v), 3) for v in self.cut_frame_in_hand_m)} m in the "
              f"hand frame, grasp->cut offset "
              f"{tuple(round(float(v), 3) for v in self.grasp_to_cut_m)} m")

    def _assert_physics_free(self, stage) -> None:
        """No tool prim may carry a physics schema. Checked, not assumed."""
        for prim in Usd.PrimRange(stage.GetPrimAtPath(self.root)):
            if (prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    or prim.HasAPI(UsdPhysics.MassAPI)
                    or prim.IsA(UsdPhysics.Joint)):
                raise RuntimeError(
                    f"tool prim {prim.GetPath()} carries a physics schema "
                    f"{list(prim.GetAppliedSchemas())}; the cutter must be visual-only")

    def physics_free(self) -> bool:
        """True when no prim of the tool carries any physics schema (measured)."""
        try:
            self._assert_physics_free(stage_utils.get_current_stage(backend="usd"))
            return self.built
        except Exception:                                   # noqa: BLE001
            return False

    # ------------------------------------------------------------------ #
    # Animation (any time): local translations only
    # ------------------------------------------------------------------ #

    def open_cutter(self) -> None:
        self._target_half_gap_m = float(self.spec.blade_open_half_gap_m)
        self.closed_commanded = False

    def close_cutter(self) -> None:
        self._target_half_gap_m = float(self.spec.blade_closed_half_gap_m)
        self.closed_commanded = True

    def tick(self) -> None:
        """Advance the blade animation by one frame. Call once per sim frame."""
        if not self.built:
            return
        delta = self._target_half_gap_m - self._half_gap_m
        if abs(delta) < 1e-9:
            return
        speed = float(self.spec.blade_speed_m_per_frame)
        self._half_gap_m += float(np.clip(delta, -speed, speed))
        for attr, sign in zip(self._blade_translate_attrs, (1.0, -1.0)):
            local = self._blade_in_hand_m(sign, self._half_gap_m)
            attr.Set(Gf.Vec3d(*(float(v) for v in local)))

    def blade_gap_m(self) -> Optional[float]:
        """The ANIMATED distance between the blades (their authored translation)."""
        return None if not self.built else 2.0 * self._half_gap_m

    def cutter_closed(self, tolerance_m: float = 0.001) -> bool:
        gap = self.blade_gap_m()
        return gap is not None and gap <= 2.0 * self.spec.blade_closed_half_gap_m + tolerance_m

    def cut_frame_world(self, hand_position: Sequence[float],
                        hand_orientation: Sequence[float]) -> np.ndarray:
        """The cut frame origin in world for a given hand pose."""
        return (np.asarray(hand_position, dtype=float)
                + _quat_rotate(np.asarray(hand_orientation, dtype=float),
                               self.cut_frame_in_hand_m))

    def part_world_positions(self) -> Dict[str, np.ndarray]:
        """MEASURED world positions of the bracket and blades (USD composition)."""
        out: Dict[str, np.ndarray] = {}
        if not self.built:
            return out
        for name, path in (("bracket", self.bracket_path), ("blade_pos", self.blade_paths[0]),
                           ("blade_neg", self.blade_paths[1])):
            position, _ = XformPrim(path).get_world_poses()
            out[name] = np.asarray(position.numpy()[0], dtype=float)
        return out

    def diagnostics(self) -> Dict[str, Any]:
        gap = self.blade_gap_m()
        return {"tool": "gripper_cutter", "built": self.built,
                "physics": "none: visual geometry, kinematically animated blades",
                "cut_offset_m": list(self.spec.cut_offset_m),
                "blade_gap_m": None if gap is None else round(gap, 4),
                "cutter_closed_commanded": self.closed_commanded,
                "cutter_closed_animated": self.cutter_closed()}


__all__ = ["GripperCutterTool", "ToolSpec", "TOOL_ROOT", "TOOL_PRIM_NAME",
           "_quat_multiply", "XformPrim"]
