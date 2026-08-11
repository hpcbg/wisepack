"""Simulated-appearance materials, read from configuration and bound to a prim.

ONE PLACE, like every other WISEPACK registry. The numbers live in
`config/isaac_materials.yaml`; this module loads them and builds the USD shading
network. A colour written inline at a call site would be a second declaration
that agrees with the first only until somebody edits one — and here that would
mean the model-free REFERENCE views and the model-free QUERY frame could be
rendered with different surfaces, which would invalidate the experiment they
exist for.

WHAT A MATERIAL DOES AND DOES NOT TOUCH
---------------------------------------
It changes how a surface responds to light. It does not change the mesh, the
dimensions, the pose, the declared symmetry, the mass, the ground-truth transform
or the instance mask — the mask comes from prim semantics, which is why changing
the material cannot and does not affect simulated segmentation.

IMPORTABLE WITHOUT ISAAC. The loader and the profile are plain Python and YAML,
so a test can assert the declared parameters on a machine with no simulator; only
`bind()` touches `pxr`, and it imports it lazily.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Where the profiles live, relative to the repository root.
MATERIALS_REGISTRY_PATH = os.path.join("config", "isaac_materials.yaml")


class MaterialError(ValueError):
    """A material profile that cannot be used, with the reason."""


@dataclass(frozen=True)
class SurfaceProfile:
    """One physically based surface, exactly as configured.

    FROZEN: a profile that could be mutated after loading would let one caller
    change what another renders, which is the drift this module exists to
    prevent.
    """

    name: str
    description: str = ""
    shader: str = "UsdPreviewSurface"
    base_color: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    metallic: float = 0.0
    roughness: float = 0.5
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.0
    specular: float = 0.5
    ior: float = 1.5
    emissive_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    opacity: float = 1.0

    @property
    def is_metal(self) -> bool:
        return self.metallic >= 0.5

    @property
    def is_mirror_like(self) -> bool:
        """Deliberately checkable: the simulated depth camera does NOT model a
        real D435's failure on specular metal, so a mirror finish would look
        like the hard case without being it."""
        return self.is_metal and self.roughness < 0.15

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "shader": self.shader, "base_color": list(self.base_color),
            "metallic": self.metallic, "roughness": self.roughness,
            "clearcoat": self.clearcoat,
            "clearcoat_roughness": self.clearcoat_roughness,
            "specular": self.specular, "ior": self.ior,
            "emissive_color": list(self.emissive_color),
            "opacity": self.opacity,
            "is_metal": self.is_metal, "is_mirror_like": self.is_mirror_like,
        }


@dataclass
class MaterialRegistry:
    profiles: Dict[str, SurfaceProfile] = field(default_factory=dict)
    default_profile: str = ""
    error: str = ""

    def require(self, name: str = "") -> SurfaceProfile:
        """The named profile, or the declared default. NEVER a silent fallback.

        A missing profile raises rather than rendering something arbitrary: the
        whole point is that the reference set and the query are the same
        surface, and quietly substituting one would break that without saying so.
        """
        if self.error:
            raise MaterialError(self.error)
        wanted = str(name or self.default_profile or "")
        if not wanted:
            raise MaterialError(
                "no material profile was requested and none is declared as the "
                f"default in {MATERIALS_REGISTRY_PATH}")
        profile = self.profiles.get(wanted)
        if profile is None:
            raise MaterialError(
                f"unknown material profile {wanted!r}; configured: "
                + (", ".join(sorted(self.profiles)) or "none"))
        return profile


def load_materials(path: str = "", repo_root: str = "") -> MaterialRegistry:
    """Load the registry. A BROKEN registry is reported, never silently empty."""
    root = repo_root or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    resolved = path or os.path.join(root, MATERIALS_REGISTRY_PATH)
    if not os.path.isfile(resolved):
        return MaterialRegistry(error=f"{resolved} does not exist")
    try:
        import yaml                                              # noqa: PLC0415
        with open(resolved, encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
    except Exception as exc:                                     # noqa: BLE001
        return MaterialRegistry(error=f"{resolved}: {exc}")

    profiles: Dict[str, SurfaceProfile] = {}
    for entry in (document.get("profiles") or []):
        name = str((entry or {}).get("name", "")).strip()
        if not name:
            return MaterialRegistry(error=f"{resolved}: a profile has no name")
        if name in profiles:
            return MaterialRegistry(
                error=f"{resolved}: duplicate material profile {name!r}")
        known = {f for f in SurfaceProfile.__dataclass_fields__}
        profiles[name] = SurfaceProfile(
            **{k: v for k, v in entry.items() if k in known})
    default = str(document.get("default_profile", "") or "")
    if default and default not in profiles:
        return MaterialRegistry(
            error=f"{resolved}: default_profile {default!r} is not declared")
    return MaterialRegistry(profiles=profiles, default_profile=default)


def bind(prim, profile: SurfaceProfile, stage=None, material_root: str = ""):
    """Build the USD shading network for `profile` and bind it to `prim`.

    ONE MATERIAL PER PROFILE, reused. Creating a fresh material per prim would
    be indistinguishable in a single-object scene and quietly wrong in a scene
    with several — and it would make "the reference and the query used the same
    surface" a coincidence rather than a fact.
    """
    from pxr import Gf, Sdf, UsdShade                             # noqa: PLC0415

    stage = stage or prim.GetStage()
    root = material_root or "/World/Looks"
    path = f"{root}/{profile.name}"
    material = UsdShade.Material.Get(stage, path)
    if not material:
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr(profile.shader)
        colour = Gf.Vec3f(*[float(v) for v in profile.base_color])
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(colour)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            float(profile.metallic))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(profile.roughness))
        shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(
            float(profile.clearcoat))
        shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(
            float(profile.clearcoat_roughness))
        shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(
            float(profile.specular))
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(profile.ior))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*[float(v) for v in profile.emissive_color]))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
            float(profile.opacity))
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(material)
    return material


__all__ = ["MATERIALS_REGISTRY_PATH", "MaterialError", "SurfaceProfile",
           "MaterialRegistry", "load_materials", "bind"]
