"""THE SUPPORTED ISAAC ROBOTS — one tracked registry, one typed model.

WHY THIS EXISTS AT ALL
----------------------
Before this module the Franka Emika Panda was not a *choice*, it was an
assumption: an asset URL in the scene builder, a prim name in the grasp welder,
a joint count in the reset validator, a home vector written out twice, and the
word "Panda" in the README, the launcher and four tests. Adding a second arm by
copying all of that would have produced two implementations that drift, and a
dashboard whose robot list is a hand-written ``<option>`` that agrees with the
Python only until someone edits one of them.

So the robot is DATA. ``config/isaac_robots.yaml`` is the single tracked source,
this module is the only parser, and everything that needs to know which robot is
in play — the launcher, the simulator, the orchestrator, the web API, the test
suite — reads it from here. There is deliberately no second robot list anywhere,
and ``tests/test_isaac_robots.py`` asserts that.

WHAT A PROFILE IS AND IS NOT
----------------------------
A profile describes what the SIMULATED robot is and where its parts are. It is
not a driver, and loading one moves nothing: the adapter in
``simulators/isaac/adapters/`` does that, and it VALIDATES every claim in the
profile against the articulation it actually loaded before a single joint is
commanded. A profile that disagrees with the asset is a startup failure with a
named reason, never a silent best-effort.

Nothing here imports Isaac, USD or numpy, so the registry is fully testable on a
machine with no GPU and no simulator — which is the same rule the rest of
``wisepack_core`` follows.

PUBLIC VERSUS INTERNAL
----------------------
``RobotProfile.to_public_dict()`` is what the web API is allowed to publish:
identity, capability and status. Asset URLs, prim paths and joint names stay
inside the process — they are not secrets in the cryptographic sense, but an
asset-server path is free reconnaissance and a prim path is of no use to an
operator deciding which arm to run.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Bumped when the SHAPE of a profile changes in a way a consumer must notice.
#: Carried into SCENE_READY beside the per-profile revision so a simulator built
#: against an older schema cannot silently authorise a run.
REGISTRY_SCHEMA = "wisepack-isaac-robots/1.0"

#: The environment override, for automation. Documented in the README and in
#: config/isaac_robots.yaml; deliberately NOT read from config/local.env, which
#: is host-specific and untracked — the robot is a public scenario choice.
ROBOT_ENV_VAR = "WISEPACK_ISAAC_ROBOT"

#: Where the registry lives, relative to the repository root.
DEFAULT_REGISTRY_RELPATH = os.path.join("config", "isaac_robots.yaml")

#: The placement skills this iteration implements. A profile that claims one
#: outside this set is rejected at load: an unimplemented skill in a config file
#: reads as a capability and is a promise nothing keeps.
KNOWN_SKILLS = ("HOME", "MOVE_TO_PICK", "GRASP", "LIFT", "MOVE_TO_CONTAINER",
                "RELEASE", "SETTLE", "VERIFY")

#: Implementation maturity, in the operator's words. `validated` means a full
#: smoke run has been measured with this profile; `experimental` means it loads
#: and moves but has not; `planned` means it is described and not implemented.
KNOWN_STATUSES = ("validated", "experimental", "planned")

#: Kinematics implementations the adapters actually provide. Naming them here
#: rather than accepting free text is what stops a profile claiming
#: "collision-aware motion planning" that nothing implements.
KNOWN_KINEMATICS = (
    #: Damped least squares over the articulation Jacobian, one step per physics
    #: frame. NOT a motion planner: no collision awareness, no trajectory.
    "differential-ik-dls",
)


class RobotConfigError(ValueError):
    """The robot registry is missing, malformed or self-inconsistent.

    Always carries the file and the offending key. A robot registry that cannot
    be parsed must stop the process before anything loads an asset — the
    alternative is an arm moving under a configuration nobody validated.
    """


# --------------------------------------------------------------------------- #
# Workcell overrides
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WorkcellOverrides:
    """Per-robot changes to the shared scene layout. Metres.

    THE POINT OF THIS BLOCK. The default layout was sized for a Panda's ~0.855 m
    reach, and dropping a shorter arm into those coordinates does not fail
    loudly — differential IK converges to the nearest achievable pose and the
    gripper opens over nothing. So a robot whose envelope differs states the
    difference here, and ``isaac_transform.layout_for_robot`` builds the layout
    from it. Every value that is ``None`` keeps the shared default.

    A robot with no overrides is not a special case: it simply declares none.
    """

    robot_max_reach_m: Optional[float] = None
    robot_min_reach_m: Optional[float] = None
    container_outer_xy_m: Optional[Tuple[float, float]] = None
    pick_row_x_m: Optional[float] = None
    pick_row_y_start_m: Optional[float] = None
    pick_row_y_pitch_m: Optional[float] = None
    table_top_z_m: Optional[float] = None
    #: Where the spectator camera sits and how it is aimed, so the DemoCamera
    #: frames THIS robot and its workcell rather than the one it was placed for.
    camera_position_m: Optional[Tuple[float, float, float]] = None
    camera_rotation_deg: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items() if v is not None}

    @staticmethod
    def from_dict(doc: Any, *, where: str) -> "WorkcellOverrides":
        if doc is None:
            return WorkcellOverrides()
        if not isinstance(doc, dict):
            raise RobotConfigError(f"{where}: workcell must be a mapping")
        known = set(WorkcellOverrides.__dataclass_fields__)
        unknown = set(doc) - known
        if unknown:
            raise RobotConfigError(
                f"{where}: unknown workcell keys {sorted(unknown)}; "
                f"known: {sorted(known)}")

        def _f(key):
            value = doc.get(key)
            return None if value is None else float(value)

        def _t(key, n):
            value = doc.get(key)
            if value is None:
                return None
            if not isinstance(value, (list, tuple)) or len(value) != n:
                raise RobotConfigError(
                    f"{where}: workcell.{key} must be a list of {n} numbers")
            return tuple(float(v) for v in value)

        return WorkcellOverrides(
            robot_max_reach_m=_f("robot_max_reach_m"),
            robot_min_reach_m=_f("robot_min_reach_m"),
            container_outer_xy_m=_t("container_outer_xy_m", 2),
            pick_row_x_m=_f("pick_row_x_m"),
            pick_row_y_start_m=_f("pick_row_y_start_m"),
            pick_row_y_pitch_m=_f("pick_row_y_pitch_m"),
            table_top_z_m=_f("table_top_z_m"),
            camera_position_m=_t("camera_position_m", 3),
            camera_rotation_deg=_t("camera_rotation_deg", 3))


# --------------------------------------------------------------------------- #
# The profile
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RobotProfile:
    """One supported Isaac robot, fully described and validated at load."""

    robot_id: str
    display_name: str
    manufacturer: str
    model: str
    enabled: bool
    implementation_status: str
    #: Which adapter class implements it. Not free text — the adapter factory
    #: maps this to a concrete class and refuses an unknown value, so a config
    #: file cannot name an implementation that does not exist.
    adapter: str

    #: Paths under the Isaac asset root, tried IN ORDER. A list rather than one
    #: string because asset layouts have moved between Isaac releases, and the
    #: loader reporting "none of these three exist" is far more useful than
    #: "cannot open <one path>".
    asset_path_candidates: List[str]
    #: USD variant selections applied when the asset is referenced in.
    asset_variants: Dict[str, str]

    #: Where the robot is referenced into the stage, and where the base
    #: transform is written. The ARTICULATION ROOT is separate and is often not
    #: the same prim: the xArm 7 asset carries PhysicsArticulationRootAPI on
    #: <root>/root_joint, so writing a world pose to the articulation asserts on
    #: a missing xformOp. Both are recorded and both are verified.
    root_prim_path: str
    articulation_root: str

    arm_joint_names: List[str]
    #: The DRIVEN gripper joints. The xArm gripper has one (`drive_joint`); its
    #: other five follow through PhysxMimicJointAPI and must NOT be commanded.
    gripper_joint_names: List[str]
    #: Joints that follow the driven one in physics. Listed so validation can
    #: assert they exist and diagnostics can report the whole gripper.
    gripper_mimic_joint_names: List[str]

    end_effector_prim: str
    end_effector_link: str
    #: Distance from the end-effector LINK origin to the point between the
    #: fingertips, along the tool approach axis, in metres. Every Cartesian goal
    #: in the sequence is a fingertip goal plus this offset. Measured from the
    #: asset, never taken from a datasheet: for the xArm gripper the finger link
    #: origins sit at the knuckles, ~70 mm short of the tips, and using them put
    #: every grasp descent 70 mm high.
    tool_centre_point_m: float

    home_joint_positions: List[float]
    open_gripper_positions: List[float]
    closed_gripper_positions: List[float]
    #: Yaw applied to the top-down orientation so the fingers close ACROSS the
    #: cylinder. A property of the shipped asset's tool frame, not of the code.
    grasp_yaw_offset_deg: float
    #: Radians. How far a joint may sit from `home_joint_positions` and still
    #: count as home. Read by the simulator's MEASURED home check.
    home_tolerance_rad: float

    kinematics: str
    kinematics_options: Dict[str, Any]

    nominal_reach_m: float
    dof: int
    supported_skills: List[str]
    supported_presets: List[str]
    workcell: WorkcellOverrides
    notes: str

    # -- derived ---------------------------------------------------------- #

    @property
    def arm_dof(self) -> int:
        return len(self.arm_joint_names)

    @property
    def gripper_dof(self) -> int:
        return len(self.gripper_joint_names) + len(self.gripper_mimic_joint_names)

    @property
    def is_motion_planner(self) -> bool:
        """Never true today, and stated rather than left to inference.

        A differential IK controller is not a motion planner. Nothing in this
        iteration does collision-aware trajectory planning, and the dashboard
        must not imply otherwise.
        """
        return False

    def supports_preset(self, preset: str) -> bool:
        #: An empty list means "no restriction beyond the backend's own physical
        #: compatibility check" — see execution.preset_physical_compatibility.
        return not self.supported_presets or preset in self.supported_presets

    def preset_refusal(self, preset: str) -> str:
        """"" when this robot may run ``preset``, else the operator's reason."""
        if self.supports_preset(preset):
            return ""
        return (f"{self.display_name} is configured for "
                f"{', '.join(self.supported_presets)}; {preset} is not among them")

    # -- identity --------------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """The COMPLETE profile. Internal — see to_public_dict for the API."""
        return {
            "id": self.robot_id,
            "display_name": self.display_name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "enabled": self.enabled,
            "implementation_status": self.implementation_status,
            "adapter": self.adapter,
            "asset_path_candidates": list(self.asset_path_candidates),
            "asset_variants": dict(self.asset_variants),
            "root_prim_path": self.root_prim_path,
            "articulation_root": self.articulation_root,
            "arm_joint_names": list(self.arm_joint_names),
            "gripper_joint_names": list(self.gripper_joint_names),
            "gripper_mimic_joint_names": list(self.gripper_mimic_joint_names),
            "end_effector_prim": self.end_effector_prim,
            "end_effector_link": self.end_effector_link,
            "tool_centre_point_m": self.tool_centre_point_m,
            "home_joint_positions": list(self.home_joint_positions),
            "open_gripper_positions": list(self.open_gripper_positions),
            "closed_gripper_positions": list(self.closed_gripper_positions),
            "grasp_yaw_offset_deg": self.grasp_yaw_offset_deg,
            "home_tolerance_rad": self.home_tolerance_rad,
            "kinematics": self.kinematics,
            "kinematics_options": dict(self.kinematics_options),
            "nominal_reach_m": self.nominal_reach_m,
            "dof": self.dof,
            "supported_skills": list(self.supported_skills),
            "supported_presets": list(self.supported_presets),
            "workcell": self.workcell.to_dict(),
            "notes": self.notes,
        }

    def to_public_dict(self) -> Dict[str, Any]:
        """What the web API may publish. No asset URLs, no prim paths.

        Everything an operator needs to choose a robot and to understand what
        they are choosing; nothing that describes this host's filesystem or the
        asset server it fetches from.
        """
        return {
            "id": self.robot_id,
            "display_name": self.display_name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "dof": self.dof,
            "arm_dof": self.arm_dof,
            "gripper_dof": self.gripper_dof,
            "enabled": self.enabled,
            "status": self.implementation_status,
            "kinematics": self.kinematics,
            "motion_planning": self.is_motion_planner,
            "nominal_reach_m": self.nominal_reach_m,
            "compatible_presets": list(self.supported_presets),
            "skills": list(self.supported_skills),
            "profile_revision": self.revision,
            "notes": self.notes,
        }

    @property
    def revision(self) -> str:
        """A deterministic digest of this profile.

        Carried in SCENE_READY beside ``robot_id`` so the orchestrator can tell
        "the right robot" from "the right robot, configured differently". An
        edited home pose or a changed TCP offset is a different physical
        machine as far as a plan is concerned, and the id alone cannot see it.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # -- construction ----------------------------------------------------- #

    @staticmethod
    def from_dict(doc: Any, *, where: str) -> "RobotProfile":
        if not isinstance(doc, dict):
            raise RobotConfigError(f"{where}: each robot must be a mapping")

        def _req(key, kind=str):
            if key not in doc or doc[key] is None:
                raise RobotConfigError(f"{where}: missing required key {key!r}")
            try:
                return kind(doc[key])
            except (TypeError, ValueError) as exc:
                raise RobotConfigError(
                    f"{where}: {key}={doc[key]!r} is not a {kind.__name__}") from exc

        def _list(key, kind=str, *, required=True):
            value = doc.get(key)
            if value is None:
                if required:
                    raise RobotConfigError(f"{where}: missing required key {key!r}")
                return []
            if not isinstance(value, (list, tuple)):
                raise RobotConfigError(f"{where}: {key} must be a list")
            try:
                return [kind(v) for v in value]
            except (TypeError, ValueError) as exc:
                raise RobotConfigError(
                    f"{where}: {key} must contain only {kind.__name__}") from exc

        known = {
            "id", "display_name", "manufacturer", "model", "enabled",
            "implementation_status", "adapter", "asset_path_candidates",
            "asset_variants", "root_prim_path", "articulation_root",
            "arm_joint_names", "gripper_joint_names", "gripper_mimic_joint_names",
            "end_effector_prim", "end_effector_link", "tool_centre_point_m",
            "home_joint_positions", "open_gripper_positions",
            "closed_gripper_positions", "grasp_yaw_offset_deg",
            "home_tolerance_rad", "kinematics", "kinematics_options",
            "nominal_reach_m", "supported_skills", "supported_presets",
            "workcell", "notes",
        }
        unknown = set(doc) - known
        if unknown:
            # Rejected, not ignored. A typo in a robot description silently
            # ignored is a robot running with a default nobody chose.
            raise RobotConfigError(
                f"{where}: unknown keys {sorted(unknown)}; known: {sorted(known)}")

        robot_id = _req("id")
        status = _req("implementation_status")
        if status not in KNOWN_STATUSES:
            raise RobotConfigError(
                f"{where}: implementation_status {status!r} is not one of "
                f"{list(KNOWN_STATUSES)}")
        kinematics = _req("kinematics")
        if kinematics not in KNOWN_KINEMATICS:
            raise RobotConfigError(
                f"{where}: kinematics {kinematics!r} is not implemented; "
                f"available: {list(KNOWN_KINEMATICS)}. Naming a controller no "
                "adapter provides would advertise a capability nothing keeps.")
        skills = _list("supported_skills")
        bad = [s for s in skills if s not in KNOWN_SKILLS]
        if bad:
            raise RobotConfigError(
                f"{where}: supported_skills {bad} are not implemented; "
                f"available: {list(KNOWN_SKILLS)}")

        arm = _list("arm_joint_names")
        grip = _list("gripper_joint_names")
        mimic = _list("gripper_mimic_joint_names", required=False)
        overlap = sorted(set(arm) & set(grip + mimic))
        if overlap:
            raise RobotConfigError(
                f"{where}: {overlap} appear as both arm and gripper joints")
        if len(set(arm)) != len(arm):
            raise RobotConfigError(f"{where}: arm_joint_names has duplicates")

        home = _list("home_joint_positions", float)
        if len(home) != len(arm):
            raise RobotConfigError(
                f"{where}: home_joint_positions has {len(home)} value(s) but "
                f"arm_joint_names has {len(arm)} joint(s)")
        openg = _list("open_gripper_positions", float)
        closeg = _list("closed_gripper_positions", float)
        for name, values in (("open_gripper_positions", openg),
                             ("closed_gripper_positions", closeg)):
            if len(values) != len(grip):
                raise RobotConfigError(
                    f"{where}: {name} has {len(values)} value(s) but "
                    f"gripper_joint_names has {len(grip)}")
        if openg == closeg:
            raise RobotConfigError(
                f"{where}: open_gripper_positions and closed_gripper_positions "
                "are identical — the gripper would never visibly move")

        tcp = float(_req("tool_centre_point_m", float))
        if tcp <= 0.0:
            raise RobotConfigError(
                f"{where}: tool_centre_point_m must be > 0; it is the distance "
                "from the end-effector link origin to the fingertips")
        reach = float(_req("nominal_reach_m", float))
        if reach <= 0.0:
            raise RobotConfigError(f"{where}: nominal_reach_m must be > 0")

        candidates = _list("asset_path_candidates")
        if not candidates:
            raise RobotConfigError(
                f"{where}: asset_path_candidates must name at least one path")

        variants = doc.get("asset_variants") or {}
        if not isinstance(variants, dict):
            raise RobotConfigError(f"{where}: asset_variants must be a mapping")
        options = doc.get("kinematics_options") or {}
        if not isinstance(options, dict):
            raise RobotConfigError(f"{where}: kinematics_options must be a mapping")

        return RobotProfile(
            robot_id=robot_id,
            display_name=_req("display_name"),
            manufacturer=_req("manufacturer"),
            model=_req("model"),
            enabled=bool(doc.get("enabled", True)),
            implementation_status=status,
            adapter=_req("adapter"),
            asset_path_candidates=candidates,
            asset_variants={str(k): str(v) for k, v in variants.items()},
            root_prim_path=_req("root_prim_path"),
            articulation_root=_req("articulation_root"),
            arm_joint_names=arm,
            gripper_joint_names=grip,
            gripper_mimic_joint_names=mimic,
            end_effector_prim=_req("end_effector_prim"),
            end_effector_link=_req("end_effector_link"),
            tool_centre_point_m=tcp,
            home_joint_positions=home,
            open_gripper_positions=openg,
            closed_gripper_positions=closeg,
            grasp_yaw_offset_deg=float(doc.get("grasp_yaw_offset_deg", 0.0)),
            home_tolerance_rad=float(doc.get("home_tolerance_rad", 0.35)),
            kinematics=kinematics,
            kinematics_options=dict(options),
            nominal_reach_m=reach,
            dof=len(arm) + len(grip) + len(mimic),
            supported_skills=skills,
            supported_presets=_list("supported_presets", required=False),
            workcell=WorkcellOverrides.from_dict(
                doc.get("workcell"), where=where),
            notes=str(doc.get("notes", "")),
        )


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RobotRegistry:
    """Every supported Isaac robot, plus which one is the default."""

    profiles: Dict[str, RobotProfile]
    default_robot_id: str
    source_path: str
    schema: str = REGISTRY_SCHEMA

    def __iter__(self):
        return iter(self.ordered)

    @property
    def ordered(self) -> List[RobotProfile]:
        """Registry order, which is the order the selector shows.

        Insertion order, not alphabetical: the file's author put the preferred
        robot first and a dropdown that silently re-sorts loses that intent.
        """
        return list(self.profiles.values())

    @property
    def enabled_ids(self) -> List[str]:
        return [p.robot_id for p in self.ordered if p.enabled]

    @property
    def revision(self) -> str:
        """Digest over every profile and the default — the REGISTRY revision."""
        blob = json.dumps(
            {"schema": self.schema, "default": self.default_robot_id,
             "robots": [p.to_dict() for p in self.ordered]},
            sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def get(self, robot_id: str) -> RobotProfile:
        """The profile for ``robot_id``, or a refusal naming what IS available.

        Raises rather than falling back to the default. A typo that quietly
        selected another arm would produce a run whose artefacts name a robot
        that never moved.
        """
        key = str(robot_id or "").strip().lower()
        if key not in self.profiles:
            raise RobotConfigError(
                f"unknown robot {robot_id!r}; configured robots: "
                f"{', '.join(sorted(self.profiles))} (see {self.source_path})")
        profile = self.profiles[key]
        if not profile.enabled:
            raise RobotConfigError(
                f"robot {profile.robot_id!r} ({profile.display_name}) is "
                f"disabled in {self.source_path}; enabled robots: "
                f"{', '.join(self.enabled_ids) or 'none'}")
        return profile

    def default(self) -> RobotProfile:
        return self.get(self.default_robot_id)

    def to_public_dict(self) -> Dict[str, Any]:
        """The payload behind GET /api/config/robots."""
        return {
            "schema": self.schema,
            "registry_revision": self.revision,
            "default_robot": self.default_robot_id,
            "robots": [p.to_public_dict() for p in self.ordered],
        }

    def resolve(self, *, explicit: Optional[str] = None,
                draft: Optional[str] = None,
                env: Optional[Dict[str, str]] = None) -> RobotProfile:
        """WHICH ROBOT, by the documented precedence.

            1. an explicit command-line value
            2. the exported environment override (WISEPACK_ISAAC_ROBOT)
            3. the scenario draft selection
            4. the configured default

        Environment sits above the draft on purpose: the override exists for
        automation, and a validator that exports it must not be overruled by
        whatever a browser last left in the draft.

        Every candidate is validated, and an unknown or disabled one raises
        instead of falling through to the next. Falling through would turn a
        typo into a run on a robot nobody selected.
        """
        environ = os.environ if env is None else env
        for value in (explicit, environ.get(ROBOT_ENV_VAR), draft):
            if value is not None and str(value).strip():
                return self.get(value)
        return self.default()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _repo_root() -> str:
    """The repository root, derived from THIS file rather than from the cwd.

    The Isaac launcher deliberately runs the simulator from a scratch directory
    (NVIDIA's streaming stack writes trace files into the process working
    directory), so a relative "config/..." lookup resolves to nothing there.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def registry_path(explicit: Optional[str] = None) -> str:
    """Where the registry is, in the order the answer is looked for."""
    if explicit:
        return explicit
    override = os.environ.get("WISEPACK_ROBOTS_CONFIG")
    if override:
        return override
    config_dir = os.environ.get("WISEPACK_CONFIG_DIR")
    if config_dir:
        return os.path.join(config_dir, "isaac_robots.yaml")
    local = os.path.join(os.getcwd(), DEFAULT_REGISTRY_RELPATH)
    if os.path.exists(local):
        return local
    return os.path.join(_repo_root(), DEFAULT_REGISTRY_RELPATH)


def _load_document(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise RobotConfigError(
            f"the Isaac robot registry is missing: {path}. It is a TRACKED "
            "file — every supported robot is defined there and nothing else "
            "carries a robot list. Restore it from the repository.")
    try:
        import yaml                                        # noqa: PLC0415
    except ImportError as exc:                             # pragma: no cover
        raise RobotConfigError(
            f"cannot read {path}: pyyaml is not installed. The Isaac backend "
            "cannot select a robot without it.") from exc
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise RobotConfigError(f"{path}: the top level must be a mapping")
    return doc


def parse_registry(doc: Dict[str, Any], *, source_path: str = "<memory>"
                   ) -> RobotRegistry:
    """Validate a registry document into typed profiles.

    Separate from file loading so the whole validation surface is testable from
    a dict, which is what ``tests/test_isaac_robots.py`` does.
    """
    known = {"schema", "default_robot", "robots"}
    unknown = set(doc) - known
    if unknown:
        raise RobotConfigError(
            f"{source_path}: unknown top-level keys {sorted(unknown)}; "
            f"known: {sorted(known)}")

    schema = str(doc.get("schema", REGISTRY_SCHEMA))
    if schema.split("/", 1)[0] != REGISTRY_SCHEMA.split("/", 1)[0]:
        raise RobotConfigError(
            f"{source_path}: schema {schema!r} is not a "
            f"{REGISTRY_SCHEMA.split('/', 1)[0]} document")

    raw = doc.get("robots")
    if not isinstance(raw, list) or not raw:
        raise RobotConfigError(
            f"{source_path}: 'robots' must be a non-empty list")

    profiles: Dict[str, RobotProfile] = {}
    for n, entry in enumerate(raw, 1):
        profile = RobotProfile.from_dict(entry, where=f"{source_path}: robots[{n}]")
        key = profile.robot_id.strip().lower()
        if key != profile.robot_id:
            raise RobotConfigError(
                f"{source_path}: robot id {profile.robot_id!r} must be lower "
                "case with no surrounding whitespace — it travels in run "
                "artefacts, NGSI-LD payloads and URLs")
        if key in profiles:
            # Rejected, not last-wins. Two profiles under one id means one of
            # them is silently unreachable, and which one depends on file order.
            raise RobotConfigError(
                f"{source_path}: duplicate robot id {key!r} at robots[{n}]")
        profiles[key] = profile

    default = doc.get("default_robot")
    if not default:
        raise RobotConfigError(
            f"{source_path}: 'default_robot' is required — exactly one robot "
            "must be the default, so an unconfigured run cannot pick one by "
            "accident of ordering")
    default = str(default).strip().lower()
    if default not in profiles:
        raise RobotConfigError(
            f"{source_path}: default_robot {default!r} is not one of "
            f"{sorted(profiles)}")
    if not profiles[default].enabled:
        raise RobotConfigError(
            f"{source_path}: default_robot {default!r} is disabled; the "
            "default must be runnable")

    return RobotRegistry(profiles=profiles, default_robot_id=default,
                         source_path=source_path, schema=schema)


_CACHE: Dict[str, RobotRegistry] = {}


def load_registry(path: Optional[str] = None, *, reload: bool = False
                  ) -> RobotRegistry:
    """Load and validate the registry ONCE per path.

    Cached because the dashboard reads it on every ``/api/state`` poll and the
    file does not change under a running process. ``reload=True`` is for tests.
    """
    resolved = registry_path(path)
    if reload or resolved not in _CACHE:
        _CACHE[resolved] = parse_registry(_load_document(resolved),
                                          source_path=resolved)
    return _CACHE[resolved]


def resolve_robot(explicit: Optional[str] = None, draft: Optional[str] = None,
                  *, path: Optional[str] = None) -> RobotProfile:
    """The module-level convenience the launcher and simulator use."""
    return load_registry(path).resolve(explicit=explicit, draft=draft)


def robot_choices(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """The public catalogue, for the web API and the dashboard selector."""
    return load_registry(path).to_public_dict()["robots"]


__all__ = [
    "REGISTRY_SCHEMA", "ROBOT_ENV_VAR", "KNOWN_SKILLS", "KNOWN_STATUSES",
    "KNOWN_KINEMATICS", "RobotConfigError", "WorkcellOverrides", "RobotProfile",
    "RobotRegistry", "parse_registry", "load_registry", "registry_path",
    "resolve_robot", "robot_choices",
]
