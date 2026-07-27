"""Execution backends: who actually moves the item.

WISEPACK separates two things that are easy to conflate and must not be:

    DATA SOURCE        where the DASHBOARD reads state from.
                       sim | ros | fiware.  (see web/snapshot.py)

    EXECUTION BACKEND  what performs the pick-and-place the operator approved.
                       simulated | isaac.  (this module)

They are orthogonal. A run can be executed by Isaac Sim and observed through
FIWARE; it can be executed by the simulated robot and observed over ROS. "Isaac"
is not another way of reading the state, and the dashboard must never present it
as one.

WHAT THE BACKEND CHANGES, AND WHAT IT DOES NOT
----------------------------------------------
Unchanged in both backends: scenario generation, both packing algorithms, Digital
Twin validation, the approval gate, re-planning, the audit trail, the KPI
definitions, and every canonical topic. The optimizer does not know a robot
exists.

Changed: what happens between "the operator approved this placement" and "this
placement is executed". SIMULATED resolves it with a seeded coin flip inside
``WorkflowEngine.step_execution``. ISAAC hands the placement to a real PhysX
scene and waits to be told what physically happened.

Exactly one backend is authoritative per run. When ``isaac`` is selected, the
orchestrator does not call ``step_execution`` at all — see
``wisepack_orchestration.isaac_bridge`` — so there is never a moment when a
simulated outcome and a physical outcome both claim the same placement.

MAPPING PHYSICS ONTO THE EXISTING WORKFLOW
------------------------------------------
Isaac reports eleven physical states. WISEPACK already has a workflow ``Stage``
vocabulary that the dashboard timeline, the FIWARE ``stage`` attribute and the
validation scripts all consume. Rather than inventing a second, parallel state
display, each Isaac state maps onto the WISEPACK stage it belongs to, and the
finer physical detail rides in the action event's ``details`` block. So a viewer
who knows nothing about Isaac still sees a coherent PICK_ITEM -> VERIFY_PICK ->
PLACE_ITEM -> VERIFY_PLACEMENT -> UPDATE_CONTAINER_STATE progression, and a
viewer who does can read the physical state underneath it.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

from .events import Stage
from .isaac_contract import IsaacState


class ExecutionBackend(str, Enum):
    """Who performs the approved placements."""

    #: The seeded robot model in ``WorkflowEngine``. No kinematics, no contacts;
    #: a pick outcome is a coin flip and every KPI derived from it is labelled
    #: ``simulated``. This is the default and its behaviour is unchanged.
    SIMULATED = "simulated"
    #: NVIDIA Isaac Sim on the host: a Franka Panda, PhysX contacts, and a real
    #: gravity-driven release into a physical container.
    ISAAC = "isaac"

    @property
    def is_physical(self) -> bool:
        return self is ExecutionBackend.ISAAC

    @property
    def label(self) -> str:
        """Short human label for the dashboard badge. Never claims more than it is."""
        return {"simulated": "SIMULATED EXECUTION",
                "isaac": "ISAAC SIM / PHYSICS"}[self.value]

    @property
    def detail(self) -> str:
        return {
            "simulated": ("pick and place outcomes are produced by a seeded "
                          "model — no kinematics, no contacts, no physics"),
            "isaac": ("placements are executed by a Franka Emika Panda in NVIDIA "
                      "Isaac Sim; the release is physical and PhysX settles the "
                      "item under gravity and contact"),
        }[self.value]


#: Item-count ceiling a bench-scale physical cell can reasonably run. Above this
#: a "run" is twenty minutes of watching an arm, and the presets that large are
#: sized for an industrial container the Panda cannot reach across anyway.
PHYSICAL_MAX_ITEMS = 8

#: Gripper opening, millimetres. An item wider than this cannot be grasped at
#: all, so offering such a preset would be offering a run that cannot happen.
PHYSICAL_MAX_DIAMETER_MM = 78


def preset_physical_compatibility(preset: str) -> tuple:
    """(compatible, reason) for running ``preset`` on a physical backend.

    The dashboard must not offer the operator a scenario the physical adapter
    cannot instantiate and reach. `mixed_pipes_dense` exists in the software
    generator and is a perfectly good PACKING benchmark — it is simply not
    something a bench-scale Panda can execute, and letting it be selected in an
    Isaac run produces a plan whose first pick is impossible.

    Import-light on purpose: the generator is imported lazily so this stays
    usable from the dashboard without pulling the whole core in at module load.
    """
    from .generator import PRESETS, preset_config                # noqa: PLC0415
    if preset not in PRESETS:
        return False, f"unknown preset {preset!r}"
    try:
        cfg = preset_config(preset, seed=0)
    except Exception as exc:                                     # noqa: BLE001
        return False, f"cannot be configured: {exc}"
    if cfg.item_count > PHYSICAL_MAX_ITEMS:
        return False, (f"{cfg.item_count} items — a physical cell runs at most "
                       f"{PHYSICAL_MAX_ITEMS} in one demonstration")
    widest = max(cfg.diameter_range_mm)
    if widest > PHYSICAL_MAX_DIAMETER_MM:
        return False, (f"items up to {widest} mm across exceed the "
                       f"{PHYSICAL_MAX_DIAMETER_MM} mm gripper opening")
    return True, ""


def physical_presets() -> dict:
    """{preset: reason} for every known preset; reason is "" when compatible."""
    from .generator import PRESETS                                # noqa: PLC0415
    return {name: preset_physical_compatibility(name)[1] for name in sorted(PRESETS)}


def parse_backend(value: Optional[str]) -> ExecutionBackend:
    """Parse a backend name from a launch argument, env var or CLI flag.

    Unknown values raise rather than falling back to ``simulated``. A typo that
    quietly selects the simulated backend would produce a run that reports
    physical execution it never performed.
    """
    if value is None or str(value).strip() == "":
        return ExecutionBackend.SIMULATED
    text = str(value).strip().lower()
    try:
        return ExecutionBackend(text)
    except ValueError as exc:
        known = ", ".join(b.value for b in ExecutionBackend)
        raise ValueError(
            f"unknown execution_backend {value!r}; known backends: {known}") from exc


# --------------------------------------------------------------------------- #
# Isaac physical state -> WISEPACK workflow stage
# --------------------------------------------------------------------------- #

#: One entry per IsaacState. Asserted exhaustive by the test suite, because a
#: state added to the contract and forgotten here would render as whatever the
#: previous stage happened to be — a timeline that silently stops advancing.
ISAAC_STATE_STAGE: Dict[IsaacState, Optional[Stage]] = {
    # READY is about the SIMULATOR, not about an item. It must not move the
    # workflow stage: the plan may still be awaiting approval when Isaac comes
    # up, and advancing the stage there would show a pick before authorisation.
    IsaacState.READY: None,

    # The scene-reset lifecycle is about the SIMULATOR, not about an item, so
    # like READY none of it moves the workflow stage. The orchestrator gates
    # authorisation on SCENE_READY separately — see isaac_bridge.
    IsaacState.RESET_REQUESTED: None,
    IsaacState.RESETTING: None,
    IsaacState.SCENE_READY: None,
    IsaacState.RESET_FAILED: None,

    IsaacState.MOVING_TO_PICK: Stage.PICK_ITEM,
    IsaacState.GRASPING: Stage.PICK_ITEM,
    # The lift is the physical evidence that the grasp actually holds, which is
    # exactly what VERIFY_PICK means in the simulated backend.
    IsaacState.LIFTING: Stage.VERIFY_PICK,
    IsaacState.MOVING_TO_CONTAINER: Stage.PLACE_ITEM,
    IsaacState.RELEASING: Stage.PLACE_ITEM,
    # Settling is where the placement is decided by physics rather than by the
    # plan, so it belongs to VERIFY_PLACEMENT, not to PLACE_ITEM.
    IsaacState.SETTLING: Stage.VERIFY_PLACEMENT,
    IsaacState.ITEM_COMPLETED: Stage.UPDATE_CONTAINER_STATE,
    IsaacState.ITEM_FAILED: Stage.VERIFY_PICK,
    IsaacState.RUN_COMPLETED: Stage.COMPLETE,
    IsaacState.RUN_FAILED: Stage.DEGRADED,
}


def stage_for_isaac_state(state: IsaacState) -> Optional[Stage]:
    """The WISEPACK stage an Isaac physical state belongs to, or None.

    None means "this report does not move the workflow stage" — currently only
    READY, which is a simulator-lifecycle fact rather than an item fact.
    """
    return ISAAC_STATE_STAGE[IsaacState(state)]


#: Isaac states that describe the robot as physically busy with an item. Used to
#: drive the dashboard's robot indicator through the same vocabulary the
#: simulated backend uses ("picking" / "placing" / "idle").
_ROBOT_STATE = {
    IsaacState.MOVING_TO_PICK: "picking",
    IsaacState.GRASPING: "picking",
    IsaacState.LIFTING: "picking",
    IsaacState.MOVING_TO_CONTAINER: "placing",
    IsaacState.RELEASING: "placing",
    IsaacState.SETTLING: "placing",
}


def robot_state_for_isaac_state(state: IsaacState) -> str:
    return _ROBOT_STATE.get(IsaacState(state), "idle")


__all__ = [
    "ExecutionBackend", "parse_backend", "PHYSICAL_MAX_ITEMS",
    "PHYSICAL_MAX_DIAMETER_MM", "preset_physical_compatibility",
    "physical_presets", "ISAAC_STATE_STAGE",
    "stage_for_isaac_state", "robot_state_for_isaac_state",
]
