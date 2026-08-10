#!/usr/bin/env python3
"""Stage D — the perceived Cylinder5 through the normal WISEPACK workflow.

    ./scripts/stage_d.sh --reuse

Takes the Stage C `PhysicalObservation` — a FoundationPose estimate expressed in
`wisepack_workarea` — and runs it through the ORDINARY planning path: apply the
observation batch, advance the scenario revision, generate a plan with the
existing optimizer, validate the Digital Twin, and stop at the approval gate.

NO SPECIAL PLANNER. There is no FoundationPose-aware planning code; the batch
enters through `WorkflowEngine.apply_observation_batch`, exactly as a planar
camera batch does. If that needed a second path, the provider boundary would
have failed.

NO ROBOT. Stage D stops at "approval pending". Nothing here loads an arm, solves
IK or commands a motion.

ISAAC GROUND TRUTH IS NOT AN INPUT. It is read once at the end, to report how
far the plan's object is from where the simulator actually put it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))

STAGE_C = os.path.join(REPO, ".cache-perception", "stage-c", "stage_c.json")
FRAME = os.path.join(REPO, ".cache-perception", "isaac-reference",
                     "stage_a_workcell", "ground_truth.json")
EVIDENCE = os.path.join(REPO, ".cache-perception", "stage-d")


def main(argv=None) -> int:
    from wisepack_core.domain import PhysicalObservation
    from wisepack_core.generator import build_scenario
    from wisepack_core.perception import (BatchStatus, ObservationBatch,
                                          PerceptionSource)
    from wisepack_core.workflow import WorkflowEngine, WorkflowConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    if not os.path.isfile(STAGE_C):
        print(f"No Stage C result at {STAGE_C}.\n"
              "  Run ./scripts/stage_c.sh first.", file=sys.stderr)
        return 2
    with open(STAGE_C, encoding="utf-8") as handle:
        stage_c = json.load(handle)
    os.makedirs(EVIDENCE, exist_ok=True)

    observation = PhysicalObservation.from_dict(stage_c["observation"])
    task = observation.task_geometry()

    print("STAGE D — perceived Cylinder5 -> plan -> Digital Twin -> approval")
    print(f"  perception     {observation.perception_method}")
    print(f"  object model   {observation.object_model_id}")
    print(f"  frame          {observation.frame_id}")
    print()
    print("  WHAT THE PLANNER WILL USE")
    print(f"    object_center (physical body)  "
          f"{[round(v, 2) for v in observation.object_center]} mm")
    print(f"    CAD model-frame origin         "
          f"{[round(v, 2) for v in (observation.x_mm, observation.y_mm, observation.z_mm)]} mm")
    print("    -> the planner takes the CENTRE. The model origin lies outside")
    print("       the body and would target empty space.")
    print(f"    tube axis                      {task['tube_axis_line']}")
    print(f"    geometry                       D{task['diameter_mm']} x "
          f"L{task['length_mm']} mm")
    print()

    # ---- the ORDINARY workflow --------------------------------------------
    batch = ObservationBatch(
        batch_id="stage-d-1", source=PerceptionSource.CAMERA.value,
        status=BatchStatus.OK, observations=[observation],
        frame_id=observation.frame_id,
        captured_at=observation.captured_at,
        detector=observation.detector,
        perception_method=observation.perception_method,
        acquisition="simulated_rgbd",
        model_id=observation.object_model_id,
        calibration_status="not_applicable")

    # THE ORDINARY ENGINE, configured for a camera source and the CAD
    # scenario. `generate_or_load_scenario` is the same entry point every run
    # uses; the perceived batch then replaces its items.
    engine = WorkflowEngine(WorkflowConfig(
        preset="cad_cylinder5_single",
        perception_source=PerceptionSource.CAMERA))
    engine.generate_or_load_scenario(build_scenario("cad_cylinder5_single"))
    revision_before = engine.scenario_revision

    engine.apply_observation_batch(batch)
    print("  WORKFLOW")
    print(f"    run_id                 {engine.run_id}")
    print(f"    scenario_revision      {revision_before} -> "
          f"{engine.scenario_revision}")

    item = engine.scenario.items[0]
    print(f"    planning item          {item.item_id}")
    print(f"    model_id               {item.model_id}")
    print(f"    geometry_source        {item.geometry_source}")
    print(f"    nominal geometry       D{item.outer_diameter_mm} x "
          f"L{item.length_mm} mm, bore {item.inner_diameter_mm}")
    print(f"    source_position        {item.source_position.to_dict()} mm")

    # THE KEY REGRESSION CHECK, in the run itself and not only in a test.
    centre = [round(v) for v in observation.object_center]
    origin = [round(observation.x_mm), round(observation.y_mm),
              round(observation.z_mm)]
    used = [item.source_position.x, item.source_position.y,
            item.source_position.z]
    if used == centre:
        print("    -> the planner is using the PHYSICAL CENTRE. Correct.")
    elif used == origin:
        print("    -> ERROR: the planner is using the CAD MODEL ORIGIN, which "
              "for this part\n       lies outside the body. This is the Stage C "
              "regression.", file=sys.stderr)
        return 1
    else:
        print(f"    -> ERROR: the planner used {used}, which is neither the "
              "centre nor the origin.", file=sys.stderr)
        return 1
    print()

    # ---- plan, twin, approval ---------------------------------------------
    #
    # `digital_twin_validate` is what SELECTS between the baseline and the
    # optimised plan, so it runs before the plan can be reported. That is the
    # engine's own order and is not rearranged here.
    engine.generate_plans()
    valid = engine.digital_twin_validate()
    plan = engine.selected
    print("  PLAN (the existing optimizer, no perception-aware path)")
    print(f"    strategy               {plan.strategy.value}")
    print(f"    containers used        {len(plan.containers)}")
    for container in plan.containers:
        print(f"    container              {container.container_id} "
              f"({container.inner_width_mm}x{container.inner_depth_mm}x"
              f"{container.inner_height_mm} mm)")
    for placement in plan.placements:
        print(f"    placement              {placement.item_id} -> "
              f"{placement.container_id} at "
              f"{placement.position.to_dict()} mm, axis {placement.axis.value}")
    print(f"    items placed           {len(plan.placements)} of "
          f"{len(engine.scenario.items)}")
    print()

    validation = {"valid": bool(valid)}
    print("  DIGITAL TWIN")
    print(f"    valid                  {validation['valid']}")
    print(f"    validated revision     {engine.scenario_revision}")
    print()

    print("  APPROVAL")
    print(f"    stage                  {engine.stage.value}")
    print(f"    approval_revision      {engine.approval_revision}")
    print(f"    scenario_revision      {engine.scenario_revision}")
    pending = engine.approval_revision != engine.scenario_revision
    print(f"    approval pending       {pending} "
          "(a plan for a new revision needs a fresh decision)")
    print()
    print("  NO ROBOT MOTION. Stage D stops here.")

    # ---- evaluation, after the fact ---------------------------------------
    settled = None
    if os.path.isfile(FRAME):
        with open(FRAME, encoding="utf-8") as handle:
            settled = json.load(handle)["settled_workarea"]["position_mm"]
        error = sum((a - b) ** 2 for a, b in
                    zip(observation.object_center, settled)) ** 0.5
        print(f"  EVALUATION (diagnostics only): the planned object sits "
              f"{error:.2f} mm from\n    where Isaac actually settled it "
              f"{[round(v, 2) for v in settled]} mm")

    document = {
        "run_id": engine.run_id,
        "scenario_revision": engine.scenario_revision,
        "model_id": item.model_id,
        "geometry_source": item.geometry_source,
        "perceived_object_center_mm": [float(v) for v in observation.object_center],
        "cad_model_frame_origin_mm": [observation.x_mm, observation.y_mm,
                                      observation.z_mm],
        "planner_used": used,
        "planner_used_the_physical_centre": used == centre,
        "tube_axis_line": task["tube_axis_line"],
        "plan": {
            "strategy": plan.strategy.value,
            "containers": [c.container_id for c in plan.containers],
            "placements": [{"item_id": p.item_id,
                            "container_id": p.container_id,
                            "position_mm": p.position.to_dict(),
                            "axis": p.axis.value} for p in plan.placements],
        },
        "digital_twin": validation,
        "approval": {"stage": engine.stage.value,
                     "approval_revision": engine.approval_revision,
                     "scenario_revision": engine.scenario_revision,
                     "pending": pending},
        "evaluation": {"settled_position_mm": settled,
                       "note": "Isaac ground truth, diagnostics only; it never "
                               "entered the planning path."},
    }
    with open(os.path.join(EVIDENCE, "stage_d.json"), "w",
              encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, default=str)
    print(f"\n  wrote {EVIDENCE}/stage_d.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
