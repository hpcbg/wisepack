#!/usr/bin/env python3
"""Generate the presentation evidence: run artefacts + SVG figures.

Runs the REAL pipeline for every scenario preset — the same generator, the same
two algorithms, the same independent validator — and writes both the machine
artefacts and the figures used in the README. Nothing here is drawn from a
constant: if a figure shows "3 containers", three containers were computed while
producing it.

Figures are SVG rather than PNG deliberately: no matplotlib dependency, they
scale in a slide deck, and they are diffable in review.

    python3 scripts/generate_demo_artifacts.py
    python3 scripts/generate_demo_artifacts.py --preset curated_volume_reduction
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from typing import Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))

from wisepack_core.artifacts import (                              # noqa: E402
    latest_latency_p50_ms, write_run_artifacts, write_validation_report,
)
from wisepack_core.domain import PackingPlan, Scenario, Strategy   # noqa: E402
from wisepack_core.events import DynamicEvent, DynamicEventType    # noqa: E402
from wisepack_core.generator import PRESETS, build_scenario        # noqa: E402
from wisepack_core.kpi import compare_strategies                   # noqa: E402
from wisepack_core.packing import (                                # noqa: E402
    OptimizerConfig, pack_baseline, pack_optimized, select_plan,
)
from wisepack_core.workflow import (                               # noqa: E402
    RobotSimConfig, WorkflowConfig, run_headless,
)

IMAGES = os.path.join(REPO, "images", "generated")
RESULTS = os.path.join(REPO, "results")

# Palette chosen to survive both a light slide and a dark terminal screenshot,
# and to stay distinguishable under the common colour-vision deficiencies
# (blue / amber / magenta rather than red / green).
INK = "#12222f"
MUT = "#5b7185"
LINE = "#d8e0ea"
BG = "#ffffff"
GROUP_COLORS = {"A": "#1266d1", "B": "#b06d00", "C": "#8e44ad",
                "D": "#0e8a6d", "E": "#c0392b"}
BASE_C = "#b06d00"
OPT_C = "#1266d1"


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def svg_open(width: int, height: int, title: str) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="{esc(title)}" font-family="ui-sans-serif,system-ui,Arial">',
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
        f'<text x="16" y="26" font-size="16" font-weight="700" fill="{INK}">'
        f'{esc(title)}</text>',
    ]


def svg_close(parts: List[str], path: str) -> str:
    parts.append("</svg>")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts) + "\n")
    return path


def plan_scale(plan: PackingPlan, width: int, height: int,
               rows: Optional[int] = None) -> float:
    """mm -> px scale for one plan, optionally forced to a shared row count.

    A comparison figure MUST pass the same ``rows`` for both plans. Physically
    identical containers rendered at different sizes because one plan happens to
    need more of them is a misleading picture, not a cosmetic issue: the eye
    reads the smaller boxes as smaller containers.
    """
    used = plan.containers_used
    if not used:
        return 1.0
    rows = rows or len(used)
    row_h = height / max(1, rows)
    cnt = used[0]
    return min((width - 30) / cnt.inner_width_mm,
               (row_h - 34) / (cnt.inner_depth_mm + cnt.inner_height_mm))


def _draw_plan(parts: List[str], plan: PackingPlan, scenario: Scenario,
               x0: int, y0: int, width: int, height: int, label: str,
               scale: Optional[float] = None,
               rows: Optional[int] = None) -> None:
    """Top and side projections of every used container, to scale."""
    used = plan.containers_used
    parts.append(f'<text x="{x0}" y="{y0 - 8}" font-size="12" font-weight="700" '
                 f'fill="{INK}">{esc(label)}</text>')
    if not used:
        return
    items = {i.item_id: i for i in scenario.items}
    rows = rows or len(used)
    row_h = height / max(1, rows)
    sc = scale if scale is not None else plan_scale(plan, width, height, rows)
    for ci, cnt in enumerate(used):
        cy = y0 + ci * row_h
        cw, cd, ch = cnt.inner_width_mm, cnt.inner_depth_mm, cnt.inner_height_mm
        bw, top_h, side_h = cw * sc, cd * sc, ch * sc
        oy_top = cy + 14
        oy_side = oy_top + top_h + 6
        parts.append(f'<text x="{x0}" y="{cy + 9}" font-size="9" fill="{MUT}">'
                     f'{esc(cnt.container_id)} · {cw}×{cd}×{ch} mm</text>')
        for oy, hh in ((oy_top, top_h), (oy_side, side_h)):
            parts.append(f'<rect x="{x0:.1f}" y="{oy:.1f}" width="{bw:.1f}" '
                         f'height="{hh:.1f}" fill="#f7f9fb" stroke="{LINE}" '
                         f'stroke-width="1"/>')
        for z in cnt.shelf_levels_mm:
            yy = oy_side + side_h - z * sc
            parts.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x0 + bw:.1f}" '
                         f'y2="{yy:.1f}" stroke="#c05a11" stroke-width="1" '
                         f'stroke-dasharray="3 3" opacity="0.75"/>')
        for p in plan.placements_for(cnt.container_id):
            item = items.get(p.item_id)
            col = GROUP_COLORS.get(item.segregation_group if item else "A", OPT_C)
            stroke = "#c05a11" if (item and item.injected) else col
            parts.append(
                f'<rect x="{x0 + p.position.x * sc:.1f}" '
                f'y="{oy_top + p.position.y * sc:.1f}" '
                f'width="{max(1.0, p.size.x * sc):.1f}" '
                f'height="{max(1.0, p.size.y * sc):.1f}" fill="{col}" '
                f'fill-opacity="0.55" stroke="{stroke}" stroke-width="0.8"/>')
            parts.append(
                f'<rect x="{x0 + p.position.x * sc:.1f}" '
                f'y="{oy_side + side_h - (p.position.z + p.size.z) * sc:.1f}" '
                f'width="{max(1.0, p.size.x * sc):.1f}" '
                f'height="{max(1.0, p.size.z * sc):.1f}" fill="{col}" '
                f'fill-opacity="0.55" stroke="{stroke}" stroke-width="0.8"/>')


def figure_packing(scenario: Scenario, plan: PackingPlan, path: str,
                   title: str) -> str:
    parts = svg_open(760, 460, title)
    used = plan.containers_used
    subtitle = (f"{len(used)} container(s) · {plan.utilization_pct:.1f}% utilization"
                f" · {len(plan.placements)} placements · "
                f"{'passes' if plan.is_valid else 'FAILS'} independent validation")
    parts.append(f'<text x="16" y="44" font-size="11" fill="{MUT}">'
                 f'{esc(subtitle)}</text>')
    _draw_plan(parts, plan, scenario, 20, 72, 720, 360, "")
    parts.append(f'<text x="16" y="452" font-size="9" fill="{MUT}">'
                 f'Top and side projection per container, to scale. '
                 f'Dashed orange = shelf plate. Colour = segregation group.</text>')
    return svg_close(parts, path)


def figure_comparison(scenario: Scenario, baseline: PackingPlan,
                      optimized: PackingPlan, path: str) -> str:
    parts = svg_open(980, 500,
                     f"Baseline vs optimized — {scenario.scenario_id}")
    reduction = (100.0 * (baseline.required_capacity_mm3
                          - optimized.required_capacity_mm3)
                 / baseline.required_capacity_mm3
                 if baseline.required_capacity_mm3 else 0.0)
    parts.append(
        f'<text x="16" y="44" font-size="11" fill="{MUT}">'
        f'Measured: {baseline.containers_required} -> '
        f'{optimized.containers_required} containers, utilization '
        f'{baseline.utilization_pct:.1f}% -> {optimized.utilization_pct:.1f}%, '
        f'volume requirement reduction {reduction:.1f}%</text>')
    # ONE shared scale and row height across both panels: the containers are
    # physically identical, so they must render identically.
    rows = max(baseline.containers_required, optimized.containers_required, 1)
    scale = min(plan_scale(baseline, 450, 380, rows),
                plan_scale(optimized, 450, 380, rows))
    _draw_plan(parts, baseline, scenario, 20, 84, 450, 380,
               f"BASELINE  {baseline.algorithm}", scale=scale, rows=rows)
    _draw_plan(parts, optimized, scenario, 510, 84, 450, 380,
               f"OPTIMIZED  {optimized.algorithm} / {optimized.strategy.value}",
               scale=scale, rows=rows)
    parts.append(f'<text x="16" y="492" font-size="9" fill="{MUT}">'
                 f'Both plans pass the same independent validator. Reduction '
                 f'denominator is required container capacity, not material volume.</text>')
    return svg_close(parts, path)


def figure_kpi_bars(rows: List[Tuple[str, float, float, str]], path: str,
                    title: str) -> str:
    """Grouped bars: baseline vs optimized for each metric."""
    height = 90 + len(rows) * 62
    parts = svg_open(760, height, title)
    parts.append(f'<text x="16" y="44" font-size="11" fill="{MUT}">'
                 f'All values MEASURED by running both algorithms.</text>')
    x_label, x_bar, bar_w = 16, 250, 420
    for n, (label, base, opt, unit) in enumerate(rows):
        y = 74 + n * 62
        top = max(base, opt, 1e-9)
        parts.append(f'<text x="{x_label}" y="{y + 14}" font-size="11" '
                     f'fill="{INK}">{esc(label)}</text>')
        for k, (value, colour, name) in enumerate(
                ((base, BASE_C, "baseline"), (opt, OPT_C, "optimized"))):
            w = max(2.0, bar_w * value / top)
            by = y + k * 20
            parts.append(f'<rect x="{x_bar}" y="{by}" width="{w:.1f}" height="16" '
                         f'fill="{colour}" fill-opacity="0.85" rx="2"/>')
            parts.append(f'<text x="{x_bar + w + 7:.1f}" y="{by + 12}" '
                         f'font-size="10" fill="{INK}">'
                         f'{value:.1f} {esc(unit)} <tspan fill="{MUT}">'
                         f'{name}</tspan></text>')
    return svg_close(parts, path)


def figure_topology(path: str) -> str:
    parts = svg_open(940, 460,
                     "WISEPACK architecture — solid: telemetry, dashed: commands")
    layers = [
        ("Perception layer", [("Task generator", "deterministic, seeded"),
                              ("Perception simulator", "SIMULATED — no vision model")]),
        ("Optimization + Digital Twin", [("Packing optimizer", "geometry_aware_ep_bfd"),
                                         ("Digital Twin validator", "independent process")]),
        ("HitL + decision support", [("py_trees orchestrator", "approval gate"),
                                     ("Operator", "approve / reject / alter"),
                                     ("Robot simulator", "SIMULATED")]),
        ("Middleware", [("ROS 2 / DDS", "Vulcanexus Fast DDS")]),
        ("Logging + analytics", [("Orion-LD", "NGSI-LD, -wip dds"),
                                 ("Dashboard", "FastAPI + SVG")]),
    ]
    NW, NH = 164, 38

    # PASS 1 — compute geometry only. Nothing is emitted yet, because edges must
    # be painted UNDER the nodes: drawn on top they cross the boxes and obscure
    # exactly the labels a reader needs.
    positions: Dict[str, Tuple[float, float, int]] = {}
    layer_headers: List[Tuple[int, str]] = []
    y = 66
    for layer_index, (layer_name, nodes) in enumerate(layers):
        layer_headers.append((y, layer_name))
        step = 700 / (len(nodes) + 1)
        for i, (name, _kind) in enumerate(nodes):
            positions[name] = (210 + step * (i + 1), y + NH / 2, layer_index)
        y += 76

    edges = [
        ("Task generator", "Packing optimizer", False),
        ("Perception simulator", "Packing optimizer", False),
        ("Packing optimizer", "Digital Twin validator", False),
        ("Digital Twin validator", "py_trees orchestrator", False),
        ("Operator", "py_trees orchestrator", True),
        ("py_trees orchestrator", "Robot simulator", True),
        ("Robot simulator", "py_trees orchestrator", False),
        ("py_trees orchestrator", "ROS 2 / DDS", False),
        ("ROS 2 / DDS", "Orion-LD", False),
        ("Orion-LD", "Dashboard", False),
        ("Dashboard", "Orion-LD", True),
    ]

    # PASS 2 — edges.
    for a, b, dashed in edges:
        if a not in positions or b not in positions:
            continue
        ax, ay, al = positions[a]
        bx, by, bl = positions[b]
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        colour = OPT_C if dashed else MUT
        if al == bl:
            # Same layer: arc BELOW the row rather than straight through the
            # boxes between them.
            drop = ay + NH / 2 + (14 if not dashed else 24)
            d = (f"M{ax},{ay + NH / 2} C{ax},{drop} {bx},{drop} "
                 f"{bx},{by + NH / 2}")
        else:
            a1 = ay + (NH / 2 if by > ay else -NH / 2)
            b1 = by + (-NH / 2 if by > ay else NH / 2)
            mid = (a1 + b1) / 2
            d = f"M{ax},{a1} C{ax},{mid} {bx},{mid} {bx},{b1}"
        parts.append(f'<path d="{d}" fill="none" stroke="{colour}" '
                     f'stroke-width="1.3"{dash}/>')

    # PASS 3 — nodes, painted over the edges.
    for y_top, layer_name in layer_headers:
        parts.append(f'<text x="16" y="{y_top + 22}" font-size="9" fill="{MUT}" '
                     f'letter-spacing="1">{esc(layer_name.upper())}</text>')
    for layer_name, nodes in layers:
        for name, kind in nodes:
            cx, cy, _ = positions[name]
            top = cy - NH / 2
            parts.append(f'<rect x="{cx - NW / 2}" y="{top}" width="{NW}" '
                         f'height="{NH}" rx="6" fill="{BG}" stroke="{OPT_C}" '
                         f'stroke-width="1.4"/>')
            parts.append(f'<text x="{cx}" y="{top + 16}" font-size="11" '
                         f'font-weight="700" text-anchor="middle" fill="{INK}">'
                         f'{esc(name)}</text>')
            parts.append(f'<text x="{cx}" y="{top + 30}" font-size="8" '
                         f'text-anchor="middle" fill="{MUT}">{esc(kind)}</text>')

    parts.append(f'<text x="16" y="452" font-size="9" fill="{MUT}">'
                 f'Every workflow action is published on /wisepack/action/event and '
                 f'reaches Orion-LD over DDS. There is no direct-HTTP audit path.</text>')
    return svg_close(parts, path)


def figure_replan(scenario: Scenario, before: PackingPlan, after: PackingPlan,
                  path: str, label: str) -> str:
    parts = svg_open(980, 500, "Dynamic re-planning — a late high-priority arrival")
    parts.append(f'<text x="16" y="44" font-size="11" fill="{MUT}">{esc(label)}</text>')
    rows = max(before.containers_required, after.containers_required, 1)
    scale = min(plan_scale(before, 450, 380, rows),
                plan_scale(after, 450, 380, rows))
    _draw_plan(parts, before, scenario, 20, 84, 450, 380, "BEFORE the event",
               scale=scale, rows=rows)
    _draw_plan(parts, after, scenario, 510, 84, 450, 380, "AFTER the re-plan",
               scale=scale, rows=rows)
    parts.append(f'<text x="16" y="492" font-size="9" fill="{MUT}">'
                 f'Orange outline = the injected item. Already-executed placements '
                 f'are frozen; only the remainder is re-optimized.</text>')
    return svg_close(parts, path)


def run_preset(preset: str, seed: int, write_report: bool = True) -> Dict:
    scenario = build_scenario(preset, seed)
    baseline = pack_baseline(scenario)
    optimized = pack_optimized(scenario, config=OptimizerConfig(seed=seed,
                                                                restarts=6))
    selected, reason = select_plan(baseline, optimized, scenario)
    return {"scenario": scenario, "baseline": baseline, "optimized": optimized,
            "selected": selected, "reason": reason}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default=None,
                        help="only this preset (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(IMAGES, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    presets = [args.preset] if args.preset else sorted(PRESETS)
    written: List[str] = []
    summary_rows: List[str] = []

    print(f"{'preset':28s} {'baseline':>12s} {'optimized':>12s} {'reduction':>10s}")
    print("-" * 66)

    for preset in presets:
        r = run_preset(preset, args.seed)
        scenario, baseline, optimized = r["scenario"], r["baseline"], r["optimized"]
        reduction = (100.0 * (baseline.required_capacity_mm3
                              - optimized.required_capacity_mm3)
                     / baseline.required_capacity_mm3
                     if baseline.required_capacity_mm3 else 0.0)
        print(f"{preset:28s} {baseline.containers_required:>4d}c "
              f"{baseline.utilization_pct:>6.1f}% "
              f"{optimized.containers_required:>4d}c "
              f"{optimized.utilization_pct:>6.1f}% {reduction:>9.1f}%")
        summary_rows.append(
            f"| `{preset}` | {baseline.containers_required} | "
            f"{baseline.utilization_pct:.1f}% | {optimized.containers_required} | "
            f"{optimized.utilization_pct:.1f}% | **{reduction:.1f}%** |")

        written.append(figure_comparison(
            scenario, baseline, optimized,
            os.path.join(IMAGES, f"comparison-{preset}.svg")))

    # The headline figures use the dense scenario.
    r = run_preset("mixed_pipes_dense", args.seed)
    scenario, baseline, optimized = r["scenario"], r["baseline"], r["optimized"]
    written.append(figure_packing(scenario, baseline,
                                  os.path.join(IMAGES, "baseline-packing.svg"),
                                  f"Baseline packing — {baseline.algorithm}"))
    written.append(figure_packing(scenario, optimized,
                                  os.path.join(IMAGES, "optimized-packing.svg"),
                                  f"Optimized packing — {optimized.algorithm}"))
    written.append(figure_kpi_bars([
        ("Containers required", float(baseline.containers_required),
         float(optimized.containers_required), "containers"),
        ("Container utilization", baseline.utilization_pct,
         optimized.utilization_pct, "%"),
        ("Required capacity", baseline.required_capacity_mm3 / 1e9,
         optimized.required_capacity_mm3 / 1e9, "m3"),
        ("Empty capacity", baseline.unused_capacity_mm3 / 1e9,
         optimized.unused_capacity_mm3 / 1e9, "m3"),
        ("Computation time", baseline.computation_time_ms,
         optimized.computation_time_ms, "ms"),
    ], os.path.join(IMAGES, "kpi-comparison.svg"),
        f"KPI comparison — {scenario.scenario_id}"))
    written.append(figure_topology(os.path.join(IMAGES, "topology.svg")))

    # Re-planning figure: capture the plan before and after a real injected event.
    event = DynamicEvent(
        event_type=DynamicEventType.ITEM_INJECT, trigger="placement:3",
        label="High-priority ILW component arrives late",
        payload={"item": {"length_mm": 1200, "outer_diameter_mm": 220,
                          "inner_diameter_mm": 186, "material": "stainless_316L",
                          "priority": 9, "dose_class": "ILW"}})
    before = run_preset("late_arrival_replan", args.seed)
    engine = run_headless(WorkflowConfig(
        preset="late_arrival_replan", seed=args.seed, auto_approve=True,
        optimizer=OptimizerConfig(seed=args.seed, restarts=6),
        robot=RobotSimConfig(seed=args.seed),
        dynamic_events=[event]))
    written.append(figure_replan(
        engine.scenario, before["optimized"], engine.selected,
        os.path.join(IMAGES, "dynamic-replan.svg"),
        f"{engine.stats.replans} re-plan(s): "
        f"{engine.stats.replan_causes[0] if engine.stats.replan_causes else 'none'}"))

    if not args.images_only:
        kpis = engine.kpis(latest_latency_p50_ms(RESULTS))
        artifacts = write_run_artifacts(
            engine.scenario, engine.baseline, engine.optimized, engine.selected,
            kpis, engine.log, RESULTS)
        report = write_validation_report(
            engine.scenario, engine.baseline, engine.optimized, engine.selected,
            kpis, engine.log, artifacts, RESULTS)
        written.append(report)
        print(f"\nrun artefacts: {artifacts.stamp}")

    # A markdown table the README can quote verbatim.
    table = os.path.join(RESULTS, "wisepack-preset-summary.md")
    with open(table, "w", encoding="utf-8") as fh:
        fh.write("| Scenario | Baseline containers | Baseline util. | "
                 "Optimized containers | Optimized util. | Volume requirement "
                 "reduction |\n|---|---|---|---|---|---|\n")
        fh.write("\n".join(summary_rows) + "\n")
    written.append(table)

    print(f"\nwrote {len(written)} file(s):")
    for path in written:
        print(f"  {os.path.relpath(path, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
