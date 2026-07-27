"""Generate WISEPACK Behaviour Tree diagrams FROM the implementation.

The node set is derived from the live source of truth — ``wisepack_core.events.Stage``
(the canonical workflow stages), the ``py_trees`` tree shape in
``wisepack_orchestration.hitl_orchestrator.build_tree``, and the anomaly
reactions in ``wisepack_core.anomaly`` — so a diagram cannot silently drift from
the code. A test (``test_behaviour_tree.py``) asserts the required nodes appear.

Outputs (SVG always; PNG when cairosvg is importable):
    images/generated/wisepack_behaviour_tree.svg / .png            (full engineering)
    images/generated/wisepack_behaviour_tree_interview.svg / .png  (simplified)

    python3 scripts/generate_behaviour_tree_images.py
"""

from __future__ import annotations

import html
import os
import sys
from typing import Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_bringup"))

from wisepack_core.anomaly import Reaction, Severity          # noqa: E402
from wisepack_core.events import Stage                        # noqa: E402

OUT = os.path.join(REPO, "images", "generated")

INK, MUT, LINE, BG = "#12222f", "#5b7185", "#d8e0ea", "#ffffff"
# Node kinds -> fill/stroke. Colours chosen to read in light and dark screenshots.
KIND = {
    "root": ("#eef4fb", "#1266d1"),
    "action": ("#ffffff", "#5b7185"),
    "gate": ("#fff6e6", "#c05a11"),        # human approval / acknowledgement
    "validate": ("#eaf6ef", "#0e8a6d"),    # Digital Twin validation
    "hold": ("#fdecea", "#c0392b"),        # anomaly / degraded hold
    "loop": ("#f2f0fb", "#8e44ad"),
    "done": ("#eef4fb", "#1266d1"),
}

#: The core invariant, printed on every diagram.
INVARIANT = ("No pick or placement may execute unless the exact current plan "
             "revision is independently validated and explicitly approved.")


def esc(t) -> str:
    return html.escape(str(t), quote=True)


# node tuple shape: (id, label, sublabel, kind)


def full_tree() -> Tuple[List, List[Tuple[str, str, str]]]:
    """The full engineering tree, node ids anchored to Stage names where they map."""
    S = Stage
    # (id, label, sublabel, kind)
    nodes = [
        ("ROOT", "WISEPACK", "py_trees Sequence (memory)", "root"),
        (S.GENERATE_OR_LOAD_SCENARIO.value, "Generate / load scenario",
         "task generator, seeded", "action"),
        (S.SCAN_SOURCE_BIN.value, "Scan source bin", "SIMULATED", "action"),
        (S.DETECT_ITEMS.value, "Perception / detect items", "SIMULATED", "action"),
        (S.GENERATE_BASELINE_PLAN.value, "Baseline planning",
         "arrival_order_shelf", "action"),
        (S.GENERATE_OPTIMIZED_PLAN.value, "Optimized planning",
         "geometry_aware_ep_bfd", "action"),
        ("COMPARE", "Compare strategies", "decision support (no state change)", "action"),
        # -- cut-aware whole-process branch (brief §20) --
        (S.GENERATE_CUT_ALTERNATIVES.value, "Generate cut alternatives",
         "bounded candidates", "action"),
        (S.DIGITAL_TWIN_VALIDATE_CUT_PLAN.value, "Validate cut plan",
         "conservation + lineage (INDEPENDENT)", "validate"),
        (S.WAIT_FOR_CUT_APPROVAL.value, "Wait for cut approval",
         "SEPARATE from packing approval", "gate"),
        (S.CUT_REQUESTED.value, "Cut requested",
         "external cutting skill (SIMULATED)", "action"),
        (S.CUT_COMPLETED.value, "Cut completed / failed", "SIMULATED", "action"),
        (S.REGISTER_DERIVED_ITEMS.value, "Register derived items",
         "actual segment sizes + lineage", "action"),
        (S.REPLAN_AFTER_CUT.value, "Re-plan after cut",
         "packing approval required again", "action"),
        # -- inventory / logistics branch (brief §20) --
        (S.CHECK_CONTAINER_AVAILABILITY.value, "Check container availability",
         "inventory-aware planning", "action"),
        (S.RESERVE_CONTAINER.value, "Reserve container",
         "FIWARE-backed inventory", "action"),
        (S.WAIT_FOR_CONTAINER.value, "Wait for container",
         "delivery / shortage", "gate"),
        (S.DIGITAL_TWIN_VALIDATE.value, "Digital Twin validation",
         "INDEPENDENT validator", "validate"),
        (S.WAIT_FOR_OPERATOR_APPROVAL.value, "Wait for operator approval",
         "the gate — never times out", "gate"),
        (S.COLLECT_FULL_CONTAINER.value, "Collect full container",
         "SIMULATED logistics", "loop"),
        ("APPROVE", "Approve", "operator", "gate"),
        ("REJECT", "Reject / alternative strategy", "-> re-plan -> gate", "gate"),
        (S.PICK_ITEM.value, "Pick item", "SIMULATED robot", "loop"),
        (S.VERIFY_PICK.value, "Verify pick", "SIMULATED", "loop"),
        (S.PLACE_ITEM.value, "Place item", "SIMULATED robot", "loop"),
        (S.VERIFY_PLACEMENT.value, "Verify placement",
         "re-validate vs geometry", "validate"),
        (S.UPDATE_CONTAINER_STATE.value, "Update container state", "", "loop"),
        (S.NEXT_ITEM.value, "Next item", "loop back", "loop"),
        ("DYN", "Dynamic event handling",
         "inject / remove / unavailable", "action"),
        ("ANOM", "Anomaly Monitoring",
         "SIMULATED, deterministic", "hold"),
        ("ANOM_PAUSE", "Anomaly PAUSE (warning)",
         "acknowledge to resume", "gate"),
        ("ANOM_HOLD", "Anomaly HOLD (critical)",
         "revoke authorisation", "hold"),
        ("ACK", "Acknowledge anomaly", "operator", "gate"),
        ("PAUSE", "Pause / resume / step", "operator supervision", "action"),
        (S.REPLAN.value, "Re-plan", "freeze executed, re-optimize", "action"),
        ("REAPPROVE", "Renewed approval required", "back to the gate", "gate"),
        (S.COMPLETE.value, "Complete", "artefacts written", "done"),
        (S.DEGRADED.value, "Degraded / failure hold",
         "held, never auto-continue", "hold"),
    ]
    edges = [
        ("ROOT", S.GENERATE_OR_LOAD_SCENARIO.value, ""),
        (S.GENERATE_OR_LOAD_SCENARIO.value, S.SCAN_SOURCE_BIN.value, ""),
        (S.SCAN_SOURCE_BIN.value, S.DETECT_ITEMS.value, ""),
        (S.DETECT_ITEMS.value, S.GENERATE_BASELINE_PLAN.value, ""),
        (S.GENERATE_BASELINE_PLAN.value, S.GENERATE_OPTIMIZED_PLAN.value, ""),
        (S.GENERATE_OPTIMIZED_PLAN.value, "COMPARE", "optional"),
        # cut-aware branch (optional, only when cutting is beneficial)
        (S.GENERATE_OPTIMIZED_PLAN.value, S.GENERATE_CUT_ALTERNATIVES.value, "cut?"),
        (S.GENERATE_CUT_ALTERNATIVES.value, S.DIGITAL_TWIN_VALIDATE_CUT_PLAN.value, ""),
        (S.DIGITAL_TWIN_VALIDATE_CUT_PLAN.value, S.WAIT_FOR_CUT_APPROVAL.value, ""),
        (S.WAIT_FOR_CUT_APPROVAL.value, S.CUT_REQUESTED.value, "approve cut"),
        (S.WAIT_FOR_CUT_APPROVAL.value, S.DIGITAL_TWIN_VALIDATE.value, "no cut"),
        (S.CUT_REQUESTED.value, S.CUT_COMPLETED.value, ""),
        (S.CUT_COMPLETED.value, S.REGISTER_DERIVED_ITEMS.value, "completed"),
        (S.CUT_COMPLETED.value, S.DIGITAL_TWIN_VALIDATE.value, "failed -> no cut"),
        (S.REGISTER_DERIVED_ITEMS.value, S.REPLAN_AFTER_CUT.value, ""),
        (S.REPLAN_AFTER_CUT.value, S.DIGITAL_TWIN_VALIDATE.value, ""),
        # inventory-aware container availability branch
        (S.GENERATE_OPTIMIZED_PLAN.value, S.DIGITAL_TWIN_VALIDATE.value, ""),
        (S.DIGITAL_TWIN_VALIDATE.value, S.CHECK_CONTAINER_AVAILABILITY.value, ""),
        (S.CHECK_CONTAINER_AVAILABILITY.value, S.RESERVE_CONTAINER.value, "available"),
        (S.CHECK_CONTAINER_AVAILABILITY.value, S.WAIT_FOR_CONTAINER.value, "shortage"),
        (S.WAIT_FOR_CONTAINER.value, S.RESERVE_CONTAINER.value, "delivered"),
        (S.RESERVE_CONTAINER.value, S.WAIT_FOR_OPERATOR_APPROVAL.value, ""),
        (S.COMPLETE.value, S.COLLECT_FULL_CONTAINER.value, "full"),
        (S.WAIT_FOR_OPERATOR_APPROVAL.value, "APPROVE", "approve"),
        (S.WAIT_FOR_OPERATOR_APPROVAL.value, "REJECT", "reject"),
        ("REJECT", S.REPLAN.value, ""),
        ("APPROVE", S.PICK_ITEM.value, ""),
        (S.PICK_ITEM.value, S.VERIFY_PICK.value, ""),
        (S.VERIFY_PICK.value, S.PLACE_ITEM.value, ""),
        (S.PLACE_ITEM.value, S.VERIFY_PLACEMENT.value, ""),
        (S.VERIFY_PLACEMENT.value, S.UPDATE_CONTAINER_STATE.value, ""),
        (S.UPDATE_CONTAINER_STATE.value, S.NEXT_ITEM.value, ""),
        (S.NEXT_ITEM.value, S.PICK_ITEM.value, "loop"),
        (S.NEXT_ITEM.value, S.COMPLETE.value, "done"),
        (S.PICK_ITEM.value, "DYN", "event"),
        (S.PICK_ITEM.value, "ANOM", "anomaly"),
        ("ANOM", "ANOM_PAUSE", "warning"),
        ("ANOM", "ANOM_HOLD", "critical"),
        ("ANOM_PAUSE", "ACK", ""),
        ("ANOM_HOLD", "ACK", ""),
        ("ACK", S.WAIT_FOR_OPERATOR_APPROVAL.value, "critical -> re-approve"),
        ("ANOM_HOLD", S.REPLAN.value, "if feasibility affected"),
        ("DYN", S.REPLAN.value, ""),
        (S.REPLAN.value, "REAPPROVE", ""),
        ("REAPPROVE", S.WAIT_FOR_OPERATOR_APPROVAL.value, ""),
        ("APPROVE", "PAUSE", "supervise"),
        ("ANOM_HOLD", S.DEGRADED.value, "unrecoverable"),
    ]
    return nodes, edges


def interview_tree() -> Tuple[List, List[Tuple[str, str, str]]]:
    """The simplified whole-process story of brief §20."""
    nodes = [
        ("GEN", "Generate waste task", "seeded scenario", "action"),
        ("INV", "Read container inventory", "FIWARE-backed", "action"),
        ("OPT", "No-cut & cut-aware optimize", "whole-process", "action"),
        ("DTV", "Digital Twin validation", "INDEPENDENT", "validate"),
        ("CUT", "Human cut decision", "SEPARATE approval", "gate"),
        ("SKILL", "Optional cutting skill", "SIMULATED", "hold"),
        ("RE", "Re-plan on actual segments", "renewed", "action"),
        ("APP", "Human packing approval", "the gate", "gate"),
        ("RES", "Reserve / deliver container", "SIMULATED logistics", "action"),
        ("PACK", "Pack", "SIMULATED robot", "loop"),
        ("COLL", "Collect full container", "SIMULATED", "loop"),
        ("FA", "FIWARE analytics", "auditable state", "done"),
        ("ANOM", "Anomaly Monitoring", "SIMULATED, hold", "hold"),
    ]
    edges = [
        ("GEN", "INV", ""), ("INV", "OPT", ""), ("OPT", "DTV", ""),
        ("DTV", "CUT", ""), ("CUT", "SKILL", "cut"), ("CUT", "APP", "no cut"),
        ("SKILL", "RE", ""), ("RE", "APP", ""), ("APP", "RES", "approve"),
        ("RES", "PACK", ""), ("PACK", "COLL", "full"), ("COLL", "FA", ""),
        ("PACK", "ANOM", "on event"), ("ANOM", "APP", "re-approve"),
    ]
    return nodes, edges


def render(nodes, edges, path: str, title: str, cols: int) -> str:
    """Grid layout in the node's declared order; curved edges under the boxes."""
    NW, NH, GAPX, GAPY = 210, 46, 40, 34
    pos: Dict[str, Tuple[float, float]] = {}
    for i, (nid, *_rest) in enumerate(nodes):
        r, c = divmod(i, cols)
        pos[nid] = (30 + c * (NW + GAPX) + NW / 2, 70 + r * (NH + GAPY) + NH / 2)
    rows = (len(nodes) + cols - 1) // cols
    W = 30 + cols * (NW + GAPX) + 10
    H = 70 + rows * (NH + GAPY) + 60

    P: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="ui-sans-serif,system-ui,Arial">',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        f'<text x="24" y="34" font-size="17" font-weight="700" fill="{INK}">{esc(title)}</text>',
    ]
    # edges first (under nodes)
    for a, b, lbl in edges:
        if a not in pos or b not in pos:
            continue
        (ax, ay), (bx, by) = pos[a], pos[b]
        a1 = ay + (NH / 2 if by >= ay else -NH / 2)
        b1 = by + (-NH / 2 if by >= ay else NH / 2)
        mid = (a1 + b1) / 2
        P.append(f'<path d="M{ax},{a1} C{ax},{mid} {bx},{mid} {bx},{b1}" '
                 f'fill="none" stroke="{MUT}" stroke-width="1.2" opacity="0.75"/>')
        if lbl:
            P.append(f'<text x="{(ax+bx)/2:.0f}" y="{mid:.0f}" font-size="8.5" '
                     f'fill="{MUT}" text-anchor="middle">{esc(lbl)}</text>')
    # nodes
    for nid, label, sub, kind in nodes:
        cx, cy = pos[nid]
        fill, stroke = KIND.get(kind, KIND["action"])
        x, y = cx - NW / 2, cy - NH / 2
        # data-node carries the canonical id (Stage name where it maps), so a
        # test can assert the node exists without depending on the display label.
        P.append(f'<rect data-node="{esc(nid)}" x="{x:.0f}" y="{y:.0f}" '
                 f'width="{NW}" height="{NH}" rx="7" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        P.append(f'<text x="{cx:.0f}" y="{y+19:.0f}" font-size="11.5" font-weight="700" '
                 f'text-anchor="middle" fill="{INK}">{esc(label)}</text>')
        if sub:
            P.append(f'<text x="{cx:.0f}" y="{y+34:.0f}" font-size="8.5" '
                     f'text-anchor="middle" fill="{MUT}">{esc(sub)}</text>')
    P.append(f'<text x="24" y="{H-30}" font-size="10" font-weight="700" '
             f'fill="{stroke_invariant()}">CORE INVARIANT</text>')
    P.append(f'<text x="24" y="{H-16}" font-size="10" fill="{INK}">{esc(INVARIANT)}</text>')
    P.append("</svg>")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(P) + "\n")
    return path


def stroke_invariant() -> str:
    return "#c0392b"


def to_png(svg_path: str) -> Optional[str]:
    try:
        import cairosvg
    except ImportError:
        return None
    png = svg_path.replace(".svg", ".png")
    cairosvg.svg2png(url=svg_path, write_to=png, output_width=1400)
    return png


def main() -> int:
    written: List[str] = []
    fn, fe = full_tree()
    svg = render(fn, fe, os.path.join(OUT, "wisepack_behaviour_tree.svg"),
                 "WISEPACK Behaviour Tree — engineering view", cols=4)
    written.append(svg)
    png = to_png(svg)
    if png:
        written.append(png)

    iv_n, iv_e = interview_tree()
    svg2 = render(iv_n, iv_e,
                  os.path.join(OUT, "wisepack_behaviour_tree_interview.svg"),
                  "WISEPACK Behaviour Tree — interview view", cols=5)
    written.append(svg2)
    png2 = to_png(svg2)
    if png2:
        written.append(png2)

    print(f"wrote {len(written)} file(s):")
    for p in written:
        size = os.path.getsize(p)
        print(f"  {os.path.relpath(p, REPO)}  ({size/1024:.0f} kB)")
    if not png:
        print("  (PNG skipped: pip install cairosvg to also emit .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
