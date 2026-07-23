"""Timestamped validation artefacts under results/.

Every run writes a fixed set of files with a shared timestamp, so a reviewer can
pick one stamp and get the whole picture: what was generated, what was planned,
what was executed, what was measured, and whether it validated.

Filenames follow ``wisepack-<kind>-<YYYYmmdd-HHMMSS>.<ext>``, matching TEMPO's
``tempo-dds-latency-*.json`` and HARMONY's ``harmony-dds-latency-*`` convention
so the same "newest artefact wins" lookup works across all three repositories.

Nothing here invents a value. If a KPI was not measured it is written as null in
JSON and as an empty cell in CSV, never as zero.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .domain import PackingPlan, Scenario
from .events import ActionLog
from .kpi import KPIReport

DEFAULT_RESULTS_DIR = os.environ.get("WISEPACK_RESULTS_DIR", "results")


def timestamp() -> str:
    """Local-time stamp used in every filename of one run."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _csv(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str, sort_keys=False) + "\n"


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


@dataclass
class ArtifactSet:
    """Paths written for one run."""

    stamp: str
    paths: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.stamp, "files": dict(self.paths)}

    def summary(self) -> str:
        lines = [f"artefacts for {self.stamp}:"]
        lines += [f"  {kind:12s} {path}" for kind, path in sorted(self.paths.items())]
        return "\n".join(lines)


def placement_csv_rows(plan: PackingPlan, scenario: Scenario):
    header = ["plan_id", "algorithm", "strategy", "placement_order", "item_id",
              "container_id", "geometry_type", "segregation_group", "axis",
              "x_mm", "y_mm", "z_mm", "size_x_mm", "size_y_mm", "size_z_mm",
              "occupied_volume_mm3", "clearance_mm", "validation_status",
              "executed", "is_approximated"]
    rows: List[Sequence[Any]] = []
    for p in plan.ordered_placements:
        item = scenario.item(p.item_id)
        rows.append([
            plan.plan_id, plan.algorithm, plan.strategy.value, p.placement_order,
            p.item_id, p.container_id,
            item.geometry_type.value if item else "",
            item.segregation_group if item else "",
            p.axis.value, p.position.x, p.position.y, p.position.z,
            p.size.x, p.size.y, p.size.z, p.occupied_volume_mm3,
            p.clearance_mm, p.validation_status.value, int(p.executed),
            int(item.is_approximated) if item else 0,
        ])
    return header, rows


def write_run_artifacts(scenario: Scenario, baseline: PackingPlan,
                        optimized: PackingPlan, selected: PackingPlan,
                        kpis: KPIReport, action_log: ActionLog,
                        results_dir: str = DEFAULT_RESULTS_DIR,
                        stamp: Optional[str] = None,
                        extra: Optional[Dict[str, Any]] = None) -> ArtifactSet:
    """Write the full artefact set for one run and return the paths."""
    stamp = stamp or timestamp()
    paths: Dict[str, str] = {}

    def out(kind: str, ext: str) -> str:
        return os.path.join(results_dir, f"wisepack-{kind}-{stamp}.{ext}")

    # -- the run record ------------------------------------------------------ #
    run_doc: Dict[str, Any] = {
        "timestamp": stamp,
        "run_id": kpis.run_id,
        "scenario": scenario.to_dict(),
        "plans": {
            "baseline": baseline.to_dict(),
            "optimized": optimized.to_dict(),
            "selected_plan_id": selected.plan_id,
            "selected_algorithm": selected.algorithm,
        },
        "kpis": kpis.to_dict(),
        "action_events": action_log.count,
        "action_sequence_monotonic": action_log.sequence_is_monotonic()[0],
        "environment": _environment(),
    }
    if extra:
        run_doc.update(extra)
    paths["run"] = _write(out("run", "json"), _json(run_doc))

    # -- actions ------------------------------------------------------------- #
    paths["actions_jsonl"] = _write(out("actions", "jsonl"),
                                    action_log.to_jsonl() + "\n")
    header, rows = action_log.csv_rows()
    paths["actions_csv"] = _write(out("actions", "csv"), _csv(header, rows))

    # -- placements (both plans, so the comparison is reproducible) ---------- #
    b_header, b_rows = placement_csv_rows(baseline, scenario)
    _, o_rows = placement_csv_rows(optimized, scenario)
    paths["placements"] = _write(out("placements", "csv"),
                                 _csv(b_header, list(b_rows) + list(o_rows)))

    # -- scenario ------------------------------------------------------------ #
    s_header, s_rows = scenario.csv_rows()
    paths["scenario_csv"] = _write(out("scenario", "csv"), _csv(s_header, s_rows))
    paths["scenario_json"] = _write(out("scenario", "json"),
                                    _json(scenario.to_dict()))

    # -- KPIs ---------------------------------------------------------------- #
    paths["kpis"] = _write(out("kpis", "json"), _json(kpis.to_dict()))
    k_header, k_rows = kpis.csv_rows()
    paths["kpis_csv"] = _write(out("kpis", "csv"), _csv(k_header, k_rows))

    return ArtifactSet(stamp, paths)


def _environment() -> Dict[str, Any]:
    """Machine facts a reviewer needs to interpret the timings."""
    import platform
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ros_distro": os.environ.get("ROS_DISTRO", "not set (no ROS in this process)"),
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "not set"),
    }


# --------------------------------------------------------------------------- #
# Validation report (markdown)
# --------------------------------------------------------------------------- #


def write_validation_report(scenario: Scenario, baseline: PackingPlan,
                            optimized: PackingPlan, selected: PackingPlan,
                            kpis: KPIReport, action_log: ActionLog,
                            artifacts: ArtifactSet,
                            results_dir: str = DEFAULT_RESULTS_DIR,
                            checks: Optional[List[Dict[str, Any]]] = None) -> str:
    """Human-readable evidence summary. This is what a reviewer opens first."""
    stamp = artifacts.stamp
    path = os.path.join(results_dir, f"wisepack-validation-{stamp}.md")
    monotonic_ok, monotonic_note = action_log.sequence_is_monotonic()

    def fmt(key: str, digits: int = 1) -> str:
        m = kpis.metrics.get(key)
        if m is None or m.value is None:
            return "not measured"
        return f"{m.value:.{digits}f}{(' ' + m.unit) if m.unit else ''}"

    lines: List[str] = [
        f"# WISEPACK validation report — {stamp}",
        "",
        f"- Run id: `{kpis.run_id}`",
        f"- Scenario: `{scenario.scenario_id}` (preset `{scenario.preset}`, "
        f"seed `{scenario.seed}`)",
        f"- Items: {len(scenario.items)}"
        + (f" — {kpis.approximated_items} use conservative bounding-box "
           "approximation" if kpis.approximated_items else ""),
        f"- Curated demonstration dataset: **{'yes' if scenario.curated else 'no'}**",
        "",
        "## Measured packing result",
        "",
        "| Metric | Baseline | Optimized |",
        "|---|---|---|",
        f"| Algorithm | `{baseline.algorithm}` | `{optimized.algorithm}` "
        f"(`{optimized.strategy.value}`) |",
        f"| Containers required | {baseline.containers_required} | "
        f"{optimized.containers_required} |",
        f"| Container utilization | {baseline.utilization_pct:.1f} % | "
        f"{optimized.utilization_pct:.1f} % |",
        f"| Required capacity | {baseline.required_capacity_mm3 / 1e9:.3f} m³ | "
        f"{optimized.required_capacity_mm3 / 1e9:.3f} m³ |",
        f"| Unplaced items | {len(baseline.unplaced_item_ids)} | "
        f"{len(optimized.unplaced_item_ids)} |",
        f"| Computation time | {baseline.computation_time_ms:.1f} ms | "
        f"{optimized.computation_time_ms:.1f} ms |",
        f"| Passes independent validator | "
        f"{'yes' if baseline.is_valid else 'NO'} | "
        f"{'yes' if optimized.is_valid else 'NO'} |",
        "",
        f"**Volume requirement reduction: {fmt('volume_requirement_reduction_pct')}**  ",
        "(denominator = required container capacity = containers × capacity each; "
        "NOT total material volume, which is identical for both algorithms)",
        "",
        f"Selected plan: `{selected.plan_id}` (`{selected.algorithm}`).",
        "",
        "## Proposal KPI assessment",
        "",
        "A proposal target is not a result. Targets this demonstrator cannot "
        "measure are marked `not_applicable` rather than scored.",
        "",
        "| KPI | Target | Measured | Source | Status |",
        "|---|---|---|---|---|",
    ]
    for row in kpis.assess_targets():
        measured = ("—" if row["measured_value"] is None
                    else f"{row['measured_value']:.1f} %")
        lines.append(
            f"| {row['key']} {row['label']} | >{row['target_pct']:.0f} % | "
            f"{measured} | {row['measured_source'] or '—'} | "
            f"`{row['status']}` |")

    lines += [
        "",
        "## Simulated execution",
        "",
        "Every figure in this section is SIMULATED. There is no perception model, "
        "no robot and no physics in this demonstrator.",
        "",
        f"- Simulated pick attempts: {fmt('simulated_pick_attempts', 0)}",
        f"- Simulated pick success rate: {fmt('simulated_pick_success_rate_pct')}",
        f"- Simulated end-to-end success rate: "
        f"{fmt('simulated_end_to_end_success_rate_pct')}",
        f"- Re-plans: {fmt('replans', 0)}",
        f"- Operator interventions: {fmt('operator_interventions', 0)}",
        "",
        "## Audit trail",
        "",
        f"- Action events published: **{action_log.count}**",
        f"- Sequence monotonic and gap-free: "
        f"**{'yes' if monotonic_ok else 'NO'}** ({monotonic_note})",
        f"- DDS → FIWARE latency (p50): {fmt('dds_to_fiware_latency_ms', 2)}",
        "",
        "### Actions by stage",
        "",
        "| Stage | Events | Total duration (ms) |",
        "|---|---|---|",
    ]
    durations = action_log.duration_by_stage_ms()
    for stage, count in sorted(action_log.by_stage().items()):
        lines.append(f"| `{stage}` | {count} | {durations.get(stage, 0.0):.1f} |")

    if checks:
        lines += ["", "## Acceptance checks", "",
                  "| Check | Result | Detail |", "|---|---|---|"]
        for check in checks:
            mark = "✅ pass" if check.get("ok") else "❌ FAIL"
            lines.append(f"| {check.get('name', '')} | {mark} | "
                         f"{check.get('detail', '')} |")

    lines += ["", "## Artefacts", "", "| Kind | File |", "|---|---|"]
    for kind, p in sorted(artifacts.paths.items()):
        lines.append(f"| {kind} | `{os.path.basename(p)}` |")

    env = _environment()
    lines += [
        "", "## Environment", "",
        f"- Python {env['python']} on {env['platform']}",
        f"- {env['cpu_count']} CPUs, ROS_DISTRO={env['ros_distro']}",
        "",
        "Timings are wall-clock on this machine and will differ on other "
        "hardware. Container counts, utilization and validation outcomes are "
        "deterministic and will not.",
        "",
    ]
    return _write(path, "\n".join(lines))


# --------------------------------------------------------------------------- #
# Reading artefacts back
# --------------------------------------------------------------------------- #


def latest_artifact(prefix: str, results_dir: str = DEFAULT_RESULTS_DIR,
                    ext: str = "json") -> Optional[Dict[str, Any]]:
    """Newest ``wisepack-<prefix>-*.<ext>`` parsed as JSON, or None.

    Same pattern as TEMPO's ``load_benchmark_kpi``: the dashboard reports the
    latest measured artefact rather than inventing a live figure, and shows
    "not measured" when no artefact exists.
    """
    try:
        names = [f for f in os.listdir(results_dir)
                 if f.startswith(f"wisepack-{prefix}-") and f.endswith(f".{ext}")]
        if not names:
            return None
        with open(os.path.join(results_dir, sorted(names)[-1]), encoding="utf-8") as fh:
            doc = json.load(fh)
        doc["_file"] = sorted(names)[-1]
        return doc
    except (OSError, ValueError):
        return None


def latest_latency_p50_ms(results_dir: str = DEFAULT_RESULTS_DIR) -> Optional[float]:
    """p50 of the ROS→FIWARE hop from the newest latency artefact, or None."""
    doc = latest_artifact("dds-fiware-latency", results_dir)
    if not doc:
        return None
    summary = doc.get("summary", {})
    for key in ("ros_to_fiware", "loop"):
        block = summary.get(key)
        if isinstance(block, dict) and block.get("median") is not None:
            return float(block["median"])
    return None


__all__ = [
    "DEFAULT_RESULTS_DIR", "timestamp", "ArtifactSet", "write_run_artifacts",
    "write_validation_report", "latest_artifact", "latest_latency_p50_ms",
    "placement_csv_rows",
]
