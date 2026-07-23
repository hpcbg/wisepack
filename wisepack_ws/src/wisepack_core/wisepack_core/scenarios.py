"""YAML scenario configuration — optional layer over the built-in presets.

Adapted from HARVEST's ``config.yaml`` approach: a single declarative file
carries the generator parameters and a list of deterministic dynamic events, so
a scenario can be changed without touching Python.

Two deliberate differences from HARVEST:

  * YAML is OPTIONAL. Every preset is defined in generator.py and the acceptance
    demo runs with no config file present. A demo that cannot start because a
    config file is missing is a demo that will fail on the reviewer's machine.
  * Dynamic-event triggers are stage- or placement-indexed rather than
    wall-clock timestamped (HARVEST uses simulated clock times, which suits a
    day-long farm simulation and not a 30-second packing cycle). See
    events.DynamicEvent for the three accepted trigger forms.

``pyyaml`` is imported lazily so that wisepack_core keeps working — presets and
all — on a machine without it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .domain import Scenario, Strategy
from .events import DynamicEvent
from .generator import PRESETS, build_scenario, preset_config, generate_scenario
from .workflow import RobotSimConfig, WorkflowConfig
from .packing import OptimizerConfig
from .validator import ValidationConfig

DEFAULT_CONFIG_DIR = os.environ.get("WISEPACK_CONFIG_DIR", "config")


class ConfigError(ValueError):
    """Raised when a scenario configuration file is malformed."""


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml                                   # noqa: PLC0415
    except ImportError as exc:                        # pragma: no cover
        raise ConfigError(
            f"cannot read {path}: pyyaml is not installed. The built-in presets "
            f"({', '.join(sorted(PRESETS))}) work without it.") from exc
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def load_config(path: str) -> Dict[str, Any]:
    """Load a .yaml or .json scenario configuration."""
    if not os.path.exists(path):
        raise ConfigError(f"configuration file not found: {path}")
    if path.endswith((".yaml", ".yml")):
        return _load_yaml(path)
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    raise ConfigError(f"unsupported configuration format: {path}")


# --------------------------------------------------------------------------- #
# Building workflow configuration from a document
# --------------------------------------------------------------------------- #


def _dynamic_events(raw: Any) -> List[DynamicEvent]:
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ConfigError("dynamic_events must be a list")
    events: List[DynamicEvent] = []
    for n, entry in enumerate(raw, 1):
        if not isinstance(entry, dict):
            raise ConfigError(f"dynamic_events[{n}] must be a mapping")
        try:
            events.append(DynamicEvent.from_dict(entry))
        except (KeyError, ValueError) as exc:
            raise ConfigError(f"dynamic_events[{n}]: {exc}") from exc
    return events


def workflow_config_from_doc(doc: Dict[str, Any]) -> WorkflowConfig:
    """Translate a configuration document into a WorkflowConfig.

    Unknown keys are rejected rather than ignored: a silently-ignored typo in a
    scenario file produces a run that looks right and measures something else.
    """
    known = {"scenario", "generator", "optimizer", "robot", "validation",
             "dynamic_events", "auto_approve", "description"}
    unknown = set(doc) - known
    if unknown:
        raise ConfigError(
            f"unknown top-level keys: {sorted(unknown)}. Known: {sorted(known)}")

    scenario_block = doc.get("scenario", {}) or {}
    preset = scenario_block.get("preset", "mixed_pipes_small")
    if preset not in PRESETS:
        raise ConfigError(
            f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
    seed = int(scenario_block.get("seed", 42))
    strategy = Strategy(scenario_block.get("strategy", "max_density"))

    gen = dict(doc.get("generator", {}) or {})
    for key in ("length_range_mm", "diameter_range_mm", "wall_fraction_range"):
        if key in gen and isinstance(gen[key], list):
            gen[key] = tuple(gen[key])

    opt_block = doc.get("optimizer", {}) or {}
    optimizer = OptimizerConfig(
        strategy=strategy,
        restarts=int(opt_block.get("restarts", 6)),
        time_budget_ms=float(opt_block.get("time_budget_ms", 4000.0)),
        improve=bool(opt_block.get("improve", True)),
        improve_rounds=int(opt_block.get("improve_rounds", 3)),
        seed=seed)

    rb = doc.get("robot", {}) or {}
    robot = RobotSimConfig(
        pick_failure_probability=float(rb.get("pick_failure_probability", 0.08)),
        max_pick_retries=int(rb.get("max_pick_retries", 2)),
        pick_duration_ms=float(rb.get("pick_duration_ms", 900.0)),
        place_duration_ms=float(rb.get("place_duration_ms", 1100.0)),
        travel_duration_ms=float(rb.get("travel_duration_ms", 600.0)),
        detection_confidence_min=float(rb.get("detection_confidence_min", 0.82)),
        detection_confidence_max=float(rb.get("detection_confidence_max", 0.99)),
        detection_miss_probability=float(rb.get("detection_miss_probability", 0.0)),
        seed=int(rb.get("seed", seed)))

    vb = doc.get("validation", {}) or {}
    validation = ValidationConfig(
        min_support_fraction=float(vb.get("min_support_fraction", 0.70)),
        min_clearance_mm=int(vb.get("min_clearance_mm", 0)))

    return WorkflowConfig(
        preset=preset,
        seed=seed,
        strategy=strategy,
        optimizer=optimizer,
        robot=robot,
        validation=validation,
        dynamic_events=_dynamic_events(doc.get("dynamic_events")),
        auto_approve=bool(doc.get("auto_approve", False)),
        generator_overrides=gen)


def load_workflow_config(path: str) -> WorkflowConfig:
    return workflow_config_from_doc(load_config(path))


def scenario_from_doc(doc: Dict[str, Any]) -> Scenario:
    """Build just the Scenario described by a configuration document."""
    cfg = workflow_config_from_doc(doc)
    scenario = build_scenario(cfg.preset, cfg.seed, **cfg.generator_overrides)
    scenario.dynamic_events = [e.to_dict() for e in cfg.dynamic_events]
    return scenario


def find_scenario_file(name: str,
                       config_dir: str = DEFAULT_CONFIG_DIR) -> Optional[str]:
    """Locate ``config/scenarios/<name>.yaml`` by preset name."""
    for ext in ("yaml", "yml", "json"):
        candidate = os.path.join(config_dir, "scenarios", f"{name}.{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def list_scenario_files(config_dir: str = DEFAULT_CONFIG_DIR) -> List[str]:
    folder = os.path.join(config_dir, "scenarios")
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.endswith((".yaml", ".yml", ".json")))


def load_or_preset(name: str, seed: Optional[int] = None,
                   config_dir: str = DEFAULT_CONFIG_DIR) -> WorkflowConfig:
    """Preferred entry point: YAML if present, built-in preset otherwise.

    This is what makes the acceptance demo robust — the config directory is a
    convenience, never a dependency.
    """
    path = find_scenario_file(name, config_dir)
    if path:
        cfg = load_workflow_config(path)
        if seed is not None:
            cfg.seed = seed
            cfg.optimizer.seed = seed
            cfg.robot.seed = seed
        return cfg
    if name not in PRESETS:
        raise ConfigError(
            f"no scenario file for {name!r} in {config_dir}/scenarios and no "
            f"built-in preset of that name; known presets: {sorted(PRESETS)}")
    return WorkflowConfig(preset=name, seed=seed if seed is not None else 42)


def save_scenario(scenario: Scenario, path: str) -> str:
    """Persist a generated scenario as JSON for later replay."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scenario.to_dict(), fh, indent=2, default=str)
        fh.write("\n")
    return path


def load_scenario(path: str) -> Scenario:
    """Replay a previously saved scenario exactly."""
    with open(path, encoding="utf-8") as fh:
        return Scenario.from_dict(json.load(fh))


__all__ = [
    "ConfigError", "DEFAULT_CONFIG_DIR", "load_config", "load_workflow_config",
    "workflow_config_from_doc", "scenario_from_doc", "find_scenario_file",
    "list_scenario_files", "load_or_preset", "save_scenario", "load_scenario",
]
