"""wisepack_core — the ROS-free WISEPACK domain core.

Import this package to get the domain model, the deterministic task generator,
both packing algorithms, the independent placement validator, the KPI
calculations and the workflow engine. Nothing here imports rclpy, so the same
code runs inside a ROS 2 node, inside the FastAPI dashboard, and inside a plain
``python3`` on a laptop with nothing installed.

    from wisepack_core import build_scenario, pack_baseline, pack_optimized
    scenario  = build_scenario("mixed_pipes_dense", seed=42)
    baseline  = pack_baseline(scenario)
    optimized = pack_optimized(scenario)

It is an ament_python package so ROS nodes can depend on it normally, but it has
no ROS build-time or run-time dependency of its own.
"""

from .domain import (                                            # noqa: F401
    SCHEMA_VERSION, ApprovalState, Axis, Box, Container, ContainerStatus,
    DomainError, GeometryType, ItemStatus, PackingPlan, Placement, Scenario,
    Source, Strategy, ValidationStatus, Vec3, WasteItem,
)
from .events import (                                            # noqa: F401
    ActionEvent, ActionLog, Actor, DynamicEvent, DynamicEventType, Result,
    Stage, Stopwatch, utc_now_iso,
)
from .execution import (                                         # noqa: F401
    ExecutionBackend, ISAAC_STATE_STAGE, parse_backend,
    robot_state_for_isaac_state, stage_for_isaac_state,
)
from .isaac_contract import (                                    # noqa: F401
    ContractError, Dimensions, IsaacCommand, IsaacCommandType, IsaacFeedback,
    IsaacState, Pose, RunGate,
)
from .isaac_transform import (                                   # noqa: F401
    DEFAULT_LAYOUT, SceneLayout, check_containment, placement_pose,
    pose_to_world, table_pose_for_index, world_to_pose,
)
from .generator import (                                         # noqa: F401
    CONTAINER_SPECS, GeneratorConfig, ISAAC_SMOKE_PRESET, MATERIALS, PRESETS,
    build_curated_scenario, build_scenario, generate_scenario, inject_item,
    make_container, preset_config,
)
from .kpi import (                                               # noqa: F401
    ExecutionStats, KPIReport, Metric, PROPOSAL_TARGETS, compare_strategies,
    compute_kpis, cut_recommendations, volume_requirement_reduction_pct,
)
from .packing import (                                           # noqa: F401
    BASELINE_ALGORITHM, OPTIMIZED_ALGORITHM, ObjectiveWeights, OptimizerConfig,
    STRATEGY_WEIGHTS, pack_baseline, pack_optimized, score_plan, select_plan,
)
from .validator import (                                         # noqa: F401
    DEFAULT_VALIDATION, PlacementValidator, ValidationConfig, ValidationReport,
    Violation,
)
from .anomaly import (                                           # noqa: F401
    AnomalyClass, AnomalyEvent, Reaction, Severity, SIMULATED_LABEL,
)
from .workflow import (                                          # noqa: F401
    AnomalyHold, ApprovalRequired, RobotSimConfig, WorkflowConfig,
    WorkflowEngine, WorkflowError, run_headless,
)

__version__ = "0.1.0"
