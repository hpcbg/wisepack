"""WISEPACK demo bring-up: orchestrator + perception simulator + Digital Twin.

Three processes, real DDS traffic between them:

    perception_sim  --/wisepack/waste/detected_count-->  (observers)
    orchestrator    --/wisepack/plan/geometry--------->  twin_validator
    twin_validator  --/wisepack/plan/status_json------>  (observers, Orion-LD)

The Digital Twin validator runs in its OWN process on purpose: it re-derives
every placement from JSON that crossed DDS, so it can genuinely disagree with the
optimizer rather than share its assumptions.

Launch-file shape follows HARMONY's complete.launch.py and TEMPO's
demo_modbus.launch.py: declared arguments, one Node per component, parameters
passed inline.

    ros2 launch wisepack_bringup demo.launch.py
    ros2 launch wisepack_bringup demo.launch.py preset:=curated_volume_reduction
    ros2 launch wisepack_bringup demo.launch.py auto_approve:=true      # headless

EXECUTION BACKEND. `execution_backend:=isaac` hands every approved placement to
NVIDIA Isaac Sim instead of resolving it with the simulated robot model:

    ros2 launch wisepack_bringup demo.launch.py execution_backend:=isaac

That is a different axis from where the DASHBOARD reads state (sim / ros /
fiware). Isaac is not another data source — it is who moves the item. Isaac Sim
itself runs on the HOST in its own bundled Python, not from this launch file and
not inside the WISEPACK container; see scripts/run_wisepack_isaac.sh.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    EnvironmentVariable, LaunchConfiguration, PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    preset = LaunchConfiguration("preset")
    seed = LaunchConfiguration("seed")
    strategy = LaunchConfiguration("strategy")
    auto_approve = LaunchConfiguration("auto_approve")
    dynamic_events = LaunchConfiguration("dynamic_events")
    results_dir = LaunchConfiguration("results_dir")
    with_perception = LaunchConfiguration("with_perception")
    with_twin = LaunchConfiguration("with_twin")
    execution_backend = LaunchConfiguration("execution_backend")
    robot = LaunchConfiguration("robot")

    # A LaunchConfiguration substitutes to a string; these parameters are
    # declared with non-string types, so state the type or startup is rejected.
    seed_param = ParameterValue(seed, value_type=int)
    auto_param = ParameterValue(auto_approve, value_type=bool)
    dyn_param = ParameterValue(dynamic_events, value_type=bool)

    args = [
        DeclareLaunchArgument("preset", default_value="mixed_pipes_dense",
                              description="Scenario preset from wisepack_core"),
        DeclareLaunchArgument("seed", default_value="42",
                              description="Deterministic random seed"),
        DeclareLaunchArgument("strategy", default_value="max_density",
                              description="max_density | retrievability | segregation"),
        DeclareLaunchArgument(
            "auto_approve", default_value="false",
            description=("Approve without an operator. ONLY for headless "
                         "acceptance runs; the audit trail attributes such an "
                         "approval to `system`, never to an operator.")),
        DeclareLaunchArgument("dynamic_events", default_value="true",
                              description="Run the scripted late-arrival event"),
        DeclareLaunchArgument("results_dir", default_value="results",
                              description="Where run artefacts are written"),
        DeclareLaunchArgument("with_perception", default_value="true"),
        DeclareLaunchArgument("with_twin", default_value="true"),
        DeclareLaunchArgument(
            "execution_backend", default_value="simulated",
            description=("Who performs the approved placements: `simulated` "
                         "(the seeded robot model, the default and unchanged) "
                         "or `isaac` (the selected manipulator in NVIDIA Isaac "
                         "Sim on the host, with physical release and PhysX "
                         "settling). This is NOT the dashboard data source.")),
        DeclareLaunchArgument(
            "robot", default_value="",
            description=("Which robot the Isaac backend executes with, by id "
                         "from config/isaac_robots.yaml (xarm7 | panda). Empty "
                         "resolves it: WISEPACK_ISAAC_ROBOT wins over the "
                         "configured default. Ignored by the simulated "
                         "backend, which has no robot.")),
        DeclareLaunchArgument("with_anomaly", default_value="true",
                              description="Start the SIMULATED anomaly source"),
        DeclareLaunchArgument(
            "perception_source",
            # The environment is the DEFAULT, not an override: an explicit
            # `perception_source:=` on the command line still wins, and with
            # neither set the value is `sim` exactly as before.
            default_value=EnvironmentVariable("WISEPACK_PERCEPTION_SOURCE",
                                              default_value="sim"),
            description=("Where the OBJECT OBSERVATIONS come from: `sim` (the "
                         "default and unchanged simulated detector) or "
                         "`harmony_camera` (a real camera through the HARMONY "
                         "Faster R-CNN detector). Empty resolves it from "
                         "WISEPACK_PERCEPTION_SOURCE, which defaults to `sim`. "
                         "This is a THIRD axis: it is neither the dashboard "
                         "data source nor the execution backend, and every "
                         "combination of the three is legal.")),
    ]
    #: The perception SIMULATOR must not run beside a real detector. Two
    #: perception writers on `/wisepack/waste/detected_count` would make the
    #: reported count depend on which arrived last — and one of them would be a
    #: seeded draw over ground truth presented beside real measurements.
    simulated_perception = PythonExpression(
        ["'", LaunchConfiguration("perception_source"), "' == 'sim'"])
    with_anomaly = LaunchConfiguration("with_anomaly")
    perception_source = LaunchConfiguration("perception_source")

    orchestrator = Node(
        package="wisepack_orchestration",
        executable="hitl_orchestrator",
        name="wisepack_hitl_orchestrator",
        parameters=[{
            "preset": preset,
            "seed": seed_param,
            "strategy": strategy,
            "auto_approve": auto_param,
            "dynamic_events": dyn_param,
            "results_dir": results_dir,
            "tick_period_s": 0.7,
            "execution_backend": execution_backend,
            "robot": robot,
            "perception_source": perception_source,
        }],
        output="screen",
    )

    perception = Node(
        package="wisepack_sim",
        executable="perception_sim",
        name="wisepack_perception_sim",
        parameters=[{"seed": seed_param}],
        output="screen",
        # Started only when perception is SIMULATED. With a real perception
        # source the detector is the authority and this node is not launched at
        # all — see `simulated_perception` above.
        condition=IfCondition(PythonExpression(
            [with_perception, " and ", simulated_perception])),
    )

    twin = Node(
        package="wisepack_optimization",
        executable="twin_validator",
        name="wisepack_twin_validator",
        # Must match the orchestrator's validation config, or the two would
        # legitimately disagree for a reason that is not a bug and the
        # disagreement signal would be worthless.
        parameters=[{"min_support_fraction": 0.70, "min_clearance_mm": 0}],
        output="screen",
        condition=IfCondition(with_twin),
    )

    # SIMULATED anomaly source. On-demand only (auto_interval_s=0):
    # it publishes when the operator injects, never autonomously.
    anomaly = Node(
        package="wisepack_anomaly",
        executable="anomaly_simulator",
        name="wisepack_anomaly_simulator",
        parameters=[{"auto_interval_s": 0.0}],
        output="screen",
        condition=IfCondition(with_anomaly),
    )

    return LaunchDescription(args + [orchestrator, perception, twin, anomaly])
