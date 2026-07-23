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
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
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
    ]

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
        }],
        output="screen",
    )

    perception = Node(
        package="wisepack_sim",
        executable="perception_sim",
        name="wisepack_perception_sim",
        parameters=[{"seed": seed_param}],
        output="screen",
        condition=IfCondition(with_perception),
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

    return LaunchDescription(args + [orchestrator, perception, twin])
