"""Launch the PLANAR empowerment monitor and (optionally) the greedy controller.

This is the 2-link arm with ground-truth outcomes -- no Gazebo, no camera,
runs in seconds. It is the place to check the estimator is alive before
bringing up the full stack. For the UR5 use gazebo_ur5.launch.py.

    ros2 launch intrinsic_motivation_ros empowerment.launch.py
    ros2 launch intrinsic_motivation_ros empowerment.launch.py control:=true
    ros2 launch intrinsic_motivation_ros empowerment.launch.py control:=true task_bonus:=2.0

The third form is the instinct-plus-information condition: empowerment
finds the object, the bonus supplies the reason to finish.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("intrinsic_motivation_ros")
    # NOT empowerment.yaml -- that one is UR5-shaped and would override
    # joint_names with six joints these nodes never see.
    params = os.path.join(share, "config", "empowerment_planar.yaml")

    control = LaunchConfiguration("control")
    task_bonus = LaunchConfiguration("task_bonus")
    bolted = LaunchConfiguration("bolted")

    return LaunchDescription([
        DeclareLaunchArgument(
            "control", default_value="false",
            description="Also run the greedy empowerment-ascent controller.",
        ),
        DeclareLaunchArgument(
            "task_bonus", default_value="0.0",
            description="Weight on the task term. 0.0 is pure empowerment.",
        ),
        DeclareLaunchArgument(
            "bolted", default_value="1.0",
            description="Object mass factor. 1.0 pushable, 0.0 bolted (control).",
        ),

        Node(
            package="intrinsic_motivation_ros",
            executable="empowerment_node_planar",
            name="empowerment_node",
            output="screen",
            parameters=[params, {
                "object_mass_factor": ParameterValue(bolted, value_type=float),
            }],
        ),

        Node(
            package="intrinsic_motivation_ros",
            executable="greedy_climber_planar",
            name="greedy_climber",
            output="screen",
            condition=IfCondition(control),
            parameters=[params, {
                "task_bonus_weight": ParameterValue(task_bonus, value_type=float),
            }],
        ),
    ])
