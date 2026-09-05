"""
Full stack: Gazebo (gz-sim) + UR5 + overhead depth camera + empowerment.

    # world, arm and camera only, to check the scene loads
    ros2 launch intrinsic_motivation_ros gazebo_ur5.launch.py

    # headless (no Gazebo GUI), e.g. over ssh
    ros2 launch intrinsic_motivation_ros gazebo_ur5.launch.py gui:=false

    # add the empowerment monitor
    ros2 launch intrinsic_motivation_ros gazebo_ur5.launch.py empowerment:=true

    # let it drive the arm (no goal, no reward)
    ros2 launch intrinsic_motivation_ros gazebo_ur5.launch.py \
        empowerment:=true control:=true

    # control condition: agent's model says the puck cannot be moved
    ros2 launch intrinsic_motivation_ros gazebo_ur5.launch.py \
        empowerment:=true control:=true bolted:=0.0

    # instinct + information: add the one bit that empowerment cannot supply
    ros2 launch intrinsic_motivation_ros gazebo_ur5.launch.py \
        empowerment:=true control:=true task_bonus:=2.0

Requires ros_gz_sim, ros_gz_bridge, ur_description, xacro,
robot_state_publisher -- all of which ship with a normal desktop install
plus:

    sudo apt install ros-$ROS_DISTRO-ros-gz ros-$ROS_DISTRO-ur-description

There is deliberately no ros2_control here. See urdf/ur5_gz.urdf.xacro for
why: the arm is driven by gz-sim's own JointTrajectoryController, so the ROS
interface is still trajectory_msgs/JointTrajectory and nothing about the
nodes changes when this is pointed at a real UR5.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

WORLD_NAME = "empowerment_table"     # must match <world name=...> in the sdf
ROBOT_NAME = "ur5"                   # must match the gz topic names in gz_bridge.yaml

# Gazebo Harmonic. This becomes `gz sim --force-version 8`, so a machine with
# an older Gazebo also installed cannot pick that one up by accident. The
# world's gz-sim-*-system plugin names and the gz.msgs.* types in
# config/gz_bridge.yaml are Harmonic's; changing this number alone is not
# enough to run anything else.
GZ_VERSION = "8"

# Where the UR5 bolts on, and the origin of the frame that
# config/empowerment.yaml is written in.
#
# Both tables are the same height (WORK_HEIGHT); the arm gets its extra
# 0.25 m from robot_riser, a block on the robot table. That clearance is
# required: SDF cannot state initial joint positions, so Gazebo spawns the
# arm at all zeros, and a UR5 at all zeros lies horizontally 0.817 m out with
# the wrist 5 mm below its own base plane. Level with a bench inside 0.87 m
# and it spawns inside the bench and jams. See the header comment in
# worlds/empowerment_table.sdf.
TABLE_HEIGHT = 1.00          # top of robot_riser: where the UR5 bolts on
WORK_HEIGHT = 0.75          # object_table top; table_height in the yaml is
                            # WORK_HEIGHT - TABLE_HEIGHT = -0.25

# Yaw of the arm on its pedestal, radians.
#
# intrinsic_core's DH convention is 180 degrees about z from
# ur_description's URDF: at the same joint angles the analytic model puts the
# tool at (-0.631, -0.109, 0.323) in the base frame while Gazebo puts it at
# (+0.631, +0.109, 0.323). Exactly negated in x and y. Spawn the arm
# unrotated and the simulated arm reaches away from the object table while
# the agent plans toward it, so the climber drives confidently in the wrong
# direction and nothing ever touches the puck.
#
# With this yaw the two agree, and the agent's frame becomes the world frame
# raised by TABLE_HEIGHT with no rotation at all. Every coordinate in
# config/empowerment.yaml is then a plain world x,y.
BASE_YAW = 3.14159265

# The camera's <pose> in worlds/empowerment_table.sdf. Repeated here because
# the TF has to match it.
#
# THREE places carry this camera and all three have to agree: the <pose> and
# <horizontal_fov> in the world SDF, this TF, and camera_position /
# camera_look_at / camera_fov_deg in config/empowerment.yaml. They have
# drifted apart before -- the world said 60 deg while the agent's model said
# 30 -- and the symptom is not an error, it is an agent predicting through
# optics it does not actually have. Change one, change all three.
#
# Oblique rather than overhead, at 36 deg below horizontal looking back along
# +x toward the arm, so the arm stops occluding the objects it reaches for.
# See the comment on the camera model in the world file.
CAMERA_XYZ = (-1.45, 0.0, WORK_HEIGHT + 0.60)
CAMERA_PITCH = 0.6283


def generate_launch_description():
    pkg = get_package_share_directory("intrinsic_motivation_ros")
    world = os.path.join(pkg, "worlds", f"{WORLD_NAME}.sdf")
    xacro_file = os.path.join(pkg, "urdf", "ur5_gz.urdf.xacro")
    initial_positions = os.path.join(pkg, "config", "initial_positions.yaml")
    bridge_cfg = os.path.join(pkg, "config", "gz_bridge.yaml")
    params = os.path.join(pkg, "config", "empowerment.yaml")
    rviz_cfg = os.path.join(pkg, "rviz", "empowerment.rviz")
    gui_cfg = os.path.join(pkg, "gui", "empowerment_gui.config")

    empowerment = LaunchConfiguration("empowerment")
    control = LaunchConfiguration("control")
    task_bonus = LaunchConfiguration("task_bonus")
    bolted = LaunchConfiguration("bolted")
    use_rviz = LaunchConfiguration("rviz")
    gui = LaunchConfiguration("gui")
    drift = LaunchConfiguration("drift")
    use_eval = LaunchConfiguration("eval")
    use_grasp = LaunchConfiguration("grasp")
    objective = LaunchConfiguration("objective")

    args = [
        DeclareLaunchArgument("empowerment", default_value="false",
                              description="Run the empowerment monitor node."),
        DeclareLaunchArgument("control", default_value="false",
                              description="Let empowerment drive the arm."),
        DeclareLaunchArgument("task_bonus", default_value="0.0",
                              description="Weight on the task term. 0.0 = pure empowerment."),
        DeclareLaunchArgument("bolted", default_value="1.0",
                              description="Agent's puck mass factor. 1.0 pushable, 0.0 bolted."),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("gui", default_value="true",
                              description="Show the Gazebo GUI. false = server only."),
        DeclareLaunchArgument("grasp", default_value="false",
                              description="Run grasp_climber: approach the object "
                                          "with plain motion, then let TRANSFER "
                                          "empowerment decide, which picks it up. "
                                          "Use objective:=own_sensor to run the "
                                          "objective that refuses to."),
        DeclareLaunchArgument("objective", default_value="transfer",
                              description="transfer | own_sensor"),
        DeclareLaunchArgument("eval", default_value="false",
                              description="Run detector_eval, which scores the "
                                          "agent's object estimates against the "
                                          "simulator's ground truth. It is a "
                                          "separate process precisely so that "
                                          "no ground truth is reachable from "
                                          "the agent's own nodes."),
        DeclareLaunchArgument("drift", default_value="true",
                              description="Drive the drifter's back-and-forth. "
                                          "false leaves it parked, which turns it "
                                          "into a second static object and is the "
                                          "control for the noisy-TV comparison."),
        DeclareLaunchArgument("spawn_delay", default_value="10.0",
                              description="Seconds to wait for Gazebo before spawning the arm. "
                                          "Raise it if spawn_ur5 reports a service timeout."),
    ]

    # gz-sim resolves package:// and model:// against this. The UR5 meshes
    # come through as absolute file:// paths (ur_macro's force_abs_paths), so
    # this is belt and braces, but it costs nothing and saves an afternoon
    # when a mesh path changes.
    share_dirs = [
        os.path.join(prefix, "share")
        for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
        if prefix
    ]
    resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.pathsep.join(filter(None, share_dirs + [
            os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
        ])),
    )

    # --- Gazebo ------------------------------------------------------------
    # -r start unpaused, -v 3 warnings and up, -s server only when gui:=false.
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py",
        ])),
        launch_arguments={
            # --gui-config frames the camera on the workcell; Gazebo's stock
            # camera is 6 m back and 6 m up and makes this scene unreadable.
            "gz_args": ["-r -v 3 --gui-config ", gui_cfg, " ", world],
            "gz_version": GZ_VERSION,
            "on_exit_shutdown": "true",
        }.items(),
        condition=IfCondition(gui),
    )
    gz_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py",
        ])),
        launch_arguments={
            "gz_args": ["-s -r -v 3 ", world],
            "gz_version": GZ_VERSION,
            "on_exit_shutdown": "true",
        }.items(),
        condition=UnlessCondition(gui),
    )

    # --- UR5 description ---------------------------------------------------
    # Kept as a bare Command so it can be handed to BOTH robot_state_publisher
    # (wrapped in ParameterValue, below) and ros_gz_sim create (as -string).
    # create used to read the URDF off /robot_description instead, which is a
    # latched transient-local topic: as a late joiner it has to be given the
    # historical sample at discovery, and when the machine is busy loading
    # Gazebo that hand-off is missed. create has no retry, so it sat printing
    # "Waiting messages on topic [robot_description]" forever and the world
    # came up with no arm in it. Passing the XML directly has no race.
    robot_description_cmd = Command([
            "xacro ", xacro_file,
            " name:=", ROBOT_NAME,
            " ur_type:=ur5",
            # quoted: launch shlex-splits this command line, and an
            # unquoted "0 0 0.75" would reach xacro as three arguments
            f" base_xyz:='0 0 {TABLE_HEIGHT}'",
            f" base_rpy:='0 0 {BASE_YAW}'",
            " initial_positions_file:=", initial_positions,
            f" trajectory_topic:=/{ROBOT_NAME}/joint_trajectory",
            f" joint_state_topic:=/{ROBOT_NAME}/joint_state",
    ])
    robot_description = ParameterValue(robot_description_cmd, value_type=str)

    # Publishes /robot_description and the arm's TF tree, for RViz and for
    # anything downstream that wants link poses. It is not in the control
    # path: joint angles come back from Gazebo over the bridge.
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True,
        }],
    )

    # ros_gz_sim create talks to /world/<name>/create, which does not exist
    # until the server has finished loading the world, and create has no
    # retry and no timeout flag. There is no event to wait on across the
    # process boundary, so this is a delay. On a loaded machine 6 s was not
    # enough and the spawn failed with
    #   [spawn_ur5]: Request to create entity from service
    #   [/world/empowerment_table/create] timed out
    # which leaves the world running with no arm in it. If you see that,
    # raise spawn_delay.
    spawn = TimerAction(period=LaunchConfiguration("spawn_delay"), actions=[
        Node(
            package="ros_gz_sim",
            executable="create",
            name="spawn_ur5",
            output="screen",
            arguments=[
                "-world", WORLD_NAME,
                "-string", robot_description_cmd,
                "-name", ROBOT_NAME,
                "-allow_renaming", "false",
            ],
            parameters=[{"use_sim_time": True}],
        ),
    ])

    # The depth image and point cloud are stamped depth_camera_link
    # (<gz_frame_id> in the world file). Nothing else publishes that frame,
    # so RViz would drop every cloud without this.
    #
    # Parented to world, not base_link: the camera is bolted to the room, and
    # hanging it off the arm would have silently swung it round when the arm
    # was yawed.
    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="depth_camera_tf",
        output="log",
        arguments=[
            "--x", str(CAMERA_XYZ[0]),
            "--y", str(CAMERA_XYZ[1]),
            "--z", str(CAMERA_XYZ[2]),
            "--roll", "0.0", "--pitch", str(CAMERA_PITCH), "--yaw", "0.0",
            "--frame-id", "world", "--child-frame-id", "depth_camera_link",
        ],
        parameters=[{"use_sim_time": True}],
    )

    # The frame config/empowerment.yaml is written in: world axes, origin at
    # the work surface. It coincides with base_link's origin but NOT its
    # orientation, because base_link carries BASE_YAW. The empowerment
    # marker is placed with tool positions straight out of the analytic
    # model, so it has to be published here and not in base_link.
    agent_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="arm_base_tf",
        output="log",
        arguments=[
            "--x", "0.0", "--y", "0.0", "--z", str(TABLE_HEIGHT),
            "--roll", "0.0", "--pitch", "0.0", "--yaw", "0.0",
            "--frame-id", "world", "--child-frame-id", "arm_base",
        ],
        parameters=[{"use_sim_time": True}],
    )

    # --- gz <-> ROS bridge ---------------------------------------------------
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_bridge",
        parameters=[{"config_file": bridge_cfg, "use_sim_time": True}],
        output="screen",
    )

    # --- our nodes -----------------------------------------------------------
    empowerment_node = Node(
        package="intrinsic_motivation_ros",
        executable="empowerment_node",
        name="empowerment_node",
        output="screen",
        condition=IfCondition(empowerment),
        parameters=[params, {
            "object_mass_factor": ParameterValue(bolted, value_type=float),
            "use_sim_time": True,
        }],
    )

    climber = Node(
        package="intrinsic_motivation_ros",
        executable="greedy_climber",
        name="greedy_climber",
        output="screen",
        condition=IfCondition(control),
        parameters=[params, {
            "task_bonus_weight": ParameterValue(task_bonus, value_type=float),
            "object_mass_factor": ParameterValue(bolted, value_type=float),
            "use_sim_time": True,
        }],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_cfg],
        condition=IfCondition(use_rviz),
        parameters=[{"use_sim_time": True}],
    )

    # The only thing that makes the drifter drift. Without it the scene has
    # two static objects and one pushable one, which is the control condition
    # rather than the experiment.
    drifter_driver = Node(
        package="intrinsic_motivation_ros",
        executable="drifter_driver",
        name="drifter_driver",
        output="screen",
        condition=IfCondition(drift),
        parameters=[{
            "use_sim_time": True,
            # 30 s, and the number is chosen for MARGIN rather than for
            # anything about the experiment; the drifter only has to move,
            # not move quickly.
            #
            # The tracker associates detections between frames by nearest
            # neighbour, with the gate capped at 0.11 m (it cannot go higher:
            # the objects are 0.24 m apart and a gate past half that starts
            # matching a track to its neighbour). So an object must not
            # travel more than 0.11 m between two detector updates.
            #
            #   period 30 s -> peak speed 2*pi*0.10/30 = 0.021 m/s
            #   loop at 1 Hz                           -> 0.021 m per tick
            #   headroom before the gate is exceeded   -> 5.2x
            #
            # That headroom is the point. The loop does keep to 1 Hz on an
            # idle machine (measured: gap 1.00-1.04 s, work 0.33 s of it),
            # but under CPU contention it stretches, and at 20 s period a
            # 3.7 s tick moved the drifter 0.110 m against the 0.11 m gate
            # and split it into two tracks. Losing a track identity is not
            # cosmetic: it resets independent_motion_std, which is the
            # statistic that identifies this object as non-contingent.
            "period": 30.0,
        }],
    )

    grasp_climber = Node(
        package="intrinsic_motivation_ros",
        executable="grasp_climber",
        name="grasp_climber",
        output="screen",
        condition=IfCondition(use_grasp),
        parameters=[params, {
            "objective": objective,
            "use_sim_time": True,
        }],
    )

    # The ONLY node that sees both the agent's estimate and the truth.
    detector_eval = Node(
        package="intrinsic_motivation_ros",
        executable="detector_eval",
        name="detector_eval",
        output="screen",
        condition=IfCondition(use_eval),
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription(args + [
        resource_path,
        gz, gz_headless,
        robot_state_publisher, spawn, camera_tf, agent_tf,
        bridge, drifter_driver, empowerment_node, climber, grasp_climber,
        detector_eval, rviz,
    ])
