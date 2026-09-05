#!/usr/bin/env python3
"""
greedy_climber -- drives the UR5 up the empowerment gradient.

Subscribes:
    /joint_states                sensor_msgs/JointState
    /camera/depth/image_raw      sensor_msgs/Image

Publishes:
    /joint_trajectory_controller/joint_trajectory   trajectory_msgs/JointTrajectory

Each cycle it evaluates every one-step action, keeps the successor with the
highest score, and sends it as a short trajectory segment. This is the
control law from sec 4.7 of the paper -- enough to swing up a pendulum with
no reward function.

Two caveats, both stated in the paper rather than discovered here:

  * Not principled. Empowerment is not a value function, so hill-climbing
    on it optimises nothing in particular. It works on simple systems and
    nobody can characterise when it stops working.

  * It will not finish a task. The arm finds the object and stays near it.
    It will not pick it up, because grasping commits to one future and so
    LOWERS the option count. Finishing has to come from elsewhere; see
    task_bonus_weight, which adds a single scalar carrying no information
    about where the object is or how to reach it.
"""

from __future__ import annotations

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from intrinsic_core.batched import BatchedDepthEmpowerment
from intrinsic_core.empowerment import EmpowermentConfig
from intrinsic_core.sensors import CameraConfig, DepthCamera, DepthDiscretiser
from intrinsic_core.ur5 import UR5, UR5Config

from .object_source import DepthObjectSource

UR5_JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


class GreedyClimber(Node):
    def __init__(self) -> None:
        super().__init__("greedy_climber")
        self._declare_parameters()
        p = self.get_parameter

        self.arm = UR5(UR5Config(
            dq=p("joint_step").value,
            active_joints=tuple(p("active_joints").value),
            tool_radius=p("tool_radius").value,
            puck_radius=p("object_radius").value,
            puck_mass_factor=p("object_mass_factor").value,
            table_height=p("table_height").value,
        ))
        # up=(0,0,1) is world up. It was (1,0,0) because the old camera
        # looked straight down, which made world-up parallel to the view
        # direction and the camera basis degenerate. The camera is oblique
        # now, and the natural choice also matches how gz-sim orients its
        # image, which matters because the detector back-projects Gazebo's
        # pixels through this same basis.
        camera_config = CameraConfig(
            position=tuple(p("camera_position").value),
            look_at=tuple(p("camera_look_at").value),
            up=(0.0, 0.0, 1.0),
            fov_deg=p("camera_fov_deg").value,
            width=p("camera_width").value,
            height=p("camera_height").value,
            table_height=p("table_height").value,
        )
        self.camera = DepthCamera(camera_config)

        # This node MOVES THE ARM, so it is the one that most needs to be
        # honest about where its knowledge comes from. It used to read the
        # puck's position off /puck/pose and never looked at the camera at
        # all. object_source has no ground-truth path of any kind.
        self.objects = DepthObjectSource(
            self,
            camera_config,
            plane_tolerance=p("plane_tolerance").value,
            min_pixels=p("min_detection_pixels").value,
            arm_exclusion_radius=p("arm_exclusion_radius").value,
            max_missing=p("track_max_missing").value,
        )
        self.estimator = BatchedDepthEmpowerment(
            arm=self.arm,
            camera=self.camera,
            discretiser=DepthDiscretiser(p("depth_resolution").value),
            config=EmpowermentConfig(
                n_steps=p("horizon").value,
                n_sequences=p("climber_sequences").value,
                exhaustive_below=0,
                seed=0,
            ),
        )

        self.joint_names = list(p("joint_names").value)
        self.q = np.array(p("initial_joint_state").value, dtype=float)
        self.object_xy = np.array(p("initial_object_xy").value, dtype=float)
        self.have_joints = False

        self.create_subscription(JointState, "joint_states", self._on_joints, 10)

        self.pub_traj = self.create_publisher(
            JointTrajectory,
            "joint_trajectory_controller/joint_trajectory",
            10,
        )
        self.pub_score = self.create_publisher(Float64, "empowerment/chosen_score", 10)

        self.period = 1.0 / max(p("rate_hz").value, 1e-3)
        self.create_timer(self.period, self._tick)

        bonus = p("task_bonus_weight").value
        self.get_logger().info(
            "greedy_climber up | "
            + ("pure empowerment: no goal, no reward" if bonus <= 0.0
               else f"empowerment + task bonus {bonus}")
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("joint_step", 0.10)
        self.declare_parameter("active_joints", [0, 1, 2, 3])
        self.declare_parameter("tool_radius", 0.05)
        self.declare_parameter("object_radius", 0.04)
        self.declare_parameter("object_mass_factor", 1.0)

        # Work surface in the arm's base frame. Non-zero whenever the arm is
        # mounted higher than the bench it works over, which it is here: the
        # pedestal has to clear the arm's all-zeros spawn pose. Feeds the
        # contact model and the camera's ground plane, so getting it wrong
        # makes the agent push at empty air and range against a phantom
        # table.
        self.declare_parameter("table_height", 0.0)

        self.declare_parameter("horizon", 2)
        self.declare_parameter("climber_sequences", 600)
        self.declare_parameter("depth_resolution", 0.02)

        self.declare_parameter("camera_position", [-0.50, 0.0, 0.95])
        self.declare_parameter("camera_look_at", [-0.50, 0.0, 0.0])
        self.declare_parameter("camera_fov_deg", 30.0)
        self.declare_parameter("camera_width", 16)
        self.declare_parameter("camera_height", 12)

        self.declare_parameter("rate_hz", 1.0)
        self.declare_parameter("joint_names", UR5_JOINT_ORDER)
        self.declare_parameter(
            "initial_joint_state", [0.0, -1.2, 1.4, -1.75, -1.57, 0.0]
        )
        self.declare_parameter("initial_object_xy", [-0.50, 0.0])

        # --- perception ---------------------------------------------------
        self.declare_parameter("plane_tolerance", 0.015)
        self.declare_parameter("min_detection_pixels", 40)
        self.declare_parameter("arm_exclusion_radius", 0.15)
        self.declare_parameter("track_max_missing", 5)

        self.declare_parameter("task_bonus_weight", 0.0)
        self.declare_parameter("task_bonus_radius", 0.06)
        self.declare_parameter("candidate_actions", 24)

    def _on_joints(self, msg: JointState) -> None:
        lookup = dict(zip(msg.name, msg.position))
        if all(n in lookup for n in self.joint_names):
            self.q = np.array([lookup[n] for n in self.joint_names])
            self.have_joints = True

    # -- scoring ------------------------------------------------------------

    def _score(self, state: np.ndarray) -> float:
        """Empowerment, optionally plus a one-bit task term.

        The task term carries NO information about where the object is or
        how to reach it. Empowerment supplies that. The bonus only says
        'contact is good', which is the intention empowerment structurally
        cannot represent.
        """
        value = self.estimator.estimate(state)

        weight = self.get_parameter("task_bonus_weight").value
        if weight > 0.0:
            tool = self.arm.tool_position(state[:6])
            reach = float(np.hypot(state[6] - tool[0], state[7] - tool[1]))
            if reach < self.get_parameter("task_bonus_radius").value:
                value += weight
        return value

    def _tick(self) -> None:
        if not self.have_joints:
            self.get_logger().info(
                "waiting for /joint_states ...", throttle_duration_sec=5.0
            )
            return

        if self.objects.ready:
            tool = self.arm.tool_position(self.q)
            self.objects.update(tool_xy=tool[:2])
            nearest = self.objects.nearest_to(tool)
            if nearest is not None:
                self.object_xy = nearest
        else:
            self.get_logger().info(
                "waiting for a depth frame ...", throttle_duration_sec=5.0
            )
            return

        state = np.concatenate([self.q, self.object_xy])

        # Evaluating all 81 successors at ~0.2 s each is too slow for a live
        # loop, so score a random subset each cycle. Over several cycles the
        # arm still climbs; it just takes a noisier path.
        n_candidates = min(
            self.get_parameter("candidate_actions").value, self.arm.n_actions
        )
        rng = np.random.default_rng()
        candidates = rng.choice(self.arm.n_actions, size=n_candidates, replace=False)

        best_action, best_score = None, -np.inf
        for action in candidates:
            successor = self.arm.step(state, int(action))
            score = self._score(successor)
            if score > best_score:
                best_action, best_score = int(action), score

        if best_action is None:
            return

        target = self.arm.step(state, best_action)
        self._send_trajectory(target[:6])
        self.pub_score.publish(Float64(data=float(best_score)))

    def _send_trajectory(self, q_target: np.ndarray) -> None:
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in q_target]
        point.velocities = [0.0] * len(q_target)

        seconds = self.period * 0.9
        point.time_from_start = Duration(
            sec=int(seconds), nanosec=int((seconds % 1.0) * 1e9)
        )
        traj.points = [point]
        self.pub_traj.publish(traj)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GreedyClimber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
