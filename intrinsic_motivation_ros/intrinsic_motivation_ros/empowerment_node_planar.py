#!/usr/bin/env python3
"""
empowerment_node -- publishes the arm's n-step empowerment, live.

Subscribes:
    /joint_states                sensor_msgs/JointState
    /object_pose                 geometry_msgs/PoseStamped   (the puck)

Publishes:
    /empowerment/value           std_msgs/Float64            bits
    /empowerment/gradient        geometry_msgs/Vector3       ascent direction
                                                             in joint space
    /empowerment/markers         visualization_msgs/MarkerArray

The node holds an analytic forward model of the arm.  Every control cycle it
replays a batch of candidate action sequences through that model, counts how
many distinguishable outcomes result, and publishes the log of that count.

Nothing here knows what the object is for.  The object enters only because
its position is part of the outcome vector, which is why empowerment rises
near it.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray

from intrinsic_core import (
    ArmConfig,
    EmpowermentConfig,
    EmpowermentEstimator,
    GridDiscretiser,
    PlanarArm,
)


class EmpowermentNode(Node):
    def __init__(self) -> None:
        super().__init__("empowerment_node")

        self._declare_parameters()
        p = self.get_parameter

        self.arm = PlanarArm(
            ArmConfig(
                l1=p("link1_length").value,
                l2=p("link2_length").value,
                dq=p("joint_step").value,
                puck_radius=p("contact_radius").value,
                puck_mass_factor=p("object_mass_factor").value,
            )
        )

        self.estimator = EmpowermentEstimator(
            rollout=self.arm.rollout,
            n_actions=9,
            discretiser=GridDiscretiser(p("outcome_resolution").value),
            config=EmpowermentConfig(
                n_steps=p("horizon").value,
                n_sequences=p("n_sequences").value,
                seed=0,
            ),
        )

        self.joint_names = list(p("joint_names").value)
        self.q = np.zeros(2)
        self.object_xy = np.array(p("initial_object_xy").value, dtype=float)
        self.have_joints = False

        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(JointState, "joint_states", self._on_joints, 10)
        self.create_subscription(PoseStamped, "object_pose", self._on_object, 10)

        self.pub_value = self.create_publisher(Float64, "empowerment/value", 10)
        self.pub_grad = self.create_publisher(Vector3, "empowerment/gradient", 10)
        self.pub_marker = self.create_publisher(
            MarkerArray, "empowerment/markers", latched
        )

        period = 1.0 / max(p("rate_hz").value, 1e-3)
        self.create_timer(period, self._tick)

        self.get_logger().info(
            f"empowerment_node up: horizon={p('horizon').value}, "
            f"resolution={p('outcome_resolution').value} m, "
            f"rate={p('rate_hz').value} Hz"
        )

    # -- parameters ---------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter("link1_length", 0.30)
        self.declare_parameter("link2_length", 0.25)
        self.declare_parameter("joint_step", 0.12)
        self.declare_parameter("contact_radius", 0.045)
        self.declare_parameter("object_mass_factor", 1.0)

        self.declare_parameter("horizon", 3)
        self.declare_parameter("n_sequences", 300)
        self.declare_parameter("outcome_resolution", 0.025)

        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("joint_names", ["joint1", "joint2"])
        self.declare_parameter("initial_object_xy", [0.30, 0.25])
        self.declare_parameter("frame_id", "base_link")

    # -- callbacks ----------------------------------------------------------

    def _on_joints(self, msg: JointState) -> None:
        lookup = dict(zip(msg.name, msg.position))
        try:
            self.q = np.array([lookup[n] for n in self.joint_names])
            self.have_joints = True
        except KeyError:
            missing = set(self.joint_names) - set(msg.name)
            self.get_logger().warn(
                f"joint_states missing {missing}", throttle_duration_sec=5.0
            )

    def _on_object(self, msg: PoseStamped) -> None:
        self.object_xy = np.array([msg.pose.position.x, msg.pose.position.y])

    # -- main loop ----------------------------------------------------------

    def _state(self, q: np.ndarray) -> np.ndarray:
        return np.array([q[0], q[1], self.object_xy[0], self.object_xy[1]])

    def _tick(self) -> None:
        if not self.have_joints:
            return

        value = self.estimator.estimate(self._state(self.q))
        self.pub_value.publish(Float64(data=float(value)))

        gradient = self._finite_difference_gradient(value)
        self.pub_grad.publish(
            Vector3(x=float(gradient[0]), y=float(gradient[1]), z=0.0)
        )
        self._publish_markers(value)

    def _finite_difference_gradient(self, here: float) -> np.ndarray:
        """Central difference in joint space.

        Four extra empowerment evaluations per cycle.  Empowerment has no
        analytic gradient -- it is a count -- so this is the only option,
        and it is the main reason the node is slow.
        """
        eps = self.get_parameter("joint_step").value
        grad = np.zeros(2)
        for i in range(2):
            up, down = self.q.copy(), self.q.copy()
            up[i] += eps
            down[i] -= eps
            e_up = self.estimator.estimate(self._state(up))
            e_dn = self.estimator.estimate(self._state(down))
            grad[i] = (e_up - e_dn) / (2.0 * eps)
        return grad

    # -- visualisation ------------------------------------------------------

    def _publish_markers(self, value: float) -> None:
        frame = self.get_parameter("frame_id").value
        ex, ey = self.arm.forward_kinematics(self.q[0], self.q[1])

        text = Marker()
        text.header.frame_id = frame
        text.header.stamp = self.get_clock().now().to_msg()
        text.ns = "empowerment"
        text.id = 0
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(ex)
        text.pose.position.y = float(ey)
        text.pose.position.z = 0.12
        text.pose.orientation.w = 1.0
        text.scale.z = 0.05
        text.color.a = 1.0
        text.color.r = text.color.g = text.color.b = 1.0
        text.text = f"{value:.2f} bits"

        self.pub_marker.publish(MarkerArray(markers=[text]))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EmpowermentNode()
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
