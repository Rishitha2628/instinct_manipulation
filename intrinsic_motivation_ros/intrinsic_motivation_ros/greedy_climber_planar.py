#!/usr/bin/env python3
"""
greedy_climber -- drives the arm up the empowerment gradient.

Subscribes:
    /joint_states                sensor_msgs/JointState
    /object_pose                 geometry_msgs/PoseStamped

Publishes:
    /joint_group_position_controller/commands   std_msgs/Float64MultiArray

Each cycle it evaluates every one-step action, keeps the one whose
successor state has the highest empowerment, and commands it.  This is the
control law from section 4.7 of the paper, which is enough to swing up a
pendulum without a reward function.

Two honest caveats, both from the paper:

  * This is not principled.  Empowerment is not a value function, so
    hill-climbing on it optimises nothing in particular.  It works on
    simple systems and nobody can characterise when it stops working.

  * It will not complete a task.  The arm will find the object and stay
    near it.  It will not pick it up, because grasping commits to one
    future and therefore *lowers* the option count.  Finishing has to come
    from a reward term; see the `task_bonus_weight` parameter.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from intrinsic_core import (
    ArmConfig,
    EmpowermentConfig,
    EmpowermentEstimator,
    GridDiscretiser,
    JOINT_ACTIONS,
    PlanarArm,
)


class GreedyClimber(Node):
    def __init__(self) -> None:
        super().__init__("greedy_climber")

        self.declare_parameter("link1_length", 0.30)
        self.declare_parameter("link2_length", 0.25)
        self.declare_parameter("joint_step", 0.12)
        self.declare_parameter("contact_radius", 0.045)
        self.declare_parameter("horizon", 3)
        self.declare_parameter("outcome_resolution", 0.025)
        self.declare_parameter("rate_hz", 4.0)
        self.declare_parameter("joint_names", ["joint1", "joint2"])
        self.declare_parameter("initial_object_xy", [0.30, 0.25])
        self.declare_parameter("task_bonus_weight", 0.0)
        self.declare_parameter("task_bonus_radius", 0.05)

        p = self.get_parameter
        self.arm = PlanarArm(
            ArmConfig(
                l1=p("link1_length").value,
                l2=p("link2_length").value,
                dq=p("joint_step").value,
                puck_radius=p("contact_radius").value,
            )
        )
        self.estimator = EmpowermentEstimator(
            rollout=self.arm.rollout,
            n_actions=len(JOINT_ACTIONS),
            discretiser=GridDiscretiser(p("outcome_resolution").value),
            config=EmpowermentConfig(n_steps=p("horizon").value, seed=0),
        )

        self.joint_names = list(p("joint_names").value)
        self.q = np.zeros(2)
        self.object_xy = np.array(p("initial_object_xy").value, dtype=float)
        self.have_joints = False

        self.create_subscription(JointState, "joint_states", self._on_joints, 10)
        self.create_subscription(PoseStamped, "object_pose", self._on_object, 10)
        self.pub_cmd = self.create_publisher(
            Float64MultiArray,
            "joint_group_position_controller/commands",
            10,
        )

        self.create_timer(1.0 / max(p("rate_hz").value, 1e-3), self._tick)
        self.get_logger().info("greedy_climber up (no goal, no reward)")

    def _on_joints(self, msg: JointState) -> None:
        lookup = dict(zip(msg.name, msg.position))
        if all(n in lookup for n in self.joint_names):
            self.q = np.array([lookup[n] for n in self.joint_names])
            self.have_joints = True

    def _on_object(self, msg: PoseStamped) -> None:
        self.object_xy = np.array([msg.pose.position.x, msg.pose.position.y])

    def _score(self, state: np.ndarray) -> float:
        """Empowerment, optionally plus a one-bit task term.

        With task_bonus_weight = 0 this is pure empowerment: the arm engages
        with the object and never finishes.  Raising it supplies the
        intention that empowerment structurally cannot.
        """
        value = self.estimator.estimate(state)

        weight = self.get_parameter("task_bonus_weight").value
        if weight > 0.0:
            ex, ey = self.arm.forward_kinematics(state[0], state[1])
            reached = np.hypot(state[2] - ex, state[3] - ey) < \
                self.get_parameter("task_bonus_radius").value
            value += weight * float(reached)
        return value

    def _tick(self) -> None:
        if not self.have_joints:
            return

        state = np.array([self.q[0], self.q[1], *self.object_xy])

        best_action, best_score = 0, -np.inf
        for action in range(len(JOINT_ACTIONS)):
            successor = self.arm.step(state, action)
            score = self._score(successor)
            if score > best_score:
                best_action, best_score = action, score

        target = self.arm.step(state, best_action)
        self.pub_cmd.publish(
            Float64MultiArray(data=[float(target[0]), float(target[1])])
        )


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
