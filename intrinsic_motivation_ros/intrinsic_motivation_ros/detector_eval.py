"""
detector_eval -- scores the detector against ground truth, out of process.

This is the only place in the system where the agent's estimate of where the
objects are and the simulator's knowledge of where they really are appear
together, and it is a separate process on purpose.

The agent's nodes used to subscribe to /puck/pose themselves. That was fine
in intent -- it was used only for logging -- but it is the wrong shape. A
topic that is present but "must not be used for anything" is exactly what
quietly gets used again in six months, and while it is subscribed there is no
way to demonstrate that the agent is not reading it short of reading the
source. Moving the comparison out here makes the separation structural: the
empowerment node and the climber do not subscribe to any topic carrying a
true object position, so they cannot consult one even by mistake.

Subscribes:
    /empowerment/detections     geometry_msgs/PoseArray   agent's estimate
    /puck/pose                  geometry_msgs/PoseStamped ground truth
    /fixed_block/pose           geometry_msgs/PoseStamped ground truth
    /drifter/joint_states       sensor_msgs/JointState    ground truth

Publishes:
    /detector/error             std_msgs/Float64   mean metres over all objects
    /detector/max_error         std_msgs/Float64   worst object
    /detector/count             std_msgs/Float64   objects the agent believes in
    /detector/error/<name>      std_msgs/Float64   per object

Read the per-object topics, not just the mean. They measure two different
things and averaging them hides that. For a stationary object the error is
the detector's accuracy. For the drifter it is dominated by STALENESS
instead: the agent's estimate is only refreshed when its loop runs, which is
irregular and well below 1 Hz, while the truth moves continuously. A large
drifter error therefore says the loop is slow, not that the detector is
inaccurate, and the fix for it is in the loop rather than in perception.

The drifter's truth arrives as a joint position rather than a pose because
its model root is a fixed anchor that never moves, so the model pose says
nothing; the block's world y is the anchor's y plus the slide value.

Nothing here feeds back into the agent. It exists so that a claim like
"located to within 6 mm without being told where anything was" has a number
behind it.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


class DetectorEval(Node):

    def __init__(self) -> None:
        super().__init__("detector_eval")

        # Where the drifter's anchor sits, so the slide joint can be turned
        # into a world position. Matches the <pose> of the drifter model in
        # worlds/empowerment_table.sdf.
        self.declare_parameter("drifter_anchor_xy", [-0.68, -0.24])
        self.anchor = np.array(
            self.get_parameter("drifter_anchor_xy").value, dtype=float
        )

        self.truth: dict[str, np.ndarray] = {}

        self.create_subscription(
            PoseStamped, "puck/pose",
            lambda m: self._pose("puck", m), 10)
        self.create_subscription(
            PoseStamped, "fixed_block/pose",
            lambda m: self._pose("fixed_block", m), 10)
        self.create_subscription(
            JointState, "drifter/joint_states", self._drifter, 10)
        self.create_subscription(
            PoseArray, "empowerment/detections", self._detections, 10)

        self.pub_error = self.create_publisher(Float64, "detector/error", 10)
        self.pub_max = self.create_publisher(Float64, "detector/max_error", 10)
        self.pub_count = self.create_publisher(Float64, "detector/count", 10)
        self.pub_per_object: dict[str, object] = {}

        self.get_logger().info(
            "detector_eval up | scoring /empowerment/detections against "
            "simulator ground truth"
        )

    # -- ground truth --------------------------------------------------------

    def _pose(self, name: str, msg: PoseStamped) -> None:
        self.truth[name] = np.array([msg.pose.position.x, msg.pose.position.y])

    def _drifter(self, msg: JointState) -> None:
        """World xy of the drifter's block, from its slide joint.

        By name, not by index. The model reports two joints -- the fixed
        anchor_to_world first, then slide -- and taking position[0] gets the
        fixed one, which reads 0.0 for ever and makes a moving object look
        bolted down. That mistake cost an afternoon here.
        """
        joints = dict(zip(msg.name, msg.position))
        if "slide" in joints:
            self.truth["drifter"] = self.anchor + np.array([0.0, joints["slide"]])

    # -- scoring -------------------------------------------------------------

    def _detections(self, msg: PoseArray) -> None:
        if not self.truth:
            self.get_logger().info(
                "waiting for ground truth ...", throttle_duration_sec=5.0
            )
            return

        estimates = [np.array([p.position.x, p.position.y]) for p in msg.poses]
        self.pub_count.publish(Float64(data=float(len(estimates))))
        if not estimates:
            return

        # Each true object scored against its nearest estimate. Deliberately
        # not a full assignment: a spurious extra detection should not be
        # able to improve the error, and it shows up in /detector/count
        # instead, which is where a miscount belongs.
        errors = {}
        for name, true in self.truth.items():
            errors[name] = min(
                float(np.linalg.norm(e - true)) for e in estimates
            )
            if name not in self.pub_per_object:
                self.pub_per_object[name] = self.create_publisher(
                    Float64, f"detector/error/{name}", 10
                )
            self.pub_per_object[name].publish(Float64(data=errors[name]))

        values = list(errors.values())
        self.pub_error.publish(Float64(data=float(np.mean(values))))
        self.pub_max.publish(Float64(data=float(np.max(values))))


def main() -> None:
    rclpy.init()
    node = DetectorEval()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
