#!/usr/bin/env python3
"""
empowerment_node -- publishes a UR5's n-step empowerment, live, from a depth camera.

Subscribes:
    /joint_states                    sensor_msgs/JointState
    /camera/depth/image_raw          sensor_msgs/Image      (32FC1 or 16UC1)

Publishes:
    /empowerment/value               std_msgs/Float64       bits
    /empowerment/markers             visualization_msgs/MarkerArray
    /empowerment/detections          geometry_msgs/PoseArray

Architecture note, since this is the part people query:

There are two models here and they do different jobs.

  Gazebo          the world. Real contact physics, real depth rendering,
                  ground truth, runs at 1x real time.

  The analytic    what the agent IMAGINES. Every cycle it must evaluate
  UR5 + camera    thousands of candidate futures, which is impossible by
  model           stepping Gazebo thousands of times. So the agent carries
                  its own fast forward model and predicts with that.

That split is not a shortcut. Empowerment is defined over a forward model,
and the paper explicitly places acquiring that model outside the formalism
(sec 4.8). Every model-based method has the same structure.

Where the object comes from, which is the part that used to be a cheat:

The agent locates objects itself, by segmenting the depth image against the
dominant plane in view (intrinsic_core.perception, via object_source). It is
never told that an object exists, how many there are, or where.

This node used to take the puck's position from /puck/pose, which is the
simulator handing over ground truth, and it made the project's central claim
only half true: the outcome half was honest and the input half was not.

There is now no ground-truth path here at all -- not a disabled one, not one
behind a parameter. This node does not subscribe to any topic carrying a true
object position, so it cannot read one even by mistake. Grading the detector
is done by a separate process, detector_eval, which is the only place the
agent's estimate and the simulator's truth are allowed to meet.

What perception supplies and what it does not:

    perception says THERE IS SOMETHING THERE.
    empowerment says WHETHER IT IS WORTH TOUCHING.

The three objects in the world are identical to the camera by construction --
same cylinder, same size, same standoff -- so nothing in this file can tell
the pushable one from the bolted one from the one that slides about on its
own. Only their response to action separates them, which is the experiment.

The empowerment number itself still comes from imagined rollouts.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray

from intrinsic_core.batched import BatchedDepthEmpowerment
from intrinsic_core.empowerment import EmpowermentConfig
from intrinsic_core.sensors import CameraConfig, DepthCamera, DepthDiscretiser
from intrinsic_core.ur5 import UR5, UR5Config

from .object_source import DepthObjectSource

UR5_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class EmpowermentNode(Node):
    def __init__(self) -> None:
        super().__init__("empowerment_node")
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

        # up=(0,0,1) is world up. It used to be (1,0,0) because the camera
        # looked straight down and world-up was parallel to the view
        # direction, which makes the camera basis degenerate. The camera is
        # oblique now, so the natural choice works again -- and it matters
        # for more than tidiness: the segmenter back-projects Gazebo's pixels
        # through this same basis, and gz-sim orients its image with world up.
        # Keeping (1,0,0) here would roll the agent's rendering 90 degrees
        # against the real one.
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

        # Objects come from the camera. See object_source: it deliberately
        # has no ground-truth path of any kind, so this node cannot read a
        # true object position even by accident.
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
            discretiser=DepthDiscretiser(
                depth_resolution=p("depth_resolution").value
            ),
            config=EmpowermentConfig(
                n_steps=p("horizon").value,
                n_sequences=p("n_sequences").value,
                exhaustive_below=p("exhaustive_below").value,
                seed=0,
            ),
        )

        self.joint_names = list(p("joint_names").value)
        self.q = np.array(p("initial_joint_state").value, dtype=float)
        self.object_xy = np.array(p("initial_object_xy").value, dtype=float)
        self.have_joints = False
        self._last_entered: float | None = None
        self._n_raw = 0

        self.create_subscription(JointState, "joint_states", self._on_joints, 10)

        self.pub_value = self.create_publisher(Float64, "empowerment/value", 10)
        self.pub_marker = self.create_publisher(
            MarkerArray, "empowerment/markers", 10
        )
        self.pub_detections = self.create_publisher(
            PoseArray, "empowerment/detections", 10
        )

        self.create_timer(1.0 / max(p("rate_hz").value, 1e-3), self._tick)

        self.get_logger().info(
            f"empowerment_node up | {self.arm.n_actions} actions, "
            f"horizon {p('horizon').value}, "
            f"camera {p('camera_width').value}x{p('camera_height').value}, "
            "objects located from the depth image, no ground truth anywhere"
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
        self.declare_parameter("n_sequences", 2000)
        self.declare_parameter("exhaustive_below", 0)
        self.declare_parameter("depth_resolution", 0.02)

        # --- perception -------------------------------------------------
        # Metres a point must stand off the fitted support plane to count as
        # an object. Floor is sensor noise: the world's camera has a 3 mm
        # gaussian and a real frame fits to a 2.7 mm residual, so anything
        # under ~1 cm starts segmenting the table itself into speckle.
        self.declare_parameter("plane_tolerance", 0.015)
        self.declare_parameter("min_detection_pixels", 40)

        # Motion nearer than this to the tool is assumed to be the agent's
        # doing, so it is not counted toward an object's independent motion.
        # Without it, a push would be recorded as the object moving by
        # itself, which inverts the distinction the drifter exists to test.
        self.declare_parameter("arm_exclusion_radius", 0.15)

        # Detector ticks an occluded object may be remembered for before it
        # is forgotten. This is the object-permanence window, and it is in
        # ticks rather than seconds, so at rate_hz 1.0 it is also seconds.
        # The default used to be 30, which kept a stale track alive long
        # enough to be reported alongside its own replacement.
        self.declare_parameter("track_max_missing", 5)

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
        self.declare_parameter("frame_id", "base_link")

    # -- callbacks ----------------------------------------------------------

    def _on_joints(self, msg: JointState) -> None:
        lookup = dict(zip(msg.name, msg.position))
        missing = [n for n in self.joint_names if n not in lookup]
        if missing:
            self.get_logger().warn(
                f"joint_states missing {missing}", throttle_duration_sec=5.0
            )
            return
        self.q = np.array([lookup[n] for n in self.joint_names])
        self.have_joints = True

    def _update_from_depth(self) -> None:
        """Locate objects in the latest frame and pick one to reason about."""
        if not self.objects.ready:
            self.get_logger().info(
                "waiting for a depth frame ...", throttle_duration_sec=5.0
            )
            return

        tool = self.arm.tool_position(self.q)
        tracks = self.objects.update(tool_xy=tool[:2])
        self._n_raw = self.objects.last_detection_count
        if not tracks:
            self.get_logger().warn(
                "nothing found above the support plane",
                throttle_duration_sec=10.0,
            )
            return

        nearest = self.objects.nearest_to(tool)
        if nearest is not None:
            self.object_xy = nearest
        self._publish_detections(tracks)

    def _publish_detections(self, tracks) -> None:
        """Where the agent believes the objects are.

        This is the agent's estimate and nothing else. Comparing it against
        the simulator's truth is detector_eval's job, in its own process, so
        that no ground truth is reachable from here.
        """
        msg = PoseArray()
        msg.header.frame_id = self.get_parameter("frame_id").value
        msg.header.stamp = self.get_clock().now().to_msg()
        table = self.get_parameter("table_height").value
        for track in tracks:
            pose = Pose()
            pose.position.x = float(track.xy[0])
            pose.position.y = float(track.xy[1])
            pose.position.z = float(table + track.height)
            pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.pub_detections.publish(msg)

    # -- main loop ----------------------------------------------------------

    def _tick(self) -> None:
        if not self.have_joints:
            self.get_logger().info(
                "waiting for /joint_states ...", throttle_duration_sec=5.0
            )
            return

        # Timing lives here permanently rather than as scaffolding, because
        # this loop does not keep to rate_hz and the distinction between "the
        # work got slow" and "we were not scheduled" is invisible from the
        # published value alone. gap is wall time since the previous tick;
        # the two phases are the actual work. If the phases stay small while
        # gap balloons, the executor is the problem, not the workload.
        entered = time.perf_counter()
        gap = None if self._last_entered is None else entered - self._last_entered
        self._last_entered = entered

        self._update_from_depth()
        detected = time.perf_counter()

        state = np.concatenate([self.q, self.object_xy])
        value = self.estimator.estimate(state)
        estimated = time.perf_counter()

        self.pub_value.publish(Float64(data=float(value)))
        self._publish_marker(value)

        if gap is not None:
            # Diagnostics must never be able to kill the agent. This block
            # already did once: last_dt is None on the very first update and
            # "{None:.2f}" raises TypeError, which propagated out of the timer
            # callback and took the whole node down a few seconds after
            # start-up. A crash in the code that measures the loop is a
            # strictly worse outcome than not measuring it.
            dt = self.objects.last_dt
            tracks = self.objects.tracker.tracks
            self.get_logger().info(
                f"tick | gap {gap:6.2f}s  perceive {detected - entered:5.3f}s  "
                f"estimate {estimated - detected:5.3f}s  "
                f"work {estimated - entered:5.3f}s  "
                f"raw {self._n_raw} tracks {len(tracks)}  "
                f"dt {'--' if dt is None else format(dt, '.2f')}  "
                f"ys {sorted(round(float(t.xy[1]), 3) for t in tracks)}",
                throttle_duration_sec=10.0,
            )

    def _publish_marker(self, value: float) -> None:
        tool = self.arm.tool_position(self.q)

        marker = Marker()
        marker.header.frame_id = self.get_parameter("frame_id").value
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "empowerment"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = float(tool[0])
        marker.pose.position.y = float(tool[1])
        marker.pose.position.z = float(tool[2]) + 0.15
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = marker.color.g = marker.color.b = 1.0
        marker.text = f"{value:.2f} bits"

        self.pub_marker.publish(MarkerArray(markers=[marker]))


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
