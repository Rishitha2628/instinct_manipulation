"""
Where the agent's knowledge of objects comes from: its own camera.

Both the monitor and the climber need exactly this, and they used to get it
from `/puck/pose` instead -- the simulator handing over the answer. Sharing
one implementation rather than two copies is not only tidiness: the pair have
to agree about the camera and the detector for the published empowerment
number to describe the arm that is actually moving, and two copies of that
setup is precisely how the field of view drifted out of step between the
world file and the model earlier in this project.

There is deliberately NO ground-truth path in here. Not a disabled one, not
one behind a parameter. A node that imports this module cannot read an
object's true position even by accident, because nothing in it subscribes to
anything that carries one. Grading the detector is the job of a separate
process (see detector_eval), which is the only place the two are allowed to
meet.
"""

from __future__ import annotations

import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from intrinsic_core.perception import DepthSegmenter, ObjectTracker, SegmenterConfig
from intrinsic_core.sensors import CameraConfig

__all__ = ["DepthObjectSource"]


class DepthObjectSource:
    """Latest depth frame in, tracked object positions out."""

    def __init__(
        self,
        node,
        camera: CameraConfig,
        plane_tolerance: float = 0.015,
        min_pixels: int = 40,
        arm_exclusion_radius: float = 0.15,
        max_missing: int = 5,
        topic: str = "camera/depth/image_raw",
        callback_group=None,
    ) -> None:
        self.node = node

        # Same CameraConfig object the agent predicts with, so the detector
        # and the renderer cannot disagree about where a pixel points. Only
        # the resolution differs, and the segmenter rebuilds its rays for
        # whatever size of frame actually turns up.
        self.segmenter = DepthSegmenter(camera, SegmenterConfig(
            plane_tolerance=plane_tolerance,
            min_pixels=min_pixels,
            # gz-sim publishes distance along the optical axis, not along the
            # ray. Confirmed by fitting a real frame both ways: 0.64 deg of
            # residual table tilt read as z-depth against 3.17 deg read as
            # range. Read it wrong and positions still look right while
            # object heights come out about 38 percent low.
            depth_is_range=False,
        ))
        self.tracker = ObjectTracker(
            arm_exclusion_radius=arm_exclusion_radius,
            max_missing=max_missing,
        )

        self.last_depth: np.ndarray | None = None
        self._last_update: float | None = None
        self.last_detection_count = 0
        self.last_dt: float | None = None
        """Raw blobs found in the most recent frame, before tracking. Compare
        it against len(tracker.tracks): if the raw count dips while the track
        count rises, the detector is losing sight of an object and the
        tracker is remembering it, which is a perception dropout rather than
        a tracking fault."""

        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(Image, topic, self._on_depth, sensor_qos,
                                 callback_group=callback_group)

    # -- sensor --------------------------------------------------------------

    def _on_depth(self, msg: Image) -> None:
        if msg.encoding not in ("32FC1", "16UC1"):
            self.node.get_logger().warn(
                f"unexpected depth encoding {msg.encoding}",
                throttle_duration_sec=10.0,
            )
            return

        dtype = np.float32 if msg.encoding == "32FC1" else np.uint16
        image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
        if msg.encoding == "16UC1":
            image = image.astype(np.float32) / 1000.0        # mm -> m
        # Left at the wire dtype. Widening a 640x480 frame to float64 here
        # costs a 2.4 MB copy per message, and every frame but the last one
        # before an update is discarded unlooked at; the segmenter widens the
        # one it is actually handed.
        self.last_depth = image

    @property
    def ready(self) -> bool:
        return self.last_depth is not None

    # -- objects -------------------------------------------------------------

    def update(self, tool_xy: np.ndarray | None = None) -> list:
        """Detect, associate, and return the current tracks."""
        if self.last_depth is None:
            return self.tracker.tracks

        detections = self.segmenter.detect(self.last_depth)
        self.last_detection_count = len(detections)

        # Real elapsed time, not the nominal loop rate. The caller's loop does
        # not keep to its configured rate, and the tracker widens its
        # association gate by dt; handing it a nominal value would defeat
        # that. See the ObjectTracker.update docstring.
        now = self.node.get_clock().now().nanoseconds * 1e-9
        dt = None if self._last_update is None else now - self._last_update
        self._last_update = now
        self.last_dt = dt

        return self.tracker.update(detections, tool_xy=tool_xy, dt=dt)

    def nearest_to(self, tool_xy: np.ndarray) -> np.ndarray | None:
        """The tracked object closest to the tool, or None if nothing is seen.

        The forward model carries a single object, so somebody has to choose.
        Nearest-to-the-tool is the one the agent is in a position to affect,
        and at this horizon the only one whose empowerment contribution can
        be non-zero anyway.

        Scoring all three against each other -- which is what the static /
        drifter / pushable scene is for -- needs the model to carry several
        objects at once, and needs the drifter's motion to enter rollouts as
        action-independent noise rather than as a puck that happens to be
        sitting still. Neither exists yet.
        """
        tracks = self.tracker.tracks
        if not tracks:
            return None
        nearest = min(
            tracks, key=lambda t: float(np.linalg.norm(t.xy - tool_xy[:2]))
        )
        return np.asarray(nearest.xy, dtype=float)
