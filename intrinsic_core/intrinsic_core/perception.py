"""
Finding objects in a depth image, without being told any exist.

Why this file exists: until now the agent read the puck's position off
`/puck/pose`, which is the simulator handing it the answer. That is the one
place the project's central claim leaked -- "the agent is never told the
object exists, what it is, or what to do with it" was true of the outcome
half and false of the input half. This closes it.

What replaces the oracle is NOT a learned detector. It is a geometric prior,
deliberately:

    the dominant plane in view is the support surface,
    and anything standing off it is a thing.

That is weak and generic in the way an innate prior should be. It knows
nothing about pucks, cylinders, colour, or count. It does not know how many
objects there are, and it will happily report a coffee cup. What it will not
do is tell you which of the things it found is worth touching -- that is
exactly the question empowerment answers, and keeping the two separate is the
point of the experiment:

    perception says THERE IS SOMETHING THERE.
    empowerment says WHETHER IT IS WORTH TOUCHING.

The three objects in worlds/empowerment_table.sdf are identical to this
module by construction: same cylinder, same size, same height off the table.
It cannot distinguish the pushable one from the bolted one from the one that
slides around on its own. Only their response to action separates them.

What is still assumed, stated plainly so it is not mistaken for a free lunch:

  * a single dominant planar surface exists and most of the view is it;
  * objects stand PROUD of that plane, toward the camera;
  * the arm can be excluded, either by a rendered self-mask or by the
    cruder rule that the arm reaches in from outside the frame while the
    objects do not touch its border.

Nothing here needs ROS. Feed it a float array of metres and it works, which
is what makes it testable against a saved frame in a unit test instead of
only inside a running simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage

from .sensors import CameraConfig, DepthCamera

__all__ = [
    "SegmenterConfig",
    "Detection",
    "DepthSegmenter",
    "Track",
    "ObjectTracker",
]


@dataclass
class SegmenterConfig:
    """Everything the geometric prior needs, and nothing about pucks."""

    plane_tolerance: float = 0.015
    """Metres a point must stand off the fitted plane to count as an object.

    Floor is set by sensor noise: the world's depth camera has a 3 mm
    gaussian, and a flat table read through it has a spread of roughly that.
    Below about 3x the noise the table itself starts segmenting into
    speckle. Above the object height (0.08 m here) nothing is ever found."""

    min_pixels: int = 6
    """Blobs smaller than this are noise. At the detection resolutions used
    here an 8 cm object is tens of pixels, so this is generous."""

    max_pixel_fraction: float = 0.30
    """Blobs larger than this fraction of the frame are not objects. Catches
    the case where plane fitting fails and half the image is 'above' it."""

    reject_border: bool = True
    border_margin: int = 1
    """The arm reaches in from outside the field of view, so it is connected
    to the frame edge; the objects sit in the middle of the table and are
    not. Crude, free, and works. A rendered self-mask is strictly better and
    is supported by passing `self_mask` to `detect`."""

    top_slice: float = 0.15
    """Fraction of a blob's height, measured down from its highest point,
    whose pixels are averaged to locate the object's axis.

    Using the whole blob biases the estimate toward the camera, because an
    oblique view sees the near side of an object and not the far side. The
    top face of a flat-topped object is centred on its axis from any angle,
    so averaging only the top slice removes most of that bias.

    Measured against a rendered sphere at 96x72, bias toward the camera:

        1.00 (whole blob)   -2.16 cm
        0.50                -1.93
        0.30                -1.42
        0.15                -0.84
        0.08                -0.59

    It never reaches zero there because intrinsic_core renders the puck as a
    SPHERE, whose top cap is only half visible from an angle. The real
    objects are flat-topped cylinders: every point of the top face has the
    same standoff, so any slice captures the whole disc and its centroid is
    the axis exactly. 0.15 is chosen to be tight enough to matter on the
    sphere while still spanning several times the camera's 3 mm noise."""

    plane_iterations: int = 8
    """Maximum trimming rounds in the plane fit; it exits early once the fit
    stops moving. Measured convergence is at round 4 on a real frame."""

    depth_is_range: bool = False
    """Whether a depth value is distance ALONG THE RAY (True) or distance
    along the optical axis (False, the usual RGBD convention and what
    gz-sim's depth_camera publishes).

    This matters only when the camera is not looking straight down, which is
    exactly the case now. Get it wrong with an oblique camera and the
    reconstructed table is a plane at the wrong tilt: the fit still succeeds,
    objects are still found, and every position is quietly skewed toward the
    edges of the frame. intrinsic_core's own DepthCamera returns range, so
    tests against rendered frames pass True and frames off the ROS topic
    pass False."""


@dataclass
class Detection:
    """One thing standing off the support surface."""

    xy: np.ndarray
    """Object axis in the camera config's frame (agent frame: world x, y)."""

    height: float
    """Metres the highest point of the blob stands off the plane. This is
    what makes a lift visible: a grasped and raised object keeps its (x, y)
    and changes this."""

    n_pixels: int
    touches_border: bool
    centroid_px: tuple[float, float]


@dataclass
class Track:
    """A detection followed across frames, so occlusion is not deletion."""

    xy: np.ndarray
    height: float
    frames_seen: int = 1
    frames_missing: int = 0
    visible: bool = True
    _free_motion: list = field(default_factory=list)

    @property
    def independent_motion_std(self) -> float:
        """Metres per frame this thing moves while nothing is near it.

        This is the number that separates the drifter from the puck without
        anything being told which is which. A pushable object sitting on a
        table has an independent motion of zero: it only ever moves because
        it was touched. Something that slides around on its own has a
        non-zero value here, learned by watching rather than by being told.

        A deterministic forward model cannot represent that, and if the
        agent predicts the drifter as 'a puck that sits still until pushed'
        then in imagination it never moves and looks identical to the static
        block. Feeding this variance into the rollouts is what makes the
        drifter's outcomes action-INDEPENDENT, which is what drives its
        mutual information with the agent's actions to zero. Without it the
        noisy-TV comparison does not mean anything."""
        if len(self._free_motion) < 2:
            return 0.0
        return float(np.std(self._free_motion))


class DepthSegmenter:
    """Objects from a depth image, via 'the big flat thing is the floor'."""

    def __init__(
        self,
        camera: CameraConfig,
        config: SegmenterConfig | None = None,
    ) -> None:
        self.camera = camera
        self.config = config or SegmenterConfig()
        self._ray_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    # -- geometry ------------------------------------------------------------

    def _rays_for(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Unit ray directions and the camera origin, at the image's own size.

        Built by DepthCamera itself rather than by a second copy of the
        pinhole maths here. The detector and the renderer then cannot
        disagree about where a pixel points, which is the sort of drift this
        project has already been bitten by once with the field of view.
        """
        h, w = shape
        if (h, w) not in self._ray_cache:
            cam = replace(self.camera, width=w, height=h)
            dc = DepthCamera(cam)
            self._ray_cache[(h, w)] = (dc._rays, dc._origin)
        return self._ray_cache[(h, w)]

    def back_project(self, depth: np.ndarray) -> np.ndarray:
        """Depth image -> (h*w, 3) points in the camera config's frame."""
        depth = np.asarray(depth, dtype=np.float64)
        rays, origin = self._rays_for(depth.shape)

        d = depth.ravel()
        if not self.config.depth_is_range:
            # Along-axis depth: scale to range along the ray. The optical
            # axis is the ray at the image centre, up to the half-pixel
            # offset, so take it from the camera basis instead.
            forward = np.array(self.camera.look_at, dtype=np.float64) - origin
            forward /= np.linalg.norm(forward)
            cos_theta = rays @ forward
            d = d / np.clip(cos_theta, 1e-6, None)

        return origin[None, :] + rays * d[:, None]

    def fit_plane(self, points: np.ndarray) -> tuple[np.ndarray, float]:
        """Robust dominant-plane fit. Returns (unit normal, offset).

        The normal is oriented toward the camera, so a point's signed
        distance is positive when it stands proud of the surface.

        Least squares with trimming rather than RANSAC: the support surface
        is the overwhelming majority of the view here, so a plain fit is
        already close and two rounds of discarding the worst residuals
        converge on the table without the extra machinery.
        """
        pts = points[np.isfinite(points).all(axis=1)]
        if len(pts) < 3:
            raise ValueError("not enough finite points to fit a plane")

        keep = np.ones(len(pts), dtype=bool)
        normal = np.array([0.0, 0.0, 1.0])
        offset = 0.0
        previous = None

        # Iterate to convergence rather than a fixed count. Measured on a real
        # 640x480 Gazebo frame, where the view also contains the floor a
        # metre below the table, the arm, and both benches:
        #
        #   iter 0   218307 inliers   tilt 10.55 deg   offset -0.3772
        #   iter 1   189356            2.94            -0.2897
        #   iter 2   173590            0.64            -0.2597
        #   iter 3   174171            0.12            -0.2520
        #   iter 4   176453            0.04            -0.2509   converged
        #
        # Three rounds, which is what this used to do, stops at 0.64 deg and
        # a centimetre of offset. The trimmed scale settles at 2.7 mm, which
        # is the world file's 3 mm sensor noise: the fit ends up measuring
        # the noise floor, which is the right place to stop.
        for _ in range(self.config.plane_iterations):
            sel = pts[keep]
            # Trimming can in principle discard everything -- a frame that is
            # mostly arm, for instance, where the residual distribution is
            # wide and multi-modal. Stop and keep the previous fit rather
            # than taking the mean of an empty set, which yields a NaN plane
            # and an empty mask that fails much later and much less clearly.
            if len(sel) < 3:
                break

            centroid = sel.mean(axis=0)
            # Normal is the least-variance direction of the centred points.
            # Taken from the eigenvector of the 3x3 scatter matrix rather
            # than from an SVD of the full (N,3): identical answer, but the
            # SVD dominated the whole detector at 151 ms of 207 ms per frame
            # on a 640x480 image, which is a third of the control cycle spent
            # on a decomposition of a matrix that is only ever rank 3.
            centred = sel - centroid
            scatter = centred.T @ centred
            normal = np.linalg.eigh(scatter)[1][:, 0]
            offset = float(normal @ centroid)

            residual = pts @ normal - offset
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            if scale <= 1e-9:
                break
            if previous is not None:
                moved = abs(offset - previous[1]) + float(
                    np.linalg.norm(normal - previous[0]))
                if moved < 1e-5:
                    break
            previous = (normal.copy(), offset)

            tighter = np.abs(residual) < 2.5 * scale
            if tighter.sum() < 3:
                break
            keep = tighter

        # Point the normal at the camera so "above the table" is positive.
        _, origin = self._rays_for((self.camera.height, self.camera.width))
        if normal @ origin - offset < 0:
            normal, offset = -normal, -offset
        return normal, offset

    # -- detection -----------------------------------------------------------

    def detect(
        self,
        depth: np.ndarray,
        self_mask: np.ndarray | None = None,
    ) -> list[Detection]:
        """Every object standing off the dominant plane.

        Args:
            depth: (h, w) float array of metres.
            self_mask: optional (h, w) bool array, True where the agent's own
                body is. Render it from forward kinematics if you want the
                principled version of "that is me, not the world"; leave it
                None to fall back on the border rule.
        """
        c = self.config
        depth = np.asarray(depth, dtype=np.float64)
        h, w = depth.shape

        points = self.back_project(depth)
        normal, offset = self.fit_plane(points)

        # Rays that hit nothing come back as inf, and inf * normal is a warning
        # followed by an inf standoff, which compares as "above the plane" and
        # would be counted as an object. It happened to be filtered out later
        # by the isfinite mask; making it explicit here means the fallback is
        # deliberate rather than a coincidence of ordering.
        finite = np.isfinite(points).all(axis=1)
        standoff = np.full(len(points), -np.inf)
        standoff[finite] = points[finite] @ normal - offset
        standoff = standoff.reshape(h, w)

        above = standoff > c.plane_tolerance
        if self_mask is not None:
            above &= ~np.asarray(self_mask, dtype=bool)

        labels, n = ndimage.label(above)
        if n == 0:
            return []

        detections: list[Detection] = []
        max_pixels = c.max_pixel_fraction * depth.size

        for label in range(1, n + 1):
            ys, xs = np.nonzero(labels == label)
            if len(ys) < c.min_pixels or len(ys) > max_pixels:
                continue

            m = c.border_margin
            touches = bool(
                (xs <= m).any() or (xs >= w - 1 - m).any()
                or (ys <= m).any() or (ys >= h - 1 - m).any()
            )
            if touches and c.reject_border:
                continue

            flat = ys * w + xs
            blob_pts = points[flat]
            blob_standoff = standoff[ys, xs]

            # Average only the top slice, so an oblique view does not drag
            # the estimate onto the near face of the object.
            top = blob_standoff.max()
            band = max(c.top_slice * top, 1e-4)
            crown = blob_standoff >= top - band
            if crown.sum() < 3:
                crown = np.ones_like(blob_standoff, dtype=bool)

            axis_xy = blob_pts[crown][:, :2].mean(axis=0)

            detections.append(Detection(
                xy=axis_xy,
                height=float(top),
                n_pixels=int(len(ys)),
                touches_border=touches,
                centroid_px=(float(xs.mean()), float(ys.mean())),
            ))

        detections.sort(key=lambda d: -d.n_pixels)
        return detections


class ObjectTracker:
    """Follows detections between frames, and survives occlusion.

    Object permanence, and it is here for an entirely practical reason: even
    with the camera moved off vertical, the arm still covers an object
    sometimes, and a detector alone reports that as the object ceasing to
    exist. An agent whose world model deletes things when a limb passes in
    front of them cannot act on them. Holding the last known position across
    a gap is the cheapest possible fix and it is also, not coincidentally,
    the thing infants take several months to develop.
    """

    def __init__(
        self,
        gate: float = 0.10,
        max_missing: int = 30,
        arm_exclusion_radius: float = 0.15,
        max_speed: float = 0.05,
        max_gate: float = 0.11,
    ) -> None:
        self.gate = gate
        """Metres. A detection further than this from every existing track
        starts a new track instead of being matched to a distant one."""
        self.max_speed = max_speed
        """Metres per second an object is assumed capable of moving, used to
        widen the gate by the elapsed time between frames."""
        self.max_gate = max_gate
        """Hard ceiling on the widened gate. The objects here are 0.24 m
        apart, so the ceiling has to stay strictly below half that spacing:
        at exactly half, a track sitting midway between two objects can match
        either, and silently following the wrong object is a far worse
        failure than briefly reporting one object twice."""
        self.max_missing = max_missing
        self.arm_exclusion_radius = arm_exclusion_radius
        """Motion is only counted toward independent_motion_std while the
        tool is at least this far away. Otherwise a push would be recorded
        as the object having moved by itself, which inverts the very
        distinction the drifter exists to test."""
        self.tracks: list[Track] = []

    def update(
        self,
        detections: list[Detection],
        tool_xy: np.ndarray | None = None,
        dt: float | None = None,
    ) -> list[Track]:
        """Associate this frame's detections with the existing tracks.

        `dt` is the seconds since the previous update. It widens the
        association gate, and it matters more than it looks: the detector
        does not run at a fixed rate. Measured live, the node ticked at a
        median 0.54 Hz with gaps as long as 14.9 s, and across a gap like
        that a slowly drifting object crosses a fixed gate, gets treated as
        new, and is reported alongside the stale track it should have
        matched. Three objects were being counted as five.

        The widening is capped at half the spacing between objects, because
        past that point a gate large enough to follow one object is also
        large enough to jump to its neighbour.
        """
        gate = self.gate
        if dt is not None:
            gate = min(self.gate + self.max_speed * float(dt), self.max_gate)

        unmatched = list(range(len(detections)))

        for track in self.tracks:
            best, best_dist = None, gate
            for i in unmatched:
                dist = float(np.linalg.norm(detections[i].xy - track.xy))
                if dist < best_dist:
                    best, best_dist = i, dist

            if best is None:
                track.visible = False
                track.frames_missing += 1
                continue

            det = detections[best]
            unmatched.remove(best)

            far_from_tool = (
                tool_xy is None
                or float(np.linalg.norm(det.xy - np.asarray(tool_xy)[:2]))
                > self.arm_exclusion_radius
            )
            if far_from_tool and track.visible:
                # Actual displacement, not the residual against the
                # prediction: this statistic is about how much the object
                # moves on its own, and a good prediction must not make a
                # drifting object look stationary.
                track._free_motion.append(best_dist)

            track.xy = det.xy
            track.height = det.height
            track.visible = True
            track.frames_seen += 1
            track.frames_missing = 0

        for i in unmatched:
            self.tracks.append(Track(xy=detections[i].xy, height=detections[i].height))

        self.tracks = [
            t for t in self.tracks if t.frames_missing <= self.max_missing
        ]
        return self.tracks
