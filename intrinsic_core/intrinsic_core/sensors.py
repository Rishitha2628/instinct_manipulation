"""
A depth camera, rendered analytically.

Why this file exists at all: empowerment is defined as the channel capacity
from the agent's ACTUATORS to its own SENSORS (Salge et al. sec 4.3). Using
ground-truth object pose as the outcome quietly skips the sensor half and
measures something easier. This renders what a depth camera would actually
see, so the outcome vector is a sensor reading.

That distinction has teeth. With ground-truth pose, the puck's contribution
is always visible. Through a camera it is not: if the arm occludes the puck,
or the puck moves less than one depth quantum, the sensor cannot resolve the
difference and those outcomes collapse into one. Empowerment then correctly
reports that the agent has no distinguishable influence -- which is the
paper's point about perception and action being two halves of one quantity.

Rendering is ray-cast against analytic primitives (a table plane, a sphere
for the puck, spheres along the arm links). No physics engine, no GPU, and
it vectorises to a few milliseconds at the resolutions used here. Replace
`DepthCamera.render` with a PyBullet `getCameraImage` call when you move to
a real simulator; the interface is a depth array either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CameraConfig", "DepthCamera", "DepthDiscretiser"]


@dataclass
class CameraConfig:
    position: tuple[float, float, float] = (0.35, -0.75, 0.85)
    look_at: tuple[float, float, float] = (-0.45, 0.0, 0.05)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)

    width: int = 32
    height: int = 24
    """Deliberately tiny. A 640x480 depth image has 307200 dimensions and
    every rollout differs in some pixel, so nothing ever collapses and
    empowerment saturates. Downsampling IS the sensor model: it says what
    the agent can actually resolve."""

    fov_deg: float = 58.0
    """HORIZONTAL field of view, matching SDF's <horizontal_fov>, ROS
    CameraInfo, and how every depth-camera datasheet quotes the number.

    This used to be read as the VERTICAL fov (half_h was taken straight from
    it and half_w scaled up by the aspect ratio), which meant the analytic
    camera and the Gazebo one never had the same optics even when the two
    config files agreed on '30'. At 4:3 the model was actually seeing 38.6
    deg horizontally against the simulator's 30. Nothing errored; the agent
    simply predicted through a lens it did not have."""
    z_near: float = 0.1
    z_far: float = 2.5

    table_height: float = 0.0
    arm_sphere_radius: float = 0.05


class DepthCamera:
    """Pinhole depth camera over a table scene."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._rays = self._build_rays()
        self._origin = np.array(self.config.position, dtype=np.float64)

    # -- ray setup ----------------------------------------------------------

    def _build_rays(self) -> np.ndarray:
        c = self.config
        eye = np.array(c.position, dtype=np.float64)
        target = np.array(c.look_at, dtype=np.float64)
        up = np.array(c.up, dtype=np.float64)

        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)

        aspect = c.width / c.height
        half_w = np.tan(np.radians(c.fov_deg) / 2.0)
        half_h = half_w / aspect

        # pixel centres in [-1, 1]
        xs = (np.arange(c.width) + 0.5) / c.width * 2.0 - 1.0
        ys = 1.0 - (np.arange(c.height) + 0.5) / c.height * 2.0
        gx, gy = np.meshgrid(xs, ys)

        dirs = (
            forward[None, None, :]
            + (gx * half_w)[..., None] * right[None, None, :]
            + (gy * half_h)[..., None] * true_up[None, None, :]
        )
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        return dirs.reshape(-1, 3)

    # -- intersections ------------------------------------------------------

    def _hit_plane(self, z: float) -> np.ndarray:
        """Distance to the horizontal plane at height z, inf where no hit."""
        dz = self._rays[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - self._origin[2]) / dz
        return np.where((dz != 0) & (t > 0), t, np.inf)

    def _hit_spheres(self, centres: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Nearest sphere hit per ray. centres (n,3), radii (n,)."""
        if len(centres) == 0:
            return np.full(len(self._rays), np.inf)

        oc = self._origin[None, None, :] - centres[None, :, :]      # (1,n,3)
        b = np.einsum("rd,rnd->rn", self._rays, np.broadcast_to(
            oc, (len(self._rays), len(centres), 3)))
        c_term = (oc ** 2).sum(axis=-1) - radii[None, :] ** 2
        disc = b ** 2 - c_term

        with np.errstate(invalid="ignore"):
            sqrt_disc = np.where(disc >= 0, np.sqrt(np.maximum(disc, 0.0)), np.nan)
            t0 = -b - sqrt_disc
            t1 = -b + sqrt_disc

        t = np.where(t0 > 1e-6, t0, np.where(t1 > 1e-6, t1, np.inf))
        t = np.where(np.isfinite(t), t, np.inf)
        return t.min(axis=1)

    # -- rendering ----------------------------------------------------------

    def render(
        self,
        arm_points: np.ndarray,
        puck_xy: np.ndarray,
        puck_radius: float,
        arm_radius: float | None = None,
    ) -> np.ndarray:
        """Depth image, shape (height, width), clipped to [z_near, z_far].

        Args:
            arm_points: (k,3) points along the arm; drawn as spheres.
            puck_xy:    (2,) puck centre on the table.
        """
        c = self.config
        arm_radius = c.arm_sphere_radius if arm_radius is None else arm_radius

        depth = self._hit_plane(c.table_height)

        puck_centre = np.array(
            [[puck_xy[0], puck_xy[1], c.table_height + puck_radius]]
        )
        depth = np.minimum(
            depth, self._hit_spheres(puck_centre, np.array([puck_radius]))
        )

        arm_points = np.atleast_2d(np.asarray(arm_points, dtype=np.float64))
        if len(arm_points):
            depth = np.minimum(
                depth,
                self._hit_spheres(
                    arm_points, np.full(len(arm_points), arm_radius)
                ),
            )

        depth = np.clip(depth, c.z_near, c.z_far)
        return depth.reshape(c.height, c.width)


    def render_batch(
        self,
        arm_points: np.ndarray,
        puck_xy: np.ndarray,
        puck_radius: float,
        arm_radius: float | None = None,
    ) -> np.ndarray:
        """Render N scenes at once. Much faster than looping `render`.

        Args:
            arm_points: (N, k, 3) arm points per scene.
            puck_xy:    (N, 2) puck centre per scene.

        Returns:
            (N, height, width) depth images.
        """
        c = self.config
        arm_radius = c.arm_sphere_radius if arm_radius is None else arm_radius

        arm_points = np.asarray(arm_points, dtype=np.float64)
        puck_xy = np.asarray(puck_xy, dtype=np.float64)
        n_scenes, k, _ = arm_points.shape

        puck_centres = np.stack([
            puck_xy[:, 0],
            puck_xy[:, 1],
            np.full(n_scenes, c.table_height + puck_radius),
        ], axis=-1)[:, None, :]

        centres = np.concatenate([arm_points, puck_centres], axis=1)   # (N,k+1,3)
        radii = np.concatenate([np.full(k, arm_radius), [puck_radius]])

        oc = self._origin[None, None, :] - centres                     # (N,k+1,3)
        b = np.einsum("rd,nsd->rns", self._rays, oc)                   # (R,N,k+1)
        c_term = (oc ** 2).sum(axis=-1)[None, :, :] - (radii ** 2)[None, None, :]
        disc = b ** 2 - c_term

        with np.errstate(invalid="ignore"):
            sq = np.sqrt(np.maximum(disc, 0.0))
            t0, t1 = -b - sq, -b + sq
        t = np.where(
            disc < 0, np.inf,
            np.where(t0 > 1e-6, t0, np.where(t1 > 1e-6, t1, np.inf)),
        )

        depth = t.min(axis=2)                                          # (R,N)
        depth = np.minimum(depth, self._hit_plane(c.table_height)[:, None])
        depth = np.clip(depth, c.z_near, c.z_far)

        return depth.T.reshape(n_scenes, c.height, c.width)


@dataclass
class DepthDiscretiser:
    """Quantise a depth image into a hashable outcome key.

    `depth_resolution` is the sensor's real depth accuracy -- roughly 5 mm
    for a RealSense D435 at half a metre, though it degrades quadratically
    with range. Setting it far below the true noise floor is the mistake
    that makes every map saturate: you are then counting sensor noise as
    distinguishable influence.
    """

    depth_resolution: float = 0.01
    ignore_beyond: float = 2.0

    def __call__(self, depth_image: np.ndarray) -> tuple:
        d = np.asarray(depth_image, dtype=np.float64).ravel()
        d = np.where(d > self.ignore_beyond, self.ignore_beyond, d)
        return tuple(np.round(d / self.depth_resolution).astype(np.int32))
