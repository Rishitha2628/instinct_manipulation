"""Locating objects from a depth image, with nobody saying where they are.

These run against frames rendered by intrinsic_core's own DepthCamera, not
against Gazebo, which is the point: the detector is testable in milliseconds
without a simulator, and a failure here is a geometry bug rather than a
bringup problem.

The camera here is the oblique one the world file now uses, because the whole
class of error this guards against -- confusing range along the ray with
depth along the optical axis, and biasing an object's position toward the
camera -- is invisible when the camera looks straight down.
"""

import numpy as np
import pytest

from intrinsic_core.perception import (
    DepthSegmenter,
    ObjectTracker,
    SegmenterConfig,
    Track,
)
from intrinsic_core.sensors import CameraConfig, DepthCamera

TABLE = -0.25
PUCK_R = 0.04

# Matches config/empowerment.yaml and the <pose> in the world file: 36 deg
# below horizontal, looking back along +x toward the arm.
CAM_POS = (-1.45, 0.0, 0.35)
CAM_LOOK = (-0.68, 0.0, -0.21)
FOV = 50.0


def _camera(width=96, height=72):
    return CameraConfig(
        position=CAM_POS, look_at=CAM_LOOK, width=width, height=height,
        fov_deg=FOV, table_height=TABLE,
    )


def _render(dc, spheres):
    """Depth image of an arbitrary set of spheres over the table.

    DepthCamera.render only takes one puck; the scene here needs three
    objects plus an arm, so this reaches for the same primitives it does.
    """
    c = dc.config
    depth = dc._hit_plane(c.table_height)
    if spheres:
        centres = np.array([s[0] for s in spheres], dtype=np.float64)
        radii = np.array([s[1] for s in spheres], dtype=np.float64)
        depth = np.minimum(depth, dc._hit_spheres(centres, radii))
    return np.clip(depth, c.z_near, c.z_far).reshape(c.height, c.width)


def _object(xy, radius=PUCK_R, lift=0.0):
    return (np.array([xy[0], xy[1], TABLE + radius + lift]), radius)


def _segmenter(cam, **kw):
    return DepthSegmenter(cam, SegmenterConfig(depth_is_range=True, **kw))


# --- plane -------------------------------------------------------------------

def test_plane_fit_recovers_the_table():
    """The support surface is found, not assumed."""
    cam = _camera()
    img = _render(DepthCamera(cam), [_object((-0.68, 0.0))])
    normal, offset = _segmenter(cam).fit_plane(_segmenter(cam).back_project(img))

    assert np.allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-3)
    assert offset == pytest.approx(TABLE, abs=1e-3)


def test_plane_fit_is_not_dragged_by_the_objects():
    """Three objects in view must not tilt the fitted surface."""
    cam = _camera()
    img = _render(DepthCamera(cam), [
        _object((-0.68, 0.0)), _object((-0.68, 0.24)), _object((-0.68, -0.24)),
    ])
    normal, offset = _segmenter(cam).fit_plane(_segmenter(cam).back_project(img))
    assert np.allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-3)
    assert offset == pytest.approx(TABLE, abs=1e-3)


# --- locating ----------------------------------------------------------------

def test_single_object_located_to_a_centimetre():
    cam = _camera()
    truth = np.array([-0.68, 0.0])
    img = _render(DepthCamera(cam), [_object(truth)])

    dets = _segmenter(cam).detect(img)
    assert len(dets) == 1
    assert np.linalg.norm(dets[0].xy - truth) < 0.02


def test_all_three_objects_found_and_are_indistinguishable():
    """Perception must find all three and NOT be able to rank them.

    This is the experiment's premise: the pushable puck, the bolted block and
    the drifter are the same object as far as any depth image is concerned.
    If the detector could tell them apart, empowerment would not be doing the
    work the project claims it does.
    """
    cam = _camera()
    truths = [np.array(t) for t in [(-0.68, 0.0), (-0.68, 0.24), (-0.68, -0.24)]]
    img = _render(DepthCamera(cam), [_object(t) for t in truths])

    dets = _segmenter(cam).detect(img)
    assert len(dets) == 3

    for truth in truths:
        nearest = min(dets, key=lambda d: np.linalg.norm(d.xy - truth))
        assert np.linalg.norm(nearest.xy - truth) < 0.02

    # Same size and same standoff, to within the sampling grid.
    assert max(d.height for d in dets) - min(d.height for d in dets) < 0.01
    assert max(d.n_pixels for d in dets) <= 2 * min(d.n_pixels for d in dets)


def test_lift_is_visible_as_height():
    """An oblique camera can see that something was picked up.

    From straight overhead this test could not exist: raising the object
    changes its depth slightly and nothing else. Here the standoff from the
    table plane tracks the lift directly, which is what any later grasping
    work has to be able to measure.
    """
    cam = _camera()
    flat = _segmenter(cam).detect(
        _render(DepthCamera(cam), [_object((-0.68, 0.0))]))[0]
    high = _segmenter(cam).detect(
        _render(DepthCamera(cam), [_object((-0.68, 0.0), lift=0.10)]))[0]

    assert high.height - flat.height == pytest.approx(0.10, abs=0.02)
    # and it did not slide sideways while being lifted
    assert np.linalg.norm(high.xy - flat.xy) < 0.02


# --- the arm -----------------------------------------------------------------

def _arm_chain():
    """Spheres from the object out to the arm's base, so it leaves the frame."""
    return [(np.array([x, 0.0, TABLE + 0.06]), 0.05)
            for x in np.linspace(-0.62, 0.30, 14)]


def test_arm_is_rejected_by_the_border_rule():
    """The arm reaches in from outside; the objects do not."""
    cam = _camera()
    img = _render(DepthCamera(cam), [_object((-0.68, 0.24))] + _arm_chain())

    kept = _segmenter(cam).detect(img)
    assert len(kept) == 1
    assert np.linalg.norm(kept[0].xy - np.array([-0.68, 0.24])) < 0.03

    # With the rule off, the arm comes back as an extra blob touching the edge.
    everything = _segmenter(cam, reject_border=False).detect(img)
    assert len(everything) > len(kept)
    assert any(d.touches_border for d in everything)


def test_self_mask_excludes_the_arm():
    """The principled alternative: predict your own body and subtract it."""
    cam = _camera()
    dc = DepthCamera(cam)
    arm = _arm_chain()

    img = _render(dc, [_object((-0.68, 0.24))] + arm)
    arm_only = _render(dc, arm)
    table = _render(dc, [])
    mask = np.abs(arm_only - table) > 1e-6

    dets = _segmenter(cam, reject_border=False).detect(img, self_mask=mask)
    assert len(dets) == 1
    assert np.linalg.norm(dets[0].xy - np.array([-0.68, 0.24])) < 0.03


# --- depth convention --------------------------------------------------------

def test_along_axis_depth_is_converted_to_range():
    """z-depth in, same answer out.

    gz-sim publishes distance along the optical axis; DepthCamera returns
    distance along the ray. With this oblique camera cos(theta) runs from
    0.87 to 1.0 across the frame, so the two differ by up to 30 cm at the
    corners. See the test below for what goes wrong if they are confused.
    """
    cam = _camera()
    dc = DepthCamera(cam)
    truth = np.array([-0.68, 0.0])
    ranges = _render(dc, [_object(truth)])

    forward = np.array(CAM_LOOK) - np.array(CAM_POS)
    forward = forward / np.linalg.norm(forward)
    cos_theta = (dc._rays @ forward).reshape(cam.height, cam.width)
    z_depth = ranges * cos_theta

    assert np.abs(z_depth - ranges).max() > 0.01, "conversion must be non-trivial"

    as_z = DepthSegmenter(cam, SegmenterConfig(depth_is_range=False)).detect(z_depth)
    as_range = _segmenter(cam).detect(ranges)
    assert len(as_z) == len(as_range) == 1
    assert np.linalg.norm(as_z[0].xy - as_range[0].xy) < 5e-3


def test_wrong_depth_convention_corrupts_height_not_position():
    """Guards the failure mode above, and pins down what it actually breaks.

    The intuition that the reconstructed table comes out tilted is wrong: it
    stays almost level (0.13 deg) and the object's (x, y) is barely affected.
    What goes is the SCALE along the view direction. Measured here, feeding
    z-depth to a segmenter that expects range puts the plane 3.2 cm out and
    shrinks an 8 cm object to 4.9 cm.

    That is the nastier failure. Position still looks right, so a detector
    smoke test passes, while the one quantity that tells you an object has
    been picked up is silently 38 percent low.
    """
    cam = _camera()
    dc = DepthCamera(cam)
    ranges = _render(dc, [_object((-0.68, 0.0))])
    forward = np.array(CAM_LOOK) - np.array(CAM_POS)
    forward = forward / np.linalg.norm(forward)
    z_depth = ranges * (dc._rays @ forward).reshape(cam.height, cam.width)

    right = _segmenter(cam).detect(ranges)[0]
    wrong = _segmenter(cam).detect(z_depth)[0]        # claims range, gets z

    assert right.height == pytest.approx(2 * PUCK_R, abs=0.005)
    assert wrong.height < 0.7 * right.height
    # position survives, which is exactly why this is easy to miss
    assert np.linalg.norm(wrong.xy - right.xy) < 0.01

    seg = _segmenter(cam)
    _, offset = seg.fit_plane(seg.back_project(z_depth))
    assert abs(offset - TABLE) > 0.02


# --- tracking ----------------------------------------------------------------

def test_track_survives_occlusion():
    """Object permanence: a limb passing in front is not the object ceasing."""
    tracker = ObjectTracker()
    cam = _camera()
    img = _render(DepthCamera(cam), [_object((-0.68, 0.0))])
    dets = _segmenter(cam).detect(img)

    tracker.update(dets)
    remembered = tracker.tracks[0].xy.copy()

    for _ in range(5):                       # object hidden by the arm
        tracker.update([])

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].visible is False
    assert np.allclose(tracker.tracks[0].xy, remembered)


def test_track_is_dropped_once_it_is_gone_for_good():
    tracker = ObjectTracker(max_missing=3)
    tracker.tracks = [Track(xy=np.array([-0.68, 0.0]), height=0.08)]
    for _ in range(5):
        tracker.update([])
    assert tracker.tracks == []


def test_independent_motion_separates_the_drifter_from_the_puck():
    """The one measurement that distinguishes them, learned by watching.

    Both objects move. Only one of them moves while nothing is near it, and
    the agent is never told which. This is what later lets the forward model
    treat the drifter's outcomes as action-independent, which is what drives
    its empowerment to zero while a curiosity signal would be drawn to it.
    """
    from intrinsic_core.perception import Detection

    def det(xy):
        return Detection(xy=np.array(xy), height=0.08, n_pixels=40,
                         touches_border=False, centroid_px=(0.0, 0.0))

    far_tool = np.array([0.30, 0.30])

    still = ObjectTracker()
    for _ in range(12):
        still.update([det((-0.68, 0.24))], tool_xy=far_tool)

    drifting = ObjectTracker()
    for i in range(12):
        y = -0.24 + 0.10 * np.sin(2 * np.pi * i / 6.0)
        drifting.update([det((-0.68, y))], tool_xy=far_tool)

    assert still.tracks[0].independent_motion_std == pytest.approx(0.0, abs=1e-9)
    assert drifting.tracks[0].independent_motion_std > 0.01


def test_pushing_is_not_counted_as_independent_motion():
    """A shoved object must not be mistaken for a self-moving one."""
    from intrinsic_core.perception import Detection

    tracker = ObjectTracker(arm_exclusion_radius=0.15)
    for i in range(12):
        xy = np.array([-0.68 + 0.01 * i, 0.0])
        # tool right on top of it: this motion is the agent's doing
        tracker.update(
            [Detection(xy=xy, height=0.08, n_pixels=40,
                       touches_border=False, centroid_px=(0.0, 0.0))],
            tool_xy=xy,
        )
    assert tracker.tracks[0].independent_motion_std == pytest.approx(0.0, abs=1e-9)


def test_gate_widens_with_elapsed_time():
    """A slow object crossing a long frame gap is still the same object.

    The detector does not run at a fixed rate; measured live it managed a
    median 0.54 Hz with gaps up to 14.9 s. Over a gap like that a drifting
    object moves further than a fixed association gate, is treated as new,
    and gets reported alongside the stale track it should have matched.
    """
    from intrinsic_core.perception import Detection

    def det(y):
        return Detection(xy=np.array([-0.68, y]), height=0.08, n_pixels=40,
                         touches_border=False, centroid_px=(0.0, 0.0))

    moved = 0.11          # further than the 0.10 m base gate

    fixed = ObjectTracker()
    fixed.update([det(-0.24)])
    fixed.update([det(-0.24 + moved)])
    assert len(fixed.tracks) == 2, "fixed gate should split, as it did live"

    aware = ObjectTracker()
    aware.update([det(-0.24)], dt=1.0)
    aware.update([det(-0.24 + moved)], dt=5.0)
    assert len(aware.tracks) == 1


def test_widened_gate_never_reaches_a_neighbour():
    """The widening must not let a track jump to the object next to it."""
    tracker = ObjectTracker()
    assert tracker.max_gate < 0.24 / 2, "objects here are 0.24 m apart"

    from intrinsic_core.perception import Detection
    tracker.update([Detection(xy=np.array([-0.68, -0.24]), height=0.08,
                              n_pixels=40, touches_border=False,
                              centroid_px=(0.0, 0.0))], dt=1.0)
    # its neighbour, 0.24 m away, after an absurd gap
    tracker.update([Detection(xy=np.array([-0.68, 0.0]), height=0.08,
                              n_pixels=40, touches_border=False,
                              centroid_px=(0.0, 0.0))], dt=600.0)
    assert len(tracker.tracks) == 2
