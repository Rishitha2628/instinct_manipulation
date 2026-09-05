"""UR5 + depth camera tests. Run: python3 -m pytest tests/test_ur5.py -v"""
import numpy as np
import pytest

from intrinsic_core import (
    BatchedDepthEmpowerment, CameraConfig, DepthCamera, DepthDiscretiser,
    EmpowermentConfig, EmpowermentEstimator, UR5, UR5Config,
)

PUCK = np.array([-0.50, 0.0])


def _arm():
    return UR5(UR5Config(dq=0.10, active_joints=(0, 1, 2, 3)))


def _camera(width=16, height=12, fov=30.0):
    return DepthCamera(CameraConfig(
        position=(-0.50, 0.0, 0.95), look_at=(-0.50, 0.0, 0.0),
        up=(1.0, 0.0, 0.0), fov_deg=fov, width=width, height=height,
    ))


def _state(arm, xyz):
    q = arm.inverse_kinematics(xyz)
    assert q is not None, xyz
    return np.concatenate([q, PUCK])


def _estimate(arm, camera, state, steps=2, depth_res=0.02, exhaustive=True):
    cfg = EmpowermentConfig(
        n_steps=steps, n_sequences=2000,
        exhaustive_below=10000 if exhaustive else 0, seed=0,
    )
    return BatchedDepthEmpowerment(
        arm, camera, DepthDiscretiser(depth_res), cfg
    ).estimate(state)


# --- kinematics -------------------------------------------------------------

def test_zero_pose_matches_ur5_geometry():
    """UR5 at all-zeros is fully extended horizontally."""
    tool = _arm().tool_position(np.zeros(6))
    assert tool[0] == pytest.approx(-0.8172, abs=1e-3)
    assert abs(tool[2]) < 0.02


def test_reach_is_within_ur5_envelope():
    arm = _arm()
    qs = np.random.default_rng(0).uniform(-np.pi, np.pi, (2000, 6))
    r = np.linalg.norm([arm.tool_position(q) for q in qs], axis=1)
    assert r.max() < 1.10          # wrist offset beyond the 0.85 wrist-centre spec
    assert r.max() > 0.85


def test_ik_inverts_fk():
    arm = _arm()
    for target in [(-0.50, 0.0, 0.12), (-0.45, 0.15, 0.12), (-0.55, -0.15, 0.20)]:
        q = arm.inverse_kinematics(target)
        assert q is not None, target
        assert arm.tool_position(q) == pytest.approx(target, abs=1e-3)


def test_link_positions_has_seven_frames():
    assert _arm().link_positions(np.zeros(6)).shape == (7, 3)


# --- depth camera -----------------------------------------------------------

def test_depth_image_shape_and_range():
    cam = _camera()
    img = cam.render(_arm().link_positions(np.zeros(6)), PUCK, 0.04)
    assert img.shape == (12, 16)
    assert img.min() >= cam.config.z_near
    assert img.max() <= cam.config.z_far


def test_puck_is_visible_when_the_arm_is_not_in_the_way():
    """The puck must appear in the image, or nothing downstream works.

    Note the arm must be off to the side. The camera looks straight DOWN at
    the puck, so an arm reaching to the puck's (x, y) occludes it completely
    -- see test_arm_occludes_puck_from_overhead_camera below.
    """
    cam = _camera()
    arm = _arm()
    aside = arm.inverse_kinematics((-0.30, 0.20, 0.35))
    assert aside is not None

    with_puck = cam.render(arm.link_positions(aside), PUCK, 0.04)
    without = cam.render(arm.link_positions(aside), np.array([5.0, 5.0]), 0.04)

    changed = np.abs(with_puck - without) > 1e-6
    assert changed.sum() > 3, "puck should occupy several pixels"
    assert with_puck[changed].min() < without[changed].min()


def test_arm_occludes_puck_from_overhead_camera():
    """A real limitation of this camera placement, documented not fixed.

    With the camera directly above the puck, an arm reaching down to the
    puck's (x, y) blocks the view entirely: zero pixels differ between a
    scene with the puck and one without.

    Empowerment still finds the puck, because pushing displaces it OUT from
    under the arm where the camera can see it again -- the visible signal is
    the puck's displacement, not the puck itself. But an oblique camera
    would be the better choice for a real rig.
    """
    cam = _camera()
    arm = _arm()
    overhead_blocked = arm.inverse_kinematics((-0.50, 0.0, 0.45))
    assert overhead_blocked is not None

    with_puck = cam.render(arm.link_positions(overhead_blocked), PUCK, 0.04)
    without = cam.render(
        arm.link_positions(overhead_blocked), np.array([5.0, 5.0]), 0.04
    )
    assert np.abs(with_puck - without).max() < 1e-9


def test_batch_render_matches_single_render():
    cam = _camera()
    arm = _arm()
    poses = np.array([arm.link_positions(q) for q in [
        np.zeros(6), np.array([0.0, -1.2, 1.4, -1.75, -1.57, 0.0]),
    ]])
    pucks = np.array([PUCK, PUCK + 0.1])
    batch = cam.render_batch(poses, pucks, 0.04)
    for i in range(2):
        single = cam.render(poses[i], pucks[i], 0.04)
        assert batch[i] == pytest.approx(single, abs=1e-9)


# --- the core claim ---------------------------------------------------------

def test_empowerment_peaks_at_the_puck():
    """The result: the arm identifies the object with no goal given."""
    arm, cam = _arm(), _camera()
    near = _estimate(arm, cam, _state(arm, (-0.50, 0.0, 0.08)))
    far = _estimate(arm, cam, _state(arm, (-0.50, 0.0, 0.35)))
    assert near > far + 1.0


def test_bolted_puck_removes_the_peak():
    """The paper's immovable-box control: influence, not mere presence."""
    arm, cam = _arm(), _camera()
    state = _state(arm, (-0.50, 0.0, 0.08))
    pushable = _estimate(arm, cam, state)
    bolted = _estimate(arm.with_fixed_puck(), cam, state)
    assert pushable - bolted > 1.0


def test_puck_contribution_vanishes_with_distance():
    arm, cam = _arm(), _camera()
    bolt = arm.with_fixed_puck()
    state = _state(arm, (-0.50, 0.0, 0.30))
    assert abs(_estimate(arm, cam, state) - _estimate(bolt, cam, state)) < 0.3


def test_puck_contribution_decays_with_sensor_resolution():
    """Sensor resolution alone can produce the 'cannot perceive' condition.

    The threshold is geometric, but not at one pixel per puck. Rendering here
    is ray-cast, not area-averaged: a ray either intersects the puck or it
    does not, so an object smaller than a pixel still flips that pixel's
    whole value whenever a ray happens to land on it. What kills the signal
    is ray SPACING coarse enough that no ray lands on the puck at all, which
    is roughly three times the puck's diameter rather than one.

    With a 30 deg horizontal FOV at 0.95 m the sampling pitch at the table
    is 0.509 m / width. Measured contribution against the 8 cm puck:

        16x12   3.2 cm pitch   +1.21 bits
         8x6    6.4 cm         +0.93
         6x4    8.5 cm         +0.87
         4x3   12.7 cm         +0.53
         2x2   25.5 cm         +0.00   nothing resolved

    Nobody imposed the paper's 'box is not perceivable' condition here. It
    falls out of the optics, which is a stronger form of the same point.

    These numbers moved when CameraConfig.fov_deg was corrected to mean the
    horizontal FOV; they were previously measured with the model seeing 38.6
    deg while claiming 30. The effect is unchanged, the thresholds are not.
    """
    arm = _arm()
    state = _state(arm, (-0.50, 0.0, 0.08))
    bolted_arm = arm.with_fixed_puck()

    def contribution(width, height):
        cam = _camera(width=width, height=height)
        return (
            _estimate(arm, cam, state, exhaustive=False)
            - _estimate(bolted_arm, cam, state, exhaustive=False)
        )

    resolved = contribution(16, 12)
    unresolved = contribution(2, 2)

    assert resolved > 0.5
    assert unresolved < 0.2
    assert resolved > unresolved * 3


# --- estimator agreement ----------------------------------------------------

def test_batched_matches_reference_estimator():
    """The fast path must not change the answer."""
    arm, cam = _arm(), _camera()
    state = _state(arm, (-0.50, 0.0, 0.08))
    cfg = EmpowermentConfig(n_steps=2, exhaustive_below=10000, seed=0)
    disc = DepthDiscretiser(0.02)

    def rollout(s, actions):
        s2 = arm.rollout_state(s, actions)
        return cam.render(arm.link_positions(s2[:6]), s2[6:8], arm.config.puck_radius)

    reference = EmpowermentEstimator(rollout, arm.n_actions, disc, cfg).estimate(state)
    batched = BatchedDepthEmpowerment(arm, cam, disc, cfg).estimate(state)
    assert batched == pytest.approx(reference, abs=1e-9)


def test_action_count_matches_active_joints():
    assert UR5(UR5Config(active_joints=(0, 1, 2, 3))).n_actions == 81
    assert UR5(UR5Config(active_joints=(0, 1, 2))).n_actions == 27
