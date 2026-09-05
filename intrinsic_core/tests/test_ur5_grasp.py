"""The grasp experiment on the real UR5 kinematics.

grasp.py established the result with a free-flying tool. This checks it
survives an actual 6-DOF arm, where directions are not equally cheap and the
set of futures available while holding something need not be the same shape
as the set available while hovering.

It does survive, and more strongly: the own-sensor penalty for grasping grows
from -0.13 bits on the toy to -2.66 here.
"""

import numpy as np
import pytest

from intrinsic_core.sensors import CameraConfig, DepthCamera
from intrinsic_core.ur5 import UR5, UR5Config
from intrinsic_core.ur5_grasp import HELD, OBJ, Q, UR5GraspWorld

OBJECT_XY = np.array([-0.68, 0.0])


def _world(**kw):
    arm = UR5(UR5Config(dq=0.10, active_joints=(0, 1, 2, 3), tool_radius=0.09,
                        puck_radius=0.04, table_height=-0.25))
    return UR5GraspWorld(arm, **kw)


def _camera(world):
    return DepthCamera(CameraConfig(
        position=(-1.45, 0.0, 0.35), look_at=(-0.68, 0.0, -0.21),
        up=(0.0, 0.0, 1.0), fov_deg=50.0, width=16, height=12,
        table_height=world.arm.config.table_height))


def _at_object(world, **kw):
    q = world.arm.inverse_kinematics(
        np.array([OBJECT_XY[0], OBJECT_XY[1], world.rest_height()]))
    assert q is not None
    return world.state_at(q, OBJECT_XY, **kw)


# --- mechanics ---------------------------------------------------------------

def test_close_grasps_at_the_object_but_not_far_away():
    w = _world()
    assert w.step(_at_object(w), w.close_action)[HELD] == 1.0

    q = w.arm.inverse_kinematics(np.array([-0.40, 0.0, w.rest_height() + 0.05]))
    assert w.step(w.state_at(q, OBJECT_XY), w.close_action)[HELD] == 0.0


def test_held_object_follows_the_tool_and_leaves_the_table():
    """Impossible to express in ur5.py, where the object is (x, y) only."""
    w = _world()
    held = w.step(_at_object(w), w.close_action)
    highest = max(range(w.n_joint_actions),
                  key=lambda a: w.arm.tool_position(w.step(held, a)[Q])[2])
    moved = w.step(held, highest)

    assert np.allclose(moved[OBJ][:2], w.arm.tool_position(moved[Q])[:2])
    assert moved[OBJ][2] > w.rest_height() + 0.05


def test_release_drops_it_and_unheld_object_never_rises():
    w = _world()
    held = w.step(_at_object(w), w.close_action)
    highest = max(range(w.n_joint_actions),
                  key=lambda a: w.arm.tool_position(w.step(held, a)[Q])[2])
    lifted = w.step(held, highest)
    assert w.step(lifted, w.open_action)[OBJ][2] == pytest.approx(w.rest_height())

    open_state = _at_object(w)
    assert w.rollout(open_state, [highest, highest])[OBJ][2] == pytest.approx(
        w.rest_height())


# --- the result --------------------------------------------------------------

def _after(world, action):
    return world.step(_at_object(world), action)


def test_transfer_empowerment_makes_grasping_the_best_action():
    """Scored on control of the OBJECT, closing beats opening and moving.

    Over all 83 actions, CLOSE ranks 1st. Checked here against OPEN and a
    spread of joint moves rather than all of them, to keep the suite fast.
    """
    w = _world()
    close = w.empowerment(_after(w, w.close_action), 2, "object")
    other = [w.empowerment(_after(w, a), 2, "object")
             for a in [w.open_action, 0, 20, 40, 55, 80]]
    assert close > max(other)


def test_own_sensor_empowerment_makes_grasping_the_worst_action():
    """Scored on the depth image, the same grasp ranks LAST of 83.

    Not a near miss: -2.66 bits against simply leaving the gripper open. The
    project's 'empowerment will not pick things up' is real, and this is it.
    """
    w = _world()
    cam = _camera(w)
    close = w.empowerment(_after(w, w.close_action), 2, "depth", camera=cam)
    other = [w.empowerment(_after(w, a), 2, "depth", camera=cam)
             for a in [w.open_action, 0, 20, 40, 55, 80]]
    assert close < min(other)


def test_bolting_the_object_removes_the_penalty_for_grasping():
    """The mechanism, isolated.

    Holding is not what own-sensor dislikes; losing the ability to PUSH is.
    Shoving gives the object semi-independent motion and so adds variety to
    the picture, and grasping destroys that independence by making the object
    a rigid function of the tool. Bolt the object, and hovering has nothing
    left to offer: the penalty disappears.
    """
    w = _world()
    cam = _camera(w)
    penalty = (w.empowerment(_at_object(w, grasped=True), 2, "depth", camera=cam)
               - w.empowerment(_at_object(w), 2, "depth", camera=cam))
    assert penalty < -1.0

    b = w.bolted()
    bolted_penalty = (
        b.empowerment(_at_object(b, grasped=True), 2, "depth", camera=_camera(b))
        - b.empowerment(_at_object(b), 2, "depth", camera=_camera(b)))
    assert bolted_penalty > penalty + 2.0
