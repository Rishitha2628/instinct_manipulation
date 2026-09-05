"""Does an intrinsic objective prefer holding an object, or hovering by it?

The headline result of this file: it depends entirely on what you call the
outcome, and the two reasonable choices disagree in sign.

Everything here enumerates exhaustively -- 8 macro-actions at horizon 3 is
512 sequences -- so these are true values, not the sampled lower bounds the
main pipeline has to live with.
"""

import numpy as np
import pytest

from intrinsic_core.grasp import (
    CLOSE,
    HELD,
    OBJ,
    OPEN,
    TOOL,
    GraspConfig,
    GraspSensor,
    GraspWorld,
    empowerment,
)
from intrinsic_core.sensors import CameraConfig

OBJECT_XY = np.array([-0.68, 0.0])


def _world(**kw):
    return GraspWorld(GraspConfig(table_height=-0.25, object_radius=0.04, **kw))


def _sensor(world):
    cam = CameraConfig(
        position=(-1.45, 0.0, 0.35), look_at=(-0.68, 0.0, -0.21),
        up=(0.0, 0.0, 1.0), fov_deg=50.0, width=16, height=12,
        table_height=world.config.table_height,
    )
    return GraspSensor(cam, world.config, depth_resolution=0.02)


def _at_object(world, grasped=False, lift=0.0):
    tool = np.array([-0.68, 0.0, world.rest_height() + lift])
    return world.initial_state(tool, OBJECT_XY, grasped=grasped, lift=lift)


# --- mechanics ---------------------------------------------------------------

def test_close_grasps_only_within_reach():
    w = _world()
    assert w.step(_at_object(w), CLOSE)[HELD] == 1.0

    far = w.initial_state(np.array([-0.40, 0.0, w.rest_height()]), OBJECT_XY)
    assert w.step(far, CLOSE)[HELD] == 0.0


def test_held_object_lifts_and_is_carried():
    """The thing the old model could not express at all.

    `ur5.py` tracks the object as (x, y) with no z, so 'off the table' has no
    representation there and no objective could ever have scored it.
    """
    w = _world()
    held = w.step(_at_object(w), CLOSE)
    lifted = w.rollout(held, [4, 4])                 # two +z moves
    assert lifted[OBJ][2] == pytest.approx(w.rest_height() + 0.10, abs=1e-9)

    carried = w.rollout(lifted, [2, 2])              # then +y
    assert carried[OBJ][1] == pytest.approx(0.10, abs=1e-9)
    # while held, the object IS the tool: that is what makes it stop being an
    # independent degree of freedom, which is the whole finding below.
    assert np.allclose(carried[OBJ], carried[TOOL])


def test_release_drops_to_the_table():
    w = _world()
    lifted = w.rollout(w.step(_at_object(w), CLOSE), [4, 4])
    assert w.step(lifted, OPEN)[OBJ][2] == pytest.approx(w.rest_height())


def test_unheld_object_cannot_be_lifted():
    w = _world()
    assert w.rollout(_at_object(w), [4, 4])[OBJ][2] == pytest.approx(w.rest_height())


def test_push_moves_it_and_bolted_does_not():
    w = _world()
    start = w.initial_state(
        np.array([-0.74, 0.0, w.rest_height()]), OBJECT_XY)
    assert w.step(start, 0)[OBJ][0] > start[OBJ][0]          # +x shove
    assert w.bolted().step(start, 0)[OBJ][0] == pytest.approx(OBJECT_XY[0])


# --- the experiment ----------------------------------------------------------

def test_own_sensor_empowerment_prefers_hovering_to_holding():
    """Measured explanation for 'empowerment will not pick things up'.

    Scored on the depth image, grasping is a small LOSS at every horizon
    (-0.23, -0.13, -0.19 bits at horizons 2, 3, 4). The project asserted this
    as a structural claim; it is really a consequence of what gets called the
    outcome, and the control below shows the mechanism.
    """
    w = _world()
    s = _sensor(w)
    hovering = empowerment(w, s, _at_object(w), 3, "depth")
    holding = empowerment(w, s, _at_object(w, grasped=True), 3, "depth")
    assert holding < hovering


def test_transfer_empowerment_prefers_holding():
    """Score control over the OBJECT and grasping becomes the best move.

    Untouched the object never moves; pushed it reaches a few dozen spots on
    the table; held it goes anywhere the tool goes, in three dimensions. So
    picking it up is derived from the objective rather than rewarded, which
    is the version of this the project actually wants.
    """
    w = _world()
    s = _sensor(w)
    hovering = empowerment(w, s, _at_object(w), 3, "object")
    holding = empowerment(w, s, _at_object(w, grasped=True), 3, "object")
    assert holding > hovering + 0.5


def test_bolting_the_object_flips_the_own_sensor_preference():
    """WHY own-sensor dislikes grasping, isolated.

    It is not that holding is bad. It is that hovering beside a PUSHABLE
    object is good: shoving gives the object semi-independent motion, which
    adds variety to the picture, and grasping destroys that independence by
    making the object a rigid function of the tool.

    Remove the ability to push -- bolt the object -- and hovering has nothing
    left to offer, so grasping wins under own-sensor too (+0.53 bits). That
    reversal is the mechanism, and it is why the fix is to change the outcome
    rather than to tune anything.
    """
    w = _world().bolted()
    s = _sensor(w)
    hovering = empowerment(w, s, _at_object(w), 3, "depth")
    holding = empowerment(w, s, _at_object(w, grasped=True), 3, "depth")
    assert holding > hovering


def test_object_out_of_reach_contributes_almost_nothing():
    """The reach problem survives the change of objective.

    Transfer empowerment makes grasping the best thing to do once you are at
    the object. It does not help you get there: from 20 cm away the object is
    outside what three macro-actions can touch, so nearly every rollout
    leaves it where it started.
    """
    w = _world()
    s = _sensor(w)
    away = w.initial_state(np.array([-0.48, 0.0, w.rest_height()]), OBJECT_XY)
    assert empowerment(w, s, away, 3, "object") < 1.5
    assert empowerment(w, s, _at_object(w), 3, "object") > 4.0
