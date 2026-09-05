"""Tests for the empowerment estimator. Run with: python3 -m pytest tests/ -v"""
import numpy as np
import pytest

from intrinsic_core import (
    ArmConfig, EmpowermentConfig, EmpowermentEstimator,
    GridDiscretiser, ScaledDiscretiser, PlanarArm, blahut_arimoto,
)


# --- Blahut-Arimoto ---------------------------------------------------------

def test_ba_noiseless_binary_channel_is_one_bit():
    capacity, _ = blahut_arimoto(np.eye(2))
    assert capacity == pytest.approx(1.0, abs=1e-4)


def test_ba_noiseless_8ary_channel_is_three_bits():
    capacity, _ = blahut_arimoto(np.eye(8))
    assert capacity == pytest.approx(3.0, abs=1e-4)


def test_ba_useless_channel_is_zero_bits():
    capacity, _ = blahut_arimoto(np.ones((4, 4)) / 4)
    assert capacity == pytest.approx(0.0, abs=1e-6)


def test_ba_binary_symmetric_channel_matches_analytic():
    p = 0.1
    bsc = np.array([[1 - p, p], [p, 1 - p]])
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    capacity, _ = blahut_arimoto(bsc)
    assert capacity == pytest.approx(1.0 - h, abs=1e-4)


# --- kinematics -------------------------------------------------------------

def test_ik_inverts_fk():
    arm = PlanarArm()
    for target in [(0.30, 0.25), (0.20, 0.30), (-0.25, 0.20)]:
        ik = arm.inverse_kinematics(*target)
        assert ik is not None, target
        assert arm.forward_kinematics(*ik) == pytest.approx(target, abs=1e-9)


def test_unreachable_points_return_none():
    arm = PlanarArm()
    assert arm.inverse_kinematics(5.0, 5.0) is None


# --- the core claim ---------------------------------------------------------

def _estimator(model, resolution=0.025, steps=3):
    return EmpowermentEstimator(
        model.rollout, 9, GridDiscretiser(resolution),
        EmpowermentConfig(n_steps=steps, seed=0),
    )


def _state(arm, xy, puck):
    ik = arm.inverse_kinematics(*xy)
    assert ik is not None
    return np.array([ik[0], ik[1], puck[0], puck[1]])


def test_empowerment_is_higher_near_a_pushable_object():
    """The result. Nobody tells the arm the puck matters."""
    arm = PlanarArm()
    puck = np.array([0.30, 0.25])
    est = _estimator(arm)

    near = est.estimate(_state(arm, (0.30, 0.25), puck))
    far = est.estimate(_state(arm, (0.30, 0.42), puck))
    assert near > far + 0.5


def test_bolted_object_removes_the_peak():
    """Paper's immovable-box condition: influence, not mere presence."""
    arm = PlanarArm()
    puck = np.array([0.30, 0.25])
    near = _state(arm, (0.30, 0.25), puck)
    far = _state(arm, (0.30, 0.42), puck)

    est = _estimator(arm.with_fixed_puck())
    assert abs(est.estimate(near) - est.estimate(far)) < 0.5


def test_unperceived_object_contributes_nothing():
    """Drop the puck dims from the outcome; the peak must vanish.

    This is the paper's 'can it perceive the box?' condition. Perceiving
    something you cannot act on is worth nothing -- and acting on something
    you cannot perceive is worth nothing either.
    """
    arm = PlanarArm()
    puck = np.array([0.30, 0.25])
    blind = ScaledDiscretiser([0.025, 0.025, np.inf, np.inf])
    est = EmpowermentEstimator(
        arm.rollout, 9, blind, EmpowermentConfig(n_steps=3, seed=0)
    )
    near = est.estimate(_state(arm, (0.30, 0.25), puck))
    far = est.estimate(_state(arm, (0.30, 0.42), puck))
    assert abs(near - far) < 0.5


def test_empowerment_decays_with_distance():
    arm = PlanarArm()
    puck = np.array([0.30, 0.25])
    est = _estimator(arm)
    values = [
        est.estimate(_state(arm, (0.30 + d, 0.25), puck))
        for d in (0.0, 0.05, 0.10)
    ]
    assert values[0] > values[1] > values[2]


# --- the parameters that bite -----------------------------------------------

def test_coarse_resolution_collapses_the_map():
    """Everything in one bin -> 0 bits. One of the two failure modes."""
    arm = PlanarArm()
    est = _estimator(arm, resolution=100.0)
    assert est.estimate(_state(arm, (0.30, 0.25), np.array([0.30, 0.25]))) == 0.0


def test_fine_resolution_saturates_the_map():
    """The other failure mode: nothing collapses, so the map goes flat high.

    Note the ceiling is NOT log2(9**3) = 9.51 bits. Joint increments commute
    -- [+1,-1,0] and [0,-1,+1] end at identical angles -- so 729 sequences
    only reach 235 distinct configurations however finely you measure. That
    degeneracy is a property of the arm, not of the discretiser, and it caps
    open-loop empowerment at about 7.9 bits on this system regardless of
    horizon.
    """
    arm = PlanarArm()
    puck = np.array([0.30, 0.25])
    fine = _estimator(arm, resolution=1e-9).estimate(_state(arm, (0.30, 0.25), puck))
    usable = _estimator(arm, resolution=0.025).estimate(_state(arm, (0.30, 0.25), puck))

    assert fine > usable                    # finer resolution, more outcomes
    assert fine < np.log2(9 ** 3)           # but well under the naive ceiling
    assert fine == pytest.approx(7.88, abs=0.05)


def test_commuting_actions_produce_identical_outcomes():
    """Documents the degeneracy above: action order does not matter."""
    arm = PlanarArm()
    state = _state(arm, (0.30, 0.25), np.array([0.30, 0.25]))
    assert arm.rollout(state, [0, 4, 8]) == pytest.approx(
        arm.rollout(state, [8, 4, 0])
    )


def test_estimator_is_deterministic_for_fixed_seed():
    arm = PlanarArm()
    s = _state(arm, (0.30, 0.25), np.array([0.30, 0.25]))
    assert _estimator(arm).estimate(s) == _estimator(arm).estimate(s)
