#!/usr/bin/env python3
"""
Find the band where empowerment actually has structure.

The paper advertises empowerment as parameter-free. On a continuous system
it is not. Two parameters decide whether the map means anything, and both
have a failure mode on each side:

    outcome_resolution   too coarse -> every outcome in one bin -> 0 bits
                         too fine   -> no outcomes merge -> flat and high

    horizon              too short  -> nothing reachable -> flat and low
                         too long   -> everything reachable -> flat again
                                       (sec 4.5.5, the Greek gods problem)

This script measures the *contrast* -- empowerment at the object minus
empowerment away from it -- across both. Contrast, not absolute value, is
what you need: a map can have plenty of bits everywhere and still tell you
nothing.

Run this before trusting any result, and again whenever you change the arm.

    python3 scripts/sweep_parameters.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intrinsic_core import (  # noqa: E402
    EmpowermentConfig,
    EmpowermentEstimator,
    GridDiscretiser,
    PlanarArm,
)

PUCK = np.array([0.30, 0.25])
AT_OBJECT = (0.30, 0.25)
AWAY = (0.30, 0.42)


def _state(arm, xy):
    ik = arm.inverse_kinematics(*xy)
    return None if ik is None else np.array([ik[0], ik[1], *PUCK])


def contrast(arm, resolution: float, horizon: int) -> tuple[float, float, float]:
    est = EmpowermentEstimator(
        arm.rollout, 9, GridDiscretiser(resolution),
        EmpowermentConfig(n_steps=horizon, n_sequences=600, seed=0),
    )
    near = est.estimate(_state(arm, AT_OBJECT))
    far = est.estimate(_state(arm, AWAY))
    return near, far, near - far


def _verdict(near: float, delta: float, ceiling: float) -> str:
    if near < 0.1:
        return "DEAD  (all outcomes in one bin)"
    if near > ceiling - 0.15:
        return "SATURATED  (nothing merges)"
    if delta < 0.25:
        return "flat  (no contrast)"
    return "usable"


def main() -> int:
    arm = PlanarArm()

    print("\nRESOLUTION SWEEP  (horizon = 3)")
    print(f"{'res (m)':>10} {'at object':>11} {'away':>8} {'contrast':>10}   verdict")
    print("-" * 62)
    ceiling = 7.88  # measured degeneracy limit for this arm, see tests
    for r in [0.001, 0.005, 0.01, 0.02, 0.025, 0.04, 0.06, 0.10, 0.20, 1.0]:
        near, far, delta = contrast(arm, r, 3)
        print(f"{r:>10.3f} {near:>11.3f} {far:>8.3f} {delta:>+10.3f}   "
              f"{_verdict(near, delta, ceiling)}")

    print("\nHORIZON SWEEP  (resolution = 0.025 m)")
    print(f"{'steps':>10} {'at object':>11} {'away':>8} {'contrast':>10}   verdict")
    print("-" * 62)
    for n in [1, 2, 3, 4, 5]:
        near, far, delta = contrast(arm, 0.025, n)
        print(f"{n:>10d} {near:>11.3f} {far:>8.3f} {delta:>+10.3f}   "
              f"{_verdict(near, delta, ceiling)}")

    print("\nPick a resolution and horizon from the middle of the usable band,")
    print("not the edge. The edges are where the artefacts live.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
