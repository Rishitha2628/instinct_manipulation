"""
A 2-link planar arm on a table with a puck.

This is deliberately the smallest system in which the paper's box
experiment (section 4.5.4) still makes sense.  The four conditions there
were: the box is pushable or not, and perceivable or not.  All four are
reproducible here by changing `puck_mass_factor` and by dropping the puck
dimensions from the discretiser.

State vector:      [q1, q2, px, py]     joint angles, puck position
Outcome vector:    [ex, ey, px, py]     end-effector position, puck position

The outcome deliberately includes the puck.  That is the entire mechanism:
near the puck, two different action sequences that put the gripper in the
same place can leave the puck in different places, so they count as
different outcomes.  Away from the puck they collapse into one.  Nobody
tells the arm the puck is interesting; it falls out of the counting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

__all__ = ["ArmConfig", "PlanarArm", "JOINT_ACTIONS"]


# Nine actions: each joint moves down / holds / moves up.
JOINT_ACTIONS = np.array(
    [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)],
    dtype=np.float64,
)


@dataclass
class ArmConfig:
    l1: float = 0.30
    l2: float = 0.25

    dq: float = 0.12
    """Radians per step per joint.  The paper's 'power constraint' in
    section 4.7.4 -- and it matters as much here as it does there."""

    q1_limits: tuple[float, float] = (-2.6, 2.6)
    q2_limits: tuple[float, float] = (-2.4, 2.4)

    puck_radius: float = 0.045
    """Contact distance.  Inside this, the gripper pushes the puck."""

    puck_mass_factor: float = 1.0
    """1.0 = puck moves fully with the gripper.
    0.0 = puck is bolted down (the paper's immovable-box condition)."""

    table_bounds: tuple[float, float, float, float] = (-0.55, 0.55, -0.10, 0.55)
    """(xmin, xmax, ymin, ymax) that the puck cannot leave."""


class PlanarArm:
    """Deterministic forward model.  No physics engine, no ROS, no learning."""

    def __init__(self, config: ArmConfig | None = None) -> None:
        self.config = config or ArmConfig()

    # -- kinematics ---------------------------------------------------------

    def forward_kinematics(self, q1: float, q2: float) -> np.ndarray:
        c = self.config
        x = c.l1 * np.cos(q1) + c.l2 * np.cos(q1 + q2)
        y = c.l1 * np.sin(q1) + c.l2 * np.sin(q1 + q2)
        return np.array([x, y])

    def reachable(self, x: float, y: float) -> bool:
        c = self.config
        r = np.hypot(x, y)
        return abs(c.l1 - c.l2) + 1e-6 < r < (c.l1 + c.l2) - 1e-6

    def inverse_kinematics(self, x: float, y: float, elbow_up: bool = True):
        """Returns (q1, q2) or None if unreachable."""
        c = self.config
        r2 = x * x + y * y
        cos_q2 = (r2 - c.l1 ** 2 - c.l2 ** 2) / (2 * c.l1 * c.l2)
        if not -1.0 <= cos_q2 <= 1.0:
            return None
        q2 = np.arccos(cos_q2)
        if not elbow_up:
            q2 = -q2
        q1 = np.arctan2(y, x) - np.arctan2(
            c.l2 * np.sin(q2), c.l1 + c.l2 * np.cos(q2)
        )
        if not (c.q1_limits[0] <= q1 <= c.q1_limits[1]):
            return None
        if not (c.q2_limits[0] <= q2 <= c.q2_limits[1]):
            return None
        return float(q1), float(q2)

    # -- dynamics -----------------------------------------------------------

    def step(self, state: np.ndarray, action: int) -> np.ndarray:
        """One control step.  state = [q1, q2, px, py]."""
        c = self.config
        q1, q2, px, py = (float(v) for v in state)

        dq1, dq2 = JOINT_ACTIONS[int(action)] * c.dq
        q1 = float(np.clip(q1 + dq1, *c.q1_limits))
        q2 = float(np.clip(q2 + dq2, *c.q2_limits))

        ex, ey = self.forward_kinematics(q1, q2)

        # Push the puck if the gripper is inside contact range.
        if c.puck_mass_factor > 0.0:
            dx, dy = px - ex, py - ey
            dist = float(np.hypot(dx, dy))
            if dist < c.puck_radius:
                if dist < 1e-9:
                    dx, dy, dist = 1.0, 0.0, 1.0
                overlap = c.puck_radius - dist
                scale = c.puck_mass_factor * overlap / dist
                px += dx * scale
                py += dy * scale
                xmin, xmax, ymin, ymax = c.table_bounds
                px = float(np.clip(px, xmin, xmax))
                py = float(np.clip(py, ymin, ymax))

        return np.array([q1, q2, px, py])

    def rollout(self, state: np.ndarray, actions: Sequence[int]) -> np.ndarray:
        """Run a sequence of actions and return the OUTCOME vector."""
        s = np.asarray(state, dtype=np.float64).copy()
        for a in actions:
            s = self.step(s, int(a))
        ex, ey = self.forward_kinematics(s[0], s[1])
        return np.array([ex, ey, s[2], s[3]])

    # -- variants of the paper's box experiment -----------------------------

    def with_fixed_puck(self) -> "PlanarArm":
        """The immovable-box condition: present, but cannot be pushed."""
        return PlanarArm(replace(self.config, puck_mass_factor=0.0))
