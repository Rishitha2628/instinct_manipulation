"""
UR5 forward kinematics and a pushable object on a table.

Uses the standard UR5 Denavit-Hartenberg parameters, so joint angles here
correspond to real /joint_states values from a ur_robot_driver setup and the
tool pose matches what MoveIt reports.

Contact is a kinematic approximation, not physics: if the tool sphere
overlaps the puck sphere, the puck is displaced along the separation axis
and stays on the table plane. That is enough for the empowerment result --
what matters is that being near the puck creates *more distinguishable
outcomes*, not that the contact forces are right. Swap `PuckModel.push` for
a PyBullet step when you move to a real simulator; nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

__all__ = ["UR5Config", "UR5", "DH_UR5", "JOINT_LIMITS_UR5"]


# Standard UR5 DH parameters: (a, d, alpha)
DH_UR5 = np.array([
    [0.0,      0.089159,  np.pi / 2],
    [-0.425,   0.0,       0.0],
    [-0.39225, 0.0,       0.0],
    [0.0,      0.10915,   np.pi / 2],
    [0.0,      0.09465,  -np.pi / 2],
    [0.0,      0.0823,    0.0],
])

# Real UR5 joints are +-2pi; these are tightened to keep the arm above the
# table and out of self-collision without needing a collision checker.
JOINT_LIMITS_UR5 = np.array([
    [-np.pi,      np.pi],
    [-np.pi,      0.0],
    [-2.6,        2.6],
    [-np.pi,      np.pi],
    [-np.pi,      np.pi],
    [-np.pi,      np.pi],
])


@dataclass
class UR5Config:
    dq: float = 0.10
    """Radians per joint per control step. The paper's power constraint
    (sec 4.7.4) -- raising it can invert the landscape, not just scale it."""

    active_joints: tuple[int, ...] = (0, 1, 2, 3)
    """Which joints the agent may move. Using all six gives 3^6 = 729
    actions per step, so horizon 3 is 387 million sequences and sampling
    becomes very thin. The wrist contributes little to where the tool
    *is*, so the default drives the arm and leaves wrist 2/3 fixed."""

    tool_radius: float = 0.05
    puck_radius: float = 0.04
    puck_mass_factor: float = 1.0
    """1.0 pushable, 0.0 bolted down (the paper's immovable-box control)."""

    table_height: float = 0.0
    table_bounds: tuple[float, float, float, float] = (-0.85, 0.85, -0.85, 0.85)

    joint_limits: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_UR5.copy())


class UR5:
    """Analytic UR5 kinematics + kinematic puck contact."""

    def __init__(self, config: UR5Config | None = None) -> None:
        self.config = config or UR5Config()
        self.n_actions = 3 ** len(self.config.active_joints)
        self._action_table = self._build_action_table()

    # -- kinematics ---------------------------------------------------------

    @staticmethod
    def _dh_transform(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,      sa,       ca,      d],
            [0.0,     0.0,      0.0,    1.0],
        ])

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Full 4x4 tool pose in the base frame."""
        T = np.eye(4)
        for i in range(6):
            a, d, alpha = DH_UR5[i]
            T = T @ self._dh_transform(float(q[i]), a, d, alpha)
        return T

    def tool_position(self, q: np.ndarray) -> np.ndarray:
        return self.forward_kinematics(q)[:3, 3]

    def link_positions(self, q: np.ndarray) -> np.ndarray:
        """Origin of every link frame -- used by the depth camera to draw
        the arm, and for a crude table-collision check."""
        out = [np.zeros(3)]
        T = np.eye(4)
        for i in range(6):
            a, d, alpha = DH_UR5[i]
            T = T @ self._dh_transform(float(q[i]), a, d, alpha)
            out.append(T[:3, 3].copy())
        return np.array(out)

    def above_table(self, q: np.ndarray, margin: float = 0.02) -> bool:
        z = self.link_positions(q)[2:, 2]      # skip base links
        return bool(np.all(z > self.config.table_height + margin))

    # -- actions ------------------------------------------------------------

    def _build_action_table(self) -> np.ndarray:
        """Each action is a delta over the active joints only."""
        n = len(self.config.active_joints)
        table = np.zeros((3 ** n, 6))
        for idx in range(3 ** n):
            rem = idx
            for k, joint in enumerate(self.config.active_joints):
                table[idx, joint] = (rem % 3) - 1     # -1, 0, +1
                rem //= 3
        return table * self.config.dq

    # -- dynamics -----------------------------------------------------------

    def step(self, state: np.ndarray, action: int) -> np.ndarray:
        """state = [q0..q5, px, py]. Returns the successor state."""
        c = self.config
        q = state[:6] + self._action_table[int(action)]
        q = np.clip(q, c.joint_limits[:, 0], c.joint_limits[:, 1])

        px, py = float(state[6]), float(state[7])

        if c.puck_mass_factor > 0.0:
            tool = self.tool_position(q)
            dx, dy = px - tool[0], py - tool[1]
            planar = float(np.hypot(dx, dy))
            contact = c.tool_radius + c.puck_radius

            # Only push if the tool is also low enough to touch the puck.
            near_table = tool[2] < c.table_height + contact
            if near_table and planar < contact:
                if planar < 1e-9:
                    dx, dy, planar = 1.0, 0.0, 1.0
                overlap = contact - planar
                scale = c.puck_mass_factor * overlap / planar
                px += dx * scale
                py += dy * scale
                xmin, xmax, ymin, ymax = c.table_bounds
                px = float(np.clip(px, xmin, xmax))
                py = float(np.clip(py, ymin, ymax))

        return np.concatenate([q, [px, py]])

    def rollout_state(self, state: np.ndarray, actions: Sequence[int]) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64).copy()
        for a in actions:
            s = self.step(s, int(a))
        return s

    def rollout_pose(self, state: np.ndarray, actions: Sequence[int]) -> np.ndarray:
        """Ground-truth outcome: [tool_xyz, puck_xy].

        Useful as a baseline, but note this is *not* what the paper asks
        for -- empowerment is the channel to the agent's own SENSORS, so
        the depth-camera outcome in sensors.py is the faithful version.
        """
        s = self.rollout_state(state, actions)
        return np.concatenate([self.tool_position(s[:6]), s[6:8]])

    # -- inverse kinematics -------------------------------------------------

    def inverse_kinematics(
        self,
        target_xyz,
        q_seed: np.ndarray | None = None,
        tol: float = 1e-3,
        restarts: int = 12,
        seed: int = 0,
        tool_axis: np.ndarray | None = None,
        axis_weight: float = 0.05,
        aim_tol: float = 2e-3,
    ) -> np.ndarray | None:
        """Numerical IK. Position-only by default; add `tool_axis` to aim it.

        Position-only is the default because the tool orientation is then
        left free and the arm keeps the redundancy that makes multiple
        approaches possible. That is the right choice for asking how much of
        the scene the arm can disturb, which is what empowerment needs.

        It is the wrong choice for grasping, and quietly so. A grasp is
        defined by orientation as much as by position: the fingers have to
        straddle the object. With the orientation unconstrained the solver
        will happily return a pose that puts the FLANGE where you asked while
        the gripper points 33 degrees off vertical, and anything that assumes
        the tool hangs straight down is then wrong by the length of the
        gripper. Measured on this arm: the caller believed the pads were on
        the object, and they were 51 mm away, which is how a grasp closes on
        empty air while every number involved looks correct.

        Pass `tool_axis` as a unit vector for the direction the tool should
        POINT -- (0, 0, -1) to reach down at a table -- and the solve trades
        a little position accuracy for getting the approach right.
        """
        from scipy.optimize import minimize

        target = np.asarray(target_xyz, dtype=np.float64)
        lo, hi = self.config.joint_limits[:, 0], self.config.joint_limits[:, 1]
        rng = np.random.default_rng(seed)

        if tool_axis is None:
            def cost(q):
                return float(np.sum((self.tool_position(q) - target) ** 2))
        else:
            axis = np.asarray(tool_axis, dtype=np.float64)
            axis = axis / np.linalg.norm(axis)

            def cost(q):
                T = self.forward_kinematics(q)
                position = float(np.sum((T[:3, 3] - target) ** 2))
                # 1 - cos(angle), zero when the tool points exactly along the
                # requested axis and 2 when it points the opposite way.
                aim = 1.0 - float(T[:3, 2] @ axis)
                return position + axis_weight * aim

        seeds = []
        if q_seed is not None:
            seeds.append(np.asarray(q_seed, dtype=np.float64))
        seeds.append(np.array([0.0, -1.2, 1.4, -1.75, -1.57, 0.0]))
        seeds.extend(rng.uniform(lo, hi, size=(restarts, 6)))

        best, best_cost = None, np.inf
        for s in seeds:
            res = minimize(
                cost, np.clip(s, lo, hi), method="L-BFGS-B",
                bounds=list(zip(lo, hi)),
                options={"maxiter": 250, "ftol": 1e-14},
            )
            if res.fun < best_cost:
                best, best_cost = res.x, res.fun

            # Stop on the POSITION error, not on the objective. With an
            # orientation term the objective also carries the aim penalty,
            # which is rarely below tol^2, so testing the objective meant the
            # early exit never fired and every call ground through all 14
            # seeds. Measured effect on the caller: its control loop dropped
            # to 0.065 Hz, roughly one command every 15 seconds, which looks
            # exactly like an arm that is ignoring you.
            reached = float(np.linalg.norm(self.tool_position(best) - target))
            if reached < tol:
                if tool_axis is None:
                    break
                aimed = float(self.forward_kinematics(best)[:3, 2] @ axis)
                if aimed > 1.0 - aim_tol:
                    break

        # With an orientation term the cost is no longer purely squared
        # metres, so the acceptance test has to look at the position error
        # itself rather than at the objective value.
        if best is None or not self.above_table(best):
            return None
        if float(np.linalg.norm(self.tool_position(best) - target)) > tol:
            return None
        return best

    # -- control conditions -------------------------------------------------

    def with_fixed_puck(self) -> "UR5":
        return UR5(replace(self.config, puck_mass_factor=0.0))
