"""
Batched empowerment for the UR5 + depth camera setup.

The generic estimator in empowerment.py calls the forward model once per
action sequence. For a 6-DOF arm with a depth camera that is far too slow:
81 actions at horizon 2 is 6561 rollouts, each needing forward kinematics
and a render.

This module does the same computation with the whole batch of sequences
resolved at once -- kinematics for all endpoints, then a single batched
ray-cast. Same numbers, roughly two orders of magnitude faster.

The estimate is identical to `EmpowermentEstimator`; `tests/` checks that
the two agree.
"""

from __future__ import annotations

import numpy as np

from .empowerment import EmpowermentConfig
from .sensors import DepthCamera, DepthDiscretiser
from .ur5 import UR5

__all__ = ["BatchedDepthEmpowerment"]


class BatchedDepthEmpowerment:
    def __init__(
        self,
        arm: UR5,
        camera: DepthCamera,
        discretiser: DepthDiscretiser,
        config: EmpowermentConfig | None = None,
    ) -> None:
        self.arm = arm
        self.camera = camera
        self.discretiser = discretiser
        self.config = config or EmpowermentConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._sequences = self._build_sequences()

    def _build_sequences(self) -> np.ndarray:
        cfg = self.config
        n_a = self.arm.n_actions
        total = n_a ** cfg.n_steps

        if total <= cfg.exhaustive_below:
            idx = np.arange(total)
            seqs = np.empty((total, cfg.n_steps), dtype=np.int64)
            for k in range(cfg.n_steps):
                seqs[:, cfg.n_steps - 1 - k] = (idx // (n_a ** k)) % n_a
            return seqs

        return self._rng.integers(0, n_a, size=(cfg.n_sequences, cfg.n_steps))

    @staticmethod
    def _batch_link_positions(q: np.ndarray) -> np.ndarray:
        """Batched forward kinematics. q (N,6) -> link origins (N,7,3)."""
        from .ur5 import DH_UR5

        n = len(q)
        T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        out = [np.zeros((n, 3))]

        for i in range(6):
            a, d, alpha = DH_UR5[i]
            ct, st = np.cos(q[:, i]), np.sin(q[:, i])
            ca, sa = np.cos(alpha), np.sin(alpha)

            Ti = np.zeros((n, 4, 4))
            Ti[:, 0, 0] = ct
            Ti[:, 0, 1] = -st * ca
            Ti[:, 0, 2] = st * sa
            Ti[:, 0, 3] = a * ct
            Ti[:, 1, 0] = st
            Ti[:, 1, 1] = ct * ca
            Ti[:, 1, 2] = -ct * sa
            Ti[:, 1, 3] = a * st
            Ti[:, 2, 1] = sa
            Ti[:, 2, 2] = ca
            Ti[:, 2, 3] = d
            Ti[:, 3, 3] = 1.0

            T = T @ Ti
            out.append(T[:, :3, 3].copy())

        return np.stack(out, axis=1)

    def _endpoints(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resolve every sequence to its final state, fully vectorised.

        Returns (arm_points (N,7,3), puck_xy (N,2)).
        """
        seqs = self._sequences
        n = len(seqs)
        cfg = self.arm.config

        q = np.repeat(state[None, :6], n, axis=0)
        puck = np.repeat(state[None, 6:8], n, axis=0)

        lo = cfg.joint_limits[:, 0]
        hi = cfg.joint_limits[:, 1]
        contact = cfg.tool_radius + cfg.puck_radius
        xmin, xmax, ymin, ymax = cfg.table_bounds

        for step in range(seqs.shape[1]):
            q = np.clip(q + self.arm._action_table[seqs[:, step]], lo, hi)

            if cfg.puck_mass_factor > 0.0:
                tool = self._batch_link_positions(q)[:, -1, :]
                delta = puck - tool[:, :2]
                planar = np.hypot(delta[:, 0], delta[:, 1])

                touching = (
                    (tool[:, 2] < cfg.table_height + contact)
                    & (planar < contact)
                    & (planar > 1e-9)
                )
                if touching.any():
                    overlap = contact - planar[touching]
                    scale = cfg.puck_mass_factor * overlap / planar[touching]
                    puck[touching] += delta[touching] * scale[:, None]
                    puck[:, 0] = np.clip(puck[:, 0], xmin, xmax)
                    puck[:, 1] = np.clip(puck[:, 1], ymin, ymax)

        return self._batch_link_positions(q), puck

    def estimate(self, state: np.ndarray) -> float:
        """Empowerment in bits, from the depth sensor's point of view."""
        arm_pts, puck_xy = self._endpoints(np.asarray(state, dtype=np.float64))

        images = self.camera.render_batch(
            arm_pts, puck_xy, self.arm.config.puck_radius
        )

        quantised = np.round(
            np.minimum(images, self.discretiser.ignore_beyond)
            / self.discretiser.depth_resolution
        ).astype(np.int32)

        flat = quantised.reshape(len(quantised), -1)
        n_distinct = len(np.unique(flat, axis=0))
        return float(np.log2(n_distinct)) if n_distinct > 0 else 0.0

    __call__ = estimate
