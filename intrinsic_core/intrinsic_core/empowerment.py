"""
n-step empowerment estimation for continuous systems.

Empowerment is the channel capacity from an agent's actuators to its own
later sensor readings:

    E(s) = max_{p(a)} I(S_{t+n} ; A_t)

For a deterministic system this collapses to the log of the number of
*distinguishable* outcomes reachable in n steps.  That is the estimator
implemented here, plus a Blahut-Arimoto variant for the stochastic case.

Reference:
    Salge, Glackin & Polani (2013), "Empowerment -- An Introduction",
    arXiv:1310.1863.  Section 4.4 for the discrete case, 4.6 for the
    problems that appear when the state space becomes continuous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "EmpowermentConfig",
    "EmpowermentEstimator",
    "blahut_arimoto",
]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class EmpowermentConfig:
    """Parameters for the estimator.

    The paper advertises empowerment as parameter-free.  In a continuous
    setting that is not true, and these are exactly the knobs section 4.6
    warns about.  They are collected here so they are visible rather than
    buried.
    """

    n_steps: int = 4
    """Horizon.  Too short and every state looks alike because you cannot
    reach anything; too long and every state looks alike because you can
    reach everything (the 'tragedy of the Greek gods', section 4.5.5)."""

    n_sequences: int = 400
    """How many action sequences to sample.  Exhaustive enumeration is
    |A|^n, which is only tractable for small n and small discrete A."""

    exhaustive_below: int = 4096
    """If |A|^n is under this, enumerate exactly instead of sampling."""

    seed: int | None = 0

    stochastic: bool = False
    """If True, use Blahut-Arimoto on an empirical transition matrix rather
    than simply counting distinct outcomes."""

    ba_tolerance: float = 1e-6
    ba_max_iter: int = 200


# --------------------------------------------------------------------------
# Blahut-Arimoto
# --------------------------------------------------------------------------

def blahut_arimoto(
    p_y_given_x: np.ndarray,
    tolerance: float = 1e-6,
    max_iter: int = 200,
) -> tuple[float, np.ndarray]:
    """Channel capacity of a discrete memoryless channel, in bits.

    Args:
        p_y_given_x: (n_inputs, n_outputs) row-stochastic matrix.  Row i is
            the distribution over outcomes given input i.

    Returns:
        (capacity_in_bits, capacity_achieving_input_distribution)

    The returned input distribution is *not* a policy the agent should
    follow.  Empowerment is a maximum over action distributions and
    measures potential, not intent (section 4.4.2 of the paper).
    """
    p_y_given_x = np.asarray(p_y_given_x, dtype=np.float64)
    n_in, n_out = p_y_given_x.shape

    if n_in == 0 or n_out == 0:
        return 0.0, np.zeros(n_in)

    # Rows must be normalised; guard against all-zero rows.
    row_sums = p_y_given_x.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    p_y_given_x = p_y_given_x / safe

    r = np.full(n_in, 1.0 / n_in)
    prev_capacity = -np.inf

    for _ in range(max_iter):
        # q(x|y) proportional to r(x) p(y|x)
        joint = r[:, None] * p_y_given_x
        marginal = joint.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(marginal > 0, joint / marginal, 0.0)

            # r(x) proportional to exp( sum_y p(y|x) log q(x|y) )
            log_q = np.where(q > 0, np.log(q), 0.0)
            exponent = (p_y_given_x * log_q).sum(axis=1)

        exponent -= exponent.max()          # stabilise
        r_new = np.exp(exponent)
        total = r_new.sum()
        if total <= 0:
            break
        r_new /= total

        # current capacity estimate, in nats then converted
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                (q > 0) & (r_new[:, None] > 0),
                np.log(q / r_new[:, None]),
                0.0,
            )
        capacity = float((r_new[:, None] * p_y_given_x * ratio).sum())

        if abs(capacity - prev_capacity) < tolerance:
            r = r_new
            prev_capacity = capacity
            break

        r = r_new
        prev_capacity = capacity

    return max(prev_capacity, 0.0) / np.log(2.0), r


# --------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------

class EmpowermentEstimator:
    """Estimates n-step empowerment for any system exposing a forward model.

    The system is supplied as two callables so that this class knows nothing
    about arms, grids, ROS, or simulators:

        rollout(state, action_sequence) -> outcome
        actions                          -> the discrete action alphabet

    and one callable that decides when two outcomes count as the same:

        discretiser(outcome) -> hashable key

    That last one is the whole content of section 4.6.  In a grid world two
    outcomes are the same square or they are not.  In a continuous system
    every outcome differs in the twelfth decimal place, so without a notion
    of "close enough" the answer is always log2(n_sequences) and means
    nothing.
    """

    def __init__(
        self,
        rollout: Callable[[np.ndarray, Sequence[int]], np.ndarray],
        n_actions: int,
        discretiser: Callable[[np.ndarray], tuple],
        config: EmpowermentConfig | None = None,
    ) -> None:
        self.rollout = rollout
        self.n_actions = int(n_actions)
        self.discretiser = discretiser
        self.config = config or EmpowermentConfig()
        self._rng = np.random.default_rng(self.config.seed)

    # -- action sequences ---------------------------------------------------

    def _action_sequences(self) -> np.ndarray:
        """Either every sequence, or a random sample of them."""
        cfg = self.config
        total = self.n_actions ** cfg.n_steps

        if total <= cfg.exhaustive_below:
            # enumerate exactly: digits of 0..total-1 in base n_actions
            idx = np.arange(total)
            seqs = np.empty((total, cfg.n_steps), dtype=np.int64)
            for k in range(cfg.n_steps):
                seqs[:, cfg.n_steps - 1 - k] = (idx // (self.n_actions ** k)) % self.n_actions
            return seqs

        return self._rng.integers(
            0, self.n_actions, size=(cfg.n_sequences, cfg.n_steps)
        )

    # -- the estimate -------------------------------------------------------

    def __call__(self, state: np.ndarray) -> float:
        return self.estimate(state)

    def estimate(self, state: np.ndarray) -> float:
        """Empowerment of `state`, in bits."""
        sequences = self._action_sequences()

        outcomes = [
            self.discretiser(self.rollout(state, seq))
            for seq in sequences
        ]

        if self.config.stochastic:
            return self._capacity_from_outcomes(sequences, outcomes)

        # Deterministic case.  This is the entire idea:
        #   count the distinct places you can end up, take the log.
        return float(np.log2(len(set(outcomes)))) if outcomes else 0.0

    def _capacity_from_outcomes(
        self,
        sequences: np.ndarray,
        outcomes: list[tuple],
    ) -> float:
        """Build an empirical p(outcome | sequence) and run Blahut-Arimoto.

        Only meaningful when the rollout is genuinely stochastic and each
        sequence has been rolled out more than once; with one sample per
        sequence the matrix is one-hot and this reduces to the counting
        estimator above (with more arithmetic).
        """
        keys = sorted(set(outcomes))
        key_index = {k: i for i, k in enumerate(keys)}

        seq_keys = [tuple(s) for s in sequences]
        unique_seqs = sorted(set(seq_keys))
        seq_index = {s: i for i, s in enumerate(unique_seqs)}

        matrix = np.zeros((len(unique_seqs), len(keys)))
        for s, o in zip(seq_keys, outcomes):
            matrix[seq_index[s], key_index[o]] += 1.0

        capacity, _ = blahut_arimoto(
            matrix,
            tolerance=self.config.ba_tolerance,
            max_iter=self.config.ba_max_iter,
        )
        return capacity

    # -- convenience --------------------------------------------------------

    def map_over(self, states: np.ndarray) -> np.ndarray:
        """Empowerment for a batch of states.  Returns a 1-D array."""
        return np.array([self.estimate(s) for s in states], dtype=np.float64)
