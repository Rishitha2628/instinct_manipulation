"""
Deciding when two continuous outcomes count as "the same".

This is the part of continuous empowerment that has no clean answer.  In a
grid world you either land on the same square or you do not.  On an arm,
every rollout ends at a slightly different joint angle, so a naive count
finds every outcome distinct and reports log2(n_sequences) everywhere --
a perfectly flat, perfectly useless map.

Two options are provided:

  GridDiscretiser   round each dimension to a fixed resolution.  Simple,
                    fast, and produces artefacts that depend on exactly
                    where the bin boundaries happen to fall.

  ScaledDiscretiser same, but each dimension gets its own resolution, so
                    that (say) 1 cm of puck movement can be made to count
                    for more than 1 cm of gripper movement.

The choice of resolution is a real parameter with real consequences.  Too
coarse and everything collapses into one bin (empowerment 0 everywhere).
Too fine and nothing collapses at all (empowerment saturated everywhere).
Both failure modes are worth seeing once; `sweep_resolution` below exists
to make that easy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GridDiscretiser", "ScaledDiscretiser", "sweep_resolution"]


@dataclass
class GridDiscretiser:
    """Round every dimension to the same resolution."""

    resolution: float = 0.02

    def __call__(self, outcome: np.ndarray) -> tuple:
        return tuple(np.round(np.asarray(outcome) / self.resolution).astype(np.int64))


@dataclass
class ScaledDiscretiser:
    """Per-dimension resolution.

    Use this to say what the agent's sensors can actually resolve, or to
    weight one part of the outcome more heavily than another.  Setting a
    dimension's resolution to np.inf drops it from the outcome entirely,
    which is how you reproduce the paper's "can it perceive the box?"
    condition.
    """

    resolutions: np.ndarray

    def __post_init__(self) -> None:
        self.resolutions = np.asarray(self.resolutions, dtype=np.float64)

    def __call__(self, outcome: np.ndarray) -> tuple:
        outcome = np.asarray(outcome, dtype=np.float64)
        keep = np.isfinite(self.resolutions)
        scaled = outcome[keep] / self.resolutions[keep]
        return tuple(np.round(scaled).astype(np.int64))


def sweep_resolution(
    estimator_factory,
    state: np.ndarray,
    resolutions,
) -> list[tuple[float, float]]:
    """Empowerment at one state across a range of resolutions.

    Returns [(resolution, empowerment_bits), ...].  A usable resolution sits
    in the middle of this curve, away from both the 0-bit floor and the
    log2(n_sequences) ceiling.
    """
    out = []
    for r in resolutions:
        est = estimator_factory(GridDiscretiser(resolution=float(r)))
        out.append((float(r), est.estimate(state)))
    return out
