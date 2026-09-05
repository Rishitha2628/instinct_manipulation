// Deciding when two continuous outcomes count as "the same".
//
// This is the part of continuous empowerment that has no clean answer. In a
// grid world you either land on the same square or you do not. On an arm,
// every rollout ends at a slightly different joint angle, so a naive count
// finds every outcome distinct and reports log2(n_sequences) everywhere -- a
// perfectly flat, perfectly useless map.
//
// Two options are provided:
//
//   GridDiscretiser    round each dimension to a fixed resolution. Simple,
//                      fast, and produces artefacts that depend on exactly
//                      where the bin boundaries happen to fall.
//
//   ScaledDiscretiser  same, but each dimension gets its own resolution, so
//                      that (say) 1 cm of puck movement can be made to count
//                      for more than 1 cm of gripper movement.
//
// The choice of resolution is a real parameter with real consequences. Too
// coarse and everything collapses into one bin (empowerment 0 everywhere).
// Too fine and nothing collapses at all (empowerment saturated everywhere).
// Both failure modes are worth seeing once; sweep_resolution exists to make
// that easy.
#ifndef INTRINSIC_CORE__DISCRETISE_HPP_
#define INTRINSIC_CORE__DISCRETISE_HPP_

#include <cstdint>
#include <functional>
#include <vector>

#include <Eigen/Core>

namespace intrinsic_core
{

/// A discretised outcome. Compared and hashed as a whole.
using OutcomeKey = std::vector<int64_t>;

/// outcome -> key. Two outcomes with equal keys are "the same" outcome.
using Discretiser = std::function<OutcomeKey(const Eigen::VectorXd &)>;

/// Round every dimension to the same resolution.
Discretiser grid_discretiser(double resolution = 0.02);

/// Per-dimension resolution.
///
/// Use this to say what the agent's sensors can actually resolve, or to
/// weight one part of the outcome more heavily than another. Setting a
/// dimension's resolution to infinity drops it from the outcome entirely,
/// which is how you reproduce the paper's "can it perceive the box?"
/// condition.
Discretiser scaled_discretiser(const Eigen::VectorXd & resolutions);

}  // namespace intrinsic_core

#endif  // INTRINSIC_CORE__DISCRETISE_HPP_
