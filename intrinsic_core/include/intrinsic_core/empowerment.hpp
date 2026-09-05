// n-step empowerment estimation for continuous systems.
//
// Empowerment is the channel capacity from an agent's actuators to its own
// later sensor readings:
//
//     E(s) = max_{p(a)} I(S_{t+n} ; A_t)
//
// For a deterministic system this collapses to the log of the number of
// *distinguishable* outcomes reachable in n steps. That is the estimator
// implemented here, plus a Blahut-Arimoto variant for the stochastic case.
//
// Reference:
//     Salge, Glackin & Polani (2013), "Empowerment -- An Introduction",
//     arXiv:1310.1863. Section 4.4 for the discrete case, 4.6 for the
//     problems that appear when the state space becomes continuous.
#ifndef INTRINSIC_CORE__EMPOWERMENT_HPP_
#define INTRINSIC_CORE__EMPOWERMENT_HPP_

#include <cstdint>
#include <functional>
#include <vector>

#include <Eigen/Core>

#include "intrinsic_core/discretise.hpp"

namespace intrinsic_core
{

/// Parameters for the estimator.
///
/// The paper advertises empowerment as parameter-free. In a continuous
/// setting that is not true, and these are exactly the knobs section 4.6
/// warns about. They are collected here so they are visible rather than
/// buried.
struct EmpowermentConfig
{
  /// Horizon. Too short and every state looks alike because you cannot reach
  /// anything; too long and every state looks alike because you can reach
  /// everything (the "tragedy of the Greek gods", section 4.5.5).
  int n_steps = 4;

  /// How many action sequences to sample. Exhaustive enumeration is |A|^n,
  /// which is only tractable for small n and small discrete A.
  int n_sequences = 400;

  /// If |A|^n is under this, enumerate exactly instead of sampling.
  int exhaustive_below = 4096;

  uint64_t seed = 0;

  /// If true, use Blahut-Arimoto on an empirical transition matrix rather
  /// than simply counting distinct outcomes.
  bool stochastic = false;

  double ba_tolerance = 1e-6;
  int ba_max_iter = 200;
};

/// Channel capacity of a discrete memoryless channel, in bits.
///
/// `p_y_given_x` is a (n_inputs, n_outputs) row-stochastic matrix; row i is
/// the distribution over outcomes given input i. Returns the capacity and
/// writes the capacity-achieving input distribution to `input_distribution`
/// when that pointer is non-null.
///
/// That distribution is *not* a policy the agent should follow. Empowerment
/// is a maximum over action distributions and measures potential, not intent
/// (section 4.4.2 of the paper).
double blahut_arimoto(
  const Eigen::MatrixXd & p_y_given_x,
  double tolerance = 1e-6,
  int max_iter = 200,
  Eigen::VectorXd * input_distribution = nullptr);

/// Run a sequence of actions from a state and return the OUTCOME vector.
using Rollout =
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const std::vector<int> &)>;

/// Estimates n-step empowerment for any system exposing a forward model.
///
/// The system is supplied as a callable so that this class knows nothing
/// about arms, grids, ROS, or simulators, plus one callable that decides
/// when two outcomes count as the same.
///
/// That last one is the whole content of section 4.6. In a grid world two
/// outcomes are the same square or they are not. In a continuous system
/// every outcome differs in the twelfth decimal place, so without a notion
/// of "close enough" the answer is always log2(n_sequences) and means
/// nothing.
class EmpowermentEstimator
{
public:
  EmpowermentEstimator(
    Rollout rollout,
    int n_actions,
    Discretiser discretiser,
    const EmpowermentConfig & config = EmpowermentConfig());

  /// Empowerment of `state`, in bits.
  double estimate(const Eigen::VectorXd & state) const;

  double operator()(const Eigen::VectorXd & state) const {return estimate(state);}

  /// Empowerment for a batch of states.
  Eigen::VectorXd map_over(const std::vector<Eigen::VectorXd> & states) const;

  const EmpowermentConfig & config() const {return config_;}

private:
  /// Either every sequence, or a random sample of them.
  std::vector<std::vector<int>> action_sequences() const;

  double capacity_from_outcomes(
    const std::vector<std::vector<int>> & sequences,
    const std::vector<OutcomeKey> & outcomes) const;

  Rollout rollout_;
  int n_actions_;
  Discretiser discretiser_;
  EmpowermentConfig config_;
};

}  // namespace intrinsic_core

#endif  // INTRINSIC_CORE__EMPOWERMENT_HPP_
