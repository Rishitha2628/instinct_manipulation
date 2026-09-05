#include "intrinsic_core/empowerment.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <set>

namespace intrinsic_core
{

double blahut_arimoto(
  const Eigen::MatrixXd & p_y_given_x_in,
  double tolerance,
  int max_iter,
  Eigen::VectorXd * input_distribution)
{
  const Eigen::Index n_in = p_y_given_x_in.rows();
  const Eigen::Index n_out = p_y_given_x_in.cols();

  if (n_in == 0 || n_out == 0) {
    if (input_distribution) {*input_distribution = Eigen::VectorXd::Zero(n_in);}
    return 0.0;
  }

  // Rows must be normalised; guard against all-zero rows.
  Eigen::MatrixXd p = p_y_given_x_in;
  for (Eigen::Index i = 0; i < n_in; ++i) {
    const double sum = p.row(i).sum();
    p.row(i) /= (sum > 0.0) ? sum : 1.0;
  }

  Eigen::VectorXd r = Eigen::VectorXd::Constant(n_in, 1.0 / static_cast<double>(n_in));
  double prev_capacity = -std::numeric_limits<double>::infinity();

  for (int iter = 0; iter < max_iter; ++iter) {
    // q(x|y) proportional to r(x) p(y|x)
    Eigen::MatrixXd joint = r.asDiagonal() * p;
    Eigen::RowVectorXd marginal = joint.colwise().sum();

    Eigen::MatrixXd q = Eigen::MatrixXd::Zero(n_in, n_out);
    for (Eigen::Index j = 0; j < n_out; ++j) {
      if (marginal[j] > 0.0) {q.col(j) = joint.col(j) / marginal[j];}
    }

    // r(x) proportional to exp( sum_y p(y|x) log q(x|y) )
    Eigen::VectorXd exponent = Eigen::VectorXd::Zero(n_in);
    for (Eigen::Index i = 0; i < n_in; ++i) {
      double acc = 0.0;
      for (Eigen::Index j = 0; j < n_out; ++j) {
        if (q(i, j) > 0.0) {acc += p(i, j) * std::log(q(i, j));}
      }
      exponent[i] = acc;
    }

    exponent.array() -= exponent.maxCoeff();      // stabilise
    Eigen::VectorXd r_new = exponent.array().exp();
    const double total = r_new.sum();
    if (total <= 0.0) {break;}
    r_new /= total;

    // current capacity estimate, in nats then converted
    double capacity = 0.0;
    for (Eigen::Index i = 0; i < n_in; ++i) {
      if (r_new[i] <= 0.0) {continue;}
      for (Eigen::Index j = 0; j < n_out; ++j) {
        if (q(i, j) > 0.0) {
          capacity += r_new[i] * p(i, j) * std::log(q(i, j) / r_new[i]);
        }
      }
    }

    const bool converged = std::abs(capacity - prev_capacity) < tolerance;
    r = r_new;
    prev_capacity = capacity;
    if (converged) {break;}
  }

  if (input_distribution) {*input_distribution = r;}
  return std::max(prev_capacity, 0.0) / std::log(2.0);
}

EmpowermentEstimator::EmpowermentEstimator(
  Rollout rollout,
  int n_actions,
  Discretiser discretiser,
  const EmpowermentConfig & config)
: rollout_(std::move(rollout)),
  n_actions_(n_actions),
  discretiser_(std::move(discretiser)),
  config_(config)
{
}

std::vector<std::vector<int>> EmpowermentEstimator::action_sequences() const
{
  // |A|^n, computed in double so that a huge alphabet cannot silently wrap
  // an integer and turn "far too many to enumerate" into "few enough".
  const double total_real =
    std::pow(static_cast<double>(n_actions_), static_cast<double>(config_.n_steps));

  if (total_real <= static_cast<double>(config_.exhaustive_below)) {
    // enumerate exactly: digits of 0..total-1 in base n_actions
    const int64_t total = static_cast<int64_t>(total_real);
    std::vector<std::vector<int>> seqs(
      static_cast<size_t>(total), std::vector<int>(static_cast<size_t>(config_.n_steps)));
    for (int64_t idx = 0; idx < total; ++idx) {
      int64_t rem = idx;
      for (int k = config_.n_steps - 1; k >= 0; --k) {
        seqs[static_cast<size_t>(idx)][static_cast<size_t>(k)] =
          static_cast<int>(rem % n_actions_);
        rem /= n_actions_;
      }
    }
    return seqs;
  }

  std::mt19937_64 rng(config_.seed);
  std::uniform_int_distribution<int> pick(0, n_actions_ - 1);
  std::vector<std::vector<int>> seqs(
    static_cast<size_t>(config_.n_sequences),
    std::vector<int>(static_cast<size_t>(config_.n_steps)));
  for (auto & seq : seqs) {
    for (auto & a : seq) {a = pick(rng);}
  }
  return seqs;
}

double EmpowermentEstimator::estimate(const Eigen::VectorXd & state) const
{
  const std::vector<std::vector<int>> sequences = action_sequences();

  std::vector<OutcomeKey> outcomes;
  outcomes.reserve(sequences.size());
  for (const auto & seq : sequences) {
    outcomes.push_back(discretiser_(rollout_(state, seq)));
  }

  if (config_.stochastic) {
    return capacity_from_outcomes(sequences, outcomes);
  }

  // Deterministic case. This is the entire idea:
  //   count the distinct places you can end up, take the log.
  if (outcomes.empty()) {return 0.0;}
  const std::set<OutcomeKey> distinct(outcomes.begin(), outcomes.end());
  return std::log2(static_cast<double>(distinct.size()));
}

double EmpowermentEstimator::capacity_from_outcomes(
  const std::vector<std::vector<int>> & sequences,
  const std::vector<OutcomeKey> & outcomes) const
{
  // Build an empirical p(outcome | sequence) and run Blahut-Arimoto.
  //
  // Only meaningful when the rollout is genuinely stochastic and each
  // sequence has been rolled out more than once; with one sample per
  // sequence the matrix is one-hot and this reduces to the counting
  // estimator above (with more arithmetic).
  std::map<OutcomeKey, Eigen::Index> key_index;
  for (const auto & o : outcomes) {
    key_index.emplace(o, 0);
  }
  Eigen::Index next = 0;
  for (auto & kv : key_index) {kv.second = next++;}

  std::map<std::vector<int>, Eigen::Index> seq_index;
  for (const auto & s : sequences) {
    seq_index.emplace(s, 0);
  }
  next = 0;
  for (auto & kv : seq_index) {kv.second = next++;}

  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(
    static_cast<Eigen::Index>(seq_index.size()),
    static_cast<Eigen::Index>(key_index.size()));
  for (size_t i = 0; i < sequences.size(); ++i) {
    matrix(seq_index.at(sequences[i]), key_index.at(outcomes[i])) += 1.0;
  }

  return blahut_arimoto(matrix, config_.ba_tolerance, config_.ba_max_iter);
}

Eigen::VectorXd EmpowermentEstimator::map_over(
  const std::vector<Eigen::VectorXd> & states) const
{
  Eigen::VectorXd out(static_cast<Eigen::Index>(states.size()));
  for (size_t i = 0; i < states.size(); ++i) {
    out[static_cast<Eigen::Index>(i)] = estimate(states[i]);
  }
  return out;
}

}  // namespace intrinsic_core
