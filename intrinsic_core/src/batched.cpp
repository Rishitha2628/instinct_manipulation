#include "intrinsic_core/batched.hpp"

#include <cmath>
#include <random>
#include <set>

namespace intrinsic_core
{

BatchedDepthEmpowerment::BatchedDepthEmpowerment(
  const UR5 & arm,
  const DepthCamera & camera,
  Discretiser discretiser,
  const EmpowermentConfig & config)
: arm_(arm), camera_(camera), discretiser_(std::move(discretiser)), config_(config)
{
  build_sequences();
}

void BatchedDepthEmpowerment::build_sequences()
{
  const int n_a = arm_.n_actions();
  const double total_real =
    std::pow(static_cast<double>(n_a), static_cast<double>(config_.n_steps));

  if (total_real <= static_cast<double>(config_.exhaustive_below)) {
    const int64_t total = static_cast<int64_t>(total_real);
    sequences_.assign(
      static_cast<size_t>(total), std::vector<int>(static_cast<size_t>(config_.n_steps)));
    for (int64_t idx = 0; idx < total; ++idx) {
      int64_t rem = idx;
      for (int k = config_.n_steps - 1; k >= 0; --k) {
        sequences_[static_cast<size_t>(idx)][static_cast<size_t>(k)] =
          static_cast<int>(rem % n_a);
        rem /= n_a;
      }
    }
    return;
  }

  std::mt19937_64 rng(config_.seed);
  std::uniform_int_distribution<int> pick(0, n_a - 1);
  sequences_.assign(
    static_cast<size_t>(config_.n_sequences),
    std::vector<int>(static_cast<size_t>(config_.n_steps)));
  for (auto & seq : sequences_) {
    for (auto & a : seq) {a = pick(rng);}
  }
}

double BatchedDepthEmpowerment::estimate(const Eigen::VectorXd & state) const
{
  std::set<OutcomeKey> distinct;
  for (const auto & seq : sequences_) {
    const Eigen::VectorXd s = arm_.rollout_state(state, seq);
    const Eigen::MatrixXd points = arm_.link_positions(s.head<6>());
    const Eigen::Vector2d puck(s[6], s[7]);
    distinct.insert(
      discretiser_(camera_.render(points, puck, arm_.config().puck_radius)));
  }
  return distinct.empty() ? 0.0 : std::log2(static_cast<double>(distinct.size()));
}

}  // namespace intrinsic_core
