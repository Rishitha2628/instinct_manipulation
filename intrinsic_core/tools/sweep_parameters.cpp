// Finds the usable band of the discretiser's resolution.
//
// Too coarse and everything collapses into one bin (0 bits everywhere); too
// fine and nothing collapses at all (saturated everywhere). A usable
// resolution sits in the middle, away from both. Run this before trusting any
// map.
#include <cstdio>
#include <vector>

#include "intrinsic_core/arm2d.hpp"
#include "intrinsic_core/discretise.hpp"
#include "intrinsic_core/empowerment.hpp"

using namespace intrinsic_core;  // NOLINT(build/namespaces)

int main()
{
  const PlanarArm arm;
  const Eigen::Vector2d puck(0.30, 0.25);

  auto state_at = [&](double x, double y) {
      const auto ik = arm.inverse_kinematics(x, y);
      Eigen::VectorXd s(4);
      s << (*ik)[0], (*ik)[1], puck[0], puck[1];
      return s;
    };
  const Eigen::VectorXd near = state_at(0.30, 0.25);
  const Eigen::VectorXd far = state_at(0.30, 0.42);

  std::printf("%12s %10s %10s %10s\n", "resolution", "near", "far", "contrast");
  for (double r : {100.0, 1.0, 0.20, 0.10, 0.05, 0.025, 0.01, 0.005, 1e-9}) {
    EmpowermentConfig cfg;
    cfg.n_steps = 3;
    cfg.seed = 0;
    const EmpowermentEstimator est(
      [&arm](const Eigen::VectorXd & s, const std::vector<int> & a) {
        return arm.rollout(s, a);
      },
      9, grid_discretiser(r), cfg);
    const double n = est.estimate(near);
    const double f = est.estimate(far);
    std::printf("%12.6f %10.3f %10.3f %10.3f\n", r, n, f, n - f);
  }
  return 0;
}
