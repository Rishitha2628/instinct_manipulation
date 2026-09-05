// 2-link planar arm empowerment map, with ground-truth outcomes.
//
// The same experiment as run_ur5_map, on the smallest system in which it
// still makes sense. It runs in seconds and is the right place to build
// intuition before touching Gazebo.
//
// Writes CSV rather than a figure: see the note in run_ur5_map.cpp.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "intrinsic_core/arm2d.hpp"
#include "intrinsic_core/discretise.hpp"
#include "intrinsic_core/empowerment.hpp"

using namespace intrinsic_core;  // NOLINT(build/namespaces)

int main(int argc, char ** argv)
{
  int grid = 40, steps = 3;
  double resolution = 0.025, puck_x = 0.30, puck_y = 0.25;
  std::string out = "empowerment_map.csv";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() {return (i + 1 < argc) ? argv[++i] : "";};
    if (a == "--grid") {grid = std::atoi(next());} else if (a == "--steps") {
      steps = std::atoi(next());
    } else if (a == "--resolution") {resolution = std::atof(next());} else if (
      a == "--out")
    {
      out = next();
    }
  }

  const PlanarArm arm;
  const PlanarArm bolted = arm.with_fixed_puck();

  EmpowermentConfig cfg;
  cfg.n_steps = steps;
  cfg.seed = 0;

  auto make = [&](const PlanarArm & a) {
      return EmpowermentEstimator(
        [a](const Eigen::VectorXd & s, const std::vector<int> & seq) {
          return a.rollout(s, seq);
        },
        9, grid_discretiser(resolution), cfg);
    };
  const auto pushable = make(arm);
  const auto fixed = make(bolted);

  FILE * f = std::fopen(out.c_str(), "w");
  if (f == nullptr) {return 1;}
  std::fprintf(f, "x,y,pushable,bolted,contribution\n");

  for (int i = 0; i < grid; ++i) {
    const double y = 0.0 + 0.50 * i / (grid - 1);
    for (int j = 0; j < grid; ++j) {
      const double x = -0.30 + 0.80 * j / (grid - 1);
      const auto ik = arm.inverse_kinematics(x, y);
      if (!ik.has_value()) {continue;}
      Eigen::VectorXd state(4);
      state << (*ik)[0], (*ik)[1], puck_x, puck_y;
      const double a = pushable.estimate(state);
      const double b = fixed.estimate(state);
      std::fprintf(f, "%.4f,%.4f,%.6f,%.6f,%.6f\n", x, y, a, b, a - b);
    }
  }
  std::fclose(f);
  std::fprintf(stderr, "wrote %s\n", out.c_str());
  return 0;
}
