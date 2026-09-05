// UR5 + depth camera empowerment map over a table.
//
// Computes empowerment at every reachable point of a table-height plane under
// two conditions, and writes the numbers:
//
//     A  puck pushable      -- the agent can move it and the camera can see it
//     B  puck bolted down   -- present and visible, but immovable (control)
//     C  A - B              -- the puck's own contribution
//
// The outcome the agent is scored on is a DEPTH IMAGE, not ground-truth
// object pose. That is the faithful reading of the paper: empowerment is the
// channel from actuators to the agent's own sensors, so anything the camera
// cannot resolve does not count.
//
// The Python original drew the figure with matplotlib. There is no plotting
// library here, so this writes CSV -- x, y, pushable, bolted, contribution --
// and the committed figure is produced from that. The numbers are the
// artefact; the picture was only ever a rendering of them.
//
// Usage:
//     run_ur5_map [--grid 18] [--sequences 2000] [--out ur5_map.csv]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "intrinsic_core/batched.hpp"
#include "intrinsic_core/empowerment.hpp"
#include "intrinsic_core/sensors.hpp"
#include "intrinsic_core/ur5.hpp"

using namespace intrinsic_core;  // NOLINT(build/namespaces)

int main(int argc, char ** argv)
{
  int grid = 18, steps = 2, sequences = 2000;
  double dq = 0.10, depth_res = 0.02, tool_height = 0.08;
  double puck_x = -0.50, puck_y = 0.0;
  std::string out = "ur5_empowerment_map.csv";

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() {return (i + 1 < argc) ? argv[++i] : "";};
    if (a == "--grid") {grid = std::atoi(next());} else if (a == "--steps") {
      steps = std::atoi(next());
    } else if (a == "--sequences") {sequences = std::atoi(next());} else if (a == "--dq") {
      dq = std::atof(next());
    } else if (a == "--depth-res") {depth_res = std::atof(next());} else if (
      a == "--tool-height")
    {
      tool_height = std::atof(next());
    } else if (a == "--out") {out = next();}
  }

  UR5Config arm_cfg;
  arm_cfg.dq = dq;
  arm_cfg.active_joints = {0, 1, 2, 3};
  const UR5 arm(arm_cfg);
  const UR5 bolted = arm.with_fixed_puck();

  CameraConfig cam;
  cam.position = {puck_x, puck_y, 0.95};
  cam.look_at = {puck_x, puck_y, 0.0};
  cam.up = {1.0, 0.0, 0.0};
  cam.fov_deg = 30.0;
  cam.width = 16;
  cam.height = 12;
  const DepthCamera camera(cam);

  EmpowermentConfig cfg;
  cfg.n_steps = steps;
  cfg.n_sequences = sequences;
  cfg.exhaustive_below = 0;
  cfg.seed = 0;
  const Discretiser disc = depth_discretiser(depth_res);

  const BatchedDepthEmpowerment pushable_est(arm, camera, disc, cfg);
  const BatchedDepthEmpowerment bolted_est(bolted, camera, disc, cfg);

  FILE * f = std::fopen(out.c_str(), "w");
  if (f == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", out.c_str());
    return 1;
  }
  std::fprintf(f, "x,y,pushable,bolted,contribution\n");

  const double lo = -0.75, hi = -0.25;
  int solved = 0, total = 0;
  for (int i = 0; i < grid; ++i) {
    const double y = -0.25 + 0.50 * i / (grid - 1);
    for (int j = 0; j < grid; ++j) {
      const double x = lo + (hi - lo) * j / (grid - 1);
      ++total;
      const auto q = arm.inverse_kinematics(Eigen::Vector3d(x, y, tool_height));
      if (!q.has_value()) {continue;}
      ++solved;

      Eigen::VectorXd state(8);
      state << *q, puck_x, puck_y;
      const double a = pushable_est.estimate(state);
      const double b = bolted_est.estimate(state);
      std::fprintf(f, "%.4f,%.4f,%.6f,%.6f,%.6f\n", x, y, a, b, a - b);
    }
    std::fprintf(stderr, "\rrow %d/%d", i + 1, grid);
  }
  std::fclose(f);
  std::fprintf(stderr, "\n%d of %d grid points reachable -> %s\n", solved, total, out.c_str());
  return 0;
}
