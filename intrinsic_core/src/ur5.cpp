#include "intrinsic_core/ur5.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace intrinsic_core
{

namespace
{
constexpr double kPi = 3.14159265358979323846;

double clamp(double v, double lo, double hi) {return std::min(std::max(v, lo), hi);}

Eigen::Matrix<double, 6, 3> make_dh()
{
  Eigen::Matrix<double, 6, 3> dh;
  dh << 0.0,      0.089159,  kPi / 2,
        -0.425,   0.0,       0.0,
        -0.39225, 0.0,       0.0,
        0.0,      0.10915,   kPi / 2,
        0.0,      0.09465,  -kPi / 2,
        0.0,      0.0823,    0.0;
  return dh;
}

Eigen::Matrix<double, 6, 2> make_limits()
{
  Eigen::Matrix<double, 6, 2> lim;
  lim << -kPi, kPi,
         -kPi, 0.0,
         -2.6, 2.6,
         -kPi, kPi,
         -kPi, kPi,
         -kPi, kPi;
  return lim;
}
}  // namespace

const Eigen::Matrix<double, 6, 3> DH_UR5 = make_dh();
const Eigen::Matrix<double, 6, 2> JOINT_LIMITS_UR5 = make_limits();

UR5::UR5(const UR5Config & config)
: config_(config)
{
  n_actions_ = 1;
  for (size_t i = 0; i < config_.active_joints.size(); ++i) {n_actions_ *= 3;}
  build_action_table();
}

Eigen::Matrix4d UR5::dh_transform(double theta, double a, double d, double alpha)
{
  const double ct = std::cos(theta), st = std::sin(theta);
  const double ca = std::cos(alpha), sa = std::sin(alpha);
  Eigen::Matrix4d T;
  T << ct, -st * ca,  st * sa, a * ct,
       st,  ct * ca, -ct * sa, a * st,
      0.0,       sa,       ca,      d,
      0.0,      0.0,      0.0,    1.0;
  return T;
}

Eigen::Matrix4d UR5::forward_kinematics(const Eigen::VectorXd & q) const
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 6; ++i) {
    T *= dh_transform(q[i], DH_UR5(i, 0), DH_UR5(i, 1), DH_UR5(i, 2));
  }
  return T;
}

Eigen::Vector3d UR5::tool_position(const Eigen::VectorXd & q) const
{
  return forward_kinematics(q).block<3, 1>(0, 3);
}

Eigen::Matrix<double, 7, 3> UR5::link_positions(const Eigen::VectorXd & q) const
{
  Eigen::Matrix<double, 7, 3> out;
  out.row(0).setZero();
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 6; ++i) {
    T *= dh_transform(q[i], DH_UR5(i, 0), DH_UR5(i, 1), DH_UR5(i, 2));
    out.row(i + 1) = T.block<3, 1>(0, 3).transpose();
  }
  return out;
}

bool UR5::above_table(const Eigen::VectorXd & q, double margin) const
{
  const Eigen::Matrix<double, 7, 3> p = link_positions(q);
  for (int i = 2; i < 7; ++i) {           // skip base links
    if (!(p(i, 2) > config_.table_height + margin)) {return false;}
  }
  return true;
}

void UR5::build_action_table()
{
  // Each action is a delta over the active joints only.
  const int n = static_cast<int>(config_.active_joints.size());
  action_table_ = Eigen::MatrixXd::Zero(n_actions_, 6);
  for (int idx = 0; idx < n_actions_; ++idx) {
    int rem = idx;
    for (int k = 0; k < n; ++k) {
      action_table_(idx, config_.active_joints[static_cast<size_t>(k)]) =
        static_cast<double>(rem % 3) - 1.0;      // -1, 0, +1
      rem /= 3;
    }
  }
  action_table_ *= config_.dq;
}

Eigen::VectorXd UR5::step(const Eigen::VectorXd & state, int action) const
{
  const UR5Config & c = config_;
  Eigen::VectorXd q = state.head<6>() + action_table_.row(action).transpose();
  for (int i = 0; i < 6; ++i) {
    q[i] = clamp(q[i], c.joint_limits(i, 0), c.joint_limits(i, 1));
  }

  double px = state[6], py = state[7];

  if (c.puck_mass_factor > 0.0) {
    const Eigen::Vector3d tool = tool_position(q);
    double dx = px - tool[0], dy = py - tool[1];
    double planar = std::hypot(dx, dy);
    const double contact = c.tool_radius + c.puck_radius;

    // Only push if the tool is also low enough to touch the puck.
    const bool near_table = tool[2] < c.table_height + contact;
    if (near_table && planar < contact) {
      if (planar < 1e-9) {dx = 1.0; dy = 0.0; planar = 1.0;}
      const double overlap = contact - planar;
      const double scale = c.puck_mass_factor * overlap / planar;
      px = clamp(px + dx * scale, c.x_min, c.x_max);
      py = clamp(py + dy * scale, c.y_min, c.y_max);
    }
  }

  Eigen::VectorXd out(8);
  out << q, px, py;
  return out;
}

Eigen::VectorXd UR5::rollout_state(
  const Eigen::VectorXd & state, const std::vector<int> & actions) const
{
  Eigen::VectorXd s = state;
  for (int a : actions) {s = step(s, a);}
  return s;
}

Eigen::VectorXd UR5::rollout_pose(
  const Eigen::VectorXd & state, const std::vector<int> & actions) const
{
  const Eigen::VectorXd s = rollout_state(state, actions);
  Eigen::VectorXd out(5);
  out << tool_position(s.head<6>()), s[6], s[7];
  return out;
}

Eigen::VectorXd UR5::solve_from(
  const Eigen::VectorXd & seed,
  const Eigen::Vector3d & target,
  const Eigen::Vector3d * tool_axis) const
{
  // Damped least squares on the geometric Jacobian. scipy's L-BFGS-B over a
  // scalar cost was the original solver; DLS reaches the same poses in far
  // fewer forward-kinematics calls, which matters because the climber solves
  // IK inside a 1 Hz control loop.
  Eigen::VectorXd q = seed;
  for (int i = 0; i < 6; ++i) {
    q[i] = clamp(q[i], config_.joint_limits(i, 0), config_.joint_limits(i, 1));
  }

  const int max_iter = 200;
  const double lambda = 1e-4;

  for (int iter = 0; iter < max_iter; ++iter) {
    // Frame origins and z-axes for every joint, plus the tool pose.
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Matrix<double, 3, 6> origins;
    Eigen::Matrix<double, 3, 6> axes;
    for (int i = 0; i < 6; ++i) {
      origins.col(i) = T.block<3, 1>(0, 3);
      axes.col(i) = T.block<3, 1>(0, 2);
      T *= dh_transform(q[i], DH_UR5(i, 0), DH_UR5(i, 1), DH_UR5(i, 2));
    }
    const Eigen::Vector3d tool = T.block<3, 1>(0, 3);
    const Eigen::Vector3d tool_z = T.block<3, 1>(0, 2);

    const Eigen::Vector3d e_pos = target - tool;
    Eigen::Vector3d e_rot = Eigen::Vector3d::Zero();
    if (tool_axis != nullptr) {
      // Rotation that would swing the tool's z onto the requested axis.
      e_rot = tool_z.cross(*tool_axis);
    }

    const bool aimed = (tool_axis == nullptr) ||
      (tool_z.dot(*tool_axis) > 1.0 - 1e-9);
    if (e_pos.norm() < 1e-9 && aimed) {break;}

    const int rows = (tool_axis == nullptr) ? 3 : 6;
    Eigen::MatrixXd J(rows, 6);
    for (int i = 0; i < 6; ++i) {
      const Eigen::Vector3d linear = axes.col(i).cross(tool - origins.col(i));
      J.block<3, 1>(0, i) = linear;
      if (rows == 6) {J.block<3, 1>(3, i) = axes.col(i);}
    }

    Eigen::VectorXd residual(rows);
    residual.head<3>() = e_pos;
    if (rows == 6) {residual.tail<3>() = e_rot;}

    const Eigen::MatrixXd JJt =
      J * J.transpose() + lambda * Eigen::MatrixXd::Identity(rows, rows);
    const Eigen::VectorXd dq = J.transpose() * JJt.ldlt().solve(residual);

    // A full DLS step can be enormous near a singularity; cap it so the
    // solve walks rather than teleports.
    const double step_norm = dq.norm();
    const double max_step = 0.2;
    const Eigen::VectorXd applied =
      (step_norm > max_step) ? (dq * (max_step / step_norm)).eval() : dq;

    for (int i = 0; i < 6; ++i) {
      q[i] = clamp(q[i] + applied[i], config_.joint_limits(i, 0),
        config_.joint_limits(i, 1));
    }
  }
  return q;
}

std::optional<Eigen::VectorXd> UR5::inverse_kinematics(
  const Eigen::Vector3d & target_xyz,
  const Eigen::VectorXd * q_seed,
  double tol,
  int restarts,
  uint64_t seed,
  const Eigen::Vector3d * tool_axis,
  double aim_tol) const
{
  Eigen::Vector3d axis;
  const Eigen::Vector3d * axis_ptr = nullptr;
  if (tool_axis != nullptr) {
    axis = tool_axis->normalized();
    axis_ptr = &axis;
  }

  std::vector<Eigen::VectorXd> seeds;
  if (q_seed != nullptr) {seeds.push_back(*q_seed);}
  Eigen::VectorXd canonical(6);
  canonical << 0.0, -1.2, 1.4, -1.75, -1.57, 0.0;
  seeds.push_back(canonical);

  std::mt19937_64 rng(seed);
  for (int r = 0; r < restarts; ++r) {
    Eigen::VectorXd s(6);
    for (int i = 0; i < 6; ++i) {
      std::uniform_real_distribution<double> pick(
        config_.joint_limits(i, 0), config_.joint_limits(i, 1));
      s[i] = pick(rng);
    }
    seeds.push_back(s);
  }

  Eigen::VectorXd best;
  double best_error = std::numeric_limits<double>::infinity();

  for (const auto & s : seeds) {
    const Eigen::VectorXd q = solve_from(s, target_xyz, axis_ptr);
    const double reached = (tool_position(q) - target_xyz).norm();
    if (reached < best_error) {best_error = reached; best = q;}

    // Stop on the POSITION error, not on the objective. With an orientation
    // term the objective also carries the aim penalty, which is rarely below
    // tol^2, so testing the objective meant the early exit never fired and
    // every call ground through all 14 seeds. Measured effect on the caller:
    // its control loop dropped to 0.065 Hz, roughly one command every 15
    // seconds, which looks exactly like an arm that is ignoring you.
    if (best_error < tol) {
      if (axis_ptr == nullptr) {break;}
      const double aimed = forward_kinematics(best).block<3, 1>(0, 2).dot(*axis_ptr);
      if (aimed > 1.0 - aim_tol) {break;}
    }
  }

  if (best.size() == 0 || !above_table(best)) {return std::nullopt;}
  if ((tool_position(best) - target_xyz).norm() > tol) {return std::nullopt;}
  return best;
}

UR5 UR5::with_fixed_puck() const
{
  UR5Config c = config_;
  c.puck_mass_factor = 0.0;
  return UR5(c);
}

}  // namespace intrinsic_core
