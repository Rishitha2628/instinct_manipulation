#include "intrinsic_core/arm2d.hpp"

#include <algorithm>
#include <cmath>

namespace intrinsic_core
{

namespace
{
Eigen::Matrix<double, 9, 2> build_joint_actions()
{
  Eigen::Matrix<double, 9, 2> table;
  int row = 0;
  for (int a = -1; a <= 1; ++a) {
    for (int b = -1; b <= 1; ++b) {
      table(row, 0) = static_cast<double>(a);
      table(row, 1) = static_cast<double>(b);
      ++row;
    }
  }
  return table;
}

double clamp(double v, double lo, double hi)
{
  return std::min(std::max(v, lo), hi);
}
}  // namespace

const Eigen::Matrix<double, 9, 2> JOINT_ACTIONS = build_joint_actions();

PlanarArm::PlanarArm(const ArmConfig & config)
: config_(config)
{
}

Eigen::Vector2d PlanarArm::forward_kinematics(double q1, double q2) const
{
  const ArmConfig & c = config_;
  return Eigen::Vector2d(
    c.l1 * std::cos(q1) + c.l2 * std::cos(q1 + q2),
    c.l1 * std::sin(q1) + c.l2 * std::sin(q1 + q2));
}

bool PlanarArm::reachable(double x, double y) const
{
  const ArmConfig & c = config_;
  const double r = std::hypot(x, y);
  return std::abs(c.l1 - c.l2) + 1e-6 < r && r < (c.l1 + c.l2) - 1e-6;
}

std::optional<Eigen::Vector2d> PlanarArm::inverse_kinematics(
  double x, double y, bool elbow_up) const
{
  const ArmConfig & c = config_;
  const double r2 = x * x + y * y;
  const double cos_q2 = (r2 - c.l1 * c.l1 - c.l2 * c.l2) / (2.0 * c.l1 * c.l2);
  if (!(cos_q2 >= -1.0 && cos_q2 <= 1.0)) {return std::nullopt;}

  double q2 = std::acos(cos_q2);
  if (!elbow_up) {q2 = -q2;}
  const double q1 = std::atan2(y, x) -
    std::atan2(c.l2 * std::sin(q2), c.l1 + c.l2 * std::cos(q2));

  if (q1 < c.q1_min || q1 > c.q1_max) {return std::nullopt;}
  if (q2 < c.q2_min || q2 > c.q2_max) {return std::nullopt;}
  return Eigen::Vector2d(q1, q2);
}

Eigen::VectorXd PlanarArm::step(const Eigen::VectorXd & state, int action) const
{
  const ArmConfig & c = config_;
  double q1 = state[0], q2 = state[1], px = state[2], py = state[3];

  q1 = clamp(q1 + JOINT_ACTIONS(action, 0) * c.dq, c.q1_min, c.q1_max);
  q2 = clamp(q2 + JOINT_ACTIONS(action, 1) * c.dq, c.q2_min, c.q2_max);

  const Eigen::Vector2d e = forward_kinematics(q1, q2);

  // Push the puck if the gripper is inside contact range.
  if (c.puck_mass_factor > 0.0) {
    double dx = px - e[0], dy = py - e[1];
    double dist = std::hypot(dx, dy);
    if (dist < c.puck_radius) {
      if (dist < 1e-9) {dx = 1.0; dy = 0.0; dist = 1.0;}
      const double overlap = c.puck_radius - dist;
      const double scale = c.puck_mass_factor * overlap / dist;
      px = clamp(px + dx * scale, c.x_min, c.x_max);
      py = clamp(py + dy * scale, c.y_min, c.y_max);
    }
  }

  Eigen::VectorXd out(4);
  out << q1, q2, px, py;
  return out;
}

Eigen::VectorXd PlanarArm::rollout(
  const Eigen::VectorXd & state, const std::vector<int> & actions) const
{
  Eigen::VectorXd s = state;
  for (int a : actions) {s = step(s, a);}
  const Eigen::Vector2d e = forward_kinematics(s[0], s[1]);
  Eigen::VectorXd outcome(4);
  outcome << e[0], e[1], s[2], s[3];
  return outcome;
}

PlanarArm PlanarArm::with_fixed_puck() const
{
  ArmConfig c = config_;
  c.puck_mass_factor = 0.0;
  return PlanarArm(c);
}

}  // namespace intrinsic_core
