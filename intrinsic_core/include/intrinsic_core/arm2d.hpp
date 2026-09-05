// A 2-link planar arm on a table with a puck.
//
// This is deliberately the smallest system in which the paper's box
// experiment (section 4.5.4) still makes sense. The four conditions there
// were: the box is pushable or not, and perceivable or not. All four are
// reproducible here by changing puck_mass_factor and by dropping the puck
// dimensions from the discretiser.
//
// State vector:    [q1, q2, px, py]   joint angles, puck position
// Outcome vector:  [ex, ey, px, py]   end-effector position, puck position
//
// The outcome deliberately includes the puck. That is the entire mechanism:
// near the puck, two different action sequences that put the gripper in the
// same place can leave the puck in different places, so they count as
// different outcomes. Away from the puck they collapse into one. Nobody
// tells the arm the puck is interesting; it falls out of the counting.
#ifndef INTRINSIC_CORE__ARM2D_HPP_
#define INTRINSIC_CORE__ARM2D_HPP_

#include <optional>
#include <vector>

#include <Eigen/Core>

namespace intrinsic_core
{

/// Nine actions: each joint moves down / holds / moves up.
extern const Eigen::Matrix<double, 9, 2> JOINT_ACTIONS;

struct ArmConfig
{
  double l1 = 0.30;
  double l2 = 0.25;

  /// Radians per step per joint. The paper's "power constraint" in section
  /// 4.7.4 -- and it matters as much here as it does there.
  double dq = 0.12;

  double q1_min = -2.6, q1_max = 2.6;
  double q2_min = -2.4, q2_max = 2.4;

  /// Contact distance. Inside this, the gripper pushes the puck.
  double puck_radius = 0.045;

  /// 1.0 = puck moves fully with the gripper.
  /// 0.0 = puck is bolted down (the paper's immovable-box condition).
  double puck_mass_factor = 1.0;

  /// The rectangle the puck cannot leave.
  double x_min = -0.55, x_max = 0.55, y_min = -0.10, y_max = 0.55;
};

/// Deterministic forward model. No physics engine, no ROS, no learning.
class PlanarArm
{
public:
  explicit PlanarArm(const ArmConfig & config = ArmConfig());

  Eigen::Vector2d forward_kinematics(double q1, double q2) const;
  bool reachable(double x, double y) const;

  /// Returns (q1, q2), or nothing if unreachable or outside the joint limits.
  std::optional<Eigen::Vector2d> inverse_kinematics(
    double x, double y, bool elbow_up = true) const;

  /// One control step. state = [q1, q2, px, py].
  Eigen::VectorXd step(const Eigen::VectorXd & state, int action) const;

  /// Run a sequence of actions and return the OUTCOME vector.
  Eigen::VectorXd rollout(
    const Eigen::VectorXd & state, const std::vector<int> & actions) const;

  /// The immovable-box condition: present, but cannot be pushed.
  PlanarArm with_fixed_puck() const;

  const ArmConfig & config() const {return config_;}

private:
  ArmConfig config_;
};

}  // namespace intrinsic_core

#endif  // INTRINSIC_CORE__ARM2D_HPP_
