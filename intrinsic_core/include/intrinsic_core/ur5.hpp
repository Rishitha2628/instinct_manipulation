// UR5 forward kinematics and a pushable object on a table.
//
// Uses the standard UR5 Denavit-Hartenberg parameters, so joint angles here
// correspond to real /joint_states values from a ur_robot_driver setup and
// the tool pose matches what MoveIt reports.
//
// Contact is a kinematic approximation, not physics: if the tool sphere
// overlaps the puck sphere, the puck is displaced along the separation axis
// and stays on the table plane. That is enough for the empowerment result --
// what matters is that being near the puck creates *more distinguishable
// outcomes*, not that the contact forces are right.
#ifndef INTRINSIC_CORE__UR5_HPP_
#define INTRINSIC_CORE__UR5_HPP_

#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace intrinsic_core
{

/// Standard UR5 DH parameters: (a, d, alpha), one row per joint.
extern const Eigen::Matrix<double, 6, 3> DH_UR5;

/// Real UR5 joints are +-2pi; these are tightened to keep the arm above the
/// table and out of self-collision without needing a collision checker.
extern const Eigen::Matrix<double, 6, 2> JOINT_LIMITS_UR5;

struct UR5Config
{
  /// Radians per joint per control step. The paper's power constraint
  /// (sec 4.7.4) -- raising it can invert the landscape, not just scale it.
  double dq = 0.10;

  /// Which joints the agent may move. Using all six gives 3^6 = 729 actions
  /// per step, so horizon 3 is 387 million sequences and sampling becomes
  /// very thin. The wrist contributes little to where the tool *is*, so the
  /// default drives the arm and leaves wrist 2/3 fixed.
  std::vector<int> active_joints{0, 1, 2, 3};

  double tool_radius = 0.05;
  double puck_radius = 0.04;

  /// 1.0 pushable, 0.0 bolted down (the paper's immovable-box control).
  double puck_mass_factor = 1.0;

  double table_height = 0.0;
  double x_min = -0.85, x_max = 0.85, y_min = -0.85, y_max = 0.85;

  Eigen::Matrix<double, 6, 2> joint_limits = JOINT_LIMITS_UR5;
};

/// Analytic UR5 kinematics + kinematic puck contact.
class UR5
{
public:
  explicit UR5(const UR5Config & config = UR5Config());

  // -- kinematics -----------------------------------------------------------

  /// Full 4x4 tool pose in the base frame.
  Eigen::Matrix4d forward_kinematics(const Eigen::VectorXd & q) const;
  Eigen::Vector3d tool_position(const Eigen::VectorXd & q) const;

  /// Origin of every link frame -- used by the depth camera to draw the arm,
  /// and for a crude table-collision check. Returns a (7,3) matrix.
  Eigen::Matrix<double, 7, 3> link_positions(const Eigen::VectorXd & q) const;

  bool above_table(const Eigen::VectorXd & q, double margin = 0.02) const;

  // -- dynamics -------------------------------------------------------------

  /// state = [q0..q5, px, py]. Returns the successor state.
  Eigen::VectorXd step(const Eigen::VectorXd & state, int action) const;
  Eigen::VectorXd rollout_state(
    const Eigen::VectorXd & state, const std::vector<int> & actions) const;

  /// Ground-truth outcome: [tool_xyz, puck_xy].
  ///
  /// Useful as a baseline, but note this is *not* what the paper asks for --
  /// empowerment is the channel to the agent's own SENSORS, so the
  /// depth-camera outcome in sensors.hpp is the faithful version.
  Eigen::VectorXd rollout_pose(
    const Eigen::VectorXd & state, const std::vector<int> & actions) const;

  // -- inverse kinematics ---------------------------------------------------

  /// Numerical IK. Position-only by default; pass `tool_axis` to aim it.
  ///
  /// Position-only is the default because the tool orientation is then left
  /// free and the arm keeps the redundancy that makes multiple approaches
  /// possible. That is the right choice for asking how much of the scene the
  /// arm can disturb, which is what empowerment needs.
  ///
  /// It is the wrong choice for grasping, and quietly so. A grasp is defined
  /// by orientation as much as by position: the fingers have to straddle the
  /// object. With the orientation unconstrained the solver will happily
  /// return a pose that puts the FLANGE where you asked while the gripper
  /// points 33 degrees off vertical, and anything that assumes the tool
  /// hangs straight down is then wrong by the length of the gripper.
  /// Measured on this arm: the caller believed the pads were on the object,
  /// and they were 51 mm away, which is how a grasp closes on empty air
  /// while every number involved looks correct.
  std::optional<Eigen::VectorXd> inverse_kinematics(
    const Eigen::Vector3d & target_xyz,
    const Eigen::VectorXd * q_seed = nullptr,
    double tol = 1e-3,
    int restarts = 12,
    uint64_t seed = 0,
    const Eigen::Vector3d * tool_axis = nullptr,
    double aim_tol = 2e-3) const;

  // -- control conditions ---------------------------------------------------

  UR5 with_fixed_puck() const;

  int n_actions() const {return n_actions_;}
  const UR5Config & config() const {return config_;}

  /// The (n_actions, 6) table of joint deltas. Exposed so that the grasp
  /// world can extend exactly this action set with open and close rather
  /// than inventing a second one.
  const Eigen::MatrixXd & action_table() const {return action_table_;}

private:
  static Eigen::Matrix4d dh_transform(double theta, double a, double d, double alpha);
  void build_action_table();

  /// One damped-least-squares solve from a single seed. Returns the joint
  /// vector it settled on, whether or not it converged.
  Eigen::VectorXd solve_from(
    const Eigen::VectorXd & seed,
    const Eigen::Vector3d & target,
    const Eigen::Vector3d * tool_axis) const;

  UR5Config config_;
  int n_actions_ = 0;
  Eigen::MatrixXd action_table_;   // (n_actions, 6)
};

}  // namespace intrinsic_core

#endif  // INTRINSIC_CORE__UR5_HPP_
