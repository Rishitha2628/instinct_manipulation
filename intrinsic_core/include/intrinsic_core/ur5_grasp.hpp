// The UR5 with an object it can push, hold, lift and drop.
//
// grasp.hpp asks whether an intrinsic objective prefers holding, using a
// free-floating tool so that the arm's kinematics do not colour the answer.
// This is the same question put back on the real arm, and it is what the ROS
// grasp_climber plans with.
#ifndef INTRINSIC_CORE__UR5_GRASP_HPP_
#define INTRINSIC_CORE__UR5_GRASP_HPP_

#include <string>
#include <vector>

#include <Eigen/Core>

#include "intrinsic_core/discretise.hpp"
#include "intrinsic_core/sensors.hpp"
#include "intrinsic_core/ur5.hpp"

namespace intrinsic_core
{

// state layout: [q (6), object_xyz (3), held (1)] -> 10
constexpr int GRASP_Q = 0;
constexpr int GRASP_OBJ = 6;
constexpr int GRASP_HELD = 9;

struct UR5GraspConfig
{
  /// How close the tool point must be to the object's centre for a close
  /// command to take.
  double grasp_radius = 0.10;

  /// 1.0 pushable, 0.0 the bolted control.
  double push_efficiency = 1.0;

  /// Metres below the flange at which the gripper actually grips.
  ///
  /// This model's UR5 has no gripper: tool_position returns the flange, and
  /// tool_radius is widened to 0.09 to stand in for the gripper's bulk. That
  /// fudge is fine for scoring how much the arm can disturb a scene, and
  /// wrong the moment you try to grasp something, because it aims the FLANGE
  /// at the object and buries the fingers 0.09 m past it -- through the
  /// table. robotiq_85_tcp sits 0.090 m along the tool axis from tool0, so
  /// set this to that and the grip point is where the fingers are.
  ///
  /// Applied along the tool's own +Z axis, so it is correct whatever the
  /// wrist is doing.
  double tool_offset = 0.0;

  double max_lift = 0.45;
  double object_resolution = 0.02;
  double depth_resolution = 0.02;
};

class UR5GraspWorld
{
public:
  explicit UR5GraspWorld(
    const UR5 & arm = UR5(),
    const UR5GraspConfig & config = UR5GraspConfig());

  int n_joint_actions() const {return arm_.n_actions();}
  int close_action() const {return arm_.n_actions();}
  int open_action() const {return arm_.n_actions() + 1;}
  int n_actions() const {return arm_.n_actions() + 2;}

  double rest_height() const;

  /// Where the gripper actually grips, not where the flange is.
  ///
  /// Along the TOOL's own axis, not straight down. Assuming straight down is
  /// only correct if the wrist happens to be pointing at the table, and with
  /// position-only IK it generally is not: measured on this arm the solver
  /// returned a pose 33 degrees off vertical, which put the real pads 51 mm
  /// from where a straight-down assumption said they were -- further than the
  /// grasp radius, so the model declared a perfect grasp while the fingers
  /// closed on nothing.
  Eigen::Vector3d grip_point(const Eigen::VectorXd & q) const;

  Eigen::VectorXd state_at(
    const Eigen::VectorXd & q,
    const Eigen::Vector2d & obj_xy,
    bool grasped = false,
    double lift = 0.0) const;

  Eigen::VectorXd step(const Eigen::VectorXd & state, int action) const;
  Eigen::VectorXd rollout(
    const Eigen::VectorXd & state, const std::vector<int> & actions) const;

  UR5GraspWorld bolted() const;

  /// TRANSFER outcome: the object's state, nothing about the arm.
  OutcomeKey object_key(const Eigen::VectorXd & state) const;

  /// OWN-SENSOR outcome: what the camera sees.
  ///
  /// Not DepthCamera::render, because that pins the object to the table and
  /// the whole question here is what happens when it is lifted off it.
  OutcomeKey depth_key(const Eigen::VectorXd & state, const DepthCamera & camera) const;

  /// log2 of the number of distinguishable outcomes.
  ///
  /// The forward model is deterministic, so channel capacity reduces to
  /// counting distinct reachable outcomes. `n_sequences <= 0` enumerates
  /// exhaustively, which is affordable at horizon 2 (83^2 = 6889) and not at
  /// horizon 3 (571787).
  double empowerment(
    const Eigen::VectorXd & state,
    int horizon = 2,
    const std::string & outcome = "object",
    const DepthCamera * camera = nullptr,
    int n_sequences = 0,
    uint64_t seed = 0) const;

  const UR5 & arm() const {return arm_;}
  const UR5GraspConfig & config() const {return config_;}

private:
  UR5 arm_;
  UR5GraspConfig config_;
};

}  // namespace intrinsic_core

#endif  // INTRINSIC_CORE__UR5_GRASP_HPP_
