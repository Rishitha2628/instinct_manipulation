// Can an intrinsic objective produce a GRASP, or does grasping look like a
// loss?
//
// This is the experiment the rest of the project could not run, because the
// model it runs on cannot represent a grasp at all. ur5.hpp has exactly one
// object interaction -- a kinematic shove in the table plane -- and the
// object's state is (x, y), so "off the table" is literally unrepresentable.
// No amount of tuning gets a lift out of that.
//
// Four things are added here, and every one of them is required before the
// question can even be asked:
//
//   1. the object gets a z, so it can leave the table;
//   2. an ATTACHMENT state, in which the object's pose is a rigid function of
//      the tool's, so holding is a thing that exists;
//   3. open/close in the action set, so the agent can choose to grasp;
//   4. MACRO-ACTIONS in task space rather than joint increments, because a
//      grasp is ten-plus primitive steps and 81^h enumeration dies long
//      before that. Eight macro-actions at horizon 3 is 512 sequences, which
//      is small enough to enumerate EXHAUSTIVELY -- so unlike the main
//      pipeline, these numbers are not sampling-limited.
//
// The agent is modelled as a free-floating tool rather than a UR5. That is
// deliberate: the question is whether the OBJECTIVE prefers holding, and an
// arm's kinematics add a large, pose-dependent background that would have to
// be subtracted back out. If holding does not win for a free-flying gripper
// that can go anywhere, it certainly will not win once joint limits and
// singularities are in the way.
//
// Two objectives are compared, and they are expected to disagree:
//
//   OWN-SENSOR empowerment, the one the project already uses: the outcome is
//   the depth image. The catch is that once you are holding something, its
//   position is a deterministic function of your own, so it stops
//   contributing any independent variety to the picture. Grasping may
//   therefore score no better than waving the gripper around in free space,
//   and WORSE than shoving, where the object moves semi-independently.
//
//   TRANSFER empowerment: the outcome is the OBJECT's state alone. Untouched,
//   the object never moves, so there is exactly one outcome and zero bits.
//   Pushed, it reaches a handful of spots on the table. Held, it goes
//   anywhere the tool can go, in three dimensions. Here grasping should be
//   the global maximum -- and derived, rather than rewarded.
#ifndef INTRINSIC_CORE__GRASP_HPP_
#define INTRINSIC_CORE__GRASP_HPP_

#include <string>
#include <vector>

#include <Eigen/Core>

#include "intrinsic_core/discretise.hpp"
#include "intrinsic_core/sensors.hpp"

namespace intrinsic_core
{

struct GraspConfig
{
  double table_height = -0.25;
  double object_radius = 0.04;
  double tool_radius = 0.03;

  /// Metres a single macro-action moves the tool. One macro-action stands for
  /// a short scripted move, not a joint increment.
  double step = 0.05;

  /// How close the tool centre must be to the object's centre for a close
  /// command to take hold.
  double grasp_radius = 0.06;

  /// 1.0 means a shove displaces the object by the full overlap. 0.0 is the
  /// bolted control.
  double push_efficiency = 1.0;

  /// Bounds on tool x, so the tool cannot wander off to infinity and make the
  /// outcome count meaningless. y and z are bounded in GraspWorld::step.
  double x_min = -0.95, x_max = -0.35;
};

// state layout: [tool_xyz (3), object_xyz (3), grasped (1)] -> 7
constexpr int TOOL = 0;
constexpr int OBJ = 3;
constexpr int HELD = 6;

/// 6 translations + close + open
constexpr int N_GRASP_ACTIONS = 8;
constexpr int CLOSE = 6;
constexpr int OPEN = 7;

/// A tool, an object it can push or hold, and a table.
class GraspWorld
{
public:
  explicit GraspWorld(const GraspConfig & config = GraspConfig());

  int n_actions() const {return N_GRASP_ACTIONS;}
  double rest_height() const;

  Eigen::VectorXd initial_state(
    const Eigen::Vector3d & tool,
    const Eigen::Vector2d & obj_xy,
    bool grasped = false,
    double lift = 0.0) const;

  Eigen::VectorXd step(const Eigen::VectorXd & state, int action) const;
  Eigen::VectorXd rollout(
    const Eigen::VectorXd & state, const std::vector<int> & actions) const;

  /// Control: the agent's model says the object cannot be moved.
  GraspWorld bolted() const;

  const GraspConfig & config() const {return config_;}

private:
  GraspConfig config_;
};

/// Renders a state, and quantises it into an outcome key.
class GraspSensor
{
public:
  GraspSensor(
    const CameraConfig & camera,
    const GraspConfig & config,
    double depth_resolution = 0.02,
    double object_resolution = 0.02);

  /// OWN-SENSOR outcome: what the camera sees.
  OutcomeKey depth_key(const Eigen::VectorXd & state) const;

  /// TRANSFER outcome: the object's state, and nothing about the tool.
  OutcomeKey object_key(const Eigen::VectorXd & state) const;

private:
  DepthCamera camera_;
  GraspConfig config_;
  double depth_resolution_;
  double object_resolution_;
};

/// log2 of the number of distinguishable outcomes, enumerated exhaustively.
///
/// With a deterministic forward model the channel capacity from actions to
/// outcomes reduces to exactly this. 8 actions at horizon 3 is 512
/// sequences, so this is the true value rather than a sampled lower bound.
///
/// `outcome` is "depth" for own-sensor, anything else for transfer.
double grasp_empowerment(
  const GraspWorld & world,
  const GraspSensor & sensor,
  const Eigen::VectorXd & state,
  int horizon = 3,
  const std::string & outcome = "depth");

/// Every sequence of `horizon` actions drawn from `n_actions`.
std::vector<std::vector<int>> enumerate_sequences(int n_actions, int horizon);

}  // namespace intrinsic_core

#endif  // INTRINSIC_CORE__GRASP_HPP_
