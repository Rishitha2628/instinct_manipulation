#include "intrinsic_core/ur5_grasp.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <stdexcept>

#include "intrinsic_core/grasp.hpp"

namespace intrinsic_core
{

UR5GraspWorld::UR5GraspWorld(const UR5 & arm, const UR5GraspConfig & config)
: arm_(arm), config_(config)
{
}

double UR5GraspWorld::rest_height() const
{
  return arm_.config().table_height + arm_.config().puck_radius;
}

Eigen::Vector3d UR5GraspWorld::grip_point(const Eigen::VectorXd & q) const
{
  const Eigen::Matrix4d T = arm_.forward_kinematics(q);
  return T.block<3, 1>(0, 3) + config_.tool_offset * T.block<3, 1>(0, 2);
}

Eigen::VectorXd UR5GraspWorld::state_at(
  const Eigen::VectorXd & q,
  const Eigen::Vector2d & obj_xy,
  bool grasped,
  double lift) const
{
  Eigen::VectorXd s = Eigen::VectorXd::Zero(10);
  s.segment<6>(GRASP_Q) = q;
  s[GRASP_OBJ + 0] = obj_xy[0];
  s[GRASP_OBJ + 1] = obj_xy[1];
  s[GRASP_OBJ + 2] = rest_height() + lift;
  s[GRASP_HELD] = grasped ? 1.0 : 0.0;
  return s;
}

Eigen::VectorXd UR5GraspWorld::step(const Eigen::VectorXd & state, int action) const
{
  const UR5GraspConfig & c = config_;
  Eigen::VectorXd s = state;

  if (action == close_action()) {
    const Eigen::Vector3d tool = grip_point(s.segment<6>(GRASP_Q));
    if (s[GRASP_HELD] == 0.0 &&
      (tool - s.segment<3>(GRASP_OBJ)).norm() <= c.grasp_radius)
    {
      s[GRASP_HELD] = 1.0;
    }
    return s;
  }

  if (action == open_action()) {
    if (s[GRASP_HELD] != 0.0) {
      s[GRASP_HELD] = 0.0;
      s[GRASP_OBJ + 2] = rest_height();
    }
    return s;
  }

  // A joint move. Reuse the arm's own action table so this is exactly the
  // action set the rest of the project uses.
  const Eigen::VectorXd q =
    s.segment<6>(GRASP_Q) + arm_.action_table().row(action).transpose();
  s.segment<6>(GRASP_Q) = q;
  const Eigen::Vector3d tool = grip_point(q);

  if (s[GRASP_HELD] != 0.0) {
    // Held: the object's pose is a rigid function of the tool's, which is
    // what removes it as an independent degree of freedom and is the entire
    // reason own-sensor empowerment dislikes grasping.
    s[GRASP_OBJ + 0] = tool[0];
    s[GRASP_OBJ + 1] = tool[1];
    s[GRASP_OBJ + 2] =
      std::min(tool[2], arm_.config().table_height + c.max_lift);
    return s;
  }

  if (c.push_efficiency > 0.0) {
    const UR5Config & a = arm_.config();
    const Eigen::Vector2d delta = tool.head<2>() - s.segment<2>(GRASP_OBJ);
    const double planar = delta.norm();
    const double contact = a.tool_radius + a.puck_radius;
    const bool low_enough = tool[2] < a.table_height + contact;
    if (low_enough && planar > 1e-9 && planar < contact) {
      const double push = c.push_efficiency * (contact - planar) / planar;
      s[GRASP_OBJ + 0] -= delta[0] * push;
      s[GRASP_OBJ + 1] -= delta[1] * push;
      s[GRASP_OBJ + 2] = rest_height();
    }
  }
  return s;
}

Eigen::VectorXd UR5GraspWorld::rollout(
  const Eigen::VectorXd & state, const std::vector<int> & actions) const
{
  Eigen::VectorXd s = state;
  for (int a : actions) {s = step(s, a);}
  return s;
}

UR5GraspWorld UR5GraspWorld::bolted() const
{
  UR5GraspConfig c = config_;
  c.push_efficiency = 0.0;
  return UR5GraspWorld(arm_, c);
}

OutcomeKey UR5GraspWorld::object_key(const Eigen::VectorXd & state) const
{
  OutcomeKey key;
  key.reserve(3);
  for (int i = 0; i < 3; ++i) {
    key.push_back(static_cast<int64_t>(
        std::llround(state[GRASP_OBJ + i] / config_.object_resolution)));
  }
  return key;
}

OutcomeKey UR5GraspWorld::depth_key(
  const Eigen::VectorXd & state, const DepthCamera & camera) const
{
  const Eigen::Matrix<double, 7, 3> arm_points =
    arm_.link_positions(state.segment<6>(GRASP_Q));

  Eigen::MatrixXd centres(8, 3);
  centres.topRows<7>() = arm_points;
  centres.row(7) = state.segment<3>(GRASP_OBJ).transpose();

  Eigen::VectorXd radii(8);
  radii.head<7>().setConstant(camera.config().arm_sphere_radius);
  radii[7] = arm_.config().puck_radius;

  const Eigen::VectorXd depth = camera.render_spheres(centres, radii);
  OutcomeKey key;
  key.reserve(static_cast<size_t>(depth.size()));
  for (Eigen::Index i = 0; i < depth.size(); ++i) {
    key.push_back(
      static_cast<int64_t>(std::llround(depth[i] / config_.depth_resolution)));
  }
  return key;
}

double UR5GraspWorld::empowerment(
  const Eigen::VectorXd & state,
  int horizon,
  const std::string & outcome,
  const DepthCamera * camera,
  int n_sequences,
  uint64_t seed) const
{
  const double total =
    std::pow(static_cast<double>(n_actions()), static_cast<double>(horizon));

  std::vector<std::vector<int>> seqs;
  if (n_sequences <= 0 || static_cast<double>(n_sequences) >= total) {
    seqs = enumerate_sequences(n_actions(), horizon);
  } else {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> pick(0, n_actions() - 1);
    seqs.assign(
      static_cast<size_t>(n_sequences), std::vector<int>(static_cast<size_t>(horizon)));
    for (auto & seq : seqs) {
      for (auto & a : seq) {a = pick(rng);}
    }
  }

  const bool use_object = (outcome == "object");
  if (!use_object && camera == nullptr) {
    throw std::invalid_argument("depth outcome needs a camera");
  }

  std::set<OutcomeKey> seen;
  for (const auto & seq : seqs) {
    const Eigen::VectorXd end = rollout(state, seq);
    seen.insert(use_object ? object_key(end) : depth_key(end, *camera));
  }
  return std::log2(static_cast<double>(seen.size()));
}

}  // namespace intrinsic_core
