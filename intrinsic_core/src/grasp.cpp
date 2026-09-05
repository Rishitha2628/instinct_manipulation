#include "intrinsic_core/grasp.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace intrinsic_core
{

namespace
{
const Eigen::Matrix<double, 6, 3> MOVES = (Eigen::Matrix<double, 6, 3>() <<
  +1, 0, 0,
  -1, 0, 0,
  0, +1, 0,
  0, -1, 0,
  0, 0, +1,
  0, 0, -1).finished();

double clamp(double v, double lo, double hi) {return std::min(std::max(v, lo), hi);}
}  // namespace

GraspWorld::GraspWorld(const GraspConfig & config)
: config_(config)
{
}

double GraspWorld::rest_height() const
{
  return config_.table_height + config_.object_radius;
}

Eigen::VectorXd GraspWorld::initial_state(
  const Eigen::Vector3d & tool,
  const Eigen::Vector2d & obj_xy,
  bool grasped,
  double lift) const
{
  Eigen::VectorXd s = Eigen::VectorXd::Zero(7);
  s.segment<3>(TOOL) = tool;
  s[OBJ + 0] = obj_xy[0];
  s[OBJ + 1] = obj_xy[1];
  s[OBJ + 2] = rest_height() + lift;
  s[HELD] = grasped ? 1.0 : 0.0;
  return s;
}

Eigen::VectorXd GraspWorld::step(const Eigen::VectorXd & state, int action) const
{
  const GraspConfig & c = config_;
  Eigen::VectorXd s = state;
  const Eigen::Vector3d tool = s.segment<3>(TOOL);
  const Eigen::Vector3d obj = s.segment<3>(OBJ);

  if (action == CLOSE) {
    if (s[HELD] == 0.0 && (tool - obj).norm() <= c.grasp_radius) {s[HELD] = 1.0;}
    return s;
  }

  if (action == OPEN) {
    if (s[HELD] != 0.0) {
      s[HELD] = 0.0;
      // dropped: falls straight down to the table
      s[OBJ + 2] = rest_height();
    }
    return s;
  }

  Eigen::Vector3d new_tool = tool + MOVES.row(action).transpose() * c.step;
  new_tool[0] = clamp(new_tool[0], c.x_min, c.x_max);
  new_tool[1] = clamp(new_tool[1], -0.40, 0.40);
  // the tool cannot go through the table
  new_tool[2] = clamp(new_tool[2], c.table_height + 0.01, c.table_height + 0.45);
  s.segment<3>(TOOL) = new_tool;

  if (s[HELD] != 0.0) {
    // Held: the object's pose is a rigid function of the tool's. This is the
    // whole point of the attachment state, and it is also exactly why
    // own-sensor empowerment may dislike it -- the object has stopped being
    // an independent degree of freedom.
    s.segment<3>(OBJ) = new_tool;
    return s;
  }

  // Not held: a shove in the table plane, only while the tool is low enough
  // to catch the object's side.
  if (c.push_efficiency > 0.0) {
    const Eigen::Vector2d delta = new_tool.head<2>() - obj.head<2>();
    const double planar = delta.norm();
    const double contact = c.tool_radius + c.object_radius;
    const bool low_enough = new_tool[2] < rest_height() + c.object_radius;
    if (low_enough && planar > 1e-9 && planar < contact) {
      const double push = c.push_efficiency * (contact - planar) / planar;
      s[OBJ + 0] = obj[0] - delta[0] * push;
      s[OBJ + 1] = obj[1] - delta[1] * push;
      s[OBJ + 2] = rest_height();
    }
  }
  return s;
}

Eigen::VectorXd GraspWorld::rollout(
  const Eigen::VectorXd & state, const std::vector<int> & actions) const
{
  Eigen::VectorXd s = state;
  for (int a : actions) {s = step(s, a);}
  return s;
}

GraspWorld GraspWorld::bolted() const
{
  GraspConfig c = config_;
  c.push_efficiency = 0.0;
  return GraspWorld(c);
}

GraspSensor::GraspSensor(
  const CameraConfig & camera,
  const GraspConfig & config,
  double depth_resolution,
  double object_resolution)
: camera_(camera),
  config_(config),
  depth_resolution_(depth_resolution),
  object_resolution_(object_resolution)
{
}

OutcomeKey GraspSensor::depth_key(const Eigen::VectorXd & state) const
{
  Eigen::MatrixXd centres(2, 3);
  centres.row(0) = state.segment<3>(TOOL).transpose();
  centres.row(1) = state.segment<3>(OBJ).transpose();
  Eigen::VectorXd radii(2);
  radii << config_.tool_radius, config_.object_radius;

  const Eigen::VectorXd depth = camera_.render_spheres(centres, radii);
  OutcomeKey key;
  key.reserve(static_cast<size_t>(depth.size()));
  for (Eigen::Index i = 0; i < depth.size(); ++i) {
    key.push_back(static_cast<int64_t>(std::llround(depth[i] / depth_resolution_)));
  }
  return key;
}

OutcomeKey GraspSensor::object_key(const Eigen::VectorXd & state) const
{
  OutcomeKey key;
  key.reserve(3);
  for (int i = 0; i < 3; ++i) {
    key.push_back(
      static_cast<int64_t>(std::llround(state[OBJ + i] / object_resolution_)));
  }
  return key;
}

std::vector<std::vector<int>> enumerate_sequences(int n_actions, int horizon)
{
  int64_t total = 1;
  for (int i = 0; i < horizon; ++i) {total *= n_actions;}
  std::vector<std::vector<int>> seqs(
    static_cast<size_t>(total), std::vector<int>(static_cast<size_t>(horizon)));
  for (int64_t idx = 0; idx < total; ++idx) {
    int64_t rem = idx;
    for (int k = horizon - 1; k >= 0; --k) {
      seqs[static_cast<size_t>(idx)][static_cast<size_t>(k)] =
        static_cast<int>(rem % n_actions);
      rem /= n_actions;
    }
  }
  return seqs;
}

double grasp_empowerment(
  const GraspWorld & world,
  const GraspSensor & sensor,
  const Eigen::VectorXd & state,
  int horizon,
  const std::string & outcome)
{
  const bool own_sensor = (outcome == "depth");
  std::set<OutcomeKey> seen;
  for (const auto & seq : enumerate_sequences(world.n_actions(), horizon)) {
    const Eigen::VectorXd end = world.rollout(state, seq);
    seen.insert(own_sensor ? sensor.depth_key(end) : sensor.object_key(end));
  }
  return std::log2(static_cast<double>(seen.size()));
}

}  // namespace intrinsic_core
