// The grasp experiment on the real UR5 kinematics.
//
// grasp.hpp established the result with a free-flying tool. This checks it
// survives an actual 6-DOF arm, where directions are not equally cheap and
// the set of futures available while holding something need not be the same
// shape as the set available while hovering.
//
// It does survive, and more strongly: the own-sensor penalty for grasping
// grows from -0.13 bits on the toy to -2.66 here.
#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "intrinsic_core/sensors.hpp"
#include "intrinsic_core/ur5.hpp"
#include "intrinsic_core/ur5_grasp.hpp"

using namespace intrinsic_core;  // NOLINT(build/namespaces)

namespace
{

const Eigen::Vector2d OBJECT_XY(-0.68, 0.0);

UR5 grasp_arm()
{
  UR5Config cfg;
  cfg.dq = 0.10;
  cfg.active_joints = {0, 1, 2, 3};
  cfg.tool_radius = 0.09;
  cfg.puck_radius = 0.04;
  cfg.table_height = -0.25;
  return UR5(cfg);
}

UR5GraspWorld make_world()
{
  return UR5GraspWorld(grasp_arm(), UR5GraspConfig());
}

DepthCamera make_camera(const UR5GraspWorld & world)
{
  CameraConfig cfg;
  cfg.position = {-1.45, 0.0, 0.35};
  cfg.look_at = {-0.68, 0.0, -0.21};
  cfg.up = {0.0, 0.0, 1.0};
  cfg.fov_deg = 50.0;
  cfg.width = 16;
  cfg.height = 12;
  cfg.table_height = world.arm().config().table_height;
  return DepthCamera(cfg);
}

Eigen::VectorXd at_object(
  const UR5GraspWorld & w, bool grasped = false, double lift = 0.0)
{
  const auto q = w.arm().inverse_kinematics(
    Eigen::Vector3d(OBJECT_XY[0], OBJECT_XY[1], w.rest_height()));
  EXPECT_TRUE(q.has_value());
  return w.state_at(*q, OBJECT_XY, grasped, lift);
}

Eigen::VectorXd after(const UR5GraspWorld & w, int action)
{
  return w.step(at_object(w), action);
}

/// The joint action that raises the tool highest, used to lift.
int highest_action(const UR5GraspWorld & w, const Eigen::VectorXd & held)
{
  int best = 0;
  double best_z = -1e9;
  for (int a = 0; a < w.n_joint_actions(); ++a) {
    const double z =
      w.arm().tool_position(w.step(held, a).segment<6>(GRASP_Q))[2];
    if (z > best_z) {best_z = z; best = a;}
  }
  return best;
}

}  // namespace

// --- mechanics -------------------------------------------------------------

TEST(UR5Grasp, CloseGraspsAtTheObjectButNotFarAway)
{
  const UR5GraspWorld w = make_world();
  EXPECT_DOUBLE_EQ(w.step(at_object(w), w.close_action())[GRASP_HELD], 1.0);

  const auto q = w.arm().inverse_kinematics(
    Eigen::Vector3d(-0.40, 0.0, w.rest_height() + 0.05));
  ASSERT_TRUE(q.has_value());
  EXPECT_DOUBLE_EQ(
    w.step(w.state_at(*q, OBJECT_XY), w.close_action())[GRASP_HELD], 0.0);
}

/// Impossible to express in ur5.hpp, where the object is (x, y) only.
TEST(UR5Grasp, HeldObjectFollowsTheToolAndLeavesTheTable)
{
  const UR5GraspWorld w = make_world();
  const Eigen::VectorXd held = w.step(at_object(w), w.close_action());
  const Eigen::VectorXd moved = w.step(held, highest_action(w, held));

  const Eigen::Vector3d tool = w.arm().tool_position(moved.segment<6>(GRASP_Q));
  EXPECT_NEAR(moved[GRASP_OBJ + 0], tool[0], 1e-9);
  EXPECT_NEAR(moved[GRASP_OBJ + 1], tool[1], 1e-9);
  EXPECT_GT(moved[GRASP_OBJ + 2], w.rest_height() + 0.05);
}

TEST(UR5Grasp, ReleaseDropsItAndUnheldObjectNeverRises)
{
  const UR5GraspWorld w = make_world();
  const Eigen::VectorXd held = w.step(at_object(w), w.close_action());
  const int highest = highest_action(w, held);
  const Eigen::VectorXd lifted = w.step(held, highest);
  EXPECT_NEAR(
    w.step(lifted, w.open_action())[GRASP_OBJ + 2], w.rest_height(), 1e-9);

  const Eigen::VectorXd open_state = at_object(w);
  EXPECT_NEAR(
    w.rollout(open_state, {highest, highest})[GRASP_OBJ + 2], w.rest_height(), 1e-9);
}

// --- the result ------------------------------------------------------------

/// Scored on control of the OBJECT, closing beats opening and moving.
///
/// Over all 83 actions, CLOSE ranks 1st. Checked here against OPEN and a
/// spread of joint moves rather than all of them, to keep the suite fast.
TEST(UR5Grasp, TransferEmpowermentMakesGraspingTheBestAction)
{
  const UR5GraspWorld w = make_world();
  const double close = w.empowerment(after(w, w.close_action()), 2, "object");
  double best_other = -1e9;
  for (int a : {w.open_action(), 0, 20, 40, 55, 80}) {
    best_other = std::max(best_other, w.empowerment(after(w, a), 2, "object"));
  }
  EXPECT_GT(close, best_other);
}

/// Scored on the depth image, the same grasp ranks LAST of 83.
///
/// Not a near miss: -2.66 bits against simply leaving the gripper open. The
/// project's "empowerment will not pick things up" is real, and this is it.
TEST(UR5Grasp, OwnSensorEmpowermentMakesGraspingTheWorstAction)
{
  const UR5GraspWorld w = make_world();
  const DepthCamera cam = make_camera(w);
  const double close =
    w.empowerment(after(w, w.close_action()), 2, "depth", &cam);
  double worst_other = 1e9;
  for (int a : {w.open_action(), 0, 20, 40, 55, 80}) {
    worst_other = std::min(worst_other, w.empowerment(after(w, a), 2, "depth", &cam));
  }
  EXPECT_LT(close, worst_other);
}

/// The mechanism, isolated.
///
/// Holding is not what own-sensor dislikes; losing the ability to PUSH is.
/// Shoving gives the object semi-independent motion and so adds variety to
/// the picture, and grasping destroys that independence by making the object
/// a rigid function of the tool. Bolt the object, and hovering has nothing
/// left to offer: the penalty disappears.
TEST(UR5Grasp, BoltingTheObjectRemovesThePenaltyForGrasping)
{
  const UR5GraspWorld w = make_world();
  const DepthCamera cam = make_camera(w);
  const double penalty =
    w.empowerment(at_object(w, true), 2, "depth", &cam) -
    w.empowerment(at_object(w), 2, "depth", &cam);
  EXPECT_LT(penalty, -1.0);

  const UR5GraspWorld b = w.bolted();
  const DepthCamera bcam = make_camera(b);
  const double bolted_penalty =
    b.empowerment(at_object(b, true), 2, "depth", &bcam) -
    b.empowerment(at_object(b), 2, "depth", &bcam);
  EXPECT_GT(bolted_penalty, penalty + 2.0);
}
