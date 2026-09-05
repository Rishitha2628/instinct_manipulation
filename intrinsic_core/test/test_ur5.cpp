// UR5 + depth camera tests.
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "intrinsic_core/batched.hpp"
#include "intrinsic_core/empowerment.hpp"
#include "intrinsic_core/sensors.hpp"
#include "intrinsic_core/ur5.hpp"

using namespace intrinsic_core;  // NOLINT(build/namespaces)

namespace
{

const Eigen::Vector2d PUCK(-0.50, 0.0);
constexpr double kPi = 3.14159265358979323846;

UR5 make_arm()
{
  UR5Config cfg;
  cfg.dq = 0.10;
  cfg.active_joints = {0, 1, 2, 3};
  return UR5(cfg);
}

DepthCamera make_camera(int width = 16, int height = 12, double fov = 30.0)
{
  CameraConfig cfg;
  cfg.position = {-0.50, 0.0, 0.95};
  cfg.look_at = {-0.50, 0.0, 0.0};
  cfg.up = {1.0, 0.0, 0.0};
  cfg.fov_deg = fov;
  cfg.width = width;
  cfg.height = height;
  return DepthCamera(cfg);
}

Eigen::VectorXd make_state(const UR5 & arm, const Eigen::Vector3d & xyz)
{
  const auto q = arm.inverse_kinematics(xyz);
  EXPECT_TRUE(q.has_value());
  Eigen::VectorXd state(8);
  state << *q, PUCK[0], PUCK[1];
  return state;
}

double estimate(
  const UR5 & arm, const DepthCamera & camera, const Eigen::VectorXd & state,
  int steps = 2, double depth_res = 0.02, bool exhaustive = true)
{
  EmpowermentConfig cfg;
  cfg.n_steps = steps;
  cfg.n_sequences = 2000;
  cfg.exhaustive_below = exhaustive ? 10000 : 0;
  cfg.seed = 0;
  return BatchedDepthEmpowerment(
    arm, camera, depth_discretiser(depth_res), cfg).estimate(state);
}

}  // namespace

// --- kinematics ------------------------------------------------------------

/// UR5 at all-zeros is fully extended horizontally.
TEST(UR5Kinematics, ZeroPoseMatchesUR5Geometry)
{
  const Eigen::Vector3d tool = make_arm().tool_position(Eigen::VectorXd::Zero(6));
  EXPECT_NEAR(tool[0], -0.8172, 1e-3);
  EXPECT_LT(std::abs(tool[2]), 0.02);
}

TEST(UR5Kinematics, ReachIsWithinUR5Envelope)
{
  const UR5 arm = make_arm();
  std::mt19937_64 rng(0);
  std::uniform_real_distribution<double> pick(-kPi, kPi);
  double max_r = 0.0;
  for (int i = 0; i < 2000; ++i) {
    Eigen::VectorXd q(6);
    for (int j = 0; j < 6; ++j) {q[j] = pick(rng);}
    max_r = std::max(max_r, arm.tool_position(q).norm());
  }
  EXPECT_LT(max_r, 1.10);     // wrist offset beyond the 0.85 wrist-centre spec
  EXPECT_GT(max_r, 0.85);
}

TEST(UR5Kinematics, IkInvertsFk)
{
  const UR5 arm = make_arm();
  const std::vector<Eigen::Vector3d> targets = {
    {-0.50, 0.0, 0.12}, {-0.45, 0.15, 0.12}, {-0.55, -0.15, 0.20}};
  for (const auto & target : targets) {
    const auto q = arm.inverse_kinematics(target);
    ASSERT_TRUE(q.has_value());
    const Eigen::Vector3d reached = arm.tool_position(*q);
    EXPECT_NEAR(reached[0], target[0], 1e-3);
    EXPECT_NEAR(reached[1], target[1], 1e-3);
    EXPECT_NEAR(reached[2], target[2], 1e-3);
  }
}

TEST(UR5Kinematics, LinkPositionsHasSevenFrames)
{
  const auto p = make_arm().link_positions(Eigen::VectorXd::Zero(6));
  EXPECT_EQ(p.rows(), 7);
  EXPECT_EQ(p.cols(), 3);
}

TEST(UR5Kinematics, ActionCountMatchesActiveJoints)
{
  UR5Config four; four.active_joints = {0, 1, 2, 3};
  UR5Config three; three.active_joints = {0, 1, 2};
  EXPECT_EQ(UR5(four).n_actions(), 81);
  EXPECT_EQ(UR5(three).n_actions(), 27);
}

// --- depth camera ----------------------------------------------------------

TEST(DepthCameraTest, ImageShapeAndRange)
{
  const DepthCamera cam = make_camera();
  const Eigen::VectorXd img =
    cam.render(make_arm().link_positions(Eigen::VectorXd::Zero(6)), PUCK, 0.04);
  EXPECT_EQ(img.size(), 12 * 16);
  EXPECT_GE(img.minCoeff(), cam.config().z_near);
  EXPECT_LE(img.maxCoeff(), cam.config().z_far);
}

/// The puck must appear in the image, or nothing downstream works.
///
/// Note the arm must be off to the side. The camera looks straight DOWN at
/// the puck, so an arm reaching to the puck's (x, y) occludes it completely.
TEST(DepthCameraTest, PuckIsVisibleWhenTheArmIsNotInTheWay)
{
  const DepthCamera cam = make_camera();
  const UR5 arm = make_arm();
  const auto aside = arm.inverse_kinematics({-0.30, 0.20, 0.35});
  ASSERT_TRUE(aside.has_value());

  const Eigen::VectorXd with_puck =
    cam.render(arm.link_positions(*aside), PUCK, 0.04);
  const Eigen::VectorXd without =
    cam.render(arm.link_positions(*aside), Eigen::Vector2d(5.0, 5.0), 0.04);

  int changed = 0;
  double min_with = 1e9, min_without = 1e9;
  for (Eigen::Index i = 0; i < with_puck.size(); ++i) {
    if (std::abs(with_puck[i] - without[i]) > 1e-6) {
      ++changed;
      min_with = std::min(min_with, with_puck[i]);
      min_without = std::min(min_without, without[i]);
    }
  }
  EXPECT_GT(changed, 3) << "puck should occupy several pixels";
  EXPECT_LT(min_with, min_without);
}

/// A real limitation of this camera placement, documented not fixed.
///
/// With the camera directly above the puck, an arm reaching down to the
/// puck's (x, y) blocks the view entirely: zero pixels differ between a scene
/// with the puck and one without.
///
/// Empowerment still finds the puck, because pushing displaces it OUT from
/// under the arm where the camera can see it again -- the visible signal is
/// the puck's displacement, not the puck itself. But an oblique camera would
/// be the better choice for a real rig.
TEST(DepthCameraTest, ArmOccludesPuckFromOverheadCamera)
{
  const DepthCamera cam = make_camera();
  const UR5 arm = make_arm();
  const auto blocked = arm.inverse_kinematics({-0.50, 0.0, 0.45});
  ASSERT_TRUE(blocked.has_value());

  const Eigen::VectorXd with_puck =
    cam.render(arm.link_positions(*blocked), PUCK, 0.04);
  const Eigen::VectorXd without =
    cam.render(arm.link_positions(*blocked), Eigen::Vector2d(5.0, 5.0), 0.04);
  EXPECT_LT((with_puck - without).cwiseAbs().maxCoeff(), 1e-9);
}

TEST(DepthCameraTest, BatchRenderMatchesSingleRender)
{
  const DepthCamera cam = make_camera();
  const UR5 arm = make_arm();
  Eigen::VectorXd q0 = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd q1(6);
  q1 << 0.0, -1.2, 1.4, -1.75, -1.57, 0.0;

  const std::vector<Eigen::MatrixXd> poses = {
    arm.link_positions(q0), arm.link_positions(q1)};
  const std::vector<Eigen::Vector2d> pucks = {
    PUCK, Eigen::Vector2d(PUCK[0] + 0.1, PUCK[1] + 0.1)};

  const auto batch = cam.render_batch(poses, pucks, 0.04);
  for (size_t i = 0; i < poses.size(); ++i) {
    const Eigen::VectorXd single = cam.render(poses[i], pucks[i], 0.04);
    EXPECT_LT((batch[i] - single).cwiseAbs().maxCoeff(), 1e-9);
  }
}

// --- the core claim --------------------------------------------------------

/// The result: the arm identifies the object with no goal given.
TEST(UR5Empowerment, PeaksAtThePuck)
{
  const UR5 arm = make_arm();
  const DepthCamera cam = make_camera();
  const double near = estimate(arm, cam, make_state(arm, {-0.50, 0.0, 0.08}));
  const double far = estimate(arm, cam, make_state(arm, {-0.50, 0.0, 0.35}));
  EXPECT_GT(near, far + 1.0);
}

/// The paper's immovable-box control: influence, not mere presence.
TEST(UR5Empowerment, BoltedPuckRemovesThePeak)
{
  const UR5 arm = make_arm();
  const DepthCamera cam = make_camera();
  const Eigen::VectorXd state = make_state(arm, {-0.50, 0.0, 0.08});
  const double pushable = estimate(arm, cam, state);
  const double bolted = estimate(arm.with_fixed_puck(), cam, state);
  EXPECT_GT(pushable - bolted, 1.0);
}

TEST(UR5Empowerment, PuckContributionVanishesWithDistance)
{
  const UR5 arm = make_arm();
  const DepthCamera cam = make_camera();
  const UR5 bolt = arm.with_fixed_puck();
  const Eigen::VectorXd state = make_state(arm, {-0.50, 0.0, 0.30});
  EXPECT_LT(std::abs(estimate(arm, cam, state) - estimate(bolt, cam, state)), 0.3);
}

/// Sensor resolution alone can produce the "cannot perceive" condition.
///
/// The threshold is geometric, but not at one pixel per puck. Rendering here
/// is ray-cast, not area-averaged: a ray either intersects the puck or it
/// does not, so an object smaller than a pixel still flips that pixel's whole
/// value whenever a ray happens to land on it. What kills the signal is ray
/// SPACING coarse enough that no ray lands on the puck at all, which is
/// roughly three times the puck's diameter rather than one.
///
/// Nobody imposed the paper's "box is not perceivable" condition here. It
/// falls out of the optics, which is a stronger form of the same point.
TEST(UR5Empowerment, PuckContributionDecaysWithSensorResolution)
{
  const UR5 arm = make_arm();
  const UR5 bolted_arm = arm.with_fixed_puck();
  const Eigen::VectorXd state = make_state(arm, {-0.50, 0.0, 0.08});

  auto contribution = [&](int width, int height) {
      const DepthCamera cam = make_camera(width, height);
      return estimate(arm, cam, state, 2, 0.02, false) -
             estimate(bolted_arm, cam, state, 2, 0.02, false);
    };

  const double resolved = contribution(16, 12);
  const double unresolved = contribution(2, 2);

  EXPECT_GT(resolved, 0.5);
  EXPECT_LT(unresolved, 0.2);
  EXPECT_GT(resolved, unresolved * 3);
}

// --- estimator agreement ---------------------------------------------------

/// The batched path must not change the answer.
TEST(UR5Empowerment, BatchedMatchesReferenceEstimator)
{
  const UR5 arm = make_arm();
  const DepthCamera cam = make_camera();
  const Eigen::VectorXd state = make_state(arm, {-0.50, 0.0, 0.08});

  EmpowermentConfig cfg;
  cfg.n_steps = 2;
  cfg.exhaustive_below = 10000;
  cfg.seed = 0;
  const Discretiser disc = depth_discretiser(0.02);

  const EmpowermentEstimator reference(
    [&arm, &cam](const Eigen::VectorXd & s, const std::vector<int> & actions) {
      const Eigen::VectorXd s2 = arm.rollout_state(s, actions);
      return cam.render(
        arm.link_positions(s2.head<6>()), Eigen::Vector2d(s2[6], s2[7]),
        arm.config().puck_radius);
    },
    arm.n_actions(), disc, cfg);

  const double batched = BatchedDepthEmpowerment(arm, cam, disc, cfg).estimate(state);
  EXPECT_NEAR(reference.estimate(state), batched, 1e-9);
}
