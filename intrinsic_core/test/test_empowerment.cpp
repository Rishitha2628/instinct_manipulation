// Tests for the empowerment estimator.
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "intrinsic_core/arm2d.hpp"
#include "intrinsic_core/discretise.hpp"
#include "intrinsic_core/empowerment.hpp"

using intrinsic_core::ArmConfig;
using intrinsic_core::EmpowermentConfig;
using intrinsic_core::EmpowermentEstimator;
using intrinsic_core::PlanarArm;
using intrinsic_core::blahut_arimoto;
using intrinsic_core::grid_discretiser;
using intrinsic_core::scaled_discretiser;

// --- Blahut-Arimoto --------------------------------------------------------

TEST(BlahutArimoto, NoiselessBinaryChannelIsOneBit)
{
  EXPECT_NEAR(blahut_arimoto(Eigen::MatrixXd::Identity(2, 2)), 1.0, 1e-4);
}

TEST(BlahutArimoto, Noiseless8aryChannelIsThreeBits)
{
  EXPECT_NEAR(blahut_arimoto(Eigen::MatrixXd::Identity(8, 8)), 3.0, 1e-4);
}

TEST(BlahutArimoto, UselessChannelIsZeroBits)
{
  Eigen::MatrixXd useless = Eigen::MatrixXd::Constant(4, 4, 0.25);
  EXPECT_NEAR(blahut_arimoto(useless), 0.0, 1e-6);
}

TEST(BlahutArimoto, BinarySymmetricChannelMatchesAnalytic)
{
  const double p = 0.1;
  Eigen::MatrixXd bsc(2, 2);
  bsc << 1 - p, p, p, 1 - p;
  const double h = -p * std::log2(p) - (1 - p) * std::log2(1 - p);
  EXPECT_NEAR(blahut_arimoto(bsc), 1.0 - h, 1e-4);
}

// --- kinematics ------------------------------------------------------------

TEST(PlanarArmKinematics, IkInvertsFk)
{
  PlanarArm arm;
  const std::vector<Eigen::Vector2d> targets = {
    {0.30, 0.25}, {0.20, 0.30}, {-0.25, 0.20}};
  for (const auto & target : targets) {
    const auto ik = arm.inverse_kinematics(target[0], target[1]);
    ASSERT_TRUE(ik.has_value());
    const Eigen::Vector2d fk = arm.forward_kinematics((*ik)[0], (*ik)[1]);
    EXPECT_NEAR(fk[0], target[0], 1e-9);
    EXPECT_NEAR(fk[1], target[1], 1e-9);
  }
}

TEST(PlanarArmKinematics, UnreachablePointsReturnNothing)
{
  PlanarArm arm;
  EXPECT_FALSE(arm.inverse_kinematics(5.0, 5.0).has_value());
}

// --- the core claim --------------------------------------------------------

namespace
{

EmpowermentEstimator make_estimator(
  const PlanarArm & arm, double resolution = 0.025, int steps = 3)
{
  EmpowermentConfig cfg;
  cfg.n_steps = steps;
  cfg.seed = 0;
  return EmpowermentEstimator(
    [arm](const Eigen::VectorXd & s, const std::vector<int> & a) {
      return arm.rollout(s, a);
    },
    9, grid_discretiser(resolution), cfg);
}

Eigen::VectorXd make_state(
  const PlanarArm & arm, const Eigen::Vector2d & xy, const Eigen::Vector2d & puck)
{
  const auto ik = arm.inverse_kinematics(xy[0], xy[1]);
  EXPECT_TRUE(ik.has_value());
  Eigen::VectorXd state(4);
  state << (*ik)[0], (*ik)[1], puck[0], puck[1];
  return state;
}

}  // namespace

/// The result. Nobody tells the arm the puck matters.
TEST(Empowerment, IsHigherNearAPushableObject)
{
  PlanarArm arm;
  const Eigen::Vector2d puck(0.30, 0.25);
  const auto est = make_estimator(arm);

  const double near = est.estimate(make_state(arm, {0.30, 0.25}, puck));
  const double far = est.estimate(make_state(arm, {0.30, 0.42}, puck));
  EXPECT_GT(near, far + 0.5);
}

/// Paper's immovable-box condition: influence, not mere presence.
TEST(Empowerment, BoltedObjectRemovesThePeak)
{
  PlanarArm arm;
  const Eigen::Vector2d puck(0.30, 0.25);
  const Eigen::VectorXd near = make_state(arm, {0.30, 0.25}, puck);
  const Eigen::VectorXd far = make_state(arm, {0.30, 0.42}, puck);

  const auto est = make_estimator(arm.with_fixed_puck());
  EXPECT_LT(std::abs(est.estimate(near) - est.estimate(far)), 0.5);
}

/// The paper's "can it perceive the box?" condition. Perceiving something you
/// cannot act on is worth nothing -- and acting on something you cannot
/// perceive is worth nothing either.
TEST(Empowerment, UnperceivedObjectContributesNothing)
{
  PlanarArm arm;
  const Eigen::Vector2d puck(0.30, 0.25);
  Eigen::VectorXd resolutions(4);
  resolutions << 0.025, 0.025,
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity();

  EmpowermentConfig cfg;
  cfg.n_steps = 3;
  cfg.seed = 0;
  EmpowermentEstimator est(
    [arm](const Eigen::VectorXd & s, const std::vector<int> & a) {
      return arm.rollout(s, a);
    },
    9, scaled_discretiser(resolutions), cfg);

  const double near = est.estimate(make_state(arm, {0.30, 0.25}, puck));
  const double far = est.estimate(make_state(arm, {0.30, 0.42}, puck));
  EXPECT_LT(std::abs(near - far), 0.5);
}

TEST(Empowerment, DecaysWithDistance)
{
  PlanarArm arm;
  const Eigen::Vector2d puck(0.30, 0.25);
  const auto est = make_estimator(arm);
  std::vector<double> values;
  for (double d : {0.0, 0.05, 0.10}) {
    values.push_back(est.estimate(make_state(arm, {0.30 + d, 0.25}, puck)));
  }
  EXPECT_GT(values[0], values[1]);
  EXPECT_GT(values[1], values[2]);
}

// --- the parameters that bite ----------------------------------------------

/// Everything in one bin -> 0 bits. One of the two failure modes.
TEST(Empowerment, CoarseResolutionCollapsesTheMap)
{
  PlanarArm arm;
  const auto est = make_estimator(arm, 100.0);
  EXPECT_DOUBLE_EQ(est.estimate(make_state(arm, {0.30, 0.25}, {0.30, 0.25})), 0.0);
}

/// The other failure mode: nothing collapses, so the map goes flat high.
///
/// Note the ceiling is NOT log2(9**3) = 9.51 bits. Joint increments commute
/// -- [+1,-1,0] and [0,-1,+1] end at identical angles -- so 729 sequences
/// only reach 235 distinct configurations however finely you measure. That
/// degeneracy is a property of the arm, not of the discretiser, and it caps
/// open-loop empowerment at about 7.9 bits on this system regardless of
/// horizon.
TEST(Empowerment, FineResolutionSaturatesTheMap)
{
  PlanarArm arm;
  const Eigen::Vector2d puck(0.30, 0.25);
  const double fine =
    make_estimator(arm, 1e-9).estimate(make_state(arm, {0.30, 0.25}, puck));
  const double usable =
    make_estimator(arm, 0.025).estimate(make_state(arm, {0.30, 0.25}, puck));

  EXPECT_GT(fine, usable);                      // finer resolution, more outcomes
  EXPECT_LT(fine, std::log2(std::pow(9, 3)));   // but well under the naive ceiling
  EXPECT_NEAR(fine, 7.88, 0.05);
}

/// Documents the degeneracy above: action order does not matter.
TEST(Empowerment, CommutingActionsProduceIdenticalOutcomes)
{
  PlanarArm arm;
  const Eigen::VectorXd state = make_state(arm, {0.30, 0.25}, {0.30, 0.25});
  const Eigen::VectorXd forward = arm.rollout(state, {0, 4, 8});
  const Eigen::VectorXd backward = arm.rollout(state, {8, 4, 0});
  EXPECT_TRUE(forward.isApprox(backward, 1e-9));
}

TEST(Empowerment, IsDeterministicForFixedSeed)
{
  PlanarArm arm;
  const Eigen::VectorXd s = make_state(arm, {0.30, 0.25}, {0.30, 0.25});
  EXPECT_DOUBLE_EQ(make_estimator(arm).estimate(s), make_estimator(arm).estimate(s));
}
