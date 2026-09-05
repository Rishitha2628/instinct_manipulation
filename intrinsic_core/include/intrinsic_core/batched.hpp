// Empowerment for the UR5 + depth camera setup.
//
// In the Python original this file existed for speed: the generic estimator
// called the forward model once per action sequence, and at 81 actions and
// horizon 2 that is 6561 rollouts, each paying numpy's per-call overhead for
// a handful of floating-point operations. Resolving the whole batch at once
// made it roughly two orders of magnitude faster.
//
// That overhead does not exist here, so this is a thin wrapper over the same
// primitives rather than a second implementation of the physics. It is kept
// because the callers and the tests are written against it, and because
// having one place that owns "sequences -> endpoints -> images -> count"
// is worth more than the speed ever was.
#ifndef INTRINSIC_CORE__BATCHED_HPP_
#define INTRINSIC_CORE__BATCHED_HPP_

#include <vector>

#include <Eigen/Core>

#include "intrinsic_core/discretise.hpp"
#include "intrinsic_core/empowerment.hpp"
#include "intrinsic_core/sensors.hpp"
#include "intrinsic_core/ur5.hpp"

namespace intrinsic_core
{

class BatchedDepthEmpowerment
{
public:
  BatchedDepthEmpowerment(
    const UR5 & arm,
    const DepthCamera & camera,
    Discretiser discretiser,
    const EmpowermentConfig & config = EmpowermentConfig());

  /// Empowerment in bits, from the depth sensor's point of view.
  double estimate(const Eigen::VectorXd & state) const;

  double operator()(const Eigen::VectorXd & state) const {return estimate(state);}

private:
  void build_sequences();

  UR5 arm_;
  DepthCamera camera_;
  Discretiser discretiser_;
  EmpowermentConfig config_;
  std::vector<std::vector<int>> sequences_;
};

}  // namespace intrinsic_core

#endif  // INTRINSIC_CORE__BATCHED_HPP_
