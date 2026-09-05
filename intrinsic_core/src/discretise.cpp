#include "intrinsic_core/discretise.hpp"

#include <cmath>

namespace intrinsic_core
{

Discretiser grid_discretiser(double resolution)
{
  return [resolution](const Eigen::VectorXd & outcome) {
    OutcomeKey key;
    key.reserve(static_cast<size_t>(outcome.size()));
    for (Eigen::Index i = 0; i < outcome.size(); ++i) {
      key.push_back(static_cast<int64_t>(std::llround(outcome[i] / resolution)));
    }
    return key;
  };
}

Discretiser scaled_discretiser(const Eigen::VectorXd & resolutions)
{
  Eigen::VectorXd res = resolutions;
  return [res](const Eigen::VectorXd & outcome) {
    OutcomeKey key;
    key.reserve(static_cast<size_t>(outcome.size()));
    for (Eigen::Index i = 0; i < outcome.size() && i < res.size(); ++i) {
      // A non-finite resolution means "this dimension is not perceived",
      // so it contributes nothing to the key at all.
      if (std::isfinite(res[i])) {
        key.push_back(static_cast<int64_t>(std::llround(outcome[i] / res[i])));
      }
    }
    return key;
  };
}

}  // namespace intrinsic_core
