#include "intrinsic_core/sensors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace intrinsic_core
{

namespace
{
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.14159265358979323846;
}  // namespace

DepthCamera::DepthCamera(const CameraConfig & config)
: config_(config), origin_(config.position)
{
  build_rays();
  table_ = hit_plane(config_.table_height);
}

void DepthCamera::build_rays()
{
  const CameraConfig & c = config_;

  Eigen::Vector3d forward = (c.look_at - c.position).normalized();
  Eigen::Vector3d right = forward.cross(c.up).normalized();
  Eigen::Vector3d true_up = right.cross(forward);

  const double aspect = static_cast<double>(c.width) / static_cast<double>(c.height);
  const double half_w = std::tan(c.fov_deg * kPi / 180.0 / 2.0);
  const double half_h = half_w / aspect;

  rays_.resize(n_pixels(), 3);
  int row = 0;
  for (int j = 0; j < c.height; ++j) {
    // pixel centres in [-1, 1]
    const double gy = 1.0 - (static_cast<double>(j) + 0.5) / c.height * 2.0;
    for (int i = 0; i < c.width; ++i) {
      const double gx = (static_cast<double>(i) + 0.5) / c.width * 2.0 - 1.0;
      Eigen::Vector3d d = forward + (gx * half_w) * right + (gy * half_h) * true_up;
      rays_.row(row++) = d.normalized().transpose();
    }
  }
}

Eigen::VectorXd DepthCamera::hit_plane(double z) const
{
  Eigen::VectorXd t(rays_.rows());
  for (Eigen::Index r = 0; r < rays_.rows(); ++r) {
    const double dz = rays_(r, 2);
    const double hit = (z - origin_[2]) / dz;
    t[r] = (dz != 0.0 && hit > 0.0) ? hit : kInf;
  }
  return t;
}

Eigen::VectorXd DepthCamera::render_spheres(
  const Eigen::MatrixXd & centres, const Eigen::VectorXd & radii) const
{
  Eigen::VectorXd depth = table_;
  for (Eigen::Index r = 0; r < rays_.rows(); ++r) {
    const Eigen::Vector3d ray = rays_.row(r).transpose();
    double nearest = depth[r];
    for (Eigen::Index s = 0; s < centres.rows(); ++s) {
      const Eigen::Vector3d oc = origin_ - centres.row(s).transpose();
      const double b = ray.dot(oc);
      const double c_term = oc.squaredNorm() - radii[s] * radii[s];
      const double disc = b * b - c_term;
      if (disc < 0.0) {continue;}
      const double sq = std::sqrt(disc);
      const double t0 = -b - sq;
      const double t1 = -b + sq;
      const double t = (t0 > 1e-6) ? t0 : ((t1 > 1e-6) ? t1 : kInf);
      if (t < nearest) {nearest = t;}
    }
    depth[r] = std::min(std::max(nearest, config_.z_near), config_.z_far);
  }
  return depth;
}

Eigen::VectorXd DepthCamera::render(
  const Eigen::MatrixXd & arm_points,
  const Eigen::Vector2d & puck_xy,
  double puck_radius,
  double arm_radius) const
{
  const CameraConfig & c = config_;
  const double r_arm = (arm_radius < 0.0) ? c.arm_sphere_radius : arm_radius;

  const Eigen::Index k = arm_points.rows();
  Eigen::MatrixXd centres(k + 1, 3);
  Eigen::VectorXd radii(k + 1);
  if (k > 0) {
    centres.topRows(k) = arm_points;
    radii.head(k).setConstant(r_arm);
  }
  centres.row(k) << puck_xy[0], puck_xy[1], c.table_height + puck_radius;
  radii[k] = puck_radius;

  Eigen::VectorXd depth = table_;
  for (Eigen::Index r = 0; r < rays_.rows(); ++r) {
    const Eigen::Vector3d ray = rays_.row(r).transpose();
    double nearest = depth[r];
    for (Eigen::Index s = 0; s < centres.rows(); ++s) {
      const Eigen::Vector3d oc = origin_ - centres.row(s).transpose();
      const double b = ray.dot(oc);
      const double c_term = oc.squaredNorm() - radii[s] * radii[s];
      const double disc = b * b - c_term;
      if (disc < 0.0) {continue;}
      const double sq = std::sqrt(disc);
      const double t0 = -b - sq;
      const double t1 = -b + sq;
      const double t = (t0 > 1e-6) ? t0 : ((t1 > 1e-6) ? t1 : kInf);
      if (t < nearest) {nearest = t;}
    }
    depth[r] = std::min(std::max(nearest, c.z_near), c.z_far);
  }
  return depth;
}

std::vector<Eigen::VectorXd> DepthCamera::render_batch(
  const std::vector<Eigen::MatrixXd> & arm_points,
  const std::vector<Eigen::Vector2d> & puck_xy,
  double puck_radius,
  double arm_radius) const
{
  std::vector<Eigen::VectorXd> out;
  out.reserve(arm_points.size());
  for (size_t i = 0; i < arm_points.size(); ++i) {
    out.push_back(render(arm_points[i], puck_xy[i], puck_radius, arm_radius));
  }
  return out;
}

Discretiser depth_discretiser(double depth_resolution, double ignore_beyond)
{
  return [depth_resolution, ignore_beyond](const Eigen::VectorXd & depth) {
    OutcomeKey key;
    key.reserve(static_cast<size_t>(depth.size()));
    for (Eigen::Index i = 0; i < depth.size(); ++i) {
      const double d = (depth[i] > ignore_beyond) ? ignore_beyond : depth[i];
      key.push_back(static_cast<int64_t>(std::llround(d / depth_resolution)));
    }
    return key;
  };
}

}  // namespace intrinsic_core
