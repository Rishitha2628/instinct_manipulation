#include "intrinsic_core/perception.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <Eigen/Eigenvalues>

namespace intrinsic_core
{

namespace
{
constexpr double kInf = std::numeric_limits<double>::infinity();

double median_of(std::vector<double> v)
{
  if (v.empty()) {return 0.0;}
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
  const double hi = v[mid];
  if (v.size() % 2 != 0) {return hi;}
  std::nth_element(v.begin(), v.begin() + static_cast<long>(mid - 1), v.end());
  return 0.5 * (hi + v[mid - 1]);
}

/// 4-connected labelling, matching scipy.ndimage.label's default structure.
/// Returns the number of labels; `labels` is filled with 0 for background.
int label_blobs(
  const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> & above,
  Eigen::MatrixXi * labels)
{
  const int h = static_cast<int>(above.rows());
  const int w = static_cast<int>(above.cols());
  *labels = Eigen::MatrixXi::Zero(h, w);

  int next = 0;
  std::vector<std::pair<int, int>> stack;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (!above(y, x) || (*labels)(y, x) != 0) {continue;}
      ++next;
      stack.clear();
      stack.emplace_back(y, x);
      (*labels)(y, x) = next;
      while (!stack.empty()) {
        const auto [cy, cx] = stack.back();
        stack.pop_back();
        const int dy[4] = {-1, 1, 0, 0};
        const int dx[4] = {0, 0, -1, 1};
        for (int k = 0; k < 4; ++k) {
          const int ny = cy + dy[k], nx = cx + dx[k];
          if (ny < 0 || ny >= h || nx < 0 || nx >= w) {continue;}
          if (!above(ny, nx) || (*labels)(ny, nx) != 0) {continue;}
          (*labels)(ny, nx) = next;
          stack.emplace_back(ny, nx);
        }
      }
    }
  }
  return next;
}
}  // namespace

double Track::independent_motion_std() const
{
  if (free_motion.size() < 2) {return 0.0;}
  const double mean =
    std::accumulate(free_motion.begin(), free_motion.end(), 0.0) /
    static_cast<double>(free_motion.size());
  double acc = 0.0;
  for (double v : free_motion) {acc += (v - mean) * (v - mean);}
  return std::sqrt(acc / static_cast<double>(free_motion.size()));
}

DepthSegmenter::DepthSegmenter(
  const CameraConfig & camera, const SegmenterConfig & config)
: camera_(camera), config_(config)
{
}

const DepthCamera & DepthSegmenter::rays_for(int height, int width) const
{
  const auto key = std::make_pair(height, width);
  auto it = ray_cache_.find(key);
  if (it == ray_cache_.end()) {
    CameraConfig cam = camera_;
    cam.width = width;
    cam.height = height;
    it = ray_cache_.emplace(key, std::make_shared<DepthCamera>(cam)).first;
  }
  return *it->second;
}

Eigen::MatrixXd DepthSegmenter::back_project(const Eigen::MatrixXd & depth) const
{
  const int h = static_cast<int>(depth.rows());
  const int w = static_cast<int>(depth.cols());
  const DepthCamera & dc = rays_for(h, w);
  const Eigen::MatrixXd & rays = dc.rays();
  const Eigen::Vector3d & origin = dc.origin();

  Eigen::Vector3d forward = (camera_.look_at - origin).normalized();

  Eigen::MatrixXd points(static_cast<Eigen::Index>(h) * w, 3);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const Eigen::Index i = static_cast<Eigen::Index>(y) * w + x;
      double d = depth(y, x);
      if (!config_.depth_is_range) {
        // Along-axis depth: scale to range along the ray. The optical axis is
        // the ray at the image centre, up to the half-pixel offset, so take
        // it from the camera basis instead.
        const double cos_theta = rays.row(i).dot(forward.transpose());
        d = d / std::max(cos_theta, 1e-6);
      }
      points.row(i) = origin.transpose() + rays.row(i) * d;
    }
  }
  return points;
}

void DepthSegmenter::fit_plane(
  const Eigen::MatrixXd & points, Eigen::Vector3d * normal_out, double * offset_out) const
{
  std::vector<Eigen::Index> finite;
  finite.reserve(static_cast<size_t>(points.rows()));
  for (Eigen::Index i = 0; i < points.rows(); ++i) {
    if (points.row(i).allFinite()) {finite.push_back(i);}
  }
  if (finite.size() < 3) {
    throw std::runtime_error("not enough finite points to fit a plane");
  }

  Eigen::MatrixXd pts(static_cast<Eigen::Index>(finite.size()), 3);
  for (size_t i = 0; i < finite.size(); ++i) {
    pts.row(static_cast<Eigen::Index>(i)) = points.row(finite[i]);
  }

  std::vector<char> keep(static_cast<size_t>(pts.rows()), 1);
  Eigen::Vector3d normal(0.0, 0.0, 1.0);
  double offset = 0.0;
  bool have_previous = false;
  Eigen::Vector3d prev_normal = normal;
  double prev_offset = 0.0;

  // Iterate to convergence rather than a fixed count. Measured on a real
  // 640x480 Gazebo frame, where the view also contains the floor a metre
  // below the table, the arm, and both benches, the fit converges at round 4
  // and the trimmed scale settles at 2.7 mm -- the world file's 3 mm sensor
  // noise. The fit ends up measuring the noise floor, which is the right
  // place to stop.
  for (int iter = 0; iter < config_.plane_iterations; ++iter) {
    std::vector<Eigen::Index> sel;
    sel.reserve(keep.size());
    for (size_t i = 0; i < keep.size(); ++i) {
      if (keep[i]) {sel.push_back(static_cast<Eigen::Index>(i));}
    }
    // Trimming can in principle discard everything -- a frame that is mostly
    // arm, for instance. Stop and keep the previous fit rather than taking
    // the mean of an empty set, which yields a NaN plane and an empty mask
    // that fails much later and much less clearly.
    if (sel.size() < 3) {break;}

    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (Eigen::Index i : sel) {centroid += pts.row(i).transpose();}
    centroid /= static_cast<double>(sel.size());

    // Normal is the least-variance direction of the centred points, taken
    // from the eigenvector of the 3x3 scatter matrix rather than from an SVD
    // of the full (N,3): identical answer, but the SVD dominated the whole
    // detector at 151 ms of 207 ms per frame on a 640x480 image.
    Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
    for (Eigen::Index i : sel) {
      const Eigen::Vector3d d = pts.row(i).transpose() - centroid;
      scatter += d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scatter);
    normal = solver.eigenvectors().col(0);
    offset = normal.dot(centroid);

    Eigen::VectorXd residual = pts * normal;
    residual.array() -= offset;

    std::vector<double> res(residual.data(), residual.data() + residual.size());
    const double med = median_of(res);
    std::vector<double> dev;
    dev.reserve(res.size());
    for (double r : res) {dev.push_back(std::abs(r - med));}
    const double scale = 1.4826 * median_of(dev);
    if (scale <= 1e-9) {break;}

    if (have_previous) {
      const double moved =
        std::abs(offset - prev_offset) + (normal - prev_normal).norm();
      if (moved < 1e-5) {break;}
    }
    prev_normal = normal;
    prev_offset = offset;
    have_previous = true;

    std::vector<char> tighter(res.size(), 0);
    size_t kept = 0;
    for (size_t i = 0; i < res.size(); ++i) {
      if (std::abs(res[i]) < 2.5 * scale) {tighter[i] = 1; ++kept;}
    }
    if (kept < 3) {break;}
    keep.swap(tighter);
  }

  // Point the normal at the camera so "above the table" is positive.
  const Eigen::Vector3d origin =
    rays_for(camera_.height, camera_.width).origin();
  if (normal.dot(origin) - offset < 0.0) {normal = -normal; offset = -offset;}

  *normal_out = normal;
  *offset_out = offset;
}

std::vector<Detection> DepthSegmenter::detect(
  const Eigen::MatrixXd & depth,
  const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> * self_mask) const
{
  const SegmenterConfig & c = config_;
  const int h = static_cast<int>(depth.rows());
  const int w = static_cast<int>(depth.cols());

  const Eigen::MatrixXd points = back_project(depth);
  Eigen::Vector3d normal;
  double offset = 0.0;
  fit_plane(points, &normal, &offset);

  // Rays that hit nothing come back as inf, and inf * normal is an inf
  // standoff, which compares as "above the plane" and would be counted as an
  // object. Making the fallback explicit means it is deliberate rather than a
  // coincidence of ordering.
  Eigen::MatrixXd standoff(h, w);
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> above(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const Eigen::Index i = static_cast<Eigen::Index>(y) * w + x;
      const double s = points.row(i).allFinite()
        ? (points.row(i).dot(normal.transpose()) - offset)
        : -kInf;
      standoff(y, x) = s;
      bool is_above = s > c.plane_tolerance;
      if (self_mask != nullptr && (*self_mask)(y, x)) {is_above = false;}
      above(y, x) = is_above;
    }
  }

  Eigen::MatrixXi labels;
  const int n = label_blobs(above, &labels);
  if (n == 0) {return {};}

  std::vector<Detection> detections;
  const double max_pixels =
    c.max_pixel_fraction * static_cast<double>(h) * static_cast<double>(w);

  for (int label = 1; label <= n; ++label) {
    std::vector<int> xs, ys;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (labels(y, x) == label) {xs.push_back(x); ys.push_back(y);}
      }
    }
    if (static_cast<int>(ys.size()) < c.min_pixels ||
      static_cast<double>(ys.size()) > max_pixels)
    {
      continue;
    }

    const int m = c.border_margin;
    bool touches = false;
    for (size_t i = 0; i < xs.size(); ++i) {
      if (xs[i] <= m || xs[i] >= w - 1 - m || ys[i] <= m || ys[i] >= h - 1 - m) {
        touches = true;
        break;
      }
    }
    if (touches && c.reject_border) {continue;}

    // Average only the top slice, so an oblique view does not drag the
    // estimate onto the near face of the object.
    double top = -kInf;
    for (size_t i = 0; i < xs.size(); ++i) {
      top = std::max(top, standoff(ys[i], xs[i]));
    }
    const double band = std::max(c.top_slice * top, 1e-4);

    std::vector<size_t> crown;
    for (size_t i = 0; i < xs.size(); ++i) {
      if (standoff(ys[i], xs[i]) >= top - band) {crown.push_back(i);}
    }
    if (crown.size() < 3) {
      crown.resize(xs.size());
      std::iota(crown.begin(), crown.end(), 0);
    }

    Eigen::Vector2d axis_xy = Eigen::Vector2d::Zero();
    for (size_t i : crown) {
      const Eigen::Index flat = static_cast<Eigen::Index>(ys[i]) * w + xs[i];
      axis_xy += points.row(flat).head<2>().transpose();
    }
    axis_xy /= static_cast<double>(crown.size());

    Detection d;
    d.xy = axis_xy;
    d.height = top;
    d.n_pixels = static_cast<int>(ys.size());
    d.touches_border = touches;
    d.centroid_x =
      std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
    d.centroid_y =
      std::accumulate(ys.begin(), ys.end(), 0.0) / static_cast<double>(ys.size());
    detections.push_back(d);
  }

  std::stable_sort(
    detections.begin(), detections.end(),
    [](const Detection & a, const Detection & b) {return a.n_pixels > b.n_pixels;});
  return detections;
}

ObjectTracker::ObjectTracker(
  double gate, int max_missing, double arm_exclusion_radius,
  double max_speed, double max_gate)
: gate_(gate),
  max_speed_(max_speed),
  max_gate_(max_gate),
  max_missing_(max_missing),
  arm_exclusion_radius_(arm_exclusion_radius)
{
}

const std::vector<Track> & ObjectTracker::update(
  const std::vector<Detection> & detections,
  const Eigen::Vector2d * tool_xy,
  const double * dt)
{
  double gate = gate_;
  if (dt != nullptr) {gate = std::min(gate_ + max_speed_ * *dt, max_gate_);}

  std::vector<size_t> unmatched(detections.size());
  std::iota(unmatched.begin(), unmatched.end(), 0);

  for (auto & track : tracks_) {
    long best = -1;
    double best_dist = gate;
    for (size_t k = 0; k < unmatched.size(); ++k) {
      const double dist = (detections[unmatched[k]].xy - track.xy).norm();
      if (dist < best_dist) {best = static_cast<long>(k); best_dist = dist;}
    }

    if (best < 0) {
      track.visible = false;
      track.frames_missing += 1;
      continue;
    }

    const Detection & det = detections[unmatched[static_cast<size_t>(best)]];
    unmatched.erase(unmatched.begin() + best);

    const bool far_from_tool = (tool_xy == nullptr) ||
      ((det.xy - *tool_xy).norm() > arm_exclusion_radius_);
    if (far_from_tool && track.visible) {
      // Actual displacement, not the residual against the prediction: this
      // statistic is about how much the object moves on its own, and a good
      // prediction must not make a drifting object look stationary.
      track.free_motion.push_back(best_dist);
    }

    track.xy = det.xy;
    track.height = det.height;
    track.visible = true;
    track.frames_seen += 1;
    track.frames_missing = 0;
  }

  for (size_t i : unmatched) {
    Track t;
    t.xy = detections[i].xy;
    t.height = detections[i].height;
    tracks_.push_back(t);
  }

  tracks_.erase(
    std::remove_if(
      tracks_.begin(), tracks_.end(),
      [this](const Track & t) {return t.frames_missing > max_missing_;}),
    tracks_.end());
  return tracks_;
}

}  // namespace intrinsic_core
