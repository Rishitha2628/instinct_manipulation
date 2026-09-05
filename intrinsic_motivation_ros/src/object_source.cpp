#include "intrinsic_motivation_ros/object_source.hpp"

#include <cstring>

namespace intrinsic_motivation_ros
{

using intrinsic_core::SegmenterConfig;

namespace
{
SegmenterConfig make_segmenter_config(double plane_tolerance, int min_pixels)
{
  SegmenterConfig cfg;
  cfg.plane_tolerance = plane_tolerance;
  cfg.min_pixels = min_pixels;
  // gz-sim publishes distance along the optical axis, not along the ray.
  // Confirmed by fitting a real frame both ways: 0.64 deg of residual table
  // tilt read as z-depth against 3.17 deg read as range. Read it wrong and
  // positions still look right while object heights come out about 38 percent
  // low.
  cfg.depth_is_range = false;
  return cfg;
}
}  // namespace

DepthObjectSource::DepthObjectSource(
  rclcpp::Node * node,
  const intrinsic_core::CameraConfig & camera,
  double plane_tolerance,
  int min_pixels,
  double arm_exclusion_radius,
  int max_missing,
  const std::string & topic,
  rclcpp::CallbackGroup::SharedPtr callback_group)
: node_(node),
  // Same CameraConfig the agent predicts with, so the detector and the
  // renderer cannot disagree about where a pixel points. Only the resolution
  // differs, and the segmenter rebuilds its rays for whatever size of frame
  // actually turns up.
  segmenter_(camera, make_segmenter_config(plane_tolerance, min_pixels)),
  tracker_(0.10, max_missing, arm_exclusion_radius)
{
  rclcpp::SubscriptionOptions options;
  if (callback_group) {options.callback_group = callback_group;}
  sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
    topic, rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::Image::SharedPtr msg) {on_depth(msg);},
    options);
}

void DepthObjectSource::on_depth(const sensor_msgs::msg::Image::SharedPtr msg)
{
  if (msg->encoding != "32FC1" && msg->encoding != "16UC1") {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 10000,
      "unexpected depth encoding %s", msg->encoding.c_str());
    return;
  }

  const int h = static_cast<int>(msg->height);
  const int w = static_cast<int>(msg->width);
  last_depth_.resize(h, w);

  if (msg->encoding == "32FC1") {
    const float * data = reinterpret_cast<const float *>(msg->data.data());
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {last_depth_(y, x) = data[y * w + x];}
    }
  } else {
    const uint16_t * data = reinterpret_cast<const uint16_t *>(msg->data.data());
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        last_depth_(y, x) = static_cast<double>(data[y * w + x]) / 1000.0;   // mm -> m
      }
    }
  }
  have_depth_ = true;
}

const std::vector<intrinsic_core::Track> & DepthObjectSource::update(
  const Eigen::Vector2d * tool_xy)
{
  if (!have_depth_) {return tracker_.tracks();}

  const auto detections = segmenter_.detect(last_depth_);
  last_detection_count_ = static_cast<int>(detections.size());

  // Real elapsed time, not the nominal loop rate. The caller's loop does not
  // keep to its configured rate, and the tracker widens its association gate
  // by dt; handing it a nominal value would defeat that.
  const double now = node_->get_clock()->now().nanoseconds() * 1e-9;
  std::optional<double> dt;
  if (last_update_.has_value()) {dt = now - *last_update_;}
  last_update_ = now;
  last_dt_ = dt;

  const double * dt_ptr = dt.has_value() ? &(*dt) : nullptr;
  return tracker_.update(detections, tool_xy, dt_ptr);
}

std::optional<Eigen::Vector2d> DepthObjectSource::nearest_to(
  const Eigen::Vector2d & tool_xy) const
{
  const auto & tracks = tracker_.tracks();
  if (tracks.empty()) {return std::nullopt;}
  const intrinsic_core::Track * best = nullptr;
  double best_dist = 1e18;
  for (const auto & t : tracks) {
    const double d = (t.xy - tool_xy).norm();
    if (d < best_dist) {best_dist = d; best = &t;}
  }
  return best->xy;
}

}  // namespace intrinsic_motivation_ros
