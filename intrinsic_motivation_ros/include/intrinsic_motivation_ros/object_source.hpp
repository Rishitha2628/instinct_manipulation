// Where the agent's knowledge of objects comes from: its own camera.
//
// Both the monitor and the climber need exactly this, and they used to get it
// from /puck/pose instead -- the simulator handing over the answer. Sharing
// one implementation rather than two copies is not only tidiness: the pair
// have to agree about the camera and the detector for the published
// empowerment number to describe the arm that is actually moving, and two
// copies of that setup is precisely how the field of view drifted out of step
// between the world file and the model earlier in this project.
//
// There is deliberately NO ground-truth path in here. Not a disabled one, not
// one behind a parameter. A node that includes this header cannot read an
// object's true position even by accident, because nothing in it subscribes
// to anything that carries one. Grading the detector is the job of a separate
// process (see detector_eval), which is the only place the two are allowed to
// meet.
#ifndef INTRINSIC_MOTIVATION_ROS__OBJECT_SOURCE_HPP_
#define INTRINSIC_MOTIVATION_ROS__OBJECT_SOURCE_HPP_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "intrinsic_core/perception.hpp"
#include "intrinsic_core/sensors.hpp"

namespace intrinsic_motivation_ros
{

/// Latest depth frame in, tracked object positions out.
class DepthObjectSource
{
public:
  DepthObjectSource(
    rclcpp::Node * node,
    const intrinsic_core::CameraConfig & camera,
    double plane_tolerance = 0.015,
    int min_pixels = 40,
    double arm_exclusion_radius = 0.15,
    int max_missing = 5,
    const std::string & topic = "camera/depth/image_raw",
    rclcpp::CallbackGroup::SharedPtr callback_group = nullptr);

  bool ready() const {return have_depth_;}

  /// Detect, associate, and return the current tracks.
  const std::vector<intrinsic_core::Track> & update(
    const Eigen::Vector2d * tool_xy = nullptr);

  /// The tracked object closest to the tool, or nothing if none is seen.
  ///
  /// The forward model carries a single object, so somebody has to choose.
  /// Nearest-to-the-tool is the one the agent is in a position to affect, and
  /// at this horizon the only one whose empowerment contribution can be
  /// non-zero anyway.
  std::optional<Eigen::Vector2d> nearest_to(const Eigen::Vector2d & tool_xy) const;

  const std::vector<intrinsic_core::Track> & tracks() const {return tracker_.tracks();}

  /// Raw blobs found in the most recent frame, before tracking. Compare it
  /// against tracks().size(): if the raw count dips while the track count
  /// rises, the detector is losing sight of an object and the tracker is
  /// remembering it, which is a perception dropout rather than a tracking
  /// fault.
  int last_detection_count() const {return last_detection_count_;}
  std::optional<double> last_dt() const {return last_dt_;}

private:
  void on_depth(const sensor_msgs::msg::Image::SharedPtr msg);

  rclcpp::Node * node_;
  intrinsic_core::DepthSegmenter segmenter_;
  intrinsic_core::ObjectTracker tracker_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

  Eigen::MatrixXd last_depth_;
  bool have_depth_ = false;
  std::optional<double> last_update_;
  std::optional<double> last_dt_;
  int last_detection_count_ = 0;
};

}  // namespace intrinsic_motivation_ros

#endif  // INTRINSIC_MOTIVATION_ROS__OBJECT_SOURCE_HPP_
