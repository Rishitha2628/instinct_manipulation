// empowerment_node -- live n-step empowerment from joint states and depth.
//
// Subscribes:
//     joint_states                 sensor_msgs/JointState
//     camera/depth/image_raw       sensor_msgs/Image
//
// Publishes:
//     empowerment/value            std_msgs/Float64      bits
//     empowerment/markers          visualization_msgs/MarkerArray
//     empowerment/detections       geometry_msgs/PoseArray
//
// Objects come from the camera, never from the simulator: see
// object_source.hpp, which has no ground-truth path of any kind.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <geometry_msgs/msg/pose_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "intrinsic_core/batched.hpp"
#include "intrinsic_core/empowerment.hpp"
#include "intrinsic_core/sensors.hpp"
#include "intrinsic_core/ur5.hpp"
#include "intrinsic_motivation_ros/object_source.hpp"

using namespace intrinsic_core;             // NOLINT(build/namespaces)
using intrinsic_motivation_ros::DepthObjectSource;

namespace
{
const std::vector<std::string> UR5_JOINT_ORDER = {
  "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};

double seconds_now()
{
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

class EmpowermentNode : public rclcpp::Node
{
public:
  EmpowermentNode()
  : Node("empowerment_node")
  {
    declare_all();

    UR5Config arm_cfg;
    arm_cfg.dq = get_parameter("joint_step").as_double();
    // Bound to a named copy first, deliberately. get_parameter returns a
    // Parameter BY VALUE and as_integer_array() hands back a reference into
    // it, so iterating the call directly walks a vector whose owner has
    // already been destroyed. It does not crash where you wrote it: the
    // garbage indices come back out as column numbers and Eigen aborts
    // inside UR5's action table, several frames away.
    const std::vector<int64_t> active = get_parameter("active_joints").as_integer_array();
    arm_cfg.active_joints.clear();
    for (int64_t j : active) {
      arm_cfg.active_joints.push_back(static_cast<int>(j));
    }
    arm_cfg.tool_radius = get_parameter("tool_radius").as_double();
    arm_cfg.puck_radius = get_parameter("object_radius").as_double();
    arm_cfg.puck_mass_factor = get_parameter("object_mass_factor").as_double();
    arm_cfg.table_height = get_parameter("table_height").as_double();
    arm_ = std::make_shared<UR5>(arm_cfg);

    // up=(0,0,1) is world up. It used to be (1,0,0) because the camera looked
    // straight down and world-up was parallel to the view direction, which
    // makes the camera basis degenerate. The camera is oblique now, so the
    // natural choice works again -- and it matters for more than tidiness:
    // the segmenter back-projects Gazebo's pixels through this same basis,
    // and gz-sim orients its image with world up.
    CameraConfig cam;
    const auto pos = get_parameter("camera_position").as_double_array();
    const auto look = get_parameter("camera_look_at").as_double_array();
    cam.position = {pos[0], pos[1], pos[2]};
    cam.look_at = {look[0], look[1], look[2]};
    cam.up = {0.0, 0.0, 1.0};
    cam.fov_deg = get_parameter("camera_fov_deg").as_double();
    cam.width = static_cast<int>(get_parameter("camera_width").as_int());
    cam.height = static_cast<int>(get_parameter("camera_height").as_int());
    cam.table_height = arm_cfg.table_height;
    camera_ = std::make_shared<DepthCamera>(cam);

    // Objects come from the camera. See object_source: it deliberately has no
    // ground-truth path of any kind, so this node cannot read a true object
    // position even by accident.
    objects_ = std::make_shared<DepthObjectSource>(
      this, cam,
      get_parameter("plane_tolerance").as_double(),
      static_cast<int>(get_parameter("min_detection_pixels").as_int()),
      get_parameter("arm_exclusion_radius").as_double(),
      static_cast<int>(get_parameter("track_max_missing").as_int()));

    EmpowermentConfig est_cfg;
    est_cfg.n_steps = static_cast<int>(get_parameter("horizon").as_int());
    est_cfg.n_sequences = static_cast<int>(get_parameter("n_sequences").as_int());
    est_cfg.exhaustive_below =
      static_cast<int>(get_parameter("exhaustive_below").as_int());
    est_cfg.seed = 0;
    estimator_ = std::make_shared<BatchedDepthEmpowerment>(
      *arm_, *camera_,
      depth_discretiser(get_parameter("depth_resolution").as_double()),
      est_cfg);

    joint_names_ = get_parameter("joint_names").as_string_array();
    const auto q0 = get_parameter("initial_joint_state").as_double_array();
    q_ = Eigen::VectorXd(6);
    for (int i = 0; i < 6; ++i) {q_[i] = q0[static_cast<size_t>(i)];}
    const auto obj0 = get_parameter("initial_object_xy").as_double_array();
    object_xy_ = Eigen::Vector2d(obj0[0], obj0[1]);

    sub_joints_ = create_subscription<sensor_msgs::msg::JointState>(
      "joint_states", 10,
      [this](const sensor_msgs::msg::JointState::SharedPtr m) {on_joints(m);});

    pub_value_ = create_publisher<std_msgs::msg::Float64>("empowerment/value", 10);
    pub_marker_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("empowerment/markers", 10);
    pub_detections_ =
      create_publisher<geometry_msgs::msg::PoseArray>("empowerment/detections", 10);

    const double rate = std::max(get_parameter("rate_hz").as_double(), 1e-3);
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate), [this]() {tick();});

    RCLCPP_INFO(
      get_logger(),
      "empowerment_node up | %d actions, horizon %ld, camera %ldx%ld, "
      "objects located from the depth image, no ground truth anywhere",
      arm_->n_actions(), get_parameter("horizon").as_int(),
      get_parameter("camera_width").as_int(),
      get_parameter("camera_height").as_int());
  }

private:
  void declare_all()
  {
    declare_parameter("joint_step", 0.10);
    declare_parameter("active_joints", std::vector<int64_t>{0, 1, 2, 3});
    declare_parameter("tool_radius", 0.05);
    declare_parameter("object_radius", 0.04);
    declare_parameter("object_mass_factor", 1.0);

    // Work surface in the arm's base frame. Non-zero whenever the arm is
    // mounted higher than the bench it works over, which it is here. Feeds
    // the contact model and the camera's ground plane, so getting it wrong
    // makes the agent push at empty air and range against a phantom table.
    declare_parameter("table_height", 0.0);

    declare_parameter("horizon", 2);
    declare_parameter("n_sequences", 2000);
    declare_parameter("exhaustive_below", 0);
    declare_parameter("depth_resolution", 0.02);

    // --- perception ---
    // Metres a point must stand off the fitted support plane to count as an
    // object. Floor is sensor noise: the world's camera has a 3 mm gaussian
    // and a real frame fits to a 2.7 mm residual, so anything under ~1 cm
    // starts segmenting the table itself into speckle.
    declare_parameter("plane_tolerance", 0.015);
    declare_parameter("min_detection_pixels", 40);

    // Motion nearer than this to the tool is assumed to be the agent's doing,
    // so it is not counted toward an object's independent motion. Without it,
    // a push would be recorded as the object moving by itself, which inverts
    // the distinction the drifter exists to test.
    declare_parameter("arm_exclusion_radius", 0.15);

    // Detector ticks an occluded object may be remembered for before it is
    // forgotten. The object-permanence window, in ticks rather than seconds,
    // so at rate_hz 1.0 it is also seconds.
    declare_parameter("track_max_missing", 5);

    declare_parameter("camera_position", std::vector<double>{-0.50, 0.0, 0.95});
    declare_parameter("camera_look_at", std::vector<double>{-0.50, 0.0, 0.0});
    declare_parameter("camera_fov_deg", 30.0);
    declare_parameter("camera_width", 16);
    declare_parameter("camera_height", 12);

    declare_parameter("rate_hz", 1.0);
    declare_parameter("joint_names", UR5_JOINT_ORDER);
    declare_parameter(
      "initial_joint_state", std::vector<double>{0.0, -1.2, 1.4, -1.75, -1.57, 0.0});
    declare_parameter("initial_object_xy", std::vector<double>{-0.50, 0.0});
    declare_parameter("frame_id", std::string("base_link"));
  }

  void on_joints(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    Eigen::VectorXd q(6);
    for (size_t i = 0; i < joint_names_.size() && i < 6; ++i) {
      const auto it =
        std::find(msg->name.begin(), msg->name.end(), joint_names_[i]);
      if (it == msg->name.end()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "joint_states missing %s",
          joint_names_[i].c_str());
        return;
      }
      q[static_cast<Eigen::Index>(i)] =
        msg->position[static_cast<size_t>(std::distance(msg->name.begin(), it))];
    }
    q_ = q;
    have_joints_ = true;
  }

  /// Locate objects in the latest frame and pick one to reason about.
  void update_from_depth()
  {
    if (!objects_->ready()) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000, "waiting for a depth frame ...");
      return;
    }

    const Eigen::Vector3d tool = arm_->tool_position(q_);
    const Eigen::Vector2d tool_xy = tool.head<2>();
    const auto & tracks = objects_->update(&tool_xy);
    n_raw_ = objects_->last_detection_count();
    if (tracks.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 10000, "nothing found above the support plane");
      return;
    }

    const auto nearest = objects_->nearest_to(tool_xy);
    if (nearest.has_value()) {object_xy_ = *nearest;}
    publish_detections(tracks);
  }

  /// Where the agent believes the objects are.
  ///
  /// This is the agent's estimate and nothing else. Comparing it against the
  /// simulator's truth is detector_eval's job, in its own process, so that no
  /// ground truth is reachable from here.
  void publish_detections(const std::vector<Track> & tracks)
  {
    geometry_msgs::msg::PoseArray msg;
    msg.header.frame_id = get_parameter("frame_id").as_string();
    msg.header.stamp = now();
    const double table = get_parameter("table_height").as_double();
    for (const auto & track : tracks) {
      geometry_msgs::msg::Pose pose;
      pose.position.x = track.xy[0];
      pose.position.y = track.xy[1];
      pose.position.z = table + track.height;
      pose.orientation.w = 1.0;
      msg.poses.push_back(pose);
    }
    pub_detections_->publish(msg);
  }

  void tick()
  {
    if (!have_joints_) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000, "waiting for /joint_states ...");
      return;
    }

    // Timing lives here permanently rather than as scaffolding, because this
    // loop does not keep to rate_hz and the distinction between "the work got
    // slow" and "we were not scheduled" is invisible from the published value
    // alone. gap is wall time since the previous tick; the two phases are the
    // actual work. If the phases stay small while gap balloons, the executor
    // is the problem, not the workload.
    const double entered = seconds_now();
    const bool have_gap = last_entered_ > 0.0;
    const double gap = entered - last_entered_;
    last_entered_ = entered;

    update_from_depth();
    const double detected = seconds_now();

    Eigen::VectorXd state(8);
    state << q_, object_xy_[0], object_xy_[1];
    const double value = estimator_->estimate(state);
    const double estimated = seconds_now();

    std_msgs::msg::Float64 out;
    out.data = value;
    pub_value_->publish(out);
    publish_marker(value);

    if (have_gap) {
      const auto dt = objects_->last_dt();
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "tick | gap %6.2fs  perceive %5.3fs  estimate %5.3fs  work %5.3fs  "
        "raw %d tracks %zu  dt %s",
        gap, detected - entered, estimated - detected, estimated - entered,
        n_raw_, objects_->tracks().size(),
        dt.has_value() ? std::to_string(*dt).c_str() : "--");
    }
  }

  void publish_marker(double value)
  {
    const Eigen::Vector3d tool = arm_->tool_position(q_);

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = get_parameter("frame_id").as_string();
    marker.header.stamp = now();
    marker.ns = "empowerment";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = tool[0];
    marker.pose.position.y = tool[1];
    marker.pose.position.z = tool[2] + 0.15;
    marker.pose.orientation.w = 1.0;
    marker.scale.z = 0.05;
    marker.color.a = 1.0;
    marker.color.r = marker.color.g = marker.color.b = 1.0;
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f bits", value);
    marker.text = text;

    visualization_msgs::msg::MarkerArray array;
    array.markers.push_back(marker);
    pub_marker_->publish(array);
  }

  std::shared_ptr<UR5> arm_;
  std::shared_ptr<DepthCamera> camera_;
  std::shared_ptr<DepthObjectSource> objects_;
  std::shared_ptr<BatchedDepthEmpowerment> estimator_;

  std::vector<std::string> joint_names_;
  Eigen::VectorXd q_;
  Eigen::Vector2d object_xy_{-0.50, 0.0};
  bool have_joints_ = false;
  double last_entered_ = 0.0;
  int n_raw_ = 0;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joints_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_value_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_detections_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EmpowermentNode>());
  rclcpp::shutdown();
  return 0;
}
