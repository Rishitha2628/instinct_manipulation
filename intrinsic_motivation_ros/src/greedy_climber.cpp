// greedy_climber -- drives the UR5 up the empowerment gradient.
//
// No goal, no reward, no demonstration. Each tick it scores a random subset
// of the successors of the current state and commands the best one.
//
// Subscribes:
//     joint_states                 sensor_msgs/JointState
//     camera/depth/image_raw       sensor_msgs/Image
//
// Publishes:
//     joint_trajectory_controller/joint_trajectory   trajectory_msgs/JointTrajectory
//     empowerment/chosen_score     std_msgs/Float64
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

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
}

class GreedyClimber : public rclcpp::Node
{
public:
  GreedyClimber()
  : Node("greedy_climber")
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

    // up=(0,0,1) is world up. It was (1,0,0) because the old camera looked
    // straight down, which made world-up parallel to the view direction and
    // the camera basis degenerate. The camera is oblique now, and the natural
    // choice also matches how gz-sim orients its image, which matters because
    // the detector back-projects Gazebo's pixels through this same basis.
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

    // This node MOVES THE ARM, so it is the one that most needs to be honest
    // about where its knowledge comes from. It used to read the puck's
    // position off /puck/pose and never looked at the camera at all.
    // object_source has no ground-truth path of any kind.
    objects_ = std::make_shared<DepthObjectSource>(
      this, cam,
      get_parameter("plane_tolerance").as_double(),
      static_cast<int>(get_parameter("min_detection_pixels").as_int()),
      get_parameter("arm_exclusion_radius").as_double(),
      static_cast<int>(get_parameter("track_max_missing").as_int()));

    EmpowermentConfig est_cfg;
    est_cfg.n_steps = static_cast<int>(get_parameter("horizon").as_int());
    est_cfg.n_sequences =
      static_cast<int>(get_parameter("climber_sequences").as_int());
    est_cfg.exhaustive_below = 0;
    est_cfg.seed = 0;
    estimator_ = std::make_shared<BatchedDepthEmpowerment>(
      *arm_, *camera_,
      depth_discretiser(get_parameter("depth_resolution").as_double()), est_cfg);

    joint_names_ = get_parameter("joint_names").as_string_array();
    const auto q0 = get_parameter("initial_joint_state").as_double_array();
    q_ = Eigen::VectorXd(6);
    for (int i = 0; i < 6; ++i) {q_[i] = q0[static_cast<size_t>(i)];}
    const auto obj0 = get_parameter("initial_object_xy").as_double_array();
    object_xy_ = Eigen::Vector2d(obj0[0], obj0[1]);

    sub_joints_ = create_subscription<sensor_msgs::msg::JointState>(
      "joint_states", 10,
      [this](const sensor_msgs::msg::JointState::SharedPtr m) {on_joints(m);});

    pub_traj_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "joint_trajectory_controller/joint_trajectory", 10);
    pub_score_ =
      create_publisher<std_msgs::msg::Float64>("empowerment/chosen_score", 10);

    period_ = 1.0 / std::max(get_parameter("rate_hz").as_double(), 1e-3);
    timer_ = create_wall_timer(
      std::chrono::duration<double>(period_), [this]() {tick();});

    const double bonus = get_parameter("task_bonus_weight").as_double();
    if (bonus <= 0.0) {
      RCLCPP_INFO(
        get_logger(), "greedy_climber up | pure empowerment: no goal, no reward");
    } else {
      RCLCPP_INFO(
        get_logger(), "greedy_climber up | empowerment + task bonus %.3f", bonus);
    }
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
    // mounted higher than the bench it works over, which it is here.
    declare_parameter("table_height", 0.0);

    declare_parameter("horizon", 2);
    declare_parameter("climber_sequences", 600);
    declare_parameter("depth_resolution", 0.02);

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

    // --- perception ---
    declare_parameter("plane_tolerance", 0.015);
    declare_parameter("min_detection_pixels", 40);
    declare_parameter("arm_exclusion_radius", 0.15);
    declare_parameter("track_max_missing", 5);

    declare_parameter("task_bonus_weight", 0.0);
    declare_parameter("task_bonus_radius", 0.06);
    declare_parameter("candidate_actions", 24);
  }

  void on_joints(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    Eigen::VectorXd q(6);
    for (size_t i = 0; i < joint_names_.size() && i < 6; ++i) {
      const auto it = std::find(msg->name.begin(), msg->name.end(), joint_names_[i]);
      if (it == msg->name.end()) {return;}
      q[static_cast<Eigen::Index>(i)] =
        msg->position[static_cast<size_t>(std::distance(msg->name.begin(), it))];
    }
    q_ = q;
    have_joints_ = true;
  }

  /// Empowerment, optionally plus a one-bit task term.
  ///
  /// The task term carries NO information about where the object is or how to
  /// reach it. Empowerment supplies that. The bonus only says "contact is
  /// good", which is the intention empowerment structurally cannot represent.
  double score(const Eigen::VectorXd & state) const
  {
    double value = estimator_->estimate(state);

    const double weight = get_parameter("task_bonus_weight").as_double();
    if (weight > 0.0) {
      const Eigen::Vector3d tool = arm_->tool_position(state.head<6>());
      const double reach = std::hypot(state[6] - tool[0], state[7] - tool[1]);
      if (reach < get_parameter("task_bonus_radius").as_double()) {value += weight;}
    }
    return value;
  }

  void tick()
  {
    if (!have_joints_) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000, "waiting for /joint_states ...");
      return;
    }

    if (!objects_->ready()) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000, "waiting for a depth frame ...");
      return;
    }
    const Eigen::Vector3d tool = arm_->tool_position(q_);
    const Eigen::Vector2d tool_xy = tool.head<2>();
    objects_->update(&tool_xy);
    const auto nearest = objects_->nearest_to(tool_xy);
    if (nearest.has_value()) {object_xy_ = *nearest;}

    Eigen::VectorXd state(8);
    state << q_, object_xy_[0], object_xy_[1];

    // Evaluating all 81 successors at ~0.2 s each is too slow for a live
    // loop, so score a random subset each cycle. Over several cycles the arm
    // still climbs; it just takes a noisier path.
    const int n_candidates = std::min<int>(
      static_cast<int>(get_parameter("candidate_actions").as_int()),
      arm_->n_actions());
    std::vector<int> all(static_cast<size_t>(arm_->n_actions()));
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng_);

    int best_action = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < n_candidates; ++k) {
      const int action = all[static_cast<size_t>(k)];
      const double s = score(arm_->step(state, action));
      if (s > best_score) {best_score = s; best_action = action;}
    }
    if (best_action < 0) {return;}

    const Eigen::VectorXd target = arm_->step(state, best_action);
    send_trajectory(target.head<6>());

    std_msgs::msg::Float64 out;
    out.data = best_score;
    pub_score_->publish(out);
  }

  void send_trajectory(const Eigen::VectorXd & q_target)
  {
    trajectory_msgs::msg::JointTrajectory traj;
    traj.joint_names = joint_names_;

    trajectory_msgs::msg::JointTrajectoryPoint point;
    for (Eigen::Index i = 0; i < q_target.size(); ++i) {
      point.positions.push_back(q_target[i]);
      point.velocities.push_back(0.0);
    }
    const double seconds = period_ * 0.9;
    point.time_from_start.sec = static_cast<int32_t>(seconds);
    point.time_from_start.nanosec =
      static_cast<uint32_t>(std::fmod(seconds, 1.0) * 1e9);
    traj.points.push_back(point);
    pub_traj_->publish(traj);
  }

  std::shared_ptr<UR5> arm_;
  std::shared_ptr<DepthCamera> camera_;
  std::shared_ptr<DepthObjectSource> objects_;
  std::shared_ptr<BatchedDepthEmpowerment> estimator_;

  std::vector<std::string> joint_names_;
  Eigen::VectorXd q_;
  Eigen::Vector2d object_xy_{-0.50, 0.0};
  bool have_joints_ = false;
  double period_ = 1.0;
  std::mt19937_64 rng_{std::random_device{}()};

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joints_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_traj_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_score_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GreedyClimber>());
  rclcpp::shutdown();
  return 0;
}
