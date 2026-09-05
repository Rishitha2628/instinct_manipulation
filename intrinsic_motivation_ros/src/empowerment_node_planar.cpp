// empowerment_node_planar -- the 2-link version, kept for the simple demo.
//
// Ground-truth outcomes rather than a rendered depth image, which is the
// wrong thing by the paper's definition but is the right place to build
// intuition: it runs in milliseconds and needs no simulator.
//
// Subscribes:
//     joint_states     sensor_msgs/JointState
//     object_pose      geometry_msgs/PoseStamped
//
// Publishes:
//     empowerment/value      std_msgs/Float64
//     empowerment/gradient   geometry_msgs/Vector3
//     empowerment/markers    visualization_msgs/MarkerArray
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "intrinsic_core/arm2d.hpp"
#include "intrinsic_core/discretise.hpp"
#include "intrinsic_core/empowerment.hpp"

using namespace intrinsic_core;             // NOLINT(build/namespaces)

class EmpowermentNodePlanar : public rclcpp::Node
{
public:
  EmpowermentNodePlanar()
  : Node("empowerment_node")
  {
    declare_parameter("link1_length", 0.30);
    declare_parameter("link2_length", 0.25);
    declare_parameter("joint_step", 0.12);
    declare_parameter("contact_radius", 0.045);
    declare_parameter("object_mass_factor", 1.0);
    declare_parameter("horizon", 3);
    declare_parameter("n_sequences", 300);
    declare_parameter("outcome_resolution", 0.025);
    declare_parameter("rate_hz", 5.0);
    declare_parameter("joint_names", std::vector<std::string>{"joint1", "joint2"});
    declare_parameter("initial_object_xy", std::vector<double>{0.30, 0.25});
    declare_parameter("frame_id", std::string("base_link"));

    ArmConfig arm_cfg;
    arm_cfg.l1 = get_parameter("link1_length").as_double();
    arm_cfg.l2 = get_parameter("link2_length").as_double();
    arm_cfg.dq = get_parameter("joint_step").as_double();
    arm_cfg.puck_radius = get_parameter("contact_radius").as_double();
    arm_cfg.puck_mass_factor = get_parameter("object_mass_factor").as_double();
    arm_ = std::make_shared<PlanarArm>(arm_cfg);

    EmpowermentConfig cfg;
    cfg.n_steps = static_cast<int>(get_parameter("horizon").as_int());
    cfg.n_sequences = static_cast<int>(get_parameter("n_sequences").as_int());
    cfg.seed = 0;
    auto arm = arm_;
    estimator_ = std::make_shared<EmpowermentEstimator>(
      [arm](const Eigen::VectorXd & s, const std::vector<int> & a) {
        return arm->rollout(s, a);
      },
      9, grid_discretiser(get_parameter("outcome_resolution").as_double()), cfg);

    joint_names_ = get_parameter("joint_names").as_string_array();
    const auto obj = get_parameter("initial_object_xy").as_double_array();
    object_xy_ = Eigen::Vector2d(obj[0], obj[1]);

    sub_joints_ = create_subscription<sensor_msgs::msg::JointState>(
      "joint_states", 10,
      [this](const sensor_msgs::msg::JointState::SharedPtr m) {on_joints(m);});
    sub_object_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "object_pose", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr m) {
        object_xy_ = Eigen::Vector2d(m->pose.position.x, m->pose.position.y);
      });

    pub_value_ = create_publisher<std_msgs::msg::Float64>("empowerment/value", 10);
    pub_grad_ =
      create_publisher<geometry_msgs::msg::Vector3>("empowerment/gradient", 10);
    auto latched = rclcpp::QoS(1).reliable().transient_local();
    pub_marker_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "empowerment/markers", latched);

    const double rate = std::max(get_parameter("rate_hz").as_double(), 1e-3);
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate), [this]() {tick();});

    RCLCPP_INFO(
      get_logger(), "empowerment_node up: horizon=%ld, resolution=%.3f m, rate=%.1f Hz",
      get_parameter("horizon").as_int(),
      get_parameter("outcome_resolution").as_double(), rate);
  }

private:
  void on_joints(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    Eigen::Vector2d q;
    for (size_t i = 0; i < joint_names_.size() && i < 2; ++i) {
      const auto it = std::find(msg->name.begin(), msg->name.end(), joint_names_[i]);
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

  Eigen::VectorXd state_of(const Eigen::Vector2d & q) const
  {
    Eigen::VectorXd s(4);
    s << q[0], q[1], object_xy_[0], object_xy_[1];
    return s;
  }

  void tick()
  {
    if (!have_joints_) {return;}

    const double value = estimator_->estimate(state_of(q_));
    std_msgs::msg::Float64 out;
    out.data = value;
    pub_value_->publish(out);

    const Eigen::Vector2d gradient = finite_difference_gradient();
    geometry_msgs::msg::Vector3 g;
    g.x = gradient[0];
    g.y = gradient[1];
    g.z = 0.0;
    pub_grad_->publish(g);

    publish_markers(value);
  }

  /// Central difference in joint space.
  ///
  /// Four extra empowerment evaluations per cycle. Empowerment has no
  /// analytic gradient -- it is a count -- so this is the only option, and it
  /// is the main reason the node is slow.
  Eigen::Vector2d finite_difference_gradient()
  {
    const double eps = get_parameter("joint_step").as_double();
    Eigen::Vector2d grad = Eigen::Vector2d::Zero();
    for (int i = 0; i < 2; ++i) {
      Eigen::Vector2d up = q_, down = q_;
      up[i] += eps;
      down[i] -= eps;
      const double e_up = estimator_->estimate(state_of(up));
      const double e_dn = estimator_->estimate(state_of(down));
      grad[i] = (e_up - e_dn) / (2.0 * eps);
    }
    return grad;
  }

  void publish_markers(double value)
  {
    const Eigen::Vector2d e = arm_->forward_kinematics(q_[0], q_[1]);

    visualization_msgs::msg::Marker text;
    text.header.frame_id = get_parameter("frame_id").as_string();
    text.header.stamp = now();
    text.ns = "empowerment";
    text.id = 0;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.position.x = e[0];
    text.pose.position.y = e[1];
    text.pose.position.z = 0.12;
    text.pose.orientation.w = 1.0;
    text.scale.z = 0.05;
    text.color.a = 1.0;
    text.color.r = text.color.g = text.color.b = 1.0;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f bits", value);
    text.text = buf;

    visualization_msgs::msg::MarkerArray array;
    array.markers.push_back(text);
    pub_marker_->publish(array);
  }

  std::shared_ptr<PlanarArm> arm_;
  std::shared_ptr<EmpowermentEstimator> estimator_;
  std::vector<std::string> joint_names_;
  Eigen::Vector2d q_{0.0, 0.0};
  Eigen::Vector2d object_xy_{0.30, 0.25};
  bool have_joints_ = false;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joints_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_object_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_value_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr pub_grad_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EmpowermentNodePlanar>());
  rclcpp::shutdown();
  return 0;
}
