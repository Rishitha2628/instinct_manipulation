// greedy_climber_planar -- the 2-link climber, kept for the simple demo.
//
// With task_bonus_weight = 0 this is pure empowerment: the arm engages with
// the object and never finishes. A completed grasp is ONE future and
// therefore lowers the option count, so finishing has to come from a reward
// term.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "intrinsic_core/arm2d.hpp"
#include "intrinsic_core/discretise.hpp"
#include "intrinsic_core/empowerment.hpp"

using namespace intrinsic_core;             // NOLINT(build/namespaces)

class GreedyClimberPlanar : public rclcpp::Node
{
public:
  GreedyClimberPlanar()
  : Node("greedy_climber")
  {
    declare_parameter("link1_length", 0.30);
    declare_parameter("link2_length", 0.25);
    declare_parameter("joint_step", 0.12);
    declare_parameter("contact_radius", 0.045);
    declare_parameter("horizon", 3);
    declare_parameter("outcome_resolution", 0.025);
    declare_parameter("rate_hz", 4.0);
    declare_parameter("joint_names", std::vector<std::string>{"joint1", "joint2"});
    declare_parameter("initial_object_xy", std::vector<double>{0.30, 0.25});
    declare_parameter("task_bonus_weight", 0.0);
    declare_parameter("task_bonus_radius", 0.05);

    ArmConfig arm_cfg;
    arm_cfg.l1 = get_parameter("link1_length").as_double();
    arm_cfg.l2 = get_parameter("link2_length").as_double();
    arm_cfg.dq = get_parameter("joint_step").as_double();
    arm_cfg.puck_radius = get_parameter("contact_radius").as_double();
    arm_ = std::make_shared<PlanarArm>(arm_cfg);

    EmpowermentConfig cfg;
    cfg.n_steps = static_cast<int>(get_parameter("horizon").as_int());
    cfg.seed = 0;
    auto arm = arm_;
    estimator_ = std::make_shared<EmpowermentEstimator>(
      [arm](const Eigen::VectorXd & s, const std::vector<int> & a) {
        return arm->rollout(s, a);
      },
      static_cast<int>(JOINT_ACTIONS.rows()),
      grid_discretiser(get_parameter("outcome_resolution").as_double()), cfg);

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

    pub_cmd_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      "joint_group_position_controller/commands", 10);

    const double rate = std::max(get_parameter("rate_hz").as_double(), 1e-3);
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate), [this]() {tick();});

    RCLCPP_INFO(get_logger(), "greedy_climber up (no goal, no reward)");
  }

private:
  void on_joints(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    Eigen::Vector2d q;
    for (size_t i = 0; i < joint_names_.size() && i < 2; ++i) {
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
  /// With task_bonus_weight = 0 this is pure empowerment: the arm engages
  /// with the object and never finishes. Raising it supplies the intention
  /// that empowerment structurally cannot.
  double score(const Eigen::VectorXd & state) const
  {
    double value = estimator_->estimate(state);

    const double weight = get_parameter("task_bonus_weight").as_double();
    if (weight > 0.0) {
      const Eigen::Vector2d e = arm_->forward_kinematics(state[0], state[1]);
      const bool reached = std::hypot(state[2] - e[0], state[3] - e[1]) <
        get_parameter("task_bonus_radius").as_double();
      value += weight * (reached ? 1.0 : 0.0);
    }
    return value;
  }

  void tick()
  {
    if (!have_joints_) {return;}

    Eigen::VectorXd state(4);
    state << q_[0], q_[1], object_xy_[0], object_xy_[1];

    int best_action = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int action = 0; action < static_cast<int>(JOINT_ACTIONS.rows()); ++action) {
      const double s = score(arm_->step(state, action));
      if (s > best_score) {best_score = s; best_action = action;}
    }

    const Eigen::VectorXd target = arm_->step(state, best_action);
    std_msgs::msg::Float64MultiArray cmd;
    cmd.data = {target[0], target[1]};
    pub_cmd_->publish(cmd);
  }

  std::shared_ptr<PlanarArm> arm_;
  std::shared_ptr<EmpowermentEstimator> estimator_;
  std::vector<std::string> joint_names_;
  Eigen::Vector2d q_{0.0, 0.0};
  Eigen::Vector2d object_xy_{0.30, 0.25};
  bool have_joints_ = false;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joints_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_object_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_cmd_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GreedyClimberPlanar>());
  rclcpp::shutdown();
  return 0;
}
