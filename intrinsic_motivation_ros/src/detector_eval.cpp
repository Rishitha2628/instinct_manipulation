// detector_eval -- scores the agent's object estimates against the
// simulator's ground truth.
//
// This is a SEPARATE PROCESS on purpose. The agent's own nodes must not be
// able to read a true object position even by accident, so the comparison
// lives here and nothing here feeds back into them.
//
// Errors are published per object rather than only as an average, because
// they mean different things and averaging them hides that. For a stationary
// object the error is the detector's accuracy. For the drifter it is
// dominated by STALENESS instead: the agent's estimate is only refreshed when
// its loop runs, which is irregular and well below 1 Hz, while the truth
// moves continuously. A large drifter error therefore says the loop is slow,
// not that the detector is inaccurate, and the fix for it is in the loop
// rather than in perception.
//
// The drifter's truth arrives as a joint position rather than a pose because
// its model root is a fixed anchor that never moves, so the model pose says
// nothing; the block's world y is the anchor's y plus the slide value.
//
// It exists so that a claim like "located to within 6 mm without being told
// where anything was" has a number behind it.
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>

class DetectorEval : public rclcpp::Node
{
public:
  DetectorEval()
  : Node("detector_eval")
  {
    // Where the drifter's anchor sits, so the slide joint can be turned into
    // a world position. Matches the <pose> of the drifter model in
    // worlds/empowerment_table.sdf.
    const auto anchor =
      declare_parameter("drifter_anchor_xy", std::vector<double>{-0.68, -0.24});
    anchor_ = Eigen::Vector2d(anchor[0], anchor[1]);

    sub_puck_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "puck/pose", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr m) {
        on_pose("puck", m);
      });
    sub_block_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "fixed_block/pose", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr m) {
        on_pose("fixed_block", m);
      });
    sub_drifter_ = create_subscription<sensor_msgs::msg::JointState>(
      "drifter/joint_states", 10,
      [this](const sensor_msgs::msg::JointState::SharedPtr m) {on_drifter(m);});
    sub_detections_ = create_subscription<geometry_msgs::msg::PoseArray>(
      "empowerment/detections", 10,
      [this](const geometry_msgs::msg::PoseArray::SharedPtr m) {on_detections(m);});

    pub_error_ = create_publisher<std_msgs::msg::Float64>("detector/error", 10);
    pub_max_ = create_publisher<std_msgs::msg::Float64>("detector/max_error", 10);
    pub_count_ = create_publisher<std_msgs::msg::Float64>("detector/count", 10);

    RCLCPP_INFO(
      get_logger(),
      "detector_eval up | scoring /empowerment/detections against simulator "
      "ground truth");
  }

private:
  void on_pose(
    const std::string & name, const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    truth_[name] = Eigen::Vector2d(msg->pose.position.x, msg->pose.position.y);
  }

  /// World xy of the drifter's block, from its slide joint.
  ///
  /// By name, not by index. The model reports two joints -- the fixed
  /// anchor_to_world first, then slide -- and taking position[0] gets the
  /// fixed one, which reads 0.0 for ever and makes a moving object look
  /// bolted down. That mistake cost an afternoon here.
  void on_drifter(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i) {
      if (msg->name[i] == "slide") {
        truth_["drifter"] = anchor_ + Eigen::Vector2d(0.0, msg->position[i]);
        return;
      }
    }
  }

  void on_detections(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    if (truth_.empty()) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000, "waiting for ground truth ...");
      return;
    }

    std::vector<Eigen::Vector2d> estimates;
    for (const auto & p : msg->poses) {
      estimates.emplace_back(p.position.x, p.position.y);
    }

    std_msgs::msg::Float64 out;
    out.data = static_cast<double>(estimates.size());
    pub_count_->publish(out);
    if (estimates.empty()) {return;}

    // Each true object scored against its nearest estimate. Deliberately not
    // a full assignment: a spurious extra detection should not be able to
    // improve the error, and it shows up in /detector/count instead, which is
    // where a miscount belongs.
    std::vector<double> values;
    for (const auto & [name, truth] : truth_) {
      double best = std::numeric_limits<double>::infinity();
      for (const auto & e : estimates) {best = std::min(best, (e - truth).norm());}
      values.push_back(best);

      auto it = pub_per_object_.find(name);
      if (it == pub_per_object_.end()) {
        it = pub_per_object_.emplace(
          name,
          create_publisher<std_msgs::msg::Float64>("detector/error/" + name, 10)).first;
      }
      out.data = best;
      it->second->publish(out);
    }

    out.data =
      std::accumulate(values.begin(), values.end(), 0.0) /
      static_cast<double>(values.size());
    pub_error_->publish(out);
    out.data = *std::max_element(values.begin(), values.end());
    pub_max_->publish(out);
  }

  Eigen::Vector2d anchor_{-0.68, -0.24};
  std::map<std::string, Eigen::Vector2d> truth_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_puck_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_block_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_drifter_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sub_detections_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_error_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_max_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_count_;
  std::map<std::string, rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr>
  pub_per_object_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DetectorEval>());
  rclcpp::shutdown();
  return 0;
}
