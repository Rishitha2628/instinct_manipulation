// drifter_driver -- makes the non-contingent object move.
//
// The drifter is the control that gives this scene its point. It is a
// cylinder identical to the puck in every way the depth camera can measure,
// sliding back and forth along y on a prismatic joint, and NOTHING the arm
// does affects it.
//
// That makes it a "noisy TV": a source of constant, unpredictable-looking
// sensory change that the agent has no influence over. Two intrinsic
// objectives disagree sharply about it, which is the whole reason it is here:
//
//   * prediction-error curiosity is ATTRACTED to it, because it keeps
//     changing and a forward model that does not know about it keeps being
//     wrong;
//   * empowerment should IGNORE it, because the mutual information between
//     the agent's own actions and the drifter's state is zero.
//
// Publishes:
//     /drifter/position           std_msgs/Float64    metres along the slide
//
// bridged to the gz topic of the same name, where the world's
// JointPositionController PIDs the joint toward it.
//
// Why a node rather than something in the SDF: SDF has no way to script a
// pose over time. A joint plus a controller needs no new Gazebo plugin, at
// the cost of something having to publish the setpoint.
//
// Why a position controller and not the JointTrajectoryController the arm
// uses: that plugin's semantics are "replace the current trajectory", and
// driving a continuous oscillation means republishing constantly. Each
// replacement restarted the motion against a stale start time, so the joint
// snapped to a point most of a cycle away -- measured at 0.196 m between two
// samples 25 ms apart, on a joint limited to 0.5 m/s. It also broke the
// detector's tracker into four tracks for three objects. One scalar setpoint
// at 20 Hz has none of that machinery to get wrong.
//
// The motion is deliberately periodic and boring. It has to be genuinely
// independent of the agent, not merely hard to predict: a random walk would
// also be non-contingent, but then "curiosity chases it" and "the world is
// noisy" become impossible to tell apart in the results.
#include <chrono>
#include <cmath>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>

namespace
{
constexpr double kPi = 3.14159265358979323846;
}

class DrifterDriver : public rclcpp::Node
{
public:
  DrifterDriver()
  : Node("drifter_driver")
  {
    // Metres either side of centre. The joint's own limit is +-0.12, so keep
    // this inside it or the controller fights the constraint at the turning
    // points.
    amplitude_ = declare_parameter("amplitude", 0.10);
    // Seconds for one full there-and-back. Slow enough that a detector
    // running at about 1 Hz sees smooth motion rather than teleporting, which
    // matters: the tracker associates detections between frames by nearest
    // neighbour inside a 0.10 m gate, and an object that jumps further than
    // that between frames becomes two objects.
    period_ = std::max(declare_parameter("period", 8.0), 0.5);
    // Setpoint rate. Fast relative to the period so the PID chases a smoothly
    // moving target rather than a staircase.
    const double rate = std::max(declare_parameter("rate_hz", 20.0), 1.0);

    pub_ = create_publisher<std_msgs::msg::Float64>("/drifter/position", 10);

    // Phase comes from the clock, not from a counter, so the motion is a
    // function of time alone: it does not care when this node started, it
    // survives a missed tick, and it is reproducible across runs.
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate), [this]() {tick();});

    RCLCPP_INFO(
      get_logger(), "drifter_driver up | +-%.3f m, %.1f s cycle, %.0f Hz setpoints",
      amplitude_, period_, rate);
  }

private:
  void tick()
  {
    const double now = get_clock()->now().nanoseconds() * 1e-9;
    const double phase = std::fmod(now, period_) / period_;
    std_msgs::msg::Float64 msg;
    msg.data = amplitude_ * std::sin(2.0 * kPi * phase);
    pub_->publish(msg);
  }

  double amplitude_ = 0.10;
  double period_ = 8.0;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DrifterDriver>());
  rclcpp::shutdown();
  return 0;
}
