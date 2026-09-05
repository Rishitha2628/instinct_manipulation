// grasp_climber -- drives the UR5 by TRANSFER empowerment, and picks the
// object up.
//
// Subscribes:
//     joint_states                 sensor_msgs/JointState
//     camera/depth/image_raw       sensor_msgs/Image
//
// Publishes:
//     joint_trajectory_controller/joint_trajectory   trajectory_msgs/JointTrajectory
//     empowerment/chosen_score     std_msgs/Float64
//     empowerment/held             std_msgs/Float64   1.0 when it believes it holds
//
// Why this exists alongside greedy_climber.
//
// greedy_climber maximises OWN-SENSOR empowerment: the outcome is the depth
// image, and the arm's own body is most of what varies in it. Measured on
// this exact model, over all 83 available actions from a pose at the object,
// closing the gripper ranks 83rd of 83. It is the single worst thing that
// agent can do, and no amount of tuning changes that.
//
// This node maximises TRANSFER empowerment: the outcome is the OBJECT's state
// alone, and nothing about the arm. Same arm, same physics, same state; only
// the definition of the outcome differs. Closing then ranks 1st of 83.
//
//     from a pose at the object, over all 83 actions
//       own-sensor   CLOSE ranks 83/83     hovering wins by 2.66 bits
//       transfer     CLOSE ranks  1/83     grasping wins by 1.05 bits
//
// The mechanism, from the bolted control: own-sensor does not dislike
// holding, it likes PUSHING. Shoving gives the object semi-independent motion
// and so multiplies the variety in the picture, and grasping destroys that
// independence by making the object a rigid function of the tool. Bolt the
// object so it cannot be pushed and the penalty vanishes (-2.66 -> +0.09).
//
// So the grasp is derived from the objective rather than rewarded. There is
// no task bonus here, nothing that says "holding is good", and no goal
// position. Set objective:=own_sensor to run the other one and watch it
// refuse.
//
// What this still does not do: find the object from across the table. The
// object's contribution is zero beyond about 20 cm because nothing the arm
// can do within the horizon touches it, so there is no gradient to climb.
// Getting to the object is plain motion toward something perception already
// sees, and approach() handles it. Empowerment's job here is deciding what to
// do once there, which is the part it is actually good at.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "intrinsic_core/sensors.hpp"
#include "intrinsic_core/ur5.hpp"
#include "intrinsic_core/ur5_grasp.hpp"
#include "intrinsic_motivation_ros/object_source.hpp"

using namespace intrinsic_core;             // NOLINT(build/namespaces)
using intrinsic_motivation_ros::DepthObjectSource;

namespace
{
const std::vector<std::string> UR5_JOINT_ORDER = {
  "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
const std::vector<std::string> GRIPPER_JOINTS = {
  "robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint"};
}  // namespace

class GraspClimber : public rclcpp::Node
{
public:
  GraspClimber()
  : Node("grasp_climber")
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
    arm_cfg.table_height = get_parameter("table_height").as_double();
    const UR5 arm(arm_cfg);

    UR5GraspConfig world_cfg;
    world_cfg.grasp_radius = get_parameter("grasp_radius").as_double();
    world_cfg.tool_offset = get_parameter("tool_offset").as_double();
    world_cfg.object_resolution = get_parameter("object_resolution").as_double();
    world_cfg.depth_resolution = get_parameter("depth_resolution").as_double();
    world_ = std::make_shared<UR5GraspWorld>(arm, world_cfg);

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

    // A tick spends most of a second inside the estimator, and on a
    // single-threaded executor nothing else in this node runs while it does.
    // Measured consequence: the arm reached the commanded pose and sat there
    // for TWELVE SECONDS while q_ still held the pose from before the
    // command, so the approach re-issued the same 6 cm step for ever and
    // never got within approach_within of the object. Sensing goes in a
    // reentrant group so it keeps running while the tick thinks.
    sensing_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    objects_ = std::make_shared<DepthObjectSource>(
      this, cam,
      get_parameter("plane_tolerance").as_double(),
      static_cast<int>(get_parameter("min_detection_pixels").as_int()),
      get_parameter("arm_exclusion_radius").as_double(),
      static_cast<int>(get_parameter("track_max_missing").as_int()),
      "camera/depth/image_raw", sensing_);

    objective_ = get_parameter("objective").as_string();
    joint_names_ = get_parameter("joint_names").as_string_array();
    const auto q0 = get_parameter("initial_joint_state").as_double_array();
    q_ = Eigen::VectorXd(6);
    for (int i = 0; i < 6; ++i) {q_[i] = q0[static_cast<size_t>(i)];}
    object_xyz_ = Eigen::Vector3d(-0.68, 0.0, world_->rest_height());

    rclcpp::SubscriptionOptions options;
    options.callback_group = sensing_;
    sub_joints_ = create_subscription<sensor_msgs::msg::JointState>(
      "joint_states", 10,
      [this](const sensor_msgs::msg::JointState::SharedPtr m) {on_joints(m);},
      options);

    pub_traj_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "joint_trajectory_controller/joint_trajectory", 10);
    pub_score_ =
      create_publisher<std_msgs::msg::Float64>("empowerment/chosen_score", 10);
    pub_held_ = create_publisher<std_msgs::msg::Float64>("empowerment/held", 10);

    period_ = 1.0 / std::max(get_parameter("rate_hz").as_double(), 1e-3);
    // The group has to be held as a member. The node keeps only a weak
    // reference to the groups it hands out, so a group created inline dies
    // immediately and the executor never discovers the timer attached to it:
    // the node comes up, reports itself ready, and then simply never ticks.
    control_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    timer_ = create_wall_timer(
      std::chrono::duration<double>(period_), [this]() {tick();}, control_);

    RCLCPP_INFO(
      get_logger(),
      "grasp_climber up | objective=%s, %d actions (%d joint + close + open), "
      "horizon %ld",
      objective_.c_str(), world_->n_actions(), world_->n_joint_actions(),
      get_parameter("horizon").as_int());
  }

private:
  // -- parameters -----------------------------------------------------------

  void declare_all()
  {
    declare_parameter("joint_step", 0.10);
    declare_parameter("active_joints", std::vector<int64_t>{0, 1, 2, 3});
    declare_parameter("tool_radius", 0.09);
    declare_parameter("object_radius", 0.04);
    declare_parameter("table_height", -0.25);
    declare_parameter("joint_names", UR5_JOINT_ORDER);
    declare_parameter("initial_joint_state", std::vector<double>(6, 0.0));

    // "transfer" scores the object's state alone; "own_sensor" scores the
    // depth image, and refuses to grasp.
    declare_parameter("objective", std::string("transfer"));

    declare_parameter("horizon", 2);
    // Inner rollouts per candidate. Exhaustive at horizon 2 is 6889 and costs
    // 610 ms per action, far too slow for a control loop. The ranking is
    // unaffected by subsampling -- measured, CLOSE still wins by +1.2 to +1.5
    // bits at every level down to 120.
    declare_parameter("n_sequences", 500);
    declare_parameter("candidate_actions", 24);

    // The model counts a close as a grasp when the object is within this of
    // the grip point, so it has to be the region the PADS can actually
    // capture, which is (jaw_open - diameter) / 2 = 12.5 mm for a 60 mm
    // object in an 85 mm jaw.
    //
    // It was 0.10, then 0.05, and both were still far more permissive than
    // the hardware: the model scored CLOSE as productive from 5 cm away,
    // empowerment duly chose it, and the jaws shut on air beside the puck.
    declare_parameter("grasp_radius", 0.015);
    // How far the gripper must have carried the object away from where it
    // closed before "is it still on the table" becomes a fair question.
    // Deliberately NOT derived from grasp_radius: that is a capture tolerance
    // of millimetres, and this is a travel distance.
    declare_parameter("carry_gate", 0.10);
    // robotiq_85_tcp is 0.090 m along the tool axis from tool0. Without this
    // the node aims the FLANGE at the object and drives the gripper through
    // the table.
    declare_parameter("tool_offset", 0.09);
    declare_parameter("object_resolution", 0.02);
    declare_parameter("depth_resolution", 0.02);

    declare_parameter("camera_position", std::vector<double>{-1.45, 0.0, 0.35});
    declare_parameter("camera_look_at", std::vector<double>{-0.68, 0.0, -0.21});
    declare_parameter("camera_fov_deg", 50.0);
    declare_parameter("camera_width", 16);
    declare_parameter("camera_height", 12);

    declare_parameter("plane_tolerance", 0.015);
    declare_parameter("min_detection_pixels", 40);
    declare_parameter("arm_exclusion_radius", 0.15);
    declare_parameter("track_max_missing", 5);

    declare_parameter("rate_hz", 1.0);

    // How far PAST first contact to drive the jaws, and how big a shortfall
    // counts as having caught something.
    //
    // The jaws are commanded well beyond the angle that just touches the
    // object, because the object itself is the mechanical stop: it is how a
    // real 2F-85 is driven, and it turns the jaw angle into a tactile sensor.
    // Measured on this model, commanding 0.30 rad:
    //
    //     empty            jaws reach 0.295 and 0.295, sum 0.590
    //     on the puck      jaws stall at 0.279 and 0.087, sum 0.366
    //
    // so the SUM of the two knuckles is short by 0.22 rad when something is
    // between the pads. The sum is what to watch, not either finger: an
    // off-centre object stops one finger early and lets the other run most of
    // the way, which is exactly the 0.279 / 0.087 split above.
    //
    // /gripper/contacts would be the obvious signal and the bridge carries
    // it, but those sensors have never published a message, not even through
    // a 45 s load-bearing lift. This works and they do not.
    declare_parameter("grip_overdrive", 0.11);
    // Seconds to travel from the current jaw angle to the commanded one.
    declare_parameter("grip_ramp", 1.5);
    declare_parameter("grip_engage_margin", 0.10);

    // The knuckle angle to command when holding is DERIVED from the object's
    // size, not set by hand. urdf/robotiq_2f85.xacro gives a jaw gap, between
    // the faces that actually make contact, of
    //     gap(theta) = jaw_open - jaw_travel * sin(theta)
    //
    // jaw_travel is 2 * (prox_len + dist_len) = 0.14, the lever to the pad's
    // LOWER EDGE, not 2 * 0.055 = 0.11 to the pad's centre. The fingers are
    // tilted by theta, so the lower inner corner leads and touches first.
    //
    // That error had teeth. With 0.11 the node commanded 0.286 rad calling it
    // a 6 mm squeeze, when at the contact point it was 14 mm of interference
    // into a rigid cylinder. Gazebo held the puck for ~15 s while the knuckle
    // integrator wound up, then spat it out at several m/s -- found 45 m
    // away.
    declare_parameter("jaw_open", 0.085);
    declare_parameter("jaw_travel", 0.14);
    // 2 mm because these are rigid bodies. The real 2F-85 has compliant pads
    // and a slipping four-bar; nothing here does, so the interference IS the
    // grip force, and 6 mm of it is a launch.
    declare_parameter("grip_squeeze", 0.002);

    // Just outside grasp_radius, so empowerment inherits a pose from which
    // closing is physically possible and only has to decide whether to.
    // Handing over at 3.5 cm left it to fine-position into a 12 mm window
    // using 0.10 rad joint steps that move the tool several centimetres,
    // which it cannot do.
    declare_parameter("approach_within", 0.02);

    // Big enough to reach the goal in ONE command, not a stream of small
    // ones. The trajectory controller droops: it will not execute a move
    // whose joint swing is below about 0.05 rad, so a 2 cm Cartesian step
    // (0.047 rad) produced no motion at all and the approach deadlocked
    // 1.6 cm from the pre-grasp pose, re-issuing the same dead command every
    // tick.
    declare_parameter("approach_step", 0.25);
    declare_parameter("descend_step", 0.20);

    // Height above the object to travel to before descending onto it.
    declare_parameter("pregrasp_clearance", 0.14);
    // Horizontal distance within which the arm is considered "above" the
    // object and may start descending. This must be comfortably INSIDE the
    // finger clearance, which is 12.5 mm for a 60 mm object.
    declare_parameter("pregrasp_tolerance", 0.010);
    declare_parameter("max_joint_swing", 1.2);
    // Direction the gripper should POINT while approaching. Straight down at
    // the table. Without this the IK is position-only and picks the
    // orientation itself -- it chose 33 degrees off vertical, which put the
    // finger pads 51 mm from the object while every position number read as a
    // perfect hit.
    declare_parameter("approach_axis", std::vector<double>{0.0, 0.0, -1.0});
  }

  // -- inputs ---------------------------------------------------------------

  double now_seconds() {return get_clock()->now().nanoseconds() * 1e-9;}

  void on_joints(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    Eigen::VectorXd q(6);
    bool complete = true;
    for (size_t i = 0; i < joint_names_.size() && i < 6; ++i) {
      const auto it = std::find(msg->name.begin(), msg->name.end(), joint_names_[i]);
      if (it == msg->name.end()) {complete = false; break;}
      q[static_cast<Eigen::Index>(i)] =
        msg->position[static_cast<size_t>(std::distance(msg->name.begin(), it))];
    }
    if (complete) {q_ = q; have_joints_ = true;}

    Eigen::Vector2d jaw;
    bool have_jaw = true;
    for (size_t i = 0; i < GRIPPER_JOINTS.size(); ++i) {
      const auto it =
        std::find(msg->name.begin(), msg->name.end(), GRIPPER_JOINTS[i]);
      if (it == msg->name.end()) {have_jaw = false; break;}
      jaw[static_cast<Eigen::Index>(i)] =
        msg->position[static_cast<size_t>(std::distance(msg->name.begin(), it))];
    }
    if (have_jaw) {
      const double now = now_seconds();
      if (now - jaw_stamp_ > 0.5) {jaw_was_ = jaw_; jaw_stamp_ = now;}
      jaw_ = jaw;
    }
  }

  /// Object pose from the camera. No ground truth anywhere.
  bool update_object()
  {
    if (!objects_->ready()) {return false;}
    const Eigen::Vector3d tool = world_->grip_point(q_);
    const Eigen::Vector2d tool_xy = tool.head<2>();
    const auto & tracks = objects_->update(&tool_xy);
    if (tracks.empty()) {return false;}

    const Track * nearest = &tracks.front();
    double best = (nearest->xy - tool_xy).norm();
    for (const auto & t : tracks) {
      const double d = (t.xy - tool_xy).norm();
      if (d < best) {best = d; nearest = &t;}
    }

    const double table = get_parameter("table_height").as_double();
    // `height` is the standoff of the object's TOP above the support plane,
    // and the object is resting ON that plane, so its centre is half of it.
    // Subtracting one RADIUS instead is only the same thing for an object as
    // tall as it is wide: the puck is 80 mm tall and 60 mm across, so that
    // put the centre 10 mm high.
    const double centre_z = table + 0.5 * nearest->height;

    // Correct the belief from perception rather than trusting the model.
    //
    // It must NOT be falsified by asking whether the object is at the tool.
    // The tracker discards every detection within arm_exclusion_radius
    // (0.15 m) of the tool, which is exactly where a held object is, so that
    // question can only ever answer "no" and the belief was being dropped a
    // few seconds after every successful grasp.
    //
    // The answerable question is whether the object is still lying where we
    // closed on it -- and that only has an answer once the gripper has MOVED
    // AWAY from the site.
    if (held_) {
      if (!grasp_site_.has_value()) {grasp_site_ = object_xyz_;}
      const double gate = get_parameter("carry_gate").as_double();
      const double carried = (tool - *grasp_site_).norm();
      double left_behind = std::numeric_limits<double>::infinity();
      for (const auto & t : tracks) {
        left_behind = std::min(left_behind, (t.xy - grasp_site_->head<2>()).norm());
      }
      if (carried > gate && left_behind <= gate) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "believed held, but the object is still on the table where the jaws "
          "closed; releasing that belief");
        held_ = false;
        grasp_site_.reset();
        object_xyz_ = Eigen::Vector3d(nearest->xy[0], nearest->xy[1], centre_z);
      } else {
        object_xyz_ = tool;      // held: the object is where the pads are
      }
      return true;
    }

    object_xyz_ = Eigen::Vector3d(nearest->xy[0], nearest->xy[1], centre_z);
    return true;
  }

  // -- the gripper's sense of touch ------------------------------------------

  /// What to actually command: past contact, so the object is the stop.
  double grip_command() const
  {
    return std::min(
      closing_angle() + get_parameter("grip_overdrive").as_double(), 0.8);
  }

  /// Have the fingers had a full ramp to reach what they were asked for?
  bool jaws_arrived()
  {
    return now_seconds() - finger_since_ >
           get_parameter("grip_ramp").as_double() + 0.5;
  }

  /// Have the fingers stopped moving?
  ///
  /// Without this the angle is read mid-travel and a finger merely on its way
  /// to 0.30 is indistinguishable from one stopped on an object. Measured
  /// after the puck had been dropped and the jaws were closing on nothing:
  /// consecutive samples read 0.289 / 0.067, which scores as a firm grasp,
  /// then 0.298 / 0.297, which is free air. Both were the same empty gripper.
  bool jaws_settled() const
  {
    return (jaw_ - jaw_was_).cwiseAbs().maxCoeff() < 0.01;
  }

  /// Is anything between the pads?
  ///
  /// Purely proprioceptive: the jaws were told to go somewhere and did not
  /// get there. Nothing about WHERE the object is, or that there is an object
  /// at all, enters the decision to close -- that stays with empowerment.
  /// This only decides whether to believe the close worked.
  bool grip_engaged() const
  {
    const double deficit = 2.0 * grip_command() - jaw_.sum();
    return deficit > get_parameter("grip_engage_margin").as_double();
  }

  /// Knuckle angle that just grips an object of the configured size.
  double closing_angle() const
  {
    const double diameter = 2.0 * get_parameter("object_radius").as_double();
    const double wanted = get_parameter("jaw_open").as_double() - diameter +
      get_parameter("grip_squeeze").as_double();
    const double ratio =
      std::clamp(wanted / get_parameter("jaw_travel").as_double(), 0.0, 1.0);
    return std::asin(ratio);
  }

  // -- control ---------------------------------------------------------------

  double score(const Eigen::VectorXd & state) const
  {
    const std::string outcome = (objective_ == "transfer") ? "object" : "depth";
    return world_->empowerment(
      state, static_cast<int>(get_parameter("horizon").as_int()), outcome,
      camera_.get(), static_cast<int>(get_parameter("n_sequences").as_int()));
  }

  void tick()
  {
    if (!have_joints_) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000, "waiting for /joint_states ...");
      return;
    }
    if (!update_object()) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000, "waiting for the object ...");
      return;
    }

    const Eigen::Vector3d tool = world_->grip_point(q_);
    const double reach = (tool - object_xyz_).norm();
    if (!held_ && reach > get_parameter("approach_within").as_double()) {
      approach(tool, reach);
      return;
    }

    Eigen::VectorXd state(10);
    state << q_, object_xyz_, (held_ ? 1.0 : 0.0);

    const int n = std::min<int>(
      static_cast<int>(get_parameter("candidate_actions").as_int()),
      world_->n_joint_actions());
    std::vector<int> pool(static_cast<size_t>(world_->n_joint_actions()));
    std::iota(pool.begin(), pool.end(), 0);
    std::shuffle(pool.begin(), pool.end(), rng_);
    std::vector<int> candidates(pool.begin(), pool.begin() + n);
    // The gripper commands are always evaluated. Leaving them to a random
    // subset would make "does it choose to grasp" a matter of luck.
    candidates.push_back(world_->close_action());
    candidates.push_back(world_->open_action());

    int best_action = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int a : candidates) {
      const double s = score(world_->step(state, a));
      if (s > best_score) {best_score = s; best_action = a;}
    }

    const Eigen::VectorXd next = world_->step(state, best_action);
    const bool was_held = held_;
    held_ = next[GRASP_HELD] != 0.0;

    // A close is believed for exactly one tick, which is what the jaws need
    // to travel, and then it has to be true. Without this the model closed on
    // empty air next to a puck it had nudged aside on the way down and then
    // sat believing it for two hundred seconds.
    if (held_ && was_held && jaws_arrived() && jaws_settled() && !grip_engaged()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "closed on nothing: jaws at [%.3f %.3f] for a commanded %.3f; reopening",
        jaw_[0], jaw_[1], grip_command());
      held_ = false;
      grasp_site_.reset();
    }
    if (held_) {descending_ = false;}
    if (held_ != was_held) {
      grasp_site_ = held_ ? std::optional<Eigen::Vector3d>(object_xyz_) : std::nullopt;
      RCLCPP_INFO(get_logger(), held_ ? "CLOSED on the object" : "released");
    }

    std_msgs::msg::Float64 msg;
    msg.data = best_score;
    pub_score_->publish(msg);
    msg.data = held_ ? 1.0 : 0.0;
    pub_held_->publish(msg);
    send(next.head<6>());
  }

  /// Move toward the object. Plain motion, NOT a decision.
  ///
  /// Two phases, and the second one is not optional. Driving the gripper
  /// straight at the object's centre sweeps it sideways through the puck and
  /// shoves it off the bench. So: travel to a pre-grasp pose directly ABOVE
  /// the object at a safe height, and only then descend onto it.
  ///
  /// This deliberately does not use empowerment, because empowerment cannot
  /// do it. The object's contribution is exactly zero beyond about 20 cm, so
  /// every imagined future leaves it where it is and a movable object scores
  /// identically to a bolted one. There is no gradient to climb.
  ///
  /// Nothing is smuggled in. The target comes from the agent's own camera,
  /// and "move toward the thing you can see" is a generic exploration policy
  /// rather than task supervision: no demonstration, no reward, and nothing
  /// saying which object matters or what to do on arrival.
  void approach(const Eigen::Vector3d & tool, double reach)
  {
    const double clearance = get_parameter("pregrasp_clearance").as_double();
    const double tolerance = get_parameter("pregrasp_tolerance").as_double();

    const double planar = (tool.head<2>() - object_xyz_.head<2>()).norm();
    const Eigen::Vector3d above =
      object_xyz_ + Eigen::Vector3d(0.0, 0.0, clearance);

    // Hysteresis on the phase switch. Without it the arm descends, drifts a
    // millimetre past the tolerance, abandons the descent, climbs all the way
    // back to pre-grasp height, and comes down again -- observed cycling
    // 2 cm -> 11 cm -> 2 cm and never closing.
    descending_ = descending_ ? (planar <= 3.0 * tolerance) : (planar <= tolerance);

    Eigen::Vector3d goal;
    const char * phase = nullptr;
    double step = 0.0;
    if (descending_) {
      goal = object_xyz_;
      phase = "descending";
      step = get_parameter("descend_step").as_double();
    } else {
      goal = above;
      phase = "to pre-grasp";     // travel high, clear of it
      step = get_parameter("approach_step").as_double();
    }

    const Eigen::Vector3d delta = goal - tool;
    const double distance = delta.norm();
    if (distance < 1e-6) {return;}

    const auto axis_param = get_parameter("approach_axis").as_double_array();
    Eigen::Vector3d axis(axis_param[0], axis_param[1], axis_param[2]);
    axis.normalize();

    // max_joint_swing rejects a solution that is not near the current
    // configuration: the IK can return a completely different arm posture
    // that happens to put the tool in the same place, and commanding it
    // swings the arm through a large reconfiguration on the way.
    //
    // Try the full step first, then shorter ones. A long step is what gets
    // the arm moving at all -- the controller ignores a move whose joint
    // swing is under about 0.05 rad -- but the further the target, the
    // likelier the IK is to answer with a different arm posture that the
    // swing guard rejects. Rejecting and returning left the arm holding
    // position for the rest of a run: 25 cm asked for, 1.63 rad of
    // reconfiguration offered, nothing done, every tick.
    const double limit = get_parameter("max_joint_swing").as_double();
    std::optional<Eigen::VectorXd> q_target;
    for (double scale : {1.0, 0.5, 0.25}) {
      const double reach_step = std::min(step * scale, distance);
      const Eigen::Vector3d grip_target = tool + delta / distance * reach_step;
      const Eigen::Vector3d target =
        grip_target - world_->config().tool_offset * axis;
      const auto candidate = world_->arm().inverse_kinematics(
        target, &q_, 1e-3, 12, 0, &axis);
      if (!candidate.has_value()) {continue;}
      if ((*candidate - q_).norm() <= limit) {q_target = candidate; break;}
    }
    if (!q_target.has_value()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "no usable IK toward [%.3f %.3f %.3f] at any step; holding position",
        goal[0], goal[1], goal[2]);
      return;
    }

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 3000,
      "%s: %.0f cm to go (object %.0f cm) | grip [%.3f %.3f %.3f] "
      "goal [%.3f %.3f %.3f] planar %.1f swing %.3f",
      phase, distance * 100.0, reach * 100.0, tool[0], tool[1], tool[2],
      goal[0], goal[1], goal[2], planar * 100.0, (*q_target - q_).norm());
    send(*q_target);
  }

  /// Arm and gripper in one trajectory.
  ///
  /// Both knuckle joints are commanded to the same angle: the 2F-85's
  /// four-bar linkage is approximated by two independent revolute joints in
  /// urdf/robotiq_2f85.xacro, and gz-sim has no mimic constraint, so the
  /// trajectory controller drives them together instead.
  void send(const Eigen::VectorXd & q_target)
  {
    const double finger = held_ ? grip_command() : 0.0;

    // Published in the order the controller declares its joints in
    // urdf/ur5_gz.urdf.xacro, which interleaves the knuckles BEFORE wrist_3:
    //     pan, lift, elbow, wrist_1, wrist_2, knuckle_l, knuckle_r, wrist_3
    // Sending arm-then-gripper instead puts wrist_3's angle where the
    // controller expects the left knuckle.
    std::vector<std::string> order(joint_names_.begin(), joint_names_.begin() + 5);
    order.insert(order.end(), GRIPPER_JOINTS.begin(), GRIPPER_JOINTS.end());
    order.push_back(joint_names_[5]);

    // Long enough for the move to actually finish. This was period * 0.9,
    // i.e. 0.9 s, sized for the 1 Hz tick the node asks for -- but a tick
    // costs 3.5 s of empowerment, and the arm only travels about 0.15 rad/s.
    // A 0.34 rad descent therefore got 0.12 rad done before the trajectory
    // ended and the controller held where it had got to: the gripper stopped
    // 8 cm above the puck and closed on air above it.
    const double swing = (q_target - q_).norm();
    const double seconds = std::clamp(swing / 0.15, 0.6, 3.0);

    // Sent as a RAMP, not as one point.
    //
    // A single point puts the whole 0.30 rad of finger travel in front of a
    // p=1200 controller, and the jaws cross it in well under a second. When
    // the pads are not quite around the object that is not a grasp, it is a
    // swipe: measured, a close moved the puck 10.5 cm in the same second the
    // jaws went from 0.03 to 0.31 rad, and two attempts later the puck was on
    // the floor. Each failed attempt left the scene worse than it found it.
    //
    // Ramping changes nothing about where the fingers end up or how hard they
    // squeeze once stopped -- the final angle and the force clamp are
    // untouched, so grip_engaged reads exactly the same stall -- it only
    // stops them arriving like a bat.
    if (std::abs(finger - finger_target_) > 1e-9) {
      finger_target_ = finger;
      finger_since_ = now_seconds();
    }
    const double start = jaw_.mean();
    const double ramp = get_parameter("grip_ramp").as_double();
    const bool moving_fingers = std::abs(finger - start) > 0.02;
    const double total = moving_fingers ? std::max(seconds, ramp) : seconds;
    const int n = moving_fingers ? 6 : 1;

    trajectory_msgs::msg::JointTrajectory traj;
    traj.joint_names = order;
    for (int k = 1; k <= n; ++k) {
      const double a = static_cast<double>(k) / n;
      const Eigen::VectorXd q_k = q_ + (q_target - q_) * a;
      const double f_k = start + (finger - start) * a;

      trajectory_msgs::msg::JointTrajectoryPoint point;
      for (int i = 0; i < 5; ++i) {point.positions.push_back(q_k[i]);}
      point.positions.push_back(f_k);
      point.positions.push_back(f_k);
      point.positions.push_back(q_k[5]);
      point.velocities.assign(point.positions.size(), 0.0);
      const double t = total * a;
      point.time_from_start.sec = static_cast<int32_t>(t);
      point.time_from_start.nanosec =
        static_cast<uint32_t>(std::fmod(t, 1.0) * 1e9);
      traj.points.push_back(point);
    }
    pub_traj_->publish(traj);
  }

  std::shared_ptr<UR5GraspWorld> world_;
  std::shared_ptr<DepthCamera> camera_;
  std::shared_ptr<DepthObjectSource> objects_;
  rclcpp::CallbackGroup::SharedPtr sensing_;
  rclcpp::CallbackGroup::SharedPtr control_;

  std::string objective_;
  std::vector<std::string> joint_names_;
  Eigen::VectorXd q_;
  bool have_joints_ = false;

  // Measured knuckle angles, and the same reading about half a second
  // earlier, which is how we tell a jaw that has STOPPED on something from
  // one still moving.
  Eigen::Vector2d jaw_{0.0, 0.0};
  Eigen::Vector2d jaw_was_{0.0, 0.0};
  double jaw_stamp_ = 0.0;
  // What the fingers were last ASKED for, and when. A ramp that gets
  // preempted by the next tick leaves them stopped part of the way there,
  // which looks exactly like a jaw stopped on an object -- and reads as
  // settled, because it really has stopped.
  double finger_target_ = 0.0;
  double finger_since_ = 0.0;

  bool held_ = false;
  // Where the object was when the jaws closed. The test for whether a grasp
  // took is "is it still there", not "is it at the tool".
  std::optional<Eigen::Vector3d> grasp_site_;
  bool descending_ = false;
  Eigen::Vector3d object_xyz_{-0.68, 0.0, 0.0};

  double period_ = 1.0;
  std::mt19937_64 rng_{std::random_device{}()};

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joints_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_traj_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_score_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_held_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GraspClimber>();
  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), 3);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
