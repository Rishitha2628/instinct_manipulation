// A depth camera, rendered analytically.
//
// Why this file exists at all: empowerment is defined as the channel
// capacity from the agent's ACTUATORS to its own SENSORS (Salge et al. sec
// 4.3). Using ground-truth object pose as the outcome quietly skips the
// sensor half and measures something easier. This renders what a depth
// camera would actually see, so the outcome vector is a sensor reading.
//
// That distinction has teeth. With ground-truth pose, the puck's
// contribution is always visible. Through a camera it is not: if the arm
// occludes the puck, or the puck moves less than one depth quantum, the
// sensor cannot resolve the difference and those outcomes collapse into one.
// Empowerment then correctly reports that the agent has no distinguishable
// influence -- which is the paper's point about perception and action being
// two halves of one quantity.
//
// Rendering is ray-cast against analytic primitives: a table plane, a sphere
// for the puck, spheres along the arm links.
#ifndef INTRINSIC_CORE__SENSORS_HPP_
#define INTRINSIC_CORE__SENSORS_HPP_

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "intrinsic_core/discretise.hpp"

namespace intrinsic_core
{

struct CameraConfig
{
  Eigen::Vector3d position{0.35, -0.75, 0.85};
  Eigen::Vector3d look_at{-0.45, 0.0, 0.05};
  Eigen::Vector3d up{0.0, 0.0, 1.0};

  /// Deliberately tiny. A 640x480 depth image has 307200 dimensions and every
  /// rollout differs in some pixel, so nothing ever collapses and empowerment
  /// saturates. Downsampling IS the sensor model: it says what the agent can
  /// actually resolve.
  int width = 32;
  int height = 24;

  /// HORIZONTAL field of view, matching SDF's <horizontal_fov>, ROS
  /// CameraInfo, and how every depth-camera datasheet quotes the number.
  ///
  /// This used to be read as the VERTICAL fov, which meant the analytic
  /// camera and the Gazebo one never had the same optics even when the two
  /// config files agreed on "30". At 4:3 the model was actually seeing 38.6
  /// deg horizontally against the simulator's 30. Nothing errored; the agent
  /// simply predicted through a lens it did not have.
  double fov_deg = 58.0;

  double z_near = 0.1;
  double z_far = 2.5;

  double table_height = 0.0;
  double arm_sphere_radius = 0.05;
};

/// Pinhole depth camera over a table scene.
class DepthCamera
{
public:
  explicit DepthCamera(const CameraConfig & config = CameraConfig());

  /// Depth image of `height * width` values in row-major order, clipped to
  /// [z_near, z_far].
  ///
  /// `arm_points` is a (k,3) matrix of points along the arm, drawn as
  /// spheres; `puck_xy` is the puck centre on the table.
  Eigen::VectorXd render(
    const Eigen::MatrixXd & arm_points,
    const Eigen::Vector2d & puck_xy,
    double puck_radius,
    double arm_radius = -1.0) const;

  /// Depth image of an arbitrary set of spheres over the table plane.
  /// The grasp world draws its tool and object this way rather than as an
  /// arm plus a puck.
  Eigen::VectorXd render_spheres(
    const Eigen::MatrixXd & centres, const Eigen::VectorXd & radii) const;

  /// Render N scenes at once. `arm_points` holds one (k,3) block per scene.
  std::vector<Eigen::VectorXd> render_batch(
    const std::vector<Eigen::MatrixXd> & arm_points,
    const std::vector<Eigen::Vector2d> & puck_xy,
    double puck_radius,
    double arm_radius = -1.0) const;

  const CameraConfig & config() const {return config_;}
  int n_pixels() const {return config_.width * config_.height;}

  /// The per-pixel unit ray directions, (n_pixels, 3), and the eye point.
  /// The detector back-projects with these rather than with a second copy of
  /// the pinhole maths, so the detector and the renderer cannot disagree
  /// about where a pixel points -- drift this project has been bitten by
  /// once already, over the field of view.
  const Eigen::MatrixXd & rays() const {return rays_;}
  const Eigen::Vector3d & origin() const {return origin_;}

private:
  void build_rays();

  /// Distance to the horizontal plane at height z, infinite where no hit.
  Eigen::VectorXd hit_plane(double z) const;

  CameraConfig config_;
  Eigen::MatrixXd rays_;      // (n_pixels, 3), unit direction per pixel
  Eigen::Vector3d origin_;
  Eigen::VectorXd table_;     // cached hit_plane(table_height)
};

/// Quantise a depth image into a hashable outcome key.
///
/// `depth_resolution` is the sensor's real depth accuracy -- roughly 5 mm for
/// a RealSense D435 at half a metre, though it degrades quadratically with
/// range. Setting it far below the true noise floor is the mistake that makes
/// every map saturate: you are then counting sensor noise as distinguishable
/// influence.
Discretiser depth_discretiser(double depth_resolution = 0.01, double ignore_beyond = 2.0);

}  // namespace intrinsic_core

#endif  // INTRINSIC_CORE__SENSORS_HPP_
