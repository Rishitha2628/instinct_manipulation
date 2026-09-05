// Finding objects in a depth image, without being told any exist.
//
// Why this file exists: until now the agent read the puck's position off
// /puck/pose, which is the simulator handing it the answer. That is the one
// place the project's central claim leaked -- "the agent is never told the
// object exists, what it is, or what to do with it" was true of the outcome
// half and false of the input half. This closes it.
//
// What replaces the oracle is NOT a learned detector. It is a geometric
// prior, deliberately:
//
//     the dominant plane in view is the support surface,
//     and anything standing off it is a thing.
//
// That is weak and generic in the way an innate prior should be. It knows
// nothing about pucks, cylinders, colour, or count. It does not know how many
// objects there are, and it will happily report a coffee cup. What it will
// not do is tell you which of the things it found is worth touching -- that
// is exactly the question empowerment answers, and keeping the two separate
// is the point of the experiment:
//
//     perception says THERE IS SOMETHING THERE.
//     empowerment says WHETHER IT IS WORTH TOUCHING.
//
// The three objects in worlds/empowerment_table.sdf are identical to this
// module by construction: same cylinder, same size, same height off the
// table. It cannot distinguish the pushable one from the bolted one from the
// one that slides around on its own. Only their response to action separates
// them.
//
// What is still assumed, stated plainly so it is not mistaken for a free
// lunch: a single dominant planar surface exists and most of the view is it;
// objects stand PROUD of that plane, toward the camera; and the arm can be
// excluded, either by a rendered self-mask or by the cruder rule that the arm
// reaches in from outside the frame while the objects do not touch its
// border.
#ifndef INTRINSIC_CORE__PERCEPTION_HPP_
#define INTRINSIC_CORE__PERCEPTION_HPP_

#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "intrinsic_core/sensors.hpp"

namespace intrinsic_core
{

/// Everything the geometric prior needs, and nothing about pucks.
struct SegmenterConfig
{
  /// Metres a point must stand off the fitted plane to count as an object.
  ///
  /// Floor is set by sensor noise: the world's depth camera has a 3 mm
  /// gaussian, and a flat table read through it has a spread of roughly that.
  /// Below about 3x the noise the table itself starts segmenting into
  /// speckle. Above the object height (0.08 m here) nothing is ever found.
  double plane_tolerance = 0.015;

  /// Blobs smaller than this are noise. At the detection resolutions used
  /// here an 8 cm object is tens of pixels, so this is generous.
  int min_pixels = 6;

  /// Blobs larger than this fraction of the frame are not objects. Catches
  /// the case where plane fitting fails and half the image is "above" it.
  double max_pixel_fraction = 0.30;

  /// The arm reaches in from outside the field of view, so it is connected to
  /// the frame edge; the objects sit in the middle of the table and are not.
  /// Crude, free, and works. A rendered self-mask is strictly better and is
  /// supported by passing one to detect().
  bool reject_border = true;
  int border_margin = 1;

  /// Fraction of a blob's height, measured down from its highest point, whose
  /// pixels are averaged to locate the object's axis.
  ///
  /// Using the whole blob biases the estimate toward the camera, because an
  /// oblique view sees the near side of an object and not the far side. The
  /// top face of a flat-topped object is centred on its axis from any angle,
  /// so averaging only the top slice removes most of that bias.
  ///
  /// Measured against a rendered sphere at 96x72, bias toward the camera:
  ///
  ///     1.00 (whole blob)   -2.16 cm
  ///     0.50                -1.93
  ///     0.30                -1.42
  ///     0.15                -0.84
  ///     0.08                -0.59
  ///
  /// It never reaches zero there because this renders the puck as a SPHERE,
  /// whose top cap is only half visible from an angle. The real objects are
  /// flat-topped cylinders: every point of the top face has the same
  /// standoff, so any slice captures the whole disc and its centroid is the
  /// axis exactly.
  double top_slice = 0.15;

  /// Maximum trimming rounds in the plane fit; it exits early once the fit
  /// stops moving. Measured convergence is at round 4 on a real frame.
  int plane_iterations = 8;

  /// Whether a depth value is distance ALONG THE RAY (true) or distance along
  /// the optical axis (false, the usual RGBD convention and what gz-sim's
  /// depth_camera publishes).
  ///
  /// This matters only when the camera is not looking straight down, which is
  /// exactly the case now. Get it wrong with an oblique camera and the
  /// reconstructed table is a plane at the wrong tilt: the fit still
  /// succeeds, objects are still found, and every position is quietly skewed
  /// toward the edges of the frame. This module's own DepthCamera returns
  /// range, so tests against rendered frames pass true and frames off the ROS
  /// topic pass false.
  bool depth_is_range = false;
};

/// One thing standing off the support surface.
struct Detection
{
  /// Object axis in the camera config's frame (agent frame: world x, y).
  Eigen::Vector2d xy{0.0, 0.0};

  /// Metres the highest point of the blob stands off the plane. This is what
  /// makes a lift visible: a grasped and raised object keeps its (x, y) and
  /// changes this.
  double height = 0.0;

  int n_pixels = 0;
  bool touches_border = false;
  double centroid_x = 0.0;
  double centroid_y = 0.0;
};

/// A detection followed across frames, so occlusion is not deletion.
struct Track
{
  Eigen::Vector2d xy{0.0, 0.0};
  double height = 0.0;
  int frames_seen = 1;
  int frames_missing = 0;
  bool visible = true;
  std::vector<double> free_motion;

  /// Metres per frame this thing moves while nothing is near it.
  ///
  /// This is the number that separates the drifter from the puck without
  /// anything being told which is which. A pushable object sitting on a table
  /// has an independent motion of zero: it only ever moves because it was
  /// touched. Something that slides around on its own has a non-zero value
  /// here, learned by watching rather than by being told.
  double independent_motion_std() const;
};

/// Objects from a depth image, via "the big flat thing is the floor".
class DepthSegmenter
{
public:
  explicit DepthSegmenter(
    const CameraConfig & camera, const SegmenterConfig & config = SegmenterConfig());

  /// Depth image -> (h*w, 3) points in the camera config's frame.
  Eigen::MatrixXd back_project(const Eigen::MatrixXd & depth) const;

  /// Robust dominant-plane fit. The normal is oriented toward the camera, so
  /// a point's signed distance is positive when it stands proud of the
  /// surface.
  ///
  /// Least squares with trimming rather than RANSAC: the support surface is
  /// the overwhelming majority of the view here, so a plain fit is already
  /// close and two rounds of discarding the worst residuals converge on the
  /// table without the extra machinery.
  void fit_plane(
    const Eigen::MatrixXd & points, Eigen::Vector3d * normal, double * offset) const;

  /// Every object standing off the dominant plane. `self_mask`, when given,
  /// is true where the agent's own body is.
  std::vector<Detection> detect(
    const Eigen::MatrixXd & depth,
    const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> * self_mask = nullptr) const;

  const SegmenterConfig & config() const {return config_;}

private:
  /// Unit ray directions and the camera origin, at the image's own size.
  const DepthCamera & rays_for(int height, int width) const;

  CameraConfig camera_;
  SegmenterConfig config_;
  mutable std::map<std::pair<int, int>, std::shared_ptr<DepthCamera>> ray_cache_;
};

/// Follows detections between frames, and survives occlusion.
///
/// Object permanence, and it is here for an entirely practical reason: even
/// with the camera moved off vertical, the arm still covers an object
/// sometimes, and a detector alone reports that as the object ceasing to
/// exist. An agent whose world model deletes things when a limb passes in
/// front of them cannot act on them. Holding the last known position across a
/// gap is the cheapest possible fix and it is also, not coincidentally, the
/// thing infants take several months to develop.
class ObjectTracker
{
public:
  ObjectTracker(
    double gate = 0.10,
    int max_missing = 30,
    double arm_exclusion_radius = 0.15,
    double max_speed = 0.05,
    double max_gate = 0.11);

  /// Associate this frame's detections with the existing tracks.
  ///
  /// `dt` is the seconds since the previous update. It widens the association
  /// gate, and it matters more than it looks: the detector does not run at a
  /// fixed rate. Measured live, the node ticked at a median 0.54 Hz with gaps
  /// as long as 14.9 s, and across a gap like that a slowly drifting object
  /// crosses a fixed gate, gets treated as new, and is reported alongside the
  /// stale track it should have matched. Three objects were being counted as
  /// five.
  ///
  /// The widening is capped at half the spacing between objects, because past
  /// that point a gate large enough to follow one object is also large enough
  /// to jump to its neighbour.
  const std::vector<Track> & update(
    const std::vector<Detection> & detections,
    const Eigen::Vector2d * tool_xy = nullptr,
    const double * dt = nullptr);

  const std::vector<Track> & tracks() const {return tracks_;}
  double max_gate() const {return max_gate_;}

private:
  double gate_;
  double max_speed_;
  double max_gate_;
  int max_missing_;
  double arm_exclusion_radius_;
  std::vector<Track> tracks_;
};

}  // namespace intrinsic_core

#endif  // INTRINSIC_CORE__PERCEPTION_HPP_
