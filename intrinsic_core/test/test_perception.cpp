// Locating objects from a depth image, with nobody saying where they are.
//
// These run against frames rendered by this project's own DepthCamera, not
// against Gazebo, which is the point: the detector is testable in
// milliseconds without a simulator, and a failure here is a geometry bug
// rather than a bringup problem.
//
// The camera here is the oblique one the world file uses, because the whole
// class of error this guards against -- confusing range along the ray with
// depth along the optical axis, and biasing an object's position toward the
// camera -- is invisible when the camera looks straight down.
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intrinsic_core/perception.hpp"
#include "intrinsic_core/sensors.hpp"

using namespace intrinsic_core;  // NOLINT(build/namespaces)

namespace
{

constexpr double TABLE = -0.25;
constexpr double PUCK_R = 0.04;

// Matches config/empowerment.yaml and the <pose> in the world file: 36 deg
// below horizontal, looking back along +x toward the arm.
const Eigen::Vector3d CAM_POS(-1.45, 0.0, 0.35);
const Eigen::Vector3d CAM_LOOK(-0.68, 0.0, -0.21);
constexpr double FOV = 50.0;

CameraConfig camera(int width = 96, int height = 72)
{
  CameraConfig c;
  c.position = CAM_POS;
  c.look_at = CAM_LOOK;
  c.width = width;
  c.height = height;
  c.fov_deg = FOV;
  c.table_height = TABLE;
  return c;
}

struct Sphere
{
  Eigen::Vector3d centre;
  double radius;
};

Sphere object_at(double x, double y, double radius = PUCK_R, double lift = 0.0)
{
  return {Eigen::Vector3d(x, y, TABLE + radius + lift), radius};
}

/// Depth image of an arbitrary set of spheres over the table.
Eigen::MatrixXd render(const DepthCamera & dc, const std::vector<Sphere> & spheres)
{
  Eigen::MatrixXd centres(static_cast<Eigen::Index>(spheres.size()), 3);
  Eigen::VectorXd radii(static_cast<Eigen::Index>(spheres.size()));
  for (size_t i = 0; i < spheres.size(); ++i) {
    centres.row(static_cast<Eigen::Index>(i)) = spheres[i].centre.transpose();
    radii[static_cast<Eigen::Index>(i)] = spheres[i].radius;
  }
  const Eigen::VectorXd flat = spheres.empty()
    ? dc.render_spheres(Eigen::MatrixXd(0, 3), Eigen::VectorXd(0))
    : dc.render_spheres(centres, radii);

  const CameraConfig & c = dc.config();
  Eigen::MatrixXd img(c.height, c.width);
  for (int y = 0; y < c.height; ++y) {
    for (int x = 0; x < c.width; ++x) {img(y, x) = flat[y * c.width + x];}
  }
  return img;
}

DepthSegmenter segmenter(const CameraConfig & cam, bool reject_border = true)
{
  SegmenterConfig cfg;
  cfg.depth_is_range = true;
  cfg.reject_border = reject_border;
  return DepthSegmenter(cam, cfg);
}

/// Spheres from the object out to the arm's base, so it leaves the frame.
std::vector<Sphere> arm_chain()
{
  std::vector<Sphere> out;
  for (int i = 0; i < 14; ++i) {
    const double x = -0.62 + (0.30 + 0.62) * i / 13.0;
    out.push_back({Eigen::Vector3d(x, 0.0, TABLE + 0.06), 0.05});
  }
  return out;
}

Detection fake_detection(double x, double y)
{
  Detection d;
  d.xy = Eigen::Vector2d(x, y);
  d.height = 0.08;
  d.n_pixels = 40;
  d.touches_border = false;
  return d;
}

}  // namespace

// --- plane -----------------------------------------------------------------

/// The support surface is found, not assumed.
TEST(PlaneFit, RecoversTheTable)
{
  const CameraConfig cam = camera();
  const Eigen::MatrixXd img = render(DepthCamera(cam), {object_at(-0.68, 0.0)});
  const DepthSegmenter seg = segmenter(cam);

  Eigen::Vector3d normal;
  double offset = 0.0;
  seg.fit_plane(seg.back_project(img), &normal, &offset);

  EXPECT_NEAR(std::abs(normal[0]), 0.0, 1e-3);
  EXPECT_NEAR(std::abs(normal[1]), 0.0, 1e-3);
  EXPECT_NEAR(std::abs(normal[2]), 1.0, 1e-3);
  EXPECT_NEAR(offset, TABLE, 1e-3);
}

/// Three objects in view must not tilt the fitted surface.
TEST(PlaneFit, IsNotDraggedByTheObjects)
{
  const CameraConfig cam = camera();
  const Eigen::MatrixXd img = render(
    DepthCamera(cam),
    {object_at(-0.68, 0.0), object_at(-0.68, 0.24), object_at(-0.68, -0.24)});
  const DepthSegmenter seg = segmenter(cam);

  Eigen::Vector3d normal;
  double offset = 0.0;
  seg.fit_plane(seg.back_project(img), &normal, &offset);

  EXPECT_NEAR(std::abs(normal[2]), 1.0, 1e-3);
  EXPECT_NEAR(offset, TABLE, 1e-3);
}

// --- locating --------------------------------------------------------------

TEST(Detect, SingleObjectLocatedToACentimetre)
{
  const CameraConfig cam = camera();
  const Eigen::Vector2d truth(-0.68, 0.0);
  const Eigen::MatrixXd img = render(DepthCamera(cam), {object_at(truth[0], truth[1])});

  const auto dets = segmenter(cam).detect(img);
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_LT((dets[0].xy - truth).norm(), 0.02);
}

/// Perception must find all three and NOT be able to rank them.
///
/// This is the experiment's premise: the pushable puck, the bolted block and
/// the drifter are the same object as far as any depth image is concerned. If
/// the detector could tell them apart, empowerment would not be doing the work
/// the project claims it does.
TEST(Detect, AllThreeObjectsFoundAndAreIndistinguishable)
{
  const CameraConfig cam = camera();
  const std::vector<Eigen::Vector2d> truths = {
    {-0.68, 0.0}, {-0.68, 0.24}, {-0.68, -0.24}};
  std::vector<Sphere> scene;
  for (const auto & t : truths) {scene.push_back(object_at(t[0], t[1]));}

  const auto dets = segmenter(cam).detect(render(DepthCamera(cam), scene));
  ASSERT_EQ(dets.size(), 3u);

  for (const auto & truth : truths) {
    double best = 1e9;
    for (const auto & d : dets) {best = std::min(best, (d.xy - truth).norm());}
    EXPECT_LT(best, 0.02);
  }

  // Same size and same standoff, to within the sampling grid.
  double h_min = 1e9, h_max = -1e9;
  int p_min = 1 << 30, p_max = 0;
  for (const auto & d : dets) {
    h_min = std::min(h_min, d.height);
    h_max = std::max(h_max, d.height);
    p_min = std::min(p_min, d.n_pixels);
    p_max = std::max(p_max, d.n_pixels);
  }
  EXPECT_LT(h_max - h_min, 0.01);
  EXPECT_LE(p_max, 2 * p_min);
}

/// An oblique camera can see that something was picked up.
///
/// From straight overhead this test could not exist: raising the object
/// changes its depth slightly and nothing else. Here the standoff from the
/// table plane tracks the lift directly, which is what any later grasping
/// work has to be able to measure.
TEST(Detect, LiftIsVisibleAsHeight)
{
  const CameraConfig cam = camera();
  const DepthCamera dc(cam);
  const auto flat = segmenter(cam).detect(render(dc, {object_at(-0.68, 0.0)}));
  const auto high =
    segmenter(cam).detect(render(dc, {object_at(-0.68, 0.0, PUCK_R, 0.10)}));
  ASSERT_FALSE(flat.empty());
  ASSERT_FALSE(high.empty());

  EXPECT_NEAR(high[0].height - flat[0].height, 0.10, 0.02);
  // and it did not slide sideways while being lifted
  EXPECT_LT((high[0].xy - flat[0].xy).norm(), 0.02);
}

// --- the arm ---------------------------------------------------------------

/// The arm reaches in from outside; the objects do not.
TEST(Detect, ArmIsRejectedByTheBorderRule)
{
  const CameraConfig cam = camera();
  std::vector<Sphere> scene = {object_at(-0.68, 0.24)};
  for (const auto & s : arm_chain()) {scene.push_back(s);}
  const Eigen::MatrixXd img = render(DepthCamera(cam), scene);

  const auto kept = segmenter(cam).detect(img);
  ASSERT_EQ(kept.size(), 1u);
  EXPECT_LT((kept[0].xy - Eigen::Vector2d(-0.68, 0.24)).norm(), 0.03);

  // With the rule off, the arm comes back as an extra blob touching the edge.
  const auto everything = segmenter(cam, false).detect(img);
  EXPECT_GT(everything.size(), kept.size());
  bool any_border = false;
  for (const auto & d : everything) {any_border = any_border || d.touches_border;}
  EXPECT_TRUE(any_border);
}

/// The principled alternative: predict your own body and subtract it.
TEST(Detect, SelfMaskExcludesTheArm)
{
  const CameraConfig cam = camera();
  const DepthCamera dc(cam);
  const std::vector<Sphere> arm = arm_chain();

  std::vector<Sphere> scene = {object_at(-0.68, 0.24)};
  for (const auto & s : arm) {scene.push_back(s);}

  const Eigen::MatrixXd img = render(dc, scene);
  const Eigen::MatrixXd arm_only = render(dc, arm);
  const Eigen::MatrixXd table = render(dc, {});

  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> mask(cam.height, cam.width);
  for (int y = 0; y < cam.height; ++y) {
    for (int x = 0; x < cam.width; ++x) {
      mask(y, x) = std::abs(arm_only(y, x) - table(y, x)) > 1e-6;
    }
  }

  const auto dets = segmenter(cam, false).detect(img, &mask);
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_LT((dets[0].xy - Eigen::Vector2d(-0.68, 0.24)).norm(), 0.03);
}

// --- depth convention ------------------------------------------------------

/// z-depth in, same answer out.
///
/// gz-sim publishes distance along the optical axis; DepthCamera returns
/// distance along the ray. With this oblique camera cos(theta) runs from 0.87
/// to 1.0 across the frame, so the two differ by up to 30 cm at the corners.
TEST(DepthConvention, AlongAxisDepthIsConvertedToRange)
{
  const CameraConfig cam = camera();
  const DepthCamera dc(cam);
  const Eigen::MatrixXd ranges = render(dc, {object_at(-0.68, 0.0)});

  const Eigen::Vector3d forward = (CAM_LOOK - CAM_POS).normalized();
  Eigen::MatrixXd z_depth(cam.height, cam.width);
  double biggest = 0.0;
  for (int y = 0; y < cam.height; ++y) {
    for (int x = 0; x < cam.width; ++x) {
      const double cos_theta = dc.rays().row(y * cam.width + x).dot(forward.transpose());
      z_depth(y, x) = ranges(y, x) * cos_theta;
      biggest = std::max(biggest, std::abs(z_depth(y, x) - ranges(y, x)));
    }
  }
  EXPECT_GT(biggest, 0.01) << "conversion must be non-trivial";

  SegmenterConfig as_z_cfg;
  as_z_cfg.depth_is_range = false;
  const auto as_z = DepthSegmenter(cam, as_z_cfg).detect(z_depth);
  const auto as_range = segmenter(cam).detect(ranges);
  ASSERT_EQ(as_z.size(), 1u);
  ASSERT_EQ(as_range.size(), 1u);
  EXPECT_LT((as_z[0].xy - as_range[0].xy).norm(), 5e-3);
}

/// Guards the failure mode above, and pins down what it actually breaks.
///
/// The intuition that the reconstructed table comes out tilted is wrong: it
/// stays almost level and the object's (x, y) is barely affected. What goes is
/// the SCALE along the view direction, which shrinks an 8 cm object to 4.9 cm.
///
/// That is the nastier failure. Position still looks right, so a detector
/// smoke test passes, while the one quantity that tells you an object has been
/// picked up is silently 38 percent low.
TEST(DepthConvention, WrongConventionCorruptsHeightNotPosition)
{
  const CameraConfig cam = camera();
  const DepthCamera dc(cam);
  const Eigen::MatrixXd ranges = render(dc, {object_at(-0.68, 0.0)});

  const Eigen::Vector3d forward = (CAM_LOOK - CAM_POS).normalized();
  Eigen::MatrixXd z_depth(cam.height, cam.width);
  for (int y = 0; y < cam.height; ++y) {
    for (int x = 0; x < cam.width; ++x) {
      z_depth(y, x) = ranges(y, x) *
        dc.rays().row(y * cam.width + x).dot(forward.transpose());
    }
  }

  const auto right = segmenter(cam).detect(ranges);
  const auto wrong = segmenter(cam).detect(z_depth);   // claims range, gets z
  ASSERT_FALSE(right.empty());
  ASSERT_FALSE(wrong.empty());

  EXPECT_NEAR(right[0].height, 2 * PUCK_R, 0.005);
  EXPECT_LT(wrong[0].height, 0.7 * right[0].height);
  // position survives, which is exactly why this is easy to miss
  EXPECT_LT((wrong[0].xy - right[0].xy).norm(), 0.01);

  const DepthSegmenter seg = segmenter(cam);
  Eigen::Vector3d normal;
  double offset = 0.0;
  seg.fit_plane(seg.back_project(z_depth), &normal, &offset);
  EXPECT_GT(std::abs(offset - TABLE), 0.02);
}

// --- tracking --------------------------------------------------------------

/// Object permanence: a limb passing in front is not the object ceasing.
TEST(Tracking, TrackSurvivesOcclusion)
{
  ObjectTracker tracker;
  const CameraConfig cam = camera();
  const auto dets = segmenter(cam).detect(render(DepthCamera(cam), {object_at(-0.68, 0.0)}));

  tracker.update(dets);
  ASSERT_EQ(tracker.tracks().size(), 1u);
  const Eigen::Vector2d remembered = tracker.tracks()[0].xy;

  for (int i = 0; i < 5; ++i) {tracker.update({});}   // hidden by the arm

  ASSERT_EQ(tracker.tracks().size(), 1u);
  EXPECT_FALSE(tracker.tracks()[0].visible);
  EXPECT_LT((tracker.tracks()[0].xy - remembered).norm(), 1e-12);
}

TEST(Tracking, TrackIsDroppedOnceItIsGoneForGood)
{
  ObjectTracker tracker(0.10, 3);
  tracker.update({fake_detection(-0.68, 0.0)});
  ASSERT_EQ(tracker.tracks().size(), 1u);
  for (int i = 0; i < 5; ++i) {tracker.update({});}
  EXPECT_TRUE(tracker.tracks().empty());
}

/// The one measurement that distinguishes them, learned by watching.
///
/// Both objects move. Only one of them moves while nothing is near it, and the
/// agent is never told which. This is what later lets the forward model treat
/// the drifter's outcomes as action-independent, which drives its empowerment
/// to zero while a curiosity signal would be drawn to it.
TEST(Tracking, IndependentMotionSeparatesTheDrifterFromThePuck)
{
  const Eigen::Vector2d far_tool(0.30, 0.30);

  ObjectTracker still;
  for (int i = 0; i < 12; ++i) {
    still.update({fake_detection(-0.68, 0.24)}, &far_tool);
  }

  ObjectTracker drifting;
  for (int i = 0; i < 12; ++i) {
    const double y = -0.24 + 0.10 * std::sin(2 * M_PI * i / 6.0);
    drifting.update({fake_detection(-0.68, y)}, &far_tool);
  }

  EXPECT_NEAR(still.tracks()[0].independent_motion_std(), 0.0, 1e-9);
  EXPECT_GT(drifting.tracks()[0].independent_motion_std(), 0.01);
}

/// A shoved object must not be mistaken for a self-moving one.
TEST(Tracking, PushingIsNotCountedAsIndependentMotion)
{
  ObjectTracker tracker(0.10, 30, 0.15);
  for (int i = 0; i < 12; ++i) {
    const Eigen::Vector2d xy(-0.68 + 0.01 * i, 0.0);
    // tool right on top of it: this motion is the agent's doing
    tracker.update({fake_detection(xy[0], xy[1])}, &xy);
  }
  EXPECT_NEAR(tracker.tracks()[0].independent_motion_std(), 0.0, 1e-9);
}

/// A slow object crossing a long frame gap is still the same object.
///
/// The detector does not run at a fixed rate; measured live it managed a
/// median 0.54 Hz with gaps up to 14.9 s. Over a gap like that a drifting
/// object moves further than a fixed association gate, is treated as new, and
/// gets reported alongside the stale track it should have matched.
TEST(Tracking, GateWidensWithElapsedTime)
{
  const double moved = 0.11;      // further than the 0.10 m base gate

  ObjectTracker fixed;
  fixed.update({fake_detection(-0.68, -0.24)});
  fixed.update({fake_detection(-0.68, -0.24 + moved)});
  EXPECT_EQ(fixed.tracks().size(), 2u) << "fixed gate should split, as it did live";

  ObjectTracker aware;
  const double one = 1.0, five = 5.0;
  aware.update({fake_detection(-0.68, -0.24)}, nullptr, &one);
  aware.update({fake_detection(-0.68, -0.24 + moved)}, nullptr, &five);
  EXPECT_EQ(aware.tracks().size(), 1u);
}

/// The widening must not let a track jump to the object next to it.
TEST(Tracking, WidenedGateNeverReachesANeighbour)
{
  ObjectTracker tracker;
  EXPECT_LT(tracker.max_gate(), 0.24 / 2) << "objects here are 0.24 m apart";

  const double one = 1.0, absurd = 600.0;
  tracker.update({fake_detection(-0.68, -0.24)}, nullptr, &one);
  // its neighbour, 0.24 m away, after an absurd gap
  tracker.update({fake_detection(-0.68, 0.0)}, nullptr, &absurd);
  EXPECT_EQ(tracker.tracks().size(), 2u);
}
