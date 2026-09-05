#!/usr/bin/env python3
"""
grasp_climber -- drives the UR5 by TRANSFER empowerment, and picks the object up.

Subscribes:
    /joint_states                sensor_msgs/JointState
    /camera/depth/image_raw      sensor_msgs/Image

Publishes:
    /joint_trajectory_controller/joint_trajectory   trajectory_msgs/JointTrajectory
    /empowerment/chosen_score    std_msgs/Float64
    /empowerment/held            std_msgs/Float64   1.0 when it believes it holds

Why this exists alongside greedy_climber, rather than replacing it.

`greedy_climber` maximises OWN-SENSOR empowerment: the outcome is the depth
image, and the arm's own body is most of what varies in it. Measured on this
exact model, over all 83 available actions from a pose at the object, closing
the gripper ranks **83rd of 83**. It is the single worst thing that agent can
do, and no amount of tuning changes that -- which is what the project's "it
will not pick the object up" claim is really describing.

This node maximises TRANSFER empowerment: the outcome is the OBJECT's state
alone, and nothing about the arm. Same arm, same physics, same state; only
the definition of the outcome differs. Closing then ranks **1st of 83**.

    from a pose at the object, over all 83 actions
      own-sensor   CLOSE ranks 83/83     hovering wins by 2.66 bits
      transfer     CLOSE ranks  1/83     grasping wins by 1.05 bits

The mechanism, from the bolted control: own-sensor does not dislike holding,
it likes PUSHING. Shoving gives the object semi-independent motion and so
multiplies the variety in the picture, and grasping destroys that
independence by making the object a rigid function of the tool. Bolt the
object so it cannot be pushed and the penalty vanishes (-2.66 -> +0.09).

So the grasp is derived from the objective rather than rewarded. There is no
task bonus here, nothing that says "holding is good", and no goal position.
Set objective:=own_sensor to run the other one and watch it refuse.

What this still does not do: find the object from across the table. The
object's contribution is zero beyond about 20 cm because nothing the arm can
do within the horizon touches it, so there is no gradient to climb. Getting
to the object is plain motion toward something perception already sees, and
`approach` handles it. Empowerment's job here is deciding what to do once
there, which is the part it is actually good at.
"""

from __future__ import annotations

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from intrinsic_core.sensors import CameraConfig, DepthCamera
from intrinsic_core.ur5 import UR5, UR5Config
from intrinsic_core.ur5_grasp import HELD, OBJ, Q, UR5GraspConfig, UR5GraspWorld

from .object_source import DepthObjectSource

UR5_JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
GRIPPER_JOINTS = [
    "robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint",
]


class GraspClimber(Node):

    def __init__(self) -> None:
        super().__init__("grasp_climber")
        self._declare_parameters()
        p = self.get_parameter

        arm = UR5(UR5Config(
            dq=p("joint_step").value,
            active_joints=tuple(p("active_joints").value),
            tool_radius=p("tool_radius").value,
            puck_radius=p("object_radius").value,
            table_height=p("table_height").value,
        ))
        self.world = UR5GraspWorld(arm, UR5GraspConfig(
            grasp_radius=p("grasp_radius").value,
            tool_offset=p("tool_offset").value,
            object_resolution=p("object_resolution").value,
            depth_resolution=p("depth_resolution").value,
        ))

        camera_config = CameraConfig(
            position=tuple(p("camera_position").value),
            look_at=tuple(p("camera_look_at").value),
            up=(0.0, 0.0, 1.0),
            fov_deg=p("camera_fov_deg").value,
            width=p("camera_width").value,
            height=p("camera_height").value,
            table_height=p("table_height").value,
        )
        self.camera = DepthCamera(camera_config)
        self.sensing = ReentrantCallbackGroup()
        self.objects = DepthObjectSource(
            self, camera_config, callback_group=self.sensing,
            plane_tolerance=p("plane_tolerance").value,
            min_pixels=p("min_detection_pixels").value,
            arm_exclusion_radius=p("arm_exclusion_radius").value,
            max_missing=p("track_max_missing").value,
        )

        self.objective = str(p("objective").value)
        self.joint_names = list(p("joint_names").value)
        self.q = np.array(p("initial_joint_state").value, dtype=float)
        self.have_joints = False
        # Measured knuckle angles. The gripper's own sense of touch.
        self.jaw = np.zeros(2)
        # ... and the same reading about half a second earlier, which is how
        # we tell a jaw that has STOPPED on something from one still moving.
        self._jaw_was = np.zeros(2)
        self._jaw_stamp = 0.0
        # What the fingers were last ASKED for, and when. A ramp that gets
        # preempted by the next tick leaves them stopped part of the way
        # there, which looks exactly like a jaw stopped on an object -- and
        # reads as settled, because it really has stopped. Measured: jaws at
        # 0.211 / 0.148, scored as a firm grasp, with the gripper 0.24 m away
        # from the puck. So the verdict also has to wait for the ramp.
        self._finger_target = 0.0
        self._finger_since = 0.0
        self.held = False
        # Where the object was when the jaws closed. The test for whether a
        # grasp took is "is it still there", not "is it at the tool".
        self.grasp_site: np.ndarray | None = None
        self.descending = False
        self.object_xyz = np.array([-0.68, 0.0, self.world.rest_height()])

        # A tick spends 0.4 to 0.6 s inside numpy, and on a single-threaded
        # executor nothing else in this node runs while it does. Measured
        # consequence: the arm reached the commanded pose and sat there for
        # TWELVE SECONDS while `self.q` still held the pose from before the
        # command, so the approach re-issued the same 6 cm step for ever and
        # never got within approach_within of the object. Sensing goes in a
        # reentrant group so it keeps running while the tick thinks.
        self.create_subscription(JointState, "joint_states", self._on_joints, 10,
                                 callback_group=self.sensing)
        self.pub_traj = self.create_publisher(
            JointTrajectory, "joint_trajectory_controller/joint_trajectory", 10)
        self.pub_score = self.create_publisher(
            Float64, "empowerment/chosen_score", 10)
        self.pub_held = self.create_publisher(Float64, "empowerment/held", 10)

        self.period = 1.0 / max(p("rate_hz").value, 1e-3)
        self.create_timer(self.period, self._tick,
                          callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(
            f"grasp_climber up | objective={self.objective}, "
            f"{self.world.n_actions} actions "
            f"({self.world.n_joint_actions} joint + close + open), "
            f"horizon {p('horizon').value}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("joint_step", 0.10)
        self.declare_parameter("active_joints", [0, 1, 2, 3])
        self.declare_parameter("tool_radius", 0.09)
        self.declare_parameter("object_radius", 0.04)
        self.declare_parameter("table_height", -0.25)
        self.declare_parameter("joint_names", UR5_JOINT_ORDER)
        self.declare_parameter("initial_joint_state", [0.0] * 6)

        # "transfer"   outcome is the object's state -> grasping ranks 1/83
        # "own_sensor" outcome is the depth image    -> grasping ranks 83/83
        self.declare_parameter("objective", "transfer")

        self.declare_parameter("horizon", 2)
        # Inner rollouts per candidate. Exhaustive at horizon 2 is 6889 and
        # costs 610 ms per action, far too slow for a control loop. The
        # ranking is unaffected by subsampling -- measured, CLOSE still wins
        # by +1.2 to +1.5 bits at every level down to 120 -- and 500 costs
        # 44 ms, so ~24 candidates fit inside a second.
        self.declare_parameter("n_sequences", 500)
        self.declare_parameter("candidate_actions", 24)
        # The model counts a close as a grasp when the object is within this
        # of the grip point, so it has to be the region the PADS can actually
        # capture, which is (jaw_open - diameter) / 2 = 12.5 mm for a 60 mm
        # object in an 85 mm jaw.
        #
        # It was 0.10, then 0.05, and both were still far more permissive
        # than the hardware: the model scored CLOSE as productive from 5 cm
        # away, empowerment duly chose it, and the jaws shut on air beside
        # the puck. The measured jaw angle now catches that every time
        # (see _grip_engaged), but catching it is not the same as not doing
        # it, and the fix is for the model's grasp to mean what the gripper's
        # grasp means.
        self.declare_parameter("grasp_radius", 0.015)
        # How far the gripper must have carried the object away from where
        # it closed before "is it still on the table" becomes a fair
        # question. Deliberately NOT derived from grasp_radius: that is a
        # capture tolerance of millimetres, and this is a travel distance.
        self.declare_parameter("carry_gate", 0.10)
        # robotiq_85_tcp is 0.090 m along the tool axis from tool0. Without
        # this the node aims the FLANGE at the object and drives the gripper
        # through the table; observed as the approach stalling at 10 cm and
        # the close never taking hold.
        self.declare_parameter("tool_offset", 0.09)
        self.declare_parameter("object_resolution", 0.02)
        self.declare_parameter("depth_resolution", 0.02)

        self.declare_parameter("camera_position", [-1.45, 0.0, 0.35])
        self.declare_parameter("camera_look_at", [-0.68, 0.0, -0.21])
        self.declare_parameter("camera_fov_deg", 50.0)
        self.declare_parameter("camera_width", 16)
        self.declare_parameter("camera_height", 12)

        self.declare_parameter("plane_tolerance", 0.015)
        self.declare_parameter("min_detection_pixels", 40)
        self.declare_parameter("arm_exclusion_radius", 0.15)
        self.declare_parameter("track_max_missing", 5)

        self.declare_parameter("rate_hz", 1.0)
        # The knuckle angle to command when holding is DERIVED from the
        # object's size, not set by hand.
        #
        # urdf/robotiq_2f85.xacro gives a jaw gap, between the faces that
        # actually make contact, of
        #     gap(theta) = jaw_open - jaw_travel * sin(theta)
        # so the angle that closes onto an object of diameter d with a small
        # squeeze is  asin((jaw_open - d + squeeze) / jaw_travel).
        #
        # Hand-setting it has now gone wrong twice in opposite directions: 0.8
        # rad is a 6 mm gap, which crushes rather than grips, and 0.08 rad is
        # a 76 mm gap, which never touches a 60 mm object at all. Both look
        # identical from outside -- the gripper moves, and nothing is held.
        #
        # jaw_travel is 2 * (prox_len + dist_len) = 0.14, the lever to the
        # pad's LOWER EDGE, not 2 * 0.055 = 0.11 to the pad's centre. The
        # fingers are tilted by theta, so the lower inner corner leads and
        # touches first; measuring the gap at the centre understates how far
        # the pads have really come in.
        #
        # That error had teeth. With 0.11 the node commanded 0.286 rad calling
        # it a 6 mm squeeze, when at the contact point it was 14 mm of
        # interference into a rigid cylinder. Gazebo held the puck for ~15 s
        # while the knuckle integrator wound up against a target it could
        # never reach -- the joints stalled at 0.18 rad, exactly where the
        # pad-edge gap equals the puck's 60 mm -- and then spat the puck out
        # at several m/s. It was found 45 m away, after which the arm cycled
        # forever between the static block and the drifter, neither liftable,
        # warning that it believed it held something it did not.
        #
        # grip_squeeze is 2 mm because these are rigid bodies. The real 2F-85
        # has compliant pads and a slipping four-bar; nothing here does, so
        # the interference IS the grip force, and 6 mm of it is a launch.
        # How far PAST first contact to drive the jaws, and how big a
        # shortfall counts as having caught something.
        #
        # The jaws are commanded well beyond the angle that just touches the
        # object, because the object itself is the mechanical stop: it is how
        # a real 2F-85 is driven, and it turns the jaw angle into a tactile
        # sensor. Measured on this model, commanding 0.30 rad:
        #
        #     empty            jaws reach 0.295 and 0.295, sum 0.590
        #     on the puck      jaws stall at 0.279 and 0.087, sum 0.366
        #
        # so the SUM of the two knuckles is short by 0.22 rad when something
        # is between the pads. The sum is what to watch, not either finger:
        # an off-centre object stops one finger early and lets the other run
        # most of the way, which is exactly the 0.279 / 0.087 split above.
        #
        # At the merely-touching angle of 0.194 the same deficit is 0.015
        # rad, which is not enough to tell a grasp from an empty close.
        #
        # /gripper/contacts would be the obvious signal and the bridge
        # carries it, but those sensors have never published a message, not
        # even through a 45 s load-bearing lift. This works and they do not.
        self.declare_parameter("grip_overdrive", 0.11)
        # Seconds to travel from the current jaw angle to the commanded one.
        self.declare_parameter("grip_ramp", 1.5)
        self.declare_parameter("grip_engage_margin", 0.10)
        self.declare_parameter("jaw_open", 0.085)
        self.declare_parameter("jaw_travel", 0.14)
        self.declare_parameter("grip_squeeze", 0.002)

        # Distance at which the node stops approaching and starts letting
        # empowerment choose. Inside the object's contribution is large;
        # beyond about 0.20 m it is exactly zero, so there is nothing there
        # for any gradient method to work with.
        # Hand over to empowerment once the pads are actually around the
        # object, which is not the same as the tool centre coinciding with
        # the object centre. The pads are 30 mm long and the object is 80 mm
        # tall, so a TCP within about 3.5 cm of the object's centre already
        # has them straddling it. 0.015 was too strict -- the arm settles
        # around 3 cm and never satisfied it, so the grasp was never offered.
        # Just outside grasp_radius, so empowerment inherits a pose from
        # which closing is physically possible and only has to decide
        # whether to. Handing over at 3.5 cm left it to fine-position into
        # a 12 mm window using 0.10 rad joint steps that move the tool
        # several centimetres, which it cannot do.
        self.declare_parameter("approach_within", 0.02)
        # Big enough to reach the goal in ONE command, not a stream of small
        # ones. The trajectory controller droops: it will not execute a move
        # whose joint swing is below about 0.05 rad, so a 2 cm Cartesian step
        # (0.047 rad) produced no motion at all and the approach deadlocked
        # 1.6 cm from the pre-grasp pose, re-issuing the same dead command
        # every tick. The scripted grasp that does work commands each pose
        # outright, and max_joint_swing is still there to reject a wild
        # reconfiguration.
        self.declare_parameter("approach_step", 0.25)
        # The descent gets its own, much smaller step. Each Cartesian step is
        # executed as a joint-space interpolation, so the tool travels an ARC
        # between the two IK solutions rather than the straight line the step
        # was planned as, and the bulge grows with the step. At 0.06 the
        # descent swept the puck 4.7 cm sideways on the way down and the jaws
        # then closed on empty air beside it: measured, the jaws reached
        # 0.1975 rad against a free-air 0.189, i.e. nothing between them.
        #
        # It cannot go much below this. The trajectory controller droops by
        # roughly 0.04 rad on the wrists, so a Cartesian step whose joint
        # swing is smaller than that produces no motion at all: at 0.02 the
        # descent froze 10 cm above the object and re-issued the same step
        # for the rest of the run. 0.04 is above the droop and still halves
        # the sideways sweep.
        self.declare_parameter("descend_step", 0.20)
        # Height above the object to travel to before descending onto it.
        self.declare_parameter("pregrasp_clearance", 0.14)
        # Horizontal distance within which the arm is considered "above" the
        # object and may start descending.
        #
        # This must be comfortably INSIDE the finger clearance, which is
        # (jaw_open - object_diameter)/2 = 12.5 mm for a 60 mm object. At the
        # old 25 mm the arm would begin descending while off-centre by more
        # than the fingers could accommodate, and they landed on top of the
        # object instead of either side of it -- the descent then stalled with
        # the fingertips resting on the puck, about 5 cm short.
        self.declare_parameter("pregrasp_tolerance", 0.010)
        # Largest joint-space move accepted for one Cartesian approach step.
        self.declare_parameter("max_joint_swing", 1.2)
        # Direction the gripper should POINT while approaching. Straight down
        # at the table. Without this the IK is position-only and picks the
        # orientation itself -- it chose 33 degrees off vertical, which put
        # the finger pads 51 mm from the object while every position number
        # read as a perfect hit.
        self.declare_parameter("approach_axis", [0.0, 0.0, -1.0])

    # -- inputs --------------------------------------------------------------

    def _on_joints(self, msg: JointState) -> None:
        lookup = dict(zip(msg.name, msg.position))
        if all(n in lookup for n in self.joint_names):
            self.q = np.array([lookup[n] for n in self.joint_names])
            self.have_joints = True
        if all(n in lookup for n in GRIPPER_JOINTS):
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._jaw_stamp > 0.5:
                self._jaw_was = self.jaw.copy()
                self._jaw_stamp = now
            self.jaw = np.array([lookup[n] for n in GRIPPER_JOINTS])

    def _update_object(self) -> bool:
        """Object pose from the camera. No ground truth anywhere."""
        if not self.objects.ready:
            return False
        tool = self.world.grip_point(self.q)
        tracks = self.objects.update(tool_xy=tool[:2])
        if not tracks:
            return False

        nearest = min(tracks, key=lambda t: float(np.linalg.norm(t.xy - tool[:2])))
        table = self.get_parameter("table_height").value
        # `height` is the standoff of the object's TOP above the support
        # plane, and the object is resting ON that plane, so its centre is
        # half of it. Subtracting one RADIUS instead is only the same thing
        # for an object as tall as it is wide: the puck is 80 mm tall and
        # 60 mm across, so that put the centre 10 mm high and the pads that
        # much nearer its top edge than intended. The camera measures the
        # height directly; the radius is a parameter the agent was told.
        centre_z = table + 0.5 * float(nearest.height)

        # Correct the belief from perception rather than trusting the model.
        # Gazebo has real contact physics and the analytic model does not, so
        # the model's "held" has to be falsifiable from the camera.
        #
        # It must NOT be falsified by asking whether the object is at the
        # tool. The tracker discards every detection within
        # arm_exclusion_radius (0.15 m) of the tool, which is exactly where a
        # held object is, so that question can only ever answer "no": the
        # nearest track becomes some other object 20 cm away and the belief
        # was being dropped a few seconds after every successful grasp. The
        # arm then flung the jaws open with the puck still between them,
        # which is how a real lift ended with the puck sliding across the
        # floor.
        #
        # The answerable question is whether the object is still lying where
        # we closed on it. If it is, the grasp missed; if that spot is empty,
        # the object left the table with the gripper.
        #
        # And that question only has an answer once the gripper has MOVED
        # AWAY from the site. In the second after closing the object is of
        # course still there, so testing immediately falsified every grasp
        # within 0.3 s; the jaws then chattered open and shut on the puck and
        # threw it off the bench.
        if self.held:
            if self.grasp_site is None:
                self.grasp_site = self.object_xyz.copy()
            gate = float(self.get_parameter("carry_gate").value)
            carried = float(np.linalg.norm(tool - self.grasp_site))
            left_behind = min(
                (float(np.linalg.norm(t.xy - self.grasp_site[:2])) for t in tracks),
                default=float("inf"))
            if carried > gate and left_behind <= gate:
                self.get_logger().warn(
                    "believed held, but the object is still on the table where "
                    "the jaws closed; releasing that belief",
                    throttle_duration_sec=5.0)
                self.held = False
                self.grasp_site = None
                self.object_xyz = np.array([nearest.xy[0], nearest.xy[1], centre_z])
            else:
                # Held: the object is wherever the pads are.
                self.object_xyz = tool.copy()
            return True

        self.object_xyz = np.array([nearest.xy[0], nearest.xy[1], centre_z])
        return True

    # -- control -------------------------------------------------------------

    def _score(self, state: np.ndarray) -> float:
        outcome = "object" if self.objective == "transfer" else "depth"
        return self.world.empowerment(
            state,
            horizon=self.get_parameter("horizon").value,
            outcome=outcome,
            camera=self.camera,
            n_sequences=self.get_parameter("n_sequences").value,
        )

    def _tick(self) -> None:
        if not self.have_joints:
            self.get_logger().info(
                "waiting for /joint_states ...", throttle_duration_sec=5.0)
            return
        if not self._update_object():
            self.get_logger().info(
                "waiting for the object ...", throttle_duration_sec=5.0)
            return

        tool = self.world.grip_point(self.q)
        reach = float(np.linalg.norm(tool - self.object_xyz))
        if not self.held and reach > self.get_parameter("approach_within").value:
            self._approach(tool, reach)
            return

        state = np.concatenate([self.q, self.object_xyz, [1.0 if self.held else 0.0]])

        n = min(self.get_parameter("candidate_actions").value,
                self.world.n_joint_actions)
        rng = np.random.default_rng()
        candidates = list(rng.choice(self.world.n_joint_actions, size=n,
                                     replace=False))
        # The gripper commands are always evaluated. Leaving them to a random
        # subset would make "does it choose to grasp" a matter of luck.
        candidates += [self.world.close_action, self.world.open_action]

        best_action, best_score = None, -np.inf
        for a in candidates:
            score = self._score(self.world.step(state, int(a)))
            if score > best_score:
                best_action, best_score = int(a), score

        nxt = self.world.step(state, best_action)
        was_held = self.held
        self.held = bool(nxt[HELD])
        # A close is believed for exactly one tick, which is what the jaws
        # need to travel, and then it has to be true. Without this the model
        # closed on empty air next to a puck it had nudged aside on the way
        # down and then sat believing it for two hundred seconds.
        if (self.held and was_held and self._jaws_arrived()
                and self._jaws_settled() and not self._grip_engaged()):
            self.get_logger().warn(
                f"closed on nothing: jaws at {np.round(self.jaw, 3)} for a "
                f"commanded {self._grip_command():.3f}; reopening",
                throttle_duration_sec=5.0)
            self.held = False
            self.grasp_site = None
        if self.held:
            self.descending = False
        if self.held != was_held:
            self.grasp_site = self.object_xyz.copy() if self.held else None
            self.get_logger().info(
                "CLOSED on the object" if self.held else "released")

        self.pub_score.publish(Float64(data=float(best_score)))
        self.pub_held.publish(Float64(data=1.0 if self.held else 0.0))
        self._send(nxt[Q])

    def _approach(self, tool: np.ndarray, reach: float) -> None:
        """Move toward the object. Plain motion, NOT a decision.

        Two phases, and the second one is not optional. Driving the gripper
        straight at the object's centre sweeps it sideways through the puck
        and shoves it off the bench -- observed exactly that: the puck rose
        16 mm on contact and then ended up on the floor. So: travel to a
        pre-grasp pose directly ABOVE the object at a safe height, and only
        then descend onto it. That is how you approach anything you intend to
        pick up, and it costs nothing here.

        This deliberately does not use empowerment, because empowerment cannot
        do it. The object's contribution is exactly zero beyond about 20 cm --
        nothing the arm can do within the horizon touches it, so every
        imagined future leaves it where it is and a movable object scores
        identically to a bolted one. There is no gradient to climb.

        Nothing is smuggled in. The target comes from the agent's own camera,
        not from ground truth, and "move toward the thing you can see" is a
        generic exploration policy rather than task supervision: no
        demonstration, no reward, and nothing saying which object matters or
        what to do on arrival. Deciding THAT is empowerment's job, and it
        starts once the arm is close enough for the question to have an
        answer.
        """
        clearance = float(self.get_parameter("pregrasp_clearance").value)

        planar = float(np.linalg.norm(tool[:2] - self.object_xyz[:2]))
        above = self.object_xyz + np.array([0.0, 0.0, clearance])
        tolerance = float(self.get_parameter("pregrasp_tolerance").value)

        # Hysteresis on the phase switch. Without it the arm descends, drifts
        # a millimetre past the tolerance, abandons the descent, climbs all
        # the way back to pre-grasp height, and comes down again -- observed
        # cycling 2 cm -> 11 cm -> 2 cm and never closing. Once committed to a
        # descent it takes a much larger error to give it up.
        if self.descending:
            self.descending = planar <= 3.0 * tolerance
        else:
            self.descending = planar <= tolerance

        if self.descending:
            goal, phase = self.object_xyz, "descending"
            step = float(self.get_parameter("descend_step").value)
        else:
            goal, phase = above, "to pre-grasp"       # travel high, clear of it
            step = float(self.get_parameter("approach_step").value)

        delta = goal - tool
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            return
        # `tool` is the grip point; step toward the goal, then convert back to
        # the flange pose the IK solves for. Both use the SAME approach axis
        # the IK is asked to achieve, so the two cannot disagree about where
        # the pads end up.
        axis = np.array(self.get_parameter("approach_axis").value, dtype=float)
        axis = axis / np.linalg.norm(axis)
        # Try the full step first, then shorter ones. A long step is what
        # gets the arm moving at all -- the controller ignores a move whose
        # joint swing is under about 0.05 rad -- but the further the target,
        # the likelier the IK is to answer with a different arm posture that
        # the swing guard below then rejects. Rejecting and returning left
        # the arm holding position for the rest of a run: 25 cm asked for,
        # 1.63 rad of reconfiguration offered, nothing done, every tick.
        # So shorten the step and ask again rather than give up.
        # max_joint_swing rejects a solution that is not near the current
        # configuration: the IK can return a completely different arm posture
        # that happens to put the tool in the same place, and commanding it
        # swings the arm through a large reconfiguration on the way, which is
        # how a 6 cm step once turned into the tool jumping from 12 cm away
        # to 26 cm away between ticks. It still has to leave room for ONE
        # genuinely large move: the arm spawns with the tool 33 degrees off
        # vertical, and turning it to face the table is 0.78 rad however
        # small the Cartesian step.
        limit = float(self.get_parameter("max_joint_swing").value)
        q = None
        for scale in (1.0, 0.5, 0.25):
            reach_step = min(step * scale, distance)
            grip_target = tool + delta / distance * reach_step
            target = grip_target - self.world.config.tool_offset * axis
            candidate = self.world.arm.inverse_kinematics(
                target, q_seed=self.q, tool_axis=axis)
            if candidate is None:
                continue
            if float(np.linalg.norm(candidate - self.q)) <= limit:
                q, step = candidate, reach_step
                break
        if q is None:
            self.get_logger().warn(
                f"no usable IK toward {np.round(goal, 3)} at any step; "
                "holding position", throttle_duration_sec=5.0)
            return

        self.get_logger().info(
            f"{phase}: {distance * 100:.0f} cm to go (object {reach * 100:.0f} cm) "
            f"| grip {np.round(tool, 3)} goal {np.round(goal, 3)} "
            f"planar {planar * 100:.1f} swing {np.linalg.norm(q - self.q):.3f} "
            f"| q {np.round(self.q, 2)} -> {np.round(q, 2)} "
            f"| axis {np.round(self.world.arm.forward_kinematics(self.q)[:3, 2], 2)}",
            throttle_duration_sec=3.0)
        self._send(q)

    def _grip_command(self) -> float:
        """What to actually command: past contact, so the object is the stop."""
        return float(min(self._closing_angle()
                         + self.get_parameter("grip_overdrive").value, 0.8))

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _jaws_arrived(self) -> bool:
        """Have the fingers had a full ramp to reach what they were asked for?"""
        return (self._now() - self._finger_since
                > float(self.get_parameter("grip_ramp").value) + 0.5)

    def _jaws_settled(self) -> bool:
        """Have the fingers stopped moving?

        Without this the angle is read mid-travel and a finger merely on its
        way to 0.30 is indistinguishable from one stopped on an object.
        Measured after the puck had been dropped and the jaws were closing on
        nothing: consecutive samples read 0.289 / 0.067, which scores as a
        firm grasp, then 0.298 / 0.297, which is free air. Both were the same
        empty gripper.
        """
        return float(np.max(np.abs(self.jaw - self._jaw_was))) < 0.01

    def _grip_engaged(self) -> bool:
        """Is anything between the pads?

        Purely proprioceptive: the jaws were told to go somewhere and did
        not get there. Nothing about WHERE the object is, or that there is
        an object at all, enters the decision to close -- that stays with
        empowerment. This only decides whether to believe the close worked.
        """
        commanded = self._grip_command()
        deficit = 2.0 * commanded - float(np.sum(self.jaw))
        return deficit > self.get_parameter("grip_engage_margin").value

    def _closing_angle(self) -> float:
        """Knuckle angle that just grips an object of the configured size."""
        p = self.get_parameter
        diameter = 2.0 * float(p("object_radius").value)
        wanted = float(p("jaw_open").value) - diameter + float(p("grip_squeeze").value)
        ratio = np.clip(wanted / float(p("jaw_travel").value), 0.0, 1.0)
        return float(np.arcsin(ratio))

    def _send(self, q_target: np.ndarray) -> None:
        """Arm and gripper in one trajectory.

        Both knuckle joints are commanded to the same angle: the 2F-85's
        four-bar linkage is approximated by two independent revolute joints
        in urdf/robotiq_2f85.xacro, and gz-sim has no mimic constraint, so
        the trajectory controller drives them together instead.
        """
        finger = self._grip_command() if self.held else 0.0

        # Published in the order the controller declares its joints in
        # urdf/ur5_gz.urdf.xacro, which interleaves the knuckles BEFORE
        # wrist_3:
        #     pan, lift, elbow, wrist_1, wrist_2, knuckle_l, knuckle_r, wrist_3
        # Sending arm-then-gripper instead puts wrist_3's angle where the
        # controller expects the left knuckle, and a finger angle where it
        # expects wrist_3. greedy_climber never hit this because it publishes
        # the six arm joints and nothing else.
        order = self.joint_names[:5] + GRIPPER_JOINTS + self.joint_names[5:6]

        # Long enough for the move to actually finish. This was period * 0.9,
        # i.e. 0.9 s, sized for the 1 Hz tick the node asks for -- but a tick
        # costs 3.5 s of empowerment, and the arm only travels about
        # 0.15 rad/s. A 0.34 rad descent therefore got 0.12 rad done before
        # the trajectory ended and the controller held where it had got to:
        # the gripper stopped 8 cm above the puck and closed on air above it.
        # Capped below the tick so a new trajectory does not preempt this one.
        swing = float(np.linalg.norm(np.asarray(q_target) - self.q))
        seconds = float(np.clip(swing / 0.15, 0.6, 3.0))

        # Sent as a RAMP, not as one point.
        #
        # A single point puts the whole 0.30 rad of finger travel in front of
        # a p=1200 controller, and the jaws cross it in well under a second.
        # When the pads are not quite around the object that is not a grasp,
        # it is a swipe: measured, a close moved the puck 10.5 cm in the same
        # second the jaws went from 0.03 to 0.31 rad, and two attempts later
        # the puck was on the floor. Each failed attempt left the scene worse
        # than it found it, so a run could not recover from one bad approach.
        #
        # Ramping over grip_ramp changes nothing about where the fingers end
        # up or how hard they squeeze once stopped -- the final angle and the
        # force clamp are untouched, so _grip_engaged reads exactly the same
        # stall -- it only stops them arriving like a bat.
        if abs(finger - self._finger_target) > 1e-9:
            self._finger_target = finger
            self._finger_since = self._now()
        start = float(np.mean(self.jaw))
        ramp = float(self.get_parameter("grip_ramp").value)
        moving_fingers = abs(finger - start) > 0.02
        total = max(seconds, ramp) if moving_fingers else seconds
        n = 6 if moving_fingers else 1

        traj = JointTrajectory()
        traj.joint_names = order
        for k in range(1, n + 1):
            a = k / n
            q_k = self.q + (np.asarray(q_target) - self.q) * a
            by_name = dict(zip(self.joint_names, (float(v) for v in q_k)))
            f_k = start + (finger - start) * a
            by_name[GRIPPER_JOINTS[0]] = f_k
            by_name[GRIPPER_JOINTS[1]] = f_k
            point = JointTrajectoryPoint()
            point.positions = [by_name[n_] for n_ in order]
            point.velocities = [0.0] * len(point.positions)
            t = total * a
            point.time_from_start = Duration(
                sec=int(t), nanosec=int((t % 1.0) * 1e9))
            traj.points.append(point)
        self.pub_traj.publish(traj)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspClimber()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor(num_threads=3))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
