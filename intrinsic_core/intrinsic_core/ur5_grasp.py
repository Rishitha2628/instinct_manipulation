"""
The grasp experiment, on the real UR5 kinematics.

`grasp.py` asked whether an intrinsic objective prefers holding an object to
hovering beside it, using a free-flying tool so that the objective could be
tested without an arm's kinematics muddying the result. It answered: own-sensor
empowerment says no, transfer empowerment says yes.

This repeats it on the actual UR5 model, because a free-flying tool can move
in any direction equally well and a 6-DOF arm cannot. Near the object the arm
is part-way into its workspace, some directions are cheap and others are close
to a singularity, and the set of futures available while holding something is
not obviously the same shape as the set available while hovering. If the
result survives that, it is about the objective. If it does not, it was about
the toy.

Three things are added to `UR5`, which cannot express a grasp as it stands:

  * the object gets a z, so it can leave the table;
  * an attachment state, in which the object's pose follows the tool;
  * close/open appended to the joint-increment action table.

The action space is the arm's OWN -- joint increments -- not the task-space
macro-actions of grasp.py. That keeps the comparison honest about what the
robot can actually do in one control step, at the cost of a much larger action
set: 81 joint combinations plus two gripper commands is 83, so horizon 2 is
6889 sequences (enumerable exhaustively) and horizon 3 is 571787 (not).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sensors import CameraConfig, DepthCamera
from .ur5 import UR5, UR5Config

__all__ = ["UR5GraspConfig", "UR5GraspWorld"]

# state layout: [q (6), object_xyz (3), held (1)] -> 10
Q = slice(0, 6)
OBJ = slice(6, 9)
HELD = 9


@dataclass
class UR5GraspConfig:
    grasp_radius: float = 0.10
    """How close the tool point must be to the object's centre for a close
    command to take. Generous, because intrinsic_core's UR5 has no gripper --
    its tool point is the flange and tool_radius is already widened to 0.09 to
    stand in for the gripper's bulk."""

    push_efficiency: float = 1.0
    """1.0 pushable, 0.0 the bolted control."""

    tool_offset: float = 0.0
    """Metres below the flange at which the gripper actually grips.

    intrinsic_core's UR5 has no gripper: `tool_position` returns the flange,
    and `tool_radius` is widened to 0.09 to stand in for the gripper's bulk.
    That fudge is fine for scoring how much the arm can disturb a scene, and
    wrong the moment you try to grasp something, because it aims the FLANGE at
    the object and buries the fingers 0.09 m past it -- through the table.
    robotiq_85_tcp sits 0.090 m along the tool axis from tool0, so set this to
    that and the grip point is where the fingers are.

    Applied along the tool's own +Z axis, so it is correct whatever the wrist
    is doing. Pair it with UR5.inverse_kinematics(tool_axis=...) to make that
    orientation something you chose rather than something the solver picked
    for you."""

    max_lift: float = 0.45
    object_resolution: float = 0.02
    depth_resolution: float = 0.02


class UR5GraspWorld:
    """UR5 kinematics, plus an object it can push, hold, lift and drop."""

    def __init__(
        self,
        arm: UR5 | None = None,
        config: UR5GraspConfig | None = None,
    ) -> None:
        self.arm = arm or UR5(UR5Config())
        self.config = config or UR5GraspConfig()
        self.n_joint_actions = self.arm.n_actions
        self.close_action = self.n_joint_actions
        self.open_action = self.n_joint_actions + 1

    @property
    def n_actions(self) -> int:
        return self.n_joint_actions + 2

    def rest_height(self) -> float:
        return self.arm.config.table_height + self.arm.config.puck_radius

    def grip_point(self, q: np.ndarray) -> np.ndarray:
        """Where the gripper actually grips, not where the flange is.

        Along the TOOL's own axis, not straight down. Assuming straight down
        is only correct if the wrist happens to be pointing at the table, and
        with position-only IK it generally is not: measured on this arm the
        solver returned a pose 33 degrees off vertical, which put the real
        pads 51 mm from where a straight-down assumption said they were --
        further than the grasp radius, so the model declared a perfect grasp
        while the fingers closed on nothing.
        """
        T = self.arm.forward_kinematics(q)
        return T[:3, 3] + self.config.tool_offset * T[:3, 2]

    # -- states --------------------------------------------------------------

    def state_at(
        self,
        q: np.ndarray,
        obj_xy: np.ndarray,
        grasped: bool = False,
        lift: float = 0.0,
    ) -> np.ndarray:
        s = np.zeros(10)
        s[Q] = q
        s[OBJ] = [obj_xy[0], obj_xy[1], self.rest_height() + lift]
        s[HELD] = 1.0 if grasped else 0.0
        return s

    # -- dynamics ------------------------------------------------------------

    def step(self, state: np.ndarray, action: int) -> np.ndarray:
        c = self.config
        s = state.copy()

        if action == self.close_action:
            tool = self.grip_point(s[Q])
            if not s[HELD] and np.linalg.norm(tool - s[OBJ]) <= c.grasp_radius:
                s[HELD] = 1.0
            return s

        if action == self.open_action:
            if s[HELD]:
                s[HELD] = 0.0
                s[OBJ] = [s[OBJ][0], s[OBJ][1], self.rest_height()]
            return s

        # A joint move. Reuse the arm's own action table so this is exactly
        # the action set the rest of the project uses.
        q = s[Q] + self.arm._action_table[action]
        s[Q] = q
        tool = self.grip_point(q)

        if s[HELD]:
            # Held: the object's pose is a rigid function of the tool's, which
            # is what removes it as an independent degree of freedom and is
            # the entire reason own-sensor empowerment dislikes grasping.
            s[OBJ] = [tool[0], tool[1],
                      min(tool[2], self.arm.config.table_height + c.max_lift)]
            return s

        if c.push_efficiency > 0.0:
            a = self.arm.config
            delta = tool[:2] - s[OBJ][:2]
            planar = float(np.linalg.norm(delta))
            contact = a.tool_radius + a.puck_radius
            low_enough = tool[2] < a.table_height + contact
            if low_enough and 1e-9 < planar < contact:
                push = c.push_efficiency * (contact - planar) / planar
                s[OBJ] = [s[OBJ][0] - delta[0] * push,
                          s[OBJ][1] - delta[1] * push,
                          self.rest_height()]
        return s

    def rollout(self, state: np.ndarray, actions) -> np.ndarray:
        s = state
        for a in actions:
            s = self.step(s, int(a))
        return s

    def bolted(self) -> "UR5GraspWorld":
        cfg = UR5GraspConfig(**{**self.config.__dict__, "push_efficiency": 0.0})
        return UR5GraspWorld(self.arm, cfg)

    # -- outcomes ------------------------------------------------------------

    def object_key(self, state: np.ndarray) -> tuple:
        """TRANSFER outcome: the object's state, nothing about the arm."""
        return tuple(np.round(state[OBJ] / self.config.object_resolution).astype(np.int32))

    def depth_key(self, state: np.ndarray, camera: DepthCamera) -> tuple:
        """OWN-SENSOR outcome: what the camera sees.

        Not DepthCamera.render, because that pins the object to the table and
        the whole question here is what happens when it is lifted off it.
        """
        a = self.arm.config
        arm_points = self.arm.link_positions(state[Q])
        centres = np.vstack([arm_points, state[OBJ][None, :]])
        radii = np.concatenate([
            np.full(len(arm_points), camera.config.arm_sphere_radius),
            [a.puck_radius],
        ])
        depth = np.minimum(
            camera._hit_plane(a.table_height),
            camera._hit_spheres(centres, radii),
        )
        depth = np.clip(depth, camera.config.z_near, camera.config.z_far)
        return tuple(np.round(depth / self.config.depth_resolution).astype(np.int32))

    # -- empowerment ---------------------------------------------------------

    def empowerment(
        self,
        state: np.ndarray,
        horizon: int = 2,
        outcome: str = "object",
        camera: DepthCamera | None = None,
        n_sequences: int | None = None,
        seed: int = 0,
    ) -> float:
        """log2 of the number of distinguishable outcomes.

        The forward model is deterministic, so channel capacity reduces to
        counting distinct reachable outcomes. `n_sequences=None` enumerates
        exhaustively, which is affordable at horizon 2 (83^2 = 6889) and not
        at horizon 3 (571787).
        """
        total = self.n_actions ** horizon
        if n_sequences is None or n_sequences >= total:
            grids = np.meshgrid(*[np.arange(self.n_actions)] * horizon,
                                indexing="ij")
            seqs = np.stack([g.ravel() for g in grids], axis=-1)
        else:
            rng = np.random.default_rng(seed)
            seqs = rng.integers(0, self.n_actions, size=(n_sequences, horizon))

        if outcome == "object":
            key = self.object_key
        else:
            if camera is None:
                raise ValueError("depth outcome needs a camera")
            key = lambda s: self.depth_key(s, camera)

        return float(np.log2(len({key(self.rollout(state, seq)) for seq in seqs})))
