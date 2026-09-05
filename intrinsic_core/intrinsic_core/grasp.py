"""
Can an intrinsic objective produce a GRASP, or does grasping look like a loss?

This is the experiment the rest of the project could not run, because the
model it runs on cannot represent a grasp at all. `ur5.py` has exactly one
object interaction -- a kinematic shove in the table plane -- and the object's
state is `(x, y)`, so "off the table" is literally unrepresentable. No amount
of tuning gets a lift out of that.

Four things are added here, and every one of them is required before the
question can even be asked:

  1. the object gets a z, so it can leave the table;
  2. an ATTACHMENT state, in which the object's pose is a rigid function of
     the tool's, so holding is a thing that exists;
  3. open/close in the action set, so the agent can choose to grasp;
  4. MACRO-ACTIONS in task space rather than joint increments, because a
     grasp is ten-plus primitive steps and 81^h enumeration dies long before
     that. Eight macro-actions at horizon 3 is 512 sequences, which is small
     enough to enumerate EXHAUSTIVELY -- so unlike the main pipeline, these
     numbers are not sampling-limited.

The agent is modelled as a free-floating tool rather than a UR5. That is
deliberate: the question is whether the OBJECTIVE prefers holding, and an
arm's kinematics add a large, pose-dependent background that would have to be
subtracted back out. If holding does not win for a free-flying gripper that
can go anywhere, it certainly will not win once joint limits and singularities
are in the way.

Two objectives are compared, and they are expected to disagree:

  OWN-SENSOR empowerment, the one the project already uses: the outcome is
  the depth image. The catch is that once you are holding something, its
  position is a deterministic function of your own, so it stops contributing
  any independent variety to the picture. Grasping may therefore score no
  better than waving the gripper around in free space, and WORSE than
  shoving, where the object moves semi-independently. That would be a
  measured explanation of the project's "empowerment will not grasp" claim,
  rather than a restatement of it.

  TRANSFER empowerment: the outcome is the OBJECT's state alone. Untouched,
  the object never moves, so there is exactly one outcome and zero bits.
  Pushed, it reaches a handful of spots on the table. Held, it goes anywhere
  the tool can go, in three dimensions. Here grasping should be the global
  maximum -- and derived, rather than rewarded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .sensors import CameraConfig, DepthCamera

__all__ = ["GraspConfig", "GraspWorld", "GraspSensor", "empowerment"]


@dataclass
class GraspConfig:
    table_height: float = -0.25
    object_radius: float = 0.04
    tool_radius: float = 0.03

    step: float = 0.05
    """Metres a single macro-action moves the tool. One macro-action stands
    for a short scripted move, not a joint increment."""

    grasp_radius: float = 0.06
    """How close the tool centre must be to the object's centre for a close
    command to take hold."""

    push_efficiency: float = 1.0
    """1.0 means a shove displaces the object by the full overlap. 0.0 is the
    bolted control."""

    workspace: tuple[float, float] = (-0.95, -0.35)
    """Bounds on tool x, so the tool cannot wander off to infinity and make
    the outcome count meaningless. y and z are bounded in GraspWorld.step."""


# state layout: [tool_xyz (3), object_xyz (3), grasped (1)] -> 7
TOOL = slice(0, 3)
OBJ = slice(3, 6)
HELD = 6

# 6 translations + close + open
N_ACTIONS = 8
_MOVES = np.array([
    [+1, 0, 0], [-1, 0, 0],
    [0, +1, 0], [0, -1, 0],
    [0, 0, +1], [0, 0, -1],
], dtype=float)
CLOSE, OPEN = 6, 7


class GraspWorld:
    """A tool, an object it can push or hold, and a table."""

    def __init__(self, config: GraspConfig | None = None) -> None:
        self.config = config or GraspConfig()

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    def rest_height(self) -> float:
        return self.config.table_height + self.config.object_radius

    def initial_state(
        self,
        tool: np.ndarray,
        obj_xy: np.ndarray,
        grasped: bool = False,
        lift: float = 0.0,
    ) -> np.ndarray:
        s = np.zeros(7)
        s[TOOL] = tool
        s[OBJ] = [obj_xy[0], obj_xy[1], self.rest_height() + lift]
        s[HELD] = 1.0 if grasped else 0.0
        return s

    def step(self, state: np.ndarray, action: int) -> np.ndarray:
        c = self.config
        s = state.copy()
        tool, obj = s[TOOL], s[OBJ]

        if action == CLOSE:
            if not s[HELD] and np.linalg.norm(tool - obj) <= c.grasp_radius:
                s[HELD] = 1.0
            return s

        if action == OPEN:
            if s[HELD]:
                s[HELD] = 0.0
                # dropped: falls straight down to the table
                s[OBJ] = [obj[0], obj[1], self.rest_height()]
            return s

        new_tool = tool + _MOVES[action] * c.step
        new_tool[0] = np.clip(new_tool[0], *c.workspace)
        new_tool[1] = np.clip(new_tool[1], -0.40, 0.40)
        # the tool cannot go through the table
        new_tool[2] = np.clip(new_tool[2], c.table_height + 0.01, c.table_height + 0.45)
        s[TOOL] = new_tool

        if s[HELD]:
            # Held: the object's pose is a rigid function of the tool's. This
            # is the whole point of the attachment state, and it is also
            # exactly why own-sensor empowerment may dislike it -- the object
            # has stopped being an independent degree of freedom.
            s[OBJ] = new_tool
            return s

        # Not held: a shove in the table plane, only while the tool is low
        # enough to catch the object's side.
        if c.push_efficiency > 0.0:
            delta = new_tool[:2] - obj[:2]
            planar = float(np.linalg.norm(delta))
            contact = c.tool_radius + c.object_radius
            low_enough = new_tool[2] < self.rest_height() + c.object_radius
            if low_enough and 1e-9 < planar < contact:
                push = c.push_efficiency * (contact - planar) / planar
                s[OBJ] = [obj[0] - delta[0] * push, obj[1] - delta[1] * push,
                          self.rest_height()]
        return s

    def rollout(self, state: np.ndarray, actions) -> np.ndarray:
        s = state
        for a in actions:
            s = self.step(s, int(a))
        return s

    def bolted(self) -> "GraspWorld":
        """Control: the agent's model says the object cannot be moved."""
        return GraspWorld(replace(self.config, push_efficiency=0.0))


class GraspSensor:
    """Renders a state, and quantises it into an outcome key."""

    def __init__(
        self,
        camera: CameraConfig,
        config: GraspConfig,
        depth_resolution: float = 0.02,
        object_resolution: float = 0.02,
    ) -> None:
        self.camera = DepthCamera(camera)
        self.config = config
        self.depth_resolution = depth_resolution
        self.object_resolution = object_resolution

    def depth_key(self, state: np.ndarray) -> tuple:
        """OWN-SENSOR outcome: what the camera sees."""
        c = self.config
        centres = np.array([state[TOOL], state[OBJ]])
        radii = np.array([c.tool_radius, c.object_radius])
        depth = np.minimum(
            self.camera._hit_plane(c.table_height),
            self.camera._hit_spheres(centres, radii),
        )
        depth = np.clip(depth, self.camera.config.z_near, self.camera.config.z_far)
        return tuple(np.round(depth / self.depth_resolution).astype(np.int32))

    def object_key(self, state: np.ndarray) -> tuple:
        """TRANSFER outcome: the object's state, and nothing about the tool."""
        return tuple(np.round(state[OBJ] / self.object_resolution).astype(np.int32))


def _sequences(n_actions: int, horizon: int) -> np.ndarray:
    grids = np.meshgrid(*[np.arange(n_actions)] * horizon, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=-1)


def empowerment(
    world: GraspWorld,
    sensor: GraspSensor,
    state: np.ndarray,
    horizon: int = 3,
    outcome: str = "depth",
) -> float:
    """log2 of the number of distinguishable outcomes, enumerated exhaustively.

    With a deterministic forward model the channel capacity from actions to
    outcomes reduces to exactly this -- see the note in the project README.
    8 actions at horizon 3 is 512 sequences, so this is the true value rather
    than a sampled lower bound.
    """
    key = sensor.depth_key if outcome == "depth" else sensor.object_key
    seen = {key(world.rollout(state, seq)) for seq in _sequences(world.n_actions, horizon)}
    return float(np.log2(len(seen)))
