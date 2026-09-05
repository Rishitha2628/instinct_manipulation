#!/usr/bin/env python3
"""
UR5 + depth camera empowerment map over a table.

Computes empowerment at every reachable point of a table-height plane under
two conditions, and writes the figure:

    A  puck pushable      -- the agent can move it and the camera can see it
    B  puck bolted down   -- present and visible, but immovable (control)
    C  A - B              -- the puck's own contribution

The outcome the agent is scored on is a DEPTH IMAGE, not ground-truth object
pose. That is the faithful reading of the paper: empowerment is the channel
from actuators to the agent's own sensors, so anything the camera cannot
resolve does not count.

Usage:
    python3 scripts/run_ur5_map.py --grid 18 --sequences 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intrinsic_core.batched import BatchedDepthEmpowerment          # noqa: E402
from intrinsic_core.empowerment import EmpowermentConfig            # noqa: E402
from intrinsic_core.sensors import (                                # noqa: E402
    CameraConfig, DepthCamera, DepthDiscretiser,
)
from intrinsic_core.ur5 import UR5, UR5Config                       # noqa: E402


def build_scene(args):
    arm = UR5(UR5Config(dq=args.dq, active_joints=(0, 1, 2, 3)))
    camera = DepthCamera(CameraConfig(
        position=(args.puck[0], args.puck[1], 0.95),
        look_at=(args.puck[0], args.puck[1], 0.0),
        up=(1.0, 0.0, 0.0),
        fov_deg=30.0,
        width=16, height=12,
    ))
    return arm, camera


def compute_map(arm, camera, disc, cfg, puck, xs, ys, z, ik_cache):
    est = BatchedDepthEmpowerment(arm, camera, disc, cfg)
    out = np.full((len(ys), len(xs)), np.nan)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            q = ik_cache.get((i, j))
            if q is None:
                continue
            out[i, j] = est.estimate(np.concatenate([q, puck]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=18)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--sequences", type=int, default=2000)
    ap.add_argument("--dq", type=float, default=0.10)
    ap.add_argument("--depth-res", type=float, default=0.02)
    ap.add_argument("--tool-height", type=float, default=0.08)
    ap.add_argument("--puck", type=float, nargs=2, default=[-0.50, 0.0])
    ap.add_argument("--out", type=str, default="ur5_empowerment_map.png")
    args = ap.parse_args()

    puck = np.array(args.puck)
    arm, camera = build_scene(args)
    bolted = arm.with_fixed_puck()

    disc = DepthDiscretiser(depth_resolution=args.depth_res)
    cfg = EmpowermentConfig(
        n_steps=args.steps, n_sequences=args.sequences,
        exhaustive_below=0, seed=0,
    )

    half = 0.28
    xs = np.linspace(puck[0] - half, puck[0] + half, args.grid)
    ys = np.linspace(puck[1] - half, puck[1] + half, args.grid)

    # IK is the expensive part and identical for both conditions -- cache it.
    print(f"solving IK for {args.grid ** 2} points ...")
    t0 = time.time()
    ik_cache, seed_q = {}, None
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            q = arm.inverse_kinematics((x, y, args.tool_height), q_seed=seed_q)
            if q is not None:
                ik_cache[(i, j)] = q
                seed_q = q
    print(f"  {len(ik_cache)}/{args.grid ** 2} reachable  ({time.time() - t0:.0f}s)")

    print("computing empowerment (pushable) ...")
    map_push = compute_map(arm, camera, disc, cfg, puck, xs, ys, args.tool_height, ik_cache)
    print(f"  done ({time.time() - t0:.0f}s)")

    print("computing empowerment (bolted control) ...")
    map_bolt = compute_map(bolted, camera, disc, cfg, puck, xs, ys, args.tool_height, ik_cache)
    print(f"  done ({time.time() - t0:.0f}s)")

    delta = map_push - map_bolt
    np.savez(Path(args.out).with_suffix(".npz"),
             xs=xs, ys=ys, pushable=map_push, bolted=map_bolt,
             delta=delta, puck=puck)

    _render(xs, ys, map_push, map_bolt, delta, puck, args)

    finite = delta[np.isfinite(delta)]
    print(f"\nwrote {args.out}")
    print(f"puck contribution: max {finite.max():+.2f} bits, "
          f"mean {finite.mean():+.2f} bits")
    return 0


def _render(xs, ys, map_push, map_bolt, delta, puck, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    vmin = np.nanmin([np.nanmin(map_push), np.nanmin(map_bolt)])
    vmax = np.nanmax([np.nanmax(map_push), np.nanmax(map_bolt)])

    for ax, (data, title, cmap) in zip(axes, [
        (map_push, "A. Puck pushable", "viridis"),
        (map_bolt, "B. Puck bolted down (control)", "viridis"),
        (delta, "C. A - B: the puck's contribution", "magma"),
    ]):
        kw = dict(origin="lower", extent=extent, cmap=cmap, aspect="equal")
        if cmap == "viridis":
            kw.update(vmin=vmin, vmax=vmax)
        im = ax.imshow(data, **kw)
        ax.scatter(*puck, s=130, facecolors="none",
                   edgecolors="white", linewidths=2.2, zorder=5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="bits")

    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        f"UR5 {args.steps}-step empowerment through a depth camera "
        f"(circle = puck; never given to the agent as a goal)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
