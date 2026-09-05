#!/usr/bin/env python3
"""
Reproduce the box experiment (Salge et al. 2013, sec. 4.5.4) on a 2-link arm.

Computes n-step empowerment at every reachable point of the workspace under
three conditions and writes a figure:

    A  pushable puck    -- the puck can be moved and its position is sensed
    B  bolted puck      -- the puck is present but immovable
    C  A minus B        -- the puck's own contribution, with the arm's
                           kinematic structure divided out

Panel C is the result.  Panels A and B on their own are dominated by the
arm's geometry: empowerment is naturally low near the workspace boundary
and near the folded-arm singularity, whether or not a puck exists.
Subtracting B isolates what the puck did.

Usage:
    python3 scripts/run_arm_map.py --grid 40 --steps 3 --out empowerment_map.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intrinsic_core import (  # noqa: E402
    ArmConfig,
    EmpowermentConfig,
    EmpowermentEstimator,
    GridDiscretiser,
    PlanarArm,
)


def compute_map(model, puck, xs, ys, cfg, disc, arm_for_ik):
    """Empowerment over a grid of end-effector positions.  NaN where unreachable."""
    est = EmpowermentEstimator(model.rollout, 9, disc, cfg)
    out = np.full((len(ys), len(xs)), np.nan)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if not arm_for_ik.reachable(x, y):
                continue
            ik = arm_for_ik.inverse_kinematics(x, y)
            if ik is None:
                continue
            state = np.array([ik[0], ik[1], puck[0], puck[1]])
            out[i, j] = est.estimate(state)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--resolution", type=float, default=0.025)
    ap.add_argument("--puck", type=float, nargs=2, default=[0.30, 0.25])
    ap.add_argument("--out", type=str, default="empowerment_map.png")
    args = ap.parse_args()

    puck = np.array(args.puck)
    arm = PlanarArm(ArmConfig())
    bolted = arm.with_fixed_puck()

    cfg = EmpowermentConfig(n_steps=args.steps, seed=0)
    disc = GridDiscretiser(args.resolution)

    xs = np.linspace(-0.50, 0.50, args.grid)
    ys = np.linspace(0.00, 0.52, args.grid)

    t0 = time.time()
    print(f"computing {args.grid}x{args.grid} map, horizon={args.steps} ...")
    map_push = compute_map(arm, puck, xs, ys, cfg, disc, arm)
    print(f"  pushable done  ({time.time() - t0:.1f}s)")
    map_bolt = compute_map(bolted, puck, xs, ys, cfg, disc, arm)
    print(f"  bolted done    ({time.time() - t0:.1f}s)")

    delta = map_push - map_bolt

    np.savez(
        Path(args.out).with_suffix(".npz"),
        xs=xs, ys=ys, pushable=map_push, bolted=map_bolt, delta=delta, puck=puck,
    )

    _render(xs, ys, map_push, map_bolt, delta, puck, args)
    print(f"wrote {args.out}")

    finite = delta[np.isfinite(delta)]
    print(f"\npuck contribution: max {finite.max():+.2f} bits, "
          f"mean {finite.mean():+.2f} bits")
    return 0


def _render(xs, ys, map_push, map_bolt, delta, puck, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    panels = [
        (map_push, "A. Pushable puck", "viridis", None),
        (map_bolt, "B. Bolted puck (control)", "viridis", None),
        (delta, "C. A - B: the puck's contribution", "magma", None),
    ]

    vmin = np.nanmin([np.nanmin(map_push), np.nanmin(map_bolt)])
    vmax = np.nanmax([np.nanmax(map_push), np.nanmax(map_bolt)])

    for ax, (data, title, cmap, _) in zip(axes, panels):
        kw = dict(origin="lower", extent=extent, cmap=cmap, aspect="equal")
        if cmap == "viridis":
            kw.update(vmin=vmin, vmax=vmax)
        im = ax.imshow(data, **kw)
        ax.scatter(*puck, s=110, facecolors="none",
                   edgecolors="white", linewidths=2.0, zorder=5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="bits")

    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        f"{args.steps}-step empowerment over a 2-link arm workspace "
        f"(circle = puck, never given to the agent as a goal)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
