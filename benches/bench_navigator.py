"""Navigator.step timing on recorded golden sequences (tests/golden/navigator_*.npz: the frames of closed-loop
runs in the 2-D simulator), steps where the frontier planner ran apart from the others; then what a plan is
made of, on the last map of a sequence and on a large one (the planner's worst case: its window is the whole
known map). Through the Python API, so the times include the conversion of the scan.

    uv run python benches/bench_navigator.py
"""

import sys
import time
from pathlib import Path

import numpy as np

from navette import (FrontierParams, FrontierPlanner, Navigator, OccupancyGrid, SafetyParams, Scan, dijkstra,
                     distance_transform, frontier_cells)

TESTS = Path(__file__).resolve().parents[1] / "tests"
sys.path.insert(0, str(TESTS))
import sim2d  # noqa: E402  (the simulator's scan format)

REPS = 5  # runs of each sequence
SEQUENCES = ("clutter_default", "doorway_live", "open_default", "forward_box_default")
SAFETY = ("radius", "margin", "v_max", "decel", "latency", "unseen_m", "reverse_max", "step")


def stats(name, t):
    t = np.array(t) * 1e3
    print(f"  {name:26s} n {len(t):5d}  median {np.median(t):7.3f}, p95 {np.percentile(t, 95):7.3f}, max {t.max():7.3f}")


def timed(f, reps=20):
    """Median time of f() (s)."""
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def replay(name):
    """Times of the steps of a sequence (plan steps, other steps), and the navigator and pose at its end."""
    g = dict(np.load(TESTS / "golden" / f"navigator_{name}.npz"))
    scans = [Scan(**sim2d.scan_arrays(g["seen_idx"], g["floor"][i], float(g["closest"][i]), float(g["max_range"][i]),
                                      float(g["near_tol"]))) for i in range(len(g["t"]))]
    args = [g[k].tolist() for k in ("t", "dt", "cam_h", "a", "v", "w", "v_prev")]
    poses = [tuple(p) for p in g["pose"].tolist()]
    plan, other = [], []
    for _ in range(REPS):
        safety = SafetyParams(**{k: float(g[k]) for k in SAFETY}, n_speeds=int(g["n_speeds"]))
        nav = Navigator(safety, behaviour=str(g["behaviour"]), odometry=bool(g["odometry"]),
                        resume_n=int(g["resume_n"]), forward_v=float(g["forward_v"]))
        for i, op in enumerate(g["op"]):
            t, dt, cam_h, a, v, w, v_prev = (x[i] for x in args)
            if op == 1:
                nav.observe(t, poses[i], scans[i], cam_h)
                continue
            t0 = time.perf_counter()
            nav.step(t, dt, poses[i], scans[i], cam_h, a, v, w, v_prev)
            (plan if g["planned"][i] else other).append(time.perf_counter() - t0)
    return plan, other, nav, poses[-1]


def plan_parts(grid, pose, label):
    """The planner on `grid` from `pose`, and its parts on the same window (the known cells + 3)."""
    obst, free = grid.masks()
    known = np.argwhere(obst | free)
    lo, hi = np.maximum(known.min(0) - 3, 0), np.minimum(known.max(0) + 4, grid.n)
    obst, free = obst[lo[0]:hi[0], lo[1]:hi[1]], free[lo[0]:hi[0], lo[1]:hi[1]]
    p = FrontierParams()
    dist = distance_transform(obst) * grid.res
    trav = free & (dist >= p.clear)
    cost = 1 + p.soft_k * np.clip(1 - (dist - p.clear) / p.soft_m, 0, 1)
    start = tuple(grid.cell(np.array([pose[:2]]))[0] - lo)
    planner = FrontierPlanner()
    parts = {
        "plan (whole)": lambda: planner.plan(0.0, pose, grid),
        "  masks (whole grid)": grid.masks,
        "  distance_transform": lambda: distance_transform(obst),
        "  dijkstra": lambda: dijkstra(trav, cost, grid.res, start),
        "  frontier_cells": lambda: frontier_cells(obst, free, 5, 40),
    }
    print(f"{label}: window {obst.shape[0]} x {obst.shape[1]} cells, {int(trav.sum())} traversable, ms")
    for name, f in parts.items():
        print(f"  {name:22s} {timed(f) * 1e3:7.3f}")


def large_map(side_m=20.0):
    """A square hall of side_m with a box every 2 m, all known: about the largest map the default 24 m grid
    holds."""
    grid = OccupancyGrid()
    n, res = grid.n, grid.res
    c = (np.arange(n) + 0.5) * res - n * res / 2  # cell centres, origin at the grid centre
    x, y = np.meshgrid(c, c, indexing="ij")
    l = np.zeros((n, n), np.float32)
    inside = (np.abs(x) < side_m / 2) & (np.abs(y) < side_m / 2)
    l[inside] = -1.5
    l[inside & ((np.abs(x) > side_m / 2 - 0.1) | (np.abs(y) > side_m / 2 - 0.1))] = 1.5
    l[inside & (np.abs(x % 2.0 - 1.0) < 0.2) & (np.abs(y % 2.0 - 1.0) < 0.2)] = 1.5
    l[(x > side_m / 2 - 0.1) & (np.abs(y) < 0.5)] = 0.0  # a door: the frontier to plan to
    l[(x > side_m / 2 - 0.2) & (x < side_m / 2 - 0.1) & (np.abs(y) < 0.5)] = -1.5
    grid.set_map(-n * res / 2, -n * res / 2, l)
    return grid


if __name__ == "__main__":
    print(f"navette: Navigator.step over recorded sequences x {REPS} runs, ms per step")
    for name in SEQUENCES:
        plan, other, nav, pose = replay(name)
        print(f"{name}")
        if plan:
            stats("steps with a plan", plan)
        stats("other steps", other)
        if name == SEQUENCES[0]:
            last = (nav.grid, pose)
    print()
    plan_parts(*last, f"planner on the last map of {SEQUENCES[0]}")
    plan_parts(large_map(), (-9.0, 0.3, 0.0), "planner on a 20 x 20 m hall (worst case)")
