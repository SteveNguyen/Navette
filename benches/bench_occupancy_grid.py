"""OccupancyGrid timing on the sequence of tests/golden/occupancy_grid_room.npz (a drive through a room,
default parameters, 180 sectors): per frame, blind_known, then update, then obstacle_points, as the navigator
calls them. Through the Python API, so the times include the conversion of the scan.

    uv run python benches/bench_occupancy_grid.py
"""

import time
from pathlib import Path

import numpy as np

from navette import OccupancyGrid, Scan

REPS = 20  # runs of the sequence

g = np.load(Path(__file__).resolve().parents[1] / "tests/golden/occupancy_grid_room.npz")
scans = [Scan(g["bearing_deg"], *(g[f][i] for f in ("seen", "obstacle", "near", "floor", "closest")))
         for i in range(len(g["t"]))]
steps = [(float(t), tuple(p), s) for t, p, s in zip(g["t"], g["pose"], scans)]

times = {"blind_known": [], "update (keyframe)": [], "update (no keyframe)": [], "obstacle_points": [], "replay": []}
for _ in range(REPS):
    grid = OccupancyGrid()
    for t, pose, scan in steps:
        t0 = time.perf_counter()
        grid.blind_known(pose, scan)
        t1 = time.perf_counter()
        is_key = grid.update(t, pose, scan)
        t2 = time.perf_counter()
        grid.obstacle_points(pose)
        t3 = time.perf_counter()
        times["blind_known"].append(t1 - t0)
        times["update (keyframe)" if is_key else "update (no keyframe)"].append(t2 - t1)
        times["obstacle_points"].append(t3 - t2)
    t0 = time.perf_counter()
    grid.replay()
    times["replay"].append(time.perf_counter() - t0)

print(f"navette: OccupancyGrid over {len(steps)} frames ({len(grid.keyframes)} keyframes) x {REPS} runs, ms per call")
for name, t in times.items():
    t = np.array(t) * 1e3
    print(f"  {name:21s} median {np.median(t):7.4f}, p99 {np.percentile(t, 99):7.4f}, max {t.max():7.4f}")
