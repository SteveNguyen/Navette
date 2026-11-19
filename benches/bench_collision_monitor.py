"""Collision monitor timing on realistic sizes: a 180-sector scan + 300 map points, requests that pass, are
slowed and are blocked (open floor, a wall 0.6 m and 0.2 m ahead), and a controller's arc scoring (39 arcs,
1 m ahead).

    uv run python benches/bench_collision_monitor.py
"""

import time

import numpy as np

from navette import SafetyParams, Scan, boundary_points, free_arc_lengths, safe_command

REPS = 300  # calls per case

BEAR = (np.arange(180) + 0.5) * 2.0 - 180
SEEN = np.abs(BEAR) < 55


def scan(wall=None):
    """Open floor in a 110 deg view, or a wall `wall` m ahead (+-20 deg)."""
    obstacle = np.full(180, np.inf)
    if wall is not None:
        obstacle[np.abs(BEAR) < 20] = wall
    return Scan(BEAR, SEEN, obstacle, np.zeros(180, bool), np.where(SEEN, 2.5, 0.0), np.where(SEEN, 0.44, np.inf))


def map_points(n=300):
    """Map obstacles around the robot (robot frame), leaving a corridor ahead free."""
    pts = np.random.default_rng(0).uniform(-1.5, 1.5, (4 * n, 2))
    return pts[(np.abs(pts[:, 1]) > 0.6) | (pts[:, 0] < -0.4)][:n]


def median_us(f):
    f()  # warm up
    t = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return np.median(t) * 1e6


extra = map_points()
assert len(extra) == 300
# (name, v, w, distance of the wall ahead)
REQUESTS = [("open floor, straight", 0.3, 0.0, None), ("open floor, turning", 0.3, 0.5, None),
            ("wall at 0.6 m, straight", 0.3, 0.0, 0.6), ("wall at 0.6 m, turning", 0.3, 0.5, 0.6),
            ("wall at 0.2 m, straight", 0.3, 0.0, 0.2), ("wall at 0.2 m, turning", 0.3, 0.5, 0.2)]
PARAMS = (("default", SafetyParams()),
          ("robot (latency 1.53 s, decel 0.64)", SafetyParams(latency=1.53, decel=0.64)))
print(f"navette: collision monitor, 180 sectors + {len(extra)} map points, us per call (median of {REPS})")
for pname, p in PARAMS:
    print(f"  SafetyParams {pname}")
    for name, v, w, wall in REQUESTS:
        sc = scan(wall)
        us = median_us(lambda: safe_command(v, w, sc, p, extra))
        print(f"    safe_command {name:24s} {us:8.1f}   -> v = {safe_command(v, w, sc, p, extra)[0]:.3f}")
    curvs = [w / v for v in (0.1, 0.2, 0.3) for w in np.linspace(-1.2, 1.2, 13)]
    for name, wall in (("open floor", None), ("wall at 0.6 m", 0.6)):
        pts = np.concatenate([boundary_points(scan(wall), p.unseen_m, near_m=p.radius + p.margin), extra])
        us = median_us(lambda: free_arc_lengths(pts, curvs, p, 1.0))
        print(f"    free_arc_lengths {len(curvs)} arcs, {len(pts)} points, {name:14s} {us:8.1f}")
