"""SLAM scenario with ground truth, on the 2-D simulator of tests/sim2d.py. Pure numpy, seeded, deterministic.

    seq = scenario(seed=0)              # navette.slam.data.Sequence: odometry, scans, cam_h, ground truth
    python tests/sim_slam.py OUT.npz [--seed N] [--loops K]   # the same, saved

World (`world()`, 6 x 4 m): two rooms joined by two 1.1 m wide corridors around a central block, i.e. a loop
(left room x 0-2.2, right room x 3.8-6, block x 2.2-3.8 / y 1.1-2.9; 0.8 m doorways at both ends of each
corridor), with three boxes. The robot goes around the loop (bottom corridor to the right room, top corridor
back) and ends at its start: a loop closure in the left room.

Driving: a scripted waypoint follower (pure pursuit, <= V_MAX, turning in place at sharp corners), NOT the
Navette navigator: the scenario must not change when the navigator does. It steers on the TRUE pose, so the
true path stays on the waypoints whatever the odometry drift (the SLAM only ever sees the odometry and the
scans; only the spins' ends depend on the odometry, as on the robot). Every
SPIN_EVERY_M of travel, and at the start and the end, it spins 360 deg in place (as the robot does to look
around): measured on the odometry heading, as the robot would.

Odometry (`Odometry`): integrates the true per-step motion (sim2d's v dt, w dt) through a drift model close to
the robot's wheels + base gyro: a wheel scale error (default +1.5 %: travelled distance over-read), a yaw scale
error (default -1 %: gyro scale), a constant heading bias (default 3 deg/min, gyro bias), per-step white noise
on distance (1 % of the step) and heading (1 % of the step + 0.03 deg). With all of them zero it reproduces the
truth exactly (same integration as sim2d.Sim.step: rotate, then advance along the new heading).

Scan: sim2d's camera-like scan, fov +-35 deg (fov_deg=35), floor seen from 0.34 m to 2.5 m, range noise
0.01 x range^2 (1 cm at 1 m, 4 cm at 2 m). One frame per step at 8 Hz.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sim2d, when run as a script

from sim2d import CAM_H, Sim, polygon, rect  # noqa: E402

from navette.slam.data import Keyframe, Sequence  # noqa: E402

DT = 1 / 8.0  # s, frame period (the robot processes ~8 Hz)
V_MAX = 0.15  # m/s, the robot's cap
W_MAX = 0.6  # rad/s, turns and spins (a 360 deg spin: ~11 s)
LOOKAHEAD = 0.35  # m, pure pursuit
TURN_IN_PLACE = math.radians(50)  # heading error above which the follower stops and turns in place
SPIN_EVERY_M = 3.0  # m of travel between spins
START = (1.1, 2.0, 0.0)  # in the left room, facing the block
# around the loop: bottom corridor, right room, top corridor, back to the start
WAYPOINTS = [(1.5, 0.6), (3.0, 0.6), (4.5, 0.6), (4.9, 2.0), (4.5, 3.4), (3.0, 3.4), (1.5, 3.4), (1.1, 2.0)]


def world():
    """Wall segments (N, 4) of the scenario's world (see the module docstring)."""
    stubs = []  # door jambs: 0.15 m stubs narrowing each corridor end to a 0.8 m doorway (wall corners to match)
    for x in (2.2, 3.8):
        stubs += [[x, 0.0, x, 0.15], [x, 1.1, x, 0.95], [x, 4.0, x, 3.85], [x, 2.9, x, 3.05]]
    boxes = [rect(0.4, 2.9, 0.8, 3.4),  # left room, top left
             rect(5.2, 1.6, 5.6, 2.3),  # right room, against the far wall side
             polygon((0.5, 0.5), (0.9, 0.4), (1.0, 0.8), (0.6, 0.9))]  # left room, bottom left, rotated
    return np.vstack([rect(0, 0, 6, 4), rect(2.2, 1.1, 3.8, 2.9), np.array(stubs, float), *boxes])


class Odometry:
    """Wheel + gyro odometry with drift, integrating the true motion increments (see the module docstring)."""

    def __init__(self, pose=(0.0, 0.0, 0.0), rng=None, scale=0.015, yaw_scale=-0.01, bias_deg_min=3.0,
                 noise_s=0.01, noise_th=0.01, noise_th0_deg=0.03):
        self.pose = np.array(pose, float)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.scale, self.yaw_scale, self.bias = scale, yaw_scale, math.radians(bias_deg_min) / 60.0
        self.noise_s, self.noise_th, self.noise_th0 = noise_s, noise_th, math.radians(noise_th0_deg)

    def step(self, ds, dth, dt):
        """True increments: ds (m, along the heading after the turn), dth (rad), over dt (s)."""
        n = self.rng.standard_normal(2)
        ds_o = ds * (1 + self.scale) + self.noise_s * abs(ds) * n[0]
        dth_o = dth * (1 + self.yaw_scale) + self.bias * dt + (self.noise_th * abs(dth) + self.noise_th0) * n[1]
        self.pose[2] += dth_o
        self.pose[0] += ds_o * math.cos(self.pose[2])
        self.pose[1] += ds_o * math.sin(self.pose[2])
        return tuple(float(x) for x in self.pose)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _closest_on_path(path, p, seg):
    """(segment index, point) of the path point nearest p, searching forward from segment `seg` only (the
    path loops back on itself: never jump to a later or earlier pass)."""
    best = None
    for k in range(seg, min(seg + 2, len(path) - 1)):
        a, b = path[k], path[k + 1]
        e = b - a
        u = float(np.clip((p - a) @ e / (e @ e), 0.0, 1.0))
        q = a + u * e
        d = float(np.hypot(*(q - p)))
        if best is None or d < best[0]:
            best = (d, k, q)
    return best[1], best[2]


def _lookahead_point(path, seg, q, dist):
    """The point `dist` metres further along the path from q (on segment seg); the path's end at most."""
    k, p = seg, q
    while k < len(path) - 1:
        left = float(np.hypot(*(path[k + 1] - p)))
        if left >= dist:
            return p + (path[k + 1] - p) * (dist / left)
        dist -= left
        k, p = k + 1, path[k + 1]
    return path[-1]


def scenario(seed=0, loops=1, odom_kw=None, noise=0.01, max_steps=20000):
    """The scenario's Sequence (see the module docstring). loops: times around the loop. odom_kw: Odometry
    overrides (e.g. all zero: perfect odometry). meta: parameters, the spins' frame ranges [start, end) and
    the true path length."""
    sim = Sim(world(), START, seed=seed, noise=noise, fov_deg=35.0, closest=0.34, max_range=2.5)
    odom_kw = dict(odom_kw or {})
    odo = Odometry(rng=np.random.default_rng([seed, 1]), **odom_kw)  # odom frame: starts at the origin
    path = np.array([START[:2]] + WAYPOINTS * loops, float)
    frames, spins = [], []
    seg, travelled, since_spin, odom = 0, 0.0, 0.0, (0.0, 0.0, 0.0)
    spin = dict(left=2 * math.pi, start=0)  # spin in progress: odometry yaw still to turn; first: at the start
    done_path, min_clearance = False, sim.clearance()
    for n in range(max_steps):
        frames.append(Keyframe(t=n * DT, odom=odom, scan=sim.scan(), cam_h=CAM_H,
                               gt=tuple(float(x) for x in sim.pose)))
        p, th = sim.pose[:2], float(sim.pose[2])
        if spin is None and not done_path and since_spin >= SPIN_EVERY_M:
            spin, since_spin = dict(left=2 * math.pi, start=n), 0.0
        if spin is None and not done_path:
            seg, q = _closest_on_path(path, p, seg)
            done_path = seg == len(path) - 2 and float(np.hypot(*(path[-1] - p))) < 0.05
            if done_path:  # back at the start: a last spin there (the loop closure's view), then stop
                spin = dict(left=2 * math.pi, start=n)
        if spin is not None:
            v, w = 0.0, W_MAX
        elif done_path:
            if abs(sim.v) < 1e-3 and abs(sim.w) < 1e-3:
                break
            v, w = 0.0, 0.0
        else:
            target = _lookahead_point(path, seg, q, LOOKAHEAD)
            alpha = _wrap(math.atan2(target[1] - p[1], target[0] - p[0]) - th)
            if abs(alpha) > TURN_IN_PLACE:
                v, w = 0.0, math.copysign(W_MAX, alpha)
            else:
                # pure pursuit: curvature 2 sin(alpha) / L; slower when the arc is tight
                v = V_MAX * max(0.3, math.cos(alpha))
                w = float(np.clip(v * 2 * math.sin(alpha) / LOOKAHEAD, -W_MAX, W_MAX))
        th_before = float(sim.pose[2])
        sim.step(v, w, DT)
        min_clearance = min(min_clearance, sim.clearance())
        ds, dth = sim.v * DT, float(sim.pose[2]) - th_before  # exactly sim2d's increments
        odom_before = odom
        odom = odo.step(ds, dth, DT)
        travelled += abs(ds)
        since_spin += abs(ds)
        if spin is not None:
            spin["left"] -= odom[2] - odom_before[2]  # measured on the odometry heading, as the robot would
            if spin["left"] <= 0:
                spins.append([spin["start"], n + 1])
                spin = None
    else:
        raise RuntimeError("scenario did not finish within max_steps")
    if min_clearance < sim.radius:  # a scripted path through a wall: a bug of the scenario
        raise RuntimeError(f"scenario collided (clearance {min_clearance:.3f} m)")
    meta = dict(source="tests/sim_slam.py", seed=seed, loops=loops, dt=DT, noise=noise, odom=odom_kw,
                start=list(START), spins=spins, travelled_m=travelled, min_clearance=min_clearance, cam_h=CAM_H)
    return Sequence.from_keyframes(frames, meta)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Generate the SLAM simulation scenario and save it (.npz).")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loops", type=int, default=1)
    args = ap.parse_args(argv)
    seq = scenario(seed=args.seed, loops=args.loops)
    seq.save(args.out)
    print(f"{args.out}: {len(seq)} frames, {seq.t[-1]:.0f} s, {seq.meta['travelled_m']:.1f} m, "
          f"{len(seq.meta['spins'])} spins")


if __name__ == "__main__":
    main()
