"""The whole per-frame core path, floor mask -> command, on recorded closed-loop sequences
(tests/golden/navigator_*_live.npz: the robot's collision monitor parameters), with a 320x192 floor mask per
frame rendered through the reference robot's lens (the rays of tests/golden/scan_real_lens.npz).

Two ways of making the same step, timed apart:
  scan + step   FreeSpace.scan(mask) then Navigator.step(scan): what the runtime's loop does
  step_mask     Navigator.step_mask(fs, mask): the same in one call, the scan staying on the Rust side

    uv run python benches/bench_step.py
"""

import sys
import time
from pathlib import Path

import numpy as np

from navette import FreeSpace, Navigator, SafetyParams

TESTS = Path(__file__).resolve().parents[1] / "tests"
sys.path.insert(0, str(TESTS))
import sim2d  # noqa: E402  (the simulator's scan format)

REPS = 3  # runs of each sequence
SEQUENCES = ("clutter_live", "doorway_live", "forward_box_live")
SAFETY = ("radius", "margin", "v_max", "decel", "latency", "unseen_m", "reverse_max", "step")
PITCH_DOWN_DEG = 15.0  # the reference robot's camera


def camera(pitch_down_deg):
    """R_wc (columns = camera axes in the world: x forward, y left, z up) of a camera pitched down, and 'up'
    in the camera frame."""
    p = np.radians(pitch_down_deg)
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    R_wc = Ry @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # x_c = -y_w, y_c = -z_w, z_c = x_w
    return R_wc, R_wc.T @ np.array([0, 0, 1.0])


def floor_masks(g, rays, R_wc, shape):
    """One floor mask per frame of a sequence: what the camera would see of that frame's scan. A pixel is floor
    if its ray reaches the floor before the obstacle of its 2 deg bearing sector (sectors outside the
    simulator's narrower view take the nearest one's). The same function is kept with the prototype's bench:
    change both together."""
    d = rays @ R_wc.T
    down = d[:, 2] < -1e-9
    cam_h = float(g["cam_h"][0])
    rng = np.where(down, cam_h / np.where(down, -d[:, 2], 1.0), np.inf) * np.hypot(d[:, 0], d[:, 1])
    sector = np.clip(((np.degrees(np.arctan2(d[:, 1], d[:, 0])) + 180) / 2.0).astype(int), 0, sim2d.N_SECTORS - 1)
    seen = g["seen_idx"]
    sector = np.clip(sector, seen[0], seen[-1]) - seen[0]  # column of g["floor"]
    masks = []
    for i in range(len(g["t"])):
        # the floor ends at an obstacle where it ends before max_range; else it goes on
        obstacle = np.where(g["floor"][i] < g["max_range"][i], g["floor"][i], np.inf)
        masks.append((down & (rng < obstacle[sector])).reshape(shape))
    return masks


def stats(name, t):
    t = np.array(t) * 1e3
    print(f"  {name:26s} n {len(t):5d}  median {np.median(t):7.3f}, p95 {np.percentile(t, 95):7.3f}, max {t.max():7.3f}")


def new_navigator(g):
    safety = SafetyParams(**{k: float(g[k]) for k in SAFETY}, n_speeds=int(g["n_speeds"]))
    return Navigator(safety, behaviour=str(g["behaviour"]), odometry=bool(g["odometry"]),
                     resume_n=int(g["resume_n"]), forward_v=float(g["forward_v"]))


def replay(name, lens, R_wc, up_c):
    g = dict(np.load(TESTS / "golden" / f"navigator_{name}.npz"))
    shape = tuple(int(x) for x in lens["mask_shape"])
    masks = floor_masks(g, lens["rays"], R_wc, shape)
    args = [g[k].tolist() for k in ("t", "dt", "cam_h", "a", "v", "w", "v_prev")]
    poses = [tuple(p) for p in g["pose"].tolist()]
    up = tuple(up_c)
    t_scan, t_step, t_both, t_mask = [], [], [], []
    for _ in range(REPS):
        fs, nav = FreeSpace(lens["rays"], lens["rays_low"], shape), new_navigator(g)
        fs2, nav2 = FreeSpace(lens["rays"], lens["rays_low"], shape), new_navigator(g)
        for i, op in enumerate(g["op"]):
            t, dt, cam_h, a, v, w, v_prev = (x[i] for x in args)
            if op == 1:  # a head-glance frame: into the map only, not timed
                scan = fs.scan(masks[i], up, cam_h)
                nav.observe(t, poses[i], scan, cam_h)
                nav2.observe(t, poses[i], scan, cam_h)
                continue
            t0 = time.perf_counter()
            scan = fs.scan(masks[i], up, cam_h)
            t1 = time.perf_counter()
            nav.step(t, dt, poses[i], scan, cam_h, a, v, w, v_prev)
            t2 = time.perf_counter()
            nav2.step_mask(fs2, masks[i], up, t, dt, poses[i], cam_h, a, v, w, v_prev)
            t3 = time.perf_counter()
            t_scan.append(t1 - t0)
            t_step.append(t2 - t1)
            t_both.append(t2 - t0)
            t_mask.append(t3 - t2)
    print(f"{name} ({str(g['behaviour'])})")
    stats("FreeSpace.scan", t_scan)
    stats("Navigator.step", t_step)
    stats("scan + step (mask -> cmd)", t_both)
    stats("Navigator.step_mask", t_mask)


if __name__ == "__main__":
    lens = dict(np.load(TESTS / "golden" / "scan_real_lens.npz"))
    R_wc, up_c = camera(PITCH_DOWN_DEG)
    print(f"navette: floor mask -> command over recorded sequences x {REPS} runs, ms per frame")
    for name in SEQUENCES:
        replay(name, lens, R_wc, up_c)
