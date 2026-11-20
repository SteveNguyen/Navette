"""Run the SLAM front end (navette.slam.frontend) over a recorded or simulated Sequence, offline.

    python -m navette.slam.run SEQUENCE.npz --out DIR [--no-loops]

Prints: the front end's counts (submaps, matches tried / accepted / kept), every match attempt with its reason,
trajectory metrics (navette.slam.metrics.summary) of odometry and of the SLAM when the Sequence has ground
truth; always, the start -> end motion of both (the return-to-start error when the run ends where it started)
and map-consistency numbers of the occupancy grid built from odometry poses vs from corrected poses (same
keyframes, replay()): obstacle cells and their fraction of the known cells (a drifted map draws a wall twice:
more, thicker obstacle cells), and the mean binary entropy of the known cells (contradicting keyframes leave
cells near 0.5; lower is more consistent).

Writes to DIR: trajectories.npz (t, odom, slam, gt if any, submap of each frame), trajectories.csv, matches.csv,
grid_odom.png / grid_slam.png (the two grids, same crop, trajectory drawn), grids.png (side by side) and
trajectories.png (odometry, SLAM, ground truth if any, submap anchors and accepted loop edges). PNGs need
OpenCV (the 'opencv' extra); without it they are skipped. Recorded runs are private (CLAUDE.md): DIR outside
the repository.
"""
import argparse
import csv
import math
import os
import time

import numpy as np

from navette.slam import metrics
from navette.slam.data import Sequence
from navette.slam.frontend import SlamFrontEnd, SlamParams

# colours (BGR) of the drawings
ODOM_C, SLAM_C, GT_C, LOOP_C, ANCHOR_C = (40, 40, 220), (200, 120, 0), (40, 160, 40), (200, 0, 200), (0, 140, 255)


def run_frontend(seq, params=None, loops=True):
    """The front end fed with every frame of seq, finished. Returns (front end, seconds)."""
    fe = SlamFrontEnd(params, loops=loops)
    t0 = time.perf_counter()
    for i in range(len(seq)):
        fe.add(seq.t[i], seq.odom[i], seq.scan(i), seq.cam_h[i], seq.cam_heading[i], bool(seq.glance[i]))
    fe.finish()
    return fe, time.perf_counter() - t0


def build_grids(seq, slam_poses, params):
    """(grid from odometry, grid from corrected poses): one navette.OccupancyGrid fed every frame at its
    odometry scan pose (glance frames forced, as the navigator), rendered, then the same keyframes moved to
    their corrected scan poses (slam_poses: base poses, + cam_heading) and replayed. Returns two log-odds
    arrays with the grid's (x0, y0, res)."""
    from navette import OccupancyGrid

    grid = OccupancyGrid(res=params.grid_res)
    kf = []
    odom_scan = seq.scan_poses()
    for i in range(len(seq)):
        if grid.update(seq.t[i], tuple(odom_scan[i]), seq.navette_scan(i), sigma=params.grid_sigma, h=seq.cam_h[i],
                       max_range=params.grid_max_range_m, force=bool(seq.glance[i])):
            kf.append(i)
    lo_odom = grid.log_odds
    corrected = np.array(slam_poses, float)
    corrected[:, 2] += seq.cam_heading
    grid.set_keyframe_poses([tuple(corrected[i]) for i in kf])
    grid.replay()
    return lo_odom, grid.log_odds, (grid.x0, grid.y0, grid.res), grid.thr


def grid_stats(lo, thr):
    """Map-consistency numbers of a log-odds grid (see the module docstring)."""
    known = np.abs(lo) > thr
    obstacle = lo > thr
    p = 1.0 / (1.0 + np.exp(-lo[known].astype(float)))
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return dict(obstacle_cells=int(obstacle.sum()), known_cells=int(known.sum()),
                obstacle_frac=float(obstacle.sum() / max(1, known.sum())), entropy=float(h.mean()) if len(h) else 0.0)


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _crop(los, thr, margin=20):
    """(i0, i1, j0, j1) cell box of the known cells of all grids, plus margin."""
    known = np.zeros_like(los[0], bool)
    for lo in los:
        known |= np.abs(lo) > thr
    ii, jj = np.nonzero(known)
    if not len(ii):
        return 0, known.shape[0], 0, known.shape[1]
    n0, n1 = known.shape
    return (max(0, ii.min() - margin), min(n0, ii.max() + margin + 1), max(0, jj.min() - margin),
            min(n1, jj.max() + margin + 1))


class _Canvas:
    """Image of the grid crop: x to the right, y up; scale px per cell."""

    def __init__(self, box, origin, scale):
        self.box, self.origin, self.scale = box, origin, scale
        i0, i1, j0, j1 = box
        self.w, self.h = (i1 - i0) * scale, (j1 - j0) * scale

    def px(self, xy):
        x0, y0, res = self.origin
        i0, _, _, j1 = self.box
        xy = np.atleast_2d(np.asarray(xy, float))
        u = ((xy[:, 0] - x0) / res - i0) * self.scale
        v = (j1 - (xy[:, 1] - y0) / res) * self.scale
        return np.round(np.column_stack([u, v])).astype(np.int32)

    def grid_image(self, lo, thr):
        import cv2

        i0, i1, j0, j1 = self.box
        c = lo[i0:i1, j0:j1].T[::-1]  # rows: y down the image
        img = np.full(c.shape, 160, np.uint8)
        img[c < -thr] = 255
        img[c > thr] = 0
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def path(self, img, poses, colour, thickness=1):
        import cv2

        cv2.polylines(img, [self.px(np.asarray(poses)[:, :2])], False, colour, thickness, cv2.LINE_AA)


def _label(img, text):
    import cv2

    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def write_pngs(out, seq, fe, lo_odom, lo_slam, origin, thr):
    """grid_odom.png, grid_slam.png, grids.png, trajectories.png (see the module docstring). False without
    OpenCV."""
    try:
        import cv2
    except ImportError:
        return False
    box = _crop([lo_odom, lo_slam], thr)
    scale = max(1, min(4, 1200 // max(1, box[1] - box[0], box[3] - box[2])))
    cv = _Canvas(box, origin, scale)
    slam = fe.poses()
    imgs = []
    for name, lo, poses, colour in (("odom", lo_odom, seq.odom, ODOM_C), ("slam", lo_slam, slam, SLAM_C)):
        img = cv.grid_image(lo, thr)
        cv.path(img, poses, colour, 1)
        _label(img, f"{name}: grid from {'odometry' if name == 'odom' else 'corrected'} poses")
        cv2.imwrite(os.path.join(out, f"grid_{name}.png"), img)
        imgs.append(img)
    sep = np.full((imgs[0].shape[0], 6, 3), 255, np.uint8)
    cv2.imwrite(os.path.join(out, "grids.png"), np.hstack([imgs[0], sep, imgs[1]]))
    # trajectories on the corrected grid, faded
    img = (0.35 * cv.grid_image(lo_slam, thr) + 0.65 * 255).astype(np.uint8)
    if seq.gt is not None:  # ground truth in the map frame: anchored on the odometry's start (map = odom then)
        cv.path(img, metrics.anchor(seq.gt, seq.odom), GT_C, 2)
    cv.path(img, seq.odom, ODOM_C, 1)
    cv.path(img, slam, SLAM_C, 1)
    nodes = np.array(fe.graph.poses(), float)
    for s in fe.submaps:
        cv2.circle(img, tuple(int(v) for v in cv.px(nodes[s.index, :2])[0]), 4 if s.kind == "spin" else 2,
                   ANCHOR_C, -1, cv2.LINE_AA)
    for m in fe.matches:
        if m.edge is not None:
            a, b = cv.px(nodes[m.base, :2])[0], cv.px(nodes[m.query, :2])[0]
            cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), LOOP_C, 2, cv2.LINE_AA)
    _label(img, "odometry (red), SLAM (blue)" + (", truth (green)" if seq.gt is not None else "")
           + ", anchors (orange), loops (magenta)")
    cv2.imwrite(os.path.join(out, "trajectories.png"), img)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m navette.slam.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", help="Sequence .npz (navette.slam.extract or tests/sim_slam.py)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--no-loops", action="store_true", help="odometry chain only (the baseline)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    seq = Sequence.load(args.sequence)
    params = SlamParams()
    fe, secs = run_frontend(seq, params, loops=not args.no_loops)
    slam = fe.poses()
    st = fe.stats()
    print(f"{args.sequence}: {len(seq)} frames, {seq.t[-1] - seq.t[0]:.0f} s, odometry path "
          f"{metrics.path_length(seq.odom):.1f} m; front end {secs:.2f} s")
    print(f"submaps {st['submaps']} ({st['spins']} spin, {st['travel']} travel); matches tried {st['tried']}, "
          f"accepted {st['accepted']}, kept by the graph {st['kept']}; final map->odom correction "
          f"{st['correction_m']:.3f} m, {st['correction_deg']:.1f} deg")
    for m in fe.matches:
        w = "" if m.window is None else f" window {m.window[0]:.2f} m {math.degrees(m.window[2]):.0f} deg"
        ref = f"{m.ref[0]}-{m.ref[-1]}" if m.ref else "-"
        print(f"  submap {m.query:3d} ({m.kind}) vs {ref:7s} score {m.score:.2f} n {m.n:4d} obs {m.n_obs:4d} ref {m.n_ref:5d} "
              f"sigma {m.sigma_xy:.3f} m {math.degrees(m.sigma_th):4.1f} deg{w}: {m.reason}")
    rows = {}
    if seq.gt is not None:
        rows["odometry"], rows["slam"] = metrics.summary(seq.odom, seq.gt), metrics.summary(slam, seq.gt)
        keys = list(rows["odometry"])
        print("metrics (vs ground truth):")
        for k in keys:
            print(f"  {k:20s} odometry {rows['odometry'][k]:8.3f}   slam {rows['slam'][k]:8.3f}")
    for name, p in (("odometry", seq.odom), ("slam", slam)):
        d, r = metrics.return_to_start(p)
        print(f"start -> end ({name}): {d:.3f} m, {math.degrees(r):.1f} deg (the error if the run ended at its start)")
    lo_odom, lo_slam, origin, thr = build_grids(seq, slam, params)
    for name, lo in (("odometry", lo_odom), ("slam", lo_slam)):
        g = grid_stats(lo, thr)
        print(f"grid from {name:8s}: obstacle cells {g['obstacle_cells']}, known {g['known_cells']}, obstacle "
              f"fraction {g['obstacle_frac']:.3f}, mean entropy {g['entropy']:.3f} bit")
    # outputs
    sub = fe.frame_submaps()
    arrays = dict(t=seq.t, odom=seq.odom, slam=slam, submap=sub)
    if seq.gt is not None:
        arrays["gt"] = seq.gt
    np.savez_compressed(os.path.join(args.out, "trajectories.npz"), **arrays)
    _write_csv(os.path.join(args.out, "trajectories.csv"),
               ["t", "odom_x", "odom_y", "odom_theta", "slam_x", "slam_y", "slam_theta", "submap"],
               [[f"{seq.t[i]:.3f}", *(f"{v:.4f}" for v in seq.odom[i]), *(f"{v:.4f}" for v in slam[i]), int(sub[i])]
                for i in range(len(seq))])
    _write_csv(os.path.join(args.out, "matches.csv"),
               ["query", "kind", "ref_first", "ref_last", "base", "score", "n", "n_obs", "n_ref", "sigma_xy", "sigma_th_deg",
                "window_xy", "window_th_deg", "accepted", "dcs_scale", "reason"],
               [[m.query, m.kind, m.ref[0] if m.ref else "", m.ref[-1] if m.ref else "", m.base, f"{m.score:.3f}",
                 m.n, m.n_obs, m.n_ref, f"{m.sigma_xy:.4f}", f"{math.degrees(m.sigma_th):.2f}",
                 "" if m.window is None else f"{m.window[0]:.3f}",
                 "" if m.window is None else f"{math.degrees(m.window[2]):.1f}", int(m.edge is not None),
                 f"{m.dcs_scale:.3f}", m.reason] for m in fe.matches])
    if write_pngs(args.out, seq, fe, lo_odom, lo_slam, origin, thr):
        print(f"wrote {args.out}: trajectories.npz/.csv, matches.csv, grids.png, grid_odom.png, grid_slam.png, "
              f"trajectories.png")
    else:
        print(f"wrote {args.out}: trajectories.npz/.csv, matches.csv (no OpenCV: no PNG)")


if __name__ == "__main__":
    main()
