"""Measure the two [perception] switches on recorded runs, each one on its own. Open loop: the replay follows the
robot's own steps (navette.runtime.replay), nothing is sent anywhere. Recorded runs are private: give their
folders on the command line, write the output outside the repository.

    python tools/flicker/compare_replay.py --robot robots/reachy_poussah.toml --out OUT RUN [RUN ...]
    python tools/flicker/compare_replay.py --robot ... --out OUT --frames 1500 1700 RUN   # a short segment

Variants: one replay each; the floor masks (and the small grey images) are computed once per frame and shared.
  A  pose_at_capture = false, floor_two_frames = false   the Python prototype's
  B  pose_at_capture = true,  floor_two_frames = false   B vs A: the pose at the capture time alone
  C  pose_at_capture = true,  floor_two_frames = true    C vs B: floor in two frames alone

Pose at the capture time (A vs B), on consecutive steps (< 0.3 s apart, same navigator run), as the flicker
study measured it: the previous image is warped onto this one through the floor plane (each frame's 'up' and
camera height, the variant's poses, the navigator's geometry: camera at the pose, heading = the optical axis
on the floor), and the high-passed images are compared by normalised cross-correlation on this frame's floor
(eroded mask, 0.4-2.5 m). Higher = better aligned. Reported while turning in place (|w| > 0.2 rad/s, |v| < 0.05
m/s; measured base motion) and while driving (|v| >= 0.05 m/s): median NCC, and how often B beats A.
Also the map sharpness: cells above the grid's obstacle threshold in the final map of each navigator run
(summed): a smeared map has thicker walls, so more of them.
Floor in two frames (C vs B):
  - what it removes: sectors shortened per step (Navigator.floor_cut), steps with any;
  - what it costs: steps where B allowed forward motion and C did not (v_req > 0), mean allowed speed.

Output: OUT/<variant>/<run>/steps.csv (the replay's) and OUT/summary.json; the summary is printed.
"""
import argparse
import importlib
import json
import os
import time

import cv2
import numpy as np

from navette import config
from navette.rays import mask_rays
from navette.runtime import loop
from navette.runtime.replay import replay

VARIANTS = {"A": dict(pose_at_capture=False, floor_two_frames=False),
            "B": dict(pose_at_capture=True, floor_two_frames=False),
            "C": dict(pose_at_capture=True, floor_two_frames=True)}
MAX_PAIR_DT = 0.3       # s, consecutive steps compared
GW, GH = 320, 180       # grey image size for the alignment
RANGE_M = (0.4, 2.5)    # floor points used for the alignment


class Recorder:
    """A navigator that logs what each step got and gave."""

    def __init__(self, nav, steps, run_id):
        self.nav, self.steps, self.run_id = nav, steps, run_id

    def step(self, t, dt, pose, scan, cam_h, a, v, w, v_prev):
        out = self.nav.step(t, dt, pose, scan, cam_h, a, v, w, v_prev)
        self.steps.append(dict(run=self.run_id, t=t, pose=tuple(pose), cam_h=cam_h, v=v, w=w, v_out=out.v,
                               v_req=out.v_req, cut=self.nav.floor_cut))
        return out

    def __getattr__(self, name):  # observe, set_request, grid... as the navigator's
        return getattr(self.nav, name)


def open_run(cfg, run_dir, cache, frames):
    """The recorded run as a Robot whose floor masks are cached by frame index (shared by the variants), with a
    small grey copy of each image and its floor mask for the alignment; limited to frames [first, last]."""
    robot = importlib.import_module(cfg.robot.adapter).open_replay(cfg, run_dir)
    cam = robot.camera
    segment, read = cam.segment, cam.read

    def cached(frame):
        if frame.index not in cache:
            drive = segment(frame)
            grey = cv2.resize(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), (GW, GH), interpolation=cv2.INTER_AREA)
            floor = cv2.resize(drive.astype(np.uint8), (GW, GH), interpolation=cv2.INTER_NEAREST)
            floor = cv2.erode(floor, np.ones((5, 5), np.uint8)) > 0
            cache[frame.index] = (drive, frame.under, grey, floor)
        drive, frame.under = cache[frame.index][:2]
        return drive

    def limited():
        while True:
            frame = read()  # EOFError after the last
            if frames is None or frame.index >= frames[0]:
                break
        if frames is not None and frame.index > frames[1]:
            raise EOFError
        return frame

    cam.segment, cam.read = cached, limited
    return robot


def gravity_frame(up_c):
    """navette-core geometry::gravity_frame: rows = the g axes (x: optical axis on the floor, y left, z up) in
    the camera frame."""
    z = up_c / np.linalg.norm(up_c)
    x = np.array([-z[2] * z[0], -z[2] * z[1], 1.0 - z[2] * z[2]])
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def highpass(g):
    b = cv2.GaussianBlur(g, (0, 0), 1.0)
    return b - cv2.GaussianBlur(b, (0, 0), 4.0)


class Warp:
    """Frame j's image warped onto frame i through the floor plane."""

    def __init__(self, calib_path):
        with open(calib_path) as f:
            cal = json.load(f)
        self.K, self.D = np.array(cal["K"], float), np.array(cal["D"], float)
        self.W0, self.H0 = cal["image_resolution"]
        self.rays = mask_rays(calib_path, GW, GH)

    def ncc(self, gi, floor_i, si, gj, sj):
        """NCC at zero shift of the high-passed grey image i and image j warped onto it; si, sj: (pose, up_c,
        cam_h). None if too few floor pixels."""
        (xi, yi, ti), up_i, h_i = si
        (xj, yj, tj), up_j, h_j = sj
        g = self.rays @ gravity_frame(up_i).T
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(g[:, 2] < -1e-6, -h_i / g[:, 2], np.nan)
        fx, fy = s * g[:, 0], s * g[:, 1]  # floor point, g frame of i
        rng = np.hypot(fx, fy)
        c, sn = np.cos(ti), np.sin(ti)
        wx, wy = xi + c * fx - sn * fy, yi + sn * fx + c * fy  # fixed frame
        c, sn = np.cos(tj), np.sin(tj)
        dx, dy = wx - xj, wy - yj
        p_g = np.stack([c * dx + sn * dy, -sn * dx + c * dy, np.full_like(dx, -h_j)], 1)  # g frame of j
        p_c = np.nan_to_num(p_g) @ gravity_frame(up_j)  # camera frame of j (R_gc^T p)
        z = p_c[:, 2]
        ok = (z > 0.05) & np.isfinite(rng) & (rng > RANGE_M[0]) & (rng < RANGE_M[1]) & floor_i.ravel()
        zz = np.where(ok, z, 1.0)
        xn, yn = p_c[:, 0] / zz, p_c[:, 1] / zz
        ok &= (np.abs(xn) < 2.0) & (np.abs(yn) < 1.5)  # rational lens: no wrap-around outside its field
        uv, _ = cv2.projectPoints(np.stack([xn, yn, np.ones_like(xn)], 1).reshape(-1, 1, 3), np.zeros(3),
                                  np.zeros(3), self.K, self.D)
        uv = uv.reshape(-1, 2)
        u = (uv[:, 0] + 0.5) * GW / self.W0 - 0.5
        v = (uv[:, 1] + 0.5) * GH / self.H0 - 0.5
        ok &= (u > 0) & (u < GW - 1) & (v > 0) & (v < GH - 1)
        if ok.sum() < 1500:
            return None
        warped = cv2.remap(gj, u.reshape(GH, GW).astype(np.float32), v.reshape(GH, GW).astype(np.float32),
                           cv2.INTER_LINEAR)
        a, b = highpass(gi).ravel()[ok], highpass(warped).ravel()[ok]
        a, b = (a - a.mean()) / (a.std() + 1e-6), (b - b.mean()) / (b.std() + 1e-6)
        return float((a * b).mean())


def pairs(steps):
    """(k, kind) for consecutive steps k-1, k of the same navigator run; kind 'turn' or 'drive'."""
    for k in range(1, len(steps)):
        s0, s1 = steps[k - 1], steps[k]
        if s1["run"] != s0["run"] or not 0 < s1["t"] - s0["t"] <= MAX_PAIR_DT:
            continue
        if abs(s1["v"]) >= 0.05:
            yield k, "drive"
        elif abs(s1["w"]) > 0.2:
            yield k, "turn"


def alignment(steps_a, steps_b, cache, warp):
    """{kind: (NCC of A, NCC of B) per pair}: same frames, the variants' poses."""
    out = {"turn": [], "drive": []}
    for k, kind in pairs(steps_a):
        sa0, sa1, sb0, sb1 = steps_a[k - 1], steps_a[k], steps_b[k - 1], steps_b[k]
        gi, floor_i = cache[sa1["frame"]][2:]
        gi, gj = gi.astype(np.float32), cache[sa0["frame"]][2].astype(np.float32)
        na = warp.ncc(gi, floor_i, (sa1["pose"], sa1["up"], sa1["cam_h"]), gj, (sa0["pose"], sa0["up"], sa0["cam_h"]))
        nb = warp.ncc(gi, floor_i, (sb1["pose"], sb1["up"], sb1["cam_h"]), gj, (sb0["pose"], sb0["up"], sb0["cam_h"]))
        if na is not None and nb is not None:
            out[kind].append((na, nb))
    return out


def sharpness(navs):
    return int(sum((np.asarray(nav.grid.log_odds) > nav.grid.thr).sum() for nav in navs))


def summarise(per_variant, cache, warp):
    s = {}
    for name, (steps, navs) in per_variant.items():
        cuts = np.array([x["cut"] for x in steps]) if steps else np.zeros(0)
        s[name] = {"steps": len(steps), "occupied_cells": sharpness(navs),
                   "floor_cut_mean": float(cuts.mean()) if len(cuts) else 0.0, "steps_with_cut": int((cuts > 0).sum()),
                   "v_out_mean": float(np.mean([x["v_out"] for x in steps])) if steps else 0.0}
    a, b = per_variant.get("A", ([], []))[0], per_variant.get("B", ([], []))[0]
    if a and b and len(a) == len(b):
        s["B_vs_A"] = {}
        for kind, v in alignment(a, b, cache, warp).items():
            v = np.array(v).reshape(-1, 2)
            s["B_vs_A"][kind] = {"pairs": len(v), "ncc_A_p50": float(np.median(v[:, 0])) if len(v) else None,
                                 "ncc_B_p50": float(np.median(v[:, 1])) if len(v) else None,
                                 "B_better": float((v[:, 1] > v[:, 0]).mean()) if len(v) else None}
    c = per_variant.get("C", ([], []))[0]
    if b and c and len(b) == len(c):
        s["C_vs_B"] = {"steps_forward_B": sum(1 for x in b if x["v_out"] > 0),
                       "steps_forward_lost": sum(1 for x, y in zip(b, c)
                                                 if x["v_req"] > 0 and x["v_out"] > 0 and y["v_out"] <= 0)}
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, help="robot config (TOML); its floor model and calibration are used")
    ap.add_argument("--out", required=True, help="output folder (outside the repository: runs are private)")
    ap.add_argument("--frames", type=int, nargs=2, metavar=("FIRST", "LAST"), help="only these frame indices")
    ap.add_argument("--variants", default="ABC", help="which variants to run (default ABC)")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    base_cfg = config.load(args.robot)
    warp = Warp(base_cfg.adapter["calib"])
    summary = {}
    for run_dir in args.runs:
        run = os.path.basename(os.path.normpath(run_dir))
        cache, per_variant, t0 = {}, {}, time.monotonic()
        for name in args.variants:
            cfg = base_cfg.replace("perception", **VARIANTS[name])
            steps, navs = [], []

            def make_navigator(behaviour, forward_v, odometry, cfg=cfg, steps=steps, navs=navs):
                nav = loop.new_navigator(cfg, behaviour, forward_v, odometry)
                navs.append(nav)
                return Recorder(nav, steps, len(navs))

            robot = open_run(cfg, run_dir, cache, args.frames)
            try:
                rows = replay(robot, cfg, run_dir, os.path.join(args.out, name, run), make_navigator,
                              loop.default_up_filter, log=lambda m: None, video=False)
            finally:
                robot.close()
            # one steps.csv row per navigator step, in order: the frame and its 'up' (loop.STEP_COLS + ATT_COLS)
            cols = loop.STEP_COLS + loop.ATT_COLS
            fi, ui = cols.index("frame"), cols.index("up_x")
            for st, row in zip(steps, rows):
                st["frame"], st["up"] = int(row[fi]), np.array([float(x) for x in row[ui:ui + 3]])
            per_variant[name] = (steps, navs)
        summary[run] = summarise(per_variant, cache, warp)
        summary[run]["elapsed_s"] = round(time.monotonic() - t0, 1)
        print(run, json.dumps(summary[run], indent=1), flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
