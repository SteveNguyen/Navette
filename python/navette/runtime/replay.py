"""Replay of a recorded run through the navigator, on a laptop. Nothing is sent anywhere.

The adapter gives the recording back as a Robot (its open_replay): the same Camera / Imu / Base contract,
read from the run folder instead of the devices. A recorded robot differs from a live one in three ways:
  - Camera.read() returns the frames one after the other without waiting and raises EOFError after the last;
  - Imu.read() and Base.odometry() give what the robot had when the last frame read arrived;
  - its Imu may have an attribute `g`: the median |acc| of the whole recording (else it is measured on the
    first perception.up_g_s of samples, as on the robot).

A robot run's steps.csv is followed: only the frames it stepped, a fresh navigator at each START (its `run`
column), its v_prev (the command actually sent), its head-glance frames into the map only. Without one, every
frame is stepped and v_prev is the previous step's output. Behaviour, forward speed and dry run come from the
run's meta.json ("args"), else from the arguments.

Output folder: steps.csv (STEP_COLS; timings are this machine's) and, with OpenCV and frames that carry a
picture, live.mp4 (camera with the drivable floor | map).
"""
import csv
import json
import os
import time

import numpy as np

from navette.interfaces import Odometry
from navette.runtime.localization import OdometryLocalization
from navette.runtime.loop import STEP_COLS, BaseMotion, camera_pose, free_space, pitch_down_deg, step_row
from navette.runtime.up import LiveUp


def robot_steps(run_dir):
    """A robot run's steps.csv -> {frame: (navigator run id, v_prev the robot passed, state)}, or None (not a
    robot run)."""
    path = os.path.join(run_dir, "steps.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows or "run" not in rows[0]:
        return None
    return {int(r["frame"]): (int(r["run"]), float(r["v_prev"]), r.get("state", "")) for r in rows}


def run_args(run_dir):
    """The "args" of a run's meta.json ({} if none). A run of the Python prototype says behaviour "v3" for
    "explore"."""
    path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        args = json.load(f).get("args", {})
    if args.get("behaviour") == "v3":
        args["behaviour"] = "explore"
    return args


def replay(robot, cfg, run_dir, out_dir, make_navigator, make_up_filter, log=print):
    """make_navigator(behaviour, forward_v, odometry) -> navigator. Returns the steps.csv rows written."""
    camera, imu, base = robot.camera, robot.imu, robot.base
    args, follow = run_args(run_dir), robot_steps(run_dir)
    os.makedirs(out_dir, exist_ok=True)
    fs = free_space(camera)
    behaviour = args.get("behaviour", cfg.behaviour.default)
    forward_v = args.get("forward_v", cfg.behaviour.forward_v)
    odometry = base is not None and not args.get("dry_run", False)  # as loop.run

    def new_nav():
        return make_navigator(behaviour, forward_v, odometry)

    up = LiveUp(make_up_filter, cfg.perception.up_g_s, cfg.perception.imu_max_age_s, g=getattr(imu, "g", None))
    loc = OdometryLocalization()
    nav, run, motion = new_nav(), 0, BaseMotion()
    video, rows, t_first, t_prev, v_prev = _Video(out_dir), [], None, None, 0.0
    while True:
        try:
            frame = camera.read()
        except EOFError:
            break
        t, idx = frame.t, frame.index
        t_first = t if t_first is None else t_first
        up.add(imu.read())
        # no base in the recording: the pose stays at the origin
        odom = base.odometry() if base is not None else Odometry(t, 0.0, 0.0, 0.0, 0.0, 0.0)
        loc.update(odom)
        if follow is not None:  # a robot run: the frames it stepped, its navigator per START, its v_prev
            if idx not in follow:
                continue
            if follow[idx][0] != run:
                nav, run, t_prev = new_nav(), follow[idx][0], None
            v_prev = follow[idx][1]
        up_c = up.up_at(t)
        if up_c is None or loc.pose() is None:  # as the robot: no step
            continue
        cam_h, cam_heading = camera_pose(frame.T_base_cam, up_c, cfg.camera.body_z)
        x, y, th = loc.pose()
        pose = (x, y, th + cam_heading)
        c0 = time.perf_counter()
        floor = camera.segment(frame)
        c1 = time.perf_counter()
        scan = fs.scan(floor, up_c, cam_h)
        c2 = time.perf_counter()
        if follow is not None and follow[idx][2] == "GLANCE":  # a head-glance frame: map only
            nav.observe(t, pose, scan, cam_h)
            continue
        dt = 0.1 if t_prev is None else t - t_prev
        a, v, w = motion.update(odom, dt) if base is not None and base.balancing else (0.0, 0.0, 0.0)
        out = nav.step(t, dt, pose, scan, cam_h, a, v, w, v_prev)
        c3 = time.perf_counter()
        t_prev, v_prev = t, out.v  # (a robot run overrides v_prev from its steps.csv)
        pitch = pitch_down_deg(up_c)
        rows.append(step_row(t - t_first, idx, pose, cam_h, pitch, out,
                             ((c1 - c0) * 1e3, (c2 - c1) * 1e3, (c3 - c2) * 1e3)))
        video.write(frame, floor, out, cam_h, pitch, getattr(nav, "grid", None), pose, scan)
    video.close()
    with open(os.path.join(out_dir, "steps.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(STEP_COLS)
        wr.writerows(rows)
    c = {k: STEP_COLS.index(k) for k in ("cam_h", "seg_ms", "scan_ms", "nav_ms")}
    ms = np.array([[float(r[c[k]]) for k in ("seg_ms", "scan_ms", "nav_ms")] for r in rows]) if rows else np.zeros((1, 3))
    h = np.array([float(r[c["cam_h"]]) for r in rows]) if rows else np.zeros(1)
    log(f"{os.path.basename(os.path.normpath(run_dir))}: {len(rows)} steps (IMU {imu.name}, pose "
        f"{'odometry' if base is not None else 'fixed'}), camera height {np.median(h):.3f} m "
        f"({h.min():.3f}-{h.max():.3f}), ms median / p95: seg {np.median(ms[:, 0]):.1f} / "
        f"{np.percentile(ms[:, 0], 95):.1f}, scan {np.median(ms[:, 1]):.1f} / {np.percentile(ms[:, 1], 95):.1f}, "
        f"nav {np.median(ms[:, 2]):.1f} / {np.percentile(ms[:, 2], 95):.1f}")
    return rows


class _Video:
    """live.mp4 of the replay, if OpenCV is installed and the frames carry a picture."""

    def __init__(self, out_dir):
        self.path, self.vw = os.path.join(out_dir, "live.mp4"), None
        try:
            from navette.runtime import preview
        except ImportError:
            preview = None
        self.preview = preview

    def write(self, frame, floor, out, cam_h, pitch, grid, pose, scan):
        if self.preview is None or frame.image is None:
            return
        import cv2

        img = self.preview.view(frame.image, floor, frame.under, out, cam_h, pitch, grid, pose, scan)
        if self.vw is None:
            self.vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (img.shape[1], img.shape[0]))
        self.vw.write(img)

    def close(self):
        if self.vw is not None:
            self.vw.release()
