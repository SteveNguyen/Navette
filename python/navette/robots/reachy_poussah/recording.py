"""A recorded run of the Reachy Mini on the Poussah base, given back as a Robot for the replay
(navette.runtime.replay): the files robot.py writes (also the Python prototype's run folders, same format).

What the replayed robot "had" at each frame, as on the robot:
  camera   frames.csv / frames/: Frame.t = the recorded arrival time minus the camera's lag; head pose: the
           last one logged at or before the capture time (frames before the first head pose are skipped)
  imu      imu_fifo.jsonl (400 Hz) when it holds data, else the daemon's 50 Hz IMU in state.jsonl; and the
           daemon's from meta imu_fallback_t on if the robot switched. Samples up to the frame's arrival.
  base     base.jsonl: the received lines are fed to a BaseLink in arrival order up to the frame's arrival,
           so the odometry is the one the robot had. No base.jsonl: no base (the pose stays at the origin).
"""
import csv
import json
import os

import numpy as np

from navette.interfaces import Frame
from navette.robots.reachy_poussah.imu import load_imu_fifo


def has_fifo(run_dir):
    """imu_fifo.jsonl present with data (a run whose FIFO reader failed may leave a header only)."""
    path = os.path.join(run_dir, "imu_fifo.jsonl")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return sum(1 for _ in zip(range(3), f)) >= 3  # header + 2 reads: enough to fit the sample clock


class Recording:
    """The files of a run folder. fifo: take the IMU from imu_fifo.jsonl (400 Hz, acc resampled at the gyro
    times) instead of the 50 Hz daemon IMU of state.jsonl."""

    def __init__(self, run_dir, fifo=False):
        self.dir = run_dir
        with open(os.path.join(run_dir, "meta.json")) as f:
            self.meta = json.load(f)
        with open(os.path.join(run_dir, "frames.csv")) as f:
            rows = list(csv.DictReader(f))
        self.frame_idx = np.array([int(r["idx"]) for r in rows])
        self.frame_t = np.array([float(r["t"]) for r in rows])

        t, acc, gyro, pose_t, poses = [], [], [], [], []
        with open(os.path.join(run_dir, "state.jsonl")) as f:
            for line in f:
                rec = json.loads(line)
                if rec["kind"] == "imu":
                    t.append(rec["t"])
                    acc.append(rec["accelerometer"])
                    gyro.append(rec["gyroscope"])
                elif rec["kind"] == "head_pose":
                    pose_t.append(rec["t"])
                    poses.append(rec["pose"])
        self.imu_t = np.array(t)
        self.acc = np.array(acc, float).reshape(-1, 3)
        self.gyro = np.array(gyro, float).reshape(-1, 3)
        if fifo:
            t_g, g, t_a, a, _ = load_imu_fifo(os.path.join(run_dir, "imu_fifo.jsonl"))
            self.imu_t, self.gyro = t_g, g
            self.acc = np.stack([np.interp(t_g, t_a, a[:, j]) for j in range(3)], axis=1)
        # measured head pose (encoder forward kinematics, body frame), logged on change only
        self.pose_t = np.array(pose_t)
        self.head_pose = np.array(poses).reshape(-1, 4, 4)

    def head_pose_at(self, t):
        """Latest measured head pose at or before t (samples are logged on change), or None."""
        i = np.searchsorted(self.pose_t, t, side="right") - 1
        return self.head_pose[i] if i >= 0 else None


def replay_imu(run_dir, rec, fifo):
    """(t, gyro, acc) the robot had: the FIFO reader's samples, then the daemon's from meta 'imu_fallback_t' on
    if the robot switched. No switch: the recording's own IMU."""
    t_fb = rec.meta.get("robot", rec.meta).get("imu_fallback_t")
    if not fifo or t_fb is None:
        return rec.imu_t, rec.gyro, rec.acc
    d = Recording(run_dir)  # daemon IMU from state.jsonl
    a, b = rec.imu_t < t_fb, d.imu_t > t_fb
    return (np.r_[rec.imu_t[a], d.imu_t[b]], np.r_[rec.gyro[a], d.gyro[b]], np.r_[rec.acc[a], d.acc[b]])


class _Now:
    """Arrival time of the frame being replayed: what the recorded Imu and Base are read up to."""
    t = -np.inf


class ReplayCamera:
    def __init__(self, rec, now, model, X, rays, rays_low, mask_shape, latency_s):
        self.rec, self.now, self.model, self.X, self.latency_s = rec, now, model, X, latency_s
        self.rays, self.rays_low, self.mask_shape = rays, rays_low, mask_shape
        self.k = 0

    def read(self):
        while self.k < len(self.rec.frame_idx):
            idx, t = int(self.rec.frame_idx[self.k]), float(self.rec.frame_t[self.k])
            self.k += 1
            # head pose at the frame's capture time (t is its arrival): the heading can move fast
            T_head = self.rec.head_pose_at(t - self.latency_s)
            if T_head is None:
                continue
            self.now.t = t
            return Frame(t - self.latency_s, idx, T_head @ self.X)
        raise EOFError

    def segment(self, frame):
        import cv2

        frame.image = cv2.imread(os.path.join(self.rec.dir, "frames", f"{frame.index:06d}.jpg"))
        drive, frame.under = self.model(frame.image)
        return drive


class ReplayImu:
    def __init__(self, t, gyro, acc, now, R, name):
        self.t, self.now, self.name, self.i = t, now, name, 0
        self.gyro, self.acc = gyro @ R.T, acc @ R.T  # IMU axes -> camera axes
        # |g| of the whole recording (runtime.replay); None: a recording without IMU, nothing is stepped
        self.g = float(np.median(np.linalg.norm(acc, axis=1))) if len(acc) else None

    def read(self):
        j = int(np.searchsorted(self.t, self.now.t, side="right"))  # samples up to the frame's arrival
        out = list(zip(self.t[self.i:j], self.gyro[self.i:j], self.acc[self.i:j]))
        self.i = max(self.i, j)
        return out


class ReplayBase:
    """The PoussahBase's odometry from the recorded lines; nothing else of the Base contract is used by a
    replay."""

    balancing = True

    def __init__(self, run_dir, now):
        from navette.robots.reachy_poussah import poussah_link as pl
        from navette.robots.reachy_poussah.poussah_base import PoussahBase

        with open(os.path.join(run_dir, "base.jsonl")) as f:
            self.rx = [(r["t"], r["rx"]) for r in map(json.loads, f) if "rx" in r]  # only what was received
        self.link, self.now, self.i = pl.BaseLink(pl.NoTransport()), now, 0
        self.base = PoussahBase(self.link, log=lambda m: None)

    def odometry(self):
        # lines received up to the frame's arrival: the robot reads the odometry right then
        while self.i < len(self.rx) and self.rx[self.i][0] <= self.now.t:
            self.link.feed(*self.rx[self.i])
            self.i += 1
        return self.base.odometry()


class ReplayRobot:
    head = None

    def __init__(self, camera, imu, base):
        self.camera, self.imu, self.base = camera, imu, base

    def on_stop(self, stop):
        pass

    def meta(self):
        return {}

    def close(self):
        pass


def open_replay_robot(cfg, run_dir, model=None, calib=None):
    """The recorded run as a Robot. model: bgr -> (drivable mask, under mask or None), default the config's
    floor model; calib: (R_cam_imu, T_head_cam, rays, rays_low), default the config's calibration files."""
    from navette.robots.reachy_poussah import robot as rb

    a = cfg.adapter
    R_cam_imu, X, rays, rays_low = calib if calib is not None else rb.load_calib(a)
    if model is None:
        model = rb.FloorModel(a["model"], a["floor_thr"], a["under_thr"])
    fifo = has_fifo(run_dir)
    rec = Recording(run_dir, fifo=fifo)
    now = _Now()
    camera = ReplayCamera(rec, now, model, X, rays, rays_low, (rb.MASK_H, rb.MASK_W), a["camera_latency_s"])
    imu = ReplayImu(*replay_imu(run_dir, rec, fifo), now, R_cam_imu, "fifo" if fifo else "daemon")
    base = ReplayBase(run_dir, now) if os.path.exists(os.path.join(run_dir, "base.jsonl")) else None
    return ReplayRobot(camera, imu, base)
