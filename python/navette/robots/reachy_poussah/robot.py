"""Reachy Mini on the Poussah base: the live robot behind the contract (navette.interfaces).

  camera   the Mini's camera (SDK), frames at half size, the floor model (ONNX) behind Camera.segment
  imu      the head IMU: imu.ImuSource (400 Hz FIFO reader, daemon IMU as the fallback), in camera axes
  base     poussah_base.PoussahBase over Bluetooth LE or USB; None with use_base=False
  head     a glance down at adapter.glance_pitch_deg, if not 0
  stop     pushing an antenna is a STOP

The Reachy Mini SDK, OpenCV, onnxruntime, bleak and pyserial are imported inside the functions that need
them: this module imports on a laptop without them.

Recordings, in the run folder (the replay reads them: recording.py):
  frames/NNNNNN.jpg, frames.csv   every frame read, at half size; idx, t = arrival time (capture + the
                                  camera's lag, adapter.camera_latency_s)
  state.jsonl                     SDK IMU / head pose / joints, changed samples only, polled at 200 Hz
  imu_fifo.jsonl, imu_fifo.out    the FIFO reader's log and its stdout
  base.jsonl                      every line received from / sent to the base (poussah_link.LoggedTransport)
and in meta.json under "robot": SDK version, camera intrinsics at the saved size, T_head_cam, head pose at the
start, imu_source, imu_fallback_t (host time the daemon IMU took over, or null), cs_latency_ms_median.
"""
import collections
import csv
import importlib.metadata
import json
import os
import threading
import time

import numpy as np

from navette.interfaces import Frame
from navette.robots.reachy_poussah.imu import ImuSource

SCALE = 0.5             # frames saved and processed at half size (1280x720 -> 640x360)
JPEG_QUALITY = 90
STATE_POLL_HZ = 200.0
GLANCE_MOVE_S = 0.5     # head move duration of a glance (each way)
ANTENNA_REST = [-0.1745, 0.1745]  # rad, the SDK's rest position of the antennas
PUSH_RAD = 0.4          # an antenna more than ~23 deg from rest is pushed
MASK_W, MASK_H = 320, 192  # floor model input and output size
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255  # RGB, ImageNet (the floor model's preprocessing)
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


class FloorModel:
    """ONNX floor model, no spinning (idle threads would burn the CPU the rest of the loop needs). threads: on
    the CM4 1 -> 86 ms, 2 -> 60, 3 -> 56, 4 -> 62: the robot uses 2, leaving cores to the scan, the IMU reader
    and the daemon."""

    def __init__(self, path, thr, under_thr, threads=1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.thr, self.under_thr = thr, under_thr

    def __call__(self, bgr):
        """(drivable mask, under mask or None) at mask size (MASK_H x MASK_W)."""
        import cv2

        from navette import drive_mask, seg_probs

        x = cv2.resize(bgr, (MASK_W, MASK_H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32)
        z = self.sess.run(None, {"input": ((x - MEAN) / STD).transpose(2, 0, 1)[None].copy()})[0][0]
        p_floor, p_under = seg_probs(z)
        under = None if p_under is None else (p_floor > self.thr) & (p_under > self.under_thr)
        return drive_mask(p_floor, p_under, self.thr, self.under_thr), under


def load_calib(adapter):
    """(R_cam_imu, T_head_cam fitted, rays, rays_low) from the calibration files of the config's [adapter]."""
    from navette.rays import mask_rays

    with open(adapter["r_cam_imu"]) as f:
        R_cam_imu = np.array(json.load(f)["R_cam_imu"])
    with open(adapter["handeye"]) as f:
        X = np.array(json.load(f)["T_head_cam_fitted"])
    return R_cam_imu, X, mask_rays(adapter["calib"], MASK_W, MASK_H), mask_rays(adapter["calib"], MASK_W, MASK_H, dv=0.5)


class ReachyCamera:
    """Camera contract on the Mini's camera. read(): the frame, stamped at its arrival, the measured head
    pose, and the frame saved (every frame read is saved: the run replays on a laptop). segment(): the floor
    model."""

    mask_shape = (MASK_H, MASK_W)

    def __init__(self, mini, model, X, rays, rays_low, out_dir, latency_s):
        self.mini, self.model, self.X, self.rays, self.rays_low = mini, model, X, rays, rays_low
        self.out_dir, self.latency_s, self.idx = out_dir, latency_s, 0
        os.makedirs(os.path.join(out_dir, "frames"))
        self.frames_f = open(os.path.join(out_dir, "frames.csv"), "w", newline="", buffering=1)
        self.frames_w = csv.writer(self.frames_f)
        self.frames_w.writerow(["idx", "t"])

    def read(self):
        import cv2

        raw = self.mini.media.get_frame()  # blocks up to ~20 ms; None: no new frame
        if raw is None:
            return None
        t = time.monotonic()  # stamp first: the frame's arrival
        w0 = time.perf_counter()
        T_head = np.asarray(self.mini.get_current_head_pose(), float)
        w1 = time.perf_counter()
        bgr = cv2.resize(raw, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(self.out_dir, "frames", f"{self.idx:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        self.frames_w.writerow([self.idx, f"{t:.6f}"])
        w2 = time.perf_counter()
        frame = Frame(t - self.latency_s, self.idx, T_head @ self.X, image=bgr,
                      ms={"head": (w1 - w0) * 1e3, "save": (w2 - w1) * 1e3})
        self.idx += 1
        return frame

    def segment(self, frame):
        drive, frame.under = self.model(frame.image)
        return drive

    def close(self):
        self.frames_f.close()


class StateRecorder(threading.Thread):
    """The SDK's IMU, head pose and joints to state.jsonl (200 Hz polling, changed samples only), keeping the
    daemon IMU samples in memory as well (ImuSource's fallback). Line-buffered file."""

    def __init__(self, mini, out_dir, log=print):
        super().__init__(daemon=True)
        self.mini, self.path, self.log = mini, os.path.join(out_dir, "state.jsonl"), log
        self.stop_event = threading.Event()
        self.imu = collections.deque(maxlen=200)  # (t, gyro rad/s, acc m/s^2)
        self.errors = 0
        self._lock = threading.Lock()

    def imu_after(self, t):
        with self._lock:
            return [s for s in self.imu if s[0] > t]

    def _poll(self, f, last):
        t = time.monotonic()
        imu = self.mini.imu  # dict or None
        samples = []
        if imu is not None:
            samples.append(("imu", {"accelerometer": imu["accelerometer"], "gyroscope": imu["gyroscope"],
                                    "quaternion": imu["quaternion"]}))
        samples.append(("head_pose", {"pose": self.mini.get_current_head_pose().tolist()}))
        head_joints, antennas = self.mini.get_current_joint_positions()
        samples.append(("joints", {"head": head_joints, "antennas": antennas}))
        for kind, payload in samples:
            if payload != last.get(kind):
                last[kind] = payload
                f.write(json.dumps({"t": round(t, 6), "kind": kind, **payload}) + "\n")
                if kind == "imu":
                    with self._lock:
                        self.imu.append((t, np.array(payload["gyroscope"], float),
                                         np.array(payload["accelerometer"], float)))

    def run(self):
        period, last = 1.0 / STATE_POLL_HZ, {}
        with open(self.path, "w", buffering=1) as f:
            next_tick = time.monotonic()
            while not self.stop_event.is_set():
                try:
                    self._poll(f, last)
                except Exception as e:  # one SDK hiccup must not end the IMU fallback and the state log
                    self.errors += 1
                    if self.errors <= 5 or self.errors % 1000 == 0:
                        self.log(f"state poll failed ({self.errors}): {e!r}")
                next_tick += period
                sleep = next_tick - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.monotonic()


def antenna_pushed(positions, rest=ANTENNA_REST, thr=PUSH_RAD):
    """True if an antenna is more than thr (rad) from its rest position."""
    return max(abs(p - r) for p, r in zip(positions, rest)) > thr


class AntennaPush:
    """Rising edge of antenna_pushed: one STOP per push, not one per poll while it is held."""

    def __init__(self):
        self.was = False

    def update(self, positions):
        now = antenna_pushed(positions)
        edge, self.was = now and not self.was, now
        return edge


def antenna_loop(mini, stop, stop_event, log):
    """stop(reason): the controller's STOP."""
    push = AntennaPush()
    while not stop_event.wait(0.05):
        try:
            if push.update(mini.get_present_antenna_joint_positions()):
                log("antenna pushed: STOP")
                stop("antenna")
        except Exception as e:  # an SDK hiccup must not end the STOP button
            log(f"antenna read failed: {e!r}")


def measured_rpy(mini):
    """Measured head roll, pitch, yaw (deg); pitch > 0: looking down."""
    m = mini.get_current_head_pose()
    return np.degrees([np.arctan2(m[2, 1], m[2, 2]), -np.arcsin(np.clip(m[2, 0], -1, 1)), np.arctan2(m[1, 0], m[0, 0])])


def set_nav_head(mini, pitch_deg, log):
    """ROBOT MOTION. Head to pitch_deg down, yaw 0, roll 0, body yaw 0, antennas at rest. Returns the measured
    roll / pitch / yaw (deg).
    No levelling loop on the measured pose: folding the measured roll / pitch error back into the command made
    the roll worse at each iteration on a head with a weak motor (0.5 -> 2.1, 2.2 -> 5.8 deg: hysteresis and
    sag, not a sign error). The camera's 'up' and height come from the IMU and the measured pose anyway."""
    from reachy_mini.utils import create_head_pose

    pose = create_head_pose(roll=0.0, pitch=pitch_deg, yaw=0.0, degrees=True)
    mini.goto_target(head=pose, antennas=ANTENNA_REST, duration=2.0, body_yaw=0.0)
    time.sleep(1.0)  # let the head settle
    r, p, y = measured_rpy(mini)
    log(f"head: measured r/p/y {r:.1f}/{p:.1f}/{y:.1f} deg")
    return [float(r), float(p), float(y)]


class ReachyHead:
    """Head contract: glance down to glance_pitch_deg and back to nav_pitch_deg. ROBOT MOTION (head), also in
    a dry run. Before turning it on, check that the base does not show at the image bottom at that pitch (it
    would read as an obstacle), and see the config's note on the balancing base."""

    move_s = GLANCE_MOVE_S

    def __init__(self, mini, glance_pitch_deg, nav_pitch_deg, log):
        self.mini, self.glance_pitch, self.nav_pitch, self.log = mini, glance_pitch_deg, nav_pitch_deg, log

    def _move(self, pitch_deg):
        """Head to pitch_deg down (yaw, roll 0) in move_s, in a thread: goto_target blocks."""
        from reachy_mini.utils import create_head_pose

        def go():
            try:
                self.mini.goto_target(head=create_head_pose(roll=0.0, pitch=pitch_deg, yaw=0.0, degrees=True),
                                      antennas=ANTENNA_REST, duration=self.move_s, body_yaw=0.0)
            except Exception as e:  # logged; the glance times out and the loop goes on
                self.log(f"head move to {pitch_deg} deg failed: {e!r}")
        threading.Thread(target=go, daemon=True).start()

    def look_down(self):
        self._move(self.glance_pitch)

    def look_back(self):
        self._move(self.nav_pitch)

    def is_back(self):
        """Measured head pitch within 5 deg of the navigation pose."""
        return abs(measured_rpy(self.mini)[1] - self.nav_pitch) < 5.0


def sdk_meta(mini):
    """Camera and SDK data for meta.json (intrinsics scaled to the saved frames)."""
    cam = mini.media.camera
    K = np.array(cam.K, dtype=float)
    K[:2, :] *= SCALE
    width, height = cam.resolution
    return {"sdk_version": importlib.metadata.version("reachy_mini"), "camera_native_resolution": [width, height],
            "camera_framerate": cam.framerate, "scale": SCALE,
            "image_resolution": [round(width * SCALE), round(height * SCALE)], "K": K.tolist(),
            "D": np.asarray(cam.D, dtype=float).tolist(),
            "T_head_cam": np.asarray(mini.T_head_cam, dtype=float).tolist(), "state_poll_hz": STATE_POLL_HZ}


def connect_base(adapter, out_dir, dry_run, log):
    """PoussahBase over BLE / USB, logged to base.jsonl, read-only in a dry run."""
    from navette.robots.reachy_poussah import poussah_link as pl
    from navette.robots.reachy_poussah.poussah_base import PoussahBase

    kind = adapter["base_link"]
    inner = pl.BleTransport(adapter["ble_name"]) if kind == "ble" else pl.SerialTransport(adapter["usb_port"])
    link = pl.BaseLink(pl.LoggedTransport(inner, os.path.join(out_dir, "base.jsonl"), dry_run),
                       rate=adapter["nav_rate_hz"], link_name=kind)
    log(f"base: connecting over {kind}" + (" (dry run: read-only)" if dry_run else ""))
    link.connect()
    t_end = time.monotonic() + 3.0
    while time.monotonic() < t_end and (link.hello is None or link.latest()[0] is None):
        time.sleep(0.05)
    nav = link.latest()[0]
    log(f"base: hello {link.hello.kv if link.hello else None}; first [nav] st {None if nav is None else nav.st}")
    if link.hello is None or link.hello.r is None:
        log("base: WARNING no [hello] with r / track: no odometry pose, the nav steps are skipped")
    ok, why = link.hello_ok()
    if not ok:
        log(f"base: WARNING {why}: START will be refused (except in a dry run)")
    return PoussahBase(link, dry_run, log=log)


class ReachyPoussah:
    """The Robot (navette.interfaces). Built by open_robot."""

    def __init__(self, mini, camera, imu, base, head, recorder, meta, log, close_mini=None):
        self.mini, self.camera, self.imu, self.base, self.head = mini, camera, imu, base, head
        self.recorder, self._meta, self.log, self._close_mini = recorder, meta, log, close_mini
        self._threads_stop = threading.Event()

    def on_stop(self, stop):
        threading.Thread(target=antenna_loop, args=(self.mini, stop, self._threads_stop, self.log), daemon=True).start()

    def meta(self):
        link = self.base.link if self.base is not None else None
        return {**self._meta, "imu_source": self.imu.name, "imu_fallback_t": self.imu.t_fallback,
                "cs_latency_ms_median": (float(np.median(link.cs_latencies)) * 1e3
                                         if link is not None and link.cs_latencies else None)}

    def close(self):
        self._threads_stop.set()
        self.imu.stop()
        self.recorder.stop_event.set()
        self.recorder.join(2.0)
        if self.base is not None:
            try:
                self.base.link.close()
            except Exception as e:
                self.log(f"base close: {e!r}")
        self.camera.close()
        if self._close_mini is not None:
            self._close_mini()


def open_robot(cfg, out_dir, dry_run=False, use_base=True, log=print, mini=None):
    """The live robot. mini: a connected ReachyMini (the Reachy Mini app passes its own), or None: one is
    opened here and closed by close(). ROBOT MOTION: the head goes to its navigation pose, also in a dry run.
    Must be called on the thread that runs the loop (ImuSource: the IMU reader is tied to that thread's life)."""
    a = cfg.adapter
    # model and calibration first: a bad path fails before anything moves
    R_cam_imu, X, rays, rays_low = load_calib(a)
    model = FloorModel(a["model"], a["floor_thr"], a["under_thr"], threads=a["model_threads"])
    close_mini = None
    if mini is None:
        from reachy_mini import ReachyMini

        ctx = ReachyMini(connection_mode="localhost_only")
        mini = ctx.__enter__()
        close_mini = lambda: ctx.__exit__(None, None, None)  # noqa: E731
    recorder = imu = base = None
    try:
        meta = {"script": "navette.robots.reachy_poussah", **sdk_meta(mini)}
        camera = ReachyCamera(mini, model, X, rays, rays_low, out_dir, a["camera_latency_s"])
        recorder = StateRecorder(mini, out_dir, log)
        recorder.start()
        imu = ImuSource(out_dir, recorder, log, R=R_cam_imu, python=a["daemon_python"])
        log("head: setting the nav pose (ROBOT MOTION" + (", also in a dry run)" if dry_run else ")"))
        meta["head_rpy_measured"] = set_nav_head(mini, a["nav_pitch_deg"], log)
        if use_base:
            base = connect_base(a, out_dir, dry_run, log)
        head = None
        if a["glance_pitch_deg"] > 0:
            head = ReachyHead(mini, a["glance_pitch_deg"], a["nav_pitch_deg"], log)
        return ReachyPoussah(mini, camera, imu, base, head, recorder, meta, log, close_mini)
    except BaseException:
        if imu is not None:
            imu.stop()
        if recorder is not None:
            recorder.stop_event.set()
        if close_mini is not None:
            close_mini()
        raise
