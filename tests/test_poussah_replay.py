"""navette.runtime.replay on a synthetic Reachy Mini / Poussah run folder (recording.py): pose from base.jsonl,
IMU from imu_fifo.jsonl when present, the robot's steps followed. The navigator and the attitude filter are
test doubles (navette.testing.fakes)."""
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from navette import config  # noqa: E402
from navette.rays import MASK_H, MASK_W, mask_rays  # noqa: E402
from navette.robots.reachy_poussah import recording  # noqa: E402
from navette.runtime.replay import replay as run_replay  # noqa: E402
from navette.testing.fakes import FakeNavigator, FakeUpFilter  # noqa: E402

ROBOT_TOML = Path(__file__).parents[1] / "robots" / "reachy_poussah.toml"
R, TRACK, V = 0.0444, 0.16, 0.2
HELLO = f"[hello] proto=1 fw=abc name=rmini_wheels_6 r={R} track={TRACK} vmax=0.40 wmax=1.00 deadman_ms=500"
# camera looking straight ahead from the head origin (T_head_cam): optical z = body x, image y = body -z
X = np.array([[0, 0, 1, 0.0437], [-1, 0, 0, 0], [0, -1, 0, 0.0512], [0, 0, 0, 1.0]])
R_CAM_IMU = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])  # IMU z (up) -> camera -y (up)
FRAMES_T = 10.0 + 0.1 * np.arange(10)


def make_run(d, base=True, fifo=False, fallback_t=None, steps=None, args=None):
    """A recording in the robot run format: 10 frames at 10 Hz from t = 10 s, still IMU, head pose identity;
    base.jsonl: the base driving forward at V from t = 9.6 s (wheel angles as the firmware signs them).
    fallback_t: the robot switched to the daemon IMU then (meta imu_fallback_t); state.jsonl then holds a daemon
    IMU tilted by 10 deg (camera looking up), so the switch shows in the pitch. steps: robot steps.csv rows
    {frame: (run, v_prev)} or {frame: (run, v_prev, state)}. args: meta.json "args"."""
    os.makedirs(d / "frames")
    meta = {"robot": {"imu_fallback_t": fallback_t}}
    if args is not None:
        meta["args"] = args
    json.dump(meta, open(d / "meta.json", "w"))
    with open(d / "frames.csv", "w") as f:
        f.write("idx,t\n" + "".join(f"{i},{t:.6f}\n" for i, t in enumerate(FRAMES_T)))
    for i in range(len(FRAMES_T)):
        cv2.imwrite(str(d / "frames" / f"{i:06d}.jpg"), np.full((360, 640, 3), 128, np.uint8))
    tilt = np.radians(10 if fallback_t is not None else 0)
    with open(d / "state.jsonl", "w") as f:
        f.write(json.dumps({"t": 9.0, "kind": "head_pose", "pose": np.eye(4).tolist()}) + "\n")
        if not fifo or fallback_t is not None:
            for t in np.arange(8.5, 11.1, 0.02):
                f.write(json.dumps({"t": round(float(t), 4), "kind": "imu",
                                    "accelerometer": [0, 9.81 * np.sin(tilt), 9.81 * np.cos(tilt)],
                                    "gyroscope": [0, 0, 0], "quaternion": [1, 0, 0, 0]}) + "\n")
    if steps is not None:
        req = any(len(v) > 3 for v in steps.values())  # (run, v_prev, state, v_req, w_req): with the request
        with open(d / "steps.csv", "w") as f:
            f.write("frame,run,v_prev,state" + (",v_req,w_req" if req else "") + "\n"
                    + "".join(f"{k},{v[0]},{v[1]},{v[2] if len(v) > 2 else 'DRIVE'}"
                              + (f",{v[3]},{v[4]}" if req else "") + "\n" for k, v in steps.items()))
    if fifo:
        with open(d / "imu_fifo.jsonl", "w") as f:
            f.write(json.dumps({"header": True, "gyro_lsb_per_dps": 65.536, "acc_lsb_per_g": 10920,
                                "g": 9.80665}) + "\n")
            for k, t in enumerate(np.arange(8.5, 11.1, 0.02)):
                n = 8  # 400 Hz
                f.write(json.dumps({"t0": float(t), "t": float(t) + 0.002, "g": [0, 0, 0] * n, "gov": False,
                                    "a": [0, 0, 10920] * n, "st": [], "an": 7 * n}) + "\n")
    if base:
        wl = -V / R  # both wheels turn 'backward' in their own sign for a forward drive (poussah_link.Odometry)
        with open(d / "base.jsonl", "w") as f:
            f.write(json.dumps({"t": 9.5, "tx": "nh"}) + "\n")
            f.write(json.dumps({"t": 9.55, "rx": HELLO}) + "\n")
            for k, t in enumerate(np.arange(9.6, 11.1, 0.02)):
                p = wl * (t - 9.6)
                line = (f"[nav] t={int(t * 1e6)} seq={k} st=2 tilt=0 rate=0 roll=0 wl={wl:.3f} wr={wl:.3f} "
                        f"pl={p:.4f} pr={p:.4f} ax=0 az=9.81 vc={V} wc=0 cs=0 f=7")
                f.write(json.dumps({"t": round(float(t), 4), "rx": line}) + "\n")


def all_floor(bgr):
    return np.ones((MASK_H, MASK_W), bool), None


def ideal_calib(tmp_path):
    """Calibration file of an ideal lens with the robot camera's focal length, 1280x720."""
    p = tmp_path / "calib.json"
    p.write_text(json.dumps({"image_resolution": [1280, 720], "K": [[822.0, 0, 640.0], [0, 822.0, 360.0], [0, 0, 1]],
                             "D": [0.0] * 5}))
    return str(p)


def replay(tmp_path, navs=None, **kw):
    """Replay a synthetic run; returns the steps.csv rows. navs: list that receives the navigators made, as
    (navigator, behaviour, forward_v, odometry)."""
    run = tmp_path / "run"
    make_run(run, **kw)
    cfg = config.load(ROBOT_TOML).replace("camera", body_z=0.3)
    calib = ideal_calib(tmp_path)
    robot = recording.open_replay_robot(cfg, str(run), model=all_floor,
                                        calib=(R_CAM_IMU, X, mask_rays(calib), mask_rays(calib, dv=0.5)))

    def make_navigator(behaviour, forward_v, odometry):
        nav = FakeNavigator(odometry)
        if navs is not None:
            navs.append((nav, behaviour, forward_v, odometry))
        return nav

    run_replay(robot, cfg, str(run), str(tmp_path / "out"), make_navigator, FakeUpFilter, log=lambda m: None)
    with open(tmp_path / "out" / "steps.csv") as f:
        return list(csv.DictReader(f))


def test_replay_uses_the_recorded_odometry(tmp_path):
    navs = []
    rows = replay(tmp_path, navs=navs)
    assert len(rows) == len(FRAMES_T)
    t = FRAMES_T[[int(r["frame"]) for r in rows]]
    x = np.array([float(r["x"]) for r in rows])
    # the last [nav] at or before each frame (50 Hz lines): V * (t - 9.6) within one line
    assert x == pytest.approx(V * (t - 9.6), abs=V * 0.021)
    assert all(abs(float(r["y"])) < 1e-3 and abs(float(r["heading_deg"])) < 0.1 for r in rows)
    assert float(rows[0]["cam_h"]) == pytest.approx(0.3 + 0.0512, abs=1e-3)
    assert os.path.exists(tmp_path / "out" / "live.mp4")
    # the measured base motion reaches the navigator (a balancing base), and the capture time is the frame's
    # arrival minus the camera's lag
    nav = navs[0][0]
    assert navs[0][1:] == ("explore", 0.15, True) and len(navs) == 1
    assert [s[6] for s in nav.steps] == pytest.approx([V] * len(rows), abs=1e-3)
    assert [s[0] for s in nav.steps] == pytest.approx(FRAMES_T - 0.038)
    assert [s[1] for s in nav.steps[1:]] == pytest.approx([0.1] * (len(rows) - 1))


def test_replay_without_base_keeps_the_pose_fixed(tmp_path):
    navs = []
    rows = replay(tmp_path, navs=navs, base=False)
    assert len(rows) == len(FRAMES_T) and all(float(r["x"]) == 0 and float(r["y"]) == 0 for r in rows)
    assert navs[0][3] is False  # no odometry: the navigator must not build a map on it
    # without the robot's steps, v_prev is the previous step's output
    assert [s[8] for s in navs[0][0].steps] == pytest.approx([0.0] + [0.1] * (len(rows) - 1))


def test_replay_follows_the_robot_steps(tmp_path):
    """The robot's stepped frames only, a fresh navigator where its run id changes (START), its v_prev."""
    navs = []
    steps = {k: (0 if k < 5 else 1, 0.01 * k) for k in range(10) if k != 3}  # frame 3 skipped by the robot
    rows = replay(tmp_path, navs=navs, steps=steps)
    assert [int(r["frame"]) for r in rows] == sorted(steps)
    assert len(navs) == 2  # the initial navigator, then one at the START (run 0 -> 1)
    v_prevs = [s[8] for nav, *_ in navs for s in nav.steps]
    assert v_prevs == pytest.approx([steps[k][1] for k in sorted(steps)])
    assert len(navs[0][0].steps) == 4 and len(navs[1][0].steps) == 5
    assert [s[1] for s in navs[0][0].steps] == pytest.approx([0.1, 0.1, 0.1, 0.2])  # dt spans the skipped frame


def test_replay_of_an_old_run_writes_the_camera_attitude(tmp_path):
    """A run recorded before the attitude columns (its steps.csv has none of ATT_COLS, nor most LIVE_COLS)
    replays; the replay's steps.csv has them last: the camera looking straight ahead, level: no roll, 'up' =
    -y (image up)."""
    from navette.runtime.loop import ATT_COLS, STEP_COLS

    rows = replay(tmp_path, steps={k: (0, 0.0) for k in range(10)})
    assert len(rows) == 10 and list(rows[0]) == STEP_COLS + ATT_COLS
    with open(tmp_path / "run" / "steps.csv") as f:
        assert not set(ATT_COLS) & set(next(csv.reader(f)))  # the recorded run had none
    for r in rows:
        assert float(r["roll_deg"]) == pytest.approx(0.0, abs=0.5)
        assert [float(r[k]) for k in ("up_x", "up_y", "up_z")] == pytest.approx([0, -1, 0], abs=0.01)


def test_replay_puts_glance_frames_into_the_map_only(tmp_path):
    navs = []
    steps = {k: (0, 0.0, "GLANCE" if k in (2, 3) else "DRIVE") for k in range(10)}
    rows = replay(tmp_path, navs=navs, steps=steps)
    nav = navs[0][0]
    assert [int(r["frame"]) for r in rows] == [0, 1, 4, 5, 6, 7, 8, 9]
    assert len(nav.observed) == 2 and len(nav.steps) == 8


def test_replay_uses_the_runs_own_behaviour_and_dry_run(tmp_path):
    """meta.json args: the robot's behaviour and forward speed; a dry run had no odometry (carried robot);
    a run of the Python prototype says 'v3' for 'explore'."""
    navs = []
    replay(tmp_path, navs=navs, args={"behaviour": "forward", "forward_v": 0.2, "dry_run": True})
    assert navs[0][1:] == ("forward", 0.2, False)
    navs = []
    replay(tmp_path / "b", navs=navs, args={"behaviour": "v3", "base": "ble"})
    assert navs[0][1:] == ("explore", 0.15, True)


def test_replay_of_a_teleop_run_gives_the_operators_requests(tmp_path):
    """A teleop run: each step gets the request the robot's navigator took (steps.csv v_req, w_req), set at the
    frame's time; other behaviours get none."""
    navs = []
    steps = {k: (0, 0.0, "TELEOP", 0.01 * k, -0.02 * k) for k in range(10)}
    replay(tmp_path, navs=navs, steps=steps, args={"behaviour": "teleop", "forward_v": 0.15})
    nav = navs[0][0]
    assert navs[0][1] == "teleop" and len(nav.steps) == 10
    assert np.array([r[:2] for r in nav.requests]) == pytest.approx(np.array([[0.01 * k, -0.02 * k]
                                                                              for k in range(10)]))
    assert [r[2] for r in nav.requests] == pytest.approx([s[0] for s in nav.steps])
    navs = []
    replay(tmp_path / "b", navs=navs, steps=steps)
    assert navs[0][0].requests == []


def test_replay_switches_to_the_daemon_imu_where_the_robot_did(tmp_path):
    """FIFO IMU level, daemon IMU tilted 10 deg: after the robot's switch the pitch moves."""
    rows = replay(tmp_path, fifo=True, fallback_t=10.3)
    assert abs(float(rows[0]["pitch_down_deg"])) < 0.5  # before: the FIFO reader's (level)
    assert float(rows[-1]["pitch_down_deg"]) < -2.0      # after: the daemon's 10 deg up


def test_replay_reads_the_fifo_imu(tmp_path):
    """No IMU in state.jsonl: the frames are processed only if imu_fifo.jsonl is used."""
    rows = replay(tmp_path, fifo=True)
    assert len(rows) == len(FRAMES_T)
    assert float(rows[-1]["pitch_down_deg"]) == pytest.approx(0.0, abs=0.5)
