"""The generic loop and its parts without hardware: 'up' from the IMU, head glance, camera pose, and the whole
loop on the fake robot (real threads, real clock) with test doubles of the core's Navigator and UpFilter."""
import csv
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from navette import config
from navette.runtime import controller as ctl
from navette.runtime import loop
from navette.runtime.glance import Glance
from navette.runtime.localization import OdometryLocalization
from navette.runtime.up import LiveUp
from navette.testing.fakes import FakeNavigator, FakeRobot, FakeUpFilter, FakeWorld, camera_rotation

ROBOT_TOML = Path(__file__).parents[1] / "robots" / "reachy_poussah.toml"
# [explore] keys the reference robot sets away from the core defaults, on purpose (see its config file)
REACHY_SWITCHES = {"near_known": True, "spin_flip_deg": 45.0, "spin_lead_s": 0.5, "spin_accel": 0.5,
                   "blocked_s": 3.0, "blocked_progress": True}
IMU_MAX_AGE_S = 0.1


# ------------------------------------------------------------------ up

def test_live_up_waits_for_g():
    up = LiveUp(FakeUpFilter, g_window_s=1.0, max_age_s=IMU_MAX_AGE_S)
    acc = np.array([0.0, 0.0, 9.7])
    up.add([(t, np.zeros(3), acc) for t in np.arange(0, 0.5, 0.01)])
    assert up.up_at(0.4) is None
    up.add([(t, np.zeros(3), acc) for t in np.arange(0.5, 1.2, 0.01)])
    assert up.upf.g == pytest.approx(9.7) and up.up_at(1.1) == pytest.approx([0, 0, 1])


def test_live_up_is_none_when_the_imu_is_stale():
    """The interpolation clamps; LiveUp refuses a query more than max_age_s past the newest sample."""
    up = LiveUp(FakeUpFilter, g_window_s=1.0, max_age_s=IMU_MAX_AGE_S)
    up.add([(t, np.zeros(3), np.array([0, 0, 9.8])) for t in np.arange(0, 1.5, 0.0025)])
    assert up.up_at(1.5 + IMU_MAX_AGE_S - 0.01) is not None
    assert up.up_at(1.5 + IMU_MAX_AGE_S + 0.01) is None
    assert up.up_at(60.0) is None


def test_live_up_interpolates_at_the_capture_time_and_takes_a_given_g():
    """A frame is captured before it arrives: 'up' is taken in the history, not at the newest sample."""
    up = LiveUp(FakeUpFilter, g=9.8)  # g given (replay): the filter starts at the first sample
    tilt = lambda t: np.array([0.0, math.sin(t), math.cos(t)]) * 9.8  # 'up' turning at 1 rad/s
    up.add([(t, np.zeros(3), tilt(t)) for t in np.arange(0, 0.3, 0.005)])
    assert up.upf.g == 9.8
    assert up.up_at(0.2) == pytest.approx([0, math.sin(0.2), math.cos(0.2)], abs=1e-4)
    assert up.up_at(0.262) == pytest.approx([0, math.sin(0.262), math.cos(0.262)], abs=1e-4)


def test_live_up_never_gives_a_non_finite_up():
    """Samples that are not measurements (zero, NaN) neither spoil the measure of g nor reach the history,
    whatever the filter answers (FakeUpFilter divides by |acc|; navette.UpFilter answers None)."""
    acc, zero, nan = np.array([0.0, 0.0, 9.7]), np.zeros(3), np.full(3, np.nan)
    up = LiveUp(FakeUpFilter, g_window_s=0.2, max_age_s=IMU_MAX_AGE_S)
    up.add([(0.0, zero, zero), (0.01, zero, nan)] + [(t, zero, acc) for t in np.arange(0.02, 0.3, 0.01)])
    assert up.upf.g == pytest.approx(9.7) and up.up_at(0.25) == pytest.approx([0, 0, 1])
    # garbage only from now on: the last valid 'up' ages, and is refused after max_age_s
    up.add([(t, zero, nan) for t in np.arange(0.3, 0.6, 0.01)])
    assert up.up_at(0.3) == pytest.approx([0, 0, 1]) and up.up_at(0.5) is None

    class NoUp:  # a filter without an 'up' yet
        def update(self, t, gyro, acc):
            return None

    up = LiveUp(lambda g: NoUp(), g=9.8)
    up.add([(t, zero, acc) for t in np.arange(0, 0.3, 0.01)])
    assert up.up_at(0.29) is None
    # a window with no usable sample at all: g is measured on the next one
    up = LiveUp(FakeUpFilter, g_window_s=0.2, max_age_s=IMU_MAX_AGE_S)
    up.add([(t, zero, nan) for t in np.arange(0, 0.3, 0.01)])
    assert up.upf is None and up.up_at(0.29) is None
    up.add([(t, zero, acc) for t in np.arange(0.3, 0.6, 0.01)])
    assert up.upf.g == pytest.approx(9.7) and up.up_at(0.59) == pytest.approx([0, 0, 1])


# ------------------------------------------------------------------ glance

class Head:
    move_s = 0.5

    def __init__(self, back=True):
        self.moves, self.back = [], back

    def look_down(self):
        self.moves.append("down")

    def look_back(self):
        self.moves.append("back")

    def is_back(self):
        return self.back


def test_glance_sequence_hold_look_back_and_cooldown():
    head = Head()
    g = Glance(head, n_frames=2, settle_s=0.3, cooldown_s=5.0)
    assert g.frame(0.0) is None and not g.active
    assert g.begin(0.0, (0, 0, 0)) and head.moves == ["down"]
    assert g.frame(0.4) == "move"           # head moving / settling
    assert g.frame(0.85) == "look"          # settled: 1st frame into the map
    assert g.frame(0.95) == "look" and head.moves == ["down", "back"]  # 2nd: head back
    assert g.frame(1.2) == "move"
    assert g.frame(1.8, (0, 0, 0)) is None and not g.active
    assert not g.begin(3.0, (0.05, 0, 0))   # cooldown, not moved
    assert g.begin(3.0, (0.2, 0, 0))        # moved 0.2 m: again
    g2 = Glance(Head())
    g2.begin(0.0, (0, 0, 0))
    assert not g2.begin(0.1, (1, 0, 0))     # one at a time
    assert Glance(Head()).begin(0.0, (0, 0, 0), force=True)


def test_glance_resends_the_move_back_until_the_head_is_back():
    head = Head(back=False)
    g = Glance(head, n_frames=1, settle_s=0.3, retries=2)
    g.begin(0.0, (0, 0, 0))
    assert g.frame(0.9) == "look" and head.moves == ["down", "back"]
    assert g.frame(1.8) == "move" and g.frame(2.7) == "move" and head.moves == ["down", "back", "back", "back"]
    assert g.frame(3.6, (0, 0, 0)) is None and g.failed  # still not back after the retries: ends anyway


# ------------------------------------------------------------------ geometry and pose

def test_camera_pose_height_along_up_and_heading():
    T = np.eye(4)
    T[:3, :3] = camera_rotation(20.0)
    T[:3, 3] = [0.04, 0.0, 0.05]
    up_c = T[:3, :3].T @ np.array([0, 0, 1.0])
    h, heading = loop.camera_pose(T, up_c, 0.3)
    assert h == pytest.approx(0.35) and heading == pytest.approx(0.0)
    assert loop.pitch_down_deg(up_c) == pytest.approx(20.0)
    yaw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # camera turned 90 deg to the left on the base
    T[:3, :3] = yaw @ T[:3, :3]
    assert loop.camera_pose(T, up_c, 0.3)[1] == pytest.approx(math.pi / 2)
    # the body leaning 10 deg forward: the camera's 4 cm forward offset now lowers it
    lean = math.radians(10)
    up_body = np.array([-math.sin(lean), 0, math.cos(lean)])
    T[:3, :3] = camera_rotation(20.0)
    h, _ = loop.camera_pose(T, T[:3, :3].T @ up_body, 0.3)
    assert h == pytest.approx(0.3 + 0.05 * math.cos(lean) - 0.04 * math.sin(lean))


def test_camera_mount_warning():
    """The loop's one-time check of the Camera contract's mounting limits: yaw off the base's forward axis,
    horizontal offset against the footprint radius."""
    T = np.eye(4)
    T[:3, :3] = camera_rotation(20.0)
    T[:3, 3] = [0.04, 0.0, 0.3]
    assert loop.camera_mount_warning(T, 0.13) is None  # 4 cm ahead, looking forward: any height
    T[:3, 3] = [0.1, 0.0, 0.3]
    assert "0.10 m from the base centre" in loop.camera_mount_warning(T, 0.13)
    assert loop.camera_mount_warning(T, 0.3) is None  # a larger robot
    yaw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    T[:3, :3] = yaw @ T[:3, :3]
    assert "looks +90 deg off" in loop.camera_mount_warning(T, 0.3)


def test_loop_warns_once_about_an_unsupported_camera_mounting(tmp_path, monkeypatch):
    r = Run(tmp_path / "ok", FakeWorld())
    r.end()
    assert not [m for m in r.logs if "WARNING" in m]
    real = FakeRobot.__init__

    def camera_ahead(self, *a, **k):  # the camera 0.4 m ahead of the base centre
        real(self, *a, **k)
        self.camera.T_base_cam[0, 3] = 0.4

    monkeypatch.setattr(FakeRobot, "__init__", camera_ahead)
    r = Run(tmp_path / "off", FakeWorld())
    time.sleep(0.3)  # more frames: the warning is not repeated
    r.end()
    warnings = [m for m in r.logs if "WARNING" in m]
    assert len(warnings) == 1 and "0.40 m from the base centre" in warnings[0]


def test_localization_is_the_odometry_pose():
    from navette.interfaces import Odometry

    loc = OdometryLocalization()
    assert loc.pose() is None
    loc.update(Odometry(1.0, 0.5, -0.2, 0.3, 0.1, 0.0))
    assert loc.pose() == (0.5, -0.2, 0.3)
    loc.update(None)
    assert loc.pose() is None


def test_base_motion_low_passes_the_acceleration():
    from navette.interfaces import Odometry

    m = loop.BaseMotion(tau_s=0.2)
    assert m.update(Odometry(0, 0, 0, 0, 0.2, 0.5), 0.1) == (0.0, 0.2, 0.5)
    a, v, w = m.update(Odometry(0.1, 0, 0, 0, 0.0, 0.0), 0.1)
    assert (v, w) == (0.0, 0.0) and a == pytest.approx(-1.0)  # half of -2 m/s^2 (dt = tau / 2)
    assert loop.BaseMotion().update(None, 0.1) == (0.0, 0.0, 0.0)


# ------------------------------------------------------------------ the navigator's parameters

def test_reachy_config_gives_the_core_defaults_exactly():
    """Robot geometry and speeds now come from the config (loop.navigator_params). For the reference robot
    they must be, bit for bit, the core's defaults (the parameters of the golden sequences and of every
    recorded run), plus what its config says of the collision monitor, and its resume gate: resume_n and
    resume_margin differ from the core's defaults (2, 0.0) on purpose (restart hysteresis for the balancing
    base), and so do the exploration switches of REACHY_SWITCHES (see the config file's comments)."""
    import navette

    cfg = config.load(ROBOT_TOML)
    p = loop.navigator_params(cfg)
    assert p["frontier"] == navette.FrontierParams()
    # the Navigator's own default, but the switches this robot turns on
    assert p["explore"] == navette.ExploreParams(rotate_in_place=True, **REACHY_SWITCHES)
    core = navette.ExploreParams()
    assert all(getattr(core, k) != v for k, v in REACHY_SWITCHES.items())  # they do differ from the core's
    # pinned to the core's defaults, it is the Navigator's default exactly
    pinned = cfg.replace("explore", **{k: getattr(core, k) for k in REACHY_SWITCHES})
    assert loop.navigator_params(pinned)["explore"] == navette.ExploreParams(rotate_in_place=True)
    assert p["explore"].v_set == [0.1, 0.2, 0.3] and p["explore"].moving_v == 0.05
    assert p["grid"] == {"robot_radius": 0.13} and navette.OccupancyGrid(**p["grid"]).n == navette.OccupancyGrid().n
    assert p["safety"] == navette.SafetyParams(radius=0.13, margin=0.05, v_max=0.4, decel=0.64, latency=0.9 + 0.63,
                                               unseen_m=0.35, reverse_max=0.0, step=0.02, n_speeds=20)
    assert p["frontier"].clear == 0.18 == p["safety"].radius + p["safety"].margin and p["frontier"].blind == 0.5
    nav = loop.new_navigator(cfg, "explore", 0.15, True)
    assert nav.safety == p["safety"] and (nav.behaviour, nav.forward_v) == ("explore", 0.15)
    assert (nav.resume_n, nav.resume_margin) == (4, 0.10)
    core = navette.Navigator()
    assert (core.resume_n, core.resume_margin) == (2, 0.0)


def test_a_bigger_slower_robot_gets_its_own_geometry_and_speeds():
    cfg = config.load(ROBOT_TOML)
    cfg = cfg.replace("footprint", radius=0.3, margin=0.1).replace("camera", blind_m=0.8)
    cfg = cfg.replace("safety", v_max=0.08).replace("controller", v_max=0.06)
    p = loop.navigator_params(cfg)
    assert p["grid"] == {"robot_radius": 0.3}
    assert p["frontier"].clear == pytest.approx(0.4) and p["frontier"].blind == 0.8
    assert p["frontier"].clear == p["safety"].radius + p["safety"].margin  # the collision monitor's clearance
    # every candidate (0.1, 0.2, 0.3) is above what the collision monitor plans for: one is kept, its v_max
    assert p["explore"].v_set == [0.08]
    # the robot is sent 0.06 m/s at most (controller.v_max): it counts as moving from half of that
    assert p["explore"].moving_v == pytest.approx(0.03)
    # candidates under safety.v_max are kept, those above dropped; the controller's cap does not remove any
    p = loop.navigator_params(cfg.replace("explore", v_set=(0.02, 0.07, 0.2)))
    assert p["explore"].v_set == [0.02, 0.07] and p["explore"].moving_v == pytest.approx(0.01)
    nav = loop.new_navigator(cfg, "explore", 0.05, True)
    assert nav.safety.radius == 0.3 and nav.safety.v_max == 0.08


# ------------------------------------------------------------------ the loop on the fake robot

def fake_config(**perception):
    cfg = config.load(ROBOT_TOML)
    cfg = cfg.replace("camera", body_z=0.35)  # the fake camera's height
    cfg = cfg.replace("page", timeout_s=0.0)  # no page in these tests
    return cfg.replace("perception", up_g_s=0.2, **perception)


class Run:
    """loop.run on a FakeRobot in a thread; the navigators it made, the controller it built."""

    def __init__(self, tmp_path, world, cfg=None, dry_run=False, base=True, head=None, **nav_kw):
        self.robot, self.navs, self.logs, self.ctrl = FakeRobot(world), [], [], None
        if not base:
            self.robot.base = None
        self.robot.head = head
        self.out, self.stop_event = tmp_path, threading.Event()
        tmp_path.mkdir(exist_ok=True)

        def make_navigator(odometry):
            self.navs.append(FakeNavigator(odometry, **nav_kw))
            return self.navs[-1]

        self.thread = threading.Thread(target=loop.run, daemon=True, args=(
            self.robot, cfg or fake_config(), str(tmp_path), self.stop_event, make_navigator, FakeUpFilter),
            kwargs={"dry_run": dry_run, "on_controller": lambda c: setattr(self, "ctrl", c), "log": self.logs.append})
        self.thread.start()
        self.wait(lambda: self.ctrl is not None and self.navs and self.navs[0].steps, "the first step")

    def wait(self, cond, what, timeout_s=5.0):
        t_end = time.monotonic() + timeout_s
        while not cond():
            assert time.monotonic() < t_end and self.thread.is_alive(), f"timeout waiting for {what}: {self.logs}"
            time.sleep(0.01)

    def end(self):
        self.stop_event.set()
        self.thread.join(8)
        assert not self.thread.is_alive()
        with open(self.out / "steps.csv") as f:
            steps = list(csv.DictReader(f))
        with open(self.out / "commands.csv") as f:
            cmds = list(csv.DictReader(f))
        return steps, cmds, json.load(open(self.out / "meta.json"))


def test_loop_drives_to_the_wall_and_stops(tmp_path):
    """Closed loop on the fake world: START, the base drives, the scan sees the wall come closer at the right
    distance (the loop's frames, signs and heights are right), the navigator double stops in front of it."""
    world = FakeWorld(wall_x=1.6)
    r = Run(tmp_path, world, v=0.3, stop_m=1.0)
    scan0 = r.navs[0].steps[-1][3]
    assert FakeNavigator.ahead(scan0) == pytest.approx(1.6, abs=0.15)  # the wall, seen from the start
    assert r.robot.base.calls == []  # nothing before START
    assert r.ctrl.start() == (True, "starting")
    r.wait(lambda: len(r.navs) == 2, "a fresh navigator at START")
    r.wait(lambda: r.ctrl.state == ctl.DRIVING and r.ctrl.sent[0] > 0, "driving")
    r.wait(lambda: r.ctrl.sent[0] == 0.0 and world.v == 0.0, "the stop at the wall", 15.0)
    world.advance()
    assert 0.4 < world.x < 0.75  # stopped once the wall was 1 m ahead (0.15 m/s cap, ~10 Hz loop, base lag 0)
    t, dt, pose, scan, cam_h, a, v, w, v_prev = r.navs[1].steps[-1]
    assert pose[0] == pytest.approx(world.x, abs=0.01) and abs(pose[1]) < 1e-6 and abs(pose[2]) < 1e-6
    assert FakeNavigator.ahead(scan) == pytest.approx(1.6 - world.x, abs=0.15)
    assert cam_h == pytest.approx(0.35, abs=1e-6) and (a, v, w) == (0.0, 0.0, 0.0)  # not a balancing base: zeros
    assert r.ctrl.state == ctl.DRIVING  # held by the navigator, not halted
    r.robot.stop("button")  # the robot's own stop control
    r.wait(lambda: r.ctrl.state == ctl.IDLE, "idle after STOP")
    steps, cmds, meta = r.end()
    assert r.navs[0].odometry is True and all(n.observed == [] for n in r.navs)
    # logs: the steps.csv columns, one row per step, both navigators
    assert list(steps[0]) == loop.STEP_COLS + loop.LIVE_COLS
    assert len(steps) == len(r.navs[0].steps) + len(r.navs[1].steps) == meta["steps"]
    assert {s["run"] for s in steps} == {"0", "1"} and {s["imu"] for s in steps} == {"fake"}
    drive = [s for s in steps if s["ctrl"] == "DRIVING" and float(s["v"]) > 0]
    assert drive and all(float(s["pitch_down_deg"]) == pytest.approx(20.0, abs=0.1) for s in steps)
    assert all(0 < float(s["lat_ms"]) < 500 for s in steps)
    assert max(float(s["v_prev"]) for s in steps) == pytest.approx(0.15)  # the capped command that was sent
    kinds = [c["cmd"] for c in cmds]
    assert kinds[0] == "start" and kinds[-2:] == ["stop", "park"] and "command" in kinds
    assert "button" in [c["reason"] for c in cmds] and {c["dry"] for c in cmds} == {"0"}
    assert max(float(c["v"]) for c in cmds) == pytest.approx(0.15)
    assert meta["args"] == {"behaviour": "explore", "forward_v": 0.15, "dry_run": False, "base": True}
    assert meta["robot"] == {"fake": True} and meta["config"]["perception"]["stale_s"] == 0.9
    assert r.robot.base.calls[-2:] == [("stop",), ("park",)]


def test_loop_exit_stops_and_parks_the_base(tmp_path):
    world = FakeWorld()
    r = Run(tmp_path, world)
    r.ctrl.start()
    r.wait(lambda: world.v > 0, "driving")
    steps, cmds, _ = r.end()  # the loop ends while driving
    assert r.robot.base.calls[-2:] == [("stop",), ("park",)] and world.v == 0.0
    assert cmds[-1]["cmd"] == "park" and cmds[-1]["reason"] == "exit"


def test_loop_base_fault_halts(tmp_path):
    world = FakeWorld()
    r = Run(tmp_path, world)
    r.ctrl.start()
    r.wait(lambda: world.v > 0, "driving")
    r.robot.base.fault("event estop")
    r.wait(lambda: r.ctrl.state == ctl.IDLE, "the halt")
    assert r.ctrl.reason == "watchdog: event estop" and world.v == 0.0
    r.end()


def test_loop_stale_imu_stops_stepping_and_the_watchdog_halts(tmp_path):
    """No IMU sample: no step, so no navigator output, so the stale-perception watchdog halts the drive."""
    world = FakeWorld()
    r = Run(tmp_path, world, cfg=fake_config(stale_s=0.5))
    r.ctrl.start()
    r.wait(lambda: world.v > 0, "driving")
    r.robot.imu.read = lambda: []  # the IMU goes silent
    r.wait(lambda: r.ctrl.state == ctl.IDLE, "the halt")
    assert "no navigator output for 0.5 s" in r.ctrl.reason and world.v == 0.0
    r.end()


@pytest.mark.parametrize("what", ["odometry", "camera_pose"])
def test_loop_takes_no_step_on_non_finite_inputs_and_the_watchdog_halts(tmp_path, what):
    """A NaN in the base's odometry or in the camera pose: no step (the navigator never sees it), so no
    output, so the stale-perception watchdog halts the drive."""
    world = FakeWorld()
    r = Run(tmp_path, world, cfg=fake_config(stale_s=0.5))
    r.ctrl.start()
    r.wait(lambda: world.v > 0, "driving")
    if what == "odometry":
        odometry = r.robot.base.odometry
        r.robot.base.odometry = lambda: odometry()._replace(v=math.nan)
    else:
        r.robot.camera.T_base_cam[2, 3] = math.nan  # the camera height is then NaN
    time.sleep(0.2)  # a step may be under way
    n = sum(len(nav.steps) for nav in r.navs)
    r.wait(lambda: r.ctrl.state == ctl.IDLE, "the halt")
    assert "no navigator output for 0.5 s" in r.ctrl.reason and world.v == 0.0
    assert sum(len(nav.steps) for nav in r.navs) == n
    steps, _, _ = r.end()
    assert all(math.isfinite(float(s[k])) for s in steps for k in ("x", "cam_h", "v"))


def test_loop_dry_run_and_no_base_have_no_odometry(tmp_path):
    """The pose only follows a base that drives: a carried robot (dry run, no base) must not smear the map."""
    r = Run(tmp_path / "a", FakeWorld(), dry_run=True)
    r.ctrl.start()
    r.wait(lambda: len(r.navs) == 2 and r.navs[1].steps, "a step after START")
    _, cmds, meta = r.end()
    assert [n.odometry for n in r.navs] == [False, False] and meta["args"]["dry_run"] is True
    assert {c["dry"] for c in cmds} == {"1"}
    r = Run(tmp_path / "b", FakeWorld(), base=False)
    r.ctrl.start()
    r.wait(lambda: r.ctrl.state == ctl.DRIVING and r.ctrl.sent[0] > 0, "driving (logged only)")
    steps, cmds, meta = r.end()
    assert r.navs[0].odometry is False and meta["args"]["base"] is False
    assert all(float(s["x"]) == 0 and float(s["y"]) == 0 for s in steps) and {c["dry"] for c in cmds} == {"1"}
    assert any(c["cmd"] == "command" and float(c["v"]) > 0 for c in cmds)


def test_loop_glance_at_start_holds_and_feeds_the_map(tmp_path):
    """With a Head: after START, once the base is DRIVING, the head looks down, the robot holds still (zero
    commands, watchdog fed), the settled frames go to Navigator.observe only, then it drives."""
    class QuickHead(Head):
        move_s = 0.15

    head, world = QuickHead(), FakeWorld()
    r = Run(tmp_path, world, head=head)
    assert head.moves == []
    r.ctrl.start()
    r.wait(lambda: len(r.navs) == 2 and len(r.navs[1].observed) == 2, "the glance")
    assert head.moves == ["down", "back"]
    assert world.v == 0.0 and r.ctrl.state == ctl.DRIVING and len(r.navs[1].observed) == 2 and r.navs[1].steps == []
    r.wait(lambda: world.v > 0, "driving after the glance")
    steps, cmds, _ = r.end()
    glance = [s for s in steps if s["state"] == "GLANCE"]
    assert len(glance) == 2 and all(float(s["v"]) == 0 and s["run"] == "1" for s in glance)
    i_first_go = next(i for i, c in enumerate(cmds) if c["cmd"] == "command" and float(c["v"]) > 0)
    assert i_first_go > 5 and all(float(c["v"]) == 0 for c in cmds[:i_first_go])  # zeros during the glance
