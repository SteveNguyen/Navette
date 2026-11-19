"""End to end without hardware: the runtime with the REAL core (navette.Navigator, navette.UpFilter, FreeSpace)
driving the fake robot in a simulated room. The fake camera gives the floor mask of a pinhole camera looking
at a floor with walls and boxes from where the fake base is, the fake IMU gives gravity
(navette.testing.fakes).

Two ways:
  - loop.run itself, with its threads, on the real clock (a few seconds each): START explores without
    touching anything, STOP stops, stale perception and a silent page halt, the forward behaviour stops before
    a box;
  - the same pieces called in loop.run's order on a simulated clock (`simulate`: loop.run takes its time from
    the system, and exploring a room takes a minute of robot time): exploration runs to its end, the
    controller stops by itself.
"""
import csv
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from navette import config
from navette.runtime import controller as ctl
from navette.runtime import loop
from navette.runtime.localization import OdometryLocalization
from navette.runtime.up import LiveUp
from navette.testing.fakes import FakeRobot, FakeWorld, ManualClock, box

ROBOT_TOML = Path(__file__).parents[1] / "robots" / "reachy_poussah.toml"
# a 2.6 x 2.2 m room with a box in a corner area; the robot starts at the origin, looking along x
ROOM = np.vstack([box(-1.0, -1.1, 1.6, 1.1), box(0.9, 0.5, 1.2, 0.8)])
# a larger room with a box 0.9 m straight ahead
BOX_AHEAD_X = 0.9
BOX_AHEAD = np.vstack([box(-1.0, -1.5, 3.0, 1.5), box(BOX_AHEAD_X, -0.3, BOX_AHEAD_X + 0.4, 0.3)])


def fake_config(stale_s=None, page_timeout_s=0.0):
    """The reference robot's config (its collision monitor, speeds and watchdogs) on the fake robot."""
    cfg = config.load(ROBOT_TOML)
    cfg = cfg.replace("camera", body_z=0.35)  # the fake camera's height
    cfg = cfg.replace("page", timeout_s=page_timeout_s)
    cfg = cfg.replace("perception", up_g_s=0.2)
    return cfg if stale_s is None else cfg.replace("perception", stale_s=stale_s)


# ------------------------------------------------------------------ loop.run, real clock

class Live:
    """loop.run on a FakeRobot in a thread, with the navigator and the attitude filter the command line
    builds (loop.default_navigator, loop.default_up_filter)."""

    def __init__(self, tmp_path, world, cfg, behaviour="explore", patch=None, web_app=None, session=None,
                 **robot_kw):
        """patch(robot): changes the fake robot before the loop starts. web_app: gets the page's routes.
        robot_kw: FakeRobot's."""
        self.robot, self.world, self.cfg, self.out = FakeRobot(world, **robot_kw), world, cfg, tmp_path
        if patch is not None:
            patch(self.robot)
        self.navs, self.logs, self.ctrl, self.stop_event = [], [], None, threading.Event()
        make = loop.default_navigator(cfg, behaviour, cfg.behaviour.forward_v)

        def make_navigator(odometry):
            self.navs.append(make(odometry))
            return self.navs[-1]

        self.thread = threading.Thread(target=loop.run, daemon=True, args=(
            self.robot, cfg, str(tmp_path), self.stop_event, make_navigator),
            kwargs={"behaviour": behaviour, "on_controller": lambda c: setattr(self, "ctrl", c),
                    "log": self.logs.append, "web_app": web_app, "session": session})
        self.thread.start()
        self.wait(lambda: "ready" in self.logs, "the loop")

    def wait(self, cond, what, timeout_s=10.0):
        t_end = time.monotonic() + timeout_s
        while not cond():
            assert time.monotonic() < t_end and self.thread.is_alive(), f"timeout waiting for {what}: {self.logs}"
            time.sleep(0.01)

    def start(self):
        assert self.ctrl.start() == (True, "starting")
        self.wait(lambda: self.ctrl.state == ctl.DRIVING, "DRIVING")

    def moving(self):
        return self.world.v != 0.0 or self.world.w != 0.0

    def end(self):
        self.stop_event.set()
        self.thread.join(8)
        assert not self.thread.is_alive()
        with open(self.out / "steps.csv") as f:
            return list(csv.DictReader(f))


def read(f):
    """f() on a navigator (or its grid) that the loop's thread is stepping. A step runs without the GIL with
    its objects borrowed, and using them meanwhile raises RuntimeError: retried."""
    for _ in range(2000):
        try:
            return f()
        except RuntimeError:
            time.sleep(0.001)
    raise AssertionError("the navigator stayed borrowed for 2 s")


def test_start_explores_without_collision_and_stop_stops(tmp_path):
    world = FakeWorld(walls=ROOM)
    r = Live(tmp_path, world, fake_config())
    assert r.robot.base.calls == [] and not r.moving()  # nothing before START
    r.start()
    r.wait(r.moving, "the first motion")
    time.sleep(4.0)  # exploring
    assert r.ctrl.state == ctl.DRIVING, r.ctrl.reason
    nav = r.navs[-1]
    assert read(lambda: (nav.behaviour, nav.odometry)) == ("explore", True)
    assert read(lambda: nav.safety).latency == pytest.approx(0.9 + 0.63)
    assert int(read(lambda: nav.grid.masks()[1]).sum()) > 400  # > 1 m^2 of floor mapped (0.05 m cells)
    r.robot.stop("button")  # the robot's own stop control
    r.wait(lambda: r.ctrl.state == ctl.IDLE, "IDLE after STOP")
    world.advance()
    pose = (world.x, world.y, world.theta)
    assert not r.moving() and r.robot.base.calls[-2:] == [("stop",), ("park",)]
    time.sleep(0.3)
    world.advance()
    assert (world.x, world.y, world.theta) == pose  # and it stays stopped
    steps = r.end()
    driven = [s for s in steps if s["ctrl"] == "DRIVING"]
    assert {s["state"] for s in driven} <= {"SPIN", "LOOK", "DRIVE"} and len(driven) > 20
    assert any(float(s["v"]) != 0 or float(s["w"]) != 0 for s in driven)
    assert abs(world.theta) > 0.3 or math.hypot(world.x, world.y) > 0.1  # it did turn or drive
    assert world.min_clearance > r.cfg.footprint.radius  # no collision
    assert all(float(s["cam_h"]) == pytest.approx(0.35, abs=1e-3) for s in steps)
    assert all(float(s["pitch_down_deg"]) == pytest.approx(20.0, abs=0.2) for s in steps)  # the real UpFilter


def test_stale_perception_halts(tmp_path):
    """The camera stops giving frames while the robot drives: no navigator output, the watchdog halts."""
    world = FakeWorld(walls=ROOM)
    r = Live(tmp_path, world, fake_config(stale_s=0.5))
    r.start()
    r.wait(r.moving, "the first motion")
    r.robot.camera.read = lambda: time.sleep(0.01)  # None: no new frame
    r.wait(lambda: r.ctrl.state == ctl.IDLE, "the halt")
    assert r.ctrl.reason == "watchdog: no navigator output for 0.5 s" and not r.moving()
    assert r.robot.base.calls[-1] == ("stop",)  # halted, not parked
    r.end()


def test_silent_page_halts(tmp_path):
    """Page dead-man: the run goes on while the page polls, and halts when it goes silent."""
    world = FakeWorld(walls=ROOM)
    r = Live(tmp_path, world, fake_config(page_timeout_s=0.6))
    r.start()
    t_end = time.monotonic() + 1.5
    while time.monotonic() < t_end:  # the page's poll
        r.ctrl.page_alive()
        time.sleep(0.1)
    assert r.ctrl.state == ctl.DRIVING and r.moving()
    r.wait(lambda: r.ctrl.state == ctl.IDLE, "the halt")
    assert "phone page silent" in r.ctrl.reason and not r.moving()
    r.end()


def test_forward_stops_before_the_box(tmp_path):
    world = FakeWorld(walls=BOX_AHEAD)
    r = Live(tmp_path, world, fake_config(), behaviour="forward")
    r.start()
    r.wait(lambda: world.x > 0.3, "the drive", 15.0)
    assert world.v == pytest.approx(0.15)  # the requested speed, under the controller's cap

    def at_rest():  # stopped for 1 s (a blind stop, then a short advance, come first)
        x = world.x
        time.sleep(1.0)
        world.advance()
        return world.x == x and not r.moving()

    r.wait(at_rest, "the stop", 20.0)
    gap = BOX_AHEAD_X - world.x  # base centre to the box's near face
    sp = read(lambda: r.navs[-1].safety)
    # The box ends up filling the image bottom: no floor is seen in front of it any more, and the collision
    # monitor lets nothing closer. That is where the camera's view of the floor starts: 0.35 m for this
    # camera (0.35 m high, lower edge of the view 45 deg down), within a frame's travel.
    assert sp.radius + sp.margin < gap < 0.42, gap
    assert abs(world.y) < 0.02 and world.min_clearance == pytest.approx(gap, abs=1e-6)
    assert r.ctrl.state == ctl.DRIVING  # held by the collision monitor, not halted
    steps = r.end()
    assert {s["state"] for s in steps} == {"FORWARD"}
    assert float(steps[-1]["v"]) == 0.0 and float(steps[-1]["v_req"]) == 0.15


def test_teleop_driven_by_the_page_joystick(tmp_path):
    """--behaviour teleop, the real core, driven only through the page's HTTP API as the phone's joystick does
    (~10 Hz posts while the thumb is down): it drives, stops by itself when the posts stop (operator dead-man),
    is held by the collision monitor before the box while the joystick still says forward, turns in place,
    and STOP stops. The session name is in meta.json."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import json

    from fastapi.testclient import TestClient

    from navette.runtime import page

    world, app = FakeWorld(walls=BOX_AHEAD), page.new_app()
    r = Live(tmp_path, world, fake_config(), behaviour="teleop", web_app=app, session="home-1")
    c = TestClient(app)
    assert "const TELEOP = true" in c.get("/").text
    assert c.post("/api/start", json={}).json()["ok"]
    r.wait(lambda: r.ctrl.state == ctl.DRIVING, "DRIVING")

    def joystick(v, w, seconds, until=None):
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end and not (until and until()):
            assert c.post("/api/teleop", json={"v": v, "w": w}).json()["ok"]
            time.sleep(0.1)

    time.sleep(0.5)
    assert not r.moving()  # no request yet
    joystick(0.15, 0.0, 1.5)
    world.advance()
    assert world.x > 0.1 and world.v == pytest.approx(0.15)
    r.wait(lambda: not r.moving(), "the operator's dead-man", 2.0)  # thumb up: no more posts
    assert r.ctrl.state == ctl.DRIVING  # not halted, just no request

    def held():
        s = c.get("/api/status").json()
        return s["v_req"] == 0.15 and s["v_ok"] == 0.0 and not r.moving()

    joystick(0.15, 0.0, 10.0, until=held)  # toward the box until the collision monitor holds it
    assert held()
    world.advance()
    gap = BOX_AHEAD_X - world.x
    sp = read(lambda: r.navs[-1].safety)
    assert sp.radius + sp.margin < gap < 0.45, gap
    theta = world.theta
    joystick(0.0, 0.5, 1.0)  # a turn in place is not held
    world.advance()
    assert world.theta - theta > 0.2 and BOX_AHEAD_X - world.x == pytest.approx(gap, abs=0.01)
    assert c.post("/api/stop", json={}).json()["ok"]
    r.wait(lambda: r.ctrl.state == ctl.IDLE, "IDLE after STOP")
    steps = r.end()
    driven = [s for s in steps if s["run"] == "1"]
    assert {s["state"] for s in driven} == {"TELEOP"} and world.min_clearance > sp.radius + sp.margin
    meta = json.load(open(tmp_path / "meta.json"))
    assert meta["session"] == "home-1" and meta["args"]["behaviour"] == "teleop"


def at_rest(r, seconds=1.0):
    """Not moving, and not moved for `seconds`."""
    r.world.advance()
    x = r.world.x
    time.sleep(seconds)
    r.world.advance()
    return r.world.x == x and not r.moving()


@pytest.mark.parametrize("bad", [np.zeros(3), np.full(3, np.nan)], ids=["zero", "nan"])
def test_a_bad_first_imu_sample_does_not_blind_the_robot(tmp_path, bad):
    """A wall 1 m ahead, forward behaviour, and an IMU whose very first sample is zero (a driver's first read)
    or NaN. That one sample used to make 'up' NaN for good: the camera height NaN, a scan that sees nothing,
    and the collision monitor granting ~0.1 m/s for ever: the robot drove through the wall, no watchdog
    tripping. Now the sample is ignored and the robot stops before the wall as with a healthy IMU."""
    world = FakeWorld(wall_x=1.0)
    first = [True]

    def patch(robot):
        read = robot.imu.read

        def bad_first():
            out = read()
            if out and first[0]:
                first[0] = False
                out[0] = (out[0][0], out[0][1], bad)
            return out
        robot.imu.read = bad_first

    r = Live(tmp_path, world, fake_config(), behaviour="forward", patch=patch)
    r.start()
    r.wait(lambda: world.x > 0.2, "the drive", 15.0)
    r.wait(lambda: at_rest(r), "the stop", 20.0)
    sp = read(lambda: r.navs[-1].safety)
    assert sp.radius + sp.margin < 1.0 - world.x < 0.5, world.x  # stopped before the wall
    assert r.ctrl.state == ctl.DRIVING and not first[0]  # held by the collision monitor
    steps = r.end()
    assert all(float(s["cam_h"]) == pytest.approx(0.35, abs=1e-3) for s in steps)
    assert all(float(s["pitch_down_deg"]) == pytest.approx(20.0, abs=0.2) for s in steps)


def test_an_imu_that_turns_to_garbage_halts_the_drive(tmp_path):
    """Only NaN samples from some moment on: no 'up', so no step and no output, and the stale-perception
    watchdog halts."""
    world = FakeWorld(walls=BOX_AHEAD)
    r = Live(tmp_path, world, fake_config(stale_s=0.5), behaviour="forward")
    r.start()
    r.wait(r.moving, "the first motion")
    read = r.robot.imu.read
    r.robot.imu.read = lambda: [(t, gyro, np.full(3, np.nan)) for t, gyro, acc in read()]
    r.wait(lambda: r.ctrl.state == ctl.IDLE, "the halt")
    assert r.ctrl.reason == "watchdog: no navigator output for 0.5 s" and not r.moving()
    world.advance()
    assert world.x < 0.3  # within the watchdog's time of the first bad sample
    steps = r.end()
    assert all(math.isfinite(float(s["cam_h"])) and math.isfinite(float(s["v"])) for s in steps)


@pytest.mark.parametrize("pitch_down_deg", [-40.0, 90.0], ids=["up", "straight_down"])
def test_a_camera_that_sees_no_floor_ahead_gives_no_translation(tmp_path, pitch_down_deg):
    """Camera pitched up (no ray below the horizon: the scan sees nothing at all) or straight down (it only
    sees the floor under the robot). Open floor, forward behaviour: the robot must not advance. The collision
    monitor alone let it creep at ~0.1 m/s on a scan that sees nothing."""
    world = FakeWorld()
    r = Live(tmp_path, world, fake_config(), behaviour="forward", pitch_down_deg=pitch_down_deg)
    r.start()
    time.sleep(2.5)
    world.advance()
    assert world.x == 0.0 and world.v == 0.0
    assert r.ctrl.state == ctl.DRIVING and r.ctrl.sent[0] == 0.0  # fresh outputs, all holding still
    steps = [s for s in r.end() if s["ctrl"] == "DRIVING"]
    assert len(steps) > 10 and all(float(s["v"]) == 0.0 and float(s["v_req"]) == 0.15 for s in steps)


# ------------------------------------------------------------------ simulated clock

def simulate(world, clock, cfg, behaviour, seconds, on_step=None):
    """loop.run's pieces in its order, single-threaded on a ManualClock: perception at the camera's rate
    (odometry, IMU -> LiveUp with the real UpFilter, mask -> scan -> real Navigator -> Controller.set_nav) and
    Controller.tick at 20 Hz. START at the first step; runs `seconds` of robot time, or until the controller
    is back to IDLE. on_step(nav, frame, floor, scan, pose, out). Returns (controller, navigator, outputs)."""
    robot = FakeRobot(world, period_s=0.1)
    camera, imu, base = robot.camera, robot.imu, robot.base
    fs = loop.free_space(camera)
    up = LiveUp(loop.default_up_filter, cfg.perception.up_g_s, cfg.perception.imu_max_age_s)
    loc = OdometryLocalization()
    ctrl = ctl.Controller(base, cfg.controller.v_max, cfg.controller.w_max, stale_s=cfg.perception.stale_s,
                          start_timeout_s=cfg.controller.start_timeout_s, clock=clock, log=lambda m: None)
    nav = loop.new_navigator(cfg, behaviour, cfg.behaviour.forward_v, True)
    outs, t_prev, t_end = [], None, clock() + seconds
    while clock() < t_end and not (outs and ctrl.state == ctl.IDLE):
        clock.t += 0.05
        frame = camera.read()
        if frame is not None:
            loc.update(base.odometry())
            up.add(imu.read())
            up_c = up.up_at(frame.t)
            if up_c is not None:
                if not outs:
                    assert ctrl.start() == (True, "starting")
                cam_h, cam_heading = loop.camera_pose(frame.T_base_cam, up_c, cfg.camera.body_z)
                x, y, th = loc.pose()
                pose = (x, y, th + cam_heading)
                floor = camera.segment(frame)
                scan = fs.scan(floor, up_c, cam_h)
                dt = 0.1 if t_prev is None else frame.t - t_prev
                out = nav.step(frame.t, dt, pose, scan, cam_h, 0.0, 0.0, 0.0, ctrl.sent[0])
                ctrl.set_nav(out, frame.t, ctrl.run)
                outs.append(out)
                t_prev = frame.t
                if on_step is not None:
                    on_step(nav, frame, floor, scan, pose, out)
        ctrl.tick()
    world.advance()
    return ctrl, nav, outs


def test_exploration_runs_to_its_end():
    """START -> the room is explored without touching anything -> the navigator says finished -> the
    controller stops and parks the base. About a minute of robot time."""
    clock = ManualClock()
    world = FakeWorld(clock, walls=ROOM)
    planned = []  # a step where the planner has a path, for the preview test below

    def on_step(nav, frame, floor, scan, pose, out):
        if not planned and nav.path is not None and len(nav.frontier):
            planned.append((np.asarray(nav.frontier).copy(), np.asarray(nav.path).copy(), nav.target))

    cfg = fake_config()
    ctrl, nav, outs = simulate(world, clock, cfg, "explore", 180.0, on_step)
    assert ctrl.state == ctl.IDLE and ctrl.reason == "exploration finished"
    assert world.v == 0.0 and world.w == 0.0
    assert outs[-1].finished and outs[-1].state == "DONE" and nav.target is None
    assert {o.state for o in outs} == {"SPIN", "LOOK", "DRIVE", "DONE"}
    assert world.min_clearance > cfg.footprint.radius + 0.1  # well clear of the walls and the box
    assert max(o.v for o in outs) > 0.1
    # the room is 2.6 x 2.2 m less the 0.3 x 0.3 m box and what lies within a cell of a wall: most of it is
    # mapped free
    free_m2 = float(nav.grid.masks()[1].sum()) * nav.grid.res ** 2
    assert 0.8 * (2.6 * 2.2 - 0.09) < free_m2 < 2.6 * 2.2, free_m2
    # the planner's frontier cells, path and target are exposed for the preview, in the map frame (the room)
    frontier, path, target = planned[0]
    assert frontier.shape[1] == 2 and path.shape[1] == 2 and len(path) >= 2 and len(target) == 2
    for pts in (frontier, path):  # cell centres: within the room, to a few cells (range error at the far walls)
        assert (pts[:, 0] > -1.15).all() and (pts[:, 0] < 1.75).all() and (np.abs(pts[:, 1]) < 1.25).all(), pts


def test_preview_shows_the_frontier_and_the_path():
    cv2 = pytest.importorskip("cv2")
    from navette.runtime import preview

    clock = ManualClock()
    world = FakeWorld(clock, walls=ROOM)
    views = []

    def on_step(nav, frame, floor, scan, pose, out):
        if not views and nav.path is not None and len(nav.frontier):
            image = np.zeros((360, 640, 3), np.uint8)
            views.append(preview.view(image, floor, None, out, 0.35, 20.0, nav, pose, scan, "text"))

    simulate(world, clock, fake_config(), "explore", 40.0, on_step)
    (img,) = views
    assert img.shape == (360, 1000, 3) and preview.jpeg(img)[:2] == b"\xff\xd8"
    top_down = img[:, 640:].reshape(-1, 3)

    def has(colour):
        return bool((top_down == colour).all(1).any())

    assert has((200, 0, 200)) and has((0, 110, 0))  # frontier cells, path
    assert has((235, 245, 235)) and has((40, 40, 150)) and has((150, 150, 150))  # free, obstacle, unknown
