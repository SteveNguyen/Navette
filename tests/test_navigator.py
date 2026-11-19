"""Navigator (map -> frontier exploration -> collision monitor -> resume gate) driving a unicycle in a walled
room, scans ray-cast (the Python prototype's tests)."""

import itertools
import math

import numpy as np
import pytest

from navette import (ExploreParams, FreeSpace, FrontierParams, NavOut, Navigator, OccupancyGrid, SafetyParams, Scan,
                     TiltTrust)

BEAR = (np.arange(180) + 0.5) * 2.0 - 180
FOV, BLIND, MAX_R = 55.0, 0.45, 2.5  # camera half-width (deg), unseen floor in front (m), scan range (m)


def room_scan(pose, size=(4.0, 3.0)):
    """Scan of an empty rectangular room [0, sx] x [0, sy] from pose (x, y, theta)."""
    x, y, th = pose
    sx, sy = size
    seen = np.abs(BEAR) < FOV
    d = np.full(180, np.inf)
    for k in np.nonzero(seen)[0]:
        a = th + math.radians(BEAR[k])
        c, s = math.cos(a), math.sin(a)
        hits = [t for t in ((sx - x) / c if c > 1e-9 else np.inf, -x / c if c < -1e-9 else np.inf,
                            (sy - y) / s if s > 1e-9 else np.inf, -y / s if s < -1e-9 else np.inf) if t > 0]
        d[k] = min(hits)
    obst = np.where(seen & (d < MAX_R), d, np.inf)
    near = np.isfinite(obst) & (obst <= BLIND + 0.05)
    return Scan(BEAR, seen, obst, near, np.where(seen, np.minimum(d, MAX_R), 0.0), np.where(seen, BLIND, np.inf))


def wall_distance(pose, size=(4.0, 3.0)):
    return min(pose[0], size[0] - pose[0], pose[1], size[1] - pose[1])


def drive(nav, pose, size, steps, dt=1 / 8.7):
    """Closed loop with a perfect base (the command is applied for the next step). Returns the NavOuts, the
    closest approach of a wall and the distance driven; stops at the first finished step."""
    pose, v, w, closest, travelled, outs = list(pose), 0.0, 0.0, np.inf, 0.0, []
    for k in range(steps):
        out = nav.step(k * dt, dt, tuple(pose), room_scan(pose, size), 0.35, 0.0, v, w, v)
        outs.append(out)
        v, w = out.v, out.w
        pose[2] += w * dt
        pose[0] += v * dt * math.cos(pose[2])
        pose[1] += v * dt * math.sin(pose[2])
        travelled += abs(v) * dt
        closest = min(closest, wall_distance(pose, size))
        if out.finished:
            break
    return outs, closest, travelled


def test_explores_the_room_without_touching_a_wall():
    nav = Navigator()
    _, closest, travelled = drive(nav, (1.0, 1.5, 0.0), (4.0, 3.0), 400)
    assert closest > nav.safety.radius, closest
    assert travelled > 1.0, travelled


def test_safety_filter_holds_a_forward_request_before_a_wall():
    # the collision monitor alone (the resume gate: tests/test_blind_zone.py), on a constant 0.3 m/s request
    nav = Navigator(behaviour="forward", forward_v=0.3, resume_n=1)
    pose = (4.0 - nav.safety.radius - 0.05, 1.5, 0.0)  # robot edge 5 cm from the wall at x = 4 (facing it)
    out = nav.step(0.0, 0.1, pose, room_scan(pose), 0.35, 0.0, 0.0, 0.0, 0.0)
    assert out.v_req == 0.3 and out.v <= 1e-9
    far = (1.0, 1.5, 0.0)  # 2.9 m of floor ahead: the request goes through
    assert nav.step(0.1, 0.1, far, room_scan(far), 0.35, 0.0, 0.0, 0.0, 0.0).v > 0.1


def test_forward_behaviour_drives_on_open_floor_and_stops_at_a_wall():
    """'forward' without odometry (a dry run without a base): forward_v straight through the collision monitor
    on the scan alone; the map is not used, so a fixed pose with obstacles seen earlier does not stop it. The
    resume gate holds the first clear step."""
    nav = Navigator(behaviour="forward", odometry=False)
    near = (4.0 - nav.safety.radius - 0.05, 1.5, 0.0)
    far = (1.0, 1.5, 0.0)
    assert nav.step(0.0, 0.1, near, room_scan(near), 0.35, 0.0, 0.0, 0.0, 0.0).v <= 1e-9
    assert nav.step(0.1, 0.1, near, room_scan(far), 0.35, 0.0, 0.0, 0.0, 0.0).v == 0.0  # gate: 1st clear step
    out = nav.step(0.2, 0.1, near, room_scan(far), 0.35, 0.0, 0.0, 0.0, 0.0)  # same pose, open scan
    assert (out.v, out.w, out.v_req, out.state) == (0.15, 0.0, 0.15, "FORWARD")
    assert len(nav.grid.keyframes) == 0


def test_forward_holds_the_start_heading():
    """forward steers back to the heading of its first step (the base drifts with w = 0), capped at 0.3 rad/s."""
    nav = Navigator(behaviour="forward", odometry=True, resume_n=1)
    far = (1.0, 1.5, 0.0)
    assert nav.step(0.0, 0.1, far, room_scan(far), 0.35, 0.0, 0.0, 0.0, 0.0).w_req == 0.0
    drifted = (1.1, 1.5, math.radians(-4.0))  # turned right (clockwise): steer left
    assert nav.step(0.1, 0.1, drifted, room_scan(drifted), 0.35, 0.0, 0.0, 0.0, 0.0).w_req > 0.05
    far_off = (1.2, 1.5, math.radians(30.0))
    assert nav.step(0.2, 0.1, far_off, room_scan(far_off), 0.35, 0.0, 0.0, 0.0, 0.0).w_req == -0.3


def test_finished_after_done_for_3_s():
    # a closed 1.6 x 1.6 m room: explored by looking around, then DONE
    nav, dt, size = Navigator(), 1 / 8.7, (1.6, 1.6)
    outs, closest, _ = drive(nav, (0.8, 0.8, 0.0), size, 1500)
    assert outs[-1].finished and closest > nav.safety.radius
    states = [o.state for o in outs]
    assert {"LOOK", "DONE"} <= set(states)
    n_done = next(k for k, s in enumerate(reversed(states)) if s != "DONE")  # DONE steps up to the finished one
    # finished when n DONE steps span more than 3 s: 27 at 8.7 Hz
    assert n_done == math.floor(3.0 / dt) + 1
    assert sum(o.finished for o in outs) == 1 and nav.path is None and nav.target is None


def test_reset_forgets_the_map_and_the_behaviour():
    nav, size = Navigator(), (1.6, 1.6)
    outs, _, _ = drive(nav, (0.8, 0.8, 0.0), size, 1500)
    grid = nav.grid
    assert outs[-1].finished and len(grid.keyframes) > 10 and grid.x0 is not None
    nav.reset()
    assert nav.grid is grid and len(grid.keyframes) == 0 and grid.x0 is None and not grid.log_odds.any()
    assert len(nav.frontier) == 0 and nav.path is None
    # as a new navigator: the same run again from another place
    pose = (0.6, 1.0, 1.0)
    fresh = Navigator()
    for k in range(40):
        args = (k * 0.1, 0.1, pose, room_scan(pose, size), 0.35, 0.0, 0.0, 0.0, 0.0)
        a, b = nav.step(*args), fresh.step(*args)
        assert (a.v, a.w, a.v_req, a.w_req, a.state, a.finished) == (b.v, b.w, b.v_req, b.w_req, b.state, b.finished)
    assert np.array_equal(nav.grid.log_odds, fresh.grid.log_odds)


def test_debug_views_follow_the_planner():
    nav = Navigator()
    assert nav.frontier.shape == (0, 2) and nav.path is None and nav.target is None
    pose = (1.0, 1.5, 0.0)
    out = nav.step(0.0, 0.1, pose, room_scan(pose), 0.35, 0.0, 0.0, 0.0, 0.0)
    assert out.state in ("DRIVE", "SPIN") and not out.finished and not out.blind_stop
    assert nav.frontier.shape[1] == 2 and len(nav.frontier) > 5
    assert nav.path.shape[1] == 2 and np.array_equal(nav.path[-1], nav.target)
    assert np.hypot(*(nav.path[0] - pose[:2])) < 0.05  # from the robot's cell
    # the target is a frontier cell of the map: free, in the grid
    grid = nav.grid
    i, j = grid.cell(nav.target[None])[0]
    assert grid.masks()[1][i, j] and grid.log_odds[i, j] < -grid.thr
    assert len(grid.obstacle_points(pose)) == 0  # the walls are beyond the trusted range from here


def test_parameters():
    grid = OccupancyGrid(size_m=8.0)
    nav = Navigator(SafetyParams(latency=1.53, decel=0.64), behaviour="explore", odometry=True, resume_n=3,
                    forward_v=0.2, grid=grid, frontier=FrontierParams(lookahead=0.4),
                    explore=ExploreParams(v_set=(0.1, 0.2), rotate_in_place=True), tilt_trust=TiltTrust(tol_m=0.05))
    assert nav.grid is grid and nav.grid.n == 160
    assert (nav.safety.latency, nav.safety.decel, nav.safety.radius) == (1.53, 0.64, 0.13)
    assert (nav.behaviour, nav.odometry, nav.resume_n, nav.forward_v) == ("explore", True, 3, 0.2)
    assert Navigator().safety == SafetyParams() and Navigator().grid.n == 480
    pose = (1.0, 1.5, 0.0)
    out = nav.step(0.0, 0.1, pose, room_scan(pose), 0.35, 0.0, 0.0, 0.0, 0.0)
    assert out.v_req in (0.0, 0.1, 0.2) and len(grid.keyframes) == 1
    # a tighter tilt tolerance trusts less of the scan: fewer cells known than with the default
    default = Navigator()
    default.step(0.0, 0.1, pose, room_scan(pose), 0.35, 0.0, 0.0, 0.0, 0.0)
    assert 0 < grid.masks()[1].sum() < default.grid.masks()[1].sum()
    with pytest.raises(ValueError):
        Navigator(behaviour="v3")
    with pytest.raises(ValueError):
        nav.step(0.0, 0.1, pose, Scan([0.0], [True], [1.0], [False], [1.0]), 0.35, 0.0, 0.0, 0.0, 0.0)


def wall_scan(d):
    """A straight wall across the path d m ahead of the centre, floor seen up to it from 0.1 m (never near):
    the collision monitor's clearance alone decides."""
    seen = np.abs(BEAR) < FOV
    free = d / np.cos(np.radians(BEAR))
    return Scan(BEAR, seen, np.where(seen, free, np.inf), np.zeros(180, bool), np.where(seen, free, 0.0),
                np.where(seen, 0.1, np.inf))


def test_resume_margin_restart_hysteresis():
    """After a stop, the resume gate counts a step only if the request fits with resume_margin more clearance;
    stops are unchanged. With the default safety, 0.15 m/s gets some v > 0 from a wall at 0.21 m on (0.31 m
    with 0.10 m more margin)."""
    fits, blocked, fits_wider = 0.25, 0.15, 0.35
    plain = Navigator(behaviour="forward", odometry=False)
    hyst = Navigator(behaviour="forward", odometry=False, resume_margin=0.1)
    assert (plain.resume_margin, hyst.resume_margin, hyst.resume_n) == (0.0, 0.1, 2)
    clock = itertools.count()

    def v(nav, d):
        return nav.step(next(clock) * 0.1, 0.1, (0.0, 0.0, 0.0), wall_scan(d), 0.35, 0.0, 0.0, 0.0, 0.0).v

    for d in (1.0, 0.5, fits):  # moving toward the wall: no change
        a, b = v(plain, d), v(hyst, d)
        assert a > 0 and a == b, d
    assert (v(plain, blocked), v(hyst, blocked)) == (0.0, 0.0)  # stops: immediate, both
    assert v(plain, fits) == 0.0 and v(plain, fits) > 0  # plain: resumes after resume_n = 2 steps
    assert all(v(hyst, fits) == 0.0 for _ in range(20))  # hysteresis: holds where it only fits
    assert v(hyst, fits_wider) == 0.0 and v(hyst, fits_wider) > 0  # 0.1 m farther: resumes after 2
    assert v(hyst, fits) > 0  # moving again: the normal margin
    with pytest.raises(ValueError, match="resume_margin"):
        Navigator(resume_margin=-0.1)
    with pytest.raises(ValueError, match="resume_margin"):
        Navigator(resume_margin=float("nan"))


def patch_scan(near, theta_deg=0.0):
    """wall_scan(2.0) with, if near, a patch filling the image bottom at 0.45 m in the sectors within 3 deg of
    straight ahead in the fixed frame, seen from heading theta_deg."""
    s = wall_scan(2.0)
    if not near:
        return s
    k = np.abs(BEAR + theta_deg) < 3.0
    obst, floor, closest, near_ = (np.array(s.obstacle), np.array(s.floor), np.array(s.closest),
                                   np.array(s.near))
    obst[k], floor[k], closest[k], near_[k] = 0.45, 0.45, 0.45, True
    return Scan(BEAR, np.array(s.seen), obst, near_, floor, closest)


def test_near_persistence():
    """near_k of near_n: a single-frame near patch is a far obstacle; one that stays stops the robot from its
    second frame; bearings compared in the fixed frame. 1 of 1 (default): the scan's own flags."""
    plain = Navigator(behaviour="forward", odometry=False, resume_n=1)
    nav = Navigator(behaviour="forward", odometry=False, resume_n=1, near_k=2, near_n=3)
    assert (plain.near_k, plain.near_n, nav.near_k, nav.near_n) == (1, 1, 2, 3)
    assert len(nav.near) == 0
    clock = itertools.count()

    def v(n, near, theta_deg=0.0):
        pose = (0.0, 0.0, math.radians(theta_deg))
        return n.step(next(clock) * 0.1, 0.1, pose, patch_scan(near, theta_deg), 0.35, 0.0, 0.0, 0.0, 0.0).v

    assert v(plain, True) == 0.0 and np.array_equal(plain.near, np.array(patch_scan(True).near))
    assert v(nav, False) > 0 and v(nav, True) > 0 and not nav.near.any()  # one frame: not near
    assert v(nav, False) > 0 and v(nav, False) > 0  # the flicker leaves the window
    assert v(nav, True, 8.0) > 0  # turned 8 deg: the same fixed bearing, 2 of the last 3 from the next one
    assert v(nav, True, 12.0) == 0.0 and nav.near.sum() == 2
    for k, n in ((0, 1), (3, 2)):
        with pytest.raises(ValueError, match="near_k"):
            Navigator(near_k=k, near_n=n)


def test_nav_out_is_dataclass_like():
    out = NavOut(0.1, 0.2, 0.3, 0.4, "DRIVE", False)
    assert (out.v, out.w, out.v_req, out.w_req, out.state, out.finished, out.blind_stop) == (
        0.1, 0.2, 0.3, 0.4, "DRIVE", False, False)
    out.state, out.v = "GLANCE", 0.0  # the robot loop marks its own states
    assert repr(out) == "NavOut(v=0.0, w=0.2, v_req=0.3, w_req=0.4, state=\"GLANCE\", finished=False, blind_stop=False)"


def test_step_mask_is_scan_then_step(tmp_path):
    """step_mask(fs, mask, up_c, ...) = step(fs.scan(mask, up_c, cam_h), ...), and returns that scan."""
    from conftest import camera, free_space, ideal_calib

    fs = free_space(ideal_calib(tmp_path))
    _, up_c = camera(30.0, 0.0)
    rng = np.random.default_rng(0)
    a, b = Navigator(), Navigator()
    for k in range(12):
        mask = np.ones(fs.shape, bool)
        mask[:int(rng.integers(40, 150))] = False  # a wall: not floor above some row
        mask[:, :int(rng.integers(0, 120))] &= rng.random() < 0.5
        pose = (0.02 * k, 0.0, 0.05 * k)
        args = (k * 0.1, 0.1, pose, 0.35, 0.0, 0.1, 0.0, 0.1)
        scan = fs.scan(mask, up_c, 0.35)
        out_a, scan_a = a.step_mask(fs, mask, up_c, *args)
        out_b = b.step(*args[:3], scan, *args[3:])
        assert repr(out_a) == repr(out_b)
        for f in ("bearing_deg", "seen", "obstacle", "near", "floor", "closest"):
            assert np.array_equal(getattr(scan_a, f), getattr(scan, f))
    assert np.array_equal(a.grid.log_odds, b.grid.log_odds) and len(a.grid.keyframes) > 3
    assert isinstance(fs, FreeSpace)
    with pytest.raises(ValueError):
        a.step_mask(fs, np.ones((3, 3), bool), up_c, *args)
