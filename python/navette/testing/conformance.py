"""Conformance checks of a robot adapter against navette.interfaces. Each check raises AssertionError saying
what is wrong. The fake robot passes them (tests/test_conformance_fake.py); run them on a new adapter before
giving it to the runtime.

    check_camera(camera)
    check_imu(imu, up_c=...)        # robot standing still
    check_base(base)                # ROBOT MOTION: drives ~0.1 m forward, turns ~0.3 rad, then stops and parks

clock / sleep: the adapter's time base. A real robot uses the defaults (time.monotonic, time.sleep); the fake
robot on a ManualClock passes the clock and its sleep, and the checks run instantly.
"""
import math
import time

import numpy as np

from navette.interfaces import DRIVING, FAULT, MAX_CAMERA_YAW_DEG, STARTING, STOPPED, camera_mount

STATES = (STOPPED, STARTING, DRIVING, FAULT)
PERIOD_S = 0.05  # command period used here (20 Hz, as the controller)
# "Stopped" = the pose stays within STILL_M (forward) and STILL_RAD over STILL_WINDOW_S. A balancing base never
# reads still at an instant: at rest it sways (Poussah with a Reachy Mini: +-2 cm, +-0.1 m/s, gyro +-0.2 rad/s),
# so its allowance is wider; a statically stable base gets 5 mm.
STILL_WINDOW_S, STILL_RAD = 1.0, 0.05
STILL_M = {True: 0.03, False: 0.005}  # by Base.balancing


def _on_clock(t, now, what):
    """t is a time.monotonic()-like stamp of something that already happened, not long ago."""
    assert isinstance(t, float) and math.isfinite(t), f"{what}: time {t!r} is not a finite float"
    assert now - 2.0 < t <= now + 1e-6, f"{what}: time {t} is not on the loop's clock (now {now}): {now - t:+.3f} s"


def check_camera(camera, n_frames=5, timeout_s=10.0, max_yaw_deg=MAX_CAMERA_YAW_DEG, max_offset_m=0.05,
                 clock=time.monotonic, sleep=time.sleep):
    """Rays and mask shape; n_frames frames: capture times on the clock and strictly increasing, indices
    increasing, a proper camera pose within the mounting limits of the Camera contract, a C-contiguous bool
    mask of the right shape.
    max_yaw_deg: yaw of the optical axis off the base's +x. max_offset_m: horizontal distance of the camera
    from the base centre; pass half your footprint radius (the default is for a 0.1 m one)."""
    h, w = camera.mask_shape
    for name in ("rays", "rays_low"):
        r = np.asarray(getattr(camera, name))
        assert r.shape == (h * w, 3), f"{name}: shape {r.shape}, expected {(h * w, 3)}"
        assert np.isfinite(r).all() and (r[:, 2] > 0).all(), f"{name}: not finite, or not in front of the camera"
    rays, low = np.asarray(camera.rays).reshape(h, w, 3), np.asarray(camera.rays_low).reshape(h, w, 3)
    assert (rays[1:, :, 1] / rays[1:, :, 2] > rays[:-1, :, 1] / rays[:-1, :, 2]).all(), \
        "rays: y must grow with the row (camera frame: y down, row-major rays)"
    assert (rays[:, 1:, 0] / rays[:, 1:, 2] > rays[:, :-1, 0] / rays[:, :-1, 2]).all(), \
        "rays: x must grow with the column (camera frame: x right)"
    assert (low[:, :, 1] / low[:, :, 2] > rays[:, :, 1] / rays[:, :, 2]).all(), \
        "rays_low: must look lower than rays (the pixel's lower edge)"
    frames, t_end = [], clock() + timeout_s
    while len(frames) < n_frames:
        assert clock() < t_end, f"only {len(frames)} frames in {timeout_s} s"
        frame = camera.read()
        if frame is None:
            sleep(0.005)
            continue
        _on_clock(frame.t, clock(), "Frame.t")
        T = np.asarray(frame.T_base_cam, float)
        assert T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1]), "Frame.T_base_cam: not a 4x4 pose"
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.linalg.det(R) > 0, \
            "Frame.T_base_cam: rotation not orthonormal / right-handed"
        yaw, offset = camera_mount(T)
        assert abs(yaw) <= max_yaw_deg, \
            f"Frame.T_base_cam: the camera looks {yaw:+.0f} deg off the base's forward (+x); more than " \
            f"{max_yaw_deg:.0f} deg is not supported (the robot drives where the scan says straight ahead)"
        assert offset <= max_offset_m, \
            f"Frame.T_base_cam: the camera is {offset:.2f} m from the base centre horizontally (limit " \
            f"{max_offset_m:.2f} m): the offset is ignored, every range would be off by that much"
        mask = camera.segment(frame)
        assert isinstance(mask, np.ndarray) and mask.dtype == bool and mask.shape == (h, w), \
            f"segment(): {type(mask).__name__} {getattr(mask, 'dtype', '')} {getattr(mask, 'shape', '')}, " \
            f"expected a bool array {(h, w)}"
        assert mask.flags.c_contiguous, "segment(): the mask must be C-contiguous (np.ascontiguousarray)"
        frames.append(frame)
    ts, idx = [f.t for f in frames], [f.index for f in frames]
    assert all(b > a for a, b in zip(ts, ts[1:])), f"Frame.t not strictly increasing: {ts}"
    assert all(b > a for a, b in zip(idx, idx[1:])), f"Frame.index not increasing: {idx}"


def check_imu(imu, up_c=None, duration_s=0.5, clock=time.monotonic, sleep=time.sleep):
    """Robot standing still for duration_s: samples arrive, each once, times on the clock and strictly
    increasing, acc ~9.81 m/s^2 (m/s^2, not g) pointing up, gyro ~0 (rad/s, not deg/s). Every sample is a
    measurement, from the very first one read() returns: finite, |acc| within 50 % of g (a driver's
    zero-filled first read is caught here).
    up_c: the expected 'up' in the camera frame (within 15 deg), if known; else only checked to point to the
    upper half of the image (camera y down: acc[1] < 0), which holds for a camera pitched less than 90 deg."""
    assert isinstance(imu.name, str), "Imu.name must be a str"
    first = imu.read()  # whatever accumulated before: old, but it must be measurements all the same
    _measurements(first, "the first read()")
    samples, t_end = [], clock() + duration_s
    while clock() < t_end:
        sleep(0.02)
        new = imu.read()
        if new:
            _on_clock(float(new[-1][0]), clock(), "IMU sample")
        samples += new
    assert len(samples) >= 5, f"{len(samples)} IMU samples in {duration_s} s"
    t = np.array([s[0] for s in samples], float)
    assert (np.diff(t) > 0).all(), "IMU sample times not strictly increasing (a sample returned twice?)"
    gyro, acc = np.array([s[1] for s in samples], float), np.array([s[2] for s in samples], float)
    assert gyro.shape == acc.shape == (len(samples), 3), "gyro and acc must be 3-vectors"
    _measurements(samples, "read()")
    g = float(np.median(np.linalg.norm(acc, axis=1)))
    assert 9.0 < g < 10.6, f"|acc| at rest is {g:.2f}: expected ~9.81 m/s^2 (the specific force, in m/s^2)"
    rate = float(np.median(np.linalg.norm(gyro, axis=1)))
    assert rate < 0.1, f"|gyro| at rest is {rate:.3f}: expected ~0 rad/s"
    up = np.median(acc, axis=0) / g
    if up_c is not None:
        angle = math.degrees(math.acos(float(np.clip(up @ np.asarray(up_c, float), -1, 1))))
        assert angle < 15, f"acc at rest is {angle:.0f} deg from the expected 'up' (camera frame, pointing up)"
    else:
        assert up[1] < 0, "acc at rest must point up: negative y in the camera frame (y down)"


def _measurements(samples, what):
    """Every IMU sample is finite, with |acc| within 50 % of g."""
    for k, (t, gyro, acc) in enumerate(samples):
        gyro, acc = np.asarray(gyro, float), np.asarray(acc, float)
        assert math.isfinite(t) and np.isfinite(gyro).all() and np.isfinite(acc).all(), \
            f"IMU sample {k} of {what} is not finite: t {t}, gyro {gyro}, acc {acc}"
        norm = float(np.linalg.norm(acc))
        assert 0.5 * 9.81 < norm < 1.5 * 9.81, \
            f"IMU sample {k} of {what}: |acc| = {norm:.2f}, not a measurement at rest: expected ~9.81 m/s^2 " \
            f"(the specific force, in m/s^2; a zero-filled first read?)"


def _odometry(base, clock, last=None):
    o = base.odometry()
    assert o is not None, "odometry() is None during a session"
    assert len(o) == 6 and all(isinstance(x, float) and math.isfinite(x) for x in o), f"odometry(): {o!r}"
    _on_clock(o.t, clock(), "Odometry.t")
    assert last is None or o.t >= last.t, f"Odometry.t went back: {last.t} -> {o.t}"
    return o


def _drive(base, v, w, duration_s, clock, sleep):
    """command(v, w) every PERIOD_S for duration_s (keeps the dead-man fed). Returns the last odometry read
    while still commanding."""
    t_end, o = clock() + duration_s, None
    while clock() < t_end:
        base.command(v, w)
        sleep(PERIOD_S)
        o = base.odometry()
    return o


def _moved(a, b):
    """(forward, left, turn) displacement from odometry a to b, in a's frame."""
    dx, dy = b.x - a.x, b.y - a.y
    c, s = math.cos(a.theta), math.sin(a.theta)
    return c * dx + s * dy, -s * dx + c * dy, math.remainder(b.theta - a.theta, 2 * math.pi)


def _watch(base, duration_s, clock, sleep):
    """Odometry sampled every PERIOD_S / 5 for duration_s, no command sent: [(time since the call, odometry)]."""
    t0, out, last = clock(), [], None
    while clock() - t0 < duration_s:
        sleep(PERIOD_S / 5)
        last = _odometry(base, clock, last)
        out.append((clock() - t0, last))
    return out


def _spread(ref, samples):
    """(forward spread m, turn spread rad) of the samples' poses, measured in ref's frame."""
    f = [_moved(ref, o)[0] for o in samples]
    th = [_moved(ref, o)[2] for o in samples]
    return max(f) - min(f), max(th) - min(th)


def check_base(base, v=0.1, w=0.3, drive_s=1.0, deadman_s=1.0, stop_s=0.3, start_timeout_s=10.0, inject_fault=None,
               clock=time.monotonic, sleep=time.sleep):
    """ROBOT MOTION on a real base (v m/s forward for drive_s, then w rad/s for drive_s, then two short
    forward drives; needs free floor).

    Checks, in order: no DRIVING and no motion before start(); start() reaches DRIVING through STARTING only;
    odometry on the clock; forward command -> x grows by about v * drive_s, measured v > 0; turn command ->
    theta grows (counter-clockwise), measured w > 0; the dead-man: the base stops by itself within deadman_s
    of the last command, and afterwards either the session goes on (the next command is applied) or the base
    reports FAULT with a reason (both allowed by the contract); stop() while driving brakes: the base is still
    (STILL_M, STILL_RAD over STILL_WINDOW_S) from stop_s after it on, having rolled at most about v * stop_s (a
    stop() that only latches leaves the base rolling until its dead-man); stop() is latched (commands
    ignored), park() returns True; then, if inject_fault is given (a function that makes the base fail), a new
    session reports FAULT with a reason.
    stop_s: the time the base needs to stop from v. It must be shorter than the base's dead-man time plus
    its braking, or a stop() that does not brake passes (the reference robot, 0.63 s of lag: 1.0)."""
    assert isinstance(base.balancing, bool), "Base.balancing must be a bool"
    st = base.status()
    assert st.state in STATES, f"status().state {st.state!r} is not one of {STATES}"
    assert st.state == STOPPED, f"before start(): state {st.state} ({st.fault}), expected STOPPED"
    a = base.odometry()
    _drive(base, v, 0.0, 0.3, clock, sleep)
    b = base.odometry()
    if a is not None and b is not None:
        assert abs(_moved(a, b)[0]) < 0.01, "command() before start() moved the base"

    _start(base, start_timeout_s, clock, sleep)
    o0 = _odometry(base, clock)

    o1 = _odometry(base, clock, _drive(base, v, 0.0, drive_s, clock, sleep))
    fwd, left, turn = _moved(o0, o1)
    assert fwd > 0, f"command(v > 0) moved the odometry by {fwd:+.3f} m along x: forward must be +x"
    assert 0.2 * v * drive_s < fwd < 1.5 * v * drive_s, \
        f"command(v={v}) for {drive_s} s moved {fwd:.3f} m (expected about {v * drive_s:.2f} m: units are m, m/s)"
    assert abs(left) < 0.5 * fwd and abs(turn) < 0.5, f"a straight command drifted: left {left:+.3f} m, turn {turn:+.2f} rad"
    assert o1.v > 0, f"Odometry.v is {o1.v:+.3f} while driving forward: must be the measured speed, > 0 forward"

    _drive(base, 0.0, 0.0, 0.5, clock, sleep)  # hold still (the base may be braking)
    o2 = _odometry(base, clock, o1)
    o3 = _odometry(base, clock, _drive(base, 0.0, w, drive_s, clock, sleep))
    turn = _moved(o2, o3)[2]
    assert turn > 0, f"command(w > 0) turned the odometry by {turn:+.3f} rad: counter-clockwise must be +theta"
    assert 0.2 * w * drive_s < turn < 1.5 * w * drive_s, \
        f"command(w={w}) for {drive_s} s turned {turn:.3f} rad (expected about {w * drive_s:.2f}: units are rad, rad/s)"
    assert o3.w > 0, f"Odometry.w is {o3.w:+.3f} while turning counter-clockwise: must be measured, > 0"

    # dead-man: one last command, then silence
    _drive(base, 0.0, 0.0, 0.5, clock, sleep)
    _drive(base, v, 0.0, 0.3, clock, sleep)
    sleep(deadman_s)
    o4 = _odometry(base, clock)
    still_m = STILL_M[base.balancing]
    df, dth = _spread(o4, [o for _, o in _watch(base, STILL_WINDOW_S, clock, sleep)])
    assert df < still_m and dth < STILL_RAD, \
        f"dead-man: the base still moves {deadman_s} s after the last command ({df * 100:.1f} cm, " \
        f"{dth:.3f} rad over the next {STILL_WINDOW_S} s; allowed {still_m * 100:.1f} cm, {STILL_RAD} rad)"
    # after the dead-man: the session goes on, or the base says FAULT (then a new session is started)
    st = base.status()
    assert st.state in (DRIVING, FAULT), \
        f"after the dead-man: state {st.state}, expected DRIVING (the next command resumes) or FAULT"
    if st.state == FAULT:
        assert st.fault, "FAULT after the dead-man without a reason (BaseStatus.fault)"
        base.stop()
        _start(base, start_timeout_s, clock, sleep)

    # stop() while driving must brake
    o6 = _odometry(base, clock, _drive(base, v, 0.0, 0.3, clock, sleep))
    assert o6.v > 0.2 * v, f"command(v={v}) after the dead-man did not move the base (v {o6.v:+.3f} m/s)"
    base.stop()
    samples = _watch(base, stop_s + STILL_WINDOW_S, clock, sleep)
    df, dth = _spread(o6, [o for t, o in samples if t >= stop_s])
    assert df < still_m and dth < STILL_RAD, \
        f"stop() did not brake: the base still moves {stop_s} s after it ({df * 100:.1f} cm, {dth:.3f} rad over " \
        f"the next {STILL_WINDOW_S} s); closing the session is not enough, the base must be told to stop"
    rolled = max(_moved(o6, o)[0] for _, o in samples)
    assert rolled <= v * stop_s + still_m + 0.01, \
        f"stop() at {v} m/s: the base rolled {rolled:.3f} m (more than v * stop_s)"
    o7 = samples[-1][1]

    _drive(base, v, 0.0, 0.3, clock, sleep)
    o8 = _odometry(base, clock, o7)
    assert abs(_moved(o7, o8)[0]) < still_m + 0.005, "stop() is not latched: a later command() moved the base"
    assert base.status().state != DRIVING, "status() says DRIVING after stop()"
    t_end = clock() + 10.0
    while not base.park():
        assert clock() < t_end, "park() did not return True within 10 s"
        sleep(PERIOD_S)
    base.stop()  # allowed in any state, any number of times

    if inject_fault is not None:
        _start(base, start_timeout_s, clock, sleep)
        inject_fault()
        t_end = clock() + 2.0
        while (st := base.status()).state != FAULT:
            assert clock() < t_end, f"no FAULT 2 s after the fault (state {st.state})"
            base.command(0.0, 0.0)
            sleep(PERIOD_S)
        assert st.fault, "FAULT without a reason (BaseStatus.fault)"
        base.stop()


def _start(base, timeout_s, clock, sleep):
    """start(), zeros every period, until DRIVING; only STARTING on the way."""
    base.start()
    t_end = clock() + timeout_s
    while (st := base.status()).state != DRIVING:
        assert st.state == STARTING, f"after start(): state {st.state} ({st.fault}), expected STARTING then DRIVING"
        assert clock() < t_end, f"not DRIVING {timeout_s} s after start()"
        base.command(0.0, 0.0)
        sleep(PERIOD_S)


def check_robot(robot, up_c=None, drive=False, clock=time.monotonic, sleep=time.sleep, **base_kw):
    """Camera and IMU checks; with drive=True (ROBOT MOTION) the base's as well."""
    check_camera(robot.camera, clock=clock, sleep=sleep)
    check_imu(robot.imu, up_c=up_c, clock=clock, sleep=sleep)
    if drive and robot.base is not None:
        check_base(robot.base, clock=clock, sleep=sleep, **base_kw)
