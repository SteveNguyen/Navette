"""A fake robot for tests without hardware: a point robot on a flat floor, with a wall across its way.

    world = FakeWorld()                       # clock: time.monotonic, or a ManualClock for instant tests
    robot = FakeRobot(world)                  # .camera .imu .base (navette.interfaces)

FakeWorld is the truth (pose of the base, wall); the fake devices read it:
  FakeBase    unicycle model, velocity applied at once; dead-man; start delay; scripted faults
  FakeCamera  ideal pinhole pitched down, floor mask of the flat floor up to the wall
  FakeImu     still head: specific force g along 'up', gyro = the base's yaw rate about 'up'

Also the test doubles for what the runtime gets injected from the core: FakeNavigator and FakeUpFilter.
They are NOT implementations of the core's classes, only the smallest thing a test can drive the loop with.
"""
import math
import time

import numpy as np

from navette.interfaces import DRIVING, FAULT, STARTING, STOPPED, BaseStatus, Frame, Odometry, StartRefused

G = 9.81


class ManualClock:
    """A clock tests move by hand: clock.t += dt, or clock.sleep(dt)."""

    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class FakeWorld:
    """Truth. Base pose (x, y, theta) in the world frame (x forward at the start, y left, theta
    counter-clockwise), moved by the velocity (v, w) the FakeBase applies; integrated when read.
    wall_x: a wall along the world's y axis at this x (None: open floor)."""

    def __init__(self, clock=time.monotonic, wall_x=None):
        self.clock, self.wall_x = clock, wall_x
        self.x = self.y = self.theta = 0.0
        self.v = self.w = 0.0
        self.t = clock()
        self.t_zero = math.inf  # time at which the base's dead-man zeroes the velocity

    def advance(self):
        """Integrate the pose up to now (exact arcs), the velocity dropping to 0 at t_zero. Returns now."""
        now = self.clock()
        for t_end in sorted({min(self.t_zero, now), now}):
            dt = t_end - self.t
            if dt > 0:
                dth = self.w * dt
                # chord of a constant-curvature step along the mid heading
                chord = self.v * dt * np.sinc(dth / (2 * np.pi))
                self.x += chord * math.cos(self.theta + dth / 2)
                self.y += chord * math.sin(self.theta + dth / 2)
                self.theta = math.remainder(self.theta + dth, 2 * math.pi)
                self.t = t_end
            if self.t >= self.t_zero:
                self.v = self.w = 0.0
        return now


class FakeBase:
    """Base contract on the FakeWorld. deadman_s: the velocity drops to 0 this long after the last command().
    start_s: time from start() to DRIVING. Scripts: refuse (start() raises StartRefused with this text),
    fault(text) (status() says FAULT from now on, until the next start()).
    v_sign, w_sign, deadman: knobs to build a wrong base that the conformance checks must catch."""

    def __init__(self, world, deadman_s=0.5, start_s=0.2, balancing=False, v_sign=1.0, w_sign=1.0, deadman=True):
        self.world, self.deadman_s, self.start_s, self.balancing = world, deadman_s, start_s, balancing
        self.v_sign, self.w_sign, self.deadman = v_sign, w_sign, deadman
        self.refuse, self.low_battery = None, False
        self._session, self._t_start, self._fault = False, 0.0, None
        self.calls = []  # ("start",), ("command", v, w), ("stop",), ("park",)

    def fault(self, text):
        self._fault = text

    def odometry(self):
        w = self.world
        now = w.advance()
        return Odometry(now, w.x, w.y, w.theta, w.v, w.w)

    def start(self):
        if self.refuse:
            raise StartRefused(self.refuse)
        self.calls.append(("start",))
        self._session, self._t_start, self._fault = True, self.world.clock(), None

    def command(self, v, w):
        self.calls.append(("command", v, w))
        if self.status_now() != DRIVING:  # outside a session, or not ready: ignored
            return None
        world = self.world
        now = world.advance()
        world.v, world.w = self.v_sign * v, self.w_sign * w
        world.t_zero = now + self.deadman_s if self.deadman else math.inf
        return len(self.calls)

    def stop(self):
        self.calls.append(("stop",))
        self._session = False
        self.world.advance()
        self.world.v = self.world.w = 0.0  # stops at once (no braking distance in this model)

    def park(self):
        self.calls.append(("park",))
        return True

    def status_now(self):
        if self._fault is not None and self._session:
            return FAULT
        if not self._session:
            return STOPPED
        return DRIVING if self.world.clock() - self._t_start >= self.start_s else STARTING

    def status(self):
        state = self.status_now()
        return BaseStatus(state, self._fault if state == FAULT else "", self.low_battery)


def pinhole_rays(mask_shape, f_px, dv=0.0):
    """(h*w, 3) viewing rays (z = 1) of an ideal lens with focal length f_px (mask pixels) centred on the
    mask; dv as navette.rays.mask_rays."""
    h, w = mask_shape
    uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    x = (uu + 0.5 - w / 2) / f_px
    y = (vv + 0.5 + dv - h / 2) / f_px
    return np.stack([x.ravel(), y.ravel(), np.ones(h * w)], -1)


def camera_rotation(pitch_down_deg):
    """R_base_cam of a camera looking forward, pitched down: columns = camera axes (x right, y down, z
    optical) in the base frame (x forward, y left, z up)."""
    p = math.radians(pitch_down_deg)
    level = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # x_c = -y_b, y_c = -z_b, z_c = x_b
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    return Ry @ level


class FakeCamera:
    """Camera contract: frames every period_s on the world's clock (read() sleeps until the next one is due
    with the real clock, never with a ManualClock), captured latency_s before they are returned. The camera is
    at the base frame's origin, `height` above the floor (the config's camera.body_z), pitched down.
    Floor mask: a pixel is floor if its ray hits the floor before the wall."""

    def __init__(self, world, mask_shape=(96, 160), f_px=104.0, pitch_down_deg=20.0, height=0.35, period_s=0.05,
                 latency_s=0.0):
        self.world, self.mask_shape, self.height = world, mask_shape, height
        self.period_s, self.latency_s = period_s, latency_s
        self.rays, self.rays_low = pinhole_rays(mask_shape, f_px), pinhole_rays(mask_shape, f_px, dv=0.5)
        self.T_base_cam = np.eye(4)
        self.T_base_cam[:3, :3] = camera_rotation(pitch_down_deg)
        self.index, self.t_next = 0, world.clock()

    def read(self):
        now = self.world.clock()
        if now < self.t_next:
            if self.world.clock is time.monotonic:
                time.sleep(min(self.t_next - now, 0.02))
            return None
        self.t_next = max(self.t_next + self.period_s, now)
        self.world.advance()
        frame = Frame(now - self.latency_s, self.index, self.T_base_cam.copy())
        frame.truth = (self.world.x, self.world.y, self.world.theta)  # where the picture was taken (this fake only)
        self.index += 1
        return frame

    def segment(self, frame):
        x, y, th = frame.truth
        d = self.rays @ self.T_base_cam[:3, :3].T  # ray directions in the base frame
        down = d[:, 2] < -1e-9
        s = np.where(down, self.height / np.where(down, -d[:, 2], 1.0), 0.0)  # ray length to the floor
        px, py = s * d[:, 0], s * d[:, 1]  # floor point in the base frame
        floor = down
        if self.world.wall_x is not None:
            floor = down & (x + math.cos(th) * px - math.sin(th) * py < self.world.wall_x)
        return floor.reshape(self.mask_shape)


class FakeImu:
    """Imu contract: samples at rate_hz up to now, in the camera frame of a FakeCamera."""

    name = "fake"

    def __init__(self, world, pitch_down_deg=20.0, rate_hz=200.0):
        self.world, self.dt = world, 1.0 / rate_hz
        self.up_c = camera_rotation(pitch_down_deg).T @ np.array([0.0, 0.0, 1.0])  # 'up' in the camera frame
        self.t_next = world.clock()

    def read(self):
        now = self.world.advance()
        out = []
        while self.t_next <= now:
            out.append((self.t_next, self.world.w * self.up_c, G * self.up_c))
            self.t_next += self.dt
        return out


class FakeRobot:
    """Robot contract: FakeCamera + FakeImu + FakeBase on one FakeWorld."""

    head = None

    def __init__(self, world, pitch_down_deg=20.0, **camera_kw):
        self.world = world
        self.camera = FakeCamera(world, pitch_down_deg=pitch_down_deg, **camera_kw)
        self.imu = FakeImu(world, pitch_down_deg)
        self.base = FakeBase(world)
        self.stop = None  # the controller's STOP, once the loop gave it (a test's stop button)

    def on_stop(self, stop):
        self.stop = stop

    def meta(self):
        return {"fake": True}

    def close(self):
        pass


# ------------------------------------------------------------------ test doubles of the core's classes

class FakeNavOut:
    def __init__(self, v, w, state, finished=False, blind_stop=False):
        self.v, self.w, self.v_req, self.w_req = v, w, v, w
        self.state, self.finished, self.blind_stop = state, finished, blind_stop


class FakeNavigator:
    """TEST DOUBLE of navette.Navigator (same step / observe signature). Drives straight at `v` and stops when
    the scan's free distance within +-10 deg is under stop_m. Records every call."""

    grid = None

    def __init__(self, odometry=True, v=0.1, stop_m=0.6):
        self.odometry, self.v, self.stop_m = odometry, v, stop_m
        self.steps, self.observed = [], []  # (t, dt, pose, scan, cam_h, a, v, w, v_prev), (t, pose, scan, cam_h)

    def step(self, t, dt, pose, scan, cam_h, a, v, w, v_prev):
        self.steps.append((t, dt, pose, scan, cam_h, a, v, w, v_prev))
        ahead = self.ahead(scan)
        return FakeNavOut(self.v if ahead > self.stop_m else 0.0, 0.0, "FORWARD")

    def observe(self, t, pose, scan, cam_h):
        self.observed.append((t, pose, scan, cam_h))

    @staticmethod
    def ahead(scan):
        """Smallest free distance within +-10 deg of straight ahead (0 if nothing is seen there)."""
        d = np.asarray(scan.free())[np.abs(np.asarray(scan.bearing_deg)) <= 10]
        return float(np.nanmin(d)) if np.isfinite(d).any() else 0.0


class FakeUpFilter:
    """TEST DOUBLE of navette.UpFilter (same update signature): 'up' is the direction of the accelerometer,
    unfiltered. Right for a still IMU only."""

    def __init__(self, g=G):
        self.g = g

    def update(self, t, gyro, acc):
        a = np.asarray(acc, float)
        return a / np.linalg.norm(a)
