"""A small 2-D room simulator for closed-loop navigation tests. Pure numpy, seeded, independent of any
navigation stack: it gives scans as plain arrays and takes (v, w) commands.

    sim = Sim(*room("open"), seed=1, noise=0.01)
    scan = sim.scan()               # dict of the six per-sector arrays of a scan
    a = sim.step(v_cmd, w_cmd, dt)  # the base follows the command; sim.pose, sim.v, sim.w are its state
    frames, summary = run(sim, step, observe)  # the closed loop around a navigation stack's step function

The same file is kept with the generators of the golden sequences (tests/golden/navigator_*.npz), which the
reference implementation drove through `run`: change both copies together, and regenerate.

World: wall segments (rooms, boxes) in a fixed frame, metres. Robot: a disk (radius 0.13 m) with differential
drive; pose = (x, y, theta), theta counter-clockwise from the fixed x axis; its speed and yaw rate follow the
command with a first-order lag. A collision is the disk touching a wall (`clearance() < radius`); the
simulator does not stop the robot there, the tests assert it never happens.

Sensor: what a forward camera looking at the floor gives, per 2 deg bearing sector (bearing 0 = ahead,
positive = left):
  - only the sectors within `fov_deg` of straight ahead are seen;
  - the floor is visible from `closest` (below: the blind zone under the image) to `max_range`;
  - the obstacle range is the nearest wall along the sector (3 rays per sector), with optional noise growing
    with the range squared (as the error of a tilted camera); an obstacle nearer than `closest` fills the
    image bottom: it reads at `closest`, flagged `near` (within `near_tol` of `closest`).
"""

import math

import numpy as np

N_SECTORS = 180
BEARING_DEG = (np.arange(N_SECTORS) + 0.5) * 2.0 - 180.0
SUB_RAYS_DEG = np.array([-2.0 / 3.0, 0.0, 2.0 / 3.0])  # rays of a sector, from its centre


def polygon(*points):
    """Wall segments (N, 4) [x1, y1, x2, y2] of a closed polygon."""
    p = np.array(points, float)
    return np.hstack([p, np.roll(p, -1, axis=0)])


def rect(x0, y0, x1, y1):
    return polygon((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def room(name):
    """(wall segments, start pose) of a named test room."""
    if name == "open":  # an empty 4 x 3 m room
        return rect(0, 0, 4, 3), (1.0, 1.5, 0.0)
    if name == "wall":  # the same, starting 0.4 m from a wall and facing it (it fills the image bottom)
        return rect(0, 0, 4, 3), (0.4, 1.5, math.pi)
    if name == "small":  # a closed 2.2 x 2 m room: soon explored
        return rect(0, 0, 2.2, 2.0), (1.1, 1.0, 0.0)
    if name == "doorway":  # a corridor 1.2 m wide, then a doorway 0.8 m wide into a 2.9 x 3.2 m room
        outline = polygon((0, 0), (4, 0), (4, 0.2), (4.1, 0.2), (4.1, -1), (7, -1), (7, 2.2), (4.1, 2.2), (4.1, 1.0),
                          (4, 1.0), (4, 1.2), (0, 1.2))
        return outline, (0.6, 0.6, 0.0)
    if name == "clutter":  # a 5 x 4 m room with four boxes
        boxes = [rect(1.8, 1.0, 2.3, 1.5), rect(3.2, 2.4, 3.6, 3.0), rect(1.0, 2.8, 1.4, 3.2),
                 polygon((3.5, 0.8), (4.0, 1.0), (3.8, 1.5), (3.3, 1.3))]
        return np.vstack([rect(0, 0, 5, 4), *boxes]), (0.7, 0.7, 0.3)
    if name == "box":  # a box 2.2 m straight ahead
        return np.vstack([rect(0, 0, 5, 3), rect(3.0, 1.2, 3.4, 1.8)]), (0.8, 1.5, 0.0)
    if name == "side_box":  # a box in the blind zone at the start, ahead on the left, clear of the straight path
        return np.vstack([rect(0, 0, 4, 3), rect(1.36, 1.75, 1.56, 1.95)]), (1.0, 1.5, 0.0)
    raise ValueError(name)


def scan_arrays(seen_idx, floor, closest, max_range, near_tol):
    """The scan (dict of per-sector arrays: bearing_deg, seen, obstacle, near, floor, closest) whose sectors
    `seen_idx` are in view, with the floor visible from `closest` up to the range `floor` (one per sector in
    view, at most max_range). The floor ends at an obstacle if it ends before max_range; the obstacle is
    `near` if within near_tol of closest. All a scan of this simulator holds is in these arguments: a recorded
    scan is stored as them."""
    seen = np.zeros(N_SECTORS, bool)
    seen[seen_idx] = True
    obstacle, full_floor = np.full(N_SECTORS, np.inf), np.zeros(N_SECTORS)
    obstacle[seen_idx] = np.where(floor < max_range, floor, np.inf)
    full_floor[seen_idx] = floor
    return dict(bearing_deg=BEARING_DEG.copy(), seen=seen, obstacle=obstacle,
                near=np.isfinite(obstacle) & (obstacle <= closest + near_tol), floor=full_floor,
                closest=np.where(seen, closest, np.inf))


class Sim:
    def __init__(self, walls, pose, seed=0, noise=0.0, radius=0.13, lag_s=0.15, fov_deg=35.0, closest=0.47,
                 max_range=2.5, near_tol=0.05, w_lag_s=None):
        """noise: range noise, standard deviation = noise x range^2 (m; 0.01: 1 cm at 1 m, 4 cm at 2 m).
        lag_s: time constant of the base's response to a command; w_lag_s: the yaw rate's own, if given (None:
        lag_s for both)."""
        self.walls = np.asarray(walls, float)
        self.pose = np.array(pose, float)
        self.v = self.w = 0.0  # current speed (m/s) and yaw rate (rad/s)
        self.rng = np.random.default_rng(seed)
        self.noise, self.radius, self.lag_s = noise, radius, lag_s
        self.w_lag_s = lag_s if w_lag_s is None else w_lag_s
        self.fov_deg, self.closest, self.max_range, self.near_tol = fov_deg, closest, max_range, near_tol

    def ray_ranges(self, angles):
        """Distance from the robot centre to the nearest wall along each fixed-frame direction (rad); inf if
        none."""
        d = np.stack([np.cos(angles), np.sin(angles)], 1)[:, None, :]  # (rays, 1, 2)
        a = self.walls[None, :, :2] - self.pose[:2]  # robot -> segment start, (1, walls, 2)
        e = (self.walls[:, 2:] - self.walls[:, :2])[None]  # segment start -> end
        # robot + t d = start + u e
        den = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (a[..., 0] * e[..., 1] - a[..., 1] * e[..., 0]) / den
            u = (a[..., 0] * d[..., 1] - a[..., 1] * d[..., 0]) / den
        hit = (den != 0) & (t > 0) & (u >= 0) & (u <= 1)
        return np.where(hit, t, np.inf).min(1)

    def scan(self, closest=None, max_range=None):
        """The scan from the current pose, as a dict of per-sector arrays: bearing_deg, seen, obstacle, near,
        floor, closest. closest, max_range: this frame's instead of the simulator's (a look down at the
        floor right in front: closer, shorter)."""
        closest = self.closest if closest is None else closest
        max_range = self.max_range if max_range is None else max_range
        seen = np.abs(BEARING_DEG) < self.fov_deg
        k = np.flatnonzero(seen)
        angles = self.pose[2] + np.radians(BEARING_DEG[k, None] + SUB_RAYS_DEG[None, :])
        d = self.ray_ranges(angles.ravel()).reshape(len(k), -1).min(1)
        if self.noise:
            d = np.maximum(d + self.noise * d * d * self.rng.standard_normal(len(k)), 0.02)
        d = np.maximum(d, closest)  # nearer than the image bottom: reads at the image bottom
        return scan_arrays(k, np.minimum(d, max_range), closest, max_range, self.near_tol)

    def step(self, v_cmd, w_cmd, dt):
        """Advance by dt (s) under the command (m/s, rad/s). Returns the forward acceleration (m/s^2)."""
        k = 1.0 - math.exp(-dt / self.lag_s)
        k_w = k if self.w_lag_s == self.lag_s else 1.0 - math.exp(-dt / self.w_lag_s)
        v_old = self.v
        self.v += k * (v_cmd - self.v)
        self.w += k_w * (w_cmd - self.w)
        self.pose[2] += self.w * dt
        self.pose[0] += self.v * dt * math.cos(self.pose[2])
        self.pose[1] += self.v * dt * math.sin(self.pose[2])
        return (self.v - v_old) / dt

    def clearance(self):
        """Distance from the robot centre to the nearest wall (m). Below `radius`: a collision."""
        a, b = self.walls[:, :2], self.walls[:, 2:]
        e = b - a
        u = np.clip(((self.pose[:2] - a) * e).sum(1) / (e * e).sum(1), 0.0, 1.0)
        return float(np.hypot(*(a + u[:, None] * e - self.pose[:2]).T).min())

    def collided(self):
        return self.clearance() < self.radius


DT = 1 / 8.7  # s, frame period (the robot's camera rate)
CAM_H = 0.35  # m, camera height given with the driving frames
# a head glance: the camera looks down at the floor right in front of the robot
GLANCE_CAM_H, GLANCE_CLOSEST, GLANCE_RANGE, GLANCE_FRAMES = 0.33, 0.15, 0.6, 2
GLANCE_AGAIN_M = 0.2  # no second glance from the same place


def run(sim, step, observe=None, steps=400, dt=DT, extra_steps=3, delay=0):
    """Closed loop: at each frame, `step(t, dt, pose, scan, cam_h, a, v, w, v_prev)` gets the simulator's
    pose (perfect odometry), scan (the dict of Sim.scan) and measured motion, and returns an object with
    v, w (the command, applied until the next frame), finished and blind_stop.
    delay: the base gets each command this many frames late (perception-to-command latency; 0: at once).
    observe(t, pose, scan, cam_h), if given, gets the frames of a head glance (looking down, robot held
    still), taken when a step reports blind_stop and the robot has moved GLANCE_AGAIN_M since the last one.
    Stops after `steps` steps, or `extra_steps` steps after the first one that reports finished.
    Returns (frames, summary): one dict per frame, in order (op "step" or "observe", the inputs given, the
    closest and max_range of its scan and, for a step, the output `out`), and dict(min_clearance, travelled,
    finished_at: index of the first finished step or None, collided)."""
    frames, k = [], 0  # k: frame count, steps and glance frames
    a = v_prev = 0.0
    glanced_at, finished_at, stop_at = None, None, steps  # glanced_at: position of the last glance
    min_clearance, travelled = sim.clearance(), 0.0
    late = [(0.0, 0.0)] * delay  # commands on their way to the base, oldest first

    def advance(v_cmd, w_cmd):
        nonlocal a, min_clearance, travelled
        late.append((v_cmd, w_cmd))
        a = sim.step(*late.pop(0), dt)
        min_clearance = min(min_clearance, sim.clearance())
        travelled += abs(sim.v) * dt

    n = 0  # steps
    while n < stop_at:
        pose, scan = tuple(float(x) for x in sim.pose), sim.scan()
        frame = dict(op="step", t=k * dt, pose=pose, scan=scan, closest=sim.closest, max_range=sim.max_range,
                     cam_h=CAM_H, dt=dt, a=a, v=sim.v, w=sim.w, v_prev=v_prev)
        out = step(frame["t"], dt, pose, scan, CAM_H, a, sim.v, sim.w, v_prev)
        frames.append(dict(frame, out=out))
        k, n = k + 1, n + 1
        if out.finished and finished_at is None:
            finished_at, stop_at = n - 1, min(steps, n + extra_steps)
        moved = glanced_at is None or math.dist(glanced_at, pose[:2]) >= GLANCE_AGAIN_M
        if observe is not None and out.blind_stop and moved:
            glanced_at = pose[:2]
            for _ in range(GLANCE_FRAMES):
                advance(0.0, 0.0)
                pose, scan = tuple(float(x) for x in sim.pose), sim.scan(GLANCE_CLOSEST, GLANCE_RANGE)
                observe(k * dt, pose, scan, GLANCE_CAM_H)
                frames.append(dict(op="observe", t=k * dt, pose=pose, scan=scan, closest=GLANCE_CLOSEST,
                                   max_range=GLANCE_RANGE, cam_h=GLANCE_CAM_H))
                k += 1
            v_prev = 0.0
            advance(0.0, 0.0)
        else:
            v_prev = out.v
            advance(out.v, out.w)
    return frames, dict(min_clearance=min_clearance, travelled=travelled, finished_at=finished_at,
                        collided=min_clearance < sim.radius)
