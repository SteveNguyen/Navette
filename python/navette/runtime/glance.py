"""Head glance at the ground the camera cannot see at its navigation pose (navette.interfaces.Head)."""
import math


class Glance:
    """Head glance down at the camera's blind zone, robot holding still: head down, settle, n_frames frames
    into the map (Navigator.observe), head back, settle. Started at START (empty map) and when a stop comes
    only from a never-seen blind zone (the navigator output's blind_stop), not again within cooldown_s unless
    the robot moved since the last one.

    Head moves do not block (the loop keeps reading frames, the IMU and feeding the controller a zero-velocity
    hold, so the stale-perception watchdog stays fed). frame(t) tells the loop what to do with each frame while
    active: 'move' (head moving or settling: hold, frame not used), 'look' (head down and settled: into the
    map, hold); None when not active.

    Before the glance ends, head.is_back() is checked and the move back sent again up to `retries` times (a
    failed or overtaken move would leave the head looking down). `failed` is set if it is still not back: the
    glance ends anyway (the camera pose is measured, so the geometry stays right; the view is shorter)."""

    def __init__(self, head, n_frames=2, settle_s=0.3, cooldown_s=5.0, moved_m=0.1, turned_deg=20.0, retries=2):
        self.head, self.retries, self.tries, self.failed = head, retries, 0, False
        self.n_frames, self.wait_s, self.cooldown_s = n_frames, head.move_s + settle_s, cooldown_s
        self.moved_m, self.turned = moved_m, math.radians(turned_deg)
        self.phase, self.t_phase, self.n = None, 0.0, 0
        self.t_end, self.pose_end = -math.inf, None
        self.count = 0

    @property
    def active(self):
        return self.phase is not None

    def begin(self, t, pose, force=False):
        """Start a glance unless one is under way, or (without force) one ended less than cooldown_s ago
        and the robot has not moved since. Returns True if started."""
        if self.active:
            return False
        if not force and t - self.t_end < self.cooldown_s and self.pose_end is not None:
            dth = abs((pose[2] - self.pose_end[2] + math.pi) % (2 * math.pi) - math.pi)
            if math.hypot(pose[0] - self.pose_end[0], pose[1] - self.pose_end[1]) < self.moved_m and dth < self.turned:
                return False
        self.phase, self.t_phase, self.count = "down", t, self.count + 1
        self.head.look_down()
        return True

    def frame(self, t, pose=None):
        if self.phase == "down":
            if t - self.t_phase < self.wait_s:
                return "move"
            self.phase, self.n = "look", 0
        if self.phase == "look":
            self.n += 1
            if self.n >= self.n_frames:
                self.phase, self.t_phase, self.tries, self.failed = "up", t, 0, False
                self.head.look_back()
            return "look"
        if self.phase == "up":
            if t - self.t_phase < self.wait_s:
                return "move"
            if not self.head.is_back():
                if self.tries < self.retries:
                    self.tries, self.t_phase = self.tries + 1, t
                    self.head.look_back()
                    return "move"
                self.failed = True
            self.phase, self.t_end, self.pose_end = None, t, pose
        return None
