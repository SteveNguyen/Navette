"""The one source of the robot's pose. Nothing else in the runtime turns odometry into a pose.

Frames as REP 105: `odom` (continuous, drifts) and `map` (corrected, may jump). The navigator's grid lives in
`map`. Today there is no correction: map = odom, and the pose is the base's odometry. A gyro-fused heading or
a SLAM pose replaces this class behind the same two calls.
"""
import math


class OdometryLocalization:
    """at_capture: the pose of a frame is the odometry interpolated at the frame's capture time, instead of the
    latest odometry when the frame arrived (the default, the Python prototype's). The odometry is read once
    per frame, when it arrives, and a frame arrives one camera latency after its capture (the reference
    robot: 38 ms): the latest odometry is that much too new, ~1 deg of heading at the base's turn rate, on
    every scan put in the map. Measured on recorded runs by aligning consecutive frames on the floor: the
    best alignment came with the poses shifted ~40 ms back. The samples are stamped with their own receive
    times (Odometry.t), so the pose at the capture time is a linear interpolation between the last two reads
    (heading unwrapped); never extrapolated (a capture after the latest sample takes the latest)."""

    def __init__(self, at_capture=False):
        self.at_capture = at_capture
        self._pose = None
        self._samples = []  # the last two distinct odometry samples (t, x, y, theta), oldest first

    def update(self, odom, t_capture=None):
        """odom: navette.interfaces.Odometry read when a frame arrived, or None (the base has none yet: the
        history is dropped as well). t_capture: the frame's capture time (Frame.t, same clock as Odometry.t);
        used with at_capture."""
        if odom is None:
            self._pose, self._samples = None, []
            return
        if not self._samples or odom.t > self._samples[-1][0]:  # a new sample (the same one can be read twice)
            self._samples = (self._samples + [(odom.t, odom.x, odom.y, odom.theta)])[-2:]
        if self.at_capture and t_capture is not None:
            self._pose = self._at(t_capture)
        else:
            self._pose = (odom.x, odom.y, odom.theta)

    def _at(self, t):
        t1, x1, y1, th1 = self._samples[-1]
        if len(self._samples) < 2 or t >= t1:
            return (x1, y1, th1)
        t0, x0, y0, th0 = self._samples[0]
        a = max(0.0, (t - t0) / (t1 - t0))  # before the older sample: the older one
        return (x0 + a * (x1 - x0), y0 + a * (y1 - y0), th0 + a * math.remainder(th1 - th0, math.tau))

    def pose(self):
        """(x, y, theta) of the base frame in the map frame (m, m, rad counter-clockwise), or None."""
        return self._pose
