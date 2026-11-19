"""The one source of the robot's pose. Nothing else in the runtime turns odometry into a pose.

Frames as REP 105: `odom` (continuous, drifts) and `map` (corrected, may jump). The navigator's grid lives in
`map`. Today there is no correction: map = odom, and the pose is the base's odometry. A gyro-fused heading or
a SLAM pose replaces this class behind the same two calls.
"""


class OdometryLocalization:
    def __init__(self):
        self._pose = None

    def update(self, odom):
        """odom: navette.interfaces.Odometry read when a frame arrived, or None (the base has none yet)."""
        self._pose = None if odom is None else (odom.x, odom.y, odom.theta)

    def pose(self):
        """(x, y, theta) of the base frame in the map frame (m, m, rad counter-clockwise), or None."""
        return self._pose
