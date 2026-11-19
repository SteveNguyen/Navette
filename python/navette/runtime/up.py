"""'up' in the camera frame at a frame's capture time, from the Imu's samples through the core's attitude
filter (navette.UpFilter, injected: `make_filter(g)` returns an object with update(t, gyro, acc) -> up)."""
import math
from collections import deque

import numpy as np

HISTORY_S = 0.5  # 'up' history kept for the query at a frame's capture time (frames arrive after their samples)


class LiveUp:
    """Attitude filter fed sample by sample. Its g is the median |acc| of the first g_window_s of samples
    (an accelerometer's scale can be off by ~1 %, which the filter's acceleration gate would see), or `g` if
    given (a replay passes the median of the whole recording)."""

    def __init__(self, make_filter, g_window_s=1.0, max_age_s=0.1, g=None):
        self.make_filter, self.g_window_s, self.max_age_s = make_filter, g_window_s, max_age_s
        self.upf = None if g is None else make_filter(g)
        self.pending, self.t_last = [], -math.inf
        self.hist = deque()  # (t, up) of the last HISTORY_S

    def add(self, samples):
        """samples: [(t, gyro, acc)] from Imu.read()."""
        if samples:
            self.t_last = max(self.t_last, samples[-1][0])
        if self.upf is None:
            self.pending += samples
            if not self.pending or self.pending[-1][0] - self.pending[0][0] < self.g_window_s:
                return
            # a zero or non-finite reading (a driver's first read, a glitch) is not a measurement of g
            norms = [n for n in (float(np.linalg.norm(s[2])) for s in self.pending) if math.isfinite(n) and n > 0]
            if not norms:
                self.pending = []  # nothing usable in this window: measure on the next one
                return
            self.upf = self.make_filter(float(np.median(norms)))
            samples, self.pending = self.pending, []
        for t, gyro, acc in samples:
            # A sample with a non-finite value is no sample: not fed, and not in the history, so that an IMU
            # delivering only those goes stale (up_at) instead of holding its last 'up' for ever.
            if not (np.isfinite(gyro).all() and np.isfinite(acc).all()):
                continue
            # None: the filter has no 'up' yet (it does not start from a zero accelerometer reading).
            # Whatever the filter, only a finite 'up' goes into the history: up_at() must never hand a NaN
            # to the scan.
            u = self.upf.update(t, gyro, acc)
            if u is not None and np.isfinite(u).all():
                self.hist.append((t, u))
        while self.hist and self.hist[0][0] < self.t_last - HISTORY_S:
            self.hist.popleft()

    def up_at(self, t):
        """'up' at t, linearly interpolated in the history (clamped at its ends); None before the filter
        has an 'up', or if the newest valid sample is more than max_age_s older than t (the interpolation
        clamps: an IMU that stopped, or only delivers garbage, would give the last 'up' forever), or if the
        result is not a finite unit vector. The caller takes no navigation step on None."""
        if not self.hist or t - self.hist[-1][0] > self.max_age_s:
            return None
        ts = np.array([h[0] for h in self.hist])
        us = np.array([h[1] for h in self.hist], float)
        u = np.array([np.interp(t, ts, us[:, j]) for j in range(3)])
        u = u / np.linalg.norm(u)
        return u if np.isfinite(u).all() else None
