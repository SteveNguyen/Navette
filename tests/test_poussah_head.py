"""The navigation head pose is checked after the move: a head that did not get there (motors disabled) refuses
the run instead of navigating on a view of the robot's own body."""
import sys
import types

import numpy as np
import pytest

from navette.robots.reachy_poussah import robot


def rotation(roll_deg, pitch_deg, yaw_deg):
    """Head pose matrix (4x4) with the convention robot.measured_rpy reads back (pitch > 0: looking down)."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = Rz @ Ry @ Rx
    return m


class FakeMini:
    """Moves its head to the commanded pitch only if its motors are enabled."""

    def __init__(self, enabled, rest=(0.0, -0.3, 13.0)):
        self.enabled, self.pose = enabled, rotation(*rest)

    def goto_target(self, head, antennas, duration, body_yaw):
        if self.enabled:
            self.pose = head

    def get_current_head_pose(self):
        return self.pose


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    utils = types.ModuleType("reachy_mini.utils")
    utils.create_head_pose = lambda roll, pitch, yaw, degrees: rotation(roll, pitch, yaw)
    monkeypatch.setitem(sys.modules, "reachy_mini", types.ModuleType("reachy_mini"))
    monkeypatch.setitem(sys.modules, "reachy_mini.utils", utils)
    monkeypatch.setattr(robot.time, "sleep", lambda s: None)


def test_head_at_the_navigation_pose_is_accepted():
    r, p, y = robot.set_nav_head(FakeMini(enabled=True), 15.0, lambda m: None)
    assert p == pytest.approx(15.0) and abs(r) < 1e-9 and abs(y) < 1e-9


def test_head_that_did_not_move_refuses_the_run():
    with pytest.raises(RuntimeError, match="not at the navigation pose"):
        robot.set_nav_head(FakeMini(enabled=False), 15.0, lambda m: None)
