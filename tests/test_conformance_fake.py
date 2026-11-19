"""The conformance suite (navette.testing.conformance) on the fake robot: it passes, and fake robots that break
the contract (wrong sign, wrong unit, no dead-man, bad clock...) are caught."""
import math
import time

import numpy as np
import pytest

from navette import interfaces as itf
from navette.testing import conformance as conf
from navette.testing.fakes import (FakeBase, FakeCamera, FakeImu, FakeRobot, FakeWorld, ManualClock,
                                   camera_rotation)


def rig(**base_kw):
    clock = ManualClock()
    world = FakeWorld(clock)
    return clock, world, FakeBase(world, **base_kw)


def kw(clock):
    return {"clock": clock, "sleep": clock.sleep}


# ------------------------------------------------------------------ the fake robot conforms

def test_fake_robot_passes_everything():
    clock = ManualClock()
    robot = FakeRobot(FakeWorld(clock, wall_x=2.0))
    conf.check_robot(robot, up_c=robot.imu.up_c, drive=True, inject_fault=lambda: robot.base.fault("motor fault"),
                     **kw(clock))
    world = robot.world
    world.advance()
    assert world.x > 0.05 and world.theta > 0.1 and world.v == 0  # it did drive and turn, and is stopped
    assert robot.base.calls[0][0] == "command" and ("start",) in robot.base.calls and ("park",) in robot.base.calls


def test_fake_robot_passes_on_the_real_clock():
    """As a real adapter is checked: time.monotonic and time.sleep (short drives)."""
    world = FakeWorld()
    conf.check_camera(FakeCamera(world), n_frames=3)
    conf.check_imu(FakeImu(world), duration_s=0.2)
    conf.check_base(FakeBase(world, deadman_s=0.2, start_s=0.1), drive_s=0.3, deadman_s=0.3)


def test_fake_camera_sees_the_wall_where_it_is():
    """The fake world's own geometry: the floor mask ends where the wall is."""
    from navette import FreeSpace

    clock = ManualClock()
    world = FakeWorld(clock, wall_x=1.5)
    cam = FakeCamera(world)
    fs = FreeSpace(cam.rays, cam.rays_low, cam.mask_shape)
    up_c = FakeImu(world).up_c
    scan = fs.scan(cam.segment(cam.read()), up_c, cam.height)
    ahead = np.abs(np.asarray(scan.bearing_deg)) < 3
    assert np.asarray(scan.obstacle)[ahead] == pytest.approx(1.5, abs=0.1)
    world.x, world.theta = 0.5, np.radians(20)  # closer, and turned left: the wall is nearer on the right
    clock.t += 0.1
    scan = fs.scan(cam.segment(cam.read()), up_c, cam.height)
    b, obstacle = np.asarray(scan.bearing_deg), np.asarray(scan.obstacle)
    assert obstacle[np.abs(b + 20) < 1.5] == pytest.approx(1.0, abs=0.1)  # straight at the wall: bearing -20 deg
    assert obstacle[np.abs(b) < 1.5] == pytest.approx(1.0 / np.cos(np.radians(20)), abs=0.1)


# ------------------------------------------------------------------ broken bases are caught

@pytest.mark.parametrize("base_kw, message", [
    ({"v_sign": -1.0}, "forward must be \\+x"),
    ({"w_sign": -1.0}, "counter-clockwise must be \\+theta"),
    ({"v_sign": 100.0}, "units are m, m/s"),          # a base that takes cm/s
    ({"w_sign": 5.0}, "units are rad, rad/s"),        # another unit (5x: stays under half a turn)
    ({"deadman": False}, "dead-man"),
    ({"start_s": 60.0}, "not DRIVING 10.0 s after start"),
])
def test_wrong_base_is_caught(base_kw, message):
    clock, world, base = rig(**base_kw)
    with pytest.raises(AssertionError, match=message):
        conf.check_base(base, **kw(clock))


def broken_base(**methods):
    clock, world, base = rig()
    for name, fn in methods.items():
        setattr(base, name, fn.__get__(base))
    return clock, base


def test_base_moving_before_start_is_caught():
    def command(self, v, w):  # no session check
        self.world.advance()
        self.world.v, self.world.w = v, w
        self.world.t_zero = self.world.clock() + self.deadman_s
    clock, base = broken_base(command=command)
    with pytest.raises(AssertionError, match="before start\\(\\) moved"):
        conf.check_base(base, **kw(clock))


def test_stop_not_latched_is_caught():
    def stop(self):  # stops, but the session stays open
        self.world.advance()
        self.world.v = self.world.w = 0.0
    clock, base = broken_base(stop=stop)
    with pytest.raises(AssertionError, match="not latched"):
        conf.check_base(base, **kw(clock))


def test_stop_that_latches_but_does_not_brake_is_caught():
    """stop() closes the session and sends nothing: the base rolls on until its dead-man fires (0.135 m from
    0.3 m/s with this fake)."""
    def stop(self):
        self.calls.append(("stop",))
        self._session = False
    clock, base = broken_base(stop=stop)
    with pytest.raises(AssertionError, match="stop\\(\\) did not brake"):
        conf.check_base(base, **kw(clock))


def test_a_base_that_faults_after_its_dead_man_conforms():
    """The contract allows both: the session goes on after the dead-man (the fake), or the base reports FAULT
    until the next start() (the reference robot's base). A FAULT without a reason is still caught."""
    def faulting(reason):
        def command(self, v, w):
            if self.status_now() == itf.DRIVING and self.world.advance() >= self.world.t_zero:
                self.fault(reason)  # the dead-man fired before this command
            return FakeBase.command(self, v, w)

        def status(self):
            if self._session and self.world.advance() >= self.world.t_zero:
                self.fault(reason)
            return FakeBase.status(self)

        def start(self):
            FakeBase.start(self)
            self.world.t_zero = math.inf  # a new session: the old dead-man is history
        return broken_base(command=command, status=status, start=start)

    clock, base = faulting("event deadman")
    conf.check_base(base, **kw(clock))
    assert base.calls.count(("start",)) == 2  # a new session after the FAULT
    clock, base = faulting("")
    with pytest.raises(AssertionError, match="FAULT after the dead-man without a reason"):
        conf.check_base(base, **kw(clock))


def test_commanded_velocity_reported_as_measured_is_caught():
    """Odometry.v must be measured: a base reporting 0 (or nothing useful) while it moves fails."""
    def odometry(self):
        o = FakeBase.odometry(self)
        return o._replace(v=0.0)
    clock, base = broken_base(odometry=odometry)
    with pytest.raises(AssertionError, match="Odometry.v"):
        conf.check_base(base, **kw(clock))


def test_odometry_on_another_clock_is_caught():
    def odometry(self):
        return FakeBase.odometry(self)._replace(t=time.time())  # wall clock
    clock, base = broken_base(odometry=odometry)
    with pytest.raises(AssertionError, match="not on the loop's clock"):
        conf.check_base(base, **kw(clock))


def test_park_never_done_and_fault_without_reason_are_caught():
    clock, base = broken_base(park=lambda self: False)
    with pytest.raises(AssertionError, match="park\\(\\) did not return True"):
        conf.check_base(base, **kw(clock))
    clock, world, base = rig()
    with pytest.raises(AssertionError, match="FAULT without a reason"):
        conf.check_base(base, inject_fault=lambda: base.fault(""), **kw(clock))
    clock, world, base = rig()
    with pytest.raises(AssertionError, match="no FAULT 2 s after the fault"):
        conf.check_base(base, inject_fault=lambda: None, **kw(clock))


def test_start_refused_is_a_base_error():
    clock, world, base = rig()
    base.refuse = "not calibrated"
    with pytest.raises(itf.StartRefused, match="not calibrated"):
        base.start()
    assert issubclass(itf.StartRefused, itf.BaseError) and base.status().state == itf.STOPPED


# ------------------------------------------------------------------ broken cameras and IMUs are caught

def camera_rig():
    clock = ManualClock()
    return clock, FakeCamera(FakeWorld(clock))


def test_wrong_camera_is_caught():
    clock, cam = camera_rig()
    cam.rays = cam.rays[:, [1, 0, 2]]  # x and y swapped
    with pytest.raises(AssertionError, match="rays: y must grow with the row"):
        conf.check_camera(cam, **kw(clock))
    clock, cam = camera_rig()
    cam.rays_low = cam.rays
    with pytest.raises(AssertionError, match="rays_low"):
        conf.check_camera(cam, **kw(clock))
    clock, cam = camera_rig()
    cam.mask_shape = (cam.mask_shape[1], cam.mask_shape[0])
    with pytest.raises(AssertionError):
        conf.check_camera(cam, **kw(clock))
    clock, cam = camera_rig()
    seg = cam.segment
    cam.segment = lambda frame: seg(frame).astype(np.uint8)
    with pytest.raises(AssertionError, match="expected a bool array"):
        conf.check_camera(cam, **kw(clock))
    clock, cam = camera_rig()
    seg = cam.segment
    cam.segment = lambda frame: np.asfortranarray(seg(frame))  # same values, column-major memory
    with pytest.raises(AssertionError, match="C-contiguous"):
        conf.check_camera(cam, **kw(clock))
    clock, cam = camera_rig()
    cam.latency_s = -5.0  # frames stamped in the future
    with pytest.raises(AssertionError, match="not on the loop's clock"):
        conf.check_camera(cam, **kw(clock))
    clock, cam = camera_rig()
    cam.read = lambda: None
    with pytest.raises(AssertionError, match="only 0 frames"):
        conf.check_camera(cam, **kw(clock))


def test_camera_mounting_outside_the_supported_limits_is_caught():
    """The navigator takes straight ahead in the scan for the driving direction and ignores the camera's
    horizontal offset (the Camera contract): a camera looking sideways or far from the base centre fails."""
    yaw = lambda deg: np.array([[np.cos(np.radians(deg)), -np.sin(np.radians(deg)), 0],  # noqa: E731
                                [np.sin(np.radians(deg)), np.cos(np.radians(deg)), 0], [0, 0, 1.0]])
    clock, cam = camera_rig()
    cam.T_base_cam[:3, :3] = yaw(90) @ camera_rotation(20.0)  # looking 90 deg to the left
    cam.T_base_cam[:3, 3] = [0.4, 0.0, 0.0]                   # and 0.4 m ahead of the base centre
    with pytest.raises(AssertionError, match="looks \\+90 deg off the base's forward"):
        conf.check_camera(cam, **kw(clock))
    clock, cam = camera_rig()
    cam.T_base_cam[:3, 3] = [0.4, 0.0, 0.3]
    with pytest.raises(AssertionError, match="0.40 m from the base centre"):
        conf.check_camera(cam, **kw(clock))
    conf.check_camera(cam, max_offset_m=0.5, **kw(clock))  # a large robot says so
    # what a head on a neck gives passes: a few degrees, a few centimetres, any height
    clock, cam = camera_rig()
    cam.T_base_cam[:3, :3] = yaw(-6) @ camera_rotation(20.0)
    cam.T_base_cam[:3, 3] = [0.04, 0.01, 0.3]
    conf.check_camera(cam, **kw(clock))
    cam.T_base_cam[:3, :3] = yaw(-12) @ camera_rotation(20.0)
    with pytest.raises(AssertionError, match="looks -12 deg off"):
        conf.check_camera(cam, **kw(clock))


def imu_rig(scale_acc=1.0, scale_gyro=1.0, bias=0.0):
    clock = ManualClock()
    imu = FakeImu(FakeWorld(clock))
    read = imu.read
    imu.read = lambda: [(t, g * scale_gyro + bias, a * scale_acc) for t, g, a in read()]
    return clock, imu


def test_wrong_imu_is_caught():
    clock, imu = imu_rig()
    conf.check_imu(imu, **kw(clock))  # no expected 'up' given: only "pointing up"
    clock, imu = imu_rig(scale_acc=1 / 9.81)  # in g
    with pytest.raises(AssertionError, match="9.81 m/s\\^2"):
        conf.check_imu(imu, **kw(clock))
    clock, imu = imu_rig(scale_acc=-1.0)  # gravity instead of the specific force
    with pytest.raises(AssertionError, match="must point up"):
        conf.check_imu(imu, **kw(clock))
    clock, imu = imu_rig(bias=np.radians(30))  # a gyro in deg/s shows its bias as tens of "rad/s"
    with pytest.raises(AssertionError, match="rad/s"):
        conf.check_imu(imu, **kw(clock))
    clock, imu = imu_rig()
    with pytest.raises(AssertionError, match="deg from the expected 'up'"):
        conf.check_imu(imu, up_c=[0.0, 0.0, 1.0], **kw(clock))  # not in the camera frame
    clock, imu = imu_rig()
    first = imu.read
    kept = []
    imu.read = lambda: kept.extend(first()) or list(kept)  # returns old samples again
    with pytest.raises(AssertionError, match="returned twice"):
        conf.check_imu(imu, **kw(clock))
    clock, imu = imu_rig()
    imu.read = lambda: []
    with pytest.raises(AssertionError, match="0 IMU samples"):
        conf.check_imu(imu, **kw(clock))


@pytest.mark.parametrize("which, bad, message", [
    (0, np.zeros(3), "sample 0 of the first read\\(\\): \\|acc\\| = 0.00, not a measurement"),  # a driver's zero-filled first read
    (0, np.full(3, np.nan), "sample 0 of the first read\\(\\) is not finite"),
    (30, np.zeros(3), "of read\\(\\): \\|acc\\| = 0.00"),                    # one bad sample later on
    (30, np.array([0.0, np.inf, 0.0]), "of read\\(\\) is not finite"),
])
def test_imu_samples_that_are_not_measurements_are_caught(which, bad, message):
    """The median over the check's samples hid them: one such sample made the attitude NaN for good."""
    clock = ManualClock()
    imu = FakeImu(FakeWorld(clock))
    read, n = imu.read, [0]

    def with_a_bad_sample():
        out = read()
        for k in range(len(out)):
            if n[0] == which:
                out[k] = (out[k][0], out[k][1], bad)
            n[0] += 1
        return out
    imu.read = with_a_bad_sample
    with pytest.raises(AssertionError, match=message):
        conf.check_imu(imu, **kw(clock))
