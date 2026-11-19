"""The robot contract: what the runtime (navette.runtime) needs from a robot, and nothing else.

A robot adapter is a `Robot`: a `Camera`, an `Imu`, a `Base` (or None) and optionally a `Head`. The runtime
never imports a robot SDK; everything robot-specific (lens, floor model, IMU driver, base protocol) sits
behind these classes. `navette.testing.conformance` checks an adapter against this contract, and
`navette.testing.fakes` is a fake robot that passes it.

Conventions (all of them, everywhere):
  clock        time.monotonic() of the process running the loop, seconds. Every `t` below is on that clock.
  base frame   x forward, y left, z up, right-handed. Angles counter-clockwise seen from above, radians.
  camera frame x right, y down, z along the optical axis (OpenCV).
  odom frame   fixed frame of the base's odometry: x forward and y left where the odometry was anchored.
  units        m, s, rad, m/s, rad/s, m/s^2.

Threads: the perception thread calls Camera.read / segment, Imu.read and Base.odometry. The command thread
(20 Hz) calls Base.status, command and park. Base.start and Base.stop come from any thread (page, stop
button), but the controller serialises start / stop / status / command / park under one lock: a driver only
has to make odometry() safe against the others.
"""
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Protocol

import numpy as np

# Base states (BaseStatus.state)
STOPPED = "STOPPED"    # no drive session: command() is ignored
STARTING = "STARTING"  # start() was called, the base is getting ready (a balancing base stands up)
DRIVING = "DRIVING"    # the base applies command()
FAULT = "FAULT"        # the base cannot drive, or should not: BaseStatus.fault says why


class BaseError(RuntimeError):
    """A base call failed (link down, write error): the command did not reach the base. The controller
    answers with stop() and goes idle; the base's own dead-man stops it if stop() cannot reach it either."""


class StartRefused(BaseError):
    """Base.start() refused before doing anything (not ready, not calibrated...): nothing was sent, nothing
    is to be undone. The message is shown to the user."""


@dataclass
class Frame:
    """One camera frame, as returned by Camera.read()."""
    t: float                 # capture time (not arrival: the adapter subtracts its camera's lag)
    index: int               # frame number, increasing; the name of the frame in the adapter's own log
    T_base_cam: np.ndarray   # 4x4 pose of the camera frame in the base frame at capture (fixed mount: constant)
    image: np.ndarray | None = None  # BGR uint8 picture for the status page preview, if the adapter has one
    under: np.ndarray | None = None  # optional second mask (mask_shape) drawn on the preview; segment() may set it
    ms: dict = field(default_factory=dict)  # optional per-stage times of read() for the run log, e.g. head, save


class Camera(Protocol):
    """The camera and whatever turns its picture into a floor mask (a segmentation model, a depth threshold).

    The height of the camera is not part of this class: the runtime computes it from T_base_cam, the IMU's
    'up' and the config's camera.body_z (height of the base frame's origin above the floor).
    Failure: read() and segment() may raise; the loop then ends and the base is stopped."""

    rays: np.ndarray       # (h*w, 3) float64 viewing rays of the mask pixel centres, camera frame, row-major, z = 1
    rays_low: np.ndarray   # (h*w, 3) the same through each pixel's lower edge (navette.rays.mask_rays dv=0.5)
    mask_shape: tuple      # (h, w) of the masks segment() returns

    def read(self) -> Frame | None:
        """The next frame, each frame once, in capture order. Blocks at most a few tens of ms; None if no new
        frame came. Must be cheap: the runtime reads the base's odometry right after it, as the pose of this
        frame (the expensive part belongs in segment())."""

    def segment(self, frame: Frame) -> np.ndarray:
        """Floor mask of a frame from read(): bool (h, w), True = floor the robot may drive on.
        Not called for frames the loop does not use (IMU not ready, head moving)."""


class Imu(Protocol):
    """Inertial unit rigidly fixed to the camera."""

    name: str  # label of the source in use, for the status page and the run log

    def read(self) -> list:
        """Every sample since the last call, oldest first, each once: [(t, gyro, acc)].
        t: sample time, strictly increasing over all calls. gyro: (3,) rad/s. acc: (3,) m/s^2, the specific
        force: at rest it points up with norm ~9.81. Both in the camera frame (the adapter applies the
        IMU-to-camera rotation). Never blocks; an empty list if nothing new. A source that stops delivering
        just returns empty lists: the runtime stops stepping when the newest sample is too old
        (perception.imu_max_age_s), and the stale-perception watchdog then halts the base."""


class Odometry(NamedTuple):
    """Base.odometry(): pose in the odom frame and measured velocity."""
    t: float      # time the measurement was received
    x: float      # m
    y: float      # m
    theta: float  # rad, counter-clockwise
    v: float      # m/s, measured (not the commanded value), > 0 forward
    w: float      # rad/s, measured, > 0 counter-clockwise


@dataclass(frozen=True)
class BaseStatus:
    state: str                 # STOPPED, STARTING, DRIVING or FAULT
    fault: str = ""            # why, when FAULT (shown to the user and kept as the halt reason)
    low_battery: bool = False  # shown on the status page


class Base(Protocol):
    """The mobile base.

    Dead-man (required): the base must stop by itself if command() is not called for a short time (the
    reference robot: 0.5 s), whatever happens to this process or to the link. The controller calls command()
    every control period from start() until stop(), with zeros when it has nothing to say (base starting, no
    navigator output yet), so that the base can tell "hold still" from "the controller is gone".

    Session: start() opens it, stop() closes it. Outside a session command() must do nothing (latched): a
    STOP from one thread cannot be undone by a command already on its way from another."""

    balancing: bool  # True for a self-balancing base: it pitches when it accelerates, so the runtime passes
    #                  its measured motion to the navigator's tilt trust. False: the navigator gets zeros.

    def odometry(self) -> Odometry | None:
        """The latest odometry; None while the base has none (no telemetry yet). Any thread, never blocks.
        The pose must be consistent with command(): driving v > 0 increases x at theta = 0, w > 0 increases
        theta."""

    def start(self) -> None:
        """Open a drive session and get ready (stand up, enable the motors). Returns at once: status() says
        STARTING, then DRIVING. Raises StartRefused if the base cannot start now, BaseError on a failure."""

    def command(self, v: float, w: float) -> int | None:
        """Velocity reference, v m/s forward and w rad/s counter-clockwise, applied until the next one (and
        zeroed by the dead-man). Ignored outside a session. Returns an optional id of the command for the run
        log (a sequence number the base echoes), or None. Raises BaseError if it could not be sent."""

    def stop(self) -> None:
        """Close the session and bring the base to rest now (controlled deceleration). Latched before
        anything is sent: command() does nothing until the next start(), even if stop() itself raises
        BaseError. Allowed in any state, any number of times."""

    def park(self) -> bool:
        """After stop(): put the base in its resting state once it has stopped moving (a balancing base sits
        down; others may have nothing to do). Called every control period until it returns True. Must return
        True within a few seconds whatever the base does. Not called after a watchdog halt: the base is then
        left as stop() left it."""

    def status(self) -> BaseStatus:
        """Called once per control period by the command thread, also while idle (a driver may use it as
        its periodic tick). During a session, anything that means the base is not doing what the controller
        asks is a FAULT with its reason: a fault or safety event reported by the base, telemetry gone silent,
        control taken by someone else. The controller then calls stop() and goes idle."""


class Head(Protocol):
    """Optional: a camera that can look down at the ground just ahead of the robot, which it cannot see at
    its navigation pose. The robot holds still during a glance."""

    move_s: float  # duration of each move (s); the runtime waits this long, plus a settling time

    def look_down(self) -> None:
        """Start moving to the glance pose. Must not block."""

    def look_back(self) -> None:
        """Start moving back to the navigation pose. Must not block."""

    def is_back(self) -> bool:
        """Measured: the head is at the navigation pose."""


class Robot(Protocol):
    """What an adapter module's open(...) returns."""

    camera: Camera
    imu: Imu
    base: Base | None   # None: no base, the commands are only logged
    head: Head | None   # None: no glance

    def on_stop(self, stop: Callable[[str], None]) -> None:
        """Give the robot's own stop control (a button, an antenna), if it has one, the function to call
        with a reason. It may be called from any thread."""

    def meta(self) -> dict:
        """Adapter data for the run's meta.json (called at the start and at the end of a run)."""

    def close(self) -> None:
        """Release everything (threads, links, files)."""
