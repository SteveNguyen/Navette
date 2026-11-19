"""Client of the Poussah base's navigation link: the wire protocol (firmware docs/nav-link-api.md). The Base
contract on top of it is poussah_base.PoussahBase.

The base speaks text lines over Bluetooth LE (Nordic UART service, the primary path) or USB serial (fallback):
it sends `[nav]` telemetry, `[event]`, `[hello]` and `[sync]` lines; we send `nh nr<hz> nq<id> nb<ms> nz`,
the velocity command `nv<v>,<w>,<seq>`, the halt `ns`, and the legacy `S0` (stand up) / `S1` (sit down).

    link = BaseLink(BleTransport("rmini_wheels_3"), rate=50)   # or SerialTransport("/dev/ttyACM0")
    link.connect()                    # opens the transport, sends nh, nr50, nq (one clock sync)
    link.sync()                       # call ~1 Hz: nq round trips keep the base -> host clock fit
    link.send_velocity(0.1, 0.0)      # nv, numbered; >= 2 Hz or the base's dead-man (500 ms) halts it
    nav, pose = link.latest()         # last [nav] (with t_host) and odometry pose (x, y, theta) or None
    ev = link.events.get_nowait()     # queue.Queue of Event
    link.halt(); link.sit()           # ns, S1

Conventions:
  - v m/s > 0 forward, w rad/s > 0 counter-clockwise (the firmware's nv convention, confirmed on the floor).
    Pose: x forward, y left, theta counter-clockwise, from where odometry was anchored.
  - Base clock: micros(), uint32, wraps every 71.6 min. Host clock: time.monotonic().
  - Threads: the transport calls `feed` from its own thread; `latest`, `send_velocity`... from any thread.

Run log: LoggedTransport writes every line received and sent to base.jsonl; in a dry run it lets only the
read-only commands through.
"""
import asyncio
import json
import math
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field, fields, replace

import numpy as np

from navette.interfaces import BaseError

NUS_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # host -> base (write without response)
NUS_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # base -> host (notify)

# [nav] st
STOPPED, LIFTOFF, BALANCING, STOPPING = 0, 1, 2, 3
# [nav] f bits
F_IMU_OK, F_TILT_TRUSTED, F_MOTORS_ON, F_DEADMAN, F_LOW_BATTERY = 1, 2, 4, 8, 16
F_CTRL_USB, F_CTRL_BLE = 32, 64  # which link is the controller (nav-link-api.md section 4)
F_GZ_OK = 128  # the base has learnt its gyro offset (about 1 s after a still boot): gz is usable for heading
F_READY_TO_STAND = 256  # Stopped, still on its rest, an S0 now passes the tilt check (recent firmware only)
CTRL_BIT = {"usb": F_CTRL_USB, "ble": F_CTRL_BLE}

NH, NS, NZ = "nh", "ns", "nz"  # hello, halt (decel ramp to 0, keeps balancing), zero pl/pr
STAND, SIT = "S0", "S1"        # legacy Commander lines
NAV_RATES = (0, 10, 25, 50, 100)
# balance gains A-D, balance angle O, margin M, tilt filter UF, wheel speed loop pv / pmVP / pmVF / pmVI, release
# pf / pb (HOVERBOARD.md of the firmware)
SETTING = re.compile(r"(A|B|C|D|O|M|UF|pv|pmVP|pmVF|pmVI|pf|pb)-?\d+(\.\d+)?")


# ------------------------------------------------------------------ parsing

@dataclass(frozen=True)
class Nav:
    """One [nav] line (units: nav-link-api.md section 3). wl/wr/pl/pr in the firmware's wheel sign: see Odometry."""
    t: int        # base micros() of the IMU sample, uint32 (wraps)
    seq: int      # lines on this link since the session started (gap = lost lines)
    st: int       # STOPPED / LIFTOFF / BALANCING / STOPPING
    tilt: float   # deg
    rate: float   # rad/s
    roll: float   # deg
    wl: float     # rad/s
    wr: float
    pl: float     # rad, accumulated (also with the motors off, on recent firmware)
    pr: float
    ax: float     # m/s^2
    az: float
    vc: float     # m/s, ramped nv reference as applied
    wc: float     # rad/s
    cs: int       # seq of the last nv accepted since boot (0: none)
    f: int        # F_* bits
    gz: float | None = None      # rad/s yaw rate, > 0 counter-clockwise, offset removed once f & F_GZ_OK;
                                 # None on a firmware without it
    lag: float | None = None     # m, signed (forward > 0): what the base still owes its position goal, i.e. the
                                 # distance it still travels when commands go to 0, before braking; None if absent
    t_recv: float | None = None  # host monotonic when the line arrived (filled by BaseLink)
    t_host: float | None = None  # `t` mapped to host monotonic (None until the first nq round trip)


@dataclass(frozen=True)
class Event:
    t: int        # base micros()
    ev: str       # state, start_refused, tipped, deadman, link_loss_sitdown, ... estop
    st: int       # state when it happened
    details: dict = field(default_factory=dict)  # from/to (int), reason (str), link (str)


@dataclass(frozen=True)
class Sync:
    id: int
    t: int        # base micros() when the nq was routed


@dataclass(frozen=True)
class Hello:
    r: float | None       # wheel radius, m (None on an early firmware)
    track: float | None   # wheel centre to centre, m
    kv: dict              # every key, typed (fw, name, vmax, wmax, deadman_ms, back, front, ...)


_LINE = re.compile(r"^\[(\w+)\]\s*(.*)$")
_KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
_NAV_OPTIONAL = ("gz", "lag")  # keys of recent firmware only
_NAV_FIELDS = [f for f in fields(Nav) if f.name not in _NAV_OPTIONAL + ("t_recv", "t_host")]  # required keys


def _value(s):
    """Quoted -> str without quotes, else int, float (nan too), or str."""
    if len(s) >= 2 and s[0] == s[-1] == '"':
        return s[1:-1]
    for conv in (int, float):
        try:
            return conv(s)
        except ValueError:
            pass
    return s


def parse_line(text):
    """One received line -> Nav | Event | Hello | Sync, or None for anything else ([diag], [poussah], a line
    cut by a link drop, a garbled number)."""
    m = _LINE.match(text.strip())
    if not m:
        return None
    kind, raw = m.group(1), dict(_KV.findall(m.group(2)))
    try:
        if kind == "nav":
            opt = {k: float(raw[k]) for k in _NAV_OPTIONAL if k in raw}
            return Nav(**{f.name: f.type(raw[f.name]) for f in _NAV_FIELDS}, **opt)
        if kind == "event":
            details = {k: _value(v) for k, v in raw.items() if k not in ("t", "ev", "st")}
            return Event(int(raw["t"]), raw["ev"], int(raw["st"]), details)
        if kind == "sync":
            return Sync(int(raw["id"]), int(raw["t"]))
        if kind == "hello":
            kv = {k: _value(v) for k, v in raw.items()}
            kv.update(fw=raw.get("fw"), name=raw.get("name"))  # a git hash of digits stays a str
            if "rates" in raw:
                kv["rates"] = tuple(int(x) for x in raw["rates"].split(","))
            r, track = kv.get("r"), kv.get("track")
            return Hello(float(r) if r is not None else None, float(track) if track is not None else None, kv)
    except (KeyError, ValueError):
        return None
    return None


# ------------------------------------------------------------------ commands

def _dec(x):
    """x in the firmware's nv number syntax: [-]digits[.digits], at most 12 characters, no exponent / '+' /
    nan. Rounded to 4 decimals (0.1 mm/s, 1e-4 rad/s); trailing zeros dropped; -0 written 0."""
    if not math.isfinite(x):
        raise ValueError(f"not finite: {x}")
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    s = "0" if s == "-0" else s
    if len(s) > 12:
        raise ValueError(f"too large for nv: {x}")
    return s


def nv(v, w, seq):
    """Velocity command: v m/s (> 0 forward), w rad/s (> 0 counter-clockwise), seq uint32 (echoed as cs)."""
    if not 0 <= seq < 2**32:
        raise ValueError(f"seq out of uint32: {seq}")
    return f"nv{_dec(v)},{_dec(w)},{seq}"


def nr(hz):
    """[nav] rate on this link; the firmware silently ignores other values, so refuse them here."""
    if hz not in NAV_RATES:
        raise ValueError(f"nav rate must be one of {NAV_RATES}: {hz}")
    return f"nr{hz}"


def nq(i):
    return f"nq{i}"


def nb(ms):
    """BLE connection interval request (the base clamps to 8..30 ms; nb8 asks 7.5 ms)."""
    return f"nb{ms}"


# ------------------------------------------------------------------ clock

class ClockSync:
    """Base micros() (uint32, wraps) -> host time.monotonic(), from nq round trips.

    Each round trip gives (host mid-time, base time), assuming the base answers halfway. Over the last
    `window` trips, the 50 % fastest (least queueing) are fitted by least squares: base - host = a + b (host - x0),
    b being the crystals' rate difference (10-20 ppm: 1 ms per minute, worth fitting). Not thread-safe:
    BaseLink calls it under its lock."""

    def __init__(self, window=30):
        self.samples = deque(maxlen=window)  # (host mid s, base unwrapped s, round trip s)
        self.fit = None                      # (a, b, x0)
        self._last = None                    # last unwrapped base time, us

    def unwrap(self, t_us):
        """uint32 us -> continuous seconds. Each step taken as the signed 32-bit difference from the previous
        value (so slightly out-of-order times, [sync] vs [nav], step back instead of jumping 71 min)."""
        if self._last is None:
            self._last = t_us
        else:
            self._last += (t_us - self._last + 2**31) % 2**32 - 2**31
        return self._last / 1e6

    def add(self, t_send, t_us, t_recv):
        self.samples.append(((t_send + t_recv) / 2, self.unwrap(t_us), t_recv - t_send))
        keep = sorted(self.samples, key=lambda s: s[2])[:max(1, (len(self.samples) + 1) // 2)]
        x = np.array([s[0] for s in keep])
        d = np.array([s[1] for s in keep]) - x
        x0 = x.mean()
        b = np.polyfit(x - x0, d, 1)[0] if np.ptp(x) > 0 else 0.0
        self.fit = (d.mean(), b, x0)  # mean d at mean x: the fit's intercept at x0

    def to_host(self, t_us):
        """Host monotonic time of base time t_us; None before the first round trip. Advances the unwrap."""
        base = self.unwrap(t_us)
        if self.fit is None:
            return None
        a, b, x0 = self.fit
        return (base - a + b * x0) / (1 + b)  # solves base - host = a + b (host - x0) for host


# ------------------------------------------------------------------ odometry

class Odometry:
    """Planar pose (x forward, y left, theta counter-clockwise) from the [nav] wheel angles pl, pr.

    Signs (firmware src/hoverboard): pl, pr (hal_poussah.cpp hal_nav_fill, WheelOdom) integrate each wheel's
    sensor angle times its polarity, "signed like [diag] vel_m1 / vel_m2", and wl = vel_m1, wr = vel_m2
    (field.cpp): so d(pl)/dt has the sign of wl, d(pr)/dt of wr. In that wheel frame field.cpp has
      pos_robot = -(pos_m1 + pos_m2) / 2,  target_m1 = -target - turn_vel,  target_m2 = -target + turn_vel,
    and nv feeds tar_rate = v / r (pos_robot follows it) and turn_vel = -w track / (2 r) (nav_drive.h). Hence
      wl = -v/r + w track/(2r),  wr = -v/r - w track/(2r)
    i.e. a wheel turning forward has a NEGATIVE angle rate, and w > 0 makes wl - wr > 0. Inverting:
      ds     = -r (dpl + dpr) / 2        forward distance of the axle centre
      dtheta =  r (dpl - dpr) / track    counter-clockwise
    Both signs were confirmed on the floor (nv v > 0 drives forward, w > 0 turns counter-clockwise). The
    wheel radius and track come from [hello].

    Heading from the base's gyro, not the wheels, whenever the base gives it (gz, with f & F_GZ_OK): the
    wheels are conical, their rolling radii differ by ~4 % with a fraction of a degree of roll, which made
    ~30 deg of false heading per 2 m driven straight (firmware measurement); on the robot the wheel heading
    drifted 262 deg in a 224 s exploration. gz is integrated over the base clock t (micros, wraps at 2^32),
    trapezoidal between two lines. The wheel difference is only the fallback (older firmware, or before the
    base has learnt its gyro offset ~1 s after a still boot); distance always comes from the wheels."""

    def __init__(self, r, track):
        self.r, self.track = r, track
        self.pose = (0.0, 0.0, 0.0)
        self._last = None  # (pl, pr, t, gz or None) of the previous update; None: next update only anchors

    def reset(self):
        """Pose back to 0; the next update re-anchors (after a reconnection, a base reboot or an nz)."""
        self.pose, self._last = (0.0, 0.0, 0.0), None

    def update(self, pl, pr, t=None, gz=None):
        """pl, pr wheel angles (rad); t base micros and gz yaw rate (rad/s), gz None unless the base's offset is
        known (f & F_GZ_OK)."""
        if self._last is not None:
            pl0, pr0, t0, gz0 = self._last
            dl, dr = pl - pl0, pr - pr0
            ds = -self.r * (dl + dr) / 2
            if gz is not None and gz0 is not None and t is not None and t0 is not None:
                dt = ((t - t0) % 2**32) * 1e-6
                dth = (gz + gz0) / 2 * dt
            else:
                dth = self.r * (dl - dr) / self.track
            x, y, th = self.pose
            # exact for a constant-curvature step: chord ds * sinc(dth/2) along the mid heading
            chord = ds * np.sinc(dth / (2 * np.pi))
            mid = th + dth / 2
            th = math.remainder(th + dth, 2 * math.pi)
            self.pose = (x + chord * math.cos(mid), y + chord * math.sin(mid), th)
        self._last = (pl, pr, t, gz)
        return self.pose


# ------------------------------------------------------------------ link

class LinkError(BaseError):
    """The transport reported a failure (disconnection, write error): commands no longer reach the base."""


class BaseLink:
    """The base, over any transport with open(), write_line(str), close(), an `error` attribute (None while
    healthy) and an `on_line(t_recv, text)` attribute it calls for each received line (set here to `feed`).

    Halt is latched: after halt() (and at start), send_velocity() sends nothing until resume(), so a STOP from
    another thread (phone page, antenna) cannot be undone by the nav loop's next command."""

    def __init__(self, transport, rate=50, link_name=None):
        """link_name: 'ble' or 'usb', the link this transport is, for the controller checks (the base
        takes drive commands from one link, the first to send nv; [nav] f says which, [event] owner link=...).
        None: unknown (tests), the checks are skipped."""
        self.transport = transport
        self.rate = rate
        self.link_name = link_name
        transport.on_line = self.feed
        self.events = queue.Queue()        # Event, in arrival order
        self.cs_latencies = deque(maxlen=100)  # s, nv sent -> first [nav] echoing its seq in cs
        self.malformed = 0                 # [nav]/[event]/... lines that did not parse
        self.hello = None
        self._lock = threading.Lock()      # state below, touched by the transport thread
        self._tx = threading.Lock()        # numbering + write order of the commands
        self._clock = ClockSync()
        self._odom = None
        self._nav = None
        self._seq = None                   # last nv seq sent
        self._sent = {}                    # nv seq -> host send time, not echoed yet
        self._nq = 0
        self._nq_sent = {}                 # nq id -> host send time
        self._halted = True                # latched stop: send_velocity() sends nothing until resume()

    def connect(self):
        """Open the transport, then nh (the base says nothing to a link before it hears from it) and nr.
        Starts halted: resume() before driving."""
        with self._tx, self._lock:
            self._clock, self._nav, self._seq, self._sent, self._halted = ClockSync(), None, None, {}, True
            self.hello = None  # this session's [hello] (hello_ok must not pass on a previous session's)
            if self._odom:
                self._odom.reset()
        self.transport.open()
        self._write(NH)
        self._write(NH)  # the first line after a USB open can be lost; a 2nd [hello] is harmless
        self._write(nr(self.rate))
        self.sync()

    def hello_ok(self):
        """(ok, reason): the base speaks the nav link protocol 1 and is calibrated (setup=7: its balance
        calibration), per [hello]."""
        if self.hello is None:
            return False, "no [hello] from the base"
        proto, setup = self.hello.kv.get("proto"), self.hello.kv.get("setup")
        if proto != 1:
            return False, f"base protocol {proto} (need 1)"
        if setup != 7:
            return False, f"base not calibrated (setup={setup}, need 7)"
        return True, ""

    def is_controller(self, nav=None):
        """True if the last [nav] (or `nav`) says this link is the base's controller; None if link_name unknown."""
        if self.link_name not in CTRL_BIT:
            return None
        nav = nav or self.latest()[0]
        return nav is not None and bool(nav.f & CTRL_BIT[self.link_name])

    def close(self):
        self.transport.close()

    # --- commands
    def _check(self):
        if getattr(self.transport, "error", None):
            raise LinkError(self.transport.error)

    def _write(self, line):
        with self._tx:
            self._check()
            self.transport.write_line(line)

    def sync(self):
        """One clock-sync round trip (nq); call about once a second."""
        with self._tx:
            self._nq += 1
            with self._lock:
                now = time.monotonic()
                self._nq_sent = {i: t for i, t in self._nq_sent.items() if now - t < 5.0}  # unanswered
                self._nq_sent[self._nq] = now
            self.transport.write_line(nq(self._nq))

    def send_velocity(self, v, w):
        """nv with the next seq; returns the seq, or None (nothing sent) while halted or before the first [nav]
        of the session. The first seq follows that [nav]'s cs (the base keeps cs across sessions), so an old
        cs is never taken for an echo of ours. Raises LinkError if the transport failed."""
        with self._tx:
            self._check()
            with self._lock:
                if self._halted or self._nav is None:
                    return None
                if self._seq is None:
                    self._seq = self._nav.cs
                self._seq = self._seq + 1 if self._seq + 1 < 2**32 else 1
                seq = self._seq
                line = nv(v, w, seq)  # NaN / too large: raises, nothing sent (that seq is skipped)
                self._sent[seq] = time.monotonic()
                while len(self._sent) > 50:  # never echoed (link down): drop the oldest
                    del self._sent[next(iter(self._sent))]
            self.transport.write_line(line)
        return seq

    def halt(self):
        """ns (controlled stop, keeps balancing), latched: no nv goes out until resume(). The latch is set
        before the write, so it holds even if the link has failed (then LinkError is raised)."""
        with self._tx:
            with self._lock:
                self._halted = True
            self._check()
            self.transport.write_line(NS)

    def resume(self):
        """Allow send_velocity() again after halt() / connect()."""
        with self._tx, self._lock:
            self._halted = False

    def setting(self, line):
        """One balance / wheel-loop setting line (Commander syntax, RAM only, accepted while Stopped), e.g. "A40",
        "O2.4", "pv2.5". Only these settings can be sent this way: nothing that moves or stops the base."""
        if not SETTING.fullmatch(line):
            raise ValueError(f"not a balance setting line: {line!r}")
        self._write(line)

    def stand(self):
        self._write(STAND)

    def sit(self):
        self._write(SIT)

    # --- state
    def latest(self):
        """(last Nav or None, pose (x, y, theta) or None before a [hello] gave r and track)."""
        with self._lock:
            return self._nav, (self._odom.pose if self._odom and self._odom._last else None)

    def reset_pose(self):
        """Pose origin at the current position (client side only: no nz sent)."""
        with self._lock:
            if self._odom:
                self._odom.reset()

    def feed(self, t_recv, text):
        """One received line (transport thread)."""
        msg = parse_line(text)
        if msg is None:
            if text.lstrip().startswith(("[nav]", "[event]", "[sync]", "[hello]")):
                self.malformed += 1
            return
        with self._lock:
            if isinstance(msg, Nav):
                msg = replace(msg, t_recv=t_recv, t_host=self._clock.to_host(msg.t))
                if self._odom:
                    self._odom.update(msg.pl, msg.pr, msg.t, msg.gz if msg.f & F_GZ_OK else None)
                if msg.cs in self._sent:  # exact match: the base applies nv in order, older ones are done
                    self.cs_latencies.append(t_recv - self._sent[msg.cs])
                    self._sent = {s: t for s, t in self._sent.items() if s > msg.cs}
                self._nav = msg
            elif isinstance(msg, Sync):
                if msg.id in self._nq_sent:
                    self._clock.add(self._nq_sent.pop(msg.id), msg.t, t_recv)
            elif isinstance(msg, Hello):
                self.hello = msg
                if msg.r and msg.track:
                    if self._odom is None:
                        self._odom = Odometry(msg.r, msg.track)
                    self._odom.r, self._odom.track = msg.r, msg.track
            else:
                self.events.put(msg)


# ------------------------------------------------------------------ logging and dry run

DRY_OK = ("nh", "nr", "nq", "nb")  # the only lines a dry run sends: read-only (hello, rate, clock, interval)


class LoggedTransport:
    """Wraps a transport: every received line and every line written go to base.jsonl with the host
    time.monotonic() (received: the transport's receive stamp):
        {"t": ..., "rx": line}    {"t": ..., "tx": line}    {"t": ..., "tx": line, "dry": true}
    dry_run: only DRY_OK lines are passed on; the others are logged with "dry": true and dropped, so nothing
    can move the base whatever calls it."""

    def __init__(self, inner, path, dry_run):
        self.inner, self.dry_run = inner, dry_run
        self.on_line = None  # set by BaseLink
        self.f = open(path, "w", buffering=1)
        self._lock = threading.Lock()  # rx (transport thread) and tx (any thread) share the file
        inner.on_line = self._rx

    @property
    def error(self):
        return self.inner.error

    def _write(self, rec):
        with self._lock:
            if not self.f.closed:
                self.f.write(json.dumps(rec) + "\n")

    def _rx(self, t, text):
        self._write({"t": t, "rx": text})
        self.on_line(t, text)

    def write_line(self, line):
        ok = not self.dry_run or line.startswith(DRY_OK)
        self._write({"t": time.monotonic(), "tx": line, **({} if ok else {"dry": True})})
        if ok:
            self.inner.write_line(line)

    def open(self):
        self.inner.open()

    def close(self):
        try:
            self.inner.close()
        finally:
            with self._lock:
                self.f.close()


class NoTransport:
    """Transport of a BaseLink that only replays recorded lines (nothing is ever written)."""
    on_line, error = None, None

    def open(self):
        pass

    def write_line(self, line):
        pass

    def close(self):
        pass


# ------------------------------------------------------------------ transports (not testable here)

def _split_lines(buf, chunk, t, on_line):
    """Line reassembly: returns the unfinished tail; calls on_line(t, text) per complete line."""
    *lines, buf = (buf + chunk).split(b"\n")
    for ln in lines:
        on_line(t, ln.decode("utf8", "replace").rstrip("\r"))
    return buf


class SerialTransport:
    """USB serial (fallback). Opened WITHOUT resetting the base: the ESP32-S3 USB-Serial/JTAG resets the chip
    while DTR=0 and RTS=1; Linux raises both on open and pyserial applies DTR before RTS, so dtr = rts = False
    passes through the reset state. Only rts = False before open(), DTR left up."""

    def __init__(self, port, baudrate=115200):
        self.port, self.baudrate = port, baudrate
        self.on_line = None
        self.error = None  # set if the reader thread dies (unplug)

    def open(self):
        import serial

        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = self.port, self.baudrate, 0.05
        self.ser.rts = False
        self.ser.open()
        self._stop = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        buf = b""
        try:
            while not self._stop.is_set():
                # block for the first byte (50 ms timeout), then drain: timestamp when data arrives
                chunk = self.ser.read(max(1, self.ser.in_waiting))
                buf = _split_lines(buf, chunk, time.monotonic(), self.on_line)
        except Exception as e:  # closed during a read, unplugged
            if not self._stop.is_set():
                self.error = repr(e)

    def write_line(self, line):
        self.ser.write((line + "\n").encode())

    def close(self):
        self._stop.set()
        self.ser.close()


class BleTransport:
    """Bluetooth LE, Nordic UART service (bleak, imported here only). An asyncio loop runs in a background
    thread; write_line() schedules the write and returns at once (writes keep their order). Connects to the
    strongest device whose name starts with `name`, then asks the connection interval `interval_ms` (nb):
    7.5 ms (nb8) carried 100 Hz [nav] on the bench. The base takes one BLE client at a time; losing it
    halts the base's nv drive and sits it down 1 s later."""

    def __init__(self, name="rmini_wheels", interval_ms=8, scan_s=5.0):
        self.name, self.interval_ms, self.scan_s = name, interval_ms, scan_s
        self.on_line = None
        self.error = None  # set on a disconnection
        self._buf = b""

    def open(self):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._connect(), self._loop).result(self.scan_s + 30)
        self.write_line(nb(self.interval_ms))

    async def _connect(self):
        from bleak import BleakClient, BleakScanner

        found = await BleakScanner.discover(timeout=self.scan_s, return_adv=True)
        bases = [(adv.rssi, d) for d, adv in found.values() if (adv.local_name or d.name or "").startswith(self.name)]
        if not bases:
            raise ConnectionError(f"no BLE device named {self.name}*")
        self._lock = asyncio.Lock()
        self.client = BleakClient(max(bases, key=lambda b: b[0])[1], disconnected_callback=self._disconnected)
        await self.client.connect()
        # write without response is limited to MTU - 3 bytes (20 before an MTU exchange); the base
        # reassembles lines from the byte stream, so a line may span several writes
        self._chunk = max(20, self.client.mtu_size - 3)
        await self.client.start_notify(NUS_TX, self._notify)

    def _disconnected(self, _client):
        self.error = self.error or "Bluetooth disconnected"

    def _notify(self, _char, data):
        self._buf = _split_lines(self._buf, bytes(data), time.monotonic(), self.on_line)

    async def _write(self, data):
        async with self._lock:  # one line's chunks never interleave with another's
            for i in range(0, len(data), self._chunk):
                await self.client.write_gatt_char(NUS_RX, data[i:i + self._chunk], response=False)

    def write_line(self, line):
        fut = asyncio.run_coroutine_threadsafe(self._write((line + "\n").encode()), self._loop)
        fut.add_done_callback(self._written)

    def _written(self, fut):
        if not fut.cancelled() and fut.exception() is not None:
            self.error = self.error or f"write: {fut.exception()!r}"

    def close(self):
        self.error = self.error or "closed"
        asyncio.run_coroutine_threadsafe(self.client.disconnect(), self._loop).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
