"""The Poussah base's wire protocol client (navette.robots.reachy_poussah.poussah_link): parsing, command
syntax, clock sync, wheel odometry, the link's latch and numbering. No hardware: a fake transport."""
import math
import re

import pytest

from navette.robots.reachy_poussah import poussah_link as bl

R, TRACK = 0.0444, 0.16


def nav_line(t=123456, seq=7, st=2, pl=0.0, pr=0.0, cs=0, f=7, wl=0.0, wr=0.0):
    """As the firmware prints it (same printf formats)."""
    return (f"[nav] t={t} seq={seq} st={st} tilt={-1.234:.3f} rate={0.0123:.4f} roll={0.5:.2f} wl={wl:.3f} "
            f"wr={wr:.3f} pl={pl:.4f} pr={pr:.4f} ax={0.12:.3f} az={9.81:.3f} vc={0.1:.3f} wc={0.0:.3f} "
            f"cs={cs} f={f}")


HELLO = ("[hello] proto=1 fw=cfb9d6c id=6 name=rmini_wheels_6 setup=6 rates=10,25,50,100 tilt_lag_ms=25 "
         "back=nan front=nan lpf_tar=0.100 lpf_turn=0.060 "
         f"r={R:.4f} track={TRACK:.3f} vmax=0.40 wmax=1.00 acc=0.50 dec=0.50 deadman_ms=500")


# ------------------------------------------------------------------ parsing

def test_parse_nav():
    n = bl.parse_line(nav_line(t=4294967295, pl=12.3457, pr=-0.0001, cs=42) + "\r\n")
    assert isinstance(n, bl.Nav)
    assert (n.t, n.seq, n.st, n.cs, n.f) == (4294967295, 7, 2, 42, 7)
    assert all(isinstance(x, int) for x in (n.t, n.seq, n.st, n.cs, n.f))
    assert n.tilt == -1.234 and n.pl == 12.3457 and n.pr == -0.0001 and n.az == 9.81
    assert n.t_recv is None and n.t_host is None


def test_parse_event_sync_hello():
    e = bl.parse_line("[event] t=5 ev=state st=1 from=0 to=1")
    assert e == bl.Event(5, "state", 1, {"from": 0, "to": 1})
    e = bl.parse_line('[event] t=5 ev=start_refused st=0 reason="base not set up (setup=6)"')
    assert e.details == {"reason": "base not set up (setup=6)"}
    assert bl.parse_line("[event] t=9 ev=link_loss_sitdown st=2 link=ble").details == {"link": "ble"}
    assert bl.parse_line("[event] t=9 ev=estop st=2") == bl.Event(9, "estop", 2, {})
    assert bl.parse_line("[sync] id=3 t=4000000000") == bl.Sync(3, 4000000000)
    h = bl.parse_line(HELLO)
    assert (h.r, h.track) == (R, TRACK)
    assert h.kv["rates"] == (10, 25, 50, 100) and h.kv["deadman_ms"] == 500 and h.kv["vmax"] == 0.4
    assert math.isnan(h.kv["back"]) and h.kv["name"] == "rmini_wheels_6" and h.kv["fw"] == "cfb9d6c"
    assert bl.parse_line("[hello] proto=1 fw=1234567 name=x").kv["fw"] == "1234567"  # stays a str
    assert bl.parse_line("[hello] proto=1 fw=abc name=x").r is None  # early firmware: no r, track


@pytest.mark.parametrize("line", [
    "[diag] vel_m1=1.0 vel_m2=2.0", "[poussah] start refused: x", "[field] whatever", "", "garbage",
    "[nav] t=1 seq=", "[nav] t=12 seq=3 st=2 tilt=1.0",  # cut by a link drop
    nav_line().replace("seq=7", "seq=7.5"),               # garbled
])
def test_parse_ignores(line):
    assert bl.parse_line(line) is None


# ------------------------------------------------------------------ commands

NUM = re.compile(r"-?\d+(\.\d+)?")  # the firmware's grammar (nav_parse_dec)


@pytest.mark.parametrize("v, w, expect", [
    (0.1, 0.0, "nv0.1,0,5"),
    (-0.25, 1.5, "nv-0.25,1.5,5"),
    (1e-7, -1e-7, "nv0,0,5"),          # no exponent, no -0
    (-0.00004, 0.00005, "nv0,0.0001,5"),
    (0.123449, -0.99996, "nv0.1234,-1,5"),
    (10.0, 0.4, "nv10,0.4,5"),
])
def test_nv_format(v, w, expect):
    cmd = bl.nv(v, w, 5)
    assert cmd == expect
    a, b, _ = cmd[2:].split(",")
    assert NUM.fullmatch(a) and NUM.fullmatch(b) and len(a) <= 12 and len(b) <= 12


def test_nv_limits():
    assert bl.nv(-999999.9999, 0, 2**32 - 1) == f"nv-999999.9999,0,{2**32 - 1}"  # 12 characters
    for bad in (float("nan"), float("inf"), -9999999.9999, 12345678.12345, 1e12):  # 13+ characters
        with pytest.raises(ValueError):
            bl.nv(bad, 0, 1)
    with pytest.raises(ValueError):
        bl.nv(0, 0, 2**32)


def test_other_commands():
    assert (bl.nr(50), bl.nq(12), bl.nb(8), bl.NS, bl.NH, bl.NZ, bl.STAND, bl.SIT) == \
           ("nr50", "nq12", "nb8", "ns", "nh", "nz", "S0", "S1")
    with pytest.raises(ValueError):
        bl.nr(20)


# ------------------------------------------------------------------ clock

def test_clock_unwrap():
    c = bl.ClockSync()
    assert c.unwrap(2**32 - 1_000_000) == pytest.approx((2**32 - 1_000_000) / 1e6)
    assert c.unwrap(500_000) == pytest.approx((2**32 + 500_000) / 1e6)       # wrapped: +1.5 s
    assert c.unwrap(2**32 - 100_000) == pytest.approx((2**32 - 100_000) / 1e6)  # slightly older: back


def test_clock_sync_across_wrap():
    """Base clock = host + 1000 s, 20 ppm fast, wrapping during the run; noisy round trips."""
    c = bl.ClockSync()
    base_of = lambda h: (h + 1000.0) * (1 + 20e-6)
    for k in range(40):
        ts = 4294.0 - 1000.0 - 20 + k   # host time; the base wraps (4294.97 s) around k = 20
        rtt = 0.010 + 0.030 * (k % 3 == 0)  # every third trip delayed
        t_us = int(base_of(ts + rtt / 2) * 1e6) % 2**32
        c.add(ts, t_us, ts + rtt)
    h = 4294.0 - 1000.0 + 25.0
    assert c.to_host(int(base_of(h) * 1e6) % 2**32) == pytest.approx(h, abs=2e-4)
    assert c.fit[1] == pytest.approx(20e-6, rel=0.05)


def test_clock_none_before_sync():
    assert bl.ClockSync().to_host(123) is None


# ------------------------------------------------------------------ odometry

def wheel_rates(v, w):
    """Wheel angle rates (rad/s) for (v, w), in the firmware's wheel sign (poussah_link.Odometry)."""
    return -v / R + w * TRACK / (2 * R), -v / R - w * TRACK / (2 * R)


def drive(odom, v, w, t, n=50):
    """Integrate the wheel angles for (v, w) during t s in n [nav] steps; returns the pose."""
    wl, wr = wheel_rates(v, w)
    pl, pr = odom._last
    for i in range(1, n + 1):
        odom.update(pl + wl * t * i / n, pr + wr * t * i / n)
    return odom.pose


def anchored():
    o = bl.Odometry(R, TRACK)
    o.update(3.0, -2.0)  # arbitrary start angles: only deltas count
    return o


def test_odom_straight():
    assert drive(anchored(), 0.2, 0.0, 5.0) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert drive(anchored(), -0.1, 0.0, 2.0) == pytest.approx((-0.2, 0.0, 0.0), abs=1e-9)


def test_odom_turn_in_place():
    x, y, th = drive(anchored(), 0.0, 0.5, 2.0)
    assert (x, y, th) == pytest.approx((0, 0, 1.0), abs=1e-9)  # counter-clockwise
    assert drive(anchored(), 0.0, -0.5, 2.0)[2] == pytest.approx(-1.0)


def test_odom_arc():
    """Quarter circle of radius 0.5 m to the left: ends at (0.5, 0.5) heading +90 deg."""
    v, w = 0.2, 0.4
    x, y, th = drive(anchored(), v, w, (math.pi / 2) / w, n=7)  # few, large steps: exact per step
    assert (x, y, th) == pytest.approx((0.5, 0.5, math.pi / 2), abs=1e-9)


def test_odom_wraps_and_resets():
    o = anchored()
    assert drive(o, 0.0, 1.0, 4.0)[2] == pytest.approx(4.0 - 2 * math.pi)
    o.reset()
    o.update(100.0, 100.0)  # re-anchor: no jump
    assert o.pose == (0.0, 0.0, 0.0)


# ------------------------------------------------------------------ BaseLink

class FakeTransport:
    def __init__(self):
        self.on_line, self.sent, self.opened, self.error = None, [], False, None

    def open(self):
        self.opened = True

    def write_line(self, line):
        self.sent.append(line)

    def close(self):
        pass


def make_link(resume=True):
    tr = FakeTransport()
    link = bl.BaseLink(tr, rate=50)
    link.connect()
    if resume:
        link.resume()
    return link, tr


def test_connect_handshake():
    link, tr = make_link()
    assert tr.opened and tr.sent == ["nh", "nh", "nr50", "nq1"]
    link.halt(), link.stand(), link.sit()
    assert tr.sent[4:] == ["ns", "S0", "S1"]


def test_hello_ok():
    """Drive only a base speaking protocol 1 and calibrated (setup=7) (nav-link-api.md section 5)."""
    link, _ = make_link()
    assert link.hello_ok() == (False, "no [hello] from the base")
    link.feed(1.0, HELLO)  # setup=6
    ok, why = link.hello_ok()
    assert not ok and "setup=6" in why
    link.feed(1.0, HELLO.replace("setup=6", "setup=7"))
    assert link.hello_ok() == (True, "")
    link.feed(1.0, HELLO.replace("setup=6", "setup=7").replace("proto=1", "proto=2"))
    ok, why = link.hello_ok()
    assert not ok and "protocol 2" in why
    link.feed(1.0, HELLO.replace("setup=6 ", ""))  # no setup key at all
    assert not link.hello_ok()[0]


def test_is_controller():
    """[nav] f bit 32 (USB) / 64 (BLE) says which link is the base's controller."""
    link, _ = make_link()
    assert link.is_controller() is None  # link_name unknown: check skipped
    link.feed(1.0, nav_line(f=7 | bl.F_CTRL_BLE))
    assert link.is_controller() is None
    ble = bl.BaseLink(FakeTransport(), link_name="ble")
    assert ble.is_controller() is False  # no [nav] yet
    ble.feed(1.0, nav_line(f=7))
    assert ble.is_controller() is False  # no controller
    ble.feed(1.0, nav_line(f=7 | bl.F_CTRL_USB))
    assert ble.is_controller() is False  # the other link
    ble.feed(1.0, nav_line(f=7 | bl.F_CTRL_BLE))
    assert ble.is_controller() is True
    assert ble.is_controller(bl.parse_line(nav_line(f=7))) is False  # an explicit [nav] wins over the latest
    usb = bl.BaseLink(FakeTransport(), link_name="usb")
    assert usb.is_controller(bl.parse_line(nav_line(f=7 | bl.F_CTRL_USB))) is True
    assert usb.is_controller(bl.parse_line(nav_line(f=7 | bl.F_CTRL_BLE))) is False


def test_seq_follows_cs_and_latency():
    link, tr = make_link()
    link.feed(1.00, nav_line(seq=1, cs=41))  # cs from an earlier session
    assert [link.send_velocity(0.1, 0) for _ in range(3)] == [42, 43, 44]
    assert tr.sent[-3:] == ["nv0.1,0,42", "nv0.1,0,43", "nv0.1,0,44"]
    t42, t44 = link._sent[42], link._sent[44]
    link.feed(t42 + 0.03, nav_line(seq=2, cs=42))
    link.feed(t42 + 0.04, nav_line(seq=3, cs=42))  # same cs: counted once
    link.feed(t44 + 0.05, nav_line(seq=4, cs=44))  # 43 skipped over: dropped, not counted
    assert list(link.cs_latencies) == pytest.approx([0.03, 0.05])
    assert link._sent == {}


def test_no_nv_before_the_first_nav_and_bad_value():
    link, tr = make_link()
    assert link.send_velocity(0.1, 0) is None and tr.sent == ["nh", "nh", "nr50", "nq1"]  # cs of the session unknown
    link.feed(1.0, nav_line(cs=3))          # small old cs (earlier session): numbering follows it
    assert link.send_velocity(0, 0) == 4
    with pytest.raises(ValueError):
        link.send_velocity(float("nan"), 0)
    assert tr.sent[-1] == "nv0,0,4"  # nothing sent for the NaN


def test_old_echo_after_numbering_is_not_latency():
    link, _ = make_link()
    link.feed(1.0, nav_line(cs=3))
    link.send_velocity(0.1, 0)              # seq 4
    link.feed(1.1, nav_line(cs=3))          # still the old cs
    assert len(link.cs_latencies) == 0 and 4 in link._sent


def test_halt_is_latched_until_resume():
    link, tr = make_link(resume=False)
    link.feed(1.0, nav_line(cs=0))
    assert link.send_velocity(0.2, 0) is None          # starts halted
    link.resume()
    assert link.send_velocity(0.2, 0) == 1
    link.halt()
    assert link.send_velocity(0.2, 0) is None and tr.sent[-1] == "ns"  # a STOP is not undone by the next nv
    link.resume()
    assert link.send_velocity(0.2, 0) == 2


def test_link_error_raises_and_still_latches_the_halt():
    link, tr = make_link()
    link.feed(1.0, nav_line(cs=0))
    tr.error = "Bluetooth disconnected"
    with pytest.raises(bl.LinkError):
        link.send_velocity(0.2, 0)
    with pytest.raises(bl.LinkError):
        link.halt()
    tr.error = None
    assert link.send_velocity(0.2, 0) is None  # the halt latched even though its write failed


def test_reconnect_clears_session_state():
    link, _ = make_link()
    link.feed(1.0, nav_line(cs=0))
    link.send_velocity(0.1, 0)
    link.connect()
    assert link._sent == {} and link._nav is None and link.send_velocity(0.1, 0) is None


def test_events_and_ignored_lines():
    link, _ = make_link()
    link.feed(1.0, "[diag] a=1")
    link.feed(1.0, "[event] t=5 ev=tipped st=2")
    link.feed(1.1, "[event] t=6 ev=state st=2 from=2 to=0")
    link.feed(1.2, "[nav] t=1 seq=")
    assert link.events.get_nowait().ev == "tipped"
    assert link.events.get_nowait().details == {"from": 2, "to": 0}
    assert link.events.empty() and link.malformed == 1


def test_pose_and_clock_through_link():
    link, tr = make_link()
    link.feed(1.0, nav_line(pl=5.0, pr=5.0))
    assert link.latest()[1] is None  # no [hello] yet: r, track unknown
    link.feed(1.0, HELLO)
    wl, wr = wheel_rates(0.2, 0.0)
    link.feed(1.02, nav_line(pl=5.0, pr=5.0))               # anchors
    link.feed(1.04, nav_line(pl=5.0 + wl, pr=5.0 + wr))     # 0.2 m forward (1 s of wheel turn)
    nav, pose = link.latest()
    assert pose == pytest.approx((0.2, 0.0, 0.0), abs=1e-3)  # pl/pr printed with 4 decimals
    assert nav.t_recv == 1.04 and nav.t_host is None
    # clock: base = host + 100 s; one round trip of nq1
    t_send = link._nq_sent[1]
    link.feed(t_send + 0.01, f"[sync] id=1 t={int((t_send + 0.005 + 100) * 1e6) % 2**32}")
    link.feed(t_send + 0.02, nav_line(t=int((t_send + 0.015 + 100) * 1e6) % 2**32))
    assert link.latest()[0].t_host == pytest.approx(t_send + 0.015, abs=2e-6)
    link.reset_pose()
    link.feed(1.1, nav_line(pl=7.0, pr=7.0))
    assert link.latest()[1] == (0.0, 0.0, 0.0)
