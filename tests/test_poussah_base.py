"""The generic Controller on the real Poussah driver (PoussahBase on a BaseLink over a fake transport, fake
clock): what reaches the base's wire for every START / STOP / watchdog case, and the dry-run transport."""
import json
import threading
import time

import pytest

from navette import interfaces as itf
from navette.robots.reachy_poussah import poussah_base as pb
from navette.robots.reachy_poussah import poussah_link as bl
from navette.runtime import controller as lr
from navette.runtime.loop import BaseMotion, NavOut

R, TRACK = 0.0444, 0.16
HELLO = f"[hello] proto=1 setup=7 fw=abc name=rmini_wheels_6 r={R} track={TRACK} vmax=0.40 wmax=1.00 deadman_ms=500"
START_TIMEOUT_S = 10.0


def nav_line(st=2, vc=0.0, wc=0.0, cs=0, seq=1, wl=0.0, wr=0.0, f=7, pl=0.0, pr=0.0):
    return (f"[nav] t=1000 seq={seq} st={st} tilt=0.0 rate=0.0 roll=0.0 wl={wl} wr={wr} pl={pl} pr={pr} ax=0.0 "
            f"az=9.81 vc={vc} wc={wc} cs={cs} f={f}")


def out(v=0.1, w=0.0, finished=False, state="DRIVE"):
    return NavOut(v, w, v, w, state, finished)


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeTransport:
    def __init__(self):
        self.on_line, self.sent, self.error = None, [], None

    def open(self):
        pass

    def write_line(self, line):
        self.sent.append(line)

    def close(self):
        pass


class Rig:
    """A Controller on a PoussahBase on a real BaseLink over a fake transport, with a fake clock."""

    def __init__(self, st=2, dry_run=False, v_max=0.15, w_max=1.0, stale_s=0.3, base=True, tmp_path=None,
                 hello=HELLO, link_name=None, f=7):
        """hello: the [hello] line fed (None: none); link_name: BaseLink's ('ble' / 'usb' enable the controller
        checks); f: the [nav] f bits of every line fed."""
        self.clock, self.cmds, self.logs, self.seq, self.f = Clock(), [], [], 0, f
        self.tr = FakeTransport()
        self.link = self.base = None
        if base:
            tr = bl.LoggedTransport(self.tr, str(tmp_path / "base.jsonl"), dry_run) if tmp_path else self.tr
            self.link = bl.BaseLink(tr, link_name=link_name)
            self.link.connect()
            if hello is not None:
                self.feed(hello)
            self.nav(st)
            self.base = pb.PoussahBase(self.link, dry_run, clock=self.clock, log=self.logs.append)
        self.ctrl = lr.Controller(self.base, v_max, w_max, dry_run, stale_s=stale_s, clock=self.clock,
                                  start_timeout_s=START_TIMEOUT_S, on_command=lambda *a: self.cmds.append(a),
                                  log=self.logs.append)

    def feed(self, line):
        self.tr.on_line(self.clock.t, line)  # as the transport thread: through LoggedTransport if wrapped

    def nav(self, st=2, vc=0.0, wc=0.0, wl=0.0, wr=0.0):
        self.seq += 1
        self.feed(nav_line(st, vc, wc, seq=self.seq, wl=wl, wr=wr, f=self.f))

    def step(self, dt=0.05, st=2, vc=0.0, wc=0.0, nav_out=None, wl=0.0, wr=0.0):
        """dt later: a fresh [nav] (unless st is None), the perception output (if any), one tick."""
        self.clock.t += dt
        if st is not None and self.link is not None:
            self.nav(st, vc, wc, wl, wr)
        if nav_out is not None:
            self.ctrl.set_nav(nav_out, self.clock.t, self.ctrl.run)
        self.ctrl.tick()

    def still(self, st=2, n=pb.STILL_LINES):
        """n [nav] lines with still wheels (what the park waits for)."""
        for _ in range(n):
            self.step(st=st)

    def settle(self):
        """Balancing with still wheels until the stand-up's settle roll is over (pb.SETTLE_MIN_S)."""
        for _ in range(int(pb.SETTLE_MIN_S / 0.05) + 2):
            self.step(st=2)

    def drive(self):
        """START from Balancing and one nv: DRIVING."""
        assert self.ctrl.start()[0]
        self.step(nav_out=out())
        self.step(nav_out=out())
        assert self.ctrl.state == lr.DRIVING

    def sent(self):
        return [s for s in self.tr.sent if s not in ("nh", "nr50") and not s.startswith("nq")]


def test_page_dead_man_halts_when_the_phone_page_goes_silent():
    """The robot's WiFi once dropped while it explored; the phone could not STOP. With page_timeout_s, a run
    halts once the page's status polls stop; polls keep it going; IDLE is not affected."""
    r = Rig()
    r.ctrl.page_timeout_s = 2.0
    r.drive()
    for _ in range(60):  # 3 s of driving with the page polling every 0.5 s
        r.step(nav_out=out())
        if round(r.clock.t * 20) % 10 == 0:
            r.ctrl.page_alive()
    assert r.ctrl.state == lr.DRIVING
    for _ in range(45):  # page silent for 2.25 s
        r.step(nav_out=out())
    assert r.ctrl.state == lr.IDLE and "phone page" in r.ctrl.reason and r.sent()[-1] == "ns"
    for _ in range(100):  # idle: no further halts, nothing sent
        r.step()
    assert r.sent()[-1] == "ns"


def test_nothing_sent_before_start():
    r = Rig()
    for _ in range(5):
        r.step(nav_out=out())
    assert r.ctrl.state == lr.IDLE and r.sent() == []


def test_start_from_sitting_stands_up_then_drives_capped():
    r = Rig(st=0)
    assert r.ctrl.start() == (True, "starting")
    assert r.sent() == ["nv0,0,1", "S0"] and r.ctrl.state == lr.STARTING  # claim control, then stand up
    r.feed("[event] t=1 ev=state st=1 from=0 to=1")  # the stand-up's own state changes: not a watchdog
    r.step(st=1)
    r.feed("[event] t=2 ev=state st=2 from=1 to=2")
    r.settle()
    assert r.ctrl.state == lr.DRIVING
    r.step(nav_out=out(0.3, 0.6))
    assert r.sent()[-1].startswith("nv0.15,0.3,") and r.ctrl.sent == pytest.approx((0.15, 0.3))


def test_after_a_stand_up_driving_waits_for_the_settle_roll():
    """With the Mini on, the base rolls forward 11-15 cm by itself during its first ~2 s of balancing: only
    zero commands until it has balanced SETTLE_MIN_S with still wheels, or SETTLE_MAX_S if they never are."""
    r = Rig(st=0)
    r.ctrl.start()
    r.step(st=2)
    n = 0
    while r.ctrl.state == lr.STARTING:  # balancing, still wheels
        r.step(st=2, nav_out=out())
        n += 1
    assert n * 0.05 == pytest.approx(pb.SETTLE_MIN_S, abs=0.06) and r.ctrl.state == lr.DRIVING
    assert all(s.startswith("nv0,0,") for s in r.sent()[2:-1])  # zeros while settling
    m = Rig(st=0)
    m.ctrl.start()
    m.step(st=2)
    n = 0
    while m.ctrl.state == lr.STARTING:  # rolling all along (wheels at 2 rad/s): the time limit decides
        m.step(st=2, wl=-2.0, wr=-2.0, nav_out=out())
        n += 1
    assert n * 0.05 == pytest.approx(pb.SETTLE_MAX_S, abs=0.06) and m.ctrl.state == lr.DRIVING


def test_start_refused_while_the_base_sits_down_after_its_own_stop():
    """The base's idle rule ([event] stop) sits it down and releases control; [nav] still says Balancing for a
    moment, and a claim then is ignored. START waits until the base has sat."""
    r = Rig(st=2)
    r.feed("[event] t=9 ev=stop st=2 reason=idle link=ble")
    r.step(st=2)  # the idle tick drains the event
    ok, msg = r.ctrl.start()
    assert not ok and "sitting down" in msg and r.sent() == []
    r.step(st=3)
    r.step(st=0)  # just landed: still rocking, the base would refuse S0 ("tilt filter off")
    ok, msg = r.ctrl.start()
    assert not ok and r.sent() == []
    r.step(dt=pb.SIT_SETTLE_S, st=0)
    assert r.ctrl.start()[0] and r.sent() == ["nv0,0,1", "S0"]


def test_balance_angle_sent_before_each_stand_up_only():
    """O2.4 after the claim, before S0 (it must reach the base while Stopped); not when already balancing; never
    in a dry run (the logged transport drops it)."""
    r = Rig(st=0)
    r.base.balance_angle_deg = 2.4
    assert r.ctrl.start()[0] and r.sent() == ["nv0,0,1", "O2.4", "S0"]
    b = Rig(st=2)
    b.base.balance_angle_deg = 2.4
    assert b.ctrl.start()[0] and b.sent() == ["nv0,0,1"]
    with pytest.raises(ValueError):
        b.link.balance_angle(9.0)


def test_balance_angle_dropped_in_a_dry_run(tmp_path):
    d = Rig(st=0, dry_run=True, tmp_path=tmp_path)
    d.base.balance_angle_deg = 2.4
    assert d.ctrl.start()[0] and all(s.startswith(bl.DRY_OK) for s in d.tr.sent)


def test_start_waits_for_the_ready_to_stand_bit_when_the_firmware_has_it():
    """f bit 256: Stopped and an S0 would pass the tilt check now. Once seen, START on a sitting base needs it
    (instead of the fixed wait after a sit)."""
    r = Rig(st=0, f=7 | bl.F_READY_TO_STAND)
    r.step(st=0)  # idle tick: the bit is seen
    r.f = 7  # just sat / still rocking: not ready
    r.step(st=0)
    ok, msg = r.ctrl.start()
    assert not ok and "not ready to stand" in msg and r.sent() == []
    r.f = 7 | bl.F_READY_TO_STAND
    r.step(st=0)
    assert r.ctrl.start()[0] and r.sent() == ["nv0,0,1", "S0"]


def test_start_right_after_our_own_park_waits_for_the_landing():
    """STOP sits the base (S1); a START pressed while it lands would get "tilt filter off": refused until it
    has been on its rest SIT_SETTLE_S."""
    r = Rig()
    r.drive()
    r.ctrl.stop()
    r.still()
    assert r.sent()[-1] == "S1"
    r.step(st=3)
    r.step(st=0)
    ok, msg = r.ctrl.start()
    assert not ok and "sitting down" in msg
    r.step(dt=pb.SIT_SETTLE_S, st=0)
    assert r.ctrl.start()[0]


def test_driving_only_once_the_base_names_us_controller():
    """Claim not answered (no controller bit in [nav]): stay STARTING (zeros), never DRIVING."""
    r = Rig(st=2, link_name="ble", f=7)
    assert r.ctrl.start()[0]
    for _ in range(5):
        r.step(st=2, nav_out=out())
    assert r.ctrl.state == lr.STARTING and all(s.startswith("nv0,0,") for s in r.sent())
    r.f = 7 | bl.F_CTRL_BLE
    r.step(st=2, nav_out=out())
    r.step(st=2, nav_out=out())
    assert r.ctrl.state == lr.DRIVING and r.sent()[-1].startswith("nv0.1,0,")


def test_stand_timeout_halts():
    r = Rig(st=0)
    r.ctrl.start()
    for _ in range(int(START_TIMEOUT_S / 0.05) + 2):
        r.step(st=0)
    s = r.sent()
    assert r.ctrl.state == lr.IDLE and "not ready" in r.ctrl.reason and "STARTING" in r.ctrl.reason
    # claim, S0, then only holds (nv0,0) while waiting, then the halt
    assert s[:2] == ["nv0,0,1", "S0"] and s[-1] == "ns" and all(x.startswith("nv0,0,") for x in s[2:-1])
    assert r.link.send_velocity(0.1, 0) is None  # latched: nothing moves until the next START


@pytest.mark.parametrize("st", [None, 3])
def test_start_refused_without_nav_or_while_stopping(st):
    r = Rig()
    if st is None:
        r.link.connect()  # new session: no [nav] yet
    else:
        r.nav(st)
    ok, msg = r.ctrl.start()
    assert not ok and r.ctrl.state == lr.IDLE and r.sent() == []


@pytest.mark.parametrize("hello, why", [
    (None, "no [hello]"),
    (HELLO.replace("setup=7", "setup=6"), "setup=6"),     # not calibrated
    (HELLO.replace("setup=7 ", ""), "setup=None"),       # early firmware: no setup key
    (HELLO.replace("proto=1", "proto=2"), "protocol 2"),
])
def test_start_refused_on_a_bad_hello(hello, why, tmp_path):
    """Drive only a calibrated base speaking protocol 1 (nav-link-api.md section 5). A dry run moves nothing,
    so it may start anyway (to watch an uncalibrated base)."""
    r = Rig(st=0, hello=hello)
    ok, msg = r.ctrl.start()
    assert not ok and why in msg and r.ctrl.state == lr.IDLE and r.sent() == [] and r.cmds == []
    d = Rig(st=0, hello=hello, dry_run=True, tmp_path=tmp_path)
    assert d.ctrl.start() == (True, "starting")
    assert all(s.startswith(bl.DRY_OK) for s in d.tr.sent)


def test_claim_before_stand_up():
    """The claim nv0,0 (any nv makes this link the controller, allowed while sitting) goes before S0: S0 from a
    non-controller link is refused. The controller logs one "start"; the wire lines are in base.jsonl."""
    r = Rig(st=0)
    assert r.ctrl.start()[0]
    assert r.sent() == ["nv0,0,1", "S0"]
    assert [c[2:6] for c in r.cmds] == [("start", 0.0, 0.0, None)]
    b = Rig(st=2)  # already balancing: claim only, no S0
    assert b.ctrl.start()[0] and b.sent() == ["nv0,0,1"]


def test_hold_streamed_while_starting():
    """While standing up, nv0,0 every tick keeps the base's dead-man (500 ms) and idle stop away. Each is
    logged with its real sequence number."""
    r = Rig(st=0)
    r.ctrl.start()
    for _ in range(5):
        r.step(st=1)
    assert r.ctrl.state == lr.STARTING
    assert r.sent() == ["nv0,0,1", "S0"] + [f"nv0,0,{k}" for k in range(2, 7)]
    assert [c[5] for c in r.cmds if c[2] == "command"] == [2, 3, 4, 5, 6]  # (seq 1: the claim inside start)
    r.settle()  # Balancing, settle roll over: DRIVING; zeros until then, nothing after (no navigator output yet)
    assert r.ctrl.state == lr.DRIVING
    s = r.sent()
    assert all(x.startswith("nv0,0,") for x in s[2:])
    assert len(r.sent()) == len(s)


def test_own_owner_event_ignored_while_starting_and_driving():
    """[event] owner link=ble answers our own claim: not a watchdog trip on the BLE link."""
    r = Rig(st=0, link_name="ble", f=7 | bl.F_CTRL_BLE)
    r.ctrl.start()
    r.feed("[event] t=1 ev=owner st=0 link=ble")
    r.step(st=1)
    assert r.ctrl.state == lr.STARTING
    r.settle()
    r.step(nav_out=out())
    r.feed("[event] t=2 ev=owner st=2 link=ble")
    r.step(nav_out=out())
    assert r.ctrl.state == lr.DRIVING and r.sent()[-1].startswith("nv0.1,0,")


@pytest.mark.parametrize("who", ["usb", "none"])
@pytest.mark.parametrize("starting", [False, True])
def test_other_owner_event_halts(who, starting):
    """Control moved to the other link, or was released (a stop procedure): halt."""
    r = Rig(st=0 if starting else 2, link_name="ble", f=7 | bl.F_CTRL_BLE)
    if starting:
        r.ctrl.start()
    else:
        r.drive()
    r.feed(f"[event] t=5 ev=owner st=2 link={who}")
    r.step(st=1 if starting else 2, nav_out=out())
    assert r.ctrl.state == lr.IDLE and "owner" in r.ctrl.reason and r.sent()[-1] == "ns"


@pytest.mark.parametrize("f", [7, 7 | bl.F_CTRL_USB])
def test_driving_without_control_halts(f):
    """[nav] f without the BLE controller bit while driving over BLE: our nv are dropped by the base, halt."""
    r = Rig(link_name="ble", f=7 | bl.F_CTRL_BLE)
    r.drive()
    r.step(nav_out=out())
    assert r.ctrl.state == lr.DRIVING  # with the bit: keeps driving
    r.f = f
    r.step(nav_out=out())
    assert r.ctrl.state == lr.IDLE and "controller" in r.ctrl.reason and r.sent()[-1] == "ns"


def test_controller_bit_not_checked_in_a_dry_run(tmp_path):
    """Dry run: no nv reaches the base, it never names us controller; that is not a watchdog trip."""
    r = Rig(link_name="ble", f=7, dry_run=True, tmp_path=tmp_path)
    r.drive()
    r.step(nav_out=out())
    assert r.ctrl.state == lr.DRIVING


def test_stale_perception_halts_and_stays_halted():
    r = Rig()
    r.drive()
    n = len(r.sent())
    r.step(0.2)  # no new output: 0.25 s since the last one, still fine
    assert r.ctrl.state == lr.DRIVING and len(r.sent()) == n + 1
    r.step(0.15)  # 0.35 s
    assert r.ctrl.state == lr.IDLE and r.sent()[-1] == "ns" and "navigator output" in r.ctrl.reason
    for _ in range(5):  # fresh perception again: nothing moves until START
        r.step(nav_out=out())
    assert r.sent()[-1] == "ns" and r.link.send_velocity(0.1, 0) is None  # latched in BaseLink too


def test_no_navout_after_start_halts():
    r = Rig()
    r.ctrl.start()
    r.step()  # DRIVING (the transition tick sends nothing): so far only the claim at START
    assert r.ctrl.state == lr.DRIVING and r.sent() == ["nv0,0,1"]
    for _ in range(7):
        r.step()
    # no output yet: hold still with nv0,0 (keeps the base's 500 ms dead-man away), then the stale halt
    s = r.sent()
    assert r.ctrl.state == lr.IDLE and s[0] == "nv0,0,1" and s[-1] == "ns" and len(s) > 2
    assert all(c.startswith("nv0,0,") for c in s[1:-1]), s


@pytest.mark.parametrize("event", ["[event] t=5 ev=tipped st=0", "[event] t=5 ev=deadman st=2",
                                   "[event] t=5 ev=estop st=2", "[event] t=5 ev=state st=2 from=2 to=3"])
def test_event_halts(event):
    r = Rig()
    r.drive()
    r.feed(event)
    r.step(nav_out=out())
    assert r.ctrl.state == lr.IDLE and r.sent()[-1] == "ns" and "event" in r.ctrl.reason


def test_liftoff_event_is_a_stand_up_step_only_while_starting():
    """A state change to Liftoff belongs to our stand-up; once Balancing was reached it means the base left
    it."""
    r = Rig()
    r.drive()
    r.feed("[event] t=5 ev=state st=1 from=2 to=1")
    r.step(nav_out=out())
    assert r.ctrl.state == lr.IDLE and "event state" in r.ctrl.reason


def test_state_to_balancing_and_old_events_ignored():
    r = Rig()
    r.feed("[event] t=1 ev=deadman st=2")  # before START: history
    r.drive()
    r.feed("[event] t=5 ev=state st=2 from=1 to=2")
    r.step(nav_out=out())
    assert r.ctrl.state == lr.DRIVING


def test_start_refused_event_while_starting_halts():
    r = Rig(st=0)
    r.ctrl.start()
    r.feed('[event] t=5 ev=start_refused st=0 reason="front start failed"')
    r.step(st=0)
    assert r.ctrl.state == lr.IDLE and "start_refused" in r.ctrl.reason


def test_nav_silence_halts():
    r = Rig()
    r.drive()
    r.step(0.2, st=None, nav_out=out())
    assert r.ctrl.state == lr.DRIVING
    r.step(0.15, st=None, nav_out=out())  # 0.35 s since the last [nav]
    assert r.ctrl.state == lr.IDLE and "[nav]" in r.ctrl.reason


def test_not_balancing_while_driving_halts():
    r = Rig()
    r.drive()
    r.step(st=3, nav_out=out())
    assert r.ctrl.state == lr.IDLE and "st 3" in r.ctrl.reason


def test_link_error_goes_idle():
    r = Rig()
    r.drive()
    r.tr.error = "Bluetooth disconnected"
    r.step(nav_out=out())
    assert r.ctrl.state == lr.IDLE and "base error: Bluetooth disconnected" in r.ctrl.reason
    r.tr.error = None
    assert r.link.send_velocity(0.1, 0) is None  # the halt latched although its write failed


def test_link_error_at_start_goes_idle_and_latches():
    r = Rig(st=0)
    r.tr.error = "Bluetooth disconnected"
    ok, msg = r.ctrl.start()
    assert not ok and "Bluetooth disconnected" in msg and r.ctrl.state == lr.IDLE
    r.tr.error = None
    assert r.link.send_velocity(0.1, 0) is None


def test_stop_waits_for_still_wheels_then_sits():
    """vc / wc (the reference) are 0 at once; S1 waits for the measured wheels."""
    r = Rig()
    r.drive()
    r.ctrl.stop("STOP (page)")
    assert r.sent()[-1] == "ns" and r.ctrl.state == lr.STOPPING
    for _ in range(4):  # reference already 0, the base still rolls at 3 rad/s (0.13 m/s)
        r.step(wl=-3.0, wr=-3.0, nav_out=out())
    assert r.ctrl.state == lr.STOPPING and r.sent()[-1] == "ns"  # no nv, no S1
    r.still(n=pb.STILL_LINES - 1)
    assert r.ctrl.state == lr.STOPPING  # still lines must be consecutive and STILL_LINES of them
    r.step(wl=-0.5, wr=0.5)  # 0.022 m/s at the rim: still
    assert r.ctrl.state == lr.IDLE and r.sent()[-2:] == ["ns", "S1"] and r.ctrl.reason == "STOP (page)"


def test_stop_counts_each_nav_line_once_and_only_after_the_halt():
    r = Rig()
    r.drive()
    r.ctrl.stop()
    for _ in range(5):  # ticks without a new [nav]: the line from before the halt does not count
        r.step(st=None)
    assert r.ctrl.state == lr.STOPPING
    r.still()
    assert r.ctrl.state == lr.IDLE and r.sent()[-1] == "S1"


def test_stop_timeout_sits_anyway_and_stop_while_idle():
    r = Rig()
    r.ctrl.stop()  # IDLE, balancing: STOP still halts and sits
    for _ in range(int(pb.STOP_TIMEOUT_S / 0.05) + 2):
        r.step(wl=-3.0, wr=-3.0)
    assert r.ctrl.state == lr.IDLE and r.sent() == ["ns", "S1"]


def test_no_sit_for_a_base_already_sitting():
    r = Rig(st=0)
    r.ctrl.stop("exit")
    r.still(st=0)
    assert r.ctrl.state == lr.IDLE and r.sent() == ["ns"]


def test_stop_right_after_start_from_sitting_still_sits():
    """The [nav] lines after STOP may predate S0 (st 0): S0 was sent, so S1 must follow."""
    r = Rig(st=0)
    r.ctrl.start()
    r.ctrl.stop("antenna")
    r.still(st=0)
    assert r.ctrl.state == lr.IDLE and r.sent() == ["nv0,0,1", "S0", "ns", "S1"]


def test_finished_is_a_stop():
    r = Rig()
    r.drive()
    r.step(nav_out=out(0.0, 0.0, finished=True, state="DONE"))
    assert r.ctrl.state == lr.STOPPING and r.ctrl.reason == "exploration finished"
    r.still()
    assert r.sent()[-2:] == ["ns", "S1"]


def test_watchdog_halt_does_not_sit():
    """A watchdog halt leaves the base balancing (ns only): no S1 however long the wheels are still."""
    r = Rig()
    r.drive()
    r.step(0.4)  # stale
    assert r.ctrl.state == lr.IDLE
    r.still(n=10)
    assert r.sent()[-1] == "ns"


def test_restart_after_watchdog_and_new_run():
    r = Rig()
    r.drive()
    run = r.ctrl.run
    r.step(0.4)  # stale
    assert r.ctrl.state == lr.IDLE
    r.drive()
    assert r.ctrl.run == run + 1 and r.sent()[-1].startswith("nv")


def test_output_of_the_previous_run_is_dropped():
    """A navigator step under way at START delivers the old run's output (here: finished)."""
    r = Rig()
    old = r.ctrl.run
    r.ctrl.start()
    r.ctrl.set_nav(out(0.0, 0.0, finished=True, state="DONE"), r.clock.t, old)
    r.step()
    r.step()
    assert r.ctrl.state == lr.DRIVING and r.ctrl.reason == ""
    r.step(nav_out=out())  # the new run's output is used
    assert r.sent()[-1].startswith("nv0.1,0,")


def test_non_finite_output_is_dropped():
    """A NaN from the planner is no output; the stale watchdog halts, nothing raises."""
    r = Rig()
    r.drive()
    n = len(r.sent())
    for _ in range(8):
        r.clock.t += 0.05
        r.nav()
        r.ctrl.set_nav(out(0.1, float("nan")), r.clock.t, r.ctrl.run)
        r.ctrl.tick()
    assert r.ctrl.state == lr.IDLE and "navigator output" in r.ctrl.reason
    assert all("nan" not in s for s in r.sent()) and r.sent()[-1] == "ns" and len(r.sent()) > n
    assert any("non-finite" in m for m in r.logs)


def test_tick_error_halts_instead_of_raising():
    """Any exception in a tick halts the base (IDLE, reason, traceback logged)."""
    r = Rig()
    r.drive()

    def broken(v, w):
        raise RuntimeError("boom")
    r.link.send_velocity = broken
    r.step(nav_out=out())  # must not raise
    assert r.ctrl.state == lr.IDLE and "boom" in r.ctrl.reason and r.sent()[-1] == "ns"
    assert any("Traceback" in m for m in r.logs)
    r.step(nav_out=out())
    assert r.sent()[-1] == "ns"  # nothing moves until START


def test_stale_s_is_a_parameter():
    r = Rig(stale_s=0.5)
    r.drive()
    r.step(0.4)
    assert r.ctrl.state == lr.DRIVING
    r.step(0.15)
    assert r.ctrl.state == lr.IDLE and "0.5 s" in r.ctrl.reason


def test_w_cap_applies_to_sent_commands():
    r = Rig(w_max=0.5)
    r.drive()
    r.step(nav_out=out(0.0, 1.0))
    assert r.sent()[-1].startswith("nv0,0.5,")


def test_dry_run_sends_only_read_only_lines(tmp_path):
    r = Rig(st=0, dry_run=True, tmp_path=tmp_path)
    assert r.ctrl.start()[0]
    r.step(st=0)  # dry run: the base never stands, go straight to driving
    assert r.ctrl.state == lr.DRIVING
    for _ in range(3):
        r.step(st=0, nav_out=out(0.3, 0.2))  # st 0 is not a watchdog trip in a dry run
    r.ctrl.stop()
    r.still(st=0)
    assert r.ctrl.state == lr.IDLE
    assert all(s.startswith(bl.DRY_OK) for s in r.tr.sent), r.tr.sent  # what reached the transport
    cmds = [c[2] for c in r.cmds]
    assert cmds == ["start", "command", "command", "command", "stop", "park"]  # the controller's calls
    recs = [json.loads(ln) for ln in open(tmp_path / "base.jsonl")]
    dry = [x["tx"] for x in recs if x.get("dry")]  # the wire lines, as they would have been sent
    assert dry[:2] == ["nv0,0,1", "S0"] and dry[2].startswith("nv0.15,0.1,") and dry[-2:] == ["ns", "S1"]
    assert [x[:2] for x in dry] == ["nv", "S0", "nv", "nv", "nv", "ns", "S1"]
    assert any(x.get("rx", "").startswith("[hello]") for x in recs)


def test_dry_run_still_watches_events(tmp_path):
    r = Rig(st=0, dry_run=True, tmp_path=tmp_path)
    r.ctrl.start()
    r.step(st=0, nav_out=out())
    r.feed("[event] t=5 ev=estop st=0")
    r.step(st=0, nav_out=out())
    assert r.ctrl.state == lr.IDLE and "estop" in r.ctrl.reason


def test_command_thread_sends_at_20hz():
    """Real thread, real clock: the latest navigator output goes out every 50 ms."""
    tr = FakeTransport()
    link = bl.BaseLink(tr)
    link.connect()
    link.feed(time.monotonic(), HELLO)  # START needs proto=1 setup=7
    stop = threading.Event()
    ctrl = lr.Controller(pb.PoussahBase(link, log=lambda m: None), 0.15, 1.0, log=lambda m: None)
    feeder_stop = threading.Event()

    def feeder():  # the base's [nav] at 50 Hz and perception at ~10 Hz
        k = 0
        while not feeder_stop.wait(0.02):
            link.feed(time.monotonic(), nav_line(st=2, seq=k))
            if k % 5 == 0:
                ctrl.set_nav(out(0.1, 0.05 * (k // 5)), time.monotonic(), ctrl.run)
            k += 1

    threading.Thread(target=feeder, daemon=True).start()
    time.sleep(0.1)
    assert ctrl.start()[0]
    th = threading.Thread(target=lr.command_loop, args=(ctrl, stop, lambda m: None), daemon=True)
    th.start()
    time.sleep(0.55)
    stop.set()
    feeder_stop.set()
    th.join(1)
    nv = [s for s in tr.sent if s.startswith("nv")]
    assert nv[0].startswith("nv0,0,")  # the claim at START
    nv = nv[1:]  # the command thread's
    assert ctrl.state == lr.DRIVING and 7 <= len(nv) <= 12, nv  # ~10 in 0.5 s
    assert len({s.split(",")[1] for s in nv}) >= 2  # follows the newest output
    assert sum(s.startswith("nq") for s in tr.sent) >= 2  # connect + the driver's 1 s sync (in status())


# ------------------------------------------------------------------ the driver alone

def test_status_states_and_low_battery():
    r = Rig(st=0)
    assert r.base.status() == itf.BaseStatus(itf.STOPPED)
    r.clock.t += 1.0  # no [nav] for 1 s
    assert r.base.status().state == itf.FAULT and "[nav]" in r.base.status().fault
    r.f = 7 | bl.F_LOW_BATTERY
    r.nav(0)
    assert r.base.status() == itf.BaseStatus(itf.STOPPED, "", True)
    r.base.start()
    assert r.base.status().state == itf.STARTING
    r.nav(2)
    assert r.base.status().state == itf.STARTING  # stood up: its settle roll first
    r.clock.t += pb.SETTLE_MIN_S
    r.nav(2)
    assert r.base.status().state == itf.DRIVING
    r.feed("[event] t=5 ev=low_battery_sitdown st=2")
    st = r.base.status()
    assert st.state == itf.FAULT and "low_battery_sitdown" in st.fault
    assert r.base.status() == st  # kept until stop()
    r.base.stop()
    assert r.base.status().state == itf.STOPPED


def test_odometry_pose_and_measured_velocity():
    """v, w come from the measured wheel speeds (the reference vc / wc ramps to 0 well before the base stops,
    and is 0 when a person pushes the robot); the pose from the wheel angles."""
    r = Rig(hello=None)
    assert r.base.odometry() is None  # no [hello]: no wheel radius, no odometry
    r = Rig()
    v, w = 0.2, 0.5
    wl, wr = -v / R + w * TRACK / (2 * R), -v / R - w * TRACK / (2 * R)  # the firmware's wheel sign
    r.clock.t += 0.02
    r.feed(nav_line(vc=0.0, wc=0.0, wl=round(wl, 4), wr=round(wr, 4), pl=-0.2 / R, pr=-0.2 / R))
    o = r.base.odometry()
    assert (o.v, o.w) == pytest.approx((v, w), abs=1e-3) and o.t == r.clock.t
    assert (o.x, o.y, o.theta) == pytest.approx((0.2, 0.0, 0.0), abs=1e-3)  # both wheels 0.2 m forward
    assert r.base.balancing is True


def test_base_motion_from_wheel_speeds():
    """The tilt trust gets the measured motion (wheel speeds), not the reference vc / wc."""
    r = Rig()
    v, w = 0.2, 0.5
    wl, wr = -v / R + w * TRACK / (2 * R), -v / R - w * TRACK / (2 * R)
    r.feed(nav_line(vc=0.0, wc=0.0, wl=round(wl, 4), wr=round(wr, 4)))
    m = BaseMotion(tau_s=0.2)
    a, vm, wm = m.update(r.base.odometry(), 0.1)
    assert (vm, wm) == pytest.approx((v, w), abs=1e-3) and a == 0.0
    r.feed(nav_line(vc=0.3, wc=0.3))  # reference non-zero, wheels still
    a, vm, wm = m.update(r.base.odometry(), 0.1)
    assert (vm, wm) == (0.0, 0.0) and a == pytest.approx(-1.0, abs=1e-3)  # low-passed: half of -2 m/s^2 (dt = tau / 2)
    assert BaseMotion().update(None, 0.1) == (0.0, 0.0, 0.0)  # no odometry: still


def test_poussah_base_passes_the_static_conformance_checks():
    """What the conformance suite can check without a moving base: status before start is STOPPED, command()
    before start() sends nothing."""
    r = Rig()
    assert r.base.status().state == itf.STOPPED
    assert r.base.command(0.1, 0.0) is None and r.sent() == []
