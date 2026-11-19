"""The generic Controller against the fake base (navette.testing.fakes): only the Base contract is visible.
The same state machine on the real Poussah driver: tests/test_poussah_base.py."""
import threading
import time

import pytest

from navette import interfaces as itf
from navette.runtime import controller as lr
from navette.runtime.loop import NavOut
from navette.testing.fakes import FakeBase, FakeWorld, ManualClock


def out(v=0.1, w=0.0, finished=False, state="DRIVE"):
    return NavOut(v, w, v, w, state, finished)


class Rig:
    """A Controller on a FakeBase, on a manual clock."""

    def __init__(self, base=True, v_max=0.15, w_max=1.0, stale_s=0.3, **kw):
        self.clock, self.cmds, self.logs = ManualClock(), [], []
        self.world = FakeWorld(self.clock)
        self.base = FakeBase(self.world, start_s=0.175) if base else None  # ready at the 4th 50 ms tick
        self.ctrl = lr.Controller(self.base, v_max, w_max, stale_s=stale_s, clock=self.clock,
                                  on_command=lambda *a: self.cmds.append(a), log=self.logs.append, **kw)

    def step(self, dt=0.05, nav_out=None):
        """dt later: the perception output (if any), one tick."""
        self.clock.t += dt
        if nav_out is not None:
            self.ctrl.set_nav(nav_out, self.clock.t, self.ctrl.run)
        self.ctrl.tick()

    def drive(self):
        """START and the first commands: DRIVING."""
        assert self.ctrl.start() == (True, "starting")
        for _ in range(6):  # the fake base needs 0.2 s to be ready
            self.step(nav_out=out())
        assert self.ctrl.state == lr.DRIVING

    def calls(self, kind=None):
        return [c for c in self.base.calls if kind is None or c[0] == kind]


def test_cap_speed_keeps_the_arc():
    assert lr.cap_speed(0.3, 0.6, 0.15, 1.0) == pytest.approx((0.15, 0.3))
    assert lr.cap_speed(-0.3, 0.6, 0.15, 1.0) == pytest.approx((-0.15, 0.3))
    assert lr.cap_speed(0.1, -0.5, 0.15, 1.0) == (0.1, -0.5)


def test_cap_speed_bounds_the_turn_rate():
    assert lr.cap_speed(0.0, 0.8, 0.15, 0.5) == pytest.approx((0.0, 0.5))    # turn in place
    assert lr.cap_speed(0.0, -0.8, 0.15, 0.5) == pytest.approx((0.0, -0.5))
    assert lr.cap_speed(0.1, 1.0, 0.15, 0.5) == pytest.approx((0.05, 0.5))   # same arc, slower
    assert lr.cap_speed(0.3, 1.5, 0.15, 0.5) == pytest.approx((0.1, 0.5))    # the tighter cap wins


def test_nothing_reaches_the_base_before_start():
    r = Rig()
    for _ in range(5):
        r.step(nav_out=out())
    assert r.ctrl.state == lr.IDLE and r.calls() == [] and r.cmds == []


def test_start_holds_with_zeros_then_drives_capped():
    r = Rig()
    assert r.ctrl.start() == (True, "starting") and r.ctrl.state == lr.STARTING
    assert r.ctrl.start()[0] is False  # not idle
    for _ in range(3):  # the base is STARTING: a zero command every tick (its dead-man must stay fed)
        r.step()
    assert r.ctrl.state == lr.STARTING and r.calls() == [("start",)] + [("command", 0.0, 0.0)] * 3
    r.step()  # ready: DRIVING, nothing sent on the transition tick
    assert r.ctrl.state == lr.DRIVING and len(r.calls("command")) == 3
    r.step()  # no navigator output yet: still zeros
    assert r.calls()[-1] == ("command", 0.0, 0.0) and len(r.calls("command")) == 4
    r.step(nav_out=out(0.3, 0.6))
    assert r.calls()[-1] == ("command", pytest.approx(0.15), pytest.approx(0.3))
    assert r.ctrl.sent == pytest.approx((0.15, 0.3))
    r.step(0.5)
    assert r.world.x > 0.05  # and the fake base moved


def test_hold_until_the_first_output_then_stale_halt():
    r = Rig()
    r.ctrl.start()
    for _ in range(5):
        r.step()
    assert r.ctrl.state == lr.DRIVING
    n = len(r.calls("command"))
    for _ in range(7):
        r.step()
    assert r.ctrl.state == lr.IDLE and "navigator output" in r.ctrl.reason
    assert all(c == ("command", 0.0, 0.0) for c in r.calls("command")) and len(r.calls("command")) > n
    assert r.calls()[-1] == ("stop",)  # a watchdog halt: stop, no park


def test_start_refused_changes_nothing():
    r = Rig()
    r.base.refuse = "not calibrated"
    assert r.ctrl.start() == (False, "not calibrated")
    assert r.ctrl.state == lr.IDLE and r.calls() == [] and r.cmds == [] and r.ctrl.run == 0


def test_start_failure_stops_the_base():
    r = Rig()

    def broken():
        raise itf.BaseError("link down")
    r.base.start = broken
    ok, msg = r.ctrl.start()
    assert not ok and "link down" in msg and r.ctrl.state == lr.IDLE and r.calls() == [("stop",)]


def test_start_timeout_halts():
    r = Rig(start_timeout_s=1.0)
    r.base.start_s = 60.0
    r.ctrl.start()
    for _ in range(25):
        r.step()
    assert r.ctrl.state == lr.IDLE and "not ready 1 s after START (STARTING)" in r.ctrl.reason
    assert r.calls()[-1] == ("stop",) and ("park",) not in r.calls()


@pytest.mark.parametrize("starting", [False, True])
def test_base_fault_halts_with_its_reason(starting):
    r = Rig()
    if starting:
        r.ctrl.start()
        r.step()
    else:
        r.drive()
    r.base.fault("event estop")
    r.step(nav_out=out())
    assert r.ctrl.state == lr.IDLE and r.ctrl.reason == "watchdog: event estop"
    assert r.calls()[-1] == ("stop",) and ("park",) not in r.calls()
    for _ in range(5):  # nothing moves until START
        r.step(nav_out=out())
    assert r.calls()[-1] == ("stop",) and r.ctrl.sent == (0.0, 0.0)


def test_base_leaving_driving_halts():
    r = Rig()
    r.drive()
    r.base._session = False  # the base dropped the session by itself
    r.step(nav_out=out())
    assert r.ctrl.state == lr.IDLE and "STOPPED while driving" in r.ctrl.reason


def test_stale_perception_halts_and_stays_halted():
    r = Rig()
    r.drive()
    n = len(r.calls("command"))
    r.step(0.2)  # no new output: 0.25 s since the last one, still fine
    assert r.ctrl.state == lr.DRIVING and len(r.calls("command")) == n + 1
    r.step(0.15)  # 0.35 s
    assert r.ctrl.state == lr.IDLE and r.calls()[-1] == ("stop",) and "navigator output" in r.ctrl.reason
    x = r.base.odometry().x
    for _ in range(5):  # fresh perception again: nothing moves until START
        r.step(nav_out=out())
    assert r.calls()[-1] == ("stop",) and r.base.odometry().x == x > 0


def test_page_dead_man():
    r = Rig(page_timeout_s=2.0)
    r.drive()
    for _ in range(60):  # 3 s of driving with the page polling every 0.5 s
        r.step(nav_out=out())
        if round(r.clock.t * 20) % 10 == 0:
            r.ctrl.page_alive()
    assert r.ctrl.state == lr.DRIVING
    for _ in range(45):  # page silent for 2.25 s
        r.step(nav_out=out())
    assert r.ctrl.state == lr.IDLE and "phone page" in r.ctrl.reason and r.calls()[-1] == ("stop",)


def test_stop_parks_and_stop_while_idle():
    r = Rig()
    r.drive()
    r.ctrl.stop("STOP (page)")
    assert r.ctrl.state == lr.STOPPING and r.calls()[-1] == ("stop",) and r.ctrl.sent == (0.0, 0.0)
    r.ctrl.stop("again")  # already stopping: ignored
    r.step(nav_out=out())
    assert r.ctrl.state == lr.IDLE and r.calls()[-2:] == [("stop",), ("park",)] and r.ctrl.reason == "STOP (page)"
    r.ctrl.stop("exit")  # idle: a STOP still stops and parks the base
    r.step()
    assert r.calls()[-2:] == [("stop",), ("park",)] and r.ctrl.state == lr.IDLE


def test_park_is_polled_until_done():
    r = Rig()
    r.drive()
    answers = [False, False, True]
    r.base.park = lambda: answers.pop(0)
    r.ctrl.stop()
    r.step()
    r.step()
    assert r.ctrl.state == lr.STOPPING
    r.step()
    assert r.ctrl.state == lr.IDLE and answers == []


def test_finished_is_a_stop():
    r = Rig()
    r.drive()
    r.step(nav_out=out(0.0, 0.0, finished=True, state="DONE"))
    assert r.ctrl.state == lr.STOPPING and r.ctrl.reason == "exploration finished"
    r.step()
    assert r.ctrl.state == lr.IDLE and r.calls()[-1] == ("park",)


def test_restart_is_a_new_run_and_old_outputs_are_dropped():
    r = Rig()
    r.drive()
    old = r.ctrl.run
    r.step(0.4)  # stale
    assert r.ctrl.state == lr.IDLE
    r.ctrl.start()
    assert r.ctrl.run == old + 1
    r.ctrl.set_nav(out(0.0, 0.0, finished=True, state="DONE"), r.clock.t, old)  # a step under way at START
    for _ in range(5):
        r.step()
    assert r.ctrl.state == lr.DRIVING and r.ctrl.reason == ""
    r.step(nav_out=out())
    assert r.calls()[-1] == ("command", 0.1, 0.0)


def test_non_finite_output_is_dropped():
    r = Rig()
    r.drive()
    for _ in range(8):
        r.step(nav_out=out(0.1, float("nan")))
    assert r.ctrl.state == lr.IDLE and "navigator output" in r.ctrl.reason
    assert all(c[1] == c[1] and c[2] == c[2] for c in r.calls("command"))  # no NaN went to the base
    assert any("non-finite" in m for m in r.logs)


def test_command_error_halts_instead_of_raising():
    r = Rig()
    r.drive()

    def broken(v, w):
        raise RuntimeError("boom")
    r.base.command = broken
    r.step(nav_out=out())  # must not raise
    assert r.ctrl.state == lr.IDLE and "boom" in r.ctrl.reason and r.calls()[-1] == ("stop",)
    assert any("Traceback" in m for m in r.logs)


def test_base_error_halts_even_if_stop_fails_too():
    r = Rig()
    r.drive()

    def down(*a):
        raise itf.BaseError("link down")
    r.base.command = r.base.stop = down
    r.step(nav_out=out())
    assert r.ctrl.state == lr.IDLE and r.ctrl.reason == "base error: link down"


def test_no_base_logs_commands():
    r = Rig(base=False)
    r.ctrl.start()
    r.step()
    r.step(nav_out=out(0.2, 0.0))
    assert r.ctrl.state == lr.DRIVING and r.cmds[-1][2:6] == ("command", 0.15, 0.0, None)
    r.ctrl.stop()
    r.step()
    assert r.ctrl.state == lr.IDLE and r.ctrl.snapshot()["base"] is None


def test_command_log_and_snapshot():
    r = Rig(dry_run=True)
    assert r.ctrl.snapshot()["base"] == {"state": None, "fault": "", "low_battery": False}  # before the first tick
    r.base.low_battery = True
    r.drive()
    r.ctrl.stop("STOP")
    r.step()
    assert [c[2] for c in r.cmds][:2] == ["start", "command"] and [c[2] for c in r.cmds][-2:] == ["stop", "park"]
    assert r.cmds[-1][1] == lr.STOPPING and r.cmds[-1][6] == "STOP"  # (t, state, cmd, v, w, seq, reason)
    seqs = [c[5] for c in r.cmds if c[2] == "command"]
    assert seqs[-1] is not None  # the id Base.command returned
    snap = r.ctrl.snapshot()
    assert snap["state"] == "IDLE" and snap["dry_run"] and snap["v_max"] == 0.15
    assert snap["base"] == {"state": itf.STOPPED, "fault": "", "low_battery": True}


def test_status_is_read_once_per_tick_and_never_by_the_page():
    """Base.status() may be a driver's periodic tick: the page must read the controller's copy."""
    r = Rig()
    n = []
    real = r.base.status
    r.base.status = lambda: n.append(1) or real()
    for _ in range(3):
        r.step()
        r.ctrl.snapshot()
        r.ctrl.snapshot()
    assert len(n) == 3


def test_command_loop_survives_anything():
    class Flaky:
        calls = 0

        def tick(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first tick")
    ctrl, stop, logs = Flaky(), threading.Event(), []
    th = threading.Thread(target=lr.command_loop, args=(ctrl, stop, logs.append), daemon=True)
    th.start()
    time.sleep(0.2)
    alive = th.is_alive()
    stop.set()
    th.join(1)
    assert alive and ctrl.calls >= 3 and "first tick" in logs[0]
