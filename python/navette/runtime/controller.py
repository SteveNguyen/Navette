"""START / STOP sequencing and the watchdogs, between the perception loop, the page / stop button and the
base. It only knows the Base contract (navette.interfaces): what a given base needs to start, stop and stay
safe is in its driver.

Safety: nothing is sent unless START; Base.stop() is latched in the driver, and every watchdog trip calls it
and goes back to IDLE, so nothing moves again until the next START.
"""
import math
import threading
import time
import traceback

from navette import interfaces as itf

IDLE, STARTING, DRIVING, STOPPING = "IDLE", "STARTING", "DRIVING", "STOPPING"


def cap_speed(v, w, v_max, w_max):
    """Scale (v, w) together so that |v| <= v_max and |w| <= w_max. Scaling both keeps the curvature w / v, so
    the base follows the arc the collision monitor checked, only slower (clipping one alone changes the arc)."""
    k = min(1.0, v_max / abs(v) if v else 1.0, w_max / abs(w) if w else 1.0)
    return v * k, w * k


class Controller:
    """States:  IDLE --start--> STARTING (Base.start(); wait for the base to say DRIVING) --> DRIVING
             any --stop--> STOPPING (Base.stop(); Base.park() every tick until done) --> IDLE
             STARTING / DRIVING --watchdog--> IDLE (Base.stop(), reason kept; no park: the base is left as
                                                    stop() leaves it, e.g. a balancing base keeps balancing)
    Watchdog (STARTING and DRIVING): the base reports a FAULT; the status page went silent (page_timeout_s);
    STARTING: the base is not DRIVING after start_timeout_s; DRIVING: the base is not DRIVING any more, no
    navigator output for stale_s (age from the frame's capture). A finished output (exploration done) is a
    STOP. A BaseError or any other exception in tick() also halts (IDLE, reason and traceback logged): the
    command thread itself never dies.

    Commands: one Base.command() per tick from START until STOP, zeros while STARTING and in DRIVING before
    the first navigator output (the base's dead-man must see a live controller that holds still).

    base: a navette.interfaces.Base, or None (no base: commands are only logged). dry_run is only reported
    (snapshot): in a dry run the adapter hands over a base that sends nothing.
    Thread-safe: start() / stop() from any thread, tick() from the command thread, set_nav() from perception.
    clock: time source (tests pass a fake); on_command(t, state, cmd, v, w, seq, reason) logs each call made
    to the base: cmd is "start", "command", "stop" or "park"."""

    def __init__(self, base, v_max, w_max, dry_run=False, stale_s=0.3, start_timeout_s=10.0,
                 clock=time.monotonic, on_command=None, log=print, page_timeout_s=0.0):
        """page_timeout_s > 0: page dead-man. While STARTING / DRIVING, halt if no page_alive() (the status
        page's poll, every 0.5 s or a quarter of this timeout) for this long: the page's STOP needs the robot's WiFi, which can drop while
        the robot keeps driving. START itself counts as a heartbeat."""
        self.base, self.v_max, self.w_max, self.dry_run = base, v_max, w_max, dry_run
        self.stale_s, self.start_timeout_s, self.clock = stale_s, start_timeout_s, clock
        self.page_timeout_s, self._t_page = page_timeout_s, -math.inf
        self.on_command = on_command or (lambda *a: None)
        self.log = log
        self.state, self.reason = IDLE, ""
        self.sent = (0.0, 0.0)  # last (v, w) sent; 0 when not driving (the behaviour's v_prev)
        self.run = 0            # +1 per START: the perception loop starts a fresh navigator (new exploration)
        self._out, self._out_t = None, None  # latest navigator output of this run and its frame's capture time
        self._t_state = clock()
        self._status = None     # the base's status at the last tick (the page reads this, never the base)
        self._lock = threading.RLock()

    # --- inputs
    def set_nav(self, out, t, run):
        """Navigator output of the frame captured at t, computed by the navigator of START number `run`.
        Outputs of another run (a step that was under way at START) and non-finite commands are dropped: no
        output, so the stale-perception watchdog halts if nothing valid follows."""
        if not (math.isfinite(out.v) and math.isfinite(out.w)):
            self.log(f"non-finite navigator output dropped: v {out.v} w {out.w}")
            return
        with self._lock:
            if run == self.run:
                self._out, self._out_t = out, t

    def page_alive(self):
        """Heartbeat from the status page (any thread)."""
        with self._lock:
            self._t_page = self.clock()

    def start(self):
        """START (page). Returns (ok, message)."""
        with self._lock:
            self._t_page = self.clock()  # START came from the page (or the API): a heartbeat
            if self.state != IDLE:
                return False, f"not idle ({self.state})"
            try:
                if self.base is not None:
                    self.base.start()
            except itf.StartRefused as e:  # nothing was sent: stay as we are
                return False, str(e)
            except itf.BaseError as e:
                self.run += 1  # whatever was under way is over
                self._fail(f"base error: {e}")
                return False, self.reason
            self.reason, self.run = "", self.run + 1
            self._out, self._out_t = None, None  # wait for the new navigator's first output
            self._command("start")
            self._set(STARTING)
            return True, "starting"

    def stop(self, reason="STOP"):
        """STOP (page, stop button, exploration finished, exit): Base.stop() now, park once it has stopped
        (tick)."""
        with self._lock:
            if self.state == STOPPING:
                return
            self.reason = reason
            self.sent = (0.0, 0.0)
            if self.base is not None:
                try:
                    self.base.stop()
                except itf.BaseError as e:
                    self._fail(f"{reason}; base error: {e}")
                    return
            self._command("stop")
            self._set(STOPPING)

    # --- command thread
    def tick(self):
        """One command period: watchdog, sequencing, command. Never raises."""
        with self._lock:
            try:
                self._tick(self.clock())
            except itf.BaseError as e:
                self._fail(f"base error: {e}")
            except Exception as e:  # a bug or a bad value must halt the base, not kill the command thread
                self.log(traceback.format_exc())
                self._fail(f"error: {e!r}")

    def _tick(self, now):
        st = self._status = self.base.status() if self.base is not None else None
        if self.state == IDLE:
            return
        if self.state in (STARTING, DRIVING) and self.page_timeout_s > 0 \
                and now - self._t_page > self.page_timeout_s:
            return self._halt(f"phone page silent for {self.page_timeout_s:.0f} s (WiFi?)")
        if self.state in (STARTING, DRIVING) and st is not None and st.state == itf.FAULT:
            return self._halt(st.fault or "base fault")
        if self.state == STARTING:
            if st is None or st.state == itf.DRIVING:
                self._set(DRIVING)
            elif now - self._t_state > self.start_timeout_s:
                self._halt(f"base not ready {self.start_timeout_s:.0f} s after START ({st.state})")
            else:  # hold still while it gets ready: zeros keep the base's dead-man fed
                self._send(0.0, 0.0)
            return
        if self.state == DRIVING:
            if st is not None and st.state != itf.DRIVING:
                return self._halt(f"base {st.state} while driving")
            # stale perception: the age of the latest output's frame, or of DRIVING itself before the first one
            if now - (self._out_t if self._out is not None else self._t_state) > self.stale_s:
                return self._halt(f"no navigator output for {self.stale_s} s")
            if self._out is None:  # no output of this run yet: hold still
                self._send(0.0, 0.0)
                return
            if self._out.finished:
                return self.stop("exploration finished")
            v, w = cap_speed(self._out.v, self._out.w, self.v_max, self.w_max)
            self.sent = (v, w)
            self._send(v, w)
            return
        if self.state == STOPPING:
            if self.base is None or self.base.park():
                self._command("park")
                self._set(IDLE)

    # --- helpers (under the lock)
    def _send(self, v, w):
        seq = self.base.command(v, w) if self.base is not None else None
        self._command("command", v, w, seq)

    def _set(self, state):
        if state != self.state:
            self.log(f"controller {self.state} -> {state}" + (f" ({self.reason})" if self.reason else ""))
        self.state, self._t_state = state, self.clock()

    def _command(self, cmd, v=0.0, w=0.0, seq=None):
        self.on_command(self.clock(), self.state, cmd, v, w, seq, self.reason)

    def _halt(self, reason):
        """Watchdog: Base.stop() (latched in the driver), IDLE; nothing moves until the next START."""
        self.reason, self.sent = f"watchdog: {reason}", (0.0, 0.0)
        if self.base is not None:
            self.base.stop()  # BaseError: tick() -> _fail (the driver latches before it writes)
        self._command("stop")
        self._set(IDLE)

    def _fail(self, reason):
        """The base call failed or tick() raised: try Base.stop() (it latches even if the write fails), IDLE.
        If the link is down, the base's own dead-man stops it."""
        self.reason, self.sent = reason, (0.0, 0.0)
        try:
            if self.base is not None:
                self.base.stop()
                self._command("stop")
        except Exception:
            pass
        self._set(IDLE)

    def snapshot(self):
        """Status for the page."""
        with self._lock:
            st = self._status
            base = None
            if self.base is not None:
                base = {"state": None if st is None else st.state, "fault": "" if st is None else st.fault,
                        "low_battery": bool(st is not None and st.low_battery)}
            return {"state": self.state, "reason": self.reason, "v": self.sent[0], "w": self.sent[1],
                    "v_max": self.v_max, "w_max": self.w_max, "dry_run": self.dry_run, "base": base}


def command_loop(ctrl, stop_event, log=print, hz=20.0):
    """Command thread: ctrl.tick() at hz on a fixed schedule (resynced if late, never bursting). Survives
    anything (tick() itself halts on errors)."""
    period = 1.0 / hz
    next_t = time.monotonic()
    while not stop_event.is_set():
        try:
            ctrl.tick()
        except Exception as e:
            log(f"command loop: {e!r}")
        next_t += period
        sleep = next_t - time.monotonic()
        if sleep > 0:
            stop_event.wait(sleep)
        else:
            next_t = time.monotonic()
