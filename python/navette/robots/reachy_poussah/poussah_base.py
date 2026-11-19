"""The Poussah self-balancing base behind the Base contract (navette.interfaces), over a poussah_link.BaseLink.

What the contract's calls mean on this base (firmware docs/nav-link-api.md, sections 4 and 5):

  start()    refused unless a [nav] was received, the base is not in its Stopping state, and (except in a dry
             run) [hello] says protocol 1 and setup=7 (calibrated). Then: open the link's latch, claim control
             with nv0,0 (any nv from a link, while the base has no controller, makes that link its controller;
             allowed while sitting; S0 from a non-controller link is refused), and S0 (stand up) if the base
             sits. Events from before are history.
  command()  nv with the next sequence number (echoed by the base as cs). The controller's zero commands while
             STARTING and before the first navigator output are what keeps this base standing: without nv for
             0.5 s its dead-man zeroes the reference, and 3 s later its idle rule sits it down.
  stop()     ns (ramp to 0, keeps balancing), latched in the link: no nv goes out until the next start().
  park()     S1 (sit down) once the wheels are still. Decided on [nav] lines received after the stop only (an
             older line can predate S0 / the ns): STILL_LINES of them in a row with still wheels, or after
             STOP_TIMEOUT_S anyway. The reference vc / wc is 0 at once and says nothing: the measured wheel
             speeds are used. S1 is skipped only if such a line says the base sits (st 0) and no S0 was sent in
             this session (an S0 still on its way stands it up).
  status()   during a session, FAULT on: any [event] except the state changes of our own stand-up (to
             Liftoff while starting, to Balancing) and the `owner` event naming our link (the answer to our
             claim); no [nav] for NAV_STALE_S; once Balancing was reached: a [nav] in another state, or without
             our link's controller bit (our nv are then dropped by the base). So a tipped base, the dead-man,
             the firmware's estop (e.g. its runaway rule: both wheels past 16 rad/s for 0.25 s), a stop
             procedure started by the other link, a low-battery sit-down all halt the controller.
             STARTING until the first Balancing [nav], then DRIVING.
             Also the periodic tick: one clock-sync round trip (nq) per SYNC_S.
  odometry() pose from the wheel angles (poussah_link.Odometry), v and w measured from the wheel speeds.

Dry run: the link's transport drops every command (poussah_link.LoggedTransport), so the base never stands
and never names us controller: start() skips the [hello] check and status() says DRIVING at once, without the
Balancing and controller checks. Events and the [nav] age are watched as in a real run.

After a watchdog halt (stop() without park()) the base keeps balancing; its idle rule sits it down 3.5 s later.
"""
import time

from navette.interfaces import DRIVING, FAULT, STARTING, STOPPED, BaseStatus, Odometry, StartRefused
from navette.robots.reachy_poussah import poussah_link as pl

NAV_STALE_S = 0.3       # no [nav] for this long during a session: FAULT
STOP_TIMEOUT_S = 3.0    # stop -> S1 at the latest, wheels still or not
STILL_MS = 0.05         # wheel rim speed (m/s) under which a wheel counts as still
STILL_LINES = 3         # consecutive post-stop [nav] lines with still wheels: the base has stopped
WHEEL_R_DEFAULT = 0.0444  # m, the firmware's default wheel radius, if no [hello] gave r
SYNC_S = 1.0            # BaseLink.sync() period


def wheels_still(nav, r):
    """Both measured wheel speeds (wl, wr rad/s) below STILL_MS at the rim (r m): the base has stopped
    rolling. (vc / wc are only the reference, which ramps to 0 well before the base stops.)"""
    return max(abs(nav.wl), abs(nav.wr)) * r < STILL_MS


class PoussahBase:
    """link: a connected poussah_link.BaseLink. dry_run: the link's transport sends nothing that moves.
    clock: time source of the [nav] receive stamps (tests pass a fake). Calls are serialised by the controller
    (navette.interfaces), except odometry() which only reads the link."""

    balancing = True

    def __init__(self, link, dry_run=False, clock=time.monotonic, log=print):
        self.link, self.dry_run, self.clock, self.log = link, dry_run, clock, log
        self._session = False    # between start() and stop()
        self._ready = False      # Balancing reached in this session
        self._fault = None       # reason, kept until stop()
        self._stood = False      # S0 sent in this session: park() must sit the base whatever [nav] says
        self._t_stop = clock()   # time of the last stop()
        self._seen = None        # last [nav] counted by park() (each line counted once)
        self._still = 0          # consecutive post-stop [nav] lines with still wheels
        self._next_sync = clock()

    # --- contract
    def odometry(self):
        nav, pose = self.link.latest()
        hello = self.link.hello
        if nav is None or pose is None:  # no [nav] yet, or no [hello] with r and track
            return None
        # wheel speeds in the firmware's wheel sign (poussah_link.Odometry): forward = both negative
        v = -hello.r * (nav.wl + nav.wr) / 2
        w = hello.r * (nav.wl - nav.wr) / hello.track
        return Odometry(nav.t_recv, pose[0], pose[1], pose[2], v, w)

    def start(self):
        nav = self.link.latest()[0]
        if nav is None:
            raise StartRefused("no [nav] from the base")
        if nav.st not in (pl.STOPPED, pl.LIFTOFF, pl.BALANCING):
            raise StartRefused(f"base state {nav.st} (Stopping): try again")
        if not self.dry_run:
            ok, why = self.link.hello_ok()
            if not ok:
                raise StartRefused(why)
        self._drain_events()  # events from before START are history (shown in the log)
        self._session, self._ready, self._fault, self._stood = True, False, None, False
        self.link.resume()
        self.link.send_velocity(0.0, 0.0)  # the claim
        if nav.st == pl.STOPPED:
            self._stood = True  # before the write: a stop must sit it even if the write fails half-way
            self.link.stand()

    def command(self, v, w):
        return self.link.send_velocity(v, w)  # None (nothing sent) while the link's latch is closed

    def stop(self):
        self._session, self._fault = False, None
        self._t_stop, self._seen, self._still = self.clock(), None, 0
        self.link.halt()  # the latch is set before the write, so it holds even if this raises LinkError

    def park(self):
        nav = self.link.latest()[0]
        post = nav is not None and nav.t_recv > self._t_stop  # received after the stop
        if post and nav is not self._seen:
            self._seen = nav
            self._still = self._still + 1 if wheels_still(nav, self._wheel_r()) else 0
        if self._still < STILL_LINES and self.clock() - self._t_stop <= STOP_TIMEOUT_S:
            return False
        sits = post and nav.st == pl.STOPPED and not self._stood
        if not sits:
            self.link.sit()
        self._stood = False
        return True

    def status(self):
        now = self.clock()
        if now >= self._next_sync:
            self._next_sync = now + SYNC_S
            self.link.sync()  # a dead link: the next write raises LinkError and the controller halts
        nav = self.link.latest()[0]
        events = self._drain_events()
        low = bool(nav is not None and nav.f & pl.F_LOW_BATTERY)
        silent = nav is None or now - nav.t_recv > NAV_STALE_S
        if not self._session:
            return BaseStatus(FAULT, f"no [nav] for {NAV_STALE_S} s", low) if silent else BaseStatus(STOPPED, "", low)
        if self._fault is None:
            self._fault = self._judge(nav, events, silent)
        if self._fault is not None:
            return BaseStatus(FAULT, self._fault, low)
        return BaseStatus(DRIVING if self._ready else STARTING, "", low)

    # --- helpers
    def _judge(self, nav, events, silent):
        """The reason this session must end, or None. Also notes that Balancing was reached."""
        for e in events:
            to = e.details.get("to")
            if e.ev == "state" and (to == pl.BALANCING or (not self._ready and to == pl.LIFTOFF)):
                continue  # a stand-up's own state changes
            if e.ev == "owner" and e.details.get("link") == self.link.link_name:
                continue  # our claim at START
            return f"event {e.ev} {e.details or ''}".strip()
        if silent:
            return f"no [nav] for {NAV_STALE_S} s"
        if not self._ready:
            self._ready = self.dry_run or nav.st == pl.BALANCING
            return None
        if not self.dry_run and nav.st != pl.BALANCING:
            return f"[nav] st {nav.st} while driving"
        if not self.dry_run and self.link.is_controller(nav) is False:
            return "this link is not the base's controller ([nav] f)"
        return None

    def _wheel_r(self):
        h = self.link.hello
        return h.r if h is not None and h.r else WHEEL_R_DEFAULT

    def _drain_events(self):
        out = []
        while not self.link.events.empty():
            e = self.link.events.get_nowait()
            self.log(f"[event] {e.ev} st={e.st} {e.details} (session {'open' if self._session else 'closed'})")
            out.append(e)
        return out
