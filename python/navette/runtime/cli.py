"""Command line of the runtime.

    python -m navette.runtime --robot robots/reachy_poussah.toml --out RUNS            the loop on the robot
    python -m navette.runtime --robot robots/reachy_poussah.toml --out OUT --replay RUN  a recorded run again

On the robot: a run folder RUNS/<YYYYmmdd-HHMMSS>/ (navette.runtime.loop + the adapter's recordings), the status
page on page.port. Nothing moves before START on the page. --dry-run: everything runs, nothing moves the base
(the adapter's base only reads); --no-base: no base at all, the commands are only logged.
--replay also takes a folder of runs (each replayed in its own output folder).
The other options override the robot config for this run.
"""
import argparse
import glob
import importlib
import os
import signal
import threading
import time
from datetime import datetime

from navette import config
from navette.runtime import loop


def _interrupt(signum, frame):
    raise KeyboardInterrupt(signal.Signals(signum).name)


def stop_on_signals():
    """SIGTERM and SIGHUP (kill, ssh session lost) raise KeyboardInterrupt like Ctrl-C, so the finally blocks
    run (the base is stopped and parked, meta.json written). Main thread only."""
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m navette.runtime", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, help="robot config (TOML)")
    ap.add_argument("--out", required=True, help="robot: parent of the run folders; replay: output folder")
    ap.add_argument("--replay", help="replay this recorded run (or folder of runs) instead of running the robot")
    ap.add_argument("--behaviour", choices=["explore", "forward"],
                    help="explore: frontier exploration; forward: straight at --forward-v through the collision "
                         "monitor. A replay uses the run's own")
    ap.add_argument("--forward-v", type=float, help="forward behaviour's request (m/s); still capped by --v-max")
    rob = ap.add_argument_group("robot")
    rob.add_argument("--dry-run", action="store_true", help="nothing moves the base: commands are logged, not sent")
    rob.add_argument("--no-base", action="store_true", help="do not connect to the base")
    rob.add_argument("--v-max", type=float, help="speed cap (m/s), after the collision monitor")
    rob.add_argument("--w-max", type=float, help="turn rate cap (rad/s)")
    rob.add_argument("--stale-s", type=float,
                     help="halt if the newest navigator output's frame was captured longer ago than this (s); the "
                          "collision monitor's latency follows it")
    rob.add_argument("--port", type=int, help="status page port")
    rob.add_argument("--page-timeout-s", type=float,
                     help="halt a run if the status page has not polled for this long; 0 = off (API-only runs)")
    return ap


def load_config(args):
    """The robot config with the command line's overrides."""
    cfg = config.load(args.robot)
    for section, key, value in (("controller", "v_max", args.v_max), ("controller", "w_max", args.w_max),
                                ("perception", "stale_s", args.stale_s), ("page", "port", args.port),
                                ("page", "timeout_s", args.page_timeout_s)):
        if value is not None:
            cfg = cfg.replace(section, **{key: value})
    return cfg


def run_robot(args, stop_event, web_app=None, log=print, **adapter_kw):
    """Open the config's robot and run the loop until stop_event. adapter_kw: passed to the adapter's open."""
    cfg = load_config(args)
    behaviour = args.behaviour or cfg.behaviour.default
    forward_v = cfg.behaviour.forward_v if args.forward_v is None else args.forward_v
    make_navigator = loop.default_navigator(cfg, behaviour, forward_v)  # before anything is opened
    out_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir)
    log(f"run folder {out_dir}")
    adapter = importlib.import_module(cfg.robot.adapter)
    robot = adapter.open(cfg, out_dir, dry_run=args.dry_run, use_base=not args.no_base, log=log, **adapter_kw)
    try:
        loop.run(robot, cfg, out_dir, stop_event, make_navigator, behaviour=behaviour, forward_v=forward_v,
                 dry_run=args.dry_run, web_app=web_app, log=log)
    finally:
        robot.close()


def main_robot(args):
    """The page on page.port (uvicorn thread), the loop in this thread."""
    import uvicorn

    from navette.runtime.page import new_app

    t_start = time.monotonic()

    def log(msg):
        try:
            print(f"[{time.monotonic() - t_start:7.2f}] {msg}", flush=True)
        except OSError:  # the terminal is gone (ssh dropped): keep running the shutdown
            pass

    stop_on_signals()  # SIGTERM / SIGHUP -> KeyboardInterrupt: the loop's finally stops the base
    port = load_config(args).page.port
    app = new_app()
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning"))
    web = threading.Thread(target=server.run, daemon=True)
    web.start()
    log(f"page: http://<robot>:{port}/")
    try:
        run_robot(args, threading.Event(), app, log)
    except KeyboardInterrupt:
        pass  # the loop's finally has stopped the base
    finally:
        server.should_exit = True
        web.join(3.0)


def main_replay(args):
    from navette.runtime.replay import replay

    cfg = load_config(args)
    if args.behaviour:
        cfg = cfg.replace("behaviour", default=args.behaviour)
    if args.forward_v is not None:
        cfg = cfg.replace("behaviour", forward_v=args.forward_v)

    def make_navigator(behaviour, forward_v, odometry):
        return loop.new_navigator(cfg, behaviour, forward_v, odometry)

    adapter = importlib.import_module(cfg.robot.adapter)
    runs = [args.replay] if os.path.exists(os.path.join(args.replay, "frames.csv")) else \
        sorted(d for d in glob.glob(os.path.join(args.replay, "*")) if os.path.exists(os.path.join(d, "frames.csv")))
    for run_dir in runs:
        out = args.out if len(runs) == 1 else os.path.join(args.out, os.path.basename(run_dir))
        robot = adapter.open_replay(cfg, run_dir)
        try:
            replay(robot, cfg, run_dir, out, make_navigator, loop.default_up_filter)
        finally:
            robot.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    return main_replay(args) if args.replay else main_robot(args)
