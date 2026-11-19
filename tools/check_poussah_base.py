"""ROBOT MOTION. Run the Base conformance suite (navette.testing.conformance.check_base) on the real Poussah
base, alone: no camera, no head. The base drives ~0.1 m forward, turns ~0.3 rad in place, drives twice more for
0.3 s, stops and sits down. Needs ~0.6 m of free floor ahead and ~0.3 m behind: the stand-up lurches ~14 cm
back, then sways ~14 cm forward during its first ~2 s of balancing.

    python tools/check_poussah_base.py robots/reachy_poussah.toml OUT_DIR

The link's lines are logged to OUT_DIR/base.jsonl. The base's own dead-man stops it if this script dies.
"""
import os
import sys
import time

from navette.config import load
from navette.robots.reachy_poussah.robot import connect_base
from navette.testing.conformance import check_base


def main(config_path, out_dir):
    cfg = load(config_path)
    os.makedirs(out_dir, exist_ok=True)
    log = lambda m: print(f"[{time.monotonic():.2f}] {m}", flush=True)
    base = connect_base(cfg.adapter, out_dir, dry_run=False, log=log)
    try:
        # This base's balancer pulls it toward a position goal that advances at the commanded speed; a stop (ns,
        # the dead-man or nv0) freezes the goal, and the base then catches up the lag it had built: with the Mini
        # on, ~8 cm after a short 0.1 m/s burst, still 0.11 m/s 1.3 s after the last command (firmware: expected).
        # deadman_s = its 0.5 s dead-man + that catch-up; stop_s likewise (0.63 s lag measured at steady speed).
        check_base(base, deadman_s=2.5, stop_s=1.5)
        log("check_base: PASSED")
        nav, pose = base.link.latest()
        log(f"final odometry pose {pose}; gz offset known: {bool(nav and nav.f & 128)}")
    finally:
        base.stop()
        t_end = time.monotonic() + 5.0
        while not base.park() and time.monotonic() < t_end:
            time.sleep(0.05)
        base.link.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
