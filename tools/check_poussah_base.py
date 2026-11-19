"""ROBOT MOTION. Run the Base conformance suite (navette.testing.conformance.check_base) on the real Poussah
base, alone: no camera, no head. The base drives ~0.1 m forward, turns ~0.3 rad in place, drives twice more for
0.3 s, stops and sits down. Needs ~0.5 m of free floor ahead (the stand-up's settle roll adds 11-15 cm).

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
        # stop_s: this base rolls on ~0.63 s after a stop with the Mini on (measured), so 0.3 s is too short
        check_base(base, stop_s=1.0)
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
