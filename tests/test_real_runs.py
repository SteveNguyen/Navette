"""Recorded robot runs replayed through navette. LOCAL ONLY: skipped unless NAVETTE_RUNS is set.

Recorded runs are private (pictures of a home): nothing of them is in this repository, and this test writes
nothing but a steps.csv in a temporary folder. Environment:

  NAVETTE_RUNS       a folder of run folders (the robot adapter's recording format)
  NAVETTE_ROBOT      the robot config to replay with, its floor model and calibration files beside it as its
                     [adapter] table says (default: robots/reachy_poussah.toml)
  NAVETTE_REFERENCE  a folder of <run>.npz: the Python prototype's own replay of the same runs, with every
                     input and output of every step (written by a script kept with the prototype). Without
                     it, the tests against the reference are skipped.

    NAVETTE_RUNS=... NAVETTE_ROBOT=... NAVETTE_REFERENCE=... \
        uv run --with onnxruntime pytest tests/test_real_runs.py -s      # -s: the numbers of each comparison

Per run that has its frames:
  1. the core on the reference's inputs: FreeSpace on its masks, Navigator on its scans. Everything exact.
  2. the runtime's replay (recording -> attitude filter -> camera pose -> mask -> scan -> Navigator) against
     the reference's, step by step.
  3. the runtime's replay against what the robot logged (steps.csv). Reported; only the geometry is asserted,
     loosely. The robot's commands are not expected to match row for row: it ran the floor model on another
     CPU, and the stack of that day (some runs predate the forward behaviour's heading hold or the resume
     gate). The geometry differs by what the robot read live: its head pose when the frame arrived (the
     replay takes the one logged at the capture time, 38 ms earlier: up to 2 deg of heading while the head
     wobbles), its IMU samples as they came (up to 1 deg of pitch while the body rocks at 50-100 deg/s).
A run copied without its frames is replayed for its geometry only (pose, camera height, pitch) against the
robot's log.

Measured on 4 runs with frames (8559 navigator calls) when this was written: (1) and (2) every output, plan
and final grid identical; 4 scans with one sector's floor end or closest point moved by 1 to 7 mm (see the
comment in test_core_on_the_reference_inputs). (3) states identical on every step; v differs on 0, 0, 6 and
83 steps of 1404, 3501, 1517 and 2137.
"""
import csv
import functools
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

RUNS = os.environ.get("NAVETTE_RUNS")
ROBOT = os.environ.get("NAVETTE_ROBOT", str(Path(__file__).parents[1] / "robots" / "reachy_poussah.toml"))
REFERENCE = os.environ.get("NAVETTE_REFERENCE")

pytestmark = pytest.mark.skipif(not RUNS, reason="NAVETTE_RUNS not set (recorded robot runs: local only)")

SAFETY = ("radius", "margin", "v_max", "decel", "latency", "unseen_m", "reverse_max", "step")
SCAN_BOOL, SCAN_FLOAT = ("seen", "near"), ("obstacle", "floor", "closest")
OUT_FLOAT = ("out_v", "out_w", "out_v_req", "out_w_req")
STEP_IN = ("dt", "a", "v", "w", "v_prev")
MASK_SHAPE = (192, 320)  # the reference robot's floor masks


def runs(frames):
    """Names of the runs that can be replayed: with their frames, or (frames=False) without (the csv and
    jsonl files only)."""
    if not RUNS:
        return []
    out = []
    for d in sorted(Path(RUNS).iterdir()):
        if all((d / f).exists() for f in ("frames.csv", "meta.json", "state.jsonl", "steps.csv")):
            if any((d / "frames").glob("*.jpg")) == frames:
                out.append(d.name)
    return out


def reference(run):
    if not REFERENCE or not (Path(REFERENCE) / f"{run}.npz").exists():
        pytest.skip("no reference replay of this run (NAVETTE_REFERENCE)")
    return dict(np.load(Path(REFERENCE) / f"{run}.npz"))


def robot_steps(run):
    with open(Path(RUNS) / run / "steps.csv") as f:
        return {int(r["frame"]): r for r in csv.DictReader(f)}


def same(a, b):
    """Equal arrays, nan = nan and inf = inf."""
    return np.array_equal(a, b, equal_nan=True)


def max_diff(a, b):
    """Largest |a - b| where both are finite; and whether the non-finite entries are the same."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    fin = np.isfinite(a) & np.isfinite(b)
    pattern = same(np.where(fin, 0.0, a), np.where(fin, 0.0, b))
    return (float(np.abs(a[fin] - b[fin]).max()) if fin.any() else 0.0), pattern


@functools.lru_cache(maxsize=None)
def replayed(run, model=True):
    """navette.runtime.replay of a run with the real core; what went through it at every Navigator call, as
    arrays under the reference's names. model=False: an all-floor mask instead of the floor model (a run
    without frames: only its geometry means something)."""
    import navette
    from navette import config
    from navette.robots.reachy_poussah import recording
    from navette.runtime import loop
    from navette.runtime import replay as rp

    # the prototype's resume gate (the core's defaults): the runs and the reference replays were made with it,
    # before the config's restart hysteresis
    cfg = config.load(ROBOT).replace("behaviour", resume_n=2, resume_margin=0.0)
    if model and not os.path.exists(cfg.adapter["model"]):
        pytest.skip(f"no floor model at {cfg.adapter['model']} (NAVETTE_ROBOT)")
    if model:
        pytest.importorskip("onnxruntime")
    robot = recording.open_replay_robot(cfg, str(Path(RUNS) / run), **({} if model else {"model": "none"}))
    tap, calls, navs = {}, [], []
    segment, free_space = robot.camera.segment, rp.free_space
    if not model:  # no picture is read
        segment = lambda frame: np.ones(robot.camera.mask_shape, bool)  # noqa: E731

    def segment_(frame):
        tap["frame"], tap["mask"] = frame.index, segment(frame)
        return tap["mask"]

    class Fs:
        def __init__(self, camera):
            self.fs = free_space(camera)

        def scan(self, floor, up_c, h):
            tap["up_c"], tap["scan"] = np.array(up_c, float), self.fs.scan(floor, up_c, h)
            return tap["scan"]

    class Nav:
        def __init__(self, behaviour, forward_v, odometry):
            self.nav = loop.new_navigator(cfg, behaviour, forward_v, odometry)
            navs.append(self.nav)

        def record(self, op, t, pose, cam_h, out=None, **inputs):
            s, nav = tap["scan"], self.nav
            calls.append(dict(
                op=op, frame=tap["frame"], nav=len(navs) - 1, t=t, pose=pose, cam_h=cam_h, up_c=tap["up_c"],
                mask=np.packbits(tap["mask"]), out=out, n_frontier=len(nav.frontier),
                target=(np.nan, np.nan) if nav.target is None else nav.target,
                path_len=-1 if nav.path is None else len(nav.path), **inputs,
                **{k: np.array(getattr(s, k)) for k in SCAN_BOOL + SCAN_FLOAT}))

        def step(self, t, dt, pose, scan, cam_h, a, v, w, v_prev):
            out = self.nav.step(t, dt, pose, scan, cam_h, a, v, w, v_prev)
            self.record(0, t, pose, cam_h, out, dt=dt, a=a, v=v, w=w, v_prev=v_prev)
            return out

        def observe(self, t, pose, scan, cam_h):
            self.nav.observe(t, pose, scan, cam_h)
            self.record(1, t, pose, cam_h)

    robot.camera.segment, rp.free_space = segment_, Fs
    try:
        with tempfile.TemporaryDirectory() as out_dir:
            rows = rp.replay(robot, cfg, str(Path(RUNS) / run), out_dir, Nav, loop.default_up_filter,
                             log=lambda m: None, video=False)
    finally:
        rp.free_space = free_space

    def col(key, none=np.nan):
        return np.array([c.get(key, none) for c in calls])

    def out(field, none):
        return np.array([none if c["out"] is None else getattr(c["out"], field) for c in calls])

    n = {k: col(k) for k in ("op", "frame", "nav", "t", "cam_h", "pose", "up_c", "mask", "path_len", "n_frontier",
                             *SCAN_BOOL, *SCAN_FLOAT, *STEP_IN)}
    n.update(target=col("target").astype(float), out_state=out("state", ""),
             **{k: out(k[4:], np.nan) for k in OUT_FLOAT},
             **{k: out(k[4:], False) for k in ("out_finished", "out_blind_stop")})
    n["grid_l"] = [np.array(nav.grid.log_odds) for nav in navs]
    n["grid_x0"] = np.array([np.nan if nav.grid.x0 is None else nav.grid.x0 for nav in navs])
    n["rows"], n["rays"], n["latency_s"] = rows, np.asarray(robot.camera.rays), cfg.adapter["camera_latency_s"]
    return n


# ------------------------------------------------------------------ 1. the core on the reference's inputs

@pytest.mark.parametrize("run", runs(frames=True))
def test_core_on_the_reference_inputs(run):
    """FreeSpace on the reference's masks and Navigator on the reference's scans, from the reference's rays
    and parameters: nothing of the runtime is involved. Exact (on the machine that wrote the reference)."""
    from navette import FreeSpace, Navigator, SafetyParams, Scan

    r = reference(run)
    fs = FreeSpace(r["rays"], r["rays_low"], MASK_SHAPE, sector_deg=float(r["sector_deg"]),
                   max_range=float(r["max_range"]), min_blob_px=int(r["min_blob_px"]), near_tol=float(r["near_tol"]))
    safety = SafetyParams(**{k: float(r[k]) for k in SAFETY}, n_speeds=int(r["n_speeds"]))
    navs, moved, bad_scan, bad_out = {}, [], [], []
    for i in range(len(r["op"])):
        mask = np.unpackbits(r["mask"][i]).reshape(MASK_SHAPE).astype(bool)
        scan = fs.scan(mask, r["up_c"][i], float(r["cam_h"][i]))
        if not all(same(getattr(scan, k), r[k][i]) for k in SCAN_BOOL + SCAN_FLOAT):
            # Tolerated: a pixel whose bearing is within one float32 ulp of a sector boundary may fall in the
            # next sector (numpy's float32 arctan2 is SIMD code that is not correctly rounded; the core calls
            # libm's atan2f, which is). That sector's floor end or closest point then moves by a few mm.
            # 4 scans of 8559 on the machine that wrote this, none changing an output.
            sectors = sum(int((np.asarray(getattr(scan, k)) != r[k][i]).sum()) for k in SCAN_FLOAT)
            diffs = [max_diff(getattr(scan, k), r[k][i]) for k in SCAN_FLOAT]  # (largest, same inf pattern)
            small = all(d < 0.02 and pattern for d, pattern in diffs)
            (moved if small and sectors <= 2 and all(same(getattr(scan, k), r[k][i]) for k in SCAN_BOOL)
             else bad_scan).append(i)
        j = int(r["nav"][i])
        if j not in navs:  # a START of the run: a new navigator
            navs[j] = Navigator(safety, behaviour=str(r["behaviour"]), odometry=bool(r["odometry"]),
                                resume_n=int(r["resume_n"]), forward_v=float(r["forward_v"]))
        nav = navs[j]
        # the reference's scan, not the one just made: a scan difference must not hide in the navigator's
        scan = Scan(fs.bearing_deg, *(r[k][i] for k in ("seen", "obstacle", "near", "floor", "closest")))
        t, pose, cam_h = float(r["t"][i]), tuple(r["pose"][i]), float(r["cam_h"][i])
        if r["op"][i] == 1:
            nav.observe(t, pose, scan, cam_h)
            got = want = ()
        else:
            dt, a, v, w, v_prev = (float(r[k][i]) for k in STEP_IN)
            o = nav.step(t, dt, pose, scan, cam_h, a, v, w, v_prev)
            got = (o.v, o.w, o.v_req, o.w_req, o.state, o.finished, o.blind_stop)
            want = (*(float(r[k][i]) for k in OUT_FLOAT), str(r["out_state"][i]), bool(r["out_finished"][i]),
                    bool(r["out_blind_stop"][i]))
        target = (np.nan, np.nan) if nav.target is None else nav.target
        plan = (-1 if nav.path is None else len(nav.path), len(nav.frontier))
        if got != want or not same(target, r["target"][i]) or plan != (r["path_len"][i], r["n_frontier"][i]):
            bad_out.append(i)
    grids = [same(nav.grid.log_odds, r["grid_l"][j]) and same(np.nan if nav.grid.x0 is None else nav.grid.x0,
                                                               r["grid_x0"][j]) for j, nav in sorted(navs.items())]
    print(f"\n{run} core on the reference's inputs: {len(r['op'])} calls ({int(r['op'].sum())} observe), "
          f"{len(navs)} navigators; scans identical but a sector boundary pixel {len(moved)}, scans differing "
          f"{len(bad_scan)}, outputs differing {len(bad_out)}, final grids identical {sum(grids)}/{len(grids)}")
    assert not bad_scan and not bad_out and all(grids), (bad_scan[:10], bad_out[:10], grids)
    assert len(moved) <= 0.002 * len(r["op"]), moved


# ------------------------------------------------------------------ 2. the runtime's replay against the reference's

@pytest.mark.parametrize("run", runs(frames=True))
def test_replay_against_the_reference(run):
    r, n = reference(run), replayed(run)
    assert same(n["op"], r["op"]) and same(n["frame"], r["frame"]) and same(n["nav"], r["nav"]), \
        "not the same frames stepped"
    steps = r["op"] == 0
    rays_diff = float(np.abs(n["rays"] - r["rays"]).max())
    mask_px = (np.unpackbits(n["mask"] ^ r["mask"], axis=1)).sum(1)
    # the reference stamps a step with the frame's arrival, navette with its capture (arrival - the camera's
    # lag): the navigator only uses differences of t
    n = dict(n, t=n["t"] + n["latency_s"])
    d = {k: max_diff(n[k], r[k]) for k in ("t", "pose", "cam_h", "up_c", *STEP_IN, *SCAN_FLOAT, *OUT_FLOAT, "target")}
    scan_bool = {k: int((n[k] != r[k]).any(1).sum()) for k in SCAN_BOOL}
    scan_float = int(sum(not same(n[k][i], r[k][i]) for k in SCAN_FLOAT for i in range(len(steps))))
    out_differs = np.zeros(len(steps), bool)
    for k in OUT_FLOAT:
        out_differs |= ~((n[k] == r[k]) | (np.isnan(n[k]) & np.isnan(r[k])))
    for k in ("out_state", "out_finished", "out_blind_stop"):
        out_differs |= n[k] != r[k]
    plan_differs = (n["path_len"] != r["path_len"]) | (n["n_frontier"] != r["n_frontier"]) \
        | ~np.array([same(a, b) for a, b in zip(n["target"], r["target"])])
    grids = [same(a, b) for a, b in zip(n["grid_l"], r["grid_l"])]
    print(f"\n{run} runtime replay vs the reference's: {int(steps.sum())} steps + {int((~steps).sum())} observe; "
          f"rays max diff {rays_diff:.1e}; masks differing {int((mask_px > 0).sum())} "
          f"(pixels: {int(mask_px.sum())}); inputs max diff: "
          + ", ".join(f"{k} {d[k][0]:.1e}" for k in ("t", "pose", "cam_h", "up_c", *STEP_IN))
          + f"; scans: seen / near differing {scan_bool['seen']} / {scan_bool['near']}, ranges not identical in "
          f"{scan_float} arrays (max diff " + ", ".join(f"{k} {d[k][0]:.1e}" for k in SCAN_FLOAT)
          + f"); outputs differing in {int(out_differs.sum())} steps (state "
          f"{int((n['out_state'] != r['out_state']).sum())}, max diff "
          + ", ".join(f"{k[4:]} {d[k][0]:.1e}" for k in OUT_FLOAT)
          + f"); plans differing {int(plan_differs.sum())}; final grids identical {sum(grids)}/{len(grids)}")
    # same pictures through the same model: the same masks
    assert rays_diff == 0.0 and mask_px.sum() == 0
    # the inputs the runtime computes (attitude, camera pose, odometry, base motion): equal to rounding
    assert all(d[k][0] < 1e-9 and d[k][1] for k in ("t", "pose", "cam_h", "up_c", *STEP_IN)), d
    # and what the core makes of them
    assert not out_differs.any() and not plan_differs.any() and all(grids)


# ------------------------------------------------------------------ 3. against what the robot logged

def robot_log_diff(run, n):
    """The replay's steps.csv rows against the robot's, on the frames both stepped: per column, the number of
    rows that differ (as written: 3 decimals) and the largest difference."""
    from navette.runtime.loop import STEP_COLS

    rec = robot_steps(run)
    rows = [dict(zip(STEP_COLS, row)) for row in n["rows"]]
    rows = [(row, rec[int(row["frame"])]) for row in rows if int(row["frame"]) in rec]
    out = {}
    for k in ("x", "y", "heading_deg", "cam_h", "pitch_down_deg", "v", "w", "v_req", "w_req"):
        a, b = (np.array([float(x[k]) for x in col]) for col in zip(*rows))
        diff = np.abs(a - b)
        if k == "heading_deg":
            diff = np.abs((a - b + 180) % 360 - 180)
        out[k] = (int((diff > 1e-9).sum()), float(diff.max()))
    out["state"] = (sum(a["state"] != b["state"] for a, b in rows), 0.0)
    return len(rows), len(rec), out


@pytest.mark.parametrize("run", runs(frames=True))
def test_replay_against_the_robot_log(run):
    n = replayed(run)
    both, robot, d = robot_log_diff(run, n)
    print(f"\n{run} runtime replay vs the robot's steps.csv: {both} steps compared (robot: {robot}); rows "
          "differing (largest difference): " + ", ".join(f"{k} {c} ({m:.3g})" for k, (c, m) in d.items()))
    assert both >= robot - 5  # the replay steps what the robot stepped (but frames before the first head pose)
    # geometry: the same recording through the same equations (the robot read its odometry and IMU live, a
    # few ms apart from what the replay takes at the frame's arrival)
    assert d["x"][1] < 0.02 and d["y"][1] < 0.02 and d["heading_deg"][1] < 2.0
    assert d["cam_h"][1] < 0.005 and d["pitch_down_deg"][1] < 1.0


@pytest.mark.parametrize("run", runs(frames=False))
def test_geometry_of_a_run_without_frames(run):
    """Pose (odometry from the base's lines + head pose), camera height and pitch (IMU through the real
    UpFilter) against the robot's log; an all-floor mask stands in for the pictures."""
    n = replayed(run, model=False)
    both, robot, d = robot_log_diff(run, n)
    geometry = ("x", "y", "heading_deg", "cam_h", "pitch_down_deg")
    print(f"\n{run} (no frames) geometry vs the robot's steps.csv: {both} steps compared (robot: {robot}); rows "
          "differing (largest difference): " + ", ".join(f"{k} {d[k][0]} ({d[k][1]:.3g})" for k in geometry))
    assert both >= robot - 5
    assert d["x"][1] < 0.02 and d["y"][1] < 0.02 and d["heading_deg"][1] < 4.0
    assert d["cam_h"][1] < 0.005 and d["pitch_down_deg"][1] < 2.0
