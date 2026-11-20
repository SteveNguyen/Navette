"""SLAM front end (navette.slam.frontend) on synthetic data only: the simulated loop of tests/sim_slam.py
(ground truth, drifting odometry) and scripted runs on tests/sim2d.py worlds.

Thresholds (measured on seeds 0-9 of the loop, 11.6 m: odometry end error 0.30-0.57 m, ATE aligned 0.11-0.19 m;
SLAM end <= 0.035 m, ATE aligned <= 0.03 m, anchored <= 0.093 m, accepted edges within 0.05 m / 3 deg of the
truth): the SLAM must beat odometry by a wide margin, and an accepted loop edge further than 0.1 m / 5 deg from
the true relative pose is a false match."""
import math

import numpy as np
import pytest
from sim2d import CAM_H, Sim, rect
from sim_slam import DT, V_MAX, W_MAX, Odometry, scenario

from navette.slam import metrics
from navette.slam.data import Keyframe, Sequence
from navette.slam.frontend import SlamFrontEnd, SlamParams, heading_span

FALSE_M, FALSE_DEG = 0.1, 5.0


def run(seq, params=None, loops=True):
    fe = SlamFrontEnd(params, loops=loops)
    for i in range(len(seq)):
        fe.add(seq.t[i], seq.odom[i], seq.scan(i), seq.cam_h[i], seq.cam_heading[i], bool(seq.glance[i]))
    fe.finish()
    return fe


def edge_errors(fe, seq):
    """(m, deg) error of each accepted loop edge's z (query anchor in the base anchor's frame) against the
    true relative pose."""
    gt = metrics.anchor(seq.gt, seq.odom)  # truth in the map frame (map = odom at the first frame)
    out = []
    for m in fe.matches:
        if m.edge is None:
            continue
        zt = metrics.relative(gt[fe.submaps[m.base].first], gt[fe.submaps[m.query].first])
        e = metrics.relative(zt, np.array(m.z))
        out.append((math.hypot(e[0], e[1]), abs(math.degrees(e[2]))))
    return out


def same_poses(a, b, tol=1e-9):
    """Equal poses, headings compared modulo 2 pi (the simulated odometry is not wrapped)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.allclose(a[:, :2], b[:, :2], atol=tol) and np.allclose(metrics.wrap(a[:, 2] - b[:, 2]), 0, atol=tol)


def scripted(walls, start, plan, seed=0):
    """Sequence of a scripted run on a sim2d world: plan = [("drive", m) | ("turn", rad, counter-clockwise)],
    with sim_slam's drifting odometry and camera-like scan."""
    sim = Sim(walls, start, seed=seed, noise=0.01, fov_deg=35.0, closest=0.34, max_range=2.5)
    odo = Odometry(rng=np.random.default_rng([seed, 1]))
    frames, odom = [], (0.0, 0.0, 0.0)
    for kind, amount in plan:
        done = 0.0
        while done < amount:
            frames.append(Keyframe(t=len(frames) * DT, odom=odom, scan=sim.scan(), cam_h=CAM_H,
                                   gt=tuple(float(x) for x in sim.pose)))
            th = float(sim.pose[2])
            sim.step(V_MAX if kind == "drive" else 0.0, 0.0 if kind == "drive" else W_MAX, DT)
            ds, dth = sim.v * DT, float(sim.pose[2]) - th
            odom = odo.step(ds, dth, DT)
            done += abs(ds) if kind == "drive" else abs(dth)
    for _ in range(10):  # settle (the sim's speed lags the command)
        frames.append(Keyframe(t=len(frames) * DT, odom=odom, scan=sim.scan(), cam_h=CAM_H,
                               gt=tuple(float(x) for x in sim.pose)))
        th = float(sim.pose[2])
        sim.step(0.0, 0.0, DT)
        odom = odo.step(sim.v * DT, float(sim.pose[2]) - th, DT)
    return Sequence.from_keyframes(frames)


@pytest.mark.parametrize("seed", range(5))
def test_loop_closure_beats_odometry_without_false_match(seed):
    seq = scenario(seed)
    fe = run(seq)
    odo, slam = metrics.summary(seq.odom, seq.gt), metrics.summary(fe.poses(), seq.gt)
    assert slam["end_m"] < 0.08 and slam["end_m"] < 0.25 * odo["end_m"]
    assert slam["ate_rmse"] < 0.05 and slam["ate_rmse"] < 0.4 * odo["ate_rmse"]
    assert slam["ate_anchored_rmse"] < 0.15 and slam["ate_anchored_rmse"] < 0.6 * odo["ate_anchored_rmse"]
    errs = edge_errors(fe, seq)
    assert len(errs) >= 1  # the return to the start is closed
    assert all(e_m < FALSE_M and e_deg < FALSE_DEG for e_m, e_deg in errs), errs
    assert all(s > 0.9 for s in fe.loop_scales())  # and the graph keeps them
    # the correction is consistent with the poses: map pose = T o odom pose
    T = fe.map_to_odom()
    assert same_poses([metrics.compose(T, seq.odom[-1])], [fe.pose()])


def test_partial_overlap_is_accepted_with_the_observed_score_only():
    """Seed 6: the spin of submap 15 matched against submaps 8-12 (an earlier pass nearby) sees much that they
    never observed. Scored over the observed points only, it is accepted, and right; scored over all its points
    (the plain score), it is rejected on its score, even at the plain score's former threshold of 0.5."""
    seq = scenario(6)
    fe = run(seq)
    m = next(m for m in fe.matches if m.query == 15 and m.ref == list(range(8, 13)))
    assert m.edge is not None and m.reason == "ok", m.reason
    assert m.n_obs < 0.8 * m.n and m.score > 0.65
    gt = metrics.anchor(seq.gt, seq.odom)
    e = metrics.relative(metrics.relative(gt[fe.submaps[m.base].first], gt[fe.submaps[m.query].first]), np.array(m.z))
    assert math.hypot(e[0], e[1]) < FALSE_M and abs(math.degrees(e[2])) < FALSE_DEG
    plain = run(seq, SlamParams(observed_score=False, min_score=0.5))
    mp = next(x for x in plain.matches if x.query == 15 and x.ref == m.ref)
    assert mp.edge is None and mp.reason.startswith("score") and mp.n_obs == mp.n


def test_no_loops_is_odometry():
    seq = scenario(0)
    fe = run(seq, loops=False)
    assert not fe.matches and fe.stats()["spins"] == 5
    assert same_poses(fe.poses(), seq.odom)


def test_deterministic():
    seq = scenario(1)
    a, b = run(seq), run(seq)
    assert np.array_equal(a.poses(), b.poses())
    assert [(m.query, m.reason, m.pose) for m in a.matches] == [(m.query, m.reason, m.pose) for m in b.matches]


def test_no_spin_is_odometry():
    """Driving around a room with 90 deg corner turns only: no spin submap, nothing matched, poses = odometry."""
    plan = [("drive", 3.0), ("turn", math.pi / 2), ("drive", 1.5), ("turn", math.pi / 2), ("drive", 3.0),
            ("turn", math.pi / 2), ("drive", 1.5)]
    seq = scripted(rect(0, 0, 5, 3.5), (1.0, 0.8, 0.0), plan)
    fe = run(seq)
    assert fe.stats()["spins"] == 0 and not fe.matches
    assert same_poses(fe.poses(), seq.odom)


def test_corridor_gets_no_wrong_lateral_correction():
    """A long corridor (ends beyond the trusted range): spin, drive 3 m, spin, drive back, spin. The return
    spin sees only two parallel walls: at most a lateral / heading correction, never a wrong one."""
    plan = [("turn", 2 * math.pi), ("drive", 3.0), ("turn", 2 * math.pi), ("turn", math.pi), ("drive", 3.0),
            ("turn", 2 * math.pi)]
    seq = scripted(rect(0, 0, 12, 1.2), (4.0, 0.6, 0.0), plan, seed=3)
    fe = run(seq)
    assert fe.stats()["spins"] >= 3
    gt = metrics.anchor(seq.gt, seq.odom)
    # lateral = across the corridor; in the map frame (odom's first frame, heading 0 = along the corridor at
    # the start) it is y, up to the start's heading error (0 here: odometry starts on the truth's heading)
    lat_odo = np.abs(seq.odom[:, 1] - gt[:, 1]).max()
    lat_slam = np.abs(fe.poses()[:, 1] - gt[:, 1]).max()
    assert lat_slam <= lat_odo + 0.03
    for e_m, e_deg in edge_errors(fe, seq):
        assert e_m < FALSE_M and e_deg < FALSE_DEG
    odo_end, slam_end = metrics.drift(seq.odom, seq.gt)["end_m"], metrics.drift(fe.poses(), seq.gt)["end_m"]
    assert slam_end <= odo_end + 0.05  # nothing pulled along the corridor either


def _still_scan():
    """An empty scan (nothing seen): the segmentation tests need poses only."""
    return dict(bearing_deg=np.array([-1.0, 1.0]), seen=np.zeros(2, bool), obstacle=np.full(2, np.inf),
                near=np.zeros(2, bool), floor=np.zeros(2), closest=np.full(2, np.inf))


def _feed(fe, poses):
    for k, p in enumerate(poses):
        fe.add(k * DT, p, _still_scan(), CAM_H)


def _turn(x, y, th0, dth, n):
    return [(x, y, th0 + dth * (k + 1) / n) for k in range(n)]


def _drive(x0, y0, th, dist, n):
    return [(x0 + dist * (k + 1) / n * math.cos(th), y0 + dist * (k + 1) / n * math.sin(th), th) for k in range(n)]


def test_heading_span():
    assert heading_span([0.0, 1.0, -0.5, 0.2]) == pytest.approx(1.5)
    assert heading_span([]) == 0.0


def test_spin_detection():
    fe = SlamFrontEnd(SlamParams(spin_deg=180.0, travel_m=1.0))
    poses = [(0.0, 0.0, 0.0)] + _drive(0.0, 0.0, 0.0, 0.5, 10)       # frames 0-10: drive 0.5 m
    poses += _turn(0.5, 0.0, 0.0, 2 * math.pi, 80)                    # 11-90: a full turn in place
    poses += _drive(0.5, 0.0, 2 * math.pi, 0.8, 16)                   # 91-106: drive away
    poses += _turn(1.3, 0.0, 2 * math.pi, math.radians(150), 30)      # 107-136: a 150 deg turn: no spin
    poses += _drive(1.3, 0.0, 2 * math.pi + math.radians(150), 0.5, 10)
    _feed(fe, poses)
    fe.finish()
    kinds = [(s.kind, s.first, s.last) for s in fe.submaps]
    # the travel before the turn ends where the turn starts (the turn's first frame is still at the spot)
    assert kinds[0] == ("travel", 0, 9)
    # the full turn closes as soon as it spans 360 deg; its frames are contiguous with the travel
    assert kinds[1][0] == "spin" and kinds[1][1] == 10 and kinds[1][2] == 90
    assert [k for k, _, _ in kinds].count("spin") == 1
    # every frame in exactly one submap, in order
    sub = fe.frame_submaps()
    assert len(sub) == len(poses) and np.all(np.diff(sub) >= 0)
    for s in fe.submaps:
        assert np.all(sub[s.first:s.last + 1] == s.index)


def test_partial_turn_closes_when_leaving_and_roundabout_is_no_spin():
    fe = SlamFrontEnd(SlamParams(spin_deg=180.0))
    poses = [(0.0, 0.0, 0.0)] + _turn(0.0, 0.0, 0.0, math.radians(200), 40)  # 200 deg in place
    poses += _drive(0.0, 0.0, math.radians(200), 0.6, 12)                    # leave: closes the spin
    # a 360 deg turn on a 0.3 m radius circle: never within 0.2 m of its start for long: not a spin
    th0 = math.radians(200)
    x0, y0 = poses[-1][:2]
    poses += [(x0 + 0.3 * (math.cos(th0 + a) - math.cos(th0)), y0 + 0.3 * (math.sin(th0 + a) - math.sin(th0)),
               th0 + a) for a in np.linspace(0.05, 2 * math.pi, 60)]
    _feed(fe, poses)
    fe.finish()
    spins = [s for s in fe.submaps if s.kind == "spin"]
    # the spin ends when the robot is more than spin_max_trans_m (0.2 m) from where the turn started: its
    # first 4 drive frames (0.05 m each) are still in it
    assert len(spins) == 1 and spins[0].first == 0 and spins[0].last == 44
