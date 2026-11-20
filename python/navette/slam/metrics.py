"""Trajectory metrics for SE(2) poses (N, 3) arrays of (x, y, theta), pure numpy. `est` is an estimate
(odometry or SLAM output), `ref` the ground truth at the same frames (same N, frame i <-> frame i).

Conventions (as evo / the TUM and KITTI benchmarks, in 2-D):
  - ATE (absolute trajectory error): position error after the rigid SE(2) alignment of est onto ref that
    minimises it (`align`: Horn / Umeyama without scale: the scale is metric here, from the wheels and the
    camera height, and a scale error is one of the things measured);
  - RPE (relative pose error) over a travelled distance delta: for each frame i, the frame j at which the
    *reference* path has gone delta metres further; error of est's motion i -> j against ref's. Independent
    of any global alignment: it measures local drift;
  - drift: end-point error with est anchored on ref's first pose (`anchor`: as odometry starts from the
    true start), in % of ref's path length;
  - return to start: for a run that ends where it started (no ground truth needed), the end pose relative to
    the start pose.
"""
import numpy as np


def wrap(a):
    """Angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    """a o b for poses (..., 3): b expressed in a's frame, put in a's parent frame."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([a[..., 0] + c * b[..., 0] - s * b[..., 1], a[..., 1] + s * b[..., 0] + c * b[..., 1],
                     wrap(a[..., 2] + b[..., 2])], -1)


def inverse(a):
    a = np.asarray(a, float)
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([-c * a[..., 0] - s * a[..., 1], s * a[..., 0] - c * a[..., 1], wrap(-a[..., 2])], -1)


def relative(a, b):
    """a^-1 o b: pose b in the frame of pose a."""
    return compose(inverse(a), b)


def align(est, ref):
    """The SE(2) transform T = (tx, ty, theta) minimising sum_i |T(est_i.xy) - ref_i.xy|^2 (positions only,
    no scale). Closed form of the 2-D Umeyama / Horn problem: with centred points p (est), q (ref), the best
    rotation is atan2(sum p x q, sum p . q), then t = mean(q) - R mean(p)."""
    p, q = np.asarray(est, float)[:, :2], np.asarray(ref, float)[:, :2]
    mp, mq = p.mean(0), q.mean(0)
    pc, qc = p - mp, q - mq
    th = np.arctan2((pc[:, 0] * qc[:, 1] - pc[:, 1] * qc[:, 0]).sum(), (pc * qc).sum())
    c, s = np.cos(th), np.sin(th)
    t = mq - np.array([c * mp[0] - s * mp[1], s * mp[0] + c * mp[1]])
    return np.array([t[0], t[1], th])


def transform(T, poses):
    """Poses (N, 3) moved by the SE(2) transform T (applied on the left: T o pose)."""
    poses = np.asarray(poses, float)
    return compose(np.broadcast_to(np.asarray(T, float), poses.shape), poses)


def anchor(est, ref):
    """est moved so that its first pose is ref's first pose (what odometry would give started at the true
    pose)."""
    return transform(compose(ref[0], inverse(est[0])), est)


def path_length(poses):
    """Travelled distance (m): sum of the position steps (in-place turns add nothing)."""
    d = np.diff(np.asarray(poses, float)[:, :2], axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def ate(est, ref, aligned=True):
    """Absolute trajectory error (m) of the positions: dict(rmse, mean, max, rot_rmse (rad, heading error),
    T: the alignment used, est_aligned). aligned=False: est as given (e.g. already anchored)."""
    T = align(est, ref) if aligned else np.zeros(3)
    e = transform(T, est)
    d = np.hypot(*(e[:, :2] - np.asarray(ref, float)[:, :2]).T)
    r = wrap(e[:, 2] - np.asarray(ref, float)[:, 2])
    return dict(rmse=float(np.sqrt(np.mean(d * d))), mean=float(d.mean()), max=float(d.max()),
                rot_rmse=float(np.sqrt(np.mean(r * r))), T=T, est_aligned=e)


def rpe_pairs(ref, delta):
    """(i, j) pairs: j is the first frame at which ref's path has gone at least delta (m) beyond frame i."""
    d = np.diff(np.asarray(ref, float)[:, :2], axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    j = np.searchsorted(s, s + delta, side="left")
    i = np.flatnonzero(j < len(s))
    return i, j[i]


def rpe(est, ref, delta):
    """Relative pose error over delta metres of travel: dict(n pairs, trans_rmse, trans_mean, trans_max (m),
    trans_pct (trans_mean / delta, %), rot_rmse, rot_mean (rad), rot_per_m (rot_mean / delta, rad/m)).
    No pairs (path shorter than delta): n = 0 and NaNs."""
    i, j = rpe_pairs(ref, delta)
    if len(i) == 0:
        nan = float("nan")
        return dict(n=0, trans_rmse=nan, trans_mean=nan, trans_max=nan, trans_pct=nan, rot_rmse=nan, rot_mean=nan,
                    rot_per_m=nan)
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    E = relative(relative(ref[i], ref[j]), relative(est[i], est[j]))
    tr, rot = np.hypot(E[:, 0], E[:, 1]), np.abs(E[:, 2])
    return dict(n=len(i), trans_rmse=float(np.sqrt(np.mean(tr * tr))), trans_mean=float(tr.mean()),
                trans_max=float(tr.max()), trans_pct=float(100 * tr.mean() / delta),
                rot_rmse=float(np.sqrt(np.mean(rot * rot))), rot_mean=float(rot.mean()),
                rot_per_m=float(rot.mean() / delta))


def return_to_start(est, ref=None):
    """(translation m, |rotation| rad) of the error of the start -> end motion. ref None: the run is known to
    end at its start pose (true motion = identity); else ref's start -> end motion is the truth."""
    est = np.asarray(est, float)
    m = relative(est[0], est[-1])
    if ref is not None:
        ref = np.asarray(ref, float)
        m = relative(relative(ref[0], ref[-1]), m)
    return float(np.hypot(m[0], m[1])), float(abs(wrap(m[2])))


def drift(est, ref):
    """End-point drift with est anchored on ref's start: dict(end_m, end_rot (rad), path_m (ref's), pct
    (end_m in % of the path))."""
    a = anchor(est, ref)
    end = float(np.hypot(*(a[-1, :2] - np.asarray(ref, float)[-1, :2])))
    length = path_length(ref)
    return dict(end_m=end, end_rot=float(abs(wrap(a[-1, 2] - ref[-1][2]))), path_m=length,
                pct=100 * end / length if length > 0 else float("nan"))


def summary(est, ref, delta=1.0):
    """The usual numbers in one flat dict (for a table): ATE after alignment, ATE anchored at the start (what
    a robot using the pose sees), RPE over delta m, end drift, path length."""
    a, an, r, d = ate(est, ref), ate(anchor(est, ref), ref, aligned=False), rpe(est, ref, delta), drift(est, ref)
    return dict(path_m=d["path_m"], ate_rmse=a["rmse"], ate_max=a["max"], ate_anchored_rmse=an["rmse"],
                ate_anchored_max=an["max"], rpe_trans_mean=r["trans_mean"], rpe_trans_pct=r["trans_pct"],
                rpe_rot_deg_per_m=float(np.degrees(r["rot_per_m"])), end_m=d["end_m"], end_pct=d["pct"],
                end_rot_deg=float(np.degrees(d["end_rot"])))
