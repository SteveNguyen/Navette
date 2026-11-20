"""PoseGraph bindings (navette._core): construction, loop correction, robust rejection of a false loop, input
errors. The solver itself is tested in the core (crates/navette-core/src/slam/pose_graph.rs)."""

import math

import numpy as np
import pytest

from navette._core import PoseGraph

INFO = np.diag([1 / 0.05**2, 1 / 0.05**2, 1 / 0.03**2])  # odometry: sigma 5 cm, 0.03 rad per step
LOOP_INFO = np.diag([1 / 0.1**2, 1 / 0.1**2, 1 / 0.05**2])


def compose(a, b):
    """Pose b given in a's frame -> the frame of a."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def between(a, b):
    """Pose of b in a's frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return (c * dx + s * dy, -s * dx + c * dy, math.remainder(b[2] - a[2], 2 * math.pi))


def square(side=5):
    """Square of 4 x side unit steps turning left at the corners. Biased odometry (+2 % distance, +0.01 rad
    per step) chained as the initial estimate (~0.3 m RMS off), plus an exact loop edge (robust) from the
    last node to the first. Returns (ground truth (N, 3), graph)."""
    # dcs_phi = 10: at the default 1, DCS would also reject this true loop (chi2 ~ 60 at the odometry
    # estimate); see small_dcs_phi_rejects_a_loop_that_corrects_much_drift in the core tests
    g = PoseGraph(dcs_phi=10.0)
    gt = [(0.0, 0.0, 0.0)]
    est = gt[0]
    g.add_node(est)
    for k in range(1, 4 * side):
        turn = math.pi / 2 if k % side == 0 else 0.0
        gt.append(compose(gt[-1], (1.0, 0.0, turn)))
        z = (1.02, 0.0, turn + 0.01)
        est = compose(est, z)
        j = g.add_node(est)
        g.add_edge(j - 1, j, z, INFO)
    g.add_edge(len(g) - 1, 0, between(gt[-1], gt[0]), LOOP_INFO.tolist(), robust=True)  # a nested list works too
    return np.array(gt), g


def rmse(poses, gt):
    return float(np.sqrt(np.mean(np.sum((np.array(poses)[:, :2] - gt[:, :2]) ** 2, axis=1))))


def test_construction_and_accessors():
    g = PoseGraph()
    assert g.dcs_phi == 1.0 and len(g) == 0 and g.n_edges() == 0
    assert g.add_node((0.0, 0.0, 0.0)) == 0
    assert g.add_node(np.array([1.0, 0.0, 0.1])) == 1
    assert g.add_edge(0, 1, (1.0, 0.0, 0.1), np.eye(3)) == 0
    assert len(g) == 2 and g.n_edges() == 1
    assert g.is_fixed(0) and not g.is_fixed(1)  # node 0 is the gauge
    assert g.poses() == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.1)]
    g.set_pose(1, (2.0, 0.5, 0.0))
    assert g.pose(1) == (2.0, 0.5, 0.0)
    r = g.optimize()
    assert r.converged and r.chi2_final < 1e-10 < r.chi2_initial and r.iterations >= 1
    assert g.pose(1) == pytest.approx((1.0, 0.0, 0.1), abs=1e-5)
    assert g.pose(0) == (0.0, 0.0, 0.0)
    assert g.edge_scales() == [1.0] and g.edge_chi2()[0] < 1e-10
    assert "OptimizeReport(" in repr(r)


def test_fixed_node_does_not_move():
    g = PoseGraph()
    for p in [(0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (2.0, 0.0, 0.0)]:
        g.add_node(p)
    g.add_edge(0, 1, (1.0, 0.0, 0.0), np.eye(3))
    g.add_edge(1, 2, (1.0, 0.0, 0.0), np.eye(3))
    g.set_fixed(2)
    g.optimize()
    assert g.pose(2) == (2.0, 0.0, 0.0)
    assert g.pose(1) == pytest.approx((1.0, 0.0, 0.0), abs=1e-5)


def test_loop_closure_corrects_the_odometry_drift():
    gt, g = square()
    before = rmse(g.poses(), gt)
    r = g.optimize(50)
    after = rmse(g.poses(), gt)
    # measured: 0.31 m -> 0.09 m, objective 24.4 -> 1.9
    assert r.converged and r.chi2_final < 0.2 * r.chi2_initial
    assert after < 0.5 * before
    assert g.edge_scales()[-1] > 0.9  # the true loop is kept
    assert all(-math.pi <= p[2] < math.pi for p in g.poses())


def test_false_loop_is_rejected():
    gt, g = square()
    g.optimize(50)
    good = g.poses()
    # node 12 is ~3.6 m from node 2, the false loop says 1 m ahead
    wrong = g.add_edge(2, 12, (1.0, 0.0, 0.0), INFO, robust=True)
    g.optimize(50)
    # measured: scale 9e-4, chi2 ~21000, the graph moves by < 1 mm
    assert g.edge_scales()[wrong] < 0.1 and g.edge_chi2()[wrong] > 1000
    assert g.edge_scales()[-2] > 0.9  # the true loop is still kept
    assert rmse(g.poses(), np.array(good)) < 0.02


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.add_node((math.nan, 0.0, 0.0)),
        lambda g: g.add_edge(0, 5, (1.0, 0.0, 0.0), np.eye(3)),
        lambda g: g.add_edge(1, 1, (1.0, 0.0, 0.0), np.eye(3)),
        lambda g: g.add_edge(0, 1, (math.inf, 0.0, 0.0), np.eye(3)),
        lambda g: g.add_edge(0, 1, (1.0, 0.0, 0.0), [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]),  # not symmetric
        lambda g: g.add_edge(0, 1, (1.0, 0.0, 0.0), np.diag([1.0, -1.0, 1.0])),  # not PSD
        lambda g: g.set_fixed(7),
        lambda g: g.pose(7),
        lambda g: g.set_pose(7, (0.0, 0.0, 0.0)),
    ],
)
def test_bad_input_is_a_value_error(call):
    g = PoseGraph()
    g.add_node((0.0, 0.0, 0.0))
    g.add_node((1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        call(g)
    assert g.n_edges() == 0


def test_bad_parameter_and_shapes():
    with pytest.raises(ValueError):
        PoseGraph(dcs_phi=0.0)
    g = PoseGraph()
    g.add_node((0.0, 0.0, 0.0))
    g.add_node((1.0, 0.0, 0.0))
    with pytest.raises((ValueError, TypeError)):
        g.add_edge(0, 1, (1.0, 0.0, 0.0), np.eye(2))  # wrong shape: pyo3 extraction error
