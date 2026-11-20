//! SE(2) pose graph (docs/SLAM.md, back end): Levenberg-Marquardt on a sparse block system, with Dynamic
//! Covariance Scaling on the edges marked robust (loop closures).
//!
//! Nodes are poses x_k = (x, y, theta) in the map frame. An edge (i, j, z, info) says "node j seen from node
//! i is at z" (z = x_i^-1 * x_j, i.e. the pose of j expressed in i's frame, the convention of odometry
//! increments and scan matches). Its error is (Grisetti et al. 2010, "A Tutorial on Graph-Based SLAM")
//!
//!   e = t2v(z^-1 * (x_i^-1 * x_j))
//!     = ( R_z^T (R_i^T (t_j - t_i) - t_z),  wrap(theta_j - theta_i - theta_z) )
//!
//! and its cost the squared Mahalanobis norm chi2 = e^T info e. The angle is wrapped so that a graph whose
//! headings cross +-pi has no 2 pi jump in its error. Poses are updated additively in (x, y, theta), and the
//! Jacobians below are the closed form of that error w.r.t. that parametrisation.
//!
//! Robust edges: Dynamic Covariance Scaling (Agarwal et al., ICRA 2013) scales the information of an edge by
//! s^2 with s = min(1, 2 phi / (phi + chi2)): an edge that disagrees with the rest of the graph (a false loop
//! closure) loses its weight in closed form. The objective minimised (and reported as chi2) is the sum of
//! rho(chi2) over the edges, with rho(c) = c for plain edges and, for robust edges,
//!   rho(c) = c                            if c <= phi
//!   rho(c) = 3 phi - 4 phi^2 / (phi + c)  otherwise,
//! the function whose derivative is s^2: the iteratively reweighted step (information times s^2) is then the
//! Gauss-Newton step of rho, so Levenberg-Marquardt accepts or rejects steps on a consistent objective. rho
//! is continuous with a continuous slope at phi, increasing and bounded by 3 phi: an outlier costs at most
//! 3 phi whatever its error. (g2o evaluates s^2 chi2 instead, which decreases for chi2 > phi: a step pushing
//! an outlier further away would look like progress.)
//!
//! Solver: the normal equations H dx = -g are 3x3 block-sparse with the structure of the graph (a chain plus
//! a few loops). They are solved by a block sparse Cholesky factorisation on a minimum-degree ordering of the
//! free nodes, computed once per optimize() call (the structure does not change between iterations). With the
//! natural order, a loop between node 0 and node 999 would make the profile of every later row span the whole
//! matrix (an envelope solver would then cost ~ dense); minimum degree eliminates the chain segments first
//! (degree 2, no fill) and keeps the fill to the loop "skeleton".

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};

use crate::geometry::{wrap_angle, Mat3, Pose2, Vec3};
use crate::params::{positive, ParamError};

/// Pose graph parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseGraphParams {
    /// Dynamic Covariance Scaling phi: a robust edge with chi2 <= phi keeps its full weight; beyond, its
    /// information is scaled by s^2, s = 2 phi / (phi + chi2). The chi2 of a correct 3-dof edge is ~3 on
    /// average, so phi = 1 already down-weights inliers a little (s ~ 0.5); what matters is that a false loop
    /// (chi2 in the hundreds) gets s ~ 0.
    pub dcs_phi: f64,
}

impl Default for PoseGraphParams {
    fn default() -> Self {
        Self { dcs_phi: 1.0 }
    }
}

impl PoseGraphParams {
    pub fn validate(&self) -> Result<(), ParamError> {
        positive("dcs_phi", self.dcs_phi)
    }
}

/// A relative pose measurement between two nodes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    pub i: usize,
    pub j: usize,
    /// Pose of node j in node i's frame.
    pub z: Pose2,
    /// Information matrix (inverse covariance) of z, in the order (x, y, theta); symmetric PSD.
    pub info: Mat3,
    /// Down-weighted by Dynamic Covariance Scaling when it disagrees with the graph (loop closures).
    pub robust: bool,
}

/// Outcome of [`PoseGraph::optimize`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizeReport {
    /// Linear systems solved (accepted and rejected Levenberg-Marquardt steps).
    pub iterations: usize,
    /// Objective (sum of rho(chi2), see the module documentation) before and after.
    pub chi2_initial: f64,
    pub chi2_final: f64,
    /// Within max_iters, the relative change of the objective fell below 1e-9 or the objective below 1e-12.
    pub converged: bool,
}

/// SE(2) pose graph. The first node added is fixed (the gauge: without a fixed node the graph can move as a
/// whole); [`PoseGraph::set_fixed`] changes it.
#[derive(Clone, Debug)]
pub struct PoseGraph {
    params: PoseGraphParams,
    poses: Vec<Pose2>,
    fixed: Vec<bool>,
    edges: Vec<Edge>,
}

/// Relative cost change below which the optimisation has converged.
const REL_TOL: f64 = 1e-9;
/// Objective below which the graph is consistent: chi2 is dimensionless, this is errors of ~1e-6 sigma.
const ABS_TOL: f64 = 1e-12;
/// Initial Levenberg-Marquardt damping, relative to the diagonal of H (Marquardt scaling).
const LAMBDA_INIT: f64 = 1e-4;
/// Damping beyond which no step can decrease the cost any more: give up.
const LAMBDA_MAX: f64 = 1e16;
/// Floor of the Marquardt scaling diagonal, so that a node with no constraint on one of its coordinates
/// still gets a positive pivot (its gradient is zero there: it does not move).
const DIAG_FLOOR: f64 = 1e-9;
/// Position of a fixed node (not in the linear system).
const FIXED: usize = usize::MAX;

fn index_err(what: &str, i: usize, n: usize) -> ParamError {
    ParamError(format!("{what} = {i}: out of range (the graph has {n} nodes)"))
}

fn pose_ok(name: &str, p: &Pose2) -> Result<(), ParamError> {
    if p.x.is_finite() && p.y.is_finite() && p.theta.is_finite() {
        Ok(())
    } else {
        Err(ParamError(format!("{name} = {p:?}: must be finite")))
    }
}

/// Information matrix check: finite, symmetric and positive semi-definite up to rounding. Returns it exactly
/// symmetrised. PSD is checked with all principal minors (the leading ones alone would accept
/// diag(0, -1, 1)) of the matrix scaled to a largest entry of 1.
fn info_ok(info: &Mat3) -> Result<Mat3, ParamError> {
    let err = |why: &str| Err(ParamError(format!("info = {info:?}: {why}")));
    if info.iter().flatten().any(|v| !v.is_finite()) {
        return err("must be finite");
    }
    let scale = info.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return Ok(*info); // no information: a valid (useless) edge
    }
    let tol = 1e-9;
    for (r, c) in [(1, 0), (2, 0), (2, 1)] {
        if (info[r][c] - info[c][r]).abs() > tol * scale {
            return err("must be symmetric");
        }
    }
    let a: Mat3 = std::array::from_fn(|r| std::array::from_fn(|c| 0.5 * (info[r][c] + info[c][r]) / scale));
    let minor2 = |p: usize, q: usize| a[p][p] * a[q][q] - a[p][q] * a[q][p];
    let det = a[0][0] * minor2(1, 2) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if (0..3).any(|k| a[k][k] < -tol)
        || [(0, 1), (0, 2), (1, 2)].iter().any(|&(p, q)| minor2(p, q) < -tol)
        || det < -tol
    {
        return err("must be positive semi-definite");
    }
    Ok(a.map(|row| row.map(|v| v * scale)))
}

impl PoseGraph {
    pub fn new(params: PoseGraphParams) -> Result<Self, ParamError> {
        params.validate()?;
        Ok(Self {
            params,
            poses: Vec::new(),
            fixed: Vec::new(),
            edges: Vec::new(),
        })
    }

    pub fn params(&self) -> &PoseGraphParams {
        &self.params
    }

    /// Add a node with its initial estimate; returns its index. The first node is fixed.
    pub fn add_node(&mut self, pose: Pose2) -> Result<usize, ParamError> {
        pose_ok("pose", &pose)?;
        self.fixed.push(self.poses.is_empty());
        self.poses.push(pose);
        Ok(self.poses.len() - 1)
    }

    /// Add an edge "node j is at z in node i's frame" with information `info`; returns its index.
    pub fn add_edge(
        &mut self,
        i: usize,
        j: usize,
        z: Pose2,
        info: Mat3,
        robust: bool,
    ) -> Result<usize, ParamError> {
        let n = self.poses.len();
        if i >= n {
            return Err(index_err("i", i, n));
        }
        if j >= n {
            return Err(index_err("j", j, n));
        }
        if i == j {
            return Err(ParamError(format!(
                "i = j = {i}: an edge joins two different nodes"
            )));
        }
        pose_ok("z", &z)?;
        let info = info_ok(&info)?;
        self.edges.push(Edge {
            i,
            j,
            z,
            info,
            robust,
        });
        Ok(self.edges.len() - 1)
    }

    /// Fix node i (the optimiser does not move it) or free it.
    pub fn set_fixed(&mut self, i: usize, fixed: bool) -> Result<(), ParamError> {
        let n = self.poses.len();
        *self.fixed.get_mut(i).ok_or_else(|| index_err("i", i, n))? = fixed;
        Ok(())
    }

    pub fn is_fixed(&self, i: usize) -> Result<bool, ParamError> {
        self.fixed
            .get(i)
            .copied()
            .ok_or_else(|| index_err("i", i, self.poses.len()))
    }

    /// Current estimate of node i. Headings moved by optimize() are wrapped to [-pi, pi).
    pub fn pose(&self, i: usize) -> Result<Pose2, ParamError> {
        self.poses
            .get(i)
            .copied()
            .ok_or_else(|| index_err("i", i, self.poses.len()))
    }

    /// Replace the estimate of node i (re-initialisation).
    pub fn set_pose(&mut self, i: usize, pose: Pose2) -> Result<(), ParamError> {
        pose_ok("pose", &pose)?;
        let n = self.poses.len();
        *self.poses.get_mut(i).ok_or_else(|| index_err("i", i, n))? = pose;
        Ok(())
    }

    pub fn poses(&self) -> &[Pose2] {
        &self.poses
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.poses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Per edge, its unscaled chi2 = e^T info e at the current estimate.
    pub fn edge_chi2(&self) -> Vec<f64> {
        self.edges
            .iter()
            .map(|e| chi2(&e.info, &error(&self.poses[e.i], &self.poses[e.j], &e.z)))
            .collect()
    }

    /// Per edge, its Dynamic Covariance Scaling factor s at the current estimate (1 for non-robust edges):
    /// a loop closure with a tiny scale is one the graph rejects.
    pub fn edge_scales(&self) -> Vec<f64> {
        let phi = self.params.dcs_phi;
        self.edges
            .iter()
            .zip(self.edge_chi2())
            .map(|(e, c)| if e.robust { dcs(phi, c).0 } else { 1.0 })
            .collect()
    }

    /// Objective at `poses`: sum over the edges of rho(chi2).
    fn cost(&self, poses: &[Pose2]) -> f64 {
        let phi = self.params.dcs_phi;
        self.edges
            .iter()
            .map(|e| {
                let c = chi2(&e.info, &error(&poses[e.i], &poses[e.j], &e.z));
                if e.robust {
                    dcs(phi, c).1
                } else {
                    c
                }
            })
            .sum()
    }

    /// Levenberg-Marquardt, at most `max_iters` linear solves. The estimates are updated in place, by
    /// accepted steps only: the objective never increases.
    pub fn optimize(&mut self, max_iters: usize) -> OptimizeReport {
        let mut f = self.cost(&self.poses);
        let mut report = OptimizeReport {
            iterations: 0,
            chi2_initial: f,
            chi2_final: f,
            converged: false,
        };
        let mut sys = System::new(self);
        if sys.n() == 0 || f <= ABS_TOL {
            report.converged = true;
            return report;
        }
        // Nielsen's damping schedule (Madsen, Nielsen & Tingleff, "Methods for non-linear least squares")
        let (mut lambda, mut nu) = (LAMBDA_INIT, 2.0);
        let mut relinearize = true;
        let mut cand = self.poses.clone();
        let mut step = vec![[0.0; 3]; sys.n()];
        while report.iterations < max_iters {
            if relinearize {
                sys.linearize(self);
                relinearize = false;
            }
            report.iterations += 1;
            if !sys.solve_damped(lambda, &mut step) {
                // not positive definite at this damping (e.g. a graph with no fixed node): damp more
                lambda *= nu;
                nu *= 2.0;
                if lambda > LAMBDA_MAX {
                    break;
                }
                continue;
            }
            cand.copy_from_slice(&self.poses);
            for (node, &p) in sys.pos.iter().enumerate() {
                if p != FIXED {
                    let (c, h) = (&mut cand[node], step[p]);
                    c.x += h[0];
                    c.y += h[1];
                    c.theta = wrap_angle(c.theta + h[2]);
                }
            }
            let f_new = self.cost(&cand);
            let small = (f - f_new).abs() <= REL_TOL * f;
            if f_new < f {
                // gain ratio: actual / predicted decrease; the quadratic model predicts
                // f - L(h) = lambda h^T D h - g^T h (since H h = -g - lambda D h)
                let pred: f64 = step
                    .iter()
                    .zip(sys.damp.iter().zip(&sys.g))
                    .flat_map(|(h, (d, g))| (0..3).map(move |k| h[k] * (lambda * d[k] * h[k] - g[k])))
                    .sum();
                let rho = (f - f_new) / pred.max(f64::MIN_POSITIVE);
                lambda *= (1.0 / 3.0f64).max(1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0;
                self.poses.copy_from_slice(&cand);
                f = f_new;
                relinearize = true;
            } else {
                lambda *= nu;
                nu *= 2.0;
            }
            if small || f <= ABS_TOL {
                report.converged = true;
                break;
            }
            if lambda > LAMBDA_MAX {
                break;
            }
        }
        report.chi2_final = f;
        report
    }
}

/// e = t2v(z^-1 (x_i^-1 x_j)), the angle wrapped to [-pi, pi).
fn error(xi: &Pose2, xj: &Pose2, z: &Pose2) -> Vec3 {
    let d = z.to_robot(xi.to_robot([xj.x, xj.y])); // R_z^T (R_i^T (t_j - t_i) - t_z)
    [d[0], d[1], wrap_angle(xj.theta - xi.theta - z.theta)]
}

/// Error and its Jacobians A = de/dx_i, B = de/dx_j (rows: e_x, e_y, e_theta; columns: x, y, theta).
/// With M = R_z^T R_i^T (the rotation by -(theta_i + theta_z)):
///   A = [ -M  R_z^T dR_i^T/dtheta (t_j - t_i) ;  0 0 -1 ],   B = [ M  0 ;  0 0 1 ].
fn linearize_edge(xi: &Pose2, xj: &Pose2, z: &Pose2) -> (Vec3, Mat3, Mat3) {
    let (si, ci) = xi.theta.sin_cos();
    let (sz, cz) = z.theta.sin_cos();
    let (sa, ca) = (xi.theta + z.theta).sin_cos();
    let (dx, dy) = (xj.x - xi.x, xj.y - xi.y);
    // dR_i^T/dtheta (t_j - t_i), with R_i^T = [[c, s], [-s, c]]; then R_z^T times it
    let v = [-si * dx + ci * dy, -ci * dx - si * dy];
    let w = [cz * v[0] + sz * v[1], -sz * v[0] + cz * v[1]];
    let a = [[-ca, -sa, w[0]], [sa, -ca, w[1]], [0.0, 0.0, -1.0]];
    let b = [[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]];
    (error(xi, xj, z), a, b)
}

fn chi2(info: &Mat3, e: &Vec3) -> f64 {
    let ie = mat_vec(info, e);
    e[0] * ie[0] + e[1] * ie[1] + e[2] * ie[2]
}

/// Dynamic Covariance Scaling: (scale s, robust cost rho) of an edge with this chi2 (module documentation).
fn dcs(phi: f64, c: f64) -> (f64, f64) {
    if c <= phi {
        (1.0, c)
    } else {
        (2.0 * phi / (phi + c), 3.0 * phi - 4.0 * phi * phi / (phi + c))
    }
}

// --- 3x3 blocks ------------------------------------------------------------------------------------------

fn mat_vec(a: &Mat3, v: &Vec3) -> Vec3 {
    std::array::from_fn(|r| a[r][0] * v[0] + a[r][1] * v[1] + a[r][2] * v[2])
}

/// a^T v
fn mat_t_vec(a: &Mat3, v: &Vec3) -> Vec3 {
    std::array::from_fn(|c| a[0][c] * v[0] + a[1][c] * v[1] + a[2][c] * v[2])
}

fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    std::array::from_fn(|r| {
        std::array::from_fn(|c| a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c])
    })
}

/// a^T b
fn mul_tn(a: &Mat3, b: &Mat3) -> Mat3 {
    std::array::from_fn(|r| {
        std::array::from_fn(|c| a[0][r] * b[0][c] + a[1][r] * b[1][c] + a[2][r] * b[2][c])
    })
}

/// a b^T
fn mul_nt(a: &Mat3, b: &Mat3) -> Mat3 {
    std::array::from_fn(|r| {
        std::array::from_fn(|c| a[r][0] * b[c][0] + a[r][1] * b[c][1] + a[r][2] * b[c][2])
    })
}

fn transpose(a: &Mat3) -> Mat3 {
    std::array::from_fn(|r| std::array::from_fn(|c| a[c][r]))
}

fn add_assign(a: &mut Mat3, b: &Mat3) {
    for r in 0..3 {
        for c in 0..3 {
            a[r][c] += b[r][c];
        }
    }
}

fn sub_assign(a: &mut Mat3, b: &Mat3) {
    for r in 0..3 {
        for c in 0..3 {
            a[r][c] -= b[r][c];
        }
    }
}

/// Lower Cholesky factor of a symmetric 3x3 block (its lower triangle is read); None if not positive
/// definite.
fn chol3(a: &Mat3) -> Option<Mat3> {
    let sq = |v: f64| {
        if v > 0.0 && v.is_finite() {
            Some(v.sqrt())
        } else {
            None
        }
    };
    let l00 = sq(a[0][0])?;
    let l10 = a[1][0] / l00;
    let l20 = a[2][0] / l00;
    let l11 = sq(a[1][1] - l10 * l10)?;
    let l21 = (a[2][1] - l20 * l10) / l11;
    let l22 = sq(a[2][2] - l20 * l20 - l21 * l21)?;
    Some([[l00, 0.0, 0.0], [l10, l11, 0.0], [l20, l21, l22]])
}

/// Solve L y = v, L lower triangular.
fn fwd(l: &Mat3, v: &Vec3) -> Vec3 {
    let y0 = v[0] / l[0][0];
    let y1 = (v[1] - l[1][0] * y0) / l[1][1];
    let y2 = (v[2] - l[2][0] * y0 - l[2][1] * y1) / l[2][2];
    [y0, y1, y2]
}

/// Solve L^T x = v, L lower triangular.
fn bwd(l: &Mat3, v: &Vec3) -> Vec3 {
    let x2 = v[2] / l[2][2];
    let x1 = (v[1] - l[2][1] * x2) / l[1][1];
    let x0 = (v[0] - l[1][0] * x1 - l[2][0] * x2) / l[0][0];
    [x0, x1, x2]
}

// --- sparse block system ---------------------------------------------------------------------------------

/// Block lower-triangular sparse matrix in elimination order: per column k, its diagonal block and its
/// off-diagonal blocks (row > k, sorted by row). Holds the lower half of H before factorisation, L after.
#[derive(Clone, Debug, Default)]
struct BlockMatrix {
    diag: Vec<Mat3>,
    cols: Vec<Vec<(usize, Mat3)>>,
}

impl BlockMatrix {
    fn zero(&mut self) {
        self.diag.iter_mut().for_each(|d| *d = [[0.0; 3]; 3]);
        self.cols
            .iter_mut()
            .flatten()
            .for_each(|(_, b)| *b = [[0.0; 3]; 3]);
    }

    /// In-place right-looking block Cholesky: on success the matrix holds L with H = L L^T. The pattern must
    /// be closed under elimination (it is: it comes from the elimination graph): for two rows r > s of
    /// column k, block (r, s) exists.
    fn factorize(&mut self) -> bool {
        for k in 0..self.diag.len() {
            let Some(l) = chol3(&self.diag[k]) else {
                return false;
            };
            self.diag[k] = l;
            let (left, right) = self.cols.split_at_mut(k + 1);
            let ck = &mut left[k];
            // L_rk = H_rk L_kk^-T: each row x of the block solves L_kk x^T = (that row of H_rk)^T
            for (_, b) in ck.iter_mut() {
                *b = std::array::from_fn(|r| fwd(&l, &b[r]));
            }
            // Schur complement: H_rs -= L_rk L_sk^T for all rows r >= s of column k
            for (bi, &(s, ls)) in ck.iter().enumerate() {
                sub_assign(&mut self.diag[s], &mul_nt(&ls, &ls));
                let cs = &mut right[s - k - 1];
                let mut p = 0;
                for &(r, lr) in &ck[bi + 1..] {
                    // both lists are sorted: walk column s forward to row r
                    while cs[p].0 != r {
                        p += 1;
                    }
                    sub_assign(&mut cs[p].1, &mul_nt(&lr, &ls));
                }
            }
        }
        true
    }

    /// Solve L L^T x = rhs in place, after factorize().
    fn solve(&self, x: &mut [Vec3]) {
        for k in 0..self.diag.len() {
            let y = fwd(&self.diag[k], &x[k]);
            x[k] = y;
            for (r, l) in &self.cols[k] {
                let ly = mat_vec(l, &y);
                (0..3).for_each(|d| x[*r][d] -= ly[d]);
            }
        }
        for k in (0..self.diag.len()).rev() {
            let mut v = x[k];
            for (r, l) in &self.cols[k] {
                let ltx = mat_t_vec(l, &x[*r]);
                (0..3).for_each(|d| v[d] -= ltx[d]);
            }
            x[k] = bwd(&self.diag[k], &v);
        }
    }
}

/// Where an edge's off-diagonal block H_ij goes: column, index in that column, and whether the stored block
/// is H_ij^T (when node i comes first in the elimination order, the stored block is (row j, col i)).
#[derive(Clone, Copy, Debug)]
struct Slot {
    col: usize,
    idx: usize,
    transposed: bool,
}

/// The normal equations of a graph: structure (ordering, sparsity) and the current H, g.
struct System {
    /// Per node, its position in the elimination order ([`FIXED`] for fixed nodes).
    pos: Vec<usize>,
    /// Per edge, the slot of its off-diagonal block (None if one of its nodes is fixed).
    slots: Vec<Option<Slot>>,
    h: BlockMatrix,
    g: Vec<Vec3>,
    /// Marquardt scaling: diag(H), floored.
    damp: Vec<Vec3>,
    /// H + lambda D, factorised in place.
    work: BlockMatrix,
}

impl System {
    fn n(&self) -> usize {
        self.g.len()
    }

    /// Ordering and symbolic factorisation. Minimum degree on the graph of the free nodes: eliminating a
    /// node joins all its remaining neighbours (they are the rows of its column of L, fill included), so the
    /// elimination graph gives the structure of L directly. Ties go to the lowest index (deterministic).
    fn new(graph: &PoseGraph) -> Self {
        let mut local = vec![FIXED; graph.len()]; // node -> index among the free nodes
        let mut nf = 0;
        for (k, l) in local.iter_mut().enumerate() {
            if !graph.fixed[k] {
                *l = nf;
                nf += 1;
            }
        }
        let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); nf];
        for e in &graph.edges {
            let (a, b) = (local[e.i], local[e.j]);
            if a != FIXED && b != FIXED {
                adj[a].insert(b);
                adj[b].insert(a);
            }
        }
        let mut heap: BinaryHeap<Reverse<(usize, usize)>> =
            (0..nf).map(|v| Reverse((adj[v].len(), v))).collect();
        let mut done = vec![false; nf];
        let mut order = Vec::with_capacity(nf); // free index of the k-th eliminated node
        let mut structure = Vec::with_capacity(nf); // its neighbours when eliminated (free indices)
        while let Some(Reverse((deg, v))) = heap.pop() {
            if done[v] || deg != adj[v].len() {
                continue; // stale heap entry
            }
            done[v] = true;
            let nbrs = std::mem::take(&mut adj[v]);
            for &a in &nbrs {
                adj[a].remove(&v);
                adj[a].extend(nbrs.iter().copied().filter(|&b| b != a));
                heap.push(Reverse((adj[a].len(), a)));
            }
            order.push(v);
            structure.push(nbrs);
        }
        let mut perm = vec![0; nf]; // free index -> position
        for (p, &v) in order.iter().enumerate() {
            perm[v] = p;
        }
        let cols: Vec<Vec<(usize, Mat3)>> = structure
            .iter()
            .map(|s| {
                let mut rows: Vec<usize> = s.iter().map(|&v| perm[v]).collect();
                rows.sort_unstable();
                rows.into_iter().map(|r| (r, [[0.0; 3]; 3])).collect()
            })
            .collect();
        let pos: Vec<usize> = local
            .iter()
            .map(|&l| if l == FIXED { FIXED } else { perm[l] })
            .collect();
        let slots = graph
            .edges
            .iter()
            .map(|e| {
                let (pi, pj) = (pos[e.i], pos[e.j]);
                if pi == FIXED || pj == FIXED {
                    return None;
                }
                let (col, row) = (pi.min(pj), pi.max(pj));
                // always found: i and j are adjacent, so the first eliminated had the other as a neighbour
                let idx = cols[col].binary_search_by_key(&row, |&(r, _)| r).ok()?;
                Some(Slot {
                    col,
                    idx,
                    transposed: pi < pj,
                })
            })
            .collect();
        let h = BlockMatrix {
            diag: vec![[[0.0; 3]; 3]; nf],
            cols,
        };
        Self {
            pos,
            slots,
            work: h.clone(),
            h,
            g: vec![[0.0; 3]; nf],
            damp: vec![[0.0; 3]; nf],
        }
    }

    /// H = sum J^T W J and g = sum J^T W e at the graph's current estimate, W = s^2 info (s = 1 for plain
    /// edges). Fixed nodes have no rows: their Jacobian blocks are dropped.
    fn linearize(&mut self, graph: &PoseGraph) {
        self.h.zero();
        self.g.iter_mut().for_each(|g| *g = [0.0; 3]);
        let phi = graph.params.dcs_phi;
        for (e, slot) in graph.edges.iter().zip(&self.slots) {
            let (err, a, b) = linearize_edge(&graph.poses[e.i], &graph.poses[e.j], &e.z);
            let mut w = e.info;
            if e.robust {
                let s = dcs(phi, chi2(&e.info, &err)).0;
                w.iter_mut().flatten().for_each(|v| *v *= s * s);
            }
            let we = mat_vec(&w, &err);
            let (wa, wb) = (mul(&w, &a), mul(&w, &b));
            let (pi, pj) = (self.pos[e.i], self.pos[e.j]);
            if pi != FIXED {
                add_assign(&mut self.h.diag[pi], &mul_tn(&a, &wa));
                let gi = mat_t_vec(&a, &we);
                (0..3).for_each(|d| self.g[pi][d] += gi[d]);
            }
            if pj != FIXED {
                add_assign(&mut self.h.diag[pj], &mul_tn(&b, &wb));
                let gj = mat_t_vec(&b, &we);
                (0..3).for_each(|d| self.g[pj][d] += gj[d]);
            }
            if let Some(s) = slot {
                let hij = mul_tn(&a, &wb); // block (row i, col j)
                let blk = if s.transposed { transpose(&hij) } else { hij };
                add_assign(&mut self.h.cols[s.col][s.idx].1, &blk);
            }
        }
        for (d, h) in self.damp.iter_mut().zip(&self.h.diag) {
            *d = std::array::from_fn(|k| h[k][k].max(DIAG_FLOOR));
        }
    }

    /// Step h solving (H + lambda D) h = -g; false if the damped matrix is not positive definite.
    fn solve_damped(&mut self, lambda: f64, step: &mut [Vec3]) -> bool {
        self.work.clone_from(&self.h);
        for (blk, d) in self.work.diag.iter_mut().zip(&self.damp) {
            (0..3).for_each(|k| blk[k][k] += lambda * d[k]);
        }
        if !self.work.factorize() {
            return false;
        }
        for (s, g) in step.iter_mut().zip(&self.g) {
            *s = [-g[0], -g[1], -g[2]];
        }
        self.work.solve(step);
        step.iter().flatten().all(|v| v.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// xorshift64* with Box-Muller normals: deterministic noise without a dependency.
    struct Rng(u64);

    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let u = self.uniform().max(1e-300);
            (-2.0 * u.ln()).sqrt() * (2.0 * PI * self.uniform()).cos()
        }
    }

    fn compose(a: &Pose2, b: &Pose2) -> Pose2 {
        let p = a.to_fixed([b.x, b.y]);
        Pose2::new(p[0], p[1], wrap_angle(a.theta + b.theta))
    }

    /// Pose of b in a's frame.
    fn between(a: &Pose2, b: &Pose2) -> Pose2 {
        let p = a.to_robot([b.x, b.y]);
        Pose2::new(p[0], p[1], wrap_angle(b.theta - a.theta))
    }

    fn diag(sxy: f64, sth: f64) -> Mat3 {
        let (a, b) = (1.0 / (sxy * sxy), 1.0 / (sth * sth));
        [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, b]]
    }

    fn noisy(rng: &mut Rng, z: &Pose2, sxy: f64, sth: f64) -> Pose2 {
        Pose2::new(
            z.x + sxy * rng.normal(),
            z.y + sxy * rng.normal(),
            z.theta + sth * rng.normal(),
        )
    }

    /// DCS phi of the tests with loops: the default (1) rejects true loops that correct more drift than a
    /// few sigmas of their own measurement (see small_dcs_phi_rejects_a_loop_that_corrects_much_drift).
    const PHI: f64 = 10.0;

    fn params() -> PoseGraphParams {
        PoseGraphParams { dcs_phi: PHI }
    }

    /// (RMS position error, max heading error) vs ground truth.
    fn ate(est: &[Pose2], gt: &[Pose2]) -> (f64, f64) {
        let se: f64 = est
            .iter()
            .zip(gt)
            .map(|(a, b)| (a.x - b.x).powi(2) + (a.y - b.y).powi(2))
            .sum();
        let th = est
            .iter()
            .zip(gt)
            .map(|(a, b)| wrap_angle(a.theta - b.theta).abs())
            .fold(0.0, f64::max);
        ((se / est.len() as f64).sqrt(), th)
    }

    /// Square of 4 x `side` unit steps, turning left at the corners: (ground truth, graph of the noisy
    /// odometry chained as the initial estimate, without a loop edge).
    fn square(rng: &mut Rng, side: usize, sxy: f64, sth: f64) -> (Vec<Pose2>, PoseGraph) {
        let mut gt = vec![Pose2::default()];
        let mut g = PoseGraph::new(params()).unwrap();
        g.add_node(gt[0]).unwrap();
        let mut est = gt[0];
        for k in 1..4 * side {
            let turn = if k % side == 0 { PI / 2.0 } else { 0.0 };
            let z = Pose2::new(1.0, 0.0, turn); // 1 m forward, then turn
            gt.push(compose(gt.last().unwrap(), &z));
            let zn = noisy(rng, &z, sxy, sth);
            est = compose(&est, &zn);
            let j = g.add_node(est).unwrap();
            g.add_edge(j - 1, j, zn, diag(sxy, sth), false).unwrap();
        }
        (gt, g)
    }

    #[test]
    fn small_dcs_phi_rejects_a_loop_that_corrects_much_drift() {
        // DCS is a redescending kernel: a true loop whose residual at the initial (odometry) estimate is
        // large w.r.t. its own information (chi2 ~ 380 here: 0.4 m of drift, sigma 0.05 m) is a local
        // minimum "rejected" at phi = 1, and accepted at phi = 10 (the value of the other tests).
        let (sxy, sth) = (0.05, 0.03);
        let mut ates = vec![];
        for phi in [1.0, PHI] {
            let mut rng = Rng(42);
            let (gt, mut g) = square(&mut rng, 5, sxy, sth);
            g.params.dcs_phi = phi;
            let n = g.len();
            let z = noisy(&mut rng, &between(&gt[n - 1], &gt[0]), sxy, sth);
            g.add_edge(n - 1, 0, z, diag(sxy, sth), true).unwrap();
            assert!(g.edge_chi2()[n - 1] > 300.0);
            g.optimize(100);
            ates.push((g.edge_scales()[n - 1], ate(g.poses(), &gt).0));
        }
        println!("phi 1: (loop scale, ATE) {:?}; phi {PHI}: {:?}", ates[0], ates[1]);
        assert!(ates[0].0 < 0.1 && ates[1].0 > 0.9 && ates[1].1 < 0.5 * ates[0].1);
    }

    #[test]
    fn jacobians_match_finite_differences() {
        let (xi, xj) = (Pose2::new(0.3, -1.2, 2.9), Pose2::new(1.1, 0.4, -2.8));
        let z = Pose2::new(0.5, 0.2, 0.4);
        let (_, a, b) = linearize_edge(&xi, &xj, &z);
        let h = 1e-6;
        for c in 0..3 {
            let bump = |p: &Pose2, s: f64| {
                let mut v = [p.x, p.y, p.theta];
                v[c] += s;
                Pose2::new(v[0], v[1], v[2])
            };
            let num = |f: &dyn Fn(f64) -> Vec3| {
                let (p, m) = (f(h), f(-h));
                [0, 1, 2].map(|r| wrap_angle(p[r] - m[r]) / (2.0 * h))
            };
            let da = num(&|s| error(&bump(&xi, s), &xj, &z));
            let db = num(&|s| error(&xi, &bump(&xj, s), &z));
            for r in 0..3 {
                assert!(
                    (da[r] - a[r][c]).abs() < 1e-6,
                    "A[{r}][{c}]: {} vs {}",
                    da[r],
                    a[r][c]
                );
                assert!(
                    (db[r] - b[r][c]).abs() < 1e-6,
                    "B[{r}][{c}]: {} vs {}",
                    db[r],
                    b[r][c]
                );
            }
        }
    }

    #[test]
    fn sparse_solve_matches_a_dense_solve() {
        // random graph: a chain plus random chords, all nodes free but node 0
        let mut rng = Rng(7);
        let mut g = PoseGraph::new(PoseGraphParams::default()).unwrap();
        let n = 30;
        for _ in 0..n {
            let p = Pose2::new(3.0 * rng.normal(), 3.0 * rng.normal(), rng.normal());
            g.add_node(p).unwrap();
        }
        let add = |g: &mut PoseGraph, i: usize, j: usize, rng: &mut Rng| {
            let z = Pose2::new(rng.normal(), rng.normal(), rng.normal());
            let info = diag(0.1 + rng.uniform(), 0.05 + rng.uniform());
            g.add_edge(i, j, z, info, false).unwrap();
        };
        for k in 1..n {
            add(&mut g, k - 1, k, &mut rng);
        }
        for _ in 0..15 {
            let i = (rng.uniform() * n as f64) as usize;
            let j = (rng.uniform() * n as f64) as usize;
            if i != j {
                add(&mut g, j, i, &mut rng);
            }
        }
        let mut sys = System::new(&g);
        sys.linearize(&g);
        let m = 3 * sys.n();
        let lambda = 1e-3;
        // dense [H + lambda D | -g] in elimination order, from the block structure
        let mut a = vec![vec![0.0; m + 1]; m];
        for k in 0..sys.n() {
            for r in 0..3 {
                for c in 0..3 {
                    a[3 * k + r][3 * k + c] = sys.h.diag[k][r][c];
                }
                a[3 * k + r][3 * k + r] += lambda * sys.damp[k][r];
                a[3 * k + r][m] = -sys.g[k][r];
            }
            for &(row, blk) in &sys.h.cols[k] {
                for r in 0..3 {
                    for c in 0..3 {
                        a[3 * row + r][3 * k + c] = blk[r][c];
                        a[3 * k + c][3 * row + r] = blk[r][c];
                    }
                }
            }
        }
        let mut step = vec![[0.0; 3]; sys.n()];
        assert!(sys.solve_damped(lambda, &mut step));
        // Gaussian elimination with partial pivoting
        for c in 0..m {
            let p = (c..m)
                .max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs()))
                .unwrap();
            a.swap(c, p);
            let (top, bottom) = a.split_at_mut(c + 1);
            let pivot = &top[c];
            for row in bottom.iter_mut() {
                let f = row[c] / pivot[c];
                row.iter_mut().zip(pivot).skip(c).for_each(|(v, p)| *v -= f * p);
            }
        }
        let mut x = vec![0.0; m];
        for r in (0..m).rev() {
            x[r] = (a[r][m] - (r + 1..m).map(|k| a[r][k] * x[k]).sum::<f64>()) / a[r][r];
        }
        let scale = x.iter().fold(1.0f64, |s, v| s.max(v.abs()));
        for r in 0..m {
            let s = step[r / 3][r % 3];
            assert!((x[r] - s).abs() < 1e-8 * scale, "{r}: {} vs {s}", x[r]);
        }
    }

    #[test]
    fn consistent_odometry_chain_is_unchanged() {
        let mut rng = Rng(1);
        let (_, mut g) = square(&mut rng, 5, 0.05, 0.02);
        let before = g.poses().to_vec();
        let r = g.optimize(20);
        assert!(
            r.converged && r.chi2_initial < 1e-20 && r.iterations == 0,
            "{r:?}"
        );
        assert_eq!(g.poses(), &before[..]);
    }

    #[test]
    fn noisy_square_loop_converges_toward_ground_truth() {
        let mut rng = Rng(42);
        let (sxy, sth) = (0.05, 0.03);
        let (gt, mut g) = square(&mut rng, 5, sxy, sth);
        // loop closure: the first node seen from the last one
        let n = g.len();
        let z = noisy(&mut rng, &between(&gt[n - 1], &gt[0]), sxy, sth);
        g.add_edge(n - 1, 0, z, diag(sxy, sth), true).unwrap();
        let (odom_xy, odom_th) = ate(g.poses(), &gt);
        let r = g.optimize(50);
        let (opt_xy, opt_th) = ate(g.poses(), &gt);
        println!(
            "square: odometry ATE {odom_xy:.3} m / {odom_th:.3} rad, optimised {opt_xy:.3} m / \
             {opt_th:.3} rad, {r:?}"
        );
        assert!(r.converged && r.chi2_final < 0.1 * r.chi2_initial, "{r:?}");
        assert!(opt_xy < 0.5 * odom_xy && opt_xy < 0.15, "{opt_xy} vs {odom_xy}");
        assert!(opt_th < odom_th);
        assert!(g.edge_scales()[g.n_edges() - 1] > 0.3); // the true loop is kept
    }

    #[test]
    fn wrong_loop_is_down_weighted_and_does_not_distort_the_graph() {
        let (sxy, sth) = (0.05, 0.03);
        let mut rng = Rng(42);
        let (gt, mut good) = square(&mut rng, 5, sxy, sth);
        let n = good.len();
        let z = noisy(&mut rng, &between(&gt[n - 1], &gt[0]), sxy, sth);
        good.add_edge(n - 1, 0, z, diag(sxy, sth), true).unwrap();
        let mut bad = good.clone();
        let mut plain = good.clone();
        // false loop: claims node 12 is 1 m in front of node 2 (they are ~3.6 m apart)
        let zw = Pose2::new(1.0, 0.0, 0.0);
        let wrong = bad.add_edge(2, 12, zw, diag(sxy, sth), true).unwrap();
        good.optimize(50);
        let r = bad.optimize(50);
        let s = bad.edge_scales();
        println!(
            "wrong loop: scale {:.2e}, chi2 {:.1}, true loop scale {:.2}, {r:?}",
            s[wrong],
            bad.edge_chi2()[wrong],
            s[n - 1]
        );
        assert!(s[wrong] < 0.1 && s[n - 1] > 0.3);
        let (d, _) = ate(bad.poses(), good.poses());
        assert!(d < 0.02, "the false loop moved the graph by {d} m RMS");
        // the same false loop as a plain edge distorts it
        plain.add_edge(2, 12, zw, diag(sxy, sth), false).unwrap();
        plain.optimize(50);
        assert!(ate(plain.poses(), &gt).0 > 0.3);
    }

    #[test]
    fn headings_across_pi_are_wrapped() {
        // node 0 fixed just below +pi; node 1 initialised just past +pi (stored as ~-pi); the edge turns +0.1
        let mut g = PoseGraph::new(PoseGraphParams::default()).unwrap();
        g.add_node(Pose2::new(0.0, 0.0, PI - 0.05)).unwrap();
        g.add_node(Pose2::new(-0.8, 0.1, -PI + 0.3)).unwrap();
        g.add_edge(0, 1, Pose2::new(1.0, 0.0, 0.1), diag(0.1, 0.1), false)
            .unwrap();
        let r = g.optimize(20);
        let p = g.pose(1).unwrap();
        assert!(r.converged && r.chi2_final < 1e-12, "{r:?}");
        assert!(wrap_angle(p.theta - (PI + 0.05)).abs() < 1e-6 && (-PI..PI).contains(&p.theta));
        // 1 m forward of node 0
        let want = Pose2::new(0.0, 0.0, PI - 0.05).to_fixed([1.0, 0.0]);
        assert!(
            (p.x - want[0]).abs() < 1e-6 && (p.y - want[1]).abs() < 1e-6,
            "{p:?}"
        );
    }

    #[test]
    fn graph_without_fixed_node_keeps_its_shape() {
        // no gauge: H is singular, the damping keeps the system solvable and the relative poses converge
        let mut rng = Rng(3);
        let (sxy, sth) = (0.05, 0.03);
        let (gt, mut g) = square(&mut rng, 4, sxy, sth);
        let n = g.len();
        let z = noisy(&mut rng, &between(&gt[n - 1], &gt[0]), sxy, sth);
        g.add_edge(n - 1, 0, z, diag(sxy, sth), false).unwrap();
        g.set_fixed(0, false).unwrap();
        let r = g.optimize(100);
        assert!(r.chi2_final < 0.1 * r.chi2_initial, "{r:?}");
        let c = g.edge_chi2();
        assert!(c.iter().all(|&c| c < 20.0), "{c:?}");
    }

    #[test]
    fn invalid_inputs_are_errors() {
        assert!(PoseGraph::new(PoseGraphParams { dcs_phi: 0.0 }).is_err());
        let mut g = PoseGraph::new(PoseGraphParams::default()).unwrap();
        assert!(g.add_node(Pose2::new(f64::NAN, 0.0, 0.0)).is_err());
        g.add_node(Pose2::default()).unwrap();
        g.add_node(Pose2::default()).unwrap();
        let eye = diag(1.0, 1.0);
        let z = Pose2::new(1.0, 0.0, 0.0);
        assert!(g.add_edge(0, 2, z, eye, false).is_err());
        assert!(g.add_edge(1, 1, z, eye, false).is_err());
        let zinf = Pose2::new(0.0, f64::INFINITY, 0.0);
        assert!(g.add_edge(0, 1, zinf, eye, false).is_err());
        let asym = [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(g.add_edge(0, 1, z, asym, false).is_err());
        let indef = [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(g.add_edge(0, 1, z, indef, false).is_err());
        let off = [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; // positive diagonal, det < 0
        assert!(g.add_edge(0, 1, z, off, false).is_err());
        let mut nan = eye;
        nan[2][2] = f64::NAN;
        assert!(g.add_edge(0, 1, z, nan, false).is_err());
        assert!(g.set_fixed(5, true).is_err() && g.pose(5).is_err() && g.set_pose(5, z).is_err());
        assert!(g.set_pose(1, Pose2::new(0.0, 0.0, f64::NAN)).is_err());
        assert_eq!(g.n_edges(), 0);
        // PSD but singular (no heading information) is accepted
        let psd = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        assert_eq!(g.add_edge(0, 1, z, psd, false), Ok(0));
        assert!(g.is_fixed(0).unwrap() && !g.is_fixed(1).unwrap());
    }

    #[test]
    fn manhattan_world_converges() {
        // M3500-like: unit steps on a 13 x 13 grid, random left / right turns; revisits give loop closures
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (sxy, sth) = (0.05, 0.01);
        let (n, half): (usize, i64) = (1000, 6);
        let mut cell = (0i64, 0i64);
        let mut heading = 0i64; // quarter turns
        let mut gt = vec![Pose2::default()];
        let mut cells = vec![cell];
        let mut g = PoseGraph::new(params()).unwrap();
        g.add_node(gt[0]).unwrap();
        let mut est = gt[0];
        let dir = |h: i64| [(1, 0), (0, 1), (-1, 0), (0, -1)][h.rem_euclid(4) as usize];
        let outside = |c: (i64, i64), h: i64| {
            let (dx, dy) = dir(h);
            (c.0 + dx).abs() > half || (c.1 + dy).abs() > half
        };
        for _ in 1..n {
            let u = rng.uniform();
            let mut turn = if u < 0.15 {
                1
            } else if u < 0.3 {
                -1
            } else {
                0
            };
            while outside(cell, heading + turn) {
                turn += 1; // stay in the grid: turn left until the next cell is inside
            }
            heading += turn;
            let (dx, dy) = dir(heading);
            cell = (cell.0 + dx, cell.1 + dy);
            let th = wrap_angle(heading as f64 * PI / 2.0);
            let next = Pose2::new(cell.0 as f64, cell.1 as f64, th);
            let z = between(gt.last().unwrap(), &next);
            gt.push(next);
            cells.push(cell);
            let zn = noisy(&mut rng, &z, sxy, sth);
            est = compose(&est, &zn);
            let j = g.add_node(est).unwrap();
            g.add_edge(j - 1, j, zn, diag(sxy, sth), false).unwrap();
        }
        // loop candidates: a node in a cell visited more than 10 nodes earlier, closing alternately to the
        // oldest and to the most recent earlier visit; 100 of them spread along the trajectory
        let cands: Vec<(usize, usize)> = (0..n)
            .filter_map(|k| {
                let earlier: Vec<usize> = (0..k.saturating_sub(10))
                    .filter(|&m| cells[m] == cells[k])
                    .collect();
                let m = if k % 2 == 0 {
                    earlier.first()
                } else {
                    earlier.last()
                };
                m.map(|&m| (m, k))
            })
            .collect();
        let n_loops = 100.min(cands.len());
        for q in 0..n_loops {
            let (m, k) = cands[q * cands.len() / n_loops];
            let z = noisy(&mut rng, &between(&gt[m], &gt[k]), sxy, sth);
            g.add_edge(m, k, z, diag(sxy, sth), true).unwrap();
        }
        let (odom_xy, odom_th) = ate(g.poses(), &gt);
        let t0 = std::time::Instant::now();
        let r = g.optimize(100);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let (opt_xy, opt_th) = ate(g.poses(), &gt);
        let min_loop_scale = g.edge_scales()[n - 1..].iter().copied().fold(1.0, f64::min);
        println!(
            "manhattan: {} nodes, {} edges ({n_loops} loops), optimize {ms:.1} ms; odometry ATE \
             {odom_xy:.3} m / {odom_th:.3} rad, optimised {opt_xy:.3} m / {opt_th:.3} rad; min loop \
             scale {min_loop_scale:.2}; {r:?}",
            g.len(),
            g.n_edges(),
        );
        assert_eq!(n_loops, 100);
        // at the global optimum, chi2 ~ the degrees of freedom: 3 (edges - free nodes) = 300 (all loops are
        // inliers with chi2 < phi, so rho = chi2); far above, LM would have stopped in a wrong local minimum
        let dof = 3.0 * (g.n_edges() - (g.len() - 1)) as f64;
        assert!(r.converged && r.chi2_final < 1.3 * dof, "{r:?}");
        assert!(opt_xy < 0.25 * odom_xy, "{opt_xy} vs {odom_xy}");
        assert!(opt_th < 0.5 * odom_th);
        assert!(min_loop_scale > 0.9, "a true loop was down-weighted");
    }
}
