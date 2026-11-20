//! Bindings of navette_core::slam::pose_graph: `PoseGraph` and its `OptimizeReport`.
//! A pose is an (x, y, theta) sequence; an information matrix a 3x3 nested sequence or numpy array.

use pyo3::prelude::*;

use navette_core::geometry::Pose2;
use navette_core::slam::pose_graph::{OptimizeReport, PoseGraph, PoseGraphParams};

use crate::convert::param_err;

fn pose2(p: [f64; 3]) -> Pose2 {
    Pose2::new(p[0], p[1], p[2])
}

fn tuple(p: Pose2) -> (f64, f64, f64) {
    (p.x, p.y, p.theta)
}

/// Outcome of PoseGraph.optimize(): iterations (linear solves, accepted and rejected steps), chi2_initial and
/// chi2_final (the robust objective: sum of chi2 over plain edges plus the DCS-bounded cost of robust ones),
/// converged (the objective stopped decreasing within max_iters).
#[pyclass(name = "OptimizeReport", module = "navette", frozen, get_all)]
pub struct PyOptimizeReport {
    iterations: usize,
    chi2_initial: f64,
    chi2_final: f64,
    converged: bool,
}

#[pymethods]
impl PyOptimizeReport {
    fn __repr__(&self) -> String {
        format!(
            "OptimizeReport(iterations={}, chi2_initial={}, chi2_final={}, converged={})",
            self.iterations,
            self.chi2_initial,
            self.chi2_final,
            if self.converged { "True" } else { "False" }
        )
    }
}

impl From<OptimizeReport> for PyOptimizeReport {
    fn from(r: OptimizeReport) -> Self {
        Self {
            iterations: r.iterations,
            chi2_initial: r.chi2_initial,
            chi2_final: r.chi2_final,
            converged: r.converged,
        }
    }
}

/// SE(2) pose graph optimised by Levenberg-Marquardt (see the core module's documentation).
///   dcs_phi   Dynamic Covariance Scaling of the robust edges: an edge with chi2 > dcs_phi has its
///             information scaled by s^2, s = 2 dcs_phi / (dcs_phi + chi2)
/// Nodes are map-frame poses; the first one is fixed (the gauge). An edge (i, j, z, info) says node j is at z
/// in node i's frame. ValueError on an invalid parameter, index or value.
#[pyclass(name = "PoseGraph", module = "navette")]
pub struct PyPoseGraph {
    inner: PoseGraph,
}

#[pymethods]
impl PyPoseGraph {
    #[new]
    #[pyo3(signature = (dcs_phi=1.0))]
    fn new(dcs_phi: f64) -> PyResult<Self> {
        Ok(Self {
            inner: PoseGraph::new(PoseGraphParams { dcs_phi }).map_err(param_err)?,
        })
    }

    #[getter]
    fn dcs_phi(&self) -> f64 {
        self.inner.params().dcs_phi
    }

    /// Add a node with its initial estimate (x, y, theta); returns its index.
    fn add_node(&mut self, pose: [f64; 3]) -> PyResult<usize> {
        self.inner.add_node(pose2(pose)).map_err(param_err)
    }

    /// Add an edge: node j is at z = (x, y, theta) in node i's frame, with the 3x3 information matrix info
    /// (symmetric positive semi-definite, order x, y, theta). robust: down-weighted by DCS when it disagrees
    /// with the graph (loop closures). Returns its index.
    #[pyo3(signature = (i, j, z, info, robust=false))]
    fn add_edge(
        &mut self,
        i: usize,
        j: usize,
        z: [f64; 3],
        info: [[f64; 3]; 3],
        robust: bool,
    ) -> PyResult<usize> {
        self.inner
            .add_edge(i, j, pose2(z), info, robust)
            .map_err(param_err)
    }

    /// Fix node i (not moved by optimize) or free it.
    #[pyo3(signature = (i, fixed=true))]
    fn set_fixed(&mut self, i: usize, fixed: bool) -> PyResult<()> {
        self.inner.set_fixed(i, fixed).map_err(param_err)
    }

    fn is_fixed(&self, i: usize) -> PyResult<bool> {
        self.inner.is_fixed(i).map_err(param_err)
    }

    /// Run Levenberg-Marquardt (at most max_iters linear solves) from the current estimates.
    #[pyo3(signature = (max_iters=20))]
    fn optimize(&mut self, py: Python<'_>, max_iters: usize) -> PyOptimizeReport {
        let inner = &mut self.inner;
        py.detach(|| inner.optimize(max_iters)).into() // without the GIL (see lib.rs)
    }

    /// Estimate (x, y, theta) of node i.
    fn pose(&self, i: usize) -> PyResult<(f64, f64, f64)> {
        self.inner.pose(i).map(tuple).map_err(param_err)
    }

    /// Replace the estimate of node i (re-initialisation).
    fn set_pose(&mut self, i: usize, pose: [f64; 3]) -> PyResult<()> {
        self.inner.set_pose(i, pose2(pose)).map_err(param_err)
    }

    /// Estimates (x, y, theta) of all nodes, by index.
    fn poses(&self) -> Vec<(f64, f64, f64)> {
        self.inner.poses().iter().map(|&p| tuple(p)).collect()
    }

    /// Per edge, its DCS scale at the current estimates (1.0 for non-robust edges); a robust edge with a
    /// tiny scale is one the graph rejects.
    fn edge_scales(&self) -> Vec<f64> {
        self.inner.edge_scales()
    }

    /// Per edge, its unscaled chi2 = e^T info e at the current estimates.
    fn edge_chi2(&self) -> Vec<f64> {
        self.inner.edge_chi2()
    }

    fn n_edges(&self) -> usize {
        self.inner.n_edges()
    }

    /// Number of nodes.
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPoseGraph>()?;
    m.add_class::<PyOptimizeReport>()?;
    Ok(())
}
