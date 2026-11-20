//! Bindings of navette_core::slam::scan_matcher: `LikelihoodGrid` and `match_points`.
//! Points are (N, 2) float64 arrays (metres), a pose is an (x, y, theta) sequence.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use navette_core::geometry::{Point2, Pose2};
use navette_core::slam::scan_matcher::{self, LikelihoodGrid, LikelihoodGridParams, MatchParams};

use crate::convert::{param_err, rows};

/// Likelihood lookup table of reference obstacle points (N, 2) in the fixed frame (see the core module):
/// value exp(-d^2 / (2 sigma^2)) per cell, d the distance to the nearest point (0 beyond 3 sigma).
///   res     cell size (m), also the translation step of match_points
///   sigma   expected point error (m)
///   margin  space around the points' bounding box (m); >= 3 sigma loses nothing
///   coarse  block size (cells) of the low-resolution bound table
///   observed  optional (M, 2) points the reference has observed (e.g. the centres of its free and occupied
///           occupancy cells): their cells, dilated by ceil(sigma / res) + 1, form the observed layer, and
///           match_points then scores only the query points in observed cells (see the core module)
/// ValueError on an invalid parameter, no finite point (or observed point), or more than 4096 cells per side.
#[pyclass(name = "LikelihoodGrid", module = "navette", frozen)]
pub struct PyLikelihoodGrid {
    inner: LikelihoodGrid,
}

#[pymethods]
impl PyLikelihoodGrid {
    #[new]
    #[pyo3(signature = (points, res=0.05, sigma=0.05, margin=0.15, coarse=4, observed=None))]
    fn new(
        py: Python<'_>,
        points: PyReadonlyArray2<'_, f64>,
        res: f64,
        sigma: f64,
        margin: f64,
        coarse: usize,
        observed: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<Self> {
        let pts: Vec<Point2> = rows(&points, "points")?;
        let obs: Option<Vec<Point2>> = observed.map(|o| rows(&o, "observed")).transpose()?;
        let params = LikelihoodGridParams {
            res,
            sigma,
            margin,
            coarse,
        };
        // without the GIL (see lib.rs): a whole map's points take a few ms
        let inner = py
            .detach(|| match &obs {
                None => LikelihoodGrid::new(&pts, params),
                Some(o) => LikelihoodGrid::new_with_observed(&pts, o, params),
            })
            .map_err(param_err)?;
        Ok(Self { inner })
    }

    /// Cell size (m).
    #[getter]
    fn res(&self) -> f64 {
        self.inner.params().res
    }

    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.params().sigma
    }

    /// Fixed-frame x of the corner of cell [0, 0].
    #[getter]
    fn x0(&self) -> f64 {
        self.inner.origin()[0]
    }

    /// Fixed-frame y of the corner of cell [0, 0].
    #[getter]
    fn y0(&self) -> f64 {
        self.inner.origin()[1]
    }

    /// Likelihood (nx, ny) float32, a copy: [i, j] = cell at x = x0 + i res, y = y0 + j res.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (nx, ny) = self.inner.shape();
        PyArray1::from_slice(py, self.inner.values()).reshape([nx, ny])
    }

    /// Observed layer (nx, ny) bool, a copy (same cells as values); None without it.
    #[getter]
    fn observed<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<bool>>>> {
        let (nx, ny) = self.inner.shape();
        self.inner
            .observed()
            .map(|m| PyArray1::from_slice(py, m).reshape([nx, ny]))
            .transpose()
    }
}

/// Best pose of the points (N, 2) (query frame, e.g. a submap's own) in the grid's frame, searched within
/// guess +- (dx, dy, dtheta) (m, m, rad) on the lattice (res, res, theta_step). theta_step None: res /
/// range of the farthest point. score_gain: k in the candidate weight exp(k (score - best)) of the
/// covariance. Non-finite points are ignored. min_observed_frac: with an observed layer, the score floor
/// k = ceil(min_observed_frac n): the score is the sum over the points in observed cells / max(their
/// number, k).
/// Returns a dict: pose (x, y, theta), score (mean likelihood, 0..1; with an observed layer over the points
/// in observed cells, floored as above), n (points used), observed (points in observed cells at the pose; n without the
/// layer), covariance (3, 3)
/// of (x, y, theta), xy_eigenvalues (min, max) of its xy block (m^2), xy_major_axis (rad), theta_step
/// (rad), evaluated (fine candidates). An unconstrained direction (corridor) has a variance near the
/// window's half-width^2 / 3. ValueError on an invalid parameter.
#[pyfunction]
#[pyo3(signature = (grid, points, guess, dx=0.5, dy=0.5, dtheta=0.3490658503988659, theta_step=None,
                    score_gain=30.0, min_observed_frac=0.3))]
#[allow(clippy::too_many_arguments)]
fn match_points<'py>(
    py: Python<'py>,
    grid: &Bound<'py, PyLikelihoodGrid>,
    points: PyReadonlyArray2<'py, f64>,
    guess: [f64; 3],
    dx: f64,
    dy: f64,
    dtheta: f64,
    theta_step: Option<f64>,
    score_gain: f64,
    min_observed_frac: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let pts: Vec<Point2> = rows(&points, "points")?;
    let params = MatchParams {
        dx,
        dy,
        dtheta,
        theta_step,
        score_gain,
        min_observed_frac,
    };
    // frozen class: the grid is shared read-only, no borrow to hold across the GIL release
    let inner = &grid.get().inner;
    let guess = Pose2::new(guess[0], guess[1], guess[2]);
    // without the GIL (see lib.rs)
    let r = py
        .detach(|| scan_matcher::match_points(inner, &pts, guess, &params))
        .map_err(param_err)?;
    let d = PyDict::new(py);
    d.set_item("pose", (r.pose.x, r.pose.y, r.pose.theta))?;
    d.set_item("score", r.score)?;
    d.set_item("n", r.n)?;
    d.set_item("observed", r.observed)?;
    d.set_item(
        "covariance",
        PyArray1::from_slice(py, r.covariance.as_flattened()).reshape([3, 3])?,
    )?;
    d.set_item("xy_eigenvalues", (r.xy_eigenvalues[0], r.xy_eigenvalues[1]))?;
    d.set_item("xy_major_axis", r.xy_major_axis)?;
    d.set_item("theta_step", r.theta_step)?;
    d.set_item("evaluated", r.evaluated)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLikelihoodGrid>()?;
    m.add_function(wrap_pyfunction!(match_points, m)?)?;
    Ok(())
}
