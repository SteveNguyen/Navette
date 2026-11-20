//! Correlative scan matcher (Olson, "Real-Time Correlative Scan Matching", ICRA 2009), multi-resolution.
//!
//! Use (docs/SLAM.md): a submap's obstacle points (its own frame) are matched against reference obstacle
//! points (the map, or an earlier submap, in the fixed frame) by an exhaustive search of the poses within the
//! odometry uncertainty around a guess. Exhaustive, not a local optimiser (ICP): the result is the global
//! best in the window, and the scores of the candidates near it give a covariance that shows how well each
//! direction is constrained (a corridor constrains across, not along).
//!
//! Reference: a [`LikelihoodGrid`], value exp(-d^2 / (2 sigma^2)) per cell, d the distance from the cell
//! centre to the nearest reference point (0 beyond 3 sigma). Cell [i, j] covers x in [x0 + i res,
//! x0 + (i + 1) res), y in [y0 + j res, y0 + (j + 1) res), row-major index i * ny + j (as occupancy_grid.rs).
//!
//! Search ([`match_points`]): poses guess + (tx res, ty res, theta offset), tx, ty integers within the window,
//! theta offsets on a lattice whose step moves the farthest point by about one cell. For one heading, the
//! cell of every point is computed once; a translation by whole cells shifts all of them by (tx, ty), so a
//! candidate costs one table lookup per point. Score = mean value over the points, in [0, 1].
//!
//! Multi-resolution (Olson's low-resolution table): a second table holds, per cell, the max of the fine
//! values over the D x D block of cells starting at it. Looked up at offset (bx, by), it bounds from above
//! the fine score of every translation in [bx, bx + D) x [by, by + D) at the same heading. Blocks are
//! visited by decreasing bound and the fine translations of a block are evaluated until the bound falls
//! below best - margin: the best is exact (as a full search), and every candidate within `margin` of it has
//! been evaluated, which is what the covariance needs.
//!
//! Covariance (Olson section IV-B): each candidate within the margin is weighted by w = exp(k (s - s_best))
//! (s the score, k = `score_gain`); the covariance is the w-weighted second moment of (x, y, theta) about the
//! w-weighted mean, plus the variance of the lattice quantisation (res^2 / 12, step^2 / 12). k plays the
//! role of the number of independent points: the exact likelihood of N independent points would be
//! ~exp(N (s - 1)), far too confident since neighbouring points of a wall are not independent. A direction
//! the scene does not constrain gets a variance bounded by the window (uniform: half-width^2 / 3): compare
//! the major axis with the window, not with an absolute threshold.

use crate::geometry::{wrap_angle, Point2, Pose2};
use crate::params::{self, ParamError};

/// Most cells per side of a [`LikelihoodGrid`] (as occupancy_grid.rs: 4096^2 cells, 64 MB per table).
pub const MAX_CELLS: usize = 4096;
/// Most headings searched (bounds the per-heading cell table: headings x points).
pub const MAX_HEADINGS: usize = 20_001;
/// Candidates weighing less than this (relative to the best) are left out of the covariance, so they need
/// not be evaluated: margin = ln(1 / MIN_WEIGHT) / k.
const MIN_WEIGHT: f64 = 1e-3;
/// The likelihood is cut to 0 beyond this many sigmas (exp(-4.5) = 0.011).
const CUTOFF_SIGMAS: f64 = 3.0;

/// Parameters of a [`LikelihoodGrid`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LikelihoodGridParams {
    /// Cell size (m); also the translation step of the search.
    pub res: f64,
    /// Width of the likelihood around a reference point (m): the expected point error.
    pub sigma: f64,
    /// Space added around the reference points' bounding box (m). Outside the grid the value is 0, which is
    /// exact beyond 3 sigma: a margin >= 3 sigma loses nothing.
    pub margin: f64,
    /// Block size D (cells) of the low-resolution table. Larger prunes more per bound but bounds more
    /// loosely.
    pub coarse: usize,
}

impl Default for LikelihoodGridParams {
    fn default() -> Self {
        Self {
            res: 0.05,
            sigma: 0.05,
            margin: 0.15,
            coarse: 4,
        }
    }
}

impl LikelihoodGridParams {
    /// res and sigma finite and > 0, margin finite and >= 0, coarse in 1..=64.
    pub fn validate(&self) -> Result<(), ParamError> {
        params::positive("LikelihoodGridParams.res", self.res)?;
        params::positive("LikelihoodGridParams.sigma", self.sigma)?;
        params::non_negative("LikelihoodGridParams.margin", self.margin)?;
        params::count("LikelihoodGridParams.coarse", self.coarse, 64)
    }
}

/// Likelihood lookup table of reference obstacle points (fixed frame), with its low-resolution bound table.
#[derive(Clone, Debug)]
pub struct LikelihoodGrid {
    params: LikelihoodGridParams,
    /// Fixed-frame position of the corner of cell [0, 0].
    origin: Point2,
    nx: usize,
    ny: usize,
    /// exp(-d^2 / (2 sigma^2)), row-major [i * ny + j].
    fine: Vec<f32>,
    /// Max of `fine` over cells [i, i + D) x [j, j + D) (clipped to the grid), row-major.
    coarse: Vec<f32>,
}

impl LikelihoodGrid {
    /// The table of `points` (fixed frame, m; non-finite ones are ignored). An error for invalid parameters,
    /// no finite point, or a bounding box (plus margins) over [`MAX_CELLS`] cells per side.
    pub fn new(points: &[Point2], params: LikelihoodGridParams) -> Result<Self, ParamError> {
        params.validate()?;
        let pts: Vec<Point2> = points
            .iter()
            .copied()
            .filter(|p| p[0].is_finite() && p[1].is_finite())
            .collect();
        if pts.is_empty() {
            return Err(ParamError("LikelihoodGrid: no finite reference point".into()));
        }
        let (res, sigma, margin) = (params.res, params.sigma, params.margin);
        let lo = |k: usize| pts.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min);
        let hi = |k: usize| pts.iter().map(|p| p[k]).fold(f64::NEG_INFINITY, f64::max);
        let origin = [lo(0) - margin, lo(1) - margin];
        // cells up to and including the one holding max + margin
        let cells = |k: usize| ((hi(k) + margin - origin[k]) / res).floor() + 1.0;
        let (nxf, nyf) = (cells(0), cells(1));
        if nxf > MAX_CELLS as f64 || nyf > MAX_CELLS as f64 {
            return Err(ParamError(format!(
                "LikelihoodGrid: the points span {nxf} x {nyf} cells, at most {MAX_CELLS} per side"
            )));
        }
        let (nx, ny) = (nxf as usize, nyf as usize);

        // Squared distance to the nearest point, brute force over each point's neighbourhood: every cell
        // within the cutoff of a point is within r cells of the point's cell. N points x (2r + 1)^2 cells
        // (49 at sigma = res), cheaper than a full distance transform for sparse obstacle points.
        let cutoff2 = (CUTOFF_SIGMAS * sigma).powi(2);
        let r = (CUTOFF_SIGMAS * sigma / res).ceil() as i64;
        let mut d2 = vec![f64::INFINITY; nx * ny];
        for p in &pts {
            let ci = ((p[0] - origin[0]) / res).floor() as i64;
            let cj = ((p[1] - origin[1]) / res).floor() as i64;
            for i in (ci - r).max(0)..=(ci + r).min(nx as i64 - 1) {
                let dx = origin[0] + (i as f64 + 0.5) * res - p[0];
                for j in (cj - r).max(0)..=(cj + r).min(ny as i64 - 1) {
                    let dy = origin[1] + (j as f64 + 0.5) * res - p[1];
                    let c = &mut d2[i as usize * ny + j as usize];
                    *c = c.min(dx * dx + dy * dy);
                }
            }
        }
        let k = -0.5 / (sigma * sigma);
        let fine: Vec<f32> = d2
            .iter()
            .map(|&d2| if d2 <= cutoff2 { (k * d2).exp() as f32 } else { 0.0 })
            .collect();
        let coarse = block_max(&fine, nx, ny, params.coarse);
        Ok(Self {
            params,
            origin,
            nx,
            ny,
            fine,
            coarse,
        })
    }

    pub fn params(&self) -> &LikelihoodGridParams {
        &self.params
    }

    /// Fixed-frame position (x0, y0) of the corner of cell [0, 0].
    pub fn origin(&self) -> Point2 {
        self.origin
    }

    /// Cells along x and along y.
    pub fn shape(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// Likelihood of every cell, row-major: cell [i, j] at index i * ny + j.
    pub fn values(&self) -> &[f32] {
        &self.fine
    }

    /// Fine value of cell [i, j]; 0 outside the grid.
    #[inline]
    fn fine_at(&self, i: i64, j: i64) -> f32 {
        // a negative index wraps to a huge usize: one comparison per axis
        if (i as usize) < self.nx && (j as usize) < self.ny {
            self.fine[i as usize * self.ny + j as usize]
        } else {
            0.0
        }
    }

    /// Bound of the fine values over [i, i + D) x [j, j + D); 0 if that block is outside the grid. A block
    /// starting before the grid (-D < i < 0) is bounded by the one starting at 0, which covers its part
    /// inside the grid.
    #[inline]
    fn coarse_at(&self, i: i64, j: i64) -> f32 {
        let d = self.params.coarse as i64;
        if i <= -d || j <= -d || i >= self.nx as i64 || j >= self.ny as i64 {
            0.0
        } else {
            self.coarse[i.max(0) as usize * self.ny + j.max(0) as usize]
        }
    }
}

/// out[i, j] = max of v over [i, i + d) x [j, j + d), clipped to the grid: two separable passes.
fn block_max(v: &[f32], nx: usize, ny: usize, d: usize) -> Vec<f32> {
    let mut rows = vec![0.0f32; nx * ny];
    for i in 0..nx {
        for j in 0..ny {
            rows[i * ny + j] = v[i * ny + j..i * ny + (j + d).min(ny)]
                .iter()
                .copied()
                .fold(0.0, f32::max);
        }
    }
    let mut out = vec![0.0f32; nx * ny];
    for i in 0..nx {
        for j in 0..ny {
            out[i * ny + j] = (i..(i + d).min(nx))
                .map(|ii| rows[ii * ny + j])
                .fold(0.0, f32::max);
        }
    }
    out
}

/// Search window and covariance weighting of [`match_points`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchParams {
    /// Half-width of the translation window along fixed x (m), around the guess. Rounded to whole cells:
    /// the translation step is the grid's res.
    pub dx: f64,
    /// Same along fixed y.
    pub dy: f64,
    /// Half-width of the heading window (rad).
    pub dtheta: f64,
    /// Heading step (rad); None: res / (range of the farthest point), so that no point moves more than
    /// about one cell between neighbouring headings. Shrunk so that the lattice ends on +-dtheta.
    pub theta_step: Option<f64>,
    /// k in the candidate weight exp(k (score - best)): the effective number of independent points. Larger
    /// gives a smaller covariance.
    pub score_gain: f64,
}

impl Default for MatchParams {
    fn default() -> Self {
        Self {
            dx: 0.5,
            dy: 0.5,
            dtheta: 20f64.to_radians(),
            theta_step: None,
            score_gain: 30.0,
        }
    }
}

impl MatchParams {
    /// dx, dy, dtheta finite and >= 0; theta_step (if given) and score_gain finite and > 0.
    pub fn validate(&self) -> Result<(), ParamError> {
        params::non_negative("MatchParams.dx", self.dx)?;
        params::non_negative("MatchParams.dy", self.dy)?;
        params::non_negative("MatchParams.dtheta", self.dtheta)?;
        if let Some(s) = self.theta_step {
            params::positive("MatchParams.theta_step", s)?;
        }
        params::positive("MatchParams.score_gain", self.score_gain)
    }
}

/// Result of [`match_points`].
#[derive(Clone, Debug, PartialEq)]
pub struct MatchResult {
    /// Best pose of the query frame in the reference frame (theta wrapped to [-pi, pi)).
    pub pose: Pose2,
    /// Its score: mean likelihood over the query points, in [0, 1].
    pub score: f64,
    /// Number of (finite) query points.
    pub n: usize,
    /// Covariance of (x, y, theta) (m^2, m rad, rad^2), row-major.
    pub covariance: [[f64; 3]; 3],
    /// Eigenvalues (min, max) of the covariance's xy block (m^2): sqrt(max) close to the window's
    /// half-width / sqrt(3) means a direction the scene does not constrain (corridor).
    pub xy_eigenvalues: [f64; 2],
    /// Direction (rad, fixed frame, in [-pi/2, pi/2]) of the xy block's major axis.
    pub xy_major_axis: f64,
    /// Heading step used (rad).
    pub theta_step: f64,
    /// Fine candidates evaluated (the pruning's work).
    pub evaluated: usize,
}

/// Best pose of the `points` (query frame, m) in the `grid`'s frame, within the window around `guess`
/// (query frame in the reference frame, e.g. from odometry). Non-finite points are ignored. With no point,
/// the guess with score 0 and the window's uniform covariance. An error for invalid parameters or more
/// than [`MAX_HEADINGS`] headings.
pub fn match_points(
    grid: &LikelihoodGrid,
    points: &[Point2],
    guess: Pose2,
    params: &MatchParams,
) -> Result<MatchResult, ParamError> {
    params.validate()?;
    let pts: Vec<Point2> = points
        .iter()
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .collect();
    let n = pts.len();
    let res = grid.params.res;
    let (wx, wy) = ((params.dx / res).round() as i64, (params.dy / res).round() as i64);

    // heading lattice: k * step, k in -nth..=nth, ending exactly on +-dtheta
    let r_max = pts.iter().map(|p| p[0].hypot(p[1])).fold(0.0, f64::max);
    let step0 = params
        .theta_step
        .unwrap_or(if r_max > 0.0 { res / r_max } else { f64::INFINITY });
    let nth = if params.dtheta > 0.0 {
        (params.dtheta / step0).ceil()
    } else {
        0.0
    };
    if 2.0 * nth + 1.0 > MAX_HEADINGS as f64 {
        return Err(ParamError(format!(
            "MatchParams: {} headings (dtheta / theta_step), at most {MAX_HEADINGS}",
            2.0 * nth + 1.0
        )));
    }
    let nth = nth as i64;
    let step = if nth > 0 { params.dtheta / nth as f64 } else { 0.0 };
    let thetas: Vec<f64> = (-nth..=nth).map(|k| k as f64 * step).collect();

    if n == 0 {
        // no information: uniform over the window
        let (vx, vy) = ((wx as f64 * res).powi(2) / 3.0, (wy as f64 * res).powi(2) / 3.0);
        let vt = params.dtheta.powi(2) / 3.0;
        return Ok(MatchResult {
            pose: Pose2::new(guess.x, guess.y, wrap_angle(guess.theta)),
            score: 0.0,
            n: 0,
            covariance: [[vx, 0.0, 0.0], [0.0, vy, 0.0], [0.0, 0.0, vt]],
            xy_eigenvalues: [vx.min(vy), vx.max(vy)],
            xy_major_axis: if vy > vx { std::f64::consts::FRAC_PI_2 } else { 0.0 },
            theta_step: step,
            evaluated: 0,
        });
    }

    // Cell of every point at every heading, translation (0, 0): heading a's cells are cells[a n..(a + 1) n].
    let o = grid.origin;
    let mut cells: Vec<[i64; 2]> = Vec::with_capacity(thetas.len() * n);
    for &dth in &thetas {
        let (s, c) = (guess.theta + dth).sin_cos();
        for p in &pts {
            let x = guess.x + c * p[0] - s * p[1];
            let y = guess.y + s * p[0] + c * p[1];
            cells.push([
                ((x - o[0]) / res).floor() as i64,
                ((y - o[1]) / res).floor() as i64,
            ]);
        }
    }
    let at = |a: usize| &cells[a * n..(a + 1) * n];
    let inv_n = 1.0 / n as f64;

    // Low resolution: one bound per (heading, block of D x D translations).
    let d = grid.params.coarse as i64;
    let mut blocks: Vec<(f64, usize, i64, i64)> = Vec::new();
    for a in 0..thetas.len() {
        let cs = at(a);
        for bx in (-wx..=wx).step_by(d as usize) {
            for by in (-wy..=wy).step_by(d as usize) {
                let sum: f64 = cs
                    .iter()
                    .map(|c| grid.coarse_at(c[0] + bx, c[1] + by) as f64)
                    .sum();
                blocks.push((sum * inv_n, a, bx, by));
            }
        }
    }
    // decreasing bound; stable, so ties keep the generation order (deterministic)
    blocks.sort_by(|p, q| q.0.total_cmp(&p.0));

    // High resolution, best blocks first, until no block can hold a candidate within the margin.
    let margin = -MIN_WEIGHT.ln() / params.score_gain;
    let mut best = (f64::NEG_INFINITY, 0usize, 0i64, 0i64);
    let mut evaluated: Vec<(f64, usize, i64, i64)> = Vec::new();
    for &(bound, a, bx, by) in &blocks {
        if bound < best.0 - margin {
            break;
        }
        let cs = at(a);
        for tx in bx..(bx + d).min(wx + 1) {
            for ty in by..(by + d).min(wy + 1) {
                let sum: f64 = cs.iter().map(|c| grid.fine_at(c[0] + tx, c[1] + ty) as f64).sum();
                let s = sum * inv_n;
                evaluated.push((s, a, tx, ty));
                if s > best.0 {
                    best = (s, a, tx, ty);
                }
            }
        }
    }

    // Covariance: weighted second moment of the candidates within the margin, about their weighted mean.
    let k = params.score_gain;
    let near: Vec<(f64, [f64; 3])> = evaluated
        .iter()
        .filter(|e| e.0 >= best.0 - margin)
        .map(|&(s, a, tx, ty)| {
            (
                (k * (s - best.0)).exp(),
                [tx as f64 * res, ty as f64 * res, thetas[a]],
            )
        })
        .collect();
    let w_sum: f64 = near.iter().map(|e| e.0).sum();
    let mut mean = [0.0; 3];
    for (w, v) in &near {
        for r in 0..3 {
            mean[r] += w * v[r] / w_sum;
        }
    }
    let mut cov = [[0.0; 3]; 3];
    for (w, v) in &near {
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += w * (v[r] - mean[r]) * (v[c] - mean[c]) / w_sum;
            }
        }
    }
    // floor: the lattice quantisation (uniform over one step)
    cov[0][0] += res * res / 12.0;
    cov[1][1] += res * res / 12.0;
    cov[2][2] += step * step / 12.0;

    // eigen-decomposition of the symmetric xy block
    let (a, b, c) = (cov[0][0], cov[0][1], cov[1][1]);
    let half_tr = 0.5 * (a + c);
    let rad = (0.25 * (a - c).powi(2) + b * b).sqrt();
    let (_, ba, tx, ty) = best;
    Ok(MatchResult {
        pose: Pose2::new(
            guess.x + tx as f64 * res,
            guess.y + ty as f64 * res,
            wrap_angle(guess.theta + thetas[ba]),
        ),
        score: best.0,
        n,
        covariance: cov,
        xy_eigenvalues: [half_tr - rad, half_tr + rad],
        xy_major_axis: 0.5 * (2.0 * b).atan2(a - c),
        theta_step: step,
        evaluated: evaluated.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// xorshift64*: deterministic, no dependency.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }

        /// Uniform in [lo, hi).
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next()
        }

        /// Standard normal (Box-Muller).
        fn normal(&mut self) -> f64 {
            let u = self.next().max(1e-300);
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.next()).cos()
        }
    }

    /// Points along the polyline `v` (closed: back to the first vertex), every ~`spacing` m, with Gaussian
    /// noise of std `noise` m on each coordinate.
    fn walls(v: &[Point2], closed: bool, spacing: f64, noise: f64, rng: &mut Rng) -> Vec<Point2> {
        let mut pts = Vec::new();
        let segs = if closed { v.len() } else { v.len() - 1 };
        for k in 0..segs {
            let (a, b) = (v[k], v[(k + 1) % v.len()]);
            let len = (b[0] - a[0]).hypot(b[1] - a[1]);
            let m = (len / spacing).round() as usize;
            for s in 0..m {
                let f = s as f64 / m as f64;
                pts.push([
                    a[0] + f * (b[0] - a[0]) + noise * rng.normal(),
                    a[1] + f * (b[1] - a[1]) + noise * rng.normal(),
                ]);
            }
        }
        pts
    }

    /// An L-shaped room, 4 x 3.5 m, 15 m of walls (no symmetry).
    const L_ROOM: [Point2; 6] = [
        [-2.0, -1.75],
        [2.0, -1.75],
        [2.0, 0.25],
        [0.0, 0.25],
        [0.0, 1.75],
        [-2.0, 1.75],
    ];

    fn grid_of(pts: &[Point2]) -> LikelihoodGrid {
        LikelihoodGrid::new(pts, LikelihoodGridParams::default()).unwrap()
    }

    /// Fixed-frame points seen from `truth`: the submap frame's points.
    fn seen_from(truth: Pose2, pts: &[Point2]) -> Vec<Point2> {
        pts.iter().map(|&p| truth.to_robot(p)).collect()
    }

    /// (position error (m), heading error (rad)).
    fn errors(r: &MatchResult, truth: Pose2) -> (f64, f64) {
        (
            (r.pose.x - truth.x).hypot(r.pose.y - truth.y),
            wrap_angle(r.pose.theta - truth.theta).abs(),
        )
    }

    #[test]
    fn likelihood_grid_values_and_bounds() {
        // binary fractions (res 1/16 m), so that cell boundaries are exact: margin 3.5 cells puts the point
        // at the centre of cell [3, 3], and the grid ends 3.5 cells after it
        let params = LikelihoodGridParams {
            res: 0.0625,
            sigma: 0.0625,
            margin: 0.21875,
            coarse: 4,
        };
        let g = LikelihoodGrid::new(&[[0.03125, 0.03125]], params).unwrap();
        assert_eq!(g.shape(), (8, 8));
        assert_eq!(g.origin(), [-0.1875, -0.1875]);
        assert_eq!(g.fine_at(3, 3), 1.0);
        let one = (-0.5f64).exp() as f32; // one sigma away
        assert!((g.fine_at(4, 3) - one).abs() < 1e-6 && (g.fine_at(3, 2) - one).abs() < 1e-6);
        assert_eq!(g.fine_at(3, 7), 0.0); // 4 sigma away: cut
        assert_eq!(g.fine_at(-1, 3), 0.0); // outside
        assert_eq!(g.fine_at(3, 8), 0.0);
        // the coarse table bounds every fine cell of its block, also for blocks starting before the grid
        for i in -5..10 {
            for j in -5..10 {
                for di in 0..4 {
                    for dj in 0..4 {
                        assert!(g.coarse_at(i, j) >= g.fine_at(i + di, j + dj));
                    }
                }
            }
        }
        let p = LikelihoodGridParams::default();
        for bad in [
            LikelihoodGridParams { res: 0.0, ..p },
            LikelihoodGridParams { sigma: -1.0, ..p },
            LikelihoodGridParams {
                margin: f64::NAN,
                ..p
            },
            LikelihoodGridParams { coarse: 0, ..p },
        ] {
            assert!(LikelihoodGrid::new(&[[0.0, 0.0]], bad).is_err());
        }
        assert!(LikelihoodGrid::new(&[[f64::NAN, 0.0]], p).is_err());
        assert!(LikelihoodGrid::new(&[[0.0, 0.0], [1000.0, 0.0]], p).is_err()); // 20000 cells
    }

    #[test]
    fn exact_recovery_on_the_lattice() {
        let mut rng = Rng(1);
        // margin 3.5 cells: the walls (multiples of res away from the lowest one) lie on cell centres, so
        // that no translation by one cell ties with the truth
        let params = LikelihoodGridParams {
            margin: 0.175,
            ..Default::default()
        };
        let g = LikelihoodGrid::new(&walls(&L_ROOM, true, 0.02, 0.0, &mut rng), params).unwrap();
        let truth = Pose2::new(0.15, -0.10, 5f64.to_radians());
        let query = seen_from(truth, &walls(&L_ROOM, true, 0.03, 0.0, &mut rng));
        let params = MatchParams {
            theta_step: Some(1f64.to_radians()),
            ..Default::default()
        };
        let r = match_points(&g, &query, Pose2::default(), &params).unwrap();
        let (dp, dth) = errors(&r, truth);
        assert!(dp < 1e-9 && dth < 1e-9, "{r:?}");
        assert!(r.score > 0.9, "{}", r.score);
        assert_eq!(r.n, query.len());
    }

    #[test]
    fn the_pruned_search_finds_the_full_search_best() {
        // the coarse table only bounds: the pruned best is the best of every candidate
        let mut rng = Rng(2);
        let g = grid_of(&walls(&L_ROOM, true, 0.03, 0.01, &mut rng));
        let truth = Pose2::new(0.23, 0.12, -0.2);
        let query = seen_from(truth, &walls(&L_ROOM, true, 0.04, 0.02, &mut rng));
        let params = MatchParams::default();
        let r = match_points(&g, &query, Pose2::default(), &params).unwrap();
        let res = g.params.res;
        let (wx, wy) = ((params.dx / res).round() as i64, (params.dy / res).round() as i64);
        let nth = (params.dtheta / r.theta_step).round() as i64;
        let mut full = f64::NEG_INFINITY;
        for k in -nth..=nth {
            for tx in -wx..=wx {
                for ty in -wy..=wy {
                    let pose = Pose2::new(tx as f64 * res, ty as f64 * res, k as f64 * r.theta_step);
                    let sum: f64 = query
                        .iter()
                        .map(|&p| {
                            let q = pose.to_fixed(p);
                            let i = ((q[0] - g.origin[0]) / res).floor() as i64;
                            let j = ((q[1] - g.origin[1]) / res).floor() as i64;
                            g.fine_at(i, j) as f64
                        })
                        .sum();
                    full = full.max(sum / query.len() as f64);
                }
            }
        }
        assert!((r.score - full).abs() < 1e-9, "{} vs {full}", r.score);
        let all = (2 * wx + 1) * (2 * wy + 1) * (2 * nth + 1);
        println!("pruned search: {} of {all} candidates evaluated", r.evaluated);
        assert!((r.evaluated as i64) < all / 4);
    }

    #[test]
    fn recovery_across_headings_with_noise() {
        let mut rng = Rng(3);
        let g = grid_of(&walls(&L_ROOM, true, 0.03, 0.01, &mut rng));
        for deg in [-18.0, -11.0, -4.5, 0.0, 3.3, 9.0, 17.0] {
            let truth = Pose2::new(
                rng.uniform(-0.4, 0.4),
                rng.uniform(-0.4, 0.4),
                f64::to_radians(deg),
            );
            let query = seen_from(truth, &walls(&L_ROOM, true, 0.03, 0.015, &mut rng));
            let r = match_points(&g, &query, Pose2::default(), &MatchParams::default()).unwrap();
            let (dp, dth) = errors(&r, truth);
            assert!(
                dp < 0.05 && dth < 1.5f64.to_radians(),
                "{deg} deg: {dp} m, {dth} rad"
            );
            assert!(r.score > 0.6, "{}", r.score);
            // a well-conditioned match: a few cm at most in every direction
            assert!(r.xy_eigenvalues[1].sqrt() < 0.05, "{:?}", r.xy_eigenvalues);
        }
    }

    #[test]
    fn outliers_are_tolerated() {
        let mut rng = Rng(4);
        let g = grid_of(&walls(&L_ROOM, true, 0.03, 0.01, &mut rng));
        let truth = Pose2::new(-0.3, 0.2, 0.15);
        let mut room = walls(&L_ROOM, true, 0.03, 0.015, &mut rng);
        // 40 % more points, spread over the room (furniture, segmentation errors)
        for _ in 0..room.len() * 2 / 5 {
            room.push([rng.uniform(-1.9, 1.9), rng.uniform(-1.6, 0.2)]);
        }
        let query = seen_from(truth, &room);
        let r = match_points(&g, &query, Pose2::default(), &MatchParams::default()).unwrap();
        let (dp, dth) = errors(&r, truth);
        assert!(dp < 0.05 && dth < 1.5f64.to_radians(), "{dp} m, {dth} rad");
    }

    #[test]
    fn a_corridor_is_constrained_across_not_along() {
        let mut rng = Rng(5);
        // two parallel walls along x, 1.2 m apart, 8 m long; the submap sees 3 m of them
        let corridor = |rng: &mut Rng, x: f64, spacing: f64, noise: f64| {
            let mut p = walls(&[[-x, -0.6], [x, -0.6]], false, spacing, noise, rng);
            p.extend(walls(&[[-x, 0.6], [x, 0.6]], false, spacing, noise, rng));
            p
        };
        let g = grid_of(&corridor(&mut rng, 4.0, 0.03, 0.01));
        let truth = Pose2::new(0.1, 0.05, 2f64.to_radians());
        let query = seen_from(truth, &corridor(&mut rng, 1.5, 0.03, 0.015));
        let r = match_points(&g, &query, Pose2::default(), &MatchParams::default()).unwrap();
        let c = r.covariance;
        println!(
            "corridor: std x {:.3} m, y {:.3} m, theta {:.2} deg; sqrt eigenvalues {:.3} {:.3} m, \
             major axis {:.1} deg",
            c[0][0].sqrt(),
            c[1][1].sqrt(),
            c[2][2].sqrt().to_degrees(),
            r.xy_eigenvalues[0].sqrt(),
            r.xy_eigenvalues[1].sqrt(),
            r.xy_major_axis.to_degrees()
        );
        // across (y) and heading recovered
        assert!((r.pose.y - truth.y).abs() < 0.03 && (r.pose.theta - truth.theta).abs() < 0.03);
        // along: as uncertain as the window allows; across: a few cm
        assert!(c[0][0] > 50.0 * c[1][1], "{c:?}");
        assert!(r.xy_eigenvalues[1].sqrt() > 0.2);
        assert!(r.xy_major_axis.abs() < 5f64.to_radians());
    }

    #[test]
    fn a_guess_outside_the_window_scores_low() {
        let mut rng = Rng(6);
        let g = grid_of(&walls(&L_ROOM, true, 0.03, 0.01, &mut rng));
        let truth = Pose2::new(1.3, -1.1, 0.0);
        let query = seen_from(truth, &walls(&L_ROOM, true, 0.03, 0.015, &mut rng));
        let params = MatchParams::default();
        let near = match_points(&g, &query, Pose2::new(1.2, -1.0, 0.1), &params).unwrap();
        let far = match_points(&g, &query, Pose2::default(), &params).unwrap();
        println!("score in window {:.3}, outside {:.3}", near.score, far.score);
        assert!(
            near.score > 0.6 && far.score < 0.35,
            "{} {}",
            near.score,
            far.score
        );
    }

    #[test]
    fn a_partial_70_degree_view_matches_the_full_room() {
        let mut rng = Rng(7);
        let g = grid_of(&walls(&L_ROOM, true, 0.03, 0.01, &mut rng));
        let truth = Pose2::new(-0.4, -0.3, 0.3);
        // only what is within +-35 deg of the robot's heading, every 5 cm (occlusion ignored)
        let query: Vec<Point2> = seen_from(truth, &walls(&L_ROOM, true, 0.05, 0.015, &mut rng))
            .into_iter()
            .filter(|p| p[1].atan2(p[0]).abs() < 35f64.to_radians())
            .collect();
        assert!(query.len() > 40, "{}", query.len());
        let r = match_points(&g, &query, Pose2::default(), &MatchParams::default()).unwrap();
        let (dp, dth) = errors(&r, truth);
        assert!(dp < 0.05 && dth < 1.5f64.to_radians(), "{dp} m, {dth} rad, {r:?}");
    }

    #[test]
    fn no_point_and_bad_parameters() {
        let g = grid_of(&L_ROOM);
        let r = match_points(&g, &[[f64::NAN, 0.0]], Pose2::default(), &MatchParams::default()).unwrap();
        assert_eq!((r.n, r.score, r.pose), (0, 0.0, Pose2::default()));
        assert!((r.covariance[0][0] - 0.25 / 3.0).abs() < 1e-12);
        let p = MatchParams::default();
        for bad in [
            MatchParams { dx: -0.1, ..p },
            MatchParams {
                dtheta: f64::NAN,
                ..p
            },
            MatchParams {
                theta_step: Some(0.0),
                ..p
            },
            MatchParams { score_gain: 0.0, ..p },
            MatchParams {
                theta_step: Some(1e-9),
                ..p
            }, // too many headings
        ] {
            assert!(match_points(&g, &L_ROOM, Pose2::default(), &bad).is_err());
        }
    }

    #[test]
    fn timing_500_points() {
        // the performance target: 500 points, +-0.5 m x +-0.5 m x +-20 deg, res 0.05 (printed, not asserted)
        let mut rng = Rng(8);
        let t0 = Instant::now();
        let g = grid_of(&walls(&L_ROOM, true, 0.02, 0.01, &mut rng));
        let t_grid = t0.elapsed();
        let truth = Pose2::new(0.27, -0.33, 0.25);
        let mut query = seen_from(truth, &walls(&L_ROOM, true, 0.03, 0.015, &mut rng));
        query.truncate(500);
        let runs = 20;
        let t0 = Instant::now();
        let mut r = None;
        for _ in 0..runs {
            r = Some(match_points(&g, &query, Pose2::default(), &MatchParams::default()).unwrap());
        }
        let r = r.unwrap();
        println!(
            "grid {:?} ({} x {} cells); match {:?} per call: {} points, {} headings, {} candidates \
             evaluated",
            t_grid,
            g.shape().0,
            g.shape().1,
            t0.elapsed() / runs,
            query.len(),
            (2.0 * 20f64.to_radians() / r.theta_step).round() as usize + 1,
            r.evaluated
        );
        let (dp, dth) = errors(&r, truth);
        assert!(dp < 0.05 && dth < 1.5f64.to_radians());
    }
}
