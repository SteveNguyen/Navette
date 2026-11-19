//! Occupancy grid (Bayesian, log-odds) in the fixed frame, fed by scan keyframes.
//!
//! Each cell holds l = log(p / (1 - p)), p = P(occupied), prior 0.5 (l = 0, unknown). A keyframe updates
//! every cell it observes once with the inverse sensor model p_z (Thrun et al., Probabilistic Robotics,
//! ch. 9):
//!     l <- clip(l + w log(p_z / (1 - p_z)), -clamp, clamp)
//!   - cells from the closest visible point to the sector's free distance: p_free
//!   - the cell at the sector's obstacle distance (if any): p_occ
//!   - the robot footprint (radius): p_free, the robot stands on floor
//!
//! Only what is seen is recorded. An obstacle filling the image bottom (`Scan::near`) may be anywhere between
//! the robot and the closest visible point, but that guess is NOT written here: persisted, it walls the robot
//! in after a turn on the spot. The planner and the collision monitor use it from the current scan only.
//!
//! The evidence of a beam at range d is scaled by w = 1 / (1 + (s_d / sigma_ref)^2), with
//! s_d = (d^2 + h^2) / h * s the range error caused by the current tilt error bound s (camera at height h):
//! p_z moves toward 0.5 at long range or under motion. Nothing beyond max_range (the trusted range) is used.
//!
//! Static world, no time decay (stale cells would become unknown and turn old walls into frontiers):
//! obstacles are only cleared by being seen free again. The clamp bounds the certainty, so ~4 contrary
//! keyframes flip a cell (a person who stood still and left).
//!
//! Keyframes (robot moved key_m, turned key_deg, or key_s elapsed) are kept: [`OccupancyGrid::replay`]
//! rebuilds the grid from them, e.g. after a SLAM pose correction.
//!
//! Frames: poses are in the fixed frame (geometry.rs). Cell [i, j] covers
//! x in [x0 + i res, x0 + (i + 1) res), y in [y0 + j res, y0 + (j + 1) res); the log-odds array is row-major,
//! index i * n + j. The grid is centred on the first pose it is given and does not move afterwards: what
//! falls outside is ignored (updates) or unknown (queries).
//!
//! Numerics follow the Python prototype (numpy), so that its grids are reproduced bit for bit: f64 geometry
//! with the same order of operations, sample ranges as `numpy.arange`, f32 log-odds, increments rounded to
//! f32 before the sum, thresholds compared in f32.

use std::f64::consts::TAU;

use crate::geometry::{wrap_angle, Point2, Pose2};
use crate::scan::Scan;

/// Log-odds of one obstacle observation at full weight: ln(0.7 / 0.3).
pub const L_OCC: f64 = 0.8472978603872037;
/// Log-odds of one free observation at full weight: ln(0.4 / 0.6).
pub const L_FREE: f64 = -0.4054651081081643;

/// Each sector is sampled along three beams, at these fractions of the sector width from its centre.
const SUB_BEAMS: [f64; 3] = [-1.0 / 3.0, 0.0, 1.0 / 3.0];
/// Footprint samples: rings of this many points at radii 0, 0.05, 0.1 m and the robot radius.
const FOOTPRINT_POINTS: usize = 24;
const FOOTPRINT_RADII: [f64; 3] = [0.0, 0.05, 0.1];

/// Grid parameters. The defaults are the ones tuned on Reachy Mini.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OccupancyGridParams {
    /// Cell size (m).
    pub res: f64,
    /// Side of the square grid (m).
    pub size_m: f64,
    /// Bound of |l|. 1.5 (p 0.18 .. 0.82): ~4 contrary keyframes (~2 s at rest) flip a cell.
    pub clamp: f64,
    /// A cell is obstacle if l > thr, free if l < -thr, unknown otherwise: one observation with weight
    /// > ~0.5 decides it.
    pub thr: f64,
    /// Range error (m) at which evidence counts half (the tilt-trust tolerance).
    pub sigma_ref: f64,
    /// Obstacles at or beyond this range (m) are not written.
    pub max_obstacle_m: f64,
    /// Robot radius (m): the footprint marked free, and the default start of the blind segment.
    pub robot_radius: f64,
    /// A new keyframe is taken when the robot moved this far (m),
    pub key_m: f64,
    /// or turned this much (deg),
    pub key_deg: f64,
    /// or after this long (s).
    pub key_s: f64,
}

impl Default for OccupancyGridParams {
    fn default() -> Self {
        Self {
            res: 0.05,
            size_m: 24.0,
            clamp: 1.5,
            thr: 0.2,
            sigma_ref: 0.10,
            max_obstacle_m: 2.5,
            robot_radius: 0.13,
            key_m: 0.1,
            key_deg: 5.0,
            key_s: 0.5,
        }
    }
}

/// One integrated scan, with everything needed to integrate it again.
#[derive(Clone, Debug, PartialEq)]
pub struct Keyframe {
    /// Time of the scan (s).
    pub t: f64,
    /// Robot pose in the fixed frame. May be corrected before [`OccupancyGrid::replay`].
    pub pose: Pose2,
    pub scan: Scan,
    /// Tilt error bound (rad); None = unweighted.
    pub sigma: Option<f64>,
    /// Camera height (m).
    pub h: f64,
    /// Trusted range (m); None = all.
    pub max_range: Option<f64>,
}

pub struct OccupancyGrid {
    params: OccupancyGridParams,
    /// Cells per side.
    n: usize,
    key_rad: f64,
    /// Log-odds, row-major: [i * n + j].
    l: Vec<f32>,
    /// Fixed-frame position of the corner of cell [0, 0]; None until the first update.
    origin: Option<Point2>,
    keyframes: Vec<Keyframe>,
    // Scratch of one keyframe integration (each observed cell is updated once per keyframe).
    /// Increment waiting for each cell; NaN = cell not touched by this keyframe.
    pending: Vec<f32>,
    /// Cells with a pending increment.
    touched: Vec<u32>,
    /// Obstacle samples (cell, increment).
    obst: Vec<(u32, f32)>,
}

impl OccupancyGrid {
    pub fn new(params: OccupancyGridParams) -> Self {
        // Python's round(): ties to even
        let n = (params.size_m / params.res).round_ties_even() as usize;
        Self {
            params,
            n,
            key_rad: params.key_deg.to_radians(),
            l: vec![0.0; n * n],
            origin: None,
            keyframes: Vec::new(),
            pending: vec![f32::NAN; n * n],
            touched: Vec::new(),
            obst: Vec::new(),
        }
    }

    pub fn params(&self) -> &OccupancyGridParams {
        &self.params
    }

    /// Cells per side.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Fixed-frame position (x0, y0) of the corner of cell [0, 0]; None until the first update.
    pub fn origin(&self) -> Option<Point2> {
        self.origin
    }

    /// Log-odds of every cell, row-major: cell [i, j] is at index i * n + j.
    pub fn log_odds(&self) -> &[f32] {
        &self.l
    }

    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    /// The keyframes, to correct their poses before [`Self::replay`].
    pub fn keyframes_mut(&mut self) -> &mut [Keyframe] {
        &mut self.keyframes
    }

    /// Cell [i, j] holding the fixed-frame point `p`; outside [0, n) if the point is outside the grid.
    /// Panics before the first update (the grid has no position yet, see [`Self::origin`]).
    pub fn cell(&self, p: Point2) -> [i64; 2] {
        let o = self
            .origin
            .expect("the grid has no origin before its first update");
        [
            ((p[0] - o[0]) / self.params.res).floor() as i64,
            ((p[1] - o[1]) / self.params.res).floor() as i64,
        ]
    }

    /// Fixed-frame centre of cell [i, j]. Panics before the first update.
    pub fn centre(&self, ij: [i64; 2]) -> Point2 {
        let o = self
            .origin
            .expect("the grid has no origin before its first update");
        centre(o, self.params.res, ij[0], ij[1])
    }

    /// Clear the map and forget the keyframes; the next update centres the grid again.
    pub fn reset(&mut self) {
        self.l.fill(0.0);
        self.origin = None;
        self.keyframes.clear();
    }

    /// Replace the whole map (tests, tools): `log_odds` row-major, n * n values (panics otherwise), with the
    /// corner of cell [0, 0] at `origin`. The keyframes are forgotten: they did not build this map.
    pub fn set_map(&mut self, origin: Point2, log_odds: &[f32]) {
        self.l.copy_from_slice(log_odds);
        self.origin = Some(origin);
        self.keyframes.clear();
    }

    /// Integrate the scan if it is a new keyframe: the first one, or the robot moved, turned or waited
    /// enough since the last one. `force`: always (the frames of a head glance, standing still, see what the
    /// driving frames cannot). Returns true if the scan was integrated.
    /// sigma: tilt error bound (rad), None = full weight at every range. h: camera height (m).
    /// max_range: trusted range (m), None = the whole scan.
    /// The scan's free distance must be finite where seen (the scan caps `floor` at its max_range).
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        t: f64,
        pose: Pose2,
        scan: &Scan,
        sigma: Option<f64>,
        h: f64,
        max_range: Option<f64>,
        force: bool,
    ) -> bool {
        if self.origin.is_none() {
            // grid centred on the first pose
            let half = self.n as f64 * self.params.res / 2.0;
            self.origin = Some([pose.x - half, pose.y - half]);
        }
        if let (Some(k), false) = (self.keyframes.last(), force) {
            let p = &self.params;
            let dth = wrap_angle(pose.theta - k.pose.theta).abs();
            if (pose.x - k.pose.x).hypot(pose.y - k.pose.y) < p.key_m
                && dth < self.key_rad
                && t - k.t < p.key_s
            {
                return false;
            }
        }
        self.keyframes.push(Keyframe {
            t,
            pose,
            scan: scan.clone(),
            sigma,
            h,
            max_range,
        });
        self.integrate(self.keyframes.len() - 1);
        true
    }

    /// Rebuild the grid from the keyframes (after their poses were corrected).
    pub fn replay(&mut self) {
        self.l.fill(0.0);
        for i in 0..self.keyframes.len() {
            self.integrate(i);
        }
    }

    /// Add keyframe `i` to the grid. Each cell it observes is updated once: with its highest weight, and an
    /// obstacle observation wins over a free one.
    fn integrate(&mut self, i: usize) {
        let Self {
            params: p,
            n,
            l,
            origin,
            keyframes,
            pending,
            touched,
            obst,
            ..
        } = self;
        let (n, kf) = (*n, &keyframes[i]);
        let origin = origin.expect("update sets the origin before the first keyframe");
        let scan = &kf.scan;
        let weight = |r: f64| weight(r, kf.sigma, kf.h, p.sigma_ref);
        // robot frame -> fixed frame, as Pose2::to_fixed with the sine and cosine computed once
        let (sin_th, cos_th) = kf.pose.theta.sin_cos();
        let to_fixed = |x: f64, y: f64| {
            [
                kf.pose.x + cos_th * x - sin_th * y,
                kf.pose.y + sin_th * x + cos_th * y,
            ]
        };
        // A free sample: keep the cell's strongest (most negative) increment. Rounding to f32 is monotonic,
        // so this is the increment of its highest weight.
        let mut free_sample = |xy: Point2, dl: f32| {
            if let Some(idx) = index(origin, p.res, n, xy) {
                let pend = &mut pending[idx];
                if pend.is_nan() {
                    touched.push(idx as u32);
                    *pend = dl;
                } else if dl < *pend {
                    *pend = dl;
                }
            }
        };

        let width = (scan.bearing_deg[1] - scan.bearing_deg[0]).to_radians();
        let r_max = kf.max_range.unwrap_or(f64::INFINITY);
        let o_max = r_max.min(p.max_obstacle_m);
        let step = p.res / 2.0;
        obst.clear();
        for (k, free) in scan.free().into_iter().enumerate() {
            if !scan.seen[k] {
                continue;
            }
            // free from the closest visible point to the free distance, cut at the trusted range
            // (a NaN free distance stays NaN and gives no sample)
            let near = scan.closest[k];
            let far = if r_max < free { r_max } else { free };
            let o = scan.obstacle[k];
            let (has_free, has_obst) = (far > near && far.is_finite(), o.is_finite() && o < o_max);
            if !has_free && !has_obst {
                continue;
            }
            let b0 = scan.bearing_deg[k].to_radians();
            // unit vectors (cos, sin) of the three beams, robot frame
            let beams = SUB_BEAMS.map(|f| {
                let (s, c) = (b0 + f * width).sin_cos();
                [c, s]
            });
            if has_free {
                for r in arange(near, far, step) {
                    let dl = (L_FREE * weight(r)) as f32;
                    for u in &beams {
                        free_sample(to_fixed(r * u[0], r * u[1]), dl);
                    }
                }
            }
            if has_obst {
                let dl = (L_OCC * weight(o)) as f32;
                for u in &beams {
                    if let Some(idx) = index(origin, p.res, n, to_fixed(o * u[0], o * u[1])) {
                        obst.push((idx as u32, dl));
                    }
                }
            }
        }
        // footprint: rings inside the robot radius, full weight
        let dl = L_FREE as f32;
        for rr in FOOTPRINT_RADII.into_iter().chain([p.robot_radius]) {
            for a in 0..FOOTPRINT_POINTS {
                let (s, c) = (a as f64 * (TAU / FOOTPRINT_POINTS as f64)).sin_cos();
                free_sample(to_fixed(rr * c, rr * s), dl);
            }
        }
        // obstacle cells: the strongest increment of each cell, replacing a free one
        obst.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
        let mut last = None;
        for &(idx, dl) in obst.iter() {
            if last != Some(idx) {
                last = Some(idx);
                let pend = &mut pending[idx as usize];
                if pend.is_nan() {
                    touched.push(idx);
                }
                *pend = dl;
            }
        }
        let clamp = p.clamp as f32;
        for idx in touched.drain(..) {
            let idx = idx as usize;
            l[idx] = (l[idx] + pending[idx]).clamp(-clamp, clamp);
            pending[idx] = f32::NAN;
        }
    }

    /// (obstacle, free) cells, row-major like [`Self::log_odds`]; the rest is unknown.
    pub fn masks(&self) -> (Vec<bool>, Vec<bool>) {
        let thr = self.params.thr as f32;
        (
            self.l.iter().map(|&l| l > thr).collect(),
            self.l.iter().map(|&l| l < -thr).collect(),
        )
    }

    /// Per sector: true if it is a near sector (obstacle filling the image bottom, `Scan::near`) that the
    /// map explains:
    ///   - its blind segment, from the robot edge `r0` (default: robot_radius) to the closest visible point,
    ///     has been observed: every cell sampled along it is free or obstacle (|l| > thr), none unknown, and
    ///   - the map holds the obstacle itself: a cell with l > thr on that ray between r0 and
    ///     closest + explain_m (e.g. a door frame seen while approaching it).
    ///
    /// For those sectors, the obstacle points of this map stand in for the collision monitor's 'somewhere in
    /// the blind zone' guess. Everything else keeps the guess: never-seen zones (start, after a turn in
    /// place), and an obstacle the map has not seen. A person stepping in front shows as `near` at about
    /// `closest`, a reading that moves with the robot, so it never outweighs the old free cells: with
    /// 'observed' alone the robot would drive through.
    ///
    /// Call it with the map BEFORE integrating the current scan, so the scan cannot explain itself.
    pub fn blind_known(&self, pose: Pose2, scan: &Scan, r0: Option<f64>, explain_m: f64) -> Vec<bool> {
        let mut known = vec![false; scan.len()];
        let Some(origin) = self.origin else {
            return known;
        };
        let p = &self.params;
        let is_near = |k: usize| scan.seen[k] && scan.near[k] && scan.closest[k].is_finite();
        // all the rays are sampled at the same ranges, up to the farthest blind segment
        let Some(far) = (0..scan.len())
            .filter(|&k| is_near(k))
            .map(|k| scan.closest[k])
            .reduce(f64::max)
        else {
            return known;
        };
        let r0 = r0.unwrap_or(p.robot_radius);
        let stop = if far > r0 { far } else { r0 } + explain_m + 1e-9;
        let thr = p.thr as f32;
        let (sin_th, cos_th) = pose.theta.sin_cos();
        for k in (0..scan.len()).filter(|&k| is_near(k)) {
            let closest = scan.closest[k];
            // the sector's centre ray, robot frame
            let (s, c) = scan.bearing_deg[k].to_radians().sin_cos();
            let (mut observed, mut explained) = (true, false);
            for r in arange(r0, stop, p.res / 2.0) {
                let (x, y) = (c * r, s * r);
                let xy = [pose.x + cos_th * x - sin_th * y, pose.y + sin_th * x + cos_th * y];
                // outside the grid: unknown
                let l = index(origin, p.res, self.n, xy).map_or(0.0, |idx| self.l[idx]);
                observed &= l.abs() > thr || r > closest;
                explained |= l > thr && r <= closest + explain_m;
            }
            known[k] = observed && explained;
        }
        known
    }

    /// Centres of the obstacle cells closer than `max_range` (m) to the robot, in the robot frame, ordered
    /// by cell (i, then j).
    pub fn obstacle_points(&self, pose: Pose2, max_range: f64) -> Vec<Point2> {
        let Some(origin) = self.origin else {
            return Vec::new();
        };
        let (res, n) = (self.params.res, self.n as i64);
        let thr = self.params.thr as f32;
        // only the cells within m of the robot's can be in range
        let m = (max_range / res).ceil() as i64 + 1;
        let [ci, cj] = self.cell([pose.x, pose.y]);
        let window = |c: i64| c.saturating_sub(m).clamp(0, n)..c.saturating_add(m + 1).clamp(0, n);
        let (sin_th, cos_th) = pose.theta.sin_cos();
        let mut pts = Vec::new();
        for i in window(ci) {
            for j in window(cj) {
                if self.l[(i * n + j) as usize] > thr {
                    // fixed frame -> robot frame, as Pose2::to_robot
                    let c = centre(origin, res, i, j);
                    let (dx, dy) = (c[0] - pose.x, c[1] - pose.y);
                    let q = [cos_th * dx + sin_th * dy, -sin_th * dx + cos_th * dy];
                    if q[0].hypot(q[1]) < max_range {
                        pts.push(q);
                    }
                }
            }
        }
        pts
    }
}

/// Evidence weight of a beam at range `r` (m): 1 with no tilt error bound, else 1 / (1 + (s_d / sigma_ref)^2)
/// with s_d = (r^2 + h^2) / h * sigma the range error (camera at height h).
#[inline]
fn weight(r: f64, sigma: Option<f64>, h: f64, sigma_ref: f64) -> f64 {
    match sigma {
        None => 1.0,
        Some(sigma) => {
            let q = (r * r + h * h) / h * sigma / sigma_ref;
            1.0 / (1.0 + q * q)
        }
    }
}

/// Row-major index of the cell holding the fixed-frame point `xy`; None outside the grid (and for NaN).
#[inline]
fn index(origin: Point2, res: f64, n: usize, xy: Point2) -> Option<usize> {
    let i = ((xy[0] - origin[0]) / res).floor();
    let j = ((xy[1] - origin[1]) / res).floor();
    let nf = n as f64;
    (i >= 0.0 && i < nf && j >= 0.0 && j < nf).then(|| i as usize * n + j as usize)
}

#[inline]
fn centre(origin: Point2, res: f64, i: i64, j: i64) -> Point2 {
    [
        (i as f64 + 0.5) * res + origin[0],
        (j as f64 + 0.5) * res + origin[1],
    ]
}

/// The values of `numpy.arange(start, stop, step)` for finite floats, bit for bit: ceil((stop - start) / step)
/// values; the second is start + step and the following ones start + i * ((start + step) - start), whose
/// rounding differs from start + i * step.
fn arange(start: f64, stop: f64, step: f64) -> impl Iterator<Item = f64> {
    let len = ((stop - start) / step).ceil().max(0.0) as usize;
    let next = start + step;
    let delta = next - start;
    (0..len).map(move |i| match i {
        0 => start,
        1 => next,
        _ => start + i as f64 * delta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECTORS: usize = 180;

    fn bearing(k: usize) -> f64 {
        (k as f64 + 0.5) * 2.0 - 180.0
    }

    /// Floor seen from `closest` to `floor` in the sectors within `fov` deg of straight ahead; obstacle at
    /// `obst_at(bearing)` (inf = none).
    fn scan(obst_at: impl Fn(f64) -> f64, fov: f64, closest: f64, floor: f64) -> Scan {
        Scan {
            bearing_deg: (0..SECTORS).map(bearing).collect(),
            seen: (0..SECTORS).map(|k| bearing(k).abs() < fov).collect(),
            obstacle: (0..SECTORS).map(|k| obst_at(bearing(k))).collect(),
            near: vec![false; SECTORS],
            floor: vec![floor; SECTORS],
            closest: vec![closest; SECTORS],
        }
    }

    fn open() -> Scan {
        scan(|_| f64::INFINITY, 38.0, 0.4, 2.5)
    }

    /// A wall 1.02 m ahead, 8 deg wide.
    fn wall() -> Scan {
        scan(
            |b| if b.abs() < 4.0 { 1.02 } else { f64::INFINITY },
            38.0,
            0.4,
            2.5,
        )
    }

    /// Open floor from 0.44 m in a wider view.
    fn wide() -> Scan {
        scan(|_| f64::INFINITY, 55.0, 0.44, 2.0)
    }

    fn grid() -> OccupancyGrid {
        OccupancyGrid::new(OccupancyGridParams {
            size_m: 8.0,
            ..Default::default()
        })
    }

    const HERE: Pose2 = Pose2 {
        x: 0.0,
        y: 0.0,
        theta: 0.0,
    };

    /// Unweighted, unforced update.
    fn update(g: &mut OccupancyGrid, t: f64, pose: Pose2, scan: &Scan) -> bool {
        g.update(t, pose, scan, None, 0.35, None, false)
    }

    /// Log-odds of the cell holding the fixed-frame point (x, y).
    fn l_at(g: &OccupancyGrid, x: f64, y: f64) -> f32 {
        let [i, j] = g.cell([x, y]);
        g.log_odds()[i as usize * g.n() + j as usize]
    }

    #[test]
    fn constants_are_the_log_odds_of_the_sensor_model() {
        assert_eq!(L_OCC, (0.7f64 / 0.3).ln());
        assert_eq!(L_FREE, (0.4f64 / 0.6).ln());
    }

    #[test]
    fn arange_is_numpy_arange() {
        // numpy.arange(0.4, 0.5, 0.025): the last value is below the stop by rounding only
        let r: Vec<f64> = arange(0.4, 0.5, 0.025).collect();
        assert_eq!(
            r,
            [0.4, 0.42500000000000004, 0.45000000000000007, 0.4750000000000001]
        );
        assert_eq!(arange(0.4, 0.4, 0.025).count(), 0);
        assert_eq!(arange(0.4, 0.3, 0.025).count(), 0);
        assert_eq!(arange(0.4, f64::NAN, 0.025).count(), 0);
        assert_eq!(arange(0.0, 0.01, 0.025).collect::<Vec<_>>(), [0.0]);
    }

    #[test]
    fn cells_and_centres() {
        let mut g = grid();
        assert_eq!((g.n(), g.origin()), (160, None));
        update(&mut g, 0.0, Pose2::new(1.0, -2.0, 0.3), &open());
        assert_eq!(g.origin(), Some([-3.0, -6.0])); // centred on the first pose
        assert_eq!(g.cell([1.0, -2.0]), [80, 80]);
        assert_eq!(g.cell([-3.01, 2.01]), [-1, 160]); // outside
        let c = g.centre([80, 79]);
        assert!((c[0] - 1.025).abs() < 1e-12 && (c[1] + 2.025).abs() < 1e-12);
        assert_eq!(g.cell(c), [80, 79]);
    }

    #[test]
    fn one_keyframe_marks_free_obstacle_footprint_and_leaves_the_rest_unknown() {
        let mut g = grid();
        assert!(update(&mut g, 0.0, HERE, &wall()));
        assert_eq!(l_at(&g, 1.02, 0.0), L_OCC as f32);
        assert_eq!(l_at(&g, 0.7, 0.0), L_FREE as f32);
        assert_eq!(l_at(&g, 0.05, 0.0), L_FREE as f32); // under the robot
        assert_eq!(l_at(&g, 0.3, 0.0), 0.0); // blind zone in front of the robot
        assert_eq!(l_at(&g, 0.0, 1.0), 0.0); // outside the view
        let (obst, free) = g.masks();
        let [i, j] = g.cell([1.02, 0.0]);
        let idx = i as usize * g.n() + j as usize;
        assert!(obst[idx] && !free[idx]);
        let count = |m: &[bool]| m.iter().filter(|&&b| b).count();
        assert_eq!(count(&free), g.log_odds().iter().filter(|&&l| l < 0.0).count());
        assert_eq!(count(&obst), g.log_odds().iter().filter(|&&l| l > 0.0).count());
    }

    #[test]
    fn a_cell_is_updated_once_per_keyframe_and_the_obstacle_wins() {
        // many samples fall in each cell (half-cell steps, three beams, neighbouring sectors): one increment
        let mut g = grid();
        update(&mut g, 0.0, HERE, &wall());
        assert!(g
            .log_odds()
            .iter()
            .all(|&l| l == 0.0 || l == L_FREE as f32 || l == L_OCC as f32));
        // the wall sectors' free samples reach the wall's cell (1.00 .. 1.05 m): it is the obstacle's
        assert_eq!(l_at(&g, 1.01, 0.0), L_OCC as f32);
    }

    #[test]
    fn keyframes_skip_repeated_views() {
        let mut g = grid();
        let mut up = |t: f64, x: f64, deg: f64, force: bool| {
            g.update(
                t,
                Pose2::new(x, 0.0, deg.to_radians()),
                &open(),
                None,
                0.35,
                None,
                force,
            )
        };
        assert!(up(0.0, 0.0, 0.0, false));
        assert!(!up(0.1, 0.02, 0.0, false)); // barely moved, too soon
        assert!(up(0.15, 0.02, 0.0, true)); // forced
        assert!(up(0.2, 0.15, 0.0, false)); // moved 13 cm
        assert!(up(0.3, 0.15, 10.0, false)); // turned 10 deg
        assert!(!up(0.4, 0.15, 370.0, false)); // a full turn more: the same heading
        assert!(up(0.9, 0.15, 10.0, false)); // 0.6 s later
        assert_eq!(g.keyframes().len(), 5);
    }

    #[test]
    fn far_evidence_is_weak_under_tilt_error_and_max_range_cuts() {
        let mut g = grid();
        g.update(0.0, HERE, &open(), Some(2f64.to_radians()), 0.35, None, false);
        let thr = g.params().thr as f32;
        assert!(l_at(&g, 0.6, 0.0) < -thr); // near: small range error
        let far = l_at(&g, 2.0, 0.0);
        assert!(far < 0.0 && far > -thr); // far: ~0.4 m range error, weak evidence
        let mut g = grid();
        let far_wall = scan(
            |b| if b.abs() < 4.0 { 1.52 } else { f64::INFINITY },
            38.0,
            0.4,
            2.5,
        );
        g.update(0.0, HERE, &far_wall, None, 0.35, Some(1.0), false);
        assert!(l_at(&g, 0.7, 0.0) < 0.0);
        assert_eq!((l_at(&g, 1.3, 0.0), l_at(&g, 1.52, 0.0)), (0.0, 0.0));
    }

    #[test]
    fn log_odds_are_clamped() {
        let mut g = grid();
        for k in 0..20 {
            update(&mut g, k as f64, HERE, &wall());
        }
        assert_eq!((l_at(&g, 1.02, 0.0), l_at(&g, 0.7, 0.0)), (1.5, -1.5));
    }

    #[test]
    fn replay_rebuilds_the_grid_and_follows_corrected_poses() {
        let mut g = grid();
        update(&mut g, 0.0, HERE, &wall());
        g.update(
            1.0,
            Pose2::new(0.3, 0.1, 0.4),
            &open(),
            Some(0.01),
            0.35,
            None,
            false,
        );
        let before = g.log_odds().to_vec();
        g.replay();
        assert_eq!(before, g.log_odds());
        // first keyframe moved 1 m to the left: its wall follows
        g.keyframes_mut()[0].pose.y = 1.0;
        g.replay();
        assert!(l_at(&g, 1.02, 1.0) > 0.0 && l_at(&g, 1.02, 0.0) <= 0.0);
    }

    #[test]
    fn reset_clears_the_map_and_recentres() {
        let mut g = grid();
        update(&mut g, 0.0, HERE, &wall());
        g.reset();
        assert!(g.log_odds().iter().all(|&l| l == 0.0) && g.keyframes().is_empty() && g.origin().is_none());
        assert!(g.obstacle_points(HERE, 1.5).is_empty());
        assert!(update(&mut g, 0.0, Pose2::new(5.0, 5.0, 0.0), &open())); // a first keyframe again
        assert_eq!(g.origin(), Some([1.0, 1.0]));
    }

    #[test]
    fn outside_the_grid_is_ignored_and_unknown() {
        let mut g = grid();
        update(&mut g, 0.0, HERE, &open());
        let before = g.log_odds().to_vec();
        // 50 m away: nothing of this keyframe is inside
        let away = Pose2::new(50.0, 0.0, 0.0);
        assert!(update(&mut g, 1.0, away, &wall()));
        assert_eq!(before, g.log_odds());
        assert!(g.obstacle_points(away, 1.5).is_empty());
        assert!(g.obstacle_points(Pose2::new(-50.0, -50.0, 0.0), 1.5).is_empty());
        let mut near = wall();
        near.near = near.obstacle.iter().map(|o| o.is_finite()).collect();
        assert!(g.blind_known(away, &near, None, 0.1).iter().all(|&k| !k));
    }

    #[test]
    fn obstacle_points_are_in_the_robot_frame_and_in_range() {
        let mut g = grid();
        update(&mut g, 0.0, HERE, &wall());
        let pts = g.obstacle_points(HERE, 1.5);
        assert!(!pts.is_empty());
        assert!(pts
            .iter()
            .all(|p| (p[0] - 1.025).abs() < 1e-9 && p[1].abs() < 0.1));
        // seen from 1 m to the left, facing right (-90 deg): the wall is 1 m ahead, 1 m to the left
        let pts = g.obstacle_points(Pose2::new(0.0, 1.0, -std::f64::consts::FRAC_PI_2), 1.5);
        assert!(!pts.is_empty());
        assert!(pts
            .iter()
            .all(|p| (p[1] - 1.025).abs() < 1e-9 && (p[0] - 1.0).abs() < 0.1));
        assert!(g.obstacle_points(HERE, 1.0).is_empty());
    }

    /// `s` with an obstacle at `range` within 6 deg of straight ahead; `near`: it fills the image bottom.
    fn box_ahead(mut s: Scan, range: f64, near: bool) -> Scan {
        for k in (0..SECTORS).filter(|&k| bearing(k).abs() < 6.0) {
            (s.obstacle[k], s.near[k], s.floor[k]) = (range, near, range);
        }
        s
    }

    #[test]
    fn blind_known_needs_the_blind_zone_observed_and_the_obstacle_in_the_map() {
        let now = box_ahead(wide(), 0.44, true);
        let none = |known: Vec<bool>| known.iter().all(|&k| !k);
        let mut g = OccupancyGrid::new(OccupancyGridParams::default());
        assert!(none(g.blind_known(HERE, &now, None, 0.1))); // before any update

        // open floor seen while driving 1 m to here: the blind zone is observed, but holds no obstacle
        for k in 0..=10 {
            update(
                &mut g,
                0.5 * k as f64,
                Pose2::new(-1.0 + 0.1 * k as f64, 0.0, 0.0),
                &wide(),
            );
        }
        assert!(none(g.blind_known(HERE, &now, None, 0.1)));

        // a box 0.3 m ahead of here, seen from 0.5 m back: the floor behind it, up to the closest visible
        // point, was hidden, so the blind segment is not all observed
        let back = Pose2::new(-0.5, 0.0, 0.0);
        let mut g = OccupancyGrid::new(OccupancyGridParams::default());
        update(&mut g, 0.0, back, &box_ahead(wide(), 0.8, false));
        assert!(none(g.blind_known(HERE, &now, None, 0.1)));

        // a box 0.42 m ahead of here, seen from 0.5 m back: observed and explained
        let mut g = OccupancyGrid::new(OccupancyGridParams::default());
        update(&mut g, 0.0, back, &box_ahead(wide(), 0.92, false));
        let known = g.blind_known(HERE, &now, None, 0.1);
        assert!(known[89] && known[90]); // +-1 deg
        assert!((0..SECTORS).all(|k| !known[k] || now.near[k]));
        // from a robot edge beyond the box, the obstacle is not on the segment any more
        assert!(none(g.blind_known(HERE, &now, Some(0.46), 0.0)));
    }
}
