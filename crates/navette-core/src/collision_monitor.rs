//! Collision monitor: makes any requested base command (v forward m/s, w rad/s, + = left) safe against the
//! scan. Every behaviour requests (v, w); [`safe_command`] returns what may be sent.
//!
//!   - w is kept: the base turns in place safely (the robot is about a circle of radius `radius`).
//!   - v is reduced until the robot, following the arc of curvature w / v, can still stop before any boundary
//!     point comes within radius + margin of its centre:
//!     v * latency + v^2 / (2 decel) <= free arc length.
//!   - Boundary points (robot frame, x forward, y left, origin = robot centre ~ camera foot point):
//!       - seen sectors: the scan's free() distance (nearest obstacle, or where the seen floor ends: unknown
//!         beyond);
//!       - unseen sectors: `unseen_m` from the centre, so an arc may not swing into what the camera has not
//!         seen;
//!       - `near` obstacles (running off the image bottom, true distance unknown): at radius + margin, so no
//!         forward motion toward them (a stool leg 0.2 m away read as 0.44 m, the image bottom). Except the
//!         sectors whose blind segment (robot edge -> closest visible point) the occupancy grid has already
//!         observed (`near_known`): the map's obstacles (`extra_points`) cover that segment, so the sector
//!         keeps its visible distance. Without this, a door frame at the image-bottom corner stopped every
//!         straight drive; Nav2 also relies on its costmap's memory there.
//!
//!     The camera's blind zone just in front of the robot (below the image) is otherwise taken as free: it
//!     was in view a moment before when driving forward. After turning in place it was not: the occupancy
//!     grid is what covers it.
//!   - Reverse: no camera behind, so v < 0 is capped at `reverse_max` (0 by default).
//!
//! Numerics follow the Python prototype, so that the same request gives the same candidate speed: arc
//! samples at step, 2 step, ... (numpy's arange(step, s_max + step, step)), candidate speeds as numpy's
//! linspace(min(v, v_max), 0, n_speeds, endpoint=False), free length = first sample in collision minus one
//! step. One deliberate difference: squares are x * x, where the prototype's scalar `x ** 2` goes through
//! libm's pow, which is not always correctly rounded (the look-ahead may differ by one ulp).

use crate::geometry::Point2;
use crate::scan::Scan;

/// Below this curvature (1/m) an arc is a straight line.
const STRAIGHT_CURV: f64 = 1e-6;

/// Robot footprint and stopping model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SafetyParams {
    /// Robot footprint radius (m).
    pub radius: f64,
    /// Extra clearance (m).
    pub margin: f64,
    /// Highest forward speed (m/s).
    pub v_max: f64,
    /// Braking deceleration (m/s^2).
    pub decel: f64,
    /// Camera frame -> command applied (s): the robot keeps its speed that long before braking.
    pub latency: f64,
    /// Distance (m) of the boundary put in sectors the camera does not see.
    pub unseen_m: f64,
    /// Allowed reverse speed (m/s). Reverse is blind.
    pub reverse_max: f64,
    /// Arc sampling step (m).
    pub step: f64,
    /// Number of candidate speeds between the request and 0.
    pub n_speeds: usize,
}

impl Default for SafetyParams {
    fn default() -> Self {
        Self {
            radius: 0.13,
            margin: 0.05,
            v_max: 0.4,
            decel: 0.5,
            latency: 0.15,
            unseen_m: 0.35,
            reverse_max: 0.0,
            step: 0.02,
            n_speeds: 20,
        }
    }
}

impl SafetyParams {
    /// Closest a boundary point may come to the robot centre (m).
    #[inline]
    pub fn clearance(&self) -> f64 {
        self.radius + self.margin
    }

    /// Look-ahead of the safety check (m): the stopping distance at v_max + 0.1 m.
    #[inline]
    pub fn lookahead(&self) -> f64 {
        self.v_max * self.latency + self.v_max * self.v_max / (2.0 * self.decel) + 0.1
    }
}

/// Distance travelled before standing still from speed `v`: `latency` at constant speed, then braking.
#[inline]
pub fn stopping_distance(v: f64, p: &SafetyParams) -> f64 {
    v * p.latency + v * v / (2.0 * p.decel)
}

/// Points the robot must keep clear of, one per scan sector (robot frame).
/// `near_m`: where to put obstacles that run off the image bottom (`near`: the camera cannot see how close
/// they are; e.g. a chair leg 0.2 m away when the image bottom is at 0.44 m). None = at the closest visible
/// point (optimistic); the collision monitor and the heading choice put them at the clearance radius.
/// `near_mask`: per sector, only these near sectors get `near_m` (None = all).
/// A seen sector whose free distance is NaN gives a NaN point, which the arc checks ignore.
pub fn boundary_points(
    scan: &Scan,
    unseen_m: f64,
    near_m: Option<f64>,
    near_mask: Option<&[bool]>,
) -> Vec<Point2> {
    let free = scan.free(); // NaN in unseen sectors, replaced below
    (0..scan.len())
        .map(|k| {
            let mut d = if scan.seen[k] { free[k] } else { unseen_m };
            if let Some(near_m) = near_m {
                if scan.seen[k] && scan.near[k] && near_mask.is_none_or(|m| m[k]) {
                    d = near_m;
                }
            }
            let b = scan.bearing_deg[k].to_radians();
            [d * b.cos(), d * b.sin()]
        })
        .collect()
}

/// Robot centre after the arc length `s` on a path of curvature `curv` (1/m, + = left), from the origin
/// heading +x: a circle of radius 1 / curv centred at (0, 1 / curv).
#[inline]
fn arc_point(curv: f64, s: f64) -> Point2 {
    if curv.abs() < STRAIGHT_CURV {
        [s, 0.0]
    } else {
        [(curv * s).sin() / curv, (1.0 - (curv * s).cos()) / curv]
    }
}

/// [`arc_point`] for each arc length of `s`.
pub fn arc(curv: f64, s: &[f64]) -> Vec<Point2> {
    s.iter().map(|&s| arc_point(curv, s)).collect()
}

/// Number of arc samples looking `s_max` ahead: the length of numpy's arange(step, s_max + step, step),
/// ceil((stop - start) / step) in floating point (so the last sample may lie up to one step past s_max).
#[inline]
fn n_samples(step: f64, s_max: f64) -> usize {
    let n = ((s_max + step - step) / step).ceil();
    if n > 0.0 {
        n as usize
    } else {
        0 // also NaN
    }
}

/// Points as separate x / y arrays: the distance loop over them vectorises.
struct Points {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Points {
    fn with_capacity(n: usize) -> Self {
        Self {
            x: Vec::with_capacity(n),
            y: Vec::with_capacity(n),
        }
    }

    /// Add the points within `reach` of the origin; farther ones (and NaN ones) cannot be hit.
    fn extend_within(&mut self, points: &[Point2], reach: f64) {
        for p in points {
            if p[0].hypot(p[1]) <= reach {
                self.x.push(p[0]);
                self.y.push(p[1]);
            }
        }
    }

    /// Arc length travelled at curvature `curv` before the sample where some point is in collision
    /// (`hit(squared distance to the robot centre)`), over `n` samples one `step` apart; None if no sample
    /// collides. The first sample is one step ahead: where the robot stands is not checked.
    #[inline]
    fn free_length(&self, curv: f64, step: f64, n: usize, hit: impl Fn(f64) -> bool) -> Option<f64> {
        for i in 0..n {
            let s = step + i as f64 * step;
            let [cx, cy] = arc_point(curv, s);
            // no early exit over the points: a branch-free loop is what the compiler vectorises
            let mut any = false;
            for (&x, &y) in self.x.iter().zip(&self.y) {
                let (dx, dy) = (cx - x, cy - y);
                any |= hit(dx * dx + dy * dy);
            }
            if any {
                return Some(s - step);
            }
        }
        None
    }
}

/// Arc length the centre can travel at curvature `curv` before coming within radius + margin of a point,
/// looking `s_max` ahead (None: [`SafetyParams::lookahead`], enough for the safety check).
/// NaN points are ignored, as in [`free_arc_lengths`].
pub fn free_arc_length(points: &[Point2], curv: f64, p: &SafetyParams, s_max: Option<f64>) -> f64 {
    let s_max = s_max.unwrap_or_else(|| p.lookahead());
    let clear = p.clearance();
    let mut pts = Points::with_capacity(points.len());
    pts.extend_within(points, f64::INFINITY);
    // the prototype's single-arc function compares the distance itself; the batch one, its square
    pts.free_length(curv, p.step, n_samples(p.step, s_max), |d2| d2.sqrt() < clear)
        .unwrap_or(s_max)
}

/// [`free_arc_length`] for several curvatures: what the collision monitor (one arc per candidate speed) and
/// a controller scoring arcs need every frame. Points farther than s_max + clearance cannot be reached and
/// are dropped first.
pub fn free_arc_lengths(points: &[Point2], curvs: &[f64], p: &SafetyParams, s_max: f64) -> Vec<f64> {
    let clear = p.clearance();
    let mut pts = Points::with_capacity(points.len());
    pts.extend_within(points, s_max + clear + p.step);
    let (n, clear2) = (n_samples(p.step, s_max), clear * clear);
    curvs
        .iter()
        .map(|&curv| {
            pts.free_length(curv, p.step, n, |d2| d2 < clear2)
                .unwrap_or(s_max)
        })
        .collect()
}

/// (v, w) requested -> (v, w) allowed, plus the free arc length at the allowed speed (m, for display).
/// `extra_points`: more points to keep clear of, robot frame (the occupancy grid's obstacles).
/// `near_known`: per sector, blind segment already observed by the map: those near sectors are not put at
/// radius + margin. Pass it only with the same map's `extra_points`.
///
/// The allowed v is the largest of `n_speeds` candidates, evenly spaced from min(v, v_max) down to (not
/// including) 0, whose stopping distance fits in the free length of its arc; 0 if none does.
pub fn safe_command(
    v: f64,
    w: f64,
    scan: &Scan,
    p: &SafetyParams,
    extra_points: Option<&[Point2]>,
    near_known: Option<&[bool]>,
) -> (f64, f64, f64) {
    if v <= 0.0 {
        let lowest = -p.reverse_max;
        return (if lowest > v { lowest } else { v }, w, 0.0);
    }
    let clear = p.clearance();
    let not_known: Option<Vec<bool>> = near_known.map(|k| k.iter().map(|&k| !k).collect());
    let boundary = boundary_points(scan, p.unseen_m, Some(clear), not_known.as_deref());
    let s_max = p.lookahead();
    let extra = extra_points.unwrap_or(&[]);
    let mut pts = Points::with_capacity(boundary.len() + extra.len());
    pts.extend_within(&boundary, s_max + clear + p.step);
    pts.extend_within(extra, s_max + clear + p.step);
    let (n, clear2) = (n_samples(p.step, s_max), clear * clear);

    // Candidate speeds, largest first: i * dv + start, as numpy's linspace computes them.
    let start = if p.v_max < v { p.v_max } else { v };
    let dv = (0.0 - start) / p.n_speeds as f64;
    let mut last: Option<(f64, f64)> = None; // (curvature, free length) of the previous candidate
    for i in 0..p.n_speeds {
        let cand = i as f64 * dv + start;
        let curv = w / cand;
        let s_free = match last {
            // w = 0: every candidate follows the same straight line
            Some((c, s)) if c == curv => s,
            _ => pts
                .free_length(curv, p.step, n, |d2| d2 < clear2)
                .unwrap_or(s_max),
        };
        if stopping_distance(cand, p) <= s_free {
            return (cand, w, s_free);
        }
        last = Some((curv, s_free));
    }
    (0.0, w, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic scan, 2 deg sectors: sectors within +-fov seen, free distance `free_at(bearing)` (an
    /// obstacle if < 2.5 m).
    fn scan(free_at: impl Fn(f64) -> f64, fov: f64) -> Scan {
        let bearing_deg: Vec<f64> = (0..180).map(|k| (k as f64 + 0.5) * 2.0 - 180.0).collect();
        let d: Vec<f64> = bearing_deg.iter().map(|&b| free_at(b)).collect();
        Scan {
            seen: bearing_deg.iter().map(|b| b.abs() < fov).collect(),
            obstacle: d
                .iter()
                .map(|&d| if d < 2.5 { d } else { f64::INFINITY })
                .collect(),
            near: vec![false; 180],
            floor: vec![2.5; 180],
            closest: vec![f64::INFINITY; 180],
            bearing_deg,
        }
    }

    /// The same scan with every obstacle running off the image bottom.
    fn all_near(sc: &Scan) -> Scan {
        let mut near = sc.clone();
        near.near = near.obstacle.iter().map(|o| o.is_finite()).collect();
        near
    }

    fn v_safe(v: f64, w: f64, sc: &Scan) -> f64 {
        safe_command(v, w, sc, &SafetyParams::default(), None, None).0
    }

    #[test]
    fn sample_count_is_numpy_arange() {
        // len(np.arange(step, s_max + step, step)), from numpy
        assert_eq!(n_samples(0.02, 0.32), 16);
        assert_eq!(n_samples(0.02, 0.837), 42);
        assert_eq!(n_samples(0.02, 0.3), 15);
        assert_eq!(n_samples(0.02, 0.0), 0);
        assert_eq!(n_samples(0.02, -1.0), 0);
        assert_eq!(n_samples(0.02, f64::NAN), 0);
    }

    #[test]
    fn arc_is_a_circle_tangent_to_x() {
        let s = [0.1, 0.5, 1.0];
        assert_eq!(arc(0.0, &s), [[0.1, 0.0], [0.5, 0.0], [1.0, 0.0]]);
        for curv in [2.0, -0.5] {
            for q in arc(curv, &s) {
                // on the circle of radius 1 / |curv| centred at (0, 1 / curv)
                let r = q[0].hypot(q[1] - 1.0 / curv);
                assert!((r - 1.0 / curv.abs()).abs() < 1e-12);
                assert_eq!(q[1] > 0.0, curv > 0.0); // + = left
            }
        }
        // quarter turn to the left at radius 1
        let q = arc(1.0, &[std::f64::consts::FRAC_PI_2])[0];
        assert!((q[0] - 1.0).abs() < 1e-12 && (q[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn free_length_is_the_first_hit_minus_one_step() {
        let p = SafetyParams::default();
        // a point 0.51 m ahead: samples every 0.02 m; the first within 0.18 m of it is at 0.34
        let pts = [[0.51, 0.0]];
        let s = free_arc_length(&pts, 0.0, &p, Some(1.0));
        assert!((s - 0.32).abs() < 1e-12, "{s}");
        assert_eq!(free_arc_lengths(&pts, &[0.0], &p, 1.0), [s]);
        // out of the look-ahead: s_max itself
        assert_eq!(free_arc_length(&pts, 0.0, &p, Some(0.2)), 0.2);
        assert_eq!(free_arc_length(&[], 0.0, &p, None), p.lookahead());
        // a point to the left stops the left arc only
        let left = [[0.3, 0.3]];
        let got = free_arc_lengths(&left, &[2.0, 0.0, -2.0], &p, 0.6);
        assert!(got[0] < 0.6 && got[1] == 0.6 && got[2] == 0.6, "{got:?}");
        // NaN and far points are ignored
        let odd = [[f64::NAN, 0.0], [f64::INFINITY, 0.0], [5.0, 0.0]];
        assert_eq!(free_arc_lengths(&odd, &[0.0], &p, 0.6), [0.6]);
        assert_eq!(free_arc_length(&odd, 0.0, &p, Some(0.6)), 0.6);
    }

    #[test]
    fn boundary_points_follow_the_scan() {
        let sc = all_near(&scan(|b| if b.abs() < 6.0 { 0.45 } else { 2.5 }, 38.0));
        let range = |pts: &[Point2], k: usize| pts[k][0].hypot(pts[k][1]);
        let plain = boundary_points(&sc, 0.35, None, None);
        assert!((range(&plain, 0) - 0.35).abs() < 1e-12); // behind: unseen
        assert!((range(&plain, 90) - 0.45).abs() < 1e-12); // ahead: the obstacle
        assert!((range(&plain, 100) - 2.5).abs() < 1e-12); // 21 deg left: floor to max range
        assert!(plain[100][1] > 0.0 && plain[80][1] < 0.0); // + bearing = left
        let near = boundary_points(&sc, 0.35, Some(0.18), None);
        assert!((range(&near, 90) - 0.18).abs() < 1e-12 && (range(&near, 100) - 2.5).abs() < 1e-12);
        let mut mask = vec![true; 180];
        mask[90] = false;
        let masked = boundary_points(&sc, 0.35, Some(0.18), Some(&mask));
        assert!((range(&masked, 90) - 0.45).abs() < 1e-12 && (range(&masked, 89) - 0.18).abs() < 1e-12);
    }

    #[test]
    fn open_floor_allows_full_speed() {
        let p = SafetyParams::default();
        let got = safe_command(0.4, 0.0, &scan(|_| 2.5, 38.0), &p, None, None);
        assert_eq!(got, (0.4, 0.0, p.lookahead()));
    }

    #[test]
    fn wall_ahead_slows_then_stops() {
        let p = SafetyParams::default();
        // 0.15 m: already inside radius + margin (0.18 m) -> no forward motion at all
        let speeds: Vec<f64> = [1.0, 0.35, 0.25, 0.15]
            .iter()
            .map(|&d| v_safe(0.4, 0.0, &scan(|_| d, 38.0)))
            .collect();
        assert_eq!(speeds[0], 0.4); // 1 m: the stopping distance at 0.4 m/s is 0.22 m
        assert!(0.4 > speeds[1] && speeds[1] > speeds[2] && speeds[2] > speeds[3] && speeds[3] == 0.0);
        // the allowed speed stops before radius + margin from the wall; the next candidate would not
        let room = 0.35 - p.radius - p.margin;
        assert!(stopping_distance(speeds[1], &p) <= room);
        assert!(stopping_distance(speeds[1] + 0.4 / p.n_speeds as f64, &p) > room - p.step);
    }

    #[test]
    fn unseen_sides_and_side_obstacles_block_the_arcs_toward_them() {
        // radius 0.1 m arc to the right: reaches the unseen side within the stopping distance
        assert!(v_safe(0.4, -4.0, &scan(|_| 2.5, 38.0)) < 0.4);
        assert_eq!(v_safe(0.4, -4.0, &scan(|_| 2.5, 180.0)), 0.4);
        // everything seen, obstacle on the left at 0.3 m, bearings 60-80 deg
        let sc = scan(|b| if 60.0 < b && b < 80.0 { 0.3 } else { 2.5 }, 180.0);
        assert_eq!(v_safe(0.4, 0.0, &sc), 0.4);
        assert!(v_safe(0.4, 2.67, &sc) < 0.4);
        assert_eq!(v_safe(0.4, -2.67, &sc), 0.4);
    }

    #[test]
    fn near_obstacle_blocks_unless_its_blind_segment_is_known() {
        let p = SafetyParams::default();
        let sc = scan(|b| if b.abs() < 6.0 { 0.45 } else { 2.5 }, 38.0);
        let near = all_near(&sc);
        assert_eq!(v_safe(0.3, 0.0, &sc), 0.3); // its foot is seen: room to stop
        assert_eq!(v_safe(0.3, 0.0, &near), 0.0); // may be much closer than its visible part
        let known = vec![true; 180];
        assert_eq!(safe_command(0.3, 0.0, &near, &p, None, Some(&known)).0, 0.3);
        // the map's own obstacle in the blind segment then stops it
        let extra = [[0.15, 0.0]];
        assert_eq!(
            safe_command(0.3, 0.0, &near, &p, Some(&extra), Some(&known)).0,
            0.0
        );
        assert_eq!(safe_command(0.3, 0.0, &sc, &p, Some(&extra), None).0, 0.0);
    }

    #[test]
    fn turn_in_place_and_reverse() {
        let p = SafetyParams::default();
        let (wall, open) = (scan(|_| 0.2, 38.0), scan(|_| 2.5, 38.0));
        assert_eq!(safe_command(0.0, 1.0, &wall, &p, None, None), (0.0, 1.0, 0.0)); // may still turn
        assert_eq!(v_safe(-0.2, 0.0, &open), 0.0); // no camera behind
        let rev = SafetyParams {
            reverse_max: 0.1,
            ..p
        };
        assert_eq!(safe_command(-0.2, 0.3, &open, &rev, None, None), (-0.1, 0.3, 0.0));
        assert_eq!(safe_command(-0.05, 0.3, &open, &rev, None, None).0, -0.05);
    }

    #[test]
    fn request_above_v_max_is_capped_and_the_result_is_a_candidate() {
        let p = SafetyParams::default();
        assert_eq!(v_safe(1.0, 0.0, &scan(|_| 2.5, 38.0)), 0.4);
        // every allowed speed is one of the n_speeds candidates of the request
        for d in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5] {
            let v = v_safe(0.3, 0.2, &scan(|_| d, 38.0));
            let i = (0.3 - v) / (0.3 / p.n_speeds as f64);
            assert!(
                (i - i.round()).abs() < 1e-9 && i.round() <= p.n_speeds as f64,
                "{d} {v}"
            );
        }
    }
}
