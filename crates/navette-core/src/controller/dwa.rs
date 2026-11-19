//! Arc selection toward a goal heading (dynamic window approach).
//!
//! Per step, with the goal heading given by a [`GoalSource`] (a fixed-frame heading, so it stays put while
//! the robot turns):
//! 1. SPIN   goal more than `spin_above_deg` off the current heading: turn in place until within
//!    `spin_done_deg`.
//! 2. ARCS   otherwise try (v, w) pairs: v in `v_set`, w in [-w_max, w_max]. An arc is admissible if the
//!    robot can stop on it (the collision monitor's stopping model, against the scan's boundary and the
//!    map's obstacles). Its score is the sum of k_heading cos(goal error after horizon_s),
//!    k_clear min(free arc, clear_cap) / clear_cap and k_speed v / v_max. The best one wins: the robot curves
//!    toward the goal and away from obstacles, and slows down where it is tight.
//! 3. BLOCKED  no admissible arc, or no motion, for `blocked_s`: the goal source is told
//!    ([`GoalSource::on_blocked`]), asked for a goal again, and the robot turns to it.
//!
//! Numerics follow the Python prototype: candidates in the same order (v, then w as numpy's linspace) and a
//! strictly better score to replace the best, so the same arc wins ties.

use std::f64::consts::{PI, TAU};

use crate::collision_monitor::{boundary_points, free_arc_lengths, stopping_distance, SafetyParams};
use crate::controller::State;
use crate::geometry::{wrap_angle, Pose2};
use crate::occupancy_grid::OccupancyGrid;
use crate::params::{self, ParamError};
use crate::scan::Scan;

/// Most yaw rate candidates (`n_w`) accepted by [`ExploreParams::validate`].
const MAX_N_W: usize = 10_000;
/// Range (m) of the map obstacles the arcs are checked against.
const MAP_RANGE_M: f64 = 1.5;

/// Arc controller parameters. The defaults are the ones tuned on Reachy Mini.
#[derive(Clone, Debug, PartialEq)]
pub struct ExploreParams {
    /// Candidate forward speeds (m/s).
    pub v_set: Vec<f64>,
    /// Candidate yaw rates span [-w_max, w_max] (rad/s),
    pub w_max: f64,
    /// in this many values.
    pub n_w: usize,
    /// The heading error is evaluated after this long on the arc (s).
    pub horizon_s: f64,
    pub k_heading: f64,
    pub k_clear: f64,
    pub k_speed: f64,
    /// Free arc length (m) that counts as fully clear.
    pub clear_cap: f64,
    /// Goal farther than this off the heading (deg): turn in place,
    pub spin_above_deg: f64,
    /// until within this (deg).
    pub spin_done_deg: f64,
    /// Yaw rate of the turns in place (rad/s).
    pub w_spin: f64,
    /// No admissible arc or no motion for this long (s): blocked.
    pub blocked_s: f64,
    /// No admissible arc but the goal more than `spin_done_deg` off: turn toward it at once (false: wait
    /// for the blocked timer).
    pub rotate_in_place: bool,
    /// Applied speed (m/s) from which the robot counts as moving (below it for `blocked_s`: blocked). It
    /// must stay under the slowest speed the robot is ever sent, or a slow robot is always "blocked".
    pub moving_v: f64,
    /// The collision monitor's `near` rule for the arcs: an obstacle running off the image bottom is put at
    /// the clearance radius only in the sectors whose blind segment the map does not explain (the
    /// `near_known` mask given to [`ArcController::step`]); explained sectors keep their visible distance and
    /// the map's obstacles cover the blind zone. False: every near sector at the clearance radius (the
    /// prototype). Why: on the real robot 14 of 15 blocks came from the current scan's near sectors (walls
    /// at an angle in a corridor, bottles, shoes), and on 512 of the 1405 steps without an admissible arc the
    /// collision monitor, which applies this rule, would have allowed some.
    pub near_known: bool,
    /// Latched in-place turns (deg; 0: no latch, the prototype). A turn in place (SPIN, the rotate-in-place
    /// turn in DRIVE, the recovery turn) keeps the direction it started with until it ends; it reverses only
    /// if the goal heading has moved more than this since the turn started, or if the robot overshot the
    /// goal (the latched way round would be more than 180 deg + this). Without it the direction is
    /// sign(goal - heading) at every step, so a goal near 180 deg off or a replan reverses a turn half-way:
    /// ~16 reversals in a 4-minute run of the real robot.
    pub spin_flip_deg: f64,
    /// The end of a turn in place is tested on the heading predicted this far ahead (s): heading + measured
    /// yaw rate x spin_lead_s (0: the current heading, the prototype). The command reaches the base
    /// ~0.4-0.9 s after the frame was taken (at ~8 Hz), so a turn stopped on the current heading overshoots.
    pub spin_lead_s: f64,
    /// Slow down before the end of a turn in place (rad/s^2; 0: always w_spin, the prototype): |w| =
    /// min(w_spin, max(spin_w_min, sqrt(2 spin_accel |angle to go|))), as Nav2's rotation shim.
    pub spin_accel: f64,
    /// Lowest yaw rate (rad/s) of that slowdown, so that a turn never stalls.
    pub spin_w_min: f64,
    /// Blocked only after `blocked_s` without progress (false: without an admissible arc or below `moving_v`,
    /// the prototype). Progress: the pose moved `progress_m` or turned `progress_deg` since the last progress
    /// mark (Nav2's pose progress checker: a rotation counts). Why: with 1 s of no admissible arc the real
    /// robot started a recovery turn on every short hesitation of the arcs (15 blocks in 4 minutes), each
    /// one a turn away from the path and back.
    pub blocked_progress: bool,
    /// Progress (m),
    pub progress_m: f64,
    /// or (deg).
    pub progress_deg: f64,
    /// Recovery that does not fight the planner (false: the prototype's, 90 deg toward the side with more
    /// free floor in view). On a block: replan first (with this scan's near sectors), then turn toward the
    /// new path heading; if there is no path or it is within 20 deg of the blocked direction, 90 deg in the
    /// direction of the previous turn in place (else toward the freer side). The second block on a target
    /// still bans it. Why: on the real robot the recovery turned away from the path, the replan pointed
    /// back to the old way and the robot turned back: the visible back-and-forth.
    pub recover_replan: bool,
    /// After that recovery turn, for this long (s) the goal does not turn back toward the blocked direction:
    /// a path heading nearer to it than to the recovery heading is replaced by the recovery heading. 0: no hold.
    pub recover_hold_s: f64,
}

impl Default for ExploreParams {
    fn default() -> Self {
        Self {
            v_set: vec![0.1, 0.2, 0.3],
            w_max: 1.2,
            n_w: 13,
            horizon_s: 1.0,
            k_heading: 1.0,
            k_clear: 0.6,
            k_speed: 0.3,
            clear_cap: 1.0,
            spin_above_deg: 70.0,
            spin_done_deg: 20.0,
            w_spin: 1.0,
            blocked_s: 1.0,
            rotate_in_place: false,
            moving_v: 0.05,
            near_known: false,
            spin_flip_deg: 0.0,
            spin_lead_s: 0.0,
            spin_accel: 0.0,
            spin_w_min: 0.3,
            blocked_progress: false,
            progress_m: 0.05,
            progress_deg: 17.0, // ~0.3 rad
            recover_replan: false,
            recover_hold_s: 0.0,
        }
    }
}

impl ExploreParams {
    /// Every value finite; v_set not empty, its speeds > 0 (w / v is the arc's curvature); n_w >= 1;
    /// clear_cap > 0; w_spin and spin_w_min > 0 (with 0 the look-around or a turn may never end); the others
    /// >= 0, the score weights any.
    pub fn validate(&self) -> Result<(), ParamError> {
        if self.v_set.is_empty() {
            return Err(ParamError("ExploreParams.v_set: must not be empty".into()));
        }
        for &v in &self.v_set {
            params::positive("ExploreParams.v_set", v)?;
        }
        params::count("ExploreParams.n_w", self.n_w, MAX_N_W)?;
        params::positive("ExploreParams.clear_cap", self.clear_cap)?;
        params::positive("ExploreParams.w_spin", self.w_spin)?;
        params::positive("ExploreParams.spin_w_min", self.spin_w_min)?;
        for (name, value) in [
            ("w_max", self.w_max),
            ("horizon_s", self.horizon_s),
            ("spin_above_deg", self.spin_above_deg),
            ("spin_done_deg", self.spin_done_deg),
            ("blocked_s", self.blocked_s),
            ("moving_v", self.moving_v),
            ("spin_flip_deg", self.spin_flip_deg),
            ("spin_lead_s", self.spin_lead_s),
            ("spin_accel", self.spin_accel),
            ("progress_m", self.progress_m),
            ("progress_deg", self.progress_deg),
            ("recover_hold_s", self.recover_hold_s),
        ] {
            params::non_negative(&format!("ExploreParams.{name}"), value)?;
        }
        for (name, value) in [
            ("k_heading", self.k_heading),
            ("k_clear", self.k_clear),
            ("k_speed", self.k_speed),
        ] {
            params::finite(&format!("ExploreParams.{name}"), value)?;
        }
        Ok(())
    }
}

/// Where the controller should head: the behaviour on top of it.
pub trait GoalSource {
    /// Fixed-frame heading (rad) to steer to, at time `t` from `pose`.
    fn choose_goal(&mut self, t: f64, pose: Pose2, grid: &OccupancyGrid) -> f64;

    /// The way to the last goal is blocked. The controller asks for a goal again right after.
    fn on_blocked(&mut self, t: f64, pose: Pose2);
}

/// A goal heading that never changes.
pub struct FixedGoal(pub f64);

impl GoalSource for FixedGoal {
    fn choose_goal(&mut self, _t: f64, _pose: Pose2, _grid: &OccupancyGrid) -> f64 {
        self.0
    }

    fn on_blocked(&mut self, _t: f64, _pose: Pose2) {}
}

/// The `num` values of numpy.linspace(start, stop, num): start + i * step, and `stop` itself for the last.
fn linspace(start: f64, stop: f64, num: usize) -> impl Iterator<Item = f64> {
    let step = (stop - start) / (num as f64 - 1.0);
    (0..num).map(move |i| match i {
        0 => start, // also num = 1, where the step is not a number
        i if i == num - 1 => stop,
        i => i as f64 * step + start,
    })
}

/// A turn in place under way, with `spin_flip_deg` > 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Turn {
    /// +1 counter-clockwise (left), -1 clockwise.
    pub dir: f64,
    /// The goal heading when the turn (or its last reversal) started (fixed frame, rad).
    pub goal: f64,
}

/// Angle (rad, in [0, 2 pi)) left to turn in direction `dir` (+1 left, -1 right) to cancel the heading
/// error `err` (goal - heading, wrapped).
fn to_go(err: f64, dir: f64) -> f64 {
    (err * dir).rem_euclid(TAU)
}

/// The command (0, w) of a turn in place toward `goal`, `err` off the heading. `dir0`: the direction the
/// prototype takes (the sign of err). With `spin_flip_deg` > 0 the direction of `latch` is kept unless the
/// goal moved more than that since it was set, or the latched way round is more than 180 deg + that (an
/// overshoot); otherwise the latch is set to `dir0`. With `spin_accel` > 0, |w| slows down near the goal.
fn turn(p: &ExploreParams, latch: &mut Option<Turn>, goal: f64, err: f64, dir0: f64) -> (f64, f64) {
    let dir = if p.spin_flip_deg > 0.0 {
        let flip = p.spin_flip_deg.to_radians();
        match *latch {
            Some(l) if wrap_angle(goal - l.goal).abs() <= flip && to_go(err, l.dir) <= PI + flip => l.dir,
            _ => {
                *latch = Some(Turn { dir: dir0, goal });
                dir0
            }
        }
    } else {
        dir0
    };
    let w = if p.spin_accel > 0.0 {
        let slowed = (2.0 * p.spin_accel * to_go(err, dir)).sqrt();
        p.w_spin.min(slowed.max(p.spin_w_min))
    } else {
        p.w_spin
    };
    (0.0, w.copysign(dir))
}

/// See the module documentation. One controller per robot; call [`ArcController::step`] once per frame.
#[derive(Clone, Debug)]
pub struct ArcController {
    pub params: ExploreParams,
    pub safety: SafetyParams,
    /// The goal heading of the last step (fixed frame, rad).
    pub goal: Option<f64>,
    /// Drive or Spin. The behaviour on top may set its own states (Look, Done) between steps.
    pub state: State,
    /// Since when (s) there has been no admissible arc or no motion (`blocked_progress`: no progress).
    pub blocked_since: Option<f64>,
    /// The latched turn in place under way (`spin_flip_deg` > 0), None when not turning.
    pub turn: Option<Turn>,
    /// With `blocked_progress`: the time and pose of the last progress (None: from the next step on).
    pub progress: Option<(f64, Pose2)>,
}

impl ArcController {
    /// An error if a parameter is invalid ([`ExploreParams::validate`], [`SafetyParams::validate`]).
    pub fn new(params: ExploreParams, safety: SafetyParams) -> Result<Self, ParamError> {
        params.validate()?;
        safety.validate()?;
        Ok(Self {
            params,
            safety,
            goal: None,
            state: State::Drive,
            blocked_since: None,
            turn: None,
            progress: None,
        })
    }

    /// Requested (v, w) at time `t` (s) from `pose`. `scan`: this frame's (None: the arcs are checked
    /// against the map alone). `v_applied`: the speed command sent last step (m/s). `w_meas`: the measured
    /// yaw rate (rad/s), for the end of the turns in place (`spin_lead_s`). `near_known`: per scan sector,
    /// its blind segment is explained by the map (`OccupancyGrid::blind_known`, asked before the frame went
    /// into the map); used only with `params.near_known`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        goals: &mut impl GoalSource,
        t: f64,
        pose: Pose2,
        scan: Option<&Scan>,
        grid: &OccupancyGrid,
        v_applied: f64,
        w_meas: f64,
        near_known: Option<&[bool]>,
    ) -> (f64, f64) {
        let goal = goals.choose_goal(t, pose, grid);
        self.goal = Some(goal);
        let (p, s) = (&self.params, &self.safety);
        if p.blocked_progress {
            // a new progress mark once the robot moved or turned enough since the last one
            let moved = self.progress.is_none_or(|(_, m)| {
                (pose.x - m.x).hypot(pose.y - m.y) >= p.progress_m
                    || wrap_angle(pose.theta - m.theta).abs() >= p.progress_deg.to_radians()
            });
            if moved {
                self.progress = Some((t, pose));
            }
        }
        let err = wrap_angle(goal - pose.theta);
        // the end of a turn is tested on the heading the robot will have when this command takes effect
        let err_end = if p.spin_lead_s > 0.0 {
            wrap_angle(goal - (pose.theta + w_meas * p.spin_lead_s))
        } else {
            err
        };
        let dir0 = 1f64.copysign(err);
        if self.state == State::Spin {
            if err_end.abs() > p.spin_done_deg.to_radians() {
                return turn(p, &mut self.turn, goal, err, dir0);
            }
            self.state = State::Drive;
        }
        if err.abs() > p.spin_above_deg.to_radians() {
            (self.state, self.blocked_since) = (State::Spin, None);
            return turn(p, &mut self.turn, goal, err, dir0);
        }

        // an obstacle running off the image bottom may be anywhere in the blind zone: at the clearance radius
        // (with params.near_known: unless the map explains that sector, as the collision monitor does)
        let not_known: Option<Vec<bool>> = near_known
            .filter(|_| p.near_known)
            .map(|k| k.iter().map(|&k| !k).collect());
        let mut pts = scan.map_or_else(Vec::new, |sc| {
            boundary_points(sc, s.unseen_m, Some(s.clearance()), not_known.as_deref())
        });
        pts.extend(grid.obstacle_points(pose, MAP_RANGE_M));
        let cand: Vec<(f64, f64)> = p
            .v_set
            .iter()
            .flat_map(|&v| linspace(-p.w_max, p.w_max, p.n_w).map(move |w| (v, w)))
            .collect();
        let curvs: Vec<f64> = cand.iter().map(|&(v, w)| w / v).collect();
        let free = free_arc_lengths(&pts, &curvs, s, p.clear_cap);
        let (mut best, mut best_score) = (None, f64::NEG_INFINITY);
        for (&(v, w), &s_free) in cand.iter().zip(&free) {
            if stopping_distance(v, s) > s_free {
                continue;
            }
            let e = wrap_angle(err - w * p.horizon_s);
            let score = p.k_heading * e.cos()
                + p.k_clear * s_free.min(p.clear_cap) / p.clear_cap
                + p.k_speed * v / s.v_max;
            if score > best_score {
                (best, best_score) = (Some((v, w)), score);
            }
        }
        if best.is_none() && p.rotate_in_place && err_end.abs() > p.spin_done_deg.to_radians() {
            // Turning in place is always safe for a round robot. It is progress toward the goal, so it does
            // not count as blocked; once facing the goal with still no arc, it does.
            self.blocked_since = None;
            return turn(p, &mut self.turn, goal, err, dir0);
        }
        if p.blocked_progress {
            self.blocked_since = self.progress.map(|(since, _)| since); // no progress since then
        } else if best.is_some() && v_applied >= p.moving_v {
            self.blocked_since = None;
        } else if self.blocked_since.is_none() {
            self.blocked_since = Some(t);
        }
        if self.blocked_since.is_some_and(|since| t - since > p.blocked_s) {
            // this way is blocked: pick another goal and turn to it
            goals.on_blocked(t, pose);
            self.blocked_since = None;
            if p.blocked_progress {
                self.progress = Some((t, pose)); // the recovery starts the count again
            }
            let goal = goals.choose_goal(t, pose, grid);
            self.goal = Some(goal);
            self.state = State::Spin;
            let err = wrap_angle(goal - pose.theta);
            let dir0 = if err == 0.0 { 1.0 } else { 1f64.copysign(err) };
            self.turn = None; // a new turn: its own direction
            return turn(p, &mut self.turn, goal, err, dir0);
        }
        self.turn = None; // not turning in place
        best.unwrap_or((0.0, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::occupancy_grid::OccupancyGridParams;

    /// 2 deg sectors; sectors within `fov` deg of straight ahead seen from `closest`, obstacle at
    /// `obst_at(bearing)` (inf = none), floor to 2.5 m.
    fn scan(obst_at: impl Fn(f64) -> f64, fov: f64, closest: f64) -> Scan {
        let bearing_deg: Vec<f64> = (0..180).map(|k| (k as f64 + 0.5) * 2.0 - 180.0).collect();
        Scan {
            seen: bearing_deg.iter().map(|b| b.abs() < fov).collect(),
            obstacle: bearing_deg.iter().map(|&b| obst_at(b)).collect(),
            near: vec![false; 180],
            floor: vec![2.5; 180],
            closest: vec![closest; 180],
            bearing_deg,
        }
    }

    fn open() -> Scan {
        scan(|_| f64::INFINITY, 38.0, 0.4)
    }

    const HERE: Pose2 = Pose2 {
        x: 0.0,
        y: 0.0,
        theta: 0.0,
    };

    /// One step of a new controller toward the fixed-frame heading `goal_deg`, on an empty map.
    fn step(params: ExploreParams, goal_deg: f64, sc: &Scan, v_applied: f64) -> ((f64, f64), ArcController) {
        let mut c = ArcController::new(params, SafetyParams::default()).unwrap();
        let grid = OccupancyGrid::new(OccupancyGridParams::default()).unwrap();
        let mut goal = FixedGoal(goal_deg.to_radians());
        (
            c.step(&mut goal, 0.0, HERE, Some(sc), &grid, v_applied, 0.0, None),
            c,
        )
    }

    #[test]
    fn bad_parameters_are_errors() {
        let ok = ExploreParams::default;
        assert_eq!(ok().validate(), Ok(()));
        let nan = f64::NAN;
        for p in [
            ExploreParams {
                v_set: vec![],
                ..ok()
            },
            ExploreParams {
                v_set: vec![0.1, 0.0],
                ..ok()
            },
            ExploreParams {
                v_set: vec![-0.1],
                ..ok()
            },
            ExploreParams {
                v_set: vec![nan],
                ..ok()
            },
            ExploreParams { n_w: 0, ..ok() },
            ExploreParams { w_max: -1.0, ..ok() },
            ExploreParams { w_max: nan, ..ok() },
            ExploreParams {
                clear_cap: 0.0,
                ..ok()
            },
            ExploreParams { w_spin: 0.0, ..ok() },
            ExploreParams {
                blocked_s: nan,
                ..ok()
            },
            ExploreParams { k_speed: nan, ..ok() },
        ] {
            let err = ArcController::new(p, SafetyParams::default()).expect_err("invalid");
            assert!(err.0.starts_with("ExploreParams."), "{err}");
        }
        let safety = SafetyParams {
            step: 0.0,
            ..Default::default()
        };
        assert!(ArcController::new(ok(), safety).is_err());
    }

    #[test]
    fn moving_threshold_is_a_parameter() {
        // a slow robot: one candidate at 0.04 m/s, applied as asked. With the default threshold (0.05) it
        // never counts as moving and is "blocked" on open floor; with a threshold under its speed it is not.
        for (moving_v, blocked) in [(0.05, true), (0.02, false)] {
            let params = ExploreParams {
                v_set: vec![0.04],
                moving_v,
                ..Default::default()
            };
            let ((v, _), c) = step(params, 0.0, &open(), 0.04);
            assert_eq!(v, 0.04);
            assert_eq!(c.blocked_since.is_some(), blocked);
        }
    }

    #[test]
    fn linspace_is_numpy_linspace() {
        // numpy.linspace(-1.2, 1.2, 13).tolist()
        let want = [
            -1.2,
            -1.0,
            -0.8,
            -0.6,
            -0.4,
            -0.20000000000000007,
            0.0,
            0.19999999999999996,
            0.3999999999999999,
            0.5999999999999999,
            0.7999999999999998,
            0.9999999999999998,
            1.2,
        ];
        assert_eq!(linspace(-1.2, 1.2, 13).collect::<Vec<_>>(), want);
        assert_eq!(linspace(0.5, 2.0, 1).collect::<Vec<_>>(), [0.5]);
        assert_eq!(linspace(0.5, 2.0, 2).collect::<Vec<_>>(), [0.5, 2.0]);
    }

    #[test]
    fn open_floor_drives_straight_at_cruise() {
        let ((v, w), c) = step(ExploreParams::default(), 0.0, &open(), 0.3);
        assert_eq!((v, w), (0.3, 0.0));
        assert_eq!(
            (c.state, c.blocked_since, c.goal),
            (State::Drive, None, Some(0.0))
        );
    }

    #[test]
    fn steers_toward_the_goal_and_spins_when_it_is_far_off() {
        let ((v, w), _) = step(ExploreParams::default(), 30.0, &open(), 0.3);
        assert!(v > 0.0 && w > 0.0);
        let ((v, w), c) = step(ExploreParams::default(), -120.0, &open(), 0.3);
        assert_eq!(((v, w), c.state), ((0.0, -1.0), State::Spin));
    }

    #[test]
    fn a_spin_ends_within_spin_done_of_the_goal() {
        let mut c = ArcController::new(ExploreParams::default(), SafetyParams::default()).unwrap();
        let grid = OccupancyGrid::new(OccupancyGridParams::default()).unwrap();
        let mut goal = FixedGoal(90f64.to_radians());
        let at = |deg: f64| Pose2::new(0.0, 0.0, deg.to_radians());
        assert_eq!(
            c.step(&mut goal, 0.0, at(0.0), Some(&open()), &grid, 0.0, 0.0, None),
            (0.0, 1.0)
        );
        // 40 deg left: below spin_above, but the spin goes on until within spin_done
        assert_eq!(
            c.step(&mut goal, 0.1, at(50.0), Some(&open()), &grid, 0.0, 0.0, None),
            (0.0, 1.0)
        );
        let (v, _) = c.step(&mut goal, 0.2, at(75.0), Some(&open()), &grid, 0.0, 0.0, None);
        assert!(v > 0.0 && c.state == State::Drive);
    }

    #[test]
    fn curves_around_an_obstacle_instead_of_stopping() {
        // a box 0.8 m ahead, slightly left: keep moving, steer right
        let sc = scan(
            |b| if -4.0 < b && b < 14.0 { 0.8 } else { f64::INFINITY },
            38.0,
            0.4,
        );
        let ((v, w), _) = step(ExploreParams::default(), 0.0, &sc, 0.3);
        assert!(v > 0.0 && w < 0.0);
    }

    #[test]
    fn rotate_in_place_when_no_arc_and_goal_off_ahead() {
        // obstacle at 0.2 m all around (no admissible arc, even the tightest), goal 45 deg left (below
        // spin_above): without rotate_in_place the controller waits for the blocked timer; with it, it
        // turns toward the goal, which does not count as blocked
        let sc = scan(|_| 0.2, 181.0, 0.15);
        for (rotate_in_place, want) in [(false, (0.0, 0.0)), (true, (0.0, 1.0))] {
            let params = ExploreParams {
                rotate_in_place,
                ..Default::default()
            };
            let (cmd, c) = step(params, 45.0, &sc, 0.0);
            assert_eq!(cmd, want);
            assert_eq!(c.blocked_since.is_none(), rotate_in_place);
        }
    }

    #[test]
    fn near_known_puts_only_unexplained_near_sectors_at_the_clearance_radius() {
        // a wall at an angle touching the image bottom on the left (near, read at 0.4 m), open ahead
        let mut sc = scan(|b| if b > 10.0 { 0.4 } else { f64::INFINITY }, 38.0, 0.4);
        sc.near = sc.bearing_deg.iter().map(|&b| b > 10.0 && b < 38.0).collect();
        let grid = OccupancyGrid::new(OccupancyGridParams::default()).unwrap();
        let run = |near_known: bool, mask: Option<&[bool]>| {
            let params = ExploreParams {
                near_known,
                ..Default::default()
            };
            let mut c = ArcController::new(params, SafetyParams::default()).unwrap();
            c.step(&mut FixedGoal(0.0), 0.0, HERE, Some(&sc), &grid, 0.3, 0.0, mask)
        };
        let all = vec![true; 180];
        let none = vec![false; 180];
        // the prototype's rule (default): every near sector at the clearance radius, whatever the mask; the
        // wall left of the path is then 0.18 m from the centre: no arc
        let base = run(false, None);
        assert_eq!(base, (0.0, 0.0));
        assert_eq!(run(false, Some(&all)), base);
        // with near_known: the same when the map explains nothing (or no mask is given) ...
        assert_eq!(run(true, Some(&none)), base);
        assert_eq!(run(true, None), base);
        // ... and an arc once the map explains those sectors (they keep their visible 0.4 m)
        let (v, _) = run(true, Some(&all));
        assert!(v > 0.0);
    }

    /// Goal headings (deg) given one per step, the last one repeated.
    struct Goals(Vec<f64>, usize);

    impl GoalSource for Goals {
        fn choose_goal(&mut self, _t: f64, _pose: Pose2, _grid: &OccupancyGrid) -> f64 {
            let g = self.0[self.1.min(self.0.len() - 1)];
            self.1 += 1;
            g.to_radians()
        }

        fn on_blocked(&mut self, _t: f64, _pose: Pose2) {}
    }

    /// The yaw rates commanded with these params from these headings (deg) toward these goals (deg), the
    /// measured yaw rate `w_meas`, on open floor.
    fn turns(params: ExploreParams, headings: &[f64], goals: &[f64], w_meas: f64) -> Vec<f64> {
        let mut c = ArcController::new(params, SafetyParams::default()).unwrap();
        let grid = OccupancyGrid::new(OccupancyGridParams::default()).unwrap();
        let mut g = Goals(goals.to_vec(), 0);
        headings
            .iter()
            .enumerate()
            .map(|(k, &h)| {
                let pose = Pose2::new(0.0, 0.0, h.to_radians());
                let t = k as f64 * 0.1;
                c.step(&mut g, t, pose, Some(&open()), &grid, 0.0, w_meas, None).1
            })
            .collect()
    }

    fn latched() -> ExploreParams {
        ExploreParams {
            spin_flip_deg: 45.0,
            ..Default::default()
        }
    }

    #[test]
    fn a_latched_turn_toward_a_goal_behind_does_not_reverse() {
        // goal ~180 deg off, jittering across +-180 (a replan): the prototype follows sign(err) and reverses
        // at each crossing; the latched turn keeps its direction (the goal moved 4 deg < spin_flip_deg)
        let headings = [0.0; 4];
        let goals = [178.0, -178.0, 178.0, -178.0];
        assert_eq!(
            turns(ExploreParams::default(), &headings, &goals, 0.0),
            [1.0, -1.0, 1.0, -1.0]
        );
        assert_eq!(turns(latched(), &headings, &goals, 0.0), [1.0; 4]);
        // turning left toward 178, the heading moves on and the goal flips to -178: still left
        let headings = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0];
        let goals = [178.0, -178.0];
        assert_eq!(turns(latched(), &headings, &goals, 0.0), [1.0; 6]);
    }

    #[test]
    fn a_latched_turn_reverses_on_a_goal_switch_or_an_overshoot() {
        // the goal moves by more than spin_flip_deg (another target): the new turn takes its own direction
        assert_eq!(turns(latched(), &[0.0; 2], &[178.0, -100.0], 0.0), [1.0, -1.0]);
        // an overshoot: turning left toward 90, the heading ends up 40 deg past it (more than spin_done):
        // the way back is the short one, not 320 deg further left
        assert_eq!(turns(latched(), &[0.0, 130.0], &[90.0], 0.0), [1.0, -1.0]);
    }

    #[test]
    fn the_end_of_a_turn_is_tested_on_the_predicted_heading() {
        // spinning toward 90 deg, now at 64 deg (26 deg to go, more than spin_done = 20) turning at 0.5 rad/s
        let lead = ExploreParams {
            spin_lead_s: 0.5,
            ..Default::default()
        };
        // the prototype: on the current heading, the turn goes on
        let want = [1.0, 1.0];
        assert_eq!(turns(ExploreParams::default(), &[0.0, 64.0], &[90.0], 0.5), want);
        // 0.5 s ahead the heading is 64 + 14.3 deg: within spin_done, the turn ends (an arc, w < w_spin)
        let w = turns(lead.clone(), &[0.0, 64.0], &[90.0], 0.5);
        assert!(w[0] == 1.0 && w[1] < 1.0, "{w:?}");
        // without a measured yaw rate the prediction is the heading itself
        assert_eq!(turns(lead, &[0.0, 64.0], &[90.0], 0.0), want);
    }

    #[test]
    fn the_turn_slows_down_near_its_goal() {
        let slow = ExploreParams {
            spin_accel: 0.5,
            spin_w_min: 0.3,
            ..Default::default()
        };
        // far from the goal: w_spin (sqrt(2 x 0.5 x 2.97) > 1)
        assert_eq!(turns(slow.clone(), &[0.0], &[170.0], 0.0), [1.0]);
        // nearer: sqrt(2 a |angle to go|), down to spin_w_min
        let (p, mut latch) = (slow, None);
        let at = |deg: f64| (2.0 * 0.5 * deg.to_radians()).sqrt();
        assert_eq!(
            turn(&p, &mut latch, 0.0, 30f64.to_radians(), 1.0),
            (0.0, at(30.0))
        );
        assert_eq!(
            turn(&p, &mut latch, 0.0, -10f64.to_radians(), -1.0),
            (0.0, -at(10.0))
        );
        assert_eq!(turn(&p, &mut latch, 0.0, 2f64.to_radians(), 1.0), (0.0, 0.3)); // the floor
        assert_eq!(latch, None); // no latch without spin_flip_deg
                                 // the prototype: w_spin whatever is left
        let p = ExploreParams::default();
        assert_eq!(turn(&p, &mut latch, 0.0, 2f64.to_radians(), 1.0), (0.0, 1.0));
    }

    /// The steps at which the controller declared a block, over `n` steps 0.1 s apart from `pose_at(k)`,
    /// with no admissible arc (obstacle at 0.2 m all around) and nothing applied (v_applied = 0).
    fn blocks(params: ExploreParams, n: usize, pose_at: impl Fn(usize) -> Pose2) -> Vec<usize> {
        let sc = scan(|_| 0.2, 181.0, 0.15);
        let mut c = ArcController::new(params, SafetyParams::default()).unwrap();
        let grid = OccupancyGrid::new(OccupancyGridParams::default()).unwrap();
        let mut goals = Blocked(0);
        (0..n)
            .filter(|&k| {
                let before = goals.0;
                c.step(
                    &mut goals,
                    k as f64 * 0.1,
                    pose_at(k),
                    Some(&sc),
                    &grid,
                    0.0,
                    0.0,
                    None,
                );
                goals.0 > before
            })
            .collect()
    }

    fn progress() -> ExploreParams {
        ExploreParams {
            blocked_progress: true,
            blocked_s: 3.0,
            ..Default::default()
        }
    }

    #[test]
    fn blocked_only_after_blocked_s_without_progress() {
        let still = |_| HERE;
        // the prototype: blocked 1 s after the arcs ran out (then a recovery turn, spinning to its goal)
        assert_eq!(blocks(ExploreParams::default(), 12, still)[0], 11);
        // progress: blocked after 3 s without moving 5 cm or turning 17 deg
        assert_eq!(blocks(progress(), 40, still)[0], 31);
        // creeping 2 cm per step (no admissible arc, nothing applied): the prototype is blocked, not the
        // progress rule
        let creep = |k: usize| Pose2::new(0.02 * k as f64, 0.0, 0.0);
        assert_eq!(blocks(ExploreParams::default(), 12, creep), [11]);
        assert!(blocks(progress(), 100, creep).is_empty());
        // turning 0.1 rad per step counts too (Nav2's pose progress checker)
        let turning = |k: usize| Pose2::new(0.0, 0.0, wrap_angle(0.1 * k as f64));
        assert!(blocks(progress(), 100, turning).is_empty());
        // turning back and forth by 10 deg is no progress
        let dither = |k: usize| Pose2::new(0.0, 0.0, if k.is_multiple_of(2) { 0.0 } else { 0.17 });
        assert_eq!(blocks(progress(), 40, dither)[0], 31);
    }

    /// Counts the blocks, and then asks to turn right.
    struct Blocked(u32);

    impl GoalSource for Blocked {
        fn choose_goal(&mut self, _t: f64, _pose: Pose2, _grid: &OccupancyGrid) -> f64 {
            if self.0 == 0 {
                0.0
            } else {
                -1.0
            }
        }

        fn on_blocked(&mut self, _t: f64, _pose: Pose2) {
            self.0 += 1;
        }
    }

    #[test]
    fn blocked_for_blocked_s_asks_for_another_goal_and_turns_to_it() {
        // a post right in front: no admissible arc ahead
        let sc = scan(|_| 0.2, 181.0, 0.15);
        let mut c = ArcController::new(ExploreParams::default(), SafetyParams::default()).unwrap();
        let grid = OccupancyGrid::new(OccupancyGridParams::default()).unwrap();
        let mut goals = Blocked(0);
        for k in 0..=10 {
            assert_eq!(
                c.step(&mut goals, k as f64 * 0.1, HERE, Some(&sc), &grid, 0.0, 0.0, None),
                (0.0, 0.0)
            );
        }
        assert_eq!(c.blocked_since, Some(0.0));
        assert_eq!(
            c.step(&mut goals, 1.1, HERE, Some(&sc), &grid, 0.0, 0.0, None),
            (0.0, -1.0)
        );
        assert_eq!(
            (goals.0, c.state, c.blocked_since, c.goal),
            (1, State::Spin, None, Some(-1.0))
        );
    }

    #[test]
    fn standing_still_with_a_free_arc_counts_as_blocked_too() {
        let mut c = ArcController::new(ExploreParams::default(), SafetyParams::default()).unwrap();
        let grid = OccupancyGrid::new(OccupancyGridParams::default()).unwrap();
        let mut goal = FixedGoal(0.0);
        let (v, _) = c.step(&mut goal, 0.0, HERE, Some(&open()), &grid, 0.0, 0.0, None);
        assert!(v > 0.0 && c.blocked_since == Some(0.0)); // asks to go, but the base is not moving
        c.step(&mut goal, 0.1, HERE, Some(&open()), &grid, 0.2, 0.0, None);
        assert_eq!(c.blocked_since, None);
    }
}
