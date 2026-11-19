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

use crate::collision_monitor::{boundary_points, free_arc_lengths, stopping_distance, SafetyParams};
use crate::controller::State;
use crate::geometry::{wrap_angle, Pose2};
use crate::occupancy_grid::OccupancyGrid;
use crate::scan::Scan;

/// Range (m) of the map obstacles the arcs are checked against.
const MAP_RANGE_M: f64 = 1.5;
/// Applied speed (m/s) from which the robot counts as moving.
const MOVING_V: f64 = 0.05;

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
        }
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

/// See the module documentation. One controller per robot; call [`ArcController::step`] once per frame.
#[derive(Clone, Debug)]
pub struct ArcController {
    pub params: ExploreParams,
    pub safety: SafetyParams,
    /// The goal heading of the last step (fixed frame, rad).
    pub goal: Option<f64>,
    /// Drive or Spin. The behaviour on top may set its own states (Look, Done) between steps.
    pub state: State,
    /// Since when (s) there has been no admissible arc or no motion.
    pub blocked_since: Option<f64>,
}

impl ArcController {
    pub fn new(params: ExploreParams, safety: SafetyParams) -> Self {
        Self {
            params,
            safety,
            goal: None,
            state: State::Drive,
            blocked_since: None,
        }
    }

    /// Requested (v, w) at time `t` (s) from `pose`. `scan`: this frame's (None: the arcs are checked
    /// against the map alone). `v_applied`: the speed command sent last step (m/s).
    pub fn step(
        &mut self,
        goals: &mut impl GoalSource,
        t: f64,
        pose: Pose2,
        scan: Option<&Scan>,
        grid: &OccupancyGrid,
        v_applied: f64,
    ) -> (f64, f64) {
        let goal = goals.choose_goal(t, pose, grid);
        self.goal = Some(goal);
        let (p, s) = (&self.params, &self.safety);
        let err = wrap_angle(goal - pose.theta);
        let spin = (0.0, p.w_spin.copysign(err));
        if self.state == State::Spin {
            if err.abs() > p.spin_done_deg.to_radians() {
                return spin;
            }
            self.state = State::Drive;
        }
        if err.abs() > p.spin_above_deg.to_radians() {
            (self.state, self.blocked_since) = (State::Spin, None);
            return spin;
        }

        // an obstacle running off the image bottom may be anywhere in the blind zone: at the clearance radius
        let mut pts = scan.map_or_else(Vec::new, |sc| {
            boundary_points(sc, s.unseen_m, Some(s.clearance()), None)
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
        if best.is_none() && p.rotate_in_place && err.abs() > p.spin_done_deg.to_radians() {
            // Turning in place is always safe for a round robot. It is progress toward the goal, so it does
            // not count as blocked; once facing the goal with still no arc, it does.
            self.blocked_since = None;
            return spin;
        }
        if best.is_some() && v_applied >= MOVING_V {
            self.blocked_since = None;
        } else if self.blocked_since.is_none() {
            self.blocked_since = Some(t);
        }
        if self.blocked_since.is_some_and(|since| t - since > p.blocked_s) {
            // this way is blocked: pick another goal and turn to it
            goals.on_blocked(t, pose);
            self.blocked_since = None;
            let goal = goals.choose_goal(t, pose, grid);
            self.goal = Some(goal);
            self.state = State::Spin;
            let err = wrap_angle(goal - pose.theta);
            return (0.0, p.w_spin.copysign(if err == 0.0 { 1.0 } else { err }));
        }
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
        let mut c = ArcController::new(params, SafetyParams::default());
        let grid = OccupancyGrid::new(OccupancyGridParams::default());
        let mut goal = FixedGoal(goal_deg.to_radians());
        (c.step(&mut goal, 0.0, HERE, Some(sc), &grid, v_applied), c)
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
        let mut c = ArcController::new(ExploreParams::default(), SafetyParams::default());
        let grid = OccupancyGrid::new(OccupancyGridParams::default());
        let mut goal = FixedGoal(90f64.to_radians());
        let at = |deg: f64| Pose2::new(0.0, 0.0, deg.to_radians());
        assert_eq!(
            c.step(&mut goal, 0.0, at(0.0), Some(&open()), &grid, 0.0),
            (0.0, 1.0)
        );
        // 40 deg left: below spin_above, but the spin goes on until within spin_done
        assert_eq!(
            c.step(&mut goal, 0.1, at(50.0), Some(&open()), &grid, 0.0),
            (0.0, 1.0)
        );
        let (v, _) = c.step(&mut goal, 0.2, at(75.0), Some(&open()), &grid, 0.0);
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
        let mut c = ArcController::new(ExploreParams::default(), SafetyParams::default());
        let grid = OccupancyGrid::new(OccupancyGridParams::default());
        let mut goals = Blocked(0);
        for k in 0..=10 {
            assert_eq!(
                c.step(&mut goals, k as f64 * 0.1, HERE, Some(&sc), &grid, 0.0),
                (0.0, 0.0)
            );
        }
        assert_eq!(c.blocked_since, Some(0.0));
        assert_eq!(c.step(&mut goals, 1.1, HERE, Some(&sc), &grid, 0.0), (0.0, -1.0));
        assert_eq!(
            (goals.0, c.state, c.blocked_since, c.goal),
            (1, State::Spin, None, Some(-1.0))
        );
    }

    #[test]
    fn standing_still_with_a_free_arc_counts_as_blocked_too() {
        let mut c = ArcController::new(ExploreParams::default(), SafetyParams::default());
        let grid = OccupancyGrid::new(OccupancyGridParams::default());
        let mut goal = FixedGoal(0.0);
        let (v, _) = c.step(&mut goal, 0.0, HERE, Some(&open()), &grid, 0.0);
        assert!(v > 0.0 && c.blocked_since == Some(0.0)); // asks to go, but the base is not moving
        c.step(&mut goal, 0.1, HERE, Some(&open()), &grid, 0.2);
        assert_eq!(c.blocked_since, None);
    }
}
