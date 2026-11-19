//! Frontier exploration behaviour: the arc controller (controller/dwa.rs) steered along the frontier
//! planner's path.
//!
//! - The planner runs every `replan_s`, with the current scan's image-bottom obstacles.
//! - No reachable frontier is not "explored": a robot that starts facing a wall sees none (the floor right in
//!   front is in the camera's blind zone). The robot first turns once on the spot (LOOK), then stops (DONE);
//!   new views bring frontiers back and exploration resumes.
//! - Blocked (no admissible arc, or no motion, for `blocked_s`): one fixed recovery, as Nav2's recovery
//!   behaviours: turn in place to a heading chosen ONCE (90 deg toward the side with more free floor in view)
//!   and held until reached, then replan. Re-choosing it every step made the robot flip left / right in
//!   place. The second block on the same target bans it.
//!
//! Numerics follow the Python prototype (the side of the recovery turn sums as numpy does).

use std::f64::consts::{FRAC_PI_2, TAU};

use crate::collision_monitor::SafetyParams;
use crate::controller::dwa::{ArcController, ExploreParams, GoalSource};
use crate::controller::State;
use crate::exploration::frontier::{FrontierParams, FrontierPlanner};
use crate::geometry::{wrap_angle, Point2, Pose2};
use crate::occupancy_grid::OccupancyGrid;
use crate::scan::Scan;

/// Blocked again with the target within this distance (m) of the one of the last block: the same target.
const SAME_TARGET_M: f64 = 0.3;

/// The sum of `a` in numpy's order (pairwise, 8 partial sums per block of at most 128 values): sums that
/// are compared must round the same way.
fn numpy_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(0.0, |s, &x| s + x);
    }
    if n > 128 {
        let half = n / 2 - (n / 2) % 8;
        return numpy_sum(&a[..half]) + numpy_sum(&a[half..]);
    }
    let mut blocks = a.chunks_exact(8);
    let mut r = [0.0; 8];
    r.copy_from_slice(blocks.next().expect("n >= 8"));
    for block in blocks.by_ref() {
        for (r, &x) in r.iter_mut().zip(block) {
            *r += x;
        }
    }
    let sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    blocks.remainder().iter().fold(sum, |s, &x| s + x)
}

/// The goal headings of frontier exploration: the planner's path, or the recovery turn in progress.
#[derive(Clone, Debug)]
pub struct FrontierGoals {
    pub planner: FrontierPlanner,
    /// The recovery turn ends within this of its heading (rad): the controller's spin_done_deg.
    spin_done: f64,
    /// Time of the last plan (s).
    last_plan: f64,
    /// The last scan: its image-bottom sectors for the planner, the side of the recovery turn.
    pub scan: Option<Scan>,
    /// Fixed-frame heading of the recovery turn in progress.
    pub recover: Option<f64>,
    /// Blocks on the current target.
    fails: u32,
    /// The target of the last block.
    fail_target: Option<Point2>,
}

impl FrontierGoals {
    /// Plan from `pose`, with the image-bottom obstacles of the last scan.
    pub fn plan(&mut self, t: f64, pose: Pose2, grid: &OccupancyGrid) {
        let (mut near, mut half) = (Vec::new(), 0.0);
        if let Some(s) = self.scan.as_ref().filter(|s| s.len() >= 2) {
            near = (0..s.len())
                .filter(|&k| s.seen[k] && s.near[k])
                .map(|k| s.bearing_deg[k].to_radians())
                .collect();
            half = ((s.bearing_deg[1] - s.bearing_deg[0]).abs() / 2.0).to_radians();
        }
        self.planner.plan(t, pose, grid, &near, half);
        self.last_plan = t;
    }

    /// Blocks on the current target so far.
    pub fn fails(&self) -> u32 {
        self.fails
    }
}

impl GoalSource for FrontierGoals {
    fn choose_goal(&mut self, t: f64, pose: Pose2, grid: &OccupancyGrid) -> f64 {
        if let Some(recover) = self.recover {
            if wrap_angle(recover - pose.theta).abs() > self.spin_done {
                return recover;
            }
            self.recover = None;
            self.plan(t, pose, grid); // turned: replan from the new view
        }
        if self.planner.path().is_none() && !self.planner.done() {
            self.plan(t, pose, grid);
        }
        self.planner.heading(pose).unwrap_or(pose.theta)
    }

    fn on_blocked(&mut self, t: f64, pose: Pose2) {
        let target = self.planner.target();
        let same = matches!((target, self.fail_target),
            (Some(a), Some(b)) if (a[0] - b[0]).hypot(a[1] - b[1]) < SAME_TARGET_M);
        self.fails = if same { self.fails + 1 } else { 1 };
        self.fail_target = target;
        if self.fails >= 2 {
            // blocked twice on the way to it: give the target up
            self.planner.ban(t);
            (self.fails, self.fail_target) = (0, None);
        }
        // more free floor in view on the left (bearing > 0) or on the right?
        let left = self.scan.as_ref().is_none_or(|s| {
            let side = |keep: fn(f64) -> bool| {
                let free: Vec<f64> = s
                    .free()
                    .into_iter()
                    .zip(&s.bearing_deg)
                    .filter(|&(_, &b)| keep(b))
                    // unseen sectors (NaN) count for nothing
                    .map(|(f, _)| {
                        if f.is_nan() {
                            0.0
                        } else {
                            f.clamp(f64::MIN, f64::MAX)
                        }
                    })
                    .collect();
                numpy_sum(&free)
            };
            side(|b| b > 0.0) >= side(|b| b < 0.0)
        });
        self.recover = Some(wrap_angle(pose.theta + if left { FRAC_PI_2 } else { -FRAC_PI_2 }));
    }
}

/// See the module documentation. Call [`FrontierExplore::step`] once per frame.
#[derive(Clone, Debug)]
pub struct FrontierExplore {
    pub controller: ArcController,
    pub goals: FrontierGoals,
    /// Time of the last step (s).
    last_t: Option<f64>,
    /// Angle turned in place (rad) since the planner last had a frontier.
    looked: f64,
}

impl FrontierExplore {
    /// `params.rotate_in_place` should be true: the path often leaves sideways from where no arc is free.
    pub fn new(params: ExploreParams, safety: SafetyParams, frontier: FrontierParams) -> Self {
        let goals = FrontierGoals {
            planner: FrontierPlanner::new(frontier),
            spin_done: params.spin_done_deg.to_radians(),
            last_plan: f64::NEG_INFINITY,
            scan: None,
            recover: None,
            fails: 0,
            fail_target: None,
        };
        Self {
            controller: ArcController::new(params, safety),
            goals,
            last_t: None,
            looked: 0.0,
        }
    }

    /// Drive, Spin, Look or Done.
    pub fn state(&self) -> State {
        self.controller.state
    }

    /// Requested (v, w) at time `t` (s) from `pose` on `grid`. `scan`: this frame's (None: the last one is
    /// kept for the planner). `v_applied`: the speed command sent last step (m/s).
    pub fn step(
        &mut self,
        t: f64,
        pose: Pose2,
        scan: Option<&Scan>,
        grid: &OccupancyGrid,
        v_applied: f64,
    ) -> (f64, f64) {
        let dt = self.last_t.map_or(0.0, |last| t - last);
        self.last_t = Some(t);
        if let Some(scan) = scan {
            self.goals.scan = Some(scan.clone());
        }
        if t - self.goals.last_plan >= self.goals.planner.params.replan_s {
            self.goals.plan(t, pose, grid);
        }
        let c = &mut self.controller;
        if self.goals.planner.done() && self.goals.recover.is_none() {
            // Turn once on the spot before giving up. Not during a recovery turn: the two turned opposite
            // ways as the planner flipped between done and a path; the recovery turn finishes first, then
            // replans.
            if self.looked < TAU {
                self.looked += c.params.w_spin * dt;
                c.state = State::Look;
                return (0.0, c.params.w_spin);
            }
            c.state = State::Done;
            return (0.0, 0.0);
        }
        self.looked = 0.0;
        if matches!(c.state, State::Done | State::Look) {
            c.state = State::Drive;
        }
        c.step(&mut self.goals, t, pose, scan, grid, v_applied)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_room as room;
    use super::*;

    const HERE: Pose2 = Pose2 {
        x: 0.0,
        y: 0.0,
        theta: 0.0,
    };

    fn explore() -> FrontierExplore {
        let params = ExploreParams {
            rotate_in_place: true,
            ..Default::default()
        };
        FrontierExplore::new(params, SafetyParams::default(), FrontierParams::default())
    }

    /// Open floor in view up to `floor_left` m on the left and `floor_right` m on the right.
    fn scan(floor_left: f64, floor_right: f64) -> Scan {
        let bearing_deg: Vec<f64> = (0..180).map(|k| (k as f64 + 0.5) * 2.0 - 180.0).collect();
        Scan {
            seen: bearing_deg.iter().map(|b| b.abs() < 38.0).collect(),
            obstacle: vec![f64::INFINITY; 180],
            near: vec![false; 180],
            floor: bearing_deg
                .iter()
                .map(|&b| if b > 0.0 { floor_left } else { floor_right })
                .collect(),
            closest: vec![0.45; 180],
            bearing_deg,
        }
    }

    #[test]
    fn numpy_sum_orders() {
        // exact sums: any order gives the same
        let a: Vec<f64> = (1..=300).map(f64::from).collect();
        for n in [0, 5, 8, 9, 90, 128, 129, 300] {
            assert_eq!(numpy_sum(&a[..n]), (n * (n + 1) / 2) as f64);
        }
        // 8 partial sums: the first is 1e16 + 1 = 1e16, the 7 others 2; a running sum would lose every 1
        let mut b = vec![1.0; 16];
        b[0] = 1e16;
        assert_eq!(numpy_sum(&b), 1e16 + 14.0);
        assert_eq!(numpy_sum(&[1e16, 1.0, 1.0, 1.0, 1.0]), 1e16); // fewer than 8: a running sum
    }

    #[test]
    fn no_frontier_turns_around_once_before_done() {
        let g = room(0.0); // closed room: nothing to explore
        let mut fe = explore();
        let cmds: Vec<(f64, f64)> = (0..100)
            .map(|k| fe.step(k as f64 * 0.1, HERE, None, &g, 0.0))
            .collect();
        let n_look = cmds.iter().filter(|c| c.0 == 0.0 && c.1 > 0.0).count();
        assert!((n_look as f64 * 0.1 * 1.0 - TAU).abs() < 0.2); // one full turn
        assert_eq!((fe.state(), cmds[99]), (State::Done, (0.0, 0.0)));
    }

    #[test]
    fn a_frontier_coming_back_resumes_exploration() {
        let mut fe = explore();
        fe.step(0.0, HERE, None, &room(0.0), 0.0);
        assert_eq!(fe.state(), State::Look);
        let (v, w) = fe.step(1.0, HERE, Some(&scan(2.0, 2.0)), &room(0.8), 0.0);
        assert_eq!(fe.state(), State::Drive);
        assert!(v > 0.0 && w.abs() < 0.3); // the door is straight ahead
    }

    #[test]
    fn recovery_turn_is_chosen_once_and_held() {
        // blocked: turn 90 deg toward the side with more free floor in view, and keep that goal while
        // turning even if the view changes
        let g = room(0.8);
        let mut fe = explore();
        fe.goals.scan = Some(scan(0.9, 0.3)); // less free floor on the right
        fe.goals.choose_goal(0.0, HERE, &g);
        fe.goals.on_blocked(0.5, HERE);
        assert_eq!(fe.goals.recover, Some(FRAC_PI_2)); // left
        fe.goals.scan = Some(scan(0.9, 0.9)); // the view changes: right now as free as left
        for yaw in [0.2, 0.5, 0.9] {
            assert_eq!(
                fe.goals.choose_goal(1.0, Pose2::new(0.0, 0.0, yaw), &g),
                FRAC_PI_2
            );
        }
        // reached: back to the planner's heading
        let there = Pose2::new(0.0, 0.0, FRAC_PI_2);
        let h = fe.goals.choose_goal(2.0, there, &g);
        assert_eq!(fe.goals.recover, None);
        assert_eq!(Some(h), fe.goals.planner.heading(there));
        // the other side, and a tie goes left
        fe.goals.scan = Some(scan(0.3, 0.9));
        fe.goals.on_blocked(3.0, HERE);
        assert_eq!(fe.goals.recover, Some(-FRAC_PI_2));
        fe.goals.scan = Some(scan(0.9, 0.9));
        fe.goals.on_blocked(4.0, HERE);
        assert_eq!(fe.goals.recover, Some(FRAC_PI_2));
    }

    #[test]
    fn recovery_turn_is_not_interrupted_by_the_look_around() {
        // planner done (closed room) while a recovery turn to the right is under way: keep turning right
        let g = room(0.0);
        let mut fe = explore();
        fe.goals.recover = Some(-FRAC_PI_2);
        for k in 0..5 {
            let (v, w) = fe.step(k as f64 * 0.1, HERE, None, &g, 0.0);
            assert!(fe.goals.planner.done() && fe.state() != State::Look && v == 0.0 && w < 0.0);
        }
    }

    #[test]
    fn second_block_on_the_same_target_bans_it() {
        let g = room(0.8);
        let mut fe = explore();
        fe.goals.choose_goal(0.0, HERE, &g);
        let first = fe.goals.planner.target().unwrap();
        fe.goals.on_blocked(0.5, HERE);
        assert!(fe.goals.planner.banned().is_empty() && fe.goals.fails() == 1); // just the recovery turn
        fe.goals.recover = None;
        fe.goals.on_blocked(1.0, HERE);
        assert_eq!(fe.goals.planner.banned(), [(1.0, first)]);
        assert_eq!(fe.goals.fails(), 0);
    }
}
