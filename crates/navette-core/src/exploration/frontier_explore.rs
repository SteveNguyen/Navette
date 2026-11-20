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
//!   With `recover_replan` (off by default) the recovery replans first and turns toward the new path, see
//!   [`ExploreParams::recover_replan`]; `recover_hold_s` then keeps it from turning straight back.
//! - Full look (only with `look360_m` > 0, off by default): every `look360_m` driven, the robot stops and turns
//!   LOOK360_DEG in place in one direction (state Look), for a complete local view (the camera sees ~70 deg).
//!   See [`ExploreParams::look360_m`] and [`FrontierExplore::full_look`].
//!
//! Numerics follow the Python prototype (the side of the recovery turn sums as numpy does).

use std::f64::consts::{FRAC_PI_2, TAU};

use crate::collision_monitor::SafetyParams;
use crate::controller::dwa::{spin_rate, ArcController, ExploreParams, GoalSource};
use crate::controller::State;
use crate::exploration::frontier::{FrontierParams, FrontierPlanner};
use crate::geometry::{wrap_angle, Point2, Pose2};
use crate::occupancy_grid::OccupancyGrid;
use crate::params::ParamError;
use crate::scan::Scan;

/// Blocked again with the target within this distance (m) of the one of the last block: the same target.
const SAME_TARGET_M: f64 = 0.3;
/// recover_replan: a new path heading within this (deg) of the blocked direction is the same way.
const RECOVER_SAME_DEG: f64 = 20.0;
/// look360_m: a full look turns this much (deg): the full circle and 20 deg of overlap, so that the view (and
/// the SLAM front end's submap of the turn) closes on itself.
const LOOK360_DEG: f64 = 380.0;

/// A full look under way (look360_m).
#[derive(Clone, Copy, Debug)]
struct FullLook {
    /// +1 counter-clockwise (left), -1 clockwise.
    dir: f64,
    /// Angle turned so far in `dir` (rad), from the measured pose.
    turned: f64,
    /// The heading at the last step (rad).
    theta: f64,
    /// Start time (s).
    t0: f64,
}

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
    /// The recovery turn ends within this of its heading (rad): the controller's spin_done_deg,
    spin_done: f64,
    /// tested on the heading predicted this far ahead (s): the controller's spin_lead_s.
    spin_lead: f64,
    /// This step's measured yaw rate x spin_lead (rad), set by [`FrontierExplore::step`].
    yaw_lead: f64,
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
    /// The controller's recover_replan and recover_hold_s.
    recover_replan: bool,
    recover_hold_s: f64,
    /// recover_replan: a block waiting for its replan (at the next choose_goal, which has the map), with the
    /// blocked direction (fixed frame, rad),
    pending: Option<f64>,
    /// the blocked direction of the recovery turn in progress,
    recover_from: Option<f64>,
    /// and after it: until when (s) the goal must not turn back toward the blocked direction, the recovery
    /// heading and the blocked direction.
    hold: Option<(f64, f64, f64)>,
    /// The goal heading last given to the controller.
    last_goal: Option<f64>,
    /// Direction of the last turn in place the controller asked for (+1 left, -1 right).
    pub last_turn: Option<f64>,
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

    /// More free floor in view on the left (bearing > 0) than on the right? (Ties and no scan: left.)
    fn freer_left(&self) -> bool {
        self.scan.as_ref().is_none_or(|s| {
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
        })
    }

    /// recover_replan: the recovery heading after a block toward `blocked` (fixed frame, rad). Replan first
    /// (this scan's near sectors, a banned target left out); the new path's heading if it leaves more than
    /// RECOVER_SAME_DEG away from the blocked direction; else 90 deg in the direction of the previous turn
    /// in place (none yet: toward the freer side), so as not to undo it.
    fn replanned_recovery(&mut self, t: f64, pose: Pose2, grid: &OccupancyGrid, blocked: f64) -> f64 {
        self.plan(t, pose, grid);
        let same = RECOVER_SAME_DEG.to_radians();
        if let Some(h) = self.planner.heading(pose) {
            if wrap_angle(h - blocked).abs() > same {
                return h;
            }
        }
        let dir = self
            .last_turn
            .unwrap_or_else(|| if self.freer_left() { 1.0 } else { -1.0 });
        wrap_angle(pose.theta + dir * FRAC_PI_2)
    }

    fn goal(&mut self, t: f64, pose: Pose2, grid: &OccupancyGrid) -> f64 {
        if let Some(blocked) = self.pending.take() {
            self.recover = Some(self.replanned_recovery(t, pose, grid, blocked));
            self.recover_from = Some(blocked);
        }
        if let Some(recover) = self.recover {
            // the heading when the command takes effect (spin_lead > 0), else the current one
            let heading = if self.spin_lead > 0.0 {
                pose.theta + self.yaw_lead
            } else {
                pose.theta
            };
            if wrap_angle(recover - heading).abs() > self.spin_done {
                return recover;
            }
            self.recover = None;
            self.plan(t, pose, grid); // turned: replan from the new view
            if let Some(blocked) = self.recover_from.take() {
                if self.recover_hold_s > 0.0 {
                    self.hold = Some((t + self.recover_hold_s, recover, blocked));
                }
            }
        }
        if self.planner.path().is_none() && !self.planner.done() {
            self.plan(t, pose, grid);
        }
        let heading = self.planner.heading(pose);
        if let Some((until, recover, blocked)) = self.hold {
            if t >= until {
                self.hold = None;
            } else if let Some(h) = heading {
                // not back toward the way that was blocked yet: keep the recovery heading
                if wrap_angle(h - blocked).abs() < wrap_angle(h - recover).abs() {
                    return recover;
                }
            }
        }
        heading.unwrap_or(pose.theta)
    }
}

impl GoalSource for FrontierGoals {
    fn choose_goal(&mut self, t: f64, pose: Pose2, grid: &OccupancyGrid) -> f64 {
        let goal = self.goal(t, pose, grid);
        self.last_goal = Some(goal);
        goal
    }

    /// The default recovery turns at once (`recover`); with recover_replan it is chosen at the next
    /// choose_goal (the controller asks right after), which replans with the map.
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
        if self.recover_replan {
            self.pending = Some(self.last_goal.unwrap_or(pose.theta));
            self.hold = None;
            return;
        }
        // more free floor in view on the left (bearing > 0) or on the right?
        let left = self.freer_left();
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
    /// look360_m: distance driven (m, path length of the poses) since the last full look or the start,
    driven: f64,
    /// the pose of the last step,
    last_pose: Option<Pose2>,
    /// the full look under way,
    full_look: Option<FullLook>,
    /// and the full looks started so far (logs, tests).
    pub full_looks: u32,
}

impl FrontierExplore {
    /// `params.rotate_in_place` should be true: the path often leaves sideways from where no arc is free.
    /// An error if a parameter is invalid.
    pub fn new(
        params: ExploreParams,
        safety: SafetyParams,
        frontier: FrontierParams,
    ) -> Result<Self, ParamError> {
        let goals = FrontierGoals {
            planner: FrontierPlanner::new(frontier)?,
            spin_done: params.spin_done_deg.to_radians(),
            spin_lead: params.spin_lead_s,
            yaw_lead: 0.0,
            last_plan: f64::NEG_INFINITY,
            scan: None,
            recover: None,
            fails: 0,
            fail_target: None,
            recover_replan: params.recover_replan,
            recover_hold_s: params.recover_hold_s,
            pending: None,
            recover_from: None,
            hold: None,
            last_goal: None,
            last_turn: None,
        };
        Ok(Self {
            controller: ArcController::new(params, safety)?,
            goals,
            last_t: None,
            looked: 0.0,
            driven: 0.0,
            last_pose: None,
            full_look: None,
            full_looks: 0,
        })
    }

    /// look360_m: the command of the full look, or None (no look under way, or it just ended: drive).
    ///
    /// Starts once `look360_m` has been driven, only while driving: not during a turn in place toward a goal
    /// (SPIN), a recovery turn, or with no frontier left (the LOOK / DONE look-around is already a turn).
    /// Direction: toward the current goal, so that the look ends facing a little past it. The angle turned is
    /// measured on the pose, not integrated from the command: the runtime caps w (controller.w_max) and the
    /// base lags. It ends after LOOK360_DEG, or after LOOK360_DEG / spin_w_min (s; the slowest rate the turn
    /// asks for) if the base does not follow. Turning in place is always allowed by the collision monitor
    /// (a request with v <= 0 keeps its w: the robot is a disk), so nothing else can stop it but the runtime
    /// (STOP builds a new navigator).
    fn full_look(&mut self, t: f64, pose: Pose2) -> Option<(f64, f64)> {
        let p = &self.controller.params;
        if self.full_look.is_none() {
            let start = self.driven >= p.look360_m
                && self.controller.state == State::Drive
                && self.goals.recover.is_none()
                && !self.goals.planner.done();
            if !start {
                return None;
            }
            let toward = self.controller.goal.map_or(0.0, |g| wrap_angle(g - pose.theta));
            self.full_look = Some(FullLook {
                dir: if toward < 0.0 { -1.0 } else { 1.0 },
                turned: 0.0,
                theta: pose.theta,
                t0: t,
            });
            self.full_looks += 1;
        }
        let look = self.full_look.as_mut().expect("set above");
        look.turned += wrap_angle(pose.theta - look.theta) * look.dir;
        look.theta = pose.theta;
        let total = LOOK360_DEG.to_radians();
        if look.turned >= total || t - look.t0 > total / p.spin_w_min {
            self.full_look = None;
            self.driven = 0.0;
            // a fresh start, as after the look-around: the time spent turning is not "no progress"
            let c = &mut self.controller;
            (c.state, c.progress, c.turn, c.blocked_since) = (State::Drive, None, None, None);
            if self.goals.planner.done() {
                self.looked = TAU; // that was the look-around: no second one
            }
            return None;
        }
        self.controller.state = State::Look;
        Some((0.0, spin_rate(p, total - look.turned).copysign(look.dir)))
    }

    /// Drive, Spin, Look or Done.
    pub fn state(&self) -> State {
        self.controller.state
    }

    /// Requested (v, w) at time `t` (s) from `pose` on `grid`. `scan`: this frame's (None: the last one is
    /// kept for the planner). `v_applied`: the speed command sent last step (m/s). `w_meas`: the measured yaw
    /// rate (rad/s). `near_known`: the scan's sectors whose blind segment the map explains, for the arc
    /// controller (see [`ArcController::step`]).
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        t: f64,
        pose: Pose2,
        scan: Option<&Scan>,
        grid: &OccupancyGrid,
        v_applied: f64,
        w_meas: f64,
        near_known: Option<&[bool]>,
    ) -> (f64, f64) {
        self.goals.yaw_lead = w_meas * self.goals.spin_lead;
        let dt = self.last_t.map_or(0.0, |last| t - last);
        self.last_t = Some(t);
        if let Some(scan) = scan {
            self.goals.scan = Some(scan.clone());
        }
        if t - self.goals.last_plan >= self.goals.planner.params.replan_s {
            self.goals.plan(t, pose, grid);
        }
        if self.controller.params.look360_m > 0.0 {
            // driven: not while a full look turns (the balancing base rolls on a few cm as it stops)
            if let (Some(last), None) = (self.last_pose, self.full_look) {
                self.driven += (pose.x - last.x).hypot(pose.y - last.y);
            }
            self.last_pose = Some(pose);
            if let Some(cmd) = self.full_look(t, pose) {
                return cmd;
            }
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
            c.progress = None; // a fresh start: the time spent looking or done is not "no progress"
        }
        let (v, w) = c.step(
            &mut self.goals,
            t,
            pose,
            scan,
            grid,
            v_applied,
            w_meas,
            near_known,
        );
        if v == 0.0 && w != 0.0 {
            self.goals.last_turn = Some(1f64.copysign(w)); // for the recovery of recover_replan
        }
        (v, w)
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
        FrontierExplore::new(params, SafetyParams::default(), FrontierParams::default()).unwrap()
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
            .map(|k| fe.step(k as f64 * 0.1, HERE, None, &g, 0.0, 0.0, None))
            .collect();
        let n_look = cmds.iter().filter(|c| c.0 == 0.0 && c.1 > 0.0).count();
        assert!((n_look as f64 * 0.1 * 1.0 - TAU).abs() < 0.2); // one full turn
        assert_eq!((fe.state(), cmds[99]), (State::Done, (0.0, 0.0)));
    }

    #[test]
    fn a_frontier_coming_back_resumes_exploration() {
        let mut fe = explore();
        fe.step(0.0, HERE, None, &room(0.0), 0.0, 0.0, None);
        assert_eq!(fe.state(), State::Look);
        let (v, w) = fe.step(1.0, HERE, Some(&scan(2.0, 2.0)), &room(0.8), 0.0, 0.0, None);
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
    fn the_recovery_turn_ends_on_the_predicted_heading() {
        // recovery turn to 90 deg, now at 64 deg turning at 0.5 rad/s: 26 deg to go now, 11.7 deg in 0.5 s
        let g = room(0.8);
        let at = Pose2::new(0.0, 0.0, 64f64.to_radians());
        for (spin_lead_s, w_meas, ends) in [(0.0, 0.5, false), (0.5, 0.0, false), (0.5, 0.5, true)] {
            let params = ExploreParams {
                rotate_in_place: true,
                spin_lead_s,
                ..Default::default()
            };
            let mut fe =
                FrontierExplore::new(params, SafetyParams::default(), FrontierParams::default()).unwrap();
            fe.goals.recover = Some(FRAC_PI_2);
            fe.step(0.0, at, None, &g, 0.0, w_meas, None);
            assert_eq!(fe.goals.recover.is_none(), ends, "lead {spin_lead_s}, w {w_meas}");
        }
    }

    #[test]
    fn recovery_turn_is_not_interrupted_by_the_look_around() {
        // planner done (closed room) while a recovery turn to the right is under way: keep turning right
        let g = room(0.0);
        let mut fe = explore();
        fe.goals.recover = Some(-FRAC_PI_2);
        for k in 0..5 {
            let (v, w) = fe.step(k as f64 * 0.1, HERE, None, &g, 0.0, 0.0, None);
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

    fn replanning(hold_s: f64) -> FrontierExplore {
        let params = ExploreParams {
            rotate_in_place: true,
            recover_replan: true,
            recover_hold_s: hold_s,
            ..Default::default()
        };
        FrontierExplore::new(params, SafetyParams::default(), FrontierParams::default()).unwrap()
    }

    /// The controller's sequence on a block: on_blocked, then choose_goal at once; the recovery heading.
    fn block(fe: &mut FrontierExplore, t: f64, pose: Pose2, g: &OccupancyGrid) -> f64 {
        fe.goals.on_blocked(t, pose);
        fe.goals.choose_goal(t, pose, g)
    }

    #[test]
    fn replanned_recovery_turns_toward_the_new_path() {
        // blocked while heading at 115 deg (the robot faces it); the replan's path leaves for the door at
        // x = +1, ~0 deg: the recovery turns there, not 90 deg to a side (the prototype: 205 or 25 deg)
        let g = room(0.8);
        let up = Pose2::new(0.0, 0.0, 2.0);
        let mut fe = replanning(0.0);
        fe.goals.scan = Some(scan(0.9, 0.3));
        fe.goals.last_goal = Some(2.0);
        let h = block(&mut fe, 0.5, up, &g);
        assert_eq!(Some(h), fe.goals.planner.heading(up));
        assert!(h.abs() < 0.2 && fe.goals.recover == Some(h));
        // the default: 90 deg toward the freer side (left), whatever the planner says
        let mut fe = explore();
        fe.goals.scan = Some(scan(0.9, 0.3));
        assert_eq!(block(&mut fe, 0.5, up, &g), wrap_angle(2.0 + FRAC_PI_2));
    }

    #[test]
    fn replanned_recovery_same_way_or_no_path_keeps_the_previous_turn_direction() {
        // blocked on the way to the door (heading 0) and the replan still says 0: 90 deg in the direction of
        // the previous turn (right), even though the left has more free floor in view
        let g = room(0.8);
        let mut fe = replanning(0.0);
        fe.goals.scan = Some(scan(0.9, 0.3));
        fe.goals.last_goal = Some(0.0);
        fe.goals.last_turn = Some(-1.0);
        assert_eq!(block(&mut fe, 0.5, HERE, &g), -FRAC_PI_2);
        // no previous turn: toward the freer side, as the prototype
        let mut fe = replanning(0.0);
        fe.goals.scan = Some(scan(0.9, 0.3));
        fe.goals.last_goal = Some(0.0);
        assert_eq!(block(&mut fe, 0.5, HERE, &g), FRAC_PI_2);
        // no path at all (closed room): the previous turn's direction
        let closed = room(0.0);
        let mut fe = replanning(0.0);
        fe.goals.last_turn = Some(-1.0);
        assert_eq!(block(&mut fe, 0.5, HERE, &closed), -FRAC_PI_2);
    }

    #[test]
    fn after_the_recovery_the_goal_does_not_turn_back_for_recover_hold_s() {
        // blocked toward the door (0 deg), recovery 90 deg right; turned: the planner points back to the
        // door, nearer the blocked direction than the recovery heading
        let g = room(0.8);
        let right = Pose2::new(0.0, 0.0, -FRAC_PI_2);
        for hold_s in [0.0, 2.0] {
            let mut fe = replanning(hold_s);
            fe.goals.last_goal = Some(0.0);
            fe.goals.last_turn = Some(-1.0);
            assert_eq!(block(&mut fe, 0.5, HERE, &g), -FRAC_PI_2);
            let path = |fe: &FrontierExplore| fe.goals.planner.heading(right).unwrap();
            let at_end = fe.goals.choose_goal(4.0, right, &g); // the turn ends
            assert!(fe.goals.recover.is_none() && path(&fe).abs() < 0.2);
            if hold_s == 0.0 {
                assert_eq!(at_end, path(&fe)); // straight back toward the door
            } else {
                assert_eq!(at_end, -FRAC_PI_2); // held
                assert_eq!(fe.goals.choose_goal(5.9, right, &g), -FRAC_PI_2);
                assert_eq!(fe.goals.choose_goal(6.0, right, &g), path(&fe)); // 2 s after the turn ended
            }
        }
    }

    fn looking(look360_m: f64, spin_accel: f64) -> FrontierExplore {
        let params = ExploreParams {
            rotate_in_place: true,
            look360_m,
            spin_accel,
            ..Default::default()
        };
        FrontierExplore::new(params, SafetyParams::default(), FrontierParams::default()).unwrap()
    }

    /// Steps 0.1 s apart from `t0`, at heading 0 (toward the door), the pose moving 5 cm per step along x
    /// (`dir` +1 or -1) from `x0`, for `n` steps; the commands.
    fn drive(
        fe: &mut FrontierExplore,
        g: &OccupancyGrid,
        t0: f64,
        x0: f64,
        dir: f64,
        n: usize,
    ) -> Vec<(f64, f64)> {
        (0..n)
            .map(|k| {
                let pose = Pose2::new(x0 + dir * 0.05 * k as f64, 0.0, 0.0);
                fe.step(t0 + 0.1 * k as f64, pose, None, g, 0.1, 0.0, None)
            })
            .collect()
    }

    /// From heading 0 at `at`, steps 0.1 s apart turning by each command (a base that follows exactly), while
    /// the state is Look; the commands.
    fn turn_while_looking(
        fe: &mut FrontierExplore,
        g: &OccupancyGrid,
        t0: f64,
        at: (f64, f64),
    ) -> Vec<(f64, f64)> {
        let (mut theta, mut cmds) = (0.0, Vec::new());
        for k in 0..200 {
            let cmd = fe.step(
                t0 + 0.1 * k as f64,
                Pose2::new(at.0, at.1, wrap_angle(theta)),
                None,
                g,
                0.0,
                0.0,
                None,
            );
            if fe.state() != State::Look {
                break;
            }
            cmds.push(cmd);
            theta += cmd.1 * 0.1;
        }
        cmds
    }

    #[test]
    fn a_full_look_after_look360_m_driven_turns_380_deg_in_one_direction() {
        let g = room(0.8);
        // off (the default): driving 1 m never stops for a look
        let mut fe = looking(0.0, 0.0);
        drive(&mut fe, &g, 0.0, -0.8, 1.0, 21);
        assert!(fe.state() == State::Drive && fe.full_looks == 0);
        // every ~1 m (0.975: below 20 x 5 cm, whatever the rounding): at 1.0 m driven it stands still and turns
        let mut fe = looking(0.975, 0.0);
        let cmds = drive(&mut fe, &g, 0.0, -0.8, 1.0, 21);
        assert!(cmds[..20].iter().all(|c| c.0 > 0.0), "{cmds:?}");
        assert_eq!((fe.state(), fe.full_looks, cmds[20].0), (State::Look, 1, 0.0));
        let turn = turn_while_looking(&mut fe, &g, 2.1, (0.2, 0.0));
        let w0 = turn[0].1;
        assert!(w0 != 0.0 && turn.iter().all(|&(v, w)| v == 0.0 && w == w0)); // w_spin, one way
        let turned = (turn.len() as f64 * w0.abs() * 0.1).to_degrees();
        assert!((380.0..380.0 + 6.0).contains(&turned), "{turned}"); // ends within a step past 380 deg
                                                                     // and drives on; the next one 1 m later, not before
        assert!(fe.state() == State::Drive && fe.full_looks == 1);
        let cmds = drive(&mut fe, &g, 10.0, 0.2, -1.0, 21);
        assert!(cmds[..20].iter().all(|c| c.0 > 0.0) && fe.full_looks == 2);
    }

    #[test]
    fn a_full_look_slows_down_with_spin_accel_and_ends_if_the_base_does_not_turn() {
        let g = room(0.8);
        let mut fe = looking(0.975, 0.5);
        drive(&mut fe, &g, 0.0, -0.8, 1.0, 21);
        let turn = turn_while_looking(&mut fe, &g, 2.1, (0.2, 0.0));
        let w: Vec<f64> = turn.iter().map(|c| c.1.abs()).collect();
        assert_eq!(w[0], 1.0); // w_spin while far from the end
        assert!(
            w.last().unwrap() < &0.5 && w.windows(2).all(|p| p[1] <= p[0]),
            "{w:?}"
        );
        // a base that does not turn (held, stalled): the look gives up after 380 deg / spin_w_min (22 s)
        let mut fe = looking(0.975, 0.5);
        drive(&mut fe, &g, 0.0, -0.8, 1.0, 21);
        let here = Pose2::new(0.2, 0.0, 0.0);
        let n = (0..400)
            .take_while(|&k| {
                fe.step(2.1 + 0.1 * k as f64, here, None, &g, 0.0, 0.0, None);
                fe.state() == State::Look
            })
            .count();
        let limit = 380f64.to_radians() / 0.3;
        assert!((n as f64 * 0.1 - limit).abs() < 0.15, "{n}");
    }

    #[test]
    fn a_full_look_waits_for_a_recovery_turn_and_leaves_the_look_around_alone() {
        // due during a recovery turn: the turn finishes first, then the look
        let g = room(0.8);
        let mut fe = looking(0.975, 0.0);
        drive(&mut fe, &g, 0.0, -0.8, 1.0, 20); // 0.95 m
        fe.goals.recover = Some(-FRAC_PI_2);
        let (_, w) = fe.step(2.0, Pose2::new(0.2, 0.0, 0.0), None, &g, 0.0, 0.0, None);
        assert!(w < 0.0 && fe.full_looks == 0 && fe.state() != State::Look);
        fe.step(2.1, Pose2::new(0.2, 0.0, -FRAC_PI_2), None, &g, 0.0, 0.0, None); // turned: recovery over
        assert!(fe.goals.recover.is_none());
        // the controller turns back toward the door (SPIN): the look waits for DRIVE
        let mut k = 0;
        while fe.state() != State::Look {
            assert!(k < 30, "no full look");
            let theta = -FRAC_PI_2 * (1.0 - k as f64 / 10.0).max(0.0);
            fe.step(
                2.2 + 0.1 * k as f64,
                Pose2::new(0.2, 0.0, theta),
                None,
                &g,
                0.0,
                0.0,
                None,
            );
            assert!(fe.full_looks == 0 || theta.abs() < 0.4, "{k} {theta}");
            k += 1;
        }
        assert_eq!(fe.full_looks, 1);
        // no frontier at all: the prototype's look-around (LOOK, then DONE), no full look
        let mut fe = looking(0.01, 0.0);
        let closed = room(0.0);
        drive(&mut fe, &closed, 0.0, -0.8, 1.0, 21);
        assert_eq!(fe.full_looks, 0);
    }

    #[test]
    fn replanned_recovery_still_bans_a_target_blocked_twice() {
        let g = room(0.8);
        let mut fe = replanning(0.0);
        fe.goals.choose_goal(0.0, HERE, &g);
        let first = fe.goals.planner.target().unwrap();
        block(&mut fe, 0.5, HERE, &g);
        assert!(fe.goals.planner.banned().is_empty() && fe.goals.fails() == 1);
        fe.goals.recover = None;
        fe.goals.on_blocked(1.0, HERE);
        assert_eq!(fe.goals.planner.banned(), [(1.0, first)]);
        // the replan then leaves it out (here it is the only frontier: the planner lifts the ban at once)
        let h = fe.goals.choose_goal(1.0, HERE, &g);
        assert!(fe.goals.planner.banned().is_empty() && h.abs() < 0.2);
    }
}
