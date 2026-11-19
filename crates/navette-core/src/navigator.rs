//! One navigation step on a free-space scan: occupancy grid -> behaviour -> collision monitor -> resume gate.
//!
//! Per frame, in this order:
//!
//! ```text
//! tilt trust   TiltTrust::update(dt, a, v, w) from the base motion; trusted range for the map
//! blind zone   OccupancyGrid::blind_known(pose, scan): which image-bottom obstacles the map explains,
//!              asked BEFORE this frame goes in (the scan must not explain itself)
//! map          OccupancyGrid::update(t, pose, scan, sigma, cam_h, trusted range)    (keyframes only)
//! behaviour    FrontierExplore::step(t, pose, scan, grid, v_prev), or the constant forward request
//!              -> requested (v, w)
//! safety       safe_command(v, w, scan, params, grid obstacles, blind_known)
//! resume gate  ResumeGate::step(v_req, v_ok)                                        -> (v, w) to send
//! ```
//!
//! Frames: pose = the robot in the fixed frame of the odometry (x forward at start, y left, theta
//! counter-clockwise, rad); the scan is in the camera's heading frame, which is the robot's while the head
//! looks straight ahead.
//!
//! A deliberate difference from the Python prototype: a step whose scan has no seen sector returns v = 0
//! (w, the request and the state as computed). The collision monitor alone puts the boundary of unseen
//! sectors at `unseen_m` all round, which still leaves room to creep forward (~0.1 m/s with the robot's
//! parameters): right for an arc that swings a little out of the view, wrong for a camera that sees no floor
//! at all (looking straight down or up, a wrong 'up', a mask with no ray below the horizon), where the robot
//! would advance blind for ever, with finite and fresh outputs that trip no watchdog. The rule is here and
//! not in `safe_command`, whose outputs stay the prototype's.

use std::f64::consts::TAU;

use crate::collision_monitor::{safe_command, SafetyParams};
use crate::controller::dwa::ExploreParams;
use crate::controller::State;
use crate::exploration::frontier::FrontierParams;
use crate::exploration::frontier_explore::FrontierExplore;
use crate::geometry::{Point2, Pose2, Vec3};
use crate::occupancy_grid::{OccupancyGrid, OccupancyGridParams};
use crate::scan::{FreeSpace, Scan, ShapeError};
use crate::state_estimation::tilt_trust::TiltTrust;
use crate::velocity_smoother::ResumeGate;

/// Exploration is finished once the behaviour has said Done this long (s).
pub const DONE_HOLD_S: f64 = 3.0;
/// 'forward' holds the heading of its first step, proportionally on the odometry pose (the base drifts
/// with w = 0: 6 deg over 0.4 m): gain (1/s; a 4 deg error -> 0.1 rad/s),
pub const HEADING_K: f64 = 1.5;
/// and bound of the correction (rad/s).
pub const HEADING_W_MAX: f64 = 0.3;
/// Range (m) of the map obstacles given to the collision monitor.
const MAP_RANGE_M: f64 = 1.5;

/// What asks for motion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Behaviour {
    /// Frontier exploration.
    Explore,
    /// A constant straight request (`forward_v`) through the collision monitor: dry runs, stop tests.
    Forward,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NavigatorParams {
    pub safety: SafetyParams,
    pub behaviour: Behaviour,
    /// False when the pose does not move with the robot (no base, a carried dry run): the map would smear,
    /// so `Forward` then uses each frame's scan alone. `Explore` always uses the map.
    pub odometry: bool,
    /// Steps allowing forward motion needed to resume after a stop (velocity_smoother).
    pub resume_n: u32,
    /// The forward behaviour's request (m/s).
    pub forward_v: f64,
    pub grid: OccupancyGridParams,
    pub frontier: FrontierParams,
    pub explore: ExploreParams,
    /// The tilt trust in its initial state (its parameters).
    pub tilt_trust: TiltTrust,
}

impl Default for NavigatorParams {
    fn default() -> Self {
        Self {
            safety: SafetyParams::default(),
            behaviour: Behaviour::Explore,
            odometry: true,
            resume_n: 2,
            forward_v: 0.15,
            grid: OccupancyGridParams::default(),
            frontier: FrontierParams::default(),
            explore: ExploreParams {
                rotate_in_place: true,
                ..Default::default()
            },
            tilt_trust: TiltTrust::default(),
        }
    }
}

/// The result of one step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavOut {
    /// Speed to send (m/s): after the collision monitor and the resume gate.
    pub v: f64,
    /// Yaw rate to send (rad/s), > 0 counter-clockwise.
    pub w: f64,
    /// What the behaviour asked for, before the collision monitor.
    pub v_req: f64,
    pub w_req: f64,
    /// The behaviour's state.
    pub state: State,
    /// Exploration is over: Done for [`DONE_HOLD_S`].
    pub finished: bool,
    /// Forward motion is stopped only by the never-seen blind zone below the image: a look down at it (see
    /// [`Navigator::observe`]) would help.
    pub blind_stop: bool,
}

/// IEEE 754 remainder of x / y (y > 0): x - n y with n the integer nearest to x / y, ties to even. Exact.
fn ieee_remainder(x: f64, y: f64) -> f64 {
    let r = x % y; // exact, sign of x, |r| < y
    let half = y / 2.0;
    // on a tie, n is even: the truncated quotient (x - r) / y is odd -> one more period
    let odd = || ((x - r) / y).round() % 2.0 != 0.0;
    if r.abs() > half || (r.abs() == half && odd()) {
        r - y.copysign(r)
    } else {
        r
    }
}

pub struct Navigator {
    params: NavigatorParams,
    /// The map is used: always by Explore, by Forward only with odometry.
    use_map: bool,
    grid: OccupancyGrid,
    explore: FrontierExplore,
    tilt_trust: TiltTrust,
    gate: ResumeGate,
    /// Forward: the heading to hold (the pose's theta at the first step).
    heading0: Option<f64>,
    /// Since when (s) the behaviour has been Done.
    done_since: Option<f64>,
}

impl Navigator {
    pub fn new(params: NavigatorParams) -> Self {
        Self {
            use_map: params.behaviour == Behaviour::Explore || params.odometry,
            grid: OccupancyGrid::new(params.grid),
            explore: FrontierExplore::new(params.explore.clone(), params.safety, params.frontier),
            tilt_trust: params.tilt_trust,
            gate: ResumeGate::new(params.resume_n),
            heading0: None,
            done_since: None,
            params,
        }
    }

    pub fn params(&self) -> &NavigatorParams {
        &self.params
    }

    pub fn grid(&self) -> &OccupancyGrid {
        &self.grid
    }

    /// The map, e.g. to correct its keyframe poses and replay them.
    pub fn grid_mut(&mut self) -> &mut OccupancyGrid {
        &mut self.grid
    }

    /// Frontier cell centres of the last plan (fixed frame), for display.
    pub fn frontier(&self) -> &[Point2] {
        self.explore.goals.planner.frontier()
    }

    /// The planned path (fixed frame), from the robot to the target.
    pub fn path(&self) -> Option<&[Point2]> {
        self.explore.goals.planner.path()
    }

    /// The frontier point the robot is going to (fixed frame).
    pub fn target(&self) -> Option<Point2> {
        self.explore.goals.planner.target()
    }

    /// Forget the map and the behaviour's state: the robot was moved by hand, the old poses mean nothing.
    pub fn reset(&mut self) {
        let p = &self.params;
        self.grid.reset();
        self.explore = FrontierExplore::new(p.explore.clone(), p.safety, p.frontier);
        self.tilt_trust = p.tilt_trust;
        self.gate = ResumeGate::new(p.resume_n);
        (self.heading0, self.done_since) = (None, None);
    }

    /// One step. `t` time (s), `dt` since the last step; `pose` in the fixed frame; `scan` of this frame;
    /// `cam_h` camera height (m); `a`, `v`, `w` measured base motion (forward acceleration m/s^2, speed m/s,
    /// yaw rate rad/s); `v_prev` the speed command sent last step.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        t: f64,
        dt: f64,
        pose: Pose2,
        scan: &Scan,
        cam_h: f64,
        a: f64,
        v: f64,
        w: f64,
        v_prev: f64,
    ) -> NavOut {
        let sp = &{ self.params.safety }; // a copy: `self` is borrowed mutably below
        self.tilt_trust.update(dt, a, v, w, None);
        let known = self.use_map.then(|| self.grid.blind_known(pose, scan, None, 0.1));
        if self.use_map {
            let trusted = self.tilt_trust.max_range(cam_h);
            self.grid.update(
                t,
                pose,
                scan,
                Some(self.tilt_trust.sigma),
                cam_h,
                Some(trusted),
                false,
            );
        }
        let (v_req, w_req, state) = match self.params.behaviour {
            Behaviour::Forward => {
                let heading0 = *self.heading0.get_or_insert(pose.theta);
                let err = ieee_remainder(pose.theta - heading0, TAU);
                let w_hold = (-HEADING_K * err).clamp(-HEADING_W_MAX, HEADING_W_MAX);
                (self.params.forward_v, w_hold, State::Forward)
            }
            Behaviour::Explore => {
                let (v_req, w_req) = self.explore.step(t, pose, Some(scan), &self.grid, v_prev);
                (v_req, w_req, self.explore.state())
            }
        };
        let mut blind_stop = false;
        let (v_ok, w_ok, _) = match &known {
            Some(known) => {
                let pts = self.grid.obstacle_points(pose, MAP_RANGE_M);
                let ok = safe_command(v_req, w_req, scan, sp, Some(&pts), Some(known));
                let unexplained = (0..scan.len()).any(|k| scan.seen[k] && scan.near[k] && !known[k]);
                if v_req > 0.0 && ok.0 <= 0.0 && unexplained {
                    // would the request go if the unseen blind segments were known free? then only they
                    // stop it
                    let all = vec![true; scan.len()];
                    blind_stop = safe_command(v_req, w_req, scan, sp, Some(&pts), Some(&all)).0 > 0.0;
                }
                ok
            }
            None => safe_command(v_req, w_req, scan, sp, None, None),
        };
        // A robot that sees nothing does not translate (see the module documentation). The gate takes it
        // as any stop; it is not a blind_stop (a look down would not help a camera that sees no floor).
        let (v_ok, blind_stop) = if scan.seen.iter().any(|&s| s) {
            (v_ok, blind_stop)
        } else {
            (0.0, false)
        };
        let v_ok = self.gate.step(v_req, v_ok);
        if state != State::Done {
            self.done_since = None;
        } else if self.done_since.is_none() {
            self.done_since = Some(t);
        }
        NavOut {
            v: v_ok,
            w: w_ok,
            v_req,
            w_req,
            state,
            // n Done steps so far span (t - done_since) + dt
            finished: self.done_since.is_some_and(|since| t - since + dt > DONE_HOLD_S),
            blind_stop,
        }
    }

    /// [`Self::step`] from the floor mask: `fs` turns `floor` (true = floor) into the scan, with 'up' in the
    /// camera frame `up_c` and the camera height `cam_h`. Returns the scan as well (logs, display).
    #[allow(clippy::too_many_arguments)]
    pub fn step_mask(
        &mut self,
        fs: &mut FreeSpace,
        floor: &[bool],
        up_c: &Vec3,
        t: f64,
        dt: f64,
        pose: Pose2,
        cam_h: f64,
        a: f64,
        v: f64,
        w: f64,
        v_prev: f64,
    ) -> Result<(NavOut, Scan), ShapeError> {
        let scan = fs.scan(floor, up_c, cam_h)?;
        Ok((self.step(t, dt, pose, &scan, cam_h, a, v, w, v_prev), scan))
    }

    /// A head-glance frame (robot holding still, head looking down at the blind zone): into the map only,
    /// always as a keyframe; no behaviour, no command. Weighted and range-gated by the current tilt trust as
    /// any frame (the base may still be settling).
    pub fn observe(&mut self, t: f64, pose: Pose2, scan: &Scan, cam_h: f64) {
        if self.use_map {
            let trusted = self.tilt_trust.max_range(cam_h);
            self.grid.update(
                t,
                pose,
                scan,
                Some(self.tilt_trust.sigma),
                cam_h,
                Some(trusted),
                true,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const BLIND: f64 = 0.45;

    fn bearing(k: usize) -> f64 {
        (k as f64 + 0.5) * 2.0 - 180.0
    }

    /// Scan of an empty room [0, sx] x [0, sy] from `pose`: 55 deg half-view, floor seen from BLIND to 2.5 m.
    fn room_scan(pose: Pose2, (sx, sy): (f64, f64)) -> Scan {
        let n = 180;
        let seen: Vec<bool> = (0..n).map(|k| bearing(k).abs() < 55.0).collect();
        let d: Vec<f64> = (0..n)
            .map(|k| {
                let (s, c) = (pose.theta + bearing(k).to_radians()).sin_cos();
                let hit = |room: f64, at: f64, dir: f64| {
                    if dir > 1e-9 {
                        (room - at) / dir
                    } else if dir < -1e-9 {
                        -at / dir
                    } else {
                        f64::INFINITY
                    }
                };
                hit(sx, pose.x, c).min(hit(sy, pose.y, s))
            })
            .collect();
        let obstacle: Vec<f64> = (0..n)
            .map(|k| {
                if seen[k] && d[k] < 2.5 {
                    d[k]
                } else {
                    f64::INFINITY
                }
            })
            .collect();
        Scan {
            bearing_deg: (0..n).map(bearing).collect(),
            near: obstacle.iter().map(|&o| o <= BLIND + 0.05).collect(),
            floor: (0..n)
                .map(|k| if seen[k] { d[k].min(2.5) } else { 0.0 })
                .collect(),
            closest: seen
                .iter()
                .map(|&s| if s { BLIND } else { f64::INFINITY })
                .collect(),
            obstacle,
            seen,
        }
    }

    const ROOM: (f64, f64) = (4.0, 3.0);

    fn forward(odometry: bool, resume_n: u32, forward_v: f64) -> Navigator {
        Navigator::new(NavigatorParams {
            behaviour: Behaviour::Forward,
            odometry,
            resume_n,
            forward_v,
            ..Default::default()
        })
    }

    fn step(nav: &mut Navigator, t: f64, pose: Pose2, scan_from: Pose2) -> NavOut {
        nav.step(
            t,
            0.1,
            pose,
            &room_scan(scan_from, ROOM),
            0.35,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    }

    #[test]
    fn remainder_is_the_ieee_one() {
        assert_eq!(ieee_remainder(0.5, TAU), 0.5);
        assert_eq!(ieee_remainder(-0.5, TAU), -0.5);
        assert_eq!(ieee_remainder(4.0, TAU), 4.0 - TAU);
        assert_eq!(ieee_remainder(-4.0, TAU), TAU - 4.0);
        assert_eq!(ieee_remainder(3.0 * TAU + 0.25, TAU), (3.0 * TAU + 0.25) % TAU);
        assert_eq!(ieee_remainder(PI, TAU), PI);
        // ties go to the even quotient: 0, 2, -2, 2
        assert_eq!(ieee_remainder(1.0, 2.0), 1.0);
        assert_eq!(ieee_remainder(3.0, 2.0), -1.0);
        assert_eq!(ieee_remainder(-3.0, 2.0), 1.0);
        assert_eq!(ieee_remainder(5.0, 2.0), 1.0);
    }

    #[test]
    fn collision_monitor_holds_a_forward_request_before_a_wall() {
        let mut nav = forward(true, 1, 0.3);
        // robot edge 5 cm from the wall at x = 4, facing it
        let near = Pose2::new(4.0 - 0.13 - 0.05, 1.5, 0.0);
        let out = step(&mut nav, 0.0, near, near);
        assert!(out.v_req == 0.3 && out.v <= 1e-9 && out.state == State::Forward);
        let far = Pose2::new(1.0, 1.5, 0.0); // 2.9 m of floor ahead: the request goes through
        assert!(step(&mut nav, 0.1, far, far).v > 0.1);
    }

    #[test]
    fn forward_without_odometry_uses_the_scan_alone_and_the_gate_holds_the_first_clear_step() {
        let mut nav = forward(false, 2, 0.15);
        let near = Pose2::new(4.0 - 0.13 - 0.05, 1.5, 0.0);
        let far = Pose2::new(1.0, 1.5, 0.0);
        assert!(step(&mut nav, 0.0, near, near).v <= 1e-9);
        // the same pose with an open scan: the obstacles seen earlier are not in a map
        assert_eq!(step(&mut nav, 0.1, near, far).v, 0.0); // gate: 1st clear step
        let out = step(&mut nav, 0.2, near, far);
        assert_eq!(
            (out.v, out.w, out.v_req, out.state),
            (0.15, 0.0, 0.15, State::Forward)
        );
        assert!(nav.grid().keyframes().is_empty());
    }

    #[test]
    fn a_scan_that_sees_nothing_gives_no_translation() {
        let blind = |pose: Pose2| {
            let mut s = room_scan(pose, ROOM);
            let n = s.len();
            (s.seen, s.near) = (vec![false; n], vec![false; n]);
            (s.obstacle, s.closest) = (vec![f64::INFINITY; n], vec![f64::INFINITY; n]);
            s.floor = vec![0.0; n];
            s
        };
        let far = Pose2::new(1.0, 1.5, 0.0);
        // the collision monitor alone lets a blind robot creep: the boundary is at unseen_m all round
        let sp = SafetyParams::default();
        assert!(safe_command(0.15, 0.0, &blind(far), &sp, None, None).0 > 0.05);
        for odometry in [true, false] {
            let mut nav = forward(odometry, 2, 0.15);
            assert!(step(&mut nav, 0.0, far, far).v > 0.1); // sees the room: drives
            let turned = Pose2::new(1.0, 1.5, -0.1);
            for k in 1..20 {
                let out = nav.step(
                    k as f64 * 0.1,
                    0.1,
                    turned,
                    &blind(turned),
                    0.35,
                    0.0,
                    0.0,
                    0.0,
                    0.15,
                );
                // no translation; the turn (here the heading hold), the request and the state are kept
                assert_eq!((out.v, out.v_req, out.state), (0.0, 0.15, State::Forward));
                assert!(out.w > 0.1 && out.w == out.w_req && !out.blind_stop);
            }
            // seeing again: the resume gate holds the first clear step, as after any stop
            assert_eq!(step(&mut nav, 2.0, far, far).v, 0.0);
            assert!(step(&mut nav, 2.1, far, far).v > 0.1);
        }
        let mut nav = Navigator::new(NavigatorParams::default());
        for k in 0..50 {
            let out = nav.step(k as f64 * 0.1, 0.1, far, &blind(far), 0.35, 0.0, 0.0, 0.0, 0.0);
            assert!(out.v == 0.0 && !out.blind_stop && out.w.is_finite());
        }
    }

    #[test]
    fn forward_holds_the_start_heading() {
        let mut nav = forward(true, 1, 0.15);
        let at = |x: f64, deg: f64| Pose2::new(x, 1.5, deg.to_radians());
        assert_eq!(step(&mut nav, 0.0, at(1.0, 0.0), at(1.0, 0.0)).w_req, 0.0);
        // turned right (clockwise): steer left
        assert!(step(&mut nav, 0.1, at(1.1, -4.0), at(1.1, -4.0)).w_req > 0.05);
        assert_eq!(
            step(&mut nav, 0.2, at(1.2, 30.0), at(1.2, 30.0)).w_req,
            -HEADING_W_MAX
        );
    }

    #[test]
    fn explores_then_finishes_after_done_for_3_s_and_reset_starts_again() {
        // a closed 1.6 x 1.6 m room: explored by looking around
        let room = (1.6, 1.6);
        let mut nav = Navigator::new(NavigatorParams::default());
        let dt = 1.0 / 8.7;
        let mut pose = Pose2::new(0.8, 0.8, 0.0);
        let (mut v, mut w) = (0.0, 0.0);
        let mut states = Vec::new();
        let mut finished_at = None;
        for k in 0..1500 {
            let out = nav.step(
                k as f64 * dt,
                dt,
                pose,
                &room_scan(pose, room),
                0.35,
                0.0,
                v,
                w,
                v,
            );
            (v, w) = (out.v, out.w); // perfect base: the command is applied for the next step
            pose.theta += w * dt;
            pose.x += v * dt * pose.theta.cos();
            pose.y += v * dt * pose.theta.sin();
            assert!(
                pose.x.min(pose.y).min(1.6 - pose.x.max(pose.y)) > 0.13,
                "touched a wall"
            );
            states.push(out.state);
            if out.finished {
                finished_at = Some(k);
                break;
            }
        }
        let finished_at = finished_at.expect("exploration finishes");
        // Done steps before finished: n dt > 3 s
        let first_done = states.iter().rposition(|&s| s != State::Done).unwrap() + 1;
        assert_eq!(
            finished_at - first_done + 1,
            (DONE_HOLD_S / dt).floor() as usize + 1
        );
        assert!(nav.path().is_none() && nav.target().is_none());
        assert!(!nav.grid().keyframes().is_empty());

        nav.reset();
        assert!(nav.grid().keyframes().is_empty() && nav.grid().origin().is_none());
        let out = nav.step(0.0, dt, pose, &room_scan(pose, room), 0.35, 0.0, 0.0, 0.0, 0.0);
        assert!(!out.finished && out.state != State::Done);
    }

    #[test]
    fn observe_goes_into_the_map_before_any_step() {
        let mut nav = Navigator::new(NavigatorParams::default());
        let pose = Pose2::new(1.0, 1.5, 0.0);
        nav.observe(0.0, pose, &room_scan(pose, ROOM), 0.33);
        nav.observe(0.1, pose, &room_scan(pose, ROOM), 0.33); // forced: a keyframe again
        assert_eq!(nav.grid().keyframes().len(), 2);
        // without a map, nothing
        let mut nav = forward(false, 2, 0.15);
        nav.observe(0.0, pose, &room_scan(pose, ROOM), 0.33);
        assert!(nav.grid().keyframes().is_empty());
    }
}
