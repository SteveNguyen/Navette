//! One navigation step on a free-space scan: occupancy grid -> behaviour -> collision monitor -> resume gate.
//!
//! Per frame, in this order:
//!
//! ```text
//! near filter  NearFilter::apply: a near sector counts only if near in near_k of the last near_n scans
//!              (fixed-frame bearings); every step below sees the filtered flags   (off by default)
//! tilt trust   TiltTrust::update(dt, a, v, w) from the base motion; trusted range for the map
//! blind zone  OccupancyGrid::blind_known(pose, scan): which image-bottom obstacles the map explains,
//!              asked BEFORE this frame goes in (the scan must not explain itself)
//! map          OccupancyGrid::update(t, pose, scan, sigma, cam_h, trusted range)    (keyframes only)
//! behaviour    FrontierExplore::step(t, pose, scan, grid, v_prev), or the constant forward request, or the
//!              operator's (teleop, set_request; (0, 0) once older than teleop_timeout_s) -> requested (v, w)
//! safety       safe_command(v, w, scan, params, grid obstacles, blind_known)
//! resume gate  ResumeGate::step(v_req, v_ok)                                        -> (v, w) to send
//!              (while stopped, with resume_margin > 0: v_ok of safe_command with margin + resume_margin)
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
//!
//! Another, active only with `resume_margin > 0` (0 by default): a restart hysteresis. After a stop the
//! request must fit with `resume_margin` more clearance, `resume_n` steps in a row, before the robot moves
//! again; stops are unchanged. A self-balancing base still travels the lag it built after any stop, and most
//! after a short start-stop burst, which the stop-and-go at the edge of a blocked zone produces.
//!
//! Another, active only with `near_k > 1` (1 of 1 by default): near persistence ([`NearFilter`]). A sector's
//! `near` flag (an obstacle filling the image bottom, which the collision monitor and the arcs put at the
//! clearance radius) counts only if that bearing was near in at least `near_k` of the last `near_n` scans.
//! The floor model flickers: single-frame false "not floor" patches at the image bottom (a bi-stable doormat,
//! tiles near an open door) stopped or blocked the robot, while the map averages them out over time.

use std::collections::VecDeque;
use std::f64::consts::TAU;

use crate::collision_monitor::{safe_command, SafetyParams};
use crate::controller::dwa::ExploreParams;
use crate::controller::State;
use crate::exploration::frontier::FrontierParams;
use crate::exploration::frontier_explore::FrontierExplore;
use crate::geometry::{Point2, Pose2, Vec3};
use crate::occupancy_grid::{OccupancyGrid, OccupancyGridParams};
use crate::params::{self, ParamError};
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
    /// The operator's request ([`Navigator::set_request`]) through the collision monitor, the resume gate and
    /// every watchdog downstream, as a planner's would go: driving by hand (phone joystick). The map is kept as
    /// with Forward (with odometry), for the collision monitor's near rule and the logs. Operator dead-man: a
    /// request older than `teleop_timeout_s` (or none yet) is (0, 0).
    Teleop,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NavigatorParams {
    pub safety: SafetyParams,
    pub behaviour: Behaviour,
    /// False when the pose does not move with the robot (no base, a carried dry run): the map would smear,
    /// so `Forward` and `Teleop` then use each frame's scan alone. `Explore` always uses the map.
    pub odometry: bool,
    /// Steps allowing forward motion needed to resume after a stop (velocity_smoother).
    pub resume_n: u32,
    /// Restart hysteresis (m): after a stop, the resume gate counts a step as allowing forward motion only
    /// if the request fits with the collision monitor's margin increased by this much. Stops stay immediate,
    /// at the normal margin. 0: no hysteresis (the request is evaluated once, as the prototype).
    /// Why: a self-balancing base still travels the lag it built after any stop, most after a short burst;
    /// stop-and-go at the edge of a blocked zone produces exactly such bursts.
    pub resume_margin: f64,
    /// The forward behaviour's request (m/s).
    pub forward_v: f64,
    /// Near persistence ([`NearFilter`]): a sector is `near` only if near in at least `near_k` of the last
    /// `near_n` scans, this one included. 1 <= near_k <= near_n; near_k = 1: no filter (the prototype).
    pub near_k: u32,
    pub near_n: u32,
    /// Teleop: the operator's request holds this long (s) after it was set, measured on the step's `t`; then
    /// (0, 0) until the next one.
    pub teleop_timeout_s: f64,
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
            resume_margin: 0.0,
            forward_v: 0.15,
            near_k: 1,
            near_n: 1,
            teleop_timeout_s: 0.5,
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

impl NavigatorParams {
    /// The parameters of every part, finite forward_v, finite resume_margin >= 0, 1 <= near_k <= near_n,
    /// finite teleop_timeout_s > 0 and the tilt trust parameters.
    pub fn validate(&self) -> Result<(), ParamError> {
        self.safety.validate()?;
        self.grid.validate()?;
        self.frontier.validate()?;
        self.explore.validate()?;
        params::finite("NavigatorParams.forward_v", self.forward_v)?;
        params::non_negative("NavigatorParams.resume_margin", self.resume_margin)?;
        if self.near_k < 1 || self.near_k > self.near_n {
            return Err(ParamError(format!(
                "NavigatorParams.near_k = {} (near_n = {}): must be in [1, near_n]",
                self.near_k, self.near_n
            )));
        }
        params::positive("NavigatorParams.teleop_timeout_s", self.teleop_timeout_s)?;
        let t = &self.tilt_trust;
        params::non_negative("TiltTrust.s0", t.s0)?;
        params::positive("TiltTrust.tau_s", t.tau_s)?;
        params::non_negative("TiltTrust.tol_m", t.tol_m)?;
        params::non_negative("TiltTrust.timing_s", t.timing_s)
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

/// Near persistence: a sector's `near` flag counts only if that bearing was near in at least `k` of the last
/// `n` scans (this one included). A sector that is near now but not confirmed keeps its obstacle distance as a
/// normal (far) obstacle: only the "may be closer than seen" flag waits.
///
/// The robot turns between frames, so bearings are compared in the fixed (odometry) frame: a sector of this
/// scan at bearing b (camera frame, deg) is at theta + b; a past scan taken at heading theta_j saw that
/// direction at theta + b - theta_j, and counts if any of its sectors within +-1 sector of it was near (the
/// heading error between frames and a patch straddling a sector edge).
///
/// Safety: a real obstacle filling the image bottom stays near frame after frame, so it is confirmed one frame
/// later with 2 of 3 (k - 1 frames in general). One frame is ~0.12 s at the robot's ~8 Hz: at <= 0.15 m/s,
/// <= 2 cm, inside the collision monitor's margin (5 cm); and the unconfirmed sector still stops the robot at
/// its seen distance (the obstacle stays in the scan). A single-frame false patch is never confirmed.
#[derive(Clone, Debug)]
pub struct NearFilter {
    k: usize,
    n: usize,
    /// The last n - 1 scans' raw near flags with the heading (rad) they were taken at, newest first.
    history: VecDeque<(f64, Vec<bool>)>,
}

impl NearFilter {
    pub fn new(k: u32, n: u32) -> Self {
        Self {
            k: k as usize,
            n: n as usize,
            history: VecDeque::new(),
        }
    }

    /// The filtered near flags of `scan`, taken at heading `theta` (rad, fixed frame); the scan's raw flags go
    /// into the history. Sectors: the full circle in equal sectors from -180 deg, as [`FreeSpace`] makes them.
    pub fn apply(&mut self, theta: f64, scan: &Scan) -> Vec<bool> {
        let near = (0..scan.len())
            .map(|i| {
                if !scan.near[i] {
                    return false;
                }
                let fixed_deg = theta.to_degrees() + scan.bearing_deg[i];
                let before = self
                    .history
                    .iter()
                    .filter(|(theta_j, near_j)| near_around(near_j, fixed_deg - theta_j.to_degrees()))
                    .count();
                1 + before >= self.k
            })
            .collect();
        self.history.push_front((theta, scan.near.clone()));
        self.history.truncate(self.n - 1);
        near
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }
}

/// Some sector of `near` (full circle, equal sectors from -180 deg) within +-1 sector of `bearing_deg` is
/// near. Any bearing: the sector index wraps.
fn near_around(near: &[bool], bearing_deg: f64) -> bool {
    let m = near.len() as i64;
    if m == 0 {
        return false;
    }
    let j = ((bearing_deg + 180.0) / (360.0 / m as f64)).floor() as i64;
    (-1..=1).any(|d| near[(j + d).rem_euclid(m) as usize])
}

pub struct Navigator {
    params: NavigatorParams,
    /// The map is used: always by Explore, by Forward and Teleop only with odometry.
    use_map: bool,
    grid: OccupancyGrid,
    explore: FrontierExplore,
    tilt_trust: TiltTrust,
    gate: ResumeGate,
    near_filter: NearFilter,
    /// The near flags the last step used (after the near filter).
    near: Vec<bool>,
    /// Teleop: the operator's last request (v, w, t set).
    request: Option<(f64, f64, f64)>,
    /// Forward: the heading to hold (the pose's theta at the first step).
    heading0: Option<f64>,
    /// Since when (s) the behaviour has been Done.
    done_since: Option<f64>,
}

impl Navigator {
    /// An error if a parameter is invalid ([`NavigatorParams::validate`]).
    pub fn new(params: NavigatorParams) -> Result<Self, ParamError> {
        params.validate()?;
        Ok(Self {
            use_map: params.behaviour == Behaviour::Explore || params.odometry,
            grid: OccupancyGrid::new(params.grid)?,
            explore: FrontierExplore::new(params.explore.clone(), params.safety, params.frontier)?,
            tilt_trust: params.tilt_trust,
            gate: ResumeGate::new(params.resume_n),
            near_filter: NearFilter::new(params.near_k, params.near_n),
            near: Vec::new(),
            request: None,
            heading0: None,
            done_since: None,
            params,
        })
    }

    /// Teleop: the operator's request (m/s, rad/s > 0 counter-clockwise), set at time `t` (s, the clock of
    /// the steps' `t`). Each step takes the last one, through the collision monitor and the resume gate, until
    /// it is older than `teleop_timeout_s`. A non-finite value is taken as (0, 0). Other behaviours ignore it.
    pub fn set_request(&mut self, v: f64, w: f64, t: f64) {
        self.request = Some(if v.is_finite() && w.is_finite() {
            (v, w, t)
        } else {
            (0.0, 0.0, t)
        });
    }

    /// The near flags of the last step's scan after the near filter (what every part of the step used);
    /// empty before the first step.
    pub fn near(&self) -> &[bool] {
        &self.near
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

    /// Blocks (recovery turns started) of the exploration behaviour since the start or the last reset.
    pub fn blocks(&self) -> u32 {
        self.explore.controller.blocks
    }

    /// Full looks (`ExploreParams::look360_m`) started by the exploration behaviour since the start or the last
    /// reset.
    pub fn full_looks(&self) -> u32 {
        self.explore.full_looks
    }

    /// Forget the map and the behaviour's state: the robot was moved by hand, the old poses mean nothing.
    pub fn reset(&mut self) {
        let p = &self.params;
        self.grid.reset();
        self.explore = FrontierExplore::new(p.explore.clone(), p.safety, p.frontier)
            .expect("the parameters were validated by Navigator::new");
        self.tilt_trust = p.tilt_trust;
        self.gate = ResumeGate::new(p.resume_n);
        self.near_filter.reset();
        self.near.clear();
        self.request = None;
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

        // near persistence first, so that the map's blind check, the behaviour and the collision monitor all
        // see the same near flags (skipped with near_k = 1: the scan as given)
        let filtered;
        let scan = if self.params.near_k > 1 {
            filtered = Scan {
                near: self.near_filter.apply(pose.theta, scan),
                ..scan.clone()
            };
            &filtered
        } else {
            scan
        };
        self.near.clone_from(&scan.near);
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
            Behaviour::Teleop => match self.request {
                // a NaN t (never) is stale; a request set after this frame's capture (t_set > t) is current
                Some((v, w, t_set)) if t - t_set <= self.params.teleop_timeout_s => (v, w, State::Teleop),
                _ => (0.0, 0.0, State::Teleop),
            },
            Behaviour::Explore => {
                let (v_req, w_req) =
                    self.explore
                        .step(t, pose, Some(scan), &self.grid, v_prev, w, known.as_deref());
                (v_req, w_req, self.explore.state())
            }
        };
        let mut blind_stop = false;
        // the map's obstacles, also for the resume check below
        let pts = known
            .as_ref()
            .map(|_| self.grid.obstacle_points(pose, MAP_RANGE_M));
        let (v_ok, w_ok, _) = match (&known, &pts) {
            (Some(known), Some(pts)) => {
                let ok = safe_command(v_req, w_req, scan, sp, Some(pts), Some(known));
                let unexplained = (0..scan.len()).any(|k| scan.seen[k] && scan.near[k] && !known[k]);
                if v_req > 0.0 && ok.0 <= 0.0 && unexplained {
                    // would the request go if the unseen blind segments were known free? then only they
                    // stop it
                    let all = vec![true; scan.len()];
                    blind_stop = safe_command(v_req, w_req, scan, sp, Some(pts), Some(&all)).0 > 0.0;
                }
                ok
            }
            _ => safe_command(v_req, w_req, scan, sp, None, None),
        };
        // A robot that sees nothing does not translate (see the module documentation). The gate takes it
        // as any stop; it is not a blind_stop (a look down would not help a camera that sees no floor).
        let (v_ok, blind_stop) = if scan.seen.iter().any(|&s| s) {
            (v_ok, blind_stop)
        } else {
            (0.0, false)
        };
        // Restart hysteresis: separate stop and resume zones. While the gate is stopped, the request must
        // also fit with resume_margin more clearance (same scan, map points and near_known) for the step to
        // count toward resuming; a step that fits only with the normal margin counts as blocked. Skipped when
        // the request is not forward or already cut (a wider margin cannot allow more) and when
        // resume_margin = 0, so that the outputs stay the prototype's.
        let v_ok = if self.params.resume_margin > 0.0 && self.gate.stopped() && v_req > 0.0 && v_ok > 0.0 {
            let wider = SafetyParams {
                margin: sp.margin + self.params.resume_margin,
                ..*sp
            };
            safe_command(v_req, w_req, scan, &wider, pts.as_deref(), known.as_deref()).0
        } else {
            v_ok
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
    /// camera frame `up_c` and the camera height `cam_h`. Returns the scan as well (logs, display), as `fs`
    /// made it: the near flags the step used are [`Self::near`].
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
        .unwrap()
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
    fn bad_parameters_are_errors() {
        let bad = |change: fn(&mut NavigatorParams)| {
            let mut p = NavigatorParams::default();
            change(&mut p);
            Navigator::new(p).err().expect("invalid").0
        };
        assert!(bad(|p| p.safety.step = 0.0).starts_with("SafetyParams.step"));
        assert!(bad(|p| p.safety.decel = 0.0).starts_with("SafetyParams.decel"));
        assert!(bad(|p| p.grid.res = 0.0).starts_with("OccupancyGridParams.res"));
        assert!(bad(|p| p.frontier.replan_s = f64::NAN).starts_with("FrontierParams.replan_s"));
        assert!(bad(|p| p.explore.v_set.clear()).starts_with("ExploreParams.v_set"));
        assert!(bad(|p| p.forward_v = f64::NAN).starts_with("NavigatorParams.forward_v"));
        assert!(bad(|p| p.resume_margin = -0.1).starts_with("NavigatorParams.resume_margin"));
        assert!(bad(|p| p.resume_margin = f64::NAN).starts_with("NavigatorParams.resume_margin"));
        assert!(bad(|p| p.tilt_trust.tau_s = 0.0).starts_with("TiltTrust.tau_s"));
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
        let mut nav = Navigator::new(NavigatorParams::default()).unwrap();
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
        let mut nav = Navigator::new(NavigatorParams::default()).unwrap();
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

    /// Scan of a straight wall across the path, `d` m ahead of the centre, floor seen up to it from 0.1 m
    /// (never `near`): the collision monitor's clearance alone decides.
    fn wall_scan(d: f64) -> Scan {
        let n = 180;
        let seen: Vec<bool> = (0..n).map(|k| bearing(k).abs() < 55.0).collect();
        let free: Vec<f64> = (0..n).map(|k| d / bearing(k).to_radians().cos()).collect();
        Scan {
            bearing_deg: (0..n).map(bearing).collect(),
            obstacle: (0..n)
                .map(|k| if seen[k] { free[k] } else { f64::INFINITY })
                .collect(),
            near: vec![false; n],
            floor: (0..n).map(|k| if seen[k] { free[k] } else { 0.0 }).collect(),
            closest: seen
                .iter()
                .map(|&s| if s { 0.1 } else { f64::INFINITY })
                .collect(),
            seen,
        }
    }

    /// Forward without odometry (the scan alone) at 0.15 m/s, with this resume gate.
    fn forward_resume(resume_n: u32, resume_margin: f64) -> Navigator {
        Navigator::new(NavigatorParams {
            behaviour: Behaviour::Forward,
            odometry: false,
            resume_n,
            resume_margin,
            ..Default::default()
        })
        .unwrap()
    }

    fn v_at(nav: &mut Navigator, k: usize, d: f64) -> f64 {
        let pose = Pose2::new(0.0, 0.0, 0.0);
        nav.step(k as f64 * 0.1, 0.1, pose, &wall_scan(d), 0.35, 0.0, 0.0, 0.0, 0.0)
            .v
    }

    // With the default parameters (radius 0.13, margin 0.05) the 0.15 m/s request gets some v > 0 from a wall
    // at 0.21 m on, none at 0.19; 0.10 m more margin moves that edge by 0.10 m.
    const FITS: f64 = 0.25; // fits with the margin, not with margin + 0.10
    const BLOCKED: f64 = 0.15;
    const FITS_WIDER: f64 = 0.35; // fits with margin + 0.10

    #[test]
    fn resume_margin_0_is_the_monitor_then_the_gate() {
        let sp = SafetyParams::default();
        assert!(safe_command(0.15, 0.0, &wall_scan(0.21), &sp, None, None).0 > 0.0);
        assert_eq!(safe_command(0.15, 0.0, &wall_scan(0.19), &sp, None, None).0, 0.0);
        let wider = SafetyParams {
            margin: sp.margin + 0.1,
            ..sp
        };
        assert!(safe_command(0.15, 0.0, &wall_scan(0.31), &wider, None, None).0 > 0.0);
        assert_eq!(
            safe_command(0.15, 0.0, &wall_scan(0.29), &wider, None, None).0,
            0.0
        );
        // a stop-and-go sequence: exactly safe_command then ResumeGate, as before the parameter existed
        let walls = [
            1.0, 0.4, FITS, BLOCKED, FITS, FITS, BLOCKED, FITS, 0.6, 0.6, BLOCKED, FITS_WIDER, 1.0,
        ];
        let mut nav = forward_resume(2, 0.0);
        let mut gate = ResumeGate::new(2);
        for (k, &d) in walls.iter().enumerate() {
            let want = gate.step(0.15, safe_command(0.15, 0.0, &wall_scan(d), &sp, None, None).0);
            assert_eq!(v_at(&mut nav, k, d), want, "step {k}");
        }
    }

    #[test]
    fn after_a_stop_the_request_must_fit_with_resume_margin_more() {
        let (mut plain, mut hyst) = (forward_resume(2, 0.0), forward_resume(2, 0.1));
        // moving toward the wall: no change, the normal margin decides (immediate stop)
        for (k, d) in [1.0, 0.5, FITS].into_iter().enumerate() {
            let v = v_at(&mut plain, k, d);
            assert!(v > 0.0 && v_at(&mut hyst, k, d) == v, "step {k}");
        }
        assert_eq!(
            (v_at(&mut plain, 3, BLOCKED), v_at(&mut hyst, 3, BLOCKED)),
            (0.0, 0.0)
        );
        // stopped, the wall back where it fitted: the plain gate resumes after 2 steps, the other holds
        assert_eq!(v_at(&mut plain, 4, FITS), 0.0);
        assert!(v_at(&mut plain, 5, FITS) > 0.0);
        for k in 4..30 {
            assert_eq!(v_at(&mut hyst, k, FITS), 0.0, "step {k}");
        }
        // resume_margin farther: it goes, after resume_n steps
        assert_eq!(v_at(&mut hyst, 30, FITS_WIDER), 0.0);
        assert!(v_at(&mut hyst, 31, FITS_WIDER) > 0.0);
        // moving again: back to the normal margin, the wall may come as close as before
        assert!(v_at(&mut hyst, 32, FITS) > 0.0);
    }

    #[test]
    fn resume_after_resume_n_steps_once_it_fits_wider() {
        for n in [1, 3, 4] {
            let mut nav = forward_resume(n, 0.1);
            assert!(v_at(&mut nav, 0, 1.0) > 0.0);
            assert_eq!(v_at(&mut nav, 1, BLOCKED), 0.0);
            let mut k = 2;
            let mut v = |d: f64| {
                k += 1;
                v_at(&mut nav, k, d)
            };
            for _ in 1..n {
                assert_eq!(v(FITS_WIDER), 0.0, "n = {n}");
            }
            // a step that fits only with the normal margin restarts the count, as a blocked one
            assert_eq!(v(FITS), 0.0, "n = {n}");
            for _ in 1..n {
                assert_eq!(v(FITS_WIDER), 0.0, "n = {n}");
            }
            assert!(v(FITS_WIDER) > 0.0, "n = {n}");
        }
    }

    /// Open floor (a wall 2 m ahead, floor seen from 0.1 m) with, if `near`, a patch filling the image bottom
    /// in the sectors at `bearing` +-3 deg (camera frame): at 0.45 m, where the camera's view starts there.
    fn patch_scan(near: bool, bearing: f64) -> Scan {
        let mut s = wall_scan(2.0);
        if near {
            for k in 0..s.len() {
                if (s.bearing_deg[k] - bearing).abs() < 3.0 {
                    (s.obstacle[k], s.floor[k], s.closest[k], s.near[k]) = (0.45, 0.45, 0.45, true);
                }
            }
        }
        s
    }

    #[test]
    fn near_filter_needs_k_of_the_last_n_scans() {
        let mut f = NearFilter::new(2, 3);
        let near_ahead = |f: &mut NearFilter, near: bool| f.apply(0.0, &patch_scan(near, 0.0))[90];
        assert!(!near_ahead(&mut f, true)); // first time: not confirmed
        assert!(near_ahead(&mut f, true)); // two in a row
        assert!(!near_ahead(&mut f, false)); // a sector that is not near now never is
        assert!(near_ahead(&mut f, true)); // near 2 scans ago and now: 2 of the last 3
        let mut f = NearFilter::new(2, 3);
        for near in [true, false, false] {
            near_ahead(&mut f, near);
        }
        // 3 scans ago: out of the window
        assert!(!near_ahead(&mut f, true));
        // 1 of 1 and 1 of n: the scan's own flags
        for n in [1, 3] {
            let mut f = NearFilter::new(1, n);
            let s = patch_scan(true, 0.0);
            assert_eq!(f.apply(0.0, &s), s.near);
        }
    }

    #[test]
    fn near_filter_compares_fixed_frame_bearings() {
        // the robot turned 10 deg left between the frames: an obstacle straight ahead before is now at -10 deg
        let turned = 10f64.to_radians();
        let mut f = NearFilter::new(2, 2);
        f.apply(0.0, &patch_scan(true, 0.0));
        let now = f.apply(turned, &patch_scan(true, -10.0));
        // -11 and -9 deg: confirmed
        let k = (0..180).position(|k| bearing(k) == -11.0).unwrap();
        assert!(now[k] && now[k + 1]);
        // the same patch with the heading not taken into account would be 10 deg off: not confirmed
        let mut f = NearFilter::new(2, 2);
        f.apply(0.0, &patch_scan(true, 0.0));
        assert!(f.apply(0.0, &patch_scan(true, -10.0)).iter().all(|&n| !n));
        // across +-180 deg: the sector index wraps
        let mut f = NearFilter::new(2, 2);
        f.apply(PI, &patch_scan(true, 0.0));
        assert!(f.apply(-PI + 1e-9, &patch_scan(true, 0.0))[90]);
    }

    fn forward_near(near_k: u32, near_n: u32) -> Navigator {
        Navigator::new(NavigatorParams {
            behaviour: Behaviour::Forward,
            odometry: false,
            resume_n: 1,
            near_k,
            near_n,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn a_single_frame_near_patch_does_not_stop_a_persistent_one_does() {
        let pose = Pose2::new(0.0, 0.0, 0.0);
        let v = |nav: &mut Navigator, k: usize, near: bool| {
            let s = patch_scan(near, 0.0);
            let out = nav.step(k as f64 * 0.1, 0.1, pose, &s, 0.35, 0.0, 0.0, 0.0, 0.0);
            assert_eq!(nav.near().len(), s.len());
            out.v
        };
        // today (1 of 1): any near frame stops the robot
        let mut plain = forward_near(1, 1);
        assert!(v(&mut plain, 0, false) > 0.0);
        assert_eq!(v(&mut plain, 1, true), 0.0);
        assert_eq!(plain.near(), patch_scan(true, 0.0).near.as_slice());
        // 2 of 3: a one-frame flicker is a normal obstacle at 0.45 m, which leaves room to go on
        let mut nav = forward_near(2, 3);
        assert!(v(&mut nav, 0, false) > 0.0);
        assert!(v(&mut nav, 1, true) > 0.0);
        assert!(nav.near().iter().all(|&n| !n));
        assert!(v(&mut nav, 2, false) > 0.0);
        // a patch that stays: stopped from its second frame
        assert!(v(&mut nav, 3, false) > 0.0);
        assert!(v(&mut nav, 4, true) > 0.0);
        assert_eq!(v(&mut nav, 5, true), 0.0);
        assert_eq!(v(&mut nav, 6, true), 0.0);
        // reset forgets the history
        nav.reset();
        assert!(nav.near().is_empty());
        assert!(v(&mut nav, 7, true) > 0.0);
    }

    #[test]
    fn near_k_must_be_in_1_to_near_n() {
        for (k, n) in [(0, 1), (2, 1), (4, 3), (0, 0)] {
            let p = NavigatorParams {
                near_k: k,
                near_n: n,
                ..Default::default()
            };
            let err = Navigator::new(p).err().unwrap().0;
            assert!(err.starts_with("NavigatorParams.near_k"), "{err}");
        }
    }

    fn teleop(odometry: bool) -> Navigator {
        Navigator::new(NavigatorParams {
            behaviour: Behaviour::Teleop,
            odometry,
            resume_n: 1,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn teleop_request_goes_through_the_collision_monitor() {
        for odometry in [true, false] {
            let mut nav = teleop(odometry);
            let near = Pose2::new(4.0 - 0.13 - 0.05, 1.5, 0.0); // 5 cm from the wall, facing it
            let far = Pose2::new(1.0, 1.5, 0.0);
            nav.set_request(0.15, 0.2, 0.0);
            let out = step(&mut nav, 0.0, near, near);
            assert_eq!(
                (out.v, out.v_req, out.w_req, out.state),
                (0.0, 0.15, 0.2, State::Teleop)
            );
            assert_eq!(out.w, 0.2); // a turn is not held (no forward motion)
            let out = step(&mut nav, 0.1, far, far);
            assert!(out.v > 0.1 && out.w == 0.2 && !out.finished);
            // reverse: capped at reverse_max (0: the camera only looks forward)
            nav.set_request(-0.1, 0.0, 0.1);
            assert_eq!(step(&mut nav, 0.2, far, far).v, 0.0);
            assert_eq!(nav.grid().keyframes().is_empty(), !odometry); // the map with odometry, as Forward
        }
    }

    #[test]
    fn teleop_operator_dead_man() {
        let mut nav = teleop(true);
        let far = Pose2::new(1.0, 1.5, 0.0);
        // no request yet: still
        let out = step(&mut nav, 0.0, far, far);
        assert_eq!((out.v, out.w, out.v_req, out.w_req), (0.0, 0.0, 0.0, 0.0));
        nav.set_request(0.15, -0.1, 1.0);
        let out = step(&mut nav, 0.9, far, far); // set after this frame's capture: current
        assert_eq!((out.v_req, out.w_req), (0.15, -0.1));
        assert_eq!(step(&mut nav, 1.5, far, far).v_req, 0.15); // 0.5 s old: still held
        let out = step(&mut nav, 1.51, far, far); // older than teleop_timeout_s: (0, 0)
        assert_eq!((out.v, out.w, out.v_req, out.w_req), (0.0, 0.0, 0.0, 0.0));
        // a new request goes again; a non-finite one is (0, 0)
        nav.set_request(0.1, 0.0, 2.0);
        assert_eq!(step(&mut nav, 2.0, far, far).v_req, 0.1);
        nav.set_request(f64::NAN, 0.3, 2.1);
        let out = step(&mut nav, 2.1, far, far);
        assert_eq!((out.v_req, out.w_req), (0.0, 0.0));
        // reset forgets it
        nav.set_request(0.1, 0.0, 3.0);
        nav.reset();
        assert_eq!(step(&mut nav, 3.0, far, far).v_req, 0.0);
        // other behaviours ignore it
        let mut fwd = forward(true, 1, 0.15);
        fwd.set_request(0.3, 0.3, 0.0);
        assert_eq!(step(&mut fwd, 0.0, far, far).v_req, 0.15);
    }

    #[test]
    fn teleop_timeout_must_be_positive() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let p = NavigatorParams {
                teleop_timeout_s: bad,
                ..Default::default()
            };
            let err = Navigator::new(p).err().unwrap().0;
            assert!(err.starts_with("NavigatorParams.teleop_timeout_s"), "{err}");
        }
    }

    #[test]
    fn observe_goes_into_the_map_before_any_step() {
        let mut nav = Navigator::new(NavigatorParams::default()).unwrap();
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
