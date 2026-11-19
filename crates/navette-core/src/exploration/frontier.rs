//! Frontier exploration planner (Yamauchi 1997) on the occupancy grid.
//!
//! Every plan:
//! 1. FRONTIERS  free cells with an unknown 4-neighbour, grouped 8-connected; groups shorter than
//!    `min_frontier_m` are dropped (no gap the robot could use or see through). Frontier cells within `blind`
//!    of the robot are ignored: the camera cannot see them from here (closest visible floor ~0.47 m). Unknown
//!    holes smaller than `min_unknown_m2` inside the mapped floor (weak evidence) make no frontier.
//! 2. COSTS      one Dijkstra from the robot over traversable cells gives the path cost to every frontier at
//!    once (A* would need one search per candidate). Traversable: known free and at least `clear` from any
//!    obstacle cell, plus the disk of radius `blind` around the robot (the floor right around it is never in
//!    view) minus the sectors where the current scan has an obstacle filling the image bottom. Cost per metre
//!    1 + soft_k near obstacles (fading over soft_m): paths keep off walls.
//! 3. TARGET     the cheapest reachable frontier cell: path cost + turn_k x the turn toward it (nearest
//!    first, preferring what is ahead). The robot commits to its target while it is still a frontier
//!    (keep_ratio 0; else a new one must cost < keep_ratio of it): each new view shifts the frontier a little,
//!    and re-choosing the nearest every second made the robot swing back and forth. A target still a frontier
//!    after target_timeout_s, or one the robot gets blocked on, is banned for ban_s (ban_r around it); if only
//!    banned frontiers are left, the oldest ban is lifted.
//! 4. FOLLOW     the goal heading for the arc controller is the path point `lookahead` ahead.
//!
//! Frames: poses, path, target and frontier points are in the fixed frame. The planner works on the window
//! of the grid holding the known cells and the robot, row-major (h, w): index i along fixed x, j along y.
//!
//! Numerics follow the Python prototype (numpy, scipy.ndimage, scipy.sparse.csgraph), so that its plans are
//! reproduced cell for cell: same distance transform, same edge weights and sums, and the same order of
//! equal-cost choices (see [`dijkstra`] and the selection in [`FrontierPlanner::plan`]).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::SQRT_2;

use crate::geometry::{wrap_angle, Point2, Pose2};
use crate::occupancy_grid::OccupancyGrid;
use crate::params::{self, ParamError};

/// Predecessor of a cell [`dijkstra`] did not reach, and of the start.
pub const NO_PRED: u32 = u32::MAX;

/// A frontier cell within this distance (m) of the current target is the same frontier: the target is kept
/// while one remains.
const SAME_TARGET_M: f64 = 0.3;

/// Planner parameters. The defaults are the ones tuned on Reachy Mini.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierParams {
    /// Distance (m) paths keep from obstacle cells: robot radius + margin.
    pub clear: f64,
    /// Width (m) beyond `clear` where cells cost more.
    pub soft_m: f64,
    /// Extra cost per metre right at `clear` (fades to 0 over `soft_m`).
    pub soft_k: f64,
    /// Radius (m) of the disk around the robot the camera cannot see.
    pub blind: f64,
    /// Frontier groups shorter than this (m) are dropped.
    pub min_frontier_m: f64,
    /// Unknown holes smaller than this (m^2) are no frontier.
    pub min_unknown_m2: f64,
    /// Metres of path cost per radian the robot must turn toward the frontier.
    pub turn_k: f64,
    /// Time between plans (s).
    pub replan_s: f64,
    /// The goal heading points at the path point this far ahead (m).
    pub lookahead: f64,
    /// Switch target only if the new one costs < keep_ratio x the current: 0 = commit until it is seen (no
    /// longer a frontier), unreachable or timed out.
    pub keep_ratio: f64,
    /// A target not reached after this long (s) is banned.
    pub target_timeout_s: f64,
    /// Duration of a ban (s).
    pub ban_s: f64,
    /// Radius of a ban (m); also how far the target may move and still be the same target.
    pub ban_r: f64,
}

impl Default for FrontierParams {
    fn default() -> Self {
        Self {
            clear: 0.18,
            soft_m: 0.3,
            soft_k: 3.0,
            blind: 0.5,
            min_frontier_m: 0.25,
            min_unknown_m2: 0.1,
            turn_k: 0.5,
            replan_s: 1.0,
            lookahead: 0.6,
            keep_ratio: 0.0,
            target_timeout_s: 30.0,
            ban_s: 60.0,
            ban_r: 0.5,
        }
    }
}

impl FrontierParams {
    /// Every value finite (a NaN compared with a timer or a distance silently disables the check);
    /// soft_m and target_timeout_s > 0; the others >= 0.
    pub fn validate(&self) -> Result<(), ParamError> {
        params::positive("FrontierParams.soft_m", self.soft_m)?;
        params::positive("FrontierParams.target_timeout_s", self.target_timeout_s)?;
        for (name, value) in [
            ("clear", self.clear),
            ("soft_k", self.soft_k),
            ("blind", self.blind),
            ("min_frontier_m", self.min_frontier_m),
            ("min_unknown_m2", self.min_unknown_m2),
            ("turn_k", self.turn_k),
            ("replan_s", self.replan_s),
            ("lookahead", self.lookahead),
            ("keep_ratio", self.keep_ratio),
            ("ban_s", self.ban_s),
            ("ban_r", self.ban_r),
        ] {
            params::non_negative(&format!("FrontierParams.{name}"), value)?;
        }
        Ok(())
    }
}

/// The 8 neighbours (di, dj, diagonal) of a cell.
const NEIGHBOURS: [(isize, isize, bool); 8] = [
    (-1, -1, true),
    (-1, 0, false),
    (-1, 1, true),
    (0, -1, false),
    (0, 1, false),
    (1, -1, true),
    (1, 0, false),
    (1, 1, true),
];

/// Row-major index of the neighbour (di, dj) of cell `c` in an (h, w) grid; None outside.
#[inline]
fn neighbour(c: usize, h: usize, w: usize, di: isize, dj: isize) -> Option<usize> {
    let i = (c / w).checked_add_signed(di).filter(|&i| i < h)?;
    let j = (c % w).checked_add_signed(dj).filter(|&j| j < w)?;
    Some(i * w + j)
}

/// Clear the connected groups of `mask` (h, w) with fewer than `min_size` cells. Groups are 8-connected if
/// `diagonal`, else 4-connected.
fn remove_small_groups(mask: &mut [bool], h: usize, w: usize, diagonal: bool, min_size: usize) {
    let mut visited = vec![false; mask.len()];
    let mut group = Vec::new(); // cells of the current group, in the order the flood fill found them
    for seed in 0..mask.len() {
        if !mask[seed] || visited[seed] {
            continue;
        }
        group.clear();
        group.push(seed);
        visited[seed] = true;
        let mut next = 0; // first cell of the group whose neighbours are still to visit
        while next < group.len() {
            let c = group[next];
            next += 1;
            for &(di, dj, diag) in &NEIGHBOURS {
                if diag && !diagonal {
                    continue;
                }
                if let Some(nb) = neighbour(c, h, w, di, dj) {
                    if mask[nb] && !visited[nb] {
                        visited[nb] = true;
                        group.push(nb);
                    }
                }
            }
        }
        if group.len() < min_size {
            for &c in &group {
                mask[c] = false;
            }
        }
    }
}

/// Frontier cells of an (h, w) grid: free cells with an unknown (neither obstacle nor free) 4-neighbour,
/// without the 8-connected groups smaller than `min_cells`. Unknown blobs (4-connected) smaller than
/// `min_unknown_cells`, holes left inside the mapped floor by weak evidence, make no frontier.
pub fn frontier_cells(
    obst: &[bool],
    free: &[bool],
    h: usize,
    w: usize,
    min_cells: usize,
    min_unknown_cells: usize,
) -> Vec<bool> {
    let mut unknown: Vec<bool> = obst.iter().zip(free).map(|(&o, &f)| !o && !f).collect();
    if min_unknown_cells > 0 {
        remove_small_groups(&mut unknown, h, w, false, min_unknown_cells);
    }
    let touches_unknown = |c: usize| {
        NEIGHBOURS
            .iter()
            .filter(|n| !n.2)
            .any(|&(di, dj, _)| neighbour(c, h, w, di, dj).is_some_and(|nb| unknown[nb]))
    };
    let mut frontier: Vec<bool> = (0..h * w).map(|c| free[c] && touches_unknown(c)).collect();
    remove_small_groups(&mut frontier, h, w, true, min_cells);
    frontier
}

/// Exact Euclidean distance (in cells) from every cell of an (h, w) grid to the nearest `obst` cell; 0 on
/// obstacle cells. The values of scipy.ndimage.distance_transform_edt(~obst): the square root of an integer.
/// Without any obstacle cell, the distance to cell (-1, 0), as scipy gives: an arbitrary but fixed choice
/// (the planner then sees the first rows as close to an obstacle; it only lasts until the map holds one).
///
/// Two passes (Meijster et al. 2000), integers only: the distance to the nearest obstacle along i in each
/// column, then for each row the lower envelope of the parabolas (j - u)^2 + g(u)^2 of its columns u.
pub fn distance_transform(obst: &[bool], h: usize, w: usize) -> Vec<f64> {
    const NONE: i64 = i64::MAX; // no obstacle in the column
    if !obst.iter().any(|&o| o) {
        return (0..h * w)
            .map(|c| (((c / w + 1).pow(2) + (c % w).pow(2)) as f64).sqrt())
            .collect();
    }
    // g[i * w + j]: distance along i from (i, j) to the nearest obstacle of column j
    let mut g = vec![NONE; h * w];
    for j in 0..w {
        let mut last = None; // nearest obstacle row at or before i
        for i in 0..h {
            if obst[i * w + j] {
                last = Some(i);
            }
            if let Some(l) = last {
                g[i * w + j] = (i - l) as i64;
            }
        }
        let mut last = None; // nearest obstacle row at or after i
        for i in (0..h).rev() {
            if obst[i * w + j] {
                last = Some(i);
            }
            if let Some(l) = last {
                g[i * w + j] = g[i * w + j].min((l - i) as i64);
            }
        }
    }
    let mut d = vec![0.0; h * w];
    // s[q]: column of the q-th parabola of the envelope, the lowest one from column t[q] on
    let (mut s, mut t) = (vec![0i64; w], vec![0i64; w]);
    for i in 0..h {
        let g = &g[i * w..(i + 1) * w];
        // squared distance from column x through the parabola of column u
        let f = |x: i64, u: i64| (x - u) * (x - u) + g[u as usize] * g[u as usize];
        let mut q: isize = -1;
        for u in (0..w as i64).filter(|&u| g[u as usize] != NONE) {
            while q >= 0 && f(t[q as usize], s[q as usize]) > f(t[q as usize], u) {
                q -= 1;
            }
            if q < 0 {
                q = 0;
                s[0] = u;
                t[0] = 0;
            } else {
                // last column where the parabola of s[q] is not above u's
                let v = s[q as usize];
                let (gu, gv) = (g[u as usize], g[v as usize]);
                let sep = (u * u - v * v + gu * gu - gv * gv).div_euclid(2 * (u - v));
                if sep + 1 < w as i64 {
                    q += 1;
                    s[q as usize] = u;
                    t[q as usize] = sep + 1;
                }
            }
        }
        // some column holds an obstacle, so the envelope is not empty and ends at q = 0, t[0] = 0
        for x in (0..w as i64).rev() {
            d[i * w + x as usize] = (f(x, s[q as usize]) as f64).sqrt();
            if x == t[q as usize] {
                q -= 1;
            }
        }
    }
    d
}

/// A cell waiting in [`dijkstra`]'s queue. Ordered so that the heap gives the smallest cost first and, among
/// equal costs, the largest cell index: the order of scipy's queue of (-cost, index) pairs.
struct Queued {
    cost: f64,
    cell: u32,
}

impl Ord for Queued {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.total_cmp(&self.cost).then(self.cell.cmp(&other.cell))
    }
}

impl PartialOrd for Queued {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Queued {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Queued {}

/// Cheapest path cost from `start` to every cell of an (h, w) grid, moving between 8-neighbours that are
/// both traversable, and each cell's predecessor on that path ([`NO_PRED`] for the start and for cells not
/// reached, whose cost is inf). An edge costs its length (res, or res sqrt(2)) x the mean `cost` of its two
/// cells.
///
/// Dijkstra with a binary heap. The cost of a cell is the minimum over its neighbours v of
/// cost(v) + edge(v, cell), a value that does not depend on the order of the queue; the predecessor is the
/// first neighbour to give that minimum, i.e. the one leaving the queue first. Costs tie exactly on a grid
/// (two neighbours at the same cost offering the same total), so the queue order is scipy's: equal costs
/// leave by decreasing cell index, and a cell is only updated by a strictly smaller cost.
pub fn dijkstra(
    trav: &[bool],
    cost: &[f64],
    h: usize,
    w: usize,
    res: f64,
    start: usize,
) -> (Vec<f64>, Vec<u32>) {
    dijkstra_to(trav, cost, h, w, res, start, &[])
}

/// [`dijkstra`], stopped as soon as every traversable cell of `targets` (distinct cells) is settled; with
/// no traversable target, the whole search.
///
/// What the early stop leaves exact: a cell is settled when it leaves the queue with its current cost, and
/// neither its cost nor its predecessor changes afterwards (later entries cost at least as much, and a cell
/// is only updated by a strictly smaller cost). Edges cost more than 0, so the cells on the path to a
/// settled cell were settled before it. Hence the cost of every target and the predecessors along the path
/// to it are those of the full search: a target not yet settled when the search stops is not traversable
/// and keeps inf, as in the full search, which never enters such a cell. The cells that are not targets may
/// hold a provisional cost or inf: only the targets and their paths may be read.
fn dijkstra_to(
    trav: &[bool],
    cost: &[f64],
    h: usize,
    w: usize,
    res: f64,
    start: usize,
    targets: &[usize],
) -> (Vec<f64>, Vec<u32>) {
    let mut is_target = vec![false; if targets.is_empty() { 0 } else { h * w }];
    let mut remaining = 0; // traversable targets not settled yet
    for &c in targets {
        is_target[c] = trav[c];
        remaining += trav[c] as usize;
    }
    let mut d = vec![f64::INFINITY; h * w];
    let mut pred = vec![NO_PRED; h * w];
    let mut queue = BinaryHeap::new();
    d[start] = 0.0;
    queue.push(Queued {
        cost: 0.0,
        cell: start as u32,
    });
    let length = [res, res * SQRT_2];
    while let Some(Queued { cost: dv, cell }) = queue.pop() {
        let v = cell as usize;
        // d[v] < dv: a cheaper way to v was found after this entry was queued, and v expanded with it
        if d[v] < dv || !trav[v] {
            continue;
        }
        // v is settled (it gets here once: its queue entries have distinct costs)
        if remaining > 0 && is_target[v] {
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }
        for &(di, dj, diag) in &NEIGHBOURS {
            let Some(nb) = neighbour(v, h, w, di, dj).filter(|&nb| trav[nb]) else {
                continue;
            };
            let next = dv + length[diag as usize] * (cost[v] + cost[nb]) / 2.0;
            if d[nb] > next {
                d[nb] = next;
                pred[nb] = cell;
                queue.push(Queued {
                    cost: next,
                    cell: nb as u32,
                });
            }
        }
    }
    (d, pred)
}

/// Index of the smallest value, the first one among equals (numpy's argmin, without NaN).
fn argmin(values: impl Iterator<Item = f64>) -> usize {
    let (mut best, mut at) = (f64::INFINITY, 0);
    for (k, v) in values.enumerate() {
        if v < best {
            (best, at) = (v, k);
        }
    }
    at
}

#[inline]
fn distance(a: Point2, b: Point2) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Frontier planner: [`Self::plan`], every `replan_s`, chooses the target and the path to it;
/// [`Self::heading`] gives the direction to follow at each step.
#[derive(Clone, Debug)]
pub struct FrontierPlanner {
    pub params: FrontierParams,
    /// Fixed-frame points from the robot's cell to the target.
    path: Option<Vec<Point2>>,
    /// Fixed-frame centre of the target frontier cell.
    target: Option<Point2>,
    /// Time the current target was first chosen (s).
    target_since: f64,
    /// The last plan found no reachable frontier.
    done: bool,
    /// Banned targets: (time of the ban, point), oldest first.
    banned: Vec<(f64, Point2)>,
    /// Frontier cell centres of the last plan (for display).
    frontier: Vec<Point2>,
}

impl FrontierPlanner {
    /// An error if a parameter is invalid ([`FrontierParams::validate`]).
    pub fn new(params: FrontierParams) -> Result<Self, ParamError> {
        params.validate()?;
        Ok(Self {
            params,
            path: None,
            target: None,
            target_since: 0.0,
            done: false,
            banned: Vec::new(),
            frontier: Vec::new(),
        })
    }

    pub fn path(&self) -> Option<&[Point2]> {
        self.path.as_deref()
    }

    pub fn target(&self) -> Option<Point2> {
        self.target
    }

    /// The last plan found no reachable frontier: nothing left to explore from here.
    pub fn done(&self) -> bool {
        self.done
    }

    pub fn banned(&self) -> &[(f64, Point2)] {
        &self.banned
    }

    pub fn frontier(&self) -> &[Point2] {
        &self.frontier
    }

    /// Give the current target up for `ban_s`: it was not reached, or the robot got blocked on the way.
    pub fn ban(&mut self, t: f64) {
        if let Some(target) = self.target.take() {
            self.banned.push((t, target));
        }
        self.path = None;
    }

    /// Choose the target and the path from `pose` on `grid` at time `t`.
    /// `near`: robot-frame bearings (rad) of the CURRENT scan's obstacles filling the image bottom
    /// (`Scan::near`), `near_half`: their sector half-width (rad). Used for this plan only, never stored in
    /// the grid (a non-persistent layer): the obstacle may be anywhere in the blind disk that way, which a
    /// stored guess would never clear.
    /// Nothing known yet, or the robot outside the grid: no path, no target.
    pub fn plan(&mut self, t: f64, pose: Pose2, grid: &OccupancyGrid, near: &[f64], near_half: f64) {
        let p = self.params;
        let Some(win) = Window::new(grid, pose) else {
            (self.path, self.target) = (None, None);
            return;
        };
        let (h, w, res) = (win.h, win.w, grid.params().res);
        let (r0i, r0j) = (win.robot[0] as f64, win.robot[1] as f64);
        // Only the cells within `reach` cells of the robot's (along i and along j) can be in the blind disk
        // or within `clear` of the robot: the distance to the robot is not computed elsewhere.
        let reach = (p.blind.max(p.clear) / res).ceil() + 1.0;
        let d_robot = |c: usize| {
            let (di, dj) = ((c / w) as f64 - r0i, (c % w) as f64 - r0j);
            (di.abs() <= reach && dj.abs() <= reach).then(|| (di.hypot(dj) * res, di, dj))
        };

        let dist: Vec<f64> = distance_transform(&win.obst, h, w)
            .into_iter()
            .map(|d| d * res)
            .collect();
        let mut in_disk = vec![false; h * w]; // within `blind` of the robot
        let mut trav = vec![false; h * w];
        for c in 0..h * w {
            let (mut blind, mut at_robot) = (false, false);
            if let Some((d, di, dj)) = d_robot(c) {
                in_disk[c] = d < p.blind;
                at_robot = d < p.clear;
                // The blind disk is assumed free because the camera never sees it, but where an obstacle
                // fills the image bottom, it IS in the blind disk: the collision monitor places it at the
                // clearance radius and holds the robot, so a path leaving that way stalls. Those sectors are
                // dropped from the blind disk, widened on each side by the angle the clearance subtends at
                // each distance.
                blind = in_disk[c] && {
                    let rel = dj.atan2(di) - pose.theta;
                    let half = near_half + p.clear.atan2(d.max(res));
                    !near.iter().any(|&b| wrap_angle(rel - b).abs() < half)
                };
            }
            trav[c] = ((win.free[c] || blind) && dist[c] >= p.clear) || (at_robot && !win.obst[c]);
        }
        let cost: Vec<f64> = dist
            .iter()
            .map(|&d| 1.0 + p.soft_k * (1.0 - (d - p.clear) / p.soft_m).clamp(0.0, 1.0))
            .collect();
        let min_cells = ((p.min_frontier_m / res).round_ties_even() as usize).max(1);
        let min_unknown = (p.min_unknown_m2 / (res * res)).round_ties_even() as usize;
        let cells: Vec<usize> = frontier_cells(&win.obst, &win.free, h, w, min_cells, min_unknown)
            .into_iter()
            .enumerate()
            .filter_map(|(c, f)| (f && !in_disk[c]).then_some(c))
            .collect();
        // Only the costs of the frontier cells and the path to the chosen one are read below: the search
        // stops once they are all known (see dijkstra_to), which saves the far side of a large known map.
        let start = win.robot[0] * w + win.robot[1];
        let (d, pred) = dijkstra_to(&trav, &cost, h, w, res, start, &cells);
        let centres: Vec<Point2> = cells.iter().map(|&c| win.centre(grid, c)).collect();
        // cost of each frontier cell: the path to it + the turn toward it
        let base: Vec<f64> = cells
            .iter()
            .zip(&centres)
            .map(|(&c, q)| {
                let bearing = (q[1] - pose.y).atan2(q[0] - pose.x) - pose.theta;
                d[c] + p.turn_k * wrap_angle(bearing).abs()
            })
            .collect();
        let reachable = base.iter().any(|c| c.is_finite());
        self.banned.retain(|&(tb, _)| t - tb < p.ban_s);
        self.frontier.clone_from(&centres);

        // Selection. It starts again after lifting or adding a ban.
        let k = loop {
            let mut fc = base.clone();
            for (_, q) in &self.banned {
                for (c, centre) in fc.iter_mut().zip(&centres) {
                    if distance(*centre, *q) < p.ban_r {
                        *c = f64::INFINITY;
                    }
                }
            }
            if !fc.iter().any(|c| c.is_finite()) {
                if reachable {
                    // only banned ones left: retry the oldest rather than stop
                    self.banned.remove(0);
                    continue;
                }
                (self.done, self.path, self.target) = (true, None, None);
                return;
            }
            self.done = false;
            let mut k = argmin(fc.iter().copied());
            if let Some(target) = self.target {
                // hysteresis: keep the current target if it is still a frontier
                let same = |c: usize| distance(centres[c], target) < SAME_TARGET_M;
                if (0..fc.len()).any(|c| same(c) && fc[c].is_finite()) {
                    let kc = argmin((0..fc.len()).map(|c| if same(c) { fc[c] } else { f64::INFINITY }));
                    if fc[k] >= p.keep_ratio * fc[kc] {
                        k = kc;
                    }
                }
            }
            match self.target {
                Some(target) if distance(centres[k], target) <= p.ban_r => {
                    if t - self.target_since > p.target_timeout_s {
                        // the same target for too long: give it up and choose again
                        self.ban(t);
                        continue;
                    }
                }
                _ => self.target_since = t,
            }
            break k;
        };
        self.target = Some(centres[k]);
        // path: back-track the predecessors from the target cell
        let mut path = Vec::new();
        let mut c = cells[k] as u32;
        while c != NO_PRED {
            path.push(win.centre(grid, c as usize));
            c = pred[c as usize];
        }
        path.reverse();
        self.path = Some(path);
    }

    /// Fixed-frame heading from `pose` to the path point `lookahead` ahead: the first point at least that far
    /// from the robot, from the path point nearest to it on (the last point if the path ends before).
    /// None without a path, or standing on that point.
    pub fn heading(&self, pose: Pose2) -> Option<f64> {
        let path = self.path.as_deref().filter(|p| !p.is_empty())?;
        let xy = [pose.x, pose.y];
        let nearest = argmin(path.iter().map(|&q| distance(q, xy)));
        let q = path[nearest..]
            .iter()
            .find(|&&q| distance(q, xy) >= self.params.lookahead)
            .unwrap_or(&path[path.len() - 1]);
        (distance(*q, xy) >= 1e-6).then(|| (q[1] - xy[1]).atan2(q[0] - xy[0]))
    }
}

/// The part of the grid a plan works on: the box of the known cells and the robot's cell, with a margin of
/// 3 cells (so that the known floor is surrounded by unknown cells where the grid allows).
struct Window {
    /// Cell of the grid at the window's [0, 0].
    lo: [usize; 2],
    h: usize,
    w: usize,
    /// The robot's cell in the window.
    robot: [usize; 2],
    /// Obstacle and free cells, row-major (h, w).
    obst: Vec<bool>,
    free: Vec<bool>,
}

impl Window {
    /// None if the grid holds no known cell yet, or the robot is outside the grid.
    fn new(grid: &OccupancyGrid, pose: Pose2) -> Option<Self> {
        grid.origin()?;
        let (n, l, thr) = (grid.n(), grid.log_odds(), grid.params().thr as f32);
        let rij = grid.cell([pose.x, pose.y]);
        if rij.iter().any(|&c| c < 0 || c >= n as i64) {
            return None;
        }
        let rij = [rij[0] as usize, rij[1] as usize];
        // box of the known cells: [min, max] along i and j
        let (mut min, mut max) = ([usize::MAX; 2], [0; 2]);
        for (i, row) in l.chunks_exact(n).enumerate() {
            for (j, &l) in row.iter().enumerate() {
                if l > thr || l < -thr {
                    min = [min[0].min(i), min[1].min(j)];
                    max = [max[0].max(i), max[1].max(j)];
                }
            }
        }
        if min[0] == usize::MAX {
            return None;
        }
        let lo = [0, 1].map(|a| min[a].min(rij[a]).saturating_sub(3));
        let hi = [0, 1].map(|a| (max[a].max(rij[a]) + 4).min(n));
        let (h, w) = (hi[0] - lo[0], hi[1] - lo[1]);
        let crop = |pick: fn(f32, f32) -> bool| -> Vec<bool> {
            (0..h * w)
                .map(|c| pick(l[(lo[0] + c / w) * n + lo[1] + c % w], thr))
                .collect()
        };
        Some(Self {
            lo,
            h,
            w,
            robot: [rij[0] - lo[0], rij[1] - lo[1]],
            obst: crop(|l, thr| l > thr),
            free: crop(|l, thr| l < -thr),
        })
    }

    /// Fixed-frame centre of the window cell of row-major index `c`.
    fn centre(&self, grid: &OccupancyGrid, c: usize) -> Point2 {
        grid.centre([(self.lo[0] + c / self.w) as i64, (self.lo[1] + c % self.w) as i64])
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_room as room;
    use super::*;
    use crate::occupancy_grid::OccupancyGridParams;

    fn grid_of(rows: &[&str], what: char) -> Vec<bool> {
        rows.iter()
            .flat_map(|r| r.chars().map(move |c| c == what))
            .collect()
    }

    fn count(mask: &[bool]) -> usize {
        mask.iter().filter(|&&m| m).count()
    }

    #[test]
    fn early_stop_gives_the_targets_and_their_paths_exactly() {
        // a pseudo-random field: obstacles, varied costs (many equal totals on a grid)
        let (h, w) = (60, 80);
        let mut x = 12345u32;
        let mut rnd = || {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x >> 8) as f64 / (1u32 << 24) as f64
        };
        let trav: Vec<bool> = (0..h * w).map(|_| rnd() > 0.25).collect();
        let cost: Vec<f64> = (0..h * w).map(|_| 1.0 + (rnd() * 3.0).floor()).collect();
        let start = (0..h * w).find(|&c| trav[c] && c > 30 * w + 40).unwrap();
        let (d, pred) = dijkstra(&trav, &cost, h, w, 0.05, start);
        // targets near the start, one of them not traversable, one not reachable (if any), the start itself
        let mut near: Vec<usize> = (0..h * w).filter(|&c| d[c] < 0.4 && c % 7 == 0).collect();
        near.push(start);
        near.extend((0..h * w).find(|&c| !trav[c]));
        near.sort_unstable();
        near.dedup();
        let (d2, pred2) = dijkstra_to(&trav, &cost, h, w, 0.05, start, &near);
        for &c in &near {
            assert_eq!(d2[c].to_bits(), d[c].to_bits());
            let mut q = c as u32;
            while q != NO_PRED && d[c].is_finite() {
                assert_eq!(pred2[q as usize], pred[q as usize]);
                q = pred[q as usize];
            }
        }
        // and it did stop early: far cells were not reached
        let reached = |d: &[f64]| d.iter().filter(|x| x.is_finite()).count();
        assert!(reached(&d2) < reached(&d) / 2, "{} {}", reached(&d2), reached(&d));
        // an unreachable traversable target: the whole search, identical
        let island = (0..h * w).find(|&c| trav[c] && d[c].is_infinite());
        if let Some(island) = island {
            assert_eq!(dijkstra_to(&trav, &cost, h, w, 0.05, start, &[island]).1, pred);
        }
        assert_eq!(dijkstra_to(&trav, &cost, h, w, 0.05, start, &[]), (d, pred));
    }

    #[test]
    fn bad_parameters_are_errors() {
        let ok = FrontierParams::default();
        assert_eq!(ok.validate(), Ok(()));
        let nan = f64::NAN;
        for p in [
            FrontierParams { replan_s: nan, ..ok },
            FrontierParams { replan_s: -1.0, ..ok },
            FrontierParams { lookahead: nan, ..ok },
            FrontierParams { soft_m: 0.0, ..ok },
            FrontierParams { clear: -0.1, ..ok },
            FrontierParams {
                blind: f64::INFINITY,
                ..ok
            },
            FrontierParams {
                target_timeout_s: 0.0,
                ..ok
            },
            FrontierParams { ban_r: nan, ..ok },
        ] {
            let err = FrontierPlanner::new(p).expect_err("invalid");
            assert!(err.0.starts_with("FrontierParams."), "{err}");
        }
    }

    #[test]
    fn frontier_needs_an_unknown_4_neighbour_and_a_minimum_size() {
        // '.' free, '#' obstacle, ' ' unknown
        let rows = ["##### ", "#...# ", "#.... ", "#...# ", "##### "];
        let (obst, free) = (grid_of(&rows, '#'), grid_of(&rows, '.'));
        let fr = frontier_cells(&obst, &free, 5, 6, 1, 0);
        // only the free cell in the gap touches unknown by an edge
        assert!(count(&fr) == 1 && fr[2 * 6 + 4]);
        assert_eq!(count(&frontier_cells(&obst, &free, 5, 6, 2, 0)), 0);
    }

    #[test]
    fn small_unknown_holes_make_no_frontier() {
        let rows = ["#####", "#...#", "#. .#", "#...#", "#####"];
        let (obst, free) = (grid_of(&rows, '#'), grid_of(&rows, '.'));
        // 4 free cells touch the hole by an edge; they form one 8-connected group
        assert_eq!(count(&frontier_cells(&obst, &free, 5, 5, 4, 0)), 4);
        assert_eq!(count(&frontier_cells(&obst, &free, 5, 5, 5, 0)), 0);
        assert_eq!(count(&frontier_cells(&obst, &free, 5, 5, 4, 2)), 0);
    }

    /// Distance to the nearest obstacle, by trying them all.
    fn brute_distance(obst: &[bool], h: usize, w: usize) -> Vec<f64> {
        (0..h * w)
            .map(|c| {
                let d2 = (0..h * w)
                    .filter(|&o| obst[o])
                    .map(|o| (c / w).abs_diff(o / w).pow(2) + (c % w).abs_diff(o % w).pow(2))
                    .min()
                    .unwrap();
                (d2 as f64).sqrt()
            })
            .collect()
    }

    #[test]
    fn distance_transform_is_exact() {
        // pseudo-random obstacle patterns (linear congruential generator), several densities and shapes
        let mut state = 12345u64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        let cases = [
            (1, 1, 1.0),
            (1, 9, 0.2),
            (9, 1, 0.2),
            (17, 23, 0.02),
            (30, 12, 0.3),
            (25, 25, 0.004),
        ];
        for (h, w, density) in cases {
            let mut obst: Vec<bool> = (0..h * w).map(|_| rand() < density).collect();
            obst[(h / 2) * w + w / 2] = true; // at least one
            assert_eq!(
                distance_transform(&obst, h, w),
                brute_distance(&obst, h, w),
                "{h} x {w}"
            );
        }
        // no obstacle: the distance to cell (-1, 0)
        let (r2, r5, r8) = (2f64.sqrt(), 5f64.sqrt(), 8f64.sqrt());
        assert_eq!(distance_transform(&[false; 6], 2, 3), [1.0, r2, r5, 2.0, r5, r8]);
    }

    #[test]
    fn dijkstra_goes_around_walls_and_equal_costs_leave_by_decreasing_index() {
        // '#' not traversable; cells 0..12 row by row, start = 4 (left of the wall)
        let rows = ["....", ".##.", "...."];
        let trav = grid_of(&rows, '.');
        let (d, pred) = dijkstra(&trav, &[1.0; 12], 3, 4, 1.0, 4);
        assert_eq!((d[4], pred[4]), (0.0, NO_PRED));
        assert_eq!((d[5], pred[5]), (f64::INFINITY, NO_PRED));
        assert_eq!((d[0], d[1], d[2]), (1.0, SQRT_2, SQRT_2 + 1.0));
        // cell 7, behind the wall, costs the same through the top (2) and the bottom (10): cells 2 and 10
        // tie, the larger index leaves the queue first and becomes the predecessor
        assert_eq!(d[2], d[10]);
        assert_eq!((d[7], pred[7]), (d[2] + SQRT_2, 10));
        // an edge costs its length x the mean cost of its two cells
        let mut cost = [1.0; 12];
        cost[0] = 3.0;
        let (d, _) = dijkstra(&trav, &cost, 3, 4, 0.5, 4);
        assert_eq!(d[0], 0.5 * 2.0);
    }

    #[test]
    fn plans_through_the_door() {
        let g = room(0.8);
        let mut pl = FrontierPlanner::new(FrontierParams::default()).unwrap();
        let pose = Pose2::new(-0.5, 0.5, 0.0);
        pl.plan(0.0, pose, &g, &[], 0.0);
        assert!(!pl.done());
        let target = pl.target().unwrap();
        assert!(target[0] > 0.9 && target[1].abs() < 0.4);
        let path = pl.path().unwrap();
        assert_eq!(path[path.len() - 1], target);
        assert_eq!(g.cell(path[0]), g.cell([pose.x, pose.y]));
        // consecutive path points are neighbouring cells
        assert!(path.windows(2).all(|s| distance(s[0], s[1]) < 0.05 * 1.5));
        let h = pl.heading(pose).unwrap();
        assert!(-std::f64::consts::FRAC_PI_2 < h && h < 0.1); // toward the door: right and ahead
        assert!(pl.frontier().iter().all(|q| q[0] > 0.9 && q[1].abs() < 0.4));
    }

    #[test]
    fn closed_or_too_narrow_means_done() {
        for door_w in [0.0, 0.25] {
            let mut pl = FrontierPlanner::new(FrontierParams::default()).unwrap();
            pl.plan(0.0, Pose2::default(), &room(door_w), &[], 0.0);
            assert!(pl.done() && pl.path().is_none() && pl.target().is_none());
            assert_eq!(pl.heading(Pose2::default()), None);
        }
    }

    #[test]
    fn nothing_known_or_robot_outside_gives_no_plan() {
        let mut pl = FrontierPlanner::new(FrontierParams::default()).unwrap();
        let empty = OccupancyGrid::new(OccupancyGridParams::default()).unwrap();
        pl.plan(0.0, Pose2::default(), &empty, &[], 0.0);
        assert!(!pl.done() && pl.path().is_none());
        pl.plan(0.0, Pose2::new(10.0, 0.0, 0.0), &room(0.8), &[], 0.0);
        assert!(!pl.done() && pl.path().is_none());
    }

    #[test]
    fn timeout_bans_the_target_and_the_only_frontier_is_retried() {
        let g = room(0.8);
        let mut pl = FrontierPlanner::new(FrontierParams {
            target_timeout_s: 5.0,
            ..Default::default()
        })
        .unwrap();
        pl.plan(0.0, Pose2::default(), &g, &[], 0.0);
        let first = pl.target().unwrap();
        pl.plan(6.0, Pose2::default(), &g, &[], 0.0);
        // not reached: banned; the door is the only frontier, so the ban is lifted at once
        assert!(!pl.done() && pl.banned().is_empty());
        assert!(distance(pl.target().unwrap(), first) < 0.5);
    }
}
