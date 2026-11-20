"""SLAM front end (docs/SLAM.md, section 3): odometry-primary, spin submaps, SE(2) pose graph.

Fed one frame at a time (time, odometry base pose, scan, camera height, camera heading), it cuts the frame
stream into submaps, adds one pose-graph node per submap, chains the nodes with odometry edges, and when a
spin submap closes, matches it against the earlier submaps around it; an accepted match is a robust loop edge,
and the graph is re-optimised. Outputs, at any time: the corrected pose of the latest frame, the map -> odom
correction (REP 105), the corrected poses of every frame fed (for OccupancyGrid.set_keyframe_poses + replay),
and diagnostics (submaps, edges, matches accepted / rejected with the reason).

Submaps (`Submap`):
  - "spin": a turn in place (frames within spin_max_trans_m of the turn's first position) whose heading span
    reached spin_deg: a wide local view (span + the camera's ~70 deg), the only kind dense enough to match
    reliably (docs/SLAM.md, family A). It closes when the robot leaves the spot, or at 360 deg; the frames
    before the turn close as a travel submap.
  - "travel": closed after travel_m of driving (or travel_max_s): a node of the odometry chain, also used
    as a matching reference (it holds the walls seen while driving), matched itself only if
    match_travel (off by default, see SlamParams).
  Each submap has an anchor = the odometry pose of its first frame; its frames are kept relative to the anchor
  (they move rigidly with it: a loop correction moves whole submaps), a local OccupancyGrid is built from its
  frames in the anchor frame, and its obstacle points (cells over points_logodds) are what matching uses.

Pose graph: node k = anchor of submap k in the map frame, node 0 fixed (map = odom at the start). Odometry
edge k-1 -> k: the odometry motion between the two anchors, information from the noise model in SlamParams
(see `odometry_sigmas`). Loop edge r -> j: the match of submap j's points against a reference built from the
earlier submaps near it (at their corrected poses); r is the reference submap whose anchor is nearest; robust
(DCS, dcs_phi ~10: at 1, true loops with a few sigma of drift are dropped).

Frames: odom and the returned poses are base poses (x, y, theta); a scan is in the camera's horizontal frame,
heading cam_heading in the base frame: its pose is (x, y, theta + cam_heading), as runtime/replay.py.
"""
import math
from dataclasses import dataclass

import numpy as np

from navette.slam.data import scan_dict, to_scan
from navette.slam.metrics import compose, inverse, relative, wrap


@dataclass
class SlamParams:
    """Front-end parameters. Lengths in m, angles in deg unless the name says rad, times in s. The odometry
    noise defaults are ~1.5x tests/sim_slam.py's drift model (+1.5 % distance, -1 % yaw scale, 3 deg/min bias),
    not yet measured on the robot (needs ground truth, docs/SLAM.md section 4); they set the matching window and
    the odometry edges' weight."""
    # --- submaps
    spin_deg: float = 180.0            # heading span of a turn in place that makes a spin submap. Not ~330:
                                       # the robot's turns in recorded explore runs span 60-260 deg, never
                                       # 330; 180 + the camera's ~70 deg is a >= 250 deg view
    spin_max_trans_m: float = 0.2      # most translation during a spin (a balancing base wobbles ~5 cm)
    travel_m: float = 1.0              # a travel submap closes after this much driving
    travel_max_s: float = 60.0         # ... or this long (bounds the frames held while standing still)
    # --- local grid of a submap (navette.OccupancyGrid, built in the anchor frame)
    grid_res: float = 0.05
    grid_size_m: float = 8.0           # 2 x (max range 2.5 + ~1 m travel) + margin
    grid_max_range_m: float | None = 2.0  # trusted scan range for submaps: the floor-scan error grows ~range^2
    grid_sigma: float | None = None    # tilt error bound passed to update() (None: unweighted, no IMU here)
    points_logodds: float = 0.6        # obstacle point = cell with log-odds above this (grid clamp is 1.5:
                                       # ~2 agreeing keyframes; the grid's own thr 0.2 is one weak hit)
    # --- odometry noise model, per edge between consecutive anchors (see odometry_sigmas)
    odom_xy_per_m: float = 0.03        # position sigma per m driven (scale error + slip)
    odom_xy_min: float = 0.01          # floor of the position sigma (wobble of a balancing base)
    odom_th_per_rad: float = 0.015     # heading sigma per rad turned (gyro / wheel yaw scale error)
    odom_th_per_m: float = 0.005       # heading sigma per m driven (rad/m: wheel-ground slip)
    odom_bias_deg_min: float = 4.0     # heading bias rate (deg/min: gyro bias), counted over the edge's time
    odom_th_min_deg: float = 0.5       # floor of the heading sigma
    # --- matching (navette.match_points on a navette.LikelihoodGrid of reference points)
    match_res: float = 0.05            # likelihood grid cell = translation search step
    match_sigma: float = 0.05          # expected point error (wall thickness + range noise)
    ref_radius_m: float = 3.0          # reference submaps: anchor within this (+ the xy window) of the guess
    loop_min_path_m: float = 2.0       # ... and at least this much driving back along the chain (_match)
    loop_min_s: float = 30.0           # ... or this long ago (in-place spins accumulate gyro bias, no path)
    ref_max_path_m: float = 4.0        # one reference = consecutive submaps over at most this much driving
    window_nsigma: float = 1.5         # window half-width = n x accumulated sigma (already a worst case), clamped:
    window_xy_min: float = 0.2
    window_xy_max: float = 1.0
    window_th_min_deg: float = 10.0
    window_th_max_deg: float = 30.0
    min_points: int = 60               # query submap points
    min_ref_points: int = 100          # reference points
    min_score: float = 0.5             # mean likelihood of the query points (0..1)
    max_cond_frac: float = 0.2         # sqrt(lambda_max xy) <= this x the xy half-window (unconstrained:
                                       # ~0.58, Olson covariance of a flat score over the window)
    max_th_cond_frac: float = 0.2      # sqrt(var theta) <= this x the heading half-window
    border_frac: float = 0.9           # reject a best within (1 - this) of the window border (truth outside?)
    match_travel: bool = False         # also match travel submaps (a ~70 deg view over 1 m: sparse). Off: on
                                       # recorded runs it accepted 0.7-0.9 m corrections from < 100 points
                                       # that made the grid worse (more obstacle cells)
    travel_min_score: float = 0.6      # stricter score for them
    # --- loop edge information = inverse(cov_scale x matcher covariance + floor^2)
    cov_scale: float = 4.0             # the matcher's score_gain (k = 30) is heuristic; 4 = sigma x2
    cov_xy_min: float = 0.02           # m, floor of a loop edge's position sigma
    cov_th_min_deg: float = 1.0        # floor of a loop edge's heading sigma
    # --- back end (navette.PoseGraph)
    dcs_phi: float = 10.0              # DCS: 1 rejects true loops once the drift is a few sigma
    opt_iters: int = 20                # LM iterations per optimisation (ms for < 1000 nodes)
    min_dcs_scale: float = 0.5         # an accepted edge the graph scales below this is counted as rejected


@dataclass
class Submap:
    """One submap. first / last: frame indices [first, last] (last None while open). anchor_odom: odometry pose
    of frame `first`. points: (N, 2) obstacle points in the anchor frame (None until closed)."""
    index: int
    kind: str                          # "spin", "travel" or "open"
    first: int
    anchor_odom: tuple
    last: int | None = None
    points: np.ndarray | None = None


@dataclass
class Match:
    """One matching attempt of submap `query` (node index = submap index). ref: the reference submaps; base: the
    node the loop edge starts from; guess, pose: anchor pose in the map frame before / from the match; window:
    (dx, dy, dtheta rad); sigma_xy: sqrt(lambda_max) of the match's xy covariance, sigma_th (rad).
    accepted: the edge was added; reason: why not (or "ok"); edge: its index; dcs_scale: its DCS scale after
    the optimisation that followed (or the latest one, see SlamFrontEnd.matches)."""
    query: int
    kind: str
    ref: list
    base: int | None = None
    odom_sigma: tuple | None = None    # accumulated odometry (sigma_xy m, sigma_theta rad) since ref[0]
    guess: tuple | None = None
    pose: tuple | None = None
    window: tuple | None = None
    score: float = 0.0
    n: int = 0
    n_ref: int = 0
    sigma_xy: float = float("nan")
    sigma_th: float = float("nan")
    accepted: bool = False
    reason: str = ""
    edge: int | None = None
    z: tuple | None = None
    dcs_scale: float = float("nan")
    ms: float = 0.0


def odometry_sigmas(p, dist, turn, dt):
    """(sigma_xy (m), sigma_theta (rad)) of the odometry motion over one edge: dist driven (m), turn = total
    |rotation| (rad, not the net change: a spin turns 2 pi for no net change), dt (s). Linear in the three, not
    root-sum-square: scale errors and the gyro bias are systematic over an edge, not white."""
    s_xy = max(p.odom_xy_min, p.odom_xy_per_m * dist)
    s_th = max(math.radians(p.odom_th_min_deg),
               p.odom_th_per_rad * turn + p.odom_th_per_m * dist + math.radians(p.odom_bias_deg_min) / 60.0 * dt)
    return s_xy, s_th


def heading_span(heading):
    """Span (max - min, rad) of unwrapped headings: the field turned through, back-and-forth counted once."""
    h = np.asarray(heading, float)
    return float(h.max() - h.min()) if len(h) else 0.0


class SlamFrontEnd:
    """See the module docstring. Use: add() per frame (in time order), then finish() at the end of a run;
    pose(), map_to_odom(), poses(), scan_poses() at any time; submaps, matches, graph for diagnostics."""

    def __init__(self, params=None, loops=True):
        from navette import PoseGraph

        self.p = params or SlamParams()
        self.loops = loops  # False: odometry chain only (the same graph, no loop edge): the baseline
        self.graph = PoseGraph(dcs_phi=self.p.dcs_phi)
        self.submaps = []
        self.matches = []
        self.odom_edges = []   # (i, j, z, sigma_xy, sigma_th)
        # per frame (index = frame number): time, odometry pose, camera heading, submap, cumulative driven
        # distance / total |rotation| / unwrapped heading (for the odometry edges and the spin detection)
        self._t, self._odom, self._cam_heading, self._submap = [], [], [], []
        self._cum_dist, self._cum_turn, self._heading = [], [], []
        self._open_frames = []  # (frame index, scan dict, cam_h, glance) of the open submap
        self._episode = 0       # first frame of the current turn-in-place episode (see _check_close)
        self._poses_cache = None

    # ------------------------------------------------------------------ input

    def add(self, t, odom, scan, cam_h, cam_heading=0.0, glance=False):
        """Feed one frame. odom: (x, y, theta) base pose in odom; scan: navette.Scan or dict of its arrays;
        cam_h: camera height (m); cam_heading: camera heading in the base frame (rad); glance: a head-glance
        frame (robot still, head down: always integrated into the submap's grid, as the navigator does)."""
        odom = tuple(float(v) for v in odom)
        i = len(self._t)
        if i:
            prev = self._odom[-1]
            dth = float(wrap(odom[2] - prev[2]))
            self._cum_dist.append(self._cum_dist[-1] + math.hypot(odom[0] - prev[0], odom[1] - prev[1]))
            self._cum_turn.append(self._cum_turn[-1] + abs(dth))
            self._heading.append(self._heading[-1] + dth)
        else:
            self._cum_dist.append(0.0)
            self._cum_turn.append(0.0)
            self._heading.append(odom[2])
        self._t.append(float(t))
        self._odom.append(odom)
        self._cam_heading.append(float(cam_heading))
        self._poses_cache = None
        if not self.submaps or self.submaps[-1].last is not None:
            self._open(i)
        self._submap.append(len(self.submaps) - 1)
        self._open_frames.append((i, scan_dict(scan), float(cam_h), bool(glance)))
        self._check_close()

    def finish(self):
        """Close the open submap (end of the run): its last turn-in-place episode as a spin if it is one, the
        rest as travel."""
        if self.submaps and self.submaps[-1].last is None:
            n, e = len(self._t) - 1, self._episode
            if e <= n and heading_span(self._heading[e:]) >= math.radians(self.p.spin_deg):
                self._close_spin(e, n)
            else:
                self._close_at("travel", n)

    # ------------------------------------------------------------------ outputs

    def poses(self):
        """(N, 3) corrected base poses of all frames fed: corrected anchor o (anchor_odom^-1 o odom)."""
        if self._poses_cache is None:
            if not self._t:
                return np.zeros((0, 3))
            nodes = np.array(self.graph.poses(), float)
            sub = np.array(self._submap)
            anchors = np.array([s.anchor_odom for s in self.submaps], float)
            self._poses_cache = compose(nodes[sub], relative(anchors[sub], np.array(self._odom, float)))
        return self._poses_cache

    def scan_poses(self):
        """(N, 3) corrected scan poses (theta + cam_heading): what OccupancyGrid keyframes hold."""
        p = self.poses().copy()
        p[:, 2] = wrap(p[:, 2] + np.array(self._cam_heading))
        return p

    def pose(self):
        """Corrected base pose of the latest frame (map frame), or None before the first."""
        return tuple(self.poses()[-1]) if self._t else None

    def map_to_odom(self):
        """The map -> odom correction T (REP 105): map pose = T o odom pose. Identity before any loop."""
        if not self._t:
            return (0.0, 0.0, 0.0)
        return tuple(float(v) for v in compose(self.poses()[-1], inverse(self._odom[-1])))

    def odometry(self):
        """(N, 3) odometry base poses of the frames fed."""
        return np.array(self._odom, float).reshape(-1, 3)

    def frame_submaps(self):
        """(N,) submap index of each frame."""
        return np.array(self._submap, int)

    # ------------------------------------------------------------------ submaps

    def _open(self, i):
        """New submap anchored at frame i; its node, and the odometry edge from the previous anchor."""
        k = len(self.submaps)
        self.submaps.append(Submap(index=k, kind="open", first=i, anchor_odom=self._odom[i]))
        if k == 0:
            self.graph.add_node(self._odom[i])  # node 0 (fixed) at its odometry pose: map = odom at the start
            return
        a = self.submaps[k - 1].first
        z = tuple(float(v) for v in relative(self._odom[a], self._odom[i]))
        s_xy, s_th = odometry_sigmas(self.p, self._cum_dist[i] - self._cum_dist[a],
                                     self._cum_turn[i] - self._cum_turn[a], self._t[i] - self._t[a])
        info = np.diag([1 / s_xy**2, 1 / s_xy**2, 1 / s_th**2])
        self.graph.add_node(tuple(compose(self.graph.pose(k - 1), z)))
        self.graph.add_edge(k - 1, k, z, info)
        self.odom_edges.append((k - 1, k, z, s_xy, s_th))

    def _check_close(self):
        """Close submaps after the new frame n. A turn-in-place episode is the run of frames staying within
        spin_max_trans_m of its first position; it becomes a spin submap when the robot leaves the spot with
        a heading span >= spin_deg (the frames before it close as travel), or at once when the span reaches
        360 deg (a full view: more turning adds nothing). Otherwise the open submap closes as travel after
        travel_m of driving or travel_max_s, at the start of the current episode (a turn just begun stays
        whole, in the next submap)."""
        p, n = self.p, len(self._t) - 1
        e = self._episode
        if math.hypot(self._odom[n][0] - self._odom[e][0], self._odom[n][1] - self._odom[e][1]) > p.spin_max_trans_m:
            # left the spot: the episode [e, n - 1] is over
            if e <= n - 1 and heading_span(self._heading[e:n]) >= math.radians(p.spin_deg):
                self._close_spin(e, n - 1)
                self._episode = n
                return
            self._episode = e = n
        elif heading_span(self._heading[e:n + 1]) >= 2 * math.pi:
            self._close_spin(e, n)
            self._episode = n + 1
            return
        first = self.submaps[-1].first
        long = self._t[n] - self._t[first] >= p.travel_max_s
        if self._cum_dist[n] - self._cum_dist[first] >= p.travel_m or long:
            if long and heading_span(self._heading[e:n + 1]) >= math.radians(p.spin_deg):
                self._close_spin(e, n)
                self._episode = n + 1
            elif e > first:
                self._close_at("travel", e - 1)
            else:
                self._close_at("travel", n)
                self._episode = n + 1

    def _close_spin(self, e, last):
        """Close frames [e, last] of the open submap as a spin; the frames before e first, as travel."""
        if e > self.submaps[-1].first:
            self._close_at("travel", e - 1)
        self._close_at("spin", last)

    def _close_at(self, kind, last):
        """Close the open submap at frame `last`; its later frames (if any) open the next submap."""
        sm = self.submaps[-1]
        keep = last - sm.first + 1
        rest = self._open_frames[keep:]
        self._open_frames = self._open_frames[:keep]
        self._close(sm, kind, last)
        if rest:
            self._open(rest[0][0])
            for f in rest:
                self._submap[f[0]] = len(self.submaps) - 1
            self._open_frames = rest

    def _close(self, sm, kind, last=None):
        """Close submap sm: build its local grid and points from the open frames, then match it."""
        sm.kind, sm.last = kind, len(self._t) - 1 if last is None else last
        sm.points = self._points(sm, self._open_frames)
        self._open_frames = []
        if self.loops and (kind == "spin" or self.p.match_travel):
            self._match(sm)

    def _points(self, sm, frames):
        """Obstacle points (N, 2) of a submap in its anchor frame, from a local OccupancyGrid of its frames."""
        from navette import OccupancyGrid

        p = self.p
        grid = OccupancyGrid(res=p.grid_res, size_m=p.grid_size_m)
        a = np.asarray(sm.anchor_odom, float)
        for i, scan, cam_h, glance in frames:
            rel = relative(a, np.asarray(self._odom[i], float))
            pose = (float(rel[0]), float(rel[1]), float(rel[2] + self._cam_heading[i]))
            grid.update(self._t[i], pose, to_scan(scan), sigma=p.grid_sigma, h=cam_h, max_range=p.grid_max_range_m,
                        force=glance)
        if grid.x0 is None:
            return np.zeros((0, 2))
        ij = np.argwhere(grid.log_odds > p.points_logodds).astype(np.int64)
        return grid.centres(ij) if len(ij) else np.zeros((0, 2))

    # ------------------------------------------------------------------ matching

    def _chain_sigmas(self, r, j):
        """Accumulated odometry (sigma_xy, sigma_theta) from node r to node j > r, conservative: heading sigmas
        add linearly (systematic), and each edge's distance turns the heading sigma accumulated before it
        into a lateral position sigma."""
        s_xy, s_th = 0.0, 0.0
        for i, k, z, e_xy, e_th in self.odom_edges[r:j]:
            dist = self._cum_dist[self.submaps[k].first] - self._cum_dist[self.submaps[i].first]
            s_xy += e_xy + dist * s_th
            s_th += e_th
        return s_xy, s_th

    def _match(self, sm):
        """Match submap sm (just closed) against the earlier submaps around it: one attempt per run of
        consecutive candidate submaps (see below); each accepted match adds a loop edge."""
        p, j = self.p, sm.index
        nodes = np.array(self.graph.poses(), float)
        guess = nodes[j]
        if len(sm.points) < p.min_points:
            self.matches.append(Match(query=j, kind=sm.kind, ref=[], guess=tuple(float(v) for v in guess),
                                      reason=f"query points {len(sm.points)} < {p.min_points}"))
            return
        # candidates: closed, with points, not the local chain (back to them: little driving AND little time;
        # there odometry is better than a match on our noisy scan, and a match would mostly repeat it), anchor
        # near the guess (+ the largest window: the anchor may be that far off)
        fj = sm.first
        cands = [s.index for s in self.submaps[:j]
                 if s.points is not None and len(s.points)
                 and (self._cum_dist[fj] - self._cum_dist[s.first] >= p.loop_min_path_m
                      or self._t[fj] - self._t[s.first] >= p.loop_min_s)
                 and math.hypot(*(nodes[s.index, :2] - guess[:2])) <= p.ref_radius_m + p.window_xy_max]
        if not cands:
            self.matches.append(Match(query=j, kind=sm.kind, ref=[], guess=tuple(float(v) for v in guess),
                                      reason="no reference submap"))
            return
        # Runs of consecutive submaps (one pass through the place), at most ref_max_path_m of driving long:
        # consistent with each other by odometry, while two passes are not until a loop joins them (a merged
        # reference would hold the same wall twice, offset by the drift between the passes, and the match
        # would be a compromise). One match per run.
        runs, run = [], [cands[0]]
        for c in cands[1:]:
            if (c == run[-1] + 1 and self._cum_dist[self.submaps[c].first]
                    - self._cum_dist[self.submaps[run[0]].first] < p.ref_max_path_m):
                run.append(c)
            else:
                runs.append(run)
                run = [c]
        runs.append(run)
        for run in runs:
            self._match_run(sm, run)

    def _match_run(self, sm, run):
        """One match of submap sm against the submaps `run` (consecutive indices); loop edge if accepted."""
        import time

        from navette import LikelihoodGrid, match_points

        p, j = self.p, sm.index
        t0 = time.perf_counter()
        nodes = np.array(self.graph.poses(), float)  # current: an earlier run's loop edge may have moved them
        guess = nodes[j]
        m = Match(query=j, kind=sm.kind, ref=list(run), guess=tuple(float(v) for v in guess))
        self.matches.append(m)
        # window: the odometry uncertainty accumulated since the run's oldest submap (the largest of the run)
        s_xy, s_th = self._chain_sigmas(run[0], j)
        wxy = float(np.clip(p.window_nsigma * s_xy, p.window_xy_min, p.window_xy_max))
        wth = float(np.clip(p.window_nsigma * s_th, math.radians(p.window_th_min_deg),
                            math.radians(p.window_th_max_deg)))
        m.window, m.odom_sigma = (wxy, wxy, wth), (s_xy, s_th)
        # edge base: the run's submap whose anchor is nearest the guess
        m.base = min(run, key=lambda c: (math.hypot(*(nodes[c, :2] - guess[:2])), c))
        # the reference: the run's points at their corrected poses (map frame)
        ref = np.vstack([compose(np.broadcast_to(nodes[c], (len(self.submaps[c].points), 3)),
                                 np.column_stack([self.submaps[c].points, np.zeros(len(self.submaps[c].points))]))
                         [:, :2] for c in run])
        m.n_ref = len(ref)
        if m.n_ref < p.min_ref_points:
            m.reason = f"reference points {m.n_ref} < {p.min_ref_points}"
            return
        lg = LikelihoodGrid(ref, res=p.match_res, sigma=p.match_sigma, margin=3 * p.match_sigma)
        res = match_points(lg, sm.points, tuple(guess), dx=wxy, dy=wxy, dtheta=wth)
        m.ms = 1000 * (time.perf_counter() - t0)
        r = m.base
        m.pose, m.score, m.n = tuple(float(v) for v in res["pose"]), float(res["score"]), int(res["n"])
        cov = np.asarray(res["covariance"], float)
        m.sigma_xy, m.sigma_th = math.sqrt(res["xy_eigenvalues"][1]), math.sqrt(cov[2, 2])
        off = np.array(m.pose) - guess
        off[2] = wrap(off[2])
        min_score = p.min_score if sm.kind == "spin" else max(p.min_score, p.travel_min_score)
        if m.score < min_score:
            m.reason = f"score {m.score:.2f} < {min_score}"
        elif m.sigma_xy > p.max_cond_frac * wxy:
            m.reason = f"ill-conditioned xy: {m.sigma_xy:.3f} m > {p.max_cond_frac} x {wxy:.2f}"
        elif m.sigma_th > p.max_th_cond_frac * wth:
            m.reason = f"ill-conditioned theta: {math.degrees(m.sigma_th):.1f} deg > {p.max_th_cond_frac} x " \
                       f"{math.degrees(wth):.0f}"
        elif max(abs(off[0]), abs(off[1])) > p.border_frac * wxy or abs(off[2]) > p.border_frac * wth:
            m.reason = "best at the window border"
        else:
            m.reason, m.accepted = "ok", True
        if not m.accepted:
            return
        # the edge r -> j: j's matched pose in r's frame. The matcher's covariance is of the map-frame pose;
        # the graph's edge error has its translation in j's frame (R_z^T R_r^T (t_j - t_r) - ...): rotate it
        c, s = math.cos(m.pose[2]), math.sin(m.pose[2])
        J = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        cov_j = p.cov_scale * J @ cov @ J.T + np.diag([p.cov_xy_min**2, p.cov_xy_min**2,
                                                       math.radians(p.cov_th_min_deg)**2])
        info = np.linalg.inv(cov_j)
        info = 0.5 * (info + info.T)  # exactly symmetric (the binding checks)
        m.z = tuple(float(v) for v in relative(nodes[r], np.array(m.pose)))
        m.edge = self.graph.add_edge(r, j, m.z, info, robust=True)
        self.graph.optimize(p.opt_iters)
        self._poses_cache = None
        m.dcs_scale = float(self.graph.edge_scales()[m.edge])
        m.ms = 1000 * (time.perf_counter() - t0)
        if m.dcs_scale < p.min_dcs_scale:
            m.reason = f"down-weighted by the graph (DCS scale {m.dcs_scale:.2f})"

    # ------------------------------------------------------------------ diagnostics

    def loop_scales(self):
        """Current DCS scale of every accepted loop edge (updated: later loops can re-weight earlier ones)."""
        scales = self.graph.edge_scales()
        for m in self.matches:
            if m.edge is not None:
                m.dcs_scale = float(scales[m.edge])
        return [m.dcs_scale for m in self.matches if m.edge is not None]

    def stats(self):
        """Counts for a report: submaps per kind, matches tried / accepted / kept by the graph, map->odom."""
        self.loop_scales()
        kinds = [s.kind for s in self.submaps]
        kept = [m for m in self.matches if m.edge is not None and m.dcs_scale >= self.p.min_dcs_scale]
        T = self.map_to_odom()
        return dict(frames=len(self._t), submaps=len(self.submaps), spins=kinds.count("spin"),
                    travel=kinds.count("travel"), tried=len(self.matches),
                    accepted=sum(m.edge is not None for m in self.matches), kept=len(kept),
                    correction_m=math.hypot(T[0], T[1]), correction_deg=math.degrees(T[2]))
