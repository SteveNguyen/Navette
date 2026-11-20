"""Offline SLAM data: a sequence of frames (time, odometry pose, scan, camera height, optional ground truth), as
plain numpy arrays saved to one .npz. It is what a SLAM front end gets at run time, minus the pictures: a
simulated scenario (tests/sim_slam.py) and a replayed robot run (navette.slam.extract) both produce one, so the
same evaluation runs on both.

Frames (docs/PLAN.md, REP 105): `odom` is the base's odometry pose (x, y, theta of the base frame in the odom
frame); the scan is in the camera's horizontal frame, whose heading in the base frame is `cam_heading` (0 when
the camera looks straight ahead, a few hundredths of a rad on the robot). The pose the navigator's grid uses
for a scan is therefore (x, y, theta + cam_heading): `scan_pose()`. Ground truth `gt`, when there is one, is the
same base pose in the world frame (whose origin differs from odom's: compare after navette.slam.metrics.anchor
or align).

A scan is the six per-sector arrays of navette.Scan (crates/navette-py/src/scan.rs): bearing_deg (one copy for
the whole sequence: the sectors are fixed by the camera's FreeSpace), seen, obstacle, near, floor, closest.
"""
import json
from dataclasses import dataclass, field

import numpy as np

# The per-frame arrays of a scan, in navette.Scan's argument order (bearing_deg is per sequence).
SCAN_KEYS = ("seen", "obstacle", "near", "floor", "closest")
_SCAN_DTYPES = dict(seen=bool, obstacle=float, near=bool, floor=float, closest=float)


@dataclass
class Keyframe:
    """One frame. scan: dict of the six arrays of a navette.Scan (as tests/sim2d.Sim.scan returns)."""
    t: float                    # s, capture time
    odom: tuple                 # (x, y, theta) of the base in the odom frame (m, m, rad)
    scan: dict
    cam_h: float                # m, camera height above the floor
    cam_heading: float = 0.0    # rad, camera heading in the base frame
    gt: tuple | None = None     # (x, y, theta) true base pose in the world frame, if known
    glance: bool = False        # a head-glance frame (camera looking down at the floor right ahead; robot still)

    def scan_pose(self):
        """(x, y, theta) of the scan's frame in odom: the pose the navigator's grid gets with this scan."""
        x, y, th = self.odom
        return (x, y, th + self.cam_heading)


def to_scan(scan):
    """navette.Scan from a dict of its arrays (imports the extension only here: data and metrics work without
    it)."""
    from navette import Scan

    return Scan(**{k: scan[k] for k in ("bearing_deg",) + SCAN_KEYS})


def scan_dict(scan):
    """Dict of the six arrays of a navette.Scan (or of a dict with them), as numpy copies."""
    get = scan.get if isinstance(scan, dict) else lambda k: getattr(scan, k)
    return {k: np.array(get(k), dtype=float if k == "bearing_deg" else _SCAN_DTYPES[k])
            for k in ("bearing_deg",) + SCAN_KEYS}


@dataclass
class Sequence:
    """N frames in columns. t (N,), odom (N, 3), cam_h (N,), cam_heading (N,), glance (N,) bool, gt (N, 3) or
    None, bearing_deg (S,), scans: dict SCAN_KEY -> (N, S) array. meta: JSON-serialisable dict (where the data
    comes from, scenario parameters...)."""
    t: np.ndarray
    odom: np.ndarray
    cam_h: np.ndarray
    cam_heading: np.ndarray
    glance: np.ndarray
    bearing_deg: np.ndarray
    scans: dict
    gt: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return Keyframe(t=float(self.t[i]), odom=tuple(map(float, self.odom[i])), scan=self.scan(i),
                        cam_h=float(self.cam_h[i]), cam_heading=float(self.cam_heading[i]),
                        gt=None if self.gt is None else tuple(map(float, self.gt[i])), glance=bool(self.glance[i]))

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def scan(self, i):
        """Frame i's scan as a dict of arrays (Sim.scan's format)."""
        d = {k: self.scans[k][i] for k in SCAN_KEYS}
        d["bearing_deg"] = self.bearing_deg
        return d

    def navette_scan(self, i):
        """Frame i's scan as a navette.Scan."""
        return to_scan(self.scan(i))

    def scan_poses(self):
        """(N, 3) poses of the scans in odom (theta + cam_heading)."""
        p = self.odom.copy()
        p[:, 2] += self.cam_heading
        return p

    @classmethod
    def from_keyframes(cls, frames, meta=None):
        """Sequence of a list of Keyframe. Ground truth is kept only if every frame has one."""
        frames = list(frames)
        if not frames:
            raise ValueError("no frames")
        bearing = np.asarray(frames[0].scan["bearing_deg"], float)
        for f in frames:
            if not np.array_equal(np.asarray(f.scan["bearing_deg"], float), bearing):
                raise ValueError(f"frame at t={f.t}: scan sectors differ from the first frame's")
        has_gt = all(f.gt is not None for f in frames)
        return cls(t=np.array([f.t for f in frames], float),
                   odom=np.array([f.odom for f in frames], float).reshape(-1, 3),
                   cam_h=np.array([f.cam_h for f in frames], float),
                   cam_heading=np.array([f.cam_heading for f in frames], float),
                   glance=np.array([f.glance for f in frames], bool),
                   bearing_deg=bearing,
                   scans={k: np.array([np.asarray(f.scan[k], _SCAN_DTYPES[k]) for f in frames]) for k in SCAN_KEYS},
                   gt=np.array([f.gt for f in frames], float).reshape(-1, 3) if has_gt else None,
                   meta=dict(meta or {}))

    def save(self, path):
        """One .npz (compressed: scans are mostly inf / False). meta is stored as JSON, so the file loads
        without pickle."""
        arrays = dict(t=self.t, odom=self.odom, cam_h=self.cam_h, cam_heading=self.cam_heading, glance=self.glance,
                      bearing_deg=self.bearing_deg, meta=np.array(json.dumps(self.meta)),
                      **{f"scan_{k}": v for k, v in self.scans.items()})
        if self.gt is not None:
            arrays["gt"] = self.gt
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(t=z["t"], odom=z["odom"], cam_h=z["cam_h"], cam_heading=z["cam_heading"], glance=z["glance"],
                       bearing_deg=z["bearing_deg"], scans={k: z[f"scan_{k}"] for k in SCAN_KEYS},
                       gt=z["gt"] if "gt" in z.files else None, meta=json.loads(str(z["meta"])))
