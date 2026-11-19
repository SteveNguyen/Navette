"""Shared test helpers (imported by the tests as `from conftest import ...`)."""

import json
from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).parent / "golden"


def golden_files(module):
    """The golden files of a module (tests/golden/<module>_*.npz, see tests/golden/README.md)."""
    return sorted(GOLDEN.glob(f"{module}_*.npz"))


def camera(pitch_down_deg, roll_deg):
    """R_wc (columns = camera axes in the world: z up, x forward, y left) and 'up' in the camera frame, for a
    camera pitched down then rolled about the world x axis
    (scipy: Rotation.from_euler("yx", [pitch, roll], degrees=True))."""
    p, r = np.radians(pitch_down_deg), np.radians(roll_deg)
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    R_wc_level = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # x_c = -y_w, y_c = -z_w, z_c = x_w
    R_wc = Rx @ Ry @ R_wc_level
    return R_wc, R_wc.T @ np.array([0, 0, 1.0])


def ideal_calib(tmp_path):
    """Calibration file of an ideal lens with the robot camera's focal length, 1280x720."""
    p = tmp_path / "calib.json"
    p.write_text(json.dumps({"image_resolution": [1280, 720], "K": [[822.0, 0, 640.0], [0, 822.0, 360.0], [0, 0, 1]],
                             "D": [0.0] * 5}))
    return str(p)


def free_space(calib_path, **kw):
    """navette.FreeSpace from a calibration file. Needs OpenCV."""
    from navette import FreeSpace
    from navette.rays import mask_rays

    return FreeSpace(mask_rays(calib_path), mask_rays(calib_path, dv=0.5), **kw)
