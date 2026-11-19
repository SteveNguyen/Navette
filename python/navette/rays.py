"""Viewing rays of an OpenCV-calibrated camera: the lens model, kept outside the core (adapter utility).

Needs OpenCV (optional dependency: `navette[opencv]`). Computed once per camera and mask size:

    from navette import FreeSpace
    from navette.rays import mask_rays
    fs = FreeSpace(mask_rays("calib.json"), mask_rays("calib.json", dv=0.5))
"""

import json

import cv2
import numpy as np

MASK_W, MASK_H = 320, 192  # floor model output


def mask_rays(calib_path, w=MASK_W, h=MASK_H, dv=0.0):
    """Undistorted viewing rays (h*w, 3), z = 1, of every pixel of a w x h mask resized from the calibrated
    image (the resize need not keep the aspect ratio: 1280x720 -> 320x192 does not).
    calib_path: json with "image_resolution" [W, H], "K" (3x3) and "D" (OpenCV distortion coefficients).
    dv: offset in mask pixels from the pixel centre (+0.5 = the pixel's lower edge)."""
    with open(calib_path) as f:
        cal = json.load(f)
    W0, H0 = cal["image_resolution"]
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    u = (uu + 0.5) * W0 / w - 0.5  # mask pixel centre -> full-resolution pixel coordinates
    v = (vv + 0.5 + dv) * H0 / h - 0.5
    pts = np.stack([u.ravel(), v.ravel()], -1)[:, None, :]
    # strong rational lens: more iterations than the default 5, or the image corners stay distorted
    xy = cv2.undistortPoints(pts, np.array(cal["K"]), np.array(cal["D"]),
                             criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9))[:, 0]
    return np.hstack([xy, np.ones((len(xy), 1))])


def pixel_rays(uv, K, D):
    """Undistorted viewing rays (N,3) in the camera frame, z component = 1, of pixels uv (N,2)."""
    uv = np.asarray(uv, float).reshape(-1, 1, 2)
    xy = cv2.undistortPoints(uv, np.asarray(K, float), np.asarray(D, float)).reshape(-1, 2)
    return np.hstack([xy, np.ones((len(xy), 1))])
