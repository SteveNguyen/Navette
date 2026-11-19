"""Picture for the status page and the replay video: the camera with the drivable floor | the map around the
robot. Needs OpenCV (optional dependency)."""
import math

import cv2
import numpy as np


def draw_map(grid, pose, scan, half=2.5, size=360):
    """Top-down view, +-half m around the robot: map frame x up in the image, y left. Free (light), obstacle
    (dark red), unknown (grey), scan ends (red: obstacle, green: end of the seen floor), robot (purple).
    grid: navette.OccupancyGrid (masks(), x0, y0, res; cell (i, j) covers x0 + i res, y0 + j res) or None."""
    img = np.full((size, size, 3), 150, np.uint8)
    k = size / (2 * half)

    def P(x, y):  # map-frame point -> pixel (col, row)
        return int(round((pose[1] + half - y) * k)), int(round((pose[0] + half - x) * k))

    if grid is not None and grid.x0 is not None:
        obst, free = grid.masks()
        n = int(round(2 * half / grid.res))
        i0 = int(math.floor((pose[0] - half - grid.x0) / grid.res))
        j0 = int(math.floor((pose[1] - half - grid.y0) / grid.res))
        ii, jj = np.meshgrid(np.arange(i0, i0 + n), np.arange(j0, j0 + n), indexing="ij")
        ok = (ii >= 0) & (ii < obst.shape[0]) & (jj >= 0) & (jj < obst.shape[1])
        cells = np.full((n, n), 0, np.uint8)  # 0 unknown, 1 free, 2 obstacle
        cells[ok] = np.where(obst[ii[ok], jj[ok]], 2, np.where(free[ii[ok], jj[ok]], 1, 0))
        lut = np.array([[150, 150, 150], [235, 245, 235], [40, 40, 150]], np.uint8)
        # rows: decreasing x (i), columns: decreasing y (j)
        img = cv2.resize(lut[cells[::-1, ::-1]], (size, size), interpolation=cv2.INTER_NEAREST)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    for b, d, o in zip(np.radians(scan.bearing_deg), scan.free(), scan.obstacle):
        if np.isnan(d):
            continue
        q = (pose[0] + d * math.cos(pose[2] + b), pose[1] + d * math.sin(pose[2] + b))
        cv2.circle(img, P(*q), 2, (0, 0, 230) if np.isfinite(o) and o <= d + 1e-6 else (0, 170, 0), -1)
    cv2.circle(img, P(pose[0], pose[1]), int(0.13 * k), (120, 0, 120), 2)
    cv2.line(img, P(pose[0], pose[1]), P(pose[0] + 0.3 * c, pose[1] + 0.3 * s), (120, 0, 120), 2)
    return img


def view(bgr, drive, under, out, cam_h, pitch, grid, pose, scan, text=""):
    """Camera 640x360 with green drivable floor and orange `under` (or None) | map 360x360."""
    cam = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
    # tints with cv2 blends + masked copies: numpy fancy indexing with float maths took most of a ~65 ms
    # preview on the CM4
    m = cv2.resize(drive.astype(np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST)
    cv2.copyTo(cv2.addWeighted(cam, 0.6, np.full_like(cam, (0, 100, 0)), 1.0, 0), m, cam)
    if under is not None:
        mu = cv2.resize(under.astype(np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST)
        cv2.copyTo(cv2.addWeighted(cam, 0.4, np.full_like(cam, (0, 110, 150)), 1.0, 0), mu, cam)
    cv2.putText(cam, f"{out.state} v {out.v:.2f} w {out.w:+.2f} (req {out.v_req:.2f} {out.w_req:+.2f}) "
                     f"h {cam_h:.3f} pitch {pitch:.1f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if text:
        cv2.putText(cam, text, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.concatenate([cam, draw_map(grid, pose, scan)], 1)


def jpeg(img, quality=70):
    """JPEG bytes, or None if the encoding failed."""
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpg.tobytes() if ok else None
