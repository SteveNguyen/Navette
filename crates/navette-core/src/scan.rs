//! Free space from a floor mask: distance to the nearest obstacle per bearing, a virtual laser scan.
//!
//! Each mask pixel's viewing ray (calibrated lens, computed once by the adapter) is turned into the
//! gravity-aligned frame with the current 'up' and intersected with the floor plane at camera height h
//! (geometry.rs frame: x forward, y left, bearing 0 = forward, positive = left). The lowest pixel of an
//! obstacle is where it touches the floor, so it projects to the right floor point; its higher pixels project
//! farther (behind it) and pixels above the horizon nowhere. So per bearing sector, the nearest projected
//! non-floor pixel is the obstacle distance.
//!
//! Limit: obstacles are found where they touch the floor; overhangs (chair seat, sofa edge) are missed,
//! unless the mask already excludes the floor under them (floor_seg 'under' output, [`drive_mask`]): then the
//! under region's near edge, where the overhang starts, is the obstacle.
//!
//! Numerics follow the Python prototype (numpy), so that its outputs are reproduced: the ray rotation is f64, the projection
//! (range, bearing) f32, the scan arrays f64.

use crate::geometry::{dot, gravity_frame, Mat3, Vec3};

/// Floor model output size: the default mask shape, (rows, columns).
pub const MASK_SHAPE: (usize, usize) = (192, 320);

/// numpy's float32 `degrees` factor: 180f / pi_f, computed in f32. It is one ulp below
/// f32(180 / pi), which `f32::to_degrees` uses, and that ulp moves pixels across sector boundaries.
const RAD2DEG: f32 = 180.0 / std::f32::consts::PI;

/// Per-bearing scan. All arrays have one entry per sector.
#[derive(Clone, Debug, PartialEq)]
pub struct Scan {
    /// Sector centres (deg), 0 = forward, positive = left.
    pub bearing_deg: Vec<f64>,
    /// Some image pixel looks at the floor in this bearing (false = outside the view: unknown).
    pub seen: Vec<bool>,
    /// Distance to the nearest obstacle (m); inf if none within max_range.
    pub obstacle: Vec<f64>,
    /// The obstacle covers the closest point the camera sees in this bearing (it fills the image bottom), so
    /// the true obstacle may be closer than `obstacle`.
    pub near: Vec<bool>,
    /// Farthest floor point seen (m), capped at max_range; beyond it nothing is known.
    pub floor: Vec<f64>,
    /// Closest floor-plane point the camera sees (m, inf if not seen); nearer than that is the blind zone
    /// below the image.
    pub closest: Vec<f64>,
}

impl Scan {
    /// Number of sectors.
    pub fn len(&self) -> usize {
        self.bearing_deg.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bearing_deg.is_empty()
    }

    /// Usable free distance = min(obstacle, floor); NaN where not seen.
    pub fn free(&self) -> Vec<f64> {
        (0..self.len())
            .map(|k| {
                let (o, f) = (self.obstacle[k], self.floor[k]);
                if !self.seen[k] || o.is_nan() || f.is_nan() {
                    f64::NAN // np.minimum propagates NaN, f64::min does not
                } else {
                    o.min(f)
                }
            })
            .collect()
    }
}

/// Scan parameters. The defaults are the ones tuned on Reachy Mini (320x192 mask, ~9 Hz).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScanParams {
    /// Sector width (deg).
    pub sector_deg: f64,
    /// Obstacles at or beyond this range (m) are dropped; `floor` is capped at it.
    pub max_range: f64,
    /// Obstacle blobs smaller than this (mask pixels) are ignored as noise.
    pub min_blob_px: usize,
    /// An obstacle within near_tol (m) of the closest visible point counts as `near`.
    pub near_tol: f64,
}

impl Default for ScanParams {
    fn default() -> Self {
        Self {
            sector_deg: 2.0,
            max_range: 2.5,
            min_blob_px: 10,
            near_tol: 0.05,
        }
    }
}

/// An input whose size does not match the mask shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeError(pub String);

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ShapeError {}

/// Floor mask -> [`Scan`], for one camera (its rays) and one mask size.
pub struct FreeSpace {
    /// Viewing ray of every mask pixel centre, camera frame, row-major.
    rays: Vec<Vec3>,
    /// Same through the pixel's LOWER edge. Obstacle pixels are projected through it: the floor/obstacle
    /// boundary is there (to +-half a pixel), whereas the pixel centre is up to a pixel beyond it, i.e. too
    /// far (2.6 cm at 1.2 m).
    rays_low: Vec<Vec3>,
    shape: (usize, usize),
    params: ScanParams,
    n: usize,
    bearing_deg: Vec<f64>,
    // Scratch buffers reused between scans (no per-scan allocation of image-sized arrays).
    obst: Vec<bool>,
    labels: Vec<u32>,
    parent: Vec<u32>,
    area: Vec<u32>,
}

impl FreeSpace {
    /// `rays`, `rays_low`: one viewing ray per mask pixel (row-major, `shape` = (rows, columns)), through the
    /// pixel centre and through its lower edge (see `navette.rays.mask_rays`, dv = 0 and dv = 0.5).
    pub fn new(
        rays: Vec<Vec3>,
        rays_low: Vec<Vec3>,
        shape: (usize, usize),
        params: ScanParams,
    ) -> Result<Self, ShapeError> {
        let px = shape.0 * shape.1;
        if rays.len() != px || rays_low.len() != px {
            return Err(ShapeError(format!(
                "rays ({}) and rays_low ({}) must have one row per pixel of the {}x{} mask ({px})",
                rays.len(),
                rays_low.len(),
                shape.0,
                shape.1
            )));
        }
        // Python's round(): ties to even
        let n = (360.0 / params.sector_deg).round_ties_even() as usize;
        let bearing_deg = (0..n)
            .map(|k| (k as f64 + 0.5) * params.sector_deg - 180.0)
            .collect();
        Ok(Self {
            rays,
            rays_low,
            shape,
            params,
            n,
            bearing_deg,
            obst: Vec::new(),
            labels: Vec::new(),
            parent: Vec::new(),
            area: Vec::new(),
        })
    }

    /// Number of sectors.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Sector centres (deg).
    pub fn bearing_deg(&self) -> &[f64] {
        &self.bearing_deg
    }

    /// Mask shape (rows, columns).
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn params(&self) -> &ScanParams {
        &self.params
    }

    fn check_mask(&self, floor: &[bool]) -> Result<(), ShapeError> {
        if floor.len() == self.shape.0 * self.shape.1 {
            Ok(())
        } else {
            Err(ShapeError(format!(
                "floor mask has {} pixels, expected {}x{}",
                floor.len(),
                self.shape.0,
                self.shape.1
            )))
        }
    }

    /// Non-floor pixels without the blobs (8-connected) smaller than min_blob_px. Row-major, like `floor`.
    pub fn obstacles(&mut self, floor: &[bool]) -> Result<&[bool], ShapeError> {
        self.check_mask(floor)?;
        self.obst.clear();
        if self.params.min_blob_px > 1 {
            self.label_blobs(floor);
            let (labels, parent, area) = (&self.labels, &self.parent, &self.area);
            let min_area = self.params.min_blob_px as u32;
            self.obst.extend(
                labels
                    .iter()
                    .map(|&l| l != 0 && area[root(parent, l) as usize] >= min_area),
            );
        } else {
            self.obst.extend(floor.iter().map(|&f| !f));
        }
        Ok(&self.obst)
    }

    /// 8-connected component labelling of the non-floor pixels (replaces cv2.connectedComponentsWithStats):
    /// one raster pass with a union-find over provisional labels. Afterwards `labels` holds a provisional
    /// label per pixel (0 = floor) and `area[root(label)]` the pixel count of its blob.
    fn label_blobs(&mut self, floor: &[bool]) {
        let (h, w) = self.shape;
        let (labels, parent, area) = (&mut self.labels, &mut self.parent, &mut self.area);
        labels.clear();
        labels.resize(h * w, 0);
        parent.clear();
        parent.push(0); // label 0 = background
        area.clear();
        area.push(0);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if floor[i] {
                    continue;
                }
                // Already-visited neighbours: W, and NW, N, NE in the row above.
                let mut l = 0;
                let mut join = |j: usize| {
                    if labels[j] != 0 {
                        let r = root(parent, labels[j]);
                        if l == 0 {
                            l = r;
                        } else if r != l {
                            // merge the two blobs under the smaller root label
                            let (lo, hi) = (l.min(r), l.max(r));
                            parent[hi as usize] = lo;
                            area[lo as usize] += area[hi as usize];
                            l = lo;
                        }
                    }
                };
                if x > 0 {
                    join(i - 1);
                }
                if y > 0 {
                    if x > 0 {
                        join(i - w - 1);
                    }
                    join(i - w);
                    if x + 1 < w {
                        join(i - w + 1);
                    }
                }
                if l == 0 {
                    l = parent.len() as u32;
                    parent.push(l);
                    area.push(0);
                }
                labels[i] = l;
                area[l as usize] += 1; // l is a root here
            }
        }
    }

    /// Floor range (NaN for rays at or above the horizon) and bearing sector of gravity-frame rays.
    /// f32 and the bearing from the ray itself (the range scale s > 0 does not change it): the f64 atan2
    /// dominated the scan on the CM4 (f32 was 1.7x faster in the numpy prototype), for a range error < 2 mm
    /// and the same sectors. The range goes back to f64, the type of the scan arrays.
    pub fn project(&self, rays_g: &[Vec3], h: f64) -> (Vec<f64>, Vec<usize>) {
        let neg_h = -h as f32;
        rays_g
            .iter()
            .map(|g| {
                let g = [g[0] as f32, g[1] as f32, g[2] as f32];
                let r = if g[2] < -1e-9 { range(&g, neg_h) } else { f64::NAN };
                (r, self.sector(g[0], g[1]))
            })
            .unzip()
    }

    /// Bearing sector of a gravity-frame direction (x forward, y left): sector 0 starts at -180 deg.
    /// The prototype's numpy arithmetic, all f32: `int((degrees(arctan2(y, x)) + 180) / sector_deg) % n`.
    /// The cast truncates towards zero and the modulo is floored (Python's); +180 deg (straight back) falls in
    /// sector n, wrapped to 0.
    #[inline]
    fn sector(&self, x: f32, y: f32) -> usize {
        let deg = y.atan2(x) * RAD2DEG;
        let k = ((deg + 180.0) / self.params.sector_deg as f32) as i64;
        k.rem_euclid(self.n as i64) as usize
    }

    /// Range and sector of the floor point seen along a camera-frame ray; None at or above the horizon.
    /// The vertical component is tested first, so that rays above the horizon cost one dot product.
    #[inline]
    fn floor_hit(&self, r_gc: &Mat3, ray: &Vec3, neg_h: f32) -> Option<(f64, usize)> {
        let gz = dot(&r_gc[2], ray) as f32;
        if gz.is_nan() || gz >= -1e-9 {
            return None;
        }
        let g = [dot(&r_gc[0], ray) as f32, dot(&r_gc[1], ray) as f32, gz];
        let r = range(&g, neg_h);
        // only finite ranges count: a camera looking straight down has no forward direction
        // (NaN x and y rows in R_gc), and then sees nothing
        r.is_finite().then(|| (r, self.sector(g[0], g[1])))
    }

    /// floor: mask (row-major, `shape`), true = floor. up_c: 'up' in the camera frame. h: camera height (m).
    pub fn scan(&mut self, floor: &[bool], up_c: &Vec3, h: f64) -> Result<Scan, ShapeError> {
        self.obstacles(floor)?;
        let r_gc = gravity_frame(up_c);
        let neg_h = -h as f32;
        let n = self.n;
        let mut seen = vec![false; n];
        let mut closest = vec![f64::INFINITY; n];
        let mut obstacle = vec![f64::INFINITY; n];
        let mut floor_r = vec![0.0f64; n];
        for (i, &obst) in self.obst.iter().enumerate() {
            // pixel centre: what the camera sees of the floor plane, and the floor itself
            if let Some((r, sec)) = self.floor_hit(&r_gc, &self.rays[i], neg_h) {
                seen[sec] = true;
                closest[sec] = closest[sec].min(r);
                if !obst {
                    floor_r[sec] = floor_r[sec].max(r);
                }
            }
            // lower edge, obstacle pixels only
            if obst {
                if let Some((r, sec)) = self.floor_hit(&r_gc, &self.rays_low[i], neg_h) {
                    obstacle[sec] = obstacle[sec].min(r);
                }
            }
        }
        let p = &self.params;
        for o in obstacle.iter_mut() {
            if *o >= p.max_range {
                *o = f64::INFINITY;
            }
        }
        let near = (0..n)
            .map(|k| obstacle[k].is_finite() && obstacle[k] <= closest[k] + p.near_tol)
            .collect();
        for f in floor_r.iter_mut() {
            *f = f.min(p.max_range);
        }
        Ok(Scan {
            bearing_deg: self.bearing_deg.clone(),
            seen,
            obstacle,
            near,
            floor: floor_r,
            closest,
        })
    }
}

/// Root of a provisional label. No path compression: merges go under the smaller label, which keeps the
/// chains short on a raster scan.
#[inline]
fn root(parent: &[u32], mut l: u32) -> u32 {
    while parent[l as usize] != l {
        l = parent[l as usize];
    }
    l
}

/// Horizontal range of the floor point along the gravity-frame ray `g` (g[2] < 0), camera at height -neg_h.
/// All f32: scale s = -h / g_z to reach the plane z = -h, times the horizontal norm.
#[inline]
fn range(g: &[f32; 3], neg_h: f32) -> f64 {
    ((neg_h / g[2]) * g[0].hypot(g[1])) as f64
}

/// Pixels the robot may drive on: floor (p_floor > thr), minus floor with something low above it
/// (p_under > under_thr). Under pixels then count as obstacles in the scan, at the under region's near edge
/// (where the overhang starts). under_thr None, or a floor-only model (p_under None): floor alone.
pub fn drive_mask(p_floor: &[f32], p_under: Option<&[f32]>, thr: f32, under_thr: Option<f32>) -> Vec<bool> {
    match (p_under, under_thr) {
        (Some(p_under), Some(under_thr)) => p_floor
            .iter()
            .zip(p_under)
            .map(|(&f, &u)| f > thr && (u <= under_thr || u.is_nan())) // NaN is not 'under', as in numpy
            .collect(),
        _ => p_floor.iter().map(|&f| f > thr).collect(),
    }
}

/// floor_seg logits (C channels of `px` pixels each, channel-major) -> (floor prob, under prob or None).
/// Floor = softmax over channels 0-1 only; a 3-channel model's channel 2 is the 'under' logit
/// (FloorNet(under=True)), read with a sigmoid.
pub fn seg_probs(z: &[f32], px: usize) -> (Vec<f32>, Option<Vec<f32>>) {
    let (z0, z1) = (&z[..px], &z[px..2 * px]);
    let floor = z0
        .iter()
        .zip(z1)
        .map(|(&a, &b)| {
            let m = a.max(b); // subtract the max: no overflow in exp
            let (e0, e1) = ((a - m).exp(), (b - m).exp());
            e1 / (e0 + e1)
        })
        .collect();
    let under = (z.len() > 2 * px).then(|| {
        z[2 * px..3 * px]
            .iter()
            .map(|&u| 1.0 / (1.0 + (-u).exp()))
            .collect()
    });
    (floor, under)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::up_from_pitch;

    /// Ideal pinhole rays for a w x h mask, focal length f (pixels), dv as in mask_rays.
    fn pinhole(w: usize, h: usize, f: f64, dv: f64) -> Vec<Vec3> {
        (0..h * w)
            .map(|i| {
                let (u, v) = ((i % w) as f64, (i / w) as f64);
                [
                    (u + 0.5 - w as f64 / 2.0) / f,
                    (v + 0.5 + dv - h as f64 / 2.0) / f,
                    1.0,
                ]
            })
            .collect()
    }

    fn free_space(params: ScanParams) -> FreeSpace {
        let (w, h, f) = (320, 192, 205.0);
        FreeSpace::new(pinhole(w, h, f, 0.0), pinhole(w, h, f, 0.5), (h, w), params).unwrap()
    }

    fn blobs(shape: (usize, usize), min_blob_px: usize, obst: &[usize]) -> Vec<bool> {
        let px = shape.0 * shape.1;
        let params = ScanParams {
            min_blob_px,
            ..Default::default()
        };
        let mut fs = FreeSpace::new(vec![[0.0; 3]; px], vec![[0.0; 3]; px], shape, params).unwrap();
        let mut floor = vec![true; px];
        for &i in obst {
            floor[i] = false;
        }
        fs.obstacles(&floor).unwrap().to_vec()
    }

    #[test]
    fn sectors_and_bearings() {
        let fs = free_space(ScanParams::default());
        assert_eq!(fs.n(), 180);
        assert_eq!(fs.bearing_deg()[0], -179.0);
        assert_eq!(fs.bearing_deg()[90], 1.0);
        assert_eq!(fs.sector(1.0, 0.0), 90); // forward: first sector on the left
        assert_eq!(fs.sector(1.0, -1e-6), 89);
        assert_eq!(fs.sector(0.0, 1.0), 135); // left
        assert_eq!(fs.sector(0.0, -1.0), 45); // right
        assert_eq!(fs.sector(-1.0, 0.0), 0); // straight back, +180 deg: wraps to sector 0
        assert_eq!(fs.sector(-1.0, -0.0), 0); // -180 deg
    }

    #[test]
    fn shape_mismatch_is_an_error() {
        assert!(FreeSpace::new(
            vec![[0.0; 3]; 5],
            vec![[0.0; 3]; 6],
            (2, 3),
            ScanParams::default()
        )
        .is_err());
        let mut fs = free_space(ScanParams::default());
        assert!(fs.scan(&[true; 10], &up_from_pitch(20.0), 0.25).is_err());
    }

    #[test]
    fn blobs_are_8_connected_and_filtered_by_area() {
        // 4x5 mask: a diagonal of 3 pixels (0, 6, 12) and a single pixel (4)
        let diag = [0, 6, 12];
        let keep3 = blobs((4, 5), 3, &[0, 6, 12, 4]);
        for (i, &keep) in keep3.iter().enumerate() {
            assert_eq!(keep, diag.contains(&i), "pixel {i}");
        }
        assert!(blobs((4, 5), 4, &[0, 6, 12, 4]).iter().all(|&k| !k));
        // min_blob_px <= 1: no filtering
        assert_eq!(blobs((4, 5), 1, &[4]).iter().filter(|&&k| k).count(), 1);
    }

    #[test]
    fn u_shape_merges_into_one_blob() {
        // two vertical bars joined at the bottom: the raster pass labels them apart, then merges them
        // . . . . .
        // X . . . X
        // X . . . X
        // X X X X X
        let u = [5, 9, 10, 14, 15, 16, 17, 18, 19];
        let keep = blobs((4, 5), 9, &u);
        assert_eq!(keep.iter().filter(|&&k| k).count(), 9);
        assert!(blobs((4, 5), 10, &u).iter().all(|&k| !k));
    }

    #[test]
    fn open_floor_is_free_to_max_range() {
        let mut fs = free_space(ScanParams::default());
        let sc = fs
            .scan(&vec![true; 192 * 320], &up_from_pitch(20.0), 0.25)
            .unwrap();
        let free = sc.free();
        #[allow(clippy::needless_range_loop)] // k indexes five parallel arrays
        for k in 0..sc.len() {
            assert!(sc.obstacle[k].is_infinite() && !sc.near[k]);
            assert_eq!(sc.seen[k], sc.closest[k].is_finite());
            assert_eq!(free[k].is_nan(), !sc.seen[k]);
            if sc.bearing_deg[k].abs() < 30.0 {
                assert!(sc.seen[k] && free[k] == 2.5);
            }
        }
    }

    #[test]
    fn wall_ahead_is_at_its_distance() {
        // floor up to a wall 1.2 m ahead: a pixel is floor iff its ray lands at x < 1.2
        let (pitch, h, d) = (20.0, 0.25, 1.2);
        let mut fs = free_space(ScanParams::default());
        let pts = crate::geometry::rays_to_floor(&fs.rays, &up_from_pitch(pitch), h);
        let floor: Vec<bool> = pts.iter().map(|p| p[0] < d).collect();
        let sc = fs.scan(&floor, &up_from_pitch(pitch), h).unwrap();
        let ahead = [89, 90]; // +-2 deg
        for k in ahead {
            assert!((sc.obstacle[k] - d).abs() < 0.02, "{}", sc.obstacle[k]);
            assert!(!sc.near[k] && sc.floor[k] <= sc.obstacle[k]);
        }
    }

    #[test]
    fn obstacle_at_the_image_bottom_is_near() {
        let mut fs = free_space(ScanParams::default());
        let sc = fs
            .scan(&vec![false; 192 * 320], &up_from_pitch(20.0), 0.25)
            .unwrap();
        for k in 0..sc.len() {
            // everything is obstacle: where the camera sees the floor plane, the obstacle starts at the
            // closest visible point (lower pixel edge: slightly nearer than the pixel centre)
            assert_eq!(sc.near[k], sc.seen[k]);
            assert_eq!(sc.floor[k], 0.0);
            if sc.seen[k] {
                assert!(sc.obstacle[k] <= sc.closest[k]);
            }
        }
    }

    #[test]
    fn camera_straight_down_sees_nothing() {
        // no forward direction: NaN frame
        let mut fs = free_space(ScanParams::default());
        let sc = fs.scan(&vec![false; 192 * 320], &[0.0, 0.0, -1.0], 0.5).unwrap();
        assert!(sc.seen.iter().all(|&s| !s) && sc.obstacle.iter().all(|o| o.is_infinite()));
        assert!(sc.closest.iter().all(|c| c.is_infinite()) && sc.free().iter().all(|f| f.is_nan()));
    }

    #[test]
    fn drive_mask_and_seg_probs() {
        let (pf, pu) = ([0.9, 0.9, 0.5], [0.1, 0.7, 0.1]);
        assert_eq!(drive_mask(&pf, Some(&pu), 0.8, Some(0.5)), [true, false, false]);
        assert_eq!(drive_mask(&pf, Some(&pu), 0.8, None), [true, true, false]);
        assert_eq!(drive_mask(&pf, None, 0.8, Some(0.5)), [true, true, false]);
        // 2 pixels; the under logit (channel 2) must not leak into the floor probability
        let (p2, u2) = seg_probs(&[0.0, 1.0, 0.0, -1.0], 2);
        let (p3, u3) = seg_probs(&[0.0, 1.0, 0.0, -1.0, 10.0, 10.0], 2);
        assert!(u2.is_none() && p2 == p3);
        assert!((p2[0] - 0.5).abs() < 1e-6 && (p2[1] - 1.0 / (1.0 + 2f32.exp())).abs() < 1e-6);
        assert!((u3.unwrap()[0] - 1.0 / (1.0 + (-10f32).exp())).abs() < 1e-6);
    }
}
