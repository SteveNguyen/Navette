# SLAM for Navette: state of the art and proposal

Status: research report (2026-11-20), for a decision. Nothing is built yet. "Measured" numbers are ours (robot
runs); "[src]" items are from the cited sources; the rest is inference, marked as such where it matters.

## 1. Problem and constraints

Goal: a pose in a `map` frame that does not drift over a home-sized run, so that the occupancy grid stays
consistent (no ghost walls when the robot comes back to a room), and later, a map that can be saved and reused.

What we have (Reachy Mini on the Poussah base):

| Input | Quality |
|---|---|
| Wheel odometry + base gyro heading | good short-term; heading drift measured ~30 deg per 2 m without the gyro, much less with it |
| Head IMU, 400 Hz | gravity 'up' for the camera; gyro usable for rotation prediction |
| One camera, 1280x720 @30 fps (processed ~8 Hz), ~70 deg useful HFOV, 0.27 m high, pitched 15-19 deg down | a large part of the image is floor; known height and pitch give metric scale on the floor plane |
| Virtual laser scan (floor segmentation, ~70 deg, ~0.34-2.5 m) | noisy: doorsteps, shiny tiles, furniture feet |
| Behaviour | 0.15 m/s max, frequent spins in place (it turns to look) |

Compute (measured, CM4, two explore runs): loop 7.5 Hz; per step: floor model 95 ms (2 threads), scan 10 ms,
navigator < 1 ms (p95 3 ms), IMU 9 ms, frame save 6 ms; load average ~3.5 of 4 cores. **A SLAM gets well under
one core**, so it has to work at keyframe rate (~1 Hz or per spin), not per frame.

What Navette already provides (docs/PLAN.md, "Localization and SLAM"):
- one pose source behind an interface (`runtime/localization.py`); REP 105 frames `odom` and `map`, today
  `map = odom`. A SLAM outputs the `map -> odom` correction; the collision monitor works in the robot frame and
  does not care about pose jumps;
- the occupancy grid keeps its keyframes (pose, scan, weights) and rebuilds from corrected poses (`replay()`):
  a loop closure is "here are the corrected keyframe poses";
- runs record frames, IMU, odometry and steps: everything can be developed and evaluated **offline**.

## 2. State of the art, by family

### A. 2D lidar-style SLAM on the virtual scan (scan matching + pose graph)
- Systems: Cartographer 2D, slam_toolbox (Karto/SPA), GMapping, Hector, CSM/PL-ICP; light non-ROS: iris_lama
  (C++, Eigen only, runs on a Pi 3B+ [src]), BreezySLAM (no loop closure). slam_toolbox peaks ~450 % CPU vs
  ~15 % for LaMa on the same bag [src].
- Closest evidence for our sensor: Kinect-as-laser (57 deg, 0.6-6 m): GMapping and Hector **failed in a
  featureless room**, misaligned scans even with good odometry, shrank long walls; "not a direct replacement" for
  a laser [src, Sensors 2014]. Camera floor-segmentation SLAM papers mostly use **360 deg** cameras [src].
- Sparse-sensing work: pool several scans with odometry into a "multiscan" (Beevers), match those; with line
  features + pose graph + Manhattan prior, SoMaSLAM reaches 0.13 m / 2.7 deg ATE on Intel Lab with 4 beams [src].
- Assessment: **per-scan matching on our 70 deg, noisy scan would likely make things worse than odometry**.
  Matching only accumulated 360 deg views (our spins) is the viable form.

### B. Monocular visual / visual-inertial (-wheel) odometry and SLAM
- Measured on ARM [src]: VINS-Mono on a CM4 ~20 FPS VGA, ~150 ms image-to-pose, uses ~1-2 cores; HybVIO on a
  Pi 4 33 ms/frame (no SLAM); ORB-SLAM2 on a Pi 3B+ ~10x slower than a laptop; OpenVINS on a Pi 4: user report
  of divergence. Learned features: SuperPoint/LightGlue ~1 s per pair on ARM CPUs; XFeat ~2 FPS on an A53.
- Ground-robot issues [src]: scale and roll/pitch unobservable for VIO on planar motion without wheels ("VINS
  on Wheels"); ORB-SLAM3 failed to initialise on ground-robot sequences; pure rotation breaks monocular
  triangulation (our spins); rolling shutter modelled only by VINS-Mono.
- Licences: ORB-SLAM3, OpenVINS, VINS, DSO, SVO Pro, HybVIO are **GPL-3**; permissive: RTAB-Map, stella_vslam
  (mono only, no IMU/wheel), Basalt/Granite (no IMU/wheel in mono). Navette is Apache-2.0.
- Assessment: off-the-shelf mono VIO/SLAM is CPU-marginal on our budget, fragile on spins and a floor-dominated
  view, and the systems that fit the problem best are GPL. Not a good backbone here.

### C. Consumer precedent: odometry first, vision corrects
- iRobot vSLAM (Karlsson 2005) / Roomba 980: one cheap camera, landmarks = corners/edges, particle filter +
  per-landmark Kalman; iRobot added an optical floor-tracking sensor because the camera alone was not reliable;
  custom DSPs [src]. Dyson 360 Eye: panoramic camera on top, MonoSLAM lineage [src]. Ceiling-camera SLAM [src].
- All use landmarks **above** the floor and lean on odometry. Our low, pitched-down camera sees mostly floor:
  appearance-based place recognition will be weak (inference).

### Loop closure, back end, robustness
- Place recognition on CPU: DBoW2/fbow (ORB bag of words, OpenCV, an offline vocabulary), iBoW-LCD (online
  vocabulary, 19-432 ms/image on a laptop core), HBST; CNN global descriptors (NetVLAD, MixVPR, CosPlace) cost
  0.3-2 s+/frame on a busy CM4 (inference from Pi 3 / embedded numbers).
- Recommended pattern: gate loop-closure candidates by **pose uncertainty** (only keyframes within a few metres),
  verify geometrically by matching scans/submaps (metric SE(2), no new dependency); appearance retrieval only if
  "kidnapped" relocalisation becomes a requirement.
- Back end: g2o / GTSAM / Ceres are heavy; Rust crates (factrs, apex-solver) pull faer/nalgebra. A custom SE(2)
  Gauss-Newton / Levenberg-Marquardt pose graph (Grisetti et al. 2010 tutorial; Konolige SPA 2010) is ~200-300
  lines and milliseconds for < 1000 keyframes (inference). Robustness to false loops: Dynamic Covariance Scaling
  (closed-form reweighting), switchable constraints, max-mixtures [src].

## 3. Proposal

**Odometry-primary SLAM with spin submaps** (family A done the robust way, in the spirit of C):

1. **Backbone:** wheel odometry + base gyro, as today (the `odom` frame).
2. **Submaps:** each spin in place (already in the behaviour) yields a 360 deg local grid around a keyframe
   pose; between spins, odometry chains the keyframes. Optionally also a keyframe every N m of driving
   (multiscan over the last metres).
3. **Constraints:** correlative grid matching of a new submap against the map / earlier submaps **within the
   pose uncertainty**; accepted only with a high score and a well-conditioned match (rejects corridors, which
   stay odometry-only along their axis).
4. **Back end:** custom SE(2) pose graph in the Rust core (no dependencies, like the rest), Levenberg-Marquardt
   with Dynamic Covariance Scaling; runs at keyframe rate; outputs corrected keyframe poses -> `map -> odom`
   correction and grid `replay()`.
5. **Later, optional:** visual corrections (floor-plane homography with gyro-predicted KLT for short-term
   odometry; keyframe appearance for relocalisation), map save / load.

Why: fits the CPU (work only at keyframes), keeps Apache-2.0 and the no-dependency core, uses the spins the
robot does anyway to get around the narrow field of view, and plugs into the seams already in place.

Risks: noisy segmentation edges in the submaps (doorsteps, shiny floors) bias matching; featureless or
symmetric rooms give ambiguous matches (gating and robust kernels mitigate, but some loops will be missed);
corridors remain odometry-only.

## 4. Evaluation (before the robot ever uses it)

- **Offline first**: run the SLAM on recorded runs (and in simulation with ground truth) and compare odometry
  only vs odometry + pose graph.
- **Ground truth in a home without motion capture**: 4-8 AprilTags lying on the floor at tape-measured positions
  (+-1 cm; the pitched-down camera sees floor tags well), one at the start for return-to-start error; ATE / RPE
  with `evo` at tag sightings; tags kept out of the SLAM front end. Also wall-to-wall distances vs tape, and
  grid agreement across repeated runs of the same loop.
- Measure drift **with loop closure disabled** too, so the gain is attributable.

## 5. Decisions to make

1. Approach: the proposal above (custom, odometry-primary, spin submaps), vs trying an existing system first
   (iris_lama, Cartographer core, RTAB-Map with our odometry) as a baseline.
2. Where it runs at first: offline / laptop only until it is shown to help, then on the CM4.
3. Ground truth: AprilTags on the floor (print, tape-measure positions) for the evaluation runs.
4. Scope of version 1: drift correction within one run; map save / relocalisation later.

## 6. Future: 360 deg looks with the Mini's body yaw (not started)

Idea: for a full look (the SLAM's submaps, the explorer's final look), turn the Mini's body on its yaw joint
instead of spinning the base: the camera's heading then comes from a motor encoder instead of wheel odometry
during a spin (wheels slip in place), and the base only balances, so the frames of one look share almost the
same base position.

Facts (sources: the Reachy Mini SDK, local clone at commit 25756e7042, 2026-11-04; Navette main):
- Body yaw range: +-160 deg (`kinematics/analytical_kinematics.py`, `max_body_yaw`; the placo kinematics use
  +-2.8 rad), and the head can yaw +-65 deg relative to the body (`max_relative_yaw`; 55 deg in the placo
  kinematics). Commanded with `goto_target(..., body_yaw=...)` or `set_target_body_yaw`; the body motor is a
  Feetech servo in position mode (`body_rotation`, id 10, P = 200).
- The head pose the SDK reports is the forward kinematics of all head joints, body yaw included (joint 0), in
  the robot's root frame. Navette's camera pose is that head pose times the hand-eye transform
  (`T_base_cam`, `robots/reachy_poussah/robot.py`), so **a yawed body is already measured per frame**.
- The runtime already folds the camera's heading into the frame pose (`runtime/loop.py`: pose heading = base
  heading + camera heading from `T_base_cam`) and already has a map-only path for frames taken while the head is
  away from the driving pose: the head glance (`runtime/glance.py`, `Navigator.observe`: into the map only, no
  behaviour step, base holding still).
- Navette today always commands body yaw 0 (`set_nav_head`, `ReachyHead`); the explorer's final look and the
  `slam` branch's `look360_m` turn the base (the final look on main integrates the commanded spin rate).

Coverage (inference): with a ~70 deg useful horizontal field of view, body yaw alone covers headings within
+-195 deg of forward, i.e. the full circle with ~30 deg of overlap behind; adding the head's +-65 deg gives
margin. Not measured: the body's speed (Feetech, position mode; the move duration is ours to set), so the time
of a full look (estimate: 6-10 views of ~0.5-1 s each, settle included).

Expected gain vs a base spin (inference, to measure): heading of each view from an encoder (0.088 deg steps)
instead of integrated wheel odometry during a turn in place; no base translation during the turn except the
balance reaction; no stop-start of a spin. Possible cost: with the head pitched 15 deg down, the head's centre
of mass sits off the yaw axis, so turning the body shifts it and the balancing base rolls to compensate (the
25 deg pitch glance rolled it 5 cm). The odometry measures that motion, and each frame keeps its own pose.

Integration sketch:
- A "look" sequence like the glance: hold still, body to a list of yaw stops (e.g. -150, -90, -30, +30, +90,
  +150 deg, head yaw relative 0), settle, n frames into the map with `observe`, body back to 0, check it is
  back. Frames during the moves are not used.
- The explorer's look (`look360_m` on the `slam` branch, the final look on main) and the SLAM front end's
  submap trigger use it instead of a base spin, behind a config flag (default: base spin, so goldens stay the
  same).
- Navigator steps (driving) only with the body back at 0: the navigator takes straight ahead in the image for
  the driving direction (`camera_mount_warning`, 10 deg limit).
- The SLAM front end gets a submap per body look directly (same base pose for all views).

Robot test protocol (each step needs a go):
1. Base off, Mini on the base: body yaw sweep -160..+160 deg at a few speeds; log the commanded vs measured yaw,
   settle time, and whether the base or a cable limits the range.
2. Base balancing, standing still: the same sweep with the head at the navigation pitch; log base odometry
   (translation, heading) and the tilt during the sweep; check the floor model sees no part of the robot.
3. One look sequence at a few spots, recorded; offline: build the submap from the body look and from a base
   spin at the same spot, compare their sharpness and their match scores against each other.

## Sources

Main references (full lists in the research notes): Sensors 2014 Kinect-as-laser (PMC4299068); iris_lama
(github.com/iris-ua/iris_lama); SoMaSLAM (arXiv 2409.15736); EG-SLAM (arXiv 2312.02353); Beevers multiscan
(RPI thesis); Karlsson et al. ICRA 2005 (vSLAM); IEEE Spectrum on Roomba 980 and Dyson 360 Eye; VINS-Mono on CM4
(arXiv 2406.13345); HybVIO (arXiv 2106.11857); "VINS on Wheels" (Wu et al. ICRA 2017); ground-robot VIO
initialisation (arXiv 2406.05969); XFeat (arXiv 2404.19174); iBoW-LCD (arXiv 1802.05909); HBST (arXiv
1802.09261); Grisetti et al. graph-SLAM tutorial (2010); Konolige et al. SPA (IROS 2010); Agarwal et al. DCS
(ICRA 2013); Sünderhauf switchable constraints (IROS 2012); Kümmerle et al. SLAM accuracy (AuRo 2009); evo
(github.com/MichaelGrupp/evo).
