# Floor-mask flicker (study of 2026-11-20)

Exploratory study: how much the floor mask changes between consecutive frames, where, and whether a temporal
input or a temporal filter would help. Four recorded explorations of the reference robot (2026-11-19 and 20),
the floor model of the time (dev v7), previous mask warped onto the current frame through the floor plane with
the odometry pose. Recorded runs are private: none is in the repository.

## Findings

- **Flicker away from mask edges** (a band along the edges is excluded: edges move a few pixels with the pose
  error): 0.59 % of floor pixels per frame pair (median 0.07 %). Standing / driving / turning in place: 0.22 /
  0.60 / 0.73 %. Concentrated: the worst 5 % of pairs hold half of it; the final episode of one run, stuck at a
  doorway, alone 30 %.
- **Where:** the same few classes the model gets wrong anyway, bi-stable from one frame to the next: a doorway
  threshold and the next room's (different) tiles, rugs and doormats, shadows under or behind objects, the
  bottom of a wall at close range called floor, sunlit tiles at the image bottom, motion blur while turning.
- **Fusion of masks:** averaging the last 2-3 motion-compensated masks cuts the flicker 3-4 times, but a region
  where the model is at a coin flip stays wrong, only steadier; the strongest variant delays every real edge
  by one frame (~0.13 s).
- **Floor in two frames** (floor only if floor in this frame and the previous one): removes every one-frame
  "phantom floor" and never delays an obstacle; free space appears one frame later. It does not unblock the
  doorway: a region the model hesitates on counts as floor only when two frames agree, so it looks blocked
  more often. Implemented as an option, off: `[perception] floor_two_frames` (`docs/PLAN.md`).
- **A temporal model input** (frame t-1 fed to the network) is not worth it now: the errors are systematic
  classes, better fixed with data and augmentation; the synthetic training data has no frame pairs. One case
  it could help later: motion blur (the previous frame is often sharp).

## Geometry found on the way

- **The pose was read ~40 ms late.** The odometry was read when a frame arrived, one camera latency (38 ms)
  after its capture: ~1 deg of heading at the base's turn rate, on every scan put in the map. Consecutive
  frames aligned best on the floor with the poses shifted ~40 ms back. Fixed by `[perception]
  pose_at_capture` (on for the reference robot).
- **The IMU 'up' filter lags the base's fast pitch wobble** (the balancing base rocks): it follows about half
  of it, which leaves 1-2 px of misalignment, a range error growing with the square of the distance. Not
  fixed: future work (filter tuning, more weight on the gyro, timestamps).

## Measuring the two switches

`tools/flicker/compare_replay.py` replays recorded runs open loop (the robot's own steps) with each switch on
its own, the floor masks computed once and shared:

- pose at the capture time (variant B vs A): normalised cross-correlation of each frame with the previous one
  warped through the floor plane (the study's method), while turning and while driving; and the number of
  occupied cells of the final maps (a smeared map has thicker walls);
- floor in two frames (C vs B): sectors it shortens per step, and steps where it took the forward motion away.

A 100-frame smoke test (one run, turns and drives): B aligned better than A on 71 % of the turning pairs and
63 % of the driving pairs. Floor in two frames shortened a sector on 92 of 101 steps (6 sectors per step on
average) without taking any forward motion away: more often than the pixel flicker suggests, so check what it
shortens (one-frame phantoms, or sector geometry on oblique walls: a 5 cm tolerance) before turning it on.
The full comparison takes ~0.09 s per frame on a laptop CPU.
