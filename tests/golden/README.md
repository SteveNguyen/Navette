# Golden files

Reference outputs of the Python prototype on seeded synthetic inputs, one set of files per module
(`<module>_*.npz`). Each file holds the inputs, the parameters and the prototype's outputs; the golden tests
(`tests/test_<module>_golden.py`, or `tests/test_<module>.py` for the small modules) replay the inputs through
navette and compare. Synthetic data only.

Tolerances, at most:

| Values | Tolerance |
|--------|-----------|
| booleans, indices, sectors, states, inf / nan pattern | exact |
| float64 | 1e-6 |
| values that go through float32 | 1e-4 at most; scan ranges: 1e-6 (bit-identical on x86-64 / glibc) |

Each test asserts the tightest tolerance its module allows (last column). "exact" is `==`: the same operations
in the same order. sin / cos / atan2 / hypot / exp need not be bit-identical between maths libraries: some
tests allow 1e-12 for it, the others (the planner, the controller, the sequences) would show a difference
there as a failure. On the machine that wrote the files (x86-64, glibc) every match is exact.

| Files | Content | Asserted |
|-------|---------|----------|
| `scan_real_lens.npz` | FreeSpace: Reachy Mini lens, 320x192 masks, default parameters, 36 cases | booleans exact; ranges 1e-6 |
| `scan_pinhole_s{1,3,7}.npz` | FreeSpace: ideal lens, small masks, sector 1 / 3 / 7 deg and other parameters, 15 cases each | booleans exact; ranges 1e-6 |
| `occupancy_grid_{room,room_weighted,leaves_bounds,random_scans}.npz` | OccupancyGrid: update sequences (ray-cast room scans, weighted by a tilt error, leaving the grid, random scans) with the grid, masks, blind_known and obstacle_points at stages, and after a pose correction + replay | keyframe decisions, float32 log-odds, masks, blind_known, cells: exact; obstacle points: 1e-9 |
| `collision_monitor_{default,live,other}.npz` | Collision monitor, one file per SafetyParams (default, the robot's live values, an unusual set): 300 random scans x requests x optional map points and near_known, with every function's output | allowed speed (the candidate): exact; free arc lengths, boundary points, arcs, stopping distances: relative 1e-12 |
| `velocity_smoother_gate.npz` | ResumeGate on random request / allowed-speed sequences | exact (output, stopped, clear count) |
| `attitude_streams.npz` | UpFilter on synthetic IMU streams (rest, slow tilt, nod, vibration, gyro bias, rate gate, irregular timestamps, impacts): 'up', bias and filtered rate after every sample, up_at queries | 1e-12 |
| `tilt_trust_motion.npz` | TiltTrust on random base motions, several parameter sets: sigma, motion term, trusted range after every step | relative 1e-12 |
| `frontier_fields.npz` | Planner building blocks on random and structured grids: frontier_cells, the distance transform (three shapes, with and without obstacles), Dijkstra cost and predecessor fields (planner costs, uniform costs, random costs) | exact, the predecessors among equal-cost paths included |
| `frontier_planner.npz` | FrontierPlanner call sequences (plan with image-bottom sectors, ban, heading) on hand-built rooms and random maps, 8 parameter sets, 881 calls | done, target, path, frontier cells, bans: exact; heading: 1e-12 |
| `dwa_arcs.npz` | ArcController toward given goal headings: 60 sequences of random scans over maps with random obstacles, several ExploreParams and SafetyParams, 1839 steps | exact (the arc, the state, the blocked timer) |
| `navigator_{open,doorway,clutter,wall,small}_{default,live}.npz` | Golden sequences, exploration: the prototype's whole stack driving the 2-D simulator (`tests/sim2d.py`) closed-loop in five rooms, with the default SafetyParams and the robot's live ones (latency 1.53 s, decel 0.64 m/s^2), range noise 1 cm at 1 m. Every input of every step and its output; the planner's target, path length and frontier size; the final grid; what the prototype reached (free cells, finish step, closest approach of a wall). `open`, `wall` (start facing a wall) and `small` run until exploration finishes; `doorway` (corridor + doorway) and `clutter` (four boxes) for 1300 steps | every step exact: v, w, v_req, w_req, state, finished, blind_stop, target, path length, frontier size; final grid bit for bit |
| `navigator_forward_{box_default,box_live,fast_live,no_odometry,side_box_glance}.npz` | Golden sequences, forward behaviour: toward a box 2.2 m ahead (0.15 m/s with both SafetyParams, 0.3 m/s with the live ones, and without odometry: no map), and past a box in the blind zone at the start (a blind stop, then the frames of a head glance through `observe`) | as above |

`tests/test_closed_loop.py` reads the `navigator_*` files for their room, seed and parameters and for the
prototype's numbers (explored cells, finish step, stop distance), and runs navette itself in the loop.

The files come from generators kept with the prototype (one `gen_<module>.py` per row; the simulator there is
a copy of `tests/sim2d.py`). Regenerate only when the reference behaviour changes on purpose.
