//! Exploration: the frontier planner, and the behaviour that follows its paths with the arc controller.

pub mod frontier;
pub mod frontier_explore;

/// For the tests: a 6 x 6 m grid, origin at its centre. Known free room x, y in [-1, 1] with walls 0.1 m
/// thick; a door of width `door_w` in the wall at x = +1 leads to unknown space.
#[cfg(test)]
pub(crate) fn test_room(door_w: f64) -> crate::occupancy_grid::OccupancyGrid {
    use crate::occupancy_grid::{OccupancyGrid, OccupancyGridParams};

    let mut g = OccupancyGrid::new(OccupancyGridParams {
        size_m: 6.0,
        ..Default::default()
    });
    let n = g.n();
    let l: Vec<f32> = (0..n * n)
        .map(|c| {
            // cell centre
            let (x, y) = ((c / n) as f64 * 0.05 - 2.975, (c % n) as f64 * 0.05 - 2.975);
            let inside = x.abs() < 1.0 && y.abs() < 1.0;
            let door = x > 0.95 && y.abs() < door_w / 2.0;
            if inside {
                -2.0
            } else if x.abs() < 1.1 && y.abs() < 1.1 && !door {
                2.0
            } else {
                0.0
            }
        })
        .collect();
    g.set_map([-3.0, -3.0], &l);
    g
}
