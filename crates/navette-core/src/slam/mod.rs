//! SLAM building blocks (docs/SLAM.md): odometry-primary, with submaps matched at keyframe rate and a small
//! SE(2) pose graph. The front end (when to make submaps, which pairs to match, applying the correction)
//! lives in the Python runtime for now (`navette.slam`); these are the parts that must be fast.

pub mod pose_graph;
pub mod scan_matcher;
