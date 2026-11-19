//! Local controllers: from a goal to a requested base command (v, w). The request still goes through the
//! collision monitor: a controller only proposes.

pub mod dwa;

/// What a behaviour is doing (for display, logs, and the navigator's 'finished' test).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// Following arcs toward the goal heading.
    Drive,
    /// Turning in place toward a goal heading far off the current one.
    Spin,
    /// Exploration: no frontier left, turning once on the spot to look around.
    Look,
    /// Exploration: looked around, still no frontier. Standing still.
    Done,
    /// The constant straight request of the 'forward' behaviour.
    Forward,
}

impl State {
    /// The state's name as the Python API gives it.
    pub fn as_str(self) -> &'static str {
        match self {
            State::Drive => "DRIVE",
            State::Spin => "SPIN",
            State::Look => "LOOK",
            State::Done => "DONE",
            State::Forward => "FORWARD",
        }
    }
}
