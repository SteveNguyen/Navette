//! Resume gate, after the collision monitor: stop at once, resume only after n consecutive steps allowing
//! forward motion (anti-chatter).
//!
//! On the robot, one noisy frame flipped v between 0 and the request 78 times in 90 s. Stops are never
//! delayed; the base's own acceleration ramp does the smoothing (Nav2: velocity smoother before the collision
//! monitor, so stops are immediate and starts are ramped).

/// See the module documentation. One gate per command stream; call [`ResumeGate::step`] once per frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResumeGate {
    /// Consecutive steps allowing forward motion needed to resume after a stop (0 or 1: no gate).
    pub n: u32,
    /// The collision monitor cut a forward request to 0 and motion has not resumed since.
    stopped: bool,
    /// Steps allowing forward motion since the stop.
    clear: u32,
}

impl Default for ResumeGate {
    fn default() -> Self {
        Self::new(2)
    }
}

impl ResumeGate {
    pub fn new(n: u32) -> Self {
        Self {
            n,
            stopped: false,
            clear: 0,
        }
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    pub fn clear(&self) -> u32 {
        self.clear
    }

    /// v allowed after the gate, for the requested `v_req` and the collision monitor's `v_ok`.
    /// A stop = the monitor cut a forward request to 0.
    pub fn step(&mut self, v_req: f64, v_ok: f64) -> f64 {
        if v_req > 0.0 && v_ok <= 0.0 {
            self.stopped = true;
            self.clear = 0;
            return v_ok;
        }
        if self.stopped && v_ok > 0.0 {
            self.clear += 1;
            if self.clear < self.n {
                return 0.0;
            }
        }
        // resumed, or no forward motion asked for (turning in place, reverse): nothing to hold back
        self.stopped = false;
        v_ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_at_once_and_resumes_after_n_clear_steps() {
        let mut g = ResumeGate::new(3);
        assert_eq!(g.step(0.2, 0.2), 0.2); // never stopped: passes
        assert_eq!(g.step(0.2, 0.0), 0.0); // stop: immediate
        assert!(g.stopped());
        assert_eq!(g.step(0.2, 0.2), 0.0); // 1 clear step
        assert_eq!(g.step(0.2, 0.1), 0.0); // 2
        assert_eq!(g.step(0.2, 0.15), 0.15); // 3: resumes at the allowed speed
        assert!(!g.stopped());
        assert_eq!(g.step(0.2, 0.2), 0.2);
    }

    #[test]
    fn a_blocked_step_restarts_the_count() {
        let mut g = ResumeGate::default();
        assert_eq!(g.step(0.2, 0.0), 0.0);
        assert_eq!(g.step(0.2, 0.2), 0.0);
        assert_eq!(g.step(0.2, 0.0), 0.0); // blocked again
        assert_eq!((g.stopped(), g.clear()), (true, 0));
        assert_eq!(g.step(0.2, 0.2), 0.0);
        assert_eq!(g.step(0.2, 0.2), 0.2);
    }

    #[test]
    fn a_step_without_forward_request_releases_the_gate() {
        // as the prototype: turning in place (v_req = 0) after a stop clears it
        let mut g = ResumeGate::default();
        assert_eq!(g.step(0.2, 0.0), 0.0);
        assert_eq!(g.step(0.0, 0.0), 0.0);
        assert!(!g.stopped());
        assert_eq!(g.step(0.2, 0.2), 0.2);
    }

    #[test]
    fn n_1_is_no_gate() {
        for n in [0, 1] {
            let mut g = ResumeGate::new(n);
            assert_eq!(g.step(0.2, 0.0), 0.0);
            assert_eq!(g.step(0.2, 0.2), 0.2);
        }
    }
}
