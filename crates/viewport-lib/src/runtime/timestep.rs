//! Fixed timestep accumulator for physics and simulation.
//!
//! Call [`FixedTimestep::advance`] each frame with the wall-clock delta. Iterate
//! the returned [`FixedStepIter`] to consume as many fixed steps as the
//! accumulator has filled. After iteration, [`FixedTimestep::alpha`] gives the
//! blend factor for transform interpolation when rendering between steps.
//!
//! # Example
//!
//! ```rust
//! use viewport_lib::runtime::FixedTimestep;
//!
//! let mut ts = FixedTimestep::new(60.0);
//!
//! let wall_dt = 0.016_f32;
//! for step_dt in ts.advance(wall_dt) {
//!     // run physics for step_dt seconds
//!     let _ = step_dt;
//! }
//! let alpha = ts.alpha(); // use this to lerp between previous and current transforms
//! ```

/// Accumulates wall-clock time and yields fixed simulation steps.
///
/// The step size is `1.0 / hz` seconds. The accumulator drains by `step_dt`
/// for each step yielded. A long frame yields multiple steps; a short frame
/// yields zero. One call to [`advance`](Self::advance) yields at most
/// `max_steps` steps (default [`DEFAULT_MAX_STEPS`]); the simulated time past
/// that is dropped and reported by [`dropped`](Self::dropped), so a hitch does
/// not leave the simulation running ever further behind.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FixedTimestep {
    /// Duration of each fixed step in seconds (`1.0 / hz`).
    pub step_dt: f32,
    accumulator: f32,
    #[cfg_attr(feature = "serde", serde(default = "default_max_steps"))]
    max_steps: u32,
    #[cfg_attr(feature = "serde", serde(default))]
    dropped: f32,
}

/// Steps one [`FixedTimestep::advance`] yields at most unless
/// [`FixedTimestep::with_max_steps`] sets another cap.
pub const DEFAULT_MAX_STEPS: u32 = 8;

#[cfg(feature = "serde")]
fn default_max_steps() -> u32 {
    DEFAULT_MAX_STEPS
}

impl FixedTimestep {
    /// Create a new accumulator that yields steps at `hz` Hz.
    ///
    /// Panics if `hz` is not positive and finite.
    pub fn new(hz: f32) -> Self {
        assert!(hz > 0.0 && hz.is_finite(), "hz must be positive and finite");
        Self {
            step_dt: 1.0 / hz,
            accumulator: 0.0,
            max_steps: DEFAULT_MAX_STEPS,
            dropped: 0.0,
        }
    }

    /// Cap the steps one [`advance`](Self::advance) yields at `max_steps`.
    ///
    /// Panics if `max_steps` is zero.
    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        assert!(max_steps > 0, "max_steps must be at least 1");
        self.max_steps = max_steps;
        self
    }

    /// The most steps one [`advance`](Self::advance) yields.
    pub fn max_steps(&self) -> u32 {
        self.max_steps
    }

    /// Add `dt` seconds to the accumulator and return an iterator over ready steps.
    ///
    /// Each call to `next()` on the returned iterator drains one step from the
    /// accumulator and yields `self.step_dt`. Exhaust the iterator before calling
    /// [`alpha`](Self::alpha) so the blend factor reflects the remaining leftover.
    ///
    /// When more than `max_steps` whole steps are pending, the excess whole
    /// steps are dropped and [`dropped`](Self::dropped) reports their length;
    /// the fraction of a step left over is kept for [`alpha`](Self::alpha).
    pub fn advance(&mut self, dt: f32) -> FixedStepIter<'_> {
        self.accumulator += dt;
        let pending = (self.accumulator / self.step_dt).floor();
        let cap = self.max_steps as f32;
        self.dropped = 0.0;
        if pending > cap {
            self.dropped = (pending - cap) * self.step_dt;
            self.accumulator -= self.dropped;
        }
        let remaining = pending.min(cap).max(0.0) as u32;
        FixedStepIter {
            ts: self,
            remaining,
        }
    }

    /// Simulated seconds the last [`advance`](Self::advance) dropped to stay
    /// within `max_steps`. Zero when nothing was dropped.
    pub fn dropped(&self) -> f32 {
        self.dropped
    }

    /// Blend factor for rendering interpolation: `accumulator / step_dt`.
    ///
    /// Ranges from `0.0` (step just completed) to just under `1.0` (about to
    /// step). Use this to lerp between the previous and current physics transform
    /// when rendering between fixed steps. Clamped to `[0.0, 1.0]`.
    ///
    /// Call this after exhausting the iterator from [`advance`](Self::advance).
    pub fn alpha(&self) -> f32 {
        (self.accumulator / self.step_dt).clamp(0.0, 1.0)
    }

    /// Reset the accumulator to zero without changing the step rate.
    pub fn reset(&mut self) {
        self.accumulator = 0.0;
        self.dropped = 0.0;
    }
}

/// Iterator over pending fixed simulation steps.
///
/// Returned by [`FixedTimestep::advance`]. Each `next()` drains one step from
/// the accumulator and yields the step duration in seconds.
pub struct FixedStepIter<'a> {
    ts: &'a mut FixedTimestep,
    remaining: u32,
}

impl<'a> Iterator for FixedStepIter<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        self.ts.accumulator = (self.ts.accumulator - self.ts.step_dt).max(0.0);
        Some(self.ts.step_dt)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.remaining as usize;
        (n, Some(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_steps_on_short_frame() {
        let mut ts = FixedTimestep::new(60.0);
        let steps: Vec<_> = ts.advance(0.005).collect();
        assert!(steps.is_empty());
    }

    #[test]
    fn test_one_step_on_normal_frame() {
        let mut ts = FixedTimestep::new(60.0);
        let steps: Vec<_> = ts.advance(1.0 / 60.0).collect();
        assert_eq!(steps.len(), 1);
        assert!((steps[0] - 1.0 / 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_steps_on_long_frame() {
        let mut ts = FixedTimestep::new(60.0);
        // Use an exact multiple of step_dt to avoid f32 rounding.
        let steps: Vec<_> = ts.advance(ts.step_dt * 3.0 + 0.0001).collect();
        assert_eq!(steps.len(), 3);
    }

    #[test]
    fn test_alpha_between_zero_and_one_after_step() {
        let mut ts = FixedTimestep::new(60.0);
        let step_dt = 1.0 / 60.0;
        // Advance slightly more than one step so there is leftover.
        ts.advance(step_dt + 0.004).for_each(drop);
        let alpha = ts.alpha();
        assert!(alpha > 0.0 && alpha < 1.0, "alpha was {alpha}");
    }

    #[test]
    fn test_accumulator_carries_over_across_frames() {
        let mut ts = FixedTimestep::new(60.0);
        let step_dt = 1.0 / 60.0;
        // First frame: half a step.
        ts.advance(step_dt * 0.5).for_each(drop);
        // Second frame: another half step. Should now yield one full step.
        let steps: Vec<_> = ts.advance(step_dt * 0.5).collect();
        assert_eq!(steps.len(), 1);
    }

    #[test]
    fn test_reset_clears_accumulator() {
        let mut ts = FixedTimestep::new(60.0);
        ts.advance(0.5).for_each(drop);
        ts.reset();
        assert_eq!(ts.alpha(), 0.0);
        let steps: Vec<_> = ts.advance(0.0).collect();
        assert!(steps.is_empty());
    }

    #[test]
    fn test_steps_capped_on_hitch() {
        let mut ts = FixedTimestep::new(60.0);
        let steps = ts.advance(2.0).count();
        assert_eq!(steps, 8);
        let alpha = ts.alpha();
        assert!((0.0..1.0).contains(&alpha), "alpha was {alpha}");
        // Whole steps past the cap are dropped; the fraction left over stays
        // for alpha, so dropped plus leftover is the excess over 8 steps.
        let excess = 2.0 - 8.0 * ts.step_dt;
        let leftover = alpha * ts.step_dt;
        assert!(
            ts.dropped() > excess - ts.step_dt,
            "dropped was {}",
            ts.dropped()
        );
        assert!(
            (ts.dropped() + leftover - excess).abs() < 1e-4,
            "dropped was {}, leftover {leftover}",
            ts.dropped()
        );
        // The next normal frame steps once and drops nothing.
        assert_eq!(ts.advance(ts.step_dt).count(), 1);
        assert_eq!(ts.dropped(), 0.0);
    }

    #[test]
    fn test_with_max_steps_sets_cap() {
        let mut ts = FixedTimestep::new(60.0).with_max_steps(2);
        assert_eq!(ts.max_steps(), 2);
        assert_eq!(ts.advance(ts.step_dt * 5.0 + 0.0001).count(), 2);
        assert!(ts.dropped() > 0.0);
    }

    #[test]
    fn test_no_drop_below_cap() {
        let mut ts = FixedTimestep::new(60.0);
        assert_eq!(ts.advance(ts.step_dt * 8.0 + 0.0001).count(), 8);
        assert_eq!(ts.dropped(), 0.0);
    }

    #[test]
    #[should_panic]
    fn test_with_max_steps_panics_on_zero() {
        let _ = FixedTimestep::new(60.0).with_max_steps(0);
    }

    #[test]
    #[should_panic]
    fn test_new_panics_on_zero_hz() {
        FixedTimestep::new(0.0);
    }

    #[test]
    #[should_panic]
    fn test_new_panics_on_negative_hz() {
        FixedTimestep::new(-60.0);
    }
}
