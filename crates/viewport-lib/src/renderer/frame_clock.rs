//! The renderer's frame clock: resolves each frame's [`FrameTime`].

use super::types::FrameTime;

/// Monotonic clock started with the renderer, and the time the current frame
/// renders at.
pub(crate) struct FrameClock {
    start: web_time::Instant,
    last_seconds: Option<f64>,
    current: FrameTime,
    /// While set, a frame without its own time renders at `FrameTime::fixed(0.0)`
    /// rather than the clock, so offscreen captures are reproducible.
    fixed_fallback: bool,
}

impl FrameClock {
    pub(crate) fn new() -> Self {
        Self {
            start: web_time::Instant::now(),
            last_seconds: None,
            current: FrameTime::fixed(0.0),
            fixed_fallback: false,
        }
    }

    /// Resolve the time a frame renders at and make it current. An explicit
    /// time wins; otherwise the fixed fallback, then the clock. A frame that
    /// does not advance renderer state (a capture beside a presented frame)
    /// keeps the current time instead of reading the clock.
    pub(crate) fn resolve(
        &mut self,
        explicit: Option<FrameTime>,
        advances: bool,
        frame_counter: u64,
    ) -> FrameTime {
        let t = match explicit {
            Some(t) => t,
            None if self.fixed_fallback => FrameTime::fixed(0.0),
            None if !advances => self.current,
            None => {
                let seconds = self.start.elapsed().as_secs_f64();
                let delta = self.last_seconds.map_or(0.0, |l| (seconds - l) as f32);
                self.last_seconds = Some(seconds);
                FrameTime::new(seconds, delta, frame_counter)
            }
        };
        self.current = t;
        t
    }

    /// The time of the frame last resolved.
    pub(crate) fn current(&self) -> FrameTime {
        self.current
    }

    /// Set the fixed fallback, returning the previous value.
    pub(crate) fn set_fixed_fallback(&mut self, on: bool) -> bool {
        std::mem::replace(&mut self.fixed_fallback, on)
    }
}

/// The camera uniform's `time` word: seconds as f32, the part of the f64
/// seconds that f32 drops, delta, and the frame index modulo 2^24 (exact in
/// an f32).
pub(crate) fn pack_time(t: &FrameTime) -> [f32; 4] {
    let s = t.seconds as f32;
    let remainder = (t.seconds - s as f64) as f32;
    let index = (t.frame_index & 0x00ff_ffff) as f32;
    [s, remainder, t.delta, index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_time_wins_and_fixed_fallback_pins_zero() {
        let mut c = FrameClock::new();
        let t = FrameTime::new(3.0, 0.5, 7);
        assert_eq!(c.resolve(Some(t), true, 1), t);
        c.set_fixed_fallback(true);
        assert_eq!(c.resolve(None, true, 2), FrameTime::fixed(0.0));
        c.set_fixed_fallback(false);
        assert_eq!(c.resolve(Some(t), true, 3), t);
        // A derivative frame keeps the current time.
        assert_eq!(c.resolve(None, false, 4), t);
        let live = c.resolve(None, true, 5);
        assert_eq!(live.frame_index, 5);
        assert!(live.seconds >= 0.0);
    }

    #[test]
    fn pack_time_splits_seconds() {
        let p = pack_time(&FrameTime::new(100_000.123_456_789, 0.016, (1 << 24) + 3));
        assert_eq!(p[3], 3.0);
        assert!(((p[0] as f64 + p[1] as f64) - 100_000.123_456_789).abs() < 1e-9);
    }
}
