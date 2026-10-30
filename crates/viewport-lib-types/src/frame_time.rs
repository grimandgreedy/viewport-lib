//! The time a frame is rendered at.

/// The clock a frame renders against: elapsed seconds, the step since the
/// previous frame and a frame index.
///
/// Set it on `FrameData::time` to pin what time-driven shaders see (the camera
/// uniform's `time` and the deform header's `time_seconds`). Two renders of the
/// same frame at the same `FrameTime` are byte-identical. Without one the
/// renderer reads its own monotonic clock, started when it was created.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FrameTime {
    /// Elapsed seconds.
    pub seconds: f64,
    /// Seconds since the previous frame.
    pub delta: f32,
    /// Frame index.
    pub frame_index: u64,
}

impl FrameTime {
    /// A frame time from its three parts.
    pub fn new(seconds: f64, delta: f32, frame_index: u64) -> Self {
        Self {
            seconds,
            delta,
            frame_index,
        }
    }

    /// A frame frozen at `seconds`: zero delta, frame index zero. For
    /// reproducible renders such as golden images and offscreen captures.
    pub fn fixed(seconds: f64) -> Self {
        Self::new(seconds, 0.0, 0)
    }
}

impl Default for FrameTime {
    fn default() -> Self {
        Self::fixed(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_has_zero_delta_and_index() {
        let t = FrameTime::fixed(2.5);
        assert_eq!(t, FrameTime::new(2.5, 0.0, 0));
        assert_eq!(FrameTime::default(), FrameTime::fixed(0.0));
    }
}
