/// Animation applied to shape opacity each frame.
///
/// The animation is resolved during `prepare()` using the `time` field on
/// `OverlayFrame`. All `start_time` and `time` values share the same
/// application-defined epoch (e.g. seconds since app launch).
///
/// viewport-lib does not own the event loop. The host application must
/// request continuous repaints while animations are active so that
/// `prepare()` is called often enough to produce smooth updates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverlayAnimation {
    /// No animation; use `opacity` as-is.
    None,
    /// Fade from 0 to `opacity` over `duration` seconds.
    FadeIn {
        /// Absolute time when the fade starts.
        start_time: f64,
        /// Duration of the fade in seconds.
        duration: f32,
    },
    /// Fade from `opacity` to 0 over `duration` seconds.
    FadeOut {
        /// Absolute time when the fade starts.
        start_time: f64,
        /// Duration of the fade in seconds.
        duration: f32,
    },
    /// Oscillate opacity between 0 and `opacity` with a sinusoidal wave.
    Pulse {
        /// Absolute time when the pulse starts.
        start_time: f64,
        /// Period of one full oscillation in seconds.
        period: f32,
    },
}

impl Default for OverlayAnimation {
    fn default() -> Self {
        OverlayAnimation::None
    }
}

/// OverlayEasing curve applied to an [`AnimTrack`]'s normalised parameter `t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OverlayEasing {
    /// Returns `t` unchanged. Constant speed.
    #[default]
    Linear,
    /// `t * t`. Starts slow, accelerates.
    EaseIn,
    /// `1 - (1 - t)^2`. Starts fast, decelerates.
    EaseOut,
    /// Smoothstep: `3t^2 - 2t^3`. Slow start, fast middle, slow end.
    EaseInOut,
    /// Sinusoidal half-wave: `sin(t * PI)`. Returns to 0 at both ends; peaks
    /// at the midpoint. Combine with [`RepeatMode::Loop`] for a continuous
    /// pulse.
    Pulse,
}

/// How an [`AnimTrack`] handles time past the end of its duration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RepeatMode {
    /// Run the track once and hold the final value. Default.
    #[default]
    Once,
    /// Restart the track from `from` each cycle.
    Loop,
    /// Reverse direction at each end so the value oscillates between
    /// `from` and `to`.
    PingPong,
}

/// A single animation track interpolating one channel from `from` to `to`
/// over `duration` seconds, with optional easing and repeat mode.
///
/// Resolved during `prepare()` using `OverlayFrame::time`. Times share the
/// same application-defined epoch as the rest of the overlay animation
/// system. Negative or zero `duration` snaps directly to `to`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimTrack<T: Copy> {
    /// Absolute time at which the track starts.
    pub start_time: f64,
    /// Length of one cycle in seconds.
    pub duration: f32,
    /// Value at `start_time` (or each loop restart).
    pub from: T,
    /// Value at `start_time + duration`.
    pub to: T,
    /// Curve applied to the normalised parameter before interpolation.
    pub easing: OverlayEasing,
    /// What happens past the end of one cycle.
    pub repeat: RepeatMode,
}

impl<T: Copy + Default> Default for AnimTrack<T> {
    fn default() -> Self {
        Self {
            start_time: 0.0,
            duration: 1.0,
            from: T::default(),
            to: T::default(),
            easing: OverlayEasing::Linear,
            repeat: RepeatMode::Once,
        }
    }
}

/// Multi-channel animation tracks attached to an [`OverlayShapeItem`].
///
/// Each `Some` track replaces the matching field on the item for the frame.
/// Tracks are independent: a shape can simultaneously translate, scale,
/// recolour, and rotate.
///
/// Animation resolution is CPU-side in `prepare()`; the host must request
/// continuous repaints while any track is active.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct OverlayAnimations {
    /// Drives the item's overall opacity multiplier. Takes precedence over
    /// the legacy [`OverlayShapeItem::animation`] field when both are set.
    pub opacity: Option<AnimTrack<f32>>,
    /// Drives `position`, the screen-pixel nudge layered on the resolved anchor.
    pub position: Option<AnimTrack<[f32; 2]>>,
    /// Drives `size` (width / height in logical pixels).
    pub size: Option<AnimTrack<[f32; 2]>>,
    /// Drives the item's solid fill colour. Applies only when `fill` is a
    /// solid colour; gradient fills are left alone.
    pub fill: Option<AnimTrack<[f32; 4]>>,
    /// Drives `border_colour`.
    pub border: Option<AnimTrack<[f32; 4]>>,
    /// Drives `rotation` in radians.
    pub rotation: Option<AnimTrack<f32>>,
    /// Arbitrary path channel driving `opacity`. Overrides the linear
    /// `opacity` track and the legacy `animation` field when set.
    pub opacity_path: Option<PathTrack<f32>>,
    /// Arbitrary path channel driving `position`. Overrides the linear
    /// `position` track when set.
    pub position_path: Option<PathTrack<[f32; 2]>>,
    /// Arbitrary path channel driving `size`. Overrides the linear
    /// `size` track when set.
    pub size_path: Option<PathTrack<[f32; 2]>>,
    /// Arbitrary path channel driving the solid fill colour. Overrides
    /// the linear `fill` track when set.
    pub fill_path: Option<PathTrack<[f32; 4]>>,
    /// Arbitrary path channel driving `border_colour`. Overrides the
    /// linear `border` track when set.
    pub border_path: Option<PathTrack<[f32; 4]>>,
    /// Arbitrary path channel driving `rotation`. Overrides the linear
    /// `rotation` track when set.
    pub rotation_path: Option<PathTrack<f32>>,
}

impl OverlayAnimations {
    /// Set the linear `opacity` track.
    pub fn with_opacity(mut self, track: AnimTrack<f32>) -> Self {
        self.opacity = Some(track);
        self
    }

    /// Set the linear `position` track (the screen-pixel nudge on the anchor).
    pub fn with_position(mut self, track: AnimTrack<[f32; 2]>) -> Self {
        self.position = Some(track);
        self
    }

    /// Set the linear `size` track.
    pub fn with_size(mut self, track: AnimTrack<[f32; 2]>) -> Self {
        self.size = Some(track);
        self
    }

    /// Set the linear solid-`fill` colour track.
    pub fn with_fill(mut self, track: AnimTrack<[f32; 4]>) -> Self {
        self.fill = Some(track);
        self
    }

    /// Set the linear `border_colour` track.
    pub fn with_border(mut self, track: AnimTrack<[f32; 4]>) -> Self {
        self.border = Some(track);
        self
    }

    /// Set the linear `rotation` track.
    pub fn with_rotation(mut self, track: AnimTrack<f32>) -> Self {
        self.rotation = Some(track);
        self
    }

    /// Set the arbitrary-path `opacity` channel.
    pub fn with_opacity_path(mut self, track: PathTrack<f32>) -> Self {
        self.opacity_path = Some(track);
        self
    }

    /// Set the arbitrary-path `position` channel.
    pub fn with_position_path(mut self, track: PathTrack<[f32; 2]>) -> Self {
        self.position_path = Some(track);
        self
    }

    /// Set the arbitrary-path `size` channel.
    pub fn with_size_path(mut self, track: PathTrack<[f32; 2]>) -> Self {
        self.size_path = Some(track);
        self
    }

    /// Set the arbitrary-path solid-`fill` colour channel.
    pub fn with_fill_path(mut self, track: PathTrack<[f32; 4]>) -> Self {
        self.fill_path = Some(track);
        self
    }

    /// Set the arbitrary-path `border_colour` channel.
    pub fn with_border_path(mut self, track: PathTrack<[f32; 4]>) -> Self {
        self.border_path = Some(track);
        self
    }

    /// Set the arbitrary-path `rotation` channel.
    pub fn with_rotation_path(mut self, track: PathTrack<f32>) -> Self {
        self.rotation_path = Some(track);
        self
    }
}

/// Trait used by [`AnimTrack`] resolution to interpolate between `from`
/// and `to`. Implemented for the channel types the overlay animation
/// system needs: `f32`, `[f32; 2]`, `[f32; 4]`.
pub trait LerpAnim: Copy {
    /// Returns `from * (1 - t) + to * t`.
    fn lerp(from: Self, to: Self, t: f32) -> Self;
}

impl LerpAnim for f32 {
    fn lerp(from: Self, to: Self, t: f32) -> Self {
        from * (1.0 - t) + to * t
    }
}

impl LerpAnim for [f32; 2] {
    fn lerp(from: Self, to: Self, t: f32) -> Self {
        [f32::lerp(from[0], to[0], t), f32::lerp(from[1], to[1], t)]
    }
}

impl LerpAnim for [f32; 4] {
    fn lerp(from: Self, to: Self, t: f32) -> Self {
        [
            f32::lerp(from[0], to[0], t),
            f32::lerp(from[1], to[1], t),
            f32::lerp(from[2], to[2], t),
            f32::lerp(from[3], to[3], t),
        ]
    }
}

impl<T: Copy + LerpAnim> AnimTrack<T> {
    /// Resolve the track at the given absolute time. Returns the
    /// interpolated value.
    pub fn sample(&self, time: f64) -> T {
        if self.duration <= 0.0 {
            return self.to;
        }
        let raw = ((time - self.start_time) as f32) / self.duration;
        let phase = resolve_phase(raw, self.repeat);
        let t = apply_easing(phase, self.easing);
        T::lerp(self.from, self.to, t)
    }
}

/// Map a raw normalised parameter (number of cycles since `start_time`) into
/// the canonical `[0, 1]` phase using the given repeat mode.
fn resolve_phase(raw: f32, repeat: RepeatMode) -> f32 {
    match repeat {
        RepeatMode::Once => raw.clamp(0.0, 1.0),
        RepeatMode::Loop => {
            let f = raw - raw.floor();
            if f < 0.0 { f + 1.0 } else { f }
        }
        RepeatMode::PingPong => {
            let two = (raw * 0.5).floor() * 2.0;
            let r = raw - two;
            if r > 1.0 { 2.0 - r } else { r }
        }
    }
}

/// Apply an easing curve to a `[0, 1]` phase.
fn apply_easing(phase: f32, easing: OverlayEasing) -> f32 {
    match easing {
        OverlayEasing::Linear => phase,
        OverlayEasing::EaseIn => phase * phase,
        OverlayEasing::EaseOut => {
            let inv = 1.0 - phase;
            1.0 - inv * inv
        }
        OverlayEasing::EaseInOut => phase * phase * (3.0 - 2.0 * phase),
        OverlayEasing::Pulse => (phase * std::f32::consts::PI).sin(),
    }
}

/// Arbitrary-path animation track. `path` is a closure called with the eased
/// parameter `t in [0, 1]` and returns the value for the channel.
///
/// Use for any motion that's more than a straight line: Bezier arcs,
/// polylines, lissajous, custom shapes. The `bezier` and `polyline` helpers
/// cover the common cases without the consumer writing the curve math.
///
/// The closure is stored in an `Arc`, so cloning the track is cheap (one
/// atomic bump). The `Send + Sync + 'static` bound is satisfied by closures
/// that capture only owned/by-value data.
#[derive(Clone)]
pub struct PathTrack<T: Copy + LerpAnim> {
    /// Absolute time at which the track starts.
    pub start_time: f64,
    /// Length of one cycle in seconds.
    pub duration: f32,
    /// Curve applied to the normalised parameter before the closure runs.
    pub easing: OverlayEasing,
    /// What happens past the end of one cycle.
    pub repeat: RepeatMode,
    /// Evaluator for the path. Called with `t in [0, 1]` after easing and
    /// repeat resolution. The closure is shared via `Arc` so the track is
    /// cheap to clone.
    pub path: std::sync::Arc<dyn Fn(f32) -> T + Send + Sync>,
}

impl<T: Copy + LerpAnim> std::fmt::Debug for PathTrack<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PathTrack")
            .field("start_time", &self.start_time)
            .field("duration", &self.duration)
            .field("easing", &self.easing)
            .field("repeat", &self.repeat)
            .field("path", &"<closure>")
            .finish()
    }
}

impl<T: Copy + LerpAnim> PathTrack<T> {
    /// Construct a track that evaluates the supplied closure at each frame.
    /// Defaults to `Linear` easing and `Once` repeat; chain `with_easing`
    /// or `with_repeat` to override.
    pub fn new(
        start_time: f64,
        duration: f32,
        path: impl Fn(f32) -> T + Send + Sync + 'static,
    ) -> Self {
        Self {
            start_time,
            duration,
            easing: OverlayEasing::Linear,
            repeat: RepeatMode::Once,
            path: std::sync::Arc::new(path),
        }
    }

    /// Builder-style easing setter.
    pub fn with_easing(mut self, easing: OverlayEasing) -> Self {
        self.easing = easing;
        self
    }

    /// Builder-style repeat-mode setter.
    pub fn with_repeat(mut self, repeat: RepeatMode) -> Self {
        self.repeat = repeat;
        self
    }

    /// Resolve the track at the given absolute time.
    pub fn sample(&self, time: f64) -> T {
        if self.duration <= 0.0 {
            return (self.path)(1.0);
        }
        let raw = ((time - self.start_time) as f32) / self.duration;
        let phase = resolve_phase(raw, self.repeat);
        let t = apply_easing(phase, self.easing);
        (self.path)(t)
    }
}

impl PathTrack<[f32; 2]> {
    /// Construct a 2D track that walks a single cubic Bezier from `p0` to
    /// `p3` with control handles `p1` and `p2`. Evaluates the standard
    /// Bernstein form at the eased parameter.
    pub fn bezier(start_time: f64, duration: f32, control_points: [[f32; 2]; 4]) -> Self {
        let [p0, p1, p2, p3] = control_points;
        Self::new(start_time, duration, move |t| {
            let one_t = 1.0 - t;
            let w0 = one_t * one_t * one_t;
            let w1 = 3.0 * one_t * one_t * t;
            let w2 = 3.0 * one_t * t * t;
            let w3 = t * t * t;
            [
                w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0],
                w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1],
            ]
        })
    }

    /// Construct a 2D track that walks a polyline at uniform per-segment
    /// parameter. With `N` points the path spans `N - 1` equal-length
    /// parameter segments; consumers wanting arc-length-uniform motion
    /// should subdivide their polyline ahead of time.
    pub fn polyline(start_time: f64, duration: f32, points: Vec<[f32; 2]>) -> Self {
        Self::new(start_time, duration, move |t| {
            let n = points.len();
            if n == 0 {
                return [0.0, 0.0];
            }
            if n == 1 {
                return points[0];
            }
            let seg_count = n - 1;
            let scaled = t.clamp(0.0, 1.0) * seg_count as f32;
            let seg = (scaled as usize).min(seg_count - 1);
            let local = scaled - seg as f32;
            let a = points[seg];
            let b = points[seg + 1];
            [
                a[0] * (1.0 - local) + b[0] * local,
                a[1] * (1.0 - local) + b[1] * local,
            ]
        })
    }
}
