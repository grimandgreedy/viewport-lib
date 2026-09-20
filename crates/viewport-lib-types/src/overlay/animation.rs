//! Time-varying animation parameters for overlay items.

/// Easing curve applied to an [`AnimTrack`]'s normalised parameter `t`.
///
/// The named variants are presets; [`OverlayEasing::CubicBezier`] is the
/// general case, with CSS `cubic-bezier()` semantics. A CSS cubic-bezier is a
/// single segment that is monotone in x, so it expresses overshoot ([`Back`])
/// but cannot express multi-oscillation curves: [`Bounce`] and [`Elastic`] are
/// evaluated natively for that reason, the same reason CSS added `linear()`.
///
/// [`Back`]: OverlayEasing::Back
/// [`Bounce`]: OverlayEasing::Bounce
/// [`Elastic`]: OverlayEasing::Elastic
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
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
    /// Pulls back before moving, and overshoots at the end. A preset over
    /// `CubicBezier { x1: 0.68, y1: -0.55, x2: 0.265, y2: 1.55 }`, which is
    /// CSS's `ease-in-out-back`.
    Back,
    /// Settles with decaying bounces, like something dropped on a surface.
    /// Not expressible as a cubic-bezier: it changes direction several times.
    Bounce,
    /// Overshoots and oscillates into place with decaying amplitude. Also not
    /// expressible as a cubic-bezier.
    Elastic,
    /// The general case: a cubic Bezier from `(0, 0)` to `(1, 1)` with control
    /// points `(x1, y1)` and `(x2, y2)`, matching CSS `cubic-bezier()`.
    ///
    /// `x1` and `x2` are clamped to `[0, 1]` so the curve stays a function of
    /// time; `y1` and `y2` are unbounded, which is what allows overshoot.
    /// Evaluating it solves for the curve parameter at the given `x` and then
    /// reads `y`, so the cost is a handful of Newton steps.
    CubicBezier {
        /// x of the first control point, clamped to `[0, 1]`.
        x1: f32,
        /// y of the first control point. Values outside `[0, 1]` overshoot.
        y1: f32,
        /// x of the second control point, clamped to `[0, 1]`.
        x2: f32,
        /// y of the second control point. Values outside `[0, 1]` overshoot.
        y2: f32,
    },
}

/// How an [`AnimTrack`] handles time past the end of its duration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
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
#[non_exhaustive]
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

/// Animation tracks attached to an overlay item or a retained group.
///
/// Each `Some` track replaces the matching field on the item for the frame.
/// Tracks are independent: an item can translate, scale, recolour, and rotate
/// at once.
///
/// # Why these five channels and no others
///
/// These are exactly the channels that survive compilation. A translate, a
/// rotation, a scale, an opacity, and a tint all ride the per-draw instance,
/// so they animate a retained group with no re-tessellation and animate every
/// overlay family identically. A size, a border colour, or a gradient stop
/// does not: driving one means re-emitting the geometry, which defeats
/// retention, and in immediate mode the consumer is already rebuilding the
/// item every frame and can set the field directly.
///
/// # Curves are ours; triggers are yours
///
/// viewport-lib evaluates the curve. When an animation starts, whether it
/// restarts, and what it responds to are the application's. `epoch` is where
/// the two meet: a track's `start_time` is a delay measured from it, so
/// authored content can say "0.2 seconds in" without knowing what time the
/// process will be started at.
///
/// Resolution is CPU-side in `prepare()`; the host must request continuous
/// repaints while any track is active.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct OverlayAnimations {
    /// Time all `start_time` values are measured from, on the same clock as
    /// `OverlayFrame::time`. `0.0` (the default) makes `start_time` an
    /// absolute time on that clock; set it when the animation begins to make
    /// `start_time` a delay instead.
    pub epoch: f64,
    /// Drives the item's overall opacity multiplier.
    pub opacity: Option<AnimTrack<f32>>,
    /// Drives `transform.translate`, the screen-pixel nudge layered on the
    /// resolved anchor.
    pub translate: Option<AnimTrack<[f32; 2]>>,
    /// Drives `transform.rotation`, in radians.
    pub rotation: Option<AnimTrack<f32>>,
    /// Drives `transform.scale`.
    pub scale: Option<AnimTrack<f32>>,
    /// Drives `tint`, the per-frame colour multiplier.
    pub tint: Option<AnimTrack<[f32; 4]>>,
    /// Arbitrary path channel driving `transform.translate`. Overrides the
    /// linear `translate` track when set.
    #[deprecated(
        since = "0.24.0",
        note = "PathTrack holds a closure, so it cannot be serialised, baked into a retained                 group, or sent across a plugin boundary. Use `translate` with an easing, or                 sample the path yourself and set the field."
    )]
    #[allow(deprecated)]
    pub translate_path: Option<PathTrack<[f32; 2]>>,
}

impl OverlayAnimations {
    /// Set the epoch all `start_time` values are measured from.
    pub fn with_epoch(mut self, epoch: f64) -> Self {
        self.epoch = epoch;
        self
    }

    /// Set the `opacity` track.
    pub fn with_opacity(mut self, track: AnimTrack<f32>) -> Self {
        self.opacity = Some(track);
        self
    }

    /// Set the `translate` track.
    pub fn with_translate(mut self, track: AnimTrack<[f32; 2]>) -> Self {
        self.translate = Some(track);
        self
    }

    /// Set the `rotation` track.
    pub fn with_rotation(mut self, track: AnimTrack<f32>) -> Self {
        self.rotation = Some(track);
        self
    }

    /// Set the `scale` track.
    pub fn with_scale(mut self, track: AnimTrack<f32>) -> Self {
        self.scale = Some(track);
        self
    }

    /// Set the `tint` track.
    pub fn with_tint(mut self, track: AnimTrack<[f32; 4]>) -> Self {
        self.tint = Some(track);
        self
    }

    /// Set the arbitrary-path translate channel.
    #[deprecated(since = "0.24.0", note = "see `OverlayAnimations::translate_path`")]
    #[allow(deprecated)]
    pub fn with_translate_path(mut self, track: PathTrack<[f32; 2]>) -> Self {
        self.translate_path = Some(track);
        self
    }

    /// Resolve every track at `time` onto the per-frame state it drives.
    ///
    /// The one place the channel list is turned into field writes, so the
    /// immediate path, the retained path, and a consumer that wants to resolve
    /// tracks itself all agree. A track replaces the field rather than adding
    /// to it, so an item's authored `translate` is the value the track
    /// interpolates away from only if the track says so.
    #[allow(deprecated)]
    pub fn apply(
        &self,
        time: f64,
        transform: &mut crate::overlay::OverlayTransform,
        opacity: &mut f32,
        tint: &mut [f32; 4],
    ) {
        let t = time - self.epoch;
        if let Some(track) = self.opacity {
            *opacity = track.sample(t);
        }
        if let Some(track) = self.translate {
            transform.translate = track.sample(t);
        }
        if let Some(track) = self.rotation {
            transform.rotation = track.sample(t);
        }
        if let Some(track) = self.scale {
            transform.scale = track.sample(t);
        }
        if let Some(track) = self.tint {
            *tint = track.sample(t);
        }
        // A path track overrides the linear track on the same channel.
        if let Some(track) = self.translate_path.as_ref() {
            transform.translate = track.sample(t);
        }
    }

    /// Whether any track is set, so a caller can skip the resolve.
    #[allow(deprecated)]
    pub fn is_empty(&self) -> bool {
        self.opacity.is_none()
            && self.translate.is_none()
            && self.rotation.is_none()
            && self.scale.is_none()
            && self.tint.is_none()
            && self.translate_path.is_none()
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

impl<T: Copy> AnimTrack<T> {
    /// A track running `from` to `to` over `duration` seconds, starting at
    /// `start_time` (a delay from [`OverlayAnimations::epoch`], or an absolute
    /// time when that is left at zero). Linear, once through.
    pub fn new(start_time: f64, duration: f32, from: T, to: T) -> Self {
        Self {
            start_time,
            duration,
            from,
            to,
            easing: OverlayEasing::Linear,
            repeat: RepeatMode::Once,
        }
    }

    /// Set the easing curve.
    pub fn with_easing(mut self, easing: OverlayEasing) -> Self {
        self.easing = easing;
        self
    }

    /// Set what happens past the end of one cycle.
    pub fn with_repeat(mut self, repeat: RepeatMode) -> Self {
        self.repeat = repeat;
        self
    }

    /// Set the start time, a delay from [`OverlayAnimations::epoch`].
    pub fn with_start_time(mut self, start_time: f64) -> Self {
        self.start_time = start_time;
        self
    }
}

impl<T: Copy + LerpAnim> AnimTrack<T> {
    /// Resolve the track at `time` measured against `epoch`, which is what
    /// [`OverlayAnimations::epoch`] is for: `start_time` is then a delay
    /// rather than a date.
    pub fn sample_from(&self, time: f64, epoch: f64) -> T {
        self.sample(time - epoch)
    }

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

/// One coordinate of a cubic Bezier from 0 to 1 with control values `a` and
/// `b`, in Bernstein form.
fn bezier_axis(t: f32, a: f32, b: f32) -> f32 {
    let u = 1.0 - t;
    3.0 * u * u * t * a + 3.0 * u * t * t * b + t * t * t
}

/// Derivative of [`bezier_axis`] with respect to `t`.
fn bezier_axis_slope(t: f32, a: f32, b: f32) -> f32 {
    let u = 1.0 - t;
    3.0 * u * u * a + 6.0 * u * t * (b - a) + 3.0 * t * t * (1.0 - b)
}

/// CSS `cubic-bezier()`: find the curve parameter whose x is `x`, then read y.
///
/// Newton from a linear guess, falling back to bisection when the slope is
/// flat enough that Newton would step wildly. Eight iterations is well past
/// convergence for the range CSS allows, and the whole thing is a handful of
/// multiplies per sample.
fn cubic_bezier_ease(x: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let x1 = x1.clamp(0.0, 1.0);
    let x2 = x2.clamp(0.0, 1.0);
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let mut t = x;
    for _ in 0..8 {
        let err = bezier_axis(t, x1, x2) - x;
        if err.abs() < 1e-5 {
            return bezier_axis(t, y1, y2);
        }
        let slope = bezier_axis_slope(t, x1, x2);
        if slope.abs() < 1e-6 {
            break;
        }
        t -= err / slope;
    }
    // Bisection fallback: guaranteed to converge because x is monotone in t.
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    let mut t = x;
    for _ in 0..24 {
        let cx = bezier_axis(t, x1, x2);
        if (cx - x).abs() < 1e-5 {
            break;
        }
        if cx < x {
            lo = t;
        } else {
            hi = t;
        }
        t = (lo + hi) * 0.5;
    }
    bezier_axis(t, y1, y2)
}

/// The standard four-segment bounce, matching the shape every easing library
/// ships under that name.
fn bounce_ease(t: f32) -> f32 {
    const N: f32 = 7.5625;
    const D: f32 = 2.75;
    if t < 1.0 / D {
        N * t * t
    } else if t < 2.0 / D {
        let t = t - 1.5 / D;
        N * t * t + 0.75
    } else if t < 2.5 / D {
        let t = t - 2.25 / D;
        N * t * t + 0.9375
    } else {
        let t = t - 2.625 / D;
        N * t * t + 0.984375
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
        OverlayEasing::Back => cubic_bezier_ease(phase, 0.68, -0.55, 0.265, 1.55),
        OverlayEasing::Bounce => bounce_ease(phase),
        OverlayEasing::Elastic => {
            if phase <= 0.0 {
                0.0
            } else if phase >= 1.0 {
                1.0
            } else {
                let c = std::f32::consts::TAU / 3.0;
                (2.0f32).powf(-10.0 * phase) * ((phase * 10.0 - 0.75) * c).sin() + 1.0
            }
        }
        OverlayEasing::CubicBezier { x1, y1, x2, y2 } => cubic_bezier_ease(phase, x1, y1, x2, y2),
    }
}

/// Arbitrary-path animation track. `path` is a closure called with the eased
/// parameter `t in [0, 1]` and returns the value for the channel.
///
/// Deprecated: it is the only overlay type that cannot be serialised, compiled
/// into a retained group, or crossed over a plugin boundary, because it holds
/// a closure. `OverlayEasing::CubicBezier` covers the curve cases it was
/// mostly used for; a genuine motion path is a `translate` track the consumer
/// re-points each frame.
///
/// Use for any motion that's more than a straight line: Bezier arcs,
/// polylines, lissajous, custom shapes. The `bezier` and `polyline` helpers
/// cover the common cases without the consumer writing the curve math.
///
/// The closure is stored in an `Arc`, so cloning the track is cheap (one
/// atomic bump). The `Send + Sync + 'static` bound is satisfied by closures
/// that capture only owned/by-value data.
#[derive(Clone)]
#[deprecated(
    since = "0.24.0",
    note = "a closure cannot be serialised, baked into a retained group, or sent across a \
            plugin boundary, which makes this the one overlay type authored content cannot \
            carry. Use an AnimTrack with an easing, or sample the path yourself."
)]
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

#[allow(deprecated)]
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

#[allow(deprecated)]
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

#[allow(deprecated)]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 2e-3
    }

    /// A cubic-bezier easing has to agree with CSS at the ends and be monotone
    /// in x, or the x-to-t solve has no unique answer.
    #[test]
    fn cubic_bezier_matches_its_endpoints_and_stays_monotone() {
        let ease = OverlayEasing::CubicBezier {
            x1: 0.25,
            y1: 0.1,
            x2: 0.25,
            y2: 1.0,
        };
        assert!(close(apply_easing(0.0, ease), 0.0));
        assert!(close(apply_easing(1.0, ease), 1.0));
        let mut prev = -1.0;
        for i in 0..=50 {
            let v = apply_easing(i as f32 / 50.0, ease);
            assert!(v >= prev - 1e-3, "not monotone at {i}: {prev} -> {v}");
            prev = v;
        }
    }

    /// The identity control points reproduce linear, which is the cheapest
    /// check that the solve is not off by a reparametrisation.
    #[test]
    fn the_identity_bezier_is_linear() {
        let ease = OverlayEasing::CubicBezier {
            x1: 1.0 / 3.0,
            y1: 1.0 / 3.0,
            x2: 2.0 / 3.0,
            y2: 2.0 / 3.0,
        };
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            assert!(close(apply_easing(t, ease), t), "at {t}");
        }
    }

    /// `Bounce` and `Elastic` land on 0 and 1 and leave the unit range in
    /// between, which is the property a cubic-bezier cannot reproduce and the
    /// reason they are native variants.
    #[test]
    fn bounce_and_elastic_are_not_expressible_as_one_bezier_segment() {
        for ease in [OverlayEasing::Bounce, OverlayEasing::Elastic] {
            assert!(close(apply_easing(0.0, ease), 0.0), "{ease:?} at 0");
            assert!(close(apply_easing(1.0, ease), 1.0), "{ease:?} at 1");
        }
        // Elastic overshoots past 1 on its way in.
        let peak = (0..100)
            .map(|i| apply_easing(i as f32 / 100.0, OverlayEasing::Elastic))
            .fold(f32::MIN, f32::max);
        assert!(peak > 1.0, "Elastic never overshoots: peak {peak}");
        // Bounce reverses direction, so it is non-monotone.
        let mut reversals = 0;
        let mut prev = apply_easing(0.0, OverlayEasing::Bounce);
        let mut rising = true;
        for i in 1..=100 {
            let v = apply_easing(i as f32 / 100.0, OverlayEasing::Bounce);
            let now = v >= prev;
            if now != rising {
                reversals += 1;
                rising = now;
            }
            prev = v;
        }
        assert!(reversals >= 3, "Bounce should reverse several times");
    }

    /// `epoch` is the whole point of authored animation: the same track data
    /// plays the same way whatever time the process was started at.
    #[test]
    fn a_start_time_is_a_delay_measured_from_the_epoch() {
        let track = AnimTrack {
            start_time: 0.5,
            duration: 1.0,
            from: 0.0f32,
            to: 1.0f32,
            ..Default::default()
        };
        // Authored "half a second in", played from two different wall clocks.
        for epoch in [0.0, 1_700_000_000.0] {
            assert!(close(track.sample_from(epoch + 0.5, epoch), 0.0));
            assert!(close(track.sample_from(epoch + 1.0, epoch), 0.5));
            assert!(close(track.sample_from(epoch + 1.5, epoch), 1.0));
        }
    }

    /// The tracks resolve onto exactly the five channels that ride the
    /// per-draw instance, and nothing else.
    #[test]
    fn apply_writes_the_instance_channels() {
        let anims = OverlayAnimations::default()
            .with_epoch(100.0)
            .with_opacity(AnimTrack {
                start_time: 0.0,
                duration: 1.0,
                from: 0.0,
                to: 1.0,
                ..Default::default()
            })
            .with_translate(AnimTrack {
                start_time: 0.0,
                duration: 1.0,
                from: [0.0, 0.0],
                to: [10.0, 20.0],
                ..Default::default()
            })
            .with_rotation(AnimTrack {
                start_time: 0.0,
                duration: 1.0,
                from: 0.0,
                to: 1.0,
                ..Default::default()
            })
            .with_scale(AnimTrack {
                start_time: 0.0,
                duration: 1.0,
                from: 1.0,
                to: 2.0,
                ..Default::default()
            })
            .with_tint(AnimTrack {
                start_time: 0.0,
                duration: 1.0,
                from: [1.0, 1.0, 1.0, 1.0],
                to: [0.0, 0.5, 1.0, 1.0],
                ..Default::default()
            });

        let mut transform = crate::overlay::OverlayTransform::IDENTITY;
        let mut opacity = 1.0f32;
        let mut tint = [1.0f32; 4];
        anims.apply(100.5, &mut transform, &mut opacity, &mut tint);

        assert!(close(opacity, 0.5));
        assert!(close(transform.translate[0], 5.0));
        assert!(close(transform.translate[1], 10.0));
        assert!(close(transform.rotation, 0.5));
        assert!(close(transform.scale, 1.5));
        assert!(close(tint[1], 0.75));
    }
}
