//! AnimationPlugin: keyframed and procedural transform animation.

use crate::interaction::selection::NodeId;
use crate::runtime::context::RuntimeStepContext;
use crate::runtime::plugin::{phase, RuntimePlugin};

/// A single keyframe: a time value and the transform at that time.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Keyframe {
    /// Time in seconds from track start.
    pub time: f32,
    /// Transform at this keyframe.
    pub transform: glam::Affine3A,
}

/// An animation track: a sequence of keyframes for one scene node.
///
/// Keyframes must be sorted by ascending time. Call
/// [`AnimationPlugin::add_track`] to register the track.
#[derive(Debug, Clone)]
pub struct AnimationTrack {
    /// Node this track drives.
    pub node_id: NodeId,
    /// Keyframes sorted by time.
    pub keyframes: Vec<Keyframe>,
    /// If true the track wraps when the playhead passes the last keyframe time.
    pub looping: bool,
}

impl AnimationTrack {
    /// Duration of the track (time of the last keyframe).
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map_or(0.0, |k| k.time)
    }

    /// Sample the track at `time`, returning the interpolated transform.
    ///
    /// Returns `None` if the track has no keyframes.
    pub fn sample(&self, mut time: f32) -> Option<glam::Affine3A> {
        if self.keyframes.is_empty() {
            return None;
        }
        if self.keyframes.len() == 1 {
            return Some(self.keyframes[0].transform);
        }

        let dur = self.duration();
        if self.looping && dur > 1e-6 {
            time = time.rem_euclid(dur);
        } else {
            time = time.clamp(0.0, dur);
        }

        // Find the pair of keyframes that bracket `time`.
        let upper = self.keyframes.partition_point(|k| k.time <= time);
        let idx = upper.saturating_sub(1).min(self.keyframes.len() - 2);

        let a = &self.keyframes[idx];
        let b = &self.keyframes[idx + 1];

        let span = b.time - a.time;
        let t = if span > 1e-6 { (time - a.time) / span } else { 0.0 };

        let (sa, ra, ta) = a.transform.to_scale_rotation_translation();
        let (sb, rb, tb) = b.transform.to_scale_rotation_translation();

        let s = sa.lerp(sb, t);
        let r = ra.slerp(rb, t);
        let p = ta.lerp(tb, t);

        Some(glam::Affine3A::from_scale_rotation_translation(s, r, p))
    }
}

/// A plugin that drives node transforms from keyframed animation tracks.
///
/// Runs in the [`RuntimePhase::Animate`] phase. Each frame it advances the
/// playhead by `dt * speed` and writes the interpolated transform for each
/// track via [`crate::TransformWriteback`].
///
/// # Example
///
/// ```rust,ignore
/// use viewport_lib::{AnimationPlugin, AnimationTrack, Keyframe, ViewportRuntime};
///
/// let mut anim = AnimationPlugin::new();
/// anim.add_track(AnimationTrack {
///     node_id: my_node,
///     keyframes: vec![
///         Keyframe { time: 0.0, transform: start_transform },
///         Keyframe { time: 2.0, transform: end_transform },
///     ],
///     looping: true,
/// });
///
/// let runtime = ViewportRuntime::new().with_plugin(anim);
/// ```
pub struct AnimationPlugin {
    tracks: Vec<AnimationTrack>,
    time: f32,
    playing: bool,
    speed: f32,
}

impl Default for AnimationPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl AnimationPlugin {
    /// Create a new AnimationPlugin with no tracks. Starts playing at speed 1.
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            time: 0.0,
            playing: true,
            speed: 1.0,
        }
    }

    /// Add an animation track.
    pub fn add_track(&mut self, track: AnimationTrack) {
        self.tracks.push(track);
    }

    /// Start or resume playback.
    pub fn play(&mut self) {
        self.playing = true;
    }

    /// Pause playback. Transforms remain at the current time.
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Reset the playhead to time zero.
    pub fn reset(&mut self) {
        self.time = 0.0;
    }

    /// Set playback speed. Negative values play backward.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    /// Current playhead position in seconds.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// True if the plugin is currently advancing the playhead each step.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Duration of the longest track in seconds.
    pub fn duration(&self) -> f32 {
        self.tracks
            .iter()
            .map(|t| t.duration())
            .fold(0.0_f32, f32::max)
    }
}

impl RuntimePlugin for AnimationPlugin {
    fn priority(&self) -> i32 {
        phase::ANIMATE
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        if self.playing {
            self.time += ctx.dt * self.speed;
        }
        for track in &self.tracks {
            if let Some(transform) = track.sample(self.time) {
                ctx.writeback.set(track.node_id, transform);
            }
        }
    }
}
