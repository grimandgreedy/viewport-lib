//! ClipPlayerPlugin: drives a [`Pose`] from an [`AnimationClip`].

use crate::runtime::context::RuntimeStepContext;
use crate::runtime::plugin::{RuntimePlugin, phase};

use super::clip::AnimationClip;
use super::skeleton::Pose;

/// Runtime plugin that samples an [`AnimationClip`] each step and writes the
/// resulting [`Pose`] into `RuntimeResources` for [`super::SkeletonPlugin`] to
/// consume.
///
/// Runs at `phase::ANIMATE`, which is before the `POST_SIM` phase where
/// `SkeletonPlugin` reads the pose.
///
/// The plugin starts each frame from a clone of `bind_pose`, then overlays the
/// clip's animated channels. Joints not touched by the clip keep their bind
/// values; channels not touched by the clip on an animated joint keep their
/// bind values too. This means a clip can be authored to drive only the
/// channels it cares about and the bind pose supplies everything else.
///
/// # Example
///
/// ```rust,ignore
/// let bind_pose = /* Pose with joint 1 placed at its bind world transform */;
/// let clip = AnimationClip { duration: 2.0, tracks: vec![/* ... */] };
/// let player = ClipPlayerPlugin::new(clip, bind_pose);
/// let runtime = ViewportRuntime::new()
///     .with_plugin(player)
///     .with_plugin(skeleton_plugin);
/// ```
pub struct ClipPlayerPlugin {
    /// Clip whose tracks are sampled and applied each step.
    pub clip: AnimationClip,
    /// Per-frame baseline. Cloned each step before the clip is applied.
    pub bind_pose: Pose,
    /// Playback speed multiplier. `1.0` = real time. Negative values play in
    /// reverse.
    pub speed: f32,
    /// Whether playback wraps at `clip.duration`. Non-looping playback clamps
    /// the playhead at the endpoints.
    pub looping: bool,
    /// Current play position in seconds.
    pub playhead: f32,
    /// When false, the playhead does not advance. Manual seeks via `playhead`
    /// still take effect.
    pub playing: bool,
}

impl ClipPlayerPlugin {
    /// Create a player that loops `clip` over `bind_pose` at real-time speed.
    pub fn new(clip: AnimationClip, bind_pose: Pose) -> Self {
        Self {
            clip,
            bind_pose,
            speed: 1.0,
            looping: true,
            playhead: 0.0,
            playing: true,
        }
    }

    /// Set the playback speed multiplier.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// Set whether playback loops at `clip.duration`.
    pub fn with_looping(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }
}

impl RuntimePlugin for ClipPlayerPlugin {
    fn priority(&self) -> i32 {
        phase::ANIMATE
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        if self.playing {
            self.playhead += ctx.dt * self.speed;
            if self.clip.duration > 0.0 {
                if self.looping {
                    self.playhead = self.playhead.rem_euclid(self.clip.duration);
                } else {
                    self.playhead = self.playhead.clamp(0.0, self.clip.duration);
                }
            }
        }

        let mut pose = self.bind_pose.clone();
        self.clip.sample_into(self.playhead, &mut pose);
        ctx.resources.insert(pose);
    }
}
