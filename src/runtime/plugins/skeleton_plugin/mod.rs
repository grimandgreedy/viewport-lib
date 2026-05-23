//! Skeletal animation: substrate types, animation clips, and built-in plugins.
//!
//! This module groups everything skeletal in one place so the feature can grow
//! (clip player, blending, IK, etc.) without scattering across the runtime
//! tree.
//!
//! - [`skeleton`]: data types and CPU LBS math (`Skeleton`, `Pose`,
//!   `JointMatrices`, `apply_skin`).
//! - [`plugin`]: the [`SkeletonPlugin`] `RuntimePlugin` impl.
//! - [`clip`]: animation clip data model (`AnimationClip`, `Track`, `Sampler`).
//! - [`clip_player`]: the [`ClipPlayerPlugin`] that drives a `Pose` from a clip.

pub mod clip;
pub mod clip_player;
pub mod plugin;
pub mod skeleton;

pub use clip::{AnimationClip, Channel, Interpolation, Sampler, Track, TrackValue, TrackValues};
pub use clip_player::ClipPlayerPlugin;
pub use plugin::SkeletonPlugin;
pub use skeleton::{Joint, JointMatrices, MAX_JOINTS, Pose, Skeleton, apply_skin};
