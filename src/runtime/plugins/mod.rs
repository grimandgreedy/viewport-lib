//! Built-in runtime plugins: animation, constraints, simple physics, and
//! skeletal animation.
//!
//! All plugins implement [`crate::RuntimePlugin`] and are registered on
//! [`crate::ViewportRuntime`] via [`crate::ViewportRuntime::with_plugin`].
//!
//! # Layout convention
//!
//! New plugins live in their own subdirectory under `plugins/`, not as a single
//! file. Use the [`skeleton_plugin`] module as the template:
//!
//! ```text
//! plugins/
//!   my_plugin/
//!     mod.rs        // re-exports the public surface
//!     plugin.rs     // the RuntimePlugin impl
//!     <feature>.rs  // substrate types, math, helpers, internal state
//! ```
//!
//! Rationale: features grow. A plugin that starts as a single file invariably
//! accumulates substrate types, helper modules, and tests. Starting in a
//! subdirectory avoids a later rename-and-rewire that breaks `use` paths
//! across the codebase. It also keeps each plugin's surface area discoverable
//! in one place rather than scattered between `runtime/<feature>.rs` and
//! `runtime/plugins/<feature>.rs`.
//!
//! All plugins follow the subdirectory layout. Use the [`skeleton_plugin`] module
//! as the template for new plugins.

pub mod animation;
pub mod constraint;
pub mod physics_lite;
pub mod skeleton_plugin;

pub use animation::{AnimationPlugin, AnimationTrack, Keyframe};
pub use constraint::{Constraint, ConstraintPlugin};
pub use physics_lite::{PhysicsBody, PhysicsLitePlugin};
pub use skeleton_plugin::{
    AnimationClip, Channel, ClipPlayerPlugin, Interpolation, Joint, JointMatrices, MAX_JOINTS,
    Pose, Sampler, Skeleton, SkeletonPlugin, SkinnedActor, SkinnedActorPart, SkinnedActorPlugin,
    SkinningPath, Track, TrackValue, TrackValues, apply_skin,
};
