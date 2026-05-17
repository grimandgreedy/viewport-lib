//! Built-in runtime plugins: animation, constraints, and simple physics.
//!
//! All plugins implement [`crate::RuntimePlugin`] and are registered on
//! [`crate::ViewportRuntime`] via [`crate::ViewportRuntime::with_plugin`].

pub mod animation;
pub mod constraint;
pub mod physics_lite;
pub mod skeleton_plugin;

pub use animation::{AnimationPlugin, AnimationTrack, Keyframe};
pub use constraint::{Constraint, ConstraintPlugin};
pub use physics_lite::{PhysicsBody, PhysicsLitePlugin};
pub use skeleton_plugin::SkeletonPlugin;
