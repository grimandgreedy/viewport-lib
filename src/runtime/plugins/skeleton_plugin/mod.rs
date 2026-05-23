//! Skeletal animation: substrate types and the built-in driver plugin.
//!
//! This module groups everything skeletal in one place so the feature can grow
//! (clip player, blending, IK, etc.) without scattering across the runtime
//! tree.
//!
//! - [`skeleton`]: data types and CPU LBS math (`Skeleton`, `Pose`,
//!   `JointMatrices`, `apply_skin`).
//! - [`plugin`]: the [`SkeletonPlugin`] `RuntimePlugin` impl.

pub mod plugin;
pub mod skeleton;

pub use plugin::SkeletonPlugin;
pub use skeleton::{Joint, JointMatrices, MAX_JOINTS, Pose, Skeleton, apply_skin};
