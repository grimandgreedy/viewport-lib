//! Deprecated location. Plugins now live in [`crate::plugins`].
//!
//! Each in-crate plugin is imported from its own module path
//! (`viewport_lib::plugins::<plugin>::*`). The re-exports here keep older
//! `viewport_lib::runtime::plugins::*` paths working for one release while
//! consumers update.

#[deprecated(since = "0.18.0", note = "use viewport_lib::plugins::animation::* instead")]
pub use crate::plugins::animation::{AnimationPlugin, AnimationTrack, Keyframe};
#[deprecated(since = "0.18.0", note = "use viewport_lib::plugins::constraint::* instead")]
pub use crate::plugins::constraint::{Constraint, ConstraintPlugin};
#[deprecated(since = "0.18.0", note = "use viewport_lib::plugins::physics_lite::* instead")]
pub use crate::plugins::physics_lite::{PhysicsBody, PhysicsLitePlugin};
#[deprecated(since = "0.18.0", note = "use viewport_lib::plugins::skeleton::* instead")]
pub use crate::plugins::skeleton::{
    AnimationClip, Channel, ClipPlayerPlugin, Interpolation, Joint, JointMatrices, MAX_JOINTS,
    Pose, Sampler, Skeleton, SkeletonPlugin, SkinnedActor, SkinnedActorPart, SkinnedActorPlugin,
    SkinningPath, Track, TrackValue, TrackValues, apply_skin,
};

#[deprecated(since = "0.18.0", note = "use viewport_lib::plugins::skeleton instead")]
pub use crate::plugins::skeleton as skeleton_plugin;
#[deprecated(since = "0.18.0", note = "use viewport_lib::plugins::animation instead")]
pub use crate::plugins::animation;
#[deprecated(since = "0.18.0", note = "use viewport_lib::plugins::constraint instead")]
pub use crate::plugins::constraint;
#[deprecated(since = "0.18.0", note = "use viewport_lib::plugins::physics_lite instead")]
pub use crate::plugins::physics_lite;
