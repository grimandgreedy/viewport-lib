//! Built-in plugins for `viewport-lib`.
//!
//! Each plugin lives in its own subdirectory and exposes its public surface
//! through its own `mod.rs`. Consumers import from the plugin's own path:
//!
//! ```ignore
//! use viewport_lib::plugins::skeleton::{SkeletonPlugin, SkinnedActorPlugin};
//! use viewport_lib::plugins::skinning::SkinningPlugin;
//! ```
//!
//! This module intentionally does not re-export its children's symbols at
//! the top level: the path identifies which plugin each type belongs to,
//! and the in-crate convention matches the shape used by external plugin
//! crates (one public root per crate, not a shared namespace).
//!
//! Plugin kinds currently shipped:
//!
//! - **Deformer plugins** register a per-vertex deformation body against
//!   the mesh shader family. The only in-crate deformer is [`skinning`].
//! - **Runtime plugins** implement
//!   [`RuntimePlugin`](crate::RuntimePlugin) and are added to a
//!   [`ViewportRuntime`](crate::ViewportRuntime) via
//!   [`with_plugin`](crate::ViewportRuntime::with_plugin). The in-crate set
//!   covers [`animation`], [`constraint`], [`physics_lite`], and the
//!   [`skeleton`] family.

pub mod animation;
pub mod constraint;
pub mod physics_lite;
pub mod skeleton;
pub mod skinning;
