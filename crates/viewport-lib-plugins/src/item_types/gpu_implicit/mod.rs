//! The GPU implicit surface item type as an [`ItemTypePlugin`]: a set of
//! signed-distance primitives combined by a blend mode and ray-marched on a
//! full-screen quad. Consumers submit [`GpuImplicitItem`]s on
//! `frame.scene.items_mut::<GpuImplicitItem>()`; the renderer routes them to
//! this plugin.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

mod types;

pub use types::{GpuImplicitItem, GpuImplicitOptions, ImplicitBlendMode, ImplicitPrimitive};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.gpu_implicit";

#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;

#[cfg(feature = "gpu")]
pub use plugin::GpuImplicitPlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
