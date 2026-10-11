//! The three curve mesh item types: streamtube, tube and ribbon.
//!
//! All three take strips of world-space points and sweep them into a connected
//! triangle mesh on the CPU. They share their pick and outline-mask pipelines,
//! but each is its own [`ItemTypePlugin`](viewport_lib::plugin_api::ItemTypePlugin)
//! with its own type name ([`STREAMTUBE_TYPE_NAME`], [`TUBE_TYPE_NAME`],
//! [`RIBBON_TYPE_NAME`]), registered and submitted separately.
//!
//! Streamtube and tube shade through the same pipeline description and
//! therefore compile it once each. Ribbon differs: it is a flat two-sided quad
//! strip, so it keys its pipelines on blend mode, casts shadows, and routes
//! transparent draws through the OIT pass.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.

mod types;

pub use types::{
    RibbonId, RibbonItem, RibbonRefItem, StreamtubeId, StreamtubeItem, StreamtubeRefItem, TubeId,
    TubeItem, TubeRefItem,
};

#[cfg(feature = "gpu")]
mod cpu_pick;
#[cfg(feature = "gpu")]
mod draw;
#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;
#[cfg(feature = "gpu")]
mod ribbon;
#[cfg(feature = "gpu")]
mod store;
#[cfg(feature = "gpu")]
mod streamtube;
#[cfg(feature = "gpu")]
mod tube;
#[cfg(feature = "gpu")]
mod uploads;

#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
#[cfg(feature = "gpu")]
pub use ribbon::RibbonPlugin;
#[cfg(feature = "gpu")]
pub use ribbon::TYPE_NAME as RIBBON_TYPE_NAME;
#[cfg(feature = "gpu")]
pub use streamtube::StreamtubePlugin;
#[cfg(feature = "gpu")]
pub use streamtube::TYPE_NAME as STREAMTUBE_TYPE_NAME;
#[cfg(feature = "gpu")]
pub use tube::TYPE_NAME as TUBE_TYPE_NAME;
#[cfg(feature = "gpu")]
pub use tube::TubePlugin;
