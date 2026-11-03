//! The vector field item type as an [`ItemTypePlugin`]: one instanced mesh per
//! sample, oriented along the sample's vector and sized and coloured from it.
//! Consumers submit [`VectorFieldItem`]s with
//! `frame.scene.items_mut::<VectorFieldItem>()`, or [`VectorFieldRefItem`]s to
//! draw a field uploaded once through
//! [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload). Both forms
//! arrive at this plugin under the one type name.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

pub mod sampling;
mod types;

pub use sampling::{
    VectorSamples, compute_face_tangent_frames, compute_vertex_tangent_frames,
    edge_one_form_vectors, face_intrinsic_vectors, tangents_from_explicit,
    vertex_intrinsic_vectors,
};
pub use types::{VectorFieldId, VectorFieldItem, VectorFieldRefItem};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.vector_field";

#[cfg(feature = "gpu")]
pub mod channels;
#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;
#[cfg(feature = "gpu")]
mod store;
#[cfg(feature = "gpu")]
mod uploads;

#[cfg(feature = "gpu")]
pub use plugin::VectorFieldPlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
