//! The tensor field item type as an [`ItemTypePlugin`]: one instanced
//! ellipsoid per sample, oriented and scaled by the sample's eigenvectors and
//! eigenvalues. Consumers submit [`TensorFieldItem`]s with
//! `frame.scene.submit::<TensorFieldItem>(..)`, or [`TensorFieldRefItem`]s
//! to draw a set uploaded once through
//! [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload). Both forms
//! arrive at this plugin under the one type name.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

pub mod eigen;
mod types;

pub use eigen::{SymmetricEigen, symmetric_eigen_3x3, symmetric_eigenvalues_3x3};
pub use types::{TensorFieldId, TensorFieldItem, TensorFieldRefItem, TensorSource};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.tensor_field";

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
pub use plugin::TensorFieldPlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
#[cfg(feature = "gpu")]
pub(crate) use store::encode_samples;
