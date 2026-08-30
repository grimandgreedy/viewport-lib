//! Pure-data types for `viewport-lib`.
//!
//! This crate holds the data vocabulary a `viewport-lib` consumer submits and
//! references: mesh and volume payloads, handle ids, scene and camera types, the
//! frame/effects config tree, overlay item data, and input events. It carries no
//! GPU dependency (no `wgpu`), so CPU-side tools such as loaders and mesh
//! processors can produce and consume these types without pulling the renderer.
//!
//! `viewport-lib` re-exports this crate's surface, so existing
//! `viewport_lib::` paths keep working.
//!
//! Its modules are populated as the type vocabulary is moved out of
//! `viewport-lib`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod camera;
pub mod colourmap;
pub mod data;
pub mod effects;
pub mod error;
pub mod ids;
pub mod input;
pub mod material;
pub mod overlay;
pub mod plugin;
pub mod render_item;
pub mod scene;

pub mod prelude {
    //! The types most code reaches for, in one glob import:
    //! `use viewport_lib_types::prelude::*;`.
    //!
    //! This is the common set, not the whole surface. Reach into the individual
    //! modules (`data`, `overlay`, `input`, `effects`, ...) for the rest.
    pub use crate::camera::Camera;
    pub use crate::colourmap::BuiltinColourmap;
    pub use crate::data::{attribute::AttributeData, mesh::MeshData};
    pub use crate::error::{ViewportError, ViewportResult};
    pub use crate::ids::{MeshId, TextureId};
    pub use crate::input::{Action, ActionFrame, ViewportEvent};
    pub use crate::material::Material;
    pub use crate::overlay::{LabelItem, OverlayShape, OverlayShapeItem};
    pub use crate::scene::{Aabb, NodeId};
}
