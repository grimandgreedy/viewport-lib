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

pub mod colourmap;
pub mod data;
pub mod ids;
