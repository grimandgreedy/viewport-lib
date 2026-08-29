//! Built-in colourmap LUT data.
//!
//! The colourmap presets, their CPU lookup tables, and the ParaView XML
//! import/export helpers now live in `viewport-lib-types` so CPU-side tools can
//! sample and bake colourmaps without the renderer. They are re-exported here so
//! the renderer's `crate::resources::material::colourmap_data::*` paths keep
//! resolving.

pub use viewport_lib_types::colourmap::{
    ColourmapId, coolwarm_rgba, export_paraview_xml_colourmap, greyscale_rgba, inferno_rgba,
    jet_rgba, lerp_colourmap_lut, magma_rgba, parse_paraview_xml_colourmap, plasma_rgba,
    rainbow_rgba, rdbu_r_rgba, turbo_rgba, viridis_rgba,
};
