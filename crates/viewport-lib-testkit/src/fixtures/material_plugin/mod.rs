//! Fixtures for [`MaterialPlugin`](viewport_lib::MaterialPlugin): shading
//! hooks composed into the lit mesh shaders, registered with
//! `DeviceResources::register_material_plugin` and selected per material
//! through `Material::shading_plugin`.
//!
//! - [`FlatColourMaterialPlugin`]: replaces lighting with a flat colour read
//!   from the plugin's params window.
//! - [`TexturedMaterialPlugin`]: authors the surface through `shade_surface`,
//!   sampling the plugin's own texture slot and tinting it from the params
//!   window.

mod flat_colour;
mod textured;

pub use flat_colour::FlatColourMaterialPlugin;
pub use textured::TexturedMaterialPlugin;
