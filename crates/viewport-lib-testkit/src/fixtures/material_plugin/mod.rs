//! Fixtures for [`MaterialPlugin`](viewport_lib::MaterialPlugin): shading
//! hooks composed into the lit mesh shaders, registered with
//! `DeviceResources::register_material_plugin` and selected per material
//! through `Material::with_plugin`.
//!
//! - [`FlatColourMaterialPlugin`]: replaces lighting with a flat colour read
//!   from a declared parameter.
//! - [`TexturedMaterialPlugin`]: authors the surface through `shade_surface`,
//!   sampling the material's plugin texture and tinting it from a declared
//!   parameter.

mod flat_colour;
mod textured;

pub use flat_colour::FlatColourMaterialPlugin;
pub use textured::TexturedMaterialPlugin;
