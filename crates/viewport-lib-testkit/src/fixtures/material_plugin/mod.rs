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
//! - [`SurfaceGlowMaterialPlugin`]: sets emissive and alpha through
//!   `shade_surface` with lighting zeroed, so a pixel reads a known blend of
//!   the emissive and the background.
//! - [`FrameInputsMaterialPlugin`]: emits the camera uniform's time and
//!   viewport and the raw vertex colour, with lighting zeroed.
//! - [`AoOverrideMaterialPlugin`]: writes a zero AO override, with or without
//!   the flag that applies it.
//! - [`RecolourTintMaterialPlugin`]: tints the lit colour through `recolour`.

mod ao_override;
mod flat_colour;
mod frame_inputs;
mod recolour_tint;
mod surface_glow;
mod textured;

pub use ao_override::AoOverrideMaterialPlugin;
pub use flat_colour::FlatColourMaterialPlugin;
pub use frame_inputs::FrameInputsMaterialPlugin;
pub use recolour_tint::RecolourTintMaterialPlugin;
pub use surface_glow::SurfaceGlowMaterialPlugin;
pub use textured::TexturedMaterialPlugin;
