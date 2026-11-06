//! Fixtures for [`ShadingHookDesc`](viewport_lib::ShadingHookDesc): a shading
//! hook registered on its own with `DeviceResources::register_shading_hook`,
//! the layer a material plugin sits on.
//!
//! - [`flat_colour_shading_hook`]: replaces lighting with a fixed flat colour.

mod flat_colour;

pub use flat_colour::flat_colour_shading_hook;
