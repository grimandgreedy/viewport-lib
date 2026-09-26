//! The vector field's writable channel.
//!
//! One channel, not four, because a field's storage is one interleaved record per
//! sample and the fields of that record are derived together. Rewriting a
//! sample's position alone would mean reading its size, scalar and colour back
//! off the GPU, or keeping a CPU copy of every sample for the sake of it. So the
//! channel is the whole sample and every write supplies every field.
//!
//! ```no_run
//! # use viewport_lib::plugin_api::Writes;
//! # use viewport_lib_item_types::channels::vector_field as vf;
//! # fn f(
//! #     renderer: &mut viewport_lib::renderer::ViewportRenderer,
//! #     queue: &viewport_lib::gpu::Queue,
//! #     id: viewport_lib_item_types::VectorFieldId,
//! #     batch: &[vf::Sample],
//! # ) -> viewport_lib::error::ViewportResult<()> {
//! renderer.write_range(vf::Samples, queue, id, 4_096, batch)?;
//! # Ok(()) }
//! ```

use super::types::VectorFieldId;
use viewport_lib::Colour;
use viewport_lib::plugin_api::Channel;

/// One sample of a vector field, as the caller supplies it for a ranged write.
///
/// Total rather than partial: the store keeps no CPU copy of a sample, so
/// "leave this field alone" would mean a read back off the GPU. Every write
/// supplies every field, and the values are final rather than mapped: `size` is
/// the resolved size before the item's global `scale`, and `colour` and `scalar`
/// are what the shader uses directly rather than inputs to the item's
/// `ColourSource`. A caller streaming through a mapping applies it itself.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sample {
    /// Object-space base of the vector.
    pub position: [f32; 3],
    /// Resolved size, before the item's global `scale`.
    pub size: f32,
    /// The vector itself, unnormalised: its length is the magnitude.
    pub vector: [f32; 3],
    /// Value the colourmap maps, when the field draws through one.
    pub scalar: f32,
    /// Final linear RGBA, used when the field does not draw through a colourmap.
    pub colour: [f32; 4],
}

impl Sample {
    /// A sample with a colour given as a [`Colour`], which converts to the
    /// linear RGBA the record holds.
    pub fn new(position: [f32; 3], vector: [f32; 3], size: f32, colour: Colour) -> Self {
        Self {
            position,
            size,
            vector,
            scalar: 0.0,
            colour: colour.to_linear_rgba(),
        }
    }
}

/// The interleaved per-sample records. A field's only channel.
#[derive(Copy, Clone, Debug, Default)]
pub struct Samples;

impl Channel for Samples {
    type Id = VectorFieldId;
    type Input = Sample;
    const NAME: &'static str = "samples";
}
