//! The tensor field's writable channel.
//!
//! One channel, like the vector field, because a field's storage is one
//! interleaved record per sample. Unlike the vector field the record is not what
//! a caller would ever hold: it is a baked rotation-scale matrix, its inverse
//! transpose, a scalar and a colour, 176 bytes derived from the sample's
//! eigendecomposition. So the sample a caller supplies is the decomposition and
//! the store builds the matrices, which is the encode step this API exists to
//! keep out of consumer code.
//!
//! ```no_run
//! # use viewport_lib::plugin_api::Writes;
//! # use viewport_lib_item_types::channels::tensor_field as tf;
//! # fn f(
//! #     renderer: &mut viewport_lib::renderer::ViewportRenderer,
//! #     queue: &viewport_lib::gpu::Queue,
//! #     id: viewport_lib_item_types::TensorFieldId,
//! #     batch: &[tf::Sample],
//! # ) -> viewport_lib::error::ViewportResult<()> {
//! renderer.write_range(tf::Samples, queue, id, 1_024, batch)?;
//! # Ok(()) }
//! ```

use super::types::TensorFieldId;
use viewport_lib::Colour;
use viewport_lib::plugin_api::Channel;

/// One sample of a tensor field, as the caller supplies it for a ranged write.
///
/// Total rather than partial: the store holds the baked matrices and not the
/// decomposition they came from, so there is nothing to read back and amend.
///
/// `extents` are final half-extents along each axis, with the item's `size`
/// source and global `scale` already applied, because a ranged write bypasses
/// both. They must be positive: a zero extent makes the normal matrix a division
/// by zero, and a write clamps it to a hair of width the way the upload path
/// does rather than sending an infinity to the GPU.
#[derive(Copy, Clone, Debug)]
pub struct Sample {
    /// Object-space centre of the glyph.
    pub position: [f32; 3],
    /// The tensor's eigenvectors, as three orthonormal columns. Not
    /// re-orthonormalised: a non-orthonormal basis shears the glyph and makes
    /// its normal matrix wrong, which is the caller's business to get right.
    pub axes: [[f32; 3]; 3],
    /// Final half-extent along each axis, in the order `axes` are given.
    pub extents: [f32; 3],
    /// Value the colourmap maps, when the field draws through one.
    pub scalar: f32,
    /// Colour used when the field does not draw through a colourmap.
    pub colour: Colour,
}

/// The interleaved per-sample records. A field's only channel.
#[derive(Copy, Clone, Debug, Default)]
pub struct Samples;

impl Channel for Samples {
    type Id = TensorFieldId;
    type Input = Sample;
    const NAME: &'static str = "samples";
}
