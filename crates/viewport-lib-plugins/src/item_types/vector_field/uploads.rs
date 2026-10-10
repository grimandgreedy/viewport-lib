//! How a renderer uploads, writes and releases vector fields.

use super::*;

/// A vector field's sample *is* its GPU record, so the bytes go across untouched.
fn vector_samples(data: &[channels::Sample]) -> std::borrow::Cow<'_, [u8]> {
    std::borrow::Cow::Borrowed(bytemuck::cast_slice(data))
}

field_sample_writes!(
    channels::Samples,
    VectorFieldId,
    TYPE_NAME,
    VectorFieldPlugin,
    encode = vector_samples
);

standard_uploads!(VectorFieldItem, VectorFieldId, TYPE_NAME, VectorFieldPlugin);
