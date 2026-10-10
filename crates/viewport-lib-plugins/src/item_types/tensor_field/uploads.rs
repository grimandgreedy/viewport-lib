//! How a renderer uploads, writes and releases tensor fields.

use super::*;

field_sample_writes!(
    channels::Samples,
    TensorFieldId,
    TYPE_NAME,
    TensorFieldPlugin,
    encode = encode_samples
);

standard_uploads!(TensorFieldItem, TensorFieldId, TYPE_NAME, TensorFieldPlugin);
