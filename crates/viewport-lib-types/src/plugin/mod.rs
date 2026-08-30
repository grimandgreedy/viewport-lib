//! Plugin descriptor vocabulary: the pure data and traits a consumer supplies
//! to define a deformer or material plugin. The registration machinery that
//! consumes these (WGSL composition, validation, GPU resources) lives in
//! `viewport-lib`.

pub mod deformer;
pub mod material;
