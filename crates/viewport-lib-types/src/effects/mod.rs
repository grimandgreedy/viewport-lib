//! Effects configuration vocabulary: the per-frame settings a consumer sets on
//! the frame's effects tree (lighting, shadows, display transform, post
//! process). Pure data; the renderer reads it to drive its passes.

pub mod lighting;
pub mod postprocess;
