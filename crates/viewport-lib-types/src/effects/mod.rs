//! Effects configuration vocabulary: the per-frame settings a consumer sets on
//! the frame's effects tree (lighting, shadows, display transform, post
//! process). Pure data; the renderer reads it to drive its passes.
//! [`light_transport`] documents the units and the exposure contract these
//! settings follow.

pub mod colour_space;
pub mod debug;
pub mod environment;
pub mod ground;
pub mod light_transport;
pub mod lighting;
pub mod lightmap;
pub mod postprocess;
pub mod scatter;
pub mod tone_curve;
pub mod tone_operators;
