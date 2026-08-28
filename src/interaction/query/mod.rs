//! Scene queries: ray-cast picking and transform snapping.

/// Pick result vocabulary shared by the CPU and GPU pick backends
/// ([`pick_result::PickHit`], [`pick_result::GpuPickHit`], and friends).
pub mod pick_result;
/// Ray-cast object picking.
pub mod picking;
/// Transform snapping helpers and constraint overlay types.
pub mod snap;
