/// Clustered-shading GPU resources (cluster grid + light index list + clear pass).
pub mod clustered;
/// GPU compute-filter pipeline for Clip/Threshold index compaction.
pub(crate) mod compute_filter;
/// Dynamic resolution intermediate render target.
pub(crate) mod dyn_res;
/// Auto-exposure GPU resources (log-luminance histogram + adaptation).
pub mod exposure;
/// Hierarchical-Z max-depth pyramid for GPU occlusion culling.
pub(crate) mod hiz;
