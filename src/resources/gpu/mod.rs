/// Clustered-shading GPU resources (cluster grid + light index list + clear pass).
pub mod clustered;
/// GPU compute-filter pipeline for Clip/Threshold index compaction.
pub(crate) mod compute_filter;
/// Dynamic resolution intermediate render target.
pub(crate) mod dyn_res;
/// GPU particle systems: compute-driven emit + sim with sprite draw.
pub mod gpu_particles;
/// Hierarchical-Z max-depth pyramid for GPU occlusion culling.
pub(crate) mod hiz;
