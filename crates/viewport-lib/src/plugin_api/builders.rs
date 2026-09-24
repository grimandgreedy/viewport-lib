//! Pipeline and bind-group construction helpers shared with plugin crates.
//!
//! These are the same helpers the built-in item types use. They exist because
//! the wgpu version legs disagree on small details of the descriptor types:
//! `depth_write_enabled` is a `bool` on one leg and an `Option<bool>` on
//! another, `entry_point` gained an `Option`, mipmap filters changed type. A
//! plugin crate that builds its pipelines with raw wgpu has to carry those
//! differences itself; one built with these does not.
//!
//! Nothing here is required. A plugin that targets a single wgpu version can
//! call `wgpu` directly and ignore this module.

pub use crate::resources::builders::{DualPipelineDesc, build_dual_pipeline};
pub use crate::resources::builders::{
    RenderPipelineDesc, clamp_linear_sampler, clamp_nearest_sampler, compute_pipeline, dcompare,
    depth_stencil, dmipmap, dwrite, pipeline_layout, render_pipeline, repeat_linear_sampler,
    sampler_entry, scene_depth_stencil, standard_scene_layout, texture_entry, texture_sampler_bgl,
    uniform_bgl, uniform_entry, uniform_texture_sampler_bgl, wgsl_module, write_mapped,
};
pub use crate::resources::device_resources::DualPipeline;

/// Prepend the `enable primitive_index;` directive when the active wgpu leg's
/// shader front end requires it. Shaders using `@builtin(primitive_index)`
/// pass their source through this before compiling.
pub use crate::resources::builders::with_primitive_index_enable;
