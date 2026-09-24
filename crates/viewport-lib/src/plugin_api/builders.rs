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

pub use crate::resources::builders::{
    ADDITIVE_BLEND, PREMULTIPLIED_BLEND, RenderPipelineDesc, build_outline_mask_pipeline,
    clamp_linear_sampler, clamp_nearest_sampler, compute_pipeline, dcompare, depth_stencil,
    dmipmap, dwrite, mesh_vertex_layout, pipeline_layout, render_pipeline, repeat_linear_sampler,
    sampler_entry, scene_depth_stencil, standard_scene_layout, texture_entry, texture_sampler_bgl,
    uniform_bgl, uniform_entry, uniform_texture_sampler_bgl, wgsl_module, write_mapped,
};
pub use crate::resources::builders::{DualPipelineDesc, build_dual_pipeline};

pub use crate::resources::device_resources::DualPipeline;
/// The depth bias the renderer's own two-sided surfaces cast shadows with.
///
/// An item type drawing cull-none geometry that casts into the cascade atlas
/// takes this on its shadow pipeline so its contact shadows land where the
/// built-in ones do.
pub use crate::resources::mesh::mesh_pipelines::CSM_SHADOW_BIAS_TWO_SIDED;

/// The vertex of the shared mesh arena, which
/// [`mesh_vertex_layout`] describes. Fill this to build geometry for a
/// pipeline declaring that layout.
pub use crate::resources::Vertex;

/// Prepend the `enable primitive_index;` directive when the active wgpu leg's
/// shader front end requires it. Shaders using `@builtin(primitive_index)`
/// pass their source through this before compiling.
pub use crate::resources::builders::with_primitive_index_enable;
