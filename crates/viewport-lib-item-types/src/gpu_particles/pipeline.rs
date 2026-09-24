//! GPU state for the particle item type: the emit and sim compute pipelines,
//! and the nine draw pipelines (three blend modes across the emissive sprite,
//! lit sprite, and instanced mesh routes).
//!
//! The bind group layouts stay in `resources`, because
//! `create_gpu_particle_system` is public API and builds each system's
//! persistent bind groups over them; this module borrows them to build
//! pipelines.

use viewport_lib::plugin_api::builders::DualPipeline;
use viewport_lib::plugin_api::builders::DualPipelineDesc;
use viewport_lib::resources::DeviceResources;

/// Every pipeline the particle hooks need, built on the first prepare that
/// sees a system.
pub(super) struct ParticleGpu {
    pub(super) emit_pipeline: viewport_lib::gpu::ComputePipeline,
    pub(super) sim_pipeline: viewport_lib::gpu::ComputePipeline,
    pub(super) sprite_pipeline_alpha: DualPipeline,
    pub(super) sprite_pipeline_additive: DualPipeline,
    pub(super) sprite_pipeline_premultiplied: DualPipeline,
    pub(super) sprite_lit_pipeline_alpha: DualPipeline,
    pub(super) sprite_lit_pipeline_additive: DualPipeline,
    pub(super) sprite_lit_pipeline_premultiplied: DualPipeline,
    pub(super) sprite_lit_fallback_bg: viewport_lib::gpu::BindGroup,
    pub(super) mesh_pipeline_alpha: DualPipeline,
    pub(super) mesh_pipeline_additive: DualPipeline,
    pub(super) mesh_pipeline_premultiplied: DualPipeline,
}

impl ParticleGpu {
    pub(super) fn new(
        device: &viewport_lib::gpu::Device,
        resources: &DeviceResources,
        layouts: &super::store::ParticleLayouts,
    ) -> Self {
        // Compute pipelines.
        let emit_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "particle_emit_shader",
            crate::shader::wgsl_source!("particle_emit"),
        );
        let sim_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "particle_sim_shader",
            crate::shader::wgsl_source!("particle_sim"),
        );

        let compute_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "particle_compute_layout",
            &[&layouts.params_bgl, &layouts.sim_bgl],
        );

        let emit_pipeline = viewport_lib::plugin_api::builders::compute_pipeline(
            device,
            "particle_emit_pipeline",
            &compute_layout,
            &emit_shader,
            "emit_main",
        );

        let sim_pipeline = viewport_lib::plugin_api::builders::compute_pipeline(
            device,
            "particle_sim_pipeline",
            &compute_layout,
            &sim_shader,
            "sim_main",
        );

        // Draw pipelines: three blend variants of the same shader.
        let sprite_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "particle_sprite_shader",
            &crate::shader::scene_shader(&[], crate::shader::wgsl_source!("particle_sprite")),
        );

        let draw_layout = viewport_lib::plugin_api::builders::standard_scene_layout(
            device,
            "particle_draw_layout",
            resources.shared_bindings().group0_layout,
            &layouts.draw_bgl,
        );

        let sample_count = resources.sample_count();
        let ldr_format = resources.target_format();
        let alpha = viewport_lib::gpu::BlendState::ALPHA_BLENDING;
        let additive = viewport_lib::plugin_api::builders::ADDITIVE_BLEND;
        let premul = viewport_lib::plugin_api::builders::PREMULTIPLIED_BLEND;

        // Particle sprites are billboards: `Less` depth test, no depth write, no
        // culling. Only the blend mode varies across the three variants.
        let make_draw = |blend: viewport_lib::gpu::BlendState, label: &str| {
            viewport_lib::plugin_api::builders::build_dual_pipeline(
                device,
                &DualPipelineDesc {
                    label,
                    layout: &draw_layout,
                    shader: &sprite_shader,
                    vertex_entry: "vs_main",
                    fragment_entry: "fs_main",
                    vertex_buffers: &[],
                    blend: Some(blend),
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    depth_write: false,
                    depth_compare: viewport_lib::gpu::CompareFunction::Less,
                    sample_count,
                    ldr_format,
                },
            )
        };

        let sprite_pipeline_alpha = make_draw(alpha, "particle_sprite_alpha");
        let sprite_pipeline_additive = make_draw(additive, "particle_sprite_additive");
        let sprite_pipeline_premultiplied = make_draw(premul, "particle_sprite_premultiplied");

        let lit_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "particle_sprite_lit_shader",
            &crate::shader::lit_shader(&[], crate::shader::wgsl_source!("particle_sprite_lit")),
        );

        let lit_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "particle_draw_lit_layout",
            &[
                resources.shared_bindings().group0_layout,
                &layouts.draw_bgl,
                &layouts.sprite_lit_bgl,
            ],
        );

        let make_lit_draw = |blend: viewport_lib::gpu::BlendState, label: &str| {
            viewport_lib::plugin_api::builders::build_dual_pipeline(
                device,
                &DualPipelineDesc {
                    label,
                    layout: &lit_layout,
                    shader: &lit_shader,
                    vertex_entry: "vs_main",
                    fragment_entry: "fs_main",
                    vertex_buffers: &[],
                    blend: Some(blend),
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    depth_write: false,
                    depth_compare: viewport_lib::gpu::CompareFunction::Less,
                    sample_count,
                    ldr_format,
                },
            )
        };

        let sprite_lit_pipeline_alpha = make_lit_draw(alpha, "particle_sprite_lit_alpha");
        let sprite_lit_pipeline_additive = make_lit_draw(additive, "particle_sprite_lit_additive");
        let sprite_lit_pipeline_premultiplied =
            make_lit_draw(premul, "particle_sprite_lit_premultiplied");

        let sprite_lit_fallback_bg =
            device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
                label: Some("gpu_particle_lit_fallback_bg"),
                layout: &layouts.sprite_lit_bgl,
                entries: &[
                    viewport_lib::gpu::BindGroupEntry {
                        binding: 0,
                        resource: viewport_lib::gpu::BindingResource::TextureView(
                            resources.fallback_texture_view(viewport_lib::TextureSlot::Normal),
                        ),
                    },
                    viewport_lib::gpu::BindGroupEntry {
                        binding: 1,
                        resource: viewport_lib::gpu::BindingResource::Sampler(
                            resources.material_sampler(),
                        ),
                    },
                ],
            });

        let mesh_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "particle_mesh_shader",
            &crate::shader::scene_shader(&[], crate::shader::wgsl_source!("particle_mesh")),
        );

        let mesh_layout = viewport_lib::plugin_api::builders::standard_scene_layout(
            device,
            "particle_mesh_draw_layout",
            resources.shared_bindings().group0_layout,
            &layouts.mesh_draw_bgl,
        );

        // Particle meshes are closed solids, so back-face culled; still no depth
        // write since particles draw transparently after the opaque pass.
        let make_mesh_draw = |blend: viewport_lib::gpu::BlendState, label: &str| {
            viewport_lib::plugin_api::builders::build_dual_pipeline(
                device,
                &DualPipelineDesc {
                    label,
                    layout: &mesh_layout,
                    shader: &mesh_shader,
                    vertex_entry: "vs_main",
                    fragment_entry: "fs_main",
                    vertex_buffers: &[viewport_lib::plugin_api::builders::mesh_vertex_layout()],
                    blend: Some(blend),
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(viewport_lib::gpu::Face::Back),
                    depth_write: false,
                    depth_compare: viewport_lib::gpu::CompareFunction::Less,
                    sample_count,
                    ldr_format,
                },
            )
        };

        let mesh_pipeline_alpha = make_mesh_draw(alpha, "particle_mesh_alpha");
        let mesh_pipeline_additive = make_mesh_draw(additive, "particle_mesh_additive");
        let mesh_pipeline_premultiplied = make_mesh_draw(premul, "particle_mesh_premultiplied");

        Self {
            emit_pipeline,
            sim_pipeline,
            sprite_pipeline_alpha,
            sprite_pipeline_additive,
            sprite_pipeline_premultiplied,
            sprite_lit_pipeline_alpha,
            sprite_lit_pipeline_additive,
            sprite_lit_pipeline_premultiplied,
            sprite_lit_fallback_bg,
            mesh_pipeline_alpha,
            mesh_pipeline_additive,
            mesh_pipeline_premultiplied,
        }
    }
}
