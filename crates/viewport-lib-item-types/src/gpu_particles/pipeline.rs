//! GPU state for the particle item type: the emit and sim compute pipelines,
//! and the nine draw pipelines (three blend modes across the emissive sprite,
//! lit sprite, and instanced mesh routes), each draw pipeline built in the
//! format a draw needs the first time it needs it.
//!
//! The bind group layouts stay in `resources`, because
//! `create_gpu_particle_system` is public API and builds each system's
//! persistent bind groups over them; this module borrows them to build
//! pipelines.

use viewport_lib::plugin_api::builders::DualPipelineDesc;
use viewport_lib::renderer::SpriteBlend;
use viewport_lib::resources::DeviceResources;

/// Draw routes, the outer grouping of [`ParticlePipelines`] members.
pub(super) const SPRITE: usize = 0;
pub(super) const SPRITE_LIT: usize = 1;
pub(super) const MESH: usize = 2;

/// The member of [`ParticlePipelines`] for a route, blend mode and format.
pub(super) fn draw_index(route: usize, blend: SpriteBlend, hdr: bool) -> usize {
    let blend = match blend {
        SpriteBlend::AlphaBlend => 0,
        SpriteBlend::Additive => 1,
        SpriteBlend::Premultiplied => 2,
    };
    (route * 3 + blend) * 2 + hdr as usize
}

/// What a particle draw pipeline build reads.
pub(super) struct ParticleRecipe {
    device: viewport_lib::gpu::Device,
    draw_layout: viewport_lib::gpu::PipelineLayout,
    sprite_shader: viewport_lib::gpu::ShaderModule,
    lit_layout: viewport_lib::gpu::PipelineLayout,
    lit_shader: viewport_lib::gpu::ShaderModule,
    mesh_layout: viewport_lib::gpu::PipelineLayout,
    mesh_shader: viewport_lib::gpu::ShaderModule,
    sample_count: u32,
    ldr_format: viewport_lib::gpu::TextureFormat,
}

/// Three routes by three blend modes by two formats.
pub(super) type ParticlePipelines = viewport_lib::plugin_api::LazyPipelines<ParticleRecipe, 18>;

fn build(r: &ParticleRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    let hdr = i % 2 == 1;
    let route = i / 6;
    let (blend, blend_name) = match (i / 2) % 3 {
        0 => (viewport_lib::gpu::BlendState::ALPHA_BLENDING, "alpha"),
        1 => (
            viewport_lib::plugin_api::builders::ADDITIVE_BLEND,
            "additive",
        ),
        _ => (
            viewport_lib::plugin_api::builders::PREMULTIPLIED_BLEND,
            "premultiplied",
        ),
    };
    let mesh_buffers = [viewport_lib::plugin_api::builders::mesh_vertex_layout()];
    // Sprites are billboards: no culling. Particle meshes are closed solids,
    // so back-face culled. Neither writes depth, since particles draw
    // transparently after the opaque pass.
    let (route_name, layout, shader, vertex_buffers, cull_mode): (_, _, _, &[_], _) = match route {
        SPRITE => ("sprite", &r.draw_layout, &r.sprite_shader, &[], None),
        SPRITE_LIT => ("sprite_lit", &r.lit_layout, &r.lit_shader, &[], None),
        _ => (
            "mesh",
            &r.mesh_layout,
            &r.mesh_shader,
            &mesh_buffers,
            Some(viewport_lib::gpu::Face::Back),
        ),
    };
    let label = format!("particle_{route_name}_{blend_name}");
    viewport_lib::plugin_api::builders::build_dual_pipeline_variant(
        &r.device,
        &DualPipelineDesc {
            label: &label,
            layout,
            shader,
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            vertex_buffers,
            blend: Some(blend),
            topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
            cull_mode,
            depth_write: false,
            depth_compare: viewport_lib::gpu::CompareFunction::Less,
            sample_count: r.sample_count,
            ldr_format: r.ldr_format,
        },
        hdr,
    )
}

/// Every pipeline the particle hooks need, made on the first prepare that
/// sees a system. The compute pipelines are built here; the draw pipelines on
/// first use.
pub(super) struct ParticleGpu {
    pub(super) emit_pipeline: viewport_lib::gpu::ComputePipeline,
    pub(super) sim_pipeline: viewport_lib::gpu::ComputePipeline,
    pub(super) pipelines: ParticlePipelines,
    pub(super) sprite_lit_fallback_bg: viewport_lib::gpu::BindGroup,
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

        let pipelines = resources.lazy_pipelines(
            ParticleRecipe {
                device: device.clone(),
                draw_layout,
                sprite_shader,
                lit_layout,
                lit_shader,
                mesh_layout,
                mesh_shader,
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
            },
            build,
        );

        Self {
            emit_pipeline,
            sim_pipeline,
            pipelines,
            sprite_lit_fallback_bg,
        }
    }
}
