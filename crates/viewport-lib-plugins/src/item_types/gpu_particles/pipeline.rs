//! GPU state for the particle item type: the emit and sim compute pipelines,
//! and the draw pipelines (three blend modes across the emissive sprite, lit
//! sprite, and instanced mesh routes, for the LDR and HDR scene passes and the
//! HDR painter pass), each built the first time a draw needs it.
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
    (route * 3 + blend_index(blend)) * 2 + hdr as usize
}

/// The member of [`ParticlePipelines`] a system placed in the painter sequence
/// draws through on the HDR path: one HDR colour target, depth read-only.
pub(super) fn painter_index(route: usize, blend: SpriteBlend) -> usize {
    SCENE_MEMBERS + route * 3 + blend_index(blend)
}

/// The scene-pass members: three routes by three blends by two formats.
const SCENE_MEMBERS: usize = 18;

fn blend_index(blend: SpriteBlend) -> usize {
    match blend {
        SpriteBlend::Additive => 1,
        SpriteBlend::Premultiplied => 2,
        // Particles have no multiply or screen pipeline; those draw as alpha.
        SpriteBlend::AlphaBlend | SpriteBlend::Multiply | SpriteBlend::Screen => 0,
    }
}

/// `particle_light(colour, clip)` for a route whose draw uniform is `ub`: the
/// colour lit by the 2D light layer of the system's painter range (layer 0
/// outside the painter sequence) at clip position `clip`, reading the light
/// layers through the painter mask group at `group`. With no group to spare
/// it returns the colour.
pub(crate) fn light_wgsl(group: Option<u32>, ub: &str) -> String {
    let Some(g) = group else {
        return "fn particle_light(colour: vec4<f32>, clip: vec4<f32>) -> vec4<f32> {\n    \
                return colour;\n}\n"
            .to_string();
    };
    format!(
        "@group({g}) @binding(2) var painter_light_tex: texture_2d_array<f32>;
@group({g}) @binding(3) var<uniform> painter_lights: PainterLights;

// An unlit system and a layer no range holds keep their colour.
fn particle_light(colour: vec4<f32>, clip: vec4<f32>) -> vec4<f32> {{
    let slice = painter_light_slice(painter_lights, {ub}.painter_layer);
    if slice < 0 || {ub}.unlit != 0u {{
        return colour;
    }}
    let texel = painter_mask_texel(clip.xy / clip.w, textureDimensions(painter_light_tex));
    let light = textureLoad(painter_light_tex, texel, slice, 0);
    return vec4<f32>(painter_light_apply(colour.rgb, light, painter_lights), colour.a);
}}
"
    )
}

/// The group each route reads the light layers through, by route: the one
/// after its own groups, where the device has it.
pub(super) fn light_groups(max_bind_groups: u32) -> [Option<u32>; 3] {
    let at = |g: u32| (g < max_bind_groups).then_some(g);
    [at(2), at(3), at(2)]
}

/// What a particle draw pipeline build reads.
pub(super) struct ParticleRecipe {
    device: viewport_lib::gpu::Device,
    draw_layout: viewport_lib::gpu::PipelineLayout,
    sprite_shader: viewport_lib::plugin_api::LazyModule,
    lit_layout: viewport_lib::gpu::PipelineLayout,
    lit_shader: viewport_lib::plugin_api::LazyModule,
    mesh_layout: viewport_lib::gpu::PipelineLayout,
    mesh_shader: viewport_lib::plugin_api::LazyModule,
    sample_count: u32,
    ldr_format: viewport_lib::gpu::TextureFormat,
    depth_format: viewport_lib::gpu::TextureFormat,
}

/// Three routes by three blend modes by two formats, then the same routes and
/// blends for the HDR painter pass.
pub(super) type ParticlePipelines = viewport_lib::plugin_api::LazyPipelines<ParticleRecipe, 27>;

fn build(r: &ParticleRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    // A painter member is the single-target build into the HDR format.
    let painter = i >= SCENE_MEMBERS;
    let (hdr, route, blend) = if painter {
        let j = i - SCENE_MEMBERS;
        (false, j / 3, j % 3)
    } else {
        (i % 2 == 1, i / 6, (i / 2) % 3)
    };
    let (blend, blend_name) = match blend {
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
        SPRITE => ("sprite", &r.draw_layout, r.sprite_shader.get(), &[], None),
        SPRITE_LIT => ("sprite_lit", &r.lit_layout, r.lit_shader.get(), &[], None),
        _ => (
            "mesh",
            &r.mesh_layout,
            r.mesh_shader.get(),
            &mesh_buffers,
            Some(viewport_lib::gpu::Face::Back),
        ),
    };
    let label = if painter {
        format!("particle_{route_name}_{blend_name}_painter")
    } else {
        format!("particle_{route_name}_{blend_name}")
    };
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
            depth_compare: viewport_lib::plugin_api::depth::DEPTH_COMPARE_STRICT,
            sample_count: r.sample_count,
            ldr_format: if painter {
                viewport_lib::resources::HDR_COLOR_FORMAT
            } else {
                r.ldr_format
            },
            depth_format: r.depth_format,
            indirect_fraction_entry: Some("fs_scene"),
        },
        hdr,
    )
}

/// What the emit and simulate compute builds read.
pub(super) struct ParticleComputeRecipe {
    device: viewport_lib::gpu::Device,
    layout: viewport_lib::gpu::PipelineLayout,
    emit_shader: viewport_lib::plugin_api::LazyModule,
    sim_shader: viewport_lib::plugin_api::LazyModule,
}

pub(super) const EMIT: usize = 0;
pub(super) const SIM: usize = 1;

fn build_compute(r: &ParticleComputeRecipe, i: usize) -> viewport_lib::gpu::ComputePipeline {
    let (label, shader, entry) = match i {
        EMIT => ("particle_emit_pipeline", &r.emit_shader, "emit_main"),
        _ => ("particle_sim_pipeline", &r.sim_shader, "sim_main"),
    };
    viewport_lib::plugin_api::builders::compute_pipeline(
        &r.device,
        label,
        &r.layout,
        shader.get(),
        entry,
    )
}

/// Every pipeline the particle hooks need, made on the first prepare that
/// sees a system. Each is built the first time a frame needs it.
pub(super) struct ParticleGpu {
    /// Emit and simulate. Until both are built a frame neither steps nor
    /// draws the systems, so they start late rather than drawing a state that
    /// was never simulated.
    pub(super) compute: viewport_lib::plugin_api::LazyPipelines<
        ParticleComputeRecipe,
        2,
        viewport_lib::gpu::ComputePipeline,
    >,
    pub(super) pipelines: ParticlePipelines,
    pub(super) sprite_lit_fallback_bg: viewport_lib::gpu::BindGroup,
    /// By route, the group the light layers bind at, if any.
    pub(super) light_groups: [Option<u32>; 3],
}

impl ParticleGpu {
    pub(super) fn new(
        device: &viewport_lib::gpu::Device,
        resources: &DeviceResources,
        layouts: &super::store::ParticleLayouts,
    ) -> Self {
        // Compute pipelines.
        let emit_shader = resources.lazy_module(
            device,
            "particle_emit_shader",
            crate::item_types::shader::wgsl_source!("particle_emit"),
        );
        let sim_shader = resources.lazy_module(
            device,
            "particle_sim_shader",
            crate::item_types::shader::wgsl_source!("particle_sim"),
        );

        let compute_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "particle_compute_layout",
            &[&layouts.params_bgl, &layouts.sim_bgl],
        );

        let compute = resources.lazy_compute_pipelines(
            ParticleComputeRecipe {
                device: device.clone(),
                layout: compute_layout,
                emit_shader,
                sim_shader,
            },
            build_compute,
        );

        // Each route reads the 2D light layers through the painter mask
        // group after its own groups, where the device has one more.
        let light_groups = light_groups(device.limits().max_bind_groups);
        let masks = resources.painter_mask_bind_group_layout();
        fn with_masks<'a>(
            light: Option<u32>,
            masks: &'a viewport_lib::gpu::BindGroupLayout,
            groups: &[&'a viewport_lib::gpu::BindGroupLayout],
        ) -> Vec<&'a viewport_lib::gpu::BindGroupLayout> {
            let mut groups = groups.to_vec();
            if light.is_some() {
                groups.push(masks);
            }
            groups
        }

        let sprite_shader = resources.lazy_module(
            device,
            "particle_sprite_shader",
            &crate::item_types::shader::scene_shader(
                &[
                    viewport_lib::plugin_api::shared_wgsl::SHARED_PAINTER_MASK_WGSL,
                    &light_wgsl(light_groups[SPRITE], "sprite_ub"),
                ],
                crate::item_types::shader::wgsl_source!("particle_sprite"),
            ),
        );

        let draw_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "particle_draw_layout",
            &with_masks(
                light_groups[SPRITE],
                masks,
                &[resources.group0_layout(), &layouts.draw_bgl],
            ),
        );

        let lit_shader = resources.lazy_module(
            device,
            "particle_sprite_lit_shader",
            &crate::item_types::shader::lit_shader(
                &[
                    viewport_lib::plugin_api::shared_wgsl::SHARED_CLIP_WGSL,
                    viewport_lib::plugin_api::shared_wgsl::SHARED_PAINTER_MASK_WGSL,
                    &light_wgsl(light_groups[SPRITE_LIT], "sprite_ub"),
                ],
                crate::item_types::shader::wgsl_source!("particle_sprite_lit"),
            ),
        );

        let lit_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "particle_draw_lit_layout",
            &with_masks(
                light_groups[SPRITE_LIT],
                masks,
                &[
                    resources.group0_layout(),
                    &layouts.draw_bgl,
                    &layouts.sprite_lit_bgl,
                ],
            ),
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

        let mesh_shader = resources.lazy_module(
            device,
            "particle_mesh_shader",
            &crate::item_types::shader::scene_shader(
                &[
                    viewport_lib::plugin_api::shared_wgsl::SHARED_PAINTER_MASK_WGSL,
                    &light_wgsl(light_groups[MESH], "draw_ub"),
                ],
                crate::item_types::shader::wgsl_source!("particle_mesh"),
            ),
        );

        let mesh_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "particle_mesh_draw_layout",
            &with_masks(
                light_groups[MESH],
                masks,
                &[resources.group0_layout(), &layouts.mesh_draw_bgl],
            ),
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
                depth_format: resources.scene_depth_format(),
            },
            build,
        );

        Self {
            compute,
            pipelines,
            sprite_lit_fallback_bg,
            light_groups,
        }
    }
}
