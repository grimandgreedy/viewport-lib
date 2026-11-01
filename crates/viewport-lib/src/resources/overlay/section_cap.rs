//! The section cap fill pipelines: flat-colour caps on the clip planes, each
//! clipped by the other planes and the clip volumes. One member draws into the
//! LDR target, one into the HDR scene target.

/// Group 1 of the cap pipelines, one per cap draw.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CapUniform {
    pub(crate) model: [[f32; 4]; 4],
    /// Linear RGBA.
    pub(crate) colour: [f32; 4],
    /// Index into the clip-plane uniform of the plane the cap lies on, which
    /// the fragment stage does not test.
    pub(crate) own_plane: u32,
    pub(crate) _pad: [u32; 3],
}

/// Member indices of [`SectionCapPipelines::pipelines`].
pub(crate) const CAP_LDR: usize = 0;
pub(crate) const CAP_HDR: usize = 1;

/// What the cap pipeline builds read.
pub(crate) struct CapRecipe {
    device: crate::gpu::Device,
    layout: crate::gpu::PipelineLayout,
    shader: crate::resources::pipeline_slot::LazyModule,
    ldr_format: crate::gpu::TextureFormat,
    depth_format: crate::gpu::TextureFormat,
    ldr_sample_count: u32,
}

/// The cap uniform layout and the two cap pipelines.
pub(crate) struct SectionCapPipelines {
    pub(crate) bgl: crate::gpu::BindGroupLayout,
    pub(crate) pipelines: crate::resources::pipeline_slot::LazyFamily<CapRecipe, 2>,
}

fn build_cap(r: &CapRecipe, i: usize) -> crate::gpu::RenderPipeline {
    let (label, format, sample_count) = match i {
        CAP_LDR => ("section_cap_pipeline", r.ldr_format, r.ldr_sample_count),
        _ => (
            "section_cap_hdr_pipeline",
            crate::gpu::TextureFormat::Rgba16Float,
            1,
        ),
    };
    let hdr = i != CAP_LDR;
    let colour = crate::gpu::ColorTargetState {
        format,
        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
        write_mask: crate::gpu::ColorWrites::ALL,
    };
    let targets = if hdr {
        crate::plugin_api::SceneColourTargets::hdr(colour, true)
    } else {
        crate::plugin_api::SceneColourTargets::single(colour)
    };
    crate::resources::builders::render_pipeline(
        &r.device,
        crate::resources::builders::RenderPipelineDesc {
            label,
            layout: &r.layout,
            vertex_module: r.shader.get(),
            vertex_entry: "vs_main",
            vertex_buffers: &[crate::resources::OverlayVertex::buffer_layout()],
            fragment: Some(crate::gpu::FragmentState {
                module: r.shader.get(),
                entry_point: Some(if hdr { "fs_scene" } else { "fs_main" }),
                // The HDR cap draws in the scene pass. Its flat colour is
                // unlit: fraction 0.
                targets: targets.as_slice(),
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            }),
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                // A cap is seen from whichever side the camera is on.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                r.depth_format,
                false,
                crate::plugin_api::depth::DEPTH_COMPARE_STRICT,
            )),
            multisample: crate::gpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
        },
    )
}

impl crate::resources::DeviceResources {
    /// Compose the cap family and ask for the member the frame draws with
    /// (`hdr` picks the HDR target). Called by the interaction prepare on a
    /// frame with caps. Needs the full tier's group 0.
    pub(crate) fn ensure_section_cap_pipeline(&mut self, device: &crate::gpu::Device, hdr: bool) {
        if self.section_cap.is_none() {
            self.note_pipeline_built(concat!(file!(), ":", line!()));
            let bgl = crate::resources::builders::uniform_bgl(
                device,
                "section_cap_bgl",
                crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
            );
            let layout = crate::resources::builders::pipeline_layout(
                device,
                "section_cap_pipeline_layout",
                &[&self.scene_tier.binds.camera_bgl, &bgl],
            );
            let recipe = CapRecipe {
                device: device.clone(),
                layout,
                shader: self.shared_module(
                    device,
                    "section_cap_shader",
                    crate::resources::builders::wgsl_source!("section_cap"),
                ),
                ldr_format: self.target_format,
                depth_format: self.scene_depth_format(),
                ldr_sample_count: self.sample_count,
            };
            self.section_cap = Some(SectionCapPipelines {
                bgl,
                pipelines: crate::resources::pipeline_slot::LazyFamily::new(
                    recipe,
                    std::sync::Arc::clone(&self.pipeline_compiler),
                    build_cap,
                ),
            });
        }
        let family = &self.section_cap.as_ref().unwrap().pipelines;
        family.get(if hdr { CAP_HDR } else { CAP_LDR });
    }

    /// The cap pipeline for the HDR or LDR target, or `None` until it is
    /// composed and built. A draw that gets `None` is skipped.
    pub(crate) fn section_cap_pipeline(&self, hdr: bool) -> Option<&crate::gpu::RenderPipeline> {
        self.section_cap
            .as_ref()?
            .pipelines
            .get(if hdr { CAP_HDR } else { CAP_LDR })
    }
}
