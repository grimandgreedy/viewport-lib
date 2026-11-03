//! Selection outlines on every tier that draws a scene: the edge-detection and
//! composite passes, and each viewport's mask and outline targets.
//!
//! The outline is three passes. Selected objects draw white into an R8 mask,
//! a fullscreen edge pass turns the mask's boundary into an anti-aliased ring
//! in the outline colour texture, and a composite blends that over the scene.
//! The mask pipelines differ per tier (the full tier's run the deformers), so
//! they live with the tier; the rest is here.

use crate::resources::pipeline_slot::LazyFamily;
use crate::resources::postprocess::targets::{DepthTarget, TargetSize, ViewportTargetAllocator};
use crate::resources::postprocess::{OutlineCompositeRecipe, build_outline_composite};

/// The edge pipeline, the composite pipelines and the layouts and sampler
/// their bind groups use, built on the first frame with a selection to
/// outline.
pub(crate) struct OutlineShared {
    /// Mask texture, sampler and edge uniform.
    pub(crate) edge_bgl: crate::gpu::BindGroupLayout,
    /// Outline colour texture and sampler.
    pub(crate) composite_bgl: crate::gpu::BindGroupLayout,
    pub(crate) sampler: crate::gpu::Sampler,
    /// Fullscreen edge detection: reads the mask, writes the ring.
    pub(crate) edge_pipeline: crate::gpu::RenderPipeline,
    /// The composite onto an LDR target, single-sample or multisampled, and
    /// onto the HDR target, each built when a pass first binds it.
    pub(crate) composite: LazyFamily<OutlineCompositeRecipe, 3>,
    /// The scene depth format the mask pass's own depth target takes.
    pub(crate) depth_format: crate::gpu::TextureFormat,
}

impl OutlineShared {
    /// The composite pipeline for an LDR pass at `sample_count`, or `None`
    /// while a worker has it.
    pub(crate) fn composite_ldr(&self, sample_count: u32) -> Option<&crate::gpu::RenderPipeline> {
        self.composite.get(if sample_count > 1 {
            crate::resources::postprocess::OUTLINE_COMPOSITE_MSAA
        } else {
            crate::resources::postprocess::OUTLINE_COMPOSITE_SINGLE
        })
    }

    pub(crate) fn composite_hdr(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.composite
            .get(crate::resources::postprocess::OUTLINE_COMPOSITE_HDR)
    }
}

impl crate::resources::DeviceResources {
    /// The tier's selection mask pipelines, one-sided and two-sided, or `None`
    /// before they are built.
    pub(crate) fn outline_mask_pipelines(
        &self,
    ) -> Option<(&crate::gpu::RenderPipeline, &crate::gpu::RenderPipeline)> {
        if self.scene_tier.is_built() {
            let outline = &self.scene_tier.outline;
            Some((
                outline.mask_pipeline.as_ref()?,
                outline.mask_two_sided_pipeline.as_ref()?,
            ))
        } else {
            let mask = self.compatibility_tier.as_deref()?.outline_mask.as_ref()?;
            Some((&mask.mask, &mask.mask_two_sided))
        }
    }

    /// Build the compatibility tier's selection mask pipelines. A no-op once built,
    /// and below the compatibility tier.
    pub(crate) fn ensure_compatibility_outline_mask(&mut self, device: &crate::gpu::Device) {
        let Some(compat) = self.compatibility_tier.as_deref() else {
            return;
        };
        if compat.outline_mask.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let bgl = crate::resources::builders::uniform_bgl(
            device,
            "outline_mask_basic_bgl",
            crate::gpu::ShaderStages::VERTEX,
        );
        let shader = crate::resources::builders::wgsl_module(
            device,
            "outline_mask_basic_shader",
            crate::resources::builders::wgsl_source!("outline_mask_basic"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "outline_mask_basic_layout",
            &[&compat.camera_bgl, &bgl],
        );
        let masks = crate::resources::mesh::mesh_pipelines::build_outline_mask_pipelines(
            device,
            &layout,
            &shader,
            self.scene_depth_format(),
            crate::gpu::TextureFormat::R8Unorm,
            self.pipeline_cache.as_ref(),
        );
        if let Some(compat) = self.compatibility_tier.as_deref_mut() {
            compat.outline_mask = Some(
                crate::resources::compatibility_tier::CompatibilityOutlineMask {
                    bgl,
                    mask: masks.mask,
                    mask_two_sided: masks.mask_two_sided,
                },
            );
        }
    }

    /// Build the shared outline passes. A no-op once built.
    pub(crate) fn ensure_outline_shared(&mut self, device: &crate::gpu::Device) {
        if self.outline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let edge_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("outline_edge_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let composite_bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "outline_composite_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );
        let sampler =
            crate::resources::builders::clamp_linear_sampler(device, "outline_composite_sampler");

        let edge_shader = self.shared_module(
            device,
            "outline_edge_shader",
            crate::resources::builders::wgsl_source!("outline_edge"),
        );
        let edge_layout = crate::resources::builders::pipeline_layout(
            device,
            "outline_edge_layout",
            &[&edge_bgl],
        );
        let edge_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "outline_edge_pipeline",
                layout: &edge_layout,
                vertex_module: edge_shader.get(),
                vertex_entry: "vs_main",
                vertex_buffers: &[],
                fragment: Some(crate::gpu::FragmentState {
                    module: edge_shader.get(),
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: crate::gpu::MultisampleState::default(),
                cache: self.pipeline_cache.as_ref(),
            },
        );

        let composite_shader = self.shared_module(
            device,
            "outline_composite_shader",
            crate::resources::builders::wgsl_source!("outline_composite"),
        );
        let composite_layout = crate::resources::builders::pipeline_layout(
            device,
            "outline_composite_layout",
            &[&composite_bgl],
        );
        let composite = LazyFamily::new(
            OutlineCompositeRecipe {
                device: device.clone(),
                layout: composite_layout,
                shader: composite_shader,
                target_format: self.target_format,
                depth_format: self.scene_depth_format(),
                sample_count: self.sample_count,
            },
            std::sync::Arc::clone(&self.pipeline_compiler),
            build_outline_composite,
        );
        self.outline = Some(OutlineShared {
            edge_bgl,
            composite_bgl,
            sampler,
            edge_pipeline,
            composite,
            depth_format: self.scene_depth_format(),
        });
    }
}

/// One viewport's outline targets: the mask the selected objects draw into,
/// the colour texture the edge pass writes, and the bind groups over them.
///
/// The mask pass tests against a depth target of the same size. On the full
/// tier that is the LDR scene depth the HDR state already holds; on the
/// compatibility tier these targets own one.
pub(crate) struct OutlineTargets {
    size: [u32; 2],
    _mask_texture: crate::gpu::Texture,
    pub(crate) mask_view: crate::gpu::TextureView,
    _colour_texture: crate::gpu::Texture,
    pub(crate) colour_view: crate::gpu::TextureView,
    depth: Option<DepthTarget>,
    /// The edge pass's colour, radius and target size.
    pub(crate) edge_uniform_buf: crate::gpu::Buffer,
    /// The mask, the sampler and the edge uniform, for the edge pass.
    pub(crate) edge_bind_group: crate::gpu::BindGroup,
    /// The outline colour texture and the sampler, for the composite.
    pub(crate) composite_bind_group: crate::gpu::BindGroup,
}

impl OutlineTargets {
    pub(crate) fn new(
        device: &crate::gpu::Device,
        shared: &OutlineShared,
        target_format: crate::gpu::TextureFormat,
        size: [u32; 2],
        own_depth: bool,
    ) -> Self {
        let size = [size[0].max(1), size[1].max(1)];
        let alloc = ViewportTargetAllocator::new(
            device,
            size[0],
            size[1],
            size[0],
            size[1],
            1,
            shared.depth_format,
        );
        let (mask_texture, mask_view) = alloc.colour(
            "outline_mask_texture",
            crate::gpu::TextureFormat::R8Unorm,
            TargetSize::Scene,
            crate::gpu::TextureUsages::empty(),
        );
        let (colour_texture, colour_view) = alloc.colour(
            "outline_colour_texture",
            target_format,
            TargetSize::Scene,
            crate::gpu::TextureUsages::empty(),
        );
        let depth = own_depth.then(|| alloc.depth("outline_mask_depth", TargetSize::Scene));
        let edge_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("outline_edge_uniform_buf"),
            size: std::mem::size_of::<crate::resources::OutlineEdgeUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let edge_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("outline_edge_bg"),
            layout: &shared.edge_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&mask_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(&shared.sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: edge_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let composite_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("outline_composite_bg"),
            layout: &shared.composite_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&colour_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(&shared.sampler),
                },
            ],
        });
        Self {
            size,
            _mask_texture: mask_texture,
            mask_view,
            _colour_texture: colour_texture,
            colour_view,
            depth,
            edge_uniform_buf,
            edge_bind_group,
            composite_bind_group,
        }
    }

    /// Whether these targets serve a mask pass at `size`, with or without a
    /// depth target of their own.
    pub(crate) fn fits(&self, size: [u32; 2], own_depth: bool) -> bool {
        self.size == [size[0].max(1), size[1].max(1)] && self.depth.is_some() == own_depth
    }

    pub(crate) fn size(&self) -> [u32; 2] {
        self.size
    }

    /// The depth target the mask pass tests against, when these targets own
    /// one.
    pub(crate) fn depth_view(&self) -> Option<&crate::gpu::TextureView> {
        self.depth.as_ref().map(|d| &d.view)
    }
}
