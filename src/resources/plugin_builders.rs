//! Plugin-facing accessors and pipeline builders on [`ViewportGpuResources`].
//!
//! See [`crate::plugin_api`] for the published types these methods return.

use crate::plugin_api::{
    MaskTargetDesc, OitTargetDesc, OpaqueTargetDesc, PickTargetDesc, ShadowTargetDesc,
    SharedBindings,
    target_desc::{OIT_ACCUM_BLEND, OIT_REVEAL_BLEND},
};
use crate::resources::ViewportGpuResources;

/// HDR colour format used by the scene buffer. Plugins targeting the HDR
/// path build pipelines against this format.
pub const HDR_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Depth-stencil format shared by every scene render pass.
pub const SCENE_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;

/// Shadow atlas depth format.
pub const SHADOW_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Outline-mask colour format.
pub const MASK_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Pick-id colour format.
pub const PICK_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

impl ViewportGpuResources {
    // ------------------------------------------------------------------
    // Target descriptors and SharedBindings accessor
    // ------------------------------------------------------------------

    /// Group-0 bind layout shared by every scene pipeline. Use as group 0
    /// when building a plugin pipeline layout.
    pub fn shared_bindings(&self) -> SharedBindings<'_> {
        SharedBindings {
            group0_layout: &self.camera_bind_group_layout,
        }
    }

    /// Render-target descriptor for the HDR opaque scene pass.
    pub fn opaque_target_desc(&self) -> OpaqueTargetDesc {
        OpaqueTargetDesc {
            color_format: HDR_COLOR_FORMAT,
            depth_format: SCENE_DEPTH_FORMAT,
            sample_count: self.sample_count,
        }
    }

    /// Render-target descriptor for the OIT pass (MRT: accum + reveal).
    pub fn oit_target_desc(&self) -> OitTargetDesc {
        OitTargetDesc {
            accum_format: HDR_COLOR_FORMAT,
            reveal_format: MASK_COLOR_FORMAT,
            depth_format: SCENE_DEPTH_FORMAT,
            accum_blend: OIT_ACCUM_BLEND,
            reveal_blend: OIT_REVEAL_BLEND,
            sample_count: self.sample_count,
        }
    }

    /// Render-target descriptor for the outline-mask pass.
    pub fn mask_target_desc(&self) -> MaskTargetDesc {
        MaskTargetDesc {
            color_format: MASK_COLOR_FORMAT,
            depth_format: SCENE_DEPTH_FORMAT,
            sample_count: 1,
        }
    }

    /// Render-target descriptor for the pick-id pass.
    pub fn pick_target_desc(&self) -> PickTargetDesc {
        PickTargetDesc {
            color_format: PICK_COLOR_FORMAT,
            depth_format: SCENE_DEPTH_FORMAT,
            sample_count: 1,
        }
    }

    /// Render-target descriptor for the shadow-atlas pass.
    pub fn shadow_target_desc(&self) -> ShadowTargetDesc {
        ShadowTargetDesc {
            depth_format: SHADOW_DEPTH_FORMAT,
            sample_count: 1,
        }
    }

    // ------------------------------------------------------------------
    // Texture-id namespace accessors
    // ------------------------------------------------------------------

    /// Borrow the `TextureView` for a texture previously uploaded via
    /// [`upload_texture`](Self::upload_texture) or
    /// [`upload_normal_map`](Self::upload_normal_map).
    ///
    /// Returns `None` if `id` does not refer to a live texture (a stale handle
    /// whose texture was freed, or one out of range).
    ///
    /// Lifetime contract: the returned view is valid until the texture is freed
    /// with [`free_texture`](Self::free_texture). Plugins that build a bind
    /// group from this view must rebuild it after any operation that could
    /// invalidate the texture (a free, device recreation). A safer pattern is to
    /// fetch the view each frame just before building / rebuilding the bind
    /// group.
    pub fn texture_view(&self, id: crate::resources::TextureId) -> Option<&wgpu::TextureView> {
        self.textures.get(id).map(|t| &t.view)
    }

    /// Borrow the sampler the texture was uploaded with.
    ///
    /// Most user textures are uploaded with a shared linear-repeat sampler;
    /// prefer [`material_sampler`](Self::material_sampler) when you need
    /// the shared lib sampler rather than the per-texture instance.
    pub fn texture_sampler(&self, id: crate::resources::TextureId) -> Option<&wgpu::Sampler> {
        self.textures.get(id).map(|t| &t.sampler)
    }

    /// Shared linear-repeat sampler used by the lib's material pipelines.
    ///
    /// Use this when building a plugin bind group that samples user
    /// textures the same way `Material` does (linear filter, repeat wrap).
    pub fn material_sampler(&self) -> &wgpu::Sampler {
        &self.material_sampler
    }

    /// Shared linear-clamp sampler used by the lib for colormap LUTs.
    ///
    /// Use this when sampling 1D LUT-style data (colourmaps, transfer
    /// functions) where the texture should not wrap.
    pub fn lut_sampler(&self) -> &wgpu::Sampler {
        &self.lut_sampler
    }

    /// Comparison sampler used for PCF shadow filtering.
    ///
    /// Plugins that sample the shadow atlas directly (rather than through
    /// `viewport_sample_csm`) use this sampler when binding the atlas.
    pub fn shadow_filter_sampler(&self) -> &wgpu::Sampler {
        &self.shadow_sampler
    }

    /// Bind group layout for the per-vertex deformation sidecar.
    ///
    /// Plugins building pipelines that draw meshes with registered deformers
    /// add this layout at group 2 so their vertex stage can read from the
    /// shared `deform_data` / `deform_instance_data` storage buffers.
    pub fn deform_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.deform.bind_group_layout
    }

    /// Number of live user-uploaded textures.
    ///
    /// `id` values in `0..texture_count()` are addressable via
    /// [`texture_view`](Self::texture_view), with the caveat that promoted
    /// IDs from async uploads may sit at the high end.
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    // ------------------------------------------------------------------
    // Pipeline builders
    // ------------------------------------------------------------------

    /// Build an opaque scene pipeline that draws into the HDR scene pass.
    ///
    /// Standard depth state: `LessEqual` test, depth write on. The pipeline
    /// layout lists [`shared_bindings`](Self::shared_bindings) as group 0,
    /// then `extra_bind_group_layouts` as groups 1.., in order. The plugin
    /// owns all groups past 0.
    pub fn build_opaque_pipeline(
        &self,
        device: &wgpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> wgpu::RenderPipeline {
        let layout = build_layout(device, opts.label, self, opts.extra_bind_group_layouts);
        let desc = self.opaque_target_desc();
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: opts.label,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: opts.shader,
                entry_point: Some(opts.vs_entry),
                buffers: opts.vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: opts.shader,
                entry_point: Some(opts.fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: desc.color_format,
                    blend: opts.color_blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: opts.primitive,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: desc.depth_format,
                depth_write_enabled: opts.depth_write,
                depth_compare: opts.depth_compare,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: desc.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        })
    }

    /// Build a transparent pipeline that draws into the OIT pass.
    ///
    /// The fragment shader must return [`OitOutput`](crate::plugin_api::shared_wgsl::SHARED_OIT_WGSL),
    /// writing both `@location(0)` (accum) and `@location(1)` (reveal).
    /// Depth state: `LessEqual` test, depth write off.
    pub fn build_oit_pipeline(
        &self,
        device: &wgpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> wgpu::RenderPipeline {
        let layout = build_layout(device, opts.label, self, opts.extra_bind_group_layouts);
        let desc = self.oit_target_desc();
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: opts.label,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: opts.shader,
                entry_point: Some(opts.vs_entry),
                buffers: opts.vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: opts.shader,
                entry_point: Some(opts.fs_entry),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: desc.accum_format,
                        blend: Some(desc.accum_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: desc.reveal_format,
                        blend: Some(desc.reveal_blend),
                        write_mask: wgpu::ColorWrites::RED,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: opts.primitive,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: desc.depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: desc.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        })
    }

    /// Build a pipeline for the outline-mask pass (R8 target).
    ///
    /// Fragment shader must write `1.0` at `@location(0)` for any covered
    /// pixel; use [`SHARED_MASK_WGSL`](crate::plugin_api::shared_wgsl::SHARED_MASK_WGSL).
    /// Depth state: `LessEqual` test, no depth write.
    pub fn build_mask_pipeline(
        &self,
        device: &wgpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> wgpu::RenderPipeline {
        let layout = build_layout(device, opts.label, self, opts.extra_bind_group_layouts);
        let desc = self.mask_target_desc();
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: opts.label,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: opts.shader,
                entry_point: Some(opts.vs_entry),
                buffers: opts.vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: opts.shader,
                entry_point: Some(opts.fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: desc.color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::RED,
                })],
                compilation_options: Default::default(),
            }),
            primitive: opts.primitive,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: desc.depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: desc.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        })
    }

    /// Build a pipeline for the pick-id pass (R32Uint target).
    ///
    /// Fragment shader must write the item's `PickId` value at
    /// `@location(0)`; use [`SHARED_PICK_WGSL`](crate::plugin_api::shared_wgsl::SHARED_PICK_WGSL).
    pub fn build_pick_pipeline(
        &self,
        device: &wgpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> wgpu::RenderPipeline {
        let layout = build_layout(device, opts.label, self, opts.extra_bind_group_layouts);
        let desc = self.pick_target_desc();
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: opts.label,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: opts.shader,
                entry_point: Some(opts.vs_entry),
                buffers: opts.vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: opts.shader,
                entry_point: Some(opts.fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: desc.color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::RED,
                })],
                compilation_options: Default::default(),
            }),
            primitive: opts.primitive,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: desc.depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: desc.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        })
    }

    /// Build a depth-only pipeline for the shadow-atlas pass.
    ///
    /// No fragment output. The fragment entry is optional; pass an empty
    /// string to use a depth-only configuration with no fragment stage.
    /// Standard depth state: `LessEqual` test, depth write on, with the
    /// lib's standard depth bias.
    pub fn build_shadow_pipeline(
        &self,
        device: &wgpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> wgpu::RenderPipeline {
        let layout = build_layout(device, opts.label, self, opts.extra_bind_group_layouts);
        let desc = self.shadow_target_desc();
        let fragment = if opts.fs_entry.is_empty() {
            None
        } else {
            Some(wgpu::FragmentState {
                module: opts.shader,
                entry_point: Some(opts.fs_entry),
                targets: &[],
                compilation_options: Default::default(),
            })
        };
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: opts.label,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: opts.shader,
                entry_point: Some(opts.vs_entry),
                buffers: opts.vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment,
            primitive: opts.primitive,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: desc.depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: desc.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        })
    }
}

/// Inputs to a plugin pipeline builder. All builders take this struct; the
/// builder picks the target descriptor and blend state.
pub struct PluginPipelineOpts<'a> {
    /// Pipeline debug label. Forwarded to wgpu.
    pub label: Option<&'a str>,
    /// Shader module containing both the vertex and fragment entry points.
    pub shader: &'a wgpu::ShaderModule,
    /// Vertex-stage entry-point name (e.g. `"vs_main"`).
    pub vs_entry: &'a str,
    /// Fragment-stage entry-point name (e.g. `"fs_main"`). For
    /// `build_shadow_pipeline`, pass `""` to skip the fragment stage.
    pub fs_entry: &'a str,
    /// Vertex buffer layouts.
    pub vertex_layouts: &'a [wgpu::VertexBufferLayout<'a>],
    /// Bind group layouts for groups 1.. (the plugin's per-object data).
    /// Group 0 is supplied automatically from
    /// [`ViewportGpuResources::shared_bindings`].
    pub extra_bind_group_layouts: &'a [&'a wgpu::BindGroupLayout],
    /// Primitive topology, cull mode, polygon mode.
    pub primitive: wgpu::PrimitiveState,
    /// Optional blend state for the opaque builder. `None` = no blending
    /// (the default for opaque pipelines). Ignored by OIT / mask / pick /
    /// shadow builders, which use their pass-specific blend state.
    pub color_blend: Option<wgpu::BlendState>,
    /// Whether the opaque builder writes depth. Ignored by the other
    /// builders. Default `true`.
    pub depth_write: bool,
    /// Depth-compare function for the opaque builder. Ignored by the other
    /// builders.
    pub depth_compare: wgpu::CompareFunction,
}

impl<'a> PluginPipelineOpts<'a> {
    /// Construct an opts struct with sensible defaults for the variable
    /// fields (`TriangleList`, back-face cull, depth write on,
    /// `LessEqual`, no blend). Callers must supply `shader`, vertex / fragment
    /// entry points, and the vertex layout.
    pub fn new(
        label: Option<&'a str>,
        shader: &'a wgpu::ShaderModule,
        vs_entry: &'a str,
        fs_entry: &'a str,
        vertex_layouts: &'a [wgpu::VertexBufferLayout<'a>],
    ) -> Self {
        Self {
            label,
            shader,
            vs_entry,
            fs_entry,
            vertex_layouts,
            extra_bind_group_layouts: &[],
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            color_blend: None,
            depth_write: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
        }
    }
}

fn build_layout(
    device: &wgpu::Device,
    label: Option<&str>,
    res: &ViewportGpuResources,
    extras: &[&wgpu::BindGroupLayout],
) -> wgpu::PipelineLayout {
    let mut bgls: Vec<&wgpu::BindGroupLayout> = Vec::with_capacity(1 + extras.len());
    bgls.push(&res.camera_bind_group_layout);
    bgls.extend(extras.iter().copied());
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label,
        bind_group_layouts: &bgls,
        push_constant_ranges: &[],
    })
}
