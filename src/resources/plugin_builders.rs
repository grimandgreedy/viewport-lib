//! Plugin-facing accessors and pipeline builders on [`DeviceResources`].
//!
//! See [`crate::plugin_api`] for the published types these methods return.

use crate::plugin_api::{
    DepthReadTargetDesc, ForegroundTargetDesc, MaskTargetDesc, OitTargetDesc, OpaqueTargetDesc,
    PickTargetDesc, ShadowTargetDesc, SharedBindings,
    target_desc::{OIT_ACCUM_BLEND, OIT_REVEAL_BLEND},
};
use crate::resources::DeviceResources;

/// HDR colour format used by the scene buffer. Plugins targeting the HDR
/// path build pipelines against this format.
pub const HDR_COLOR_FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::Rgba16Float;

/// Depth-stencil format shared by every scene render pass.
pub const SCENE_DEPTH_FORMAT: crate::gpu::TextureFormat =
    crate::gpu::TextureFormat::Depth24PlusStencil8;

/// Shadow atlas depth format.
pub const SHADOW_DEPTH_FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::Depth32Float;

/// Outline-mask colour format.
pub const MASK_COLOR_FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::R8Unorm;

/// Pick-id colour format, shared by the object-id (`@location(0)`) and
/// primitive-id (`@location(1)`) pick targets.
pub const PICK_COLOR_FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::R32Uint;

/// Pick depth-channel format (`@location(2)`): the fragment framebuffer `z`
/// written as a float so the renderer can reconstruct world position on
/// read-back.
pub const PICK_DEPTH_CHANNEL_FORMAT: crate::gpu::TextureFormat =
    crate::gpu::TextureFormat::R32Float;

impl DeviceResources {
    // ------------------------------------------------------------------
    // Target descriptors and SharedBindings accessor
    // ------------------------------------------------------------------

    /// Group-0 bind layout shared by every scene pipeline. Use as group 0
    /// when building a plugin pipeline layout.
    pub fn shared_bindings(&self) -> SharedBindings<'_> {
        SharedBindings {
            group0_layout: &self.binds.camera_bgl,
            sample_count: self.sample_count,
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

    /// Render-target descriptor for the foreground pass.
    pub fn foreground_target_desc(&self) -> ForegroundTargetDesc {
        ForegroundTargetDesc {
            color_format: HDR_COLOR_FORMAT,
            depth_format: SCENE_DEPTH_FORMAT,
            sample_count: 1,
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

    /// Render-target descriptor for the read-only-depth plugin pass.
    pub fn depth_read_target_desc(&self) -> DepthReadTargetDesc {
        DepthReadTargetDesc {
            color_format: HDR_COLOR_FORMAT,
            depth_format: SCENE_DEPTH_FORMAT,
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
            object_id_format: PICK_COLOR_FORMAT,
            primitive_id_format: PICK_COLOR_FORMAT,
            depth_channel_format: PICK_DEPTH_CHANNEL_FORMAT,
            depth_stencil_format: SCENE_DEPTH_FORMAT,
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
    pub fn texture_view(
        &self,
        id: crate::resources::TextureId,
    ) -> Option<&crate::gpu::TextureView> {
        self.content.textures.get(id).map(|t| &t.view)
    }

    /// Borrow the sampler the texture was uploaded with.
    ///
    /// Most user textures are uploaded with a shared linear-repeat sampler;
    /// prefer [`material_sampler`](Self::material_sampler) when you need
    /// the shared lib sampler rather than the per-texture instance.
    pub fn texture_sampler(&self, id: crate::resources::TextureId) -> Option<&crate::gpu::Sampler> {
        self.content.textures.get(id).map(|t| &t.sampler)
    }

    /// Shared linear-repeat sampler used by the lib's material pipelines.
    ///
    /// Use this when building a plugin bind group that samples user
    /// textures the same way `Material` does (linear filter, repeat wrap).
    pub fn material_sampler(&self) -> &crate::gpu::Sampler {
        &self.material.sampler
    }

    /// Non-filtering clamp sampler for the read-only-depth pass.
    ///
    /// Pair it with the scene depth-only view to build a bind group matching
    /// [`depth_read_bind_group_layout`](Self::depth_read_bind_group_layout).
    /// The renderer already builds that bind group each frame and hands it
    /// over as
    /// [`DepthReadContext::scene_depth_bind_group`](crate::plugin_api::DepthReadContext::scene_depth_bind_group);
    /// use this only when building your own.
    pub fn depth_read_sampler(&self) -> &crate::gpu::Sampler {
        &self.material.depth_read_sampler
    }

    /// Bind group layout for the read-only-depth pass: binding 0 is the scene
    /// depth texture (`texture_depth_2d`), binding 1 the non-filtering sampler.
    /// Matches [`DepthReadContext::scene_depth_bind_group`](crate::plugin_api::DepthReadContext::scene_depth_bind_group).
    ///
    /// A convenience for plugins that have a spare bind group: place this at
    /// whatever free slot of `extra_bind_group_layouts` when building the
    /// pipeline with [`build_depth_read_pipeline`](Self::build_depth_read_pipeline),
    /// declare the matching `@group(K) @binding(0/1)` in the shader, and bind
    /// the ready-made bind group there. A plugin already using all four bind
    /// groups skips this and folds the depth texture + sampler into an existing
    /// group instead (see
    /// [`SHARED_DEPTH_READ_WGSL`](crate::plugin_api::shared_wgsl::SHARED_DEPTH_READ_WGSL)).
    pub fn depth_read_bind_group_layout(&self) -> &crate::gpu::BindGroupLayout {
        &self.material.depth_read_bgl
    }

    /// Shared linear-clamp sampler used by the lib for colormap LUTs.
    ///
    /// Use this when sampling 1D LUT-style data (colourmaps, transfer
    /// functions) where the texture should not wrap.
    pub fn lut_sampler(&self) -> &crate::gpu::Sampler {
        &self.material.lut_sampler
    }

    /// Comparison sampler used for PCF shadow filtering.
    ///
    /// Plugins that sample the shadow atlas directly (rather than through
    /// `viewport_sample_csm`) use this sampler when binding the atlas.
    pub fn shadow_filter_sampler(&self) -> &crate::gpu::Sampler {
        &self.shadow.sampler
    }

    /// Bind group layout for the per-vertex deformation sidecar.
    ///
    /// Plugins building pipelines that draw meshes with registered deformers
    /// add this layout at group 2 so their vertex stage can read from the
    /// shared `deform_data` / `deform_instance_data` storage buffers.
    pub fn deform_bind_group_layout(&self) -> &crate::gpu::BindGroupLayout {
        &self.deform.bind_group_layout
    }

    /// Number of live user-uploaded textures.
    ///
    /// `id` values in `0..texture_count()` are addressable via
    /// [`texture_view`](Self::texture_view), with the caveat that promoted
    /// IDs from async uploads may sit at the high end.
    pub fn texture_count(&self) -> usize {
        self.content.textures.len()
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
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(device, opts.label, self, opts.extra_bind_group_layouts);
        let desc = self.opaque_target_desc();
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: opts.shader,
                    entry_point: Some(opts.vs_entry),
                    buffers: opts.vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: opts.shader,
                    entry_point: Some(opts.fs_entry),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: desc.color_format,
                        blend: opts.color_blend,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: opts.primitive,
                depth_stencil: Some(crate::resources::builders::depth_stencil(
                    desc.depth_format,
                    opts.depth_write,
                    opts.depth_compare,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }

    /// Build a pipeline that draws into the foreground pass.
    ///
    /// Same shape as [`build_opaque_pipeline`](Self::build_opaque_pipeline)
    /// (group 0 = shared bindings, then `extra_bind_group_layouts`), but
    /// single-sampled: the foreground pass runs after the SSAA resolve. The
    /// bound group-0 camera carries the foreground projection and disabled
    /// clip planes; depth is tested against the pass's own cleared target.
    pub fn build_foreground_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(device, opts.label, self, opts.extra_bind_group_layouts);
        let desc = self.foreground_target_desc();
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: opts.shader,
                    entry_point: Some(opts.vs_entry),
                    buffers: opts.vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: opts.shader,
                    entry_point: Some(opts.fs_entry),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: desc.color_format,
                        blend: opts.color_blend,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: opts.primitive,
                depth_stencil: Some(crate::resources::builders::depth_stencil(
                    desc.depth_format,
                    opts.depth_write,
                    opts.depth_compare,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }

    /// Build a transparent pipeline that draws into the OIT pass.
    ///
    /// The fragment shader must return [`OitOutput`](crate::plugin_api::shared_wgsl::SHARED_OIT_WGSL),
    /// writing both `@location(0)` (accum) and `@location(1)` (reveal).
    /// Depth state: `LessEqual` test, depth write off.
    pub fn build_oit_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(device, opts.label, self, opts.extra_bind_group_layouts);
        let desc = self.oit_target_desc();
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: opts.shader,
                    entry_point: Some(opts.vs_entry),
                    buffers: opts.vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: opts.shader,
                    entry_point: Some(opts.fs_entry),
                    targets: &[
                        Some(crate::gpu::ColorTargetState {
                            format: desc.accum_format,
                            blend: Some(desc.accum_blend),
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                        Some(crate::gpu::ColorTargetState {
                            format: desc.reveal_format,
                            blend: Some(desc.reveal_blend),
                            write_mask: crate::gpu::ColorWrites::RED,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: opts.primitive,
                depth_stencil: Some(crate::resources::builders::depth_stencil(
                    desc.depth_format,
                    false,
                    crate::gpu::CompareFunction::LessEqual,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }

    /// Build a pipeline that draws into the read-only-depth pass.
    ///
    /// One colour target (the HDR scene buffer) with the caller's blend state,
    /// and the scene depth attachment bound read-only: the pipeline tests
    /// against opaque depth (`opts.depth_compare`, `LessEqual` by default) but
    /// never writes it, since the pass binds depth read-only. Set
    /// `opts.color_blend` to alpha blending for soft particles.
    ///
    /// The plugin lists its own bind group layouts in
    /// `opts.extra_bind_group_layouts` as usual. The scene depth read is not a
    /// fixed group: the plugin either adds
    /// [`depth_read_bind_group_layout`](Self::depth_read_bind_group_layout) at a
    /// spare slot, or folds the two depth bindings into one of its existing
    /// layouts. It reconstructs depth through
    /// [`SHARED_DEPTH_READ_WGSL`](crate::plugin_api::shared_wgsl::SHARED_DEPTH_READ_WGSL).
    pub fn build_depth_read_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(device, opts.label, self, opts.extra_bind_group_layouts);
        let desc = self.depth_read_target_desc();
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: opts.shader,
                    entry_point: Some(opts.vs_entry),
                    buffers: opts.vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: opts.shader,
                    entry_point: Some(opts.fs_entry),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: desc.color_format,
                        blend: opts.color_blend,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: opts.primitive,
                depth_stencil: Some(crate::resources::builders::depth_stencil(
                    desc.depth_format,
                    false,
                    opts.depth_compare,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }

    /// Build a pipeline for the outline-mask pass (R8 target).
    ///
    /// Fragment shader must write `1.0` at `@location(0)` for any covered
    /// pixel; use [`SHARED_MASK_WGSL`](crate::plugin_api::shared_wgsl::SHARED_MASK_WGSL).
    /// Depth state: `LessEqual` test, no depth write.
    pub fn build_mask_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(device, opts.label, self, opts.extra_bind_group_layouts);
        let desc = self.mask_target_desc();
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: opts.shader,
                    entry_point: Some(opts.vs_entry),
                    buffers: opts.vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: opts.shader,
                    entry_point: Some(opts.fs_entry),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: desc.color_format,
                        blend: None,
                        write_mask: crate::gpu::ColorWrites::RED,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: opts.primitive,
                depth_stencil: Some(crate::resources::builders::depth_stencil(
                    desc.depth_format,
                    false,
                    crate::gpu::CompareFunction::LessEqual,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }

    /// Build a pipeline for the pick-id pass.
    ///
    /// The pass has three colour targets (object id, primitive id, depth) plus a
    /// depth-stencil attachment; this matches the pipeline to all of them. The
    /// fragment shader must write all three: the item's `PickId` at
    /// `@location(0)`, a sub-object index (or 0) at `@location(1)`, and the
    /// framebuffer `z` at `@location(2)`. Use
    /// [`SHARED_PICK_WGSL`](crate::plugin_api::shared_wgsl::SHARED_PICK_WGSL),
    /// whose `viewport_pick_fs` produces exactly that output.
    pub fn build_pick_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(device, opts.label, self, opts.extra_bind_group_layouts);
        let desc = self.pick_target_desc();
        // Integer and float single-channel targets, no blending: a fragment
        // either writes an exact id/depth or leaves the attachment at its clear
        // value. Order and formats mirror the internal pick pipeline.
        let color_target = |format| {
            Some(crate::gpu::ColorTargetState {
                format,
                blend: None,
                write_mask: crate::gpu::ColorWrites::ALL,
            })
        };
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: opts.shader,
                    entry_point: Some(opts.vs_entry),
                    buffers: opts.vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: opts.shader,
                    entry_point: Some(opts.fs_entry),
                    targets: &[
                        color_target(desc.object_id_format),
                        color_target(desc.primitive_id_format),
                        color_target(desc.depth_channel_format),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: opts.primitive,
                depth_stencil: Some(crate::resources::builders::depth_stencil(
                    desc.depth_stencil_format,
                    true,
                    crate::gpu::CompareFunction::LessEqual,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }

    /// Build a depth-only pipeline for the shadow-atlas pass.
    ///
    /// No fragment output. The fragment entry is optional; pass an empty
    /// string to use a depth-only configuration with no fragment stage.
    /// Standard depth state: `LessEqual` test, depth write on, with the
    /// lib's standard depth bias.
    pub fn build_shadow_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(device, opts.label, self, opts.extra_bind_group_layouts);
        let desc = self.shadow_target_desc();
        let fragment = if opts.fs_entry.is_empty() {
            None
        } else {
            Some(crate::gpu::FragmentState {
                module: opts.shader,
                entry_point: Some(opts.fs_entry),
                targets: &[],
                compilation_options: Default::default(),
            })
        };
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: opts.shader,
                    entry_point: Some(opts.vs_entry),
                    buffers: opts.vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment,
                primitive: opts.primitive,
                depth_stencil: Some(crate::gpu::DepthStencilState {
                    format: desc.depth_format,
                    depth_write_enabled: crate::resources::builders::dwrite(true),
                    depth_compare: crate::resources::builders::dcompare(
                        crate::gpu::CompareFunction::LessEqual,
                    ),
                    stencil: crate::gpu::StencilState::default(),
                    bias: crate::gpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }
}

/// Inputs to a plugin pipeline builder. All builders take this struct; the
/// builder picks the target descriptor and blend state.
pub struct PluginPipelineOpts<'a> {
    /// Pipeline debug label. Forwarded to wgpu.
    pub label: Option<&'a str>,
    /// Shader module containing both the vertex and fragment entry points.
    pub shader: &'a crate::gpu::ShaderModule,
    /// Vertex-stage entry-point name (e.g. `"vs_main"`).
    pub vs_entry: &'a str,
    /// Fragment-stage entry-point name (e.g. `"fs_main"`). For
    /// `build_shadow_pipeline`, pass `""` to skip the fragment stage.
    pub fs_entry: &'a str,
    /// Vertex buffer layouts.
    pub vertex_layouts: &'a [crate::gpu::VertexBufferLayout<'a>],
    /// Bind group layouts for groups 1.. (the plugin's per-object data).
    /// Group 0 is supplied automatically from
    /// [`DeviceResources::shared_bindings`].
    pub extra_bind_group_layouts: &'a [&'a crate::gpu::BindGroupLayout],
    /// Primitive topology, cull mode, polygon mode.
    pub primitive: crate::gpu::PrimitiveState,
    /// Optional blend state for the opaque builder. `None` = no blending
    /// (the default for opaque pipelines). Ignored by OIT / mask / pick /
    /// shadow builders, which use their pass-specific blend state.
    pub color_blend: Option<crate::gpu::BlendState>,
    /// Whether the opaque builder writes depth. Ignored by the other
    /// builders. Default `true`.
    pub depth_write: bool,
    /// Depth-compare function for the opaque builder. Ignored by the other
    /// builders.
    pub depth_compare: crate::gpu::CompareFunction,
}

impl<'a> PluginPipelineOpts<'a> {
    /// Construct an opts struct with sensible defaults for the variable
    /// fields (`TriangleList`, back-face cull, depth write on,
    /// `LessEqual`, no blend). Callers must supply `shader`, vertex / fragment
    /// entry points, and the vertex layout.
    pub fn new(
        label: Option<&'a str>,
        shader: &'a crate::gpu::ShaderModule,
        vs_entry: &'a str,
        fs_entry: &'a str,
        vertex_layouts: &'a [crate::gpu::VertexBufferLayout<'a>],
    ) -> Self {
        Self {
            label,
            shader,
            vs_entry,
            fs_entry,
            vertex_layouts,
            extra_bind_group_layouts: &[],
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(crate::gpu::Face::Back),
                ..Default::default()
            },
            color_blend: None,
            depth_write: true,
            depth_compare: crate::gpu::CompareFunction::LessEqual,
        }
    }
}

fn build_layout(
    device: &crate::gpu::Device,
    label: Option<&str>,
    res: &DeviceResources,
    extras: &[&crate::gpu::BindGroupLayout],
) -> crate::gpu::PipelineLayout {
    let mut bgls: Vec<&crate::gpu::BindGroupLayout> = Vec::with_capacity(1 + extras.len());
    bgls.push(&res.binds.camera_bgl);
    bgls.extend(extras.iter().copied());
    crate::resources::builders::pipeline_layout(device, label, &bgls)
}
