//! Dynamic resolution render target for the LDR render path.
//!
//! When `render_scale < 1.0`, the scene is rendered into a scaled intermediate
//! texture and then upscaled to the surface via bilinear filtering.

use crate::resources::DeviceResources;

/// Per-viewport intermediate render target for dynamic resolution rendering.
///
/// Owned by the viewport slot; created or recreated whenever the render scale
/// or surface size changes.
pub(crate) struct DynResTarget {
    /// Scaled colour texture (render_scale x surface_size).
    pub _colour_texture: crate::gpu::Texture,
    /// View of `colour_texture`.
    pub colour_view: crate::gpu::TextureView,
    /// Depth texture matching the scaled resolution.
    pub _depth_texture: crate::gpu::Texture,
    /// View of `depth_texture`.
    pub depth_view: crate::gpu::TextureView,
    /// Depth-aspect view for sampling (HiZ occlusion prev-depth copy).
    pub depth_only_view: crate::gpu::TextureView,
    /// Bind group for the upscale pass: colour_texture + linear sampler.
    pub upscale_bind_group: crate::gpu::BindGroup,
    /// Dimensions of the intermediate target `[w, h]`.
    pub scaled_size: [u32; 2],
    /// Native surface dimensions this target was created for `[w, h]`.
    pub surface_size: [u32; 2],
}

impl DeviceResources {
    /// Ensure the shared upscale pipeline and sampler exist, creating them on
    /// first call. Idempotent.
    pub(crate) fn ensure_dyn_res_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.post.dyn_res_upscale_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "dyn_res_upscale_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );

        let sampler =
            crate::resources::builders::clamp_linear_sampler(device, "dyn_res_linear_sampler");

        let shader = crate::resources::builders::wgsl_module(
            device,
            "dyn_res_upscale_shader",
            crate::resources::builders::wgsl_source!("dyn_res_upscale"),
        );

        let layout =
            crate::resources::builders::pipeline_layout(device, "dyn_res_upscale_layout", &[&bgl]);

        let pipeline = crate::resources::builders::build_fullscreen_pipeline(
            device,
            "dyn_res_upscale_pipeline",
            &layout,
            &shader,
            self.target_format,
            None,
        );

        self.post.dyn_res_upscale_bgl = Some(bgl);
        self.post.dyn_res_upscale_pipeline = Some(pipeline);
        self.post.dyn_res_linear_sampler = Some(sampler);
    }

    /// Ensure the depth-stencil compatible upscale pipeline exists for use inside
    /// eframe's paint render pass, which always has a `Depth24PlusStencil8` attachment.
    ///
    /// Identical to [`ensure_dyn_res_pipeline`](Self::ensure_dyn_res_pipeline) except
    /// `depth_stencil` is set to read-only `Depth24PlusStencil8` so the pipeline is
    /// compatible with any render pass that carries that depth attachment.
    /// [`ensure_dyn_res_pipeline`](Self::ensure_dyn_res_pipeline) must be called first.
    pub(crate) fn ensure_dyn_res_ds_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.post.dyn_res_upscale_ds_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let bgl = self
            .post
            .dyn_res_upscale_bgl
            .as_ref()
            .expect("ensure_dyn_res_pipeline must be called before ensure_dyn_res_ds_pipeline");
        let shader = crate::resources::builders::wgsl_module(
            device,
            "dyn_res_upscale_ds_shader",
            crate::resources::builders::wgsl_source!("dyn_res_upscale"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "dyn_res_upscale_ds_layout",
            &[bgl],
        );
        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "dyn_res_upscale_ds_pipeline",
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: None,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Always,
                )),
                multisample: crate::gpu::MultisampleState::default(),
                cache: None,
            },
        );
        self.post.dyn_res_upscale_ds_pipeline = Some(pipeline);
    }

    /// Create a [`DynResTarget`] at `scaled_size`, bound for upscaling to
    /// `surface_size`. The shared pipeline must already exist (call
    /// [`ensure_dyn_res_pipeline`](Self::ensure_dyn_res_pipeline) first).
    pub(crate) fn create_dyn_res_target(
        &self,
        device: &crate::gpu::Device,
        scaled_size: [u32; 2],
        surface_size: [u32; 2],
    ) -> DynResTarget {
        let [sw, sh] = scaled_size;

        let colour_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("dyn_res_colour"),
            size: crate::gpu::Extent3d {
                width: sw,
                height: sh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: self.target_format,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let colour_view = colour_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("dyn_res_depth"),
            size: crate::gpu::Extent3d {
                width: sw,
                height: sh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Depth24PlusStencil8,
            // TEXTURE_BINDING so the HiZ occlusion prev-depth copy can sample it.
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        let depth_only_view = depth_texture.create_view(&crate::gpu::TextureViewDescriptor {
            aspect: crate::gpu::TextureAspect::DepthOnly,
            ..Default::default()
        });

        let bgl = self.post.dyn_res_upscale_bgl.as_ref().unwrap();
        let sampler = self.post.dyn_res_linear_sampler.as_ref().unwrap();
        let upscale_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("dyn_res_upscale_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&colour_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        DynResTarget {
            _colour_texture: colour_texture,
            colour_view,
            _depth_texture: depth_texture,
            depth_view,
            depth_only_view,
            upscale_bind_group,
            scaled_size,
            surface_size,
        }
    }

    /// Create a [`HdrCallbackTarget`] at `size` for use with the eframe HDR callback path.
    ///
    /// The shared pipeline and sampler must already exist. I.e., call
    /// [`ensure_dyn_res_pipeline`](Self::ensure_dyn_res_pipeline) first.
    pub(crate) fn create_hdr_callback_target(
        &self,
        device: &crate::gpu::Device,
        size: [u32; 2],
    ) -> HdrCallbackTarget {
        let [w, h] = size;
        let texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("hdr_callback_target"),
            size: crate::gpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: self.target_format,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let blit_bind_group = {
            let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
            let bgl = self.post.dyn_res_upscale_bgl.as_ref().unwrap();
            let sampler = self.post.dyn_res_linear_sampler.as_ref().unwrap();
            device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("hdr_callback_blit_bg"),
                layout: bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::gpu::BindingResource::TextureView(&view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        };

        HdrCallbackTarget {
            texture,
            blit_bind_group,
            size,
        }
    }
}

/// Per-viewport intermediate render target for the HDR eframe callback path.
///
/// Allocated when [`prepare_hdr_callback`](crate::ViewportRenderer::prepare_hdr_callback)
/// is first called for a viewport and recreated when the viewport size changes.
/// The full HDR pipeline (OIT, EDL, tone-map) renders into `texture`; `blit_bind_group`
/// is then used by
/// [`paint_hdr_blit`](crate::ViewportRenderer::paint_hdr_blit) to copy the result
/// into the egui render pass.
pub(crate) struct HdrCallbackTarget {
    /// Intermediate LDR colour texture (same format as `target_format`).
    ///
    /// Stored so we can create a fresh `TextureView` each frame inside
    /// `prepare_hdr_callback`, avoiding a simultaneous mutable + immutable borrow.
    pub texture: crate::gpu::Texture,
    /// Bind group for the blit pass: `texture` view + linear sampler.
    pub blit_bind_group: crate::gpu::BindGroup,
    /// Dimensions `[w, h]`.
    pub size: [u32; 2],
}
