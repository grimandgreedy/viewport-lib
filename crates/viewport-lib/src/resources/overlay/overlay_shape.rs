//! Lazy pipeline creation for SDF overlay shapes.

use crate::renderer::OverlayTextureId;

/// SDF overlay shape pipelines (the instanced shape pipeline and the backdrop
/// pipeline), the fill layout they share with the text pipeline, and their
/// samplers. Lazily built on the first frame with non-empty shapes.
#[derive(Default)]
pub(crate) struct OverlayShapeResources {
    /// Render pipeline for screen-space SDF shapes (rounded rects, circles,
    /// etc.), whatever their fill: one instance each.
    pub(crate) pipeline: Option<crate::gpu::RenderPipeline>,
    /// The shape pipeline for each blend mode other than source-over that a
    /// frame has used.
    pub(crate) variants: OverlayBlendVariants,
    /// The shape pipeline's layout and shader, kept to build blend variants.
    pub(crate) layout: Option<crate::gpu::PipelineLayout>,
    pub(crate) shader: Option<crate::gpu::ShaderModule>,
    /// Bind group layout for the shape pipeline's group 0: shadow layers,
    /// clip shapes, viewport and group instance.
    pub(crate) shadow_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Render pipeline for backdrop-blur shapes: an analytic shape's quad or a
    /// vector path's triangles, sampling the blurred scene.
    pub(crate) tex_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Bind group layout for the backdrop pipeline (group 0: the blurred scene
    /// + sampler).
    pub(crate) tex_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Bind group layout for the backdrop pipeline's clip masks (group 1: the
    /// clip-shape uniform, shared with the shape pipeline's binding 1).
    pub(crate) tex_clip_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Clamp-to-edge linear sampler for the backdrop bind groups.
    pub(crate) tex_sampler: Option<crate::gpu::Sampler>,
    /// Group 1 of the shape and text pipelines, which the shared fill module
    /// reads: the fill records, the image a texture fill samples, and its
    /// sampler.
    pub(crate) fill_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Clamp-to-edge linear sampler for texture fills.
    pub(crate) fill_sampler: Option<crate::gpu::Sampler>,
    /// One white texel, bound as the fill image for every draw that samples
    /// no texture.
    pub(crate) blank_image: Option<crate::gpu::TextureView>,
}

impl OverlayShapeResources {
    /// The shape pipeline that draws with `blend`, if it has been built.
    pub(crate) fn pipeline_for(
        &self,
        blend: crate::renderer::OverlayBlend,
    ) -> Option<&crate::gpu::RenderPipeline> {
        if blend == crate::renderer::OverlayBlend::SourceOver {
            self.pipeline.as_ref()
        } else {
            self.variants.get(blend)
        }
    }
}

/// Overlay pipelines for the blend modes other than source-over, keyed by
/// mode. A frame builds the ones it uses; there are few enough modes that a
/// linear lookup is the cheapest.
#[derive(Default)]
pub(crate) struct OverlayBlendVariants(
    Vec<(crate::renderer::OverlayBlend, crate::gpu::RenderPipeline)>,
);

impl OverlayBlendVariants {
    pub(crate) fn get(
        &self,
        blend: crate::renderer::OverlayBlend,
    ) -> Option<&crate::gpu::RenderPipeline> {
        self.0.iter().find(|(b, _)| *b == blend).map(|(_, p)| p)
    }

    pub(crate) fn insert(
        &mut self,
        blend: crate::renderer::OverlayBlend,
        pipeline: crate::gpu::RenderPipeline,
    ) {
        self.0.push((blend, pipeline));
    }
}

/// The colour-target blend state and fragment entry point that draw an overlay
/// pipeline with `blend`.
///
/// Source-over keeps the overlay pass's straight-alpha state: the fragment
/// writes straight colour and the blend multiplies by its alpha. Every other
/// mode draws through `fs_premul`, which writes premultiplied colour, because
/// a mode such as source-in needs the source scaled by its own alpha and by
/// the destination alpha, and one blend factor cannot express both.
///
/// The destination is premultiplied in every case.
pub(crate) fn overlay_blend_target(
    blend: crate::renderer::OverlayBlend,
) -> (crate::gpu::BlendState, &'static str) {
    use crate::gpu::BlendFactor as F;
    use crate::renderer::OverlayBlend as B;
    let comp = |src_factor, dst_factor| crate::gpu::BlendComponent {
        src_factor,
        dst_factor,
        operation: crate::gpu::BlendOperation::Add,
    };
    // Porter-Duff: the source weighted by `fa` and the destination by `fb`,
    // the same on colour and alpha.
    let pd = |fa, fb| crate::gpu::BlendState {
        color: comp(fa, fb),
        alpha: comp(fa, fb),
    };
    let state = match blend {
        B::Additive => crate::gpu::BlendState {
            color: comp(F::One, F::One),
            alpha: comp(F::Zero, F::One),
        },
        // Sc * Dc + Dc * (1 - Sa). Missing the Sc * (1 - Da) term, which is
        // zero over an opaque destination.
        B::Multiply => crate::gpu::BlendState {
            color: comp(F::Dst, F::OneMinusSrcAlpha),
            alpha: comp(F::One, F::OneMinusSrcAlpha),
        },
        // Sc + Dc - Sc * Dc.
        B::Screen => crate::gpu::BlendState {
            color: comp(F::One, F::OneMinusSrc),
            alpha: comp(F::One, F::OneMinusSrcAlpha),
        },
        B::Clear => pd(F::Zero, F::Zero),
        B::Source => pd(F::One, F::Zero),
        B::Destination => pd(F::Zero, F::One),
        B::DestinationOver => pd(F::OneMinusDstAlpha, F::One),
        B::SourceIn => pd(F::DstAlpha, F::Zero),
        B::DestinationIn => pd(F::Zero, F::SrcAlpha),
        B::SourceOut => pd(F::OneMinusDstAlpha, F::Zero),
        B::DestinationOut => pd(F::Zero, F::OneMinusSrcAlpha),
        B::SourceAtop => pd(F::DstAlpha, F::OneMinusSrcAlpha),
        B::DestinationAtop => pd(F::OneMinusDstAlpha, F::SrcAlpha),
        B::Xor => pd(F::OneMinusDstAlpha, F::OneMinusSrcAlpha),
        B::Plus => pd(F::One, F::One),
        // Source-over, and any mode added later that this match does not
        // know, draws as the base pipeline.
        _ => {
            return (
                crate::gpu::BlendState {
                    color: comp(F::SrcAlpha, F::OneMinusSrcAlpha),
                    alpha: comp(F::One, F::OneMinusSrcAlpha),
                },
                "fs_main",
            );
        }
    };
    (state, "fs_premul")
}

/// Fullscreen separable Gaussian blur pipeline used to produce the blurred
/// scene texture for `backdrop_blur` overlay shapes.
#[derive(Default)]
pub(crate) struct BackdropBlurResources {
    /// Fullscreen separable Gaussian blur pipeline.
    pub(crate) pipeline: Option<crate::gpu::RenderPipeline>,
    /// Bind group layout (group 0: source texture + sampler + uniforms).
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
    /// Linear clamp sampler shared by blur passes.
    pub(crate) sampler: Option<crate::gpu::Sampler>,
}

impl crate::resources::DeviceResources {
    /// Lazily create the SDF overlay shape render pipeline.
    ///
    /// No-op if already created. Called from `prepare_viewport_internal()` when
    /// `frame.overlays.shapes` is non-empty.
    pub(crate) fn ensure_overlay_shape_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.overlay_shape.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        // Group 0: binding 0 = the stacked shadow layers (a data texture),
        // binding 1 = the clip-mask shapes (a uniform array), binding 3 = the
        // per-draw group instance at a dynamic offset. No storage buffers, so
        // the overlay family runs on devices without them.
        let shadow_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("overlay_shape_shadow_bgl"),
            entries: &[
                shadow_layer_entry(0),
                clip_shapes_entry(1),
                // binding 2: viewport-size uniform (logical w, h) for the
                // vertex shader's local-pixel-to-NDC transform.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                group_instance_entry(3),
            ],
        });

        let fill_bgl = self.overlay_fill_bgl(device).clone();
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "overlay_shape_layout",
            &[&shadow_bgl, &fill_bgl],
        );

        let shader = crate::resources::builders::wgsl_module(
            device,
            "overlay_shape.wgsl",
            crate::resources::builders::wgsl_source!("overlay_shape"),
        );

        let pipeline = self.overlay_shape_variant(
            device,
            &layout,
            &shader,
            crate::renderer::OverlayBlend::SourceOver,
        );

        self.overlay_shape.shadow_bgl = Some(shadow_bgl);
        self.overlay_shape.pipeline = Some(pipeline);
        self.overlay_shape.layout = Some(layout);
        self.overlay_shape.shader = Some(shader);
    }

    /// The shape pipeline for `blend`, built on first use. Source-over is the
    /// base pipeline; every other mode is a variant that differs only in its
    /// blend state and fragment entry point.
    pub(crate) fn ensure_overlay_shape_blend(
        &mut self,
        device: &crate::gpu::Device,
        blend: crate::renderer::OverlayBlend,
    ) {
        self.ensure_overlay_shape_pipeline(device);
        if self.overlay_shape.variants.get(blend).is_some()
            || blend == crate::renderer::OverlayBlend::SourceOver
        {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let (Some(layout), Some(shader)) = (
            self.overlay_shape.layout.clone(),
            self.overlay_shape.shader.clone(),
        ) else {
            return;
        };
        let pipeline = self.overlay_shape_variant(device, &layout, &shader, blend);
        self.overlay_shape.variants.insert(blend, pipeline);
    }

    fn overlay_shape_variant(
        &self,
        device: &crate::gpu::Device,
        layout: &crate::gpu::PipelineLayout,
        shader: &crate::gpu::ShaderModule,
        blend: crate::renderer::OverlayBlend,
    ) -> crate::gpu::RenderPipeline {
        let (blend_state, fs_entry) = overlay_blend_target(blend);
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "overlay_shape_pipeline",
                layout,
                vertex_module: shader,
                vertex_entry: "vs_main",
                // One record per shape; the quad is expanded from the vertex
                // index. The per-draw group instance is a uniform (binding 3),
                // so the whole attribute budget is the shape's.
                vertex_buffers: &[OverlayShapeInstance::buffer_layout()],
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    self.scene_depth_format(),
                    false,
                    crate::gpu::CompareFunction::Always,
                )),
                multisample: crate::gpu::MultisampleState::default(),
                fragment: Some(crate::gpu::FragmentState {
                    module: shader,
                    entry_point: Some(fs_entry),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(blend_state),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                cache: None,
            },
        )
    }

    /// Lazily create the backdrop-blur shape pipeline.
    ///
    /// No-op if already created. Called from the overlay shape prepare when a
    /// shape sets a backdrop blur.
    pub(crate) fn ensure_overlay_shape_tex_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.overlay_shape.tex_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "overlay_shape_tex_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );

        let sampler =
            crate::resources::builders::clamp_linear_sampler(device, "overlay_shape_tex_sampler");

        // Group 1: the clip-mask shapes, plus the shared shadow layers, so a
        // backdrop shape takes the same layers as any other.
        let clip_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("overlay_shape_tex_clip_bgl"),
            entries: &[
                clip_shapes_entry(0),
                // binding 1: viewport-size uniform (logical w, h) for the
                // vertex shader's local-pixel-to-NDC transform.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: the stacked shadow layers, the same texture the
                // solid pass reads, so a textured shape takes the same layers.
                shadow_layer_entry(2),
            ],
        });

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "overlay_shape_tex_layout",
            &[&bgl, &clip_bgl],
        );

        let shader = crate::resources::builders::wgsl_module(
            device,
            "overlay_shape_tex.wgsl",
            crate::resources::builders::wgsl_source!("overlay_shape_tex"),
        );

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "overlay_shape_tex_pipeline",
                layout: &layout,
                vertex_module: &shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[OverlayBackdropVertex::buffer_layout()],
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    self.scene_depth_format(),
                    false,
                    crate::gpu::CompareFunction::Always,
                )),
                multisample: crate::gpu::MultisampleState::default(),
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(crate::gpu::BlendState {
                            color: crate::gpu::BlendComponent {
                                src_factor: crate::gpu::BlendFactor::SrcAlpha,
                                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                                operation: crate::gpu::BlendOperation::Add,
                            },
                            alpha: crate::gpu::BlendComponent {
                                src_factor: crate::gpu::BlendFactor::One,
                                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                                operation: crate::gpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                cache: None,
            },
        );

        self.overlay_shape.tex_bgl = Some(bgl);
        self.overlay_shape.tex_clip_bgl = Some(clip_bgl);
        self.overlay_shape.tex_sampler = Some(sampler);
        self.overlay_shape.tex_pipeline = Some(pipeline);
    }

    /// The fill layout (group 1 of the shape and text pipelines), built on
    /// first use.
    pub(crate) fn overlay_fill_bgl(
        &mut self,
        device: &crate::gpu::Device,
    ) -> &crate::gpu::BindGroupLayout {
        self.overlay_shape.fill_bgl.get_or_insert_with(|| {
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("overlay_fill_bgl"),
                entries: &[
                    data_texture_entry(0),
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Texture {
                            sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Sampler(
                            crate::gpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
            })
        })
    }

    /// Build the fill layout, its sampler, and the blank fill image if they do
    /// not exist yet. The blank image is written here, which is why this takes
    /// the queue and runs from the overlay prepare.
    pub(crate) fn ensure_overlay_fill_resources(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        self.overlay_fill_bgl(device);
        if self.overlay_shape.fill_sampler.is_none() {
            self.overlay_shape.fill_sampler = Some(
                crate::resources::builders::clamp_linear_sampler(device, "overlay_fill_sampler"),
            );
        }
        if self.overlay_shape.blank_image.is_none() {
            let entry = build_overlay_texture_entry(
                device,
                Some(queue),
                1,
                1,
                crate::gpu::TextureFormat::Rgba8Unorm,
                Some(&[255u8; 4]),
            );
            self.overlay_shape.blank_image = Some(entry.view);
        }
    }

    /// A fill bind group over `records` and `image` (the blank image when
    /// `None`), from `cache` when one was built for the same pair. Built
    /// groups are moved into `used`, so a pass that starts each frame from an
    /// empty `used` drops the ones it stopped drawing with.
    pub(crate) fn overlay_fill_bind_group(
        &self,
        device: &crate::gpu::Device,
        cache: &mut OverlayFillBindGroups,
        used: &mut OverlayFillBindGroups,
        records: &crate::gpu::TextureView,
        image: Option<&crate::gpu::TextureView>,
    ) -> crate::gpu::BindGroup {
        let image = image.unwrap_or_else(|| {
            self.overlay_shape
                .blank_image
                .as_ref()
                .expect("ensure_overlay_fill_resources runs before a fill is bound")
        });
        let key = (records.clone(), image.clone());
        if let Some(bg) = used.get(&key) {
            return bg.clone();
        }
        let bg = cache.remove(&key).unwrap_or_else(|| {
            device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("overlay_fill_bg"),
                layout: self
                    .overlay_shape
                    .fill_bgl
                    .as_ref()
                    .expect("built with the pipelines"),
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::gpu::BindingResource::TextureView(records),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::TextureView(image),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::Sampler(
                            self.overlay_shape
                                .fill_sampler
                                .as_ref()
                                .expect("ensure_overlay_fill_resources runs first"),
                        ),
                    },
                ],
            })
        });
        used.insert(key, bg.clone());
        bg
    }

    /// Upload an image as a texture for overlay shape fills.
    ///
    /// Returns an `OverlayTextureId` that can be stored in
    /// `OverlayShapeItem::texture`. The texture stays resident until you release
    /// it with [`free_overlay_texture`](Self::free_overlay_texture). For a source
    /// that changes every frame, prefer
    /// [`create_streaming_overlay_texture`](Self::create_streaming_overlay_texture)
    /// plus [`update_overlay_texture`](Self::update_overlay_texture), which reuse
    /// one GPU texture instead of allocating a new one per frame.
    ///
    /// The format follows `data`: an sRGB image is sampled with the hardware
    /// decode, a linear one (a lookup table, a data image) as stored, and float
    /// pixels at half-float precision, so values above 1 survive.
    ///
    /// # Errors
    ///
    /// The errors [`TextureData::validate`](crate::TextureData::validate)
    /// returns, and
    /// [`ViewportError::UnsupportedTextureData`](crate::error::ViewportError::UnsupportedTextureData)
    /// for a normal map or a compressed payload, which an overlay cannot use.
    pub fn upload_overlay_texture(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: crate::TextureData,
    ) -> crate::error::ViewportResult<OverlayTextureId> {
        self.require_live()?;
        let (width, height, format, pixels) = overlay_texture_pixels(data)?;
        let entry =
            build_overlay_texture_entry(device, Some(queue), width, height, format, Some(&pixels));
        Ok(self
            .content
            .overlay_textures
            .insert(entry, overlay_texture_bytes(width, height, format)))
    }

    /// Allocate a reusable overlay texture and return a stable
    /// `OverlayTextureId`.
    ///
    /// The texture starts undefined; feed it pixels with
    /// [`update_overlay_texture`](Self::update_overlay_texture) before pointing an
    /// `OverlayShapeItem` at it, and release it with
    /// [`free_overlay_texture`](Self::free_overlay_texture) when the source stops.
    /// Unlike [`upload_overlay_texture`](Self::upload_overlay_texture), updating
    /// this handle reuses the same GPU texture rather than stranding one per
    /// frame, so it is the right choice for a live source (decoded video, a
    /// capture feed, a per-frame heatmap).
    ///
    /// `colour_space` is the space of the RGBA8 bytes it will be fed: `Srgb`
    /// for video and captured or authored colour, `Linear` for data such as a
    /// heatmap's values. It is fixed for the texture's life, so each update
    /// stays a plain byte copy.
    pub fn create_streaming_overlay_texture(
        &mut self,
        device: &crate::gpu::Device,
        width: u32,
        height: u32,
        colour_space: crate::ColourSpace,
    ) -> OverlayTextureId {
        let format = match colour_space {
            crate::ColourSpace::Srgb => crate::gpu::TextureFormat::Rgba8UnormSrgb,
            crate::ColourSpace::Linear => crate::gpu::TextureFormat::Rgba8Unorm,
        };
        let entry = build_overlay_texture_entry(device, None, width, height, format, None);
        self.content
            .overlay_textures
            .insert(entry, overlay_texture_bytes(width, height, format))
    }

    /// Replace the contents of an overlay texture in place.
    ///
    /// When `width` / `height` match the current allocation this writes the
    /// pixels into the existing GPU texture (no new allocation). When they
    /// differ the texture and its view are reallocated in the same slot, so `id`
    /// stays valid. Works on any live overlay texture, whether it came from
    /// [`create_streaming_overlay_texture`](Self::create_streaming_overlay_texture)
    /// or [`upload_overlay_texture`](Self::upload_overlay_texture).
    ///
    /// Returns `false` (and does nothing) if `id` does not resolve to a live
    /// texture, for example because it was already freed, or if the texture holds
    /// float pixels, which RGBA8 bytes cannot fill. The texture keeps the colour
    /// space it was created with, including across a resize.
    ///
    /// # Panics
    ///
    /// Panics if `rgba_data.len() != width * height * 4`.
    pub fn update_overlay_texture(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: OverlayTextureId,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> bool {
        assert_eq!(
            rgba_data.len(),
            (width * height * 4) as usize,
            "update_overlay_texture: rgba_data length does not match width * height * 4"
        );

        let Some(entry) = self.content.overlay_textures.get_mut(id) else {
            return false;
        };

        // External (borrowed-view) entries are not owned by the registry: the
        // caller feeds them by rendering into their own texture, not through here.
        let Some(texture) = entry.texture.as_ref() else {
            return false;
        };

        let format = texture.format();
        if !matches!(
            format,
            crate::gpu::TextureFormat::Rgba8UnormSrgb | crate::gpu::TextureFormat::Rgba8Unorm
        ) {
            return false;
        }
        let size = texture.size();
        if size.width != width || size.height != height {
            *entry = build_overlay_texture_entry(
                device,
                Some(queue),
                width,
                height,
                format,
                Some(rgba_data),
            );
            self.content
                .overlay_textures
                .set_bytes(id, overlay_texture_bytes(width, height, format));
            return true;
        }

        write_overlay_texture(queue, texture, width, height, rgba_data);
        true
    }

    /// Release an overlay texture, freeing its GPU memory and its slot.
    ///
    /// After this the `id` resolves to nothing, so an `OverlayShapeItem` (or
    /// `OverlayPolylineItem`) still pointing at it is skipped for that frame
    /// rather than drawn: clear or repoint its `texture` before the next frame.
    /// Returns `true` if a texture was freed, `false` if `id` was already
    /// invalid.
    pub fn free_overlay_texture(&mut self, id: OverlayTextureId) -> bool {
        self.content.overlay_textures.remove(id).is_some()
    }

    /// Register an external, caller-owned colour `TextureView` as an
    /// [`OverlayTextureId`], so a GPU texture already on the device can be drawn as
    /// an overlay image with no CPU round-trip. Point an
    /// [`OverlayShapeItem`](crate::vplt::overlay::OverlayShapeItem) at the returned
    /// id with `with_texture` and it composites in the overlay z-order like any
    /// other overlay image.
    ///
    /// The intended source is an
    /// [`OffscreenViewportTarget`](crate::OffscreenViewportTarget) that a viewport
    /// rendered into: pass its `render_view()` (the sRGB view), not `sample_view()`.
    /// A sample is decoded according to the view's format, so the sRGB view
    /// turns the stored colour back into linear values; the non-sRGB view would
    /// hand the shader the encoded values and the image would draw wrong.
    ///
    /// The registry does not own the texture: it never writes to or resizes it, and
    /// [`update_overlay_texture`](Self::update_overlay_texture) rejects the returned
    /// id. When the source view is recreated (for example
    /// [`OffscreenViewportTarget::resize`](crate::OffscreenViewportTarget::resize)
    /// returns `true`) re-point the id in place with
    /// [`update_overlay_texture_view`](Self::update_overlay_texture_view), or free
    /// this id and register the new view. `width` / `height` are the view's pixel
    /// size, used for nine-slice UVs.
    ///
    /// The entry holds a clone of `view`, which keeps the underlying texture alive
    /// for as long as the id is registered. Dropping your
    /// [`OffscreenViewportTarget`](crate::OffscreenViewportTarget) (or otherwise
    /// releasing the source texture) does not reclaim its GPU memory while an
    /// overlay id still references it: call
    /// [`free_overlay_texture`](Self::free_overlay_texture) when the source stops,
    /// the same as for an owned overlay texture.
    pub fn register_overlay_texture_view(
        &mut self,
        view: &crate::gpu::TextureView,
        width: u32,
        height: u32,
    ) -> OverlayTextureId {
        let entry = build_external_overlay_texture_entry(view, width, height);
        // An external entry borrows the caller's texture, so it carries no resident
        // GPU-memory charge of its own.
        self.content.overlay_textures.insert(entry, 0)
    }

    /// Re-point an external overlay texture id at a new `TextureView`, keeping the
    /// id stable. Call this when the source view is recreated (a resized
    /// [`OffscreenViewportTarget`](crate::OffscreenViewportTarget)) so existing
    /// [`OverlayShapeItem`](crate::vplt::overlay::OverlayShapeItem)s keep working
    /// without re-pointing.
    ///
    /// Returns `false` (and does nothing) if `id` does not resolve to a live entry,
    /// or resolves to an owned (CPU-uploaded / streaming) texture rather than an
    /// external one: those are updated with
    /// [`update_overlay_texture`](Self::update_overlay_texture) instead.
    pub fn update_overlay_texture_view(
        &mut self,
        id: OverlayTextureId,
        view: &crate::gpu::TextureView,
        width: u32,
        height: u32,
    ) -> bool {
        let Some(entry) = self.content.overlay_textures.get_mut(id) else {
            return false;
        };
        if entry.texture.is_some() {
            return false;
        }
        *entry = build_external_overlay_texture_entry(view, width, height);
        true
    }

    /// Start an asynchronous overlay texture upload.
    ///
    /// Returns a [`JobId`](crate::resources::JobId) immediately. Texture
    /// creation and `queue.write_texture` run on a worker thread on cloned
    /// `Device` and `Queue` handles; the apply step inserts the prepared
    /// `OverlayShapeTextureEntry` into the store. Take the resulting
    /// [`OverlayTextureId`] via
    /// [`upload_result_overlay_texture`](Self::upload_result_overlay_texture).
    ///
    /// The format follows `data` exactly as for
    /// [`upload_overlay_texture`](Self::upload_overlay_texture), and `data`
    /// transfers into the worker.
    ///
    /// # Errors
    ///
    /// The same as [`upload_overlay_texture`](Self::upload_overlay_texture),
    /// reported before any job is submitted.
    pub fn begin_upload_overlay_texture(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: crate::TextureData,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        self.require_live()?;
        let (width, height, format, pixels) = overlay_texture_pixels(data)?;
        let slot = crate::resources::ResultSlot::<OverlayTextureId>::new();
        let slot_for_apply = slot.clone();
        let device_for_worker = device.clone();
        let queue_for_worker = queue.clone();
        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.1);
                let entry = build_overlay_texture_entry(
                    &device_for_worker,
                    Some(&queue_for_worker),
                    width,
                    height,
                    format,
                    Some(&pixels),
                );
                progress.set(0.95);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut crate::resources::DeviceResources| {
                        let id = resources
                            .content
                            .overlay_textures
                            .insert(entry, overlay_texture_bytes(width, height, format));
                        slot_for_apply.set(id);
                    }),
                ))
            })
        };
        self.job_results
            .overlay_texture
            .lock()
            .expect("overlay texture result map poisoned")
            .insert(id, slot);
        Ok(id)
    }

    /// Take the [`OverlayTextureId`] produced by a completed
    /// [`begin_upload_overlay_texture`](Self::begin_upload_overlay_texture) job.
    pub fn upload_result_overlay_texture(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<OverlayTextureId> {
        let mut map = self
            .job_results
            .overlay_texture
            .lock()
            .expect("overlay texture result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(tid) => {
                map.remove(&id);
                Ok(tid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Lazily create the separable Gaussian blur pipeline used to blur the
    /// scene texture for `backdrop_blur` overlay shapes.
    ///
    /// No-op if already created.
    pub(crate) fn ensure_backdrop_blur_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.backdrop_blur.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("backdrop_blur_bgl"),
            entries: &[
                // binding 0: source texture
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
                // binding 1: sampler
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 2: blur parameters uniform
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let sampler =
            crate::resources::builders::clamp_linear_sampler(device, "backdrop_blur_sampler");

        let layout =
            crate::resources::builders::pipeline_layout(device, "backdrop_blur_layout", &[&bgl]);

        let shader = crate::resources::builders::wgsl_module(
            device,
            "backdrop_blur.wgsl",
            crate::resources::builders::wgsl_source!("backdrop_blur"),
        );

        let pipeline = crate::resources::builders::build_fullscreen_pipeline(
            device,
            "backdrop_blur_pipeline",
            &layout,
            &shader,
            self.target_format,
            None,
        );

        self.backdrop_blur.bgl = Some(bgl);
        self.backdrop_blur.sampler = Some(sampler);
        self.backdrop_blur.pipeline = Some(pipeline);
    }
}

/// Clip-mask shapes a frame can hold. The shaders declare a uniform array this
/// long, which at 64 bytes an entry fills the 16 KiB uniform binding every tier
/// has. A frame's later masks are dropped and counted in
/// `FrameStats::clip_masks_dropped`.
pub(crate) const MAX_OVERLAY_CLIP_SHAPES: usize = 256;

/// The clip uniform's size: `MAX_OVERLAY_CLIP_SHAPES` entries.
pub(crate) const CLIP_UNIFORM_BYTES: u64 =
    (MAX_OVERLAY_CLIP_SHAPES * std::mem::size_of::<ClipShapeGpu>()) as u64;

/// Shadow layers per row of the shadow-layer texture. Each layer is
/// `SHADOW_LAYER_TEXELS` `Rgba32Float` texels side by side, so a row is 768
/// texels wide, inside every tier's texture limit. The shaders hold the same
/// constant.
pub(crate) const SHADOW_LAYERS_PER_ROW: u32 = 256;

/// Texels one shadow layer takes: `colour`, `params`, `params2`.
const SHADOW_LAYER_TEXELS: u32 = 3;

/// Texels per row of the shadow-layer texture.
pub(crate) const SHADOW_ROW_TEXELS: u32 = SHADOW_LAYERS_PER_ROW * SHADOW_LAYER_TEXELS;

/// Texels per row of the fill-record texture the shared fill module reads.
/// `overlay_fill.wgsl` holds the same constant.
pub(crate) const FILL_TEXELS_PER_ROW: u32 = 1024;

/// Layout entry for an `Rgba32Float` data texture read with `textureLoad`: the
/// shadow layers and the fill records.
fn data_texture_entry(binding: u32) -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding,
        visibility: crate::gpu::ShaderStages::FRAGMENT,
        ty: crate::gpu::BindingType::Texture {
            sample_type: crate::gpu::TextureSampleType::Float { filterable: false },
            view_dimension: crate::gpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// Layout entry for the shadow-layer texture, read with `textureLoad`.
fn shadow_layer_entry(binding: u32) -> crate::gpu::BindGroupLayoutEntry {
    data_texture_entry(binding)
}

/// Layout entry for the clip-shape uniform array.
pub(crate) fn clip_shapes_entry(binding: u32) -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding,
        visibility: crate::gpu::ShaderStages::FRAGMENT,
        ty: crate::gpu::BindingType::Buffer {
            ty: crate::gpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(CLIP_UNIFORM_BYTES),
        },
        count: None,
    }
}

/// Layout entry for the per-draw group instance: one `OverlayInstance` read at
/// a dynamic offset into the frame's padded instance list.
pub(crate) fn group_instance_entry(binding: u32) -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding,
        visibility: crate::gpu::ShaderStages::VERTEX,
        ty: crate::gpu::BindingType::Buffer {
            ty: crate::gpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                crate::resources::OverlayInstance,
            >() as u64),
        },
        count: None,
    }
}

/// The group instance as bound: one record at the dynamic offset a draw sets.
pub(crate) fn group_instance_binding(buf: &crate::gpu::Buffer) -> crate::gpu::BindingResource<'_> {
    crate::gpu::BindingResource::Buffer(crate::gpu::BufferBinding {
        buffer: buf,
        offset: 0,
        size: std::num::NonZeroU64::new(
            std::mem::size_of::<crate::resources::OverlayInstance>() as u64
        ),
    })
}

/// The clip uniform as bound: exactly the array the shaders declare.
pub(crate) fn clip_binding(buf: &crate::gpu::Buffer) -> crate::gpu::BindingResource<'_> {
    crate::gpu::BindingResource::Buffer(crate::gpu::BufferBinding {
        buffer: buf,
        offset: 0,
        size: std::num::NonZeroU64::new(CLIP_UNIFORM_BYTES),
    })
}

/// `bytes` (whole `Rgba32Float` texels, at least one) padded to whole rows of
/// `row_texels`, and the row count.
pub(crate) fn data_texels(bytes: &[u8], row_texels: u32) -> (Vec<u8>, u32) {
    let row_bytes = (row_texels * 16) as usize;
    let rows = (bytes.len().div_ceil(row_bytes) as u32).max(1);
    let mut texels = bytes.to_vec();
    texels.resize(rows as usize * row_bytes, 0);
    (texels, rows)
}

/// `layers` (at least one) as shadow-layer texels, padded to whole rows, and
/// the row count.
#[cfg(test)]
pub(crate) fn shadow_layer_texels(layers: &[OverlayShadowLayerGpu]) -> (Vec<u8>, u32) {
    data_texels(bytemuck::cast_slice(layers), SHADOW_ROW_TEXELS)
}

/// An `Rgba32Float` data texture `row_texels` wide and `rows` rows high.
pub(crate) fn create_data_texture(
    device: &crate::gpu::Device,
    label: &'static str,
    row_texels: u32,
    rows: u32,
) -> crate::gpu::Texture {
    device.create_texture(&crate::gpu::TextureDescriptor {
        label: Some(label),
        size: crate::gpu::Extent3d {
            width: row_texels,
            height: rows,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: crate::gpu::TextureDimension::D2,
        format: crate::gpu::TextureFormat::Rgba32Float,
        usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Write `rows` rows of `texels` (from `data_texels`) into `texture`.
pub(crate) fn write_data_texels(
    queue: &crate::gpu::Queue,
    texture: &crate::gpu::Texture,
    texels: &[u8],
    row_texels: u32,
    rows: u32,
) {
    queue.write_texture(
        crate::gpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: crate::gpu::Origin3d::ZERO,
            aspect: crate::gpu::TextureAspect::All,
        },
        texels,
        crate::gpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row_texels * 16),
            rows_per_image: Some(rows),
        },
        crate::gpu::Extent3d {
            width: row_texels,
            height: rows,
            depth_or_array_layers: 1,
        },
    );
}

/// A data texture holding `bytes`, for a retained group compiled once.
/// Returns the view and the bytes it holds.
pub(crate) fn data_texture(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    label: &'static str,
    row_texels: u32,
    bytes: &[u8],
) -> (crate::gpu::TextureView, u64) {
    let (texels, rows) = data_texels(bytes, row_texels);
    let texture = create_data_texture(device, label, row_texels, rows);
    write_data_texels(queue, &texture, &texels, row_texels, rows);
    let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
    (view, texels.len() as u64)
}

/// A shadow-layer texture holding `layers` (at least one), for a retained
/// group compiled once. Returns the view and the bytes it holds.
pub(crate) fn shadow_layer_texture(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    layers: &[OverlayShadowLayerGpu],
) -> (crate::gpu::TextureView, u64) {
    data_texture(
        device,
        queue,
        "overlay_shadow_layers",
        SHADOW_ROW_TEXELS,
        bytemuck::cast_slice(layers),
    )
}

/// A fill-record texture holding `records`, for a retained group compiled
/// once. Returns the view and the bytes it holds.
pub(crate) fn fill_record_texture(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    records: &[[f32; 4]],
) -> (crate::gpu::TextureView, u64) {
    data_texture(
        device,
        queue,
        "overlay_fill_records",
        FILL_TEXELS_PER_ROW,
        bytemuck::cast_slice(records),
    )
}

/// Fill bind groups keyed by the record texture and the image they bind.
pub(crate) type OverlayFillBindGroups = std::collections::HashMap<
    (crate::gpu::TextureView, crate::gpu::TextureView),
    crate::gpu::BindGroup,
>;

/// One stacked shadow layer as uploaded to the shadow-layer texture.
///
/// The solid overlay-shape shader reads a contiguous run of these per shape
/// (outer layers first, then inner) starting at the instance's `shadow_meta.x`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayShadowLayerGpu {
    /// RGBA shadow colour (pre-multiplied opacity).
    pub colour: [f32; 4],
    /// `[blur_px, offset_x, offset_y, is_inner]`. `is_inner` is 0 for an
    /// outer drop shadow, 1 for an inner (inset) shadow.
    pub params: [f32; 4],
    /// `[spread_px, falloff, 0, 0]`. `spread` grows the silhouette before the
    /// blur is applied; `falloff` is the exponent shaping the blur curve.
    pub params2: [f32; 4],
}

/// One clip-mask shape as uploaded to the clip uniform, shared by the shape
/// and text pipelines. All geometry is in framebuffer pixels so the fragment shader
/// can evaluate the mask's SDF directly against `clip_position.xy`. An item's
/// `clip_index` selects its entry; `params.z` chains to a parent entry (or `-1`),
/// so nested masks compose by intersection.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ClipShapeGpu {
    /// Mask centre in framebuffer pixels.
    pub centre: [f32; 2],
    /// Mask half-extents in framebuffer pixels.
    pub half_size: [f32; 2],
    /// Shape-specific radii, matching the draw encoding (length terms scaled to
    /// framebuffer pixels).
    pub radii: [f32; 4],
    /// `[shape_type, rotation_radians, parent_index, invert]`. `parent_index` is
    /// `-1.0` for no parent; `invert` is reserved (`0.0`).
    pub params: [f32; 4],
    /// Rotation pivot in framebuffer pixels from the centre.
    pub pivot: [f32; 2],
    pub _pad: [f32; 2],
}

/// One analytic SDF overlay shape, as the shape pipeline's per-instance
/// record. The vertex shader expands a quad from the vertex index and
/// `quad`; the fragment shader maps each pixel back into the shape's own frame
/// through `linv` and evaluates the signed-distance function there, so one
/// record draws a shape under any 2D affine transform. Its fill is a record in
/// the shared fill table, or the flat `fill_colour`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayShapeInstance {
    /// The axis-aligned quad that covers the transformed shape and its outer
    /// shadows, in logical pixels before the group transform: centre `xy`,
    /// half-extents `zw`.
    pub quad: [f32; 4],
    /// The pivot in the same pixel space (`xy`) and the shape's half-extents in
    /// its own frame (`zw`).
    pub pivot_half: [f32; 4],
    /// The inverse of the item's linear map, row-major `[a, b, c, d]`: a
    /// pixel offset `q` from the pivot lands at `[a b; c d] . q + pivot_local`
    /// in the shape's frame.
    pub linv: [f32; 4],
    /// RGBA fill colour (opacity and tint folded in), drawn when the shape
    /// names no fill record.
    pub fill_colour: [f32; 4],
    /// Shape-specific radii. For RoundedRect: [top-right, bottom-right,
    /// bottom-left, top-left]. For Rect: a uniform radius in every slot.
    /// Unused components are zero.
    pub radii: [f32; 4],
    /// `[base_index, outer_count, inner_count, shape_type]`: where this
    /// shape's run of stacked shadow layers starts in the shared texture, how
    /// many of each kind it has (outer first), and the encoded shape type.
    pub shadow_meta: [f32; 4],
    /// Clip rectangle in framebuffer pixels (x0, y0, x1, y1). All-zero means
    /// no clipping. Fragments outside the rect are discarded.
    pub clip_rect: [f32; 4],
    /// `[clip_index, pivot_local_x, pivot_local_y, item_scale]`: the clip-mask
    /// index (or `-1` for none), the pivot in the shape's own frame as an
    /// offset from its centre, and the item's uniform scale, which its shadow
    /// offsets are given in.
    pub misc: [f32; 4],
    /// `[fill_index, 0, 0, 0]`: the first texel of the shape's record in the
    /// fill table, or `-1` to draw `fill_colour`.
    pub fill: [f32; 4],
}

impl OverlayShapeInstance {
    /// Nine `vec4` attributes, instance-stepped; no vertex stream.
    pub fn buffer_layout() -> crate::gpu::VertexBufferLayout<'static> {
        use std::mem::offset_of;
        const ATTRIBUTES: [crate::gpu::VertexAttribute; 9] = {
            const fn at(offset: usize, shader_location: u32) -> crate::gpu::VertexAttribute {
                crate::gpu::VertexAttribute {
                    offset: offset as u64,
                    shader_location,
                    format: crate::gpu::VertexFormat::Float32x4,
                }
            }
            // Offsets come from `offset_of!` so reordering fields cannot desync
            // them from the struct.
            [
                at(offset_of!(OverlayShapeInstance, quad), 0),
                at(offset_of!(OverlayShapeInstance, pivot_half), 1),
                at(offset_of!(OverlayShapeInstance, linv), 2),
                at(offset_of!(OverlayShapeInstance, fill_colour), 3),
                at(offset_of!(OverlayShapeInstance, radii), 4),
                at(offset_of!(OverlayShapeInstance, shadow_meta), 5),
                at(offset_of!(OverlayShapeInstance, clip_rect), 6),
                at(offset_of!(OverlayShapeInstance, misc), 7),
                at(offset_of!(OverlayShapeInstance, fill), 8),
            ]
        };
        crate::gpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayShapeInstance>() as crate::gpu::BufferAddress,
            step_mode: crate::gpu::VertexStepMode::Instance,
            attributes: &ATTRIBUTES,
        }
    }

    /// The byte offset of instance `start` in a buffer of these, for binding a
    /// draw's range without a first-instance offset.
    pub fn byte_offset(start: u32) -> crate::gpu::BufferAddress {
        start as crate::gpu::BufferAddress
            * std::mem::size_of::<OverlayShapeInstance>() as crate::gpu::BufferAddress
    }
}

const _: () = assert!(std::mem::size_of::<OverlayShapeInstance>() == 144);

/// Per-vertex data for backdrop-blur overlay shapes: an analytic shape's quad
/// (six vertices, evaluated by its distance field) or a vector path's
/// triangles (a sharp rect grown past the path, so the distance test reads
/// fully inside). The blurred scene is sampled at the screen position.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayBackdropVertex {
    /// Position in logical pixels.
    pub position: [f32; 2],
    /// Position relative to the shape pivot (or the path's centre), in logical
    /// pixels.
    pub local_pos: [f32; 2],
    /// RGBA tint laid over the filtered backdrop, its alpha the mix amount.
    pub tint: [f32; 4],
    /// Half-extents of the shape bounding box in logical pixels.
    pub half_size: [f32; 2],
    /// Shape-specific radii (same encoding as `OverlayShapeInstance`).
    pub radii: [f32; 4],
    /// `[shape_type, clip_index, pivot_local_x, pivot_local_y]`: the encoded
    /// shape type (same values as `OverlayShapeInstance`), the clip-mask index
    /// (or `-1.0` for none), and the pivot in the shape's own frame.
    pub shape_meta: [f32; 4],
    /// Framebuffer-pixel clip bbox `(x0, y0, x1, y1)`; all zero means no box
    /// clip. Same encoding as `OverlayShapeInstance::clip_rect`.
    pub clip_rect: [f32; 4],
    /// `[base_index, outer_count, inner_count]`: where this shape's run of
    /// stacked shadow layers starts in the shared texture and how many of each
    /// kind it has. Same encoding as `OverlayShapeInstance::shadow_meta`.
    pub shadow_index: [f32; 3],
    /// `[0, saturation, brightness, hue_shift]`: the backdrop colour filters.
    pub filters: [f32; 4],
    /// The inverse of the item's linear map, row-major `[a, b, c, d]`, mapping
    /// `local_pos` (a pixel offset from the pivot) into the shape's own frame.
    /// Identity for a tessellated path, whose triangles are already placed.
    pub linv: [f32; 4],
    /// `[item_scale, 0, 0, 0]`: the uniform scale the shadow offsets are given
    /// in.
    pub item_params: [f32; 4],
}

impl OverlayBackdropVertex {
    pub fn buffer_layout() -> crate::gpu::VertexBufferLayout<'static> {
        crate::gpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayBackdropVertex>() as crate::gpu::BufferAddress,
            step_mode: crate::gpu::VertexStepMode::Vertex,
            // Offsets come from `offset_of!` so reordering fields can't desync
            // them.
            attributes: &[
                // location 0: position vec2f (logical pixels)
                crate::gpu::VertexAttribute {
                    offset: std::mem::offset_of!(OverlayBackdropVertex, position) as u64,
                    shader_location: 0,
                    format: crate::gpu::VertexFormat::Float32x2,
                },
                // location 1: local_pos vec2f
                crate::gpu::VertexAttribute {
                    offset: std::mem::offset_of!(OverlayBackdropVertex, local_pos) as u64,
                    shader_location: 1,
                    format: crate::gpu::VertexFormat::Float32x2,
                },
                // location 2: tint vec4f
                crate::gpu::VertexAttribute {
                    offset: std::mem::offset_of!(OverlayBackdropVertex, tint) as u64,
                    shader_location: 2,
                    format: crate::gpu::VertexFormat::Float32x4,
                },
                // location 3: half_size vec2f
                crate::gpu::VertexAttribute {
                    offset: std::mem::offset_of!(OverlayBackdropVertex, half_size) as u64,
                    shader_location: 3,
                    format: crate::gpu::VertexFormat::Float32x2,
                },
                // location 4: radii vec4f
                crate::gpu::VertexAttribute {
                    offset: std::mem::offset_of!(OverlayBackdropVertex, radii) as u64,
                    shader_location: 4,
                    format: crate::gpu::VertexFormat::Float32x4,
                },
                // location 5: shape_meta vec4f (shape_type, clip_index, pivot)
                crate::gpu::VertexAttribute {
                    offset: std::mem::offset_of!(OverlayBackdropVertex, shape_meta) as u64,
                    shader_location: 5,
                    format: crate::gpu::VertexFormat::Float32x4,
                },
                // location 6: clip_rect vec4f (framebuffer-pixel clip bbox)
                crate::gpu::VertexAttribute {
                    offset: std::mem::offset_of!(OverlayBackdropVertex, clip_rect) as u64,
                    shader_location: 6,
                    format: crate::gpu::VertexFormat::Float32x4,
                },
                // location 7: shadow_index vec3f (base, outer_ct, inner_ct)
                crate::gpu::VertexAttribute {
                    offset: std::mem::offset_of!(OverlayBackdropVertex, shadow_index) as u64,
                    shader_location: 7,
                    format: crate::gpu::VertexFormat::Float32x3,
                },
                // location 8: filters vec4f (unused, saturation, brightness, hue)
                crate::gpu::VertexAttribute {
                    offset: std::mem::offset_of!(OverlayBackdropVertex, filters) as u64,
                    shader_location: 8,
                    format: crate::gpu::VertexFormat::Float32x4,
                },
                // location 9: linv vec4f (the inverse item linear map)
                crate::gpu::VertexAttribute {
                    offset: std::mem::offset_of!(OverlayBackdropVertex, linv) as u64,
                    shader_location: 9,
                    format: crate::gpu::VertexFormat::Float32x4,
                },
                // location 10: item_params vec4f (item_scale, 0, 0, 0)
                crate::gpu::VertexAttribute {
                    offset: std::mem::offset_of!(OverlayBackdropVertex, item_params) as u64,
                    shader_location: 10,
                    format: crate::gpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Texture entry for an overlay shape texture fill.
///
/// Stored in `DeviceResources::overlay_textures`. The `view` is what the tex
/// bind group samples. For an owned entry (CPU upload / streaming) `texture` is
/// `Some` and the `view` is rebuilt whenever the texture is reallocated for a
/// size change. For an external entry the registry borrows a caller-owned view:
/// `texture` is `None`, the registry never writes to or resizes it, and the
/// caller re-registers when the source view is recreated.
pub(crate) struct OverlayShapeTextureEntry {
    /// The owned GPU texture, or `None` for an external (borrowed-view) entry.
    pub texture: Option<crate::gpu::Texture>,
    pub view: crate::gpu::TextureView,
    /// Pixel size of the sampled view, used for nine-slice UVs. Taken from the
    /// owned texture, or supplied by the caller for an external view.
    pub size: [u32; 2],
}

/// Bytes per texel of an overlay texture format: 4 for the 8-bit formats, 8
/// for half float.
fn overlay_texel_bytes(format: crate::gpu::TextureFormat) -> u32 {
    format.block_copy_size(None).unwrap_or(4)
}

/// Resident-byte charge for an overlay texture of the given size and format.
fn overlay_texture_bytes(width: u32, height: u32, format: crate::gpu::TextureFormat) -> u64 {
    (width as u64) * (height as u64) * overlay_texel_bytes(format) as u64
}

/// Check that an overlay can take `data` and turn it into a format and the
/// bytes to write. An overlay has no normal slot and samples no compressed
/// format, so both are rejected rather than drawn wrongly.
fn overlay_texture_pixels(
    data: crate::TextureData,
) -> crate::error::ViewportResult<(u32, u32, crate::gpu::TextureFormat, Vec<u8>)> {
    use crate::{TexturePayload, TextureRejection, TextureRole, UploadSlot};
    data.validate()?;
    let reject = |reason| {
        Err(crate::error::ViewportError::UnsupportedTextureData {
            slot: UploadSlot::Overlay,
            reason,
        })
    };
    if data.role() == TextureRole::NormalMap {
        return reject(TextureRejection::NormalMap);
    }
    if matches!(data.payload(), TexturePayload::Compressed { .. }) {
        return reject(TextureRejection::UnsupportedPayload);
    }
    let (width, height) = (data.width(), data.height());
    let (format, mut levels) =
        crate::resources::material::textures::texture_format_and_levels(data);
    let pixels = levels.pop().expect("an uncompressed payload is one level");
    Ok((width, height, format, pixels))
}

/// Write `bytes` (row-major, top-to-bottom, one full image in `texture`'s
/// format) into the whole of `texture`.
fn write_overlay_texture(
    queue: &crate::gpu::Queue,
    texture: &crate::gpu::Texture,
    width: u32,
    height: u32,
    rgba_data: &[u8],
) {
    queue.write_texture(
        crate::gpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: crate::gpu::Origin3d::ZERO,
            aspect: crate::gpu::TextureAspect::All,
        },
        rgba_data,
        crate::gpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * overlay_texel_bytes(texture.format())),
            rows_per_image: Some(height),
        },
        crate::gpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Create an overlay texture in `format` and its default view, optionally
/// writing initial pixels. Pass `queue` and `pixels` together to upload
/// contents; pass both `None` to leave the texture undefined (a streaming
/// texture filled later by `update_overlay_texture`).
fn build_overlay_texture_entry(
    device: &crate::gpu::Device,
    queue: Option<&crate::gpu::Queue>,
    width: u32,
    height: u32,
    format: crate::gpu::TextureFormat,
    pixels: Option<&[u8]>,
) -> OverlayShapeTextureEntry {
    let texture = device.create_texture(&crate::gpu::TextureDescriptor {
        label: Some("overlay_shape_tex"),
        size: crate::gpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: crate::gpu::TextureDimension::D2,
        format,
        usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    if let (Some(queue), Some(pixels)) = (queue, pixels) {
        write_overlay_texture(queue, &texture, width, height, pixels);
    }

    let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
    OverlayShapeTextureEntry {
        texture: Some(texture),
        view,
        size: [width, height],
    }
}

/// Build an overlay texture entry that borrows an external, caller-owned
/// `TextureView` (for example an `OffscreenViewportTarget::render_view()`), so a
/// GPU texture already on the device can be drawn as an overlay image with no CPU
/// round-trip. The registry does not own the underlying texture: it never writes
/// to or resizes it, and the caller must re-register when the view is recreated.
///
/// `width` / `height` are the view's pixel size, used only for nine-slice UVs.
fn build_external_overlay_texture_entry(
    view: &crate::gpu::TextureView,
    width: u32,
    height: u32,
) -> OverlayShapeTextureEntry {
    OverlayShapeTextureEntry {
        texture: None,
        view: view.clone(),
        size: [width.max(1), height.max(1)],
    }
}

/// Per-frame GPU data for batched SDF overlay shape rendering.
pub(crate) struct OverlayShapeGpuData {
    /// Every analytic shape, one `OverlayShapeInstance` each, in draw order.
    /// Written once the draw list is built.
    pub instance_buf: Option<crate::gpu::Buffer>,
    /// Group 1 bind groups: the frame's fill records with the blank image at
    /// index 0, then one per image a textured shape samples.
    pub fill_bind_groups: Vec<crate::gpu::BindGroup>,
    /// Bind group (group 0) holding the stacked shadow layers, the clip shapes
    /// and the group-instance uniform for the shape pipeline. Present whenever
    /// `instance_buf` is `Some`; immediate draws bind it at dynamic offset 0,
    /// the identity instance.
    pub shadow_bind_group: Option<crate::gpu::BindGroup>,
    /// Backing shadow-layer texture for `shadow_bind_group`. Kept alive
    /// alongside it.
    pub _shadow_tex: Option<crate::gpu::TextureView>,
    /// Backing clip-shape uniform for `shadow_bind_group` (binding 1).
    pub _clip_buf: Option<crate::gpu::Buffer>,
    /// Clip-mask bind group (group 1) for the backdrop pipeline. Present
    /// whenever a backdrop shape draws this frame.
    pub tex_clip_bind_group: Option<crate::gpu::BindGroup>,
    /// Backing clip-shape uniform for `tex_clip_bind_group`. Kept alive
    /// alongside it.
    pub _tex_clip_buf: Option<crate::gpu::Buffer>,
    /// Vertex buffer for backdrop-blur shapes (frosted glass).
    /// The bind group is created at render time once the blurred scene texture
    /// is available.
    pub blur_vertex_buf: Option<crate::gpu::Buffer>,
    /// Number of blur backdrop vertices.
    pub blur_vertex_count: u32,
    /// Maximum `backdrop_blur` value across all blur shapes this frame.
    /// Used as the spread parameter for the Gaussian blur passes.
    pub max_blur_radius: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shaders declare the clip array and the shadow-layer row width as
    /// literals; they have to agree with the constants the buffers are sized by.
    #[test]
    fn shader_constants_match() {
        let clip = format!("array<ClipShape, {MAX_OVERLAY_CLIP_SHAPES}>");
        let row = format!("SHADOW_LAYERS_PER_ROW: i32 = {SHADOW_LAYERS_PER_ROW};");
        let fill_row = format!("FILL_TEXELS_PER_ROW: i32 = {FILL_TEXELS_PER_ROW};");
        for (name, src) in [
            (
                "overlay_shape",
                crate::resources::builders::wgsl_source!("overlay_shape"),
            ),
            (
                "overlay_shape_tex",
                crate::resources::builders::wgsl_source!("overlay_shape_tex"),
            ),
            (
                "overlay_text",
                crate::resources::builders::wgsl_source!("overlay_text"),
            ),
        ] {
            assert!(src.contains(&clip), "{name} does not declare {clip}");
            if name != "overlay_text" {
                assert!(src.contains(&row), "{name} does not declare {row}");
            }
            if name != "overlay_shape_tex" {
                assert!(
                    src.contains(&fill_row),
                    "{name} does not declare {fill_row}"
                );
            }
        }
        assert_eq!(CLIP_UNIFORM_BYTES, 16 << 10);
    }

    #[test]
    fn shadow_layers_pad_to_whole_rows() {
        let layer = OverlayShadowLayerGpu {
            colour: [1.0; 4],
            params: [2.0; 4],
            params2: [3.0; 4],
        };
        let row_bytes = (SHADOW_LAYERS_PER_ROW * SHADOW_LAYER_TEXELS * 16) as usize;
        let (texels, rows) = shadow_layer_texels(&[layer]);
        assert_eq!((texels.len(), rows), (row_bytes, 1));
        let many = vec![layer; SHADOW_LAYERS_PER_ROW as usize + 1];
        let (texels, rows) = shadow_layer_texels(&many);
        assert_eq!((texels.len(), rows), (2 * row_bytes, 2));
        // Layer 256 opens the second row.
        assert_eq!(
            &texels[row_bytes..row_bytes + 48],
            bytemuck::bytes_of(&layer)
        );
    }
}
