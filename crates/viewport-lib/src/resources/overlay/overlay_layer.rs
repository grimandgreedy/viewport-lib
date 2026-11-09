//! Pipelines for compositing overlay layers.

/// How a layer composite draws: which fragment entry point, under which
/// blend state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LayerPipelineKey {
    /// The premultiplied content under a blend state: source-over or one of
    /// the fixed-function modes.
    Composite(crate::renderer::OverlayBlend),
    /// The content blended with a copy of the destination by a blend formula.
    CompositeDst,
    /// The blurred destination inside the layer's clip.
    Backdrop,
}

/// The layer composite's shader, layouts and the pipeline for every
/// [`LayerPipelineKey`] a frame has used. Built on the first frame with a
/// layer that needs a scratch target.
#[derive(Default)]
pub(crate) struct OverlayLayerResources {
    /// Group 0: the content, its two blurred copies, the destination copy, the
    /// blurred destination, and a linear sampler.
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
    layout: Option<crate::gpu::PipelineLayout>,
    shader: Option<crate::gpu::ShaderModule>,
    pipelines: Vec<(LayerPipelineKey, crate::gpu::RenderPipeline)>,
    /// Linear clamp sampler for the blurred copies.
    pub(crate) sampler: Option<crate::gpu::Sampler>,
    /// One transparent texel, bound where a layer has no blurred copy or no
    /// destination copy.
    pub(crate) blank: Option<crate::gpu::TextureView>,
}

impl OverlayLayerResources {
    /// The pipeline for `key`, if it has been built.
    pub(crate) fn pipeline(&self, key: LayerPipelineKey) -> Option<&crate::gpu::RenderPipeline> {
        self.pipelines
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, p)| p)
    }
}

/// One layer composite, read per instance by `overlay_layer.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayLayerInstance {
    /// The quad, in logical pixels `[x0, y0, x1, y1]`.
    pub rect: [f32; 4],
    /// `[opacity, blend mode code, flags, 0]`.
    pub params: [f32; 4],
    /// The colour matrix, as its five columns.
    pub matrix: [[f32; 4]; 5],
    /// The drop shadow's straight linear colour.
    pub shadow_colour: [f32; 4],
    /// `[offset_x, offset_y, 0, 0]`: the drop shadow's offset in logical
    /// pixels.
    pub shadow: [f32; 4],
    /// `[saturation, brightness, hue_shift, has_backdrop]`.
    pub backdrop: [f32; 4],
    /// The clip box in framebuffer pixels; all zero for none.
    pub clip_rect: [f32; 4],
    /// `[clip_index, 0, 0, 0]`: the clip mask, or `-1`.
    pub clip: [f32; 4],
}

/// Flag bits in `OverlayLayerInstance::params[2]`.
pub(crate) mod layer_flags {
    /// Read the blurred content instead of the sharp one.
    pub(crate) const BLUR: u32 = 1;
    /// Cast a drop shadow from the shadow-blurred content.
    pub(crate) const SHADOW: u32 = 2;
    /// Apply the colour matrix.
    pub(crate) const MATRIX: u32 = 4;
}

impl OverlayLayerInstance {
    pub(crate) fn buffer_layout() -> crate::gpu::VertexBufferLayout<'static> {
        const ATTRS: [crate::gpu::VertexAttribute; 12] = {
            let mut a = [crate::gpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: crate::gpu::VertexFormat::Float32x4,
            }; 12];
            let mut i = 0;
            while i < 12 {
                a[i].offset = (i * 16) as u64;
                a[i].shader_location = i as u32;
                i += 1;
            }
            a
        };
        crate::gpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayLayerInstance>() as crate::gpu::BufferAddress,
            step_mode: crate::gpu::VertexStepMode::Instance,
            attributes: &ATTRS,
        }
    }

    /// The byte offset of instance `index`, which a draw binds the stream at.
    pub(crate) fn byte_offset(index: u32) -> crate::gpu::BufferAddress {
        index as crate::gpu::BufferAddress
            * std::mem::size_of::<OverlayLayerInstance>() as crate::gpu::BufferAddress
    }
}

/// The code `overlay_layer.wgsl` switches on for a blend formula; zero for a
/// mode a blend state draws.
pub(crate) fn blend_code(blend: crate::renderer::OverlayBlend) -> f32 {
    use crate::renderer::OverlayBlend as B;
    match blend {
        B::Multiply => 1.0,
        B::Screen => 2.0,
        B::Overlay => 3.0,
        B::Darken => 4.0,
        B::Lighten => 5.0,
        B::ColourDodge => 6.0,
        B::ColourBurn => 7.0,
        B::HardLight => 8.0,
        B::SoftLight => 9.0,
        B::Difference => 10.0,
        B::Exclusion => 11.0,
        B::Hue => 12.0,
        B::Saturation => 13.0,
        B::Colour => 14.0,
        B::Luminosity => 15.0,
        _ => 0.0,
    }
}

impl crate::resources::DeviceResources {
    /// The layer composite's layouts and shader, built on first use.
    pub(crate) fn ensure_overlay_layer_resources(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        // The backdrop pipeline's group 1 (clip shapes, viewport, shadow
        // layers) is the composite's group 1 too.
        self.ensure_overlay_shape_tex_pipeline(device);
        if self.overlay_layer.bgl.is_some() {
            return;
        }
        let texture = |binding| crate::gpu::BindGroupLayoutEntry {
            binding,
            visibility: crate::gpu::ShaderStages::FRAGMENT,
            ty: crate::gpu::BindingType::Texture {
                sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                view_dimension: crate::gpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("overlay_layer_bgl"),
            entries: &[
                texture(0),
                texture(1),
                texture(2),
                texture(3),
                texture(4),
                crate::gpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let clip_bgl = self
            .overlay_shape
            .tex_clip_bgl
            .as_ref()
            .expect("built above");
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "overlay_layer_layout",
            &[&bgl, clip_bgl],
        );
        let shader = crate::resources::builders::wgsl_module(
            device,
            "overlay_layer.wgsl",
            crate::resources::builders::wgsl_source!("overlay_layer"),
        );
        let sampler =
            crate::resources::builders::clamp_linear_sampler(device, "overlay_layer_sampler");
        let blank = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("overlay_layer_blank"),
            size: crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba8Unorm,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            crate::gpu::TexelCopyTextureInfo {
                texture: &blank,
                mip_level: 0,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            &[0u8; 4],
            crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.overlay_layer.bgl = Some(bgl);
        self.overlay_layer.layout = Some(layout);
        self.overlay_layer.shader = Some(shader);
        self.overlay_layer.sampler = Some(sampler);
        self.overlay_layer.blank = Some(blank.create_view(&Default::default()));
    }

    /// The composite pipeline for `key`, built on first use.
    pub(crate) fn ensure_overlay_layer_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        key: LayerPipelineKey,
    ) {
        if self.overlay_layer.pipeline(key).is_some() {
            return;
        }
        let (Some(layout), Some(shader)) = (
            self.overlay_layer.layout.as_ref(),
            self.overlay_layer.shader.as_ref(),
        ) else {
            return;
        };
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        use crate::gpu::BlendFactor as F;
        let premul_over = crate::gpu::BlendState {
            color: crate::gpu::BlendComponent {
                src_factor: F::One,
                dst_factor: F::OneMinusSrcAlpha,
                operation: crate::gpu::BlendOperation::Add,
            },
            alpha: crate::gpu::BlendComponent {
                src_factor: F::One,
                dst_factor: F::OneMinusSrcAlpha,
                operation: crate::gpu::BlendOperation::Add,
            },
        };
        let (blend, entry) = match key {
            LayerPipelineKey::Composite(blend)
                if blend != crate::renderer::OverlayBlend::SourceOver
                    && !blend.reads_destination() =>
            {
                (
                    crate::resources::overlay::overlay_shape::overlay_blend_target(blend).0,
                    "fs_composite",
                )
            }
            LayerPipelineKey::Composite(_) => (premul_over, "fs_composite"),
            LayerPipelineKey::CompositeDst => (premul_over, "fs_composite_dst"),
            LayerPipelineKey::Backdrop => (premul_over, "fs_backdrop"),
        };
        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "overlay_layer_pipeline",
                layout,
                vertex_module: shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[OverlayLayerInstance::buffer_layout()],
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
                    entry_point: Some(entry),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(blend),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                cache: None,
            },
        );
        self.overlay_layer.pipelines.push((key, pipeline));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_is_twelve_vec4s() {
        assert_eq!(std::mem::size_of::<OverlayLayerInstance>(), 12 * 16);
    }

    #[test]
    fn every_destination_mode_has_a_formula() {
        use crate::renderer::OverlayBlend as B;
        for b in [
            B::Overlay,
            B::Darken,
            B::Lighten,
            B::ColourDodge,
            B::ColourBurn,
            B::HardLight,
            B::SoftLight,
            B::Difference,
            B::Exclusion,
            B::Hue,
            B::Saturation,
            B::Colour,
            B::Luminosity,
        ] {
            assert!(b.reads_destination());
            assert!(blend_code(b) > 0.0, "{b:?}");
        }
        assert!(!B::Multiply.reads_destination());
    }
}
