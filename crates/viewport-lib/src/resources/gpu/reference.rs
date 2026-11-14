//! The reference target a viewport draws its scene into under a reference
//! presentation, and the pass that scales it into the output.

use crate::resources::DeviceResources;

/// The uniform `reference_present.wgsl` reads.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PresentUniform {
    pub origin_px: [f32; 2],
    pub scale: [f32; 2],
    pub reference_size: [f32; 2],
    pub bars: f32,
    pub _pad: f32,
    pub bar_colour: [f32; 4],
}

/// One viewport's reference target: a colour target of the reference size in
/// the renderer's format, the uniform placing it in the output, and, for
/// screen overlays drawn over the scaled result, a depth target of the
/// output's size for their pass.
pub(crate) struct ReferenceTarget {
    pub _texture: crate::gpu::Texture,
    pub view: crate::gpu::TextureView,
    pub size: [u32; 2],
    pub uniform_buf: crate::gpu::Buffer,
    pub bind_group: crate::gpu::BindGroup,
    /// The uniform last written, so an unchanged placement is not rewritten.
    pub written: Option<PresentUniform>,
    /// Depth for the overlay pass over the output, sized to it.
    pub output_depth: Option<crate::resources::postprocess::targets::DepthTarget>,
}

impl DeviceResources {
    /// Build the present pipeline and its layout on first use.
    pub(crate) fn ensure_reference_present_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.post.reference_present_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("reference_present_bgl"),
            entries: &[
                crate::resources::builders::texture_entry(0, crate::gpu::ShaderStages::FRAGMENT),
                crate::resources::builders::uniform_entry(1, crate::gpu::ShaderStages::FRAGMENT),
            ],
        });
        let shader = crate::resources::builders::wgsl_module(
            device,
            "reference_present_shader",
            crate::resources::builders::wgsl_source!("reference_present"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "reference_present_layout",
            &[&bgl],
        );
        let pipeline = crate::resources::builders::build_fullscreen_pipeline(
            device,
            "reference_present_pipeline",
            &layout,
            &shader,
            self.target_format,
            None,
        );
        self.post.reference_present_bgl = Some(bgl);
        self.post.reference_present_pipeline = Some(pipeline);
    }

    /// A reference target of `size` pixels. The present pipeline must exist.
    pub(crate) fn create_reference_target(
        &self,
        device: &crate::gpu::Device,
        size: [u32; 2],
    ) -> ReferenceTarget {
        let texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("reference_target"),
            size: crate::gpu::Extent3d {
                width: size[0].max(1),
                height: size[1].max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: self.target_format,
            // COPY_SRC and TEXTURE_BINDING so the scene's after-scene encodes
            // and destination-reading draws can read it as they read an
            // output of their own.
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING
                | crate::gpu::TextureUsages::COPY_SRC
                | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("reference_present_uniform"),
            size: std::mem::size_of::<PresentUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = self
            .post
            .reference_present_bgl
            .as_ref()
            .expect("ensure_reference_present_pipeline builds reference_present_bgl");
        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("reference_present_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        ReferenceTarget {
            _texture: texture,
            view,
            size,
            uniform_buf,
            bind_group,
            written: None,
            output_depth: None,
        }
    }
}
