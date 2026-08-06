//! Lightmap bake primitives.
//!
//! The texel G-buffer pass ([`rasterize_texel_gbuffer`]) is the hinge a lightmap
//! bake turns on: it rasterises a mesh into its lightmap UV (UV1) space so every
//! atlas texel a chart covers carries the world position and world normal of the
//! surface point it represents. A GI solve then shoots hemisphere rays from those
//! positions and stores the result back into the atlas by texel.
//!
//! This is the one part of a lightmapper that must be a renderer pass. Unwrap +
//! pack, the solve, denoise, and encode belong in an offline bake tool; this
//! primitive lives in core because it is a rasterisation into a GPU target.

use crate::gpu::util::DeviceExt;
use glam::{Mat3, Mat4};

/// Mesh geometry to rasterise into the atlas. The four per-vertex slices are
/// index-parallel: `positions`, `normals`, and `uv1` are one entry per vertex,
/// and `indices` lists triangle corners into them. `uv1` is the lightmap UV set
/// (unique per texel, in `[0, 1]`), not the art UV.
pub struct TexelGeometry<'a> {
    /// Object-space vertex positions.
    pub positions: &'a [[f32; 3]],
    /// Object-space vertex normals.
    pub normals: &'a [[f32; 3]],
    /// Lightmap UV (UV1) per vertex.
    pub uv1: &'a [[f32; 2]],
    /// Triangle indices into the per-vertex slices.
    pub indices: &'a [u32],
    /// Object-to-world transform applied to positions and normals.
    pub model: Mat4,
}

/// Per-texel surface data from [`rasterize_texel_gbuffer`], row-major
/// `width * height` with the atlas origin at the top-left.
pub struct TexelGBuffer {
    /// Atlas width in texels.
    pub width: u32,
    /// Atlas height in texels.
    pub height: u32,
    /// World position in `xyz`; `w` is 1.0 on a covered texel, 0.0 on an empty
    /// one. Check `w` before using a texel: empty texels hold no surface point.
    pub world_pos: Vec<[f32; 4]>,
    /// World normal in `xyz` (normalised on covered texels); `w` is unused.
    pub world_normal: Vec<[f32; 4]>,
}

impl TexelGBuffer {
    /// Whether texel `(x, y)` covers a surface point.
    pub fn is_covered(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        self.world_pos[(y * self.width + x) as usize][3] != 0.0
    }

    /// Number of covered texels.
    pub fn covered_count(&self) -> usize {
        self.world_pos.iter().filter(|p| p[3] != 0.0).count()
    }
}

/// Rasterise `geom` into a `width` x `height` texel G-buffer.
///
/// Each triangle is drawn at its UV1 coordinates in clip space, so it lands in
/// the atlas rect its UVs address; the fragment stage writes the interpolated
/// world position and normal. Texels no triangle covers stay invalid (`w == 0`
/// in [`TexelGBuffer::world_pos`]). Runs headless: no surface, no swapchain.
///
/// An empty mesh (no indices) returns an all-empty buffer without touching the
/// GPU.
pub fn rasterize_texel_gbuffer(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    geom: &TexelGeometry,
    width: u32,
    height: u32,
) -> TexelGBuffer {
    let width = width.max(1);
    let height = height.max(1);
    let texels = (width * height) as usize;

    let vertex_count = geom
        .positions
        .len()
        .min(geom.normals.len())
        .min(geom.uv1.len());
    if geom.indices.is_empty() || vertex_count == 0 {
        return TexelGBuffer {
            width,
            height,
            world_pos: vec![[0.0; 4]; texels],
            world_normal: vec![[0.0; 4]; texels],
        };
    }

    // Interleave position, normal, uv1 (8 floats per vertex) for one vertex
    // buffer matching the shader's vertex layout.
    let mut verts: Vec<f32> = Vec::with_capacity(vertex_count * 8);
    for i in 0..vertex_count {
        verts.extend_from_slice(&geom.positions[i]);
        verts.extend_from_slice(&geom.normals[i]);
        verts.extend_from_slice(&geom.uv1[i]);
    }
    let vertex_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
        label: Some("texel_gbuffer_verts"),
        contents: bytemuck::cast_slice(&verts),
        usage: crate::gpu::BufferUsages::VERTEX,
    });
    let index_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
        label: Some("texel_gbuffer_indices"),
        contents: bytemuck::cast_slice(geom.indices),
        usage: crate::gpu::BufferUsages::INDEX,
    });

    // Uniforms: model and its inverse-transpose (for normals under scale).
    let normal_mat = Mat4::from_mat3(Mat3::from_mat4(geom.model).inverse().transpose());
    let mut uniforms = [0.0f32; 32];
    uniforms[..16].copy_from_slice(&geom.model.to_cols_array());
    uniforms[16..].copy_from_slice(&normal_mat.to_cols_array());
    let uniform_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
        label: Some("texel_gbuffer_uniforms"),
        contents: bytemuck::cast_slice(&uniforms),
        usage: crate::gpu::BufferUsages::UNIFORM,
    });

    let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some("texel_gbuffer_bgl"),
        entries: &[crate::gpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: crate::gpu::ShaderStages::VERTEX,
            ty: crate::gpu::BindingType::Buffer {
                ty: crate::gpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
        label: Some("texel_gbuffer_bg"),
        layout: &bgl,
        entries: &[crate::gpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let shader = crate::resources::builders::wgsl_module(
        device,
        "texel_gbuffer",
        crate::resources::builders::wgsl_source!("texel_gbuffer"),
    );
    let layout =
        crate::resources::builders::pipeline_layout(device, Some("texel_gbuffer_layout"), &[&bgl]);

    let attrs = [
        crate::gpu::VertexAttribute {
            format: crate::gpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        crate::gpu::VertexAttribute {
            format: crate::gpu::VertexFormat::Float32x3,
            offset: 12,
            shader_location: 1,
        },
        crate::gpu::VertexAttribute {
            format: crate::gpu::VertexFormat::Float32x2,
            offset: 24,
            shader_location: 2,
        },
    ];
    let vbuf_layouts = [crate::gpu::VertexBufferLayout {
        array_stride: 32,
        step_mode: crate::gpu::VertexStepMode::Vertex,
        attributes: &attrs,
    }];
    let targets = [
        Some(crate::gpu::ColorTargetState {
            format: crate::gpu::TextureFormat::Rgba32Float,
            blend: None,
            write_mask: crate::gpu::ColorWrites::ALL,
        }),
        Some(crate::gpu::ColorTargetState {
            format: crate::gpu::TextureFormat::Rgba32Float,
            blend: None,
            write_mask: crate::gpu::ColorWrites::ALL,
        }),
    ];
    let pipeline = crate::resources::builders::render_pipeline(
        device,
        crate::resources::builders::RenderPipelineDesc {
            label: "texel_gbuffer_pipeline",
            layout: &layout,
            vertex: crate::gpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vbuf_layouts,
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(crate::gpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &targets,
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            }),
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                // Bake both facings: a triangle's winding in UV space is
                // unrelated to its facing in world space.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: crate::gpu::MultisampleState::default(),
            cache: None,
        },
    );

    let make_target = |label: &str| {
        let tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some(label),
            size: crate::gpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba32Float,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        (tex, view)
    };
    let (pos_tex, pos_view) = make_target("texel_gbuffer_pos");
    let (nrm_tex, nrm_view) = make_target("texel_gbuffer_nrm");

    let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
        label: Some("texel_gbuffer_encoder"),
    });
    {
        let clear = crate::gpu::Operations {
            load: crate::gpu::LoadOp::Clear(crate::gpu::Color::TRANSPARENT),
            store: crate::gpu::StoreOp::Store,
        };
        let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(feature = "wgpu29")]
            multiview_mask: None,
            label: Some("texel_gbuffer_pass"),
            color_attachments: &[
                Some(crate::gpu::RenderPassColorAttachment {
                    view: &pos_view,
                    resolve_target: None,
                    ops: clear,
                    depth_slice: None,
                }),
                Some(crate::gpu::RenderPassColorAttachment {
                    view: &nrm_view,
                    resolve_target: None,
                    ops: clear,
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buf.slice(..));
        pass.set_index_buffer(index_buf.slice(..), crate::gpu::IndexFormat::Uint32);
        pass.draw_indexed(0..geom.indices.len() as u32, 0, 0..1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    let world_pos = readback_rgba32f(device, queue, &pos_tex, width, height);
    let world_normal = readback_rgba32f(device, queue, &nrm_tex, width, height);
    TexelGBuffer {
        width,
        height,
        world_pos,
        world_normal,
    }
}

/// Read an `Rgba32Float` texture back to `width * height` `[f32; 4]` texels,
/// handling the copy row-alignment.
fn readback_rgba32f(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    texture: &crate::gpu::Texture,
    width: u32,
    height: u32,
) -> Vec<[f32; 4]> {
    let bytes_per_pixel = 16u32; // Rgba32Float: four 32-bit channels.
    let unpadded_row = width * bytes_per_pixel;
    let align = crate::gpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_row = (unpadded_row + align - 1) & !(align - 1);
    let buffer_size = (padded_row * height) as u64;

    let staging = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("texel_gbuffer_readback"),
        size: buffer_size,
        usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
        label: Some("texel_gbuffer_readback_encoder"),
    });
    encoder.copy_texture_to_buffer(
        crate::gpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: crate::gpu::Origin3d::ZERO,
            aspect: crate::gpu::TextureAspect::All,
        },
        crate::gpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        crate::gpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(crate::gpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    device
        .poll(crate::gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    let _ = rx.recv().unwrap_or(Err(crate::gpu::BufferAsyncError));

    let mut out: Vec<[f32; 4]> = Vec::with_capacity((width * height) as usize);
    {
        let mapped = staging.slice(..).get_mapped_range();
        let data: &[u8] = &mapped;
        for row in 0..height as usize {
            let start = row * padded_row as usize;
            let row_bytes = &data[start..start + unpadded_row as usize];
            for texel in row_bytes.chunks_exact(16) {
                let f = |o: usize| {
                    f32::from_le_bytes([texel[o], texel[o + 1], texel[o + 2], texel[o + 3]])
                };
                out.push([f(0), f(4), f(8), f(12)]);
            }
        }
    }
    staging.unmap();
    out
}
