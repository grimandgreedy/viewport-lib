//! Bridge between GTK4 and `viewport-lib::ViewportRenderer`.
//!
//! Renders the scene into an offscreen texture on our own wgpu device,
//! reads the pixels back to CPU, and returns a raw RGBA pixel buffer.
//! The caller (main.rs) wraps the pixels in a `gdk4::MemoryTexture` for
//! display in a `gtk4::Picture` widget.
//!
//! This is the same CPU-readback pattern used by the Slint example - the
//! only difference is the return type (raw bytes instead of `slint::Image`).

use std::collections::HashMap;

use viewport_lib::{
    Camera, CameraUniform, FrameData, GizmoAxis, GizmoMode, LightingSettings, MeshData,
    SceneRenderItem, ViewportRenderer,
};
use wgpu;

/// Offscreen viewport renderer that produces a raw RGBA pixel buffer.
pub struct SceneRenderer {
    renderer: ViewportRenderer,
    camera: Camera,
    uploaded: HashMap<u64, usize>,
    color_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
    staging_buffer: Option<wgpu::Buffer>,
    row_stride: u32,
    tex_w: u32,
    tex_h: u32,
}

impl SceneRenderer {
    const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;
    const BYTES_PER_PIXEL: u32 = 4;
    const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue) -> Self {
        let renderer = ViewportRenderer::new(device, Self::COLOR_FORMAT);
        let camera = Camera {
            center: glam::Vec3::ZERO,
            distance: 12.0,
            orientation: glam::Quat::from_rotation_y(0.6) * glam::Quat::from_rotation_x(0.4),
            ..Camera::default()
        };

        Self {
            renderer,
            camera,
            uploaded: HashMap::new(),
            color_texture: None,
            depth_view: None,
            staging_buffer: None,
            row_stride: 0,
            tex_w: 0,
            tex_h: 0,
        }
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    fn aligned_row_stride(width: u32) -> u32 {
        let unpadded = width * Self::BYTES_PER_PIXEL;
        let align = Self::COPY_BYTES_PER_ROW_ALIGNMENT;
        (unpadded + align - 1) / align * align
    }

    /// Render the scene and return a tightly-packed RGBA pixel buffer.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        objects: &[(u64, [f32; 3])],
    ) -> Vec<u8> {
        let w = width.max(1);
        let h = height.max(1);

        // Resize textures and staging buffer if needed.
        if w != self.tex_w || h != self.tex_h {
            let color = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gtk4_viewport_color"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Self::COLOR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });

            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gtk4_viewport_depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Self::DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

            let row_stride = Self::aligned_row_stride(w);
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gtk4_viewport_staging"),
                size: (row_stride * h) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            self.depth_view = Some(depth.create_view(&wgpu::TextureViewDescriptor::default()));
            self.color_texture = Some(color);
            self.staging_buffer = Some(staging);
            self.row_stride = row_stride;
            self.tex_w = w;
            self.tex_h = h;
        }

        // Upload meshes for any new objects.
        let box_mesh = unit_box_mesh();
        for &(id, _) in objects {
            if !self.uploaded.contains_key(&id) {
                let idx = self
                    .renderer
                    .resources_mut()
                    .upload_mesh_data(device, &box_mesh)
                    .expect("built-in mesh");
                self.uploaded.insert(id, idx);
            }
        }

        // Update camera aspect.
        self.camera.aspect = w as f32 / h as f32;

        // Build scene items.
        let scene_items: Vec<SceneRenderItem> = objects
            .iter()
            .filter_map(|&(id, position)| {
                let mesh_index = *self.uploaded.get(&id)?;
                let model = glam::Mat4::from_translation(glam::Vec3::from(position));
                let mut item = SceneRenderItem::default();
                item.mesh_index = mesh_index;
                item.model = model.to_cols_array_2d();
                Some(item)
            })
            .collect();

        let camera_uniform = CameraUniform {
            view_proj: self.camera.view_proj_matrix().to_cols_array_2d(),
            eye_pos: self.camera.eye_position().into(),
            _pad: 0.0,
        };

        let frame_data = {
            let mut fd = FrameData::default();
            fd.camera_uniform = camera_uniform;
            fd.lighting = LightingSettings::default();
            fd.eye_pos = self.camera.eye_position().into();
            fd.scene_items = scene_items;
            fd.show_grid = true;
            fd.show_axes_indicator = true;
            fd.viewport_size = [w as f32, h as f32];
            fd.camera_orientation = self.camera.orientation;
            fd
        };

        self.renderer.prepare(device, queue, &frame_data);

        // Render into offscreen texture then copy to staging buffer.
        let color_texture = self.color_texture.as_ref().unwrap();
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = self.depth_view.as_ref().unwrap();
        let staging = self.staging_buffer.as_ref().unwrap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gtk4_viewport_encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gtk4_viewport_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.12,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
            self.renderer.paint_to(&mut render_pass, &frame_data);
        }

        // Copy texture -> staging buffer.
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.row_stride),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer and read pixels.
        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        rx.recv().unwrap().expect("Failed to map staging buffer");

        let data = buffer_slice.get_mapped_range();
        let row_bytes = (w * Self::BYTES_PER_PIXEL) as usize;
        let stride = self.row_stride as usize;

        // Strip row padding to get a tightly-packed RGBA pixel buffer.
        let mut pixels = Vec::with_capacity(row_bytes * h as usize);
        for row in 0..h as usize {
            let start = row * stride;
            pixels.extend_from_slice(&data[start..start + row_bytes]);
        }
        drop(data);
        staging.unmap();

        pixels
    }
}

// ---------------------------------------------------------------------------
// Box mesh helper
// ---------------------------------------------------------------------------

fn unit_box_mesh() -> MeshData {
    #[rustfmt::skip]
    let positions: Vec<[f32; 3]> = vec![
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
        [ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
        [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5],
        [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
    ];

    #[rustfmt::skip]
    let normals: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    ];

    #[rustfmt::skip]
    let indices: Vec<u32> = vec![
        0,  1,  2,  0,  2,  3,
        4,  5,  6,  4,  6,  7,
        8,  9,  10, 8,  10, 11,
        12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19,
        20, 21, 22, 20, 22, 23,
    ];

    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}
