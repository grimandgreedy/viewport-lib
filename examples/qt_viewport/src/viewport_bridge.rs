//! CPU-readback viewport renderer for Qt integration.
//! Identical pattern to the Slint and GTK4 bridges - renders offscreen and
//! returns raw RGBA pixels.

use std::collections::HashMap;
use viewport_lib::{
    Camera, CameraUniform, FrameData, GizmoAxis, GizmoMode, LightingSettings, MeshData, SceneRenderItem,
    ViewportRenderer,
};

pub struct SceneRenderer {
    renderer: ViewportRenderer,
    pub camera: Camera,
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
    const BPP: u32 = 4;
    const ALIGN: u32 = 256;

    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            renderer: ViewportRenderer::new(device, Self::COLOR_FORMAT),
            camera: Camera {
                center: glam::Vec3::ZERO,
                distance: 12.0,
                yaw: 0.6,
                pitch: 0.4,
                ..Camera::default()
            },
            uploaded: HashMap::new(),
            color_texture: None,
            depth_view: None,
            staging_buffer: None,
            row_stride: 0,
            tex_w: 0,
            tex_h: 0,
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        objects: &[(u64, [f32; 3])],
    ) -> Vec<u8> {
        let w = w.max(1);
        let h = h.max(1);

        if w != self.tex_w || h != self.tex_h {
            let color = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("qt_color"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format: Self::COLOR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("qt_depth"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format: Self::DEPTH_FORMAT, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
            });
            let row_stride = (w * Self::BPP + Self::ALIGN - 1) / Self::ALIGN * Self::ALIGN;
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("qt_staging"), size: (row_stride * h) as u64,
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

        let box_mesh = unit_box_mesh();
        for &(id, _) in objects {
            if !self.uploaded.contains_key(&id) {
                let idx = self.renderer.resources_mut().upload_mesh_data(device, &box_mesh).expect("built-in mesh");
                self.uploaded.insert(id, idx);
            }
        }

        self.camera.aspect = w as f32 / h as f32;

        let scene_items: Vec<SceneRenderItem> = objects.iter().filter_map(|&(id, pos)| {
            let mesh_index = *self.uploaded.get(&id)?;
            let mut item = SceneRenderItem::default();
            item.mesh_index = mesh_index;
            item.model = glam::Mat4::from_translation(glam::Vec3::from(pos)).to_cols_array_2d();
            Some(item)
        }).collect();

        let mut frame_data = FrameData::default();
        frame_data.camera_uniform = CameraUniform {
            view_proj: self.camera.view_proj_matrix().to_cols_array_2d(),
            eye_pos: self.camera.eye_position().into(),
            _pad: 0.0,
        };
        frame_data.lighting = LightingSettings::default();
        frame_data.eye_pos = self.camera.eye_position().into();
        frame_data.scene_items = scene_items;
        frame_data.show_grid = true;
        frame_data.show_axes_indicator = true;
        frame_data.viewport_size = [w as f32, h as f32];
        frame_data.camera_orientation = self.camera.orientation;

        self.renderer.prepare(device, queue, &frame_data);

        let ct = self.color_texture.as_ref().unwrap();
        let cv = ct.create_view(&wgpu::TextureViewDescriptor::default());
        let dv = self.depth_view.as_ref().unwrap();
        let stg = self.staging_buffer.as_ref().unwrap();

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("qt_enc") });
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("qt_rp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &cv, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.12, a: 1.0 }), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: dv, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }), stencil_ops: None,
                }),
                timestamp_writes: None, occlusion_query_set: None,
            });
            rp.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
            self.renderer.paint_to(&mut rp, &frame_data);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: ct, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: stg, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.row_stride), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(std::iter::once(enc.finish()));

        let slice = stg.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(5)) }).unwrap();
        rx.recv().unwrap().expect("map failed");

        let data = slice.get_mapped_range();
        let rb = (w * Self::BPP) as usize;
        let stride = self.row_stride as usize;
        let mut pixels = Vec::with_capacity(rb * h as usize);
        for row in 0..h as usize {
            pixels.extend_from_slice(&data[row * stride..row * stride + rb]);
        }
        drop(data);
        stg.unmap();
        pixels
    }
}

fn unit_box_mesh() -> MeshData {
    #[rustfmt::skip]
    let p: Vec<[f32; 3]> = vec![
        [-0.5,-0.5, 0.5],[0.5,-0.5, 0.5],[0.5, 0.5, 0.5],[-0.5, 0.5, 0.5],
        [0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[0.5, 0.5,-0.5],
        [-0.5, 0.5, 0.5],[0.5, 0.5, 0.5],[0.5, 0.5,-0.5],[-0.5, 0.5,-0.5],
        [-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[0.5,-0.5, 0.5],[-0.5,-0.5, 0.5],
        [0.5,-0.5, 0.5],[0.5,-0.5,-0.5],[0.5, 0.5,-0.5],[0.5, 0.5, 0.5],
        [-0.5,-0.5,-0.5],[-0.5,-0.5, 0.5],[-0.5, 0.5, 0.5],[-0.5, 0.5,-0.5],
    ];
    #[rustfmt::skip]
    let n: Vec<[f32; 3]> = vec![
        [0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0],
        [0.0,0.0,-1.0],[0.0,0.0,-1.0],[0.0,0.0,-1.0],[0.0,0.0,-1.0],
        [0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0],
        [0.0,-1.0,0.0],[0.0,-1.0,0.0],[0.0,-1.0,0.0],[0.0,-1.0,0.0],
        [1.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,0.0],
        [-1.0,0.0,0.0],[-1.0,0.0,0.0],[-1.0,0.0,0.0],[-1.0,0.0,0.0],
    ];
    #[rustfmt::skip]
    let i: Vec<u32> = vec![
        0,1,2,0,2,3, 4,5,6,4,6,7, 8,9,10,8,10,11,
        12,13,14,12,14,15, 16,17,18,16,18,19, 20,21,22,20,22,23,
    ];
    let mut mesh = MeshData::default();
    mesh.positions = p;
    mesh.normals = n;
    mesh.indices = i;
    mesh
}
