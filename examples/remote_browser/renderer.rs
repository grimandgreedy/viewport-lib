//! Headless render thread. Owns the GPU device and ViewportRenderer.

use std::sync::mpsc::{Receiver, Sender};

use viewport_lib::{
    Camera, CameraFrame, FrameData, LightingSettings, Material, MeshId, SceneFrame,
    SceneRenderItem, ViewportRenderer, primitives,
};

pub struct RenderRequest {
    pub camera: Camera,
    pub width: u32,
    pub height: u32,
}

pub struct RenderFrame {
    /// Tightly-packed RGBA8 pixels, width * height * 4 bytes.
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub fn run(rx: Receiver<RenderRequest>, tx: Sender<RenderFrame>) {
    let Some((device, queue)) = headless_device() else {
        eprintln!("renderer: no GPU adapter available");
        return;
    };

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let scene_items = {
        let res = renderer.resources_mut();
        let sphere = res
            .upload_mesh_data(&device, &primitives::sphere(0.6, 24, 12))
            .unwrap();
        let cube = res
            .upload_mesh_data(&device, &primitives::cube(1.0))
            .unwrap();
        let torus = res
            .upload_mesh_data(&device, &primitives::torus(0.5, 0.18, 32, 16))
            .unwrap();
        build_scene(sphere, cube, torus)
    };

    loop {
        let mut req = match rx.recv() {
            Ok(r) => r,
            Err(_) => break,
        };
        while let Ok(newer) = rx.try_recv() {
            req = newer;
        }

        let camera_frame =
            CameraFrame::from_camera(&req.camera, [req.width as f32, req.height as f32]);
        let mut frame = FrameData::new(
            camera_frame,
            SceneFrame::from_surface_items(scene_items.clone()),
        );
        frame.effects.lighting = LightingSettings::default();

        let pixels =
            renderer.render_offscreen(&device, &queue, &frame, req.width, req.height);

        if tx.send(RenderFrame { pixels, width: req.width, height: req.height }).is_err() {
            break;
        }
    }
}

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("remote-browser-server"),
            ..Default::default()
        },
    ))
    .ok()?;
    Some((device, queue))
}

fn build_scene(m_sphere: MeshId, m_cube: MeshId, m_torus: MeshId) -> Vec<SceneRenderItem> {
    let make = |mesh_id: MeshId, [x, y, z]: [f32; 3], color: [f32; 3]| {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.model =
            glam::Mat4::from_translation(glam::Vec3::new(x, y, z)).to_cols_array_2d();
        item.material = Material::from_color(color);
        item
    };
    vec![
        make(m_sphere, [-2.5, 0.0, 0.0], [0.9, 0.5, 0.2]),
        make(m_cube, [0.0, 0.0, 0.0], [0.4, 0.6, 0.9]),
        make(m_torus, [2.5, 0.0, 0.0], [0.3, 0.8, 0.4]),
    ]
}
