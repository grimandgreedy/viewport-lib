//! Primitives showcase — every geometry primitive displayed in a single viewport.
//!
//! Navigation:
//!   Left drag         — orbit
//!   Right drag        — pan
//!   Middle drag       — orbit  |  Middle + Shift drag — pan
//!   Scroll            — zoom
//!   Ctrl  + Scroll    — orbit (yaw + pitch from scroll X / Y)
//!   Shift + Scroll    — pan   (right / up from scroll X / Y)

use std::sync::Arc;

use viewport_lib::{
    Camera, CameraFrame, FrameData, LightingSettings, Material, SceneFrame,
    SceneRenderItem, ViewportRenderer, primitives,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

// ---------------------------------------------------------------------------
// Navigation constants — match the egui template exactly
// ---------------------------------------------------------------------------

const ORBIT_SENSITIVITY: f32 = 0.005;
const ZOOM_SENSITIVITY: f32 = 0.001;
const MIN_DISTANCE: f32 = 0.1;
const MAX_DISTANCE: f32 = 1e15;

// ---------------------------------------------------------------------------
// Scene layout helper
// ---------------------------------------------------------------------------

fn item(mesh_index: usize, x: f32, y: f32, z: f32, color: [f32; 3]) -> SceneRenderItem {
    let mut s = SceneRenderItem::default();
    s.mesh_index = mesh_index;
    s.model = glam::Mat4::from_translation(glam::Vec3::new(x, y, z)).to_cols_array_2d();
    s.material = Material { base_color: color, ..Material::default() };
    s.two_sided = true;
    s
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

#[derive(Default)]
struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,

    renderer: ViewportRenderer,
    camera: Camera,

    // Input tracking
    left_pressed: bool,
    right_pressed: bool,
    middle_pressed: bool,
    shift_held: bool,
    ctrl_held: bool,
    last_cursor: PhysicalPosition<f64>,

    // Pre-built scene (constant across frames)
    scene_items: Vec<SceneRenderItem>,
}

impl AppState {
    fn make_depth_view(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("depth"),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Orbit the camera: yaw around world Y, pitch around camera-local X.
    fn orbit(&mut self, dx: f32, dy: f32) {
        let q_yaw = glam::Quat::from_rotation_y(-dx * ORBIT_SENSITIVITY);
        let q_pitch = glam::Quat::from_rotation_x(-dy * ORBIT_SENSITIVITY);
        self.camera.orientation = (q_yaw * self.camera.orientation * q_pitch).normalize();
    }

    /// Pan the camera: move look-at centre in the camera's right/up plane.
    fn pan(&mut self, dx: f32, dy: f32) {
        let h = self.surface_config.height as f32;
        let pan_scale =
            2.0 * self.camera.distance * (self.camera.fov_y / 2.0).tan() / h.max(1.0);
        self.camera.center -= self.camera.right() * dx * pan_scale;
        self.camera.center += self.camera.up() * dy * pan_scale;
    }
}

// ---------------------------------------------------------------------------
// winit ApplicationHandler
// ---------------------------------------------------------------------------

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("viewport-lib — Primitives Showcase")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 800u32)),
                )
                .expect("window"),
        );

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("primitives_device"),
            ..Default::default()
        }))
        .expect("device");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format =
            caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let depth_view = AppState::make_depth_view(&device, config.width, config.height);

        let mut renderer = ViewportRenderer::new(&device, format);
        let res = renderer.resources_mut();

        // ---- Upload all primitive meshes ----
        macro_rules! mesh {
            ($data:expr) => {
                res.upload_mesh_data(&device, &$data).expect("upload mesh")
            };
        }

        // Row 0 — basic solids
        let m_cube      = mesh!(primitives::cube(1.0));
        let m_cuboid    = mesh!(primitives::cuboid(2.0, 0.75, 1.0));
        let m_sphere    = mesh!(primitives::sphere(0.6, 32, 16));
        let m_icosphere = mesh!(primitives::icosphere(0.6, 3));

        // Row 1 — round / capped
        let m_ellipsoid  = mesh!(primitives::ellipsoid(0.9, 0.5, 0.6, 28, 14));
        let m_hemisphere = mesh!(primitives::hemisphere(0.65, 32, 16));
        let m_cone       = mesh!(primitives::cone(0.55, 1.1, 28));
        let m_cylinder   = mesh!(primitives::cylinder(0.4, 1.1, 28));

        // Row 2 — curved surfaces
        let m_capsule = mesh!(primitives::capsule(0.38, 1.4, 24, 16));
        let m_torus   = mesh!(primitives::torus(0.55, 0.2, 40, 24));
        let m_disk    = mesh!(primitives::disk(0.65, 40));
        let m_ring    = mesh!(primitives::ring(0.3, 0.65, 48));

        // Row 3 — flat / composite
        let m_plane      = mesh!(primitives::plane(1.8, 1.8));
        let m_grid_plane = mesh!(primitives::grid_plane(1.8, 1.8, 8, 8));
        let m_frustum =
            mesh!(primitives::frustum(std::f32::consts::FRAC_PI_3, 16.0 / 9.0, 0.3, 2.5));
        let m_arrow = mesh!(primitives::arrow(0.07, 0.18, 0.28, 24));

        // Row 4 — spring (two variants)
        let m_spring_a = mesh!(primitives::spring(0.35, 0.08, 5.0, 14));
        let m_spring_b = mesh!(primitives::spring(0.28, 0.12, 3.0, 18));

        // ---- Build scene ----
        // 4-column × 5-row grid, 3.5 units between cells.
        let cx = [-5.25f32, -1.75, 1.75, 5.25];
        let rz = [-7.0f32, -3.5, 0.0, 3.5, 7.0];

        let scene_items: Vec<SceneRenderItem> = vec![
            // Row 0: basic solids
            item(m_cube,      cx[0], 0.0, rz[0], [0.75, 0.75, 0.75]),
            item(m_cuboid,    cx[1], 0.0, rz[0], [0.35, 0.55, 0.90]),
            item(m_sphere,    cx[2], 0.0, rz[0], [0.90, 0.50, 0.20]),
            item(m_icosphere, cx[3], 0.0, rz[0], [0.25, 0.75, 0.40]),

            // Row 1: round / capped
            item(m_ellipsoid,  cx[0], 0.0, rz[1], [0.70, 0.30, 0.85]),
            item(m_hemisphere, cx[1], 0.0, rz[1], [0.50, 0.80, 0.35]),
            item(m_cone,       cx[2], 0.0, rz[1], [0.85, 0.20, 0.25]),
            item(m_cylinder,   cx[3], 0.0, rz[1], [0.20, 0.70, 0.80]),

            // Row 2: curved surfaces
            item(m_capsule, cx[0], 0.0, rz[2], [0.90, 0.45, 0.65]),
            item(m_torus,   cx[1], 0.0, rz[2], [0.85, 0.70, 0.15]),
            item(m_disk,    cx[2], 0.0, rz[2], [0.40, 0.65, 0.95]),
            item(m_ring,    cx[3], 0.0, rz[2], [0.55, 0.90, 0.30]),

            // Row 3: flat / composite
            item(m_plane,      cx[0], 0.0, rz[3], [0.60, 0.55, 0.75]),
            item(m_grid_plane, cx[1], 0.0, rz[3], [0.80, 0.50, 0.25]),
            item(m_frustum,    cx[2], 0.0, rz[3], [0.30, 0.65, 0.90]),
            item(m_arrow,      cx[3], 0.0, rz[3], [0.90, 0.25, 0.30]),

            // Row 4: spring variants
            item(m_spring_a, cx[0], 0.0, rz[4], [0.85, 0.30, 0.75]),
            item(m_spring_b, cx[1], 0.0, rz[4], [0.55, 0.35, 0.90]),
        ];

        let camera = Camera {
            center: glam::Vec3::ZERO,
            distance: 28.0,
            orientation: glam::Quat::from_rotation_x(-0.55),
            ..Camera::default()
        };

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_config: config,
            depth_view,
            renderer,
            camera,
            left_pressed: false,
            right_pressed: false,
            middle_pressed: false,
            shift_held: false,
            ctrl_held: false,
            last_cursor: PhysicalPosition::new(0.0, 0.0),
            scene_items,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(sz) => {
                if sz.width > 0 && sz.height > 0 {
                    state.surface_config.width = sz.width;
                    state.surface_config.height = sz.height;
                    state.surface.configure(&state.device, &state.surface_config);
                    state.depth_view =
                        AppState::make_depth_view(&state.device, sz.width, sz.height);
                    state.window.request_redraw();
                }
            }

            WindowEvent::ModifiersChanged(mods) => {
                state.shift_held = mods.state().shift_key();
                state.ctrl_held = mods.state().control_key();
            }

            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                let pressed = btn_state == ElementState::Pressed;
                match button {
                    MouseButton::Left => state.left_pressed = pressed,
                    MouseButton::Middle => state.middle_pressed = pressed,
                    MouseButton::Right => state.right_pressed = pressed,
                    _ => {}
                }
                state.window.request_redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                let dx = (position.x - state.last_cursor.x) as f32;
                let dy = (position.y - state.last_cursor.y) as f32;
                state.last_cursor = position;

                let any_drag =
                    state.left_pressed || state.middle_pressed || state.right_pressed;
                if !any_drag || (dx.abs() < 0.001 && dy.abs() < 0.001) {
                    return;
                }

                // Pan: right-drag, or middle + shift.
                let is_pan =
                    state.right_pressed || (state.middle_pressed && state.shift_held);

                if is_pan {
                    state.pan(dx, dy);
                } else {
                    // Orbit: left-drag or middle-drag (without shift).
                    state.orbit(dx, dy);
                }

                state.window.request_redraw();
            }

            WindowEvent::MouseWheel { delta, .. } => {
                // For orbit/pan from scroll we use the raw line count (~1.0 per notch)
                // so that ORBIT_SENSITIVITY stays consistent with drag-based orbit.
                // For zoom we scale up to pixel-equivalent amounts.
                let (raw_dx, raw_dy, zoom_dy) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x, y, y * 28.0),
                    MouseScrollDelta::PixelDelta(px) => {
                        let x = px.x as f32;
                        let y = px.y as f32;
                        (x, y, y)
                    }
                };

                if state.ctrl_held {
                    // Ctrl+scroll → orbit (X = yaw, Y = pitch).
                    state.orbit(raw_dx, raw_dy);
                } else if state.shift_held {
                    // Shift+scroll → pan (X = right, Y = up).
                    state.pan(raw_dx, raw_dy);
                } else {
                    // Plain scroll → zoom.
                    state.camera.distance =
                        (state.camera.distance * (1.0 - zoom_dy * ZOOM_SENSITIVITY))
                            .clamp(MIN_DISTANCE, MAX_DISTANCE);
                }

                state.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                let frame = match state.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state.surface.configure(&state.device, &state.surface_config);
                        return;
                    }
                    Err(e) => {
                        eprintln!("Surface error: {e:?}");
                        return;
                    }
                };

                let view =
                    frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let w = state.surface_config.width as f32;
                let h = state.surface_config.height as f32;

                state.camera.aspect = if h > 0.0 { w / h } else { 1.0 };

                let mut frame_data = FrameData::new(
                    CameraFrame::from_camera(&state.camera, [w, h]),
                    SceneFrame::from_surface_items(state.scene_items.clone()),
                );
                frame_data.effects.lighting = LightingSettings::default();
                frame_data.viewport.show_grid = true;
                frame_data.viewport.show_axes_indicator = true;

                state.renderer.prepare(&state.device, &state.queue, &frame_data);

                let mut encoder =
                    state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("primitives_encoder"),
                    });

                {
                    let mut pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("primitives_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.09,
                                        g: 0.09,
                                        b: 0.11,
                                        a: 1.0,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &state.depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Discard,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });

                    pass.set_viewport(0.0, 0.0, w, h, 0.0, 1.0);
                    state.renderer.paint_to(&mut pass, &frame_data);
                }

                state.queue.submit(std::iter::once(encoder.finish()));
                frame.present();
            }

            _ => {}
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("viewport_lib=warn")),
        )
        .init();

    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run");
}
