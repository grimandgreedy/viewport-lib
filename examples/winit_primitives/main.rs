//! Primitives showcase : every geometry primitive displayed in a single viewport.
//!
//! Navigation:
//!   Left drag         : orbit
//!   Right drag        : pan
//!   Middle drag       : orbit  |  Middle + Shift drag : pan
//!   Scroll            : zoom
//!   Ctrl  + Scroll    : orbit (yaw + pitch from scroll X / Y)
//!   Shift + Scroll    : pan   (right / up from scroll X / Y)

use std::sync::Arc;

use viewport_lib::{ButtonState, ScrollUnits};
use viewport_lib::{
    Camera, CameraFrame, FrameData, LightingSettings, Material, MeshId, OrbitCameraController,
    SceneFrame, SceneRenderItem, ViewportContext, ViewportEvent, ViewportRenderer, primitives,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

// ---------------------------------------------------------------------------
// Scene layout helper
// ---------------------------------------------------------------------------

fn item(mesh_id: MeshId, x: f32, y: f32, z: f32, color: [f32; 3]) -> SceneRenderItem {
    let mut s = SceneRenderItem::default();
    s.mesh_id = mesh_id;
    s.model = glam::Mat4::from_translation(glam::Vec3::new(x, y, z)).to_cols_array_2d();
    s.material = Material::from_color(color);
    s.material.backface_policy = viewport_lib::BackfacePolicy::Identical;
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
    controller: OrbitCameraController,

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
                        .with_title("viewport-lib : Primitives Showcase")
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
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
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

        // Row 0 : basic solids
        let m_cube = mesh!(primitives::cube(1.0));
        let m_cuboid = mesh!(primitives::cuboid(2.0, 0.75, 1.0));
        let m_sphere = mesh!(primitives::sphere(0.6, 32, 16));
        let m_icosphere = mesh!(primitives::icosphere(0.6, 3));

        // Row 1 : round / capped
        let m_ellipsoid = mesh!(primitives::ellipsoid(0.9, 0.5, 0.6, 28, 14));
        let m_hemisphere = mesh!(primitives::hemisphere(0.65, 32, 16));
        let m_cone = mesh!(primitives::cone(0.55, 1.1, 28));
        let m_cylinder = mesh!(primitives::cylinder(0.4, 1.1, 28));

        // Row 2 : curved surfaces
        let m_capsule = mesh!(primitives::capsule(0.38, 1.4, 24, 16));
        let m_torus = mesh!(primitives::torus(0.55, 0.2, 40, 24));
        let m_disk = mesh!(primitives::disk(0.65, 40));
        let m_ring = mesh!(primitives::ring(0.3, 0.65, 48));

        // Row 3 : flat / composite
        let m_plane = mesh!(primitives::plane(1.8, 1.8));
        let m_grid_plane = mesh!(primitives::grid_plane(1.8, 1.8, 8, 8));
        let m_frustum = mesh!(primitives::frustum(
            std::f32::consts::FRAC_PI_3,
            16.0 / 9.0,
            0.3,
            2.5
        ));
        let m_arrow = mesh!(primitives::arrow(0.07, 0.18, 0.28, 24));

        // Row 4 : spring (two variants)
        let m_spring_a = mesh!(primitives::spring(0.35, 0.08, 5.0, 14));
        let m_spring_b = mesh!(primitives::spring(0.28, 0.12, 3.0, 18));

        // ---- Build scene ----
        // 4-column × 5-row grid, 3.5 units between cells.
        let cx = [-5.25f32, -1.75, 1.75, 5.25];
        let rz = [-7.0f32, -3.5, 0.0, 3.5, 7.0];

        let scene_items: Vec<SceneRenderItem> = vec![
            // Row 0: basic solids
            item(m_cube, cx[0], 0.0, rz[0], [0.75, 0.75, 0.75]),
            item(m_cuboid, cx[1], 0.0, rz[0], [0.35, 0.55, 0.90]),
            item(m_sphere, cx[2], 0.0, rz[0], [0.90, 0.50, 0.20]),
            item(m_icosphere, cx[3], 0.0, rz[0], [0.25, 0.75, 0.40]),
            // Row 1: round / capped
            item(m_ellipsoid, cx[0], 0.0, rz[1], [0.70, 0.30, 0.85]),
            item(m_hemisphere, cx[1], 0.0, rz[1], [0.50, 0.80, 0.35]),
            item(m_cone, cx[2], 0.0, rz[1], [0.85, 0.20, 0.25]),
            item(m_cylinder, cx[3], 0.0, rz[1], [0.20, 0.70, 0.80]),
            // Row 2: curved surfaces
            item(m_capsule, cx[0], 0.0, rz[2], [0.90, 0.45, 0.65]),
            item(m_torus, cx[1], 0.0, rz[2], [0.85, 0.70, 0.15]),
            item(m_disk, cx[2], 0.0, rz[2], [0.40, 0.65, 0.95]),
            item(m_ring, cx[3], 0.0, rz[2], [0.55, 0.90, 0.30]),
            // Row 3: flat / composite
            item(m_plane, cx[0], 0.0, rz[3], [0.60, 0.55, 0.75]),
            item(m_grid_plane, cx[1], 0.0, rz[3], [0.80, 0.50, 0.25]),
            item(m_frustum, cx[2], 0.0, rz[3], [0.30, 0.65, 0.90]),
            item(m_arrow, cx[3], 0.0, rz[3], [0.90, 0.25, 0.30]),
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

        // Prime the controller for the first frame of events.
        let mut controller = OrbitCameraController::viewport_primitives();
        controller.begin_frame(ViewportContext {
            hovered: true,
            focused: true,
            viewport_size: [config.width as f32, config.height as f32],
        });

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_config: config,
            depth_view,
            renderer,
            camera,
            controller,
            scene_items,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(sz) => {
                if sz.width > 0 && sz.height > 0 {
                    state.surface_config.width = sz.width;
                    state.surface_config.height = sz.height;
                    state
                        .surface
                        .configure(&state.device, &state.surface_config);
                    state.depth_view =
                        AppState::make_depth_view(&state.device, sz.width, sz.height);
                    state.window.request_redraw();
                }
            }

            WindowEvent::ModifiersChanged(mods) => {
                let mut m = viewport_lib::Modifiers::default();
                m.shift = mods.state().shift_key();
                m.ctrl = mods.state().control_key();
                m.alt = mods.state().alt_key();
                state
                    .controller
                    .push_event(ViewportEvent::ModifiersChanged(m));
            }

            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                let vp_button = match button {
                    MouseButton::Left => viewport_lib::MouseButton::Left,
                    MouseButton::Middle => viewport_lib::MouseButton::Middle,
                    MouseButton::Right => viewport_lib::MouseButton::Right,
                    _ => return,
                };
                let vp_state = if btn_state == ElementState::Pressed {
                    ButtonState::Pressed
                } else {
                    ButtonState::Released
                };
                state.controller.push_event(ViewportEvent::MouseButton {
                    button: vp_button,
                    state: vp_state,
                });
                state.window.request_redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                state.controller.push_event(ViewportEvent::PointerMoved {
                    position: glam::Vec2::new(position.x as f32, position.y as f32),
                });
                state.window.request_redraw();
            }

            WindowEvent::CursorLeft { .. } => {
                state.controller.push_event(ViewportEvent::PointerLeft);
            }

            WindowEvent::Focused(false) => {
                state.controller.push_event(ViewportEvent::FocusLost);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let (d, units) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        (glam::Vec2::new(x, y), ScrollUnits::Lines)
                    }
                    MouseScrollDelta::PixelDelta(px) => (
                        glam::Vec2::new(px.x as f32, px.y as f32),
                        ScrollUnits::Pixels,
                    ),
                };
                state
                    .controller
                    .push_event(ViewportEvent::Wheel { delta: d, units });
                state.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                let frame = match state.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state
                            .surface
                            .configure(&state.device, &state.surface_config);
                        return;
                    }
                    Err(e) => {
                        eprintln!("Surface error: {e:?}");
                        return;
                    }
                };

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let w = state.surface_config.width as f32;
                let h = state.surface_config.height as f32;

                // Apply accumulated events to camera.
                state.controller.apply_to_camera(&mut state.camera);
                state.camera.set_aspect_ratio(w, h);

                let mut frame_data = FrameData::new(
                    CameraFrame::from_camera(&state.camera, [w, h]),
                    SceneFrame::from_surface_items(state.scene_items.clone()),
                );
                frame_data.effects.lighting = LightingSettings::default();
                frame_data.viewport.show_grid = true;
                frame_data.viewport.show_axes_indicator = true;

                state
                    .renderer
                    .prepare(&state.device, &state.queue, &frame_data);

                let mut encoder =
                    state
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("primitives_encoder"),
                        });

                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &state.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Discard,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    pass.set_viewport(0.0, 0.0, w, h, 0.0, 1.0);
                    state.renderer.paint_to(&mut pass, &frame_data);
                }

                state.queue.submit(std::iter::once(encoder.finish()));
                frame.present();

                // Begin accumulation for the next frame's events.
                state.controller.begin_frame(ViewportContext {
                    hovered: true,
                    focused: true,
                    viewport_size: [w, h],
                });
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
