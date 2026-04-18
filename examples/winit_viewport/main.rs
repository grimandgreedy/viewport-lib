//! Minimal `winit` + `wgpu` integration for `viewport-lib`.
//!
//! This example renders directly to a window surface and keeps input handling
//! in the host app. It is the best reference if you want the smallest possible
//! setup without framework-specific glue.

use std::sync::Arc;

use viewport_lib::{
    Camera, CameraFrame, FrameData, LightingSettings, SceneFrame, SceneRenderItem,
    ViewportRenderer, primitives,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

// ---------------------------------------------------------------------------
// Camera control constants shared with the other viewport examples.
// ---------------------------------------------------------------------------

const ORBIT_SENSITIVITY: f32 = 0.005;
const ZOOM_SENSITIVITY: f32 = 0.001;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("viewport_lib=debug")),
        )
        .init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Event loop error");
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
    mesh_index: usize,

    // Mouse tracking state for input handling.
    left_pressed: bool,
    middle_pressed: bool,
    right_pressed: bool,
    shift_held: bool,
    last_cursor: PhysicalPosition<f64>,
}

impl AppState {
    fn create_depth_view(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("winit_depth"),
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
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

// ---------------------------------------------------------------------------
// ApplicationHandler - winit 0.30 event-driven API
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
                        .with_title("viewport-lib - winit Example")
                        .with_inner_size(winit::dpi::LogicalSize::new(900u32, 600u32)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("Failed to find a suitable GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("winit_viewport_device"),
            ..Default::default()
        }))
        .expect("Failed to create wgpu device");

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_view = AppState::create_depth_view(&device, config.width, config.height);

        let mut renderer = ViewportRenderer::new(&device, format);

        let mesh_index = renderer
            .resources_mut()
            .upload_mesh_data(&device, &primitives::cube(1.0))
            .expect("built-in mesh");

        let camera = Camera {
            center: glam::Vec3::ZERO,
            distance: 12.0,
            orientation: glam::Quat::from_rotation_y(0.6) * glam::Quat::from_rotation_x(-0.4),
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
            mesh_index,
            left_pressed: false,
            middle_pressed: false,
            right_pressed: false,
            shift_held: false,
            last_cursor: PhysicalPosition::new(0.0, 0.0),
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
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    state.surface_config.width = new_size.width;
                    state.surface_config.height = new_size.height;
                    state
                        .surface
                        .configure(&state.device, &state.surface_config);
                    state.depth_view =
                        AppState::create_depth_view(&state.device, new_size.width, new_size.height);
                    state.window.request_redraw();
                }
            }

            // --- Input handling ---
            //
            // winit gives us raw OS-level events. We translate them into camera
            // operations using the same pattern as every other framework adapter:
            //   1. Track button state on press/release
            //   2. Compute pixel deltas on cursor move
            //   3. Map button+modifier combos to orbit/pan/zoom
            WindowEvent::ModifiersChanged(mods) => {
                state.shift_held = mods.state().shift_key();
            }

            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                let pressed = btn_state == ElementState::Pressed;
                match button {
                    MouseButton::Left => state.left_pressed = pressed,
                    MouseButton::Middle => state.middle_pressed = pressed,
                    MouseButton::Right => state.right_pressed = pressed,
                    _ => {}
                }
                if pressed {
                    // Reset last_cursor on press so the first delta isn't a jump.
                    // (The actual position is set by the next CursorMoved event,
                    // but this prevents a stale value from causing a large delta.)
                }
                state.window.request_redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                let dx = (position.x - state.last_cursor.x) as f32;
                let dy = (position.y - state.last_cursor.y) as f32;
                state.last_cursor = position;

                let any_drag = state.left_pressed || state.middle_pressed || state.right_pressed;
                if !any_drag || (dx.abs() < 0.001 && dy.abs() < 0.001) {
                    return;
                }

                // Pan: right-drag, or shift+middle-drag.
                let is_pan = state.right_pressed || (state.middle_pressed && state.shift_held);

                if is_pan {
                    let cam = &mut state.camera;
                    let viewport_h = state.surface_config.height as f32;
                    cam.pan_pixels(glam::vec2(dx, dy), viewport_h);
                } else {
                    // Orbit: left-drag or middle-drag (without shift).
                    // Quaternion arcball: world-Y yaw + camera-local-X pitch.
                    state.camera.orbit(dx * ORBIT_SENSITIVITY, dy * ORBIT_SENSITIVITY);
                }

                state.window.request_redraw();
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 28.0,
                    MouseScrollDelta::PixelDelta(px) => px.y as f32,
                };
                state.camera.zoom_by_factor(1.0 - scroll_y * ZOOM_SENSITIVITY);
                state.window.request_redraw();
            }

            // --- Render ---
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

                state.camera.set_aspect_ratio(w, h);

                // Build scene: 4 cubes in a grid.
                let positions = [
                    [-1.5, 0.0, -1.5],
                    [ 1.5, 0.0, -1.5],
                    [-1.5, 0.0,  1.5],
                    [ 1.5, 0.0,  1.5],
                ];
                let scene_items: Vec<SceneRenderItem> = positions
                    .iter()
                    .map(|&pos| {
                        let mut item = SceneRenderItem::default();
                        item.mesh_index = state.mesh_index;
                        item.model = glam::Mat4::from_translation(glam::Vec3::from(pos))
                            .to_cols_array_2d();
                        item
                    })
                    .collect();

                let mut frame_data = FrameData::new(
                    CameraFrame::from_camera(&state.camera, [w, h]),
                    SceneFrame::from_surface_items(scene_items),
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
                            label: Some("winit_viewport_encoder"),
                        });

                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("winit_viewport_render_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
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

                    render_pass.set_viewport(0.0, 0.0, w, h, 0.0, 1.0);
                    state.renderer.paint_to(&mut render_pass, &frame_data);
                }

                state.queue.submit(std::iter::once(encoder.finish()));
                frame.present();
            }

            _ => {}
        }
    }
}
