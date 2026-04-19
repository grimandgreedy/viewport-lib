//! Minimal `winit` + `wgpu` integration for `viewport-lib`.
//!
//! This example renders directly to a window surface and keeps input handling
//! in the host app. It is the best reference if you want the smallest possible
//! setup without framework-specific glue.

use std::sync::Arc;

use viewport_lib::{
    Camera, CameraFrame, FrameData, LightingSettings, OrbitCameraController, SceneFrame,
    SceneRenderItem, ViewportContext, ViewportEvent, ViewportRenderer, primitives,
};
use viewport_lib::{ButtonState, ScrollUnits};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

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
    controller: OrbitCameraController,
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
            mesh_index,
            controller,
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

            WindowEvent::ModifiersChanged(mods) => {
                let mut m = viewport_lib::Modifiers::default();
                m.shift = mods.state().shift_key();
                m.ctrl = mods.state().control_key();
                m.alt = mods.state().alt_key();
                state.controller.push_event(ViewportEvent::ModifiersChanged(m));
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
                    MouseScrollDelta::LineDelta(x, y) => (glam::Vec2::new(x, y), ScrollUnits::Lines),
                    MouseScrollDelta::PixelDelta(px) => (
                        glam::Vec2::new(px.x as f32, px.y as f32),
                        ScrollUnits::Pixels,
                    ),
                };
                state.controller.push_event(ViewportEvent::Wheel { delta: d, units });
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

                // Apply accumulated events to camera.
                state.controller.apply_to_camera(&mut state.camera);
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
