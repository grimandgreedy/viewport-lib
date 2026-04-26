//! Multi-viewport example : quad-view CAD layout.
//!
//! Shows how to use the split `prepare_scene` / `prepare_viewport` / `paint_viewport`
//! API to render the same scene from four independent cameras in one frame:
//!
//! ```text
//! ┌──────────────┬──────────────┐
//! │  Perspective │  Top (ortho) │
//! │   (orbit)    │  looking -Z  │
//! ├──────────────┼──────────────┤
//! │ Front (ortho)│ Right (ortho)│
//! │  looking -Y  │  looking -X  │
//! └──────────────┴──────────────┘
//! ```
//!
//! The scene is built once and shared across all viewports. Each viewport gets
//! its own camera and `OrbitCameraController`. Mouse input is routed to
//! whichever quadrant the cursor is currently in.

use std::sync::Arc;

use viewport_lib::{ButtonState, Modifiers, MouseButton, ScrollUnits};
use viewport_lib::{
    Camera, CameraFrame, FrameData, LightingSettings, MeshId, OrbitCameraController, Projection,
    SceneFrame, SceneRenderItem, ViewportContext, ViewportEvent, ViewportId, ViewportRenderer,
    primitives,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton as WinitButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Event loop error");
}

// ---------------------------------------------------------------------------
// Quadrant helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Quad {
    TopLeft = 0,
    TopRight = 1,
    BottomLeft = 2,
    BottomRight = 3,
}

fn quad_from_pos(x: f32, y: f32, w: f32, h: f32) -> Quad {
    let half_w = w / 2.0;
    let half_h = h / 2.0;
    match (x >= half_w, y >= half_h) {
        (false, false) => Quad::TopLeft,
        (true, false) => Quad::TopRight,
        (false, true) => Quad::BottomLeft,
        (true, true) => Quad::BottomRight,
    }
}

/// Return (x_offset, y_offset, width, height) for a quadrant in physical pixels.
fn quad_rect(quad: Quad, w: u32, h: u32) -> (u32, u32, u32, u32) {
    let hw = w / 2;
    let hh = h / 2;
    // Use ceiling for the second half to handle odd dimensions.
    let hw2 = w - hw;
    let hh2 = h - hh;
    match quad {
        Quad::TopLeft => (0, 0, hw, hh),
        Quad::TopRight => (hw, 0, hw2, hh),
        Quad::BottomLeft => (0, hh, hw, hh2),
        Quad::BottomRight => (hw, hh, hw2, hh2),
    }
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
    mesh_id: MeshId,

    /// One camera + controller per quadrant.
    cameras: [Camera; 4],
    controllers: [OrbitCameraController; 4],

    /// Viewport handles (one per quadrant, created in order).
    viewports: [ViewportId; 4],

    /// Which quadrant the mouse is currently over.
    hovered_quad: Quad,
    cursor_pos: glam::Vec2,
}

impl AppState {
    fn create_depth_view(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
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
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Build the scene items (a small grid of coloured cubes). Shared across all viewports.
    fn build_scene(&self) -> SceneFrame {
        let positions_colors: &[([f32; 3], [f32; 4])] = &[
            ([-2.0, -2.0, 0.0], [0.9, 0.3, 0.3, 1.0]),
            ([2.0, -2.0, 0.0], [0.3, 0.9, 0.3, 1.0]),
            ([-2.0, 2.0, 0.0], [0.3, 0.3, 0.9, 1.0]),
            ([2.0, 2.0, 0.0], [0.9, 0.9, 0.3, 1.0]),
            ([0.0, 0.0, 2.0], [0.9, 0.5, 0.1, 1.0]),
        ];
        let items: Vec<SceneRenderItem> = positions_colors
            .iter()
            .map(|&(pos, color)| {
                let mut item = SceneRenderItem::default();
                item.mesh_id = self.mesh_id;
                item.model = glam::Mat4::from_translation(glam::Vec3::from(pos)).to_cols_array_2d();
                item.material.base_color = [color[0], color[1], color[2]];
                item
            })
            .collect();
        SceneFrame::from_surface_items(items)
    }

    /// Build a `FrameData` for the given quadrant.
    fn build_frame(&self, quad: Quad, vp_id: ViewportId, w: u32, h: u32) -> FrameData {
        let (ox, oy, qw, qh) = quad_rect(quad, w, h);
        let _ = (ox, oy); // pixel offsets handled by render-pass viewport/scissor

        let cam = &self.cameras[quad as usize];
        let size = [qw as f32, qh as f32];
        let camera_frame = CameraFrame::from_camera(cam, size).with_viewport_index(vp_id.0);

        let mut fd = FrameData::new(camera_frame, self.build_scene());
        fd.effects.lighting = LightingSettings::default();
        fd.viewport.show_grid = true;
        fd.viewport.show_axes_indicator = true;
        fd
    }
}

// ---------------------------------------------------------------------------
// ApplicationHandler
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
                        .with_title("viewport-lib : Multi-Viewport (quad-view)")
                        .with_inner_size(winit::dpi::LogicalSize::new(1200u32, 900u32)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("No suitable GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("multi_viewport_device"),
            ..Default::default()
        }))
        .expect("Failed to create device");

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

        let depth_view = AppState::create_depth_view(&device, config.width, config.height);

        let mut renderer = ViewportRenderer::new(&device, format);
        let mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&device, &primitives::cube(1.0))
            .expect("cube mesh");

        // Create four viewport slots.
        let vp0 = renderer.create_viewport(&device); // TopLeft:     perspective
        let vp1 = renderer.create_viewport(&device); // TopRight:    top (ortho)
        let vp2 = renderer.create_viewport(&device); // BottomLeft:  front (ortho)
        let vp3 = renderer.create_viewport(&device); // BottomRight: right (ortho)

        // Perspective camera : standard orbit view.
        let cam_persp = Camera {
            center: glam::Vec3::ZERO,
            distance: 12.0,
            orientation: glam::Quat::from_rotation_z(0.6) * glam::Quat::from_rotation_x(1.1),
            ..Camera::default()
        };

        // Top view: looking straight down (-Z direction).
        let cam_top = Camera {
            projection: Projection::Orthographic,
            center: glam::Vec3::ZERO,
            distance: 10.0,
            // from_rotation_x(-PI/2) puts the camera directly above looking down.
            orientation: glam::Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            ..Camera::default()
        };

        // Front view: looking along -Y.
        let cam_front = Camera {
            projection: Projection::Orthographic,
            center: glam::Vec3::ZERO,
            distance: 10.0,
            orientation: glam::Quat::IDENTITY,
            ..Camera::default()
        };

        // Right view: looking along -X (rotate 90° around Z then front).
        let cam_right = Camera {
            projection: Projection::Orthographic,
            center: glam::Vec3::ZERO,
            distance: 10.0,
            orientation: glam::Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2),
            ..Camera::default()
        };

        let make_controller = || OrbitCameraController::viewport_primitives();

        let hw = config.width as f32 / 2.0;
        let hh = config.height as f32 / 2.0;
        let mut ctrl0 = make_controller();
        ctrl0.begin_frame(ViewportContext {
            hovered: true,
            focused: true,
            viewport_size: [hw, hh],
        });
        let mut ctrl1 = make_controller();
        ctrl1.begin_frame(ViewportContext {
            hovered: false,
            focused: false,
            viewport_size: [hw, hh],
        });
        let mut ctrl2 = make_controller();
        ctrl2.begin_frame(ViewportContext {
            hovered: false,
            focused: false,
            viewport_size: [hw, hh],
        });
        let mut ctrl3 = make_controller();
        ctrl3.begin_frame(ViewportContext {
            hovered: false,
            focused: false,
            viewport_size: [hw, hh],
        });

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_config: config,
            depth_view,
            renderer,
            mesh_id,
            cameras: [cam_persp, cam_top, cam_front, cam_right],
            controllers: [ctrl0, ctrl1, ctrl2, ctrl3],
            viewports: [vp0, vp1, vp2, vp3],
            hovered_quad: Quad::TopLeft,
            cursor_pos: glam::Vec2::ZERO,
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
                let m = Modifiers {
                    shift: mods.state().shift_key(),
                    ctrl: mods.state().control_key(),
                    alt: mods.state().alt_key(),
                };
                // Send to active quadrant only.
                state.controllers[state.hovered_quad as usize]
                    .push_event(ViewportEvent::ModifiersChanged(m));
            }

            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                let vp_btn = match button {
                    WinitButton::Left => MouseButton::Left,
                    WinitButton::Middle => MouseButton::Middle,
                    WinitButton::Right => MouseButton::Right,
                    _ => return,
                };
                let vp_state = if btn_state == ElementState::Pressed {
                    ButtonState::Pressed
                } else {
                    ButtonState::Released
                };
                state.controllers[state.hovered_quad as usize].push_event(
                    ViewportEvent::MouseButton {
                        button: vp_btn,
                        state: vp_state,
                    },
                );
                state.window.request_redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                let x = position.x as f32;
                let y = position.y as f32;
                let w = state.surface_config.width as f32;
                let h = state.surface_config.height as f32;

                let new_quad = quad_from_pos(x, y, w, h);
                if new_quad != state.hovered_quad {
                    // Leave old quadrant.
                    state.controllers[state.hovered_quad as usize]
                        .push_event(ViewportEvent::PointerLeft);
                    state.hovered_quad = new_quad;
                }

                // Translate cursor into quadrant-local coordinates.
                let (ox, oy, qw, qh) = quad_rect(
                    new_quad,
                    state.surface_config.width,
                    state.surface_config.height,
                );
                let local_x = x - ox as f32;
                let local_y = y - oy as f32;
                state.cursor_pos = glam::Vec2::new(local_x, local_y);

                state.controllers[new_quad as usize].push_event(ViewportEvent::PointerMoved {
                    position: state.cursor_pos,
                });
                let _ = (qw, qh);
                state.window.request_redraw();
            }

            WindowEvent::CursorLeft { .. } => {
                state.controllers[state.hovered_quad as usize]
                    .push_event(ViewportEvent::PointerLeft);
            }

            WindowEvent::Focused(false) => {
                for ctrl in &mut state.controllers {
                    ctrl.push_event(ViewportEvent::FocusLost);
                }
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
                state.controllers[state.hovered_quad as usize]
                    .push_event(ViewportEvent::Wheel { delta: d, units });
                state.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                let surf_frame = match state.surface.get_current_texture() {
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

                let output_view = surf_frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let w = state.surface_config.width;
                let h = state.surface_config.height;

                // Apply accumulated events to each camera.
                for (i, (cam, ctrl)) in state
                    .cameras
                    .iter_mut()
                    .zip(state.controllers.iter_mut())
                    .enumerate()
                {
                    ctrl.apply_to_camera(cam);
                    let (_, _, qw, qh) = quad_rect(
                        [
                            Quad::TopLeft,
                            Quad::TopRight,
                            Quad::BottomLeft,
                            Quad::BottomRight,
                        ][i],
                        w,
                        h,
                    );
                    cam.set_aspect_ratio(qw as f32, qh as f32);
                }

                // Build per-viewport FrameData.
                let quads = [
                    Quad::TopLeft,
                    Quad::TopRight,
                    Quad::BottomLeft,
                    Quad::BottomRight,
                ];
                let frames: [FrameData; 4] =
                    std::array::from_fn(|i| state.build_frame(quads[i], state.viewports[i], w, h));

                // Prepare: scene once, then one per viewport.
                let (scene_fx, _) = frames[0].effects.split();
                state
                    .renderer
                    .prepare_scene(&state.device, &state.queue, &frames[0], &scene_fx);
                for (i, frame) in frames.iter().enumerate() {
                    state.renderer.prepare_viewport(
                        &state.device,
                        &state.queue,
                        state.viewports[i],
                        frame,
                    );
                }

                // Render: one pass, four viewport/scissor rects.
                let mut encoder =
                    state
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("multi_viewport_encoder"),
                        });
                {
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("multi_viewport_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &output_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.08,
                                    g: 0.08,
                                    b: 0.10,
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

                    for (i, frame) in frames.iter().enumerate() {
                        let quad = quads[i];
                        let (ox, oy, qw, qh) = quad_rect(quad, w, h);
                        rp.set_viewport(ox as f32, oy as f32, qw as f32, qh as f32, 0.0, 1.0);
                        rp.set_scissor_rect(ox, oy, qw, qh);
                        state
                            .renderer
                            .paint_viewport_to(&mut rp, state.viewports[i], frame);
                    }
                }

                state.queue.submit(std::iter::once(encoder.finish()));
                surf_frame.present();

                // Begin next frame's event accumulation for each controller.
                for (i, ctrl) in state.controllers.iter_mut().enumerate() {
                    let quad = quads[i];
                    let (_, _, qw, qh) = quad_rect(quad, w, h);
                    ctrl.begin_frame(ViewportContext {
                        hovered: quad == state.hovered_quad,
                        focused: quad == state.hovered_quad,
                        viewport_size: [qw as f32, qh as f32],
                    });
                }
            }

            _ => {}
        }
    }
}
