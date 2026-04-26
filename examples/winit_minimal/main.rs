//! Minimal viewport-lib example using winit + wgpu.
//!
//! Navigation:
//!   Left drag / Middle drag   : orbit
//!   Right drag                : pan
//!   Scroll                    : zoom
//!
//! Object manipulation (click a primitive to select it):
//!   G / R / S                 : grab / rotate / scale
//!   X / Y / Z                 : constrain to axis
//!   Enter or click            : confirm
//!   Escape                    : cancel

use std::sync::Arc;

use viewport_lib::{
    ButtonState, Camera, CameraFrame, FrameData, LightingSettings, ManipResult, MeshId,
    ManipulationContext, ManipulationController, Material, OrbitCameraController, SceneFrame,
    SceneRenderItem, ScrollUnits, ViewportContext, ViewportEvent, ViewportRenderer, primitives,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

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
    manip: ManipulationController,

    // Scene state
    scene_items: Vec<SceneRenderItem>,
    /// Index of the selected primitive, if any.
    selected: Option<usize>,
    /// Snapshot of transforms taken when a manipulation starts.
    transforms_snapshot: Vec<[[f32; 4]; 4]>,

    // Per-frame cursor tracking (needed for ManipulationContext)
    cursor_pos: Option<glam::Vec2>,
    cursor_prev: Option<glam::Vec2>,
    left_pressed_this_frame: bool,
    left_held: bool,
    drag_started_this_frame: bool,
    clicked_this_frame: bool,
    /// Approximate pixel threshold for click vs drag detection.
    press_origin: Option<glam::Vec2>,
}

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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("viewport-lib : Minimal")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
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
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
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
        let depth_view = make_depth_view(&device, config.width, config.height);

        let mut renderer = ViewportRenderer::new(&device, format);
        let res = renderer.resources_mut();

        let m_sphere = res
            .upload_mesh_data(&device, &primitives::sphere(0.6, 24, 12))
            .unwrap();
        let m_cube = res
            .upload_mesh_data(&device, &primitives::cube(1.0))
            .unwrap();
        let m_torus = res
            .upload_mesh_data(&device, &primitives::torus(0.5, 0.18, 32, 16))
            .unwrap();

        let mut make_item = |mesh_id: MeshId, [x, y, z]: [f32; 3], color: [f32; 3]| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model = glam::Mat4::from_translation(glam::Vec3::new(x, y, z)).to_cols_array_2d();
            item.material = Material::from_color(color);
            item.two_sided = true;
            item
        };

        let scene_items = vec![
            make_item(m_sphere, [-2.5, 0.0, 0.0], [0.9, 0.5, 0.2]),
            make_item(m_cube, [0.0, 0.0, 0.0], [0.4, 0.6, 0.9]),
            make_item(m_torus, [2.5, 0.0, 0.0], [0.3, 0.8, 0.4]),
        ];

        let camera = Camera {
            distance: 10.0,
            ..Camera::default()
        };

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
            manip: ManipulationController::new(),
            scene_items,
            selected: None,
            transforms_snapshot: Vec::new(),
            cursor_pos: None,
            cursor_prev: None,
            left_pressed_this_frame: false,
            left_held: false,
            drag_started_this_frame: false,
            clicked_this_frame: false,
            press_origin: None,
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
                    state.depth_view = make_depth_view(&state.device, sz.width, sz.height);
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
                let pressed = btn_state == ElementState::Pressed;
                let vp_state = if pressed {
                    ButtonState::Pressed
                } else {
                    ButtonState::Released
                };

                state.controller.push_event(ViewportEvent::MouseButton {
                    button: vp_button,
                    state: vp_state,
                });

                if button == MouseButton::Left {
                    if pressed {
                        state.left_held = true;
                        state.press_origin = state.cursor_pos;
                        state.drag_started_this_frame = true;
                        state.left_pressed_this_frame = true;
                    } else {
                        // Distinguish click from drag by displacement.
                        let is_click = state
                            .press_origin
                            .zip(state.cursor_pos)
                            .map(|(o, c)| (c - o).length() < 5.0)
                            .unwrap_or(false);
                        if is_click {
                            state.clicked_this_frame = true;
                        }
                        state.left_held = false;
                        state.press_origin = None;
                    }
                }

                state.window.request_redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                let pos = glam::Vec2::new(position.x as f32, position.y as f32);
                state.cursor_prev = state.cursor_pos;
                state.cursor_pos = Some(pos);
                state
                    .controller
                    .push_event(ViewportEvent::PointerMoved { position: pos });
                state.window.request_redraw();
            }

            WindowEvent::CursorLeft { .. } => {
                state.cursor_pos = None;
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
                        eprintln!("surface error: {e:?}");
                        return;
                    }
                };

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let w = state.surface_config.width as f32;
                let h = state.surface_config.height as f32;

                // Build ManipulationContext for this frame.
                let selection_center = state.selected.map(|i| {
                    let col = state.scene_items[i].model[3];
                    glam::Vec3::new(col[0], col[1], col[2])
                });
                let pointer_delta = state
                    .cursor_pos
                    .zip(state.cursor_prev)
                    .map(|(c, p)| c - p)
                    .unwrap_or(glam::Vec2::ZERO);

                let manip_ctx = ManipulationContext {
                    camera: state.camera.clone(),
                    viewport_size: glam::Vec2::new(w, h),
                    cursor_viewport: state.cursor_pos,
                    pointer_delta,
                    selection_center,
                    gizmo: None,
                    drag_started: state.drag_started_this_frame,
                    dragging: state.left_held,
                    clicked: state.clicked_this_frame,
                };

                // Drive manipulation : suppress orbit while active.
                let action_frame = if state.manip.is_active() {
                    let frame = state.controller.resolve();
                    state.camera.set_aspect_ratio(w, h);
                    frame
                } else {
                    let frame = state.controller.apply_to_camera(&mut state.camera);
                    state.camera.set_aspect_ratio(w, h);
                    frame
                };

                match state.manip.update(&action_frame, manip_ctx) {
                    ManipResult::Update(delta) => {
                        if let Some(idx) = state.selected {
                            let current =
                                glam::Mat4::from_cols_array_2d(&state.scene_items[idx].model);
                            let delta_mat = glam::Mat4::from_scale_rotation_translation(
                                delta.scale,
                                delta.rotation,
                                delta.translation,
                            );
                            state.scene_items[idx].model = (delta_mat * current).to_cols_array_2d();
                        }
                    }
                    ManipResult::Commit => {}
                    ManipResult::Cancel | ManipResult::ConstraintChanged => {
                        for (item, snap) in state
                            .scene_items
                            .iter_mut()
                            .zip(state.transforms_snapshot.iter())
                        {
                            item.model = *snap;
                        }
                    }
                    ManipResult::None => {
                        // Take a fresh snapshot each frame while idle.
                        state.transforms_snapshot =
                            state.scene_items.iter().map(|i| i.model).collect();
                    }
                }

                // Reset per-frame flags.
                state.drag_started_this_frame = false;
                state.clicked_this_frame = false;
                state.left_pressed_this_frame = false;

                let mut frame_data = FrameData::new(
                    CameraFrame::from_camera(&state.camera, [w, h]),
                    SceneFrame::from_surface_items(state.scene_items.clone()),
                );
                frame_data.effects.lighting = LightingSettings::default();

                state
                    .renderer
                    .prepare(&state.device, &state.queue, &frame_data);

                let mut encoder = state
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
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
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.run_app(&mut App::default()).expect("run");
}
