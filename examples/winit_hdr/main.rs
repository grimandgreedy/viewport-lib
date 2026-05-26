//! winit-hdr: exercises the renderer.render() path against a raw wgpu surface.
//!
//! renderer.render() owns the full pipeline: uploads uniforms, runs the shadow
//! pass, renders the scene into an internal HDR texture, applies post-process
//! (bloom, tone mapping, FXAA), and blits the final image to the provided
//! output view. No manual render pass or depth buffer needed from the caller.
//!
//! Navigation:
//!   Left drag / Middle drag   : orbit
//!   Right drag                : pan
//!   Scroll                    : zoom
//!
//! Press B to toggle bloom, F to toggle FXAA, S to toggle SSAO.

use std::sync::Arc;

use viewport_lib::{
    ButtonState, Camera, CameraFrame, EffectsFrame, FrameData, LightingSettings, Material, MeshId,
    OrbitCameraController, OverlayFill, OverlayShape, OverlayShapeItem, PostProcessSettings,
    SceneFrame, SceneRenderItem, ScrollUnits, ViewportContext, ViewportEvent, ViewportRenderer,
    primitives,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
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
    renderer: ViewportRenderer,
    camera: Camera,
    controller: OrbitCameraController,

    scene_items: Vec<SceneRenderItem>,

    // Post-process toggles
    bloom: bool,
    fxaa: bool,
    ssao: bool,

    // Diagnostics toggle (D key)
    diag: bool,
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
                        .with_title("viewport-lib : HDR (render)")
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
        let required_features = if adapter
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
        {
            wgpu::Features::INDIRECT_FIRST_INSTANCE
        } else {
            eprintln!("INDIRECT_FIRST_INSTANCE not supported -- GPU culling will be disabled");
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features,
            ..Default::default()
        }))
        .expect("device");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        // Prefer sRGB: the tone mapper outputs linear values and relies on the
        // hardware sRGB write conversion to encode gamma correctly.
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

        let make_item = |mesh_id: MeshId, [x, y, z]: [f32; 3], colour: [f32; 3]| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model = glam::Mat4::from_translation(glam::Vec3::new(x, y, z)).to_cols_array_2d();
            item.material = Material::from_colour(colour);
            // Mild emissive just above 1.0 puts a small amount of HDR energy into
            // the scene. Bloom extracts this and makes the glow visible without
            // washing out the object colour.
            item.material.emissive = [colour[0] * 1.2, colour[1] * 1.2, colour[2] * 1.2];
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
            renderer,
            camera,
            controller,
            scene_items,
            bloom: true,
            fxaa: false,
            ssao: false,
            diag: false,
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
                let pos = glam::Vec2::new(position.x as f32, position.y as f32);
                state
                    .controller
                    .push_event(ViewportEvent::PointerMoved { position: pos });
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

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyB) => {
                            state.bloom = !state.bloom;
                            eprintln!("bloom: {}", state.bloom);
                        }
                        PhysicalKey::Code(KeyCode::KeyF) => {
                            state.fxaa = !state.fxaa;
                            eprintln!("fxaa: {}", state.fxaa);
                        }
                        PhysicalKey::Code(KeyCode::KeyS) => {
                            state.ssao = !state.ssao;
                            eprintln!("ssao: {}", state.ssao);
                        }
                        PhysicalKey::Code(KeyCode::KeyD) => {
                            state.diag = !state.diag;
                            eprintln!("diagnostics: {}", state.diag);
                        }
                        _ => {}
                    }
                    state.window.request_redraw();
                }
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

                state.controller.apply_to_camera(&mut state.camera);
                state.camera.set_aspect_ratio(w, h);

                let mut frame_data = FrameData::new(
                    CameraFrame::from_camera(&state.camera, [w, h]),
                    SceneFrame::from_surface_items(state.scene_items.clone()),
                );
                let mut effects = EffectsFrame::default();
                effects.lighting = LightingSettings::default();
                effects.post_process = {
                    let mut _t = PostProcessSettings::default();
                    _t.enabled = true;
                    _t.bloom = state.bloom;
                    _t.bloom_threshold = 1.0;
                    _t.bloom_intensity = 0.15;
                    _t.fxaa = state.fxaa;
                    _t.ssao = state.ssao;
                    _t
                };
                frame_data.effects = effects;

                // Backdrop blur demo: a large frosted-glass circle in the centre.
                frame_data.overlays.shapes.push(OverlayShapeItem {
                    position: [w * 0.5 - 100.0, h * 0.5 - 100.0],
                    size: [200.0, 200.0],
                    shape: OverlayShape::Circle,
                    fill: OverlayFill::Solid([1.0, 1.0, 1.0, 0.1]),
                    border_colour: [1.0, 1.0, 1.0, 0.4],
                    border_width: 1.5,
                    backdrop_blur: 20.0,
                    ..Default::default()
                });

                // render() owns the full HDR pipeline:
                //   prepare -> shadow pass -> HDR scene -> post-process -> tone map -> output_view
                // It returns a CommandBuffer ready to submit.
                let cmd = state
                    .renderer
                    .owned()
                    .render(&state.device, &state.queue, &view, &frame_data);

                state.queue.submit(std::iter::once(cmd));
                frame.present();

                if state.diag {
                    let s = state.renderer.last_frame_stats();
                    eprintln!(
                        "cull_active={} visible={:?} draws={} batches={}",
                        s.gpu_culling_active,
                        s.gpu_visible_instances,
                        s.draw_calls,
                        s.instanced_batches,
                    );
                }

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
