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
use std::time::Instant;
use viewport_lib as vpl;
use viewport_lib::Colour;
use viewport_lib::input::adapters::{AdapterCtx, winit as vpl_winit};

use vpl::wgpu;
use vpl::{
    BindingPreset, ButtonState, Camera, CameraFrame, EffectsFrame, FrameData, InputOwnership,
    LightingSettings, Material, MeshId, MeshItem, OrbitCameraController, OverlayFill, OverlayShape,
    OverlayShapeItem, PostProcessSettings, SceneFrame, ViewportContext, ViewportEvent,
    ViewportInput, ViewportRenderer, primitives,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
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
    renderer: ViewportRenderer,
    camera: Camera,
    controller: OrbitCameraController,
    /// When the last frame was drawn, for the frame time fly movement reads.
    last_frame: Instant,
    /// Owns the bindings and resolves events into an `ActionFrame`. The
    /// controller keeps only its sensitivities and applies the frame.
    input: ViewportInput,

    scene_items: Vec<MeshItem>,

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
        let required_features = ViewportRenderer::recommended_device_features(&adapter);
        if !required_features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
            eprintln!("INDIRECT_FIRST_INSTANCE not supported -- GPU culling will be disabled");
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features,
            required_limits: ViewportRenderer::recommended_device_limits(&adapter),
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

        let mut make_item = |mesh_id: MeshId, [x, y, z]: [f32; 3], colour: [f32; 3]| {
            let mut material = Material::from_colour(Colour::from_linear_rgb_array(colour));
            // Mild emissive just above 1.0 puts a small amount of HDR energy into
            // the scene. Bloom extracts this and makes the glow visible without
            // washing out the object colour.
            material.emissive =
                Colour::linear_rgb(colour[0] * 1.2, colour[1] * 1.2, colour[2] * 1.2);
            let mut item = MeshItem::new(mesh_id, res.upload_material(material).unwrap());
            item.model = glam::Mat4::from_translation(glam::Vec3::new(x, y, z)).into();
            item
        };

        let scene_items = vec![
            make_item(m_sphere, [-2.5, 0.0, 0.0], [0.9, 0.5, 0.2]),
            make_item(m_cube, [0.0, 0.0, 0.0], [0.4, 0.6, 0.9]),
            make_item(m_torus, [2.5, 0.0, 0.0], [0.3, 0.8, 0.4]),
        ];

        let camera = Camera::orbit(glam::DVec3::ZERO, 10.0, Camera::default().orientation());

        let controller = OrbitCameraController::new_stateless();
        let mut input = ViewportInput::from_preset(BindingPreset::Viewer);
        input.begin_frame(ViewportContext::new(
            InputOwnership::full_window(),
            [config.width as f32, config.height as f32],
        ));

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_config: config,
            renderer,
            camera,
            controller,
            last_frame: Instant::now(),
            input,
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

                let action_frame = state.input.resolve();
                let now = Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.controller.apply(&mut state.camera, &action_frame, dt);
                state.camera.set_aspect_ratio(w, h);

                let mut frame_data = FrameData::new(
                    CameraFrame::from_camera(&state.camera, [w, h]),
                    SceneFrame::from_surface_items(state.scene_items.clone()),
                );
                let mut effects = EffectsFrame::default();
                effects.lighting = LightingSettings::default();
                effects.post_process = {
                    let mut _t = PostProcessSettings::default();
                    _t.bloom.enabled = state.bloom;
                    _t.bloom.threshold = 1.0;
                    _t.bloom.intensity = 0.15;
                    _t.fxaa = state.fxaa;
                    _t.ssao = state.ssao;
                    _t
                };
                frame_data.effects = effects;

                // Backdrop blur demo: a large frosted-glass circle in the centre.
                frame_data.overlays.shapes.push(
                    OverlayShapeItem::new(
                        OverlayShape::Circle,
                        [w * 0.5 - 100.0, h * 0.5 - 100.0],
                        [200.0, 200.0],
                    )
                    .with_fill(OverlayFill::Solid(Colour::linear(1.0, 1.0, 1.0, 0.1)))
                    .with_outline(
                        Colour::linear(1.0, 1.0, 1.0, 0.4),
                        1.5,
                        vpl::OutlineMode::Inset,
                    )
                    .with_backdrop_blur(20.0),
                );

                // render() owns the full HDR pipeline:
                //   prepare -> shadow pass -> HDR scene -> post-process -> tone map -> output_view
                // It returns a CommandBuffer ready to submit.
                let cmd =
                    state
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

                state
                    .input
                    .begin_frame(ViewportContext::new(InputOwnership::full_window(), [w, h]));
                // Pan reads the viewport height off the controller.
                state.controller.set_viewport_size([w, h]);
            }

            // Pointer, wheel, keys, modifiers and focus. The viewport fills the
            // window, so it owns everything; positions stay in physical pixels,
            // the units the surface size is given in.
            other => {
                let mut events = Vec::new();
                vpl_winit::translate(&other, &AdapterCtx::default(), &mut events);
                if !events.is_empty() {
                    state.window.request_redraw();
                }
                for ev in events {
                    // The toggles match the character the key types, so they
                    // follow the keyboard layout.
                    if let ViewportEvent::Key {
                        state: ButtonState::Pressed,
                        logical: Some(ch),
                        ..
                    } = ev
                    {
                        match ch {
                            'b' => {
                                state.bloom = !state.bloom;
                                eprintln!("bloom: {}", state.bloom);
                            }
                            'f' => {
                                state.fxaa = !state.fxaa;
                                eprintln!("fxaa: {}", state.fxaa);
                            }
                            's' => {
                                state.ssao = !state.ssao;
                                eprintln!("ssao: {}", state.ssao);
                            }
                            'd' => {
                                state.diag = !state.diag;
                                eprintln!("diagnostics: {}", state.diag);
                            }
                            _ => {}
                        }
                    }
                    state.input.push_event(ev);
                }
            }
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.run_app(&mut App::default()).expect("run");
}
