//! The example itself: three primitives on a grid, orbited with touch.
//!
//! The same code runs on Android, on iOS, and on the desktop, and the input
//! handling is the same on all three: translate the winit event with `from_winit`,
//! push it into a `ViewportInput`, and apply the frame it resolves to the camera.
//! One finger orbits, two pan, a pinch zooms and a twist turns; none of that is
//! written here.

use std::sync::Arc;
use std::time::Instant;

use viewport_lib::interaction::input::adapters::from_winit;
use viewport_lib::{
    Camera, CameraFrame, FrameData, LightingSettings, Material, OrbitCameraController,
    PostProcessSettings, SceneFrame, SceneRenderItem, ViewportContext, ViewportInput,
    ViewportRenderer, primitives, viewport_default_bindings,
};
use winit::application::ApplicationHandler;
use winit::error::EventLoopError;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

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
    renderer: ViewportRenderer,
    camera: Camera,
    controller: OrbitCameraController,
    input: ViewportInput,
    scene_items: Vec<SceneRenderItem>,
    /// Elapsed-seconds clock. The viewport reads no clock of its own, and the
    /// double-tap and long-press recognisers need one.
    start: Instant,
}

// ---------------------------------------------------------------------------
// ApplicationHandler
// ---------------------------------------------------------------------------

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        // On a phone or tablet the window fills the screen; size attributes are
        // ignored.
        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default())
                .expect("window"),
        );

        // Android ships both a Vulkan and a GLES driver and wgpu will take
        // either. Ask for Vulkan so the backend does not change between
        // devices.
        #[cfg(target_os = "android")]
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });
        #[cfg(not(target_os = "android"))]
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
            wgpu::Features::empty()
        };
        // Mobile adapters sit below the wgpu defaults on several limits, so take
        // whatever this one reports rather than the downlevel defaults.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features,
            required_limits: adapter.limits(),
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
        let mut renderer = ViewportRenderer::new(&device, format);
        let res = renderer.resources_mut();

        let m_sphere = res
            .upload_mesh_data(&device, &primitives::sphere(0.6, 24, 12))
            .unwrap();
        let m_cube = res
            .upload_mesh_data(&device, &primitives::cube(1.0))
            .unwrap();
        let m_cylinder = res
            .upload_mesh_data(&device, &primitives::cylinder(0.4, 1.2, 20))
            .unwrap();

        let make_item = |mesh_id, [x, y, z]: [f32; 3], colour: [f32; 3]| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model = glam::Mat4::from_translation(glam::Vec3::new(x, y, z)).to_cols_array_2d();
            item.material = Material::from_colour(colour);
            item
        };

        let scene_items = vec![
            make_item(m_sphere, [-2.5, 0.0, 0.0], [0.9, 0.5, 0.2]),
            make_item(m_cube, [0.0, 0.0, 0.0], [0.4, 0.6, 0.9]),
            make_item(m_cylinder, [2.5, 0.0, 0.0], [0.3, 0.8, 0.4]),
        ];

        let camera = Camera {
            distance: 10.0,
            ..Camera::default()
        };

        // The controller applies a resolved frame; the bindings live on the
        // resolver beside it, which is the one place that decides what a gesture
        // means.
        let controller = OrbitCameraController::new_stateless();
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(ViewportContext {
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
            input,
            scene_items,
            start: Instant::now(),
        });

        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // Backgrounding the app invalidates the surface. Drop everything here;
        // resumed() will build it again.
        self.state = None;
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
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
                if state.surface_config.width == 0 {
                    return;
                }
                let frame = match state.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state
                            .surface
                            .configure(&state.device, &state.surface_config);
                        return;
                    }
                    Err(_) => return,
                };

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let w = state.surface_config.width as f32;
                let h = state.surface_config.height as f32;

                let actions = state.input.resolve();
                state.controller.apply(&mut state.camera, &actions);
                state.camera.set_aspect_ratio(w, h);

                let mut frame_data = FrameData::new(
                    CameraFrame::from_camera(&state.camera, [w, h]),
                    SceneFrame::from_surface_items(state.scene_items.clone()),
                );
                frame_data.effects.lighting = LightingSettings::default();
                let mut post_process = PostProcessSettings::default();
                post_process.bloom.enabled = true;
                post_process.bloom.threshold = 1.0;
                post_process.bloom.intensity = 0.15;
                frame_data.effects.post_process = post_process;
                frame_data.viewport.show_grid = true;
                frame_data.viewport.show_axes_indicator = true;

                let cmd =
                    state
                        .renderer
                        .owned()
                        .render(&state.device, &state.queue, &view, &frame_data);
                state.queue.submit(std::iter::once(cmd));
                frame.present();

                state.input.begin_frame_at(
                    ViewportContext {
                        hovered: true,
                        focused: true,
                        viewport_size: [w, h],
                    },
                    state.start.elapsed().as_secs_f32(),
                );
            }

            // Everything else, touch included, goes through the adapter. The
            // viewport fills the window, so there is no rect to offset by.
            other => {
                let scale = state.window.scale_factor() as f32;
                if let Some(ev) = from_winit(&other, scale) {
                    state.input.push_event(ev);
                    state.window.request_redraw();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Event loop
// ---------------------------------------------------------------------------

/// Run the example on an already-built event loop.
///
/// Android reuses one process across activity restarts, so a second call can
/// come back as `RecreationAttempt`. That is not a failure: return quietly and
/// let the running loop carry on.
pub fn run(event_loop: EventLoop<()>) {
    match event_loop.run_app(&mut App::default()) {
        Ok(()) | Err(EventLoopError::RecreationAttempt) => {}
        Err(e) => panic!("run: {e}"),
    }
}

/// Build the default event loop and run the example. Used on iOS and on the
/// desktop; Android builds its loop from the `AndroidApp` handle instead.
pub fn start() {
    match EventLoop::builder().build() {
        Ok(event_loop) => run(event_loop),
        Err(EventLoopError::RecreationAttempt) => {}
        Err(e) => panic!("event loop: {e}"),
    }
}
