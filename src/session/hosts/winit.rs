//! [`ViewportApp`]: a fullscreen winit runner over a [`ViewportSession`].
//!
//! This owns the window, the wgpu bring-up, and the event loop, and drives a
//! [`ViewportSession`] each frame. It is the standalone-application path: the
//! wiring that every hand-written winit example repeats lives here once.

use std::sync::Arc;
use std::time::Instant;

use ::winit::application::ApplicationHandler;
use ::winit::event::WindowEvent;
use ::winit::event_loop::{ActiveEventLoop, EventLoop};
use ::winit::window::{Window, WindowAttributes, WindowId};

use crate::OrbitCameraController;
use crate::interaction::input::ViewportContext;
use crate::interaction::input::adapters::from_winit;
use crate::session::ViewportSession;

/// Window configuration for a [`ViewportApp`].
///
/// Non-exhaustive: build with [`AppConfig::default`] and the `with_*` methods so
/// new options can be added without breaking construction.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Window title.
    pub title: String,
    /// Initial window width in logical pixels.
    pub width: u32,
    /// Initial window height in logical pixels.
    pub height: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            title: "viewport-lib".to_string(),
            width: 1280,
            height: 720,
        }
    }
}

impl AppConfig {
    /// Set the window title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the initial window size in logical pixels.
    pub fn with_window_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
}

/// What the per-frame callback receives: the session (via deref) plus timing.
pub struct FrameCtx<'a> {
    session: &'a mut ViewportSession,
    /// Seconds since the previous frame.
    pub dt: f32,
    /// Seconds since the app started.
    pub time: f32,
}

impl FrameCtx<'_> {
    /// Seconds since the previous frame.
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// Seconds since the app started.
    pub fn time(&self) -> f32 {
        self.time
    }
}

impl std::ops::Deref for FrameCtx<'_> {
    type Target = ViewportSession;

    fn deref(&self) -> &ViewportSession {
        self.session
    }
}

impl std::ops::DerefMut for FrameCtx<'_> {
    fn deref_mut(&mut self) -> &mut ViewportSession {
        self.session
    }
}

/// A fullscreen winit application driving a [`ViewportSession`].
///
/// ```rust,ignore
/// use viewport_lib::session::hosts::{AppConfig, ViewportApp};
/// use viewport_lib::{Material, primitives};
///
/// let mut cube = None;
/// ViewportApp::new(AppConfig::default().with_title("demo"))
///     .setup(|session, device| {
///         let mesh = session.resources_mut()
///             .upload_mesh_data(device, &primitives::cube(1.0)).unwrap();
///         cube = Some(session.scene_mut().add(
///             Some(mesh), glam::Mat4::IDENTITY, Material::from_colour([0.85, 0.25, 0.2])));
///     })
///     .run(move |ctx| {
///         if let Some(id) = cube {
///             ctx.scene_mut().set_local_transform(id, glam::Mat4::from_rotation_z(ctx.time()));
///         }
///     });
/// ```
pub struct ViewportApp {
    config: AppConfig,
    setup: Option<Box<dyn FnOnce(&mut ViewportSession, &crate::gpu::Device)>>,
}

impl ViewportApp {
    /// Create a runner with the given window configuration.
    pub fn new(config: AppConfig) -> Self {
        Self {
            config,
            setup: None,
        }
    }

    /// Register a one-time setup callback, run after the session is created with
    /// the wgpu device in hand: upload meshes, build the initial scene, attach a
    /// runtime.
    pub fn setup(
        mut self,
        setup: impl FnOnce(&mut ViewportSession, &crate::gpu::Device) + 'static,
    ) -> Self {
        self.setup = Some(Box::new(setup));
        self
    }

    /// Run the event loop, calling `callback` once per frame before rendering.
    /// Blocks until the window closes.
    pub fn run(self, callback: impl FnMut(&mut FrameCtx) + 'static) {
        let event_loop = EventLoop::new().expect("event loop");
        let mut runner = Runner {
            config: self.config,
            setup: self.setup,
            callback,
            state: None,
            orbit: OrbitCameraController::viewport_all(),
            last_frame: Instant::now(),
            start: Instant::now(),
        };
        event_loop.run_app(&mut runner).expect("run app");
    }
}

struct RunState {
    window: Arc<Window>,
    surface: crate::gpu::Surface<'static>,
    device: crate::gpu::Device,
    queue: crate::gpu::Queue,
    surface_config: crate::gpu::SurfaceConfiguration,
    session: ViewportSession,
}

struct Runner<F> {
    config: AppConfig,
    setup: Option<Box<dyn FnOnce(&mut ViewportSession, &crate::gpu::Device)>>,
    callback: F,
    state: Option<RunState>,
    orbit: OrbitCameraController,
    last_frame: Instant,
    start: Instant,
}

impl<F: FnMut(&mut FrameCtx)> ApplicationHandler for Runner<F> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title(self.config.title.clone())
                        .with_inner_size(::winit::dpi::LogicalSize::new(
                            self.config.width,
                            self.config.height,
                        )),
                )
                .expect("window"),
        );

        let instance = crate::gpu::Instance::new(&crate::gpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            },
        ))
        .expect("adapter");
        let required_features = crate::ViewportRenderer::recommended_device_features(&adapter);
        let (device, queue) =
            pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
                required_features,
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
        let surface_config = crate::gpu::SurfaceConfiguration {
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: crate::gpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let mut session = ViewportSession::new(&device, format);
        if let Some(setup) = self.setup.take() {
            setup(&mut session, &device);
        }
        let scale = window.scale_factor() as f32;
        session.set_pixels_per_point(scale);
        session.begin_frame(ViewportContext {
            hovered: true,
            focused: true,
            viewport_size: [
                surface_config.width as f32 / scale,
                surface_config.height as f32 / scale,
            ],
        });

        self.last_frame = Instant::now();
        self.start = Instant::now();
        self.state = Some(RunState {
            window,
            surface,
            device,
            queue,
            surface_config,
            session,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    state.surface_config.width = size.width;
                    state.surface_config.height = size.height;
                    state
                        .surface
                        .configure(&state.device, &state.surface_config);
                    state.window.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                // Logical viewport size; pixels_per_point carries the DPI scale so
                // the surface (physical) is sized correctly and overlays stay crisp.
                let scale = state.window.scale_factor() as f32;
                let w = state.surface_config.width as f32 / scale;
                let h = state.surface_config.height as f32 / scale;

                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;
                let time = (now - self.start).as_secs_f32();

                {
                    let mut ctx = FrameCtx {
                        session: &mut state.session,
                        dt,
                        time,
                    };
                    (self.callback)(&mut ctx);
                }

                // Sync to the live surface size before assembly so the renderer's
                // internal depth/HDR targets match the swapchain texture even when
                // a resize landed since the last begin_frame.
                state.session.set_viewport_size([w, h]);
                state.session.set_pixels_per_point(scale);
                state.session.step_runtime(dt);
                state.session.update_orbit(&mut self.orbit);

                let frame = match state.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(crate::gpu::SurfaceError::Lost | crate::gpu::SurfaceError::Outdated) => {
                        state
                            .surface
                            .configure(&state.device, &state.surface_config);
                        return;
                    }
                    Err(e) => {
                        tracing::error!("surface error: {e:?}");
                        return;
                    }
                };
                let view = frame
                    .texture
                    .create_view(&crate::gpu::TextureViewDescriptor::default());
                let cmd = state.session.render(&state.device, &state.queue, &view);
                state.queue.submit(std::iter::once(cmd));
                frame.present();

                state.session.begin_frame(ViewportContext {
                    hovered: true,
                    focused: true,
                    viewport_size: [w, h],
                });
                state.window.request_redraw();
            }

            other => {
                let scale = state.window.scale_factor() as f32;
                if let Some(ev) = from_winit(&other, scale) {
                    state.session.handle_event(ev);
                    state.window.request_redraw();
                }
            }
        }
    }
}
