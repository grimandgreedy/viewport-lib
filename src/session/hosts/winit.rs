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

use crate::interaction::input::ViewportContext;
use crate::interaction::input::adapters::from_winit;
use crate::session::ViewportSession;
use crate::{FrameData, OrbitCameraController, OverlayFrame};

/// When the runner asks for the next frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RedrawMode {
    /// Redraw every frame: a continuous animation loop, paced by the present
    /// mode. Idle scenes still burn a frame's worth of work each vsync. This is
    /// the default, and what an animated scene wants.
    #[default]
    Continuous,
    /// Redraw only when there is something new: an input event, a resize, or an
    /// explicit [`FrameCtx::request_redraw`] from the callback. The loop idles
    /// between events. A callback that animates must call `request_redraw` each
    /// frame to keep the loop going.
    OnDemand,
}

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
    /// Surface present mode. Default: [`PresentMode::AutoVsync`].
    ///
    /// [`PresentMode::AutoVsync`]: crate::gpu::PresentMode::AutoVsync
    pub present_mode: crate::gpu::PresentMode,
    /// When the runner schedules the next frame. Default:
    /// [`RedrawMode::Continuous`].
    pub redraw_mode: RedrawMode,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            title: "viewport-lib".to_string(),
            width: 1280,
            height: 720,
            present_mode: crate::gpu::PresentMode::AutoVsync,
            redraw_mode: RedrawMode::Continuous,
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

    /// Set the surface present mode directly.
    pub fn with_present_mode(mut self, present_mode: crate::gpu::PresentMode) -> Self {
        self.present_mode = present_mode;
        self
    }

    /// Vsync on ([`PresentMode::AutoVsync`]) or off
    /// ([`PresentMode::AutoNoVsync`]). Off renders as fast as the GPU allows and
    /// may tear; useful for uncapped-frame-rate measurement.
    ///
    /// [`PresentMode::AutoVsync`]: crate::gpu::PresentMode::AutoVsync
    /// [`PresentMode::AutoNoVsync`]: crate::gpu::PresentMode::AutoNoVsync
    pub fn with_vsync(mut self, vsync: bool) -> Self {
        self.present_mode = if vsync {
            crate::gpu::PresentMode::AutoVsync
        } else {
            crate::gpu::PresentMode::AutoNoVsync
        };
        self
    }

    /// Set when the runner schedules the next frame. Default:
    /// [`RedrawMode::Continuous`].
    pub fn with_redraw_mode(mut self, redraw_mode: RedrawMode) -> Self {
        self.redraw_mode = redraw_mode;
        self
    }
}

/// What the per-frame callback receives: the session (via deref) plus timing.
///
/// The callback runs before the frame is assembled, and assembly clears the
/// overlay frame, so overlays pushed directly through
/// [`frame_data_mut`](ViewportSession::frame_data_mut) would be wiped before
/// render. Use [`overlays_mut`](Self::overlays_mut) (or [`inject`](Self::inject)
/// for other per-frame, non-mesh items): the runner applies them after assembly
/// and before render, at the same point the session's `update_orbit_with` seam
/// runs.
pub struct FrameCtx<'a> {
    session: &'a mut ViewportSession,
    /// Seconds since the previous frame.
    pub dt: f32,
    /// Seconds since the app started.
    pub time: f32,
    device: &'a crate::gpu::Device,
    queue: &'a crate::gpu::Queue,
    overlays: OverlayFrame,
    injects: Vec<Box<dyn FnOnce(&mut FrameData)>>,
    request_exit: bool,
    request_redraw: bool,
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

    /// The wgpu device the runner created.
    ///
    /// For per-frame work that needs the device: GPU picking
    /// ([`pick_gpu`](ViewportSession::pick_gpu),
    /// [`pick_rect_gpu`](ViewportSession::pick_rect_gpu),
    /// [`pick_begin`](ViewportSession::pick_begin)) or a mesh upload through
    /// [`resources_mut`](ViewportSession::resources_mut). Those methods also take
    /// `&mut self` on the session, which the callback reaches by deref, so borrow
    /// the handle first (a wgpu `Device` is a cheap `Arc`-backed clone):
    ///
    /// ```rust,ignore
    /// let device = ctx.device().clone();
    /// let queue = ctx.queue().clone();
    /// if let Some(hit) = ctx.pick_gpu(&device, &queue, cursor, PickMask::ALL) {
    ///     // ...
    /// }
    /// ```
    pub fn device(&self) -> &crate::gpu::Device {
        self.device
    }

    /// The wgpu queue the runner created. See [`device`](Self::device).
    pub fn queue(&self) -> &crate::gpu::Queue {
        self.queue
    }

    /// Ask the runner to close the window and end the event loop after this
    /// frame. The current frame still renders; the loop exits before the next.
    pub fn request_exit(&mut self) {
        self.request_exit = true;
    }

    /// Ask the runner to schedule another frame after this one.
    ///
    /// Only meaningful under [`RedrawMode::OnDemand`], where the loop otherwise
    /// idles between events: a callback that animates calls this each frame to
    /// keep drawing. Under [`RedrawMode::Continuous`] the next frame is already
    /// scheduled, so this is a no-op.
    pub fn request_redraw(&mut self) {
        self.request_redraw = true;
    }

    /// Overlays to draw this frame: shapes, labels, polylines, and images.
    ///
    /// Push into the returned frame instead of `frame_data_mut().overlays`. The
    /// runner installs these after the frame is assembled and before render, so
    /// they survive the overlay reset that assembly performs. The buffer is
    /// per-frame: it starts empty each callback, so re-push anything that should
    /// persist.
    pub fn overlays_mut(&mut self) -> &mut OverlayFrame {
        &mut self.overlays
    }

    /// Queue a closure to run against the assembled [`FrameData`] before render.
    ///
    /// This is the general form of [`overlays_mut`](Self::overlays_mut), for
    /// per-frame non-mesh items that assembly rebuilds (point clouds, glyphs,
    /// volumes pushed by hand into `frame.scene`). Closures run in the order
    /// queued, after the buffered overlays are installed.
    pub fn inject(&mut self, f: impl FnOnce(&mut FrameData) + 'static) {
        self.injects.push(Box::new(f));
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
    /// Window has keyboard focus. Tracked from `WindowEvent::Focused`.
    focused: bool,
    /// Cursor is over the window. Tracked from `CursorEntered`/`CursorLeft`; the
    /// input resolver gates hover-based gestures on this.
    hovered: bool,
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
            present_mode: self.config.present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let mut session = ViewportSession::new(&device, format);
        if let Some(setup) = self.setup.take() {
            setup(&mut session, &device);
        }
        // Focus/hover start conservative: the window is focused on creation, but
        // the cursor is only "over" it once a CursorEntered arrives.
        let focused = true;
        let hovered = false;
        let scale = window.scale_factor() as f32;
        session.set_pixels_per_point(scale);
        session.begin_frame(ViewportContext {
            hovered,
            focused,
            viewport_size: [
                surface_config.width as f32 / scale,
                surface_config.height as f32 / scale,
            ],
        });

        // Kick the first frame; under OnDemand nothing else would.
        window.request_redraw();

        self.last_frame = Instant::now();
        self.start = Instant::now();
        self.state = Some(RunState {
            window,
            surface,
            device,
            queue,
            surface_config,
            session,
            focused,
            hovered,
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

                // Resolve accumulated input before the callback so a callback
                // reading action_frame() (for click-to-pick and the like) sees
                // this frame's input, not the previous frame's. update_orbit
                // resolves again below, but resolve() is side-effect-free and the
                // camera is only applied once, so the repeat is free.
                state.session.resolve();

                let mut ctx = FrameCtx {
                    session: &mut state.session,
                    dt,
                    time,
                    device: &state.device,
                    queue: &state.queue,
                    overlays: OverlayFrame::default(),
                    injects: Vec::new(),
                    request_exit: false,
                    request_redraw: false,
                };
                (self.callback)(&mut ctx);
                let FrameCtx {
                    overlays,
                    injects,
                    request_exit,
                    request_redraw,
                    ..
                } = ctx;

                // Sync to the live surface size before assembly so the renderer's
                // internal depth/HDR targets match the swapchain texture even when
                // a resize landed since the last begin_frame.
                state.session.set_viewport_size([w, h]);
                state.session.set_pixels_per_point(scale);
                state.session.step_runtime(dt);
                // Assembly clears frame.overlays, so install the callback's
                // overlays and injects here, against the assembled frame.
                state
                    .session
                    .update_orbit_with(&mut self.orbit, move |frame| {
                        frame.overlays = overlays;
                        for inject in injects {
                            inject(frame);
                        }
                    });

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

                if request_exit {
                    event_loop.exit();
                    return;
                }

                state.session.begin_frame(ViewportContext {
                    hovered: state.hovered,
                    focused: state.focused,
                    viewport_size: [w, h],
                });
                // Continuous keeps the loop spinning; OnDemand only redraws when
                // the callback asked to (an animating callback calls
                // request_redraw each frame).
                if self.config.redraw_mode == RedrawMode::Continuous || request_redraw {
                    state.window.request_redraw();
                }
            }

            WindowEvent::Focused(focused) => {
                state.focused = focused;
                state.window.request_redraw();
            }

            WindowEvent::CursorEntered { .. } => {
                state.hovered = true;
                state.window.request_redraw();
            }

            WindowEvent::CursorLeft { .. } => {
                state.hovered = false;
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
