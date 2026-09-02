//! `ViewportAppV2` is an experimental multi-window winit runner over
//! [`ViewportInstance`].
//!
//! Where [`ViewportApp`](crate::ViewportApp) owns exactly one window, this owns a
//! set of OS windows keyed by a library [`WindowId`], each driving its own
//! [`ViewportInstance`] against its own surface, over a single shared wgpu device.
//! Events are routed to the right window; each window redraws, navigates, and times
//! its frames independently.
//!
//! This type is experimental and its API may change. The stable single-window
//! [`ViewportApp`](crate::ViewportApp) is unaffected. See the runners module docs
//! for the composable [`ViewportInstance`] you drive from your own loop.
//!
//! # The winit boundary
//!
//! A consumer never touches winit through this runner: windows are named by the
//! library-owned [`WindowId`], configured with [`WindowConfig`], and the per-frame
//! callback receives viewport-lib types. For direct winit access the runner does not
//! model, reach the raw window (a later addition) or use the re-exported
//! `viewport_lib::winit` so you never add a second, mismatched winit dependency.

use std::collections::HashMap;
use std::sync::Arc;
use web_time::Instant;

use ::winit::application::ApplicationHandler;
use ::winit::event::WindowEvent;
use ::winit::event_loop::{ActiveEventLoop, EventLoop};
use ::winit::window::{Window, WindowAttributes, WindowId as WinitWindowId};

use crate::interaction::input::adapters::from_winit;
use crate::interaction::input::{ViewportContext, ViewportEvent};
use crate::runners::ViewportInstance;
use crate::runners::viewport_app::RedrawMode;
use crate::{ExposureMode, FrameData, OrbitCameraController, OverlayFrame};

/// See [`ViewportApp`](crate::ViewportApp)'s equivalent: fill the auto-exposure `dt`
/// from the frame time so smooth adaptation works under the continuous-frame runner.
fn auto_fill_exposure_dt(frame: &mut FrameData, dt: f32) {
    if let ExposureMode::Automatic(ref mut auto) = frame.effects.display.exposure.mode {
        auto.dt = dt;
    }
}

/// A window handle owned by the runner, not winit's.
///
/// Names a window opened through [`ViewportAppV2`] so consumer code and event
/// routing identify windows without depending on winit. Stable and comparable for
/// the life of the window; a value from a closed window never aliases a later one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WindowId(u64);

impl WindowId {
    /// The raw counter value. Useful as a map key or for logging; it carries no
    /// meaning beyond identity.
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// Per-window configuration for a window opened through [`ViewportAppV2`].
///
/// Non-exhaustive: build with [`WindowConfig::default`] and the `with_*` methods so
/// new options can be added without breaking construction. Fields are viewport-lib
/// or wgpu types, never winit types, so a consumer needs no winit dependency.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct WindowConfig {
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
    /// When this window schedules its next frame. Default:
    /// [`RedrawMode::Continuous`]. Set per window, so one animating window can run
    /// continuously beside a static window that only redraws on input.
    pub redraw_mode: RedrawMode,
}

impl Default for WindowConfig {
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

impl WindowConfig {
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

    /// Vsync on ([`PresentMode::AutoVsync`]) or off ([`PresentMode::AutoNoVsync`]).
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

    /// Set when this window schedules its next frame. Default:
    /// [`RedrawMode::Continuous`].
    pub fn with_redraw_mode(mut self, redraw_mode: RedrawMode) -> Self {
        self.redraw_mode = redraw_mode;
        self
    }
}

/// App-level configuration for a [`ViewportAppV2`].
///
/// Non-exhaustive: build with [`AppConfigV2::default`] and the `with_*` methods.
/// Per-window options live on [`WindowConfig`]; this holds settings that span the
/// whole app.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct AppConfigV2 {
    /// End the event loop when the last window closes. Default: `true`. Set `false`
    /// to keep the loop alive with no windows (for example to reopen one later).
    pub exit_on_last_window_close: bool,
}

impl Default for AppConfigV2 {
    fn default() -> Self {
        Self {
            exit_on_last_window_close: true,
        }
    }
}

impl AppConfigV2 {
    /// Set whether the loop ends when the last window closes.
    pub fn with_exit_on_last_window_close(mut self, exit: bool) -> Self {
        self.exit_on_last_window_close = exit;
        self
    }
}

/// The per-window setup callback: build this window's scene with the shared device
/// in hand. Mirrors [`ViewportApp::setup`](crate::ViewportApp::setup), one per
/// window.
type WindowFactory = Box<dyn FnOnce(&mut ViewportInstance, &crate::gpu::Device)>;

/// The per-window per-frame callback.
type WindowCallback = Box<dyn FnMut(&mut FrameCtxV2)>;

/// A window declared before [`run`](ViewportAppV2::run), realised in `resumed`.
struct WindowBuilder {
    config: WindowConfig,
    factory: WindowFactory,
    callback: WindowCallback,
}

/// What a window's per-frame callback receives: that window's [`ViewportInstance`]
/// (via deref) plus timing and the window's identity.
///
/// The callback is per window: it drives one window's instance, and is called once
/// per frame for that window. Frame-global work that must run once regardless of
/// window count does not belong here.
pub struct FrameCtxV2<'a> {
    session: &'a mut ViewportInstance,
    window_id: WindowId,
    /// Seconds since this window's previous frame.
    dt: f32,
    /// Seconds since the app started.
    time: f32,
    device: &'a crate::gpu::Device,
    queue: &'a crate::gpu::Queue,
    viewport_size: [f32; 2],
    overlays: OverlayFrame,
    injects: Vec<Box<dyn FnOnce(&mut FrameData)>>,
    events: Vec<ViewportEvent>,
    request_exit: bool,
    request_redraw: bool,
}

impl FrameCtxV2<'_> {
    /// Which window this callback invocation is for.
    pub fn window_id(&self) -> WindowId {
        self.window_id
    }

    /// Seconds since this window's previous frame.
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// Seconds since the app started.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// The logical viewport size in points for this window this frame.
    pub fn viewport_size(&self) -> [f32; 2] {
        self.viewport_size
    }

    /// The shared wgpu device.
    pub fn device(&self) -> &crate::gpu::Device {
        self.device
    }

    /// The shared wgpu queue.
    pub fn queue(&self) -> &crate::gpu::Queue {
        self.queue
    }

    /// Ask the runner to end the event loop after this frame.
    pub fn request_exit(&mut self) {
        self.request_exit = true;
    }

    /// Ask the runner to schedule another frame for this window after this one.
    ///
    /// Only meaningful under [`RedrawMode::OnDemand`]; under
    /// [`RedrawMode::Continuous`] the next frame is already scheduled.
    pub fn request_redraw(&mut self) {
        self.request_redraw = true;
    }

    /// Overlays to draw this frame for this window: shapes, labels, polylines, and
    /// images. Installed after assembly and before render, so they survive the
    /// overlay reset assembly performs. Per-frame: starts empty each callback.
    pub fn overlays_mut(&mut self) -> &mut OverlayFrame {
        &mut self.overlays
    }

    /// Queue a closure to run against the assembled [`FrameData`] before render, for
    /// per-frame non-mesh items assembly rebuilds. Runs after buffered overlays.
    pub fn inject(&mut self, f: impl FnOnce(&mut FrameData) + 'static) {
        self.injects.push(Box::new(f));
    }

    /// The input events that arrived for this window since its last frame, in order.
    /// The runner already fed these to the window's instance and resolved them into
    /// [`action_frame`](ViewportInstance::action_frame); this is the raw stream on
    /// top of that.
    pub fn events(&self) -> &[ViewportEvent] {
        &self.events
    }
}

impl std::ops::Deref for FrameCtxV2<'_> {
    type Target = ViewportInstance;

    fn deref(&self) -> &ViewportInstance {
        self.session
    }
}

impl std::ops::DerefMut for FrameCtxV2<'_> {
    fn deref_mut(&mut self) -> &mut ViewportInstance {
        self.session
    }
}

/// An experimental multi-window winit runner driving one [`ViewportInstance`] per
/// OS window.
///
/// Declare windows with [`window`](Self::window), then [`run`](Self::run):
///
/// ```rust,ignore
/// use viewport_lib::{AppConfigV2, Material, ViewportAppV2, WindowConfig, primitives};
///
/// ViewportAppV2::new(AppConfigV2::default())
///     .window(
///         WindowConfig::default().with_title("left"),
///         |vp, device| {
///             let mesh = vp.resources_mut()
///                 .upload_mesh_data(device, &primitives::cube(1.0)).unwrap();
///             vp.scene_mut().add(Some(mesh), glam::Mat4::IDENTITY,
///                 Material::from_colour([0.85, 0.25, 0.2]));
///             vp.camera_mut().distance = 6.0;
///         },
///         |_ctx| {},
///     )
///     .window(
///         WindowConfig::default().with_title("right"),
///         |vp, device| { /* a different scene */ },
///         |_ctx| {},
///     )
///     .run();
/// ```
pub struct ViewportAppV2 {
    config: AppConfigV2,
    windows: Vec<WindowBuilder>,
}

impl ViewportAppV2 {
    /// Create a runner with the given app configuration.
    pub fn new(config: AppConfigV2) -> Self {
        Self {
            config,
            windows: Vec::new(),
        }
    }

    /// Declare a window: its configuration, a one-time setup callback run with the
    /// shared device to build its scene, and a per-frame callback that drives it.
    ///
    /// Call once per window before [`run`](Self::run). The setup callback is the
    /// per-window equivalent of [`ViewportApp::setup`](crate::ViewportApp::setup);
    /// the frame callback is per window and runs once per frame for that window.
    pub fn window(
        mut self,
        config: WindowConfig,
        factory: impl FnOnce(&mut ViewportInstance, &crate::gpu::Device) + 'static,
        callback: impl FnMut(&mut FrameCtxV2) + 'static,
    ) -> Self {
        self.windows.push(WindowBuilder {
            config,
            factory: Box::new(factory),
            callback: Box::new(callback),
        });
        self
    }

    /// Run the event loop, driving every declared window. Blocks until the loop
    /// ends (the last window closes, unless configured otherwise, or a callback
    /// calls [`request_exit`](FrameCtxV2::request_exit)).
    pub fn run(self) {
        let event_loop = EventLoop::new().expect("event loop");
        let mut handler = AppHandlerV2 {
            config: self.config,
            pending: self.windows,
            gpu: None,
            windows: HashMap::new(),
            winit_ids: HashMap::new(),
            next_id: 0,
            start: Instant::now(),
        };
        event_loop.run_app(&mut handler).expect("run app");
    }
}

/// The shared wgpu state, created once on the first window and reused for the rest.
struct Gpu {
    instance: crate::gpu::Instance,
    adapter: crate::gpu::Adapter,
    device: crate::gpu::Device,
    queue: crate::gpu::Queue,
}

/// Everything one window owns. The multi-window generalisation of the single-window
/// runner's `RunState`: input, orbit, timing, and focus/hover are all per window.
struct WindowState {
    window: Arc<Window>,
    surface: crate::gpu::Surface<'static>,
    surface_config: crate::gpu::SurfaceConfiguration,
    session: ViewportInstance,
    redraw_mode: RedrawMode,
    orbit: OrbitCameraController,
    callback: WindowCallback,
    /// Events translated for this window since its last frame.
    events: Vec<ViewportEvent>,
    focused: bool,
    hovered: bool,
    last_frame: Instant,
}

struct AppHandlerV2 {
    config: AppConfigV2,
    /// Windows declared before `run`, realised on first `resumed`.
    pending: Vec<WindowBuilder>,
    gpu: Option<Gpu>,
    windows: HashMap<WindowId, WindowState>,
    /// Translates an incoming winit window id to the library id it was assigned.
    winit_ids: HashMap<WinitWindowId, WindowId>,
    next_id: u64,
    start: Instant,
}

impl AppHandlerV2 {
    /// Create one window from its builder, configure its surface against the shared
    /// device, build its instance, and insert it into the window set.
    fn create_window(&mut self, event_loop: &ActiveEventLoop, builder: WindowBuilder) {
        let WindowBuilder {
            config,
            factory,
            callback,
        } = builder;

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title(config.title.clone())
                        .with_inner_size(::winit::dpi::LogicalSize::new(config.width, config.height)),
                )
                .expect("window"),
        );

        // The surface must be created before the adapter request so it can be the
        // compatibility target for the first window. Subsequent windows reuse the
        // instance already in `gpu`.
        let surface = if let Some(gpu) = self.gpu.as_ref() {
            gpu.instance.create_surface(window.clone()).expect("surface")
        } else {
            // First window: create a temporary instance-less path is not possible,
            // so build the instance here via ensure_gpu after making the surface.
            let instance = crate::gpu::Instance::new(&crate::gpu::InstanceDescriptor::default());
            let surface = instance.create_surface(window.clone()).expect("surface");
            // Seed the shared gpu from this instance/surface.
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
                    required_limits: crate::ViewportRenderer::recommended_device_limits(&adapter),
                    ..Default::default()
                }))
                .expect("device");
            self.gpu = Some(Gpu {
                instance,
                adapter,
                device,
                queue,
            });
            surface
        };

        let gpu = self.gpu.as_ref().unwrap();
        let size = window.inner_size();
        let caps = surface.get_capabilities(&gpu.adapter);
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
            present_mode: config.present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&gpu.device, &surface_config);

        let mut session = ViewportInstance::new(&gpu.device, format);
        factory(&mut session, &gpu.device);

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

        let id = WindowId(self.next_id);
        self.next_id += 1;
        self.winit_ids.insert(window.id(), id);
        window.request_redraw();

        self.windows.insert(
            id,
            WindowState {
                window,
                surface,
                surface_config,
                session,
                redraw_mode: config.redraw_mode,
                orbit: OrbitCameraController::viewport_all(),
                callback,
                events: Vec::new(),
                focused,
                hovered,
                last_frame: Instant::now(),
            },
        );
    }

    /// Draw one window: sync size/DPI, run its callback, assemble, and present.
    fn redraw(&mut self, event_loop: &ActiveEventLoop, id: WindowId) {
        let Some(state) = self.windows.get_mut(&id) else {
            return;
        };
        let Some(gpu) = self.gpu.as_ref() else {
            return;
        };

        let scale = state.window.scale_factor() as f32;
        let w = state.surface_config.width as f32 / scale;
        let h = state.surface_config.height as f32 / scale;

        let now = Instant::now();
        let dt = (now - state.last_frame).as_secs_f32();
        state.last_frame = now;
        let time = (now - self.start).as_secs_f32();

        state.session.set_viewport_size([w, h]);
        state.session.set_pixels_per_point(scale);
        state.session.resolve();

        let mut ctx = FrameCtxV2 {
            session: &mut state.session,
            window_id: id,
            dt,
            time,
            device: &gpu.device,
            queue: &gpu.queue,
            viewport_size: [w, h],
            overlays: OverlayFrame::default(),
            injects: Vec::new(),
            events: std::mem::take(&mut state.events),
            request_exit: false,
            request_redraw: false,
        };
        (state.callback)(&mut ctx);
        let FrameCtxV2 {
            overlays,
            injects,
            request_exit,
            request_redraw,
            ..
        } = ctx;

        state.session.step_runtime(dt);
        state.session.update_orbit_with(&mut state.orbit, move |frame| {
            frame.overlays = overlays;
            for inject in injects {
                inject(frame);
            }
            auto_fill_exposure_dt(frame, dt);
        });

        let frame = match state.surface.get_current_texture() {
            Ok(f) => f,
            Err(crate::gpu::SurfaceError::Lost | crate::gpu::SurfaceError::Outdated) => {
                state.surface.configure(&gpu.device, &state.surface_config);
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
        let cmd = state.session.render(&gpu.device, &gpu.queue, &view);
        gpu.queue.submit(std::iter::once(cmd));
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
        if state.redraw_mode == RedrawMode::Continuous || request_redraw {
            state.window.request_redraw();
        }
    }

    /// Close one window: drop its state and, if the set is now empty and configured
    /// to, end the loop.
    fn close_window(&mut self, event_loop: &ActiveEventLoop, id: WindowId) {
        if let Some(state) = self.windows.remove(&id) {
            self.winit_ids.remove(&state.window.id());
        }
        if self.windows.is_empty() && self.config.exit_on_last_window_close {
            event_loop.exit();
        }
    }
}

impl ApplicationHandler for AppHandlerV2 {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.windows.is_empty() {
            return;
        }
        let builders = std::mem::take(&mut self.pending);
        for builder in builders {
            self.create_window(event_loop, builder);
        }
        self.start = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        winit_id: WinitWindowId,
        event: WindowEvent,
    ) {
        let Some(&id) = self.winit_ids.get(&winit_id) else {
            return;
        };

        // Typed characters: winit resolves layout/shift/dead keys into `text`. Feed
        // the window's session directly (its numeric-input buffer keeps only digits,
        // `.`, `-`) and buffer for the callback.
        if let WindowEvent::KeyboardInput {
            event: key_event, ..
        } = &event
        {
            if key_event.state == ::winit::event::ElementState::Pressed {
                if let Some(text) = &key_event.text {
                    let chars: Vec<char> = text.chars().filter(|c| !c.is_control()).collect();
                    if !chars.is_empty() {
                        if let Some(state) = self.windows.get_mut(&id) {
                            for c in chars {
                                let ev = ViewportEvent::Character(c);
                                state.session.handle_event(ev.clone());
                                state.events.push(ev);
                            }
                            state.window.request_redraw();
                        }
                    }
                }
            }
        }

        match event {
            WindowEvent::CloseRequested => self.close_window(event_loop, id),

            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    if let Some(gpu) = self.gpu.as_ref() {
                        if let Some(state) = self.windows.get_mut(&id) {
                            state.surface_config.width = size.width;
                            state.surface_config.height = size.height;
                            state.surface.configure(&gpu.device, &state.surface_config);
                            state.window.request_redraw();
                        }
                    }
                }
            }

            WindowEvent::RedrawRequested => self.redraw(event_loop, id),

            WindowEvent::Focused(focused) => {
                if let Some(state) = self.windows.get_mut(&id) {
                    state.focused = focused;
                    state.window.request_redraw();
                }
            }

            WindowEvent::CursorEntered { .. } => {
                if let Some(state) = self.windows.get_mut(&id) {
                    state.hovered = true;
                    state.window.request_redraw();
                }
            }

            WindowEvent::CursorLeft { .. } => {
                if let Some(state) = self.windows.get_mut(&id) {
                    state.hovered = false;
                    state.window.request_redraw();
                }
            }

            other => {
                if let Some(state) = self.windows.get_mut(&id) {
                    let scale = state.window.scale_factor() as f32;
                    if let Some(ev) = from_winit(&other, scale) {
                        state.session.handle_event(ev.clone());
                        state.events.push(ev);
                        state.window.request_redraw();
                    }
                }
            }
        }
    }
}
