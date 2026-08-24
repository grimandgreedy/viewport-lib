//! `ViewportApp` is a fullscreen winit runner over a [`ViewportInstance`].
//!
//! This owns the window, the wgpu bring-up, and the event loop, and drives a
//! [`ViewportInstance`] each frame. It is the standalone-application path: the
//! wiring that every hand-written winit example repeats lives here once.

use std::sync::Arc;
use std::time::Instant;

use ::winit::application::ApplicationHandler;
use ::winit::event::WindowEvent;
use ::winit::event_loop::{ActiveEventLoop, EventLoop};
use ::winit::window::{Window, WindowAttributes, WindowId};

use crate::interaction::input::adapters::from_winit;
use crate::interaction::input::{ViewportContext, ViewportEvent};
use crate::runners::ViewportInstance;
use crate::{ExposureMode, FrameData, OrbitCameraController, OverlayFrame};

/// Auto-fill the auto-exposure `dt` from the frame time so smooth adaptation
/// (`dt > 0`) works out of the box under the continuous-frame runner. Consumers
/// that render only when dirty pass `dt = 0` themselves (the snap default) and
/// never reach this path.
fn auto_fill_exposure_dt(frame: &mut FrameData, dt: f32) {
    if let ExposureMode::Automatic(ref mut auto) = frame.effects.display.exposure.mode {
        auto.dt = dt;
    }
}

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

/// What the per-frame callback receives: the viewport instance (via deref) plus timing.
///
/// The callback runs before the frame is assembled, and assembly clears the
/// overlay frame, so overlays pushed directly through
/// [`frame_data_mut`](ViewportInstance::frame_data_mut) would be wiped before
/// render. Use [`overlays_mut`](Self::overlays_mut) (or [`inject`](Self::inject)
/// for other per-frame, non-mesh items): the runner applies them after assembly
/// and before render, at the same point the instance's `update_orbit_with` seam
/// runs.
pub struct FrameCtx<'a> {
    session: &'a mut ViewportInstance,
    /// Seconds since the previous frame.
    pub dt: f32,
    /// Seconds since the app started.
    pub time: f32,
    device: &'a crate::gpu::Device,
    queue: &'a crate::gpu::Queue,
    viewport_size: [f32; 2],
    overlays: OverlayFrame,
    injects: Vec<Box<dyn FnOnce(&mut FrameData)>>,
    events: Vec<ViewportEvent>,
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

    /// The logical viewport size in points for this frame.
    ///
    /// This is the current frame's size, freshly computed from the surface size
    /// and DPI scale, so it is correct even on a frame where a resize just landed.
    /// Overlay UI drawn from the callback uses it to place itself relative to the
    /// viewport edges: clamping a panel on screen, flipping a submenu near an edge,
    /// or anchoring to a corner or the centre.
    pub fn viewport_size(&self) -> [f32; 2] {
        self.viewport_size
    }

    /// The wgpu device the runner created.
    ///
    /// For per-frame work that needs the device: GPU picking
    /// ([`pick_gpu`](ViewportInstance::pick_gpu),
    /// [`pick_rect_gpu`](ViewportInstance::pick_rect_gpu),
    /// [`pick_begin`](ViewportInstance::pick_begin)) or a mesh upload through
    /// [`resources_mut`](ViewportInstance::resources_mut). Those methods also take
    /// `&mut self` on the instance, which the callback reaches by deref, so borrow
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

    /// The input events that arrived since the last frame, in order.
    ///
    /// The runner already feeds these to the instance (so orbit navigation and
    /// bound actions work), and resolves them into
    /// [`action_frame`](ViewportInstance::action_frame), which stays the
    /// convenient path for the cursor, left click, and camera. This is the raw
    /// stream on top of that, for input the resolved frame does not surface:
    /// the secondary and middle mouse buttons, and individual keys such as the
    /// arrows, Enter, and Escape. A right-click context menu or keyboard menu
    /// navigation reads them here.
    ///
    /// Every translated event is included, whether or not the camera also acted
    /// on it, so match on the ones you want:
    ///
    /// ```rust,ignore
    /// use viewport_lib::{ButtonState, KeyCode, MouseButton, ViewportEvent};
    ///
    /// for ev in ctx.events() {
    ///     match ev {
    ///         ViewportEvent::MouseButton { button: MouseButton::Right, state: ButtonState::Pressed } => {
    ///             // open a context menu at ctx.action_frame().pointer.cursor
    ///         }
    ///         ViewportEvent::Key { key: KeyCode::Escape, state: ButtonState::Pressed, .. } => {
    ///             // close the menu
    ///         }
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn events(&self) -> &[ViewportEvent] {
        &self.events
    }
}

impl std::ops::Deref for FrameCtx<'_> {
    type Target = ViewportInstance;

    fn deref(&self) -> &ViewportInstance {
        self.session
    }
}

impl std::ops::DerefMut for FrameCtx<'_> {
    fn deref_mut(&mut self) -> &mut ViewportInstance {
        self.session
    }
}

/// What the input handler installed with [`ViewportApp::with_input`] receives:
/// the raw events buffered since the last frame plus the instance (via deref).
///
/// Forward the events the viewport should act on with [`forward`](Self::forward)
/// and drop the ones your UI consumed. Reach the camera, scene, and input state
/// through the deref to [`ViewportInstance`] (for example
/// [`camera_mut`](ViewportInstance::camera_mut) to drive your own controller).
pub struct InputCtx<'a> {
    session: &'a mut ViewportInstance,
    events: &'a [ViewportEvent],
}

impl InputCtx<'_> {
    /// The input events that arrived since the last frame, in order. These have
    /// not been sent to the viewport: forward the ones it should process with
    /// [`forward`](Self::forward).
    pub fn events(&self) -> &[ViewportEvent] {
        self.events
    }

    /// Send one event to the viewport's input so it drives picking, selection,
    /// and manipulation this frame. Skip an event to keep it from the viewport.
    pub fn forward(&mut self, event: ViewportEvent) {
        self.session.handle_event(event);
    }
}

impl std::ops::Deref for InputCtx<'_> {
    type Target = ViewportInstance;

    fn deref(&self) -> &ViewportInstance {
        self.session
    }
}

impl std::ops::DerefMut for InputCtx<'_> {
    fn deref_mut(&mut self) -> &mut ViewportInstance {
        self.session
    }
}

/// A fullscreen winit application driving a [`ViewportInstance`].
///
/// ```rust,ignore
/// use viewport_lib::{AppConfig, Material, ViewportApp, primitives};
///
/// let mut cube = None;
/// ViewportApp::new(AppConfig::default().with_title("demo"))
///     .setup(|viewport, device| {
///         let mesh = viewport.resources_mut()
///             .upload_mesh_data(device, &primitives::cube(1.0)).unwrap();
///         cube = Some(viewport.scene_mut().add(
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
    setup: Option<Box<dyn FnOnce(&mut ViewportInstance, &crate::gpu::Device)>>,
    input: Option<InputHandler>,
}

/// The per-frame input handler installed with [`ViewportApp::with_input`].
type InputHandler = Box<dyn FnMut(&mut InputCtx)>;

impl ViewportApp {
    /// Create a runner with the given window configuration.
    pub fn new(config: AppConfig) -> Self {
        Self {
            config,
            setup: None,
            input: None,
        }
    }

    /// Handle raw input yourself before it reaches the viewport.
    ///
    /// By default the runner feeds every event to the instance and drives a
    /// built-in orbit controller, so the callback sees input only after the
    /// viewport has already acted on it. Install a handler to take that over: the
    /// runner stops auto-feeding events and stops driving orbit, and instead calls
    /// this handler once per frame with the events buffered since the last frame
    /// plus mutable access to the instance (via deref). Forward the events the
    /// viewport should process with [`InputCtx::forward`], drop the ones your UI
    /// consumed, and drive your own camera controller against
    /// [`camera_mut`](ViewportInstance::camera_mut).
    ///
    /// This is the same control a hand-written [`ViewportInstance`] loop has, with
    /// the runner still owning the window, the wgpu bring-up, and the render loop.
    ///
    /// ```rust,ignore
    /// let mut orbit = OrbitCameraController::viewport_all();
    /// ViewportApp::new(config)
    ///     .with_input(move |ictx| {
    ///         for ev in ictx.events() {
    ///             if menu.borrow().is_open() && menu.borrow().hits(ev) {
    ///                 menu.borrow_mut().handle(ev); // consume: do not forward
    ///                 continue;
    ///             }
    ///             orbit.push_event(ev.clone());     // navigation
    ///             ictx.forward(ev.clone());         // viewport picking/selection
    ///         }
    ///         orbit.apply_to_camera(ictx.camera_mut());
    ///     })
    ///     .run(move |ctx| { /* overlays */ });
    /// ```
    pub fn with_input(mut self, handler: impl FnMut(&mut InputCtx) + 'static) -> Self {
        self.input = Some(Box::new(handler));
        self
    }

    /// Register a one-time setup callback, run after the instance is created with
    /// the wgpu device in hand: upload meshes, build the initial scene, attach a
    /// runtime.
    pub fn setup(
        mut self,
        setup: impl FnOnce(&mut ViewportInstance, &crate::gpu::Device) + 'static,
    ) -> Self {
        self.setup = Some(Box::new(setup));
        self
    }

    /// Run the event loop, calling `callback` once per frame before rendering.
    /// Blocks until the window closes.
    pub fn run(self, callback: impl FnMut(&mut FrameCtx) + 'static) {
        let event_loop = EventLoop::new().expect("event loop");
        let mut handler = AppHandler {
            config: self.config,
            setup: self.setup,
            callback,
            input: self.input,
            state: None,
            orbit: OrbitCameraController::viewport_all(),
            events: Vec::new(),
            last_frame: Instant::now(),
            start: Instant::now(),
        };
        event_loop.run_app(&mut handler).expect("run app");
    }
}

struct RunState {
    window: Arc<Window>,
    surface: crate::gpu::Surface<'static>,
    device: crate::gpu::Device,
    queue: crate::gpu::Queue,
    surface_config: crate::gpu::SurfaceConfiguration,
    session: ViewportInstance,
    /// Window has keyboard focus. Tracked from `WindowEvent::Focused`.
    focused: bool,
    /// Cursor is over the window. Tracked from `CursorEntered`/`CursorLeft`; the
    /// input resolver gates hover-based gestures on this.
    hovered: bool,
}

struct AppHandler<F> {
    config: AppConfig,
    setup: Option<Box<dyn FnOnce(&mut ViewportInstance, &crate::gpu::Device)>>,
    callback: F,
    /// Installed by [`ViewportApp::with_input`]. When set, the runner stops
    /// auto-feeding events and driving orbit; the handler owns the input step.
    input: Option<InputHandler>,
    state: Option<RunState>,
    orbit: OrbitCameraController,
    /// Events translated since the last frame, handed to the callback via
    /// [`FrameCtx::events`] and cleared each redraw.
    events: Vec<ViewportEvent>,
    last_frame: Instant,
    start: Instant,
}

impl<F: FnMut(&mut FrameCtx)> ApplicationHandler for AppHandler<F> {
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
                required_limits: crate::ViewportRenderer::recommended_device_limits(&adapter),
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

        let mut session = ViewportInstance::new(&device, format);
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

        // A typed character becomes a Character event, alongside the Key event the match
        // below still produces. winit has already resolved the keyboard layout, shift
        // state, and dead keys into `event.text`, so this is the committed text; control
        // keys (backspace, enter, arrows) carry no text and flow as keys. Route it exactly
        // like every other translated event: without an input handler, feed the session
        // directly (its numeric-input buffer keeps only digits, `.`, and `-`); with one,
        // buffer it so the handler decides what the viewport sees each frame.
        if let WindowEvent::KeyboardInput { event: key_event, .. } = &event {
            if key_event.state == ::winit::event::ElementState::Pressed {
                if let Some(text) = &key_event.text {
                    for c in text.chars().filter(|c| !c.is_control()) {
                        let ev = ViewportEvent::Character(c);
                        if self.input.is_none() {
                            state.session.handle_event(ev.clone());
                        }
                        self.events.push(ev);
                    }
                    state.window.request_redraw();
                }
            }
        }

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

                // Sync the instance to the live surface size and DPI before the
                // callback runs, so a callback that reads viewport_size or picks
                // (pick_gpu and friends project through this size) sees this
                // frame's value even when a resize just landed, not last frame's.
                // It also has to happen before assembly so the renderer's internal
                // depth/HDR targets match the swapchain texture.
                state.session.set_viewport_size([w, h]);
                state.session.set_pixels_per_point(scale);

                // An installed input handler owns this frame's input: it forwards
                // the events the viewport should see and drives its own camera.
                // Run it before resolve so the forwarded events land in this
                // frame's ActionFrame.
                if let Some(handler) = self.input.as_mut() {
                    let mut ictx = InputCtx {
                        session: &mut state.session,
                        events: &self.events,
                    };
                    handler(&mut ictx);
                }

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
                    viewport_size: [w, h],
                    overlays: OverlayFrame::default(),
                    injects: Vec::new(),
                    events: std::mem::take(&mut self.events),
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

                state.session.step_runtime(dt);
                // Assembly clears frame.overlays, so install the callback's
                // overlays and injects here, against the assembled frame. With an
                // input handler the app already drove the camera, so assemble
                // without touching it; otherwise drive the built-in orbit.
                if self.input.is_some() {
                    let vctx = ViewportContext {
                        hovered: state.hovered,
                        focused: state.focused,
                        viewport_size: [w, h],
                    };
                    state.session.frame_with(vctx, move |frame| {
                        frame.overlays = overlays;
                        for inject in injects {
                            inject(frame);
                        }
                        auto_fill_exposure_dt(frame, dt);
                    });
                } else {
                    state
                        .session
                        .update_orbit_with(&mut self.orbit, move |frame| {
                            frame.overlays = overlays;
                            for inject in injects {
                                inject(frame);
                            }
                            auto_fill_exposure_dt(frame, dt);
                        });
                }

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
                    // Without an input handler, feed the resolver directly (orbit /
                    // bound actions) so navigation works out of the box. With one,
                    // the handler owns forwarding, so only buffer the raw event; it
                    // decides each frame what the viewport should see.
                    if self.input.is_none() {
                        state.session.handle_event(ev.clone());
                    }
                    self.events.push(ev);
                    state.window.request_redraw();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::input::{ButtonState, MouseButton, ScrollUnits, ViewportContext};

    fn headless_instance() -> Option<ViewportInstance> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;
        let (device, _queue) =
            pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
                label: Some("input_ctx_tests"),
                required_limits: crate::ViewportRenderer::recommended_device_limits(&adapter),
                ..Default::default()
            }))
            .ok()?;
        Some(ViewportInstance::new(
            &device,
            crate::gpu::TextureFormat::Bgra8UnormSrgb,
        ))
    }

    // The core guarantee of the input handler: only the events it forwards reach
    // the viewport. A withheld event (here a scroll) leaves no trace in the
    // resolved frame, so a UI that consumes input keeps it off the viewport.
    #[test]
    fn input_ctx_forwards_only_selected_events() {
        let Some(mut session) = headless_instance() else {
            eprintln!("skipping input_ctx_forwards_only_selected_events: no GPU adapter");
            return;
        };
        session.begin_frame(ViewportContext {
            hovered: true,
            focused: true,
            viewport_size: [256.0, 256.0],
        });

        // Buffered this frame: a click (move + press + release) and a scroll.
        let events = vec![
            ViewportEvent::PointerMoved {
                position: glam::Vec2::new(40.0, 40.0),
            },
            ViewportEvent::MouseButton {
                button: MouseButton::Left,
                state: ButtonState::Pressed,
            },
            ViewportEvent::MouseButton {
                button: MouseButton::Left,
                state: ButtonState::Released,
            },
            ViewportEvent::Wheel {
                delta: glam::Vec2::new(0.0, 5.0),
                units: ScrollUnits::Lines,
            },
        ];

        {
            let mut ictx = InputCtx {
                session: &mut session,
                events: &events,
            };
            assert_eq!(ictx.events().len(), 4);
            // Forward the click, withhold the scroll (the last event).
            for ev in events[..3].iter().cloned() {
                ictx.forward(ev);
            }
        }

        let action = session.resolve();
        assert!(action.pointer.clicked, "forwarded click should resolve");
        assert_eq!(
            action.navigation.zoom, 0.0,
            "withheld scroll must not reach the viewport"
        );
    }

    // With no forwarding at all, the viewport sees nothing: the same buffered
    // events produce an idle frame. This is a menu-open frame that swallows input.
    #[test]
    fn input_ctx_dropping_all_events_leaves_viewport_idle() {
        let Some(mut session) = headless_instance() else {
            eprintln!(
                "skipping input_ctx_dropping_all_events_leaves_viewport_idle: no GPU adapter"
            );
            return;
        };
        session.begin_frame(ViewportContext {
            hovered: true,
            focused: true,
            viewport_size: [256.0, 256.0],
        });

        let events = vec![
            ViewportEvent::MouseButton {
                button: MouseButton::Left,
                state: ButtonState::Pressed,
            },
            ViewportEvent::MouseButton {
                button: MouseButton::Left,
                state: ButtonState::Released,
            },
            ViewportEvent::Wheel {
                delta: glam::Vec2::new(0.0, 5.0),
                units: ScrollUnits::Lines,
            },
        ];

        {
            let mut ictx = InputCtx {
                session: &mut session,
                events: &events,
            };
            // Forward nothing: the app consumed every event.
            let _ = &mut ictx;
        }

        let action = session.resolve();
        assert!(!action.pointer.clicked, "no forwarded click");
        assert_eq!(action.navigation.zoom, 0.0, "no forwarded scroll");
    }
}
