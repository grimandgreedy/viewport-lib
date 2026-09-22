//! `ViewportAppV2` is an experimental multi-window winit runner over
//! [`ViewportInstance`].
//!
//! Where [`ViewportApp`](crate::ViewportApp) owns exactly one window, this owns a
//! set of OS windows keyed by a library [`WindowId`], each driving its own
//! [`ViewportInstance`] against its own surface, over a single shared wgpu device.
//! Events are routed to the right window; each window redraws, navigates, and times
//! its frames independently. Windows can be opened and closed at runtime from a
//! callback.
//!
//! This type is experimental and its API may change. The stable single-window
//! [`ViewportApp`](crate::ViewportApp) is unaffected. See the runners module docs
//! for the composable [`ViewportInstance`] you drive from your own loop.
//!
//! Desktop-only: like [`ViewportApp`](crate::ViewportApp) it blocks on the event loop
//! and brings the device up synchronously (`pollster::block_on`), and multiple OS
//! windows are a desktop concept. On the web (one canvas) drive a
//! [`ViewportInstance`] from your own loop instead. It builds on every wgpu leg.
//! Per-window suspend/resume for mobile is out of scope; per-window
//! surface loss is handled (the surface is reconfigured on `Lost`/`Outdated`).
//!
//! # The winit boundary
//!
//! A consumer never touches winit through this runner: windows are named by the
//! library-owned [`WindowId`], configured with [`WindowConfig`], and the per-frame
//! callback receives viewport-lib types. For direct winit access the runner does not
//! model, reach the raw window with [`FrameCtxV2::raw_window`] or use the re-exported
//! `viewport_lib::winit` so you never add a second, mismatched winit dependency.

use std::collections::HashMap;
use std::sync::Arc;
use web_time::Instant;

use ::winit::application::ApplicationHandler;
use ::winit::event::WindowEvent;
use ::winit::event_loop::{ActiveEventLoop, EventLoop};
use ::winit::window::{Fullscreen, Window, WindowAttributes, WindowId as WinitWindowId};

use crate::interaction::input::adapters::{from_winit, from_winit_device};
use crate::interaction::input::{CursorShape, ViewportContext, ViewportEvent};
use crate::runners::ViewportInstance;
use crate::runners::viewport_app::{RedrawMode, cursor_icon};
use crate::{
    BlitTexture, ExposureMode, FrameData, OrbitCameraController, OverlayFrame, ViewportRenderer,
};

/// See [`ViewportApp`](crate::ViewportApp)'s equivalent: fill the auto-exposure `dt`
/// from the frame time so smooth adaptation works under the continuous-frame runner.
fn auto_fill_exposure_dt(frame: &mut FrameData, dt: f32) {
    if let ExposureMode::Automatic(ref mut auto) = frame.effects.display.exposure.mode {
        auto.dt = dt;
    }
}

// --- Pure bookkeeping helpers -------------------------------------------------
// The multi-window lifecycle decisions that do not touch winit or the GPU live here
// as free functions so they are unit-testable without spinning up real OS windows
// (winit's `WindowId` and a live surface cannot be constructed in a test).

/// Allocate the next library window id from a monotonically-increasing counter.
fn alloc_window_id(counter: &mut u64) -> WindowId {
    let id = WindowId(*counter);
    *counter += 1;
    id
}

/// Whether the event loop should end after a window closes: only when no windows
/// remain and the app is configured to exit on the last close.
fn should_end_loop(remaining_windows: usize, exit_on_last_close: bool) -> bool {
    remaining_windows == 0 && exit_on_last_close
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
    /// Hand the OS close button to the callback instead of closing immediately.
    /// Default: `false` (the close button closes the window). When `true`, a
    /// close request sets [`FrameCtxV2::close_requested`] on the next frame and does
    /// not close the window; the callback decides whether to call
    /// [`FrameCtxV2::close_window`] (for example after a save prompt).
    pub intercept_close: bool,
}

impl Default for AppConfigV2 {
    fn default() -> Self {
        Self {
            exit_on_last_window_close: true,
            intercept_close: false,
        }
    }
}

impl AppConfigV2 {
    /// Set whether the loop ends when the last window closes.
    pub fn with_exit_on_last_window_close(mut self, exit: bool) -> Self {
        self.exit_on_last_window_close = exit;
        self
    }

    /// Set whether the OS close button is handed to the callback (see
    /// [`intercept_close`](Self::intercept_close)).
    pub fn with_intercept_close(mut self, intercept: bool) -> Self {
        self.intercept_close = intercept;
        self
    }
}

/// The per-window setup callback: build this window's scene with the shared device
/// in hand. Mirrors [`ViewportApp::setup`](crate::ViewportApp::setup), one per
/// window.
type WindowFactory = Box<dyn FnOnce(&mut ViewportInstance, &crate::gpu::Device)>;

/// The per-window per-frame callback.
type WindowCallback = Box<dyn FnMut(&mut FrameCtxV2)>;

/// The optional per-window input handler. When set, the runner stops auto-feeding
/// events and driving orbit for that window; the handler owns the input step.
type WindowInputHandler = Box<dyn FnMut(&mut InputCtxV2)>;

/// The optional per-window paint hook. Runs after the window's own instance has
/// rendered, in a render pass over the surface, so a consumer can blit its own
/// offscreen viewports into rects of the window (in-window viewports).
type WindowPaint = Box<dyn FnMut(&mut PaintCtxV2)>;

/// A window declared before [`run`](ViewportAppV2::run), or requested at runtime,
/// realised into a [`WindowState`] when the runner holds the event loop.
struct WindowBuilder {
    config: WindowConfig,
    factory: WindowFactory,
    input: Option<WindowInputHandler>,
    callback: WindowCallback,
    paint: Option<WindowPaint>,
}

/// A window lifecycle request raised from a callback and applied once the runner
/// holds the [`ActiveEventLoop`] (winit windows can only be created from it).
enum WindowCommand {
    Open {
        id: WindowId,
        builder: WindowBuilder,
    },
    Close(WindowId),
}

/// What a window's per-frame callback receives: that window's [`ViewportInstance`]
/// (via deref) plus timing, the window's identity, and the app controls (open/close
/// windows, current-window control).
///
/// The callback is per window: it drives one window's instance, and is called once
/// per frame for that window. Frame-global work that must run once regardless of
/// window count does not belong here.
pub struct FrameCtxV2<'a> {
    session: &'a mut ViewportInstance,
    window: &'a Window,
    window_id: WindowId,
    dt: f32,
    time: f32,
    device: &'a crate::gpu::Device,
    queue: &'a crate::gpu::Queue,
    viewport_size: [f32; 2],
    surface_size: [u32; 2],
    surface_format: crate::gpu::TextureFormat,
    close_requested: bool,
    overlays: OverlayFrame,
    injects: Vec<Box<dyn FnOnce(&mut FrameData)>>,
    events: Vec<ViewportEvent>,
    commands: Vec<WindowCommand>,
    next_id: u64,
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

    /// The window surface size in physical pixels. Size an
    /// [`OffscreenViewportTarget`](crate::OffscreenViewportTarget) to this (or a
    /// sub-rect of it) so an in-window viewport renders at native resolution.
    pub fn surface_size(&self) -> [u32; 2] {
        self.surface_size
    }

    /// The window surface colour format. Build an
    /// [`OffscreenViewportTarget`](crate::OffscreenViewportTarget) and its
    /// `ViewportInstance` for this format so a blit into the window composites
    /// correctly.
    pub fn surface_format(&self) -> crate::gpu::TextureFormat {
        self.surface_format
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

    /// True when the OS asked to close this window since the last frame and
    /// [`AppConfigV2::intercept_close`] is set. The window has not been closed:
    /// call [`close_window`](Self::close_window) with this window's id to close it,
    /// or ignore it to keep the window open (for example while a save prompt is up).
    pub fn close_requested(&self) -> bool {
        self.close_requested
    }

    /// Open a new window at runtime. Returns its [`WindowId`] immediately; the window
    /// is realised after this frame. Takes the same setup and per-frame callbacks as
    /// [`ViewportAppV2::window`].
    pub fn open_window(
        &mut self,
        config: WindowConfig,
        factory: impl FnOnce(&mut ViewportInstance, &crate::gpu::Device) + 'static,
        callback: impl FnMut(&mut FrameCtxV2) + 'static,
    ) -> WindowId {
        let id = alloc_window_id(&mut self.next_id);
        self.commands.push(WindowCommand::Open {
            id,
            builder: WindowBuilder {
                config,
                factory: Box::new(factory),
                input: None,
                callback: Box::new(callback),
                paint: None,
            },
        });
        id
    }

    /// Close a window at runtime. Applied after this frame. Closing the last window
    /// ends the loop unless [`AppConfigV2::exit_on_last_window_close`] is `false`.
    pub fn close_window(&mut self, id: WindowId) {
        self.commands.push(WindowCommand::Close(id));
    }

    /// Set this window's title.
    pub fn set_title(&self, title: &str) {
        self.window.set_title(title);
    }

    /// Toggle borderless fullscreen for this window. Exclusive fullscreen (with a
    /// chosen video mode) is available through [`raw_window`](Self::raw_window).
    pub fn set_fullscreen(&self, fullscreen: bool) {
        self.window
            .set_fullscreen(fullscreen.then(|| Fullscreen::Borderless(None)));
    }

    /// Show or hide this window's decorations (title bar and border).
    pub fn set_decorations(&self, decorations: bool) {
        self.window.set_decorations(decorations);
    }

    /// Show or hide the cursor over this window.
    pub fn set_cursor_visible(&self, visible: bool) {
        self.window.set_cursor_visible(visible);
    }

    /// Set the shape the pointer takes over this window: a grab hand over a handle a
    /// tool can pick up, a crosshair over a picker, a resize arrow over an edge. Set it
    /// each frame from whatever the pointer is over; the shape persists until changed.
    pub fn set_cursor(&self, shape: CursorShape) {
        self.window.set_cursor(cursor_icon(shape));
    }

    /// Ask the OS to give this window keyboard focus.
    pub fn focus_window(&self) {
        self.window.focus_window();
    }

    /// The raw winit window, for platform-specific control the runner does not model
    /// (window icon, exact monitor video modes, drag_window, and so on). This is the
    /// escape hatch: prefer the modelled methods above where they exist.
    pub fn raw_window(&self) -> &Window {
        self.window
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

/// What a window's input handler receives when installed with
/// [`ViewportAppV2::window_with_input`]: the raw events buffered for this window
/// since its last frame plus the instance (via deref). Forward the ones the viewport
/// should act on with [`forward`](Self::forward) and drop the ones the UI consumed.
pub struct InputCtxV2<'a> {
    session: &'a mut ViewportInstance,
    window_id: WindowId,
    events: &'a [ViewportEvent],
}

impl InputCtxV2<'_> {
    /// Which window this input handler invocation is for.
    pub fn window_id(&self) -> WindowId {
        self.window_id
    }

    /// The input events that arrived for this window since its last frame, in order.
    /// These have not been sent to the viewport: forward the ones it should process.
    pub fn events(&self) -> &[ViewportEvent] {
        self.events
    }

    /// Send one event to this window's viewport so it drives picking, selection, and
    /// manipulation this frame. Skip an event to keep it from the viewport.
    pub fn forward(&mut self, event: ViewportEvent) {
        self.session.handle_event(event);
    }
}

impl std::ops::Deref for InputCtxV2<'_> {
    type Target = ViewportInstance;

    fn deref(&self) -> &ViewportInstance {
        self.session
    }
}

impl std::ops::DerefMut for InputCtxV2<'_> {
    fn deref_mut(&mut self) -> &mut ViewportInstance {
        self.session
    }
}

/// What a window's paint hook (installed with [`ViewportAppV2::window_with_paint`])
/// receives: a render pass over the window surface, after the window's own instance
/// has drawn, so the consumer can composite its own content into rects of the window.
///
/// The common use is in-window viewports: render other scenes into
/// [`OffscreenViewportTarget`](crate::OffscreenViewportTarget)s in the per-frame
/// callback, build a [`BlitTexture`] for each with
/// [`create_blit`](crate::ViewportRenderer::create_blit), then draw them here with
/// [`blit_rect`](Self::blit_rect). The consumer owns the rectangles; the runner only
/// yields the pass. Rects are in physical pixels (see [`surface_size`](Self::surface_size)).
pub struct PaintCtxV2<'a, 'rp> {
    window_id: WindowId,
    viewport_size: [f32; 2],
    surface_size: [u32; 2],
    device: &'a crate::gpu::Device,
    queue: &'a crate::gpu::Queue,
    renderer: &'a ViewportRenderer,
    rp: &'a mut crate::gpu::RenderPass<'rp>,
}

impl<'rp> PaintCtxV2<'_, 'rp> {
    /// Which window this paint hook invocation is for.
    pub fn window_id(&self) -> WindowId {
        self.window_id
    }

    /// The logical viewport size in points.
    pub fn viewport_size(&self) -> [f32; 2] {
        self.viewport_size
    }

    /// The surface size in physical pixels. Blit rects are in these units.
    pub fn surface_size(&self) -> [u32; 2] {
        self.surface_size
    }

    /// The shared wgpu device.
    pub fn device(&self) -> &crate::gpu::Device {
        self.device
    }

    /// The shared wgpu queue.
    pub fn queue(&self) -> &crate::gpu::Queue {
        self.queue
    }

    /// Blit a prepared texture into a physical-pixel rect of the window surface.
    ///
    /// Sets the pass viewport and scissor to `(x, y, w, h)` and draws `blit` there.
    /// Build `blit` once per source with
    /// [`create_blit`](crate::ViewportRenderer::create_blit) (in the per-frame
    /// callback, via [`renderer_mut`](ViewportInstance::renderer_mut)), rebuilding it
    /// when the source view changes.
    pub fn blit_rect(&mut self, blit: &BlitTexture, x: u32, y: u32, w: u32, h: u32) {
        self.rp
            .set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
        self.rp.set_scissor_rect(x, y, w, h);
        self.renderer.blit(self.rp, blit);
    }

    /// The window's renderer, for [`blit`](crate::ViewportRenderer::blit) /
    /// [`blit_with_depth`](crate::ViewportRenderer::blit_with_depth) variants beyond
    /// [`blit_rect`](Self::blit_rect).
    pub fn renderer(&self) -> &ViewportRenderer {
        self.renderer
    }

    /// The surface render pass (loaded, so the window's own render is preserved), for
    /// compositing the runner does not wrap in [`blit_rect`](Self::blit_rect).
    pub fn pass(&mut self) -> &mut crate::gpu::RenderPass<'rp> {
        self.rp
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
    /// the frame callback is per window and runs once per frame for that window. The
    /// runner drives a built-in orbit controller for this window; take input over
    /// with [`window_with_input`](Self::window_with_input).
    pub fn window(
        self,
        config: WindowConfig,
        factory: impl FnOnce(&mut ViewportInstance, &crate::gpu::Device) + 'static,
        callback: impl FnMut(&mut FrameCtxV2) + 'static,
    ) -> Self {
        self.push_window(config, factory, None, Box::new(callback), None)
    }

    /// Like [`window`](Self::window), but installs a per-window input handler. The
    /// runner stops auto-feeding events and driving orbit for this window; the
    /// handler runs once per frame with the events buffered since the last frame and
    /// forwards the ones the viewport should see (see [`InputCtxV2`]). Drive your own
    /// camera against [`camera_mut`](ViewportInstance::camera_mut).
    pub fn window_with_input(
        self,
        config: WindowConfig,
        factory: impl FnOnce(&mut ViewportInstance, &crate::gpu::Device) + 'static,
        input: impl FnMut(&mut InputCtxV2) + 'static,
        callback: impl FnMut(&mut FrameCtxV2) + 'static,
    ) -> Self {
        self.push_window(
            config,
            factory,
            Some(Box::new(input)),
            Box::new(callback),
            None,
        )
    }

    /// Like [`window`](Self::window), but installs a per-window paint hook that runs
    /// after this window's own instance has rendered, in a render pass over the
    /// surface. Use it to composite in-window viewports: render other scenes into
    /// [`OffscreenViewportTarget`](crate::OffscreenViewportTarget)s in `callback`,
    /// then blit them into rects in `paint` (see [`PaintCtxV2`]). The window's own
    /// scene draws first (leave it empty to make the window a pure compositor).
    pub fn window_with_paint(
        self,
        config: WindowConfig,
        factory: impl FnOnce(&mut ViewportInstance, &crate::gpu::Device) + 'static,
        callback: impl FnMut(&mut FrameCtxV2) + 'static,
        paint: impl FnMut(&mut PaintCtxV2) + 'static,
    ) -> Self {
        self.push_window(
            config,
            factory,
            None,
            Box::new(callback),
            Some(Box::new(paint)),
        )
    }

    fn push_window(
        mut self,
        config: WindowConfig,
        factory: impl FnOnce(&mut ViewportInstance, &crate::gpu::Device) + 'static,
        input: Option<WindowInputHandler>,
        callback: WindowCallback,
        paint: Option<WindowPaint>,
    ) -> Self {
        self.windows.push(WindowBuilder {
            config,
            factory: Box::new(factory),
            input,
            callback,
            paint,
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
    input: Option<WindowInputHandler>,
    callback: WindowCallback,
    paint: Option<WindowPaint>,
    /// Events translated for this window since its last frame.
    events: Vec<ViewportEvent>,
    focused: bool,
    hovered: bool,
    /// Set when the OS asked to close and `intercept_close` is on; surfaced to the
    /// callback once, then cleared.
    close_requested: bool,
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
    /// Allocate the next library window id.
    fn alloc_id(&mut self) -> WindowId {
        alloc_window_id(&mut self.next_id)
    }

    /// Create one window from its builder under a pre-assigned id, configure its
    /// surface against the shared device (bringing the device up on the first
    /// window), build its instance, and insert it into the window set.
    fn create_window(
        &mut self,
        event_loop: &ActiveEventLoop,
        id: WindowId,
        builder: WindowBuilder,
    ) {
        let WindowBuilder {
            config,
            factory,
            input,
            callback,
            paint,
        } = builder;

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title(config.title.clone())
                        .with_inner_size(::winit::dpi::LogicalSize::new(
                            config.width,
                            config.height,
                        )),
                )
                .expect("window"),
        );

        // Reuse the shared instance if the device is already up; otherwise this is
        // the first window and it seeds the shared device from its own surface.
        let surface = if let Some(gpu) = self.gpu.as_ref() {
            gpu.instance
                .create_surface(window.clone())
                .expect("surface")
        } else {
            let instance = crate::gpu::default_instance();
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
        let surface_config = crate::gpu::runner_surface_config(
            format,
            size.width.max(1),
            size.height.max(1),
            config.present_mode,
            caps.alpha_modes[0],
        );
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
                input,
                callback,
                paint,
                events: Vec::new(),
                focused,
                hovered,
                close_requested: false,
                last_frame: Instant::now(),
            },
        );
    }

    /// Draw one window: sync size/DPI, run its input handler and callback, assemble,
    /// present, then apply any window commands the callback raised.
    fn redraw(&mut self, event_loop: &ActiveEventLoop, id: WindowId) {
        let commands: Vec<WindowCommand>;
        let next_id: u64;
        let do_exit: bool;

        {
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

            // An installed input handler owns this window's input this frame: it
            // forwards the events the viewport should see. Run it before resolve so
            // the forwarded events land in this frame's ActionFrame.
            if let Some(handler) = state.input.as_mut() {
                let mut ictx = InputCtxV2 {
                    session: &mut state.session,
                    window_id: id,
                    events: &state.events,
                };
                handler(&mut ictx);
            }
            state.session.resolve();

            let mut ctx = FrameCtxV2 {
                session: &mut state.session,
                window: &state.window,
                window_id: id,
                dt,
                time,
                device: &gpu.device,
                queue: &gpu.queue,
                viewport_size: [w, h],
                surface_size: [state.surface_config.width, state.surface_config.height],
                surface_format: state.surface_config.format,
                close_requested: std::mem::take(&mut state.close_requested),
                overlays: OverlayFrame::default(),
                injects: Vec::new(),
                events: std::mem::take(&mut state.events),
                commands: Vec::new(),
                next_id: self.next_id,
                request_exit: false,
                request_redraw: false,
            };
            (state.callback)(&mut ctx);
            let FrameCtxV2 {
                overlays,
                injects,
                commands: raised,
                next_id: advanced,
                request_exit,
                request_redraw,
                ..
            } = ctx;
            commands = raised;
            next_id = advanced;
            do_exit = request_exit;

            state.session.step_runtime(dt);
            // With an input handler the app already drove the camera, so assemble
            // without touching it; otherwise drive the built-in orbit.
            if state.input.is_some() {
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
                    .update_orbit_with(&mut state.orbit, move |frame| {
                        frame.overlays = overlays;
                        for inject in injects {
                            inject(frame);
                        }
                        auto_fill_exposure_dt(frame, dt);
                    });
            }

            let frame = match crate::gpu::acquire_surface(&state.surface) {
                crate::gpu::SurfaceFrame::Acquired(f) => f,
                crate::gpu::SurfaceFrame::Recreate => {
                    state.surface.configure(&gpu.device, &state.surface_config);
                    return;
                }
                crate::gpu::SurfaceFrame::Skip => return,
            };
            let view = frame
                .texture
                .create_view(&crate::gpu::TextureViewDescriptor::default());
            let cmd = state.session.render(&gpu.device, &gpu.queue, &view);
            gpu.queue.submit(std::iter::once(cmd));

            // Optional per-window paint hook: composite consumer content (in-window
            // viewports) over the primary render, in a loaded pass on the surface.
            if let Some(paint) = state.paint.as_mut() {
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                            label: Some("viewport_app_v2_paint"),
                        });
                {
                    let mut rp = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(any(wgpu29, wgpu30))]
                        multiview_mask: None,
                        label: Some("viewport_app_v2_paint_pass"),
                        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: crate::gpu::Operations {
                                load: crate::gpu::LoadOp::Load,
                                store: crate::gpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    let renderer: &ViewportRenderer = state.session.renderer_mut();
                    let mut pctx = PaintCtxV2 {
                        window_id: id,
                        viewport_size: [w, h],
                        surface_size: [state.surface_config.width, state.surface_config.height],
                        device: &gpu.device,
                        queue: &gpu.queue,
                        renderer,
                        rp: &mut rp,
                    };
                    paint(&mut pctx);
                }
                gpu.queue.submit(std::iter::once(encoder.finish()));
            }

            crate::gpu::present(&gpu.queue, frame);

            state.session.begin_frame(ViewportContext {
                hovered: state.hovered,
                focused: state.focused,
                viewport_size: [w, h],
            });
            if state.redraw_mode == RedrawMode::Continuous || request_redraw {
                state.window.request_redraw();
            }
        }

        // The window borrow has ended; apply the callback's window commands.
        self.next_id = next_id;
        for command in commands {
            match command {
                WindowCommand::Open { id, builder } => self.create_window(event_loop, id, builder),
                WindowCommand::Close(id) => self.close_window(event_loop, id),
            }
        }
        if do_exit {
            event_loop.exit();
        }
    }

    /// Close one window: drop its state and, if the set is now empty and configured
    /// to, end the loop.
    fn close_window(&mut self, event_loop: &ActiveEventLoop, id: WindowId) {
        if let Some(state) = self.windows.remove(&id) {
            self.winit_ids.remove(&state.window.id());
        }
        if should_end_loop(self.windows.len(), self.config.exit_on_last_window_close) {
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
            let id = self.alloc_id();
            self.create_window(event_loop, id, builder);
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
        // the window's session directly (unless an input handler owns forwarding) and
        // buffer for the callback.
        if let WindowEvent::KeyboardInput {
            event: key_event, ..
        } = &event
        {
            if key_event.state == ::winit::event::ElementState::Pressed {
                if let Some(text) = &key_event.text {
                    let chars: Vec<char> = text.chars().filter(|c| !c.is_control()).collect();
                    if !chars.is_empty() {
                        if let Some(state) = self.windows.get_mut(&id) {
                            let auto_feed = state.input.is_none();
                            for c in chars {
                                let ev = ViewportEvent::Character(c);
                                if auto_feed {
                                    state.session.handle_event(ev.clone());
                                }
                                state.events.push(ev);
                            }
                            state.window.request_redraw();
                        }
                    }
                }
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                if self.config.intercept_close {
                    if let Some(state) = self.windows.get_mut(&id) {
                        state.close_requested = true;
                        state.window.request_redraw();
                    }
                } else {
                    self.close_window(event_loop, id);
                }
            }

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

            // DPI changed (moved to a display with a different scale, or the OS scale
            // changed). Redraw so this window re-reads its scale_factor; a Resized
            // usually follows and reconfigures the surface. Per-window, so only the
            // affected window redraws.
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(state) = self.windows.get_mut(&id) {
                    state.window.request_redraw();
                }
            }

            other => {
                if let Some(state) = self.windows.get_mut(&id) {
                    // A touch device never sends CursorEntered, so the contact itself
                    // is what makes the viewport hovered: without this the resolver
                    // gates every touch-driven gesture out.
                    if let WindowEvent::Touch(touch) = &other {
                        state.hovered = !matches!(
                            touch.phase,
                            ::winit::event::TouchPhase::Ended
                                | ::winit::event::TouchPhase::Cancelled
                        );
                    }
                    let scale = state.window.scale_factor() as f32;
                    if let Some(ev) = from_winit(&other, scale) {
                        if state.input.is_none() {
                            state.session.handle_event(ev.clone());
                        }
                        state.events.push(ev);
                        state.window.request_redraw();
                    }
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: ::winit::event::DeviceId,
        event: ::winit::event::DeviceEvent,
    ) {
        // Raw motion is not tied to a window; route it to the focused one, since
        // mouselook targets the window with keyboard focus.
        let Some(ev) = from_winit_device(&event) else {
            return;
        };
        if let Some(state) = self.windows.values_mut().find(|s| s.focused) {
            if state.input.is_none() {
                state.session.handle_event(ev.clone());
            }
            state.events.push(ev);
            state.window.request_redraw();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The multi-window bookkeeping that does not need winit or a GPU is unit-tested
    // here. Window creation, event routing (winit's WindowId has no public
    // constructor), and per-window redraw isolation need a real event loop and are
    // exercised by the app-multi-window example instead.

    #[test]
    fn window_ids_are_unique_and_monotonic() {
        let mut counter = 0u64;
        let a = alloc_window_id(&mut counter);
        let b = alloc_window_id(&mut counter);
        let c = alloc_window_id(&mut counter);
        assert_eq!(a.raw(), 0);
        assert_eq!(b.raw(), 1);
        assert_eq!(c.raw(), 2);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_eq!(counter, 3);
    }

    #[test]
    fn empty_set_exit_policy() {
        // Exit only when the last window closed and configured to do so.
        assert!(should_end_loop(0, true));
        assert!(!should_end_loop(0, false));
        // Windows remain: never exit on a close.
        assert!(!should_end_loop(1, true));
        assert!(!should_end_loop(3, false));
    }
}
