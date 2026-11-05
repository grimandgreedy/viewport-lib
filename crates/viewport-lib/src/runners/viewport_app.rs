//! `ViewportApp` is the winit runner over [`ViewportInstance`], for one window
//! or several.
//!
//! It owns a set of OS windows keyed by a library [`WindowId`], each driving its own
//! [`ViewportInstance`] against its own surface, over a single shared wgpu device.
//! Events are routed to the right window; each window redraws, navigates, and times
//! its frames independently. Windows can be opened and closed at runtime from a
//! callback.
//!
//! A single-window app declares one [`window`](crate::runners::viewport_app::ViewportApp::window). See the
//! runners module docs for the composable [`ViewportInstance`] you drive from your
//! own loop.
//!
//! Desktop-only: it blocks on the event loop and brings the device up
//! synchronously (`pollster::block_on`), and multiple OS windows are a desktop
//! concept. On the web (one canvas) drive a [`ViewportInstance`] from your own loop
//! instead. It builds on every wgpu leg. Per-window suspend/resume for mobile is
//! out of scope; per-window surface loss is handled (the surface is reconfigured
//! on `Lost`/`Outdated`).
//!
//! # Device loss
//!
//! The runner owns the device, so it also recovers from losing it. It watches
//! the device with one [`DeviceLostWatcher`](crate::DeviceLostWatcher) shared by
//! every window's renderer. When the device is lost it requests a new adapter
//! and device, recreates each window's surface and [`ViewportInstance`], runs
//! each window's setup callback again on the new instance, and puts the camera
//! back where it was. Anything else the app built on the old device (offscreen
//! targets, its own textures) has to be rebuilt by the app: build it from the
//! setup callback, or check [`FrameCtx::device_rebuilt`] in the frame callback.
//! A runtime attached in the setup callback is attached anew on the new
//! instance, so its plugins start from scratch.
//!
//! # A runtime under the runner
//!
//! Attach a runtime in the setup callback. Each frame the runner steps it
//! before the frame callback and runs its GPU plugins when it renders. The
//! step's events are the app's to apply: read them from
//! [`FrameCtx::runtime_output_mut`]. Skinned meshes and camera follow both
//! need this:
//!
//! ```rust,ignore
//! use std::{cell::RefCell, rc::Rc};
//! use viewport_lib::runtime::{CameraFollowTarget, apply_camera_commands};
//!
//! let skinning: Rc<RefCell<Option<SkinnedMeshHandle>>> = Rc::default();
//! let setup_skinning = skinning.clone();
//! ViewportApp::new(AppConfig::default())
//!     .window(
//!         WindowConfig::default(),
//!         move |setup| {
//!             let (device, queue) = (setup.device().clone(), setup.queue().clone());
//!             // ... upload the mesh, build `feature` and `follow` ...
//!             setup.set_runtime(ViewportRuntime::new().with_camera_follow(follow));
//!             *setup_skinning.borrow_mut() = setup.install_plugin(feature, &device, &queue).ok();
//!         },
//!         move |ctx| {
//!             let mut output = std::mem::take(ctx.runtime_output_mut());
//!             if let Some(handle) = skinning.borrow_mut().as_mut() {
//!                 let (device, queue) = (ctx.device().clone(), ctx.queue().clone());
//!                 handle.apply(ctx.resources_mut(), &device, &queue, &mut output);
//!             }
//!             if let Some(target) = output.events.read::<CameraFollowTarget>().last() {
//!                 let mut orbit = ctx.camera().orbit_target();
//!                 orbit.centre = target.0.as_dvec3();
//!                 ctx.camera_mut().set_orbit_target(orbit);
//!             }
//!             apply_camera_commands(&mut output.events, ctx.camera_mut());
//!         },
//!     )
//!     .run();
//! ```
//!
//! # The winit boundary
//!
//! A consumer never touches winit through this runner: windows are named by the
//! library-owned [`WindowId`], configured with [`WindowConfig`], and the per-frame
//! callback receives viewport-lib types. For direct winit access the runner does not
//! model, reach the raw window with [`FrameCtx::raw_window`] or use the re-exported
//! `viewport_lib::winit` so you never add a second, mismatched winit dependency.
//!
//! [`WindowId`]: crate::runners::viewport_app::WindowId
//! [`WindowConfig`]: crate::runners::viewport_app::WindowConfig
//! [`FrameCtx::raw_window`]: crate::runners::viewport_app::FrameCtx::raw_window

use std::collections::HashMap;
use std::sync::Arc;
use web_time::Instant;

use ::winit::application::ApplicationHandler;
use ::winit::event::WindowEvent;
use ::winit::event_loop::{ActiveEventLoop, EventLoop};
use ::winit::window::{
    CursorIcon, Fullscreen, Window, WindowAttributes, WindowId as WinitWindowId,
};

use crate::interaction::input::adapters::AdapterCtx;
use crate::interaction::input::adapters::winit::{translate, translate_device};
use crate::interaction::input::{
    CursorShape, InputOwnership, KeyboardOwnership, PointerOwnership, ViewportContext,
    ViewportEvent,
};
use crate::runners::ViewportInstance;
use crate::{
    BlitTexture, ExposureMode, FrameData, OrbitCameraController, OverlayFrame, ViewportRenderer,
};

/// Fill the auto-exposure `dt` from the frame time so smooth adaptation works under
/// the continuous-frame runner.
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
    /// frame to keep the loop going. Under `ExposureMode::Automatic` the runner
    /// draws one frame more after each, which shades with the exposure the
    /// first metered.
    OnDemand,
}

/// Whether to ask for another frame once this one is done.
///
/// A frame whose surface had to be reconfigured was never shown, so it is
/// retried whatever the mode: under [`RedrawMode::OnDemand`] nothing else would
/// ask for it. A frame skipped for a transient reason (a timeout, an occluded
/// window) follows the mode, so an occluded on-demand window does not spin.
pub(crate) fn wants_redraw(mode: RedrawMode, requested: bool, reconfigured: bool) -> bool {
    mode == RedrawMode::Continuous || requested || reconfigured
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
/// Names a window opened through [`ViewportApp`] so consumer code and event
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

/// Per-window configuration for a window opened through [`ViewportApp`].
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
            title: "Viewport".to_string(),
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

/// App-level configuration for a [`ViewportApp`].
///
/// Non-exhaustive: build with [`AppConfig::default`] and the `with_*` methods.
/// Per-window options live on [`WindowConfig`]; this holds settings that span the
/// whole app.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// End the event loop when the last window closes. Default: `true`. Set `false`
    /// to keep the loop alive with no windows (for example to reopen one later).
    pub exit_on_last_window_close: bool,
    /// Hand the OS close button to the callback instead of closing immediately.
    /// Default: `false` (the close button closes the window). When `true`, a
    /// close request sets [`FrameCtx::close_requested`] on the next frame and does
    /// not close the window; the callback decides whether to call
    /// [`FrameCtx::close_window`] (for example after a save prompt).
    pub intercept_close: bool,
    /// File the runner keeps the GPU pipeline cache in. Default: `None`, no
    /// cache. See [`with_pipeline_cache`](Self::with_pipeline_cache).
    pub pipeline_cache_path: Option<std::path::PathBuf>,
    /// Renderer settings for every window. The runner replaces
    /// `target_format` with each surface's format and draws without MSAA, so
    /// `sample_count` is ignored; a pipeline cache loaded from
    /// `pipeline_cache_path` replaces `pipeline_cache_data`.
    pub renderer: crate::RendererConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            exit_on_last_window_close: true,
            intercept_close: false,
            pipeline_cache_path: None,
            renderer: crate::RendererConfig::new(crate::gpu::TextureFormat::Bgra8UnormSrgb),
        }
    }
}

impl AppConfig {
    /// Set the renderer settings every window's renderer is built with, for
    /// example the pipeline compilation policy.
    pub fn with_renderer_config(mut self, config: crate::RendererConfig) -> Self {
        self.renderer = config;
        self
    }

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

    /// Keep the GPU pipeline cache in `path` between runs. The windows share
    /// one device and so one cache.
    ///
    /// The runner loads the file before it builds the renderer and writes it
    /// back after the first frame and again on exit, so a later launch skips
    /// the shader compilation the first one paid for. It only has an effect on
    /// a backend with a pipeline cache (Vulkan); elsewhere nothing is read or
    /// written. The file is specific to the GPU and driver, and data that no
    /// longer matches is discarded, so a per-user cache directory is the place
    /// for it.
    pub fn with_pipeline_cache(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.pipeline_cache_path = Some(path.into());
        self
    }
}

/// The per-window setup callback: build this window's scene through its
/// [`SetupCtx`], one per window. Runs again on a fresh instance after the
/// device is lost and recreated.
type WindowFactory = Box<dyn FnMut(&mut SetupCtx)>;

/// The per-window per-frame callback.
type WindowCallback = Box<dyn FnMut(&mut FrameCtx)>;

/// The optional per-window input handler. When set, the runner stops auto-feeding
/// events and driving orbit for that window; the handler owns the input step.
type WindowInputHandler = Box<dyn FnMut(&mut InputCtx)>;

/// The optional per-window paint hook. Runs after the window's own instance has
/// rendered, in a render pass over the surface, so a consumer can blit its own
/// offscreen viewports into rects of the window (in-window viewports).
type WindowPaint = Box<dyn FnMut(&mut PaintCtx)>;

/// A window declared before [`run`](ViewportApp::run), or requested at runtime,
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

/// What a window's setup callback receives: the window's new
/// [`ViewportInstance`] (via deref) and the shared device and queue.
///
/// The runner builds it, once when the window is created and again on a new
/// instance after the device is lost and recreated
/// ([`device_rebuilt`](Self::device_rebuilt) tells the two apart). Build
/// everything the window draws here: upload meshes, fill the scene, attach a
/// runtime with [`set_runtime`](ViewportInstance::set_runtime) and install
/// features with [`install_plugin`](ViewportInstance::install_plugin).
///
/// ```rust,ignore
/// |setup| {
///     let device = setup.device().clone();
///     let mesh = setup.upload_mesh(&device, &primitives::cube(1.0)).unwrap();
///     setup.scene_mut().add(Some(mesh), glam::Mat4::IDENTITY, MaterialId::DEFAULT);
///     setup.set_runtime(ViewportRuntime::new().with_gpu_plugin(MyGpuPlugin::default()));
/// }
/// ```
pub struct SetupCtx<'a> {
    session: &'a mut ViewportInstance,
    device: &'a crate::gpu::Device,
    queue: &'a crate::gpu::Queue,
    device_rebuilt: bool,
}

impl SetupCtx<'_> {
    /// The shared wgpu device.
    pub fn device(&self) -> &crate::gpu::Device {
        self.device
    }

    /// The shared wgpu queue.
    pub fn queue(&self) -> &crate::gpu::Queue {
        self.queue
    }

    /// True when this is the rerun after the runner replaced a lost device.
    /// The instance is new either way: the old one, its runtime and
    /// everything built on the old device are gone.
    pub fn device_rebuilt(&self) -> bool {
        self.device_rebuilt
    }
}

impl std::ops::Deref for SetupCtx<'_> {
    type Target = ViewportInstance;

    fn deref(&self) -> &ViewportInstance {
        self.session
    }
}

impl std::ops::DerefMut for SetupCtx<'_> {
    fn deref_mut(&mut self) -> &mut ViewportInstance {
        self.session
    }
}

/// What a window's per-frame callback receives: that window's [`ViewportInstance`]
/// (via deref) plus timing, the window's identity, and the app controls (open/close
/// windows, current-window control).
///
/// The callback is per window: it drives one window's instance, and is called once
/// per frame for that window. Frame-global work that must run once regardless of
/// window count does not belong here.
pub struct FrameCtx<'a> {
    session: &'a mut ViewportInstance,
    runtime_output: crate::runtime::RuntimeOutput,
    device_rebuilt: bool,
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

impl FrameCtx<'_> {
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

    /// True on the first frame after the runner replaced a lost device. The
    /// instance is new and its setup callback has run again; anything the
    /// frame callback built on the old device (offscreen targets, its own
    /// textures) has to be built again on [`device`](Self::device).
    pub fn device_rebuilt(&self) -> bool {
        self.device_rebuilt
    }

    /// The shared wgpu queue.
    pub fn queue(&self) -> &crate::gpu::Queue {
        self.queue
    }

    /// This frame's runtime output: what
    /// [`step_runtime`](ViewportInstance::step_runtime) returned when the runner
    /// stepped the window's runtime, just before this callback. Read or drain
    /// its events here (skinned pose updates, the camera follow target and
    /// commands); the runner applies none of them itself. Empty when no
    /// runtime is attached.
    ///
    /// The runtime steps before the callback, so scene edits made here reach
    /// it on the next frame.
    pub fn runtime_output_mut(&mut self) -> &mut crate::runtime::RuntimeOutput {
        &mut self.runtime_output
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
    /// [`AppConfig::intercept_close`] is set. The window has not been closed:
    /// call [`close_window`](Self::close_window) with this window's id to close it,
    /// or ignore it to keep the window open (for example while a save prompt is up).
    pub fn close_requested(&self) -> bool {
        self.close_requested
    }

    /// Open a new window at runtime. Returns its [`WindowId`] immediately; the window
    /// is realised after this frame. Takes the same setup and per-frame callbacks as
    /// [`ViewportApp::window`]; the setup callback gets a [`SetupCtx`].
    pub fn open_window(
        &mut self,
        config: WindowConfig,
        setup: impl FnMut(&mut SetupCtx) + 'static,
        callback: impl FnMut(&mut FrameCtx) + 'static,
    ) -> WindowId {
        let id = alloc_window_id(&mut self.next_id);
        self.commands.push(WindowCommand::Open {
            id,
            builder: WindowBuilder {
                config,
                factory: Box::new(setup),
                input: None,
                callback: Box::new(callback),
                paint: None,
            },
        });
        id
    }

    /// Close a window at runtime. Applied after this frame. Closing the last window
    /// ends the loop unless [`AppConfig::exit_on_last_window_close`] is `false`.
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

    /// Turn the platform input method on or off for this window. Turn it on while a
    /// text field has focus, so composition arrives as [`ViewportEvent::Ime`], and off
    /// when it loses focus. Off by default.
    pub fn set_ime_allowed(&self, allowed: bool) {
        self.window.set_ime_allowed(allowed);
    }

    /// Tell the platform where the caret is, in logical points from the window's
    /// top-left, so the input method opens its candidate window beside it. Set it
    /// each frame while composition is on.
    pub fn set_ime_cursor_area(&self, position: [f32; 2], size: [f32; 2]) {
        self.window.set_ime_cursor_area(
            ::winit::dpi::LogicalPosition::new(position[0], position[1]),
            ::winit::dpi::LogicalSize::new(size[0], size[1]),
        );
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

/// What a window's input handler receives when installed with
/// [`ViewportApp::window_with_input`]: the raw events buffered for this window
/// since its last frame plus the instance (via deref). Forward the ones the viewport
/// should act on with [`forward`](Self::forward) and drop the ones the UI consumed.
pub struct InputCtx<'a> {
    session: &'a mut ViewportInstance,
    window_id: WindowId,
    events: &'a [ViewportEvent],
    dt: f32,
}

impl InputCtx<'_> {
    /// Seconds since this window's previous frame. Pass it to a camera
    /// controller's `apply` for held input such as fly movement.
    pub fn dt(&self) -> f32 {
        self.dt
    }

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
    /// the frame's resolved actions. Skip an event to keep it from the viewport.
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

/// What a window's paint hook (installed with [`ViewportApp::window_with_paint`])
/// receives: a render pass over the window surface, after the window's own instance
/// has drawn, so the consumer can composite its own content into rects of the window.
///
/// The common use is in-window viewports: render other scenes into
/// [`OffscreenViewportTarget`](crate::OffscreenViewportTarget)s in the per-frame
/// callback, build a [`BlitTexture`] for each with
/// [`create_blit`](crate::ViewportRenderer::create_blit), then draw them here with
/// [`blit_rect`](Self::blit_rect). The consumer owns the rectangles; the runner only
/// yields the pass. Rects are in physical pixels (see [`surface_size`](Self::surface_size)).
pub struct PaintCtx<'a, 'rp> {
    window_id: WindowId,
    viewport_size: [f32; 2],
    surface_size: [u32; 2],
    device: &'a crate::gpu::Device,
    queue: &'a crate::gpu::Queue,
    renderer: &'a ViewportRenderer,
    rp: &'a mut crate::gpu::RenderPass<'rp>,
}

impl<'rp> PaintCtx<'_, 'rp> {
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

    /// Composite a prepared texture over a physical-pixel rect of the window
    /// surface, instead of replacing it as [`blit_rect`](Self::blit_rect) does.
    ///
    /// For drawing a viewport inside the window's own render: prepare the source
    /// with [`create_blit_composite`](crate::ViewportRenderer::create_blit_composite)
    /// and render it with a transparent background, and whatever the window
    /// already drew shows through wherever the inner viewport drew nothing.
    pub fn blit_composite_rect(&mut self, blit: &BlitTexture, x: u32, y: u32, w: u32, h: u32) {
        self.rp
            .set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
        self.rp.set_scissor_rect(x, y, w, h);
        self.renderer.blit_composite(self.rp, blit);
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

/// A winit runner driving one [`ViewportInstance`] per OS window, for one window
/// or several.
///
/// Declare windows with [`window`](Self::window), then [`run`](Self::run):
///
/// ```rust,ignore
/// use viewport_lib::{AppConfig, Colour, Material, ViewportApp, WindowConfig, primitives};
///
/// ViewportApp::new(AppConfig::default())
///     .window(
///         WindowConfig::default().with_title("left"),
///         |setup| {
///             let device = setup.device().clone();
///             let mesh = setup.resources_mut()
///                 .upload_mesh_data(&device, &primitives::cube(1.0)).unwrap();
///             let red = setup.resources_mut()
///                 .upload_material(Material::from_colour(Colour::linear_rgb(0.85, 0.25, 0.2))).unwrap();
///             setup.scene_mut().add(Some(mesh), glam::Mat4::IDENTITY, red);
///             setup.camera_mut().set_orbit_distance(6.0);
///         },
///         |_ctx| {},
///     )
///     .window(
///         WindowConfig::default().with_title("right"),
///         |setup| { /* a different scene */ },
///         |_ctx| {},
///     )
///     .run();
/// ```
pub struct ViewportApp {
    config: AppConfig,
    windows: Vec<WindowBuilder>,
}

impl ViewportApp {
    /// Create a runner with the given app configuration.
    pub fn new(config: AppConfig) -> Self {
        Self {
            config,
            windows: Vec::new(),
        }
    }

    /// Declare a window: its configuration, a setup callback that builds its
    /// scene, and a per-frame callback that drives it.
    ///
    /// Call once per window before [`run`](Self::run). The setup callback runs
    /// with a [`SetupCtx`] after the window's instance is created: upload
    /// meshes, build the initial scene, attach a runtime with
    /// [`set_runtime`](ViewportInstance::set_runtime), install features with
    /// [`install_plugin`](ViewportInstance::install_plugin).
    ///
    /// After a device loss the runner builds a new instance and runs the setup
    /// callback again on it, with [`SetupCtx::device_rebuilt`] true. The old
    /// instance goes with everything on it, the runtime included, so a setup
    /// callback that attaches a runtime attaches a new one there, and runtime
    /// plugin state (animation clocks, simulation) starts again. Only the
    /// camera is carried over. Keep anything that must survive a device loss
    /// outside the instance.
    ///
    /// Each frame the runner steps the window's runtime, then runs the frame
    /// callback (its output is on [`FrameCtx::runtime_output_mut`]), then
    /// renders, which runs the runtime's GPU plugins. The runner drives a
    /// built-in orbit controller for this window; take input over with
    /// [`window_with_input`](Self::window_with_input).
    pub fn window(
        self,
        config: WindowConfig,
        setup: impl FnMut(&mut SetupCtx) + 'static,
        callback: impl FnMut(&mut FrameCtx) + 'static,
    ) -> Self {
        self.push_window(config, setup, None, Box::new(callback), None)
    }

    /// Like [`window`](Self::window), but installs a per-window input handler. The
    /// runner stops auto-feeding events and driving orbit for this window; the
    /// handler runs once per frame with the events buffered since the last frame and
    /// forwards the ones the viewport should see (see [`InputCtx`]). Drive your own
    /// camera against [`camera_mut`](ViewportInstance::camera_mut).
    pub fn window_with_input(
        self,
        config: WindowConfig,
        setup: impl FnMut(&mut SetupCtx) + 'static,
        input: impl FnMut(&mut InputCtx) + 'static,
        callback: impl FnMut(&mut FrameCtx) + 'static,
    ) -> Self {
        self.push_window(
            config,
            setup,
            Some(Box::new(input)),
            Box::new(callback),
            None,
        )
    }

    /// Like [`window`](Self::window), but installs a per-window paint hook that runs
    /// after this window's own instance has rendered, in a render pass over the
    /// surface. Use it to composite in-window viewports: render other scenes into
    /// [`OffscreenViewportTarget`](crate::OffscreenViewportTarget)s in `callback`,
    /// then blit them into rects in `paint` (see [`PaintCtx`]). The window's own
    /// scene draws first (leave it empty to make the window a pure compositor).
    pub fn window_with_paint(
        self,
        config: WindowConfig,
        setup: impl FnMut(&mut SetupCtx) + 'static,
        callback: impl FnMut(&mut FrameCtx) + 'static,
        paint: impl FnMut(&mut PaintCtx) + 'static,
    ) -> Self {
        self.push_window(
            config,
            setup,
            None,
            Box::new(callback),
            Some(Box::new(paint)),
        )
    }

    fn push_window(
        mut self,
        config: WindowConfig,
        setup: impl FnMut(&mut SetupCtx) + 'static,
        input: Option<WindowInputHandler>,
        callback: WindowCallback,
        paint: Option<WindowPaint>,
    ) -> Self {
        self.windows.push(WindowBuilder {
            config,
            factory: Box::new(setup),
            input,
            callback,
            paint,
        });
        self
    }

    /// Run the event loop, driving every declared window. Blocks until the loop
    /// ends (the last window closes, unless configured otherwise, or a callback
    /// calls [`request_exit`](FrameCtx::request_exit)).
    ///
    /// # Panics
    ///
    /// With the error [`try_run`](Self::try_run) returns.
    pub fn run(self) {
        if let Err(e) = self.try_run() {
            panic!("{e}");
        }
    }

    /// As [`run`](Self::run), returning the error that ended the loop instead
    /// of panicking: [`ViewportError::DeviceUnsupported`](crate::ViewportError::DeviceUnsupported)
    /// when the device is below the tier a renderer can be built on, or
    /// [`ViewportError::DeviceLost`](crate::ViewportError::DeviceLost) when a
    /// device was lost while a window's renderer was being built.
    pub fn try_run(self) -> Result<(), crate::ViewportError> {
        let event_loop = EventLoop::new().expect("event loop");
        let mut handler = AppHandler {
            config: self.config,
            pending: self.windows,
            gpu: None,
            windows: HashMap::new(),
            winit_ids: HashMap::new(),
            next_id: 0,
            start: Instant::now(),
            error: None,
        };
        event_loop.run_app(&mut handler).expect("run app");
        match handler.error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// The shared wgpu state, created once on the first window and reused for the rest.
struct Gpu {
    instance: crate::gpu::Instance,
    adapter: crate::gpu::Adapter,
    device: crate::gpu::Device,
    queue: crate::gpu::Queue,
    /// Shared by every window's renderer: wgpu keeps one lost callback per
    /// device.
    lost: crate::DeviceLostWatcher,
    /// The device's pipeline cache, made by the first window and shared by
    /// every window's renderer, or `None` on a device without
    /// `Features::PIPELINE_CACHE`.
    pipeline_cache: std::sync::OnceLock<Option<crate::gpu::PipelineCache>>,
}

impl Gpu {
    /// Bring up an instance, an adapter that can present to `window`, and a
    /// device, returning the window's surface with them. The instance carries
    /// the event loop's display, which GL needs to present under Wayland.
    fn request(
        event_loop: &ActiveEventLoop,
        window: Arc<Window>,
        marks: &mut super::InitMarks,
    ) -> (Self, crate::gpu::Surface<'static>) {
        let instance = crate::gpu::instance_with_display(
            crate::gpu::Backends::default(),
            event_loop.owned_display_handle(),
        );
        let surface = instance.create_surface(window).expect("surface");
        marks.mark("instance_and_surface");
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            },
        ))
        .expect("adapter");
        marks.mark("request_adapter");
        let required_features = crate::ViewportRenderer::recommended_device_features(&adapter);
        let (device, queue) =
            pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
                required_features,
                required_limits: crate::ViewportRenderer::recommended_device_limits(&adapter),
                ..Default::default()
            }))
            .expect("device");
        marks.mark("request_device");
        let lost = crate::DeviceLostWatcher::install(&device);
        let gpu = Self {
            instance,
            adapter,
            device,
            queue,
            lost,
            pipeline_cache: std::sync::OnceLock::new(),
        };
        (gpu, surface)
    }

    /// The format a window's surface is configured with: the first sRGB one.
    fn surface_format(&self, surface: &crate::gpu::Surface<'static>) -> crate::gpu::TextureFormat {
        let caps = surface.get_capabilities(&self.adapter);
        caps.formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .or_else(|| caps.formats.first().copied())
            .expect("the adapter cannot present to this surface")
    }

    /// Build a window's instance on this device.
    fn build_session(
        &self,
        app: &AppConfig,
        format: crate::gpu::TextureFormat,
    ) -> Result<ViewportInstance, crate::ViewportError> {
        let mut config =
            super::runner_renderer_config(&app.renderer, format, None, &self.adapter, &self.device)
                .with_device_lost_watcher(self.lost.clone());
        // Every window's renderer shares the device's one cache, made with the
        // first window from the saved data, so they reuse each other's
        // pipelines and the data saved on exit is the same from any of them.
        // A cache the host set on its config is kept.
        if config.pipeline_cache.is_none() {
            let cache = self.pipeline_cache.get_or_init(|| {
                let data = super::load_pipeline_cache(app.pipeline_cache_path.as_deref())
                    .or_else(|| app.renderer.pipeline_cache_data.clone());
                crate::resources::create_pipeline_cache(&self.device, data.as_deref())
            });
            if let Some(cache) = cache {
                config = config.with_pipeline_cache(cache.clone());
            }
        }
        ViewportInstance::try_with_config(&self.device, &config)
    }
}

/// Everything one window owns. The multi-window generalisation of the single-window
/// runner's `RunState`: input, orbit, timing, and focus/hover are all per window.
struct WindowState {
    window: Arc<Window>,
    surface: crate::gpu::Surface<'static>,
    surface_config: crate::gpu::SurfaceConfiguration,
    session: ViewportInstance,
    /// The setup callback, kept to run again after the device is recreated.
    factory: WindowFactory,
    /// Set when the device was recreated since this window's last frame.
    device_rebuilt: bool,
    redraw_mode: RedrawMode,
    /// The last frame was the extra one an automatic exposure asks for.
    exposure_follow_up: bool,
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
    /// Startup timing, held until the window's first frame is presented.
    startup_marks: Option<super::InitMarks>,
}

impl WindowState {
    /// Who owns this window's input. The viewport fills the window with no chrome
    /// over it, so the pointer is the viewport's whenever it is in the window, and
    /// the keyboard whenever the window has focus.
    fn ownership(&self) -> InputOwnership {
        InputOwnership {
            pointer: if self.hovered {
                PointerOwnership::Owned
            } else {
                PointerOwnership::Elsewhere
            },
            keyboard: if self.focused {
                KeyboardOwnership::Owned
            } else {
                KeyboardOwnership::Elsewhere
            },
        }
    }
}

struct AppHandler {
    config: AppConfig,
    /// Windows declared before `run`, realised on first `resumed`.
    pending: Vec<WindowBuilder>,
    gpu: Option<Gpu>,
    windows: HashMap<WindowId, WindowState>,
    /// Translates an incoming winit window id to the library id it was assigned.
    winit_ids: HashMap<WinitWindowId, WindowId>,
    next_id: u64,
    start: Instant,
    /// The error that ended the loop, returned from `try_run`.
    error: Option<crate::ViewportError>,
}

impl AppHandler {
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
        if self
            .gpu
            .as_ref()
            .is_some_and(|gpu| gpu.lost.is_lost_or_destroyed())
        {
            self.recover_device(event_loop);
        }
        let WindowBuilder {
            config,
            factory,
            input,
            callback,
            paint,
        } = builder;
        let mut marks = super::InitMarks::new();

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
        marks.mark("window");

        // Reuse the shared instance if the device is already up; otherwise this is
        // the first window and it seeds the shared device from its own surface.
        let surface = if let Some(gpu) = self.gpu.as_ref() {
            gpu.instance
                .create_surface(window.clone())
                .expect("surface")
        } else {
            let (gpu, surface) = Gpu::request(event_loop, window.clone(), &mut marks);
            self.gpu = Some(gpu);
            surface
        };

        let gpu = self.gpu.as_ref().unwrap();
        let size = window.inner_size();
        let caps = surface.get_capabilities(&gpu.adapter);
        let format = gpu.surface_format(&surface);
        let surface_config = crate::gpu::runner_surface_config(
            format,
            size.width.max(1),
            size.height.max(1),
            config.present_mode,
            caps.alpha_modes
                .first()
                .copied()
                .unwrap_or(crate::gpu::CompositeAlphaMode::Auto),
        );
        surface.configure(&gpu.device, &surface_config);
        marks.mark("surface_configure");

        let mut session = match gpu.build_session(&self.config, format) {
            Ok(session) => session,
            Err(e) => {
                self.fail(event_loop, e);
                return;
            }
        };
        marks.mark("renderer");
        let mut factory = factory;
        factory(&mut SetupCtx {
            session: &mut session,
            device: &gpu.device,
            queue: &gpu.queue,
            device_rebuilt: false,
        });
        marks.mark("setup");

        let focused = true;
        let hovered = false;
        let scale = window.scale_factor() as f32;
        session.set_pixels_per_point(scale);
        // A new window has focus and has not yet seen the cursor.
        session.begin_frame(ViewportContext::new(
            InputOwnership {
                pointer: PointerOwnership::Elsewhere,
                keyboard: KeyboardOwnership::Owned,
            },
            [
                surface_config.width as f32 / scale,
                surface_config.height as f32 / scale,
            ],
        ));

        self.winit_ids.insert(window.id(), id);
        window.request_redraw();

        self.windows.insert(
            id,
            WindowState {
                window,
                surface,
                surface_config,
                session,
                factory,
                device_rebuilt: false,
                redraw_mode: config.redraw_mode,
                exposure_follow_up: false,
                orbit: OrbitCameraController::new_stateless(),
                input,
                callback,
                paint,
                events: Vec::new(),
                focused,
                hovered,
                close_requested: false,
                last_frame: Instant::now(),
                startup_marks: Some(marks),
            },
        );
    }

    /// End the loop with `error`, which `try_run` returns.
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: crate::ViewportError) {
        tracing::error!("ViewportApp: {error}");
        self.error.get_or_insert(error);
        event_loop.exit();
    }

    /// Replace a lost device: a new instance, adapter and device, then for each
    /// window a new surface and instance with its setup callback run again and
    /// its camera put back.
    fn recover_device(&mut self, event_loop: &ActiveEventLoop) {
        let reason = self
            .gpu
            .as_ref()
            .and_then(|gpu| gpu.lost.info())
            .map(|info| format!("{:?}: {}", info.reason, info.message))
            .unwrap_or_default();
        tracing::warn!("ViewportApp: the GPU device was lost ({reason}); recreating it");
        let states: Vec<(WindowId, WindowState)> = self.windows.drain().collect();
        let mut marks = super::InitMarks::new();
        let mut new_gpu: Option<Gpu> = None;
        let mut failed = None;
        for (id, mut state) in states {
            // A window holds one surface at a time, so the old device's goes
            // before the new one is made.
            drop(state.surface);
            state.surface = match new_gpu.as_ref() {
                Some(gpu) => gpu
                    .instance
                    .create_surface(state.window.clone())
                    .expect("surface"),
                None => {
                    let (gpu, surface) = Gpu::request(event_loop, state.window.clone(), &mut marks);
                    new_gpu = Some(gpu);
                    surface
                }
            };
            let gpu = new_gpu.as_ref().unwrap();
            state.surface_config.format = gpu.surface_format(&state.surface);
            state.surface.configure(&gpu.device, &state.surface_config);
            if failed.is_none() {
                match gpu.build_session(&self.config, state.surface_config.format) {
                    Ok(mut session) => {
                        (state.factory)(&mut SetupCtx {
                            session: &mut session,
                            device: &gpu.device,
                            queue: &gpu.queue,
                            device_rebuilt: true,
                        });
                        *session.camera_mut() = state.session.camera().clone();
                        let scale = state.window.scale_factor() as f32;
                        session.set_pixels_per_point(scale);
                        session.begin_frame(ViewportContext::new(
                            state.ownership(),
                            [
                                state.surface_config.width as f32 / scale,
                                state.surface_config.height as f32 / scale,
                            ],
                        ));
                        state.session = session;
                        state.device_rebuilt = true;
                        state.window.request_redraw();
                    }
                    Err(e) => failed = Some(e),
                }
            }
            self.windows.insert(id, state);
        }
        self.gpu = new_gpu;
        if let Some(e) = failed {
            self.fail(event_loop, e);
        }
    }

    /// Draw one window: sync size/DPI, run its input handler and callback, assemble,
    /// present, then apply any window commands the callback raised.
    fn redraw(&mut self, event_loop: &ActiveEventLoop, id: WindowId) {
        if self
            .gpu
            .as_ref()
            .is_some_and(|gpu| gpu.lost.is_lost_or_destroyed())
        {
            self.recover_device(event_loop);
        }
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
                let mut ictx = InputCtx {
                    session: &mut state.session,
                    window_id: id,
                    events: &state.events,
                    dt,
                };
                handler(&mut ictx);
            }
            state.session.resolve();
            // Step before the callback, so it can read this frame's runtime
            // output. Scene edits the callback makes reach the runtime next frame.
            let runtime_output = state.session.step_runtime(dt);

            let mut ctx = FrameCtx {
                session: &mut state.session,
                runtime_output,
                device_rebuilt: std::mem::take(&mut state.device_rebuilt),
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
            let FrameCtx {
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

            // With an input handler the app already drove the camera, so assemble
            // without touching it; otherwise drive the built-in orbit.
            let assembled = if state.input.is_some() {
                let vctx = ViewportContext::new(state.ownership(), [w, h]);
                state.session.frame_with(vctx, move |frame| {
                    frame.overlays = overlays;
                    for inject in injects {
                        inject(frame);
                    }
                    auto_fill_exposure_dt(frame, dt);
                })
            } else {
                state
                    .session
                    .update_orbit_with(&mut state.orbit, dt, move |frame| {
                        frame.overlays = overlays;
                        for inject in injects {
                            inject(frame);
                        }
                        auto_fill_exposure_dt(frame, dt);
                    })
            };
            let automatic_exposure = matches!(
                assembled.effects.display.exposure.mode,
                ExposureMode::Automatic(_)
            );

            // A frame that cannot be acquired is not drawn, but the rest of the
            // turn still runs: the callback's window commands and exit request,
            // the next frame's input window, and the redraw request.
            let mut reconfigured = false;
            let acquired = match crate::gpu::acquire_surface(&state.surface) {
                crate::gpu::SurfaceFrame::Acquired(f) => Some(f),
                crate::gpu::SurfaceFrame::Recreate => {
                    state.surface.configure(&gpu.device, &state.surface_config);
                    reconfigured = true;
                    None
                }
                crate::gpu::SurfaceFrame::Skip => None,
            };
            if let Some(frame) = acquired {
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
                                label: Some("viewport_app_paint"),
                            });
                    {
                        let mut rp = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                            #[cfg(any(wgpu29, wgpu30))]
                            multiview_mask: None,
                            label: Some("viewport_app_paint_pass"),
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
                        let mut pctx = PaintCtx {
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
                if let Some(marks) = state.startup_marks.take() {
                    marks.finish("first_frame");
                    if let Some(path) = self.config.pipeline_cache_path.as_deref() {
                        super::save_pipeline_cache(path, &state.session);
                    }
                }
            }

            // begin_frame_at rather than begin_frame: the runner already owns a clock,
            // and double tap and long press need one.
            state
                .session
                .begin_frame_at(ViewportContext::new(state.ownership(), [w, h]), time);
            // A loss shows after the submit; the next redraw replaces the device.
            let lost = gpu.lost.is_lost_or_destroyed();
            // A frame shades with the automatic exposure the frame before it
            // metered, so an on-demand window draws one frame more to show it.
            let exposure_follow_up = automatic_exposure && !state.exposure_follow_up;
            state.exposure_follow_up = exposure_follow_up;
            if lost
                || wants_redraw(
                    state.redraw_mode,
                    request_redraw || exposure_follow_up,
                    reconfigured,
                )
            {
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
        // The last window takes the last renderer with it, so save the shared
        // pipeline cache while one is still alive.
        if self.windows.len() == 1 {
            if let (Some(path), Some(state)) = (
                self.config.pipeline_cache_path.as_deref(),
                self.windows.get(&id),
            ) {
                super::save_pipeline_cache(path, &state.session);
            }
        }
        if let Some(state) = self.windows.remove(&id) {
            self.winit_ids.remove(&state.window.id());
        }
        if should_end_loop(self.windows.len(), self.config.exit_on_last_window_close) {
            event_loop.exit();
        }
    }
}

impl ApplicationHandler for AppHandler {
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Any window's renderer returns the shared cache.
        if let (Some(path), Some(state)) = (
            self.config.pipeline_cache_path.as_deref(),
            self.windows.values().next(),
        ) {
            super::save_pipeline_cache(path, &state.session);
        }
    }

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

            // Ownership changes take effect at once, so the events that follow in
            // this batch are routed on it rather than on the last frame's.
            WindowEvent::Focused(focused) => {
                if let Some(state) = self.windows.get_mut(&id) {
                    state.focused = focused;
                    state.session.set_input_ownership(state.ownership());
                    // Losing focus becomes FocusLost, which releases held keys and
                    // buttons: their releases go to whichever window took focus.
                    let mut evs = Vec::new();
                    translate(&event, &AdapterCtx::default(), &mut evs);
                    for ev in evs {
                        if state.input.is_none() {
                            state.session.handle_event(ev.clone());
                        }
                        state.events.push(ev);
                    }
                    state.window.request_redraw();
                }
            }

            WindowEvent::CursorEntered { .. } => {
                if let Some(state) = self.windows.get_mut(&id) {
                    state.hovered = true;
                    state.session.set_input_ownership(state.ownership());
                    state.window.request_redraw();
                }
            }

            WindowEvent::CursorLeft { .. } => {
                if let Some(state) = self.windows.get_mut(&id) {
                    state.hovered = false;
                    state.session.set_input_ownership(state.ownership());
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
                        state.session.set_input_ownership(state.ownership());
                    }
                    // A key press is followed by the text it typed: winit resolves
                    // layout, shift and dead keys into it.
                    let scale = state.window.scale_factor() as f32;
                    let mut evs = Vec::new();
                    translate(&other, &AdapterCtx::new(scale, glam::Vec2::ZERO), &mut evs);
                    if !evs.is_empty() {
                        for ev in evs {
                            if state.input.is_none() {
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

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: ::winit::event::DeviceId,
        event: ::winit::event::DeviceEvent,
    ) {
        // Raw motion is not tied to a window; route it to the focused one, since
        // mouselook targets the window with keyboard focus.
        let mut evs = Vec::new();
        translate_device(&event, &mut evs);
        if evs.is_empty() {
            return;
        }
        if let Some(state) = self.windows.values_mut().find(|s| s.focused) {
            for ev in evs {
                if state.input.is_none() {
                    state.session.handle_event(ev.clone());
                }
                state.events.push(ev);
            }
            state.window.request_redraw();
        }
    }
}

/// The winit cursor for a [`CursorShape`].
pub(crate) fn cursor_icon(shape: CursorShape) -> CursorIcon {
    match shape {
        CursorShape::Default => CursorIcon::Default,
        CursorShape::Pointer => CursorIcon::Pointer,
        CursorShape::Text => CursorIcon::Text,
        CursorShape::Crosshair => CursorIcon::Crosshair,
        CursorShape::Move => CursorIcon::Move,
        CursorShape::NotAllowed => CursorIcon::NotAllowed,
        CursorShape::Grab => CursorIcon::Grab,
        CursorShape::Grabbing => CursorIcon::Grabbing,
        CursorShape::ResizeHorizontal => CursorIcon::EwResize,
        CursorShape::ResizeVertical => CursorIcon::NsResize,
        CursorShape::ResizeNeSw => CursorIcon::NeswResize,
        CursorShape::ResizeNwSe => CursorIcon::NwseResize,
        CursorShape::ResizeColumn => CursorIcon::ColResize,
        CursorShape::ResizeRow => CursorIcon::RowResize,
        // A shape this winit version does not name falls back to the arrow rather than
        // leaving whatever was set before.
        _ => CursorIcon::Default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::input::{ButtonState, MouseButton, ScrollUnits};

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

    fn headless_instance() -> Option<ViewportInstance> {
        let instance = crate::gpu::instance_from_env();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
                #[cfg(wgpu30)]
                apply_limit_buckets: false,
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
        session.begin_frame(ViewportContext::new(
            InputOwnership::full_window(),
            [256.0, 256.0],
        ));

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
                window_id: WindowId(0),
                events: &events,
                dt: 1.0 / 60.0,
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
        session.begin_frame(ViewportContext::new(
            InputOwnership::full_window(),
            [256.0, 256.0],
        ));

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
                window_id: WindowId(0),
                events: &events,
                dt: 1.0 / 60.0,
            };
            // Forward nothing: the app consumed every event.
            let _ = &mut ictx;
        }

        let action = session.resolve();
        assert!(!action.pointer.clicked, "no forwarded click");
        assert_eq!(action.navigation.zoom, 0.0, "no forwarded scroll");
    }

    /// A frame lost to a surface reconfigure is asked for again in every mode;
    /// a transient skip follows the mode.
    #[test]
    fn a_reconfigured_frame_is_retried_on_demand() {
        assert!(wants_redraw(RedrawMode::OnDemand, false, true));
        assert!(!wants_redraw(RedrawMode::OnDemand, false, false));
        assert!(wants_redraw(RedrawMode::OnDemand, true, false));
        assert!(wants_redraw(RedrawMode::Continuous, false, false));
    }
}
