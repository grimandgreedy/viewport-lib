//! The showcase framework: a trait each showcase implements, plus the context
//! objects the host passes in. The host (`main.rs`) owns the window, the
//! `ViewportInstance`, and the offscreen texture; each showcase owns its own
//! scene contents, camera controllers, and interaction logic, so a showcase is
//! self-contained in one file.

use eframe::{egui, wgpu};
use viewport_lib::{ViewportContext, ViewportInstance};

use crate::camera::{CameraRig, MoveKeys};

/// Passed to [`Showcase::setup`]: the session plus the device, for uploading
/// meshes and building the initial scene.
pub struct SetupCtx<'a> {
    pub session: &'a mut ViewportInstance,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
}

/// Passed to [`Showcase::update`] each frame. Read timing and raw key presses
/// here; drive the camera and interaction through `session`.
pub struct ShowcaseCtx<'a> {
    pub session: &'a mut ViewportInstance,
    /// The shared orbit/fly camera, owned by the host and retained across
    /// showcases. Drive it with [`drive_camera`](Self::drive_camera).
    camera: &'a mut CameraRig,
    /// wgpu device and queue, for per-frame GPU work (GPU picking, in-place
    /// mesh/texture updates).
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    /// Physical pixels per logical point. Overlay coordinates are in physical
    /// render-target pixels, while pick cursors are in logical points, so scale
    /// by this to draw an overlay that lines up with the cursor.
    pub pixels_per_point: f32,
    /// Seconds since the previous frame.
    pub dt: f32,
    /// Whether the viewport rect is hovered / focused this frame.
    pub hovered: bool,
    pub focused: bool,
    /// Viewport size in logical points (what `begin_frame` was given).
    pub viewport_size: [f32; 2],
    keys_pressed: &'a [egui::Key],
    keys_down: &'a [egui::Key],
}

impl<'a> ShowcaseCtx<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session: &'a mut ViewportInstance,
        camera: &'a mut CameraRig,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        pixels_per_point: f32,
        dt: f32,
        hovered: bool,
        focused: bool,
        viewport_size: [f32; 2],
        keys_pressed: &'a [egui::Key],
        keys_down: &'a [egui::Key],
    ) -> Self {
        Self {
            session,
            camera,
            device,
            queue,
            pixels_per_point,
            dt,
            hovered,
            focused,
            viewport_size,
            keys_pressed,
            keys_down,
        }
    }

    /// True if `key` transitioned to pressed this frame (ignores auto-repeat).
    pub fn key_pressed(&self, key: egui::Key) -> bool {
        self.keys_pressed.contains(&key)
    }

    /// True while `key` is held down (for continuous movement input).
    pub fn key_down(&self, key: egui::Key) -> bool {
        self.keys_down.contains(&key)
    }

    /// Run the shared orbit/fly camera and assemble the frame. Call this once per
    /// `update`, after any click-selection and before applying manipulation, so
    /// the frame is assembled from the moved camera. Handles the backtick toggle
    /// and WASD/arrow movement internally.
    pub fn drive_camera(&mut self) {
        let toggle = self.key_pressed(egui::Key::Backtick);
        let move_keys = MoveKeys {
            forward: self.key_down(egui::Key::ArrowUp),
            back: self.key_down(egui::Key::ArrowDown),
            left: self.key_down(egui::Key::ArrowLeft),
            right: self.key_down(egui::Key::ArrowRight),
        };
        let view_ctx = ViewportContext {
            hovered: self.hovered,
            focused: self.focused,
            viewport_size: self.viewport_size,
        };
        self.camera
            .drive(&mut *self.session, self.dt, view_ctx, toggle, move_keys);
    }
}

/// One self-contained demo. The host drives it: `setup` once when it becomes
/// active, `update` every frame before rendering, `ui` for an optional controls
/// panel.
pub trait Showcase {
    /// Short label for the selector.
    fn name(&self) -> &str;

    /// Build the scene. Meshes uploaded here persist until the next switch.
    fn setup(&mut self, ctx: &mut SetupCtx);

    /// Drive the camera and interaction, and leave the session assembled: call
    /// one of `session.update_orbit(...)` / `frame(...)` so the host can render
    /// the frame right after.
    fn update(&mut self, ctx: &mut ShowcaseCtx);

    /// One or two sentences on what this showcase demonstrates. Shown in the
    /// info box over the top-left corner of the viewport.
    fn description(&self) -> &str {
        ""
    }

    /// Showcase-specific controls (e.g. G/R/S manipulation), shown in the `?`
    /// modal below the general camera controls. Default: nothing.
    fn controls(&mut self, _ui: &mut egui::Ui) {}

    /// Whether this showcase draws a live controls panel. When true the host
    /// shows a right-side panel and calls [`panel`](Self::panel). Default: false.
    fn has_controls(&self) -> bool {
        false
    }

    /// Live controls (sliders, toggles, pickers) drawn in the right-side panel.
    /// Called before `update` each frame, so `update` sees the new state.
    fn panel(&mut self, _ui: &mut egui::Ui) {}

    /// Optional controls drawn over the top-centre of the viewport, e.g. a mode
    /// chip. Default: nothing.
    fn top_overlay(&mut self, _ui: &mut egui::Ui) {}
}

/// Reset the shared session between showcases: drop all scene nodes, clear the
/// selection, and clear retained extras. The session, renderer, and camera are
/// reused; the next showcase's `setup` rebuilds the scene.
pub fn reset_session(session: &mut ViewportInstance) {
    let ids: Vec<_> = session.scene().nodes().map(|n| n.id()).collect();
    session.scene_mut().remove_many(&ids);
    session.selection_mut().clear();
    session.clear_extras();
    // Persistent viewport chrome (grid, background, wireframe) and clip objects
    // are retained on the session, so reset them or one showcase's settings leak
    // into the next. Each showcase re-sets what it needs in `setup`.
    *session.viewport_frame_mut() = viewport_lib::ViewportFrame::default();
    session.effects_mut().clip_objects.clear();
}
