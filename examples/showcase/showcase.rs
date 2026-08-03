//! The showcase framework: a trait each showcase implements, plus the context
//! objects the host passes in. The host (`main.rs`) owns the window, the
//! `ViewportSession`, and the offscreen texture; each showcase owns its own
//! scene contents, camera controllers, and interaction logic, so a showcase is
//! self-contained in one file.

use eframe::{egui, wgpu};
use viewport_lib::ViewportSession;

/// Passed to [`Showcase::setup`]: the session plus the device, for uploading
/// meshes and building the initial scene.
pub struct SetupCtx<'a> {
    pub session: &'a mut ViewportSession,
    pub device: &'a wgpu::Device,
}

/// Passed to [`Showcase::update`] each frame. Read timing and raw key presses
/// here; drive the camera and interaction through `session`.
pub struct ShowcaseCtx<'a> {
    pub session: &'a mut ViewportSession,
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
    pub fn new(
        session: &'a mut ViewportSession,
        dt: f32,
        hovered: bool,
        focused: bool,
        viewport_size: [f32; 2],
        keys_pressed: &'a [egui::Key],
        keys_down: &'a [egui::Key],
    ) -> Self {
        Self {
            session,
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

    /// Controls / key bindings, shown in the `?` modal. Default: nothing.
    fn controls(&mut self, _ui: &mut egui::Ui) {}

    /// Optional controls drawn over the top-right corner of the viewport, e.g. a
    /// camera-mode toggle. Default: nothing.
    fn viewport_overlay(&mut self, _ui: &mut egui::Ui) {}
}

/// Reset the shared session between showcases: drop all scene nodes, clear the
/// selection, and clear retained extras. The session, renderer, and camera are
/// reused; the next showcase's `setup` rebuilds the scene.
pub fn reset_session(session: &mut ViewportSession) {
    let ids: Vec<_> = session.scene().nodes().map(|n| n.id()).collect();
    session.scene_mut().remove_many(&ids);
    session.selection_mut().clear();
    session.clear_extras();
}
