//! Showcase 21: Image Textures.
//!
//! Demonstrates UV-mapped image textures uploaded via `upload_texture`.
//!
//! Layout (Z-up, 2x2 grid in X-Z plane at Y=0):
//!   Back-left   (-3, 0, +3):  Plane (two-sided) : Percy photo
//!   Back-right  (+3, 0, +3):  UV sphere : procedural checkerboard
//!   Front-left  (-3, 0, -3):  Cube : procedural colour-gradient per face
//!   Front-right (+3, 0, -3):  Torus : procedural stripe pattern
//!
//! The Percy photo is stored as pre-converted raw RGBA bytes alongside this
//! file, so no image-parsing dependency or build script is required.

use crate::App;
use crate::eframe::egui;
use viewport_lib as vpl;
use vpl::{Material, NodeId, ViewportRenderer, scene::Scene};

// Percy photo : pre-converted to raw RGBA (2203 x 2009).
const PERCY_WIDTH: u32 = 2203;
const PERCY_HEIGHT: u32 = 2009;
const PERCY_RGBA: &[u8] = include_bytes!("percy.rgba");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct TextureState {
    pub scene: Scene,
    pub built: bool,
    pub plane_node: NodeId,
}

impl Default for TextureState {
    fn default() -> Self {
        Self {
            scene: Scene::new(),
            built: false,
            plane_node: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_texture_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.texture_state.scene = Scene::new();

        let res = renderer.resources_mut();

        // --- Percy photo on a plane ---
        let percy_tex = res
            .upload_texture(
                &self.device,
                &self.queue,
                PERCY_WIDTH,
                PERCY_HEIGHT,
                PERCY_RGBA,
            )
            .expect("percy texture upload");

        let plane =
            vpl::geometry::primitives::plane(4.5, 4.5 * PERCY_HEIGHT as f32 / PERCY_WIDTH as f32);
        let plane_id = res
            .upload_mesh_data(&self.device, &plane)
            .expect("plane mesh upload");

        self.texture_state.plane_node = self.texture_state.scene.add_named(
            "Percy Plane",
            Some(plane_id),
            glam::Mat4::from_translation(glam::Vec3::new(-3.0, 0.0, 3.0))
                * glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2 * 3.0),
            {
                let mut m = Material::default();
                m.texture_id = Some(percy_tex);
                m.ambient = 0.9;
                m.diffuse = 0.4;
                m
            },
        );
        // (two-sided set in build_frame_data via texture_state.plane_node)

        // --- Checkerboard on a sphere ---
        let checker = make_checkerboard(256, 8, [220, 220, 220, 255], [40, 40, 40, 255]);
        let checker_tex = res
            .upload_texture(&self.device, &self.queue, 256, 256, &checker)
            .expect("checker texture upload");

        let sphere = vpl::geometry::primitives::sphere(1.8, 48, 24);
        let sphere_id = res
            .upload_mesh_data(&self.device, &sphere)
            .expect("sphere mesh upload");

        self.texture_state.scene.add_named(
            "Checker Sphere",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(3.0, 0.0, 3.0)),
            {
                let mut m = Material::default();
                m.texture_id = Some(checker_tex);
                m.ambient = 0.3;
                m.diffuse = 0.8;
                m.specular = 0.4;
                m.shininess = 32.0;
                m
            },
        );

        // --- Colour-gradient on a cube ---
        let gradient = make_gradient(256);
        let gradient_tex = res
            .upload_texture(&self.device, &self.queue, 256, 256, &gradient)
            .expect("gradient texture upload");

        let cube = vpl::geometry::primitives::cube(3.0);
        let cube_id = res
            .upload_mesh_data(&self.device, &cube)
            .expect("cube mesh upload");

        self.texture_state.scene.add_named(
            "Gradient Cube",
            Some(cube_id),
            glam::Mat4::from_translation(glam::Vec3::new(-3.0, 0.0, -3.0)),
            {
                let mut m = Material::default();
                m.texture_id = Some(gradient_tex);
                m.ambient = 0.3;
                m.diffuse = 0.8;
                m
            },
        );

        // --- Stripe pattern on a torus ---
        let stripes = make_stripes(256, 16, [180, 100, 30, 255], [230, 200, 140, 255]);
        let stripes_tex = res
            .upload_texture(&self.device, &self.queue, 256, 256, &stripes)
            .expect("stripes texture upload");

        let torus = vpl::geometry::primitives::torus(1.5, 0.5, 48, 24);
        let torus_id = res
            .upload_mesh_data(&self.device, &torus)
            .expect("torus mesh upload");

        self.texture_state.scene.add_named(
            "Stripe Torus",
            Some(torus_id),
            glam::Mat4::from_translation(glam::Vec3::new(3.0, 0.0, -3.0))
                * glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2 * 3.0),
            {
                let mut m = Material::default();
                m.texture_id = Some(stripes_tex);
                m.ambient = 0.3;
                m.diffuse = 0.8;
                m.specular = 0.6;
                m.shininess = 64.0;
                m
            },
        );

        self.texture_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_textures(_app: &mut App, ui: &mut egui::Ui) {
    ui.label("UV-mapped image textures on four primitives:");
    ui.label("  Back-left:  plane : Percy photo");
    ui.label("  Left:    sphere : procedural checkerboard");
    ui.label("  Right:   cube : procedural colour gradient");
    ui.label("  Front:   torus : procedural stripes");
    ui.separator();
    ui.label("Textures are uploaded as raw RGBA via upload_texture().");
    ui.label("The photo is stored as pre-converted raw RGBA bytes.");
}

// ---------------------------------------------------------------------------
// Procedural texture generators
// ---------------------------------------------------------------------------

/// Checkerboard: `cells` x `cells` grid alternating between `a` and `b`.
fn make_checkerboard(size: usize, cells: usize, a: [u8; 4], b: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(size * size * 4);
    for row in 0..size {
        for col in 0..size {
            let cx = col * cells / size;
            let cy = row * cells / size;
            let colour = if (cx + cy) % 2 == 0 { a } else { b };
            out.extend_from_slice(&colour);
        }
    }
    out
}

/// Smooth HSV hue gradient scrolling across both axes.
fn make_gradient(size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size * size * 4);
    for row in 0..size {
        for col in 0..size {
            let h = (col as f32 / size as f32 + row as f32 / size as f32 * 0.3).fract();
            let (r, g, b) = hsv_to_rgb(h, 0.75, 0.92);
            out.push((r * 255.0) as u8);
            out.push((g * 255.0) as u8);
            out.push((b * 255.0) as u8);
            out.push(255);
        }
    }
    out
}

/// Horizontal stripes alternating between `a` and `b`.
fn make_stripes(size: usize, count: usize, a: [u8; 4], b: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(size * size * 4);
    for row in 0..size {
        let band = row * count / size;
        let colour = if band % 2 == 0 { a } else { b };
        for _ in 0..size {
            out.extend_from_slice(&colour);
        }
    }
    out
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h6 = h * 6.0;
    let i = h6.floor() as u32 % 6;
    let f = h6 - h6.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.texture_state.built
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    app.build_texture_scene(renderer);
}

// ---------------------------------------------------------------------------
// Per-frame scene contents
// ---------------------------------------------------------------------------

/// Collect this showcase's render items and lighting for the frame. `_out` carries
/// the few extra frame settings a showcase can set alongside its items.
pub(crate) fn scene(
    app: &mut crate::App,
    _frame: &crate::eframe::Frame,
    _out: &mut crate::SceneOverrides,
) -> crate::SceneContents {
    let (items, bg_colour, lighting, scene_gen, sel_gen) = {
        let mut items = app
            .texture_state
            .scene
            .collect_render_items(&vpl::Selection::new());
        let plane_node = app.texture_state.plane_node;
        if let Some(item) = items
            .iter_mut()
            .find(|i| i.settings.pick_id == vpl::PickId(plane_node))
        {
            item.material.backface_policy = vpl::BackfacePolicy::Identical;
        }
        let sg = app.texture_state.scene.version();
        let lighting = {
            let mut _t = vpl::LightingSettings::default();
            _t.hemisphere_intensity = 0.5;
            _t.sky_colour = [1.0, 1.0, 1.0];
            _t.ground_colour = [1.0, 1.0, 1.0];
            _t
        };
        (items, None, lighting, sg, 0)
    };
    crate::SceneContents {
        items,
        bg_colour,
        lighting,
        scene_gen,
        sel_gen,
    }
}

// ---------------------------------------------------------------------------
// Per-frame frame-data tweaks
// ---------------------------------------------------------------------------

/// Fold this showcase's own contributions into the assembled frame: extra
/// render items, overlays, and effect settings that are re-submitted every
/// frame rather than baked into the scene.
pub(crate) fn frame(
    _app: &mut crate::App,
    _fd: &mut vpl::FrameData,
    _ctx: &crate::FrameCtx,
) {}

// ---------------------------------------------------------------------------
// Viewport overlay and per-frame tick
// ---------------------------------------------------------------------------

/// Draw this showcase's own egui overlay on top of the rendered viewport:
/// selection rectangles, mode readouts, and in-scene labels.
pub(crate) fn overlay(_app: &mut crate::App, _ui: &mut crate::eframe::egui::Ui, _cx: &crate::ViewportCtx) {}

/// Advance this showcase's animation and ask for another frame. Runs after the
/// viewport has been drawn, so it only affects the next frame.
pub(crate) fn tick(_app: &mut crate::App, _cx: &crate::ViewportCtx) {}

/// Route a viewport click for this showcase. The host calls this for a plain
/// click that no gizmo or widget has already consumed; `pos` is in viewport
/// pixels.
pub(crate) fn on_click(_app: &mut crate::App, _cx: &crate::ClickCtx) {}

/// Handle drag gestures this showcase owns, before the camera controller runs.
pub(crate) fn drag_input(_app: &mut crate::App, _cx: &crate::ViewportCtx) {}

/// Advance this showcase's own camera animation or object motion for the frame.
pub(crate) fn advance(_app: &mut crate::App, _cx: &crate::ViewportCtx) {}

/// Update this showcase's interactive widgets for the frame.
pub(crate) fn widgets(_app: &mut crate::App, _cx: &crate::ViewportCtx) {}

/// Flush any per-frame GPU writes this showcase has queued.
pub(crate) fn flush_gpu(_app: &mut crate::App, _cx: &crate::ViewportCtx) {}

/// Cache gizmo placement for next frame's hit-testing.
pub(crate) fn cache_gizmo(_app: &mut crate::App, _cx: &crate::ViewportCtx) {}

/// Take over the whole viewport for this frame. Returning false leaves the
/// host's normal single-viewport path in charge.
pub(crate) fn viewport_override(
    _app: &mut crate::App,
    _ui: &mut crate::eframe::egui::Ui,
    _cx: &crate::ViewportCtx,
) -> bool {
    false
}
