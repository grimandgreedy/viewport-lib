//! Showcase 29: Depth-Composited Render Images
//!
//! Demonstrates `ScreenImageItem::depth` for depth-compositing a
//! CPU-rendered image against 3D scene geometry.
//!
//! The scene has three spheres at clearly different distances from the camera:
//!
//! - **Green (left, near)**: closer than the overlay : pokes through in DC mode.
//! - **Blue (center, mid)**: sits at exactly the overlay depth.
//! - **Orange (right, far)**: farther than the overlay : hidden behind it in DC mode.
//!
//! The overlay is a semi-transparent heatmap gradient (blue -> red) computed at
//! the world-space depth of the center sphere.  Toggle modes to see how the
//! depth test admits the near sphere and occludes the far sphere.

use crate::{App, MeshId};
use crate::eframe::egui;
use viewport_lib as vpl;
use vpl::{
    AnchorX, AnchorY, Camera, LightKind, LightSource, LightingSettings, Material, SceneRenderItem,
    ScreenImageItem, ViewportRenderer,
};

// Overlay image dimensions.
const IMG_W: u32 = 512;
const IMG_H: u32 = 256;

// Sphere radius (shared mesh, different translations).
const SPHERE_RADIUS: f32 = 1.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DcMode {
    /// Regular overlay : drawn on top of all 3D content (no depth test).
    Plain,
    /// Depth-composite : depth-tested against scene geometry.
    DepthComposite,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct DcState {
    pub built: bool,
    pub mesh_id: MeshId,
    pub mode: DcMode,
    pub pixels: Vec<[u8; 4]>,
    pub depths: Vec<f32>,
}

impl Default for DcState {
    fn default() -> Self {
        Self {
            built: false,
            mesh_id: MeshId::INVALID,
            mode: DcMode::DepthComposite,
            pixels: Vec::new(),
            depths: Vec::new(),
        }
    }
}

impl App {
    pub(crate) fn build_dc_scene(&mut self, renderer: &mut ViewportRenderer) {
        // One sphere mesh reused at three positions via the model matrix.
        let mesh = vpl::primitives::sphere(SPHERE_RADIUS, 32, 16);
        self.dc_state.mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &mesh)
            .expect("dc sphere mesh");

        // Tight near/far so NDC depths are meaningfully spread across the
        // visible range.  The default znear=0.01 / zfar=1000 compresses all
        // nearby geometry into a tiny slice near 1.0, making the depth test
        // produce identical-looking results for "near" and "far" overlay depths.
        self.camera = Camera {
            center: glam::Vec3::ZERO,
            distance: 10.0,
            orientation: glam::Quat::from_rotation_x(0.35),
            znear: Some(0.5),
            zfar: 30.0,
            ..Camera::default()
        };

        // Compute the NDC depth for the center (mid) sphere so the overlay
        // lives at exactly that world-space plane.
        //
        // NDC depth formula for wgpu perspective (depth in [0, 1]):
        //   z_ndc = far * (d - near) / (d * (far - near))
        // where d is the view-space depth of the point (positive, along -Z view axis).
        let ndc_overlay = {
            let rot = self.camera.orientation;
            let eye = self.camera.center + rot * (glam::Vec3::Z * self.camera.distance);
            let forward = (self.camera.center - eye).normalize();
            // Mid sphere sits at the origin.
            let view_depth = (glam::Vec3::ZERO - eye).dot(forward);
            let n = self.camera.effective_znear();
            let f = self.camera.zfar;
            (f * (view_depth - n) / (view_depth * (f - n))).clamp(0.0, 1.0)
        };

        // Build the overlay: a cool-to-warm heatmap gradient, all pixels at
        // the same NDC depth (the mid-sphere plane).
        let count = (IMG_W * IMG_H) as usize;
        let mut pixels = vec![[0u8; 4]; count];
        let depths = vec![ndc_overlay; count];

        for y in 0..IMG_H {
            for x in 0..IMG_W {
                let t = x as f32 / (IMG_W - 1) as f32;
                let (r, g, b) = heatmap_rgb(t);
                pixels[(y * IMG_W + x) as usize] = [r, g, b, 210];
            }
        }

        self.dc_state.pixels = pixels;
        self.dc_state.depths = depths;
        self.dc_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_dc(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Mode:");
    ui.radio_value(
        &mut app.dc_state.mode,
        DcMode::Plain,
        "Plain overlay (always on top)",
    );
    ui.radio_value(
        &mut app.dc_state.mode,
        DcMode::DepthComposite,
        "Depth composite (depth-tested)",
    );

    ui.separator();
    ui.label("The heatmap sits at the depth of the blue center sphere.");
    ui.separator();
    ui.label("Green (left)  : closer than overlay : visible through it in DC mode.");
    ui.label("Blue (center) : at the overlay depth.");
    ui.label("Orange (right): farther than overlay : hidden behind it in DC mode.");
}

impl App {
    pub(crate) fn dc_scene_items(&self) -> Vec<SceneRenderItem> {
        if !self.dc_state.built {
            return vec![];
        }

        // Three spheres: same mesh, different translations and colours.
        let spheres = [
            (glam::Vec3::new(-3.5, 0.0, 4.5), [0.10f32, 0.52, 0.18]), // green  - near
            (glam::Vec3::new(0.0, 0.0, 0.0), [0.10, 0.26, 0.68]),     // blue   - mid
            (glam::Vec3::new(3.5, 0.0, -5.0), [0.75, 0.28, 0.05]),    // orange - far
        ];

        spheres
            .iter()
            .map(|(pos, colour)| {
                let mut item = SceneRenderItem::default();
                item.mesh_id = self.dc_state.mesh_id;
                item.model = glam::Mat4::from_translation(*pos).to_cols_array_2d();
                item.material = {
                    let mut m = Material::from_colour(*colour);
                    m.roughness = 0.3;
                    m
                };
                item
            })
            .collect()
    }

    pub(crate) fn dc_push_screen_image(&self, fd: &mut vpl::FrameData) {
        if !self.dc_state.built {
            return;
        }
        let mut img = ScreenImageItem::default();
        img.pixels = self.dc_state.pixels.clone();
        img.width = IMG_W;
        img.height = IMG_H;
        img.anchor_x = AnchorX::Middle;
        img.anchor_y = AnchorY::Middle;
        img.scale = 1.7; // ~870 x 435 screen pixels; covers all three spheres
        img.alpha = 1.0;
        if self.dc_state.mode == DcMode::DepthComposite {
            img.depth = Some(self.dc_state.depths.clone());
        }
        fd.scene.screen_images.push(img);
    }

    pub(crate) fn dc_lighting() -> LightingSettings {
        {
            let mut _t = LightingSettings::default();
            _t.lights = vec![{
                let mut _t = LightSource::default();
                _t.kind = LightKind::Directional {
                    direction: [-0.4, -0.7, -1.0],
                };
                _t.colour = [1.0, 0.97, 0.93].into();
                _t.intensity = 2.2;
                _t
            }];
            _t.hemisphere_intensity = 0.45;
            _t.sky_colour = [0.08, 0.10, 0.18];
            _t.ground_colour = [0.15, 0.15, 0.20];
            _t
        }
    }
}

/// Map `t` in `[0, 1]` to a 5-stop cool-to-warm heatmap colour.
///
/// Stops: deep-blue -> cyan -> green -> yellow -> red.
fn heatmap_rgb(t: f32) -> (u8, u8, u8) {
    // (r, g, b) stops normalised to [0, 1].
    const STOPS: [(f32, f32, f32); 5] = [
        (0.00, 0.00, 0.80), // deep blue
        (0.00, 0.75, 0.85), // cyan
        (0.05, 0.80, 0.05), // green
        (0.90, 0.88, 0.00), // yellow
        (0.88, 0.06, 0.06), // red
    ];
    let seg = (t * 4.0).clamp(0.0, 3.9999);
    let i = seg as usize;
    let f = seg - i as f32;
    let a = STOPS[i];
    let b = STOPS[i + 1];
    let lerp = |a: f32, b: f32| ((a + (b - a) * f).clamp(0.0, 1.0) * 255.0) as u8;
    (lerp(a.0, b.0), lerp(a.1, b.1), lerp(a.2, b.2))
}

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.dc_state.built
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    app.build_dc_scene(renderer);
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
        (app.dc_scene_items(), None, crate::App::dc_lighting(), 0, 0)
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
    app: &mut crate::App,
    fd: &mut vpl::FrameData,
    _ctx: &crate::FrameCtx,
) {
    // Depth-composite screen image (Showcase 29) : submitted every frame.
    app.dc_push_screen_image(&mut *fd);
}

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
