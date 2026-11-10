//! Showcase 35: Overlay Composition.
//!
//! Screen overlays drawn over a point cloud with world-anchored callout
//! labels. The rows are built in `rows.rs`, on two sheets the controls choose
//! between (or stack):
//!
//! Sheet 1, shapes, fills and strokes:
//! 1. Solid shapes: every closed analytic shape, then the inset, outer and
//!    centred border modes.
//! 2. More shapes: lines with each `LineCap`, stars, regular polygons, crosses.
//! 3. Textures: images on shapes, a texture spinning inside a still circle, a
//!    tiled texture, and texture fills on a vector path and closed polylines.
//! 4. Gradients: two-stop linear, radial and conical, four- and eight-stop
//!    ramps, a sweep gauge, a ring, and a registered WGSL hatch fill.
//! 5. Shadows and strokes: drop shadow, glow, inner shadow, a dashed vector
//!    stroke, a hairline grid under a 3x group scale, a per-point-width brush,
//!    an inverse fill, and dashed and dotted polylines.
//! 6. Transforms: rotation, skew, stretch, a whole 2x2 matrix, a rotation
//!    pivot (metronome), a flipped texture, and a skewed label.
//! 7. Animation: translate, scale, tint, rotation and opacity tracks, a
//!    repeating fade, and a dot on a polyline-traced Bezier path.
//!
//! Sheet 2, compositing, text and meshes:
//! 8. Masks and polylines: a soft rotated mask clip, nine-slice against a
//!    plain stretch, a gradient-filled closed polyline, a per-frame blob, and
//!    1 and 1.5 px strokes beside 4x copies.
//! 9. Blend modes: additive, multiply and screen; difference, overlay and hue
//!    (destination-reading); destination-out inside a layer.
//! 10. Layers: a faded group beside the same items faded singly, a group
//!     shadow, a frosted backdrop-blur card and a filtered backdrop, a blurred
//!     group, and a greyscale colour-matrix twin.
//! 11. Text: quarter-pixel pen positions, a scaled label re-rasterised at its
//!     drawn size, a texture-filled label, a tinted glyph run, colour emoji.
//! 12. Meshes: a heatmap as one mesh beside the same cells as shapes.
//!
//! Both sheets end with a strip of the previous frame's overlay counters.

mod rows;

pub(crate) use rows::{HeatmapMode, Sheet};

use crate::App;
use crate::eframe::egui;
use viewport_lib as vpl;
use viewport_lib_plugins::item_types::point_cloud::PointCloudItem;
use vpl::BuiltinColourmap;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct OvlState {
    pub start_time: std::time::Instant,
    pub colourmap: vpl::BuiltinColourmap,
    pub sheet: Sheet,
    pub show_labels: bool,
    pub show_emoji: bool,
    pub shape_corner_radius: f32,
    pub shape_border_width: f32,
    pub backdrop_blur_radius: f32,
    /// Which halves of the heatmap row draw: the mesh, the shapes, or both.
    pub heatmap: HeatmapMode,
    pub assets: rows::PageAssets,
    pub cloud_positions: Vec<[f32; 3]>,
    pub cloud_scalars: Vec<f32>,
    pub cloud_built: bool,
    /// The renderer's statistics for the previous frame, for the stats strip.
    pub last_stats: vpl::FrameStats,
}

impl Default for OvlState {
    fn default() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            colourmap: vpl::BuiltinColourmap::Viridis,
            sheet: Sheet::Shapes,
            show_labels: true,
            show_emoji: true,
            shape_corner_radius: 8.0,
            shape_border_width: 1.5,
            backdrop_blur_radius: 10.0,
            heatmap: HeatmapMode::Both,
            assets: rows::PageAssets::default(),
            cloud_positions: Vec::new(),
            cloud_scalars: Vec::new(),
            cloud_built: false,
            last_stats: vpl::FrameStats::default(),
        }
    }
}

/// System colour-emoji fonts to try, in order. The first one found is used.
pub(crate) const EMOJI_FONT_PATHS: &[&str] = &[
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf",
];

/// Resolve a handful of emoji to glyph ids in `font_bytes`, dropping any the
/// font does not cover. Empty if the bytes do not parse.
pub(crate) fn emoji_glyph_ids(font_bytes: &[u8]) -> Vec<u16> {
    let Ok(face) = ttf_parser::Face::parse(font_bytes, 0) else {
        return Vec::new();
    };
    [
        '\u{1F600}',
        '\u{1F60D}',
        '\u{1F44D}',
        '\u{1F389}',
        '\u{1F680}',
        '\u{1F30D}',
        '\u{1F525}',
        '\u{2B50}',
        '\u{1F355}',
        '\u{1F3A8}',
    ]
    .iter()
    .filter_map(|&c| face.glyph_index(c).map(|g| g.0))
    .collect()
}

const ALL_COLOURMAPS: &[(BuiltinColourmap, &str)] = &[
    (BuiltinColourmap::Viridis, "Viridis"),
    (BuiltinColourmap::Plasma, "Plasma"),
    (BuiltinColourmap::Turbo, "Turbo"),
    (BuiltinColourmap::Inferno, "Inferno"),
    (BuiltinColourmap::Magma, "Magma"),
    (BuiltinColourmap::Greyscale, "Greyscale"),
    (BuiltinColourmap::Coolwarm, "Coolwarm"),
    (BuiltinColourmap::Rainbow, "Rainbow"),
    (BuiltinColourmap::Jet, "Jet"),
    (BuiltinColourmap::RdBu, "RdBu"),
];

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_overlay(app: &mut App, ui: &mut egui::Ui, _frame: &crate::eframe::Frame) {
    let st = &mut app.ovl_state;
    ui.label("Sheet:");
    ui.radio_value(&mut st.sheet, Sheet::Shapes, "Shapes, fills and strokes");
    ui.radio_value(
        &mut st.sheet,
        Sheet::Compositing,
        "Compositing, text and meshes",
    );
    ui.radio_value(&mut st.sheet, Sheet::Both, "Both (tall window)");

    ui.separator();
    ui.add(egui::Slider::new(&mut st.shape_corner_radius, 0.0..=30.0).text("Corner radius"));
    ui.add(egui::Slider::new(&mut st.shape_border_width, 0.0..=6.0).text("Border width"));
    ui.add(egui::Slider::new(&mut st.backdrop_blur_radius, 0.0..=30.0).text("Backdrop blur"));

    ui.separator();
    ui.label("Heatmap row:");
    ui.horizontal(|ui| {
        ui.radio_value(&mut st.heatmap, HeatmapMode::Both, "Both");
        ui.radio_value(&mut st.heatmap, HeatmapMode::Mesh, "One mesh");
        ui.radio_value(&mut st.heatmap, HeatmapMode::Shapes, "Shapes");
    });
    ui.label(egui::RichText::new("The stats strip shows the draw calls for each.").weak());

    ui.separator();
    ui.checkbox(&mut st.show_labels, "Callout labels");
    ui.add_enabled(
        st.assets.emoji_font.is_some(),
        egui::Checkbox::new(&mut st.show_emoji, "Colour emoji"),
    );
    if st.assets.emoji_font.is_none() {
        ui.label(egui::RichText::new("No system colour-emoji font found.").weak());
    }

    ui.separator();
    ui.label("Point cloud colourmap:");
    let selected_name = ALL_COLOURMAPS
        .iter()
        .find(|(c, _)| *c == st.colourmap)
        .map(|(_, n)| *n)
        .unwrap_or("Unknown");
    egui::ComboBox::from_id_salt("ovl_colourmap")
        .selected_text(selected_name)
        .show_ui(ui, |ui| {
            for (cmap, name) in ALL_COLOURMAPS {
                ui.selectable_value(&mut st.colourmap, *cmap, *name);
            }
        });
}

// ---------------------------------------------------------------------------
// Point cloud geometry (built once)
// ---------------------------------------------------------------------------

/// A sinusoidal wave surface sampled on a 70 x 70 grid: x in [-pi/2, pi/2],
/// y in [0, pi], z = 1.5 sin(x). The scalar is z, matching the range
/// [-1.5, 1.5].
pub(crate) fn build_ovl_cloud() -> (Vec<[f32; 3]>, Vec<f32>) {
    use std::f32::consts::PI;
    let (nx, ny) = (70usize, 70usize);
    let mut positions = Vec::with_capacity(nx * ny);
    let mut scalars = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        let y = iy as f32 / (ny - 1) as f32 * PI;
        for ix in 0..nx {
            let x = (ix as f32 / (nx - 1) as f32 - 0.5) * PI;
            let z = x.sin() * 1.5;
            positions.push([x, y, z]);
            scalars.push(z);
        }
    }
    (positions, scalars)
}

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.ovl_state.cloud_built
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    let (positions, scalars) = build_ovl_cloud();
    app.ovl_state.cloud_positions = positions;
    app.ovl_state.cloud_scalars = scalars;
    app.ovl_state.cloud_built = true;
    let mut assets = rows::build_assets(renderer, &app.device, &app.queue);
    if let Some(bytes) = EMOJI_FONT_PATHS.iter().find_map(|p| std::fs::read(p).ok()) {
        if let Ok(handle) = renderer.resources_mut().upload_font(&bytes) {
            let ids = emoji_glyph_ids(&bytes);
            if !ids.is_empty() {
                assets.emoji_font = Some(handle);
                assets.emoji_glyphs = ids;
            }
        }
    }
    app.ovl_state.assets = assets;
    app.camera = vpl::Camera::orbit(
        glam::DVec3::new(0.0, 1.57, 0.0),
        9.0,
        glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.0),
    );
}

// ---------------------------------------------------------------------------
// Per-frame scene contents
// ---------------------------------------------------------------------------

/// Collect this showcase's render items and lighting for the frame. The point
/// cloud is added in [`frame`], so the scene itself is empty.
pub(crate) fn scene(
    _app: &mut crate::App,
    _frame: &crate::eframe::Frame,
    _out: &mut crate::SceneOverrides,
) -> crate::SceneContents {
    crate::SceneContents {
        items: Vec::new(),
        bg_colour: None,
        lighting: vpl::LightingSettings::default(),
        scene_gen: 0,
        sel_gen: 0,
    }
}

// ---------------------------------------------------------------------------
// Per-frame frame-data tweaks
// ---------------------------------------------------------------------------

/// Fold this showcase's overlays and point cloud into the assembled frame.
pub(crate) fn frame(app: &mut crate::App, fd: &mut vpl::FrameData, ctx: &crate::FrameCtx) {
    if let Some(rs) = ctx.frame.wgpu_render_state() {
        let guard = rs.renderer.read();
        if let Some(renderer) = guard.callback_resources.get::<vpl::ViewportRenderer>() {
            app.ovl_state.last_stats = renderer.last_frame_stats();
        }
    }
    let st = &app.ovl_state;
    fd.overlays = rows::build_page(&rows::PageInput {
        t: st.start_time.elapsed().as_secs_f32(),
        sheet: st.sheet,
        corner_radius: st.shape_corner_radius,
        border_width: st.shape_border_width,
        backdrop_blur: st.backdrop_blur_radius,
        heatmap: st.heatmap,
        show_labels: st.show_labels,
        show_emoji: st.show_emoji,
        assets: &st.assets,
        stats: st.last_stats,
    });
    // Layers, backdrops and destination-reading blends draw through scratch
    // targets and copies of what the pass has drawn, which needs the renderer
    // to own the encoder: the HDR callback path does, the LDR paint path does
    // not.
    fd.effects.display.mode = vpl::PipelineMode::Hdr;
    if st.cloud_built {
        let mut pc = PointCloudItem::default();
        pc.positions = st.cloud_positions.clone();
        pc.colour = vpl::ColourSource::Scalar {
            values: st.cloud_scalars.clone(),
            range: Some((-1.5, 1.5)),
            colourmap: Some(vpl::ColourmapId(st.colourmap as usize)),
        };
        pc.size = vpl::SizeSource::Uniform(4.0);
        fd.scene.items_mut::<PointCloudItem>().push(pc);
    }
}

// ---------------------------------------------------------------------------
// Per-frame tick
// ---------------------------------------------------------------------------

/// Keep repainting so the overlay clock and its animation tracks advance
/// without user input.
pub(crate) fn tick(_app: &mut crate::App, cx: &crate::ViewportCtx) {
    cx.egui.request_repaint();
}

/// Take over the whole viewport for this frame. Returning false leaves the
/// host's normal single-viewport path in charge.
pub(crate) fn viewport_override(
    _app: &mut crate::App,
    _ui: &mut crate::eframe::egui::Ui,
    _cx: &crate::ViewportCtx,
) -> bool {
    false
}

/// Drive the orbit controller for this showcase. Returning false leaves the
/// host to run the usual suppress-or-apply path.
pub(crate) fn drive_camera(_app: &mut crate::App, _cx: &crate::ViewportCtx) -> bool {
    false
}

/// Whether the orbit controller should resolve without moving the camera this
/// frame. This showcase never suppresses it.
pub(crate) fn suppress_orbit(_app: &crate::App, _cx: &crate::ViewportCtx) -> bool {
    false
}

// ---------------------------------------------------------------------------
// Showcase entry point
// ---------------------------------------------------------------------------

/// Stateless handle for this showcase; the scene state lives on [`crate::App`].
pub(crate) struct ScOverlay;

/// The registry's handle to this showcase.
pub(crate) static SHOWCASE: ScOverlay = ScOverlay;

impl crate::Showcase for ScOverlay {
    fn needs_build(&self, app: &crate::App) -> bool {
        needs_build(app)
    }
    fn build(&self, app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
        build(app, renderer)
    }
    fn scene(
        &self,
        app: &mut crate::App,
        frame: &crate::eframe::Frame,
        out: &mut crate::SceneOverrides,
    ) -> crate::SceneContents {
        scene(app, frame, out)
    }
    fn frame(&self, app: &mut crate::App, fd: &mut vpl::FrameData, ctx: &crate::FrameCtx) {
        frame(app, fd, ctx)
    }
    fn tick(&self, app: &mut crate::App, cx: &crate::ViewportCtx) {
        tick(app, cx)
    }
    fn viewport_override(
        &self,
        app: &mut crate::App,
        ui: &mut crate::eframe::egui::Ui,
        cx: &crate::ViewportCtx,
    ) -> bool {
        viewport_override(app, ui, cx)
    }
    fn drive_camera(&self, app: &mut crate::App, cx: &crate::ViewportCtx) -> bool {
        drive_camera(app, cx)
    }
    fn suppress_orbit(&self, app: &crate::App, cx: &crate::ViewportCtx) -> bool {
        suppress_orbit(app, cx)
    }
    fn controls(
        &self,
        app: &mut crate::App,
        ui: &mut crate::eframe::egui::Ui,
        frame: &crate::eframe::Frame,
    ) {
        controls_overlay(app, ui, frame)
    }
}
