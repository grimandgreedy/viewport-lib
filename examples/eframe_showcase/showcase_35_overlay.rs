// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct OvlState {
    pub start_time: std::time::Instant,
    pub colourmap: viewport_lib::BuiltinColourmap,
    pub bar_orientation: viewport_lib::ScalarBarOrientation,
    pub bar_anchor: viewport_lib::ScalarBarAnchor,
    pub tick_count: u32,
    pub bar_size: f32,
    pub bg_colour: [f32; 4],
    pub show_ruler: bool,
    pub show_labels: bool,
    pub show_shapes: bool,
    pub shape_corner_radius: f32,
    pub shape_border_width: f32,
    pub backdrop_blur_radius: f32,
    pub show_tex_shapes: bool,
    pub tex_id: Option<viewport_lib::OverlayTextureId>,
    pub carlgauss_tex_id: Option<viewport_lib::OverlayTextureId>,
    pub nine_slice_tex_id: Option<viewport_lib::OverlayTextureId>,
    pub cloud_positions: Vec<[f32; 3]>,
    pub cloud_scalars: Vec<f32>,
    pub cloud_built: bool,
}

impl Default for OvlState {
    fn default() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            colourmap: viewport_lib::BuiltinColourmap::Viridis,
            bar_orientation: viewport_lib::ScalarBarOrientation::Vertical,
            bar_anchor: viewport_lib::ScalarBarAnchor::BottomRight,
            tick_count: 5,
            bar_size: 200.0,
            bg_colour: [0.0, 0.0, 0.0, 0.63],
            show_ruler: true,
            show_labels: true,
            show_shapes: true,
            shape_corner_radius: 8.0,
            shape_border_width: 1.5,
            backdrop_blur_radius: 14.0,
            show_tex_shapes: true,
            tex_id: None,
            carlgauss_tex_id: None,
            nine_slice_tex_id: None,
            cloud_positions: Vec::new(),
            cloud_scalars: Vec::new(),
            cloud_built: false,
        }
    }
}

// Showcase 35: Overlay Composition : ScalarBarItem + RulerItem + LabelItem together.

use crate::App;
use eframe::egui;
use viewport_lib::{
    BorderMode, BuiltinColourmap, ColourmapId, LabelAnchor, LabelItem, LineCap, OverlayAnimation,
    OverlayFill, OverlayShape, OverlayShapeItem, RulerItem, ScalarBarAnchor, ScalarBarItem,
    ScalarBarOrientation, TriangleDirection,
};

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
// Demo texture generation
// ---------------------------------------------------------------------------

// Carl Gauss portrait: pre-converted raw RGBA (1500 x 1000).
pub(crate) const CARLGAUSS_WIDTH: u32 = 1500;
pub(crate) const CARLGAUSS_HEIGHT: u32 = 1000;
pub(crate) const CARLGAUSS_RGBA: &[u8] = include_bytes!("carlgauss.rgba");

/// Generate a 128x128 RGBA8 image for the texture masking demo.
///
/// The image is a smooth colour-wheel gradient: hue rotates with angle and
/// brightness increases from centre to edge, producing a ring-like look that
/// shows up well under all SDF shapes.
pub(crate) fn build_demo_texture() -> (u32, u32, Vec<u8>) {
    let size: u32 = 128;
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            // Normalised coords in [-1, 1].
            let nx = (x as f32 / (size - 1) as f32) * 2.0 - 1.0;
            let ny = (y as f32 / (size - 1) as f32) * 2.0 - 1.0;

            // Hue from angle (0..1).
            let angle = ny.atan2(nx); // -pi..pi
            let hue = (angle / (2.0 * std::f32::consts::PI) + 0.5).fract();

            // Saturation ramps up from centre.
            let r = (nx * nx + ny * ny).sqrt().min(1.0);
            let sat = r;

            // HSV to RGB (value = 1).
            let [red, green, blue] = hsv_to_rgb(hue, sat, 1.0);

            pixels.push((red * 255.0) as u8);
            pixels.push((green * 255.0) as u8);
            pixels.push((blue * 255.0) as u8);
            pixels.push(255u8); // fully opaque
        }
    }
    (size, size, pixels)
}

/// Procedural 64x32 button texture for the 9-slice demo. Looks like a
/// shipping UI button asset: rounded corners with a 10px radius, a vertical
/// teal-blue gradient body, a thin bright bevel highlight along the top
/// edge, and a darker rim hugging the rounded outline. Pixels outside the
/// rounded shape are fully transparent so the corner shape is visible
/// against any backdrop.
///
/// At source size the asset reads as one button. Stretched to a much wider
/// rect without 9-slice, the rounded corners obviously elongate into ovals
/// and the rim and bevel smear. With 9-slice on, the corners stay 10px in
/// both directions and only the centre body stretches.
pub(crate) fn build_nine_slice_texture() -> (u32, u32, Vec<u8>) {
    let w: u32 = 64;
    let h: u32 = 32;
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    let radius = 10.0_f32;
    let rim = 2.0_f32;
    let body_top = [0.30_f32, 0.62, 0.90];
    let body_bottom = [0.10_f32, 0.25, 0.55];
    let rim_col = [0.04_f32, 0.08, 0.18];
    let bevel = [0.80_f32, 0.92, 1.0];

    // Signed distance to the rounded-rect shape (negative inside).
    //   q = abs(p) - half_size + r
    //   d = min(max(qx, qy), 0) + length(max(q, 0)) - r
    let sd_rounded = |fx: f32, fy: f32| -> f32 {
        let half_w = w as f32 * 0.5;
        let half_h = h as f32 * 0.5;
        let qx = (fx - half_w).abs() - half_w + radius;
        let qy = (fy - half_h).abs() - half_h + radius;
        let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
        let inside = qx.max(qy).min(0.0);
        inside + outside - radius
    };

    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;
            let d = sd_rounded(fx, fy);
            let i = ((y * w + x) * 4) as usize;
            if d > 0.5 {
                // Outside the rounded shape: fully transparent.
                pixels[i + 3] = 0;
                continue;
            }
            // Body gradient runs top -> bottom.
            let v = (y as f32 / (h - 1) as f32).clamp(0.0, 1.0);
            let body = [
                body_top[0] * (1.0 - v) + body_bottom[0] * v,
                body_top[1] * (1.0 - v) + body_bottom[1] * v,
                body_top[2] * (1.0 - v) + body_bottom[2] * v,
            ];
            // Pick rim, bevel, or body based on inset distance.
            let c = if d > -rim {
                rim_col
            } else if d > -(rim + 1.0) && fy < h as f32 * 0.5 {
                // 1px bright bevel just inside the rim, top half only.
                bevel
            } else {
                body
            };
            // Smoothstep alpha at the outer edge for a clean AA outline.
            let a = (1.0 - (d + 0.5).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            pixels[i] = (c[0] * 255.0) as u8;
            pixels[i + 1] = (c[1] * 255.0) as u8;
            pixels[i + 2] = (c[2] * 255.0) as u8;
            pixels[i + 3] = (a * 255.0) as u8;
        }
    }
    (w, h, pixels)
}

/// Evaluate the row-5 demo's closed-Bezier "infinity" path at parameter
/// `t in [0, 1]`, centred on `(cx, cy)`. Four cubic Bezier segments stitched
/// into a stylised figure-eight: the curve sweeps out, crosses through the
/// centre, sweeps to the other side, and comes back. Used by both the
/// `PathTrack` closure driving the moving dot and by the polyline trace
/// overlay so the two stay in sync.
fn infinity_bezier_point(t: f32, cx: f32, cy: f32) -> [f32; 2] {
    // Four cubic segments. The curve passes through (cx, cy) at the start
    // of segments 2 and 4, producing the crossing of the figure-eight.
    let segment_count = 4.0_f32;
    let raw = (t.clamp(0.0, 1.0) * segment_count).min(segment_count - 1e-6);
    let seg = raw.floor() as usize;
    let u = raw - seg as f32;
    let lobe_w = 80.0_f32;
    let lobe_h = 50.0_f32;
    let (p0, p1, p2, p3) = match seg {
        0 => (
            // From centre, sweep up-and-right out to the right lobe's tip.
            [cx, cy],
            [cx + lobe_w * 0.4, cy - lobe_h * 1.4],
            [cx + lobe_w * 1.2, cy - lobe_h * 0.9],
            [cx + lobe_w * 1.2, cy],
        ),
        1 => (
            // From right tip, swing down-and-back through the centre.
            [cx + lobe_w * 1.2, cy],
            [cx + lobe_w * 1.2, cy + lobe_h * 0.9],
            [cx + lobe_w * 0.4, cy + lobe_h * 1.4],
            [cx, cy],
        ),
        2 => (
            // From centre, sweep down-and-left out to the left lobe's tip.
            [cx, cy],
            [cx - lobe_w * 0.4, cy + lobe_h * 1.4],
            [cx - lobe_w * 1.2, cy + lobe_h * 0.9],
            [cx - lobe_w * 1.2, cy],
        ),
        _ => (
            // From left tip, swing up-and-back through the centre.
            [cx - lobe_w * 1.2, cy],
            [cx - lobe_w * 1.2, cy - lobe_h * 0.9],
            [cx - lobe_w * 0.4, cy - lobe_h * 1.4],
            [cx, cy],
        ),
    };
    let one = 1.0 - u;
    let w0 = one * one * one;
    let w1 = 3.0 * one * one * u;
    let w2 = 3.0 * one * u * u;
    let w3 = u * u * u;
    [
        w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0],
        w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1],
    ]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor() as u32 % 6;
    let f = h * 6.0 - (h * 6.0).floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_overlay(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Colourmap:");
    let selected_name = ALL_COLOURMAPS
        .iter()
        .find(|(c, _)| *c == app.ovl_state.colourmap)
        .map(|(_, n)| *n)
        .unwrap_or("Unknown");
    egui::ComboBox::from_id_salt("ovl_colourmap")
        .selected_text(selected_name)
        .show_ui(ui, |ui| {
            for (cmap, name) in ALL_COLOURMAPS {
                ui.selectable_value(&mut app.ovl_state.colourmap, *cmap, *name);
            }
        });

    ui.separator();
    ui.label("Bar orientation:");
    if ui
        .radio(
            app.ovl_state.bar_orientation == ScalarBarOrientation::Vertical,
            "Vertical",
        )
        .clicked()
    {
        app.ovl_state.bar_orientation = ScalarBarOrientation::Vertical;
    }
    if ui
        .radio(
            app.ovl_state.bar_orientation == ScalarBarOrientation::Horizontal,
            "Horizontal",
        )
        .clicked()
    {
        app.ovl_state.bar_orientation = ScalarBarOrientation::Horizontal;
    }

    ui.separator();
    ui.label("Bar anchor:");
    for (label, anchor) in [
        ("Top-Left", ScalarBarAnchor::TopLeft),
        ("Top-Right", ScalarBarAnchor::TopRight),
        ("Bottom-Left", ScalarBarAnchor::BottomLeft),
        ("Bottom-Right", ScalarBarAnchor::BottomRight),
    ] {
        if ui
            .radio(app.ovl_state.bar_anchor == anchor, label)
            .clicked()
        {
            app.ovl_state.bar_anchor = anchor;
        }
    }

    ui.separator();
    ui.label("Bar size (px):");
    ui.add(egui::Slider::new(&mut app.ovl_state.bar_size, 80.0..=400.0).suffix(" px"));

    ui.separator();
    ui.label("Tick count:");
    ui.add(egui::Slider::new(&mut app.ovl_state.tick_count, 2..=10));

    ui.separator();
    ui.label("Background colour:");
    let mut rgb = [
        app.ovl_state.bg_colour[0],
        app.ovl_state.bg_colour[1],
        app.ovl_state.bg_colour[2],
    ];
    if ui.color_edit_button_rgb(&mut rgb).changed() {
        app.ovl_state.bg_colour[0] = rgb[0];
        app.ovl_state.bg_colour[1] = rgb[1];
        app.ovl_state.bg_colour[2] = rgb[2];
    }
    ui.label("Background opacity:");
    ui.add(egui::Slider::new(
        &mut app.ovl_state.bg_colour[3],
        0.0..=1.0,
    ));

    ui.separator();
    ui.checkbox(&mut app.ovl_state.show_ruler, "Show ruler");
    ui.checkbox(&mut app.ovl_state.show_labels, "Show callout labels");
    ui.checkbox(&mut app.ovl_state.show_shapes, "Show SDF shapes");
    if app.ovl_state.show_shapes {
        ui.add(
            egui::Slider::new(&mut app.ovl_state.shape_corner_radius, 0.0..=40.0)
                .text("Corner radius"),
        );
        ui.add(
            egui::Slider::new(&mut app.ovl_state.shape_border_width, 0.0..=6.0)
                .text("Border width"),
        );
    }

    ui.separator();
    ui.label("Backdrop blur:");
    ui.add(egui::Slider::new(&mut app.ovl_state.backdrop_blur_radius, 0.0..=40.0).text("radius"));

    ui.separator();
    ui.checkbox(
        &mut app.ovl_state.show_tex_shapes,
        "Show texture-masked shapes",
    );
    if app.ovl_state.tex_id.is_none() {
        ui.label(egui::RichText::new("(texture not yet uploaded)").weak());
    }
}

// ---------------------------------------------------------------------------
// Point cloud geometry (built once)
// ---------------------------------------------------------------------------

/// Generate a sinusoidal wave surface sampled on a 70x70 grid.
///
/// x in [-pi/2, pi/2], y in [0, pi], z = sin(x) * 1.5.
/// Returns (positions, scalars) where scalar = z, matching the bar range [-1.5, 1.5].
pub(crate) fn build_ovl_cloud() -> (Vec<[f32; 3]>, Vec<f32>) {
    use std::f32::consts::PI;
    let nx = 70usize;
    let ny = 70usize;
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
// Frame overlay items
// ---------------------------------------------------------------------------

pub(crate) fn build_overlay_frame(
    app: &App,
) -> (
    Vec<OverlayShapeItem>,
    Vec<LabelItem>,
    ScalarBarItem,
    Option<RulerItem>,
    Vec<viewport_lib::OverlayPolylineItem>,
) {
    let colourmap_id = ColourmapId(app.ovl_state.colourmap as usize);

    let bar = ScalarBarItem::new(colourmap_id, -1.5, 1.5)
        .with_title("Height (m)")
        .with_anchor(app.ovl_state.bar_anchor)
        .with_orientation(app.ovl_state.bar_orientation)
        .with_tick_count(app.ovl_state.tick_count)
        .with_bar_length(app.ovl_state.bar_size)
        .with_background_colour(app.ovl_state.bg_colour);

    let mut labels = Vec::new();
    if app.ovl_state.show_labels {
        for (text, pos, colour) in [
            (
                "Peak +1.5 m",
                [1.57_f32, 0.0, 1.5_f32],
                [0.3_f32, 1.0, 0.5, 1.0_f32],
            ),
            ("Trough -1.5 m", [-1.57, 3.14, -1.5], [1.0, 0.4, 0.3, 1.0]),
            ("Origin", [0.0, 0.0, 0.0], [0.9, 0.9, 0.9, 1.0]),
        ] {
            labels.push(
                LabelItem::new(text)
                    .with_world_anchor(pos)
                    .with_colour(colour)
                    .with_leader_line(true)
                    .with_leader_colour([colour[0], colour[1], colour[2], 0.7])
                    .with_background(true)
                    .with_background_colour([0.0, 0.0, 0.0, 0.5])
                    .with_border_radius(4.0)
                    .with_padding(4.0)
                    .with_anchor_align(LabelAnchor::Left),
            );
        }
    }

    let ruler = if app.ovl_state.show_ruler {
        Some(
            RulerItem::new([1.57, 0.0, 0.0], [-1.57, 3.14, 0.0])
                .with_colour([1.0, 0.85, 0.2, 1.0])
                .with_label_colour([1.0, 0.85, 0.2, 1.0])
                .with_label_format("{:.2} m")
                .with_end_caps(true),
        )
    } else {
        None
    };

    // SDF overlay shapes: a row of five shapes, vertically centred on a
    // common midline with equal spacing between them.
    let mut shapes = Vec::new();
    let mut polylines: Vec<viewport_lib::OverlayPolylineItem> = Vec::new();
    if app.ovl_state.show_shapes {
        let cr = app.ovl_state.shape_corner_radius;
        let bw = app.ovl_state.shape_border_width;

        let row_h = 70.0_f32; // row height all shapes centre on
        let y_mid = 20.0 + row_h * 0.5; // vertical midpoint
        let gap = 16.0_f32; // horizontal gap between shapes
        let mut x = 20.0_f32; // running left edge

        let mut items: Vec<(f32, f32, OverlayShape, [f32; 4], [f32; 4])> = vec![
            // Rounded rect
            (
                120.0,
                70.0,
                OverlayShape::Rect { corner_radius: cr },
                [0.12, 0.12, 0.18, 0.85],
                [0.8, 0.8, 0.8, 0.9],
            ),
            // Per-corner radii rect
            (
                120.0,
                70.0,
                OverlayShape::RoundedRect {
                    radii: [cr, 0.0, cr, 0.0],
                },
                [0.05, 0.15, 0.25, 0.85],
                [0.3, 0.7, 1.0, 0.9],
            ),
            // Circle
            (
                70.0,
                70.0,
                OverlayShape::Circle,
                [0.2, 0.5, 0.15, 0.85],
                [0.4, 1.0, 0.3, 0.9],
            ),
            // Ellipse
            (
                120.0,
                60.0,
                OverlayShape::Ellipse,
                [0.3, 0.1, 0.3, 0.85],
                [0.8, 0.4, 1.0, 0.9],
            ),
            // Capsule
            (
                120.0,
                40.0,
                OverlayShape::Capsule,
                [0.3, 0.2, 0.05, 0.85],
                [1.0, 0.8, 0.3, 0.9],
            ),
            // Ring
            (
                70.0,
                70.0,
                OverlayShape::Ring {
                    inner_radius_frac: 0.65,
                },
                [0.15, 0.35, 0.5, 0.85],
                [0.3, 0.8, 1.0, 0.9],
            ),
            // Arc (270-degree sweep)
            (
                70.0,
                70.0,
                OverlayShape::Arc {
                    inner_radius_frac: 0.6,
                    start_angle: 0.0,
                    end_angle: std::f32::consts::PI * 1.5,
                },
                [0.5, 0.2, 0.1, 0.85],
                [1.0, 0.5, 0.2, 0.9],
            ),
            // Triangle (pointing up)
            (
                60.0,
                60.0,
                OverlayShape::Triangle {
                    direction: TriangleDirection::Up,
                },
                [0.4, 0.4, 0.1, 0.85],
                [1.0, 1.0, 0.3, 0.9],
            ),
        ];

        for (w, h, shape, colour, border_colour) in items.drain(..) {
            shapes.push(
                OverlayShapeItem::new(shape, [x, y_mid - h * 0.5], [w, h])
                    .with_fill(OverlayFill::Solid(colour))
                    .with_border(border_colour, bw)
                    .with_z_order(0),
            );
            x += w + gap;
        }

        // ---------------------------------------------------------------------------
        // Texture-masked shapes (second row, below the solid shapes).
        // These demonstrate upload_overlay_texture + OverlayShapeItem::texture.
        // ---------------------------------------------------------------------------
        if app.ovl_state.show_tex_shapes {
            let bw2 = bw;
            let row2_h = 90.0_f32;
            let y2_mid = 20.0 + row_h + 24.0 + row2_h * 0.5;
            let mut x2 = 20.0_f32;

            // Circle: colour-wheel gradient texture, white border.
            if let Some(tid) = app.ovl_state.tex_id {
                let sz = 90.0_f32;
                shapes.push(
                    OverlayShapeItem::new(OverlayShape::Circle, [x2, y2_mid - sz * 0.5], [sz, sz])
                        .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]))
                        .with_border([1.0, 1.0, 1.0, 0.9], bw2)
                        .with_texture(tid),
                );
                x2 += sz + gap;
            }

            // Rounded rect: Carl Gauss portrait.
            if let Some(tid) = app.ovl_state.carlgauss_tex_id {
                shapes.push(
                    OverlayShapeItem::new(
                        OverlayShape::Rect { corner_radius: cr },
                        [x2, y2_mid - row2_h * 0.5],
                        [140.0, row2_h],
                    )
                    .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]))
                    .with_border([0.8, 0.8, 0.8, 0.9], bw2)
                    .with_texture(tid),
                );
                x2 += 140.0 + gap;
            }

            // Texture-transform: the demo colour-wheel rotating *inside* a
            // static circle. The shape stays still; the texture spins.
            if let Some(tid) = app.ovl_state.tex_id {
                let sz = 90.0_f32;
                let t = app.ovl_state.start_time.elapsed().as_secs_f32();
                shapes.push(
                    OverlayShapeItem::new(OverlayShape::Circle, [x2, y2_mid - sz * 0.5], [sz, sz])
                        .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]))
                        .with_border([1.0, 1.0, 1.0, 0.9], bw2)
                        .with_texture(tid)
                        .with_texture_transform(viewport_lib::TextureTransform {
                            rotation: t * 0.5,
                            ..Default::default()
                        }),
                );
                x2 += sz + gap;
            }

            // Texture-transform: same colour-wheel tiled 3x3 across a wider
            // rect using TileMode::Tile and scale = 1/3.
            if let Some(tid) = app.ovl_state.tex_id {
                shapes.push(
                    OverlayShapeItem::new(
                        OverlayShape::Rect { corner_radius: cr },
                        [x2, y2_mid - row2_h * 0.5],
                        [180.0, row2_h],
                    )
                    .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]))
                    .with_border([1.0, 1.0, 1.0, 0.9], bw2)
                    .with_texture(tid)
                    .with_texture_transform(viewport_lib::TextureTransform {
                        scale: [3.0, 3.0],
                        tile_mode: viewport_lib::TileMode::Tile,
                        ..Default::default()
                    }),
                );
            }
        }

        // ---------------------------------------------------------------------------
        // Gradient fill shapes (third row).
        // ---------------------------------------------------------------------------
        {
            use std::f32::consts::PI;
            let row3_h = 70.0_f32;
            let y3_mid = 20.0 + row_h + 24.0 + 90.0 + 24.0 + row3_h * 0.5;
            let mut x3 = 20.0_f32;

            // Rounded rect: left-to-right blue-to-teal gradient.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: cr },
                    [x3, y3_mid - row3_h * 0.5],
                    [120.0, row3_h],
                )
                .with_fill(OverlayFill::LinearGradient {
                    start_colour: [0.05, 0.15, 0.55, 0.9],
                    end_colour: [0.05, 0.65, 0.65, 0.9],
                    angle: 0.0,
                })
                .with_border([0.3, 0.7, 1.0, 0.8], bw),
            );
            x3 += 120.0 + gap;

            // Per-corner rect: diagonal gradient (PI/4 = 45 degrees).
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::RoundedRect {
                        radii: [cr, 0.0, cr, 0.0],
                    },
                    [x3, y3_mid - row3_h * 0.5],
                    [120.0, row3_h],
                )
                .with_fill(OverlayFill::LinearGradient {
                    start_colour: [0.5, 0.05, 0.15, 0.9],
                    end_colour: [1.0, 0.6, 0.1, 0.9],
                    angle: PI / 4.0,
                })
                .with_border([1.0, 0.5, 0.2, 0.8], bw),
            );
            x3 += 120.0 + gap;

            // Circle: top-to-bottom (angle = PI/2) dark-to-bright gradient.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Circle,
                    [x3, y3_mid - row3_h * 0.5],
                    [row3_h, row3_h],
                )
                .with_fill(OverlayFill::LinearGradient {
                    start_colour: [0.05, 0.35, 0.05, 0.9],
                    end_colour: [0.5, 1.0, 0.3, 0.9],
                    angle: PI / 2.0,
                })
                .with_border([0.4, 1.0, 0.3, 0.8], bw),
            );
            x3 += row3_h + gap;

            // Ellipse: horizontal gradient, purple-to-pink.
            shapes.push(
                OverlayShapeItem::new(OverlayShape::Ellipse, [x3, y3_mid - 30.0], [120.0, 60.0])
                    .with_fill(OverlayFill::LinearGradient {
                        start_colour: [0.35, 0.05, 0.55, 0.9],
                        end_colour: [0.9, 0.3, 0.6, 0.9],
                        angle: 0.0,
                    })
                    .with_border([0.8, 0.4, 1.0, 0.8], bw),
            );
            x3 += 120.0 + gap;

            // Capsule: horizontal gradient, dark grey to white.
            shapes.push(
                OverlayShapeItem::new(OverlayShape::Capsule, [x3, y3_mid - 20.0], [120.0, 40.0])
                    .with_fill(OverlayFill::LinearGradient {
                        start_colour: [0.15, 0.15, 0.15, 0.9],
                        end_colour: [0.85, 0.85, 0.85, 0.9],
                        angle: 0.0,
                    })
                    .with_border([0.6, 0.6, 0.6, 0.8], bw),
            );
            x3 += 120.0 + gap;

            // Ring: diagonal gradient creates a highlight effect.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Ring {
                        inner_radius_frac: 0.65,
                    },
                    [x3, y3_mid - row3_h * 0.5],
                    [row3_h, row3_h],
                )
                .with_fill(OverlayFill::LinearGradient {
                    start_colour: [0.1, 0.3, 0.6, 0.9],
                    end_colour: [0.7, 0.9, 1.0, 0.9],
                    angle: -PI / 4.0,
                })
                .with_border([0.3, 0.7, 1.0, 0.8], bw),
            );
            x3 += row3_h + gap;

            // Triangle: bottom-to-top warm gradient.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Triangle {
                        direction: TriangleDirection::Up,
                    },
                    [x3, y3_mid - row3_h * 0.5],
                    [60.0, row3_h],
                )
                .with_fill(OverlayFill::LinearGradient {
                    start_colour: [0.7, 0.15, 0.05, 0.9],
                    end_colour: [1.0, 0.9, 0.1, 0.9],
                    angle: PI / 2.0,
                })
                .with_border([1.0, 0.6, 0.2, 0.8], bw),
            );
            x3 += 60.0 + gap;

            // Circle with a radial gradient: bright centre, dark edge.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Circle,
                    [x3, y3_mid - row3_h * 0.5],
                    [row3_h, row3_h],
                )
                .with_fill(OverlayFill::RadialGradient {
                    centre_colour: [1.0, 0.95, 0.7, 1.0],
                    edge_colour: [0.2, 0.05, 0.0, 0.9],
                })
                .with_border([1.0, 0.8, 0.4, 0.8], bw),
            );
            x3 += row3_h + gap;

            // Circle with a conical (sweep) gradient: colour-wheel arc.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Circle,
                    [x3, y3_mid - row3_h * 0.5],
                    [row3_h, row3_h],
                )
                .with_fill(OverlayFill::ConicalGradient {
                    start_colour: [0.95, 0.2, 0.4, 1.0],
                    end_colour: [0.2, 0.6, 1.0, 1.0],
                    offset_angle: 0.0,
                })
                .with_border([0.9, 0.9, 0.9, 0.8], bw),
            );
            x3 += row3_h + gap;

            // Multi-stop linear gradient (sunset ramp): 4 stops at uneven
            // positions, demonstrating the OverlayFill::LinearGradientMulti
            // path.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: cr },
                    [x3, y3_mid - row3_h * 0.5],
                    [120.0, row3_h],
                )
                .with_fill(OverlayFill::LinearGradientMulti {
                    stops: vec![
                        viewport_lib::GradientStop::new(0.0, [0.05, 0.05, 0.20, 1.0]),
                        viewport_lib::GradientStop::new(0.4, [0.55, 0.10, 0.45, 1.0]),
                        viewport_lib::GradientStop::new(0.75, [0.95, 0.45, 0.20, 1.0]),
                        viewport_lib::GradientStop::new(1.0, [1.0, 0.95, 0.55, 1.0]),
                    ],
                    angle: 0.0,
                })
                .with_border([1.0, 0.85, 0.4, 0.8], bw),
            );
            x3 += 120.0 + gap;

            // Multi-stop conical gradient: 4 stops around the sweep, shows
            // how positions can be packed asymmetrically.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Circle,
                    [x3, y3_mid - row3_h * 0.5],
                    [row3_h, row3_h],
                )
                .with_fill(OverlayFill::ConicalGradientMulti {
                    stops: vec![
                        viewport_lib::GradientStop::new(0.0, [0.95, 0.2, 0.2, 1.0]),
                        viewport_lib::GradientStop::new(0.25, [1.0, 0.85, 0.2, 1.0]),
                        viewport_lib::GradientStop::new(0.6, [0.2, 0.85, 0.45, 1.0]),
                        viewport_lib::GradientStop::new(1.0, [0.3, 0.5, 1.0, 1.0]),
                    ],
                    offset_angle: 0.0,
                })
                .with_border([0.9, 0.9, 0.9, 0.8], bw),
            );
            let _ = x3;
        }

        // ---------------------------------------------------------------------------
        // Shadow/glow shapes (fourth row).
        // ---------------------------------------------------------------------------
        {
            let row4_h = 70.0_f32;
            let y4_mid = 20.0 + row_h + 24.0 + 90.0 + 24.0 + 70.0 + 24.0 + row4_h * 0.5;
            let mut x4 = 20.0_f32;

            // Rounded rect with drop shadow.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: cr },
                    [x4, y4_mid - row4_h * 0.5],
                    [120.0, row4_h],
                )
                .with_fill(OverlayFill::Solid([0.15, 0.15, 0.2, 0.95]))
                .with_border([0.5, 0.5, 0.6, 0.8], bw)
                .with_shadow([0.0, 0.0, 0.0, 0.5], 12.0, [4.0, 4.0]),
            );
            x4 += 120.0 + gap + 16.0; // extra gap for shadow bleed

            // Circle with blue glow (no offset).
            let sz = row4_h;
            shapes.push(
                OverlayShapeItem::new(OverlayShape::Circle, [x4, y4_mid - sz * 0.5], [sz, sz])
                    .with_fill(OverlayFill::Solid([0.1, 0.15, 0.35, 0.95]))
                    .with_border([0.3, 0.5, 1.0, 0.9], bw)
                    .with_shadow([0.2, 0.4, 1.0, 0.6], 16.0, [0.0, 0.0]),
            );
            x4 += sz + gap + 16.0;

            // Capsule with warm glow.
            shapes.push(
                OverlayShapeItem::new(OverlayShape::Capsule, [x4, y4_mid - 20.0], [120.0, 40.0])
                    .with_fill(OverlayFill::Solid([0.3, 0.15, 0.05, 0.95]))
                    .with_border([1.0, 0.6, 0.2, 0.9], bw)
                    .with_shadow([1.0, 0.5, 0.1, 0.45], 14.0, [0.0, 2.0]),
            );
            x4 += 120.0 + gap + 16.0;

            // Ellipse with offset shadow.
            shapes.push(
                OverlayShapeItem::new(OverlayShape::Ellipse, [x4, y4_mid - 30.0], [120.0, 60.0])
                    .with_fill(OverlayFill::Solid([0.2, 0.3, 0.15, 0.95]))
                    .with_border([0.5, 0.9, 0.3, 0.9], bw)
                    .with_shadow([0.0, 0.0, 0.0, 0.45], 10.0, [3.0, 5.0]),
            );
            x4 += 120.0 + gap + 16.0;

            // Triangle with green glow.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Triangle {
                        direction: TriangleDirection::Up,
                    },
                    [x4, y4_mid - row4_h * 0.5],
                    [60.0, row4_h],
                )
                .with_fill(OverlayFill::Solid([0.05, 0.25, 0.1, 0.95]))
                .with_border([0.3, 1.0, 0.4, 0.9], bw)
                .with_shadow([0.1, 0.8, 0.2, 0.5], 14.0, [0.0, 0.0]),
            );
            x4 += 60.0 + gap + 16.0;

            // Pressed-button effect using shadow_inset: a dark inner shadow
            // offset slightly down makes the surface read as recessed.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: cr },
                    [x4, y4_mid - row4_h * 0.5],
                    [120.0, row4_h],
                )
                .with_fill(OverlayFill::Solid([0.22, 0.24, 0.30, 1.0]))
                .with_border([0.05, 0.07, 0.12, 0.9], 1.0)
                .with_shadow([0.0, 0.0, 0.0, 0.7], 14.0, [0.0, 4.0])
                .with_shadow_inset(true),
            );
            let _ = x4;
        }

        // ---------------------------------------------------------------------------
        // Border mode + animation shapes (fifth row).
        // ---------------------------------------------------------------------------
        {
            let row5_h = 70.0_f32;
            let y5_mid =
                20.0 + row_h + 24.0 + 90.0 + 24.0 + 70.0 + 24.0 + 70.0 + 24.0 + row5_h * 0.5;
            let mut x5 = 20.0_f32;

            // Inset border (default).
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: cr },
                    [x5, y5_mid - row5_h * 0.5],
                    [100.0, row5_h],
                )
                .with_fill(OverlayFill::Solid([0.15, 0.15, 0.2, 0.9]))
                .with_border([0.9, 0.9, 0.3, 1.0], 3.0)
                .with_border_mode(BorderMode::Inset),
            );
            x5 += 100.0 + gap;

            // Outer border.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: cr },
                    [x5, y5_mid - row5_h * 0.5],
                    [100.0, row5_h],
                )
                .with_fill(OverlayFill::Solid([0.15, 0.15, 0.2, 0.9]))
                .with_border([0.3, 0.9, 0.5, 1.0], 3.0)
                .with_border_mode(BorderMode::Outer),
            );
            x5 += 100.0 + gap;

            // Center border on a circle.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Circle,
                    [x5, y5_mid - row5_h * 0.5],
                    [row5_h, row5_h],
                )
                .with_fill(OverlayFill::Solid([0.15, 0.15, 0.2, 0.9]))
                .with_border([0.5, 0.5, 1.0, 1.0], 3.0)
                .with_border_mode(BorderMode::Center),
            );
            x5 += row5_h + gap;

            // Pulsing circle.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Circle,
                    [x5, y5_mid - row5_h * 0.5],
                    [row5_h, row5_h],
                )
                .with_fill(OverlayFill::Solid([0.2, 0.5, 1.0, 0.9]))
                .with_border([0.4, 0.7, 1.0, 0.9], bw)
                .with_animation(OverlayAnimation::Pulse {
                    start_time: 0.0,
                    period: 2.0,
                }),
            );
            x5 += row5_h + gap;

            // Repeating fade-in capsule: restarts every 4 seconds.
            let now = app.ovl_state.start_time.elapsed().as_secs_f64();
            let cycle = 4.0;
            let fade_start = (now / cycle).floor() * cycle;
            shapes.push(
                OverlayShapeItem::new(OverlayShape::Capsule, [x5, y5_mid - 20.0], [120.0, 40.0])
                    .with_fill(OverlayFill::Solid([0.6, 0.2, 0.1, 0.9]))
                    .with_border([1.0, 0.5, 0.3, 0.9], bw)
                    .with_animation(OverlayAnimation::FadeIn {
                        start_time: fade_start,
                        duration: 3.0,
                    }),
            );
            x5 += 120.0 + gap;

            // Rotating circle filled with a multi-stop conical gradient.
            // The gradient rotates with the shape because the SDF is
            // evaluated in the unrotated local frame.
            let t = app.ovl_state.start_time.elapsed().as_secs_f32();
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Circle,
                    [x5, y5_mid - row5_h * 0.5],
                    [row5_h, row5_h],
                )
                .with_fill(OverlayFill::ConicalGradientMulti {
                    stops: vec![
                        viewport_lib::GradientStop::new(0.0, [0.27, 0.0, 0.33, 1.0]),
                        viewport_lib::GradientStop::new(0.25, [0.13, 0.32, 0.55, 1.0]),
                        viewport_lib::GradientStop::new(0.5, [0.12, 0.62, 0.50, 1.0]),
                        viewport_lib::GradientStop::new(0.75, [0.78, 0.79, 0.21, 1.0]),
                        viewport_lib::GradientStop::new(1.0, [0.99, 0.91, 0.14, 1.0]),
                    ],
                    offset_angle: 0.0,
                })
                .with_rotation(t * 0.8)
                .with_border([1.0, 1.0, 1.0, 0.7], bw),
            );
            x5 += row5_h + gap;

            // Rotating cross.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Cross {
                        arm_width_frac: 0.35,
                    },
                    [x5, y5_mid - row5_h * 0.5],
                    [row5_h, row5_h],
                )
                .with_fill(OverlayFill::Solid([0.3, 0.8, 0.5, 0.9]))
                .with_rotation(-t * 1.2)
                .with_border([0.5, 1.0, 0.7, 0.9], bw),
            );
            x5 += row5_h + gap;

            // Multi-channel animation demos. Each shape isolates
            // one channel of OverlayAnimations so the system is obvious:
            // position, size, fill, rotation.
            let base_x = x5;

            // Position: sliding rect, PingPong EaseInOut.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 4.0 },
                    [base_x, y5_mid - 14.0],
                    [44.0, 28.0],
                )
                .with_fill(OverlayFill::Solid([0.95, 0.65, 0.25, 0.95]))
                .with_border([1.0, 0.85, 0.4, 0.9], bw)
                .with_animations(viewport_lib::OverlayAnimations {
                    position: Some(viewport_lib::AnimTrack {
                        start_time: 0.0,
                        duration: 1.8,
                        from: [base_x, y5_mid - 14.0],
                        to: [base_x + 50.0, y5_mid - 14.0],
                        easing: viewport_lib::OverlayEasing::EaseInOut,
                        repeat: viewport_lib::RepeatMode::PingPong,
                    }),
                    ..Default::default()
                }),
            );
            x5 += 100.0 + gap;

            // Size: pulsating circle, PingPong Pulse easing.
            let pulse_cx = x5 + row5_h * 0.5;
            let pulse_cy = y5_mid;
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Circle,
                    [pulse_cx - 22.0, pulse_cy - 22.0],
                    [44.0, 44.0],
                )
                .with_fill(OverlayFill::Solid([0.45, 0.85, 1.0, 0.95]))
                .with_border([0.7, 0.95, 1.0, 0.9], bw)
                .with_animations(viewport_lib::OverlayAnimations {
                    size: Some(viewport_lib::AnimTrack {
                        start_time: 0.0,
                        duration: 1.4,
                        from: [44.0, 44.0],
                        to: [64.0, 64.0],
                        easing: viewport_lib::OverlayEasing::Pulse,
                        repeat: viewport_lib::RepeatMode::Loop,
                    }),
                    position: Some(viewport_lib::AnimTrack {
                        // Recentre while the size grows so the circle pulses
                        // about its centre rather than drifting south-east.
                        start_time: 0.0,
                        duration: 1.4,
                        from: [pulse_cx - 22.0, pulse_cy - 22.0],
                        to: [pulse_cx - 32.0, pulse_cy - 32.0],
                        easing: viewport_lib::OverlayEasing::Pulse,
                        repeat: viewport_lib::RepeatMode::Loop,
                    }),
                    ..Default::default()
                }),
            );
            x5 += row5_h + gap;

            // Fill colour: colour cycles smoothly between two colours,
            // Loop with EaseInOut for a soft transition.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: cr },
                    [x5, y5_mid - row5_h * 0.5],
                    [70.0, row5_h],
                )
                .with_fill(OverlayFill::Solid([0.95, 0.25, 0.5, 0.95]))
                .with_border([1.0, 1.0, 1.0, 0.7], bw)
                .with_animations(viewport_lib::OverlayAnimations {
                    fill: Some(viewport_lib::AnimTrack {
                        start_time: 0.0,
                        duration: 1.6,
                        from: [0.95, 0.25, 0.5, 0.95],
                        to: [0.25, 0.55, 0.95, 0.95],
                        easing: viewport_lib::OverlayEasing::EaseInOut,
                        repeat: viewport_lib::RepeatMode::PingPong,
                    }),
                    ..Default::default()
                }),
            );
            x5 += 70.0 + gap;

            // Rotation: a Star slowly spinning via the rotation channel
            // (Linear, Loop).
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Star {
                        points: 5,
                        inner_radius_frac: 0.45,
                    },
                    [x5, y5_mid - row5_h * 0.5],
                    [row5_h, row5_h],
                )
                .with_fill(OverlayFill::Solid([0.95, 0.9, 0.3, 0.95]))
                .with_border([1.0, 0.95, 0.5, 0.9], bw)
                .with_animations(viewport_lib::OverlayAnimations {
                    rotation: Some(viewport_lib::AnimTrack {
                        start_time: 0.0,
                        duration: 4.0,
                        from: 0.0,
                        to: std::f32::consts::TAU,
                        easing: viewport_lib::OverlayEasing::Linear,
                        repeat: viewport_lib::RepeatMode::Loop,
                    }),
                    ..Default::default()
                }),
            );
            x5 += row5_h + gap;

            // A small circle following a closed cubic Bezier
            // figure-eight, with the path itself traced underneath using
            // the new OverlayPolylineItem primitive.
            {
                let cx = x5 + 130.0;
                let cy = y5_mid;
                let dot_size = 22.0_f32;

                let path = viewport_lib::PathTrack::<[f32; 2]>::new(0.0, 4.5, move |t| {
                    let p = infinity_bezier_point(t, cx, cy);
                    [p[0] - dot_size * 0.5, p[1] - dot_size * 0.5]
                })
                .with_repeat(viewport_lib::RepeatMode::Loop);

                // Sample the same Bezier closure into the polyline trace.
                let trace = viewport_lib::OverlayPolylineItem::from_path(
                    |t| infinity_bezier_point(t, cx, cy),
                    160,
                    2.0,
                    [1.0, 1.0, 1.0, 0.45],
                );
                polylines.push(trace.with_closed(true).with_z_order(-1));

                shapes.push(
                    OverlayShapeItem::new(
                        OverlayShape::Circle,
                        [cx - dot_size * 0.5, cy - dot_size * 0.5],
                        [dot_size, dot_size],
                    )
                    .with_fill(OverlayFill::Solid([0.95, 0.45, 0.85, 1.0]))
                    .with_border([1.0, 0.7, 0.95, 0.9], bw)
                    .with_animations(viewport_lib::OverlayAnimations {
                        position_path: Some(path),
                        ..Default::default()
                    }),
                );
                x5 += 260.0 + gap;
            }
            let _ = x5;
        }

        // ---------------------------------------------------------------------------
        // New shape types (row 6): Line, Star, RegularPolygon, Cross.
        // ---------------------------------------------------------------------------
        {
            let row6_h = 70.0_f32;
            let y6_mid = 20.0
                + row_h
                + 24.0
                + 90.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0
                + row6_h * 0.5;
            let mut x6 = 20.0_f32;

            // Diagonal line (round cap).
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Line {
                        thickness: 6.0,
                        cap: LineCap::Round,
                    },
                    [x6, y6_mid - row6_h * 0.5],
                    [100.0, row6_h],
                )
                .with_fill(OverlayFill::Solid([0.2, 0.7, 1.0, 0.9]))
                .with_border([0.5, 0.9, 1.0, 0.9], bw),
            );
            x6 += 100.0 + gap;

            // Horizontal line (square cap): thin 3px stroke.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Line {
                        thickness: 4.0,
                        cap: LineCap::Square,
                    },
                    [x6, y6_mid - 2.0],
                    [120.0, 4.0],
                )
                .with_fill(OverlayFill::Solid([1.0, 0.6, 0.2, 0.9]))
                .with_border([1.0, 0.8, 0.4, 0.9], 0.0),
            );
            x6 += 120.0 + gap;

            // 5-pointed star.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Star {
                        points: 5,
                        inner_radius_frac: 0.45,
                    },
                    [x6, y6_mid - row6_h * 0.5],
                    [row6_h, row6_h],
                )
                .with_fill(OverlayFill::Solid([1.0, 0.85, 0.1, 0.9]))
                .with_border([1.0, 1.0, 0.5, 0.9], bw),
            );
            x6 += row6_h + gap;

            // 6-pointed star.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Star {
                        points: 6,
                        inner_radius_frac: 0.5,
                    },
                    [x6, y6_mid - row6_h * 0.5],
                    [row6_h, row6_h],
                )
                .with_fill(OverlayFill::Solid([0.9, 0.3, 0.9, 0.9]))
                .with_border([1.0, 0.6, 1.0, 0.9], bw),
            );
            x6 += row6_h + gap;

            // Pentagon.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::RegularPolygon { sides: 5 },
                    [x6, y6_mid - row6_h * 0.5],
                    [row6_h, row6_h],
                )
                .with_fill(OverlayFill::Solid([0.2, 0.8, 0.4, 0.9]))
                .with_border([0.4, 1.0, 0.6, 0.9], bw),
            );
            x6 += row6_h + gap;

            // Hexagon.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::RegularPolygon { sides: 6 },
                    [x6, y6_mid - row6_h * 0.5],
                    [row6_h, row6_h],
                )
                .with_fill(OverlayFill::Solid([0.1, 0.5, 0.9, 0.9]))
                .with_border([0.3, 0.7, 1.0, 0.9], bw),
            );
            x6 += row6_h + gap;

            // Cross (wide arms).
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Cross {
                        arm_width_frac: 0.35,
                    },
                    [x6, y6_mid - row6_h * 0.5],
                    [row6_h, row6_h],
                )
                .with_fill(OverlayFill::Solid([0.9, 0.2, 0.2, 0.9]))
                .with_border([1.0, 0.5, 0.5, 0.9], bw),
            );
            x6 += row6_h + gap;

            // Cross with gradient fill and thin arms.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Cross {
                        arm_width_frac: 0.2,
                    },
                    [x6, y6_mid - row6_h * 0.5],
                    [row6_h, row6_h],
                )
                .with_fill(OverlayFill::LinearGradient {
                    start_colour: [1.0, 0.3, 0.1, 0.9],
                    end_colour: [0.1, 0.3, 1.0, 0.9],
                    angle: std::f32::consts::PI * 0.25,
                })
                .with_border([0.8, 0.8, 0.8, 0.9], bw),
            );
            let _ = x6;
        }

        // Clipping group: a rotating decagon whose right half is hidden by
        // a half-shape clip mask. As the polygon spins, vertices sweep
        // across the cut line, making the clip edge obvious in a small
        // footprint. The mask itself is invisible; a thin white outline
        // marks the clip rect.
        {
            let poly_size = 70.0_f32;
            let px = 20.0_f32;
            // Sit one row below row 6 (new shape types row). Row spacing
            // matches the 24px gap used between the other rows.
            let py = 20.0
                + row_h
                + 24.0
                + 90.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0;
            // Mask: the left half of the decagon's bounding box. Anything
            // with clip_id == 7 is only visible inside this rect.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 0.0 },
                    [px, py],
                    [poly_size * 0.5, poly_size],
                )
                .with_clip_mask(7),
            );
            // Outline so the clip edge is visible.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 0.0 },
                    [px, py],
                    [poly_size * 0.5, poly_size],
                )
                .with_fill(OverlayFill::Solid([0.0, 0.0, 0.0, 0.0]))
                .with_border([1.0, 1.0, 1.0, 0.5], 1.0)
                .with_border_mode(BorderMode::Outer),
            );
            // The rotating decagon, sized to the full square. Only the
            // left half passes the clip.
            let t = app.ovl_state.start_time.elapsed().as_secs_f32();
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::RegularPolygon { sides: 10 },
                    [px, py],
                    [poly_size, poly_size],
                )
                .with_rotation(t * 0.6)
                .with_fill(OverlayFill::Solid([0.9, 0.55, 0.2, 0.95]))
                .with_border([1.0, 0.8, 0.3, 0.9], bw)
                .with_clip(7),
            );
        }

        // 9-slice A/B comparison: two panels, same texture, same final size.
        // Left panel disables 9-slice so the source texture stretches across
        // the whole rect; the corner squares smear and the highlight bar
        // distorts. Right panel turns 9-slice on with 12px insets: the dark
        // corners stay at their authored size and only the edges/centre
        // stretch.
        if let Some(tid) = app.ovl_state.nine_slice_tex_id {
            let py = 20.0
                + row_h
                + 24.0
                + 90.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0;
            let panel_w = 180.0_f32;
            let panel_h = 70.0_f32;
            let x_left = 20.0 + 70.0 + gap; // sit beside the clip demo
            let x_right = x_left + panel_w + gap;

            // Left: plain stretched texture (no 9-slice).
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 0.0 },
                    [x_left, py],
                    [panel_w, panel_h],
                )
                .with_texture(tid)
                .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0])),
            );
            // Right: same source, same size, 9-slice on.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 0.0 },
                    [x_right, py],
                    [panel_w, panel_h],
                )
                .with_texture(tid)
                .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]))
                .with_nine_slice(viewport_lib::NineSlice {
                    insets_px: [10.0, 10.0, 10.0, 10.0],
                    centre_mode: viewport_lib::TileMode::Stretch,
                    edge_mode: viewport_lib::TileMode::Stretch,
                }),
            );
        }

        // Closed polyline fills: the left polygon uses OverlayFill directly;
        // the right heart uses the Carl Gauss overlay texture.
        {
            let py = 20.0
                + row_h
                + 24.0
                + 90.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0;
            let x_poly = 20.0 + 70.0 + gap + 180.0 + gap + 180.0 + gap + 32.0;
            let y_poly = py + 35.0;

            polylines.push(
                viewport_lib::OverlayPolylineItem::new(vec![
                    [x_poly, y_poly - 30.0],
                    [x_poly + 48.0, y_poly - 12.0],
                    [x_poly + 42.0, y_poly + 28.0],
                    [x_poly - 18.0, y_poly + 22.0],
                    [x_poly - 36.0, y_poly - 10.0],
                ])
                .with_thickness(2.0)
                .with_colour([1.0, 1.0, 1.0, 0.8])
                .with_closed(true)
                .with_fill(OverlayFill::LinearGradient {
                    start_colour: [0.12, 0.65, 0.95, 0.78],
                    end_colour: [0.95, 0.35, 0.7, 0.82],
                    angle: std::f32::consts::PI * 0.25,
                })
                .with_z_order(1),
            );

            if let Some(tid) = app.ovl_state.carlgauss_tex_id {
                let tx = x_poly + 128.0;
                let heart_points = (0..40)
                    .map(|i| {
                        let t = i as f32 / 40.0 * std::f32::consts::TAU;
                        let s = t.sin();
                        let c = t.cos();
                        [
                            tx + 2.1 * 16.0 * s.powi(3),
                            y_poly
                                - 2.1
                                    * (13.0 * c
                                        - 5.0 * (2.0 * t).cos()
                                        - 2.0 * (3.0 * t).cos()
                                        - (4.0 * t).cos()),
                        ]
                    })
                    .collect::<Vec<_>>();
                let heart_uvs = heart_points
                    .iter()
                    .map(|p| [(p[0] - (tx - 36.0)) / 72.0, (p[1] - (y_poly - 36.0)) / 72.0])
                    .collect::<Vec<_>>();
                polylines.push(
                    viewport_lib::OverlayPolylineItem::new(heart_points)
                        .with_thickness(2.0)
                        .with_colour([1.0, 0.78, 0.9, 0.9])
                        .with_closed(true)
                        .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 0.95]))
                        .with_texture(tid)
                        .with_uvs(heart_uvs)
                        .with_z_order(1),
                );
            }

            // Stroke controls: a marching-ants dashed marquee, a round-capped
            // dashed sine wave, and a dotted circle.
            {
                let t = app.ovl_state.start_time.elapsed().as_secs_f32();
                let x_stroke = x_poly + 128.0 + 64.0;

                polylines.push(
                    viewport_lib::OverlayPolylineItem::new(vec![
                        [x_stroke, y_poly - 28.0],
                        [x_stroke + 64.0, y_poly - 28.0],
                        [x_stroke + 64.0, y_poly + 28.0],
                        [x_stroke, y_poly + 28.0],
                    ])
                    .with_thickness(2.0)
                    .with_colour([1.0, 1.0, 1.0, 0.85])
                    .with_closed(true)
                    .with_stroke_pattern(viewport_lib::StrokePattern::Dashed {
                        dash_length: 8.0,
                        gap_length: 6.0,
                        offset: t * 20.0,
                    })
                    .with_z_order(1),
                );

                let x_wave = x_stroke + 64.0 + 28.0;
                polylines.push(
                    viewport_lib::OverlayPolylineItem::new(
                        (0..=40)
                            .map(|i| {
                                let f = i as f32 / 40.0;
                                [
                                    x_wave + f * 90.0,
                                    y_poly + (f * std::f32::consts::TAU).sin() * 22.0,
                                ]
                            })
                            .collect(),
                    )
                    .with_thickness(5.0)
                    .with_colour([0.3, 0.9, 0.6, 0.9])
                    .with_cap(viewport_lib::PolylineCap::Round)
                    .with_stroke_pattern(viewport_lib::StrokePattern::Dashed {
                        dash_length: 18.0,
                        gap_length: 9.0,
                        offset: 0.0,
                    })
                    .with_z_order(1),
                );

                let x_dot = x_wave + 90.0 + 44.0;
                polylines.push(
                    viewport_lib::OverlayPolylineItem::new(
                        (0..48)
                            .map(|i| {
                                let a = i as f32 / 48.0 * std::f32::consts::TAU;
                                [x_dot + a.cos() * 26.0, y_poly + a.sin() * 26.0]
                            })
                            .collect(),
                    )
                    .with_thickness(4.0)
                    .with_colour([1.0, 0.75, 0.3, 0.9])
                    .with_closed(true)
                    .with_stroke_pattern(viewport_lib::StrokePattern::Dotted {
                        spacing: 10.0,
                        offset: 0.0,
                    })
                    .with_z_order(1),
                );
            }

            // Closure-generated closed fills: an animated blob resampled every
            // frame with set_points_from_path, and a textured closed path built
            // with closed_from_path using bounds UVs.
            {
                let t = app.ovl_state.start_time.elapsed().as_secs_f32();
                // Clear of the dotted circle's right edge (centre x_poly + 418,
                // radius 26) plus a gap and the blob's own max radius (34).
                let x_blob = x_poly + 512.0;

                // A wobbling closed blob: radius modulated by time so the shape
                // is resampled from the closure each frame.
                let blob_path = move |u: f32| {
                    let a = u * std::f32::consts::TAU;
                    let r = 28.0 + 6.0 * (a * 3.0 + t * 2.0).sin();
                    [x_blob + a.cos() * r, y_poly + a.sin() * r]
                };
                let mut blob = viewport_lib::OverlayPolylineItem::closed_from_path(
                    blob_path,
                    64,
                    Some(OverlayFill::RadialGradient {
                        centre_colour: [0.9, 0.7, 0.2, 0.85],
                        edge_colour: [0.7, 0.2, 0.5, 0.85],
                    }),
                    [1.0, 1.0, 1.0, 0.85],
                    2.0,
                );
                blob.z_order = 1;
                // Redundant here since closed_from_path already sampled the
                // closure, but shows the per-frame refresh entry point.
                blob.set_points_from_path(blob_path, 64);
                polylines.push(blob);

                if let Some(tid) = app.ovl_state.carlgauss_tex_id {
                    let x_tex = x_blob + 90.0;
                    let mut textured = viewport_lib::OverlayPolylineItem::closed_from_path(
                        |u| {
                            let a = u * std::f32::consts::TAU;
                            // Rounded pentagon.
                            let r = 30.0;
                            let a5 = (a * 5.0).cos() * 3.0;
                            [x_tex + a.cos() * (r + a5), y_poly + a.sin() * (r + a5)]
                        },
                        60,
                        Some(OverlayFill::Solid([1.0, 1.0, 1.0, 0.95])),
                        [1.0, 0.9, 0.7, 0.9],
                        2.0,
                    );
                    // Leaving uvs None maps the path bounds to [0, 1] UVs.
                    textured.texture = Some(tid);
                    textured.z_order = 1;
                    polylines.push(textured);
                }
            }
        }

        // ---------------------------------------------------------------------------
        // New feature row: stacked shadow layers, rotation pivot, texture flip.
        // ---------------------------------------------------------------------------
        {
            let t = app.ovl_state.start_time.elapsed().as_secs_f32();
            let row7_h = 70.0_f32;
            let y7_mid = 20.0
                + row_h
                + 24.0
                + 90.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0
                + 70.0
                + 24.0
                + 90.0
                + 24.0
                + row7_h * 0.5;
            let mut x7 = 20.0_f32;

            // Stacked shadow layers: two offset coloured outer glows (a red
            // halo pushed down-right and a blue halo pushed up-left) make the
            // stacking obvious at a glance, plus a dark inner shadow for a
            // recessed feel. One item carrying several distinct shadow layers.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: cr },
                    [x7, y7_mid - row7_h * 0.5],
                    [120.0, row7_h],
                )
                .with_fill(OverlayFill::Solid([0.16, 0.17, 0.22, 1.0]))
                .with_border([0.7, 0.72, 0.8, 0.9], 1.0)
                .with_shadows(vec![
                    viewport_lib::ShadowLayer::new([0.95, 0.25, 0.2, 0.85], 20.0, [16.0, 14.0]),
                    viewport_lib::ShadowLayer::new([0.2, 0.5, 1.0, 0.85], 20.0, [-16.0, -14.0]),
                ])
                .with_inner_shadows(vec![viewport_lib::ShadowLayer::new(
                    [0.0, 0.0, 0.0, 0.55],
                    14.0,
                    [0.0, 5.0],
                )]),
            );
            x7 += 120.0 + gap + 48.0;

            // Rotation pivot: a metronome. The capsule pivots about its bottom
            // end (not its centre) and swings side to side. The bounding quad
            // grows to contain the swing, so nothing is clipped.
            let hand_h = 60.0_f32;
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Capsule,
                    [x7, y7_mid - hand_h * 0.5],
                    [16.0, hand_h],
                )
                .with_fill(OverlayFill::Solid([0.95, 0.75, 0.2, 0.95]))
                .with_border([1.0, 0.9, 0.5, 0.9], bw)
                .with_rotation((t * 2.0).sin() * 0.7)
                .with_rotation_pivot([0.0, hand_h * 0.5]),
            );
            x7 += 16.0 + gap + 48.0;

            // Texture flip / mirror: the Carl Gauss portrait mirrored
            // vertically via the convenience flip helper, no second upload.
            if let Some(tid) = app.ovl_state.carlgauss_tex_id {
                shapes.push(
                    OverlayShapeItem::new(
                        OverlayShape::Rect { corner_radius: cr },
                        [x7, y7_mid - row7_h * 0.5],
                        [110.0, row7_h],
                    )
                    .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]))
                    .with_border([0.8, 0.8, 0.8, 0.9], bw)
                    .with_texture(tid)
                    .with_texture_flip(false, true),
                );
            }
        }

        // Backdrop blur circle (top-right area, 140px : 2x the normal shape size).
        if app.ovl_state.backdrop_blur_radius > 0.0 {
            let t = app.ovl_state.start_time.elapsed().as_secs_f32();
            shapes.push(
                OverlayShapeItem::new(OverlayShape::Circle, [x + gap, 20.0], [140.0, 140.0])
                    .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 0.12]))
                    .with_border([1.0, 1.0, 1.0, 0.3], 1.0)
                    .with_backdrop_blur(app.ovl_state.backdrop_blur_radius),
            );
            // Second frosted panel with backdrop colour filters: desaturated,
            // slightly dimmed, and hue-rotated, the way a cool glass surface
            // reads in AAA UI. The hue drifts over time to make the effect
            // obvious.
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect {
                        corner_radius: 16.0,
                    },
                    [x + gap, 176.0],
                    [140.0, 96.0],
                )
                .with_fill(OverlayFill::Solid([0.4, 0.5, 0.7, 0.14]))
                .with_border([1.0, 1.0, 1.0, 0.25], 1.0)
                .with_backdrop_blur(app.ovl_state.backdrop_blur_radius)
                .with_backdrop_filters(0.35, 0.9, 0.6 * t.sin()),
            );
        }
    }

    (shapes, labels, bar, ruler, polylines)
}
