// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct OvlState {
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
    pub show_tex_shapes: bool,
    pub tex_id: Option<viewport_lib::OverlayTextureId>,
    pub carlgauss_tex_id: Option<viewport_lib::OverlayTextureId>,
    pub cloud_positions: Vec<[f32; 3]>,
    pub cloud_scalars: Vec<f32>,
    pub cloud_built: bool,
}

impl Default for OvlState {
    fn default() -> Self {
        Self {
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
            show_tex_shapes: true,
            tex_id: None,
            carlgauss_tex_id: None,
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
    BuiltinColourmap, ColourmapId, LabelAnchor, LabelItem, OverlayFill, OverlayShape,
    OverlayShapeItem, RulerItem, ScalarBarAnchor, ScalarBarItem, ScalarBarOrientation,
    TriangleDirection,
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
    ui.add(egui::Slider::new(&mut app.ovl_state.bg_colour[3], 0.0..=1.0));

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
    ui.checkbox(&mut app.ovl_state.show_tex_shapes, "Show texture-masked shapes");
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
) -> (Vec<OverlayShapeItem>, Vec<LabelItem>, ScalarBarItem, Option<RulerItem>) {
    let colourmap_id = ColourmapId(app.ovl_state.colourmap as usize);

    let bar = ScalarBarItem {
        colourmap_id,
        scalar_min: -1.5,
        scalar_max: 1.5,
        title: Some("Height (m)".into()),
        anchor: app.ovl_state.bar_anchor,
        orientation: app.ovl_state.bar_orientation,
        tick_count: app.ovl_state.tick_count,
        bar_length_px: app.ovl_state.bar_size,
        background_colour: app.ovl_state.bg_colour,
        ..ScalarBarItem::default()
    };

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
            labels.push(LabelItem {
                world_anchor: Some(pos),
                text: text.into(),
                colour,
                leader_line: true,
                leader_colour: [colour[0], colour[1], colour[2], 0.7],
                background: true,
                background_colour: [0.0, 0.0, 0.0, 0.5],
                border_radius: 4.0,
                padding: 4.0,
                anchor_align: LabelAnchor::Leading,
                ..LabelItem::default()
            });
        }
    }

    let ruler = if app.ovl_state.show_ruler {
        Some(RulerItem {
            start: [1.57, 0.0, 0.0],
            end: [-1.57, 3.14, 0.0],
            colour: [1.0, 0.85, 0.2, 1.0],
            label_colour: [1.0, 0.85, 0.2, 1.0],
            label_format: Some("{:.2} m".into()),
            end_caps: true,
            ..RulerItem::default()
        })
    } else {
        None
    };

    // SDF overlay shapes: a row of five shapes, vertically centred on a
    // common midline with equal spacing between them.
    let mut shapes = Vec::new();
    if app.ovl_state.show_shapes {
        let cr = app.ovl_state.shape_corner_radius;
        let bw = app.ovl_state.shape_border_width;

        let row_h = 70.0_f32; // row height all shapes centre on
        let y_mid = 20.0 + row_h * 0.5; // vertical midpoint
        let gap = 16.0_f32; // horizontal gap between shapes
        let mut x = 20.0_f32; // running left edge

        let mut items: Vec<(f32, f32, OverlayShape, [f32; 4], [f32; 4])> = vec![
            // Rounded rect
            (120.0, 70.0,
             OverlayShape::Rect { corner_radius: cr },
             [0.12, 0.12, 0.18, 0.85],
             [0.8, 0.8, 0.8, 0.9]),
            // Per-corner radii rect
            (120.0, 70.0,
             OverlayShape::RoundedRect { radii: [cr, 0.0, cr, 0.0] },
             [0.05, 0.15, 0.25, 0.85],
             [0.3, 0.7, 1.0, 0.9]),
            // Circle
            (70.0, 70.0,
             OverlayShape::Circle,
             [0.2, 0.5, 0.15, 0.85],
             [0.4, 1.0, 0.3, 0.9]),
            // Ellipse
            (120.0, 60.0,
             OverlayShape::Ellipse,
             [0.3, 0.1, 0.3, 0.85],
             [0.8, 0.4, 1.0, 0.9]),
            // Capsule
            (120.0, 40.0,
             OverlayShape::Capsule,
             [0.3, 0.2, 0.05, 0.85],
             [1.0, 0.8, 0.3, 0.9]),
            // Ring
            (70.0, 70.0,
             OverlayShape::Ring { inner_radius_frac: 0.65 },
             [0.15, 0.35, 0.5, 0.85],
             [0.3, 0.8, 1.0, 0.9]),
            // Arc (270-degree sweep)
            (70.0, 70.0,
             OverlayShape::Arc {
                 inner_radius_frac: 0.6,
                 start_angle: 0.0,
                 end_angle: std::f32::consts::PI * 1.5,
             },
             [0.5, 0.2, 0.1, 0.85],
             [1.0, 0.5, 0.2, 0.9]),
            // Triangle (pointing up)
            (60.0, 60.0,
             OverlayShape::Triangle { direction: TriangleDirection::Up },
             [0.4, 0.4, 0.1, 0.85],
             [1.0, 1.0, 0.3, 0.9]),
        ];

        for (w, h, shape, colour, border_colour) in items.drain(..) {
            shapes.push(OverlayShapeItem {
                position: [x, y_mid - h * 0.5],
                size: [w, h],
                shape,
                fill: OverlayFill::Solid(colour),
                border_colour,
                border_width: bw,
                z_order: 0,
                ..Default::default()
            });
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
                shapes.push(OverlayShapeItem {
                    position: [x2, y2_mid - sz * 0.5],
                    size: [sz, sz],
                    shape: OverlayShape::Circle,
                    fill: OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]),
                    border_colour: [1.0, 1.0, 1.0, 0.9],
                    border_width: bw2,
                    texture: Some(tid),
                    ..Default::default()
                });
                x2 += sz + gap;
            }

            // Rounded rect: Carl Gauss portrait.
            if let Some(tid) = app.ovl_state.carlgauss_tex_id {
                shapes.push(OverlayShapeItem {
                    position: [x2, y2_mid - row2_h * 0.5],
                    size: [140.0, row2_h],
                    shape: OverlayShape::Rect { corner_radius: cr },
                    fill: OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]),
                    border_colour: [0.8, 0.8, 0.8, 0.9],
                    border_width: bw2,
                    texture: Some(tid),
                    ..Default::default()
                });
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
            shapes.push(OverlayShapeItem {
                position: [x3, y3_mid - row3_h * 0.5],
                size: [120.0, row3_h],
                shape: OverlayShape::Rect { corner_radius: cr },
                fill: OverlayFill::LinearGradient {
                    start_colour: [0.05, 0.15, 0.55, 0.9],
                    end_colour:   [0.05, 0.65, 0.65, 0.9],
                    angle: 0.0,
                },
                border_colour: [0.3, 0.7, 1.0, 0.8],
                border_width: bw,
                ..Default::default()
            });
            x3 += 120.0 + gap;

            // Per-corner rect: diagonal gradient (PI/4 = 45 degrees).
            shapes.push(OverlayShapeItem {
                position: [x3, y3_mid - row3_h * 0.5],
                size: [120.0, row3_h],
                shape: OverlayShape::RoundedRect { radii: [cr, 0.0, cr, 0.0] },
                fill: OverlayFill::LinearGradient {
                    start_colour: [0.5, 0.05, 0.15, 0.9],
                    end_colour:   [1.0, 0.6, 0.1, 0.9],
                    angle: PI / 4.0,
                },
                border_colour: [1.0, 0.5, 0.2, 0.8],
                border_width: bw,
                ..Default::default()
            });
            x3 += 120.0 + gap;

            // Circle: top-to-bottom (angle = PI/2) dark-to-bright gradient.
            shapes.push(OverlayShapeItem {
                position: [x3, y3_mid - row3_h * 0.5],
                size: [row3_h, row3_h],
                shape: OverlayShape::Circle,
                fill: OverlayFill::LinearGradient {
                    start_colour: [0.05, 0.35, 0.05, 0.9],
                    end_colour:   [0.5, 1.0, 0.3, 0.9],
                    angle: PI / 2.0,
                },
                border_colour: [0.4, 1.0, 0.3, 0.8],
                border_width: bw,
                ..Default::default()
            });
            x3 += row3_h + gap;

            // Ellipse: horizontal gradient, purple-to-pink.
            shapes.push(OverlayShapeItem {
                position: [x3, y3_mid - 30.0],
                size: [120.0, 60.0],
                shape: OverlayShape::Ellipse,
                fill: OverlayFill::LinearGradient {
                    start_colour: [0.35, 0.05, 0.55, 0.9],
                    end_colour:   [0.9, 0.3, 0.6, 0.9],
                    angle: 0.0,
                },
                border_colour: [0.8, 0.4, 1.0, 0.8],
                border_width: bw,
                ..Default::default()
            });
            x3 += 120.0 + gap;

            // Capsule: horizontal gradient, dark grey to white.
            shapes.push(OverlayShapeItem {
                position: [x3, y3_mid - 20.0],
                size: [120.0, 40.0],
                shape: OverlayShape::Capsule,
                fill: OverlayFill::LinearGradient {
                    start_colour: [0.15, 0.15, 0.15, 0.9],
                    end_colour:   [0.85, 0.85, 0.85, 0.9],
                    angle: 0.0,
                },
                border_colour: [0.6, 0.6, 0.6, 0.8],
                border_width: bw,
                ..Default::default()
            });
            x3 += 120.0 + gap;

            // Ring: diagonal gradient creates a highlight effect.
            shapes.push(OverlayShapeItem {
                position: [x3, y3_mid - row3_h * 0.5],
                size: [row3_h, row3_h],
                shape: OverlayShape::Ring { inner_radius_frac: 0.65 },
                fill: OverlayFill::LinearGradient {
                    start_colour: [0.1, 0.3, 0.6, 0.9],
                    end_colour:   [0.7, 0.9, 1.0, 0.9],
                    angle: -PI / 4.0,
                },
                border_colour: [0.3, 0.7, 1.0, 0.8],
                border_width: bw,
                ..Default::default()
            });
            x3 += row3_h + gap;

            // Triangle: bottom-to-top warm gradient.
            shapes.push(OverlayShapeItem {
                position: [x3, y3_mid - row3_h * 0.5],
                size: [60.0, row3_h],
                shape: OverlayShape::Triangle { direction: TriangleDirection::Up },
                fill: OverlayFill::LinearGradient {
                    start_colour: [0.7, 0.15, 0.05, 0.9],
                    end_colour:   [1.0, 0.9, 0.1, 0.9],
                    angle: PI / 2.0,
                },
                border_colour: [1.0, 0.6, 0.2, 0.8],
                border_width: bw,
                ..Default::default()
            });
            let _ = x3;
        }
    }

    (shapes, labels, bar, ruler)
}
