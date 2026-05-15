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
    BuiltinColourmap, ColourmapId, LabelAnchor, LabelItem, RulerItem, ScalarBarAnchor, ScalarBarItem,
    ScalarBarOrientation,
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
}

// ---------------------------------------------------------------------------
// Point cloud geometry (built once)
// ---------------------------------------------------------------------------

/// Generate a sinusoidal wave surface sampled on a 70×70 grid.
///
/// x ∈ [-π/2, π/2], y ∈ [0, π], z = sin(x) * 1.5.
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

pub(crate) fn build_overlay_frame(app: &App) -> (Vec<LabelItem>, ScalarBarItem, Option<RulerItem>) {
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

    (labels, bar, ruler)
}
