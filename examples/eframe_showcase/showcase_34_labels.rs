//! Showcase 34: Labels : exploded gearbox assembly with annotated parts
//! and feature demonstration rows showcasing every LabelItem capability.

use crate::App;
use crate::geometry::make_box_with_uvs;
use viewport_lib::{LabelAnchor, LabelItem, Material, ViewportRenderer};

// ---------------------------------------------------------------------------
// Gearbox parts definition
// ---------------------------------------------------------------------------

struct GearboxPart {
    name: &'static str,
    detail: &'static str,
    pos: [f32; 3],
    explode_y: f32,
    size: [f32; 3],
    color: [f32; 3],
    label_color: [f32; 4],
}

const PARTS: &[GearboxPart] = &[
    GearboxPart {
        name: "Upper Casing",
        detail: "Al 6061-T6",
        pos: [0.0, 0.0, 0.0],
        explode_y: 4.5,
        size: [3.0, 0.4, 2.5],
        color: [0.55, 0.58, 0.62],
        label_color: [0.75, 0.78, 0.82, 1.0],
    },
    GearboxPart {
        name: "Lower Casing",
        detail: "Al 6061-T6",
        pos: [0.0, 0.0, 0.0],
        explode_y: -4.5,
        size: [3.0, 0.4, 2.5],
        color: [0.50, 0.53, 0.57],
        label_color: [0.75, 0.78, 0.82, 1.0],
    },
    GearboxPart {
        name: "Gasket",
        detail: "Viton, 1.5 mm",
        pos: [0.0, 0.0, 0.0],
        explode_y: 3.0,
        size: [2.8, 0.06, 2.3],
        color: [0.15, 0.15, 0.15],
        label_color: [0.6, 0.6, 0.6, 1.0],
    },
    GearboxPart {
        name: "Input Shaft",
        detail: "4340 Steel, HRC 58",
        pos: [-0.8, 0.0, 0.0],
        explode_y: 0.0,
        size: [0.25, 3.5, 0.25],
        color: [0.7, 0.72, 0.75],
        label_color: [1.0, 0.85, 0.4, 1.0],
    },
    GearboxPart {
        name: "Output Shaft",
        detail: "4340 Steel, HRC 58",
        pos: [0.8, 0.0, 0.0],
        explode_y: 0.0,
        size: [0.25, 3.5, 0.25],
        color: [0.65, 0.67, 0.70],
        label_color: [1.0, 0.85, 0.4, 1.0],
    },
    GearboxPart {
        name: "Pinion (Z=16)",
        detail: "3.5:1 ratio, Module 2",
        pos: [-0.8, 0.4, 0.0],
        explode_y: 1.8,
        size: [0.6, 0.3, 0.6],
        color: [0.9, 0.75, 0.2],
        label_color: [1.0, 0.9, 0.3, 1.0],
    },
    GearboxPart {
        name: "Spur Gear (Z=56)",
        detail: "3.5:1 ratio, Module 2",
        pos: [0.8, 0.4, 0.0],
        explode_y: 1.8,
        size: [1.2, 0.3, 1.2],
        color: [0.85, 0.70, 0.15],
        label_color: [1.0, 0.9, 0.3, 1.0],
    },
    GearboxPart {
        name: "Idler Gear (Z=24)",
        detail: "Hardened 8620",
        pos: [0.0, -0.3, 0.0],
        explode_y: -1.8,
        size: [0.8, 0.25, 0.8],
        color: [0.80, 0.65, 0.10],
        label_color: [1.0, 0.9, 0.3, 1.0],
    },
    GearboxPart {
        name: "Bearing A",
        detail: "6205-2RS, Input",
        pos: [-0.8, 1.2, 0.0],
        explode_y: 2.5,
        size: [0.45, 0.18, 0.45],
        color: [0.3, 0.65, 0.9],
        label_color: [0.4, 0.75, 1.0, 1.0],
    },
    GearboxPart {
        name: "Bearing B",
        detail: "6205-2RS, Output",
        pos: [0.8, 1.2, 0.0],
        explode_y: 2.5,
        size: [0.45, 0.18, 0.45],
        color: [0.3, 0.65, 0.9],
        label_color: [0.4, 0.75, 1.0, 1.0],
    },
    GearboxPart {
        name: "Mounting Flange",
        detail: "M8 x 6 bolt pattern",
        pos: [0.0, -1.6, 0.0],
        explode_y: -6.0,
        size: [3.4, 0.2, 2.8],
        color: [0.45, 0.48, 0.52],
        label_color: [0.7, 0.7, 0.75, 1.0],
    },
];

// ---------------------------------------------------------------------------
// Build scene (geometry + world-anchored part labels only)
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_labels_scene(&mut self, renderer: &mut ViewportRenderer) {
        use viewport_lib::scene::Scene;

        self.lbl_scene = Scene::new();
        self.lbl_labels = Vec::new();

        for part in PARTS {
            let mesh = make_box_with_uvs(part.size[0], part.size[1], part.size[2]);
            let id = renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &mesh)
                .expect("gearbox part upload");

            let exploded_pos = [
                part.pos[0],
                part.pos[1] + part.explode_y,
                part.pos[2],
            ];

            self.lbl_scene.add_named(
                part.name,
                Some(id),
                glam::Mat4::from_translation(glam::Vec3::from(exploded_pos)),
                Material::from_color(part.color),
            );

            self.lbl_labels.push(LabelItem {
                world_anchor: Some(exploded_pos),
                text: format!("{}: {}", part.name, part.detail),
                color: part.label_color,
                font_size: 12.0,
                leader_line: true,
                background: true,
                background_color: [0.05, 0.05, 0.1, 0.75],
                leader_color: [
                    part.label_color[0],
                    part.label_color[1],
                    part.label_color[2],
                    0.35,
                ],
                z_order: 0,
                ..Default::default()
            });
        }

        self.lbl_built = true;
    }

    /// Generate screen-anchored labels (title, legend, feature demos) sized to
    /// the actual viewport dimensions.  Called each frame from `build_frame_data`.
    pub(crate) fn build_label_screen_overlays(&self, vp_w: f32, vp_h: f32) -> Vec<LabelItem> {
        let mut out = Vec::new();
        let demo_bg = [0.08, 0.08, 0.15, 0.7];
        let cx = vp_w * 0.5; // viewport centre X

        // -- Title (centered at top) --
        if self.lbl_show_hud_labels {
            out.push(LabelItem {
                screen_anchor: Some([cx, 36.0]),
                text: "Gearbox Assembly: Exploded View".into(),
                color: [1.0, 1.0, 1.0, 1.0],
                font_size: 48.0,
                background: true,
                background_color: [0.1, 0.1, 0.2, 0.0],
                border_radius: 4.0,
                anchor_align: LabelAnchor::Center,
                z_order: 200,
                ..Default::default()
            });

            // Legend (centered at bottom).
            out.push(LabelItem {
                screen_anchor: Some([cx, vp_h - 24.0]),
                text: "Grey=Casing  Gold=Gears  Blue=Bearings  Yellow=Shafts".into(),
                color: [0.8, 0.8, 0.8, 1.0],
                font_size: 11.0,
                background: true,
                background_color: [0.0, 0.0, 0.0, 0.65],
                border_radius: 3.0,
                anchor_align: LabelAnchor::Center,
                z_order: 200,
                ..Default::default()
            });
        }

        if !self.lbl_show_feature_demos {
            return out;
        }

        // =================================================================
        // Feature demo rows (z_order = 100)
        // Positioned along the left edge of the viewport in a column.
        // =================================================================

        let col = 16.0; // left margin
        let heading_color = [0.6, 0.7, 0.8, 1.0];

        // Helper: section heading.
        let heading = |y: f32, text: &str| -> LabelItem {
            LabelItem {
                screen_anchor: Some([col, y]),
                text: text.into(),
                color: heading_color,
                font_size: 11.0,
                z_order: 100,
                ..Default::default()
            }
        };

        // -- Row 1: Anchor alignment --
        let mut y = 50.0;
        out.push(heading(y, "anchor_align:"));
        y += 16.0;
        for (i, (align, name)) in [
            (LabelAnchor::Leading, "Leading"),
            (LabelAnchor::Center, "Center"),
            (LabelAnchor::Trailing, "Trailing"),
        ]
        .iter()
        .enumerate()
        {
            out.push(LabelItem {
                screen_anchor: Some([col + 90.0, y + i as f32 * 20.0]),
                text: format!("{} aligned", name),
                color: [0.9, 0.95, 1.0, 1.0],
                font_size: 12.0,
                background: true,
                background_color: demo_bg,
                anchor_align: *align,
                z_order: 100,
                ..Default::default()
            });
        }

        // -- Row 2: Opacity --
        y += 78.0;
        out.push(heading(y, "opacity:"));
        y += 16.0;
        for (i, opacity) in [1.0f32, 0.75, 0.5, 0.25].iter().enumerate() {
            out.push(LabelItem {
                screen_anchor: Some([col + i as f32 * 65.0, y]),
                text: format!("{:.0}%", opacity * 100.0),
                color: [1.0, 0.8, 0.3, 1.0],
                font_size: 13.0,
                background: true,
                background_color: demo_bg,
                opacity: *opacity,
                z_order: 100,
                ..Default::default()
            });
        }

        // -- Row 3: Offset --
        y += 34.0;
        out.push(heading(y, "offset:"));
        y += 16.0;
        for (i, (off, desc)) in [
            ([0.0f32, 0.0], "[0,0]"),
            ([20.0, 0.0], "[20,0]"),
            ([0.0, 12.0], "[0,12]"),
            ([20.0, -10.0], "[20,-10]"),
        ]
        .iter()
        .enumerate()
        {
            out.push(LabelItem {
                screen_anchor: Some([col + i as f32 * 65.0, y]),
                text: (*desc).into(),
                color: [0.6, 1.0, 0.7, 1.0],
                font_size: 11.0,
                background: true,
                background_color: demo_bg,
                offset: *off,
                z_order: 100,
                ..Default::default()
            });
        }

        // -- Row 4: Max width (word wrap) --
        y += 34.0;
        out.push(heading(y, "max_width:"));
        y += 16.0;
        let wrap_text = "The quick brown fox jumps over the lazy dog";
        for (i, max_w) in [None, Some(140.0f32), Some(90.0)].iter().enumerate() {
            let label_text = match max_w {
                None => format!("None: {}", wrap_text),
                Some(w) => format!("{:.0}px: {}", w, wrap_text),
            };
            out.push(LabelItem {
                screen_anchor: Some([col, y + i as f32 * 55.0]),
                text: label_text,
                color: [1.0, 0.7, 0.9, 1.0],
                font_size: 11.0,
                background: true,
                background_color: demo_bg,
                max_width: *max_w,
                z_order: 100,
                ..Default::default()
            });
        }

        // -- Row 5: Border radius --
        y += 170.0;
        out.push(heading(y, "border_radius:"));
        y += 16.0;
        for (i, radius) in [0.0f32, 3.0, 6.0, 12.0].iter().enumerate() {
            out.push(LabelItem {
                screen_anchor: Some([col + i as f32 * 65.0, y]),
                text: format!("{:.0}px", radius),
                color: [0.8, 0.85, 1.0, 1.0],
                font_size: 12.0,
                background: true,
                background_color: [0.15, 0.15, 0.3, 0.8],
                border_radius: *radius,
                z_order: 100,
                ..Default::default()
            });
        }

        // -- Row 6: Padding --
        y += 34.0;
        out.push(heading(y, "padding:"));
        y += 16.0;
        for (i, pad) in [0.0f32, 3.0, 8.0, 16.0].iter().enumerate() {
            out.push(LabelItem {
                screen_anchor: Some([col + i as f32 * 72.0, y]),
                text: format!("{:.0}px", pad),
                color: [0.9, 1.0, 0.8, 1.0],
                font_size: 12.0,
                background: true,
                background_color: [0.15, 0.3, 0.15, 0.8],
                padding: *pad,
                z_order: 100,
                ..Default::default()
            });
        }

        // -- Row 7: Font size --
        y += 34.0;
        out.push(heading(y, "font_size:"));
        y += 16.0;
        for (i, size) in [10.0f32, 13.0, 16.0, 20.0].iter().enumerate() {
            out.push(LabelItem {
                screen_anchor: Some([col + i as f32 * 65.0, y]),
                text: format!("{:.0}px", size),
                color: [1.0, 1.0, 1.0, 1.0],
                font_size: *size,
                background: true,
                background_color: demo_bg,
                z_order: 100,
                ..Default::default()
            });
        }

        // -- Row 8: Z-order (overlapping labels) --
        y += 38.0;
        out.push(heading(y, "z_order (overlap):"));
        y += 16.0;
        out.push(LabelItem {
            screen_anchor: Some([col, y]),
            text: "z=98 (back)".into(),
            color: [1.0, 1.0, 1.0, 1.0],
            font_size: 12.0,
            background: true,
            background_color: [0.6, 0.15, 0.15, 0.9],
            border_radius: 3.0,
            z_order: 98,
            ..Default::default()
        });
        out.push(LabelItem {
            screen_anchor: Some([col + 45.0, y + 6.0]),
            text: "z=99 (mid)".into(),
            color: [1.0, 1.0, 1.0, 1.0],
            font_size: 12.0,
            background: true,
            background_color: [0.15, 0.5, 0.15, 0.9],
            border_radius: 3.0,
            z_order: 99,
            ..Default::default()
        });
        out.push(LabelItem {
            screen_anchor: Some([col + 90.0, y + 12.0]),
            text: "z=100 (front)".into(),
            color: [1.0, 1.0, 1.0, 1.0],
            font_size: 12.0,
            background: true,
            background_color: [0.15, 0.15, 0.6, 0.9],
            border_radius: 3.0,
            z_order: 100,
            ..Default::default()
        });

        out
    }

    pub(crate) fn controls_labels(&mut self, ui: &mut eframe::egui::Ui) {
        ui.label("Exploded gearbox assembly with");
        ui.label("native overlay label features.");
        ui.separator();

        ui.checkbox(&mut self.lbl_show_part_labels, "Part labels");
        ui.checkbox(&mut self.lbl_show_feature_demos, "Feature demos");
        ui.checkbox(&mut self.lbl_show_hud_labels, "Title + legend");

        ui.separator();
        ui.label("Feature rows (left side):");
        ui.label("  anchor_align: Leading/Center/Trailing");
        ui.label("  opacity: 100% -> 25%");
        ui.label("  offset: pixel nudge from anchor");
        ui.label("  max_width: word wrapping");
        ui.label("  border_radius: rounded corners");
        ui.label("  padding: 0 -> 16px");
        ui.label("  font_size: 10 -> 20px");
        ui.label("  z_order: overlapping draw order");

        ui.separator();
        ui.label("Parts (world-anchored):");
        ui.label("  Casing, gasket, shafts, gears,");
        ui.label("  bearings, mounting flange");
    }
}
