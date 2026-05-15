//! Showcase 24: Surface Appearance : BackfacePolicy.
//!
//! Four rows of shapes demonstrate every [`BackfacePolicy`] variant:
//! - **Row 1 : Toruses** clipped through the ring.
//! - **Row 2 : Spheres** clipped through the center.
//! - **Row 3 : Cones** clipped through the middle.
//! - **Row 4 : Springs** clipped through the coils.
//! - **Row 5 : Spheres** (normal winding) -- zoom inside to see back faces with each policy.
//!
//! Columns (left to right):
//! - **Cull** (default): back faces invisible, interior hollow.
//! - **Identical**: back faces shaded the same as front faces.
//! - **DifferentColor**: back faces shaded red.
//! - **Tint**: back faces darkened by 40%.
//! - **Checker**: checker pattern on back faces.
//! - **Hatching**: diagonal hatching on back faces.
//! - **Crosshatch**: crosshatch pattern on back faces.
//! - **Stripes**: horizontal stripes on back faces.

use crate::App;
use eframe::egui;
use glam::Mat4;
use viewport_lib::{
    BackfacePattern, BackfacePolicy, ClipObject, LightSource, LightingSettings, Material,
    PatternConfig, SceneRenderItem, ViewportRenderer, scene::Scene,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct SaState {
    pub scene: Scene,
    pub built: bool,
    pub clip_on: bool,
}

impl Default for SaState {
    fn default() -> Self {
        Self {
            scene: Scene::new(),
            built: false,
            clip_on: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

/// All backface policies demonstrated, one per column.
fn policies() -> Vec<(BackfacePolicy, &'static str)> {
    vec![
        (BackfacePolicy::Cull, "Cull"),
        (BackfacePolicy::Identical, "Identical"),
        (
            BackfacePolicy::DifferentColor([1.0, 0.1, 0.1]),
            "DifferentColor",
        ),
        (BackfacePolicy::Tint(0.4), "Tint"),
        (
            BackfacePolicy::Pattern(PatternConfig {
                pattern: BackfacePattern::Checker,
                color: [0.9, 0.2, 0.1],
                ..Default::default()
            }),
            "Checker",
        ),
        (
            BackfacePolicy::Pattern(PatternConfig {
                pattern: BackfacePattern::Hatching,
                color: [0.1, 0.5, 0.9],
                ..Default::default()
            }),
            "Hatching",
        ),
        (
            BackfacePolicy::Pattern(PatternConfig {
                pattern: BackfacePattern::Crosshatch,
                color: [0.1, 0.7, 0.2],
                ..Default::default()
            }),
            "Crosshatch",
        ),
        (
            BackfacePolicy::Pattern(PatternConfig {
                pattern: BackfacePattern::Stripes,
                color: [0.8, 0.6, 0.1],
                ..Default::default()
            }),
            "Stripes",
        ),
    ]
}

fn make_material(policy: BackfacePolicy) -> Material {
    let mut mat = Material::from_color([0.7, 0.7, 0.7]);
    mat.backface_policy = policy;
    mat
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    // -------------------------------------------------------------------------
    // One-time scene build
    // -------------------------------------------------------------------------

    pub(crate) fn build_sa_scene(&mut self, renderer: &mut ViewportRenderer) {
        use viewport_lib::geometry::primitives;

        self.sa_state.scene = Scene::new();
        let policies = policies();
        let col_count = policies.len();
        let col_x: Vec<f32> = (0..col_count)
            .map(|i| (i as f32 - (col_count - 1) as f32 / 2.0) * 3.0)
            .collect();

        // --- Row 1 : Toruses (z = +4.5) ---
        let torus_mesh = primitives::torus(0.8, 0.35, 32, 16);
        for (i, (policy, label)) in policies.iter().enumerate() {
            let mesh_id = renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &torus_mesh)
                .expect("sa torus upload");
            self.sa_state.scene.add_named(
                &format!("Torus {label}"),
                Some(mesh_id),
                Mat4::from_translation(glam::Vec3::new(col_x[i], 0.0, 4.5)),
                make_material(*policy),
            );
        }

        // --- Row 2 : Spheres (z = +1.5) ---
        let sphere_mesh = primitives::sphere(1.2, 32, 16);
        for (i, (policy, label)) in policies.iter().enumerate() {
            let mesh_id = renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &sphere_mesh)
                .expect("sa sphere upload");
            self.sa_state.scene.add_named(
                &format!("Sphere {label}"),
                Some(mesh_id),
                Mat4::from_translation(glam::Vec3::new(col_x[i], 0.0, 1.5)),
                make_material(*policy),
            );
        }

        // --- Row 3 : Cones (z = -1.5) ---
        let cone_mesh = primitives::cone(0.9, 2.0, 32);
        for (i, (policy, label)) in policies.iter().enumerate() {
            let mesh_id = renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &cone_mesh)
                .expect("sa cone upload");
            self.sa_state.scene.add_named(
                &format!("Cone {label}"),
                Some(mesh_id),
                Mat4::from_translation(glam::Vec3::new(col_x[i], 0.0, -1.5)),
                make_material(*policy),
            );
        }

        // --- Row 4 : Springs (z = -4.5) ---
        let spring_mesh = primitives::spring(0.6, 0.2, 3.0, 16);
        for (i, (policy, label)) in policies.iter().enumerate() {
            let mesh_id = renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &spring_mesh)
                .expect("sa spring upload");
            self.sa_state.scene.add_named(
                &format!("Spring {label}"),
                Some(mesh_id),
                Mat4::from_translation(glam::Vec3::new(col_x[i], 0.0, -4.5)),
                make_material(*policy),
            );
        }

        // --- Row 5 : Spheres with inverted winding (z = -7.5) ---
        // Reversed winding makes the exterior surface a back face, so each policy is
        // visible from outside without needing a clip plane. Normals are also flipped
        // so the shader's N=-N restores the correct outward normal for lighting.
        let mut sphere_inner_mesh = primitives::sphere(1.2, 32, 16);
        for chunk in sphere_inner_mesh.indices.chunks_mut(3) {
            chunk.swap(1, 2);
        }
        for n in &mut sphere_inner_mesh.normals {
            *n = [-n[0], -n[1], -n[2]];
        }
        for (i, (policy, label)) in policies.iter().enumerate() {
            let mesh_id = renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &sphere_inner_mesh)
                .expect("sa sphere inv upload");
            self.sa_state.scene.add_named(
                &format!("InnerSphere {label}"),
                Some(mesh_id),
                Mat4::from_translation(glam::Vec3::new(col_x[i], 1.5, -7.5)),
                make_material(*policy),
            );
        }

        self.sa_state.built = true;
    }

    // -------------------------------------------------------------------------
    // Per-frame helpers (called from build_frame_data)
    // -------------------------------------------------------------------------

    pub(crate) fn sa_scene_items(&mut self) -> Vec<SceneRenderItem> {
        self.sa_state
            .scene
            .collect_render_items(&viewport_lib::Selection::new())
    }

    pub(crate) fn sa_clip_objects(&self) -> Vec<ClipObject> {
        if self.sa_state.clip_on {
            vec![ClipObject::plane([0.0, 1.0, 0.0], 0.0)]
        } else {
            vec![]
        }
    }

    /// Draw column labels above each BackfacePolicy column using world-space projection.
    pub(crate) fn draw_sa_labels(&self, ui: &egui::Ui, rect: egui::Rect) {
        let policies = policies();
        let col_count = policies.len();
        let col_x: Vec<f32> = (0..col_count)
            .map(|i| (i as f32 - (col_count - 1) as f32 / 2.0) * 3.0)
            .collect();

        let painter = ui.painter_at(rect);
        let view = self.camera.view_matrix();
        let proj = self.camera.proj_matrix();
        let vp_size = [rect.width(), rect.height()];

        // Project a point just above the top row (z = 4.5 + 1.8 clearance).
        for (i, (_, label)) in policies.iter().enumerate() {
            let world_pos = glam::Vec3::new(col_x[i], 0.0, 6.3);
            let clip = proj * view * world_pos.extend(1.0);
            if clip.w <= 0.0 {
                continue;
            }
            let ndc = glam::Vec3::new(clip.x, clip.y, clip.z) / clip.w;
            if ndc.x.abs() > 1.0 || ndc.y.abs() > 1.0 {
                continue;
            }
            let screen = glam::Vec2::new(
                (ndc.x * 0.5 + 0.5) * vp_size[0],
                (1.0 - (ndc.y * 0.5 + 0.5)) * vp_size[1],
            );
            let pos = egui::pos2(rect.left() + screen.x, rect.top() + screen.y);
            let galley = painter.layout_no_wrap(
                label.to_string(),
                egui::FontId::proportional(13.0),
                egui::Color32::from_rgba_unmultiplied(220, 220, 220, 220),
            );
            let text_pos = pos - egui::vec2(galley.size().x / 2.0, galley.size().y);
            let bg_rect = egui::Rect::from_min_size(
                text_pos - egui::vec2(3.0, 2.0),
                galley.size() + egui::vec2(6.0, 4.0),
            );
            painter.rect_filled(bg_rect, 3.0, egui::Color32::from_black_alpha(120));
            painter.galley(text_pos, galley, egui::Color32::WHITE);
        }
    }

    /// Draw row labels to the left of each object row using world-space projection.
    pub(crate) fn draw_sa_row_labels(&self, ui: &egui::Ui, rect: egui::Rect) {
        let rows: &[(&str, f32, f32)] = &[
            ("Torus", 0.0, 4.5),
            ("Sphere", 0.0, 1.5),
            ("Cone", 0.0, -1.5),
            ("Spring", 0.0, -4.5),
            ("Sphere (inverted winding)", 1.5, -7.5),
        ];

        let policies = policies();
        let col_count = policies.len();
        // Left edge: one half-column to the left of the first column.
        let left_x = (-(col_count as f32 - 1.0) / 2.0) * 3.0 - 2.0;

        let painter = ui.painter_at(rect);
        let view = self.camera.view_matrix();
        let proj = self.camera.proj_matrix();
        let vp_size = [rect.width(), rect.height()];

        for (label, y, z) in rows {
            let world_pos = glam::Vec3::new(left_x, *y, *z);
            let clip = proj * view * world_pos.extend(1.0);
            if clip.w <= 0.0 {
                continue;
            }
            let ndc = glam::Vec3::new(clip.x, clip.y, clip.z) / clip.w;
            if ndc.x.abs() > 1.0 || ndc.y.abs() > 1.0 {
                continue;
            }
            let screen = glam::Vec2::new(
                (ndc.x * 0.5 + 0.5) * vp_size[0],
                (1.0 - (ndc.y * 0.5 + 0.5)) * vp_size[1],
            );
            let pos = egui::pos2(rect.left() + screen.x, rect.top() + screen.y);
            let galley = painter.layout_no_wrap(
                label.to_string(),
                egui::FontId::proportional(13.0),
                egui::Color32::from_rgba_unmultiplied(220, 220, 220, 220),
            );
            let text_pos = pos - egui::vec2(galley.size().x, galley.size().y / 2.0);
            let bg_rect = egui::Rect::from_min_size(
                text_pos - egui::vec2(3.0, 2.0),
                galley.size() + egui::vec2(6.0, 4.0),
            );
            painter.rect_filled(bg_rect, 3.0, egui::Color32::from_black_alpha(120));
            painter.galley(text_pos, galley, egui::Color32::WHITE);
        }
    }

    pub(crate) fn sa_lighting() -> LightingSettings {
        LightingSettings {
            lights: vec![
                LightSource::default(),
                viewport_lib::LightSource {
                    kind: viewport_lib::LightKind::Directional {
                        direction: [-0.5, -0.3, 0.8],
                    },
                    intensity: 0.6,
                    ..viewport_lib::LightSource::default()
                },
            ],
            hemisphere_intensity: 0.3,
            sky_color: [1.0, 1.0, 1.0],
            ground_color: [0.4, 0.4, 0.4],
            ..LightingSettings::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_surface_appearance(app: &mut App, ui: &mut egui::Ui) {
    ui.label("BackfacePolicy controls how back faces are rendered.");
    ui.add_space(4.0);
    ui.indent("bp_desc", |ui| {
        ui.label("Cull (default): back faces invisible. Correct for\nmost closed meshes -- the interior is never seen.");
        ui.add_space(2.0);
        ui.label("Identical: back faces shaded same as front.");
        ui.label("DifferentColor: back faces shaded in a set color.");
        ui.label("Tint: back faces darkened by a factor.");
        ui.label("Pattern: procedural pattern on back faces\n(Checker / Hatching / Crosshatch / Stripes).");
    });
    ui.separator();
    ui.label("Rows (top to bottom):");
    ui.indent("row_desc", |ui| {
        ui.label("Torus, Sphere, Cone, Spring -- clipped at y=0\nto reveal interior back faces.");
        ui.add_space(2.0);
        ui.label("Sphere (inverted winding) -- winding reversed\nso the back face is on the outside, making\neach policy visible without a clip plane.");
    });
    ui.separator();

    ui.checkbox(&mut app.sa_state.clip_on, "Clip plane (y = 0)");
    ui.label("Slices the top four rows to reveal\nhow each policy treats back faces.");
}
