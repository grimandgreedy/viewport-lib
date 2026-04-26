//! Showcase 24: Surface Appearance : BackfacePolicy.
//!
//! Three rows of shapes each use a different [`BackfacePolicy`] per column:
//! - **Top row : Toruses** clipped through the ring.
//! - **Middle row : Spheres** clipped through the center.
//! - **Bottom row : Cones** clipped through the middle.
//! - **Bottom-most row : Springs** clipped through the coils.
//!
//! Columns:
//! - **Left : Cull** (default): back faces are invisible. The clipped interior is hollow.
//! - **Centre : Identical**: back faces are shaded the same as front faces.
//! - **Right : DifferentColor**: back faces are shaded red, front faces gray.

use crate::App;
use eframe::egui;
use glam::Mat4;
use viewport_lib::{
    BackfacePolicy, ClipObject, LightSource, LightingSettings, Material, MeshId,
    SceneRenderItem, ViewportRenderer, scene::Scene,
};

/// The three backface policies demonstrated in each column.
const POLICIES: [(BackfacePolicy, &str); 3] = [
    (BackfacePolicy::Cull, "Cull"),
    (BackfacePolicy::Identical, "Identical"),
    (BackfacePolicy::DifferentColor([1.0, 0.1, 0.1]), "DifferentColor"),
];

/// X positions for the three columns.
const COL_X: [f32; 3] = [-3.0, 0.0, 3.0];

fn make_material(policy: BackfacePolicy) -> Material {
    let mut mat = Material::from_color([0.7, 0.7, 0.7]);
    mat.backface_policy = policy;
    mat
}

impl App {
    // -------------------------------------------------------------------------
    // One-time scene build
    // -------------------------------------------------------------------------

    pub(crate) fn build_sa_scene(&mut self, renderer: &mut ViewportRenderer) {
        use viewport_lib::geometry::primitives;

        self.sa_scene = Scene::new();

        // --- Top row : Toruses (z = +3) ---
        let torus_z = 3.0;
        let torus_mesh = primitives::torus(0.8, 0.35, 32, 16);
        for (i, &(policy, label)) in POLICIES.iter().enumerate() {
            let mesh_id = MeshId::from_index(
                renderer
                    .resources_mut()
                    .upload_mesh_data(&self.device, &torus_mesh)
                    .expect("sa torus upload"),
            );
            self.sa_scene.add_named(
                &format!("Torus {label}"),
                Some(mesh_id),
                Mat4::from_translation(glam::Vec3::new(COL_X[i], 0.0, torus_z)),
                make_material(policy),
            );
        }

        // --- Middle row : Spheres (z = 0) ---
        let sphere_mesh = primitives::sphere(1.2, 32, 16);
        for (i, &(policy, label)) in POLICIES.iter().enumerate() {
            let mesh_id = MeshId::from_index(
                renderer
                    .resources_mut()
                    .upload_mesh_data(&self.device, &sphere_mesh)
                    .expect("sa sphere upload"),
            );
            self.sa_scene.add_named(
                &format!("Sphere {label}"),
                Some(mesh_id),
                Mat4::from_translation(glam::Vec3::new(COL_X[i], 0.0, 0.0)),
                make_material(policy),
            );
        }

        // --- Bottom row : Cones (z = -3) ---
        let cone_z = -3.0;
        let cone_mesh = primitives::cone(0.9, 2.0, 32);
        for (i, &(policy, label)) in POLICIES.iter().enumerate() {
            let mesh_id = MeshId::from_index(
                renderer
                    .resources_mut()
                    .upload_mesh_data(&self.device, &cone_mesh)
                    .expect("sa cone upload"),
            );
            self.sa_scene.add_named(
                &format!("Cone {label}"),
                Some(mesh_id),
                Mat4::from_translation(glam::Vec3::new(COL_X[i], 0.0, cone_z)),
                make_material(policy),
            );
        }

        // --- Bottom-most row : Springs (z = -6) ---
        let spring_z = -6.0;
        let spring_mesh = primitives::spring(0.6, 0.2, 3.0, 16);
        for (i, &(policy, label)) in POLICIES.iter().enumerate() {
            let mesh_id = MeshId::from_index(
                renderer
                    .resources_mut()
                    .upload_mesh_data(&self.device, &spring_mesh)
                    .expect("sa spring upload"),
            );
            self.sa_scene.add_named(
                &format!("Spring {label}"),
                Some(mesh_id),
                Mat4::from_translation(glam::Vec3::new(COL_X[i], 0.0, spring_z)),
                make_material(policy),
            );
        }

        self.sa_built = true;
    }

    // -------------------------------------------------------------------------
    // Controls panel
    // -------------------------------------------------------------------------

    pub(crate) fn controls_surface_appearance(&mut self, ui: &mut egui::Ui) {
        ui.label("BackfacePolicy (clip plane reveals inside):");
        ui.indent("bp_desc", |ui| {
            ui.label("Left:   Cull (interior hollow)");
            ui.label("Centre: Identical (lit normally)");
            ui.label("Right:  DifferentColor (red/orange)");
        });
        ui.separator();
        ui.label("Toruses | Spheres | Cones | Springs (top to bottom)");
        ui.separator();

        ui.checkbox(&mut self.sa_clip_on, "Clip plane (y = 0)");
        ui.label("Slices all shapes to reveal\nhow each policy treats back faces.");
    }

    // -------------------------------------------------------------------------
    // Per-frame helpers (called from build_frame_data)
    // -------------------------------------------------------------------------

    pub(crate) fn sa_scene_items(&mut self) -> Vec<SceneRenderItem> {
        self.sa_scene
            .collect_render_items(&viewport_lib::Selection::new())
    }

    pub(crate) fn sa_clip_objects(&self) -> Vec<ClipObject> {
        if self.sa_clip_on {
            // Clip along Y so the plane slices through all three rows equally,
            // revealing the interior of every shape.
            vec![ClipObject::plane([0.0, 1.0, 0.0], 0.0)]
        } else {
            vec![]
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
