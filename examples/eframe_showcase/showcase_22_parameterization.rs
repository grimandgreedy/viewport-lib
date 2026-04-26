//! Showcase 22: Parameterization Visualization.
//!
//! Demonstrates the `ParamVis` API for inspecting UV quality on four mesh types:
//!   - Torus   (z =  4.5)
//!   - Sphere  (z =  1.5)
//!   - Cube    (z = -1.5)
//!   - Plane   (z = -4.5, two-sided)
//!
//! Within each row, columns correspond to the four `ParamVisMode` variants:
//!   Left -> Checker · Grid · LocalChecker · LocalRadial <- Right
//!
//! The controls panel lets you:
//! - Toggle UV-vis on/off to compare patterns against plain PBR shading.
//! - Adjust the tile scale shared across all objects.

use crate::App;
use crate::geometry::{make_box_with_uvs, make_uv_sphere};
use eframe::egui;
use viewport_lib::{
    Material, MeshData, MeshId, ParamVis, ParamVisMode, ViewportRenderer,
    scene::{Scene, material::BackfacePolicy},
};

/// One entry per `ParamVisMode` variant (column order).
const MODES: [(ParamVisMode, &str); 4] = [
    (ParamVisMode::Checker, "Checker"),
    (ParamVisMode::Grid, "Grid"),
    (ParamVisMode::LocalChecker, "LocalChecker"),
    (ParamVisMode::LocalRadial, "LocalRadial"),
];

const X_POSITIONS: [f32; 4] = [-4.5, -1.5, 1.5, 4.5];

// Row Z offsets (front-to-back order in UI: torus at back, plane at front).
const Z_TORUS: f32 = 4.5;
const Z_SPHERE: f32 = 1.5;
const Z_CUBE: f32 = -1.5;
const Z_PLANE: f32 = -4.5;

// Node-ID layout: [0..4] = torus, [4..8] = sphere, [8..12] = cube, [12..16] = plane.
const IDX_TORUS: usize = 0;
const IDX_SPHERE: usize = 4;
const IDX_CUBE: usize = 8;
const IDX_PLANE: usize = 12;

impl App {
    /// Build Showcase 22: Parameterization Visualization.
    ///
    /// Four rows × four columns = 16 scene nodes.
    /// Each object needs its own uploaded mesh slot for independent per-object GPU state.
    pub(crate) fn build_param_vis_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.param_vis_scene = Scene::new();

        let torus_data = viewport_lib::primitives::torus(1.1, 0.45, 48, 24);
        let sphere_data = make_uv_sphere(48, 24, 1.0);
        let cube_data = make_box_with_uvs(1.6, 1.6, 1.6);
        let plane_data = viewport_lib::primitives::plane(2.8, 2.8);

        let upload_mesh = |renderer: &mut ViewportRenderer, data: &MeshData| -> MeshId {
            MeshId::from_index(
                renderer
                    .resources_mut()
                    .upload_mesh_data(&self.device, data)
                    .expect("param-vis mesh upload"),
            )
        };

        let scale = self.param_vis_scale;

        let mut add_row = |scene: &mut Scene,
                           renderer: &mut ViewportRenderer,
                           mesh_data: &MeshData,
                           z: f32,
                           base_idx: usize,
                           node_ids: &mut [u64; 16],
                           color: [f32; 3],
                           two_sided: bool| {
            for (col, (mode, label)) in MODES.iter().enumerate() {
                let mesh_id = upload_mesh(renderer, mesh_data);
                let mat = {
                    let mut m = Material::pbr(color, 0.0, 0.4);
                    m.param_vis = Some(ParamVis { mode: *mode, scale });
                    m.backface_policy = if two_sided {
                        BackfacePolicy::Identical
                    } else {
                        BackfacePolicy::Cull
                    };
                    m
                };
                let node_id = scene.add_named(
                    *label,
                    Some(mesh_id),
                    glam::Mat4::from_translation(glam::Vec3::new(X_POSITIONS[col], 0.0, z)),
                    mat,
                );
                node_ids[base_idx + col] = node_id;
            }
        };

        add_row(
            &mut self.param_vis_scene,
            renderer,
            &torus_data,
            Z_TORUS,
            IDX_TORUS,
            &mut self.param_vis_node_ids,
            [0.55, 0.70, 0.65],
            false,
        );
        add_row(
            &mut self.param_vis_scene,
            renderer,
            &sphere_data,
            Z_SPHERE,
            IDX_SPHERE,
            &mut self.param_vis_node_ids,
            [0.7, 0.7, 0.7],
            false,
        );
        add_row(
            &mut self.param_vis_scene,
            renderer,
            &cube_data,
            Z_CUBE,
            IDX_CUBE,
            &mut self.param_vis_node_ids,
            [0.72, 0.65, 0.55],
            false,
        );
        add_row(
            &mut self.param_vis_scene,
            renderer,
            &plane_data,
            Z_PLANE,
            IDX_PLANE,
            &mut self.param_vis_node_ids,
            [0.65, 0.65, 0.80],
            true,
        );

        self.param_vis_built = true;
    }

    /// Push the current scale and on/off state to all 16 param-vis nodes.
    fn update_param_vis_materials(&mut self) {
        let rows: [(usize, [f32; 3], bool); 4] = [
            (IDX_TORUS, [0.55, 0.70, 0.65], false),
            (IDX_SPHERE, [0.7, 0.7, 0.7], false),
            (IDX_CUBE, [0.72, 0.65, 0.55], false),
            (IDX_PLANE, [0.65, 0.65, 0.80], true),
        ];
        for (base_idx, color, two_sided) in rows {
            for (col, (mode, _)) in MODES.iter().enumerate() {
                let mat = {
                    let mut m = Material::pbr(color, 0.0, 0.4);
                    m.param_vis = if self.param_vis_on {
                        Some(ParamVis {
                            mode: *mode,
                            scale: self.param_vis_scale,
                        })
                    } else {
                        None
                    };
                    m.backface_policy = if two_sided {
                        BackfacePolicy::Identical
                    } else {
                        BackfacePolicy::Cull
                    };
                    m
                };
                self.param_vis_scene
                    .set_material(self.param_vis_node_ids[base_idx + col], mat);
            }
        }
    }

    pub(crate) fn controls_param_vis(&mut self, ui: &mut egui::Ui) {
        ui.label("Four mesh rows, four ParamVisMode columns:");
        ui.label("  Back row    : torus");
        ui.label("  ·           : sphere");
        ui.label("  ·           : cube");
        ui.label("  Front row   : plane (two-sided)");
        ui.separator();
        ui.label("Columns (left -> right):");
        ui.label("  Checker · Grid · LocalChecker · LocalRadial");

        ui.separator();

        let vis_changed = ui.checkbox(&mut self.param_vis_on, "UV vis on").changed();

        ui.separator();

        ui.label("Scale (tiles per UV unit):");
        let scale_changed = ui
            .add(
                egui::Slider::new(&mut self.param_vis_scale, 1.0..=32.0)
                    .step_by(0.5)
                    .logarithmic(false),
            )
            .changed();

        if (vis_changed || scale_changed) && self.param_vis_built {
            self.update_param_vis_materials();
        }

        ui.separator();
        ui.weak("Checker: alternating black/white in UV space.");
        ui.weak("Grid: thin lines at UV integer boundaries.");
        ui.weak("LocalChecker: polar checkerboard at UV (0.5, 0.5).");
        ui.weak("LocalRadial: concentric rings at UV (0.5, 0.5).");
    }
}
