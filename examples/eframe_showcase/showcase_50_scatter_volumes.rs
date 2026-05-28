//! Showcase 50: Scatter Volumes (participating media).
//!
//! V1 of the volumetric-effects plan. Demonstrates the new `ScatterVolume`
//! item type with a Box ("global fog") and a Sphere ("localized haze")
//! placed in a small corridor scene. Sliders control density and colour;
//! a toggle moves the camera inside the global volume to verify the
//! camera-inside-volume rendering path.

use crate::App;
use eframe::egui;
use viewport_lib::{
    Material, ScatterVolume, ScatterVolumeItem, ViewportRenderer,
    scene::{Scene, aabb::Aabb},
};

pub(crate) struct SvolState {
    pub scene: Scene,
    pub built: bool,

    pub global_enabled: bool,
    pub global_density: f32,
    pub global_colour: [f32; 3],

    pub sphere_enabled: bool,
    pub sphere_density: f32,
    pub sphere_colour: [f32; 3],
    pub sphere_radius: f32,

    pub camera_inside: bool,
    pub show_global_outline: bool,
    pub show_sphere_outline: bool,
}

impl Default for SvolState {
    fn default() -> Self {
        Self {
            scene: Scene::new(),
            built: false,
            global_enabled: true,
            global_density: 0.08,
            global_colour: [0.78, 0.82, 0.88],
            sphere_enabled: true,
            sphere_density: 0.4,
            sphere_colour: [0.95, 0.65, 0.45],
            sphere_radius: 1.6,
            camera_inside: false,
            show_global_outline: false,
            show_sphere_outline: true,
        }
    }
}

impl App {
    pub(crate) fn build_svol_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.svol_state.scene = Scene::new();

        // Floor plane and two walls forming a short corridor centred on the origin.
        let floor = viewport_lib::geometry::primitives::cuboid(20.0, 20.0, 0.2);
        let floor_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &floor)
            .expect("svol floor upload");
        let mut floor_mat = Material::from_colour([0.55, 0.55, 0.55]);
        floor_mat.roughness = 0.9;
        self.svol_state.scene.add_named(
            "Floor",
            Some(floor_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.1)),
            floor_mat,
        );

        // Tall pillars marching down +X to give depth-cueing reference points.
        let pillar = viewport_lib::geometry::primitives::cuboid(0.6, 0.6, 3.0);
        let pillar_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &pillar)
            .expect("svol pillar upload");
        let pillar_mat = Material::from_colour([0.78, 0.74, 0.66]);
        for i in 0..6 {
            let x = -6.0 + i as f32 * 2.5;
            for side in [-1.0_f32, 1.0] {
                self.svol_state.scene.add_named(
                    "Pillar",
                    Some(pillar_id),
                    glam::Mat4::from_translation(glam::Vec3::new(x, side * 2.5, 1.5)),
                    pillar_mat.clone(),
                );
            }
        }

        // A bright sphere mid-corridor as a colour reference (gets veiled by fog).
        let sphere = viewport_lib::geometry::primitives::sphere(0.8, 32, 16);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("svol sphere upload");
        let mut hot_mat = Material::from_colour([1.0, 0.5, 0.2]);
        hot_mat.roughness = 0.4;
        self.svol_state.scene.add_named(
            "Reference sphere",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(2.0, 0.0, 1.5)),
            hot_mat,
        );

        self.svol_state.built = true;
    }

    /// Push scatter volumes onto the frame based on UI state.
    pub(crate) fn submit_svol_volumes(&self, fd: &mut viewport_lib::FrameData) {
        let s = &self.svol_state;
        if s.global_enabled {
            // When the camera-inside toggle is on, lift the box ceiling so it
            // encloses the eye. When off, cap the box below the typical orbit
            // height so the eye sits outside and looks down into the fog.
            let (z_min, z_max) = if s.camera_inside {
                (-1.0_f32, 10.0)
            } else {
                (-1.0_f32, 2.5)
            };
            let v = ScatterVolume::box_uniform(
                Aabb {
                    min: glam::Vec3::new(-12.0, -12.0, z_min),
                    max: glam::Vec3::new(12.0, 12.0, z_max),
                },
                s.global_density,
                s.global_colour,
            );
            let mut item = ScatterVolumeItem::new(v);
            item.settings.selected = s.show_global_outline;
            fd.scene.scatter_volumes.push(item);
        }
        if s.sphere_enabled {
            let v = ScatterVolume::sphere_uniform(
                [4.0, 0.0, 1.8],
                s.sphere_radius,
                s.sphere_density,
                s.sphere_colour,
            );
            let mut item = ScatterVolumeItem::new(v);
            item.settings.selected = s.show_sphere_outline;
            fd.scene.scatter_volumes.push(item);
        }
    }
}

pub(crate) fn controls_svol(app: &mut App, ui: &mut egui::Ui) {
    let s = &mut app.svol_state;

    ui.heading("Global fog (Box)");
    ui.checkbox(&mut s.global_enabled, "Enabled");
    ui.horizontal(|ui| {
        ui.label("Density");
        ui.add(egui::Slider::new(&mut s.global_density, 0.0..=0.4).step_by(0.005));
    });
    ui.horizontal(|ui| {
        ui.label("Colour");
        ui.color_edit_button_rgb(&mut s.global_colour);
    });

    ui.separator();
    ui.heading("Localized haze (Sphere)");
    ui.checkbox(&mut s.sphere_enabled, "Enabled");
    ui.horizontal(|ui| {
        ui.label("Density");
        ui.add(egui::Slider::new(&mut s.sphere_density, 0.0..=1.0).step_by(0.01));
    });
    ui.horizontal(|ui| {
        ui.label("Radius");
        ui.add(egui::Slider::new(&mut s.sphere_radius, 0.5..=4.0).step_by(0.05));
    });
    ui.horizontal(|ui| {
        ui.label("Colour");
        ui.color_edit_button_rgb(&mut s.sphere_colour);
    });

    ui.separator();
    ui.checkbox(&mut s.camera_inside, "Camera inside global volume");
    ui.checkbox(&mut s.show_global_outline, "Show global outline");
    ui.checkbox(&mut s.show_sphere_outline, "Show sphere outline");
    ui.label(
        "V1 ships uniform density + flat colour + camera-inside handling.\n\
         Lighting (V2), ramps + emission (V3), and noise (V4) are future phases.",
    );
}
