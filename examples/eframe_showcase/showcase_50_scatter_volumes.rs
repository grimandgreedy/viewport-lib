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
    DensityRemap, Emission, EmissionCurve, Material, ScatterVolume, ScatterVolumeItem,
    ViewportRenderer,
    scene::{Scene, aabb::Aabb},
};

pub(crate) struct SvolState {
    pub scene: Scene,
    pub built: bool,

    pub global_enabled: bool,
    pub global_density: f32,
    pub global_colour: [f32; 3],
    pub global_anisotropy: f32,
    pub global_unlit: bool,
    pub global_receive_shadows: bool,
    pub global_soft_edges: bool,
    pub global_soft_lo: f32,
    pub global_soft_hi: f32,

    pub sphere_enabled: bool,
    pub sphere_density: f32,
    pub sphere_colour: [f32; 3],
    pub sphere_radius: f32,
    pub sphere_anisotropy: f32,

    pub fire_enabled: bool,
    pub fire_density: f32,
    pub fire_colour: [f32; 3],
    pub fire_radius: f32,
    pub fire_emission: f32,
    pub fire_falloff: f32,

    pub show_global_outline: bool,
    pub show_sphere_outline: bool,

    pub sun_dir: [f32; 3],
    pub sun_colour: [f32; 3],
    pub sun_intensity: f32,
    pub shadows_enabled: bool,
}

impl Default for SvolState {
    fn default() -> Self {
        Self {
            scene: Scene::new(),
            built: false,
            global_enabled: true,
            global_density: 0.08,
            global_colour: [0.78, 0.82, 0.88],
            global_anisotropy: 0.3,
            global_unlit: false,
            global_receive_shadows: true,
            global_soft_edges: false,
            global_soft_lo: 4.0,
            global_soft_hi: 12.0,
            sphere_enabled: true,
            sphere_density: 0.4,
            sphere_colour: [0.95, 0.65, 0.45],
            sphere_radius: 1.6,
            sphere_anisotropy: 0.0,
            fire_enabled: false,
            fire_density: 1.2,
            fire_colour: [1.0, 0.55, 0.18],
            fire_radius: 1.0,
            fire_emission: 4.0,
            fire_falloff: 1.4,
            show_global_outline: false,
            show_sphere_outline: true,
            sun_dir: [-0.6, 0.2, 0.6],
            sun_colour: [1.0, 0.96, 0.86],
            sun_intensity: 2.0,
            shadows_enabled: true,
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
            let mut v = ScatterVolume::box_uniform(
                Aabb {
                    min: glam::Vec3::new(-12.0, -12.0, -1.0),
                    max: glam::Vec3::new(12.0, 12.0, 6.0),
                },
                s.global_density,
                s.global_colour,
            );
            v.anisotropy = s.global_anisotropy;
            if s.global_soft_edges {
                v.density_remap = DensityRemap::Smoothstep {
                    lo: s.global_soft_lo,
                    hi: s.global_soft_hi,
                };
            }
            let mut item = ScatterVolumeItem::new(v);
            item.settings.selected = s.show_global_outline;
            item.settings.unlit = s.global_unlit;
            item.settings.receive_shadows = s.global_receive_shadows;
            fd.scene.scatter_volumes.push(item);
        }
        if s.fire_enabled {
            let center = [-4.0_f32, 0.0, 1.2];
            let mut v = ScatterVolume::sphere_uniform(
                center,
                s.fire_radius,
                s.fire_density,
                s.fire_colour,
            );
            v.emission = Emission::Strength {
                strength: s.fire_emission,
                curve: EmissionCurve::Power(2.0),
            };
            v.density_remap = DensityRemap::ExpFalloff {
                center,
                falloff: s.fire_falloff,
            };
            let mut item = ScatterVolumeItem::new(v);
            // Fire reads as self-illuminated; nearby surfaces are not lit by
            // the volume itself in V3 (a virtual point light for surface
            // illumination is reserved for a follow-up phase).
            item.settings.unlit = false;
            fd.scene.scatter_volumes.push(item);
        }
        if s.sphere_enabled {
            let mut v = ScatterVolume::sphere_uniform(
                [4.0, 0.0, 1.8],
                s.sphere_radius,
                s.sphere_density,
                s.sphere_colour,
            );
            v.anisotropy = s.sphere_anisotropy;
            let mut item = ScatterVolumeItem::new(v);
            item.settings.selected = s.show_sphere_outline;
            fd.scene.scatter_volumes.push(item);
        }
    }
}

pub(crate) fn controls_svol(app: &mut App, ui: &mut egui::Ui) {
    let s = &mut app.svol_state;

    ui.heading("Sun");
    ui.horizontal(|ui| {
        ui.label("Dir X");
        ui.add(egui::Slider::new(&mut s.sun_dir[0], -1.0..=1.0).step_by(0.01));
    });
    ui.horizontal(|ui| {
        ui.label("Dir Y");
        ui.add(egui::Slider::new(&mut s.sun_dir[1], -1.0..=1.0).step_by(0.01));
    });
    ui.horizontal(|ui| {
        ui.label("Dir Z");
        ui.add(egui::Slider::new(&mut s.sun_dir[2], 0.05..=1.0).step_by(0.01));
    });
    ui.horizontal(|ui| {
        ui.label("Intensity");
        ui.add(egui::Slider::new(&mut s.sun_intensity, 0.0..=5.0).step_by(0.05));
    });
    ui.horizontal(|ui| {
        ui.label("Colour");
        ui.color_edit_button_rgb(&mut s.sun_colour);
    });
    ui.checkbox(&mut s.shadows_enabled, "Cast shadows (god rays)");

    ui.separator();
    ui.heading("Global fog (Box)");
    ui.checkbox(&mut s.global_enabled, "Enabled");
    ui.horizontal(|ui| {
        ui.label("Density");
        ui.add(egui::Slider::new(&mut s.global_density, 0.0..=0.4).step_by(0.005));
    });
    ui.horizontal(|ui| {
        ui.label("Anisotropy g");
        ui.add(egui::Slider::new(&mut s.global_anisotropy, -0.9..=0.9).step_by(0.01));
    });
    ui.horizontal(|ui| {
        ui.label("Colour");
        ui.color_edit_button_rgb(&mut s.global_colour);
    });
    ui.checkbox(&mut s.global_unlit, "Unlit (skip in-scattering)");
    ui.checkbox(&mut s.global_receive_shadows, "Receive shadows (shafts)");
    ui.checkbox(&mut s.global_soft_edges, "Smoothstep soft edges (V3)");
    if s.global_soft_edges {
        ui.horizontal(|ui| {
            ui.label("Inner radius");
            ui.add(egui::Slider::new(&mut s.global_soft_lo, 0.0..=15.0).step_by(0.1));
        });
        ui.horizontal(|ui| {
            ui.label("Outer radius");
            ui.add(egui::Slider::new(&mut s.global_soft_hi, 0.5..=20.0).step_by(0.1));
        });
    }

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
        ui.label("Anisotropy g");
        ui.add(egui::Slider::new(&mut s.sphere_anisotropy, -0.9..=0.9).step_by(0.01));
    });
    ui.horizontal(|ui| {
        ui.label("Colour");
        ui.color_edit_button_rgb(&mut s.sphere_colour);
    });

    ui.separator();
    ui.heading("Fire (Sphere with emission + ExpFalloff)");
    ui.checkbox(&mut s.fire_enabled, "Enabled");
    ui.horizontal(|ui| {
        ui.label("Density");
        ui.add(egui::Slider::new(&mut s.fire_density, 0.0..=4.0).step_by(0.05));
    });
    ui.horizontal(|ui| {
        ui.label("Radius");
        ui.add(egui::Slider::new(&mut s.fire_radius, 0.2..=2.5).step_by(0.05));
    });
    ui.horizontal(|ui| {
        ui.label("Emission");
        ui.add(egui::Slider::new(&mut s.fire_emission, 0.0..=10.0).step_by(0.1));
    });
    ui.horizontal(|ui| {
        ui.label("Falloff");
        ui.add(egui::Slider::new(&mut s.fire_falloff, 0.2..=4.0).step_by(0.05));
    });
    ui.horizontal(|ui| {
        ui.label("Colour");
        ui.color_edit_button_rgb(&mut s.fire_colour);
    });

    ui.separator();
    if ui.button("Teleport camera inside global volume").clicked() {
        app.camera.center = glam::Vec3::new(0.0, 0.0, 2.5);
        app.camera.distance = 0.5;
    }
    if ui.button("Teleport camera outside global volume").clicked() {
        app.camera.center = glam::Vec3::new(0.0, 0.0, 1.5);
        app.camera.distance = 14.0;
    }
    let s = &mut app.svol_state;
    ui.checkbox(&mut s.show_global_outline, "Show global outline");
    ui.checkbox(&mut s.show_sphere_outline, "Show sphere outline");
    ui.label(
        "V3 adds density remaps and emission. Soft-edge fog uses a radial\n\
         smoothstep; fire uses ExpFalloff + Power emission. Colour ramps\n\
         (ColourSource::Ramp) and the fire-as-light virtual-light hook are\n\
         reserved for a follow-up phase.",
    );
}
