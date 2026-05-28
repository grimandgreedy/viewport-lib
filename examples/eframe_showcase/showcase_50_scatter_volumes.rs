//! Showcase 50: Scatter Volumes (participating media).
//!
//! Demonstrates the `ScatterVolume` item type with a Box ("global fog"), a
//! Sphere ("localized haze"), and an optional fire sphere (colour ramp +
//! emission + animated noise) placed in a small corridor scene. Sliders
//! control density, colour, lighting, animation, and the quality settings
//! (step count, half-resolution rendering, temporal accumulation). A toggle
//! moves the camera inside the global volume to verify the
//! camera-inside-volume rendering path.

use crate::App;
use eframe::egui;
use viewport_lib::{
    BuiltinColourmap, ColourSource, DensityRemap, Emission, EmissionCurve, Material, NoiseDriver,
    ScatterQuality, ScatterVolume, ScatterVolumeItem, ViewportRenderer,
    scene::{Scene, aabb::Aabb},
};

/// Scene preset. Each variant is a complete look (volumes, sky, sun) the
/// user can switch between with a single click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SvolPreset {
    FoggyCorridor,
    CloudySky,
    CampfireNight,
    GodRays,
    StressTest,
}

pub(crate) struct SvolState {
    pub scene: Scene,
    pub built: bool,
    pub preset: SvolPreset,

    // Global volume: doubles as the atmospheric setup -- a huge fog box in
    // the fog/god-ray/campfire presets, or a wide flat cloud slab in
    // CloudySky. Configured entirely by `apply_preset`.
    pub global_enabled: bool,
    pub global_min: [f32; 3],
    pub global_max: [f32; 3],
    pub global_density: f32,
    pub global_colour: [f32; 3],
    pub global_anisotropy: f32,
    pub global_unlit: bool,
    pub global_receive_shadows: bool,
    pub global_soft_edges: bool,
    pub global_soft_lo: f32,
    pub global_soft_hi: f32,
    pub global_noise: bool,

    pub sphere_enabled: bool,
    pub sphere_density: f32,
    pub sphere_colour: [f32; 3],
    pub sphere_radius: f32,
    pub sphere_anisotropy: f32,
    pub sphere_use_texture: bool,
    pub sphere_texture_id: Option<viewport_lib::VolumeId>,

    pub fire_enabled: bool,
    pub fire_density: f32,
    pub fire_colour: [f32; 3],
    pub fire_radius: f32,
    pub fire_emission: f32,
    pub fire_falloff: f32,
    pub fire_use_ramp: bool,
    pub fire_ramp_id: viewport_lib::ColourmapId,
    pub fire_animate: bool,
    pub fire_noise_scale: f32,
    pub fire_noise_octaves: u32,
    pub fire_noise_time_scale: f32,
    pub fire_noise_scroll_z: f32,

    pub show_global_outline: bool,
    pub show_sphere_outline: bool,

    pub sun_dir: [f32; 3],
    pub sun_colour: [f32; 3],
    pub sun_intensity: f32,
    pub shadows_enabled: bool,

    /// Hemisphere ambient -- set by the active preset, exposed in Advanced.
    pub sky_colour: [f32; 3],
    pub ground_colour: [f32; 3],
    pub hemisphere_intensity: f32,

    pub quality: ScatterQuality,
    pub blue_noise_jitter: bool,
    pub downsample: bool,
    pub temporal: bool,
    pub temporal_blend: f32,
    pub fire_step_budget_override: bool,
    pub fire_step_budget: u32,
}

impl SvolPreset {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::FoggyCorridor => "Foggy",
            Self::CloudySky => "Cloudy sky",
            Self::CampfireNight => "Campfire",
            Self::GodRays => "God rays",
            Self::StressTest => "Stress test",
        }
    }

    /// Apply this preset to a [`SvolState`], overwriting the scene-dependent
    /// fields. Fire knobs are reset to preset-specific defaults so the
    /// "Campfire" preset shows off the fire and other presets disable it.
    pub(crate) fn apply(self, s: &mut SvolState) {
        s.preset = self;
        // Defaults that most presets share; per-preset overrides follow.
        s.global_enabled = true;
        s.global_min = [-200.0, -200.0, -2.0];
        s.global_max = [200.0, 200.0, 50.0];
        s.global_density = 0.04;
        s.global_colour = [0.78, 0.82, 0.88];
        s.global_anisotropy = 0.3;
        s.global_unlit = false;
        s.global_receive_shadows = true;
        s.global_soft_edges = false;
        s.global_noise = false;
        s.sphere_enabled = false;
        s.sphere_density = 0.4;
        s.sphere_colour = [0.95, 0.65, 0.45];
        s.sphere_radius = 1.6;
        s.sphere_anisotropy = 0.0;
        s.sphere_use_texture = false;
        s.fire_enabled = true;
        s.fire_density = 2.0;
        s.fire_colour = [1.0, 0.55, 0.18];
        s.fire_radius = 2.0;
        s.fire_emission = 4.0;
        s.fire_falloff = 1.2;
        s.fire_use_ramp = true;
        s.fire_animate = true;
        s.fire_noise_scale = 2.5;
        s.fire_noise_octaves = 4;
        s.fire_noise_time_scale = 2.5;
        s.fire_noise_scroll_z = 1.5;
        s.sun_dir = [-0.6, 0.2, 0.6];
        s.sun_colour = [1.0, 0.96, 0.86];
        s.sun_intensity = 2.0;
        s.shadows_enabled = true;
        s.sky_colour = [0.45, 0.55, 0.7];
        s.ground_colour = [0.2, 0.18, 0.15];
        s.hemisphere_intensity = 0.35;
        s.show_sphere_outline = false;
        s.show_global_outline = false;

        match self {
            Self::FoggyCorridor => {
                s.sphere_enabled = true;
                s.global_density = 0.05;
            }
            Self::CloudySky => {
                // Wide, thin cloud slab high above the platform. Low base
                // density keeps the layer wispy; the procedural noise has
                // most of the visual weight, carving puffs and gaps.
                s.global_min = [-300.0, -300.0, 28.0];
                s.global_max = [300.0, 300.0, 38.0];
                s.global_density = 0.05;
                s.global_colour = [1.0, 1.0, 1.0];
                s.global_anisotropy = 0.7;
                s.global_noise = true;
                s.global_receive_shadows = false;
                // Light blue daytime sky.
                s.sky_colour = [0.55, 0.7, 0.9];
                s.ground_colour = [0.35, 0.45, 0.3];
                s.hemisphere_intensity = 0.7;
                s.sun_colour = [1.0, 0.98, 0.92];
                s.sun_intensity = 3.0;
                // Low-angle sun so the forward-scattering anisotropy shows
                // as bright silver-lining on the cloud edges facing the sun.
                s.sun_dir = [-0.7, 0.15, 0.45];
            }
            Self::CampfireNight => {
                s.global_density = 0.025;
                s.global_colour = [0.45, 0.5, 0.62];
                s.fire_enabled = true;
                s.sky_colour = [0.05, 0.08, 0.16];
                s.ground_colour = [0.03, 0.03, 0.05];
                s.hemisphere_intensity = 0.15;
                s.sun_colour = [0.35, 0.4, 0.6];
                s.sun_intensity = 0.25;
                s.sun_dir = [-0.4, 0.3, 0.7];
            }
            Self::GodRays => {
                s.global_density = 0.09;
                s.global_anisotropy = 0.6;
                s.global_receive_shadows = true;
                s.sun_intensity = 3.5;
                s.sun_dir = [-0.85, 0.25, 0.35];
            }
            Self::StressTest => {
                s.global_density = 0.04;
                // Volumes are spawned procedurally in submit_svol_volumes.
            }
        }
    }
}

impl Default for SvolState {
    fn default() -> Self {
        let mut s = Self {
            scene: Scene::new(),
            built: false,
            preset: SvolPreset::FoggyCorridor,
            global_enabled: true,
            global_min: [-200.0, -200.0, -2.0],
            global_max: [200.0, 200.0, 50.0],
            global_density: 0.05,
            global_colour: [0.78, 0.82, 0.88],
            global_anisotropy: 0.3,
            global_unlit: false,
            global_receive_shadows: true,
            global_soft_edges: false,
            global_soft_lo: 40.0,
            global_soft_hi: 200.0,
            global_noise: false,
            sphere_enabled: true,
            sphere_density: 0.4,
            sphere_colour: [0.95, 0.65, 0.45],
            sphere_radius: 1.6,
            sphere_anisotropy: 0.0,
            sphere_use_texture: false,
            sphere_texture_id: None,
            fire_enabled: false,
            fire_density: 2.0,
            fire_colour: [1.0, 0.55, 0.18],
            fire_radius: 2.0,
            fire_emission: 4.0,
            fire_falloff: 1.2,
            fire_use_ramp: true,
            fire_ramp_id: viewport_lib::ColourmapId(0),
            fire_animate: true,
            fire_noise_scale: 2.5,
            fire_noise_octaves: 4,
            fire_noise_time_scale: 2.5,
            fire_noise_scroll_z: 1.5,
            show_global_outline: false,
            show_sphere_outline: false,
            sun_dir: [-0.6, 0.2, 0.6],
            sun_colour: [1.0, 0.96, 0.86],
            sun_intensity: 2.0,
            shadows_enabled: true,
            sky_colour: [0.45, 0.55, 0.7],
            ground_colour: [0.2, 0.18, 0.15],
            hemisphere_intensity: 0.35,
            quality: ScatterQuality::Medium,
            blue_noise_jitter: true,
            downsample: false,
            temporal: false,
            temporal_blend: 0.85,
            fire_step_budget_override: false,
            fire_step_budget: 24,
        };
        SvolPreset::FoggyCorridor.apply(&mut s);
        s
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

        self.svol_state.fire_ramp_id =
            renderer.resources().builtin_colourmap_id(BuiltinColourmap::Inferno);

        // Bake a 32^3 procedural density (a hollow spherical shell with
        // sinusoidal modulation) and upload it as a 3D R32Float texture.
        // The sphere volume can opt in to using this instead of fbm noise.
        let dim: u32 = 32;
        let n = dim as usize;
        let mut data: Vec<f32> = Vec::with_capacity(n * n * n);
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let fx = (x as f32 + 0.5) / n as f32 - 0.5;
                    let fy = (y as f32 + 0.5) / n as f32 - 0.5;
                    let fz = (z as f32 + 0.5) / n as f32 - 0.5;
                    let r = (fx * fx + fy * fy + fz * fz).sqrt();
                    let shell = (1.0 - (r * 4.0 - 1.6).abs()).max(0.0);
                    let swirl = 0.5
                        + 0.5
                            * ((fx * 24.0 + fz * 16.0).sin()
                                * (fy * 18.0 + fx * 12.0).sin());
                    data.push((shell * swirl).clamp(0.0, 1.0));
                }
            }
        }
        let tex_id = renderer
            .resources_mut()
            .upload_volume(&self.device, &self.queue, &data, [dim, dim, dim]);
        self.svol_state.sphere_texture_id = Some(tex_id);

        self.svol_state.built = true;
    }

    /// Push scatter volumes onto the frame based on UI state.
    pub(crate) fn submit_svol_volumes(&self, fd: &mut viewport_lib::FrameData) {
        let s = &self.svol_state;
        if s.global_enabled {
            let mut v = ScatterVolume::box_uniform(
                Aabb {
                    min: glam::Vec3::from(s.global_min),
                    max: glam::Vec3::from(s.global_max),
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
            if s.global_noise {
                // Tuned for cloud-puffs scale: low frequency base layer + a
                // couple of octaves of detail. Slow horizontal scroll so the
                // cloud field drifts rather than evolves in place.
                let mut nd = NoiseDriver::default();
                nd.scale = 0.025;
                nd.octaves = 3;
                nd.lacunarity = 2.2;
                nd.time_scale = 0.05;
                nd.scroll_velocity = [0.6, 0.2, 0.0];
                v.noise = Some(nd);
            }
            let mut item = ScatterVolumeItem::new(v);
            item.settings.selected = s.show_global_outline;
            item.settings.unlit = s.global_unlit;
            item.settings.receive_shadows = s.global_receive_shadows;
            fd.scene.scatter_volumes.push(item);
        }

        if matches!(s.preset, SvolPreset::StressTest) {
            // Spawn a grid of small sphere volumes to demonstrate that
            // tile-based culling keeps off-screen volumes free.
            let palette: [[f32; 3]; 4] = [
                [0.95, 0.6, 0.4],
                [0.45, 0.65, 0.95],
                [0.85, 0.9, 0.5],
                [0.7, 0.5, 0.9],
            ];
            let mut idx = 0;
            for gy in -2..=1 {
                for gx in -2..=2 {
                    let centre = [gx as f32 * 4.0, gy as f32 * 4.0 + 1.0, 1.5];
                    let colour = palette[idx % palette.len()];
                    idx += 1;
                    let mut v = ScatterVolume::sphere_uniform(centre, 1.2, 0.5, colour);
                    v.anisotropy = 0.2;
                    fd.scene.scatter_volumes.push(ScatterVolumeItem::new(v));
                }
            }
        }
        if s.fire_enabled {
            let center = [-4.0_f32, 0.0, 1.2];
            let mut v = ScatterVolume::sphere_uniform(
                center,
                s.fire_radius,
                s.fire_density,
                s.fire_colour,
            );
            if s.fire_use_ramp {
                v.colour = ColourSource::Ramp(s.fire_ramp_id);
            }
            v.emission = Emission::Strength {
                strength: s.fire_emission,
                curve: EmissionCurve::Power(2.0),
            };
            v.density_remap = DensityRemap::ExpFalloff {
                center,
                falloff: s.fire_falloff,
            };
            if s.fire_animate {
                let mut nd = NoiseDriver::default();
                nd.scale = s.fire_noise_scale;
                nd.octaves = s.fire_noise_octaves;
                nd.time_scale = s.fire_noise_time_scale;
                nd.scroll_velocity = [0.0, 0.0, s.fire_noise_scroll_z];
                v.noise = Some(nd);
            }
            if s.fire_step_budget_override {
                v.step_budget = Some(s.fire_step_budget);
            }
            let mut item = ScatterVolumeItem::new(v);
            // Self-emissive volumes read better as unlit: in-scattering the
            // sun and ambient through the LUT's dark low-density colours
            // produces a darker veil around the bright core that reads as
            // a discrete sphere silhouette. The emission contribution alone
            // carries the look.
            item.settings.unlit = true;
            fd.scene.scatter_volumes.push(item);
        }
        if s.sphere_enabled {
            let mut v = ScatterVolume::sphere_uniform(
                [4.0, 0.0, 1.8],
                s.sphere_radius,
                s.sphere_density,
                s.sphere_colour,
            );
            if s.sphere_use_texture {
                v.density_texture = s.sphere_texture_id;
            }
            v.anisotropy = s.sphere_anisotropy;
            let mut item = ScatterVolumeItem::new(v);
            item.settings.selected = s.show_sphere_outline;
            fd.scene.scatter_volumes.push(item);
        }
    }
}

pub(crate) fn controls_svol(app: &mut App, ui: &mut egui::Ui) {
    // Shader time advances per frame; keep frames flowing while the fire
    // animation toggle is on so the noise field actually evolves.
    if app.svol_state.fire_enabled && app.svol_state.fire_animate {
        ui.ctx().request_repaint();
    }

    // -- Performance: always visible at the top. --
    let s = &mut app.svol_state;
    ui.heading("Performance");
    ui.horizontal(|ui| {
        for (label, q) in [
            ("Low", ScatterQuality::Low),
            ("Medium", ScatterQuality::Medium),
            ("High", ScatterQuality::High),
        ] {
            if ui.selectable_label(s.quality == q, label).clicked() {
                s.quality = q;
            }
        }
    });
    ui.checkbox(&mut s.downsample, "Half-resolution");
    ui.checkbox(&mut s.temporal, "Temporal accumulation");
    if s.temporal {
        ui.horizontal(|ui| {
            ui.label("History");
            ui.add(egui::Slider::new(&mut s.temporal_blend, 0.0..=0.95).step_by(0.01));
        });
    }
    ui.separator();

    // -- Scene preset picker. --
    ui.heading("Scene");
    let current = s.preset;
    let mut pending: Option<SvolPreset> = None;
    for preset in [
        SvolPreset::FoggyCorridor,
        SvolPreset::CloudySky,
        SvolPreset::CampfireNight,
        SvolPreset::GodRays,
        SvolPreset::StressTest,
    ] {
        if ui.selectable_label(current == preset, preset.label()).clicked() {
            pending = Some(preset);
        }
    }
    if let Some(p) = pending {
        p.apply(s);
    }
    ui.separator();

    // -- Fire: kept fully exposed so users see how a single volume's
    //    parameters compose into a recognisable look. --
    ui.heading("Fire (Sphere + emission)");
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
        ui.label("Tint");
        ui.color_edit_button_rgb(&mut s.fire_colour);
    });
    ui.checkbox(&mut s.fire_use_ramp, "Inferno colourmap");
    ui.checkbox(&mut s.fire_animate, "Animate");
    if s.fire_animate {
        ui.horizontal(|ui| {
            ui.label("Noise scale");
            ui.add(egui::Slider::new(&mut s.fire_noise_scale, 0.5..=8.0).step_by(0.1));
        });
        ui.horizontal(|ui| {
            ui.label("Time scale");
            ui.add(egui::Slider::new(&mut s.fire_noise_time_scale, 0.0..=5.0).step_by(0.05));
        });
        ui.horizontal(|ui| {
            ui.label("Vertical scroll");
            ui.add(egui::Slider::new(&mut s.fire_noise_scroll_z, -2.0..=2.0).step_by(0.05));
        });
    }
    ui.separator();

    // -- Advanced: everything else, collapsed by default. --
    egui::CollapsingHeader::new("Advanced")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut s.blue_noise_jitter, "Blue-noise jitter");
            ui.checkbox(&mut s.fire_step_budget_override, "Override fire step budget");
            if s.fire_step_budget_override {
                ui.horizontal(|ui| {
                    ui.label("Fire steps");
                    ui.add(egui::Slider::new(&mut s.fire_step_budget, 4..=64));
                });
            }

            ui.separator();
            ui.label(egui::RichText::new("Sun").strong());
            ui.horizontal(|ui| {
                ui.label("Direction");
                ui.add(egui::Slider::new(&mut s.sun_dir[0], -1.0..=1.0).step_by(0.01));
                ui.add(egui::Slider::new(&mut s.sun_dir[1], -1.0..=1.0).step_by(0.01));
                ui.add(egui::Slider::new(&mut s.sun_dir[2], 0.05..=1.0).step_by(0.01));
            });
            ui.horizontal(|ui| {
                ui.label("Intensity");
                ui.add(egui::Slider::new(&mut s.sun_intensity, 0.0..=5.0).step_by(0.05));
                ui.color_edit_button_rgb(&mut s.sun_colour);
            });
            ui.checkbox(&mut s.shadows_enabled, "Shadows");

            ui.separator();
            ui.label(egui::RichText::new("Hemisphere").strong());
            ui.horizontal(|ui| {
                ui.label("Sky");
                ui.color_edit_button_rgb(&mut s.sky_colour);
                ui.label("Ground");
                ui.color_edit_button_rgb(&mut s.ground_colour);
            });
            ui.horizontal(|ui| {
                ui.label("Intensity");
                ui.add(egui::Slider::new(&mut s.hemisphere_intensity, 0.0..=1.5).step_by(0.01));
            });

            ui.separator();
            ui.label(egui::RichText::new("Global volume").strong());
            ui.checkbox(&mut s.global_enabled, "Enabled");
            ui.horizontal(|ui| {
                ui.label("Density");
                ui.add(egui::Slider::new(&mut s.global_density, 0.0..=0.4).step_by(0.005));
            });
            ui.horizontal(|ui| {
                ui.label("Anisotropy");
                ui.add(egui::Slider::new(&mut s.global_anisotropy, -0.9..=0.9).step_by(0.01));
            });
            ui.horizontal(|ui| {
                ui.label("Colour");
                ui.color_edit_button_rgb(&mut s.global_colour);
            });
            ui.checkbox(&mut s.global_unlit, "Unlit");
            ui.checkbox(&mut s.global_receive_shadows, "Shafts (receive shadows)");
            ui.checkbox(&mut s.global_noise, "Procedural noise");

            ui.separator();
            ui.label(egui::RichText::new("Localized haze (sphere)").strong());
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
                ui.label("Anisotropy");
                ui.add(egui::Slider::new(&mut s.sphere_anisotropy, -0.9..=0.9).step_by(0.01));
            });
            ui.horizontal(|ui| {
                ui.label("Colour");
                ui.color_edit_button_rgb(&mut s.sphere_colour);
            });
            ui.checkbox(&mut s.sphere_use_texture, "Baked 3D density texture");

            ui.separator();
            ui.label(egui::RichText::new("Debug").strong());
            ui.checkbox(&mut s.show_global_outline, "Global outline");
            ui.checkbox(&mut s.show_sphere_outline, "Sphere outline");
        });

    ui.separator();
    if ui.button("Camera inside global volume").clicked() {
        app.camera.center = glam::Vec3::new(0.0, 0.0, 2.5);
        app.camera.distance = 0.5;
    }
    if ui.button("Camera back to platform").clicked() {
        app.camera.center = glam::Vec3::new(0.0, 0.0, 1.5);
        app.camera.distance = 14.0;
    }
}
