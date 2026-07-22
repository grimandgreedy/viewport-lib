//! Showcase 54: Custom Shading Plugins.
//!
//! Demonstrates the `MaterialPlugin` API: WGSL shading hooks registered at
//! runtime and selected per material via `Material::shading_plugin`. Five
//! spheres share one mesh and one light rig:
//!
//!   1. Built-in PBR (the reference look).
//!   2. Toon plugin, variant A: banded diffuse, sliders drive the group-3
//!      params window live.
//!   3. Toon plugin, variant B: same plugin and pipelines, its own params
//!      window (independent band count and tint), proving per-material params.
//!   4. Toon plugin, textured variant: a procedural stripe texture bound to
//!      the plugin's `material_texture_0` slot.
//!   5. Rim plugin: a `recolor` hook that adds a view-dependent rim on top of
//!      the built-in lighting.
//!
//! The plugin draws keep scene shadows, AO, and alpha modes; the toon
//! terminator bands the shadow edge because `light.shadow` arrives as its own
//! factor.

use crate::App;
use eframe::egui;
use viewport_lib::{
    LightKind, LightSource, LightingSettings, Material, MaterialPluginId,
    MaterialPluginParamsHandle, MeshId, SceneRenderItem, ViewportRenderer,
};

// The plugin definitions live in `examples/plugins/toon_plugin.rs` so other
// examples can register the same reference plugins.
#[path = "../plugins/toon_plugin.rs"]
mod toon_plugin;

use toon_plugin::{RimPlugin, ToonPlugin, toon_params};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct CustomShadingState {
    pub built: bool,
    pub mesh_id: Option<MeshId>,
    pub toon_a: Option<MaterialPluginId>,
    pub toon_b: Option<MaterialPluginId>,
    pub toon_tex: Option<MaterialPluginId>,
    pub rim: Option<MaterialPluginId>,
    pub toon_a_handle: Option<MaterialPluginParamsHandle>,
    pub toon_b_handle: Option<MaterialPluginParamsHandle>,
    pub rim_handle: Option<MaterialPluginParamsHandle>,
    // UI-tunable; written into the params windows every frame.
    pub bands_a: f32,
    pub ambient_a: f32,
    pub tint_a: [f32; 3],
    pub bands_b: f32,
    pub tint_b: [f32; 3],
    pub rim_colour: [f32; 3],
    pub rim_power: f32,
}

impl Default for CustomShadingState {
    fn default() -> Self {
        Self {
            built: false,
            mesh_id: None,
            toon_a: None,
            toon_b: None,
            toon_tex: None,
            rim: None,
            toon_a_handle: None,
            toon_b_handle: None,
            rim_handle: None,
            bands_a: 3.0,
            ambient_a: 0.25,
            tint_a: [1.0, 1.0, 1.0],
            bands_b: 6.0,
            tint_b: [1.0, 0.62, 0.25],
            rim_colour: [0.2, 0.5, 1.0],
            rim_power: 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_custom_shading_scene(&mut self, renderer: &mut ViewportRenderer) {
        let mesh = viewport_lib::primitives::sphere(1.0, 48, 24);
        let mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &mesh)
            .expect("upload sphere");

        let resources = renderer.resources_mut();
        // Registration is idempotent per name, so re-entering the showcase
        // reuses the existing plugins; only mint the extra variants once.
        let toon_a = resources
            .register_material_plugin(&self.device, &ToonPlugin)
            .expect("register toon plugin");
        let rim = resources
            .register_material_plugin(&self.device, &RimPlugin)
            .expect("register rim plugin");

        if self.cs_state.toon_b.is_none() {
            let s = &self.cs_state;
            self.cs_state.toon_b = Some(
                resources
                    .create_material_plugin_variant(
                        &self.device,
                        toon_a,
                        &toon_params(s.bands_b, 0.15, 1.0, s.tint_b),
                        &[],
                    )
                    .expect("toon variant B"),
            );

            // Procedural stripe texture for the textured variant.
            let tex_id = {
                let size = 64usize;
                let mut rgba = vec![0u8; size * size * 4];
                for y in 0..size {
                    for x in 0..size {
                        let on = (x / 8) % 2 == 0;
                        let v: [u8; 4] = if on {
                            [250, 250, 250, 255]
                        } else {
                            [70, 70, 90, 255]
                        };
                        rgba[(y * size + x) * 4..(y * size + x) * 4 + 4].copy_from_slice(&v);
                    }
                }
                resources
                    .upload_texture(&self.device, &self.queue, 64, 64, &rgba)
                    .expect("upload stripes")
            };
            self.cs_state.toon_tex = Some(
                resources
                    .create_material_plugin_variant(
                        &self.device,
                        toon_a,
                        &toon_params(4.0, 0.25, 4.0, [1.0, 1.0, 1.0]),
                        &[tex_id],
                    )
                    .expect("toon textured variant"),
            );
        }

        self.cs_state.toon_a_handle = resources.material_plugin_params_handle(toon_a);
        self.cs_state.toon_b_handle = self
            .cs_state
            .toon_b
            .and_then(|id| resources.material_plugin_params_handle(id));
        self.cs_state.rim_handle = resources.material_plugin_params_handle(rim);

        self.cs_state.mesh_id = Some(mesh_id);
        self.cs_state.toon_a = Some(toon_a);
        self.cs_state.rim = Some(rim);
        self.cs_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Render items
// ---------------------------------------------------------------------------

pub(crate) fn custom_shading_items(app: &App) -> Vec<SceneRenderItem> {
    let Some(mesh_id) = app.cs_state.mesh_id else {
        return vec![];
    };
    let s = &app.cs_state;
    let entries: [(f32, [f32; 3], Option<MaterialPluginId>); 5] = [
        (-6.0, [0.75, 0.3, 0.3], None), // built-in PBR reference
        (-3.0, [0.75, 0.3, 0.3], s.toon_a),
        (0.0, [0.75, 0.3, 0.3], s.toon_b),
        (3.0, [0.85, 0.85, 0.85], s.toon_tex),
        (6.0, [0.25, 0.25, 0.3], s.rim),
    ];
    entries
        .iter()
        .map(|(x, colour, plugin)| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(*x, 0.0, 1.0)).to_cols_array_2d();
            item.material = Material::pbr(*colour, 0.1, 0.55);
            item.material.shading_plugin = *plugin;
            item
        })
        .collect()
}

pub(crate) fn custom_shading_lighting() -> LightingSettings {
    let mut sun = LightSource::default();
    sun.kind = LightKind::Directional {
        direction: [0.5, 0.35, 1.0],
    };
    sun.colour = [1.0, 0.97, 0.9];
    sun.intensity = 1.2;

    let mut t = LightingSettings::default();
    t.lights = vec![sun];
    t.hemisphere_intensity = 0.25;
    t
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_custom_shading(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Custom shading via MaterialPlugin");
    ui.separator();
    ui.label(
        "Two WGSL plugins registered at runtime. Each sphere's Material\n\
         selects a plugin (or none); the toon spheres share one plugin and\n\
         its pipelines but carry independent params windows, and the striped\n\
         sphere binds a texture to the plugin's texture slot. Shadows, AO,\n\
         and transparency keep working through plugin draws.",
    );

    ui.separator();
    ui.label("Toon sphere A (default variant)");
    ui.add(egui::Slider::new(&mut app.cs_state.bands_a, 1.0..=12.0).text("bands"));
    ui.add(egui::Slider::new(&mut app.cs_state.ambient_a, 0.0..=1.0).text("ambient"));
    ui.horizontal(|ui| {
        ui.label("tint:");
        ui.color_edit_button_rgb(&mut app.cs_state.tint_a);
    });

    ui.separator();
    ui.label("Toon sphere B (second variant, same plugin)");
    ui.add(egui::Slider::new(&mut app.cs_state.bands_b, 1.0..=12.0).text("bands"));
    ui.horizontal(|ui| {
        ui.label("tint:");
        ui.color_edit_button_rgb(&mut app.cs_state.tint_b);
    });

    ui.separator();
    ui.label("Rim sphere (recolor hook on built-in PBR)");
    ui.horizontal(|ui| {
        ui.label("rim colour:");
        ui.color_edit_button_rgb(&mut app.cs_state.rim_colour);
    });
    ui.add(egui::Slider::new(&mut app.cs_state.rim_power, 0.5..=8.0).text("rim power"));

    // Push the UI values into the params windows. The writes are three small
    // uniform uploads; doing them unconditionally keeps the sliders live.
    let s = &app.cs_state;
    if let Some(h) = &s.toon_a_handle {
        h.write(
            &app.queue,
            &toon_params(s.bands_a, s.ambient_a, 1.0, s.tint_a),
        );
    }
    if let Some(h) = &s.toon_b_handle {
        h.write(&app.queue, &toon_params(s.bands_b, 0.15, 1.0, s.tint_b));
    }
    if let Some(h) = &s.rim_handle {
        h.write(
            &app.queue,
            &[[
                s.rim_colour[0],
                s.rim_colour[1],
                s.rim_colour[2],
                s.rim_power,
            ]],
        );
    }
}
