//! Showcase 54: Custom Shading Plugins.
//!
//! Demonstrates the `MaterialPlugin` API: WGSL shading hooks registered at
//! runtime and selected per material via `Material::shading_plugin`. Eight
//! spheres share one light rig:
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
//!   6. Detail-layer plugin: a tiled detail albedo blended over the base,
//!      gated by a per-vertex mask painted into
//!      `MeshData::extension_attributes` (the custom vertex-attribute demo).
//!   7. Parallax plugin: plugin-owned height + albedo textures with a
//!      tangent-space parallax march, all inside the hook body.
//!   8. Dissolve plugin: a `shade_surface` body driving the gated alpha
//!      output on a Mask material, with an emissive glow at the edge.
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
#[path = "../plugins/surface_detail_plugin.rs"]
mod surface_detail_plugin;
#[path = "../plugins/toon_plugin.rs"]
mod toon_plugin;

use surface_detail_plugin::{
    DetailLayerPlugin, DissolvePlugin, ParallaxPlugin, detail_params, dissolve_params,
    parallax_params,
};
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
    pub detail: Option<MaterialPluginId>,
    pub parallax: Option<MaterialPluginId>,
    pub dissolve: Option<MaterialPluginId>,
    pub detail_mesh_id: Option<MeshId>,
    pub toon_a_handle: Option<MaterialPluginParamsHandle>,
    pub toon_b_handle: Option<MaterialPluginParamsHandle>,
    pub rim_handle: Option<MaterialPluginParamsHandle>,
    pub detail_handle: Option<MaterialPluginParamsHandle>,
    pub parallax_handle: Option<MaterialPluginParamsHandle>,
    pub dissolve_handle: Option<MaterialPluginParamsHandle>,
    // UI-tunable; written into the params windows every frame.
    pub bands_a: f32,
    pub ambient_a: f32,
    pub tint_a: [f32; 3],
    pub bands_b: f32,
    pub tint_b: [f32; 3],
    pub rim_colour: [f32; 3],
    pub rim_power: f32,
    pub detail_tiling: f32,
    pub detail_strength: f32,
    pub detail_attr_mask: f32,
    pub parallax_height: f32,
    pub parallax_tiling: f32,
    pub dissolve_threshold: f32,
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
            detail: None,
            parallax: None,
            dissolve: None,
            detail_mesh_id: None,
            toon_a_handle: None,
            toon_b_handle: None,
            rim_handle: None,
            detail_handle: None,
            parallax_handle: None,
            dissolve_handle: None,
            bands_a: 3.0,
            ambient_a: 0.25,
            tint_a: [1.0, 1.0, 1.0],
            bands_b: 6.0,
            tint_b: [1.0, 0.62, 0.25],
            rim_colour: [0.2, 0.5, 1.0],
            rim_power: 3.0,
            detail_tiling: 6.0,
            detail_strength: 0.8,
            detail_attr_mask: 1.0,
            parallax_height: 0.06,
            parallax_tiling: 3.0,
            dissolve_threshold: 0.35,
        }
    }
}

// ---------------------------------------------------------------------------
// Prewarm
// ---------------------------------------------------------------------------

/// Register the five shading plugins and build their pipeline sets up front.
///
/// Each plugin set is roughly nine render pipelines plus two shader-module
/// compilations, all built synchronously. `build_custom_shading_scene` runs on
/// the frame the showcase is opened, so compiling the sets there stalls that
/// frame. Doing it here at startup, where the cost hides behind window
/// creation, keeps opening the showcase smooth. Registration is idempotent per
/// plugin name, so `build_custom_shading_scene` reuses these same plugins and
/// their already-warm pipelines.
pub(crate) fn prewarm_custom_shading_plugins(
    device: &eframe::wgpu::Device,
    renderer: &mut ViewportRenderer,
) {
    let resources = renderer.resources_mut();
    let ids = [
        resources.register_material_plugin(device, &ToonPlugin),
        resources.register_material_plugin(device, &RimPlugin),
        resources.register_material_plugin(device, &DetailLayerPlugin),
        resources.register_material_plugin(device, &ParallaxPlugin),
        resources.register_material_plugin(device, &DissolvePlugin),
    ]
    .into_iter()
    .filter_map(Result::ok)
    .collect::<Vec<_>>();
    resources.warm_material_plugin_pipelines(device, &ids);
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

        // The detail sphere carries a per-vertex mask in the extension
        // attribute channel: 1 toward the top pole fading to 0 below the
        // equator, so the detail layer visibly follows painted vertex data.
        let mut detail_mesh = viewport_lib::primitives::sphere(1.0, 48, 24);
        detail_mesh.extension_attributes = Some(
            detail_mesh
                .positions
                .iter()
                .map(|p| {
                    let t = (p[2] * 0.5 + 0.5).clamp(0.0, 1.0);
                    let mask = (t - 0.25).clamp(0.0, 0.5) * 2.0;
                    [mask, 0.0, 0.0, 0.0]
                })
                .collect(),
        );
        let detail_mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &detail_mesh)
            .expect("upload detail sphere");

        let resources = renderer.resources_mut();
        // Registration is idempotent per name, so re-entering the showcase
        // reuses the existing plugins; only mint the extra variants once.
        let toon_a = resources
            .register_material_plugin(&self.device, &ToonPlugin)
            .expect("register toon plugin");
        let rim = resources
            .register_material_plugin(&self.device, &RimPlugin)
            .expect("register rim plugin");
        let detail = resources
            .register_material_plugin(&self.device, &DetailLayerPlugin)
            .expect("register detail plugin");
        let parallax_default = resources
            .register_material_plugin(&self.device, &ParallaxPlugin)
            .expect("register parallax plugin");
        let dissolve = resources
            .register_material_plugin(&self.device, &DissolvePlugin)
            .expect("register dissolve plugin");

        // Normally these sets are already built by the startup call to
        // prewarm_custom_shading_plugins, so this is a cheap idempotent
        // check. It still matters if the showcase is reached without that
        // prewarm: building the sets here (rather than letting the first
        // rendered frame pay for them a few at a time) avoids the cold
        // plugins drawing built-in shading until their set is ready.
        resources.warm_material_plugin_pipelines(
            &self.device,
            &[toon_a, rim, detail, parallax_default, dissolve],
        );

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

        if self.cs_state.parallax.is_none() {
            // Detail albedo: a diagonal crosshatch.
            let detail_tex = {
                let size = 64usize;
                let mut rgba = vec![0u8; size * size * 4];
                for y in 0..size {
                    for x in 0..size {
                        let d = ((x + y) / 6) % 2 == 0;
                        let v: [u8; 4] = if d {
                            [235, 235, 235, 255]
                        } else {
                            [120, 110, 100, 255]
                        };
                        rgba[(y * size + x) * 4..(y * size + x) * 4 + 4].copy_from_slice(&v);
                    }
                }
                resources
                    .upload_texture(&self.device, &self.queue, 64, 64, &rgba)
                    .expect("upload detail albedo")
            };
            self.cs_state.detail = Some(
                resources
                    .create_material_plugin_variant(
                        &self.device,
                        detail,
                        &detail_params(
                            self.cs_state.detail_tiling,
                            self.cs_state.detail_strength,
                            self.cs_state.detail_attr_mask,
                        ),
                        &[detail_tex],
                    )
                    .expect("detail variant"),
            );

            // Parallax height + albedo: a brick-like grid. Mortar grooves are
            // low (dark height) and each brick face is high.
            let (height_tex, albedo_tex) = {
                let size = 64usize;
                let mut height = vec![0u8; size * size * 4];
                let mut albedo = vec![0u8; size * size * 4];
                for y in 0..size {
                    for x in 0..size {
                        let row = y / 16;
                        let xo = if row % 2 == 0 { x } else { x + 8 };
                        let in_mortar = (xo % 32) < 3 || (y % 16) < 3;
                        let h: u8 = if in_mortar { 30 } else { 230 };
                        let a: [u8; 4] = if in_mortar {
                            [140, 135, 130, 255]
                        } else {
                            [180, 85, 60, 255]
                        };
                        let i = (y * size + x) * 4;
                        height[i..i + 4].copy_from_slice(&[h, h, h, 255]);
                        albedo[i..i + 4].copy_from_slice(&a);
                    }
                }
                (
                    resources
                        .upload_texture(&self.device, &self.queue, 64, 64, &height)
                        .expect("upload parallax height"),
                    resources
                        .upload_texture(&self.device, &self.queue, 64, 64, &albedo)
                        .expect("upload parallax albedo"),
                )
            };
            self.cs_state.parallax = Some(
                resources
                    .create_material_plugin_variant(
                        &self.device,
                        parallax_default,
                        &parallax_params(
                            self.cs_state.parallax_height,
                            self.cs_state.parallax_tiling,
                        ),
                        &[height_tex, albedo_tex],
                    )
                    .expect("parallax variant"),
            );
        }

        self.cs_state.toon_a_handle = resources.material_plugin_params_handle(toon_a);
        self.cs_state.toon_b_handle = self
            .cs_state
            .toon_b
            .and_then(|id| resources.material_plugin_params_handle(id));
        self.cs_state.rim_handle = resources.material_plugin_params_handle(rim);
        self.cs_state.detail_handle = self
            .cs_state
            .detail
            .and_then(|id| resources.material_plugin_params_handle(id));
        self.cs_state.parallax_handle = self
            .cs_state
            .parallax
            .and_then(|id| resources.material_plugin_params_handle(id));
        self.cs_state.dissolve = Some(dissolve);
        self.cs_state.dissolve_handle = resources.material_plugin_params_handle(dissolve);

        self.cs_state.mesh_id = Some(mesh_id);
        self.cs_state.detail_mesh_id = Some(detail_mesh_id);
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
    let detail_mesh = s.detail_mesh_id.unwrap_or(mesh_id);
    let entries: [(f32, [f32; 3], Option<MaterialPluginId>, MeshId); 8] = [
        (-9.0, [0.75, 0.3, 0.3], None, mesh_id), // built-in PBR reference
        (-6.0, [0.75, 0.3, 0.3], s.toon_a, mesh_id),
        (-3.0, [0.75, 0.3, 0.3], s.toon_b, mesh_id),
        (0.0, [0.85, 0.85, 0.85], s.toon_tex, mesh_id),
        (3.0, [0.25, 0.25, 0.3], s.rim, mesh_id),
        // Detail layer reads the per-vertex mask baked into this mesh's
        // extension attributes.
        (6.0, [0.4, 0.5, 0.35], s.detail, detail_mesh),
        (9.0, [0.8, 0.8, 0.8], s.parallax, mesh_id),
        (12.0, [0.55, 0.35, 0.25], s.dissolve, mesh_id),
    ];
    entries
        .iter()
        .map(|(x, colour, plugin, mesh)| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = *mesh;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(*x, 0.0, 1.0)).to_cols_array_2d();
            item.material = Material::pbr(*colour, 0.1, 0.55);
            item.material.shading_plugin = *plugin;
            // The dissolve sphere runs on a Mask material so the hook's
            // alpha output discards fragments below the cutoff.
            if *plugin == s.dissolve && plugin.is_some() {
                item.material.alpha_mode = viewport_lib::material::AlphaMode::Mask(0.5);
            }
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
        "Four WGSL plugins registered at runtime. Each sphere's Material\n\
         selects a plugin (or none); the toon spheres share one plugin and\n\
         its pipelines but carry independent params windows, and the striped\n\
         sphere binds a texture to the plugin's texture slot. The detail\n\
         sphere gates its layer with a per-vertex mask from\n\
         MeshData::extension_attributes; the parallax sphere marches its own\n\
         height texture. Shadows, AO, and transparency keep working through\n\
         plugin draws.",
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

    ui.separator();
    ui.label("Detail sphere (extension-attribute mask)");
    ui.add(egui::Slider::new(&mut app.cs_state.detail_tiling, 1.0..=16.0).text("tiling"));
    ui.add(egui::Slider::new(&mut app.cs_state.detail_strength, 0.0..=1.0).text("strength"));
    ui.add(
        egui::Slider::new(&mut app.cs_state.detail_attr_mask, 0.0..=1.0)
            .text("vertex mask (0 = everywhere, 1 = painted mask)"),
    );

    ui.separator();
    ui.label("Parallax sphere (plugin-owned height texture)");
    ui.add(egui::Slider::new(&mut app.cs_state.parallax_height, 0.0..=0.15).text("height scale"));
    ui.add(egui::Slider::new(&mut app.cs_state.parallax_tiling, 1.0..=8.0).text("tiling"));

    ui.separator();
    ui.label("Dissolve sphere (gated alpha + emissive edge)");
    ui.add(egui::Slider::new(&mut app.cs_state.dissolve_threshold, 0.0..=1.0).text("threshold"));

    // Push the UI values into the params windows. The writes are five small
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
    if let Some(h) = &s.detail_handle {
        h.write(
            &app.queue,
            &detail_params(s.detail_tiling, s.detail_strength, s.detail_attr_mask),
        );
    }
    if let Some(h) = &s.parallax_handle {
        h.write(
            &app.queue,
            &parallax_params(s.parallax_height, s.parallax_tiling),
        );
    }
    if let Some(h) = &s.dissolve_handle {
        h.write(
            &app.queue,
            &dissolve_params(s.dissolve_threshold, 0.15, 1.0, [2.5, 1.2, 0.3]),
        );
    }
}
