//! Showcase 7: Normal Maps + AO Maps : build and controls.
//!
//! Demonstrates normal-mapped and AO-mapped surfaces on a variety of shapes:
//! a sphere, a cube, and a flat wall panel, all on a tiled ground plane.
//! Toggles enable/disable normal maps and AO maps across all objects. A
//! normal-strength slider (glTF `normalScale`) dials the tangent-normal XY up or
//! down, and an occlusion-strength slider shows glTF `occlusionStrength` mapping
//! onto the existing `ao_range` field (`ao_range = [1 - s, 1]`).

use crate::App;
use crate::geometry::{
    make_box_with_uvs, make_brick_ao_map, make_brick_normal_map, make_tile_ao_map,
    make_tile_normal_map, make_uv_sphere,
};
use crate::eframe::egui;
use viewport_lib as vpl;
use vpl::{BackfacePolicy, Material, NodeId, ViewportRenderer, scene::Scene};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct NormalMapsState {
    pub built: bool,
    pub scene: Scene,
    /// (node_id, normal_map_id, ao_map_id) for every mapped object.
    pub mapped_nodes: Vec<(NodeId, vpl::TextureId, vpl::TextureId)>,
    pub normal_on: bool,
    /// Scales the tangent-space normal XY before the TBN transform
    /// (`Material::normal_strength`, glTF `normalScale`). 1.0 is the authored
    /// strength; 0.0 flattens to the geometric normal; >1.0 exaggerates relief.
    pub normal_strength: f32,
    pub ao_on: bool,
    /// Min/max remap applied to the AO map's R sample. Identity `[0.0, 1.0]`
    /// passes the sample through unchanged. Shrinking the range from the
    /// minimum side brightens cavities; raising the minimum compresses the
    /// cavity factor toward fully lit.
    pub ao_range: [f32; 2],
    /// glTF `occlusionStrength` view of `ao_range`: setting this to `s` writes
    /// `ao_range = [1 - s, 1]`, which is exactly `mix(1.0, sample, s)`. Shows
    /// that the renderer needs no dedicated occlusion-strength field.
    pub occlusion_strength: f32,
    pub clip_enabled: bool,
    pub cap_fill: bool,
}

impl Default for NormalMapsState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            mapped_nodes: Vec::new(),
            normal_on: true,
            normal_strength: 1.0,
            ao_on: true,
            ao_range: [0.0, 1.0],
            occlusion_strength: 1.0,
            clip_enabled: false,
            cap_fill: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build Showcase 7: Normal Maps + AO Maps demo.
    pub(crate) fn build_nm_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.nm_state.scene = Scene::new();
        self.nm_state.mapped_nodes.clear();

        // Upload maps at 128x128 for better detail.
        let brick_nm_data = make_brick_normal_map(128, 128);
        let brick_nm_id = renderer
            .resources_mut()
            .upload_texture(
                &self.device,
                &self.queue,
                vpl::TextureData::normal_map(128, 128, brick_nm_data.to_vec()),
            )
            .expect("brick normal map upload");

        // upload_data_texture, not upload_texture: an AO map holds a cavity
        // factor, not colour, so it must stay linear. The sRGB path would decode
        // it on sample and darken the occlusion.
        let brick_ao_data = make_brick_ao_map(128, 128);
        let brick_ao_id = renderer
            .resources_mut()
            .upload_texture(
                &self.device,
                &self.queue,
                vpl::TextureData::linear(128, 128, brick_ao_data.to_vec()),
            )
            .expect("brick ao map upload");

        let tile_nm_data = make_tile_normal_map(128, 128);
        let tile_nm_id = renderer
            .resources_mut()
            .upload_texture(
                &self.device,
                &self.queue,
                vpl::TextureData::normal_map(128, 128, tile_nm_data.to_vec()),
            )
            .expect("tile normal map upload");

        let tile_ao_data = make_tile_ao_map(128, 128);
        let tile_ao_id = renderer
            .resources_mut()
            .upload_texture(
                &self.device,
                &self.queue,
                vpl::TextureData::linear(128, 128, tile_ao_data.to_vec()),
            )
            .expect("tile ao map upload");

        // --- Meshes ---
        let sphere = make_uv_sphere(48, 24, 1.0);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("sphere mesh upload");

        let cube_mesh = make_box_with_uvs(1.6, 1.6, 1.6);
        let cube_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &cube_mesh)
            .expect("nm cube mesh upload");

        // Flat wall panel to show brick normal map on a flat surface.
        let wall_mesh = make_box_with_uvs(4.0, 0.3, 3.0);
        let wall_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &wall_mesh)
            .expect("wall mesh upload");

        // Ground plane with tile pattern.
        let ground_mesh = make_box_with_uvs(12.0, 12.0, 0.15);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("ground mesh upload");

        // --- Scene objects ---

        // Ground : tiled normal + AO.
        let ground_node = self.nm_state.scene.add_named(
            "Ground (Tile)",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.075)),
            {
                let mut mat = Material::pbr([0.85, 0.85, 0.85], 0.0, 0.85);
                mat.normal_map_id = Some(tile_nm_id);
                mat.ao_map_id = Some(tile_ao_id);
                mat
            },
        );
        self.nm_state
            .mapped_nodes
            .push((ground_node, tile_nm_id, tile_ao_id));

        // Brick sphere : left.
        let sphere_node = self.nm_state.scene.add_named(
            "Sphere (Brick)",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 1.0)),
            {
                let mut mat = Material::pbr([0.9, 0.88, 0.85], 0.0, 0.5);
                mat.normal_map_id = Some(brick_nm_id);
                mat.ao_map_id = Some(brick_ao_id);
                mat.backface_policy = BackfacePolicy::Identical;
                mat
            },
        );
        self.nm_state
            .mapped_nodes
            .push((sphere_node, brick_nm_id, brick_ao_id));

        // Tile cube : right.
        let cube_node = self.nm_state.scene.add_named(
            "Cube (Tile)",
            Some(cube_id),
            glam::Mat4::from_translation(glam::Vec3::new(2.5, 0.0, 0.8)),
            {
                let mut mat = Material::pbr([0.85, 0.87, 0.9], 0.1, 0.6);
                mat.normal_map_id = Some(tile_nm_id);
                mat.ao_map_id = Some(tile_ao_id);
                mat
            },
        );
        self.nm_state
            .mapped_nodes
            .push((cube_node, tile_nm_id, tile_ao_id));

        // Brick wall panel : behind.
        let wall_node = self.nm_state.scene.add_named(
            "Wall (Brick)",
            Some(wall_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, -2.0, 3.5)),
            {
                let mut mat = Material::pbr([0.92, 0.9, 0.87], 0.0, 0.7);
                mat.normal_map_id = Some(brick_nm_id);
                mat.ao_map_id = Some(brick_ao_id);
                mat
            },
        );
        self.nm_state
            .mapped_nodes
            .push((wall_node, brick_nm_id, brick_ao_id));

        // Plain sphere for comparison : separate mesh upload to avoid per-object
        // uniform clobbering (two items sharing a mesh_index would overwrite
        // each other's model matrix and texture bindings in the uniform buffer).
        let plain_sphere = make_uv_sphere(48, 24, 1.0);
        let plain_sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &plain_sphere)
            .expect("plain sphere mesh upload");
        self.nm_state.scene.add_named(
            "Sphere (No Maps)",
            Some(plain_sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(-3.0, 2.5, 1.0)),
            {
                let mut mat = Material::pbr([0.9, 0.88, 0.85], 0.0, 0.5);
                mat.backface_policy = BackfacePolicy::Identical;
                mat
            },
        );

        self.nm_state.normal_on = true;
        self.nm_state.normal_strength = 1.0;
        self.nm_state.ao_on = true;
        self.nm_state.occlusion_strength = 1.0;
        self.nm_state.clip_enabled = false;
        self.nm_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_normal_maps(app: &mut App, ui: &mut egui::Ui) {
    if ui
        .checkbox(&mut app.nm_state.normal_on, "Normal map")
        .changed()
    {
        let on = app.nm_state.normal_on;
        for &(node_id, nm_id, _) in &app.nm_state.mapped_nodes.clone() {
            if let Some(node) = app.nm_state.scene.node(node_id) {
                let mut mat = *node.material();
                mat.normal_map_id = if on { Some(nm_id) } else { None };
                app.nm_state.scene.set_material(node_id, mat);
            }
        }
    }

    if app.nm_state.normal_on {
        let changed = ui
            .add(
                egui::Slider::new(&mut app.nm_state.normal_strength, 0.0..=2.0)
                    .text("Normal strength"),
            )
            .on_hover_text("Material::normal_strength (glTF normalScale). 0 = flat, 1 = authored, 2 = deepened.")
            .changed();
        if changed {
            let s = app.nm_state.normal_strength;
            for &(node_id, _, _) in &app.nm_state.mapped_nodes.clone() {
                if let Some(node) = app.nm_state.scene.node(node_id) {
                    let mut mat = *node.material();
                    mat.normal_strength = s;
                    app.nm_state.scene.set_material(node_id, mat);
                }
            }
        }
    }

    if ui.checkbox(&mut app.nm_state.ao_on, "AO map").changed() {
        let on = app.nm_state.ao_on;
        for &(node_id, _, ao_id) in &app.nm_state.mapped_nodes.clone() {
            if let Some(node) = app.nm_state.scene.node(node_id) {
                let mut mat = *node.material();
                mat.ao_map_id = if on { Some(ao_id) } else { None };
                app.nm_state.scene.set_material(node_id, mat);
            }
        }
    }

    if app.nm_state.ao_on {
        let mut changed = false;
        // glTF occlusionStrength expressed through the existing ao_range field:
        // mix(1.0, sample, s) == mix(1 - s, 1, sample). No dedicated field needed.
        if ui
            .add(
                egui::Slider::new(&mut app.nm_state.occlusion_strength, 0.0..=1.0)
                    .text("Occlusion strength"),
            )
            .on_hover_text("glTF occlusionStrength: writes ao_range = [1 - s, 1].")
            .changed()
        {
            let s = app.nm_state.occlusion_strength;
            app.nm_state.ao_range = [1.0 - s, 1.0];
            changed = true;
        }
        ui.horizontal(|ui| {
            ui.label("AO range");
            changed |= ui
                .add(egui::Slider::new(&mut app.nm_state.ao_range[0], 0.0..=1.0).text("min"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut app.nm_state.ao_range[1], 0.0..=1.0).text("max"))
                .changed();
            if ui.button("reset").clicked() {
                app.nm_state.ao_range = [0.0, 1.0];
                changed = true;
            }
        });
        if changed {
            // Clamp max >= min for a well-formed range.
            if app.nm_state.ao_range[1] < app.nm_state.ao_range[0] {
                app.nm_state.ao_range[1] = app.nm_state.ao_range[0];
            }
            let range = app.nm_state.ao_range;
            for &(node_id, _, _) in &app.nm_state.mapped_nodes.clone() {
                if let Some(node) = app.nm_state.scene.node(node_id) {
                    let mut mat = *node.material();
                    mat.ao_range = range;
                    app.nm_state.scene.set_material(node_id, mat);
                }
            }
        }
    }

    ui.checkbox(&mut app.nm_state.clip_enabled, "Clip plane");
    if app.nm_state.clip_enabled {
        ui.checkbox(&mut app.nm_state.cap_fill, "Cap fill");
    }

    ui.separator();
}

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.nm_state.built
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    app.build_nm_scene(renderer);
    app.camera = vpl::Camera {
        center: glam::Vec3::new(0.0, 0.0, 0.8),
        distance: 10.0,
        orientation: glam::Quat::from_rotation_z(0.5)
            * glam::Quat::from_rotation_x(1.0),
        ..vpl::Camera::default()
    };
}

// ---------------------------------------------------------------------------
// Per-frame scene contents
// ---------------------------------------------------------------------------

/// Collect this showcase's render items and lighting for the frame. `out` carries
/// the few extra frame settings a showcase can set alongside its items.
pub(crate) fn scene(
    app: &mut crate::App,
    _frame: &crate::eframe::Frame,
    out: &mut crate::SceneOverrides,
) -> crate::SceneContents {
    let (items, bg_colour, lighting, scene_gen, sel_gen) = {
        let items = app.nm_state.scene.collect_render_items(&vpl::Selection::new());
        if app.nm_state.clip_enabled {
            out.clip_objects.push(vpl::ClipObject::plane([1.0, 0.0, 0.0], 0.0));
        }
        let lighting = {
            let mut _t = vpl::LightingSettings::default();
            _t.lights = vec![
                {
                    let mut _t = vpl::LightSource::default();
                    _t.kind = vpl::LightKind::Directional {
                        direction: [0.5, 0.3, 1.0],
                    };
                    _t.intensity = 0.4;
                    _t
                },
                {
                    let mut _t = vpl::LightSource::default();
                    _t.kind = vpl::LightKind::Point {
                        position: [3.0, 3.0, 3.0],
                        range: 15.0,
                        radius: 0.1,
                    };
                    _t.colour = [1.0, 0.97, 0.93].into();
                    _t.intensity = 20.0;
                    // Fill light for the normal-map highlights; not a
                    // shadow caster, so the directional's shadow stays
                    // a single clean shape.
                    _t.cast_shadows = false;
                    _t
                },
            ];
            _t.shadows.enabled = true;
            _t.hemisphere_intensity = 0.4;
            _t.sky_colour = [1.0, 1.0, 1.0];
            _t.ground_colour = [1.0, 1.0, 1.0];
            _t
        };
        let sg = app.nm_state.scene.version();
        (items, None, lighting, sg, 0)
    };
    crate::SceneContents {
        items,
        bg_colour,
        lighting,
        scene_gen,
        sel_gen,
    }
}

// ---------------------------------------------------------------------------
// Per-frame frame-data tweaks
// ---------------------------------------------------------------------------

/// Fold this showcase's own contributions into the assembled frame: extra
/// render items, overlays, and effect settings that are re-submitted every
/// frame rather than baked into the scene.
pub(crate) fn frame(
    app: &mut crate::App,
    fd: &mut vpl::FrameData,
    _ctx: &crate::FrameCtx,
) {
    fd.effects.display.mode = vpl::PipelineMode::Direct;
    // Cap far plane for better cascade distribution, but track orbit
    // distance so the scene doesn't disappear when zooming out.
    let mut rc = vpl::RenderCamera::from_camera(&app.camera);
    rc.far = (app.camera.distance * 3.0).max(60.0);
    rc.projection = glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
    fd.camera.render_camera = rc;
}

// ---------------------------------------------------------------------------
// Viewport overlay and per-frame tick
// ---------------------------------------------------------------------------

/// Draw this showcase's own egui overlay on top of the rendered viewport:
/// selection rectangles, mode readouts, and in-scene labels.


/// Advance this showcase's animation and ask for another frame. Runs after the
/// viewport has been drawn, so it only affects the next frame.


/// Route a viewport click for this showcase. The host calls this for a plain
/// click that no gizmo or widget has already consumed; `pos` is in viewport
/// pixels.


/// Handle drag gestures this showcase owns, before the camera controller runs.


/// Advance this showcase's own camera animation or object motion for the frame.


/// Update this showcase's interactive widgets for the frame.


/// Flush any per-frame GPU writes this showcase has queued.


/// Cache gizmo placement for next frame's hit-testing.


/// Take over the whole viewport for this frame. Returning false leaves the
/// host's normal single-viewport path in charge.
pub(crate) fn viewport_override(
    _app: &mut crate::App,
    _ui: &mut crate::eframe::egui::Ui,
    _cx: &crate::ViewportCtx,
) -> bool {
    false
}

/// Drive the orbit controller for this showcase. Returning false leaves the
/// host to run the usual suppress-or-apply path.
pub(crate) fn drive_camera(_app: &mut crate::App, _cx: &crate::ViewportCtx) -> bool {
    false
}

/// Whether the orbit controller should resolve without moving the camera this
/// frame. This showcase never suppresses it.
pub(crate) fn suppress_orbit(_app: &crate::App, _cx: &crate::ViewportCtx) -> bool {
    false
}

// ---------------------------------------------------------------------------
// Showcase entry point
// ---------------------------------------------------------------------------

/// Stateless handle for this showcase; the scene state lives on [`crate::App`].
pub(crate) struct ScNormalMaps;

/// The registry's handle to this showcase.
pub(crate) static SHOWCASE: ScNormalMaps = ScNormalMaps;

impl crate::Showcase for ScNormalMaps {
    fn needs_build(&self, app: &crate::App) -> bool {
        needs_build(app)
    }
    fn build(&self, app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
        build(app, renderer)
    }
    fn scene(&self, app: &mut crate::App, frame: &crate::eframe::Frame, out: &mut crate::SceneOverrides) -> crate::SceneContents {
        scene(app, frame, out)
    }
    fn frame(&self, app: &mut crate::App, fd: &mut vpl::FrameData, ctx: &crate::FrameCtx) {
        frame(app, fd, ctx)
    }
    fn viewport_override(&self, app: &mut crate::App, ui: &mut crate::eframe::egui::Ui, cx: &crate::ViewportCtx) -> bool {
        viewport_override(app, ui, cx)
    }
    fn drive_camera(&self, app: &mut crate::App, cx: &crate::ViewportCtx) -> bool {
        drive_camera(app, cx)
    }
    fn suppress_orbit(&self, app: &crate::App, cx: &crate::ViewportCtx) -> bool {
        suppress_orbit(app, cx)
    }
    fn controls(&self, app: &mut crate::App, ui: &mut crate::eframe::egui::Ui, _frame: &crate::eframe::Frame) {
        controls_normal_maps(app, ui)
    }
}
