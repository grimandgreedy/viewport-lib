//! Showcase 49: Lighting and Shading Consistency.
//!
//! A side-by-side grid of item types sharing one `LightingSettings` and one set
//! of `ItemSettings` broadcast values. Toggling a flag in the controls panel
//! applies it to every item, so the user can compare how the same setting
//! behaves across types.
//!
//! Layout (Z-up). Each cell holds one item type with a small label. Grid lies
//! in the X-Z plane at Y=0: columns spaced 4 units in +X, rows spaced 4 units
//! in +Z (top row is highest).
//!
//! Controls:
//!   - Global scene lighting: light direction, intensity, hemisphere ambient
//!   - Broadcast `ItemSettings`: hidden, unlit, opacity, wireframe
//!
//! The point is to demonstrate the consistency guarantees in
//! `docs/plans/lighting-shading-consistency-plan.md`: flipping a single
//! broadcast toggle produces a coherent visual response across every item type
//! that supports it, and a documented no-op on types that don't.
//!
//! This showcase exercises the `ItemSettings` surface only. Material-level
//! shading variants (PBR vs Phong vs Toon) are mesh-only and live in
//! showcases 5, 6, 7, 19. Volume / GPU-MC / transparent volume mesh / slice
//! items are intentionally deferred to later phases; see plan section
//! "Showcase coverage by phase".

use eframe::egui;
use viewport_lib::{
    ColourmapId, FrameData, GaussianSplatData, GaussianSplatId, GaussianSplatItem, GlyphItem,
    GlyphType, GpuImplicitItem, GpuImplicitOptions, ImplicitBlendMode, ImplicitPrimitive,
    ItemSettings, LightSource, LightingSettings, Material, MeshId, PointCloudItem, PolylineItem,
    ProjectedTetId, RibbonItem, SceneRenderItem, ShDegree, StreamtubeItem, TensorGlyphItem,
    TransparentVolumeMeshItem, TubeItem, ViewportRenderer, VolumeId, VolumeItem, VolumeSurfaceSliceItem,
};

use crate::App;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct LcState {
    pub built: bool,
    pub scene: viewport_lib::scene::Scene,
    pub box_mesh_id: Option<MeshId>,
    pub disk_mesh_id: Option<MeshId>,
    pub splat_id: Option<GaussianSplatId>,
    pub volume_id: Option<VolumeId>,
    pub tvm_id: Option<ProjectedTetId>,
    pub volume_scalar_range: (f32, f32),

    pub bcast_hidden: bool,
    pub bcast_unlit: bool,
    pub bcast_opacity: f32,
    pub bcast_wireframe: bool,

    pub light_yaw_deg: f32,
    pub light_pitch_deg: f32,
    pub light_intensity: f32,
    pub second_light_enabled: bool,
    pub second_light_yaw_deg: f32,
    pub second_light_intensity: f32,
    pub hemisphere_intensity: f32,
    pub sky_colour: [f32; 3],
    pub ground_colour: [f32; 3],
}

impl Default for LcState {
    fn default() -> Self {
        Self {
            built: false,
            scene: viewport_lib::scene::Scene::new(),
            box_mesh_id: None,
            disk_mesh_id: None,
            splat_id: None,
            volume_id: None,
            tvm_id: None,
            volume_scalar_range: (0.0, 1.0),
            bcast_hidden: false,
            bcast_unlit: false,
            bcast_opacity: 1.0,
            bcast_wireframe: false,
            light_yaw_deg: 35.0,
            light_pitch_deg: 50.0,
            light_intensity: 1.0,
            second_light_enabled: false,
            second_light_yaw_deg: -120.0,
            second_light_intensity: 0.4,
            hemisphere_intensity: 0.4,
            sky_colour: [0.85, 0.92, 1.0],
            ground_colour: [0.40, 0.35, 0.30],
        }
    }
}

// ---------------------------------------------------------------------------
// Grid layout helpers
// ---------------------------------------------------------------------------

const COL_DX: f32 = 4.0;
const ROW_DZ: f32 = 4.0;

/// World position for grid cell `(col, row)`. col 0..=4 left-to-right along +X;
/// row 0..=2 top-to-bottom along -Z (row 0 is highest, row 2 is lowest).
/// Y = 0 puts the whole grid in a single vertical plane facing the camera.
fn cell(col: u32, row: u32) -> glam::Vec3 {
    let cx = (col as f32 - 2.0) * COL_DX;
    let cz = (1.0 - row as f32) * ROW_DZ;
    glam::Vec3::new(cx, 0.0, cz)
}

fn broadcast(state: &LcState, base: &mut ItemSettings) {
    base.hidden = state.bcast_hidden;
    base.unlit = state.bcast_unlit;
    base.opacity = state.bcast_opacity;
    base.wireframe = state.bcast_wireframe;
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_lc_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.lc_state.scene = viewport_lib::scene::Scene::new();

        let box_mesh = self.upload_box(renderer);
        self.lc_state.box_mesh_id = Some(box_mesh);

        // --- Cell (0, 0): surface mesh ---
        {
            let p = cell(0, 0);
            let mat = Material::from_colour([0.85, 0.55, 0.35]);
            self.lc_state.scene.add_named(
                "Surface mesh",
                Some(box_mesh),
                glam::Mat4::from_translation(p),
                mat,
            );
        }

        // --- Cell (4, 1): Gaussian splats (pre-upload) ---
        {
            let mut sd = GaussianSplatData::default();
            sd.sh_degree = ShDegree::Zero;
            // Ring of splats in the camera-facing X-Z plane (Z-up world).
            for i in 0..7 {
                let t = i as f32 / 6.0;
                let theta = t * std::f32::consts::TAU;
                sd.positions.push([
                    0.7 * theta.cos(),
                    (t - 0.5) * 0.6,
                    0.7 * theta.sin(),
                ]);
                sd.scales.push([0.22, 0.22, 0.22]);
                sd.rotations.push([0.0, 0.0, 0.0, 1.0]);
                sd.opacities.push(0.9);
                // ShDegree::Zero: 3 floats per splat = base RGB.
                sd.sh_coefficients
                    .extend_from_slice(&[0.5 + 0.4 * t, 0.7 - 0.4 * t, 0.95]);
            }
            if let Ok(sid) = renderer
                .resources_mut()
                .upload_gaussian_splats(&self.device, &self.queue, &sd)
            {
                self.lc_state.splat_id = Some(sid);
            }
        }

        // --- 16^3 scalar volume used by `VolumeItem` and `VolumeSurfaceSliceItem` ---
        {
            let dims = [16u32; 3];
            let (nx, ny, nz) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
            let mut data = vec![0.0f32; nx * ny * nz];
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let fx = (x as f32 / (nx - 1) as f32) * 2.0 - 1.0;
                        let fy = (y as f32 / (ny - 1) as f32) * 2.0 - 1.0;
                        let fz = (z as f32 / (nz - 1) as f32) * 2.0 - 1.0;
                        data[x + nx * (y + ny * z)] = 0.7 - (fx * fx + fy * fy + fz * fz).sqrt();
                    }
                }
            }
            let vid = renderer
                .resources_mut()
                .upload_volume(&self.device, &self.queue, &data, dims);
            self.lc_state.volume_id = Some(vid);
            self.lc_state.volume_scalar_range = (-0.4, 0.7);
        }

        // --- Small disk mesh for the volume surface slice ---
        {
            let disk = viewport_lib::primitives::plane(1.6, 1.6);
            if let Ok(mid) = renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &disk)
            {
                self.lc_state.disk_mesh_id = Some(mid);
            }
        }

        // --- Single-tet projected-tet mesh for `TransparentVolumeMeshItem` ---
        //
        // TVM carries no model matrix, so the tet is uploaded at its grid-cell
        // world position (row 2, col 2) by translating the vertices here.
        {
            use viewport_lib::resources::volume_mesh::{VolumeMeshData, CELL_SENTINEL};
            let p = cell(2, 2);
            let mut data = VolumeMeshData::default();
            data.positions = vec![
                [p.x - 0.9, p.y - 0.9, p.z - 0.9],
                [p.x + 0.9, p.y - 0.9, p.z - 0.9],
                [p.x, p.y + 0.9, p.z - 0.9],
                [p.x, p.y, p.z + 0.9],
            ];
            data.cells = vec![[0, 1, 2, 3, CELL_SENTINEL, CELL_SENTINEL, CELL_SENTINEL, CELL_SENTINEL]];
            data.cell_scalars
                .insert("scalar".to_string(), vec![0.6]);
            renderer
                .resources_mut()
                .ensure_colourmaps_initialized(&self.device, &self.queue);
            if let Ok((tid, _, _)) = renderer.resources_mut().upload_projected_tet_mesh(
                &self.device,
                &data,
                "scalar",
                ColourmapId(0),
            ) {
                self.lc_state.tvm_id = Some(tid);
            }
        }

        self.lc_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

pub(crate) fn controls_lc(app: &mut App, ui: &mut egui::Ui) {
    let s = &mut app.lc_state;

    ui.label("Scene lighting (shared by every lit item type)");
    ui.separator();
    ui.label("Primary directional light");
    ui.add(egui::Slider::new(&mut s.light_yaw_deg, -180.0..=180.0).text("yaw"));
    ui.add(egui::Slider::new(&mut s.light_pitch_deg, -89.0..=89.0).text("pitch"));
    ui.add(egui::Slider::new(&mut s.light_intensity, 0.0..=3.0).text("intensity"));
    ui.separator();
    ui.checkbox(&mut s.second_light_enabled, "Second directional light");
    if s.second_light_enabled {
        ui.add(egui::Slider::new(&mut s.second_light_yaw_deg, -180.0..=180.0).text("yaw 2"));
        ui.add(egui::Slider::new(&mut s.second_light_intensity, 0.0..=3.0).text("intensity 2"));
    }
    ui.separator();
    ui.label("Hemisphere ambient");
    ui.add(egui::Slider::new(&mut s.hemisphere_intensity, 0.0..=1.5).text("intensity"));
    ui.horizontal(|ui| {
        ui.label("sky");
        ui.color_edit_button_rgb(&mut s.sky_colour);
        ui.label("ground");
        ui.color_edit_button_rgb(&mut s.ground_colour);
    });

    ui.separator();
    ui.heading("Broadcast ItemSettings");
    ui.weak("Every flag below is applied to every item type each frame.");
    ui.weak("Where the type has no support, the flag is accepted-but-no-op.");
    ui.checkbox(&mut s.bcast_hidden, "hidden  (every item disappears)");
    ui.checkbox(&mut s.bcast_unlit, "unlit  (skip lighting where applicable)");
    ui.add(egui::Slider::new(&mut s.bcast_opacity, 0.0..=1.0).text("opacity"));
    ui.checkbox(&mut s.bcast_wireframe, "wireframe (where supported)");

    ui.separator();
    ui.weak("Grid (X-Y plane):");
    ui.label("Row 0: Surface mesh | Point cloud | Glyph | Tensor glyph | Polyline");
    ui.label("Row 1: Streamtube | Tube | Ribbon | GPU implicit | Gaussian splat");
    ui.label("Row 2: Volume | Vol. surface slice | Transp. vol. mesh");
    ui.separator();
    ui.weak("Observe how the four broadcast flags behave per row:");
    ui.weak("- hidden: every cell disappears");
    ui.weak("- unlit: lit cells flatten; intrinsically flat cells stay the same");
    ui.weak("- opacity: every cell becomes translucent (including TVM, volume)");
    ui.weak("- wireframe: mesh-family cells switch; non-mesh cells stay");
}

// ---------------------------------------------------------------------------
// Lighting helper
// ---------------------------------------------------------------------------

fn lc_lighting(state: &LcState) -> LightingSettings {
    let yaw = state.light_yaw_deg.to_radians();
    let pitch = state.light_pitch_deg.to_radians();
    let dir = [
        pitch.cos() * yaw.cos(),
        pitch.cos() * yaw.sin(),
        pitch.sin(),
    ];
    let mut lights = vec![{
        let mut l = LightSource::default();
        l.kind = viewport_lib::LightKind::Directional { direction: dir };
        l.intensity = state.light_intensity;
        l
    }];
    if state.second_light_enabled {
        let yaw2 = state.second_light_yaw_deg.to_radians();
        let dir2 = [yaw2.cos(), yaw2.sin(), 0.4];
        let mut l = LightSource::default();
        l.kind = viewport_lib::LightKind::Directional { direction: dir2 };
        l.intensity = state.second_light_intensity;
        l.colour = [1.0, 0.9, 0.7];
        lights.push(l);
    }
    let mut light = LightingSettings::default();
    light.lights = lights;
    light.hemisphere_intensity = state.hemisphere_intensity;
    light.sky_colour = state.sky_colour;
    light.ground_colour = state.ground_colour;
    light
}

// ---------------------------------------------------------------------------
// Scene-item collection (mesh-family items via Scene).
// ---------------------------------------------------------------------------

pub(crate) fn lc_collect_scene_items(
    app: &mut App,
) -> (Vec<SceneRenderItem>, LightingSettings, u64, u64) {
    let mut items = app
        .lc_state
        .scene
        .collect_render_items(&viewport_lib::Selection::new());
    for it in items.iter_mut() {
        broadcast(&app.lc_state, &mut it.settings);
    }
    let scene_gen = app.lc_state.scene.version();
    let lighting = lc_lighting(&app.lc_state);
    (items, lighting, scene_gen, 0)
}

// ---------------------------------------------------------------------------
// Per-frame submission for non-mesh items.
// ---------------------------------------------------------------------------

pub(crate) fn submit_lc_items(app: &App, fd: &mut FrameData) {
    if !app.lc_state.built {
        return;
    }
    let s = &app.lc_state;

    // Cell (1, 0): point cloud. Spiral in the camera-facing X-Z plane with a
    // small +/- Y depth wiggle so the points read as 3D.
    {
        let p = cell(1, 0);
        let mut pc = PointCloudItem::default();
        let n = 80;
        for i in 0..n {
            let t = i as f32 / n as f32;
            let theta = t * std::f32::consts::TAU * 2.0;
            let r = 0.7 * t;
            pc.positions.push([
                p.x + r * theta.cos(),
                p.y + (t - 0.5) * 0.6,
                p.z + r * theta.sin(),
            ]);
        }
        pc.point_size = 8.0;
        pc.default_colour = [0.55, 0.85, 1.0, 1.0];
        broadcast(s, &mut pc.settings);
        fd.scene.point_clouds.push(pc);
    }

    // Cell (2, 0): glyphs on a ring in the camera-facing X-Z plane, tangent vectors
    // along the same plane so the arrows lie flat against the screen.
    {
        let p = cell(2, 0);
        let mut g = GlyphItem::default();
        g.glyph_type = GlyphType::Arrow;
        let n = 8;
        for i in 0..n {
            let theta = (i as f32 / n as f32) * std::f32::consts::TAU;
            let r = 0.7;
            g.positions
                .push([p.x + r * theta.cos(), p.y, p.z + r * theta.sin()]);
            g.vectors
                .push([-theta.sin() * 0.6, 0.0, theta.cos() * 0.6]);
        }
        g.use_default_colour = true;
        g.default_colour = [0.95, 0.7, 0.3, 1.0];
        g.scale = 0.9;
        broadcast(s, &mut g.settings);
        fd.scene.glyphs.push(g);
    }

    // Cell (3, 0): tensor glyphs.
    {
        let p = cell(3, 0);
        let mut tg = TensorGlyphItem::default();
        for i in 0..3 {
            let dx = (i as f32 - 1.0) * 0.7;
            tg.positions.push([p.x + dx, p.y, p.z]);
            tg.eigenvalues.push([0.55, 0.35, 0.20]);
            tg.eigenvectors.push([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]);
        }
        broadcast(s, &mut tg.settings);
        fd.scene.tensor_glyphs.push(tg);
    }

    // Cell (4, 0): polyline (spiral) in the X-Z plane with a small Y depth wiggle.
    {
        let p = cell(4, 0);
        let mut pl = PolylineItem::default();
        let n = 32;
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            let theta = t * std::f32::consts::TAU * 1.5;
            pl.positions.push([
                p.x + 0.7 * theta.cos(),
                p.y + (t - 0.5) * 0.6,
                p.z + 0.7 * theta.sin(),
            ]);
        }
        pl.strip_lengths = vec![n as u32];
        pl.default_colour = [0.95, 0.95, 0.55, 1.0];
        pl.line_width = 2.5;
        broadcast(s, &mut pl.settings);
        fd.scene.polylines.push(pl);
    }

    // Cell (0, 1): streamtube traced as a loop in X-Z with a Y depth wiggle.
    {
        let p = cell(0, 1);
        let mut st = StreamtubeItem::default();
        let n = 24;
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            let theta = t * std::f32::consts::TAU;
            st.positions.push([
                p.x + 0.6 * theta.cos(),
                p.y + (t - 0.5) * 0.8,
                p.z + 0.6 * theta.sin(),
            ]);
        }
        st.strip_lengths = vec![n as u32];
        st.colour = [0.6, 0.95, 0.85, 1.0];
        st.radius = 0.08;
        broadcast(s, &mut st.settings);
        fd.scene.streamtube_items.push(st);
    }

    // Cell (1, 1): tube with tapering radius along the same X-Z loop pattern.
    {
        let p = cell(1, 1);
        let mut tb = TubeItem::default();
        let n = 24;
        let mut radii = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            let theta = t * std::f32::consts::TAU;
            tb.positions.push([
                p.x + 0.6 * theta.cos(),
                p.y + (t - 0.5) * 0.8,
                p.z + 0.6 * theta.sin(),
            ]);
            radii.push(0.04 + 0.06 * (1.0 - t));
        }
        tb.strip_lengths = vec![n as u32];
        tb.radius = 0.06;
        tb.radius_attribute = Some(radii);
        tb.colour = [0.95, 0.55, 0.85, 1.0];
        broadcast(s, &mut tb.settings);
        fd.scene.tube_items.push(tb);
    }

    // Cell (2, 1): ribbon along the X-Z loop. Twist vectors point radially outward
    // in the same plane so the ribbon face is roughly perpendicular to the camera.
    {
        let p = cell(2, 1);
        let mut rb = RibbonItem::default();
        let n = 24;
        let mut twists = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            let theta = t * std::f32::consts::TAU;
            rb.positions.push([
                p.x + 0.6 * theta.cos(),
                p.y + (t - 0.5) * 0.8,
                p.z + 0.6 * theta.sin(),
            ]);
            twists.push([theta.cos() * 0.15, 0.0, theta.sin() * 0.15]);
        }
        rb.strip_lengths = vec![n as u32];
        rb.width = 0.18;
        rb.twist_attribute = Some(twists);
        rb.colour = [0.95, 0.85, 0.55, 1.0];
        broadcast(s, &mut rb.settings);
        fd.scene.ribbon_items.push(rb);
    }

    // Cell (3, 1): GPU implicit (sphere).
    {
        let p = cell(3, 1);
        let mut item = GpuImplicitItem::default();
        let mut prim = ImplicitPrimitive::zeroed();
        prim.kind = 1; // sphere
        prim.blend = 0.0;
        prim.params[0] = p.x;
        prim.params[1] = p.y;
        prim.params[2] = p.z;
        prim.params[3] = 0.9; // radius
        prim.colour = [0.7, 0.5, 0.95, 1.0];
        item.primitives.push(prim);
        item.blend_mode = ImplicitBlendMode::Union;
        item.march_options = GpuImplicitOptions {
            max_steps: 64,
            step_scale: 0.9,
            hit_threshold: 1e-3,
            max_distance: 60.0,
        };
        broadcast(s, &mut item.settings);
        fd.scene.gpu_implicit.push(item);
    }

    // Cell (4, 1): Gaussian splats.
    if let Some(sid) = s.splat_id {
        let p = cell(4, 1);
        let mut item = GaussianSplatItem::default();
        item.id = sid;
        item.model = glam::Mat4::from_translation(p).to_cols_array_2d();
        broadcast(s, &mut item.settings);
        fd.scene.gaussian_splats.push(item);
    }

    // Cell (0, 2): volume (ray-march). C6 wires `settings.unlit` into
    // `enable_shading`, so the broadcast `unlit` toggle disables gradient Phong here.
    if let Some(vid) = s.volume_id {
        let p = cell(0, 2);
        let mut v = VolumeItem::default();
        v.volume_id = vid;
        v.colour_lut = Some(ColourmapId(0));
        v.scalar_range = s.volume_scalar_range;
        v.bbox_min = [-1.0, -1.0, -1.0];
        v.bbox_max = [1.0, 1.0, 1.0];
        v.enable_shading = true;
        v.opacity_scale = 1.0;
        v.threshold_min = s.volume_scalar_range.0;
        v.threshold_max = s.volume_scalar_range.1;
        v.model = glam::Mat4::from_translation(p)
            .mul_mat4(&glam::Mat4::from_scale(glam::Vec3::splat(0.9)))
            .to_cols_array_2d();
        broadcast(s, &mut v.settings);
        fd.scene.volumes.push(v);
    }

    // Cell (1, 2): volume surface slice (disk through the small scalar field).
    // C7: `settings.opacity` now composes with the type's own opacity field.
    //
    // The default `plane` primitive lies in the X-Y plane (Z-up = horizontal).
    // Rotate it 90 degrees around X so the disk faces the camera in the X-Z plane.
    if let (Some(vid), Some(disk_id)) = (s.volume_id, s.disk_mesh_id) {
        let p = cell(1, 2);
        let mut ss = VolumeSurfaceSliceItem::default();
        ss.volume_id = vid;
        ss.mesh_id = disk_id;
        ss.bbox_min = [p.x - 0.9, p.y - 0.9, p.z - 0.9];
        ss.bbox_max = [p.x + 0.9, p.y + 0.9, p.z + 0.9];
        ss.scalar_range = s.volume_scalar_range;
        ss.colour_lut = Some(ColourmapId(0));
        ss.opacity = 1.0;
        ss.model = (glam::Mat4::from_translation(p)
            * glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2))
        .to_cols_array_2d();
        broadcast(s, &mut ss.settings);
        fd.scene.volume_surface_slices.push(ss);
    }

    // Cell (2, 2): transparent volume mesh (single tet). C5: `unlit` is documented
    // as a no-op on this type. Setting `bcast_unlit` produces no visible change
    // here, while every other lit cell goes flat.
    if let Some(tvm_id) = s.tvm_id {
        // TVM has no model field; the tet vertices were uploaded around the origin.
        // The user can compare alpha/hidden behaviour against the rest of the grid.
        let _ = cell(2, 2);
        let mut tv = TransparentVolumeMeshItem::new(tvm_id);
        tv.density = 3.0;
        tv.scalar_range = Some((0.0, 1.0));
        broadcast(s, &mut tv.settings);
        fd.scene.transparent_volume_meshes.push(tv);
    }
}
