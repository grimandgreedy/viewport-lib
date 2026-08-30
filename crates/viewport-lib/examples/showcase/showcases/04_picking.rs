//! Picking: choose which levels to pick (object, point-like, edge-like,
//! face-like), then click or drag a box to select. The scene holds one of every
//! pickable item type (meshes, point cloud, glyphs, polyline, volume, gaussian
//! splats, a volume mesh, tensor glyphs, sprites, streamtube / tube / ribbon,
//! a volume surface slice, a screen image, a GPU implicit surface, a GPU
//! marching-cubes surface, and a decal). Every one highlights at the object
//! level (outline / selected tint) and, where it has sub-elements, at the
//! sub-object level (face fill, vertex/point/instance sprites, segment/strip,
//! cell, voxel).
//!
//! Uses the GPU pick backend (`pick_gpu` / `pick_rect_gpu`), so the wgpu device
//! and queue are read each frame from `ShowcaseCtx`. Sub-object highlighting is
//! driven by a `SubSelection` turned into a per-frame `SubSelectionRef`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use viewport_lib as vpl;

use eframe::egui;
use glam::{Mat4, Vec2, Vec3};
use vpl::{
    AnchorX, AnchorY, BuiltinColourmap, CellSelectionInfo, ColourmapId, DecalItem,
    GaussianSplatData, GaussianSplatId, GaussianSplatItem, GlyphItem, GlyphType, GpuImplicitItem,
    GpuImplicitOptions, GpuMarchingCubesJob, ImplicitBlendMode, ImplicitPrimitive, ItemSettings,
    Material, McVolumeId, MeshId, NodeId, OverlayFill, OverlayShape, OverlayShapeItem, PickId,
    PickMask, PointCloudItem, PolylineItem, PolylineSelectionInfo, RibbonItem, ScreenImageItem,
    ShDegree, SpriteItem, StreamtubeItem, SubObjectRef, SubSelection, SubSelectionRef,
    TensorGlyphItem, TextureId, TubeItem, VolumeData, VolumeId, VolumeItem, VolumeMeshData,
    VolumeMeshItem, VolumeSelectionInfo, VolumeSurfaceSliceItem, primitives,
};

use crate::showcase::{SetupCtx, Showcase, ShowcaseCtx};

// Pick ids for the injected (non-scene-node) items. High values so they never
// collide with scene node ids (a scene node's pick_id is forced to its node id).
const PC: u64 = 1001;
const GLYPH: u64 = 1002;
const POLY: u64 = 1003;
const VOLUME: u64 = 1004;
const SPLAT: u64 = 1005;
const VMESH: u64 = 1006;
const TENSOR: u64 = 1007;
const SPRITE: u64 = 1008;
const STREAMTUBE: u64 = 1009;
const TUBE: u64 = 1010;
const RIBBON: u64 = 1011;
const SLICE: u64 = 1012;
const SCREEN: u64 = 1013;
const IMPLICIT: u64 = 1014;
const MC: u64 = 1015;
const DECAL: u64 = 1016;

// World placement of the ray-marched volume: a 4x4x4 box (16^3 grid at 0.25
// spacing) translated into the upper-left of the scene. Surface slice and voxel
// highlighting share these numbers.
const VOL_ORIGIN: Vec3 = Vec3::new(-13.0, 3.0, -1.0);
const VOL_EXTENT: f32 = 4.0;

fn vol_model() -> Mat4 {
    Mat4::from_translation(VOL_ORIGIN)
}

fn splat_model() -> Mat4 {
    Mat4::from_translation(Vec3::new(-11.0, 0.0, 0.5))
}

struct HitInfo {
    name: String,
    sub: String,
    pos: Vec3,
    scalar: Option<f32>,
}

pub struct PickingShowcase {
    pick_object: bool,
    pick_point: bool,
    pick_edge: bool,
    pick_face: bool,
    /// id -> (display name, scene node if it is a mesh).
    labels: HashMap<u64, (String, Option<NodeId>)>,

    // Injected each frame so their `settings.selected` can be toggled. Simple
    // Vec-backed items are stored whole and cloned; items that own a GPU handle
    // are rebuilt each frame from the handle plus stored CPU data.
    pc: PointCloudItem,
    glyphs: GlyphItem,
    polyline: PolylineItem,
    tensor: TensorGlyphItem,
    sprites: SpriteItem,
    streamtube: StreamtubeItem,
    tube: TubeItem,
    ribbon: RibbonItem,

    // GPU handles plus the CPU data kept for sub-object highlighting.
    volume_id: Option<VolumeId>,
    volume_data: Option<Arc<VolumeData>>,
    splat_id: Option<GaussianSplatId>,
    splat_positions: Vec<[f32; 3]>,
    vmesh_mesh_id: Option<MeshId>,
    vmesh_face_to_cell: Vec<u32>,
    vmesh_data: Option<Arc<VolumeMeshData>>,
    slice_mesh_id: Option<MeshId>,
    mc_id: Option<McVolumeId>,
    mc_data: Option<Arc<VolumeData>>,
    screen_pixels: Vec<[u8; 4]>,
    decal_tex: Option<TextureId>,
    decal_transform: Mat4,

    // CPU data for sub-object highlighting, keyed by object id.
    mesh_lookup: HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    model_matrices: HashMap<u64, Mat4>,
    point_positions: HashMap<u64, Vec<[f32; 3]>>,
    instance_lookup: HashMap<u64, Vec<[f32; 3]>>,

    // Selection state.
    selected_objects: HashSet<u64>,
    sub: SubSelection,

    marker: Option<NodeId>,
    marker_pos: Option<Vec3>,
    last: Option<HitInfo>,
    rect_status: Option<String>,

    drag_start: Option<Vec2>,
    drag_cur: Option<Vec2>,
    dragging_prev: bool,
}

impl PickingShowcase {
    pub fn new() -> Self {
        Self {
            pick_object: true,
            pick_point: false,
            pick_edge: false,
            pick_face: false,
            labels: HashMap::new(),
            pc: PointCloudItem::default(),
            glyphs: GlyphItem::default(),
            polyline: PolylineItem::default(),
            tensor: TensorGlyphItem::default(),
            sprites: SpriteItem::default(),
            streamtube: StreamtubeItem::default(),
            tube: TubeItem::default(),
            ribbon: RibbonItem::default(),
            volume_id: None,
            volume_data: None,
            splat_id: None,
            splat_positions: Vec::new(),
            vmesh_mesh_id: None,
            vmesh_face_to_cell: Vec::new(),
            vmesh_data: None,
            slice_mesh_id: None,
            mc_id: None,
            mc_data: None,
            screen_pixels: Vec::new(),
            decal_tex: None,
            decal_transform: Mat4::IDENTITY,
            mesh_lookup: HashMap::new(),
            model_matrices: HashMap::new(),
            point_positions: HashMap::new(),
            instance_lookup: HashMap::new(),
            selected_objects: HashSet::new(),
            sub: SubSelection::new(),
            marker: None,
            marker_pos: None,
            last: None,
            rect_status: None,
            drag_start: None,
            drag_cur: None,
            dragging_prev: false,
        }
    }

    fn mask(&self) -> PickMask {
        let mut m = PickMask::empty();
        if self.pick_object {
            m |= PickMask::OBJECT;
        }
        if self.pick_point {
            m |= PickMask::POINT_LIKE;
        }
        if self.pick_edge {
            m |= PickMask::EDGE_LIKE;
        }
        if self.pick_face {
            m |= PickMask::FACE_LIKE;
        }
        m
    }

    /// Clear every selection channel (object outline, injected-item selected
    /// flags, sub-object set).
    fn clear_selection(&mut self, session: &mut vpl::ViewportInstance) {
        session.selection_mut().clear();
        self.selected_objects.clear();
        self.sub.clear();
    }

    /// Record a hit: sub-object hits go to the SubSelection; object hits outline
    /// the mesh (via the session selection) or flag the injected item.
    fn record(&mut self, session: &mut vpl::ViewportInstance, id: u64, sub: Option<SubObjectRef>) {
        match sub {
            Some(s) => self.sub.add(id, s),
            None => match self.labels.get(&id).and_then(|(_, n)| *n) {
                Some(node) => session.selection_mut().add(node),
                None => {
                    self.selected_objects.insert(id);
                }
            },
        }
    }

    /// Push every injected (non-scene-node) item into the frame, flagging the
    /// selected ones. They must be in the frame for both rendering and picking.
    fn inject_items(&self, session: &mut vpl::ViewportInstance) {
        let sel = |id: u64| self.selected_objects.contains(&id);
        let fd = session.frame_data_mut();

        // Point cloud.
        let mut pc = self.pc.clone();
        pc.settings.selected = sel(PC);
        fd.scene.point_clouds.push(pc);

        // Arrow glyphs.
        let mut glyphs = self.glyphs.clone();
        glyphs.settings.selected = sel(GLYPH);
        fd.scene.glyphs.push(glyphs);

        // Multi-strip polyline.
        let mut polyline = self.polyline.clone();
        polyline.settings.selected = sel(POLY);
        fd.scene.polylines.push(polyline);

        // Tensor glyphs.
        let mut tensor = self.tensor.clone();
        tensor.settings.selected = sel(TENSOR);
        fd.scene.tensor_glyphs.push(tensor);

        // Sprites.
        let mut sprites = self.sprites.clone();
        sprites.settings.selected = sel(SPRITE);
        fd.scene.sprite_items.push(sprites);

        // Streamtube / tube / ribbon.
        let mut st = self.streamtube.clone();
        st.settings.selected = sel(STREAMTUBE);
        fd.scene.streamtube_items.push(st);
        let mut tb = self.tube.clone();
        tb.settings.selected = sel(TUBE);
        fd.scene.tube_items.push(tb);
        let mut rb = self.ribbon.clone();
        rb.settings.selected = sel(RIBBON);
        fd.scene.ribbon_items.push(rb);

        // Ray-marched volume.
        if let (Some(vol_id), Some(data)) = (self.volume_id, self.volume_data.as_ref()) {
            let mut vol = VolumeItem::default();
            vol.volume_id = vol_id;
            vol.model = vol_model().to_cols_array_2d();
            vol.bbox_min = [0.0, 0.0, 0.0];
            vol.bbox_max = [VOL_EXTENT; 3];
            vol.scalar_range = (0.0, 1.0);
            vol.threshold_min = 0.15;
            vol.threshold_max = 1.0;
            vol.opacity_scale = 0.6;
            vol.enable_shading = false;
            vol.settings.unlit = false;
            vol.settings.pick_id = PickId(VOLUME);
            vol.settings.selected = sel(VOLUME);
            vol.volume_data = Some(data.clone());
            fd.scene.volumes.push(vol);
        }

        // Gaussian splats.
        if let Some(splat_id) = self.splat_id {
            let mut item = GaussianSplatItem::default();
            item.source = splat_id;
            item.model = splat_model().to_cols_array_2d();
            item.settings.pick_id = PickId(SPLAT);
            item.settings.selected = sel(SPLAT);
            item.settings.unlit = false;
            fd.scene.gaussian_splats.push(item);
        }

        // Volume mesh (capsule): opaque boundary surface, so point-like picking
        // returns Cell sub-objects via face_to_cell.
        if let Some(mesh_id) = self.vmesh_mesh_id {
            let mut item = VolumeMeshItem::new(mesh_id, self.vmesh_face_to_cell.clone());
            item.settings.pick_id = PickId(VMESH);
            item.settings.selected = sel(VMESH);
            item.settings.unlit = false;
            fd.scene.volume_meshes.push(item);
        }

        // Volume surface slice: a plane sampling the volume, tilted inside the
        // volume bbox.
        if let (Some(vol_id), Some(mesh_id)) = (self.volume_id, self.slice_mesh_id) {
            let mut item = VolumeSurfaceSliceItem::default();
            item.volume_id = vol_id;
            item.mesh_id = mesh_id;
            item.bbox_min = VOL_ORIGIN.to_array();
            item.bbox_max = (VOL_ORIGIN + Vec3::splat(VOL_EXTENT)).to_array();
            item.scalar_range = (0.0, 1.0);
            item.model = (Mat4::from_translation(VOL_ORIGIN + Vec3::new(2.0, 2.0, 2.0))
                * Mat4::from_rotation_x(60_f32.to_radians()))
            .to_cols_array_2d();
            item.settings.pick_id = PickId(SLICE);
            item.settings.selected = sel(SLICE);
            item.settings.unlit = false;
            fd.scene.volume_surface_slices.push(item);
        }

        // Screen image: a checkerboard pinned to the top-right corner.
        if !self.screen_pixels.is_empty() {
            let mut img = ScreenImageItem::default();
            img.pixels = self.screen_pixels.clone();
            img.width = 48;
            img.height = 48;
            img.anchor_x = AnchorX::Right;
            img.anchor_y = AnchorY::Top;
            img.scale = 2.0;
            img.settings.pick_id = PickId(SCREEN);
            img.settings.selected = sel(SCREEN);
            img.settings.unlit = false;
            fd.scene.screen_images.push(img);
        }

        // GPU implicit: two smooth-blended spheres.
        {
            let centers = [[10.3_f32, 5.0, 0.0], [11.7, 5.0, 0.0]];
            let colours = [[0.68_f32, 0.22, 0.06, 1.0], [0.12, 0.28, 0.68, 1.0]];
            let mut item = GpuImplicitItem::default();
            for i in 0..2 {
                let mut prim = ImplicitPrimitive::zeroed();
                prim.kind = 1; // sphere
                prim.blend = 0.8;
                prim.params[0] = centers[i][0];
                prim.params[1] = centers[i][1];
                prim.params[2] = centers[i][2];
                prim.params[3] = 1.1; // radius
                prim.colour = colours[i];
                item.primitives.push(prim);
            }
            item.blend_mode = ImplicitBlendMode::SmoothUnion;
            item.march_options = GpuImplicitOptions {
                max_steps: 64,
                step_scale: 0.9,
                hit_threshold: 1e-3,
                max_distance: 100.0,
            };
            item.settings.pick_id = PickId(IMPLICIT);
            item.settings.selected = sel(IMPLICIT);
            item.settings.unlit = false;
            fd.scene.gpu_implicit.push(item);
        }

        // GPU marching cubes: gyroid surface.
        if let Some(mc_id) = self.mc_id {
            let mut mat = Material::flat([0.10, 0.52, 0.18]);
            mat.roughness = 0.4;
            let mut settings = ItemSettings::default();
            settings.unlit = false;
            settings.pick_id = PickId(MC);
            settings.selected = sel(MC);
            fd.scene.gpu_mc_jobs.push(GpuMarchingCubesJob {
                volume_id: mc_id,
                isovalue: 0.0,
                material: mat,
                settings,
                cpu_data: self.mc_data.clone(),
            });
        }

        // Decal: a target sticker straddling the cube's top face.
        if let Some(tex) = self.decal_tex {
            let mut d = DecalItem::default();
            d.transform = self.decal_transform.to_cols_array_2d();
            d.texture_id = tex;
            d.settings.pick_id = PickId(DECAL);
            d.settings.selected = sel(DECAL);
            fd.scene.decals.push(d);
        }
    }
}

impl Showcase for PickingShowcase {
    fn name(&self) -> &str {
        "Picking"
    }

    fn setup(&mut self, ctx: &mut SetupCtx) {
        ctx.session
            .resources_mut()
            .ensure_colourmaps_initialized(ctx.device, ctx.queue);

        let cube_data = primitives::cube(1.4);
        let sphere_data = primitives::sphere(0.9, 40, 20);
        let cube = ctx
            .session
            .resources_mut()
            .upload_mesh_data(ctx.device, &cube_data)
            .unwrap();
        let sphere = ctx
            .session
            .resources_mut()
            .upload_mesh_data(ctx.device, &sphere_data)
            .unwrap();

        // Marker first so it takes node id 0 (PickId(0) == NONE, skipped by GPU
        // picking) and keeps the real meshes off node id 0.
        let mut marker_mat = Material::from_colour([1.0, 0.9, 0.15]);
        marker_mat.emissive = [1.0, 0.85, 0.1];
        let marker = ctx.session.scene_mut().add(
            Some(sphere),
            Mat4::from_scale(Vec3::splat(0.001)),
            marker_mat,
        );
        let mut hidden = ItemSettings::default();
        hidden.hidden = true;
        hidden.unlit = true;
        ctx.session.scene_mut().set_appearance(marker, hidden);
        self.marker = Some(marker);

        // Meshes. A scene node's pick_id is its node id, so key by the node id.
        let cube_pos = Vec3::new(-11.0, -5.0, 0.7);
        let sphere_pos = Vec3::new(-5.5, 0.0, 0.9);
        let cube_node = add_mesh(ctx, cube, cube_pos, [0.62, 0.18, 0.14]);
        let sphere_node = add_mesh(ctx, sphere, sphere_pos, [0.16, 0.26, 0.60]);
        self.labels
            .insert(cube_node, ("Cube".into(), Some(cube_node)));
        self.labels
            .insert(sphere_node, ("Sphere".into(), Some(sphere_node)));
        self.mesh_lookup.insert(
            cube_node,
            (cube_data.positions.clone(), cube_data.indices.clone()),
        );
        self.mesh_lookup.insert(
            sphere_node,
            (sphere_data.positions.clone(), sphere_data.indices.clone()),
        );
        self.model_matrices
            .insert(cube_node, Mat4::from_translation(cube_pos));
        self.model_matrices
            .insert(sphere_node, Mat4::from_translation(sphere_pos));

        // Point cloud (cloud points).
        let (pos, sca) = noisy_sphere(Vec3::new(0.0, 5.0, 1.0), 0.9, 400);
        self.pc.positions = pos.clone();
        self.pc.scalars = sca;
        self.pc.colourmap_id = Some(ColourmapId(BuiltinColourmap::Viridis as usize));
        self.pc.point_size = 9.0;
        self.pc.settings.pick_id = PickId(PC);
        self.point_positions.insert(PC, pos);
        self.model_matrices.insert(PC, Mat4::IDENTITY);
        self.labels.insert(PC, ("Point cloud".into(), None));

        // Arrow glyphs (instances).
        let gpos: Vec<[f32; 3]> = (0..9)
            .map(|i| {
                [
                    -5.5 + (i % 3) as f32 * 0.7 - 0.7,
                    -5.0 + (i / 3) as f32 * 0.7 - 0.7,
                    0.4,
                ]
            })
            .collect();
        let n = gpos.len();
        self.glyphs.positions = gpos.clone();
        self.glyphs.vectors = vec![[0.0, 0.0, 1.0]; n];
        self.glyphs.scale = 0.7;
        self.glyphs.use_default_colour = true;
        self.glyphs.default_colour = [0.55, 0.10, 0.75, 1.0];
        self.glyphs.glyph_type = GlyphType::Arrow;
        self.glyphs.settings.pick_id = PickId(GLYPH);
        self.instance_lookup.insert(GLYPH, gpos);
        self.model_matrices.insert(GLYPH, Mat4::IDENTITY);
        self.labels.insert(GLYPH, ("Glyphs".into(), None));

        // Multi-strip polyline (nodes / segments / strips).
        let (positions, strips) = helix_strips(Vec3::new(5.5, 5.0, 0.4), 3);
        self.polyline.positions = positions.clone();
        self.polyline.strip_lengths = strips.clone();
        self.polyline.default_colour = [0.10, 0.52, 0.18, 1.0];
        self.polyline.line_width = 4.0;
        self.polyline.settings.pick_id = PickId(POLY);
        self.model_matrices.insert(POLY, Mat4::IDENTITY);
        self.labels.insert(POLY, ("Polyline".into(), None));

        // Tensor glyphs (instances): four ellipsoids with varied shape / basis.
        let tpos = vec![
            [4.9_f32, -0.6, 0.4],
            [4.9, 0.6, 0.4],
            [6.1, -0.6, 0.4],
            [6.1, 0.6, 0.4],
        ];
        self.tensor.positions = tpos.clone();
        self.tensor.eigenvalues = vec![
            [0.8, 0.7, 0.6],
            [1.2, 0.3, 0.3],
            [0.3, 0.3, 1.2],
            [1.0, 0.6, 0.2],
        ];
        let c = std::f32::consts::FRAC_1_SQRT_2;
        self.tensor.eigenvectors = vec![
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[c, c, 0.0], [-c, c, 0.0], [0.0, 0.0, 1.0]],
            [[0.866, 0.0, 0.5], [0.0, 1.0, 0.0], [-0.5, 0.0, 0.866]],
            [[0.5, 0.866, 0.0], [-0.866, 0.5, 0.0], [0.0, 0.0, 1.0]],
        ];
        self.tensor.scale = 0.5;
        self.tensor.colourmap_id = Some(ColourmapId(BuiltinColourmap::Coolwarm as usize));
        self.tensor.settings.pick_id = PickId(TENSOR);
        self.instance_lookup.insert(TENSOR, tpos);
        self.labels.insert(TENSOR, ("Tensor glyphs".into(), None));

        // Sprites (instances): an arc of graduated size / colour.
        {
            let count = 8usize;
            let centre = Vec3::new(0.0, 0.0, 1.0);
            let radius = 1.6_f32;
            let mut positions = Vec::with_capacity(count);
            let mut sizes = Vec::with_capacity(count);
            let mut colours = Vec::with_capacity(count);
            for i in 0..count {
                let t = i as f32 / (count - 1) as f32;
                let angle = std::f32::consts::FRAC_PI_4 + std::f32::consts::PI * 0.5 * t;
                positions.push([
                    centre.x + radius * angle.cos(),
                    centre.y,
                    centre.z + radius * angle.sin(),
                ]);
                sizes.push(14.0 + 20.0 * t);
                colours.push([1.0, 0.95 - 0.5 * t, 0.15 + 0.1 * t, 1.0]);
            }
            self.sprites.positions = positions.clone();
            self.sprites.sizes = sizes;
            self.sprites.colours = colours;
            self.sprites.default_size = 24.0;
            self.sprites.depth_write = true;
            self.sprites.settings.pick_id = PickId(SPRITE);
            self.instance_lookup.insert(SPRITE, positions);
            self.labels.insert(SPRITE, ("Sprites".into(), None));
        }

        // Streamtube / tube / ribbon (segments / strips).
        {
            let (pos, lens) = spiral_strips(Vec3::new(0.0, -5.0, 0.6), 2);
            self.streamtube.positions = pos;
            self.streamtube.strip_lengths = lens;
            self.streamtube.radius = 0.12;
            self.streamtube.colour = [0.05, 0.50, 0.40, 1.0];
            self.streamtube.settings.pick_id = PickId(STREAMTUBE);
            self.labels.insert(STREAMTUBE, ("Streamtube".into(), None));
        }
        {
            let (pos, lens) = wave_strips(Vec3::new(11.0, 0.0, 0.4), 2);
            self.tube.positions = pos;
            self.tube.strip_lengths = lens;
            self.tube.radius = 0.14;
            self.tube.colour = [0.75, 0.28, 0.05, 1.0];
            self.tube.settings.pick_id = PickId(TUBE);
            self.labels.insert(TUBE, ("Tube".into(), None));
        }
        {
            let (pos, lens) = wave_strips(Vec3::new(5.5, -5.0, 0.4), 2);
            self.ribbon.positions = pos;
            self.ribbon.strip_lengths = lens;
            self.ribbon.width = 0.4;
            self.ribbon.colour = [0.38, 0.12, 0.62, 1.0];
            self.ribbon.settings.pick_id = PickId(RIBBON);
            self.labels.insert(RIBBON, ("Ribbon".into(), None));
        }

        // Ray-marched volume: a 16^3 sphere-shaped scalar field.
        {
            let dims = [16u32, 16, 16];
            let data = sphere_field(dims);
            let vol_id = ctx
                .session
                .resources_mut()
                .upload_volume(ctx.device, ctx.queue, &data, dims);
            self.volume_id = Some(vol_id);
            self.volume_data = Some(Arc::new(VolumeData {
                data,
                dims,
                origin: [0.0, 0.0, 0.0],
                spacing: [VOL_EXTENT / dims[0] as f32; 3],
            }));
            self.labels.insert(VOLUME, ("Volume".into(), None));

            // Surface slice shares the volume; it needs its own plane mesh.
            let plane = primitives::plane(3.0, 3.0);
            if let Ok(id) = ctx
                .session
                .resources_mut()
                .upload_mesh_data(ctx.device, &plane)
            {
                self.slice_mesh_id = Some(id);
                self.labels.insert(SLICE, ("Surface slice".into(), None));
            }
        }

        // Gaussian splats: a 3x3 grid in local space.
        {
            let mut positions = Vec::with_capacity(9);
            for row in -1..=1 {
                for col in -1..=1 {
                    positions.push([col as f32 * 0.7, 0.0, row as f32 * 0.7]);
                }
            }
            let data = splat_data(&positions);
            if let Ok(id) = ctx
                .session
                .resources_mut()
                .upload_gaussian_splat(ctx.device, ctx.queue, &data)
            {
                self.splat_id = Some(id);
                self.instance_lookup.insert(SPLAT, positions.clone());
                self.model_matrices.insert(SPLAT, splat_model());
                self.labels.insert(SPLAT, ("Gaussian splats".into(), None));
            }
            self.splat_positions = positions;
        }

        // Volume mesh: a capsule hex mesh rendered as an opaque boundary surface.
        {
            let data = capsule_volume_mesh(Vec3::new(-5.5, 5.0, 0.5));
            if let Ok(item) = ctx
                .session
                .resources_mut()
                .upload_volume_mesh(ctx.device, &data)
            {
                self.vmesh_mesh_id = Some(item.boundary_mesh_id);
                self.vmesh_face_to_cell = item.face_to_cell;
                self.vmesh_data = Some(Arc::new(data));
                self.labels.insert(VMESH, ("Volume mesh".into(), None));
            }
        }

        // GPU marching cubes: a gyroid field.
        {
            let vol = gyroid_volume(Vec3::new(11.0, -5.0, 0.0));
            let vol = Arc::new(vol);
            if let Ok(id) = ctx
                .session
                .resources_mut()
                .upload_volume_for_mc(ctx.device, ctx.queue, &vol)
            {
                self.mc_id = Some(id);
                self.mc_data = Some(vol);
                self.labels.insert(MC, ("Marching cubes".into(), None));
            }
        }

        // Screen image: a small checkerboard sprite in the corner.
        self.screen_pixels = checkerboard(48, 48);
        self.labels.insert(SCREEN, ("Screen image".into(), None));

        // GPU implicit surface (blended spheres).
        self.labels.insert(IMPLICIT, ("GPU implicit".into(), None));

        // Decal straddling the cube's +Z face (cube half-extent 0.7).
        {
            let (dw, dh, rgba) = decal_texture();
            if let Ok(tex) = ctx
                .session
                .resources_mut()
                .upload_texture(ctx.device, ctx.queue, dw, dh, &rgba)
            {
                self.decal_tex = Some(tex);
                self.decal_transform = Mat4::from_translation(cube_pos + Vec3::new(0.0, 0.0, 0.7))
                    * Mat4::from_scale(Vec3::new(1.0, 1.0, 0.5));
                self.labels.insert(DECAL, ("Decal".into(), None));
            }
        }

        ctx.session
            .set_selection_outline(true, [1.0, 0.85, 0.2, 1.0], 2.5);
        let cam = ctx.session.camera_mut();
        cam.distance = 34.0;
        cam.orientation = glam::Quat::from_rotation_x(0.85);
    }

    fn update(&mut self, ctx: &mut ShowcaseCtx) {
        let device = ctx.device;
        let queue = ctx.queue;
        let mask = self.mask();

        ctx.drive_camera();

        let session = &mut *ctx.session;

        // This is a picking showcase, not a manipulation one. The session stamps
        // a transform gizmo whenever a scene node is selected (mesh outlines go
        // through the session selection), so clear it here to keep the gizmo out.
        session.frame_data_mut().interaction.gizmo_model = None;

        // Inject every non-mesh item (rendering + picking both read the frame).
        self.inject_items(session);

        // Click vs rect: a big enough drag is a rect pick, anything else a click.
        let pointer = session.action_frame().pointer;
        if pointer.drag_started {
            self.drag_start = pointer.cursor;
        }
        if pointer.dragging {
            self.drag_cur = pointer.cursor;
        }
        let released = self.dragging_prev && !pointer.dragging;
        self.dragging_prev = pointer.dragging;

        let mut single_pick: Option<Vec2> = None;
        if !mask.is_empty() {
            if released {
                let a = self.drag_start;
                let b = self.drag_cur.or(pointer.cursor);
                if let (Some(a), Some(b)) = (a, b) {
                    let (min, max) = (a.min(b), a.max(b));
                    if (max - min).length() > 6.0 {
                        let result = session.pick_rect_gpu(device, queue, min, max, mask);
                        self.clear_selection(session);
                        for id in result.objects.clone() {
                            self.record(session, id, None);
                        }
                        for (id, sub) in result.elements.clone() {
                            self.record(session, id, Some(sub));
                        }
                        self.rect_status = Some(format!(
                            "{} objects, {} elements",
                            result.objects.len(),
                            result.elements.len()
                        ));
                        self.last = None;
                        self.marker_pos = None;
                    } else {
                        single_pick = Some(b);
                    }
                }
                self.drag_start = None;
                self.drag_cur = None;
            } else if pointer.clicked {
                single_pick = pointer.cursor;
            }

            if let Some(cursor) = single_pick {
                match session.pick_gpu(device, queue, cursor, mask) {
                    Some(hit) => {
                        let name = self
                            .labels
                            .get(&hit.id)
                            .map(|(n, _)| n.clone())
                            .unwrap_or_else(|| format!("#{}", hit.id));
                        self.last = Some(HitInfo {
                            name,
                            sub: sub_desc(hit.sub_object),
                            pos: hit.world_pos,
                            scalar: hit.scalar_value,
                        });
                        self.marker_pos = Some(hit.world_pos);
                        self.rect_status = None;
                        self.clear_selection(session);
                        self.record(session, hit.id, hit.sub_object);
                    }
                    None => {
                        self.last = None;
                        self.marker_pos = None;
                        self.clear_selection(session);
                    }
                }
            }
        }

        // Sub-object highlight: build the per-frame ref from the CPU lookups.
        if !self.sub.is_empty() {
            // PolylineSelectionInfo is not Clone, so rebuild the maps each frame.
            let mut polyline_lookup = HashMap::new();
            polyline_lookup.insert(
                POLY,
                PolylineSelectionInfo {
                    positions: self.polyline.positions.clone(),
                    strip_lengths: self.polyline.strip_lengths.clone(),
                },
            );
            let mut curve_family_lookup = HashMap::new();
            curve_family_lookup.insert(
                STREAMTUBE,
                PolylineSelectionInfo {
                    positions: self.streamtube.positions.clone(),
                    strip_lengths: self.streamtube.strip_lengths.clone(),
                },
            );
            curve_family_lookup.insert(
                TUBE,
                PolylineSelectionInfo {
                    positions: self.tube.positions.clone(),
                    strip_lengths: self.tube.strip_lengths.clone(),
                },
            );
            curve_family_lookup.insert(
                RIBBON,
                PolylineSelectionInfo {
                    positions: self.ribbon.positions.clone(),
                    strip_lengths: self.ribbon.strip_lengths.clone(),
                },
            );

            let mut voxel_lookup = HashMap::new();
            if self.volume_id.is_some() {
                voxel_lookup.insert(
                    VOLUME,
                    VolumeSelectionInfo {
                        dims: [16, 16, 16],
                        bbox_min: [0.0, 0.0, 0.0],
                        bbox_max: [VOL_EXTENT; 3],
                        model: vol_model().to_cols_array_2d(),
                    },
                );
            }
            let mut cell_lookup = HashMap::new();
            if let Some(data) = self.vmesh_data.as_ref() {
                cell_lookup.insert(
                    VMESH,
                    CellSelectionInfo {
                        positions: data.positions.clone(),
                        cells: data.cells.clone(),
                    },
                );
            }

            let fd = session.frame_data_mut();
            fd.interaction.sub_selection = Some(
                SubSelectionRef::new(
                    &self.sub,
                    self.mesh_lookup.clone(),
                    self.model_matrices.clone(),
                    self.point_positions.clone(),
                )
                .with_voxels(voxel_lookup)
                .with_cells(cell_lookup)
                .with_polylines(polyline_lookup)
                .with_curve_families(curve_family_lookup)
                .with_instances(self.instance_lookup.clone()),
            );
            fd.interaction.sub_highlight_face_fill_colour = [1.0, 0.85, 0.0, 0.3];
            fd.interaction.sub_highlight_edge_colour = [1.0, 0.85, 0.0, 1.0];
            fd.interaction.sub_highlight_edge_width_px = 4.0;
            fd.interaction.sub_highlight_vertex_size_px = 13.0;
        }

        // Draw the drag rectangle (overlay positions are logical points, the
        // same space as the pick cursor).
        if pointer.dragging {
            if let (Some(a), Some(b)) = (self.drag_start, self.drag_cur) {
                let min = a.min(b);
                let size = a.max(b) - a.min(b);
                let rect = OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 0.0 },
                    [min.x, min.y],
                    [size.x, size.y],
                )
                .with_fill(OverlayFill::Solid([0.3, 0.6, 1.0, 0.12]))
                .with_border([0.5, 0.8, 1.0, 0.9], 1.5);
                session.frame_data_mut().overlays.shapes.push(rect);
            }
        }

        // Move the marker onto the last single-click hit point (or hide it).
        if let Some(marker) = self.marker {
            let mut settings = ItemSettings::default();
            settings.unlit = true;
            match self.marker_pos {
                Some(p) => {
                    session.scene_mut().set_local_transform(
                        marker,
                        Mat4::from_translation(p) * Mat4::from_scale(Vec3::splat(0.12)),
                    );
                }
                None => settings.hidden = true,
            }
            session.scene_mut().set_appearance(marker, settings);
        }
    }

    fn description(&self) -> &str {
        "Pick levels: enable object / point / edge / face in the panel, then \
         click or drag a box. One of every pickable item type is in the scene, \
         and each highlights at the object and sub-object level."
    }

    fn has_controls(&self) -> bool {
        true
    }

    fn panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Pick levels");
        ui.checkbox(&mut self.pick_object, "Object");
        ui.checkbox(
            &mut self.pick_point,
            "Point-like (vertex / point / cell / voxel / splat / instance / node)",
        );
        ui.checkbox(&mut self.pick_edge, "Edge-like (edge / segment)");
        ui.checkbox(&mut self.pick_face, "Face-like (face / strip)");

        ui.separator();
        ui.weak("Click to pick, drag a box to marquee-select.");
        ui.separator();
        ui.heading("Last hit");
        if let Some(h) = &self.last {
            ui.label(format!("{}: {}", h.name, h.sub));
            ui.label(format!(
                "pos ({:.2}, {:.2}, {:.2})",
                h.pos.x, h.pos.y, h.pos.z
            ));
            if let Some(s) = h.scalar {
                ui.label(format!("scalar {s:.3}"));
            }
        } else if let Some(status) = &self.rect_status {
            ui.label(format!("Rect: {status}"));
        } else {
            ui.weak("click or drag an item");
        }
    }
}

// ---------------------------------------------------------------------------
// Scene helpers
// ---------------------------------------------------------------------------

fn add_mesh(ctx: &mut SetupCtx, mesh: MeshId, pos: Vec3, colour: [f32; 3]) -> NodeId {
    ctx.session.scene_mut().add(
        Some(mesh),
        Mat4::from_translation(pos),
        Material::pbr(colour, 0.1, 0.5),
    )
}

fn sub_desc(sub: Option<SubObjectRef>) -> String {
    match sub {
        None => "object".into(),
        Some(SubObjectRef::Face(i)) => format!("face {i}"),
        Some(SubObjectRef::Vertex(i)) => format!("vertex {i}"),
        Some(SubObjectRef::Edge(i)) => format!("edge {i}"),
        Some(SubObjectRef::Point(i)) => format!("point {i}"),
        Some(SubObjectRef::Voxel(i)) => format!("voxel {i}"),
        Some(SubObjectRef::Cell(i)) => format!("cell {i}"),
        Some(SubObjectRef::Splat(i)) => format!("splat {i}"),
        Some(SubObjectRef::Instance(i)) => format!("instance {i}"),
        Some(SubObjectRef::Segment(i)) => format!("segment {i}"),
        Some(SubObjectRef::Strip(i)) => format!("strip {i}"),
        _ => "sub-object".into(),
    }
}

/// A noisy solid sphere of points, scalar = height, for the point-cloud item.
fn noisy_sphere(centre: Vec3, radius: f32, n: usize) -> (Vec<[f32; 3]>, Vec<f32>) {
    let mut pos = Vec::with_capacity(n);
    let mut sca = Vec::with_capacity(n);
    for i in 0..n {
        let a = hashf(i as u32, 1) * std::f32::consts::TAU;
        let b = hashf(i as u32, 2) * std::f32::consts::PI;
        let r = radius * hashf(i as u32, 3).cbrt();
        let p = centre + Vec3::new(b.sin() * a.cos(), b.sin() * a.sin(), b.cos()) * r;
        pos.push([p.x, p.y, p.z]);
        sca.push(p.z);
    }
    (pos, sca)
}

/// Three vertical helix strips packed into one multi-strip polyline.
fn helix_strips(centre: Vec3, count: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut positions = Vec::new();
    let mut strips = Vec::new();
    let per = 48u32;
    for s in 0..count {
        let ox = (s as f32 - (count as f32 - 1.0) * 0.5) * 1.1;
        for i in 0..per {
            let t = i as f32 / (per - 1) as f32;
            let a = t * std::f32::consts::TAU * 2.0 + s as f32;
            positions.push([
                centre.x + ox + a.cos() * 0.35,
                centre.y + a.sin() * 0.35,
                centre.z + t * 1.8,
            ]);
        }
        strips.push(per);
    }
    (positions, strips)
}

/// Corkscrew strips along +Z, for the streamtube.
fn spiral_strips(centre: Vec3, count: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
    let per = 30u32;
    let mut positions = Vec::new();
    let mut strips = Vec::new();
    for s in 0..count {
        let ox = (s as f32 - (count as f32 - 1.0) * 0.5) * 1.4;
        for i in 0..per {
            let t = i as f32 / (per - 1) as f32;
            let a = t * std::f32::consts::TAU * 1.5;
            positions.push([
                centre.x + ox + a.cos(),
                centre.y + a.sin(),
                centre.z + t * 2.5,
            ]);
        }
        strips.push(per);
    }
    (positions, strips)
}

/// Sine-wave strips along +X, for the tube and ribbon.
fn wave_strips(centre: Vec3, count: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
    let per = 24u32;
    let mut positions = Vec::new();
    let mut strips = Vec::new();
    for s in 0..count {
        let oy = (s as f32 - (count as f32 - 1.0) * 0.5) * 1.4;
        for i in 0..per {
            let t = i as f32 / (per - 1) as f32;
            let x = centre.x - 1.5 + t * 3.0;
            let z = centre.z + (t * std::f32::consts::TAU).sin() * 0.7;
            positions.push([x, centre.y + oy, z]);
        }
        strips.push(per);
    }
    (positions, strips)
}

/// A 16^3 (or other) radial falloff scalar field for the ray-marched volume.
fn sphere_field(dims: [u32; 3]) -> Vec<f32> {
    let n = (dims[0] * dims[1] * dims[2]) as usize;
    let mut data = vec![0.0f32; n];
    let c = Vec3::new(
        dims[0] as f32 * 0.5,
        dims[1] as f32 * 0.5,
        dims[2] as f32 * 0.5,
    );
    let radius = dims[0] as f32 * 0.5;
    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let flat = (ix + iy * dims[0] + iz * dims[0] * dims[1]) as usize;
                let p = Vec3::new(ix as f32 + 0.5, iy as f32 + 0.5, iz as f32 + 0.5);
                data[flat] = (1.0 - (p - c).length() / radius).max(0.0);
            }
        }
    }
    data
}

/// A gyroid field spanning one period, centred at `centre` in world space.
fn gyroid_volume(centre: Vec3) -> VolumeData {
    // One half-period (pi) in each axis so the surface stays a compact ~3-unit
    // cube instead of the ~6 units a full period would span.
    let n = 32u32;
    let step = std::f32::consts::PI / (n - 1) as f32;
    let half = std::f32::consts::FRAC_PI_2;
    let origin = [centre.x - half, centre.y - half, centre.z - half];
    let mut data = Vec::with_capacity((n * n * n) as usize);
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let x = ix as f32 * step;
                let y = iy as f32 * step;
                let z = iz as f32 * step;
                data.push(x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos());
            }
        }
    }
    VolumeData {
        data,
        dims: [n, n, n],
        origin,
        spacing: [step; 3],
    }
}

/// Gaussian splat data: N small spheres, teal-to-orange across the set.
fn splat_data(positions: &[[f32; 3]]) -> GaussianSplatData {
    const SH0_C: f32 = 0.28209479177;
    let n = positions.len();
    let sh_coefficients = (0..n)
        .flat_map(|i| {
            let t = i as f32 / (n as f32 - 1.0).max(1.0);
            let r = 0.05 + 0.70 * t;
            let g = 0.45 - 0.17 * t;
            let b = 0.45 - 0.40 * t;
            [(r - 0.5) / SH0_C, (g - 0.5) / SH0_C, (b - 0.5) / SH0_C]
        })
        .collect();
    GaussianSplatData {
        positions: positions.to_vec(),
        scales: vec![[0.18, 0.18, 0.18]; n],
        rotations: vec![[1.0, 0.0, 0.0, 0.0]; n],
        opacities: vec![0.85; n],
        sh_coefficients,
        sh_degree: ShDegree::Zero,
    }
}

/// A capsule hex mesh centred at `centre`, elongated along world Y. 5x5
/// cross-section grid, 8 axial layers: 128 hex cells with an elliptic
/// square-to-disk mapping.
fn capsule_volume_mesh(centre: Vec3) -> VolumeMeshData {
    const N: usize = 5;
    const NZ: usize = 9;
    let radius = 0.7f32;
    let cyl_half = 1.0f32;
    let z_extent = cyl_half + radius;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(N * N * NZ);
    for k in 0..NZ {
        let t = k as f32 / (NZ - 1) as f32;
        let axial = -z_extent + 2.0 * z_extent * t;
        let r = if axial.abs() <= cyl_half {
            radius
        } else {
            let dz = axial.abs() - cyl_half;
            (radius * radius - dz * dz).max(0.0).sqrt()
        };
        for j in 0..N {
            for i in 0..N {
                let u = 2.0 * i as f32 / (N - 1) as f32 - 1.0;
                let v = 2.0 * j as f32 / (N - 1) as f32 - 1.0;
                let dx = r * u * (1.0 - v * v / 2.0).sqrt();
                let dz = r * v * (1.0 - u * u / 2.0).sqrt();
                positions.push([centre.x + dx, centre.y + axial, centre.z + dz]);
            }
        }
    }

    let vi = |i: usize, j: usize, k: usize| -> u32 { (i + j * N + k * N * N) as u32 };
    let n_lat = N - 1;
    let n_axial = NZ - 1;
    let total = n_lat * n_lat * n_axial;
    let mut cells: Vec<[u32; 8]> = Vec::with_capacity(total);
    let mut scalars: Vec<f32> = Vec::with_capacity(total);
    for k in 0..n_axial {
        for j in 0..n_lat {
            for i in 0..n_lat {
                cells.push([
                    vi(i, j, k),
                    vi(i + 1, j, k),
                    vi(i + 1, j + 1, k),
                    vi(i, j + 1, k),
                    vi(i, j, k + 1),
                    vi(i + 1, j, k + 1),
                    vi(i + 1, j + 1, k + 1),
                    vi(i, j + 1, k + 1),
                ]);
                let idx = i + j * n_lat + k * n_lat * n_lat;
                scalars.push(idx as f32 / (total - 1).max(1) as f32);
            }
        }
    }

    let mut data = VolumeMeshData::default();
    data.positions = positions;
    data.cells = cells;
    data.cell_scalars.insert("scalar".to_string(), scalars);
    data
}

/// An RGBA checkerboard for the screen-image item.
fn checkerboard(w: u32, h: u32) -> Vec<[u8; 4]> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            if (x / 6 + y / 6) % 2 == 0 {
                [255, 255, 255, 220]
            } else {
                [30, 30, 30, 220]
            }
        })
        .collect()
}

/// A target-sticker texture (concentric red / white rings on transparent) for
/// the decal. Returns `(w, h, rgba)`.
fn decal_texture() -> (u32, u32, Vec<u8>) {
    const W: u32 = 64;
    const H: u32 = 64;
    let mut rgba = vec![0u8; (W * H * 4) as usize];
    let cx = (W as f32 - 1.0) * 0.5;
    let cy = (H as f32 - 1.0) * 0.5;
    let max_r = W as f32 * 0.5;
    for y in 0..H {
        for x in 0..W {
            let r = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() / max_r;
            if r > 0.95 {
                continue;
            }
            let (cr, cg, cb) = if (r * 4.0) as u32 % 2 == 0 {
                (230u8, 40, 40)
            } else {
                (245u8, 245, 245)
            };
            let idx = ((y * W + x) * 4) as usize;
            rgba[idx] = cr;
            rgba[idx + 1] = cg;
            rgba[idx + 2] = cb;
            rgba[idx + 3] = 255;
        }
    }
    (W, H, rgba)
}

/// Small deterministic hash in [0, 1) from an index and a salt.
fn hashf(i: u32, salt: u32) -> f32 {
    let mut x = i
        .wrapping_mul(747796405)
        .wrapping_add(salt.wrapping_mul(2891336453));
    x ^= x >> 16;
    x = x.wrapping_mul(2246822519);
    x ^= x >> 13;
    (x & 0x00ff_ffff) as f32 / 0x0100_0000 as f32
}
