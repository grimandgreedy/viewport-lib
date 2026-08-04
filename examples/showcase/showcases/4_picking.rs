//! Picking: choose which levels to pick (object, point-like, edge-like,
//! face-like), then click or drag a box to select. Every item type highlights
//! at both the object level (outline / selected tint) and the sub-object level
//! (face fill, vertex/point sprites, segment/strip, glyph instance).
//!
//! Uses the GPU pick backend (`pick_gpu` / `pick_rect_gpu`), so the wgpu device
//! and queue are read each frame from `ShowcaseCtx`. Sub-object highlighting is
//! driven by a `SubSelection` turned into a per-frame `SubSelectionRef`.

use std::collections::{HashMap, HashSet};

use eframe::egui;
use glam::{Mat4, Vec2, Vec3};
use viewport_lib::{
    BuiltinColourmap, ColourmapId, GlyphItem, GlyphType, ItemSettings, Material, MeshId, NodeId,
    OverlayFill, OverlayShape, OverlayShapeItem, PickId, PickMask, PointCloudItem, PolylineItem,
    PolylineSelectionInfo, SubObjectRef, SubSelection, SubSelectionRef, primitives,
};

use crate::showcase::{SetupCtx, Showcase, ShowcaseCtx};

// Pick ids for the injected (non-scene-node) items. High values so they never
// collide with scene node ids (a scene node's pick_id is forced to its node id).
const PC: u64 = 1001;
const GLYPH: u64 = 1002;
const POLY: u64 = 1003;

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

    // Injected each frame so their `settings.selected` can be toggled.
    pc: PointCloudItem,
    glyphs: GlyphItem,
    polyline: PolylineItem,

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
    fn clear_selection(&mut self, session: &mut viewport_lib::ViewportSession) {
        session.selection_mut().clear();
        self.selected_objects.clear();
        self.sub.clear();
    }

    /// Record a hit: sub-object hits go to the SubSelection; object hits outline
    /// the mesh (via the session selection) or flag the injected item.
    fn record(
        &mut self,
        session: &mut viewport_lib::ViewportSession,
        id: u64,
        sub: Option<SubObjectRef>,
    ) {
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
}

impl Showcase for PickingShowcase {
    fn name(&self) -> &str {
        "Picking"
    }

    fn setup(&mut self, ctx: &mut SetupCtx) {
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
        let cube_pos = Vec3::new(-3.5, 0.0, 0.7);
        let sphere_pos = Vec3::new(-1.3, 0.0, 0.9);
        let cube_node = add_mesh(ctx, cube, cube_pos, [0.85, 0.4, 0.35]);
        let sphere_node = add_mesh(ctx, sphere, sphere_pos, [0.4, 0.55, 0.85]);
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
        let (pos, sca) = noisy_sphere(Vec3::new(1.2, 0.0, 1.1), 0.9, 400);
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
            .map(|i| [3.4 + (i % 3) as f32 * 0.7, -0.7 + (i / 3) as f32 * 0.7, 0.4])
            .collect();
        let n = gpos.len();
        self.glyphs.positions = gpos.clone();
        self.glyphs.vectors = vec![[0.0, 0.0, 1.0]; n];
        self.glyphs.scale = 0.7;
        self.glyphs.use_default_colour = true;
        self.glyphs.default_colour = [0.75, 0.2, 1.0, 1.0];
        self.glyphs.glyph_type = GlyphType::Arrow;
        self.glyphs.settings.pick_id = PickId(GLYPH);
        self.instance_lookup.insert(GLYPH, gpos);
        self.model_matrices.insert(GLYPH, Mat4::IDENTITY);
        self.labels.insert(GLYPH, ("Glyphs".into(), None));

        // Multi-strip polyline (nodes / segments / strips).
        let (positions, strips) = helix_strips(Vec3::new(0.5, -3.0, 0.4), 3);
        self.polyline.positions = positions.clone();
        self.polyline.strip_lengths = strips.clone();
        self.polyline.default_colour = [0.25, 0.85, 0.4, 1.0];
        self.polyline.line_width = 4.0;
        self.polyline.settings.pick_id = PickId(POLY);
        self.model_matrices.insert(POLY, Mat4::IDENTITY);
        self.labels.insert(POLY, ("Polyline".into(), None));

        ctx.session
            .set_selection_outline(true, [1.0, 0.85, 0.2, 1.0], 2.5);
        let cam = ctx.session.camera_mut();
        cam.distance = 16.0;
        cam.orientation = glam::Quat::from_rotation_x(0.8);
    }

    fn update(&mut self, ctx: &mut ShowcaseCtx) {
        let device = ctx.device;
        let queue = ctx.queue;
        let mask = self.mask();

        ctx.drive_camera();

        let session = &mut *ctx.session;

        // Inject the non-mesh items, flagging the selected ones. They must be in
        // the frame for both rendering and picking.
        self.pc.settings.selected = self.selected_objects.contains(&PC);
        self.glyphs.settings.selected = self.selected_objects.contains(&GLYPH);
        self.polyline.settings.selected = self.selected_objects.contains(&POLY);
        let fd = session.frame_data_mut();
        fd.scene.point_clouds.push(self.pc.clone());
        fd.scene.glyphs.push(self.glyphs.clone());
        fd.scene.polylines.push(self.polyline.clone());

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
            // PolylineSelectionInfo is not Clone, so rebuild it each frame.
            let mut polyline_lookup = HashMap::new();
            polyline_lookup.insert(
                POLY,
                PolylineSelectionInfo {
                    positions: self.polyline.positions.clone(),
                    strip_lengths: self.polyline.strip_lengths.clone(),
                },
            );
            let fd = session.frame_data_mut();
            fd.interaction.sub_selection = Some(
                SubSelectionRef::new(
                    &self.sub,
                    self.mesh_lookup.clone(),
                    self.model_matrices.clone(),
                    self.point_positions.clone(),
                )
                .with_polylines(polyline_lookup)
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
         click or drag a box. Meshes, a point cloud, glyphs, and a multi-strip \
         polyline all highlight at the object and sub-object level."
    }

    fn has_controls(&self) -> bool {
        true
    }

    fn panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Pick levels");
        ui.checkbox(&mut self.pick_object, "Object");
        ui.checkbox(
            &mut self.pick_point,
            "Point-like (vertex / point / instance / node)",
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
