//! GPU object-ID picking: render the scene to an offscreen R32Uint texture
//! and read back the single pixel under the cursor.

use super::*;

/// Item types the GPU pick pass can draw with the shared surface pick pipeline.
///
/// Each variant knows which pick masks it can answer. A type contributes draws
/// only when the caller asked for a level it can resolve, so the mask selects
/// which geometry is rasterised rather than filtering the read-back id. Types
/// with their own pick pipeline (glyphs, sprites, polylines) are handled in
/// their own pass blocks, not here.
#[derive(Clone, Copy)]
enum PickItemType {
    /// Mesh-backed surfaces: scene surfaces and volume-mesh boundaries, resolved
    /// against `mesh_store`.
    Surface,
    /// Tube-family geometry: streamtubes, tubes, and ribbons. These build an
    /// owned connected mesh each frame into the renderer's tube gpu-data vecs
    /// rather than living in `mesh_store`. Object-level only.
    Curve,
    /// Glyph and tensor-glyph sets: instanced base meshes drawn with a dedicated
    /// pick pipeline that reuses the render vertex transform. Object-level only.
    Glyph,
    /// Sprite sets: camera-facing quads expanded in the vertex shader, drawn with
    /// a dedicated pick pipeline that reuses the render expansion. Object-level.
    Sprite,
    /// Polylines: screen-space thick lines expanded per segment in the vertex
    /// shader, drawn with a dedicated pick pipeline. Object-level.
    Polyline,
}

/// Test the screen-space overlay images against a single click position and
/// return the topmost hit's pick id, mirroring the CPU pick's screen-image
/// section (`point.rs`, section 10). Screen images are drawn on top of all 3D
/// geometry with no world-space representation, so this is a plain rect test
/// rather than anything the render-based id pass could answer; both the GPU
/// object pick and the GPU rect pick call this instead of drawing a pass for
/// them.
fn screen_image_hit_at(
    items: &[crate::ScreenImageItem],
    viewport_size: glam::Vec2,
    click_pos: glam::Vec2,
) -> Option<u64> {
    for item in items {
        if item.settings.pick_id == crate::renderer::PickId::NONE
            || item.width == 0
            || item.height == 0
        {
            continue;
        }
        let img_w = item.width as f32 * item.scale;
        let img_h = item.height as f32 * item.scale;
        let [sx, sy] = crate::renderer::types::viewport_anchored_top_left(
            item.anchor_x,
            item.anchor_y,
            [img_w, img_h],
            [viewport_size.x, viewport_size.y],
        );
        if click_pos.x >= sx
            && click_pos.x <= sx + img_w
            && click_pos.y >= sy
            && click_pos.y <= sy + img_h
        {
            return Some(item.settings.pick_id.0);
        }
    }
    None
}

/// Collect the pick ids of every screen-space overlay image whose screen rect
/// overlaps the query rect, mirroring [`screen_image_hit_at`] but for a rect
/// query instead of a point.
fn screen_image_hits_in_rect(
    items: &[crate::ScreenImageItem],
    viewport_size: glam::Vec2,
    rect_min: glam::Vec2,
    rect_max: glam::Vec2,
) -> Vec<u64> {
    let mut hits = Vec::new();
    for item in items {
        if item.settings.pick_id == crate::renderer::PickId::NONE
            || item.width == 0
            || item.height == 0
        {
            continue;
        }
        let img_w = item.width as f32 * item.scale;
        let img_h = item.height as f32 * item.scale;
        let [sx, sy] = crate::renderer::types::viewport_anchored_top_left(
            item.anchor_x,
            item.anchor_y,
            [img_w, img_h],
            [viewport_size.x, viewport_size.y],
        );
        let overlaps = rect_min.x <= sx + img_w
            && rect_max.x >= sx
            && rect_min.y <= sy + img_h
            && rect_max.y >= sy;
        if overlaps {
            hits.push(item.settings.pick_id.0);
        }
    }
    hits
}

/// Snap priority for a resolved sub-object, higher wins. A point-like feature
/// (surface vertex, curve / cloud node, glyph instance, splat) snaps ahead of a
/// one-dimensional edge / segment, which snaps ahead of a surface face or plain
/// object hit. Used by [`ViewportRenderer::snap_query_gpu`] to reduce the window
/// to the most specific feature the cursor is near.
fn snap_priority(sub: Option<SubObjectRef>) -> i32 {
    match sub {
        Some(
            SubObjectRef::Vertex(_)
            | SubObjectRef::Point(_)
            | SubObjectRef::Instance(_)
            | SubObjectRef::Splat(_),
        ) => 3,
        Some(SubObjectRef::Edge(_) | SubObjectRef::Segment(_)) => 2,
        _ => 1,
    }
}

impl PickItemType {
    /// Whether this type answers any level requested in `mask`. A type is drawn
    /// only when the caller asked for something it can resolve.
    fn satisfies(self, mask: PickMask) -> bool {
        match self {
            PickItemType::Surface => mask.intersects(
                PickMask::OBJECT
                    | PickMask::FACE
                    | PickMask::VERTEX
                    | PickMask::EDGE
                    | PickMask::CELL,
            ),
            // Streamtubes, tubes, and ribbons answer the whole object mask plus
            // the node/segment/strip levels a curve query may ask.
            PickItemType::Curve => mask.intersects(
                PickMask::OBJECT | PickMask::POLY_NODE | PickMask::SEGMENT | PickMask::STRIP,
            ),
            // Glyph and sprite sets answer the object mask plus the per-instance
            // level.
            PickItemType::Glyph | PickItemType::Sprite => {
                mask.intersects(PickMask::OBJECT | PickMask::INSTANCE)
            }
            // Polylines are object-level; they answer the whole object mask plus
            // the node/segment/strip levels a curve query may ask.
            PickItemType::Polyline => mask.intersects(
                PickMask::OBJECT | PickMask::POLY_NODE | PickMask::SEGMENT | PickMask::STRIP,
            ),
        }
    }
}

/// How to turn a pick's read-back sub-primitive index into a [`SubObjectRef`],
/// keyed by the hit object's `pick_id`. Built at submit time from the same
/// collections the pass draws, then consulted on read-back. Types that only
/// answer object level (decals, scatter volumes, voxel volumes) are absent
/// from the map and resolve to no sub-object.
#[derive(Clone, Copy)]
enum PickSubKind {
    /// Mesh surface or volume-mesh boundary: `primitive_index` is the triangle.
    /// FACE is the channel value directly; CELL maps through the retained
    /// boundary face-to-cell table; VERTEX / EDGE come from pipeline variants
    /// that write the nearest corner / edge id into the channel per pixel. No
    /// CPU geometry is consulted.
    Surface,
    /// Item drawn by a registered
    /// [`ItemTypePlugin`](crate::plugin_api::ItemTypePlugin):
    /// `primitive_index` is
    /// whatever the plugin's pick fragment wrote (the triangle index with
    /// `viewport_pick_prim_fs`). Refined by the named plugin's
    /// `resolve_sub_object`; a plugin without the hook stays object-level. The
    /// payload is the plugin's `type_name`, the `item_type_plugins` key.
    Plugin(&'static str),
    /// Glyph, tensor-glyph, or sprite set: `instance_index` is the instance.
    Instance,
    /// Polyline: `instance_index` is the segment; strip is resolved against the
    /// retained polyline items.
    Polyline,
    /// Streamtube, tube, or ribbon: `primitive_index` is a connected-mesh
    /// triangle, mapped to a segment / strip through the item's `tri_segment` /
    /// `tri_strip` tables.
    Curve,
    /// Point cloud: `instance_index` is the point.
    CloudPoint,
    /// Gaussian splat set: `instance_index` is the splat.
    Splat,
    /// Ray-marched volume: the primitive channel carries the flat index of the
    /// first in-threshold voxel the fragment marched to.
    Voxel,
}

/// One glyph or tensor-glyph set to draw into the pick pass. The group-1 bind
/// group (the set uniform + a per-set object-id uniform) is owned; the pipeline,
/// instance bind group, and mesh buffers are borrowed from prepared state.
struct GlyphPickDraw<'a> {
    pipeline: &'a crate::gpu::RenderPipeline,
    id_bind_group: crate::gpu::BindGroup,
    instance_bind_group: &'a crate::gpu::BindGroup,
    vertex_buffer: &'a crate::gpu::Buffer,
    index_buffer: &'a crate::gpu::Buffer,
    index_count: u32,
    instance_count: u32,
}

/// One sprite set to draw into the pick pass. The group-2 pick-id bind group is
/// owned; the sprite bind group and position buffer are borrowed from prepared
/// state. The pipeline and group-0 camera bind group are shared across sets.
struct SpritePickDraw<'a> {
    id_bind_group: crate::gpu::BindGroup,
    sprite_bind_group: &'a crate::gpu::BindGroup,
    vertex_buffer: &'a crate::gpu::Buffer,
    sprite_count: u32,
}

/// One polyline to draw into the pick pass. Same shape as [`SpritePickDraw`]:
/// the group-2 pick-id bind group is owned, the render bind group and segment
/// buffer are borrowed. The pipeline and group-0 pick camera bind group are
/// shared across polylines.
struct PolylinePickDraw<'a> {
    id_bind_group: crate::gpu::BindGroup,
    render_bind_group: &'a crate::gpu::BindGroup,
    vertex_buffer: &'a crate::gpu::Buffer,
    segment_count: u32,
}

/// One voxel volume to draw into the pick pass. The owned group-2 bind group
/// holds the object id; the group-1 render bind group (volume uniform + 3D
/// texture + samplers) and the unit-cube buffers are borrowed from prepared
/// `VolumeGpuData`.
struct VolumePickDraw<'a> {
    id_bind_group: crate::gpu::BindGroup,
    render_bind_group: &'a crate::gpu::BindGroup,
    vertex_buffer: &'a crate::gpu::Buffer,
    index_buffer: &'a crate::gpu::Buffer,
}

/// One GPU implicit-surface item to draw into the pick pass. The owned group-2
/// bind group holds the object id; the group-1 render bind group (the implicit
/// uniform) is borrowed from prepared `ImplicitGpuItem`. The pipeline and the
/// full camera bind group (group 0) are shared across items.
struct ImplicitPickDraw<'a> {
    id_bind_group: crate::gpu::BindGroup,
    render_bind_group: &'a crate::gpu::BindGroup,
}

/// One GPU marching-cubes item to draw into the pick pass. The owned group-1 bind
/// group holds the object id; each slab contributes a borrowed (vertex buffer,
/// indirect-args buffer) pair drawn with the reused MC surface indirect args.
struct McPickDraw<'a> {
    id_bind_group: crate::gpu::BindGroup,
    slabs: Vec<(&'a crate::gpu::Buffer, &'a crate::gpu::Buffer)>,
}

/// One point cloud to draw into the pick pass. The owned group-2 bind group
/// holds the object id; the group-1 render bind group (uniform + LUT + radius
/// buffer) and the position buffer are borrowed from prepared
/// `PointCloudGpuData`.
struct PointCloudPickDraw<'a> {
    id_bind_group: crate::gpu::BindGroup,
    render_bind_group: &'a crate::gpu::BindGroup,
    vertex_buffer: &'a crate::gpu::Buffer,
    point_count: u32,
}

/// One Gaussian splat set to draw into the pick pass. The owned group-2 bind
/// group holds the object id; the group-1 render bind group (the per-viewport
/// sorted-index / position / scale / rotation storage buffers) is borrowed
/// from the splat store's prepared viewport sort.
struct GaussianSplatPickDraw<'a> {
    id_bind_group: crate::gpu::BindGroup,
    render_bind_group: &'a crate::gpu::BindGroup,
    count: u32,
}

/// One image slice to draw into the pick pass. The owned group-2 bind group
/// holds the object id; the group-1 render bind group (`ImageSliceUniform` +
/// volume texture) is borrowed from prepared `ImageSliceGpuData`.
struct ImageSlicePickDraw<'a> {
    id_bind_group: crate::gpu::BindGroup,
    render_bind_group: &'a crate::gpu::BindGroup,
}

/// One volume surface slice to draw into the pick pass. The owned group-2 bind
/// group holds the object id; the group-1 render bind group is borrowed from
/// prepared `VolumeSurfaceSliceGpuData`; the mesh is resolved against
/// `mesh_store` at draw time.
struct VolumeSurfaceSlicePickDraw<'a> {
    id_bind_group: crate::gpu::BindGroup,
    render_bind_group: &'a crate::gpu::BindGroup,
    mesh_id: crate::resources::mesh::mesh_store::MeshId,
}

/// Geometry source for one surface-pipeline pick draw. Surfaces reference a mesh
/// in `mesh_store`; tube-family items reference the owned per-frame buffers built
/// during prepare.
enum PickGeom<'a> {
    /// A mesh handle resolved against `mesh_store`.
    Mesh(crate::resources::mesh::mesh_store::MeshId),
    /// Direct buffer references for a tube-family connected mesh.
    Tube {
        vertex_buffer: &'a crate::gpu::Buffer,
        index_buffer: &'a crate::gpu::Buffer,
        index_count: u32,
        /// Per-triangle segment-endpoint payload for the POLY_NODE pick variant;
        /// `None` when the item built no node data.
        node_buffer: Option<&'a crate::gpu::Buffer>,
    },
}

/// Which types have pickable geometry this frame, plus the shared proxy mesh
/// handles resolved while ensuring their pick pipelines. Computed by
/// [`ViewportRenderer::ensure_pick_pipelines`], a `&mut self` step that must
/// finish (and drop its mutable borrow) before
/// [`ViewportRenderer::build_pick_draws`] borrows `self` immutably to build
/// the draw lists; splitting the two keeps `build_pick_draws`'s borrow shared,
/// so the point and rect pick passes can call further `&self` helpers
/// (instance/camera bind groups, draw recording) while the returned
/// `PickDrawSet` is still alive.
struct PickPipelineFlags {
    has_pickable_glyphs: bool,
    has_pickable_tensor: bool,
    has_pickable_sprites: bool,
    has_pickable_polylines: bool,
    has_pickable_volumes: bool,
    has_pickable_implicit: bool,
    has_pickable_mc: bool,
    has_pickable_point_clouds: bool,
    has_pickable_splats: bool,
    has_pickable_image_slices: bool,
    has_pickable_volume_surface_slices: bool,
    decal_cube: Option<crate::resources::mesh::mesh_store::MeshId>,
    scatter_cube: Option<crate::resources::mesh::mesh_store::MeshId>,
    scatter_sphere: Option<crate::resources::mesh::mesh_store::MeshId>,
}

/// A pre-built group-2 bind group for a per-pixel sub-object pick variant, one
/// slot per entry in `PickDrawSet::draws`. Owned before the pass begins so the
/// bind group outlives it. `None` slots draw with the default surface pipeline.
enum PickSublevelBind {
    /// Mesh vertex + index storage for the surface VERTEX variant.
    Vertex(crate::gpu::BindGroup),
    /// Mesh vertex + index storage for the surface EDGE variant (same layout as
    /// `Vertex`, different pipeline).
    Edge(crate::gpu::BindGroup),
    /// Per-triangle node payload for the curve POLY_NODE variant.
    Node(crate::gpu::BindGroup),
}

/// Whether a surface / volume-mesh draw should write its nearest corner (global
/// vertex index) instead of the hit face. True only when the query asks for
/// `VERTEX` as its finest surface level: no `FACE`, and no `CELL` that this
/// object could actually answer (a volume mesh with a `face_to_cell` map).
/// Mirrors the resolve-side priority FACE > CELL > VERTEX.
fn surface_writes_vertex(mask: PickMask, has_face_to_cell: bool, feature: bool) -> bool {
    feature
        && mask.intersects(PickMask::VERTEX)
        && !mask.intersects(PickMask::FACE)
        && !(has_face_to_cell && mask.intersects(PickMask::CELL))
}

/// Whether a surface / volume-mesh draw should write its nearest edge id
/// (`face * 3 + local_edge`) instead of the hit face. True only when `EDGE` is the
/// finest surface level requested, matching the resolve priority
/// FACE > CELL > VERTEX > EDGE: no `FACE`, no `VERTEX`, and no `CELL` this object
/// could answer.
fn surface_writes_edge(mask: PickMask, has_face_to_cell: bool, feature: bool) -> bool {
    feature
        && mask.intersects(PickMask::EDGE)
        && !mask.intersects(PickMask::FACE | PickMask::VERTEX)
        && !(has_face_to_cell && mask.intersects(PickMask::CELL))
}

/// Whether a curve draw should write its nearest segment endpoint (global node
/// index) instead of the hit triangle. True only when `POLY_NODE` is the finest
/// curve level requested, matching the resolve priority STRIP > SEGMENT > NODE.
fn curve_writes_node(mask: PickMask, feature: bool) -> bool {
    feature
        && mask.intersects(PickMask::POLY_NODE)
        && !mask.intersects(PickMask::STRIP | PickMask::SEGMENT)
}

/// Boundary-face-to-cell maps for volume meshes, keyed by `pick_id`. Built from
/// the frame when the pass is submitted and parked on the pending pick, so the
/// resolve step (which runs later, with no frame in hand) can turn a hit boundary
/// face into a cell without reading the per-frame `pick_*_items` cache. Plain
/// surfaces have no cells, so they are absent (a `None` lookup resolves as no
/// cell).
type SurfacePickMeta = std::collections::HashMap<u64, std::sync::Arc<[u32]>>;

/// Collect the boundary-face-to-cell map for every pickable volume mesh in the
/// frame. Called when the pass is submitted, so it reads the live frame rather
/// than a retained clone. The map is a small `u32` list, not geometry.
fn build_surface_pick_meta(frame: &FrameData) -> SurfacePickMeta {
    let mut meta: SurfacePickMeta = std::collections::HashMap::new();
    for vm in frame.scene.volume_meshes.iter() {
        let ri = vm.to_render_item();
        if ri.settings.hidden || ri.settings.pick_id == PickId::NONE {
            continue;
        }
        meta.insert(
            ri.settings.pick_id.0,
            std::sync::Arc::from(vm.face_to_cell.as_slice()),
        );
    }
    meta
}

/// Every pick pipeline's draw list plus the sub-object decode map, built once
/// per query and shared by the point and rect pick passes. Building this is
/// query-shape-agnostic: it does not know whether the caller wants a single
/// pixel or a whole rect region back, only which types answer `mask`.
struct PickDrawSet<'a> {
    draws: Vec<(PickGeom<'a>, PickInstance)>,
    glyph_draws: Vec<GlyphPickDraw<'a>>,
    sprite_draws: Vec<SpritePickDraw<'a>>,
    polyline_draws: Vec<PolylinePickDraw<'a>>,
    volume_draws: Vec<VolumePickDraw<'a>>,
    implicit_draws: Vec<ImplicitPickDraw<'a>>,
    mc_draws: Vec<McPickDraw<'a>>,
    point_cloud_draws: Vec<PointCloudPickDraw<'a>>,
    splat_draws: Vec<GaussianSplatPickDraw<'a>>,
    image_slice_draws: Vec<ImageSlicePickDraw<'a>>,
    volume_surface_slice_draws: Vec<VolumeSurfaceSlicePickDraw<'a>>,
    /// Whether any registered plugin has pickable geometry this frame. Plugin
    /// draws are not collected here; they are issued directly from
    /// `dispatch_plugin_pick` during `record_pick_pass_draws`.
    has_plugin_pick: bool,
    /// Per-object decode of the primitive channel, keyed by `pick_id`. Only
    /// consulted by the point pick path.
    kinds: std::collections::HashMap<u64, PickSubKind>,
    /// Cell / vertex refinement data for surface and volume-mesh hits, keyed by
    /// `pick_id`.
    surface_meta: SurfacePickMeta,
    /// The query mask. Decides which per-pixel sub-level pipeline variant each
    /// surface / curve draw uses (face vs nearest vertex, segment vs nearest node).
    mask: PickMask,
    /// Whether the device had `SHADER_PRIMITIVE_INDEX` when the draws were
    /// built. Only consulted by the point pick path.
    primitive_index_supported: bool,
}

impl PickDrawSet<'_> {
    /// `true` when nothing would be drawn: the pass has nothing to submit, so
    /// the caller can report a miss without touching the GPU.
    fn is_empty(&self) -> bool {
        self.draws.is_empty()
            && self.glyph_draws.is_empty()
            && self.sprite_draws.is_empty()
            && self.polyline_draws.is_empty()
            && self.volume_draws.is_empty()
            && self.implicit_draws.is_empty()
            && self.mc_draws.is_empty()
            && self.point_cloud_draws.is_empty()
            && self.splat_draws.is_empty()
            && self.image_slice_draws.is_empty()
            && self.volume_surface_slice_draws.is_empty()
            && !self.has_plugin_pick
    }
}

/// The three colour targets (object id, primitive id, depth) plus the real
/// depth-stencil buffer the pick pass renders into, all sized to the full
/// viewport regardless of how much of it the query's scissor covers. Shared
/// by the point and rect pick passes so the render-pass descriptor is written
/// once.
struct PickTargets {
    id_texture: crate::gpu::Texture,
    id_view: crate::gpu::TextureView,
    prim_texture: crate::gpu::Texture,
    prim_view: crate::gpu::TextureView,
    depth_colour_texture: crate::gpu::Texture,
    depth_colour_view: crate::gpu::TextureView,
    /// Real depth-stencil buffer; only `ds_view` is read (by the render pass),
    /// kept here so the texture stays alive for the view's lifetime.
    _ds_texture: crate::gpu::Texture,
    ds_view: crate::gpu::TextureView,
}

impl PickTargets {
    fn new(device: &crate::gpu::Device, vp_w: u32, vp_h: u32) -> Self {
        let extent = crate::gpu::Extent3d {
            width: vp_w,
            height: vp_h,
            depth_or_array_layers: 1,
        };

        let id_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("pick_id_texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::R32Uint,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let id_view = id_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

        // Primitive-id target. The pick pipelines write a sub-object index here
        // (0 for object-level types); it is attached so the pipeline's
        // location-1 output has a target, but only the point pick path reads
        // it back.
        let prim_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("pick_prim_texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::R32Uint,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let prim_view = prim_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

        let depth_colour_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("pick_depth_colour_texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::R32Float,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_colour_view =
            depth_colour_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

        let _ds_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("pick_ds_texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Depth24PlusStencil8,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let ds_view = _ds_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

        Self {
            id_texture,
            id_view,
            prim_texture,
            prim_view,
            depth_colour_texture,
            depth_colour_view,
            _ds_texture,
            ds_view,
        }
    }

    /// Begin the pick render pass against these targets: object id, primitive
    /// id, and depth colour attachments (all cleared to the "no hit" value),
    /// plus the real depth-stencil buffer (cleared to far). The caller sets
    /// its own scissor rect before drawing.
    fn begin_render_pass<'e>(
        &self,
        encoder: &'e mut crate::gpu::CommandEncoder,
    ) -> crate::gpu::RenderPass<'e> {
        encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(feature = "wgpu29")]
            multiview_mask: None,
            label: Some("pick_pass"),
            color_attachments: &[
                Some(crate::gpu::RenderPassColorAttachment {
                    view: &self.id_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::gpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: crate::gpu::StoreOp::Store,
                    },
                }),
                Some(crate::gpu::RenderPassColorAttachment {
                    view: &self.prim_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::gpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: crate::gpu::StoreOp::Store,
                    },
                }),
                Some(crate::gpu::RenderPassColorAttachment {
                    view: &self.depth_colour_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::gpu::Color {
                            r: 1.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: crate::gpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                view: &self.ds_view,
                depth_ops: Some(crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Clear(1.0),
                    store: crate::gpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }
}

impl ViewportRenderer {
    // -----------------------------------------------------------------------
    // GPU object-ID picking
    // -----------------------------------------------------------------------

    /// GPU object-ID pick: renders the scene to an offscreen `R32Uint` texture
    /// and reads back the single pixel under `cursor`.
    ///
    /// This is O(1) in mesh complexity : every object is rendered with a flat
    /// `u32` ID, and only one pixel is read back. For triangle-level queries
    /// (barycentric scalar probe, exact world position), use the CPU
    /// [`crate::interaction::query::picking::pick_scene_cpu`] path instead.
    ///
    /// The pipeline is lazily initialised on first call : zero overhead when
    /// this method is never invoked.
    ///
    /// # Arguments
    /// * `device` : wgpu device
    /// * `queue` : wgpu queue
    /// * `cursor` : cursor position in viewport-local pixels (top-left origin)
    /// * `frame` : current grouped frame data (camera, scene surfaces, viewport size)
    ///
    /// # Returns
    /// `Some(GpuPickHit)` if an object is under the cursor, `None` if empty space.
    pub fn pick_scene_gpu(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        cursor: glam::Vec2,
        frame: &FrameData,
    ) -> Option<GpuPickHit> {
        self.pick_scene_gpu_masked(device, queue, cursor, frame, PickMask::all())
    }

    /// GPU object-ID pick restricted to the item types `mask` selects.
    ///
    /// A type is drawn into the pick pass only when it answers a bit in `mask`,
    /// so a typed query (say an instance-only mask) is never occluded by an
    /// object of a type the caller did not ask for. Types with no pick pipeline
    /// yet do not draw, so they read back as no hit rather than a wrong hit.
    ///
    /// This blocks: it submits the id pass then drains the GPU queue
    /// (`device.poll(Wait)`) before reading the pixel. On a GPU-bound scene the
    /// wait costs the pick pass plus the frame backlog. For a non-blocking pick,
    /// use [`pick_object_begin`](Self::pick_object_begin) /
    /// [`pick_object_poll`](Self::pick_object_poll), which read the result on a
    /// later call instead of stalling here.
    pub(crate) fn pick_scene_gpu_masked(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        cursor: glam::Vec2,
        frame: &FrameData,
        mask: PickMask,
    ) -> Option<GpuPickHit> {
        // In Playback mode, throttle picking to every 4th frame to reduce overhead
        // during animation. Interactive, Paused, and Capture modes always pick.
        if self.runtime_mode == crate::renderer::stats::RuntimeMode::Playback
            && self.frame_counter % 4 != 0
        {
            return None;
        }

        let pending = match self.pick_scene_gpu_begin(device, queue, cursor, frame, mask) {
            PickBegin::Miss => return None,
            PickBegin::Pending(p) => p,
        };

        // Synchronous resolve: wait on this pick's submission (and thus its
        // staging maps), then read the pixel. The wait is targeted at the pick
        // submission index, so it does not drain frames queued after it.
        device
            .poll(crate::gpu::PollType::Wait {
                submission_index: Some(pending.submission.clone()),
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        pending.read_hit()
    }

    /// Blocking GPU point pick that resolves sub-object identity: submits the id
    /// pass, waits on this pick's own submission, reads back the pixel, and maps
    /// the primitive channel to a [`SubObjectRef`] per the mask. This is the
    /// object-plus-sub-object counterpart to
    /// [`pick_scene_gpu_masked`](Self::pick_scene_gpu_masked), which stays
    /// object-level for backward compatibility.
    pub(crate) fn pick_object_gpu_blocking(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        cursor: glam::Vec2,
        frame: &FrameData,
        mask: PickMask,
    ) -> Option<PickHit> {
        // Mirror the Playback throttle in pick_scene_gpu_masked.
        if self.runtime_mode == crate::renderer::stats::RuntimeMode::Playback
            && self.frame_counter % 4 != 0
        {
            return None;
        }

        // Screen-space overlay images have no world-space geometry to draw into
        // the id pass, and they always render on top of the 3D scene, so a hit
        // here takes priority over anything the render-based pass would find
        // (matching the CPU backend, where these carry toi = 0.0 : see
        // `point.rs` section 10). OBJECT-only, the same as the CPU backend.
        let viewport_size = glam::Vec2::from(frame.camera.viewport_size);
        if mask.intersects(PickMask::OBJECT) {
            if let Some(id) = screen_image_hit_at(&frame.scene.screen_images, viewport_size, cursor)
            {
                let view_proj_inv = frame.camera.render_camera.view_proj().inverse();
                let (ray_origin, ray_dir) = crate::interaction::query::picking::screen_to_ray(
                    cursor,
                    viewport_size,
                    view_proj_inv,
                );
                #[allow(deprecated)]
                return Some(PickHit {
                    id,
                    sub_object: None,
                    world_pos: ray_origin + ray_dir * 0.001,
                    normal: -ray_dir,
                    scalar_value: None,
                    sub_object_world_pos: None,
                });
            }
        }

        let pending = match self.pick_scene_gpu_begin(device, queue, cursor, frame, mask) {
            PickBegin::Miss => return None,
            PickBegin::Pending(p) => p,
        };
        device
            .poll(crate::gpu::PollType::Wait {
                submission_index: Some(pending.submission.clone()),
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        pending
            .read_hit()
            .map(|h| self.resolve_pending_hit(&pending, h))
    }

    /// Encode and submit the id pass for a point pick, map its staging buffers,
    /// and return the in-flight [`PendingPick`] without waiting. Shared by the
    /// blocking [`pick_scene_gpu_masked`](Self::pick_scene_gpu_masked) and the
    /// async [`pick_object_begin`](Self::pick_object_begin); the caller decides
    /// where the read-back wait lands. Returns [`PickBegin::Miss`] when the cursor
    /// is out of bounds or nothing pickable would draw, so there is no pass to
    /// submit.
    fn pick_scene_gpu_begin(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        cursor: glam::Vec2,
        frame: &FrameData,
        mask: PickMask,
    ) -> PickBegin {
        // Read scene items from the surface submission.
        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items.as_ref(),
        };

        // Ensure freshly uploaded geometry is resident: sync mesh uploads defer
        // their slab write to `process_uploads`, and picking can run without an
        // intervening frame. Flush before any sub-pass work submits.
        self.resources.geometry.flush(queue);

        let ppp = frame.camera.pixels_per_point;
        let vp_w = (frame.camera.viewport_size[0] * ppp).round() as u32;
        let vp_h = (frame.camera.viewport_size[1] * ppp).round() as u32;

        // --- bounds check (logical coordinates match the logical cursor) ---
        if cursor.x < 0.0
            || cursor.y < 0.0
            || cursor.x >= frame.camera.viewport_size[0]
            || cursor.y >= frame.camera.viewport_size[1]
            || vp_w == 0
            || vp_h == 0
        {
            return PickBegin::Miss;
        }

        // Physical pixel under the cursor. Used both to scissor the pass to a
        // single pixel and to read that pixel back, so the two always agree.
        // Clamp to the last valid texel: rounding can push the cursor's logical
        // edge one past the texture extent.
        let px = ((cursor.x * ppp).round() as u32).min(vp_w - 1);
        let py = ((cursor.y * ppp).round() as u32).min(vp_h - 1);

        let flags = self.ensure_pick_pipelines(device, frame, mask);
        let draw_set = self.build_pick_draws(device, queue, frame, mask, scene_items, &flags);
        if draw_set.is_empty() {
            return PickBegin::Miss;
        }

        // The buffer half of each pair only needs to live through
        // `create_bind_group` (which retains its own GPU-side reference), not
        // beyond, so neither is named further.
        let (_, pick_instance_bg) =
            self.build_pick_instance_bind_group(device, queue, &draw_set.draws);
        let (_, pick_camera_bg) = self.build_pick_camera_bind_group(device, queue, frame);
        let sublevel = self.build_pick_sublevel_binds(device, &draw_set);

        let targets = PickTargets::new(device, vp_w, vp_h);

        // A second flush catches geometry that pick-draw building itself uploaded
        // (scatter volumes, decal boxes), so every mesh the pass rasterises is
        // resident before the encoder records.
        self.resources.geometry.flush(queue);
        // --- render pass ---
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("pick_pass_encoder"),
        });
        {
            let mut pick_pass = targets.begin_render_pass(&mut encoder);

            // Only the pixel under the cursor is read back, so scissor the pass to
            // that one pixel. Rasterisation still visits every triangle, but the
            // fragment stage runs only inside the scissor, collapsing fragment and
            // depth-test work to a single pixel regardless of object count or
            // overdraw. The clear applies to the full attachment (it is not
            // scissored), so pixels outside the region stay at the "no hit" clear
            // value; nothing outside the region is ever read.
            pick_pass.set_scissor_rect(px, py, 1, 1);
            self.record_pick_pass_draws(
                &mut pick_pass,
                &pick_camera_bg,
                &pick_instance_bg,
                &draw_set,
                &sublevel,
                frame,
            );
        }

        // --- copy 1x1 pixels to staging buffers ---
        // R32Uint: 4 bytes per pixel, min bytes_per_row = 256 (wgpu alignment)
        let bytes_per_row_aligned = 256u32; // wgpu requires multiples of 256

        let id_staging = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("pick_id_staging"),
            size: bytes_per_row_aligned as u64,
            usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let prim_staging = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("pick_prim_staging"),
            size: bytes_per_row_aligned as u64,
            usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let depth_staging = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("pick_depth_staging"),
            size: bytes_per_row_aligned as u64,
            usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // `px` / `py`: physical pixel under the cursor, computed above and shared
        // with the scissor so the pass and the read-back target the same texel.
        encoder.copy_texture_to_buffer(
            crate::gpu::TexelCopyTextureInfo {
                texture: &targets.id_texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d { x: px, y: py, z: 0 },
                aspect: crate::gpu::TextureAspect::All,
            },
            crate::gpu::TexelCopyBufferInfo {
                buffer: &id_staging,
                layout: crate::gpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row_aligned),
                    rows_per_image: Some(1),
                },
            },
            crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        encoder.copy_texture_to_buffer(
            crate::gpu::TexelCopyTextureInfo {
                texture: &targets.prim_texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d { x: px, y: py, z: 0 },
                aspect: crate::gpu::TextureAspect::All,
            },
            crate::gpu::TexelCopyBufferInfo {
                buffer: &prim_staging,
                layout: crate::gpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row_aligned),
                    rows_per_image: Some(1),
                },
            },
            crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        encoder.copy_texture_to_buffer(
            crate::gpu::TexelCopyTextureInfo {
                texture: &targets.depth_colour_texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d { x: px, y: py, z: 0 },
                aspect: crate::gpu::TextureAspect::All,
            },
            crate::gpu::TexelCopyBufferInfo {
                buffer: &depth_staging,
                layout: crate::gpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row_aligned),
                    rows_per_image: Some(1),
                },
            },
            crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let submission = queue.submit(std::iter::once(encoder.finish()));

        // Start the maps now but do not wait. `map_async` callbacks fire once the
        // device has processed this submission; the caller drives that in a
        // blocking wait (`pick_scene_gpu_masked`) or a targeted one
        // (`pick_object_poll`) and reads the pixel through `PendingPick::read_hit`
        // once all three maps have landed. The callbacks flip `ready` to 3.
        let ready = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for staging in [&id_staging, &prim_staging, &depth_staging] {
            let ready = ready.clone();
            staging
                .slice(..)
                .map_async(crate::gpu::MapMode::Read, move |r| {
                    if r.is_ok() {
                        ready.fetch_add(1, std::sync::atomic::Ordering::Release);
                    }
                });
        }

        PickBegin::Pending(PendingPick {
            id_staging,
            prim_staging,
            depth_staging,
            ready,
            submission,
            cursor,
            viewport_size: glam::Vec2::from(frame.camera.viewport_size),
            view_proj: frame.camera.render_camera.view_proj(),
            mask,
            kinds: draw_set.kinds,
            surface_meta: draw_set.surface_meta,
            primitive_index_supported: draw_set.primitive_index_supported,
        })
    }

    /// Lazily build every pick pipeline the mask-selected types need, and
    /// resolve the shared decal / scatter-volume proxy meshes. `&mut self`:
    /// this is the only mutating step in the pick pipeline; everything after
    /// it (`build_pick_draws` and the render pass) only reads `self`.
    fn ensure_pick_pipelines(
        &mut self,
        device: &crate::gpu::Device,
        frame: &FrameData,
        mask: PickMask,
    ) -> PickPipelineFlags {
        // --- lazy pipeline init ---
        self.resources.ensure_pick_pipeline(device);
        // Surface VERTEX and curve POLY_NODE write their final sub-id per pixel
        // from a dedicated pipeline variant. Build them when the mask asks for
        // that level; each is a no-op without SHADER_PRIMITIVE_INDEX.
        if mask.intersects(PickMask::VERTEX) {
            self.resources.ensure_pick_vertex_pipeline(device);
        }
        if mask.intersects(PickMask::EDGE) {
            self.resources.ensure_pick_edge_pipeline(device);
        }
        if mask.intersects(PickMask::POLY_NODE) {
            self.resources.ensure_pick_node_pipeline(device);
        }
        let glyph_wanted = PickItemType::Glyph.satisfies(mask);
        let has_pickable_glyphs = glyph_wanted
            && self
                .glyph_gpu_data
                .iter()
                .any(|g| g.pick_id != PickId::NONE && g.instance_count > 0);
        let has_pickable_tensor = glyph_wanted
            && self
                .tensor_glyph_gpu_data
                .iter()
                .any(|g| g.pick_id != PickId::NONE && g.instance_count > 0);
        if has_pickable_glyphs {
            self.resources.ensure_glyph_pick_pipeline(device);
        }
        if has_pickable_tensor {
            self.resources.ensure_tensor_glyph_pick_pipeline(device);
        }
        let has_pickable_sprites = PickItemType::Sprite.satisfies(mask)
            && self
                .sprite_gpu_data
                .iter()
                .any(|g| g.pick_id != PickId::NONE && g.sprite_count > 0);
        if has_pickable_sprites {
            self.resources.ensure_sprite_pick_pipeline(device);
        }
        let has_pickable_polylines = PickItemType::Polyline.satisfies(mask)
            && self
                .polyline_gpu_data
                .iter()
                .any(|g| g.pick_id != PickId::NONE && g.segment_count > 0);
        if has_pickable_polylines {
            self.resources.ensure_polyline_pick_pipeline(device);
        }

        // Voxel volumes raymarch their bounding cube to the first in-threshold
        // voxel. Answers OBJECT and the VOXEL sub-object level; wireframe volumes
        // render an OBB polyline instead, so they are picked as polylines, not here.
        let has_pickable_volumes = mask.intersects(PickMask::OBJECT | PickMask::VOXEL)
            && self
                .volume_gpu_data
                .iter()
                .any(|v| !v.wireframe && v.pick_id != PickId::NONE);
        if has_pickable_volumes {
            self.resources.ensure_volume_pick_pipeline(device);
        }

        // GPU implicit SDF surfaces raymarch the isosurface on a full-screen
        // quad. Object-level.
        let has_pickable_implicit = mask.intersects(PickMask::OBJECT)
            && self
                .implicit_gpu_data
                .iter()
                .any(|g| g.pick_id != PickId::NONE);
        if has_pickable_implicit {
            self.resources.ensure_implicit_pick_pipeline(device);
        }

        // GPU marching-cubes surfaces rasterise their generated vertex buffer.
        // Object-level.
        let has_pickable_mc = mask.intersects(PickMask::OBJECT)
            && self.mc_gpu_data.iter().any(|m| m.pick_id != PickId::NONE);
        if has_pickable_mc {
            self.resources.ensure_mc_pick_pipeline(device);
        }

        // Point clouds: each renders as a screen-space quad per point (approach
        // B), so the pick reuses that expansion. CLOUD_POINT sub-object comes
        // from the forwarded instance index.
        let has_pickable_point_clouds = mask.intersects(PickMask::OBJECT | PickMask::CLOUD_POINT)
            && self
                .point_cloud_gpu_data
                .iter()
                .any(|g| g.pick_id != PickId::NONE && g.point_count > 0);
        if has_pickable_point_clouds {
            self.resources.ensure_point_cloud_pick_pipeline(device);
        }

        // Gaussian splats: each renders as an instanced billboard per splat.
        // Occlusion is resolved by the pick pass's own depth test, so the
        // existing per-viewport sorted-index buffer (built for back-to-front
        // render blending) can be reused without re-sorting.
        let has_pickable_splats = mask.intersects(PickMask::OBJECT | PickMask::SPLAT)
            && self
                .gaussian_splat_draw_data
                .iter()
                .any(|dd| !dd.wireframe && dd.pick_id != PickId::NONE && dd.count > 0);
        if has_pickable_splats {
            self.resources.ensure_gaussian_splat_pick_pipeline(device);
        }

        // Image slices and volume surface slices: textured world-space quads.
        // Object-level only.
        let has_pickable_image_slices = mask.intersects(PickMask::OBJECT)
            && self
                .image_slice_gpu_data
                .iter()
                .any(|g| g.pick_id != PickId::NONE);
        if has_pickable_image_slices {
            self.resources.ensure_image_slice_pick_pipeline(device);
        }
        let has_pickable_volume_surface_slices = mask.intersects(PickMask::OBJECT)
            && self
                .volume_surface_slice_gpu_data
                .iter()
                .any(|g| g.pick_id != PickId::NONE);
        if has_pickable_volume_surface_slices {
            self.resources
                .ensure_volume_surface_slice_pick_pipeline(device);
        }

        // Decals rasterise their projection box (the unit cube mapped by
        // `transform`) as an object-level proxy. Ensure the shared cube mesh
        // exists when a decal is pickable and the query asks for OBJECT. Computed
        // as a plain `MeshId` before `draws` borrows `self`, so pushing the decal
        // draws later needs no further `self` mutation.
        let decal_cube = if mask.intersects(PickMask::OBJECT)
            && frame
                .scene
                .decals
                .iter()
                .any(|d| !d.settings.hidden && d.settings.pick_id != PickId::NONE)
        {
            self.ensure_decal_pick_cube(device)
        } else {
            None
        };

        // Scatter volumes pick against their actual shape: a box (the shared cube)
        // or a sphere (the shared icosphere), world-space and unrotated, matching
        // the CPU analytic `ray_intersect`. Ensure whichever proxies are needed.
        let (scatter_cube, scatter_sphere) = if mask.intersects(PickMask::OBJECT) {
            let mut want_box = false;
            let mut want_sphere = false;
            for it in frame
                .scene
                .scatter_volumes
                .iter()
                .filter(|s| !s.settings.hidden && s.settings.pick_id != PickId::NONE)
            {
                match it.volume.shape {
                    crate::scene::scatter_volume::ScatterShape::Box(_) => want_box = true,
                    crate::scene::scatter_volume::ScatterShape::Sphere { .. } => want_sphere = true,
                }
            }
            let cube = if want_box {
                self.ensure_decal_pick_cube(device)
            } else {
                None
            };
            let sphere = if want_sphere {
                self.ensure_scatter_pick_sphere(device)
            } else {
                None
            };
            (cube, sphere)
        } else {
            (None, None)
        };

        PickPipelineFlags {
            has_pickable_glyphs,
            has_pickable_tensor,
            has_pickable_sprites,
            has_pickable_polylines,
            has_pickable_volumes,
            has_pickable_implicit,
            has_pickable_mc,
            has_pickable_point_clouds,
            has_pickable_splats,
            has_pickable_image_slices,
            has_pickable_volume_surface_slices,
            decal_cube,
            scatter_cube,
            scatter_sphere,
        }
    }

    /// Build every pick pipeline's draw list for this query, plus the
    /// sub-object decode map. Shared by the point pick pass
    /// ([`pick_scene_gpu_begin`](Self::pick_scene_gpu_begin)) and the rect pick
    /// pass ([`pick_rect_gpu`](Self::pick_rect_gpu)): both draw the same
    /// mask-selected geometry into the same three-channel target, differing
    /// only in the scissor rect and how much of the read-back they need.
    /// Takes `&self`: pipelines are already built by
    /// [`ensure_pick_pipelines`](Self::ensure_pick_pipelines), so this only
    /// reads prepared per-frame GPU data and issues the small per-draw pick-id
    /// buffer uploads.
    fn build_pick_draws<'a>(
        &'a self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        mask: PickMask,
        scene_items: &'a [SceneRenderItem],
        flags: &PickPipelineFlags,
    ) -> PickDrawSet<'a> {
        let has_pickable_glyphs = flags.has_pickable_glyphs;
        let has_pickable_tensor = flags.has_pickable_tensor;
        let has_pickable_sprites = flags.has_pickable_sprites;
        let has_pickable_polylines = flags.has_pickable_polylines;
        let has_pickable_volumes = flags.has_pickable_volumes;
        let has_pickable_implicit = flags.has_pickable_implicit;
        let has_pickable_mc = flags.has_pickable_mc;
        let has_pickable_point_clouds = flags.has_pickable_point_clouds;
        let has_pickable_splats = flags.has_pickable_splats;
        let has_pickable_image_slices = flags.has_pickable_image_slices;
        let has_pickable_volume_surface_slices = flags.has_pickable_volume_surface_slices;
        let decal_cube = flags.decal_cube;
        let scatter_cube = flags.scatter_cube;
        let scatter_sphere = flags.scatter_sphere;

        // --- build PickInstance data ---
        // Every mesh-backed pickable item draws through the surface pipeline:
        // scene surfaces, plus volume-mesh boundaries (both opaque and
        // transparent, which render their boundary as a surface mesh). Items that
        // are hidden or have pick_id 0 are skipped; clear value 0 means "no hit".
        let pickable =
            |item: &SceneRenderItem| !item.settings.hidden && item.settings.pick_id != PickId::NONE;
        let to_instance = |item: &SceneRenderItem| {
            let m = item.model;
            PickInstance {
                model_c0: m[0],
                model_c1: m[1],
                model_c2: m[2],
                model_c3: m[3],
                object_id: item.settings.pick_id.0 as u32,
                _pad: [0; 3],
            }
        };

        let instance_from = |model: [[f32; 4]; 4], pick_id: PickId| PickInstance {
            model_c0: model[0],
            model_c1: model[1],
            model_c2: model[2],
            model_c3: model[3],
            object_id: pick_id.0 as u32,
            _pad: [0; 3],
        };

        let mut draws: Vec<(PickGeom, PickInstance)> = Vec::new();

        // Surfaces and volume-mesh boundaries draw through the Surface pipeline;
        // skip building their instance data when the mask asks for none of the
        // levels that type answers.
        if PickItemType::Surface.satisfies(mask) {
            for item in scene_items.iter().filter(|i| pickable(i)) {
                draws.push((PickGeom::Mesh(item.mesh_id), to_instance(item)));
            }
            for ri in frame
                .scene
                .volume_meshes
                .iter()
                .map(|vm| vm.to_render_item())
                .filter(pickable)
            {
                draws.push((PickGeom::Mesh(ri.mesh_id), to_instance(&ri)));
            }
        }

        // Streamtubes, tubes, and ribbons build owned connected meshes into these
        // vecs during prepare(); each entry carries its source item's pick_id and
        // model. The streamtube shader applies the model to the buffer positions,
        // so the pick pass uses the same matrix and its silhouette matches.
        if PickItemType::Curve.satisfies(mask) {
            for family in [
                self.streamtube_gpu_data.as_slice(),
                self.tube_gpu_data.as_slice(),
                self.ribbon_gpu_data.as_slice(),
            ] {
                for gpu in family
                    .iter()
                    .filter(|g| g.pick_id != PickId::NONE && g.index_count > 0)
                {
                    draws.push((
                        PickGeom::Tube {
                            vertex_buffer: &gpu.vertex_buffer,
                            index_buffer: &gpu.index_buffer,
                            index_count: gpu.index_count,
                            node_buffer: gpu.node_pick_buffer.as_ref(),
                        },
                        instance_from(gpu.model, gpu.pick_id),
                    ));
                }
            }
        }

        // Decals: rasterise the unit-cube projection box under each decal's
        // transform, tagged with its pick_id. Object-level. The box silhouette
        // can extend past the projected footprint into empty space, so a click
        // near a decal but off its receiver can still select it, matching the CPU
        // decal pick. Degenerate (non-invertible) transforms are skipped.
        if let Some(cube_id) = decal_cube {
            for d in frame
                .scene
                .decals
                .iter()
                .filter(|d| !d.settings.hidden && d.settings.pick_id != PickId::NONE)
            {
                if glam::Mat4::from_cols_array_2d(&d.transform)
                    .determinant()
                    .abs()
                    < 1e-12
                {
                    continue;
                }
                draws.push((
                    PickGeom::Mesh(cube_id),
                    instance_from(d.transform, d.settings.pick_id),
                ));
            }
        }

        // Scatter volumes: box -> cube proxy (exact), sphere -> icosphere proxy.
        // The shapes are world-space and unrotated, so a translate + scale places
        // the proxy on the shape. Matches the CPU analytic scatter pick.
        for it in frame
            .scene
            .scatter_volumes
            .iter()
            .filter(|s| !s.settings.hidden && s.settings.pick_id != PickId::NONE)
        {
            match it.volume.shape {
                crate::scene::scatter_volume::ScatterShape::Box(b) => {
                    let Some(cube_id) = scatter_cube else {
                        continue;
                    };
                    let min = b.min;
                    let max = b.max;
                    let extent = max - min;
                    if extent.min_element() <= 0.0 {
                        continue;
                    }
                    let model = glam::Mat4::from_translation((min + max) * 0.5)
                        * glam::Mat4::from_scale(extent);
                    draws.push((
                        PickGeom::Mesh(cube_id),
                        instance_from(model.to_cols_array_2d(), it.settings.pick_id),
                    ));
                }
                crate::scene::scatter_volume::ScatterShape::Sphere { center, radius } => {
                    let Some(sphere_id) = scatter_sphere else {
                        continue;
                    };
                    if radius <= 0.0 {
                        continue;
                    }
                    let model = glam::Mat4::from_translation(glam::Vec3::from(center))
                        * glam::Mat4::from_scale(glam::Vec3::splat(radius));
                    draws.push((
                        PickGeom::Mesh(sphere_id),
                        instance_from(model.to_cols_array_2d(), it.settings.pick_id),
                    ));
                }
            }
        }

        // Glyph and tensor-glyph sets draw with their own pick pipelines. Each
        // set builds a group-1 bind group (the set uniform + a per-set object-id
        // uniform) here so it outlives the render pass. The buffers behind the
        // bind group stay alive through it, so the temporary id buffer can drop.
        let mut glyph_draws: Vec<GlyphPickDraw> = Vec::new();
        if has_pickable_glyphs || has_pickable_tensor {
            let id_bgl = self
                .resources
                .pick
                .glyph_pick_id_bgl
                .as_ref()
                .expect("glyph pick id bgl");
            let make_id_bg = |pick_id: PickId, uniform_buf: &crate::gpu::Buffer| {
                let id_data = [pick_id.0 as u32, 0u32, 0u32, 0u32];
                let id_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("glyph_pick_id_buf"),
                    size: std::mem::size_of_val(&id_data) as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
                device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("glyph_pick_id_bg"),
                    layout: id_bgl,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 3,
                            resource: id_buf.as_entire_binding(),
                        },
                    ],
                })
            };
            if has_pickable_glyphs {
                let pipeline = self
                    .resources
                    .pick
                    .glyph_pipeline
                    .as_ref()
                    .expect("glyph pick pipeline");
                for gpu in self
                    .glyph_gpu_data
                    .iter()
                    .filter(|g| g.pick_id != PickId::NONE && g.instance_count > 0)
                {
                    glyph_draws.push(GlyphPickDraw {
                        pipeline,
                        id_bind_group: make_id_bg(gpu.pick_id, &gpu._uniform_buf),
                        instance_bind_group: &gpu.instance_bind_group,
                        vertex_buffer: gpu.mesh_vertex_buffer,
                        index_buffer: gpu.mesh_index_buffer,
                        index_count: gpu.mesh_index_count,
                        instance_count: gpu.instance_count,
                    });
                }
            }
            if has_pickable_tensor {
                let pipeline = self
                    .resources
                    .pick
                    .tensor_glyph_pipeline
                    .as_ref()
                    .expect("tensor glyph pick pipeline");
                for gpu in self
                    .tensor_glyph_gpu_data
                    .iter()
                    .filter(|g| g.pick_id != PickId::NONE && g.instance_count > 0)
                {
                    glyph_draws.push(GlyphPickDraw {
                        pipeline,
                        id_bind_group: make_id_bg(gpu.pick_id, &gpu._uniform_buf),
                        instance_bind_group: &gpu.instance_bind_group,
                        vertex_buffer: gpu.mesh_vertex_buffer,
                        index_buffer: gpu.mesh_index_buffer,
                        index_count: gpu.mesh_index_count,
                        instance_count: gpu.instance_count,
                    });
                }
            }
        }

        // Sprite sets draw with their own pipeline. Each set gets a group-2 bind
        // group holding its object id; the pipeline and camera bind group are
        // shared, so only the id, sprite bind group, and position buffer vary.
        let mut sprite_draws: Vec<SpritePickDraw> = Vec::new();
        if has_pickable_sprites {
            let id_bgl = self
                .resources
                .sprite
                .pick_id_bgl
                .as_ref()
                .expect("sprite pick id bgl");
            for gpu in self
                .sprite_gpu_data
                .iter()
                .filter(|g| g.pick_id != PickId::NONE && g.sprite_count > 0)
            {
                let id_data = [gpu.pick_id.0 as u32, 0u32, 0u32, 0u32];
                let id_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("sprite_pick_id_buf"),
                    size: std::mem::size_of_val(&id_data) as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
                let id_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("sprite_pick_id_bg"),
                    layout: id_bgl,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: id_buf.as_entire_binding(),
                    }],
                });
                sprite_draws.push(SpritePickDraw {
                    id_bind_group,
                    sprite_bind_group: &gpu.bind_group,
                    vertex_buffer: &gpu.vertex_buffer,
                    sprite_count: gpu.sprite_count,
                });
            }
        }

        // Polylines draw with their own pipeline: group 0 is the shared minimal
        // pick camera, group 1 is the polyline render bind group (uniform + LUT),
        // group 2 is the per-draw object id.
        let mut polyline_draws: Vec<PolylinePickDraw> = Vec::new();
        if has_pickable_polylines {
            let id_bgl = self
                .resources
                .pick
                .polyline_pick_id_bgl
                .as_ref()
                .expect("polyline pick id bgl");
            for gpu in self
                .polyline_gpu_data
                .iter()
                .filter(|g| g.pick_id != PickId::NONE && g.segment_count > 0)
            {
                let id_data = [gpu.pick_id.0 as u32, 0u32, 0u32, 0u32];
                let id_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("polyline_pick_id_buf"),
                    size: std::mem::size_of_val(&id_data) as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
                let id_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("polyline_pick_id_bg"),
                    layout: id_bgl,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: id_buf.as_entire_binding(),
                    }],
                });
                polyline_draws.push(PolylinePickDraw {
                    id_bind_group,
                    render_bind_group: &gpu.bind_group,
                    vertex_buffer: &gpu.vertex_buffer,
                    segment_count: gpu.segment_count,
                });
            }
        }

        // Voxel volumes: one draw of the bounding cube per pickable, non-wireframe
        // volume, with a group-2 object-id uniform. The group-1 render bind group
        // (volume uniform + 3D texture) is reused from prepared `VolumeGpuData`.
        let mut volume_draws: Vec<VolumePickDraw> = Vec::new();
        if has_pickable_volumes && self.resources.pick.volume_pipeline.is_some() {
            let id_bgl = self
                .resources
                .pick
                .volume_pick_id_bgl
                .as_ref()
                .expect("volume pick id bgl built with the pipeline");
            for gpu in self
                .volume_gpu_data
                .iter()
                .filter(|v| !v.wireframe && v.pick_id != PickId::NONE)
            {
                let id_data = [gpu.pick_id.0 as u32, 0u32, 0u32, 0u32];
                let id_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("volume_pick_id_buf"),
                    size: std::mem::size_of_val(&id_data) as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
                let id_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("volume_pick_id_bg"),
                    layout: id_bgl,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: id_buf.as_entire_binding(),
                    }],
                });
                volume_draws.push(VolumePickDraw {
                    id_bind_group,
                    render_bind_group: &gpu.bind_group,
                    vertex_buffer: &gpu.vertex_buffer,
                    index_buffer: &gpu.index_buffer,
                });
            }
        }

        // GPU implicit SDF surfaces: one full-screen raymarch draw per pickable
        // item, with a group-2 object-id uniform. Group 1 is the reused implicit
        // render bind group (the SDF uniform).
        let mut implicit_draws: Vec<ImplicitPickDraw> = Vec::new();
        if has_pickable_implicit && self.resources.pick.implicit_pipeline.is_some() {
            let id_bgl = self
                .resources
                .pick
                .implicit_pick_id_bgl
                .as_ref()
                .expect("implicit pick id bgl built with the pipeline");
            for gpu in self
                .implicit_gpu_data
                .iter()
                .filter(|g| g.pick_id != PickId::NONE)
            {
                let id_data = [gpu.pick_id.0 as u32, 0u32, 0u32, 0u32];
                let id_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("implicit_pick_id_buf"),
                    size: std::mem::size_of_val(&id_data) as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
                let id_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("implicit_pick_id_bg"),
                    layout: id_bgl,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: id_buf.as_entire_binding(),
                    }],
                });
                implicit_draws.push(ImplicitPickDraw {
                    id_bind_group,
                    render_bind_group: &gpu.bind_group,
                });
            }
        }

        // GPU marching-cubes surfaces: one indirect draw per slab of each pickable
        // item, with a group-1 object-id uniform. The generated MC vertex buffer and
        // surface indirect args are reused from the render path.
        let mut mc_draws: Vec<McPickDraw> = Vec::new();
        if has_pickable_mc && self.resources.pick.mc_pipeline.is_some() {
            let id_bgl = self
                .resources
                .pick
                .mc_pick_id_bgl
                .as_ref()
                .expect("mc pick id bgl built with the pipeline");
            for mc in self
                .mc_gpu_data
                .iter()
                .filter(|m| m.pick_id != PickId::NONE)
            {
                let Some(vol) = self.resources.mc.volumes.get(mc.volume_idx) else {
                    continue;
                };
                if vol.slabs.is_empty() {
                    continue;
                }
                let id_data = [mc.pick_id.0 as u32, 0u32, 0u32, 0u32];
                let id_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("mc_pick_id_buf"),
                    size: std::mem::size_of_val(&id_data) as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
                let id_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("mc_pick_id_bg"),
                    layout: id_bgl,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: id_buf.as_entire_binding(),
                    }],
                });
                let slabs = vol
                    .slabs
                    .iter()
                    .map(|s| (&s.vertex_buf, &s.indirect_buf))
                    .collect();
                mc_draws.push(McPickDraw {
                    id_bind_group,
                    slabs,
                });
            }
        }

        // Point clouds: each item draws its screen-space quad expansion with a
        // group-2 object-id uniform. The group-1 render bind group (uniform +
        // LUT + radius buffer) is reused unchanged.
        let mut point_cloud_draws: Vec<PointCloudPickDraw> = Vec::new();
        if has_pickable_point_clouds && self.resources.pick.point_cloud_pipeline.is_some() {
            let id_bgl = self
                .resources
                .pick
                .point_cloud_pick_id_bgl
                .as_ref()
                .expect("point cloud pick id bgl built with the pipeline");
            for gpu in self
                .point_cloud_gpu_data
                .iter()
                .filter(|g| g.pick_id != PickId::NONE && g.point_count > 0)
            {
                let id_data = [gpu.pick_id.0 as u32, 0u32, 0u32, 0u32];
                let id_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("point_cloud_pick_id_buf"),
                    size: std::mem::size_of_val(&id_data) as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
                let id_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("point_cloud_pick_id_bg"),
                    layout: id_bgl,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: id_buf.as_entire_binding(),
                    }],
                });
                point_cloud_draws.push(PointCloudPickDraw {
                    id_bind_group,
                    render_bind_group: &gpu.bind_group,
                    vertex_buffer: &gpu.vertex_buffer,
                    point_count: gpu.point_count,
                });
            }
        }

        // Gaussian splats: each item draws its covariance-projected billboard
        // expansion with a group-2 object-id uniform. The group-1 render bind
        // group is the same per-viewport sorted-index bind group the render
        // path draws with; occlusion is resolved by the pick pass's own depth
        // test, so the sort order does not matter here.
        let mut splat_draws: Vec<GaussianSplatPickDraw> = Vec::new();
        if has_pickable_splats && self.resources.pick.gaussian_splat_pipeline.is_some() {
            let id_bgl = self
                .resources
                .pick
                .gaussian_splat_pick_id_bgl
                .as_ref()
                .expect("gaussian splat pick id bgl built with the pipeline");
            for dd in self
                .gaussian_splat_draw_data
                .iter()
                .filter(|dd| !dd.wireframe && dd.pick_id != PickId::NONE && dd.count > 0)
            {
                let Some(set) = self
                    .resources
                    .content
                    .gaussian_splat_store
                    .get_by_index(dd.store_index)
                else {
                    continue;
                };
                let Some(Some(vp_sort)) = set.viewport_sort.get(dd.viewport_index) else {
                    continue;
                };
                let id_data = [dd.pick_id.0 as u32, 0u32, 0u32, 0u32];
                let id_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("gaussian_splat_pick_id_buf"),
                    size: std::mem::size_of_val(&id_data) as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
                let id_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("gaussian_splat_pick_id_bg"),
                    layout: id_bgl,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: id_buf.as_entire_binding(),
                    }],
                });
                splat_draws.push(GaussianSplatPickDraw {
                    id_bind_group,
                    render_bind_group: &vp_sort.render_bg,
                    count: dd.count,
                });
            }
        }

        // Image slices: each item draws its quad-from-vertex-index expansion
        // with a group-2 object-id uniform. Object-level only.
        let mut image_slice_draws: Vec<ImageSlicePickDraw> = Vec::new();
        if has_pickable_image_slices && self.resources.pick.image_slice_pipeline.is_some() {
            let id_bgl = self
                .resources
                .pick
                .image_slice_pick_id_bgl
                .as_ref()
                .expect("image slice pick id bgl built with the pipeline");
            for gpu in self
                .image_slice_gpu_data
                .iter()
                .filter(|g| g.pick_id != PickId::NONE)
            {
                let id_data = [gpu.pick_id.0 as u32, 0u32, 0u32, 0u32];
                let id_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("image_slice_pick_id_buf"),
                    size: std::mem::size_of_val(&id_data) as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
                let id_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("image_slice_pick_id_bg"),
                    layout: id_bgl,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: id_buf.as_entire_binding(),
                    }],
                });
                image_slice_draws.push(ImageSlicePickDraw {
                    id_bind_group,
                    render_bind_group: &gpu.bind_group,
                });
            }
        }

        // Volume surface slices: each item draws its mesh with a group-2
        // object-id uniform. Object-level only.
        let mut volume_surface_slice_draws: Vec<VolumeSurfaceSlicePickDraw> = Vec::new();
        if has_pickable_volume_surface_slices
            && self.resources.pick.volume_surface_slice_pipeline.is_some()
        {
            let id_bgl = self
                .resources
                .pick
                .volume_surface_slice_pick_id_bgl
                .as_ref()
                .expect("volume surface slice pick id bgl built with the pipeline");
            for gpu in self
                .volume_surface_slice_gpu_data
                .iter()
                .filter(|g| g.pick_id != PickId::NONE)
            {
                let id_data = [gpu.pick_id.0 as u32, 0u32, 0u32, 0u32];
                let id_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("volume_surface_slice_pick_id_buf"),
                    size: std::mem::size_of_val(&id_data) as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
                let id_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("volume_surface_slice_pick_id_bg"),
                    layout: id_bgl,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: id_buf.as_entire_binding(),
                    }],
                });
                volume_surface_slice_draws.push(VolumeSurfaceSlicePickDraw {
                    id_bind_group,
                    render_bind_group: &gpu.bind_group,
                    mesh_id: gpu.mesh_id,
                });
            }
        }

        // Registered plugins draw their own pick-ids into the pass. They answer
        // the same level set as built-in surfaces (object plus the mesh
        // sub-object levels, refined through `ItemTypePlugin::resolve_sub_object`
        // on read-back), so run the pass for them whenever the mask asks for any
        // of those and a plugin has a non-empty collection this frame. Drawing
        // them under sub-object-only masks also keeps their geometry in the
        // depth test, so items behind a plugin item cannot be picked through it.
        // Their draws are not in `draws`/`glyph_draws`/etc.; they are issued via
        // `dispatch_plugin_pick`.
        let has_plugin_pick = mask.intersects(
            PickMask::OBJECT | PickMask::FACE | PickMask::VERTEX | PickMask::EDGE | PickMask::CELL,
        ) && self.any_plugin_items_submitted(frame);

        let kinds = self.build_pick_sub_kinds(frame, scene_items);
        let surface_meta = build_surface_pick_meta(frame);
        let primitive_index_supported = device
            .features()
            .contains(crate::gpu::PRIMITIVE_INDEX_FEATURE);

        PickDrawSet {
            draws,
            glyph_draws,
            sprite_draws,
            polyline_draws,
            volume_draws,
            implicit_draws,
            mc_draws,
            point_cloud_draws,
            splat_draws,
            image_slice_draws,
            volume_surface_slice_draws,
            has_plugin_pick,
            kinds,
            surface_meta,
            mask,
            primitive_index_supported,
        }
    }

    /// Build the group-1 `PickInstance` storage buffer + bind group for the
    /// surface-pipeline `draws` list. Shared by the point and rect pick passes.
    fn build_pick_instance_bind_group(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        draws: &[(PickGeom, PickInstance)],
    ) -> (crate::gpu::Buffer, crate::gpu::BindGroup) {
        let pick_instances: Vec<PickInstance> = draws.iter().map(|(_, inst)| *inst).collect();
        let pick_instance_bytes = bytemuck::cast_slice(&pick_instances);
        let pick_instance_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("pick_instance_buf"),
            size: pick_instance_bytes.len().max(80) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&pick_instance_buf, 0, pick_instance_bytes);

        let pick_instance_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("pick_instance_bg"),
            layout: self
                .resources
                .pick
                .bind_group_layout_1
                .as_ref()
                .expect("ensure_pick_pipeline must be called first"),
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: pick_instance_buf.as_entire_binding(),
            }],
        });
        (pick_instance_buf, pick_instance_bg)
    }

    /// Build the group-0 minimal pick camera uniform buffer + bind group
    /// (camera + clip volume). Shared by the point and rect pick passes.
    fn build_pick_camera_bind_group(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) -> (crate::gpu::Buffer, crate::gpu::BindGroup) {
        let camera_uniform = frame.camera.render_camera.camera_uniform();
        let camera_bytes = bytemuck::bytes_of(&camera_uniform);
        let pick_camera_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("pick_camera_buf"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&pick_camera_buf, 0, camera_bytes);

        let pick_camera_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("pick_camera_bg"),
            layout: self
                .resources
                .pick
                .camera_bgl
                .as_ref()
                .expect("ensure_pick_pipeline must be called first"),
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: pick_camera_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: self.resources.binds.clip_volume_buf.as_entire_binding(),
                },
            ],
        });
        (pick_camera_buf, pick_camera_bg)
    }

    /// Build the per-draw group-2 bind groups for the per-pixel sub-object
    /// variants (surface VERTEX, curve POLY_NODE), aligned with `draw_set.draws`.
    /// A slot is `Some` only when that draw's type and the query mask select a
    /// per-pixel variant and its pipeline exists. Built before the pass so each
    /// bind group outlives it.
    fn build_pick_sublevel_binds(
        &self,
        device: &crate::gpu::Device,
        draw_set: &PickDrawSet,
    ) -> Vec<Option<PickSublevelBind>> {
        let feature = draw_set.primitive_index_supported;
        draw_set
            .draws
            .iter()
            .map(|(geom, inst)| match geom {
                PickGeom::Mesh(mesh_id) => {
                    let bgl = self.resources.pick.vertex_mesh_bgl.as_ref()?;
                    let obj = inst.object_id as u64;
                    let has_f2c = draw_set
                        .surface_meta
                        .get(&obj)
                        .is_some_and(|m| !m.is_empty());
                    // VERTEX and EDGE share the mesh vertex + index storage bind
                    // group; only the pipeline differs. At most one fires (their
                    // priority guards are exclusive).
                    let want_vertex = surface_writes_vertex(draw_set.mask, has_f2c, feature)
                        && self.resources.pick.vertex_pipeline.is_some();
                    let want_edge = surface_writes_edge(draw_set.mask, has_f2c, feature)
                        && self.resources.pick.edge_pipeline.is_some();
                    if !want_vertex && !want_edge {
                        return None;
                    }
                    let mesh = self.resources.mesh_store.get(*mesh_id)?;
                    let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("pick_vertex_mesh_bg"),
                        layout: bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: self.resources.geometry.vertex_binding(mesh.vertex_span),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: self.resources.geometry.index_binding(mesh.index_span),
                            },
                        ],
                    });
                    if want_vertex {
                        Some(PickSublevelBind::Vertex(bg))
                    } else {
                        Some(PickSublevelBind::Edge(bg))
                    }
                }
                PickGeom::Tube { node_buffer, .. } => {
                    let bgl = self.resources.pick.node_bgl.as_ref()?;
                    if !curve_writes_node(draw_set.mask, feature) {
                        return None;
                    }
                    let node_buf = (*node_buffer)?;
                    let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("pick_node_bg"),
                        layout: bgl,
                        entries: &[crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: node_buf.as_entire_binding(),
                        }],
                    });
                    Some(PickSublevelBind::Node(bg))
                }
            })
            .collect()
    }

    /// Record every pick pipeline's draw calls into `pick_pass`. Shared by the
    /// point and rect pick passes: both draw the same mask-selected geometry,
    /// differing only in the scissor rect the caller set before calling this.
    fn record_pick_pass_draws<'rp>(
        &'rp self,
        pick_pass: &mut crate::gpu::RenderPass<'rp>,
        pick_camera_bg: &'rp crate::gpu::BindGroup,
        pick_instance_bg: &'rp crate::gpu::BindGroup,
        draw_set: &PickDrawSet<'rp>,
        sublevel: &'rp [Option<PickSublevelBind>],
        frame: &'rp FrameData,
    ) {
        // Surface-pipeline draws: scene surfaces, volume-mesh boundaries, and
        // tube-family geometry all rasterise with the shared pick pipeline.
        // Type-level mask filtering already happened while building `draws`,
        // so an unbuilt or unrequested type contributes nothing and reads
        // back as no hit. Instance index in the storage buffer = position in
        // `draws`. A draw with a `sublevel` bind group switches to the per-pixel
        // VERTEX / NODE pipeline variant (writing the final sub-id into the
        // primitive channel) instead of the default face / segment pipeline.
        let default_pipeline = self
            .resources
            .pick
            .pipeline
            .as_ref()
            .expect("ensure_pick_pipeline must be called first");

        for (instance_slot, (geom, _)) in draw_set.draws.iter().enumerate() {
            let slot = instance_slot as u32;
            let variant = sublevel.get(instance_slot).and_then(|s| s.as_ref());
            match geom {
                PickGeom::Mesh(mesh_id) => {
                    let Some(mesh) = self.resources.mesh_store.get(*mesh_id) else {
                        continue;
                    };
                    match variant {
                        Some(PickSublevelBind::Vertex(bg)) => {
                            pick_pass.set_pipeline(
                                self.resources.pick.vertex_pipeline.as_ref().unwrap(),
                            );
                            pick_pass.set_bind_group(0, pick_camera_bg, &[]);
                            pick_pass.set_bind_group(1, pick_instance_bg, &[]);
                            pick_pass.set_bind_group(2, bg, &[]);
                        }
                        Some(PickSublevelBind::Edge(bg)) => {
                            pick_pass
                                .set_pipeline(self.resources.pick.edge_pipeline.as_ref().unwrap());
                            pick_pass.set_bind_group(0, pick_camera_bg, &[]);
                            pick_pass.set_bind_group(1, pick_instance_bg, &[]);
                            pick_pass.set_bind_group(2, bg, &[]);
                        }
                        _ => {
                            pick_pass.set_pipeline(default_pipeline);
                            pick_pass.set_bind_group(0, pick_camera_bg, &[]);
                            pick_pass.set_bind_group(1, pick_instance_bg, &[]);
                        }
                    }
                    pick_pass.set_vertex_buffer(
                        0,
                        self.resources.geometry.vertex_slice(mesh.vertex_span),
                    );
                    pick_pass.set_index_buffer(
                        self.resources.geometry.index_slice(mesh.index_span),
                        crate::gpu::IndexFormat::Uint32,
                    );
                    pick_pass.draw_indexed(0..mesh.index_count, 0, slot..slot + 1);
                }
                PickGeom::Tube {
                    vertex_buffer,
                    index_buffer,
                    index_count,
                    ..
                } => {
                    match variant {
                        Some(PickSublevelBind::Node(bg)) => {
                            pick_pass
                                .set_pipeline(self.resources.pick.node_pipeline.as_ref().unwrap());
                            pick_pass.set_bind_group(0, pick_camera_bg, &[]);
                            pick_pass.set_bind_group(1, pick_instance_bg, &[]);
                            pick_pass.set_bind_group(2, bg, &[]);
                        }
                        _ => {
                            pick_pass.set_pipeline(default_pipeline);
                            pick_pass.set_bind_group(0, pick_camera_bg, &[]);
                            pick_pass.set_bind_group(1, pick_instance_bg, &[]);
                        }
                    }
                    pick_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    pick_pass
                        .set_index_buffer(index_buffer.slice(..), crate::gpu::IndexFormat::Uint32);
                    pick_pass.draw_indexed(0..*index_count, 0, slot..slot + 1);
                }
            }
        }

        // Glyph / tensor-glyph sets: each draws its instanced base mesh with a
        // dedicated pipeline that reuses the render vertex transform and writes
        // the set's object id.
        for gd in &draw_set.glyph_draws {
            pick_pass.set_pipeline(gd.pipeline);
            pick_pass.set_bind_group(0, pick_camera_bg, &[]);
            pick_pass.set_bind_group(1, &gd.id_bind_group, &[]);
            pick_pass.set_bind_group(2, gd.instance_bind_group, &[]);
            pick_pass.set_vertex_buffer(0, gd.vertex_buffer.slice(..));
            pick_pass.set_index_buffer(gd.index_buffer.slice(..), crate::gpu::IndexFormat::Uint32);
            pick_pass.draw_indexed(0..gd.index_count, 0, 0..gd.instance_count);
        }

        // Sprite sets: camera-facing quads expanded in the vertex shader. The
        // pipeline and full camera bind group (group 0) are shared; each set
        // varies its sprite bind group, pick-id, and position buffer.
        if let Some(sprite_pipeline) = self.resources.sprite.pick_pipeline.as_ref() {
            if !draw_set.sprite_draws.is_empty() {
                pick_pass.set_pipeline(sprite_pipeline);
                pick_pass.set_bind_group(0, &self.resources.binds.camera_bg, &[]);
                for sd in &draw_set.sprite_draws {
                    pick_pass.set_bind_group(1, sd.sprite_bind_group, &[]);
                    pick_pass.set_bind_group(2, &sd.id_bind_group, &[]);
                    pick_pass.set_vertex_buffer(0, sd.vertex_buffer.slice(..));
                    pick_pass.draw(0..6, 0..sd.sprite_count);
                }
            }
        }

        // Polylines: screen-space thick lines, one draw per polyline of its
        // segment quads. Group 0 is the shared minimal pick camera.
        if let Some(polyline_pipeline) = self.resources.pick.polyline_pipeline.as_ref() {
            if !draw_set.polyline_draws.is_empty() {
                pick_pass.set_pipeline(polyline_pipeline);
                pick_pass.set_bind_group(0, pick_camera_bg, &[]);
                for pd in &draw_set.polyline_draws {
                    pick_pass.set_bind_group(1, pd.render_bind_group, &[]);
                    pick_pass.set_bind_group(2, &pd.id_bind_group, &[]);
                    pick_pass.set_vertex_buffer(0, pd.vertex_buffer.slice(..));
                    pick_pass.draw(0..6, 0..pd.segment_count);
                }
            }
        }

        // Voxel volumes: raymarch each bounding cube. Group 0 is the full
        // scene camera bind group (the volume pick fragment reads view_proj
        // and the clip volume); group 1 is the reused volume render bind
        // group; group 2 is the per-item object id.
        if let Some(volume_pipeline) = self.resources.pick.volume_pipeline.as_ref() {
            if !draw_set.volume_draws.is_empty() {
                pick_pass.set_pipeline(volume_pipeline);
                pick_pass.set_bind_group(0, &self.resources.binds.camera_bg, &[]);
                for vd in &draw_set.volume_draws {
                    pick_pass.set_bind_group(1, vd.render_bind_group, &[]);
                    pick_pass.set_bind_group(2, &vd.id_bind_group, &[]);
                    pick_pass.set_vertex_buffer(0, vd.vertex_buffer.slice(..));
                    pick_pass.set_index_buffer(
                        vd.index_buffer.slice(..),
                        crate::gpu::IndexFormat::Uint32,
                    );
                    // The volume cube is 36 indices (12 triangles).
                    pick_pass.draw_indexed(0..36, 0, 0..1);
                }
            }
        }

        // GPU implicit SDF surfaces: raymarch the isosurface on a full-screen
        // quad. Group 0 is the full scene camera bind group (the fragment reads
        // inv_view_proj to reconstruct the ray); group 1 is the reused implicit
        // uniform; group 2 is the per-item object id.
        if let Some(implicit_pipeline) = self.resources.pick.implicit_pipeline.as_ref() {
            if !draw_set.implicit_draws.is_empty() {
                pick_pass.set_pipeline(implicit_pipeline);
                pick_pass.set_bind_group(0, &self.resources.binds.camera_bg, &[]);
                for id in &draw_set.implicit_draws {
                    pick_pass.set_bind_group(1, id.render_bind_group, &[]);
                    pick_pass.set_bind_group(2, &id.id_bind_group, &[]);
                    pick_pass.draw(0..6, 0..1);
                }
            }
        }

        // GPU marching-cubes surfaces: rasterise each slab's generated vertex
        // buffer via its surface indirect args. Group 0 is the shared minimal
        // pick camera; group 1 is the per-item object id.
        if let Some(mc_pipeline) = self.resources.pick.mc_pipeline.as_ref() {
            if !draw_set.mc_draws.is_empty() {
                pick_pass.set_pipeline(mc_pipeline);
                pick_pass.set_bind_group(0, pick_camera_bg, &[]);
                for md in &draw_set.mc_draws {
                    pick_pass.set_bind_group(1, &md.id_bind_group, &[]);
                    for (vertex_buf, indirect_buf) in &md.slabs {
                        pick_pass.set_vertex_buffer(0, vertex_buf.slice(..));
                        pick_pass.draw_indirect(indirect_buf, 0);
                    }
                }
            }
        }

        // Point clouds: each item draws its screen-space quad expansion.
        // Group 0 is the full scene camera bind group (the expansion needs
        // the viewport size); group 1 is the reused render bind group;
        // group 2 is the per-item object id.
        if let Some(point_cloud_pipeline) = self.resources.pick.point_cloud_pipeline.as_ref() {
            if !draw_set.point_cloud_draws.is_empty() {
                pick_pass.set_pipeline(point_cloud_pipeline);
                pick_pass.set_bind_group(0, &self.resources.binds.camera_bg, &[]);
                for pcd in &draw_set.point_cloud_draws {
                    pick_pass.set_bind_group(1, pcd.render_bind_group, &[]);
                    pick_pass.set_bind_group(2, &pcd.id_bind_group, &[]);
                    pick_pass.set_vertex_buffer(0, pcd.vertex_buffer.slice(..));
                    pick_pass.draw(0..6, 0..pcd.point_count);
                }
            }
        }

        // Gaussian splats: each item draws its covariance-projected
        // billboard expansion. Group 0 is the minimal pick camera; group 1
        // is the reused per-viewport sorted-index render bind group; group
        // 2 is the per-item object id.
        if let Some(splat_pipeline) = self.resources.pick.gaussian_splat_pipeline.as_ref() {
            if !draw_set.splat_draws.is_empty() {
                pick_pass.set_pipeline(splat_pipeline);
                pick_pass.set_bind_group(0, pick_camera_bg, &[]);
                for sd in &draw_set.splat_draws {
                    pick_pass.set_bind_group(1, sd.render_bind_group, &[]);
                    pick_pass.set_bind_group(2, &sd.id_bind_group, &[]);
                    pick_pass.draw(0..6, 0..sd.count);
                }
            }
        }

        // Image slices: each item draws its quad-from-vertex-index
        // expansion. Group 0 is the minimal pick camera; group 1 is the
        // reused render bind group; group 2 is the per-item object id.
        if let Some(image_slice_pipeline) = self.resources.pick.image_slice_pipeline.as_ref() {
            if !draw_set.image_slice_draws.is_empty() {
                pick_pass.set_pipeline(image_slice_pipeline);
                pick_pass.set_bind_group(0, pick_camera_bg, &[]);
                for isd in &draw_set.image_slice_draws {
                    pick_pass.set_bind_group(1, isd.render_bind_group, &[]);
                    pick_pass.set_bind_group(2, &isd.id_bind_group, &[]);
                    pick_pass.draw(0..6, 0..1);
                }
            }
        }

        // Volume surface slices: each item draws its mesh. Group 0 is the
        // minimal pick camera; group 1 is the reused render bind group;
        // group 2 is the per-item object id.
        if let Some(vss_pipeline) = self.resources.pick.volume_surface_slice_pipeline.as_ref() {
            if !draw_set.volume_surface_slice_draws.is_empty() {
                pick_pass.set_pipeline(vss_pipeline);
                pick_pass.set_bind_group(0, pick_camera_bg, &[]);
                for vsd in &draw_set.volume_surface_slice_draws {
                    let Some(mesh) = self.resources.mesh_store.get(vsd.mesh_id) else {
                        continue;
                    };
                    pick_pass.set_bind_group(1, vsd.render_bind_group, &[]);
                    pick_pass.set_bind_group(2, &vsd.id_bind_group, &[]);
                    pick_pass.set_vertex_buffer(
                        0,
                        self.resources.geometry.vertex_slice(mesh.vertex_span),
                    );
                    pick_pass.set_index_buffer(
                        self.resources.geometry.index_slice(mesh.index_span),
                        crate::gpu::IndexFormat::Uint32,
                    );
                    pick_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
        }

        // Item-type plugins render their own pick-ids last. They build their
        // pipelines against the full shared group-0 layout, so bind the full
        // camera bind group (the same one the sprite draws use) before
        // handing them the pass.
        if draw_set.has_plugin_pick {
            pick_pass.set_bind_group(0, &self.resources.binds.camera_bg, &[]);
            self.dispatch_plugin_pick(pick_pass, frame, draw_set.mask);
        }
    }

    /// Upload once (and return) the shared unit-cube mesh used as the box pick
    /// proxy (decals and box scatter volumes). Reused across picks; re-uploaded
    /// if the cached handle was freed (e.g. after device recreation). Returns
    /// `None` only if the upload fails.
    fn ensure_decal_pick_cube(
        &mut self,
        device: &crate::gpu::Device,
    ) -> Option<crate::resources::mesh::mesh_store::MeshId> {
        if let Some(id) = self.decal_pick_cube {
            if self.resources.mesh_store.get(id).is_some() {
                return Some(id);
            }
        }
        let id = self
            .resources
            .upload_mesh_data(device, &unit_cube_mesh_data())
            .ok()?;
        self.decal_pick_cube = Some(id);
        Some(id)
    }

    /// Upload once (and return) the shared unit-radius icosphere used as the pick
    /// proxy for sphere scatter volumes. Reused across picks; re-uploaded if the
    /// cached handle was freed. Returns `None` only if the upload fails.
    fn ensure_scatter_pick_sphere(
        &mut self,
        device: &crate::gpu::Device,
    ) -> Option<crate::resources::mesh::mesh_store::MeshId> {
        if let Some(id) = self.scatter_pick_sphere {
            if self.resources.mesh_store.get(id).is_some() {
                return Some(id);
            }
        }
        // Two subdivisions: a close spherical silhouette for object-level picking
        // without much geometry.
        let mesh = crate::geometry::primitives::icosphere(1.0, 2);
        let id = self.resources.upload_mesh_data(device, &mesh).ok()?;
        self.scatter_pick_sphere = Some(id);
        Some(id)
    }

    /// GPU object-id rect pick: renders the mask-selected geometry, scissored
    /// to `rect_min..rect_max`, and reads back the whole region to collect the
    /// unique object ids touched by the rect.
    ///
    /// Object-level only, and only when `mask` intersects `OBJECT`: unlike the
    /// point pick path, this does not decode the primitive channel into
    /// sub-object identity (that would mean resolving face / cell / vertex /
    /// instance per pixel over the whole rect, not just the one hit pixel).
    /// A caller that passes a mask with no `OBJECT` bit gets an empty result
    /// without any GPU work, rather than a silent reinterpretation of what it
    /// asked for.
    ///
    /// This blocks: it submits the id pass then waits on its own submission
    /// index before reading the region back (the same targeted wait
    /// `pick_scene_gpu_masked` uses, not a full queue drain).
    pub(crate) fn pick_rect_gpu(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        rect_min: glam::Vec2,
        rect_max: glam::Vec2,
        frame: &FrameData,
        mask: PickMask,
    ) -> crate::renderer::picking::PickRectResult {
        let wants_object = mask.intersects(PickMask::OBJECT);

        // Screen-space overlay images have no world-space geometry, so they are
        // tested directly against the logical-space query rect rather than
        // drawn into the id pass; see `screen_image_hits_in_rect`. OBJECT-only,
        // matching the CPU backend and the point path.
        let viewport_size_logical = glam::Vec2::from(frame.camera.viewport_size);
        let logical_lo = glam::Vec2::new(rect_min.x.min(rect_max.x), rect_min.y.min(rect_max.y));
        let logical_hi = glam::Vec2::new(rect_min.x.max(rect_max.x), rect_min.y.max(rect_max.y));
        let mut screen_image_objects = if wants_object {
            screen_image_hits_in_rect(
                &frame.scene.screen_images,
                viewport_size_logical,
                logical_lo,
                logical_hi,
            )
        } else {
            Vec::new()
        };

        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items.as_ref(),
        };

        let ppp = frame.camera.pixels_per_point;
        let vp_w = (frame.camera.viewport_size[0] * ppp).round() as u32;
        let vp_h = (frame.camera.viewport_size[1] * ppp).round() as u32;
        if vp_w == 0 || vp_h == 0 {
            return crate::renderer::picking::PickRectResult {
                objects: screen_image_objects,
                elements: Vec::new(),
            };
        }

        // Physical rect bounds, clamped to the viewport. `rect_min`/`rect_max`
        // are not assumed ordered.
        let lo = logical_lo * ppp;
        let hi = logical_hi * ppp;
        let rx = (lo.x.floor().max(0.0) as u32).min(vp_w);
        let ry = (lo.y.floor().max(0.0) as u32).min(vp_h);
        let rx_end = (hi.x.ceil().max(0.0) as u32).min(vp_w);
        let ry_end = (hi.y.ceil().max(0.0) as u32).min(vp_h);
        if rx_end <= rx || ry_end <= ry {
            return crate::renderer::picking::PickRectResult {
                objects: screen_image_objects,
                elements: Vec::new(),
            };
        }
        let rw = rx_end - rx;
        let rh = ry_end - ry;

        let flags = self.ensure_pick_pipelines(device, frame, mask);
        let draw_set = self.build_pick_draws(device, queue, frame, mask, scene_items, &flags);
        if draw_set.is_empty() {
            return crate::renderer::picking::PickRectResult {
                objects: screen_image_objects,
                elements: Vec::new(),
            };
        }

        let primitive_index_supported = draw_set.primitive_index_supported;
        // Decode map for the primitive channel, built from the same collections
        // the pass draws. Only needed when the caller asked for a sub-object level.
        let kinds = if mask.intersects(!PickMask::OBJECT) {
            self.build_pick_sub_kinds(frame, scene_items)
        } else {
            std::collections::HashMap::new()
        };

        let (_, pick_instance_bg) =
            self.build_pick_instance_bind_group(device, queue, &draw_set.draws);
        let (_, pick_camera_bg) = self.build_pick_camera_bind_group(device, queue, frame);
        let sublevel = self.build_pick_sublevel_binds(device, &draw_set);
        let targets = PickTargets::new(device, vp_w, vp_h);

        self.resources.geometry.flush(queue);
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("pick_rect_pass_encoder"),
        });
        {
            let mut pick_pass = targets.begin_render_pass(&mut encoder);
            // Only the requested rect is read back, so the fragment stage and
            // depth test collapse to that region regardless of object count or
            // overdraw, the same fast path the point pick uses at 1x1.
            pick_pass.set_scissor_rect(rx, ry, rw, rh);
            self.record_pick_pass_draws(
                &mut pick_pass,
                &pick_camera_bg,
                &pick_instance_bg,
                &draw_set,
                &sublevel,
                frame,
            );
        }

        // Read back the object-id channel plus, when a sub-object level was
        // asked for, the primitive channel: each pixel's `(object_id,
        // primitive_id)` decodes to a `SubObjectRef` the same way the point path
        // does, but from the rasterised index alone (no per-pixel ray).
        let wants_sub = mask.intersects(!PickMask::OBJECT);
        let bytes_per_row = (rw * 4).div_ceil(256) * 256;
        let make_staging = |label: &str| {
            device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some(label),
                size: (bytes_per_row as u64) * (rh as u64),
                usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let copy_region = |encoder: &mut crate::gpu::CommandEncoder,
                           texture: &crate::gpu::Texture,
                           staging: &crate::gpu::Buffer| {
            encoder.copy_texture_to_buffer(
                crate::gpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: crate::gpu::Origin3d { x: rx, y: ry, z: 0 },
                    aspect: crate::gpu::TextureAspect::All,
                },
                crate::gpu::TexelCopyBufferInfo {
                    buffer: staging,
                    layout: crate::gpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(rh),
                    },
                },
                crate::gpu::Extent3d {
                    width: rw,
                    height: rh,
                    depth_or_array_layers: 1,
                },
            );
        };

        let id_staging = make_staging("pick_rect_id_staging");
        copy_region(&mut encoder, &targets.id_texture, &id_staging);
        let prim_staging = wants_sub.then(|| {
            let s = make_staging("pick_rect_prim_staging");
            copy_region(&mut encoder, &targets.prim_texture, &s);
            s
        });
        // Plugin sub-object refinement needs each hit pixel's world position
        // (vertex / edge snap in `resolve_sub_object`), reconstructed from the
        // depth channel. Only read it back when a plugin item is in the decode
        // map; the built-in kinds decode from the primitive channel alone.
        let has_plugin_kinds = kinds.values().any(|k| matches!(k, PickSubKind::Plugin(_)));
        let depth_staging = (wants_sub && has_plugin_kinds).then(|| {
            let s = make_staging("pick_rect_depth_staging");
            copy_region(&mut encoder, &targets.depth_colour_texture, &s);
            s
        });

        let submission = queue.submit(std::iter::once(encoder.finish()));
        id_staging
            .slice(..)
            .map_async(crate::gpu::MapMode::Read, |_| {});
        if let Some(prim) = &prim_staging {
            prim.slice(..).map_async(crate::gpu::MapMode::Read, |_| {});
        }
        if let Some(depth) = &depth_staging {
            depth.slice(..).map_async(crate::gpu::MapMode::Read, |_| {});
        }
        device
            .poll(crate::gpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();

        let mut seen: std::collections::HashSet<u32> =
            screen_image_objects.iter().map(|&id| id as u32).collect();
        let mut objects = std::mem::take(&mut screen_image_objects);
        let mut elements: Vec<(u64, SubObjectRef)> = Vec::new();
        let mut seen_elem: std::collections::HashSet<(u64, SubObjectRef)> =
            std::collections::HashSet::new();
        {
            let id_data = id_staging.slice(..).get_mapped_range();
            let prim_view = prim_staging
                .as_ref()
                .map(|s| s.slice(..).get_mapped_range());
            let depth_view = depth_staging
                .as_ref()
                .map(|s| s.slice(..).get_mapped_range());
            let view_proj_inv = frame.camera.render_camera.view_proj().inverse();
            for row in 0..rh as usize {
                let row_start = row * bytes_per_row as usize;
                for col in 0..rw as usize {
                    let px_off = row_start + col * 4;
                    let id = u32::from_le_bytes([
                        id_data[px_off],
                        id_data[px_off + 1],
                        id_data[px_off + 2],
                        id_data[px_off + 3],
                    ]);
                    if id == 0 {
                        continue;
                    }
                    if wants_object && seen.insert(id) {
                        objects.push(id as u64);
                    }
                    if let Some(prim_data) = &prim_view {
                        let prim = u32::from_le_bytes([
                            prim_data[px_off],
                            prim_data[px_off + 1],
                            prim_data[px_off + 2],
                            prim_data[px_off + 3],
                        ]);
                        // Pixel world position from the depth channel, for the
                        // plugin refinement hook.
                        let world_pos = depth_view.as_ref().map(|depth_data| {
                            let depth = f32::from_le_bytes([
                                depth_data[px_off],
                                depth_data[px_off + 1],
                                depth_data[px_off + 2],
                                depth_data[px_off + 3],
                            ]);
                            let phys_x = (rx + col as u32) as f32 + 0.5;
                            let phys_y = (ry + row as u32) as f32 + 0.5;
                            let ndc_x = 2.0 * phys_x / vp_w as f32 - 1.0;
                            let ndc_y = 1.0 - 2.0 * phys_y / vp_h as f32;
                            view_proj_inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, depth))
                        });
                        if let Some(sub) = self.resolve_gpu_sub_object_rect(
                            id as u64,
                            prim,
                            mask,
                            &kinds,
                            &draw_set.surface_meta,
                            primitive_index_supported,
                            world_pos,
                        ) {
                            if seen_elem.insert((id as u64, sub)) {
                                elements.push((id as u64, sub));
                            }
                        }
                    }
                }
            }
        }
        id_staging.unmap();
        if let Some(prim) = &prim_staging {
            prim.unmap();
        }
        if let Some(depth) = &depth_staging {
            depth.unmap();
        }

        crate::renderer::picking::PickRectResult { objects, elements }
    }

    /// Best snap candidate within `radius_px` screen pixels of `cursor`.
    ///
    /// Renders the mask-selected geometry into the pick pass scissored to a
    /// square window around the cursor, reads the object / primitive / depth
    /// channels over the window, and reduces the covered pixels to one candidate
    /// by feature priority (point-like vertex / node > edge / segment > surface /
    /// object), tie-broken by screen-space distance to the cursor. The returned
    /// `world_pos` is the exact feature coordinate from
    /// [`snap_world_pos`](Self::snap_world_pos), falling back to the pixel's
    /// reconstructed world position. Blocking, like
    /// [`pick_rect_gpu`](Self::pick_rect_gpu): it waits on the pass before
    /// reading back.
    pub(crate) fn snap_query_gpu(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        cursor: glam::Vec2,
        radius_px: f32,
        frame: &FrameData,
        mask: PickMask,
    ) -> Option<crate::renderer::picking::SnapHit> {
        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items.as_ref(),
        };

        let ppp = frame.camera.pixels_per_point;
        let vp_w = (frame.camera.viewport_size[0] * ppp).round() as u32;
        let vp_h = (frame.camera.viewport_size[1] * ppp).round() as u32;
        if vp_w == 0 || vp_h == 0 {
            return None;
        }

        // Physical window around the cursor, clamped to the viewport. The radius
        // is a logical (screen-point) tolerance, so it scales by `ppp` into the
        // physical pick target the same way the cursor does.
        let radius = radius_px.max(0.0);
        let lo = (cursor - glam::Vec2::splat(radius)) * ppp;
        let hi = (cursor + glam::Vec2::splat(radius)) * ppp;
        let rx = (lo.x.floor().max(0.0) as u32).min(vp_w);
        let ry = (lo.y.floor().max(0.0) as u32).min(vp_h);
        let rx_end = ((hi.x.ceil().max(0.0) as u32) + 1).min(vp_w);
        let ry_end = ((hi.y.ceil().max(0.0) as u32) + 1).min(vp_h);
        if rx_end <= rx || ry_end <= ry {
            return None;
        }
        let rw = rx_end - rx;
        let rh = ry_end - ry;

        let flags = self.ensure_pick_pipelines(device, frame, mask);
        let draw_set = self.build_pick_draws(device, queue, frame, mask, scene_items, &flags);
        if draw_set.is_empty() {
            return None;
        }

        let primitive_index_supported = draw_set.primitive_index_supported;
        // Decode map for the primitive channel; only built when the caller asked
        // for a sub-object level to snap to.
        let kinds = if mask.intersects(!PickMask::OBJECT) {
            self.build_pick_sub_kinds(frame, scene_items)
        } else {
            std::collections::HashMap::new()
        };

        let (_, pick_instance_bg) =
            self.build_pick_instance_bind_group(device, queue, &draw_set.draws);
        let (_, pick_camera_bg) = self.build_pick_camera_bind_group(device, queue, frame);
        let sublevel = self.build_pick_sublevel_binds(device, &draw_set);
        let targets = PickTargets::new(device, vp_w, vp_h);

        self.resources.geometry.flush(queue);
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("snap_query_pass_encoder"),
        });
        {
            let mut pick_pass = targets.begin_render_pass(&mut encoder);
            pick_pass.set_scissor_rect(rx, ry, rw, rh);
            self.record_pick_pass_draws(
                &mut pick_pass,
                &pick_camera_bg,
                &pick_instance_bg,
                &draw_set,
                &sublevel,
                frame,
            );
        }

        // Read the object, primitive, and depth channels over the window: the
        // object and primitive decode to a `SubObjectRef` exactly as the rect
        // path does, and the depth reconstructs each pixel's world position.
        let bytes_per_row = (rw * 4).div_ceil(256) * 256;
        let make_staging = |label: &str| {
            device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some(label),
                size: (bytes_per_row as u64) * (rh as u64),
                usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let copy_region = |encoder: &mut crate::gpu::CommandEncoder,
                           texture: &crate::gpu::Texture,
                           staging: &crate::gpu::Buffer| {
            encoder.copy_texture_to_buffer(
                crate::gpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: crate::gpu::Origin3d { x: rx, y: ry, z: 0 },
                    aspect: crate::gpu::TextureAspect::All,
                },
                crate::gpu::TexelCopyBufferInfo {
                    buffer: staging,
                    layout: crate::gpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(rh),
                    },
                },
                crate::gpu::Extent3d {
                    width: rw,
                    height: rh,
                    depth_or_array_layers: 1,
                },
            );
        };

        let id_staging = make_staging("snap_query_id_staging");
        let prim_staging = make_staging("snap_query_prim_staging");
        let depth_staging = make_staging("snap_query_depth_staging");
        copy_region(&mut encoder, &targets.id_texture, &id_staging);
        copy_region(&mut encoder, &targets.prim_texture, &prim_staging);
        copy_region(&mut encoder, &targets.depth_colour_texture, &depth_staging);

        let submission = queue.submit(std::iter::once(encoder.finish()));
        for staging in [&id_staging, &prim_staging, &depth_staging] {
            staging
                .slice(..)
                .map_async(crate::gpu::MapMode::Read, |_| {});
        }
        device
            .poll(crate::gpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();

        let view_proj_inv = frame.camera.render_camera.view_proj().inverse();
        // Best candidate: higher feature priority wins, ties broken by the
        // pixel's screen-space distance to the cursor. Held as plain data so the
        // exact feature coordinate is resolved after the staging maps are freed.
        let mut best: Option<(i32, f32, u64, Option<SubObjectRef>, glam::Vec3)> = None;
        {
            let id_data = id_staging.slice(..).get_mapped_range();
            let prim_data = prim_staging.slice(..).get_mapped_range();
            let depth_data = depth_staging.slice(..).get_mapped_range();
            for row in 0..rh as usize {
                let row_start = row * bytes_per_row as usize;
                for col in 0..rw as usize {
                    let px_off = row_start + col * 4;
                    let id = u32::from_le_bytes([
                        id_data[px_off],
                        id_data[px_off + 1],
                        id_data[px_off + 2],
                        id_data[px_off + 3],
                    ]);
                    if id == 0 {
                        continue;
                    }

                    // Screen-space distance from this pixel's centre to the
                    // cursor, in logical points; skip pixels outside the circular
                    // tolerance so the window reads as a radius, not a square.
                    let phys_x = (rx + col as u32) as f32 + 0.5;
                    let phys_y = (ry + row as u32) as f32 + 0.5;
                    let screen = glam::Vec2::new(phys_x / ppp, phys_y / ppp);
                    let dist = (screen - cursor).length();
                    if dist > radius {
                        continue;
                    }

                    let prim = u32::from_le_bytes([
                        prim_data[px_off],
                        prim_data[px_off + 1],
                        prim_data[px_off + 2],
                        prim_data[px_off + 3],
                    ]);
                    // World position before resolution: the plugin refinement
                    // hook consumes it for vertex / edge snapping.
                    let depth = f32::from_le_bytes([
                        depth_data[px_off],
                        depth_data[px_off + 1],
                        depth_data[px_off + 2],
                        depth_data[px_off + 3],
                    ]);
                    let ndc_x = 2.0 * phys_x / vp_w as f32 - 1.0;
                    let ndc_y = 1.0 - 2.0 * phys_y / vp_h as f32;
                    let world = view_proj_inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, depth));
                    let sub = self.resolve_gpu_sub_object_rect(
                        id as u64,
                        prim,
                        mask,
                        &kinds,
                        &draw_set.surface_meta,
                        primitive_index_supported,
                        Some(world),
                    );
                    let priority = snap_priority(sub);

                    let better = match best {
                        None => true,
                        Some((bp, bd, ..)) => priority > bp || (priority == bp && dist < bd),
                    };
                    if !better {
                        continue;
                    }
                    best = Some((priority, dist, id as u64, sub, world));
                }
            }
        }
        id_staging.unmap();
        prim_staging.unmap();
        depth_staging.unmap();

        let (_, _, object_id, sub_object, pixel_world) = best?;
        // Snap to the exact feature coordinate when it is known; otherwise the
        // reconstructed pixel world position is the snap target.
        let world_pos = self
            .snap_world_pos(object_id, sub_object, pixel_world, frame)
            .unwrap_or(pixel_world);
        Some(crate::renderer::picking::SnapHit {
            world_pos,
            object_id,
            sub_object,
        })
    }

    /// Begin a non-blocking GPU object pick under `cursor`: submit the id pass
    /// and park the in-flight staging buffers on the renderer. The result is read
    /// on a later [`pick_object_poll`](Self::pick_object_poll) call, so the
    /// calling thread never blocks on the GPU queue.
    ///
    /// Any pick already in flight is dropped and replaced. Returns `true` when a
    /// pass was submitted, `false` when the cursor is out of bounds or nothing
    /// pickable would draw (in which case `pick_object_poll` reports no hit).
    ///
    /// `mask` selects item types and sub-object levels exactly as
    /// [`pick_object`](Self::pick_object) does; the resolved hit carries the
    /// same `sub_object` the blocking path would produce.
    pub fn pick_object_begin(
        &mut self,
        cursor: glam::Vec2,
        frame: &FrameData,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mask: PickMask,
    ) -> bool {
        match self.pick_scene_gpu_begin(device, queue, cursor, frame, mask) {
            PickBegin::Miss => {
                self.pending_pick = None;
                false
            }
            PickBegin::Pending(p) => {
                self.pending_pick = Some(p);
                true
            }
        }
    }

    /// Poll the pick started by [`pick_object_begin`](Self::pick_object_begin).
    ///
    /// Non-blocking: it drives the device's map callbacks with a non-waiting poll
    /// and returns [`PickPoll::Pending`] until both staging maps have landed
    /// (typically the next frame, once the render loop's own submissions have
    /// flushed the pick pass through). When they have, it reads the pixel, clears
    /// the pending slot, and returns [`PickPoll::Ready`] with the hit (or `None`
    /// for empty space). [`PickPoll::Idle`] means no pick is in flight.
    ///
    /// This never waits on the GPU queue, so unlike the blocking
    /// [`pick_scene_gpu_masked`](Self::pick_scene_gpu_masked) it cannot stall the
    /// calling thread behind queued frames. It does assume the app keeps
    /// rendering (or otherwise polls the device); a caller that submits no other
    /// work between polls should use the blocking path instead.
    pub fn pick_object_poll(
        &mut self,
        device: &crate::gpu::Device,
    ) -> crate::renderer::picking::PickPoll {
        use crate::renderer::picking::PickPoll;
        if self.pending_pick.is_none() {
            return PickPoll::Idle;
        }

        // Drive map callbacks without waiting on the queue. This processes work
        // that has already completed; it does not block for the pick submission.
        let _ = device.poll(crate::gpu::PollType::Poll);

        let pending = self.pending_pick.as_ref().expect("pending checked above");
        if pending.ready.load(std::sync::atomic::Ordering::Acquire) < 3 {
            return PickPoll::Pending;
        }

        let pending = self.pending_pick.take().expect("pending checked above");
        let hit = pending
            .read_hit()
            .map(|h| self.resolve_pending_hit(&pending, h));
        PickPoll::Ready(hit)
    }

    /// Build the object-level [`PickHit`] from a raw GPU hit, then fill its
    /// `sub_object` from the read-back primitive channel using the pick's
    /// per-object decode map. Shared by the blocking and async paths.
    fn resolve_pending_hit(&self, pending: &PendingPick, gpu_hit: GpuPickHit) -> PickHit {
        let mut hit = gpu_hit.to_pick_hit(pending.cursor, pending.viewport_size, pending.view_proj);
        hit.sub_object = self.resolve_gpu_sub_object(
            gpu_hit.object_id.0,
            gpu_hit.sub_primitive,
            pending.mask,
            &pending.kinds,
            &pending.surface_meta,
            pending.primitive_index_supported,
            Some(hit.world_pos),
        );
        hit
    }

    /// Decode a read-back `(object_id, sub_primitive)` into a [`SubObjectRef`]
    /// per the hit object's type. Object-level types (and any object not in the
    /// decode map) return `None`.
    ///
    /// The surface, curve, and plugin paths read `primitive_index`, so they
    /// only resolve when the device had `SHADER_PRIMITIVE_INDEX`; instanced
    /// types and polylines read `instance_index`, which needs no device
    /// feature. `world_pos` is the hit's reconstructed world position, consumed
    /// by the plugin path (vertex/edge snapping in `resolve_sub_object`).
    fn resolve_gpu_sub_object(
        &self,
        object_id: u64,
        sub_primitive: u32,
        mask: PickMask,
        kinds: &std::collections::HashMap<u64, PickSubKind>,
        surface_meta: &SurfacePickMeta,
        primitive_index_supported: bool,
        world_pos: Option<glam::Vec3>,
    ) -> Option<SubObjectRef> {
        match kinds.get(&object_id).copied()? {
            PickSubKind::Instance => {
                if mask.intersects(PickMask::INSTANCE) {
                    Some(SubObjectRef::Instance(sub_primitive))
                } else {
                    None
                }
            }
            PickSubKind::Polyline => {
                if mask.intersects(PickMask::STRIP) {
                    let strip = self
                        .polyline_gpu_data
                        .iter()
                        .find(|g| g.pick_id.0 == object_id)
                        .map(|g| strip_for_segment(sub_primitive, &g.strip_lengths))
                        .unwrap_or(0);
                    Some(SubObjectRef::Strip(strip))
                } else if mask.intersects(PickMask::SEGMENT | PickMask::POLY_NODE) {
                    Some(SubObjectRef::Segment(sub_primitive))
                } else {
                    None
                }
            }
            PickSubKind::Curve => {
                // Curve sub-object picking needs the primitive index. Without the
                // feature the GPU path stays object-level: no silent per-click CPU
                // test.
                if !primitive_index_supported {
                    return None;
                }
                // The POLY_NODE variant wrote the final node index into the channel.
                if curve_writes_node(mask, primitive_index_supported) {
                    return Some(SubObjectRef::Point(sub_primitive));
                }
                // Otherwise the channel is the hit triangle; map it to the item's
                // segment / strip through the persistent per-triangle tables.
                let gpu = self
                    .streamtube_gpu_data
                    .iter()
                    .chain(self.tube_gpu_data.iter())
                    .chain(self.ribbon_gpu_data.iter())
                    .find(|g| g.pick_id.0 == object_id)?;
                if mask.intersects(PickMask::STRIP) {
                    gpu.tri_strip
                        .get(sub_primitive as usize)
                        .copied()
                        .map(SubObjectRef::Strip)
                } else if mask.intersects(PickMask::SEGMENT) {
                    gpu.tri_segment
                        .get(sub_primitive as usize)
                        .copied()
                        .map(SubObjectRef::Segment)
                } else {
                    None
                }
            }
            PickSubKind::CloudPoint => {
                if mask.intersects(PickMask::CLOUD_POINT) {
                    Some(SubObjectRef::Point(sub_primitive))
                } else {
                    None
                }
            }
            PickSubKind::Splat => {
                if mask.intersects(PickMask::SPLAT) {
                    Some(SubObjectRef::Splat(sub_primitive))
                } else {
                    None
                }
            }
            PickSubKind::Voxel => {
                if mask.intersects(PickMask::VOXEL) {
                    Some(SubObjectRef::Voxel(sub_primitive))
                } else {
                    None
                }
            }
            PickSubKind::Surface => self.resolve_surface_sub_object(
                object_id,
                sub_primitive,
                mask,
                surface_meta,
                primitive_index_supported,
            ),
            PickSubKind::Plugin(name) => self.resolve_plugin_sub_object(
                name,
                object_id,
                sub_primitive,
                mask,
                primitive_index_supported,
                world_pos,
            ),
        }
    }

    /// Refine a hit on a plugin-drawn item through the owning plugin's
    /// `resolve_sub_object` hook. Needs `SHADER_PRIMITIVE_INDEX` (without it
    /// the plugin's pick fragment wrote a constant 0) and the hit's
    /// reconstructed world position. A plugin without the hook returns `None`
    /// and the hit stays object-level.
    fn resolve_plugin_sub_object(
        &self,
        name: &'static str,
        object_id: u64,
        sub_primitive: u32,
        mask: PickMask,
        primitive_index_supported: bool,
        world_pos: Option<glam::Vec3>,
    ) -> Option<SubObjectRef> {
        if !primitive_index_supported {
            return None;
        }
        if !mask.intersects(PickMask::FACE | PickMask::VERTEX | PickMask::EDGE | PickMask::CELL) {
            return None;
        }
        let plugin = self.item_type_plugins.get(name)?;
        plugin.resolve_sub_object(PickId(object_id), sub_primitive, world_pos?, mask)
    }

    /// Decode a read-back `(object_id, sub_primitive)` into a [`SubObjectRef`]
    /// for rect picking, using only the primitive channel (no per-pixel ray).
    ///
    /// A rect query has one primitive id per pixel but no single cursor ray. Every
    /// level reads the primitive channel directly: instance / cloud-point / splat /
    /// voxel (the index is the element), polyline and curve segment / strip, surface
    /// face / cell, and surface `VERTEX` / curve `POLY_NODE`. The last two need no
    /// cursor ray because their pipeline variants already wrote the nearest corner /
    /// node index into the channel per pixel. Surface and curve sub-objects need
    /// `SHADER_PRIMITIVE_INDEX` (there is no per-pixel CPU refine for a rect).
    ///
    /// `world_pos` is this pixel's world position reconstructed from the depth
    /// channel; the rect and snap paths supply it when a plugin item is in the
    /// decode map, and the plugin path forwards it to `resolve_sub_object`.
    fn resolve_gpu_sub_object_rect(
        &self,
        object_id: u64,
        sub_primitive: u32,
        mask: PickMask,
        kinds: &std::collections::HashMap<u64, PickSubKind>,
        surface_meta: &SurfacePickMeta,
        primitive_index_supported: bool,
        world_pos: Option<glam::Vec3>,
    ) -> Option<SubObjectRef> {
        match kinds.get(&object_id).copied()? {
            PickSubKind::Instance => mask
                .intersects(PickMask::INSTANCE)
                .then_some(SubObjectRef::Instance(sub_primitive)),
            PickSubKind::CloudPoint => mask
                .intersects(PickMask::CLOUD_POINT)
                .then_some(SubObjectRef::Point(sub_primitive)),
            PickSubKind::Splat => mask
                .intersects(PickMask::SPLAT)
                .then_some(SubObjectRef::Splat(sub_primitive)),
            PickSubKind::Voxel => mask
                .intersects(PickMask::VOXEL)
                .then_some(SubObjectRef::Voxel(sub_primitive)),
            PickSubKind::Polyline => {
                if mask.intersects(PickMask::STRIP) {
                    let strip = self
                        .polyline_gpu_data
                        .iter()
                        .find(|g| g.pick_id.0 == object_id)
                        .map(|g| strip_for_segment(sub_primitive, &g.strip_lengths))
                        .unwrap_or(0);
                    Some(SubObjectRef::Strip(strip))
                } else if mask.intersects(PickMask::SEGMENT | PickMask::POLY_NODE) {
                    Some(SubObjectRef::Segment(sub_primitive))
                } else {
                    None
                }
            }
            PickSubKind::Curve => {
                if !primitive_index_supported {
                    return None;
                }
                // POLY_NODE variant: the channel is the final node index.
                if curve_writes_node(mask, primitive_index_supported) {
                    return Some(SubObjectRef::Point(sub_primitive));
                }
                let gpu = self
                    .streamtube_gpu_data
                    .iter()
                    .chain(self.tube_gpu_data.iter())
                    .chain(self.ribbon_gpu_data.iter())
                    .find(|g| g.pick_id.0 == object_id)?;
                if mask.intersects(PickMask::STRIP) {
                    gpu.tri_strip
                        .get(sub_primitive as usize)
                        .copied()
                        .map(SubObjectRef::Strip)
                } else if mask.intersects(PickMask::SEGMENT) {
                    gpu.tri_segment
                        .get(sub_primitive as usize)
                        .copied()
                        .map(SubObjectRef::Segment)
                } else {
                    None
                }
            }
            PickSubKind::Surface => {
                if !primitive_index_supported {
                    return None;
                }
                let has_f2c = surface_meta.get(&object_id).is_some_and(|m| !m.is_empty());
                // VERTEX variant: the channel is the final global vertex index.
                if surface_writes_vertex(mask, has_f2c, primitive_index_supported) {
                    return Some(SubObjectRef::Vertex(sub_primitive));
                }
                // EDGE variant: the channel is the nearest edge id.
                if surface_writes_edge(mask, has_f2c, primitive_index_supported) {
                    return Some(SubObjectRef::Edge(sub_primitive));
                }
                if mask.intersects(PickMask::FACE) {
                    Some(SubObjectRef::Face(sub_primitive))
                } else if mask.intersects(PickMask::CELL) {
                    surface_meta
                        .get(&object_id)
                        .filter(|m| !m.is_empty())
                        .and_then(|m| m.get(sub_primitive as usize).copied())
                        .map(SubObjectRef::Cell)
                } else {
                    None
                }
            }
            PickSubKind::Plugin(name) => self.resolve_plugin_sub_object(
                name,
                object_id,
                sub_primitive,
                mask,
                primitive_index_supported,
                world_pos,
            ),
        }
    }

    /// Resolve a surface / volume-mesh-boundary hit to a face, cell, or vertex.
    ///
    /// Needs `SHADER_PRIMITIVE_INDEX`: the primitive channel names the hit face
    /// (or, when the VERTEX variant drew, the final nearest-corner vertex index).
    /// Without the feature the GPU path stays object-level. `FACE` is the channel
    /// value; `CELL` maps it through the boundary face-to-cell map in
    /// `surface_meta`; `VERTEX` is the
    /// channel value written by the vertex pipeline variant. No CPU geometry.
    fn resolve_surface_sub_object(
        &self,
        object_id: u64,
        sub_primitive: u32,
        mask: PickMask,
        surface_meta: &SurfacePickMeta,
        primitive_index_supported: bool,
    ) -> Option<SubObjectRef> {
        let wants_face = mask.intersects(PickMask::FACE);
        let wants_cell = mask.intersects(PickMask::CELL);
        let wants_vertex = mask.intersects(PickMask::VERTEX);
        let wants_edge = mask.intersects(PickMask::EDGE);
        if !(wants_face || wants_cell || wants_vertex || wants_edge) {
            return None;
        }
        if !primitive_index_supported {
            return None;
        }

        let meta = surface_meta.get(&object_id);
        let has_f2c = meta.is_some_and(|m| !m.is_empty());

        // The VERTEX variant already wrote the final nearest-corner vertex index.
        if surface_writes_vertex(mask, has_f2c, primitive_index_supported) {
            return Some(SubObjectRef::Vertex(sub_primitive));
        }
        // The EDGE variant wrote the nearest edge id (`face * 3 + local_edge`).
        if surface_writes_edge(mask, has_f2c, primitive_index_supported) {
            return Some(SubObjectRef::Edge(sub_primitive));
        }

        // Otherwise the channel is the hit face. FACE takes priority, then CELL
        // through the persistent boundary map.
        if wants_face {
            return Some(SubObjectRef::Face(sub_primitive));
        }
        if wants_cell {
            return meta
                .filter(|m| !m.is_empty())
                .and_then(|m| m.get(sub_primitive as usize).copied())
                .map(SubObjectRef::Cell);
        }
        None
    }

    /// Build the `pick_id -> PickSubKind` decode map from the same collections
    /// the pass draws. Ids only, so no geometry is retained here.
    fn build_pick_sub_kinds(
        &self,
        frame: &FrameData,
        scene_items: &[SceneRenderItem],
    ) -> std::collections::HashMap<u64, PickSubKind> {
        let mut kinds: std::collections::HashMap<u64, PickSubKind> =
            std::collections::HashMap::new();

        // Registered plugin items. Inserted first so a pick-id collision with a
        // built-in item resolves to the built-in kind (ids are consumer-assigned
        // and expected unique; this just makes the overlap deterministic).
        for (&name, _) in self.item_type_plugins.iter() {
            let Some(items) = frame.scene.plugin_items.get(name) else {
                continue;
            };
            for i in 0..items.len() {
                let settings = items.item_settings(i);
                if !settings.hidden && settings.pick_id != PickId::NONE {
                    kinds.insert(settings.pick_id.0, PickSubKind::Plugin(name));
                }
            }
        }

        // Surfaces and opaque/transparent volume-mesh boundaries.
        for item in scene_items
            .iter()
            .filter(|i| !i.settings.hidden && i.settings.pick_id != PickId::NONE)
        {
            kinds.insert(item.settings.pick_id.0, PickSubKind::Surface);
        }
        for vm in frame.scene.volume_meshes.iter() {
            let ri = vm.to_render_item();
            if !ri.settings.hidden && ri.settings.pick_id != PickId::NONE {
                kinds.insert(ri.settings.pick_id.0, PickSubKind::Surface);
            }
        }

        // Curve families.
        for family in [
            self.streamtube_gpu_data.as_slice(),
            self.tube_gpu_data.as_slice(),
            self.ribbon_gpu_data.as_slice(),
        ] {
            for gpu in family
                .iter()
                .filter(|g| g.pick_id != PickId::NONE && g.index_count > 0)
            {
                kinds.insert(gpu.pick_id.0, PickSubKind::Curve);
            }
        }

        // Instanced families.
        for gpu in self
            .glyph_gpu_data
            .iter()
            .filter(|g| g.pick_id != PickId::NONE && g.instance_count > 0)
        {
            kinds.insert(gpu.pick_id.0, PickSubKind::Instance);
        }
        for gpu in self
            .tensor_glyph_gpu_data
            .iter()
            .filter(|g| g.pick_id != PickId::NONE && g.instance_count > 0)
        {
            kinds.insert(gpu.pick_id.0, PickSubKind::Instance);
        }
        for gpu in self
            .sprite_gpu_data
            .iter()
            .filter(|g| g.pick_id != PickId::NONE && g.sprite_count > 0)
        {
            kinds.insert(gpu.pick_id.0, PickSubKind::Instance);
        }

        // Polylines.
        for gpu in self
            .polyline_gpu_data
            .iter()
            .filter(|g| g.pick_id != PickId::NONE && g.segment_count > 0)
        {
            kinds.insert(gpu.pick_id.0, PickSubKind::Polyline);
        }

        // Point clouds.
        for gpu in self
            .point_cloud_gpu_data
            .iter()
            .filter(|g| g.pick_id != PickId::NONE && g.point_count > 0)
        {
            kinds.insert(gpu.pick_id.0, PickSubKind::CloudPoint);
        }

        // Gaussian splat sets.
        for dd in self
            .gaussian_splat_draw_data
            .iter()
            .filter(|dd| !dd.wireframe && dd.pick_id != PickId::NONE && dd.count > 0)
        {
            kinds.insert(dd.pick_id.0, PickSubKind::Splat);
        }

        // Ray-marched volumes: the pick shader writes the hit voxel's flat index
        // into the primitive channel, decoded to SubObjectRef::Voxel.
        for gpu in self
            .volume_gpu_data
            .iter()
            .filter(|v| !v.wireframe && v.pick_id != PickId::NONE)
        {
            kinds.insert(gpu.pick_id.0, PickSubKind::Voxel);
        }

        kinds
    }
}

/// Outcome of encoding a point pick pass. `Miss` short-circuits before any GPU
/// work when there is nothing to draw; `Pending` carries the submitted staging
/// buffers for the caller to read back.
enum PickBegin {
    Miss,
    Pending(PendingPick),
}

/// An in-flight GPU object pick: the submitted staging buffers plus the cursor
/// state needed to turn the read-back pixel into a `PickHit`. Held on the
/// renderer between [`ViewportRenderer::pick_object_begin`] and
/// [`ViewportRenderer::pick_object_poll`].
pub(crate) struct PendingPick {
    id_staging: crate::gpu::Buffer,
    prim_staging: crate::gpu::Buffer,
    depth_staging: crate::gpu::Buffer,
    /// Reaches 3 when all three `map_async` callbacks have signalled success
    /// (object id, primitive id, depth).
    ready: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Submission index of the id pass, so a reader can wait on this pick alone
    /// rather than draining the whole queue.
    submission: crate::gpu::SubmissionIndex,
    cursor: glam::Vec2,
    viewport_size: glam::Vec2,
    view_proj: glam::Mat4,
    /// The mask the pick was submitted with. Decides which sub-object level the
    /// read-back primitive index resolves to (face vs cell vs vertex, etc.).
    mask: PickMask,
    /// Per-object decode of the primitive channel, keyed by `pick_id`.
    kinds: std::collections::HashMap<u64, PickSubKind>,
    /// Cell / vertex refinement data for surface and volume-mesh hits, keyed by
    /// `pick_id`. Copied from the frame at submit time so the resolve step needs
    /// no per-frame pick cache.
    surface_meta: SurfacePickMeta,
    /// Whether the device had `SHADER_PRIMITIVE_INDEX` when the pass ran. When
    /// false the surface / curve pipelines wrote a constant 0 into the primitive
    /// channel, so those types stay object-level (the instance / segment channels
    /// do not depend on the feature).
    primitive_index_supported: bool,
}

impl PendingPick {
    /// Read the object id and depth from the mapped staging buffers and unmap
    /// them. Only valid once both maps have completed (the caller has waited or
    /// seen `ready == 3`). Returns `None` for id 0, which is the clear value and
    /// so means empty space or a non-pickable surface.
    fn read_hit(&self) -> Option<GpuPickHit> {
        let object_id = {
            let data = self.id_staging.slice(..).get_mapped_range();
            u32::from_le_bytes([data[0], data[1], data[2], data[3]])
        };
        self.id_staging.unmap();

        let sub_primitive = {
            let data = self.prim_staging.slice(..).get_mapped_range();
            u32::from_le_bytes([data[0], data[1], data[2], data[3]])
        };
        self.prim_staging.unmap();

        let depth = {
            let data = self.depth_staging.slice(..).get_mapped_range();
            f32::from_le_bytes([data[0], data[1], data[2], data[3]])
        };
        self.depth_staging.unmap();

        if object_id == 0 {
            return None;
        }
        Some(GpuPickHit {
            object_id: PickId(object_id as u64),
            depth,
            sub_primitive,
        })
    }
}

/// A unit cube spanning `[-0.5, 0.5]^3` as `MeshData`, used as the decal pick
/// proxy. The pick pass reads only vertex positions; the per-face normals exist
/// solely to satisfy mesh-upload validation. Indices wind all six faces.
fn unit_cube_mesh_data() -> crate::resources::MeshData {
    let positions = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let normals = vec![[0.0, 0.0, 1.0]; 8];
    let indices = vec![
        0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 3, 7, 7, 4, 0, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7, 3,
        0, 4, 5, 5, 1, 0,
    ];
    let mut mesh = crate::resources::MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}
