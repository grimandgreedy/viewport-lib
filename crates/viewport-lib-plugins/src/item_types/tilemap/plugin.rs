//! The tilemap plugin: its store, its per-frame layers and its draw hooks.

use super::grid::GridLayout;
use super::store::{
    ChunkKey, LAYER_STRIDE, StoredTilemap, TilemapLayouts, TilemapStore, chunk_bounds,
    layer_uniform, resolve_bindings, unpack_cell,
};
use super::*;
use viewport_lib::plugin_api::{
    ItemCollections, ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext,
    PainterBatch, PainterContext, PerViewport, PickContext, PickPassContext, PickRay, PluginItem,
    RectPickContext,
};
use viewport_lib::renderer::{PickHit, PickId, PickMask, PickRectResult, SubObjectRef};

/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::item_types::shader::{scene_shader, wgsl_source};
    vec![
        (
            "tilemap.wgsl",
            scene_shader(
                &[
                    viewport_lib::plugin_api::shared_wgsl::SHARED_PAINTER_MASK_WGSL,
                    super::TILEMAP_COMMON,
                ],
                wgsl_source!("tilemap"),
            ),
        ),
        (
            "tilemap_pick.wgsl",
            scene_shader(&[super::TILEMAP_COMMON], wgsl_source!("tilemap_pick")),
        ),
    ]
}

impl PluginItem for TilemapItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

/// One layer drawn this frame.
struct FrameLayer {
    item: TilemapItem,
    /// The tile material index whose pipelines it draws through.
    family: Option<usize>,
    /// Dynamic offset of the layer's uniform.
    offset: u32,
    /// Chunks that draw: occupied and, after the cull, in view.
    visible: Vec<ChunkKey>,
    /// Drawn as the outlines of its cells instead of its tiles, in
    /// wireframe mode or with `settings.wireframe`. Still picks.
    wireframe: bool,
}

impl FrameLayer {
    fn painter(&self) -> Option<(u32, bool, TilemapPainter)> {
        match self.item.draw {
            TilemapDraw::Blended {
                layer,
                depth_test,
                painter,
            } => Some((layer, depth_test, painter)),
            TilemapDraw::Cutout { .. } => None,
        }
    }
}

/// One painter batch: a blended layer whole, or one row run of one chunk of
/// a layer sorted by row.
struct Piece {
    /// Index into the view's `frame`.
    layer: usize,
    /// For a row: its chunk, first instance and instance count.
    run: Option<(ChunkKey, u32, u32)>,
    painter_layer: u32,
    key: f32,
}

/// One viewport's layers and their uniforms, rebuilt by each prepare for
/// that viewport and narrowed by its cull.
#[derive(Default)]
struct TilemapView {
    frame: Vec<FrameLayer>,
    /// The blended layers' batches; a batch handle is an index into it.
    pieces: Vec<Piece>,
    /// Indices into `pieces`, in the order they draw.
    painter_sequence: Vec<usize>,
    layer_buf: Option<viewport_lib::gpu::Buffer>,
    layer_bg: Option<viewport_lib::gpu::BindGroup>,
    /// The camera position, for the CPU pick.
    eye: glam::DVec3,
}

impl TilemapView {
    fn cutout_layers(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.frame.len()).filter(|&i| self.frame[i].painter().is_none())
    }

    /// The cutout layers drawn as tiles rather than outlines.
    fn filled_cutout_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.cutout_layers().filter(|&i| !self.frame[i].wireframe)
    }

    /// Draw calls this view issues in the colour passes.
    fn draws(&self) -> u32 {
        let cutout: usize = self
            .filled_cutout_layers()
            .map(|i| self.frame[i].visible.len())
            .sum();
        let painter: usize = self
            .pieces
            .iter()
            .map(|p| match p.run {
                Some(_) => 1,
                None => self.frame[p.layer].visible.len(),
            })
            .sum();
        (cutout + painter) as u32
    }

    /// Rebuild the painter batches from the visible chunks.
    fn build_pieces(&mut self, maps: &TilemapStore) {
        self.pieces.clear();
        for (i, layer) in self.frame.iter().enumerate() {
            let Some((painter_layer, _, painter)) = layer.painter() else {
                continue;
            };
            if layer.wireframe {
                continue;
            }
            match painter {
                TilemapPainter::Layer { key } => self.pieces.push(Piece {
                    layer: i,
                    run: None,
                    painter_layer,
                    key,
                }),
                TilemapPainter::PerRow { offset } => {
                    let Some(stored) = maps.get(layer.item.source) else {
                        continue;
                    };
                    let item = &layer.item;
                    let base = item.drawn_origin(item.plane.to_plane(self.eye)).y;
                    let cell = item.cell_size[1] as f64;
                    for &key in &layer.visible {
                        let Some(chunk) = stored.chunks.get(&key) else {
                            continue;
                        };
                        for &(edge, first, count) in &chunk.rows {
                            let y = base + edge * cell;
                            self.pieces.push(Piece {
                                layer: i,
                                run: Some((key, first, count)),
                                painter_layer,
                                key: (offset as f64 - y) as f32,
                            });
                        }
                    }
                }
            }
        }
        let mut order: Vec<usize> = (0..self.pieces.len()).collect();
        order.sort_by(|&a, &b| {
            let (a, b) = (&self.pieces[a], &self.pieces[b]);
            a.painter_layer
                .cmp(&b.painter_layer)
                .then(a.key.total_cmp(&b.key))
        });
        self.painter_sequence = order;
    }
}

/// Draws [`TilemapItem`]s. Register it with
/// `renderer.with_item_type_plugin(device, Box::new(TilemapPlugin::default()))`
/// or through `item_types::install`.
#[derive(Default)]
pub struct TilemapPlugin {
    maps: TilemapStore,
    deps_gate: viewport_lib::resources::ResourceGate,
    layouts: Option<TilemapLayouts>,
    gpu: Option<pipeline::TilemapGpu>,
    views: PerViewport<TilemapView>,
    /// The viewport prepared or culled last, for the queries that name none.
    last_viewport: usize,
    /// Draw calls in the colour passes of the last prepared or culled view.
    drawn_chunks: u32,
    upload_bytes: u64,
    /// Registered tile materials, by [`TileMaterialId`] index.
    materials: Vec<crate::item_types::sprite::TileMaterialEntry>,
    /// Registered sprite deformers, by bit.
    deformers: Vec<crate::item_types::sprite::SpriteDeformerEntry>,
}

impl TilemapPlugin {
    /// Register a tile material: validate its vertex hook against the
    /// tilemap shaders and keep it for the layers that name it.
    pub(crate) fn register_material(
        &mut self,
        device: &viewport_lib::gpu::Device,
        plugin: &dyn viewport_lib::MaterialPlugin,
    ) -> viewport_lib::error::ViewportResult<TileMaterialId> {
        let (colour, pick) = pipeline::tile_sources();
        let entry = crate::item_types::sprite::validate_tile_material(
            device,
            &self.materials,
            &self.deformers,
            plugin,
            &[("colour", colour), ("pick", pick)],
        )?;
        self.materials.push(entry);
        Ok(TileMaterialId(self.materials.len() as u32 - 1))
    }

    /// Check that `entry`, after the deformers already registered, composes
    /// into every tilemap module.
    pub(crate) fn check_deformer(
        &self,
        device: &viewport_lib::gpu::Device,
        entry: &crate::item_types::sprite::SpriteDeformerEntry,
    ) -> viewport_lib::error::ViewportResult<()> {
        let mut trial = self.deformers.clone();
        trial.push(entry.clone());
        let (colour, pick) = pipeline::tile_sources();
        let deformed = |s: Result<String, String>| {
            s.and_then(|s| crate::item_types::sprite::tile_deformer_module(s, &trial))
        };
        let mut modules = vec![
            ("tile_colour".to_string(), deformed(Ok(colour.clone()))),
            ("tile_pick".to_string(), deformed(Ok(pick.clone()))),
        ];
        for m in &self.materials {
            for (label, base) in [("colour", &colour), ("pick", &pick)] {
                let composed = crate::item_types::sprite::compose_tile(base, m);
                modules.push((format!("tile_{}_{label}", m.name), deformed(composed)));
            }
        }
        crate::item_types::sprite::validate_deformer_modules(device, &entry.name, modules)
    }

    /// Keep `entry`, checked with [`check_deformer`](Self::check_deformer).
    /// The pipelines are rebuilt with it at the next prepare.
    pub(crate) fn add_deformer(&mut self, entry: crate::item_types::sprite::SpriteDeformerEntry) {
        self.deformers.push(entry);
        self.gpu = None;
    }

    /// The tile material registered as `name`.
    pub(crate) fn material_id(&self, name: &str) -> Option<TileMaterialId> {
        self.materials
            .iter()
            .position(|m| m.name == name)
            .map(|i| TileMaterialId(i as u32))
    }

    fn layouts(&mut self, device: &viewport_lib::gpu::Device) -> &TilemapLayouts {
        self.layouts
            .get_or_insert_with(|| TilemapLayouts::new(device))
    }

    /// Upload `map` and return its handle.
    pub(crate) fn upload(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        map: &Tilemap,
    ) -> TilemapId {
        let stored = self.build(device, queue, resources, map.clone());
        self.maps.insert_sized(stored)
    }

    fn build(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        map: Tilemap,
    ) -> StoredTilemap {
        let layouts = self.layouts(device);
        let binds = resolve_bindings(
            device,
            resources,
            layouts,
            map.tileset.texture_id,
            map.tileset.normal_atlas,
            map.tileset.filter,
            true,
        );
        StoredTilemap::build(device, queue, &binds, map)
    }

    /// Build a map's buffers on a worker thread.
    pub(crate) fn begin_upload(
        &mut self,
        jobs: &viewport_lib::resources::Jobs<'_>,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        map: Tilemap,
    ) -> viewport_lib::resources::JobId {
        let layouts = self.layouts(device);
        let binds = resolve_bindings(
            device,
            resources,
            layouts,
            map.tileset.texture_id,
            map.tileset.normal_atlas,
            map.tileset.filter,
            true,
        );
        let device = device.clone();
        let queue = queue.clone();
        jobs.submit_cpu(move || StoredTilemap::build(&device, &queue, &binds, map))
    }

    /// Store the map a finished job built.
    pub(crate) fn take_upload_result(
        &mut self,
        jobs: &viewport_lib::resources::Jobs<'_>,
        id: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<TilemapId> {
        match jobs.status(id) {
            viewport_lib::resources::UploadStatus::Pending { .. } => {
                Err(viewport_lib::error::ViewportError::JobNotReady)
            }
            viewport_lib::resources::UploadStatus::Failed(e) => Err(e),
            _ => jobs
                .take::<StoredTilemap>(id)
                .map(|stored| self.maps.insert_sized(stored))
                .ok_or(viewport_lib::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                }),
        }
    }

    /// Replace the map behind a live handle.
    pub(crate) fn replace(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        id: TilemapId,
        map: &Tilemap,
    ) -> viewport_lib::error::ViewportResult<()> {
        if !self.maps.contains(id) {
            return Err(self.maps.stale(id));
        }
        let stored = self.build(device, queue, resources, map.clone());
        self.maps.replace_sized(id, stored);
        Ok(())
    }

    /// Drop a stored map. `false` when the handle does not resolve.
    pub(crate) fn drop_stored(&mut self, id: TilemapId) -> bool {
        self.maps.remove(id).is_some()
    }

    /// Put each `(col, row, tile)` into the map and re-upload every chunk
    /// that changed, once. `Ok(false)` when any cell was outside the cell
    /// range, whose cells are skipped, or the map keeps no CPU copy, which
    /// writes nothing.
    pub fn set_tiles(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        id: TilemapId,
        tiles: impl IntoIterator<Item = (i32, i32, Tile)>,
    ) -> viewport_lib::error::ViewportResult<bool> {
        let Some(stored) = self.maps.get_mut(id) else {
            return Err(self.maps.stale(id));
        };
        if !stored.map.keep_cpu_copy {
            return Ok(false);
        }
        let mut all_written = true;
        let mut touched: Vec<[i32; 2]> = Vec::new();
        for (col, row, tile) in tiles {
            match stored.map.set(col, row, tile) {
                Some(chunk) => {
                    if !touched.contains(&chunk) {
                        touched.push(chunk);
                    }
                }
                None => all_written = false,
            }
        }
        for chunk in touched {
            stored.upload_chunk(device, queue, chunk);
        }
        self.maps.recharge(id);
        Ok(all_written)
    }

    /// Replace the whole chunk at `coord` and re-upload it, creating it when
    /// the map has none there. `Ok(false)` when the chunk lies outside the
    /// cell range.
    pub fn set_chunk(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        id: TilemapId,
        coord: [i32; 2],
        chunk: TileChunk,
    ) -> viewport_lib::error::ViewportResult<bool> {
        let Some(stored) = self.maps.get_mut(id) else {
            return Err(self.maps.stale(id));
        };
        let written = stored.replace_chunk(device, queue, coord, chunk);
        self.maps.recharge(id);
        Ok(written)
    }

    /// Drop the chunk at `coord` and free its buffer. `Ok(false)` when the
    /// map had no chunk there.
    pub fn remove_chunk(
        &mut self,
        id: TilemapId,
        coord: [i32; 2],
    ) -> viewport_lib::error::ViewportResult<bool> {
        let Some(stored) = self.maps.get_mut(id) else {
            return Err(self.maps.stale(id));
        };
        let removed = stored.remove_chunk(coord);
        self.maps.recharge(id);
        Ok(removed)
    }

    /// The CPU copy of an uploaded map.
    pub fn tilemap(&self, id: TilemapId) -> Option<&Tilemap> {
        self.maps.get(id).map(|s| &s.map)
    }

    /// Chunk uploads over the life of an uploaded map: one per occupied chunk
    /// at upload, then one per chunk an edit touched.
    pub fn chunk_uploads(&self, id: TilemapId) -> Option<u64> {
        self.maps.get(id).map(|s| s.chunk_uploads)
    }

    /// Bytes the chunk instance buffers of an uploaded map hold: sixteen
    /// per occupied cell, give or take room left by edits that emptied
    /// cells.
    pub fn chunk_buffer_bytes(&self, id: TilemapId) -> Option<u64> {
        self.maps.get(id).map(|s| s.chunk_buffer_bytes())
    }

    /// Draw calls in the colour passes of the last prepared frame, after
    /// the cull: one per drawn chunk, and one per drawn row of a layer
    /// sorted by row.
    pub fn drawn_chunks(&self) -> u32 {
        self.drawn_chunks
    }

    /// Rebind stored maps whose atlas or normal atlas was freed or swapped
    /// since the last frame. The map stays; a freed atlas leaves it drawing
    /// untextured, a freed normal atlas lit flat.
    fn revalidate_stores(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
    ) {
        use viewport_lib::resources::Revalidate;
        let action = self.deps_gate.poll(resources);
        if action == Revalidate::Valid {
            return;
        }
        let Some(layouts) = self.layouts.as_ref() else {
            return;
        };
        for (_, stored) in self.maps.iter_mut() {
            let set = &stored.map.tileset;
            let (texture, normals) = (set.texture_id, set.normal_atlas);
            if texture.is_none() && normals.is_none() {
                continue;
            }
            if action == Revalidate::CheckEach
                && [texture, normals]
                    .into_iter()
                    .flatten()
                    .all(|id| resources.has_texture(id))
            {
                continue;
            }
            let binds = resolve_bindings(
                device, resources, layouts, texture, normals, set.filter, false,
            );
            stored.rebind(device, queue, &binds);
            let set = &mut stored.map.tileset;
            if texture.is_some_and(|id| !resources.has_texture(id)) {
                set.texture_id = None;
            }
            if normals.is_some_and(|id| !resources.has_texture(id)) {
                set.normal_atlas = None;
            }
        }
    }

    /// Draw `layers` (indices into the view's `frame`), each whole, with
    /// `member`.
    fn draw_layers(
        &self,
        view: &TilemapView,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        layers: impl Iterator<Item = usize>,
        member: impl Fn(&FrameLayer) -> usize,
    ) {
        self.draw_runs(view, pass, layers.map(|i| (i, None)), member);
    }

    /// Draw each `(layer, run)` with `member`, binding per layer: the
    /// layer's visible chunks whole, or for a run its rows of one chunk.
    fn draw_runs(
        &self,
        view: &TilemapView,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        runs: impl Iterator<Item = (usize, Option<(ChunkKey, u32, u32)>)>,
        member: impl Fn(&FrameLayer) -> usize,
    ) {
        let (Some(gpu), Some(layer_bg)) = (&self.gpu, &view.layer_bg) else {
            return;
        };
        let mut bound = None;
        let mut bound_layer = None;
        for (i, run) in runs {
            let Some(layer) = view.frame.get(i) else {
                continue;
            };
            let Some(stored) = self.maps.get(layer.item.source) else {
                continue;
            };
            let m = member(layer);
            if bound != Some((layer.family, m)) {
                // Still compiling: this layer draws next frame.
                let Some(pl) = gpu.set(layer.family).and_then(|set| set.get(m)) else {
                    continue;
                };
                pass.set_pipeline(pl);
                bound = Some((layer.family, m));
            }
            if bound_layer != Some(i) {
                pass.set_bind_group(1, &stored.bind_group, &[]);
                pass.set_bind_group(2, layer_bg, &[layer.offset]);
                bound_layer = Some(i);
            }
            let stride = super::store::INSTANCE_STRIDE;
            match run {
                None => {
                    for key in &layer.visible {
                        let Some(buffer) = stored
                            .chunks
                            .get(key)
                            .and_then(|c| c.buffer.as_ref().map(|b| (b, c.count)))
                        else {
                            continue;
                        };
                        pass.set_vertex_buffer(0, buffer.0.slice(..buffer.1 as u64 * stride));
                        pass.draw(0..6, 0..buffer.1);
                    }
                }
                Some((key, first, count)) => {
                    let Some(buffer) = stored.chunks.get(&key).and_then(|c| c.buffer.as_ref())
                    else {
                        continue;
                    };
                    let start = first as u64 * stride;
                    pass.set_vertex_buffer(0, buffer.slice(start..start + count as u64 * stride));
                    pass.draw(0..6, 0..count);
                }
            }
        }
    }

    /// The `(layer, run)` pairs of painter batches `handles`.
    fn piece_runs<'a>(
        view: &'a TilemapView,
        handles: impl Iterator<Item = usize> + 'a,
    ) -> impl Iterator<Item = (usize, Option<(ChunkKey, u32, u32)>)> + 'a {
        handles
            .filter_map(|h| view.pieces.get(h))
            .map(|p| (p.layer, p.run))
    }

    /// World points of `item` drawn for a camera at `eye` at plane offsets
    /// `points`, in cells.
    fn to_world(
        item: &TilemapItem,
        eye: glam::DVec3,
        points: impl IntoIterator<Item = glam::DVec2>,
    ) -> impl Iterator<Item = glam::Vec3> {
        let plane = item.plane;
        let o = item.drawn_origin(plane.to_plane(eye));
        let cell = glam::DVec2::new(item.cell_size[0] as f64, item.cell_size[1] as f64);
        let e = item.elevation as f64;
        points
            .into_iter()
            .map(move |p| plane.to_world(o + p * cell, e).as_vec3())
    }

    /// The outline of cell (`col`, `row`) of `item` laid out by `layout`, as
    /// drawn for a camera at `eye`, counter-clockwise.
    fn cell_outline(
        item: &TilemapItem,
        layout: &GridLayout,
        eye: glam::DVec3,
        col: i32,
        row: i32,
    ) -> Vec<glam::Vec3> {
        Self::to_world(item, eye, layout.outline(col, row)).collect()
    }
}

impl ItemTypePlugin for TilemapPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn requires(&self) -> viewport_lib::Requirements {
        viewport_lib::Requirements::compatibility_tier()
    }

    fn init_gpu(
        &mut self,
        device: &viewport_lib::gpu::Device,
        _resources: &dyn viewport_lib::plugin_api::PluginResources,
    ) {
        self.layouts = Some(TilemapLayouts::new(device));
    }

    fn warm(
        &mut self,
        device: &viewport_lib::gpu::Device,
        resources: &viewport_lib::DeviceResources,
    ) {
        let layouts = self
            .layouts
            .get_or_insert_with(|| TilemapLayouts::new(device));
        let gpu = self.gpu.get_or_insert_with(|| {
            pipeline::TilemapGpu::new(device, resources, layouts, &self.deformers)
        });
        gpu.pipelines.request_all();
    }

    fn resident_bytes(&self) -> u64 {
        self.maps.allocated_bytes()
    }

    fn draw_calls(&self) -> u32 {
        self.drawn_chunks
    }

    fn upload_bytes(&self) -> u64 {
        self.upload_bytes
    }

    fn prepare(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        _items: &ItemCollections<'_>,
    ) -> Vec<viewport_lib::gpu::CommandBuffer> {
        self.upload_bytes = 0;
        self.revalidate_stores(device, queue, ctx.resources);
        self.last_viewport = ctx.viewport_index;
        let view = self.views.get_mut(ctx.viewport_index);
        view.frame.clear();
        view.painter_sequence.clear();
        view.eye = ctx.camera.eye_position();
        for (_, stored) in self.maps.iter_mut() {
            self.upload_bytes += std::mem::take(&mut stored.uploaded_bytes);
        }

        let origin = ctx.world_origin();
        let mut uniforms: Vec<u8> = Vec::new();
        for item in ctx.items_of::<TilemapItem>() {
            if item.settings.hidden {
                continue;
            }
            let Some(stored) = self.maps.get(item.source) else {
                continue;
            };
            let offset = uniforms.len() as u32;
            let mut data = layer_uniform(item, stored, origin, ctx.time.seconds);
            let family = item
                .material
                .as_ref()
                .and_then(|m| self.materials.get(m.id.0 as usize).map(|e| (m, e)));
            if let Some((m, entry)) = family {
                data.lanes = crate::item_types::sprite::material_lanes(&entry.params, &m.params);
            }
            let family = family.map(|(m, _)| m.id.0 as usize);
            uniforms.extend_from_slice(bytemuck::bytes_of(&data));
            uniforms.resize(offset as usize + LAYER_STRIDE as usize, 0);
            let visible = stored.chunks.keys().copied().collect();
            view.frame.push(FrameLayer {
                item: item.clone(),
                family,
                offset,
                visible,
                wireframe: ctx.wireframe_mode || item.settings.wireframe,
            });
        }
        if view.frame.is_empty() {
            self.drawn_chunks = 0;
            return Vec::new();
        }

        let layouts = self
            .layouts
            .get_or_insert_with(|| TilemapLayouts::new(device));
        self.gpu
            .get_or_insert_with(|| {
                pipeline::TilemapGpu::new(device, ctx.resources, layouts, &self.deformers)
            })
            .ensure_families(device, ctx.resources, layouts, &self.materials);
        let needed = uniforms.len() as u64;
        if view.layer_buf.as_ref().is_none_or(|b| b.size() < needed) {
            let size = needed.next_power_of_two().max(LAYER_STRIDE * 4);
            let buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
                label: Some("tilemap_layer_buf"),
                size,
                usage: viewport_lib::gpu::BufferUsages::UNIFORM
                    | viewport_lib::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            view.layer_bg = Some(device.create_bind_group(
                &viewport_lib::gpu::BindGroupDescriptor {
                    label: Some("tilemap_layer_bg"),
                    layout: &layouts.layer,
                    entries: &[viewport_lib::gpu::BindGroupEntry {
                        binding: 0,
                        resource: viewport_lib::gpu::BindingResource::Buffer(
                            viewport_lib::gpu::BufferBinding {
                                buffer: &buf,
                                offset: 0,
                                size: std::num::NonZeroU64::new(std::mem::size_of::<
                                    super::store::LayerUniformData,
                                >()
                                    as u64),
                            },
                        ),
                    }],
                },
            ));
            view.layer_buf = Some(buf);
        }
        if let Some(buf) = &view.layer_buf {
            queue.write_buffer(buf, 0, &uniforms);
        }
        self.upload_bytes += needed;

        view.build_pieces(&self.maps);
        self.drawn_chunks = view.draws();
        Vec::new()
    }

    fn cull(
        &mut self,
        frustum: &viewport_lib::Frustum,
        ctx: &ItemFrameContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let eye = ctx.camera.eye_position();
        self.last_viewport = ctx.viewport_index;
        let view = self.views.get_mut(ctx.viewport_index);
        for layer in &mut view.frame {
            let Some(stored) = self.maps.get(layer.item.source) else {
                layer.visible.clear();
                continue;
            };
            layer.visible = stored
                .chunks
                .keys()
                .copied()
                .filter(|&c| {
                    let (min, max) = chunk_bounds(&layer.item, stored, c, eye);
                    !frustum.cull_aabb(&viewport_lib::Aabb { min, max })
                })
                .collect();
        }
        view.build_pieces(&self.maps);
        self.drawn_chunks = view.draws();
    }

    fn paint(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let Some(view) = self.views.get(ctx.viewport_index) else {
            return;
        };
        let member = if ctx.is_hdr_pass() {
            pipeline::CUTOUT_SCENE
        } else {
            pipeline::CUTOUT_LDR
        };
        // The light layers at group 3, which cutout layers sample as painter
        // layer 0.
        pass.set_bind_group(3, ctx.mask_bind_group, &[]);
        self.draw_layers(view, pass, view.filled_cutout_layers(), |_| member);
    }

    fn draws_ldr(&self) -> bool {
        true
    }

    fn draws_painter(&self, viewport_index: usize) -> bool {
        self.views
            .get(viewport_index)
            .is_some_and(|v| !v.painter_sequence.is_empty())
    }

    /// One batch per blended layer at its key, or one per drawn row of each
    /// visible chunk for a layer sorted by row; the handle indexes the view's
    /// pieces.
    fn painter_batches(&self, viewport_index: usize, out: &mut Vec<PainterBatch>) {
        let Some(view) = self.views.get(viewport_index) else {
            return;
        };
        for &i in &view.painter_sequence {
            let piece = &view.pieces[i];
            out.push(PainterBatch::new(piece.painter_layer, piece.key, i as u32));
        }
    }

    fn paint_painter(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PainterContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let Some(view) = self.views.get(ctx.viewport_index) else {
            return;
        };
        let handles: Box<dyn Iterator<Item = usize> + '_> = match ctx.batches {
            Some(batches) => Box::new(batches.iter().map(|&b| b as usize)),
            None => Box::new(
                view.painter_sequence
                    .iter()
                    .copied()
                    .filter(|&i| ctx.includes(view.pieces[i].painter_layer)),
            ),
        };
        let runs = Self::piece_runs(view, handles);

        pass.set_bind_group(3, ctx.mask_bind_group, &[]);
        if ctx.is_hdr_pass() {
            self.draw_runs(view, pass, runs, |l| match l.painter() {
                Some((_, false, _)) => pipeline::BLEND_OFF_DEPTH_READ,
                _ => pipeline::BLEND_TEST_DEPTH_READ,
            });
        } else {
            self.draw_runs(view, pass, runs, |l| match l.painter() {
                Some((_, false, _)) => pipeline::BLEND_OFF_LDR,
                _ => pipeline::BLEND_TEST_LDR,
            });
        }
    }

    fn render_pick(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PickPassContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        if !ctx.mask.intersects(PickMask::OBJECT | PickMask::TILE) {
            return;
        }
        let Some(view) = self.views.get(ctx.viewport_index) else {
            return;
        };
        // Cutout layers here; the blended ones pick in the painter sequence
        // through `render_pick_painter`.
        let pickable = |i: &usize| view.frame[*i].item.settings.pick_id != PickId::NONE;
        let order = view.cutout_layers().filter(pickable).collect::<Vec<_>>();
        self.draw_layers(view, pass, order.into_iter(), |_| pipeline::PICK);
    }

    fn render_pick_painter(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PickPassContext<'_>,
        run: &viewport_lib::plugin_api::PainterRun<'_>,
        _items: &ItemCollections<'_>,
    ) {
        if !ctx.mask.intersects(PickMask::OBJECT | PickMask::TILE) {
            return;
        }
        let Some(view) = self.views.get(ctx.viewport_index) else {
            return;
        };
        let pickable = |i: &usize| {
            view.pieces
                .get(*i)
                .is_some_and(|p| view.frame[p.layer].item.settings.pick_id != PickId::NONE)
        };
        let handles: Vec<usize> = match run.batches {
            Some(batches) => batches
                .iter()
                .map(|&b| b as usize)
                .filter(pickable)
                .collect(),
            None => view
                .painter_sequence
                .iter()
                .copied()
                .filter(|&i| run.includes(view.pieces[i].painter_layer))
                .filter(pickable)
                .collect(),
        };
        let runs = Self::piece_runs(view, handles.into_iter());
        self.draw_runs(view, pass, runs, |l| match l.painter() {
            Some((_, false, _)) => pipeline::PICK_PAINTER_NO_TEST,
            _ => pipeline::PICK_PAINTER,
        });
    }

    fn outline_mask(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &OutlineMaskContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let Some(view) = self.views.get(ctx.viewport_index) else {
            return;
        };
        // A selected layer's drawn tiles go into the mask, so the outline
        // traces the occupied cells rather than the map's bounds.
        let selected = (0..view.frame.len())
            .filter(|&i| view.frame[i].item.settings.selected)
            .collect::<Vec<_>>();
        self.draw_layers(view, pass, selected.into_iter(), |_| pipeline::OUTLINE_MASK);
    }

    /// The outline of each occupied cell of the visible chunks, drawn in
    /// place of the tiles in wireframe mode or for a layer with
    /// `settings.wireframe`. Past a cap of cells a layer outlines each
    /// visible chunk instead.
    fn wireframe_polylines(
        &self,
        _items: &ItemCollections<'_>,
        ctx: &ItemFrameContext<'_>,
    ) -> Vec<viewport_lib::renderer::PolylineItem> {
        /// Above this many cells an outline each stops being readable.
        const MAX_OUTLINED_CELLS: usize = 4096;

        let Some(view) = self.views.get(ctx.viewport_index) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for layer in &view.frame {
            let item = &layer.item;
            if !layer.wireframe {
                continue;
            }
            let Some(stored) = self.maps.get(item.source) else {
                continue;
            };
            let cells: usize = layer
                .visible
                .iter()
                .filter_map(|k| stored.chunks.get(k))
                .map(|c| c.count as usize)
                .sum();
            let mut positions: Vec<[f32; 3]> = Vec::new();
            let mut strips: Vec<u32> = Vec::new();
            let mut ring = |c: &[glam::Vec3]| {
                positions.extend(c.iter().chain(c.first()).map(|p| p.to_array()));
                strips.push(c.len() as u32 + 1);
            };
            let layout = &stored.layout;
            for key in &layer.visible {
                let coord = [key.1, key.0];
                if cells > MAX_OUTLINED_CELLS {
                    let cells = stored.map.chunk_cells(coord);
                    let outline: Vec<_> =
                        Self::to_world(item, view.eye, layout.region_outline(cells)).collect();
                    ring(&outline);
                    continue;
                }
                let Some(chunk) = stored.chunks.get(key) else {
                    continue;
                };
                let cs = stored.map.chunk_size();
                for tile in 0..cs * cs {
                    if !chunk.holds(tile) {
                        continue;
                    }
                    if let Some([col, row]) = stored.map.cell_of(coord, tile) {
                        ring(&Self::cell_outline(item, layout, view.eye, col, row));
                    }
                }
            }
            if strips.is_empty() {
                continue;
            }
            let mut poly = viewport_lib::renderer::PolylineItem::default();
            poly.positions = positions;
            poly.strip_lengths = strips;
            poly.default_colour = viewport_lib::Colour::linear(0.75, 0.75, 0.75, 1.0);
            poly.line_width = 1.0;
            out.push(poly);
        }
        out
    }

    fn resolve_sub_object(
        &self,
        _pick_id: PickId,
        primitive_index: u32,
        _world_pos: glam::Vec3,
        mask: PickMask,
    ) -> Option<SubObjectRef> {
        // The pick fragment writes the cell, packed as the vertex stream
        // carries it.
        let [col, row] = unpack_cell(primitive_index);
        mask.intersects(PickMask::TILE)
            .then_some(SubObjectRef::Tile { col, row })
    }

    fn pick(&self, ray: &PickRay, ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
        let wants_tile = ctx.mask.intersects(PickMask::TILE);
        if !wants_tile && !ctx.mask.intersects(PickMask::OBJECT) {
            return None;
        }
        // Cast through the centre of the pixel under the cursor, the pixel
        // the GPU pick reads, so a cursor on an edge two cells share picks
        // the same cell on either backend.
        let (origin, dir) = if ctx.viewport_size.x > 0.0 && ctx.viewport_size.y > 0.0 {
            let (o, d) = viewport_lib::interaction::query::picking::screen_to_ray(
                ctx.click_pos.floor() + 0.5,
                ctx.viewport_size,
                ctx.view_proj.inverse(),
            );
            (o.as_dvec3(), d.as_dvec3())
        } else {
            (ray.origin.as_dvec3(), ray.direction.as_dvec3())
        };
        let dir = dir.normalize_or_zero();
        let view = self.views.get(self.last_viewport)?;
        let mut best: Option<(f32, PickHit)> = None;
        for layer in &view.frame {
            let item = &layer.item;
            if item.settings.pick_id == PickId::NONE {
                continue;
            }
            let Some(stored) = self.maps.get(item.source) else {
                continue;
            };
            let plane = item.plane;
            let n = plane.normal().as_dvec3();
            let denom = dir.dot(n);
            if denom.abs() < 1e-12 {
                continue;
            }
            let t = (item.elevation as f64 - origin.dot(n)) / denom;
            if t < 0.0 {
                continue;
            }
            let hit = origin + dir * t;
            let local = plane.to_plane(hit) - item.drawn_origin(plane.to_plane(view.eye));
            let local =
                local / glam::DVec2::new(item.cell_size[0] as f64, item.cell_size[1] as f64);
            if !local.is_finite() {
                continue;
            }
            let [col, row] = stored.layout.cell_at(local);
            let range = MIN_TILE_CELL as i64..=MAX_TILE_CELL as i64;
            if !range.contains(&col) || !range.contains(&row) {
                continue;
            }
            let (col, row) = (col as i32, row as i32);
            if !stored.holds(col, row) {
                continue;
            }
            // Distance along the router's ray, which the other hits use.
            let toi = (hit - ray.origin.as_dvec3())
                .dot(ray.direction.as_dvec3().normalize_or_zero())
                .max(0.0) as f32;
            // Ties go to the later layer, which drew on top.
            if best.as_ref().is_some_and(|(b, _)| *b < toi) {
                continue;
            }
            let mut pick = PickHit::object_hit(item.settings.pick_id.0, hit, plane.normal());
            if wants_tile {
                pick.sub_object = Some(SubObjectRef::Tile { col, row });
            }
            best = Some((toi, pick));
        }
        best
    }

    /// Every occupied cell whose centre projects inside the rectangle, as a
    /// [`SubObjectRef::Tile`] under [`PickMask::TILE`].
    fn pick_rect(&self, ctx: &RectPickContext<'_>) -> PickRectResult {
        use viewport_lib::plugin_api::pick_helpers::project_to_screen;
        let mut result = PickRectResult::default();
        let wants_tile = ctx.mask.intersects(PickMask::TILE);
        let wants_object = ctx.mask.intersects(PickMask::OBJECT);
        if !wants_tile && !wants_object {
            return result;
        }
        let Some(view) = self.views.get(self.last_viewport) else {
            return result;
        };
        let inside = |p: glam::Vec2| {
            p.x >= ctx.rect_min.x
                && p.x <= ctx.rect_max.x
                && p.y >= ctx.rect_min.y
                && p.y <= ctx.rect_max.y
        };
        let screen = |p: glam::Vec3| project_to_screen(p, ctx.view_proj, ctx.viewport_size);
        for layer in &view.frame {
            let item = &layer.item;
            if item.settings.pick_id == PickId::NONE {
                continue;
            }
            let Some(stored) = self.maps.get(item.source) else {
                continue;
            };
            let id = item.settings.pick_id.0;
            let mut item_hit = false;
            let cs = stored.map.chunk_size();
            for (key, chunk) in &stored.chunks {
                let coord = [key.1, key.0];
                // Skip a chunk whose projected box misses the rectangle.
                let layout = &stored.layout;
                let region = layout.region_outline(stored.map.chunk_cells(coord));
                let projected: Option<Vec<glam::Vec2>> = Self::to_world(item, view.eye, region)
                    .map(|c| screen(c))
                    .collect();
                if let Some(p) = projected {
                    let lo = p.iter().fold(glam::Vec2::MAX, |a, &b| a.min(b));
                    let hi = p.iter().fold(glam::Vec2::MIN, |a, &b| a.max(b));
                    if hi.x < ctx.rect_min.x
                        || lo.x > ctx.rect_max.x
                        || hi.y < ctx.rect_min.y
                        || lo.y > ctx.rect_max.y
                    {
                        continue;
                    }
                }
                for tile in 0..cs * cs {
                    if !chunk.holds(tile) {
                        continue;
                    }
                    let Some([col, row]) = stored.map.cell_of(coord, tile) else {
                        continue;
                    };
                    let centre = Self::to_world(item, view.eye, [layout.centre(col, row)])
                        .next()
                        .expect("one point");
                    if !screen(centre).is_some_and(inside) {
                        continue;
                    }
                    item_hit = true;
                    if wants_tile {
                        result.elements.push((id, SubObjectRef::Tile { col, row }));
                    }
                }
            }
            if wants_object && item_hit {
                result.objects.push(id);
            }
        }
        result
    }

    /// The cell's quad as drawn, for the sub-object highlight.
    fn sub_object_outline(
        &self,
        items: &ItemCollections<'_>,
        pick_id: PickId,
        sub_object: SubObjectRef,
        out: &mut Vec<glam::Vec3>,
    ) {
        let SubObjectRef::Tile { col, row } = sub_object else {
            return;
        };
        let Some(item) = items
            .of::<TilemapItem>()
            .iter()
            .find(|i| !i.settings.hidden && i.settings.pick_id == pick_id)
        else {
            return;
        };
        let Some(stored) = self.maps.get(item.source) else {
            return;
        };
        let eye = self
            .views
            .get(self.last_viewport)
            .map_or(glam::DVec3::ZERO, |v| v.eye);
        out.extend(Self::cell_outline(item, &stored.layout, eye, col, row));
    }
}
