//! The tilemap plugin: its store, its per-frame layers and its draw hooks.

use super::store::{
    LAYER_STRIDE, StoredTilemap, TilemapLayouts, TilemapStore, chunk_bounds, layer_uniform,
    resolve_bindings,
};
use super::*;
use viewport_lib::plugin_api::{
    DepthReadContext, ItemCollections, ItemFrameContext, ItemTypePlugin, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItem,
};
use viewport_lib::renderer::{PickHit, PickId, PickMask, SubObjectRef};

/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::item_types::shader::{scene_shader, wgsl_source};
    vec![
        (
            "tilemap.wgsl",
            scene_shader(&[super::TILEMAP_COMMON], wgsl_source!("tilemap")),
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
    /// Dynamic offset of the layer's uniform.
    offset: u32,
    /// Chunks that draw: occupied and, after the cull, in view.
    visible: Vec<u32>,
}

impl FrameLayer {
    fn painter(&self) -> Option<(u32, bool)> {
        match self.item.draw {
            TilemapDraw::Blended { layer, depth_test } => Some((layer, depth_test)),
            TilemapDraw::Cutout { .. } => None,
        }
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
    frame: Vec<FrameLayer>,
    /// Indices into `frame` of the blended layers, in the order they draw.
    painter_sequence: Vec<usize>,
    layer_buf: Option<viewport_lib::gpu::Buffer>,
    layer_bg: Option<viewport_lib::gpu::BindGroup>,
    /// The camera position of the last prepare, for the CPU pick.
    eye: glam::DVec3,
    drawn_chunks: u32,
    upload_bytes: u64,
}

impl TilemapPlugin {
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
            resources,
            layouts,
            map.tileset.texture_id,
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
            resources,
            layouts,
            map.tileset.texture_id,
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
    /// that changed, once. `Ok(false)` when any cell was off the map; the
    /// cells on it are still written.
    pub fn set_tiles(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        id: TilemapId,
        tiles: impl IntoIterator<Item = (u32, u32, Tile)>,
    ) -> viewport_lib::error::ViewportResult<bool> {
        let Some(stored) = self.maps.get_mut(id) else {
            return Err(self.maps.stale(id));
        };
        let mut all_on_map = true;
        let mut touched: Vec<u32> = Vec::new();
        for (col, row, tile) in tiles {
            match stored.map.set(col, row, tile) {
                Some(chunk) => {
                    if !touched.contains(&chunk) {
                        touched.push(chunk);
                    }
                }
                None => all_on_map = false,
            }
        }
        for chunk in touched {
            stored.upload_chunk(device, queue, chunk);
        }
        self.maps.recharge(id);
        Ok(all_on_map)
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

    /// Chunk draws in the colour passes of the last prepared frame, after
    /// the cull.
    pub fn drawn_chunks(&self) -> u32 {
        self.drawn_chunks
    }

    /// Rebind stored maps whose atlas was freed or swapped since the last
    /// frame. The map stays; a freed atlas leaves it drawing untextured.
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
            let Some(texture) = stored.map.tileset.texture_id else {
                continue;
            };
            if action == Revalidate::CheckEach && resources.has_texture(texture) {
                continue;
            }
            let binds = resolve_bindings(
                resources,
                layouts,
                Some(texture),
                stored.map.tileset.filter,
                false,
            );
            stored.rebind(device, queue, &binds);
            if !resources.has_texture(texture) {
                stored.map.tileset.texture_id = None;
            }
        }
    }

    fn count_drawn(&mut self) {
        self.drawn_chunks = self.frame.iter().map(|l| l.visible.len() as u32).sum();
    }

    /// Draw `layers` (indices into `frame`) with `member`, binding per layer.
    fn draw_layers(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        layers: impl Iterator<Item = usize>,
        member: impl Fn(&FrameLayer) -> usize,
    ) {
        let (Some(gpu), Some(layer_bg)) = (&self.gpu, &self.layer_bg) else {
            return;
        };
        let mut bound = None;
        for i in layers {
            let layer = &self.frame[i];
            let Some(stored) = self.maps.get(layer.item.source) else {
                continue;
            };
            let m = member(layer);
            if bound != Some(m) {
                // Still compiling: this layer draws next frame.
                let Some(pl) = gpu.pipelines.get(m) else {
                    continue;
                };
                pass.set_pipeline(pl);
                bound = Some(m);
            }
            pass.set_bind_group(1, &stored.bind_group, &[]);
            pass.set_bind_group(2, layer_bg, &[layer.offset]);
            for &chunk in &layer.visible {
                let chunk = &stored.chunks[chunk as usize];
                let Some(buffer) = &chunk.buffer else {
                    continue;
                };
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..6, 0..chunk.count);
            }
        }
    }

    fn cutout_layers(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.frame.len()).filter(|&i| self.frame[i].painter().is_none())
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
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::TilemapGpu::new(device, resources, layouts));
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
        self.frame.clear();
        self.painter_sequence.clear();
        self.upload_bytes = 0;
        self.revalidate_stores(device, queue, ctx.resources);
        self.eye = ctx.camera.eye_position();
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
            let data = layer_uniform(item, stored, origin);
            uniforms.extend_from_slice(bytemuck::bytes_of(&data));
            uniforms.resize(offset as usize + LAYER_STRIDE as usize, 0);
            let visible = (0..stored.chunks.len() as u32)
                .filter(|&c| stored.chunks[c as usize].count > 0)
                .collect();
            self.frame.push(FrameLayer {
                item: item.clone(),
                offset,
                visible,
            });
        }
        if self.frame.is_empty() {
            self.count_drawn();
            return Vec::new();
        }

        let layouts = self
            .layouts
            .get_or_insert_with(|| TilemapLayouts::new(device));
        self.gpu
            .get_or_insert_with(|| pipeline::TilemapGpu::new(device, ctx.resources, layouts));
        let needed = uniforms.len() as u64;
        if self.layer_buf.as_ref().is_none_or(|b| b.size() < needed) {
            let size = needed.next_power_of_two().max(LAYER_STRIDE * 4);
            let buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
                label: Some("tilemap_layer_buf"),
                size,
                usage: viewport_lib::gpu::BufferUsages::UNIFORM
                    | viewport_lib::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.layer_bg = Some(device.create_bind_group(
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
            self.layer_buf = Some(buf);
        }
        if let Some(buf) = &self.layer_buf {
            queue.write_buffer(buf, 0, &uniforms);
        }
        self.upload_bytes += needed;

        let painters: Vec<(usize, u32)> = self
            .frame
            .iter()
            .enumerate()
            .filter_map(|(i, l)| l.painter().map(|(layer, _)| (i, layer)))
            .collect();
        self.painter_sequence =
            crate::item_types::sprite::painter_batch_order(painters.iter().map(|&(_, l)| l))
                .into_iter()
                .map(|k| painters[k].0)
                .collect();
        self.count_drawn();
        Vec::new()
    }

    fn cull(
        &mut self,
        frustum: &viewport_lib::Frustum,
        ctx: &ItemFrameContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let eye = ctx.camera.eye_position();
        for layer in &mut self.frame {
            let Some(stored) = self.maps.get(layer.item.source) else {
                layer.visible.clear();
                continue;
            };
            layer.visible = (0..stored.chunks.len() as u32)
                .filter(|&c| stored.chunks[c as usize].count > 0)
                .filter(|&c| {
                    let (min, max) = chunk_bounds(&layer.item, &stored.map, c, eye);
                    !frustum.cull_aabb(&viewport_lib::Aabb { min, max })
                })
                .collect();
        }
        self.count_drawn();
    }

    fn paint(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        if ctx.is_hdr_pass() {
            self.draw_layers(pass, self.cutout_layers(), |_| pipeline::CUTOUT_SCENE);
            return;
        }
        self.draw_layers(pass, self.cutout_layers(), |_| pipeline::CUTOUT_LDR);
        // The LDR path has no later pass, so blended layers draw here, after
        // the cutout ones.
        self.draw_layers(pass, self.painter_sequence.iter().copied(), |l| {
            match l.painter() {
                Some((_, false)) => pipeline::BLEND_OFF_LDR,
                _ => pipeline::BLEND_TEST_LDR,
            }
        });
    }

    fn draws_ldr(&self) -> bool {
        true
    }

    fn draws_depth_read(&self) -> bool {
        !self.painter_sequence.is_empty()
    }

    fn paint_depth_read(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        _ctx: &DepthReadContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        self.draw_layers(pass, self.painter_sequence.iter().copied(), |l| {
            match l.painter() {
                Some((_, false)) => pipeline::BLEND_OFF_DEPTH_READ,
                _ => pipeline::BLEND_TEST_DEPTH_READ,
            }
        });
    }

    fn render_pick(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PickPassContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        if !ctx.mask.intersects(PickMask::OBJECT | PickMask::INSTANCE) {
            return;
        }
        // Cutout layers first, then the blended ones in the order they drew.
        let pickable = |i: &usize| self.frame[*i].item.settings.pick_id != PickId::NONE;
        let order = self
            .cutout_layers()
            .chain(self.painter_sequence.iter().copied())
            .filter(pickable)
            .collect::<Vec<_>>();
        self.draw_layers(pass, order.into_iter(), |l| match l.painter() {
            None => pipeline::PICK,
            Some((_, true)) => pipeline::PICK_PAINTER,
            Some((_, false)) => pipeline::PICK_PAINTER_NO_TEST,
        });
    }

    fn resolve_sub_object(
        &self,
        pick_id: PickId,
        primitive_index: u32,
        _world_pos: glam::Vec3,
        mask: PickMask,
    ) -> Option<SubObjectRef> {
        if !mask.intersects(PickMask::INSTANCE) {
            return None;
        }
        let layer = self
            .frame
            .iter()
            .find(|l| l.item.settings.pick_id == pick_id)?;
        let map = &self.maps.get(layer.item.source)?.map;
        let (chunk, tile) = map.locate(primitive_index & 0xffff, primitive_index >> 16)?;
        Some(SubObjectRef::Tile { chunk, tile })
    }

    fn pick(&self, ray: &PickRay, ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
        let wants_tile = ctx.mask.intersects(PickMask::INSTANCE);
        if !wants_tile && !ctx.mask.intersects(PickMask::OBJECT) {
            return None;
        }
        let origin = ray.origin.as_dvec3();
        let dir = ray.direction.as_dvec3().normalize_or_zero();
        let mut best: Option<(f32, PickHit)> = None;
        for layer in &self.frame {
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
            let local = plane.to_plane(hit) - item.drawn_origin(plane.to_plane(self.eye));
            let col = (local.x / item.cell_size[0] as f64).floor();
            let row = (local.y / item.cell_size[1] as f64).floor();
            if col < 0.0 || row < 0.0 {
                continue;
            }
            let (col, row) = (col as u32, row as u32);
            if stored.map.get(col, row).is_none_or(|t| t.is_empty()) {
                continue;
            }
            let toi = t as f32;
            // Ties go to the later layer, which drew on top.
            if best.as_ref().is_some_and(|(b, _)| *b < toi) {
                continue;
            }
            let mut pick = PickHit::object_hit(item.settings.pick_id.0, hit, plane.normal());
            if wants_tile {
                pick.sub_object = stored
                    .map
                    .locate(col, row)
                    .map(|(chunk, tile)| SubObjectRef::Tile { chunk, tile });
            }
            best = Some((toi, pick));
        }
        best
    }
}
