//! The point cloud item type: a set of world-space points drawn as
//! screen-space discs, coloured by a scalar through a colourmap or by
//! per-point colours.
//!
//! Submit [`PointCloudItem`]s with `frame.scene.submit::<PointCloudItem>(..)`,
//! or [`PointCloudRefItem`]s to draw a cloud uploaded once through
//! [`PointCloudUploads::upload_point_cloud`](crate::PointCloudUploads::upload_point_cloud).
//! Both forms arrive at this plugin under the one type name.

pub mod channels;
mod pipeline;
mod store;
mod types;

use store::{PointCloudStore, build_point_cloud, resolve_bindings};
use viewport_lib::gpu;
use viewport_lib::plugin_api::{
    ItemCollections, ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext,
    PickContext, PickPassContext, PickRay, PluginItem, RectPickContext,
};
use viewport_lib::renderer::{PickHit, PickId, PickMask, PickRectResult, SubObjectRef};
use viewport_lib::resources::HDR_COLOR_FORMAT;

pub use types::{PointCloudId, PointCloudItem, PointCloudRefItem, PointRenderMode};

/// Which channel a write names, as the store sees it. The public face is the
/// marker types in [`channels`].
pub(crate) use store::PointChannel;

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.point_cloud";

/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::shader::{scene_shader, wgsl_source};
    vec![
        (
            "point_cloud.wgsl",
            scene_shader(&[], wgsl_source!("point_cloud")),
        ),
        (
            "point_cloud_pick.wgsl",
            scene_shader(&[], wgsl_source!("point_cloud_pick")),
        ),
    ]
}

impl PluginItem for PointCloudItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

impl PluginItem for PointCloudRefItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

/// The point cloud item type. Register it with
/// [`ViewportRenderer::with_item_type_plugin`](viewport_lib::renderer::ViewportRenderer::with_item_type_plugin),
/// or take the whole set with [`install`](crate::install).
#[derive(Default)]
pub struct PointCloudPlugin {
    /// The pre-uploaded clouds, owned by the type that draws them.
    stored: PointCloudStore,
    /// Group-1 layout every upload builds its bind group against. Created on
    /// registration, because an upload can arrive before the first frame.
    bgl: Option<gpu::BindGroupLayout>,
    gpu: Option<pipeline::PointCloudGpu>,
    /// Per drawn item, rebuilt each prepare: the inline items first, then the
    /// references, the order the draw loop used before the two collections met
    /// at this plugin.
    frame: Vec<pipeline::PointCloudFrame>,
    /// Every inline item from the last prepared frame, hidden included, matching
    /// what the CPU pick cache used to retain. Reference items are not here:
    /// their positions live on the GPU, so they answer the GPU pick only.
    pick_items: Vec<PointCloudItem>,
    /// Selection-outline coverage for this frame.
    outlines: Vec<pipeline::PointCloudOutline>,
}

impl ItemTypePlugin for PointCloudPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn init_gpu(
        &mut self,
        device: &gpu::Device,
        _shared: &viewport_lib::plugin_api::SharedBindings<'_>,
    ) {
        self.bgl = Some(store::build_bgl(device));
    }

    fn resident_bytes(&self) -> u64 {
        self.stored.allocated_bytes()
    }

    fn on_device_recreated(&mut self, device: &gpu::Device, _queue: &gpu::Queue) {
        self.gpu = None;
        self.bgl = Some(store::build_bgl(device));
        self.frame.clear();
        self.outlines.clear();
    }

    fn prepare(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &ItemCollections<'_>,
    ) -> Vec<gpu::CommandBuffer> {
        self.frame.clear();
        self.outlines.clear();
        let items = items.of::<PointCloudItem>();
        let refs = ctx.refs_of::<PointCloudRefItem>();
        self.pick_items.clear();
        self.pick_items.extend_from_slice(items);
        if items.is_empty() && refs.is_empty() {
            return Vec::new();
        }
        let bgl = self.bgl.get_or_insert_with(|| store::build_bgl(device));
        let stored = &self.stored;
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::PointCloudGpu::new(device, ctx.resources, bgl));

        for item in items {
            if item.settings.hidden || item.positions.is_empty() {
                continue;
            }
            let binds = resolve_bindings(ctx.resources, bgl, item);
            let draw = build_point_cloud(device, queue, &binds, item, 0).draw();
            let pick_bind_group = (draw.pick_id != PickId::NONE)
                .then(|| gpu.pick_bind_group(device, queue, draw.pick_id));
            self.frame.push(pipeline::PointCloudFrame {
                draw,
                pick_bind_group,
            });
        }

        // Pre-uploaded references. The model matrix lives at offset 0 of the
        // uniform, so a reference re-places a stored cloud without re-uploading
        // its points.
        for ref_item in refs {
            if ref_item.settings.hidden {
                continue;
            }
            let Some(entry) = stored.get(ref_item.source) else {
                continue;
            };
            entry.write_model(queue, &ref_item.model);
            let mut draw = entry.draw();
            // The pick id comes from the reference, not from the upload: the
            // same stored cloud can be drawn twice under two ids.
            draw.pick_id = ref_item.settings.pick_id;
            let pick_bind_group = (draw.pick_id != PickId::NONE)
                .then(|| gpu.pick_bind_group(device, queue, draw.pick_id));
            self.frame.push(pipeline::PointCloudFrame {
                draw,
                pick_bind_group,
            });
        }

        if ctx.outline_selected {
            self.outlines = build_outlines(device, ctx, items, gpu);
        }
        Vec::new()
    }

    fn paint(
        &self,
        pass: &mut gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let Some(gpu) = &self.gpu else { return };
        if self.frame.is_empty() {
            return;
        }
        pass.set_pipeline(
            gpu.pipeline
                .for_format(ctx.target_format == HDR_COLOR_FORMAT),
        );
        for entry in &self.frame {
            pass.set_bind_group(1, &entry.draw.bind_group, &[]);
            pass.set_vertex_buffer(0, entry.draw.vertex_buffer.slice(..));
            // Six vertices per point (a billboard quad), one instance per point.
            pass.draw(0..6, 0..entry.draw.point_count);
        }
    }

    fn draws_ldr(&self) -> bool {
        true
    }

    fn outline_mask(
        &self,
        pass: &mut gpu::RenderPass<'_>,
        _ctx: &OutlineMaskContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let Some(gpu) = &self.gpu else { return };
        if self.outlines.is_empty() {
            return;
        }
        pass.set_pipeline(&gpu.mask_pipeline);
        for entry in &self.outlines {
            pass.set_bind_group(1, &entry.bind_group, &[]);
            pass.set_vertex_buffer(0, entry.position_buf.slice(..));
            pass.set_vertex_buffer(1, entry.size_buf.slice(..));
            pass.draw(0..6, 0..entry.instance_count);
        }
    }

    fn pick(&self, ray: &PickRay, ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
        let wants_cloud = ctx.mask.intersects(PickMask::CLOUD_POINT);
        if !wants_cloud && !ctx.mask.intersects(PickMask::OBJECT) {
            return None;
        }
        let mut best: Option<(f32, PickHit)> = None;
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                continue;
            }
            // The largest disc the size source can produce, with a floor so a
            // one-pixel cloud is still clickable. Taking the largest keeps the
            // hit circle over every point rather than only the small ones.
            let radius_px = item.size.output_range().1.max(4.0);
            let Some(mut hit) = viewport_lib::picking::pick_gaussian_splat_cpu(
                ctx.click_pos,
                item.settings.pick_id.0,
                &item.positions,
                glam::Mat4::from_cols_array_2d(&item.model),
                ctx.view_proj,
                ctx.viewport_size,
                radius_px,
            ) else {
                continue;
            };
            let toi = (hit.world_pos - ray.origin).dot(ray.direction).max(0.0);
            if !wants_cloud {
                hit.sub_object = None;
            }
            if best.as_ref().is_none_or(|(t, _)| toi < *t) {
                best = Some((toi, hit));
            }
        }
        best
    }

    fn pick_rect(&self, ctx: &RectPickContext<'_>) -> PickRectResult {
        let mut result = PickRectResult::default();
        let wants_cloud = ctx.mask.intersects(PickMask::CLOUD_POINT);
        let wants_object = ctx.mask.intersects(PickMask::OBJECT);
        if !wants_cloud && !wants_object {
            return result;
        }
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                continue;
            }
            let model = glam::Mat4::from_cols_array_2d(&item.model);
            let id = item.settings.pick_id.0;
            let mut item_hit = false;
            for (index, pos) in item.positions.iter().enumerate() {
                let world = model.transform_point3(glam::Vec3::from(*pos));
                let Some(p) = viewport_lib::plugin_api::pick_helpers::project_to_screen(
                    world,
                    ctx.view_proj,
                    ctx.viewport_size,
                ) else {
                    continue;
                };
                if p.x < ctx.rect_min.x
                    || p.x > ctx.rect_max.x
                    || p.y < ctx.rect_min.y
                    || p.y > ctx.rect_max.y
                {
                    continue;
                }
                if wants_cloud {
                    result
                        .elements
                        .push((id, SubObjectRef::Point(index as u32)));
                }
                item_hit = true;
            }
            if wants_object && item_hit {
                result.objects.push(id);
            }
        }
        result
    }

    fn render_pick(
        &self,
        pass: &mut gpu::RenderPass<'_>,
        ctx: &PickPassContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        if !ctx
            .mask
            .intersects(PickMask::OBJECT | PickMask::CLOUD_POINT)
        {
            return;
        }
        let Some(gpu) = &self.gpu else { return };
        let mut bound = false;
        for entry in &self.frame {
            let Some(pick_bg) = &entry.pick_bind_group else {
                continue;
            };
            if entry.draw.point_count == 0 {
                continue;
            }
            if !bound {
                pass.set_pipeline(&gpu.pick_pipeline);
                bound = true;
            }
            pass.set_bind_group(1, &entry.draw.bind_group, &[]);
            pass.set_bind_group(2, pick_bg, &[]);
            pass.set_vertex_buffer(0, entry.draw.vertex_buffer.slice(..));
            pass.draw(0..6, 0..entry.draw.point_count);
        }
    }

    fn resolve_sub_object(
        &self,
        _pick_id: PickId,
        primitive_index: u32,
        _world_pos: glam::Vec3,
        mask: PickMask,
    ) -> Option<SubObjectRef> {
        // The pick fragment writes the point's instance index into the
        // primitive channel; no device feature involved.
        mask.intersects(PickMask::CLOUD_POINT)
            .then_some(SubObjectRef::Point(primitive_index))
    }
    fn sub_object_position(
        &self,
        items: &ItemCollections<'_>,
        pick_id: PickId,
        sub_object: SubObjectRef,
    ) -> Option<glam::Vec3> {
        viewport_lib::plugin_api::pick_helpers::inline_point_position(
            items,
            pick_id,
            sub_object,
            |item: &PointCloudItem| (item.settings.pick_id, &item.positions, &item.model),
        )
    }
}

impl PointCloudPlugin {
    /// Number of items the last `prepare` produced draw data for.
    pub fn drawn_count(&self) -> usize {
        self.frame.len()
    }

    /// Pre-upload a point cloud and return its handle.
    pub fn upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        item: &PointCloudItem,
    ) -> PointCloudId {
        self.upload_with_capacity(device, queue, resources, item, 0)
    }

    /// Pre-upload a point cloud with room for `capacity` points.
    ///
    /// The cloud draws the points it was given and holds the rest as headroom, so
    /// a feed that grows to a known size allocates once here and then only
    /// writes. Capacity below the point count is raised to it, and the reserved
    /// bytes count against `resident_bytes` whether or not anything draws them.
    pub fn upload_with_capacity(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        item: &PointCloudItem,
        capacity: u32,
    ) -> PointCloudId {
        let bgl = self.bgl.get_or_insert_with(|| store::build_bgl(device));
        let binds = resolve_bindings(resources, bgl, item);
        let gpu = build_point_cloud(device, queue, &binds, item, capacity);
        self.stored.insert_sized(gpu)
    }

    /// Write part of one channel of a stored cloud.
    ///
    /// The bytes are already encoded for the channel: the caller-facing
    /// conversion happens in the `Writes` implementation, which is the only place
    /// that knows the input type.
    pub(crate) fn write_channel(
        &mut self,
        queue: &gpu::Queue,
        id: PointCloudId,
        which: store::PointChannel,
        name: &'static str,
        first_element: u32,
        data: &[u8],
    ) -> viewport_lib::error::ViewportResult<()> {
        let Some(gpu) = self.stored.get_mut(id) else {
            return Err(self.stored.stale(id));
        };
        gpu.write_channel(queue, which, name, first_element, data)
    }

    /// Point one channel of a stored cloud at a caller-owned buffer, or back at
    /// the cloud's own.
    pub(crate) fn set_channel_source(
        &mut self,
        device: &gpu::Device,
        id: PointCloudId,
        which: store::PointChannel,
        name: &'static str,
        source: Option<gpu::Buffer>,
    ) -> viewport_lib::error::ViewportResult<()> {
        let Some(gpu) = self.stored.get_mut(id) else {
            return Err(self.stored.stale(id));
        };
        if gpu.set_channel_source(device, which, name, source)? {
            // The bind group was replaced, so a per-viewport cache built over the
            // old one is stale. Nothing keys on it for this type today, and that
            // is exactly why it is bumped here rather than left for whoever adds
            // the first cache to discover.
            self.stored.bump_revision(id);
        }
        Ok(())
    }

    /// Whether one channel of a stored cloud draws from a caller-owned buffer.
    pub(crate) fn channel_has_source(
        &self,
        id: PointCloudId,
        which: store::PointChannel,
    ) -> Option<bool> {
        self.stored.get(id).map(|gpu| gpu.channel_has_source(which))
    }

    /// Grow a stored cloud to hold at least `capacity` points.
    pub(crate) fn reserve_stored(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: PointCloudId,
        capacity: u32,
    ) -> viewport_lib::error::ViewportResult<()> {
        let Some(gpu) = self.stored.get_mut(id) else {
            return Err(self.stored.stale(id));
        };
        if gpu.reserve(device, queue, capacity) {
            // The allocations moved, so anything built over them is stale and the
            // store's charge is wrong.
            self.stored.bump_revision(id);
            self.stored.recharge(id);
        }
        Ok(())
    }

    /// Set how many of a stored cloud's points draw.
    pub(crate) fn set_stored_len(
        &mut self,
        id: PointCloudId,
        len: u32,
    ) -> viewport_lib::error::ViewportResult<()> {
        let Some(gpu) = self.stored.get_mut(id) else {
            return Err(self.stored.stale(id));
        };
        gpu.set_live_len(len)
    }

    /// What one channel of a stored cloud holds, and how much of it draws.
    pub(crate) fn stored_extent(
        &self,
        id: PointCloudId,
        which: store::PointChannel,
    ) -> Option<viewport_lib::plugin_api::Extent> {
        self.stored.get(id).map(|gpu| gpu.extent(which))
    }

    /// Drop a pre-uploaded cloud. `false` when the handle does not resolve.
    pub fn drop_stored(&mut self, id: PointCloudId) -> bool {
        self.stored.remove(id).is_some()
    }

    /// Replace the points behind a live handle, keeping the handle.
    ///
    /// Writes into the existing buffers when the new cloud has the same shape
    /// as the old one: same point count, same colourmap, same set of present
    /// channels. That is the normal case for a feed replacing values, and it
    /// costs a handful of buffer writes rather than reallocating every buffer
    /// and the bind group. Anything else rebuilds.
    pub fn replace(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        id: PointCloudId,
        item: &PointCloudItem,
    ) -> viewport_lib::error::ViewportResult<()> {
        if !self.stored.contains(id) {
            return Err(self.stored.stale(id));
        }
        if let Some(gpu) = self.stored.get_mut(id)
            && store::try_replace_in_place(queue, gpu, item)
        {
            // The bytes are unchanged, so the store's charge still holds, but
            // the contents are not: stamp a new revision so a cache keyed on
            // one cannot serve the old cloud.
            self.stored.bump_revision(id);
            return Ok(());
        }
        let bgl = self.bgl.get_or_insert_with(|| store::build_bgl(device));
        let binds = resolve_bindings(resources, bgl, item);
        let gpu = build_point_cloud(device, queue, &binds, item, 0);
        self.stored.replace_sized(id, gpu);
        Ok(())
    }

    /// Build a cloud's buffers on a worker thread. The handle is minted when
    /// [`take_upload_result`](Self::take_upload_result) collects the job.
    ///
    /// The colourmap view and shared sampler the upload binds are resolved
    /// here, on the calling thread, and cloned into the worker: they are the
    /// renderer's and a worker has no `DeviceResources` borrow. The buffer
    /// building and the bind group then run off the frame thread.
    pub fn begin_upload(
        &mut self,
        jobs: &viewport_lib::resources::Jobs<'_>,
        device: &gpu::Device,
        queue: &gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        item: PointCloudItem,
    ) -> viewport_lib::resources::JobId {
        let bgl = self.bgl.get_or_insert_with(|| store::build_bgl(device));
        let binds = resolve_bindings(resources, bgl, &item);
        let device = device.clone();
        let queue = queue.clone();
        jobs.submit_cpu(move || build_point_cloud(&device, &queue, &binds, &item, 0))
    }

    /// Store the cloud a finished job built and hand back its handle.
    pub fn take_upload_result(
        &mut self,
        jobs: &viewport_lib::resources::Jobs<'_>,
        id: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<PointCloudId> {
        match jobs.status(id) {
            viewport_lib::resources::UploadStatus::Pending { .. } => {
                Err(viewport_lib::error::ViewportError::JobNotReady)
            }
            viewport_lib::resources::UploadStatus::Failed(e) => Err(e),
            _ => match jobs.take::<store::PointCloudGpuData>(id) {
                Some(gpu) => Ok(self.stored.insert_sized(gpu)),
                None => Err(viewport_lib::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                }),
            },
        }
    }
}

/// Build this frame's outline coverage: every point of a selected item, or just
/// the sub-selected points of an item that is not itself selected.
fn build_outlines(
    device: &gpu::Device,
    ctx: &ItemFrameContext<'_>,
    items: &[PointCloudItem],
    gpu: &pipeline::PointCloudGpu,
) -> Vec<pipeline::PointCloudOutline> {
    let mut outlines = Vec::new();
    for item in items {
        if item.settings.hidden || item.positions.is_empty() {
            continue;
        }
        let pixel_radius = (item.size.output_range().1 * 0.5).max(1.0);
        if item.settings.selected {
            outlines.push(gpu.outline_entry(
                device,
                item.model,
                ctx.viewport_size,
                pixel_radius,
                &item.positions,
            ));
        } else if item.settings.pick_id != PickId::NONE {
            let selected: Vec<[f32; 3]> = ctx
                .sub_selection
                .iter()
                .flat_map(|s| s.items().iter())
                .filter_map(|(node_id, sub)| {
                    if *node_id != item.settings.pick_id.0 {
                        return None;
                    }
                    match sub {
                        SubObjectRef::Point(i) => item.positions.get(*i as usize).copied(),
                        _ => None,
                    }
                })
                .collect();
            if selected.is_empty() {
                continue;
            }
            outlines.push(gpu.outline_entry(
                device,
                item.model,
                ctx.viewport_size,
                pixel_radius,
                &selected,
            ));
        }
    }
    outlines
}
