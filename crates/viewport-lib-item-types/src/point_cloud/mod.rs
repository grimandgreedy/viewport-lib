//! The point cloud item type: a set of world-space points drawn as
//! screen-space discs, coloured by a scalar through a colourmap or by
//! per-point colours.
//!
//! Submit [`PointCloudItem`]s with `frame.scene.submit::<PointCloudItem>(..)`,
//! or [`PointCloudRefItem`]s to draw a cloud uploaded once through
//! [`PointCloudUploads::upload_point_cloud`](crate::PointCloudUploads::upload_point_cloud).
//! Both forms arrive at this plugin under the one type name.

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
        (
            "point_disc_mask.wgsl",
            scene_shader(&[], wgsl_source!("point_disc_mask")),
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
            let gpu_data = build_point_cloud(device, queue, &binds, item);
            let pick_bind_group = (gpu_data.pick_id != PickId::NONE)
                .then(|| gpu.pick_bind_group(device, queue, gpu_data.pick_id));
            self.frame.push(pipeline::PointCloudFrame {
                gpu: gpu_data,
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
            let entry = entry.clone();
            queue.write_buffer(&entry._uniform_buf, 0, bytemuck::bytes_of(&ref_item.model));
            // The pick id comes from the reference, not from the upload: the
            // same stored cloud can be drawn twice under two ids.
            let pick_id = ref_item.settings.pick_id;
            let pick_bind_group =
                (pick_id != PickId::NONE).then(|| gpu.pick_bind_group(device, queue, pick_id));
            self.frame.push(pipeline::PointCloudFrame {
                gpu: entry,
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
            pass.set_bind_group(1, &entry.gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, entry.gpu.vertex_buffer.slice(..));
            // Six vertices per point (a billboard quad), one instance per point.
            pass.draw(0..6, 0..entry.gpu.point_count);
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
            // The disc is `point_size` pixels across, with a floor so a
            // one-pixel cloud is still clickable.
            let radius_px = item.point_size.max(4.0);
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
            if entry.gpu.point_count == 0 {
                continue;
            }
            if !bound {
                pass.set_pipeline(&gpu.pick_pipeline);
                bound = true;
            }
            pass.set_bind_group(1, &entry.gpu.bind_group, &[]);
            pass.set_bind_group(2, pick_bg, &[]);
            pass.set_vertex_buffer(0, entry.gpu.vertex_buffer.slice(..));
            pass.draw(0..6, 0..entry.gpu.point_count);
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
        let bgl = self.bgl.get_or_insert_with(|| store::build_bgl(device));
        let binds = resolve_bindings(resources, bgl, item);
        let gpu = build_point_cloud(device, queue, &binds, item);
        self.stored.insert_sized(gpu)
    }

    /// Drop a pre-uploaded cloud. `false` when the handle does not resolve.
    pub fn drop_stored(&mut self, id: PointCloudId) -> bool {
        self.stored.remove(id).is_some()
    }

    /// Replace the points behind a live handle, keeping the handle.
    pub fn replace(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        id: PointCloudId,
        item: &PointCloudItem,
    ) -> bool {
        if !self.stored.contains(id) {
            return false;
        }
        let bgl = self.bgl.get_or_insert_with(|| store::build_bgl(device));
        let binds = resolve_bindings(resources, bgl, item);
        let gpu = build_point_cloud(device, queue, &binds, item);
        self.stored.replace_sized(id, gpu).is_some()
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
        jobs.submit_cpu(move || build_point_cloud(&device, &queue, &binds, &item))
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
        let pixel_radius = (item.point_size * 0.5).max(1.0);
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
