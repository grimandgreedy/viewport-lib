//! The streamtube item type as an [`ItemTypePlugin`]: polyline strips swept
//! into a connected tube mesh at a uniform radius. Consumers submit
//! [`StreamtubeItem`]s on `SceneFrame::streamtube_items`, or
//! [`StreamtubeRefItem`]s on `SceneFrame::streamtube_refs` to draw a streamtube
//! uploaded once through `upload_streamtube`; the renderer routes both fields
//! to this plugin.

use super::cpu_pick::{self, CurveLevels};
use super::draw::{
    build_frame, outline_mask_curve_mesh, paint_curve_mesh, radius_in_pixels,
    render_pick_curve_mesh, resolve_curve_sub_object,
};
use super::pipeline::{CurveFrame, CurveMeshGpu};
use super::types::StreamtubeId;
use crate::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, RectPickContext,
};
use crate::renderer::{
    PickHit, PickId, PickMask, PickRectResult, StreamtubeItem, StreamtubeRefItem, SubObjectRef,
};

pub(crate) const TYPE_NAME: &str = "vpl.streamtube";

#[derive(Default)]
pub(crate) struct StreamtubePlugin {
    /// The pre-uploaded curves, owned by the type that draws them.
    stored: super::store::StreamtubeStore,
    /// The group-1 layout every upload builds its bind group against. Created
    /// on registration, because an upload can arrive before the first frame.
    layouts: Option<super::store::StreamtubeResources>,
    gpu: Option<CurveMeshGpu>,
    /// Per drawn item, rebuilt each prepare: the inline items first, then the
    /// references.
    frame: Vec<CurveFrame>,
    /// Every inline item from the last prepared frame. Reference items are not
    /// here: their geometry lives on the GPU, so they answer the GPU pick only.
    pick_items: Vec<StreamtubeItem>,
}

impl ItemTypePlugin for StreamtubePlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn init_gpu(
        &mut self,
        device: &crate::gpu::Device,
        _shared: &crate::plugin_api::SharedBindings<'_>,
    ) {
        self.layouts = Some(super::store::StreamtubeResources::new(device));
    }

    fn resident_bytes(&self) -> u64 {
        self.stored.allocated_bytes()
    }

    fn on_device_recreated(&mut self, device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {
        self.layouts = Some(super::store::StreamtubeResources::new(device));
        self.gpu = None;
        self.frame.clear();
    }

    fn prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<crate::gpu::CommandBuffer> {
        self.frame.clear();
        let items = items
            .as_any()
            .downcast_ref::<Vec<StreamtubeItem>>()
            .expect("streamtube collection is the SceneFrame field");
        let refs = ctx.refs_of::<StreamtubeRefItem>();
        self.pick_items.clear();
        self.pick_items.extend_from_slice(items);
        if items.is_empty() && refs.is_empty() {
            return Vec::new();
        }
        let layouts = self
            .layouts
            .get_or_insert_with(|| super::store::StreamtubeResources::new(device));
        let stored = &self.stored;
        let gpu = self
            .gpu
            .get_or_insert_with(|| CurveMeshGpu::new(device, ctx.resources, layouts, "streamtube"));

        for item in items {
            if item.settings.hidden || item.positions.is_empty() || item.strip_lengths.is_empty() {
                continue;
            }
            let wireframe = ctx.wireframe_mode || item.settings.wireframe;
            let binds = super::store::resolve_tube_bindings(ctx.resources, layouts, None);
            let mut gpu_data =
                super::store::build_streamtube(device, queue, &binds, item, wireframe);
            if gpu_data.index_count == 0 {
                continue;
            }
            gpu_data.pick_id = item.settings.pick_id;
            gpu_data.model = item.model;
            gpu_data.cast_shadows = item.settings.cast_shadows;
            self.frame.push(build_frame(
                device,
                queue,
                gpu,
                gpu_data,
                ctx.outline_selected && item.settings.selected,
            ));
        }

        // Pre-uploaded references: the payload lives in the store, the model
        // matrix and pick id come from the reference, so one stored streamtube
        // can be drawn twice at two places under two ids.
        for ref_item in refs {
            if ref_item.settings.hidden {
                continue;
            }
            let Some(entry) = stored.get(ref_item.source) else {
                continue;
            };
            let mut gpu_data = entry.clone();
            if gpu_data.index_count == 0 {
                continue;
            }
            queue.write_buffer(
                &gpu_data._uniform_buf,
                0,
                bytemuck::bytes_of(&ref_item.model),
            );
            gpu_data.pick_id = ref_item.settings.pick_id;
            gpu_data.model = ref_item.model;
            gpu_data.wireframe = ctx.wireframe_mode || ref_item.settings.wireframe;
            gpu_data.cast_shadows = ref_item.settings.cast_shadows;
            self.frame.push(build_frame(
                device,
                queue,
                gpu,
                gpu_data,
                ctx.outline_selected && ref_item.settings.selected,
            ));
        }
        Vec::new()
    }

    fn paint(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        paint_curve_mesh(pass, ctx, self.gpu.as_ref(), &self.frame);
    }

    fn draws_ldr(&self) -> bool {
        true
    }

    fn outline_mask(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &OutlineMaskContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        outline_mask_curve_mesh(pass, self.gpu.as_ref().map(|g| &g.pick), &self.frame);
    }

    fn pick(&self, ray: &PickRay, ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
        let levels = CurveLevels::from_mask(ctx.mask);
        if levels.is_empty() {
            return None;
        }
        let mut best: Option<(f32, PickHit)> = None;
        let mut consider = |toi: f32, hit: PickHit| {
            if best.as_ref().is_none_or(|(t, _)| toi < *t) {
                best = Some((toi, hit));
            }
        };
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                continue;
            }
            let radius_px = radius_in_pixels(&item.positions, item.radius.max(0.01), ctx);
            if levels.node || levels.strip || levels.object {
                if let Some(hit) = cpu_pick::node_hit(
                    levels,
                    ctx.click_pos,
                    item.settings.pick_id,
                    &item.positions,
                    &item.strip_lengths,
                    ctx.view_proj,
                    ctx.viewport_size,
                    radius_px.max(8.0),
                ) {
                    let toi = (hit.world_pos - ray.origin).dot(ray.direction).max(0.0);
                    consider(toi, hit);
                }
            }
            if levels.segment || levels.strip || levels.object {
                if let Some((world_pos, hit)) = cpu_pick::segment_hit(
                    levels,
                    ctx.click_pos,
                    item.settings.pick_id,
                    &item.positions,
                    &item.strip_lengths,
                    ctx.view_proj,
                    ctx.viewport_size,
                    radius_px,
                ) {
                    let toi = (world_pos - ray.origin).dot(ray.direction).max(0.0);
                    consider(toi, hit);
                }
            }
        }
        best
    }

    fn pick_rect(&self, ctx: &RectPickContext<'_>) -> PickRectResult {
        let levels = CurveLevels::from_mask(ctx.mask);
        let mut result = PickRectResult::default();
        if levels.is_empty() {
            return result;
        }
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                continue;
            }
            cpu_pick::rect_pick_line_curve(
                levels,
                &mut result,
                item.settings.pick_id,
                &item.positions,
                &item.strip_lengths,
                ctx.view_proj,
                ctx.viewport_size,
                ctx.rect_min,
                ctx.rect_max,
            );
        }
        result
    }

    fn render_pick(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &PickPassContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        render_pick_curve_mesh(pass, ctx, self.gpu.as_ref().map(|g| &g.pick), &self.frame);
    }

    fn resolve_sub_object(
        &self,
        pick_id: PickId,
        primitive_index: u32,
        _world_pos: glam::Vec3,
        mask: PickMask,
    ) -> Option<SubObjectRef> {
        resolve_curve_sub_object(&self.frame, pick_id, primitive_index, mask)
    }
    fn sub_object_position(
        &self,
        items: &dyn PluginItemCollection,
        pick_id: PickId,
        sub_object: SubObjectRef,
    ) -> Option<glam::Vec3> {
        crate::renderer::picking::helpers::inline_point_position(
            items,
            pick_id,
            sub_object,
            |item: &StreamtubeItem| (item.settings.pick_id, &item.positions, &item.model),
        )
    }
}

impl StreamtubePlugin {
    /// Build one curve's GPU data against the plugin's layout.
    fn build(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
        item: &StreamtubeItem,
    ) -> super::store::StreamtubeGpuData {
        let layouts = self
            .layouts
            .get_or_insert_with(|| super::store::StreamtubeResources::new(device));
        let binds = super::store::resolve_tube_bindings(resources, layouts, None);
        super::store::build_streamtube(device, queue, &binds, item, false)
    }

    /// Pre-upload a curve and return its handle.
    pub(crate) fn upload(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
        item: &StreamtubeItem,
    ) -> StreamtubeId {
        let gpu = self.build(device, queue, resources, item);
        self.stored.insert_sized(gpu)
    }

    /// Drop a stored curve. `false` when the handle does not resolve.
    pub(crate) fn drop_stored(&mut self, id: StreamtubeId) -> bool {
        self.stored.remove(id).is_some()
    }

    /// Replace the geometry behind a live handle, keeping the handle.
    pub(crate) fn replace(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
        id: StreamtubeId,
        item: &StreamtubeItem,
    ) -> bool {
        if !self.stored.contains(id) {
            return false;
        }
        let gpu = self.build(device, queue, resources, item);
        self.stored.replace_sized(id, gpu).is_some()
    }

    /// Sweep the curve mesh on a worker thread. The handle is minted when
    /// [`take_upload_result`](Self::take_upload_result) collects the job.
    ///
    /// The layout and the colourmap the upload binds are resolved here and
    /// cloned into the worker: they are the renderer's and a worker has no
    /// `DeviceResources` borrow.
    pub(crate) fn begin_upload(
        &mut self,
        jobs: &crate::resources::Jobs<'_>,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
        item: StreamtubeItem,
    ) -> crate::resources::JobId {
        let layouts = self
            .layouts
            .get_or_insert_with(|| super::store::StreamtubeResources::new(device));
        let item = item;
        let binds = super::store::resolve_tube_bindings(resources, layouts, None);
        let device = device.clone();
        let queue = queue.clone();
        jobs.submit_cpu(move || {
            super::store::build_streamtube(&device, &queue, &binds, &item, false)
        })
    }

    /// Store the curve a finished job built and hand back its handle.
    pub(crate) fn take_upload_result(
        &mut self,
        jobs: &crate::resources::Jobs<'_>,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<StreamtubeId> {
        match jobs.status(id) {
            crate::resources::UploadStatus::Pending { .. } => {
                Err(crate::error::ViewportError::JobNotReady)
            }
            crate::resources::UploadStatus::Failed(e) => Err(e),
            _ => match jobs.take::<super::store::StreamtubeGpuData>(id) {
                Some(gpu) => Ok(self.stored.insert_sized(gpu)),
                None => Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                }),
            },
        }
    }

    /// Number of items the last `prepare` produced draw data for.
    #[cfg(test)]
    pub(crate) fn drawn_count(&self) -> usize {
        self.frame.len()
    }
}
