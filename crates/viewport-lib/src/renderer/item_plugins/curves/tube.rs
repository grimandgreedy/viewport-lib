//! The tube item type as an [`ItemTypePlugin`]: polyline strips swept into a
//! connected tube mesh with a configurable cross-section, optional per-point
//! radius, and per-vertex scalar colouring. Consumers submit [`TubeItem`]s on
//! `SceneFrame::tube_items`, or [`TubeRefItem`]s on `SceneFrame::tube_refs` to
//! draw a tube uploaded once through `upload_tube`; the renderer routes both
//! fields to this plugin.
//!
//! The mesh shades through the same pipeline description the streamtube type
//! uses, so this plugin compiles its own copy of it rather than borrowing the
//! streamtube plugin's.

use super::cpu_pick::{self, CurveLevels};
use super::draw::{
    build_frame, outline_mask_curve_mesh, paint_curve_mesh, radius_in_pixels,
    render_pick_curve_mesh, resolve_curve_sub_object,
};
use super::pipeline::{CurveFrame, CurveMeshGpu};
use crate::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, RectPickContext,
};
use crate::renderer::{
    PickHit, PickId, PickMask, PickRectResult, SubObjectRef, TubeItem, TubeRefItem,
};

pub(crate) const TYPE_NAME: &str = "viewport.tube";

#[derive(Default)]
pub(crate) struct TubePlugin {
    gpu: Option<CurveMeshGpu>,
    /// Per drawn item, rebuilt each prepare: the inline items first, then the
    /// references.
    frame: Vec<CurveFrame>,
    /// Every inline item from the last prepared frame. Reference items are not
    /// here: their geometry lives on the GPU, so they answer the GPU pick only.
    pick_items: Vec<TubeItem>,
}

/// The conservative pick radius for a tube: the largest of the uniform radius
/// and any per-point radius, so the tolerance covers the widest part of the
/// sweep.
fn max_radius(item: &TubeItem) -> f32 {
    item.radius_attribute
        .as_ref()
        .and_then(|ra| ra.iter().copied().reduce(f32::max))
        .unwrap_or(0.0)
        .max(item.radius)
        .max(0.01)
}

impl ItemTypePlugin for TubePlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn on_device_recreated(&mut self, _device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {
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
            .downcast_ref::<Vec<TubeItem>>()
            .expect("tube collection is the SceneFrame field");
        let refs = ctx
            .ref_items
            .and_then(|r| r.as_any().downcast_ref::<Vec<TubeRefItem>>())
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        self.pick_items.clear();
        self.pick_items.extend_from_slice(items);
        if items.is_empty() && refs.is_empty() {
            return Vec::new();
        }
        let gpu = self
            .gpu
            .get_or_insert_with(|| CurveMeshGpu::new(device, ctx.resources, "tube"));

        for item in items {
            if item.settings.hidden || item.positions.is_empty() || item.strip_lengths.is_empty() {
                continue;
            }
            let wireframe = ctx.wireframe_mode || item.settings.wireframe;
            let mut gpu_data = ctx
                .resources
                .upload_tube_per_frame(device, queue, item, wireframe);
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
        // matrix and pick id come from the reference, so one stored tube can be
        // drawn twice at two places under two ids.
        for ref_item in refs {
            if ref_item.settings.hidden {
                continue;
            }
            let Some(entry) = ctx.resources.content.tube_store.get(ref_item.source) else {
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
            let radius_px = radius_in_pixels(&item.positions, max_radius(item), ctx);
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
}

#[cfg(test)]
impl TubePlugin {
    /// Number of items the last `prepare` produced draw data for.
    pub(crate) fn drawn_count(&self) -> usize {
        self.frame.len()
    }
}
