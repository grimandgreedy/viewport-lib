//! The point cloud item type as an [`ItemTypePlugin`]: a set of world-space
//! points drawn as screen-space discs, coloured by a scalar through a colourmap
//! or by per-point colours. Consumers submit [`PointCloudItem`]s on
//! `SceneFrame::point_clouds`, or [`PointCloudRefItem`]s on
//! `SceneFrame::point_cloud_refs` to draw a cloud uploaded once through
//! `upload_point_cloud`; the renderer routes both fields to this plugin.

mod pipeline;

use crate::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, RectPickContext,
};
use crate::renderer::{
    PickHit, PickId, PickMask, PickRectResult, PointCloudItem, PointCloudRefItem, SubObjectRef,
};
use crate::resources::HDR_COLOR_FORMAT;

pub(crate) const TYPE_NAME: &str = "viewport.point_cloud";

impl PluginItemCollection for Vec<PointCloudItem> {
    fn len(&self) -> usize {
        self.len()
    }
    fn item_settings(&self, index: usize) -> &crate::scene::material::ItemSettings {
        &self[index].settings
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl PluginItemCollection for Vec<PointCloudRefItem> {
    fn len(&self) -> usize {
        self.len()
    }
    fn item_settings(&self, index: usize) -> &crate::scene::material::ItemSettings {
        &self[index].settings
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Default)]
pub(crate) struct PointCloudPlugin {
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

    fn on_device_recreated(&mut self, _device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {
        self.gpu = None;
        self.frame.clear();
        self.outlines.clear();
    }

    fn prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<crate::gpu::CommandBuffer> {
        self.frame.clear();
        self.outlines.clear();
        let items = items
            .as_any()
            .downcast_ref::<Vec<PointCloudItem>>()
            .expect("point cloud collection is the SceneFrame field");
        let refs = ctx
            .ref_items
            .and_then(|r| r.as_any().downcast_ref::<Vec<PointCloudRefItem>>())
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        self.pick_items.clear();
        self.pick_items.extend_from_slice(items);
        if items.is_empty() && refs.is_empty() {
            return Vec::new();
        }
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::PointCloudGpu::new(device, ctx.resources));

        for item in items {
            if item.settings.hidden || item.positions.is_empty() {
                continue;
            }
            let gpu_data = ctx
                .resources
                .upload_point_cloud_per_frame(device, queue, item);
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
            let Some(entry) = ctx.resources.content.point_cloud_store.get(ref_item.source) else {
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
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &dyn PluginItemCollection,
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
        pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &OutlineMaskContext<'_>,
        _items: &dyn PluginItemCollection,
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
            let Some(mut hit) = crate::interaction::query::picking::pick_point_cloud_cpu(
                ctx.click_pos,
                item.settings.pick_id.0,
                item,
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
                let Some(p) = crate::plugin_api::pick_helpers::project_to_screen(
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
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &PickPassContext<'_>,
        _items: &dyn PluginItemCollection,
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
}

#[cfg(test)]
impl PointCloudPlugin {
    /// Number of items the last `prepare` produced draw data for.
    pub(crate) fn drawn_count(&self) -> usize {
        self.frame.len()
    }
}

/// Build this frame's outline coverage: every point of a selected item, or just
/// the sub-selected points of an item that is not itself selected.
fn build_outlines(
    device: &crate::gpu::Device,
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
                ctx.resources,
                item.model,
                ctx.viewport_size,
                pixel_radius,
                &item.positions,
            ));
        } else if item.settings.pick_id != PickId::NONE {
            let selected: Vec<[f32; 3]> = ctx
                .sub_selection
                .iter()
                .flat_map(|s| s.items.iter())
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
                ctx.resources,
                item.model,
                ctx.viewport_size,
                pixel_radius,
                &selected,
            ));
        }
    }
    outlines
}
