//! The polyline item type as an [`ItemTypePlugin`]: strips of world-space
//! points drawn as screen-space thick lines. Consumers submit [`PolylineItem`]s
//! on `SceneFrame::polylines`, or [`PolylineRefItem`]s on
//! `SceneFrame::polyline_refs` to draw a polyline uploaded once through
//! `upload_polyline`; the renderer routes both fields to this plugin.
//!
//! The pipelines this draws with belong to `resources`, not to the plugin: they
//! are the shared line substrate that isolines, scatter-volume bounds, volume
//! bounding boxes, clip-object outlines and the splat and sprite wireframe
//! overlays also render through. See [`pipeline`] for why.

mod decoration;
mod pipeline;
pub(crate) mod types;

use crate::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, RectPickContext,
};
use crate::renderer::{
    PickHit, PickId, PickMask, PickRectResult, PolylineItem, PolylineRefItem, SubObjectRef,
};
use crate::resources::{HDR_COLOR_FORMAT, PolylineKey};

pub(crate) const TYPE_NAME: &str = "viewport.polyline";

impl PluginItemCollection for Vec<PolylineItem> {
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

impl PluginItemCollection for Vec<PolylineRefItem> {
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
pub(crate) struct PolylinePlugin {
    gpu: Option<pipeline::PolylineGpu>,
    /// Per drawn item, rebuilt each prepare: the inline items first, then the
    /// references.
    frame: Vec<pipeline::PolylineFrame>,
    /// The vector-quantity decoration: arrow glyphs generated from the
    /// `node_vectors` / `edge_vectors` of the drawn items.
    decoration: decoration::Decoration,
    /// Every inline item from the last prepared frame, hidden included,
    /// matching what the CPU pick cache used to retain. Reference items are not
    /// here: their segments live on the GPU, so they answer the GPU pick only.
    pick_items: Vec<PolylineItem>,
}

impl ItemTypePlugin for PolylinePlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn on_device_recreated(&mut self, _device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {
        self.gpu = None;
        self.frame.clear();
        self.decoration.reset();
    }

    fn prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<crate::gpu::CommandBuffer> {
        self.frame.clear();
        self.decoration.clear_frame();
        let items = items
            .as_any()
            .downcast_ref::<Vec<PolylineItem>>()
            .expect("polyline collection is the SceneFrame field");
        let refs = ctx.refs_of::<PolylineRefItem>();
        self.pick_items.clear();
        self.pick_items.extend_from_slice(items);
        if items.is_empty() && refs.is_empty() {
            return Vec::new();
        }
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::PolylineGpu::new(device, ctx.resources));
        let vp_size = [ctx.viewport_size.x, ctx.viewport_size.y];

        for item in items {
            if item.settings.hidden || item.positions.is_empty() {
                continue;
            }
            let mut gpu_data = ctx
                .resources
                .upload_polyline_per_frame(device, queue, item, vp_size);
            gpu_data.wireframe = ctx.wireframe_mode || item.settings.wireframe;
            gpu_data.skip_clip = item.settings.ignore_clip;
            let pick_bind_group = (gpu_data.pick_id != PickId::NONE)
                .then(|| gpu.pick_bind_group(device, queue, gpu_data.pick_id));
            self.frame.push(pipeline::PolylineFrame {
                gpu: gpu_data,
                pick_bind_group,
                outlined: ctx.outline_selected && item.settings.selected,
            });

            self.decoration.add_for_item(device, queue, ctx, item);
        }

        // Pre-uploaded references. The model matrix lives at offset 0 of the
        // uniform and the viewport size at offset 96: the screen-space miter
        // expansion has to see the current viewport, not the placeholder the
        // upload used.
        let vp = [vp_size[0].max(1.0), vp_size[1].max(1.0)];
        for ref_item in refs {
            if ref_item.settings.hidden {
                continue;
            }
            let Some(entry) = ctx.resources.content.polyline_store.get(ref_item.source) else {
                continue;
            };
            let mut gpu_data = entry.clone();
            queue.write_buffer(
                &gpu_data._uniform_buf,
                0,
                bytemuck::bytes_of(&ref_item.model),
            );
            queue.write_buffer(&gpu_data._uniform_buf, 96, bytemuck::bytes_of(&vp));
            gpu_data.wireframe = ctx.wireframe_mode || ref_item.settings.wireframe;
            // The pick id comes from the reference, not from the upload: the
            // same stored polyline can be drawn twice under two ids.
            let pick_id = ref_item.settings.pick_id;
            let pick_bind_group =
                (pick_id != PickId::NONE).then(|| gpu.pick_bind_group(device, queue, pick_id));
            gpu_data.pick_id = pick_id;
            self.frame.push(pipeline::PolylineFrame {
                gpu: gpu_data,
                pick_bind_group,
                outlined: ctx.outline_selected && ref_item.settings.selected,
            });
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
        let is_hdr = ctx.target_format == HDR_COLOR_FORMAT;
        for entry in &self.frame {
            let gd = &entry.gpu;
            if gd.segment_count == 0 {
                continue;
            }
            let key = PolylineKey {
                skip_clip: gd.skip_clip,
                wireframe: gd.wireframe,
            };
            let pipeline = gpu.pipelines.get(key).for_format(is_hdr);
            pass.set_pipeline(pipeline);
            if gd.wireframe {
                let Some(wf_bg) = gd.wireframe_bind_group.as_ref() else {
                    continue;
                };
                pass.set_bind_group(1, wf_bg, &[]);
                pass.draw(0..2, 0..gd.segment_count);
            } else {
                pass.set_bind_group(1, &gd.bind_group, &[]);
                pass.set_vertex_buffer(0, gd.vertex_buffer.slice(..));
                pass.draw(0..6, 0..gd.segment_count);
            }
        }
        // The vector decoration draws after the lines it belongs to, the order
        // the shared scivis loop gave it.
        self.decoration.paint(pass, is_hdr);
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
        let mut bound = false;
        for entry in &self.frame {
            if !entry.outlined || entry.gpu.segment_count == 0 {
                continue;
            }
            if !bound {
                pass.set_pipeline(&gpu.mask_pipeline);
                bound = true;
            }
            pass.set_bind_group(1, &entry.gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, entry.gpu.vertex_buffer.slice(..));
            pass.draw(0..6, 0..entry.gpu.segment_count);
        }
    }

    /// Node and segment proximity, whichever lands closer.
    ///
    /// Both tests run: a click near a vertex answers with the node under
    /// `POLY_NODE`, and a click anywhere along a segment answers with the
    /// segment under `SEGMENT`, with `STRIP` folding either to the strip.
    fn pick(&self, ray: &PickRay, ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
        use crate::plugin_api::pick_helpers::pick_closest_polyline_segment;

        let wants_node = ctx.mask.intersects(PickMask::POLY_NODE);
        let wants_segment = ctx.mask.intersects(PickMask::SEGMENT);
        let wants_strip = ctx.mask.intersects(PickMask::STRIP);
        let wants_object = ctx.mask.intersects(PickMask::OBJECT);
        if !wants_node && !wants_segment && !wants_strip && !wants_object {
            return None;
        }
        let mut best: Option<(f32, PickHit)> = None;
        let mut consider = |toi: f32, hit: PickHit| {
            if best.as_ref().is_none_or(|(t, _)| toi < *t) {
                best = Some((toi, hit));
            }
        };

        // Node proximity.
        if wants_node || wants_strip || wants_object {
            for item in &self.pick_items {
                if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                    continue;
                }
                let radius_px = (item.line_width + 4.0).max(8.0);
                let Some(mut hit) = crate::interaction::query::picking::pick_gaussian_splat_cpu(
                    ctx.click_pos,
                    item.settings.pick_id.0,
                    &item.positions,
                    glam::Mat4::IDENTITY,
                    ctx.view_proj,
                    ctx.viewport_size,
                    radius_px,
                ) else {
                    continue;
                };
                let toi = (hit.world_pos - ray.origin).dot(ray.direction).max(0.0);
                if wants_node {
                    // `pick_gaussian_splat_cpu` already reports the node index.
                } else if wants_strip {
                    if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                        hit.sub_object = Some(SubObjectRef::Strip(
                            crate::renderer::picking::helpers::strip_for_node(
                                idx,
                                &item.strip_lengths,
                            ),
                        ));
                    }
                } else {
                    hit.sub_object = None;
                }
                consider(toi, hit);
            }
        }

        // Segment proximity: screen-space distance to the whole segment, so a
        // click anywhere along it registers, not just near the midpoint.
        if wants_segment || wants_strip || wants_object {
            for item in &self.pick_items {
                if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                    continue;
                }
                let threshold_px = (item.line_width / 2.0 + 4.0).max(4.0);
                let Some((seg_idx, world_pos)) = pick_closest_polyline_segment(
                    ctx.click_pos,
                    ctx.viewport_size,
                    ctx.view_proj,
                    &item.positions,
                    &item.strip_lengths,
                    threshold_px,
                ) else {
                    continue;
                };
                let toi = (world_pos - ray.origin).dot(ray.direction).max(0.0);
                let sub_object = if wants_segment {
                    Some(SubObjectRef::Segment(seg_idx))
                } else if wants_strip {
                    Some(SubObjectRef::Strip(
                        crate::renderer::picking::helpers::strip_for_segment(
                            seg_idx,
                            &item.strip_lengths,
                        ),
                    ))
                } else {
                    None
                };
                #[allow(deprecated)]
                consider(
                    toi,
                    PickHit {
                        id: item.settings.pick_id.0,
                        sub_object,
                        world_pos,
                        normal: glam::Vec3::Z,
                        scalar_value: None,
                        sub_object_world_pos: None,
                    },
                );
            }
        }
        best
    }

    fn pick_rect(&self, ctx: &RectPickContext<'_>) -> PickRectResult {
        use crate::plugin_api::pick_helpers::{project_to_screen, segment_in_rect};
        use crate::renderer::picking::helpers::{strip_for_node, strip_for_segment};

        let mut result = PickRectResult::default();
        let wants_node = ctx.mask.intersects(PickMask::POLY_NODE);
        let wants_segment = ctx.mask.intersects(PickMask::SEGMENT);
        let wants_strip = ctx.mask.intersects(PickMask::STRIP);
        let wants_object = ctx.mask.intersects(PickMask::OBJECT);
        if !wants_node && !wants_segment && !wants_strip && !wants_object {
            return result;
        }
        let in_rect = |p: glam::Vec2| {
            p.x >= ctx.rect_min.x
                && p.x <= ctx.rect_max.x
                && p.y >= ctx.rect_min.y
                && p.y <= ctx.rect_max.y
        };
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                continue;
            }
            let id = item.settings.pick_id.0;
            let mut item_hit = false;
            // Strips are collected in a set: one strip covers many nodes and
            // segments, and the caller wants it reported once.
            let mut strips_hit = std::collections::HashSet::<u32>::new();
            let screen: Vec<Option<glam::Vec2>> = item
                .positions
                .iter()
                .map(|p| project_to_screen(glam::Vec3::from(*p), ctx.view_proj, ctx.viewport_size))
                .collect();

            // Node pass.
            if wants_node || wants_strip || wants_object {
                for (node_idx, p) in screen.iter().enumerate() {
                    if !p.is_some_and(in_rect) {
                        continue;
                    }
                    item_hit = true;
                    if wants_node {
                        result
                            .elements
                            .push((id, SubObjectRef::Point(node_idx as u32)));
                    } else if wants_strip {
                        strips_hit.insert(strip_for_node(node_idx as u32, &item.strip_lengths));
                    }
                }
            }

            // Segment pass: full segment against the rectangle, so a segment
            // crossing the box counts even when neither endpoint is inside.
            if wants_segment || (wants_strip && !wants_node) || wants_object {
                let mut try_segment = |seg: u32, a: usize, b: usize, item_hit: &mut bool| {
                    let (Some(sa), Some(sb)) = (screen[a], screen[b]) else {
                        return;
                    };
                    if !segment_in_rect(sa, sb, ctx.rect_min, ctx.rect_max) {
                        return;
                    }
                    *item_hit = true;
                    if wants_segment {
                        result.elements.push((id, SubObjectRef::Segment(seg)));
                    } else if wants_strip {
                        strips_hit.insert(strip_for_segment(seg, &item.strip_lengths));
                    }
                };
                if item.strip_lengths.is_empty() {
                    for j in 0..item.positions.len().saturating_sub(1) {
                        try_segment(j as u32, j, j + 1, &mut item_hit);
                    }
                } else {
                    let mut node_off = 0usize;
                    let mut seg_off = 0u32;
                    for &len in &item.strip_lengths {
                        let len = len as usize;
                        for j in 0..len.saturating_sub(1) {
                            try_segment(
                                seg_off + j as u32,
                                node_off + j,
                                node_off + j + 1,
                                &mut item_hit,
                            );
                        }
                        seg_off += len.saturating_sub(1) as u32;
                        node_off += len;
                    }
                }
            }

            if wants_strip {
                for s in strips_hit {
                    result.elements.push((id, SubObjectRef::Strip(s)));
                }
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
        if !ctx.mask.intersects(
            PickMask::OBJECT | PickMask::POLY_NODE | PickMask::SEGMENT | PickMask::STRIP,
        ) {
            return;
        }
        let Some(gpu) = &self.gpu else { return };
        let mut bound = false;
        for entry in &self.frame {
            let Some(pick_bg) = &entry.pick_bind_group else {
                continue;
            };
            if entry.gpu.segment_count == 0 {
                continue;
            }
            if !bound {
                pass.set_pipeline(&gpu.pick_pipeline);
                bound = true;
            }
            pass.set_bind_group(1, &entry.gpu.bind_group, &[]);
            pass.set_bind_group(2, pick_bg, &[]);
            pass.set_vertex_buffer(0, entry.gpu.vertex_buffer.slice(..));
            pass.draw(0..6, 0..entry.gpu.segment_count);
        }
    }

    fn resolve_sub_object(
        &self,
        pick_id: PickId,
        primitive_index: u32,
        _world_pos: glam::Vec3,
        mask: PickMask,
    ) -> Option<SubObjectRef> {
        // The pick fragment writes the segment index into the primitive
        // channel; no device feature involved.
        if mask.intersects(PickMask::STRIP) {
            let strip = self
                .frame
                .iter()
                .find(|f| f.gpu.pick_id == pick_id)
                .map(|f| {
                    crate::renderer::picking::helpers::strip_for_segment(
                        primitive_index,
                        &f.gpu.strip_lengths,
                    )
                })
                .unwrap_or(0);
            return Some(SubObjectRef::Strip(strip));
        }
        mask.intersects(PickMask::SEGMENT | PickMask::POLY_NODE)
            .then_some(SubObjectRef::Segment(primitive_index))
    }
}

#[cfg(test)]
impl PolylinePlugin {
    /// Number of items the last `prepare` produced draw data for.
    pub(crate) fn drawn_count(&self) -> usize {
        self.frame.len()
    }
}
