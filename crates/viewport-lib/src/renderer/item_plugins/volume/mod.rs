//! The volume item type as an [`ItemTypePlugin`]: an uploaded 3D scalar field
//! ray-marched through its bounding box with colour and opacity transfer
//! functions. Consumers submit [`VolumeItem`]s on `SceneFrame::volumes`; the
//! renderer routes that field to this plugin.
//!
//! A wireframe volume draws no ray-march at all: the core line substrate
//! renders an oriented bounding box polyline for it instead, so this plugin
//! only skips the item.

mod pipeline;
pub(crate) mod types;

use crate::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, RectPickContext,
};
use crate::renderer::{PickHit, PickId, PickMask, SubObjectRef, VolumeItem};
use crate::resources::HDR_COLOR_FORMAT;

pub(crate) const TYPE_NAME: &str = "vpl.volume";

impl PluginItemCollection for Vec<VolumeItem> {
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
pub(crate) struct VolumePlugin {
    gpu: Option<pipeline::VolumeGpu>,
    /// Per drawn item, rebuilt each prepare.
    frame: Vec<pipeline::VolumeFrame>,
    /// All items from the last prepared frame, hidden included, matching what
    /// the CPU pick cache used to retain.
    pick_items: Vec<VolumeItem>,
    /// Whether the frame's selection outline is active; the mask hook draws
    /// nothing when it is off.
    outline_active: bool,
}

impl ItemTypePlugin for VolumePlugin {
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
        self.outline_active = ctx.outline_selected;
        let items = items
            .as_any()
            .downcast_ref::<Vec<VolumeItem>>()
            .expect("volume collection is the SceneFrame field");
        self.pick_items.clear();
        self.pick_items.extend_from_slice(items);
        if items.is_empty() {
            return Vec::new();
        }
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::VolumeGpu::new(device, queue, ctx.resources));
        // Under budget pressure, double the step size (half the sample count)
        // to cut the ray-march cost.
        let step_multiplier = if ctx.quality_reduced { 2.0 } else { 1.0 };
        for item in items {
            if item.settings.hidden {
                continue;
            }
            let wireframe = ctx.wireframe_mode || item.settings.wireframe;
            self.frame.push(gpu.upload_item(
                device,
                queue,
                ctx.resources,
                item,
                ctx.clip_objects,
                step_multiplier,
                wireframe,
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
        let Some(gpu) = &self.gpu else { return };
        if self.frame.is_empty() {
            return;
        }
        let mut bound = false;
        for entry in &self.frame {
            if entry.wireframe {
                continue;
            }
            if !bound {
                pass.set_pipeline(
                    gpu.pipeline
                        .for_format(ctx.target_format == HDR_COLOR_FORMAT),
                );
                bound = true;
            }
            bind_cube(pass, entry);
            pass.draw_indexed(0..36, 0, 0..1);
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
        if !self.outline_active {
            return;
        }
        let mut bound = false;
        for entry in &self.frame {
            if !entry.selected {
                continue;
            }
            if !bound {
                pass.set_pipeline(&gpu.mask_pipeline);
                bound = true;
            }
            bind_cube(pass, entry);
            pass.draw_indexed(0..36, 0, 0..1);
        }
    }

    fn pick(&self, ray: &PickRay, ctx: &PickContext) -> Option<(f32, PickHit)> {
        let wants_voxel = ctx.mask.intersects(PickMask::VOXEL);
        if !wants_voxel && !ctx.mask.intersects(PickMask::OBJECT) {
            return None;
        }
        let mut best: Option<(f32, PickHit)> = None;
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE {
                continue;
            }
            let Some(vol_data) = item.volume_data.as_deref() else {
                continue;
            };
            let Some(mut hit) = crate::interaction::query::picking::pick_volume_cpu(
                ray.origin,
                ray.direction,
                item.settings.pick_id.0,
                item,
                vol_data,
            ) else {
                continue;
            };
            let toi = (hit.world_pos - ray.origin).dot(ray.direction).max(0.0);
            if !wants_voxel {
                hit.sub_object = None;
            }
            if best.as_ref().is_none_or(|(t, _)| toi < *t) {
                best = Some((toi, hit));
            }
        }
        best
    }

    /// Project every in-threshold voxel centre and hit the item when any of
    /// them lands in the rect. Exact rather than conservative, at the cost of
    /// walking the grid.
    fn pick_rect(&self, ctx: &RectPickContext) -> crate::renderer::PickRectResult {
        let mut result = crate::renderer::PickRectResult::default();
        let wants_voxel = ctx.mask.intersects(PickMask::VOXEL);
        let wants_object = ctx.mask.intersects(PickMask::OBJECT);
        if !wants_voxel && !wants_object {
            return result;
        }
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE {
                continue;
            }
            let Some(vol_data) = item.volume_data.as_deref() else {
                continue;
            };
            let [nx, ny, nz] = vol_data.dims;
            if nx == 0 || ny == 0 || nz == 0 || vol_data.data.is_empty() {
                continue;
            }
            let mvp = ctx.view_proj * glam::Mat4::from_cols_array_2d(&item.model);
            let bbox_min = glam::Vec3::from(item.bbox_min);
            let bbox_max = glam::Vec3::from(item.bbox_max);
            let cell = (bbox_max - bbox_min) / glam::Vec3::new(nx as f32, ny as f32, nz as f32);
            let id = item.settings.pick_id.0;
            let mut item_hit = false;

            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let flat = (ix + iy * nx + iz * nx * ny) as usize;
                        let scalar = vol_data.data[flat];
                        if scalar.is_nan()
                            || scalar < item.threshold_min
                            || scalar > item.threshold_max
                        {
                            continue;
                        }
                        let centre = bbox_min
                            + cell
                                * glam::Vec3::new(
                                    ix as f32 + 0.5,
                                    iy as f32 + 0.5,
                                    iz as f32 + 0.5,
                                );
                        let projected = crate::plugin_api::pick_helpers::project_to_screen(
                            centre,
                            mvp,
                            ctx.viewport_size,
                        );
                        let in_rect = projected.is_some_and(|p| {
                            p.x >= ctx.rect_min.x
                                && p.x <= ctx.rect_max.x
                                && p.y >= ctx.rect_min.y
                                && p.y <= ctx.rect_max.y
                        });
                        if in_rect {
                            if wants_voxel {
                                result.elements.push((id, SubObjectRef::Voxel(flat as u32)));
                            }
                            item_hit = true;
                        }
                    }
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
        // Wireframe volumes render an OBB polyline instead of the ray-march,
        // so they are picked as polylines, not here.
        if !ctx.mask.intersects(PickMask::OBJECT | PickMask::VOXEL) {
            return;
        }
        let Some(gpu) = &self.gpu else { return };
        let mut bound = false;
        for entry in &self.frame {
            if entry.wireframe {
                continue;
            }
            let Some((_, pick_bg)) = &entry.pick else {
                continue;
            };
            if !bound {
                pass.set_pipeline(&gpu.pick_pipeline);
                bound = true;
            }
            bind_cube(pass, entry);
            pass.set_bind_group(2, pick_bg, &[]);
            pass.draw_indexed(0..36, 0, 0..1);
        }
    }

    /// The pick fragment writes the flat index of the first in-threshold voxel
    /// it marched to, so refining a hit is just relabelling that channel.
    fn resolve_sub_object(
        &self,
        _item: PickId,
        sub_primitive: u32,
        _world_pos: glam::Vec3,
        mask: PickMask,
    ) -> Option<SubObjectRef> {
        mask.intersects(PickMask::VOXEL)
            .then_some(SubObjectRef::Voxel(sub_primitive))
    }
    /// The volume's bounding box, transformed by its model. A ray-marched
    /// volume has no surface to trace, so the box is both its wireframe and,
    /// when selected, its only selection affordance.
    fn wireframe_polylines(
        &self,
        items: &dyn PluginItemCollection,
        ctx: &ItemFrameContext<'_>,
    ) -> Vec<crate::renderer::PolylineItem> {
        let Some(volumes) = items.as_any().downcast_ref::<Vec<VolumeItem>>() else {
            return Vec::new();
        };
        volumes
            .iter()
            .filter(|item| !item.settings.hidden && (ctx.wireframe_mode || item.settings.wireframe))
            .map(obb_polyline)
            .collect()
    }
}

/// Bind the per-item group-1 data and the cube proxy buffers shared by the
/// render, mask, and pick draws.
fn bind_cube(pass: &mut crate::gpu::RenderPass<'_>, entry: &pipeline::VolumeFrame) {
    pass.set_bind_group(1, &entry.bind_group, &[]);
    pass.set_vertex_buffer(0, entry.vertex_buffer.slice(..));
    pass.set_index_buffer(
        entry.index_buffer.slice(..),
        crate::gpu::IndexFormat::Uint32,
    );
}

/// The volume's bbox corners transformed by its model, as a box wireframe.
///
/// A `VolumeItem`'s bounds are axis-aligned in object space but its model may
/// rotate them, so this walks the eight corners through the matrix rather than
/// using [`aabb_wireframe_polyline`](crate::aabb_wireframe_polyline).
fn obb_polyline(item: &VolumeItem) -> crate::renderer::PolylineItem {
    let model = glam::Mat4::from_cols_array_2d(&item.model);
    let mn = glam::Vec3::from(item.bbox_min);
    let mx = glam::Vec3::from(item.bbox_max);
    let local = [
        glam::Vec3::new(mn.x, mn.y, mn.z),
        glam::Vec3::new(mx.x, mn.y, mn.z),
        glam::Vec3::new(mn.x, mx.y, mn.z),
        glam::Vec3::new(mx.x, mx.y, mn.z),
        glam::Vec3::new(mn.x, mn.y, mx.z),
        glam::Vec3::new(mx.x, mn.y, mx.z),
        glam::Vec3::new(mn.x, mx.y, mx.z),
        glam::Vec3::new(mx.x, mx.y, mx.z),
    ];
    let corners: [[f32; 3]; 8] =
        std::array::from_fn(|i| model.transform_point3(local[i]).to_array());
    crate::renderer::obb_wireframe_polyline(&corners, [0.75, 0.75, 0.75, 1.0])
}
