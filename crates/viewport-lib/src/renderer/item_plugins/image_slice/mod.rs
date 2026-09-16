//! The image slice item type as an [`ItemTypePlugin`]: an axis-aligned,
//! colourmapped slice of an uploaded 3D volume, drawn as a shader-generated
//! quad. Consumers submit [`ImageSliceItem`]s on `SceneFrame::image_slices`;
//! the renderer routes that field to this plugin.

mod pipeline;

use crate::plugin_api::pick_helpers::{project_to_screen, segment_in_rect};
use crate::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, RectPickContext,
};
use crate::renderer::{ImageSliceItem, PickHit, PickId, PickMask, SliceAxis};
use crate::resources::HDR_COLOR_FORMAT;

pub(crate) const TYPE_NAME: &str = "viewport.image_slice";

impl PluginItemCollection for Vec<ImageSliceItem> {
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
pub(crate) struct ImageSlicePlugin {
    gpu: Option<pipeline::ImageSliceGpu>,
    /// Per visible item, rebuilt each prepare.
    frame: Vec<pipeline::ImageSliceFrame>,
    /// All items from the last prepared frame (hidden included, matching the
    /// CPU pick cache), for the CPU pick and rect-pick answers.
    pick_items: Vec<ImageSliceItem>,
    /// Whether the frame's selection outline is active; the mask hook draws
    /// nothing when it is off.
    outline_active: bool,
}

impl ItemTypePlugin for ImageSlicePlugin {
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
            .downcast_ref::<Vec<ImageSliceItem>>()
            .expect("image slice collection is the SceneFrame field");
        self.pick_items.clear();
        self.pick_items.extend_from_slice(items);
        if items.is_empty() {
            return Vec::new();
        }
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::ImageSliceGpu::new(device, ctx.resources));
        for item in items {
            if item.settings.hidden {
                continue;
            }
            if let Some(entry) = gpu.upload_item(device, queue, ctx.resources, item) {
                self.frame.push(entry);
            }
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
            pass.set_bind_group(1, &entry.bind_group, &[]);
            pass.draw(0..6, 0..1);
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
            pass.set_bind_group(1, &entry.bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
    }

    fn pick(&self, ray: &PickRay, ctx: &PickContext) -> Option<(f32, PickHit)> {
        if !ctx.mask.intersects(PickMask::OBJECT) {
            return None;
        }
        let mut best: Option<(f32, PickHit)> = None;
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE {
                continue;
            }
            let (axis_idx, plane_pos) = slice_plane(item);
            let plane_n = {
                let mut n = glam::Vec3::ZERO;
                n[axis_idx] = 1.0;
                n
            };
            let denom = plane_n.dot(ray.direction);
            if denom.abs() < 1e-6 {
                continue;
            }
            let toi = (plane_pos - ray.origin[axis_idx]) / denom;
            if toi <= 0.0 {
                continue;
            }
            let hit_pos = ray.origin + ray.direction * toi;
            let [bmin, bmax] = [item.bbox_min, item.bbox_max];
            let in_bounds = (0..3)
                .filter(|&i| i != axis_idx)
                .all(|i| hit_pos[i] >= bmin[i] - 1e-4 && hit_pos[i] <= bmax[i] + 1e-4);
            if in_bounds && best.as_ref().is_none_or(|(t, _)| toi < *t) {
                #[allow(deprecated)]
                let hit = PickHit {
                    id: item.settings.pick_id.0,
                    sub_object: None,
                    world_pos: hit_pos,
                    normal: plane_n,
                    scalar_value: None,
                    sub_object_world_pos: None,
                };
                best = Some((toi, hit));
            }
        }
        best
    }

    fn pick_rect(&self, ctx: &RectPickContext) -> Vec<PickId> {
        if !ctx.mask.intersects(PickMask::OBJECT) {
            return Vec::new();
        }
        let in_rect = |p: glam::Vec2| {
            p.x >= ctx.rect_min.x
                && p.x <= ctx.rect_max.x
                && p.y >= ctx.rect_min.y
                && p.y <= ctx.rect_max.y
        };
        let mut out = Vec::new();
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE {
                continue;
            }
            let corners = slice_corners(item);
            let sc: [Option<glam::Vec2>; 4] = std::array::from_fn(|i| {
                project_to_screen(
                    glam::Vec3::from(corners[i]),
                    ctx.view_proj,
                    ctx.viewport_size,
                )
            });
            let hit = sc.iter().any(|p| p.is_some_and(in_rect))
                || (0..4).any(|i| match (sc[i], sc[(i + 1) % 4]) {
                    (Some(a), Some(b)) => segment_in_rect(a, b, ctx.rect_min, ctx.rect_max),
                    (Some(a), None) => in_rect(a),
                    (None, Some(b)) => in_rect(b),
                    (None, None) => false,
                });
            if hit {
                out.push(item.settings.pick_id);
            }
        }
        out
    }

    fn render_pick(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &PickPassContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        if !ctx.mask.intersects(PickMask::OBJECT) {
            return;
        }
        let Some(gpu) = &self.gpu else { return };
        let mut bound = false;
        for entry in &self.frame {
            let Some((_, pick_bg)) = &entry.pick else {
                continue;
            };
            if !bound {
                pass.set_pipeline(&gpu.pick_pipeline);
                bound = true;
            }
            pass.set_bind_group(1, &entry.bind_group, &[]);
            pass.set_bind_group(2, pick_bg, &[]);
            pass.draw(0..6, 0..1);
        }
    }
}

/// The slice's plane: the axis it is perpendicular to and its world position
/// along that axis.
fn slice_plane(item: &ImageSliceItem) -> (usize, f32) {
    let [bmin, bmax] = [item.bbox_min, item.bbox_max];
    let t = item.offset;
    match item.axis {
        SliceAxis::X => (0, bmin[0] + t * (bmax[0] - bmin[0])),
        SliceAxis::Y => (1, bmin[1] + t * (bmax[1] - bmin[1])),
        SliceAxis::Z => (2, bmin[2] + t * (bmax[2] - bmin[2])),
    }
}

/// The slice quad's four world-space corners.
fn slice_corners(item: &ImageSliceItem) -> [[f32; 3]; 4] {
    let [bmin, bmax] = [item.bbox_min, item.bbox_max];
    let (axis_idx, pos) = slice_plane(item);
    match axis_idx {
        0 => [
            [pos, bmin[1], bmin[2]],
            [pos, bmax[1], bmin[2]],
            [pos, bmax[1], bmax[2]],
            [pos, bmin[1], bmax[2]],
        ],
        1 => [
            [bmin[0], pos, bmin[2]],
            [bmax[0], pos, bmin[2]],
            [bmax[0], pos, bmax[2]],
            [bmin[0], pos, bmax[2]],
        ],
        _ => [
            [bmin[0], bmin[1], pos],
            [bmax[0], bmin[1], pos],
            [bmax[0], bmax[1], pos],
            [bmin[0], bmax[1], pos],
        ],
    }
}
