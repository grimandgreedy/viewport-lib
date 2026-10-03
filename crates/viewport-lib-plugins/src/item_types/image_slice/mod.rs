//! The image slice item type as an [`ItemTypePlugin`]: an axis-aligned,
//! colourmapped slice of an uploaded 3D volume, drawn as a shader-generated
//! quad. Consumers submit [`ImageSliceItem`]s with
//! `frame.scene.items_mut::<ImageSliceItem>()`;
//! the renderer routes that field to this plugin.

mod pipeline;
mod types;

pub use types::{ImageSliceItem, SliceAxis};
use viewport_lib::gpu;
use viewport_lib::plugin_api::pick_helpers::{project_to_screen, segment_in_rect};
use viewport_lib::plugin_api::{
    ItemCollections, ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext,
    PickContext, PickPassContext, PickRay, PluginItem, RectPickContext,
};
use viewport_lib::renderer::{PickHit, PickId, PickMask};
use viewport_lib::resources::HDR_COLOR_FORMAT;

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.image_slice";

/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::item_types::shader::{scene_shader, wgsl_source};
    vec![
        (
            "image_slice.wgsl",
            scene_shader(&[], wgsl_source!("image_slice")),
        ),
        (
            "image_slice_pick.wgsl",
            scene_shader(&[], wgsl_source!("image_slice_pick")),
        ),
        (
            "image_slice_mask.wgsl",
            scene_shader(&[], wgsl_source!("image_slice_mask")),
        ),
    ]
}

impl PluginItem for ImageSliceItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

#[derive(Default)]
pub struct ImageSlicePlugin {
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

    fn warm(&mut self, device: &gpu::Device, resources: &viewport_lib::DeviceResources) {
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::ImageSliceGpu::new(device, resources));
        gpu.pipelines.request_all();
    }

    fn on_device_recreated(&mut self, _device: &gpu::Device, _queue: &gpu::Queue) {
        self.gpu = None;
        self.frame.clear();
    }

    fn prepare(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &ItemCollections<'_>,
    ) -> Vec<gpu::CommandBuffer> {
        self.frame.clear();
        self.outline_active = ctx.outline_selected;
        let items = items.of::<ImageSliceItem>();
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
        pass: &mut gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let Some(gpu) = &self.gpu else { return };
        if self.frame.is_empty() {
            return;
        }
        // Still compiling: the slices draw next frame.
        let colour = if ctx.target_format == HDR_COLOR_FORMAT {
            pipeline::COLOUR_HDR
        } else {
            pipeline::COLOUR_LDR
        };
        let Some(pl) = gpu.pipelines.get(colour) else {
            return;
        };
        pass.set_pipeline(pl);
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
        pass: &mut gpu::RenderPass<'_>,
        _ctx: &OutlineMaskContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let Some(gpu) = self.gpu.as_ref().filter(|g| g.drawn()) else {
            return;
        };
        if !self.outline_active {
            return;
        }
        let mut bound = false;
        for entry in &self.frame {
            if !entry.selected {
                continue;
            }
            if !bound {
                let Some(pl) = gpu.pipelines.get(pipeline::MASK) else {
                    return;
                };
                pass.set_pipeline(pl);
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
                let hit = PickHit::object_hit(item.settings.pick_id.0, hit_pos, plane_n);
                best = Some((toi, hit));
            }
        }
        best
    }

    fn pick_rect(&self, ctx: &RectPickContext) -> viewport_lib::renderer::PickRectResult {
        let mut result = viewport_lib::renderer::PickRectResult::default();
        if !ctx.mask.intersects(PickMask::OBJECT) {
            return result;
        }
        let in_rect = |p: glam::Vec2| {
            p.x >= ctx.rect_min.x
                && p.x <= ctx.rect_max.x
                && p.y >= ctx.rect_min.y
                && p.y <= ctx.rect_max.y
        };
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
                result.objects.push(item.settings.pick_id.0);
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
        if !ctx.mask.intersects(PickMask::OBJECT) {
            return;
        }
        let Some(gpu) = self.gpu.as_ref().filter(|g| g.drawn()) else {
            return;
        };
        let mut bound = false;
        for entry in &self.frame {
            let Some((_, pick_bg)) = &entry.pick else {
                continue;
            };
            if !bound {
                let Some(pl) = gpu.pipelines.get(pipeline::PICK) else {
                    return;
                };
                pass.set_pipeline(pl);
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
