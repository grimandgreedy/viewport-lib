//! The GPU implicit surface item type as an [`ItemTypePlugin`]: a set of
//! signed-distance primitives combined by a blend mode and ray-marched on a
//! full-screen quad. Consumers submit [`GpuImplicitItem`]s on
//! `SceneFrame::gpu_implicit`; the renderer routes that field to this plugin.

mod pipeline;

use crate::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, RectPickContext,
};
use crate::renderer::{GpuImplicitItem, PickHit, PickId, PickMask};
use crate::resources::{HDR_COLOR_FORMAT, ImplicitBlendMode, ImplicitPrimitive};

pub(crate) const TYPE_NAME: &str = "viewport.gpu_implicit";

impl PluginItemCollection for Vec<GpuImplicitItem> {
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

/// The march parameters and primitive set of one prepared item, snapshotted
/// for the out-of-band CPU pick and rect-pick answers.
struct ImplicitPickItem {
    id: u64,
    primitives: Vec<ImplicitPrimitive>,
    blend_mode: ImplicitBlendMode,
    max_steps: u32,
    step_scale: f32,
    hit_threshold: f32,
    max_distance: f32,
}

#[derive(Default)]
pub(crate) struct GpuImplicitPlugin {
    gpu: Option<pipeline::GpuImplicitGpu>,
    /// Per drawn item, rebuilt each prepare.
    frame: Vec<pipeline::GpuImplicitFrame>,
    /// The pickable subset of the same items, for the CPU pick paths.
    pick_items: Vec<ImplicitPickItem>,
    /// Whether the frame's selection outline is active; the mask hook draws
    /// nothing when it is off.
    outline_active: bool,
}

impl ItemTypePlugin for GpuImplicitPlugin {
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
        self.pick_items.clear();
        self.outline_active = ctx.outline_selected;
        let items = items
            .as_any()
            .downcast_ref::<Vec<GpuImplicitItem>>()
            .expect("gpu implicit collection is the SceneFrame field");
        if items.is_empty() {
            return Vec::new();
        }
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::GpuImplicitGpu::new(device, ctx.resources));
        for item in items {
            if item.settings.hidden || item.primitives.is_empty() {
                continue;
            }
            self.frame.push(gpu.upload_item(device, queue, item));
            if item.settings.pick_id != PickId::NONE {
                self.pick_items.push(ImplicitPickItem {
                    id: item.settings.pick_id.0,
                    primitives: item.primitives.clone(),
                    blend_mode: item.blend_mode,
                    max_steps: item.march_options.max_steps,
                    step_scale: item.march_options.step_scale,
                    hit_threshold: item.march_options.hit_threshold,
                    max_distance: item.march_options.max_distance,
                });
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
            let Some((toi, world_pos)) = march(ray.origin, ray.direction, item) else {
                continue;
            };
            if best.as_ref().is_none_or(|(t, _)| toi < *t) {
                #[allow(deprecated)]
                let hit = PickHit {
                    id: item.id,
                    sub_object: None,
                    world_pos,
                    normal: glam::Vec3::Z,
                    scalar_value: None,
                    sub_object_world_pos: None,
                };
                best = Some((toi, hit));
            }
        }
        best
    }

    /// Conservative screen-space test: project a bounding sphere per primitive
    /// and hit the item when any of the sphere's projected box corners lands in
    /// the rect. Approximate (the marched surface is smaller than the bound)
    /// but avoids per-pixel SDF marching for a rect query.
    fn pick_rect(&self, ctx: &RectPickContext) -> crate::renderer::PickRectResult {
        let mut result = crate::renderer::PickRectResult::default();
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
            let mut hit = false;
            'prim_loop: for prim in &item.primitives {
                let (centre, radius) = match prim.kind {
                    1 => {
                        // Sphere
                        let c = glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]);
                        (c, prim.params[3].abs())
                    }
                    2 => {
                        // Box: centre + the half-extent diagonal as radius
                        let c = glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]);
                        let h = glam::Vec3::new(prim.params[4], prim.params[5], prim.params[6]);
                        (c, h.length())
                    }
                    3 => {
                        // Plane: not bounded, skip.
                        continue;
                    }
                    4 => {
                        // Capsule: segment midpoint + (half-length + radius)
                        let a = glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]);
                        let b = glam::Vec3::new(prim.params[4], prim.params[5], prim.params[6]);
                        let r = prim.params[3].abs();
                        ((a + b) * 0.5, (b - a).length() * 0.5 + r)
                    }
                    _ => continue,
                };
                for dx in [-radius, radius] {
                    for dy in [-radius, radius] {
                        for dz in [-radius, radius] {
                            let corner = centre + glam::Vec3::new(dx, dy, dz);
                            let projected = crate::plugin_api::pick_helpers::project_to_screen(
                                corner,
                                ctx.view_proj,
                                ctx.viewport_size,
                            );
                            if projected.is_some_and(in_rect) {
                                hit = true;
                                break 'prim_loop;
                            }
                        }
                    }
                }
            }
            if hit {
                result.objects.push(item.id);
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

#[cfg(test)]
impl GpuImplicitPlugin {
    /// Number of items the last `prepare` produced draw data for.
    pub(crate) fn drawn_count(&self) -> usize {
        self.frame.len()
    }
}

// ---------------------------------------------------------------------------
// CPU SDF evaluation, mirroring implicit.wgsl
// ---------------------------------------------------------------------------

/// Evaluate one primitive's signed distance from `p`.
fn eval_primitive(p: glam::Vec3, prim: &ImplicitPrimitive) -> f32 {
    match prim.kind {
        1 => {
            // Sphere: centre=params[0..3], radius=params[3]
            let centre = glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]);
            (p - centre).length() - prim.params[3]
        }
        2 => {
            // Box: centre=params[0..3], half-extents=params[4..7]
            let centre = glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]);
            let half = glam::Vec3::new(prim.params[4], prim.params[5], prim.params[6]);
            let q = (p - centre).abs() - half;
            q.max(glam::Vec3::ZERO).length() + q.x.max(q.y).max(q.z).min(0.0)
        }
        3 => {
            // Plane: normal=params[0..3], offset=params[3]
            let n =
                glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]).normalize_or_zero();
            p.dot(n) + prim.params[3]
        }
        4 => {
            // Capsule: a=params[0..3], radius=params[3], b=params[4..7]
            let a = glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]);
            let r = prim.params[3];
            let b = glam::Vec3::new(prim.params[4], prim.params[5], prim.params[6]);
            let pa = p - a;
            let ba = b - a;
            let h = (pa.dot(ba) / ba.dot(ba).max(1e-10)).clamp(0.0, 1.0);
            (pa - ba * h).length() - r
        }
        _ => f32::MAX,
    }
}

/// Polynomial smooth minimum.
#[inline]
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    a * h + b * (1.0 - h) - k * h * (1.0 - h)
}

/// Evaluate the combined SDF for all primitives in one item.
fn scene_sdf(p: glam::Vec3, item: &ImplicitPickItem) -> f32 {
    let mut d = item.max_distance;
    for (i, prim) in item.primitives.iter().enumerate() {
        let pd = eval_primitive(p, prim);
        match item.blend_mode {
            ImplicitBlendMode::Union => {
                d = d.min(pd);
            }
            ImplicitBlendMode::SmoothUnion => {
                let k = if prim.blend > 0.0 { prim.blend } else { 1e-5 };
                d = smin(d, pd, k);
            }
            ImplicitBlendMode::Intersection => {
                if i == 0 {
                    d = pd;
                } else {
                    d = d.max(pd);
                }
            }
        }
    }
    d
}

/// CPU ray-march against the SDF. Returns `(toi, world_pos)` on hit.
fn march(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    item: &ImplicitPickItem,
) -> Option<(f32, glam::Vec3)> {
    let max_steps = item.max_steps.min(512) as usize;
    let scale = item.step_scale.clamp(0.01, 1.0);
    let hit_thr = item.hit_threshold;
    let max_dist = item.max_distance;
    let min_step = hit_thr * 0.5;

    let mut t = 0.0f32;
    for _ in 0..max_steps {
        if t > max_dist {
            break;
        }
        let p = ray_origin + ray_dir * t;
        let d = scene_sdf(p, item);
        if d < hit_thr {
            return Some((t, p));
        }
        t += d.abs().max(min_step) * scale;
    }
    None
}
