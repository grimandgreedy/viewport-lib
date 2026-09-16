//! The GPU marching cubes item type as an [`ItemTypePlugin`]: an isosurface
//! extracted from an uploaded scalar field by three compute passes each frame
//! and drawn with indirect draws. Consumers submit [`GpuMarchingCubesItem`]s on
//! `SceneFrame::gpu_mc_items`; the renderer routes that field to this plugin.
//!
//! The volumes stay in the lib's store (`upload_volume_for_mc` and `McVolumeId`
//! are consumer API); the plugin owns the pipelines and the per-frame extraction.

mod pipeline;

use crate::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, RectPickContext, ShadowCastContext,
};
use crate::renderer::{GpuMarchingCubesItem, PickHit, PickId, PickMask};
use crate::resources::HDR_COLOR_FORMAT;

pub(crate) const TYPE_NAME: &str = "viewport.gpu_marching_cubes";

impl PluginItemCollection for Vec<GpuMarchingCubesItem> {
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

/// The scalar field and isovalue of one prepared item, snapshotted for the
/// out-of-band CPU pick and rect-pick answers.
struct McPickItem {
    id: u64,
    isovalue: f32,
    volume_data: std::sync::Arc<crate::geometry::marching_cubes::VolumeData>,
}

#[derive(Default)]
pub(crate) struct GpuMarchingCubesPlugin {
    gpu: Option<pipeline::McGpu>,
    /// Per drawn item, rebuilt each prepare.
    frame: Vec<pipeline::McFrame>,
    /// Object-id uniforms for the pick pass, keyed alongside `frame`.
    pick_bgs: Vec<Option<(crate::gpu::Buffer, crate::gpu::BindGroup)>>,
    /// The pickable subset of the same items, for the CPU pick paths.
    pick_items: Vec<McPickItem>,
    /// Whether the frame's selection outline is active; the mask hook draws
    /// nothing when it is off.
    outline_active: bool,
    /// The frame's global wireframe toggle, applied alongside the per-item flag.
    wireframe_mode: bool,
}

impl ItemTypePlugin for GpuMarchingCubesPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn on_device_recreated(&mut self, _device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {
        self.gpu = None;
        self.frame.clear();
        self.pick_bgs.clear();
    }

    fn prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<crate::gpu::CommandBuffer> {
        self.frame.clear();
        self.pick_bgs.clear();
        self.pick_items.clear();
        self.outline_active = ctx.outline_selected;
        self.wireframe_mode = ctx.wireframe_mode;
        let items = items
            .as_any()
            .downcast_ref::<Vec<GpuMarchingCubesItem>>()
            .expect("gpu marching cubes collection is the SceneFrame field");
        if items.is_empty() {
            return Vec::new();
        }
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::McGpu::new(device, ctx.resources));

        let (frame, buf) = gpu.run_jobs(device, ctx.resources, items);
        self.frame = frame;

        for entry in &self.frame {
            self.pick_bgs
                .push(gpu.pick_bind_group(device, queue, entry.pick_id));
        }
        for job in items {
            if job.settings.pick_id != PickId::NONE {
                if let Some(cpu_data) = &job.cpu_data {
                    self.pick_items.push(McPickItem {
                        id: job.settings.pick_id.0,
                        isovalue: job.isovalue,
                        volume_data: cpu_data.clone(),
                    });
                }
            }
        }

        buf.into_iter().collect()
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
            if entry.hidden {
                continue;
            }
            if entry.wireframe || self.wireframe_mode {
                pass.set_pipeline(gpu.wireframe_pipeline.for_format(is_hdr));
                for (slab, wire_bg) in entry.slabs.iter().zip(entry.wire_slab_bgs.iter()) {
                    pass.set_bind_group(1, wire_bg, &[]);
                    pass.draw_indirect(&slab.wire_indirect_buf, 0);
                }
            } else {
                pass.set_pipeline(gpu.surface_pipeline.for_format(is_hdr));
                pass.set_bind_group(1, &entry.render_bg, &[]);
                for slab in &entry.slabs {
                    pass.set_vertex_buffer(0, slab.vertex_buf.slice(..));
                    pass.draw_indirect(&slab.indirect_buf, 0);
                }
            }
        }
    }

    fn draws_ldr(&self) -> bool {
        true
    }

    /// Always casts from the solid slab data regardless of `wireframe`:
    /// shadows reflect the actual surface, not its display mode. MC vertices
    /// are world-space, so there is no group-1 bind group and no per-item
    /// cascade cull (the frame data carries no world AABB).
    fn cast_shadow_pass(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &ShadowCastContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let Some(gpu) = &self.gpu else { return };
        let mut bound = false;
        for entry in &self.frame {
            if entry.hidden || !entry.cast_shadows {
                continue;
            }
            if !bound {
                pass.set_pipeline(&gpu.shadow_pipeline);
                bound = true;
            }
            for slab in &entry.slabs {
                pass.set_vertex_buffer(0, slab.vertex_buf.slice(..));
                pass.draw_indirect(&slab.indirect_buf, 0);
            }
        }
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
            if entry.hidden || !entry.selected {
                continue;
            }
            if !bound {
                pass.set_pipeline(&gpu.mask_pipeline);
                bound = true;
            }
            for slab in &entry.slabs {
                pass.set_vertex_buffer(0, slab.vertex_buf.slice(..));
                pass.draw_indirect(&slab.indirect_buf, 0);
            }
        }
    }

    fn pick(&self, ray: &PickRay, ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
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

    /// Walk the cells where the scalar field straddles the isovalue (the cells
    /// the compute stage would emit triangles for) and hit the item when any
    /// such cell centre projects into the rect.
    fn pick_rect(&self, ctx: &RectPickContext<'_>) -> crate::renderer::PickRectResult {
        let mut result = crate::renderer::PickRectResult::default();
        if !ctx.mask.intersects(PickMask::OBJECT) {
            return result;
        }
        for item in &self.pick_items {
            let vol = &item.volume_data;
            let isovalue = item.isovalue;
            let [nx, ny, nz] = vol.dims;
            let origin = glam::Vec3::from(vol.origin);
            let spacing = glam::Vec3::from(vol.spacing);

            let mut hit = false;
            'mc_rect: for iz in 0..nz.saturating_sub(1) {
                for iy in 0..ny.saturating_sub(1) {
                    for ix in 0..nx.saturating_sub(1) {
                        // A cell straddles the isovalue when not all 8 corners
                        // are on the same side. Check for both above and below.
                        let mut has_below = false;
                        let mut has_above = false;
                        'corners: for dz in 0u32..=1 {
                            for dy in 0u32..=1 {
                                for dx in 0u32..=1 {
                                    let s = vol.sample(ix + dx, iy + dy, iz + dz);
                                    if s < isovalue {
                                        has_below = true;
                                    } else {
                                        has_above = true;
                                    }
                                    if has_below && has_above {
                                        break 'corners;
                                    }
                                }
                            }
                        }
                        if !(has_below && has_above) {
                            continue;
                        }
                        let cell_centre = origin
                            + spacing
                                * glam::Vec3::new(
                                    ix as f32 + 0.5,
                                    iy as f32 + 0.5,
                                    iz as f32 + 0.5,
                                );
                        let projected = crate::plugin_api::pick_helpers::project_to_screen(
                            cell_centre,
                            ctx.view_proj,
                            ctx.viewport_size,
                        );
                        if projected.is_some_and(|p| {
                            p.x >= ctx.rect_min.x
                                && p.x <= ctx.rect_max.x
                                && p.y >= ctx.rect_min.y
                                && p.y <= ctx.rect_max.y
                        }) {
                            hit = true;
                            break 'mc_rect;
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
        for (entry, pick) in self.frame.iter().zip(self.pick_bgs.iter()) {
            if entry.hidden {
                continue;
            }
            let Some((_, pick_bg)) = pick else { continue };
            if !bound {
                pass.set_pipeline(&gpu.pick_pipeline);
                bound = true;
            }
            pass.set_bind_group(1, pick_bg, &[]);
            for slab in &entry.slabs {
                pass.set_vertex_buffer(0, slab.vertex_buf.slice(..));
                pass.draw_indirect(&slab.indirect_buf, 0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU volume ray-march, mirroring the compute isosurface extraction
// ---------------------------------------------------------------------------

/// Slab test: returns (t_enter, t_exit) for a ray vs axis-aligned box, or None.
fn ray_aabb_slab(
    ray_orig: glam::Vec3,
    ray_dir: glam::Vec3,
    bbox_min: glam::Vec3,
    bbox_max: glam::Vec3,
) -> Option<(f32, f32)> {
    // Avoid division by zero for axis-aligned rays.
    let inv = glam::Vec3::new(
        if ray_dir.x.abs() > 1e-30 {
            1.0 / ray_dir.x
        } else {
            f32::INFINITY * ray_dir.x.signum()
        },
        if ray_dir.y.abs() > 1e-30 {
            1.0 / ray_dir.y
        } else {
            f32::INFINITY * ray_dir.y.signum()
        },
        if ray_dir.z.abs() > 1e-30 {
            1.0 / ray_dir.z
        } else {
            f32::INFINITY * ray_dir.z.signum()
        },
    );
    let t1 = (bbox_min - ray_orig) * inv;
    let t2 = (bbox_max - ray_orig) * inv;
    let tmin = t1.min(t2);
    let tmax = t1.max(t2);
    let t_enter = tmin.x.max(tmin.y).max(tmin.z);
    let t_exit = tmax.x.min(tmax.y).min(tmax.z);
    if t_enter <= t_exit && t_exit >= 0.0 {
        Some((t_enter, t_exit))
    } else {
        None
    }
}

/// Bisect to refine the isovalue crossing between t_lo and t_hi (8 iterations).
fn bisect_mc_crossing(
    ray_orig: glam::Vec3,
    ray_dir: glam::Vec3,
    vol: &crate::geometry::marching_cubes::VolumeData,
    isovalue: f32,
    mut t_lo: f32,
    mut t_hi: f32,
) -> f32 {
    let s0 = crate::geometry::marching_cubes::trilinear_sample(
        vol,
        (ray_orig + ray_dir * t_lo).to_array(),
    ) - isovalue;
    let mut lo_sign = s0 < 0.0;
    for _ in 0..8 {
        let mid = (t_lo + t_hi) * 0.5;
        let s = crate::geometry::marching_cubes::trilinear_sample(
            vol,
            (ray_orig + ray_dir * mid).to_array(),
        ) - isovalue;
        if (s < 0.0) == lo_sign {
            t_lo = mid;
        } else {
            t_hi = mid;
            lo_sign = !lo_sign;
        }
    }
    (t_lo + t_hi) * 0.5
}

/// CPU ray-march against a MC isosurface. Returns `(toi, world_pos)` on hit.
///
/// Steps through the volume AABB at half-cell intervals and refines any
/// isovalue crossing to 8 bisection steps.
fn march(
    ray_orig: glam::Vec3,
    ray_dir: glam::Vec3,
    item: &McPickItem,
) -> Option<(f32, glam::Vec3)> {
    use crate::geometry::marching_cubes::trilinear_sample;

    let vol = &item.volume_data;
    let isovalue = item.isovalue;
    let [nx, ny, nz] = vol.dims;
    let origin = glam::Vec3::from(vol.origin);
    let spacing = glam::Vec3::from(vol.spacing);
    let extent = spacing * glam::Vec3::new(nx as f32, ny as f32, nz as f32);

    let (t_enter, t_exit) = ray_aabb_slab(ray_orig, ray_dir, origin, origin + extent)?;
    let t_start = t_enter.max(0.0);
    if t_start >= t_exit {
        return None;
    }

    // Step at half the smallest cell spacing so we don't skip thin features.
    let step = spacing.min_element() * 0.5;
    let mut t = t_start;
    let mut prev = trilinear_sample(vol, (ray_orig + ray_dir * t).to_array()) - isovalue;

    loop {
        t += step;
        if t > t_exit {
            break;
        }
        let p = ray_orig + ray_dir * t;
        let cur = trilinear_sample(vol, p.to_array()) - isovalue;
        if prev * cur <= 0.0 {
            // Sign change detected: bisect and return.
            let t_hit = bisect_mc_crossing(ray_orig, ray_dir, vol, isovalue, t - step, t);
            let world_pos = ray_orig + ray_dir * t_hit;
            return Some((t_hit, world_pos));
        }
        prev = cur;
    }
    None
}
