//! The GPU marching cubes item type as an [`ItemTypePlugin`]: an isosurface
//! extracted from an uploaded scalar field by three compute passes each frame
//! and drawn with indirect draws. Consumers submit [`GpuMarchingCubesItem`]s
//! with `frame.scene.submit::<GpuMarchingCubesItem>(..)`.
//!
//! The plugin owns the pipelines, the per-frame extraction, and the uploaded
//! volumes themselves: [`McVolumes::upload_volume_for_mc`](crate::McVolumes::upload_volume_for_mc)
//! hands back a [`McVolumeId`] that names a volume held here.

mod pipeline;
mod store;
mod types;

use viewport_lib::plugin_api::{
    ItemCollections, ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext,
    PickContext, PickPassContext, PickRay, PluginItem, RectPickContext, ShadowCastContext,
};
use viewport_lib::renderer::{PickHit, PickId, PickMask};
use viewport_lib::resources::HDR_COLOR_FORMAT;
use viewport_lib_geometry::marching_cubes::VolumeData;

use store::{McExternalScalarSource, McVolumeGpuData, McVolumeStore, build_mc_volume_gpu_data};

pub use types::{GpuMarchingCubesItem, McVolumeId};

/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::shader::{lit_shader, scene_shader, wgsl_source};
    use viewport_lib::plugin_api::shared_wgsl;
    vec![
        ("mc_classify.wgsl", wgsl_source!("mc_classify").to_string()),
        (
            "mc_prefix_sum.wgsl",
            wgsl_source!("mc_prefix_sum").to_string(),
        ),
        ("mc_generate.wgsl", wgsl_source!("mc_generate").to_string()),
        ("mc_shadow.wgsl", wgsl_source!("mc_shadow").to_string()),
        (
            "mc_surface.wgsl",
            lit_shader(&[shared_wgsl::SHARED_CSM_WGSL], wgsl_source!("mc_surface")),
        ),
        (
            "mc_wireframe.wgsl",
            scene_shader(&[], wgsl_source!("mc_wireframe")),
        ),
        (
            "mc_outline_mask.wgsl",
            scene_shader(&[], wgsl_source!("mc_outline_mask")),
        ),
        ("mc_pick.wgsl", scene_shader(&[], wgsl_source!("mc_pick"))),
    ]
}

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.gpu_marching_cubes";

impl PluginItem for GpuMarchingCubesItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

/// The scalar field and isovalue of one prepared item, snapshotted for the
/// out-of-band CPU pick and rect-pick answers.
struct McPickItem {
    id: u64,
    isovalue: f32,
    volume_data: std::sync::Arc<viewport_lib_geometry::marching_cubes::VolumeData>,
}

/// The registered item type. `install` builds one; a consumer taking only
/// this type registers it with `with_item_type_plugin`.
#[derive(Default)]
pub struct GpuMarchingCubesPlugin {
    /// The uploaded scalar volumes, owned by the type that triangulates them.
    volumes: McVolumeStore,
    gpu: Option<pipeline::McGpu>,
    /// Per drawn item, rebuilt each prepare.
    frame: Vec<pipeline::McFrame>,
    /// Object-id uniforms for the pick pass, keyed alongside `frame`.
    pick_bgs: Vec<Option<(viewport_lib::gpu::Buffer, viewport_lib::gpu::BindGroup)>>,
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

    fn resident_bytes(&self) -> u64 {
        self.volumes.allocated_bytes()
    }

    fn on_device_recreated(
        &mut self,
        _device: &viewport_lib::gpu::Device,
        _queue: &viewport_lib::gpu::Queue,
    ) {
        self.gpu = None;
        self.frame.clear();
        self.pick_bgs.clear();
    }

    fn prepare(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &ItemCollections<'_>,
    ) -> Vec<viewport_lib::gpu::CommandBuffer> {
        self.frame.clear();
        self.pick_bgs.clear();
        self.pick_items.clear();
        self.outline_active = ctx.outline_selected;
        self.wireframe_mode = ctx.wireframe_mode;
        let items = items.of::<GpuMarchingCubesItem>();
        if items.is_empty() {
            return Vec::new();
        }
        let volumes = &self.volumes;
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::McGpu::new(device, ctx.resources));

        let (frame, buf) = gpu.run_jobs(device, volumes, items);
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
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &ItemCollections<'_>,
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
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        _ctx: &ShadowCastContext<'_>,
        _items: &ItemCollections<'_>,
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
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        _ctx: &OutlineMaskContext<'_>,
        _items: &ItemCollections<'_>,
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

    fn surface_mask(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &viewport_lib::plugin_api::SurfaceMaskContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let Some(gpu) = &self.gpu else { return };
        let mut bound = false;
        for entry in &self.frame {
            // A wireframe item drew lines, not the surface the stamp covers:
            // stamping the solid would mark whatever shows between them.
            if entry.wireframe || self.wireframe_mode {
                continue;
            }
            let Some(value) = ctx.stamp_for(&entry.settings) else {
                continue;
            };
            if !bound {
                pass.set_pipeline(&gpu.surface_mask_pipeline);
                bound = true;
            }
            pass.set_stencil_reference(value);
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
                let hit = PickHit::object_hit(item.id, world_pos, glam::Vec3::Z);
                best = Some((toi, hit));
            }
        }
        best
    }

    /// Walk the cells where the scalar field straddles the isovalue (the cells
    /// the compute stage would emit triangles for) and hit the item when any
    /// such cell centre projects into the rect.
    fn pick_rect(&self, ctx: &RectPickContext<'_>) -> viewport_lib::renderer::PickRectResult {
        let mut result = viewport_lib::renderer::PickRectResult::default();
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
                        let projected = viewport_lib::plugin_api::pick_helpers::project_to_screen(
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
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PickPassContext<'_>,
        _items: &ItemCollections<'_>,
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

impl GpuMarchingCubesPlugin {
    /// Upload a scalar field, pre-allocating every slab's intermediate and
    /// output buffer, and return its handle. Reached from
    /// [`McVolumes::upload_volume_for_mc`](crate::McVolumes::upload_volume_for_mc).
    pub(crate) fn upload(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        vol: &VolumeData,
    ) -> viewport_lib::error::ViewportResult<McVolumeId> {
        let gpu_data = build_mc_volume_gpu_data(device, queue, vol)?;
        Ok(self.volumes.insert_sized(gpu_data))
    }

    /// Drop a volume and its slab buffers. False if the handle was stale.
    pub(crate) fn free(&mut self, id: McVolumeId) -> bool {
        self.volumes.remove(id).is_some()
    }

    /// Point a volume's scalar field at a caller-supplied buffer, refreshed
    /// into the slab buffers before every dispatch.
    pub(crate) fn set_scalar_source(
        &mut self,
        id: McVolumeId,
        buffer: viewport_lib::gpu::Buffer,
        offset_bytes: u64,
    ) -> viewport_lib::error::ViewportResult<()> {
        if !buffer
            .usage()
            .contains(viewport_lib::gpu::BufferUsages::COPY_SRC)
        {
            return Err(
                viewport_lib::error::ViewportError::ExternalBufferUsageMissing {
                    missing: "COPY_SRC",
                },
            );
        }
        let store_len = self.volumes.slot_count();
        let vol =
            self.volumes
                .get_mut(id)
                .ok_or(viewport_lib::error::ViewportError::StaleHandle {
                    index: id.index(),
                    count: store_len,
                })?;
        let [nx, ny, nz] = vol.dims;
        let needed_bytes = nx as u64 * ny as u64 * nz as u64 * 4;
        let available_bytes = buffer.size().saturating_sub(offset_bytes);
        if offset_bytes % 4 != 0 || needed_bytes > available_bytes {
            return Err(viewport_lib::error::ViewportError::McScalarSourceMismatch {
                needed_bytes,
                available_bytes,
                offset_bytes,
            });
        }
        vol.external_scalar = Some(McExternalScalarSource {
            buffer,
            offset_bytes,
        });
        Ok(())
    }

    /// Detach the external scalar source. The slab buffers keep whatever was
    /// last copied in, so the isosurface freezes at the final field.
    pub(crate) fn clear_scalar_source(
        &mut self,
        id: McVolumeId,
    ) -> viewport_lib::error::ViewportResult<()> {
        let store_len = self.volumes.slot_count();
        let vol =
            self.volumes
                .get_mut(id)
                .ok_or(viewport_lib::error::ViewportError::StaleHandle {
                    index: id.index(),
                    count: store_len,
                })?;
        vol.external_scalar = None;
        Ok(())
    }

    /// Submit the slab sizing and buffer allocation to a worker thread. The
    /// volume is inserted, and its handle minted, when
    /// [`take_upload_result`](Self::take_upload_result) collects the job.
    pub(crate) fn begin_upload(
        &self,
        jobs: &viewport_lib::resources::Jobs<'_>,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        vol: VolumeData,
    ) -> viewport_lib::resources::JobId {
        let device = device.clone();
        let queue = queue.clone();
        jobs.try_submit_cpu(move |progress| {
            progress.set(0.1);
            let gpu_data = build_mc_volume_gpu_data(&device, &queue, &vol)?;
            progress.set(0.95);
            Ok(gpu_data)
        })
    }

    /// Take a finished async upload's volume into the store and hand back its
    /// handle.
    pub(crate) fn take_upload_result(
        &mut self,
        jobs: &viewport_lib::resources::Jobs<'_>,
        id: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<McVolumeId> {
        match jobs.status(id) {
            viewport_lib::resources::UploadStatus::Pending { .. } => {
                Err(viewport_lib::error::ViewportError::JobNotReady)
            }
            viewport_lib::resources::UploadStatus::Unknown => {
                Err(viewport_lib::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                })
            }
            viewport_lib::resources::UploadStatus::Failed(e) => Err(e),
            viewport_lib::resources::UploadStatus::Ready => {
                match jobs.take::<McVolumeGpuData>(id) {
                    Some(gpu_data) => Ok(self.volumes.insert_sized(gpu_data)),
                    None => Err(viewport_lib::error::ViewportError::JobResultMissing {
                        reason: "unknown id or wrong upload type",
                    }),
                }
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
    vol: &viewport_lib_geometry::marching_cubes::VolumeData,
    isovalue: f32,
    mut t_lo: f32,
    mut t_hi: f32,
) -> f32 {
    let s0 = viewport_lib_geometry::marching_cubes::trilinear_sample(
        vol,
        (ray_orig + ray_dir * t_lo).to_array(),
    ) - isovalue;
    let mut lo_sign = s0 < 0.0;
    for _ in 0..8 {
        let mid = (t_lo + t_hi) * 0.5;
        let s = viewport_lib_geometry::marching_cubes::trilinear_sample(
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
    use viewport_lib_geometry::marching_cubes::trilinear_sample;

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
