//! The Gaussian splat item type as an [`ItemTypePlugin`]: alpha-blended,
//! back-to-front sorted billboard splats over a set uploaded via
//! `upload_gaussian_splat`. Consumers submit [`GaussianSplatItem`]s on
//! `SceneFrame::gaussian_splats`; the renderer routes that field to this
//! plugin. The per-viewport GPU depth sort runs in `prepare` and its command
//! buffer is returned to the renderer's prepare submission.
//!
//! The wireframe representation stays with the core line substrate: when an
//! item (or the frame) is in wireframe mode, `paint` skips the splat
//! rasterisation and the polyline overlay produced in the renderer's prepare
//! draws instead.

mod pipeline;
pub(crate) mod types;

use std::sync::Arc;

use crate::plugin_api::pick_helpers::project_to_screen;
use crate::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, RectPickContext,
};
use crate::renderer::{GaussianSplatItem, PickHit, PickId, PickMask, PickRectResult, SubObjectRef};
use crate::resources::{GaussianSplatId, HDR_COLOR_FORMAT, SplatOutlineMaskUniform};

pub(crate) const TYPE_NAME: &str = "viewport.gaussian_splat";

impl PluginItemCollection for Vec<GaussianSplatItem> {
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

/// Per-set sort scratch, keyed by the set's buffer revision so a
/// `replace_gaussian_splat` (new buffers behind the same handle) rebuilds it.
struct SetSorts {
    revision: u64,
    viewports: Vec<Option<pipeline::SortState>>,
}

/// One visible item this prepare.
struct FrameDraw {
    source: GaussianSplatId,
    viewport_index: usize,
    count: u32,
    wireframe: bool,
    pick: Option<(crate::gpu::Buffer, crate::gpu::BindGroup)>,
}

/// Snapshot of one item for the out-of-band CPU pick paths. The CPU-side
/// splat data is shared with the store, not copied.
struct PickSplatItem {
    pick_id: PickId,
    model: [[f32; 4]; 4],
    positions: Arc<Vec<[f32; 3]>>,
    scales: Arc<Vec<[f32; 3]>>,
}

#[derive(Default)]
pub(crate) struct GaussianSplatPlugin {
    gpu: Option<pipeline::SplatGpu>,
    sorts: std::collections::HashMap<GaussianSplatId, SetSorts>,
    frame: Vec<FrameDraw>,
    outlines: Vec<pipeline::SplatOutlineEntry>,
    pick_items: Vec<PickSplatItem>,
}

impl ItemTypePlugin for GaussianSplatPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn on_device_recreated(&mut self, _device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {
        self.gpu = None;
        self.sorts.clear();
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
        self.pick_items.clear();
        let items = items
            .as_any()
            .downcast_ref::<Vec<GaussianSplatItem>>()
            .expect("gaussian splat collection is the SceneFrame field");
        let store = &ctx.resources.content.gaussian_splat_store;

        // Drop sort state for freed sets (slot reuse issues a new generation,
        // so the old handle stops resolving).
        self.sorts.retain(|id, _| store.get(*id).is_some());

        // Snapshot every resolvable item for the CPU pick paths, hidden ones
        // included, matching the pick cache the built-in pickers read.
        for item in items {
            let Some(set) = store.get(item.source) else {
                continue;
            };
            self.pick_items.push(PickSplatItem {
                pick_id: item.settings.pick_id,
                model: item.model,
                positions: set.cpu_positions.clone(),
                scales: set.cpu_scales.clone(),
            });
        }
        if items.is_empty() {
            return Vec::new();
        }

        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::SplatGpu::new(device, ctx.resources));

        let vp_idx = ctx.viewport_index;
        let eye = ctx.camera.eye_position;
        let vp_w = ctx.viewport_size.x.max(1.0);
        let vp_h = ctx.viewport_size.y.max(1.0);
        let mut encoder: Option<crate::gpu::CommandEncoder> = None;

        for item in items {
            if item.settings.hidden {
                continue;
            }
            let Some((set, revision)) = store.get_with_revision(item.source) else {
                continue;
            };

            // Per-(set, viewport) sort scratch, invalidated when the set's
            // buffers were replaced behind the handle.
            let sorts = self.sorts.entry(item.source).or_insert_with(|| SetSorts {
                revision,
                viewports: Vec::new(),
            });
            if sorts.revision != revision {
                sorts.revision = revision;
                sorts.viewports.clear();
            }
            while sorts.viewports.len() <= vp_idx {
                sorts.viewports.push(None);
            }
            if sorts.viewports[vp_idx].is_none() {
                sorts.viewports[vp_idx] = Some(gpu.make_sort_state(device, set));
            }
            let sort = sorts.viewports[vp_idx].as_ref().unwrap();

            let enc = encoder.get_or_insert_with(|| {
                device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                    label: Some("splat_sort_encoder"),
                })
            });
            gpu.encode_sort(device, queue, enc, set, sort, eye, item.model, vp_w, vp_h);

            let pick = (item.settings.pick_id != PickId::NONE).then(|| {
                let id_data: [u32; 4] = [item.settings.pick_id.0 as u32, 0, 0, 0];
                let pick_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("gaussian_splat_pick_id_buf"),
                    size: 16,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&pick_buf, 0, bytemuck::bytes_of(&id_data));
                let pick_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("gaussian_splat_pick_id_bg"),
                    layout: &gpu.pick_id_bgl,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: pick_buf.as_entire_binding(),
                    }],
                });
                (pick_buf, pick_bg)
            });
            self.frame.push(FrameDraw {
                source: item.source,
                viewport_index: vp_idx,
                count: set.count,
                wireframe: ctx.wireframe_mode || item.settings.wireframe,
                pick,
            });
        }

        // Selection outline coverage: point-sprite discs per selected set,
        // or per sub-selected splat.
        if ctx.outline_selected {
            self.build_outlines(device, ctx, items);
        }

        encoder.map(|e| vec![e.finish()]).unwrap_or_default()
    }

    fn paint(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let Some(gpu) = &self.gpu else { return };
        let mut bound = false;
        for fd in &self.frame {
            if fd.wireframe {
                continue;
            }
            let Some(sort) = self.sort_state(fd) else {
                continue;
            };
            if !bound {
                pass.set_pipeline(
                    gpu.pipeline
                        .for_format(ctx.target_format == HDR_COLOR_FORMAT),
                );
                bound = true;
            }
            pass.set_bind_group(1, &sort.render_bg, &[]);
            pass.draw(0..6, 0..fd.count);
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

    fn pick(&self, ray: &PickRay, ctx: &PickContext) -> Option<(f32, PickHit)> {
        let wants_splat = ctx.mask.intersects(PickMask::SPLAT);
        if !wants_splat && !ctx.mask.intersects(PickMask::OBJECT) {
            return None;
        }
        let mut best: Option<(f32, PickHit)> = None;
        for item in &self.pick_items {
            if item.pick_id == PickId::NONE || item.positions.is_empty() {
                continue;
            }
            let model = glam::Mat4::from_cols_array_2d(&item.model);
            // Derive pick radius from the mean per-splat scale so that a
            // click anywhere inside the visible disc registers as a hit.
            let mean_max_scale: f32 = if item.scales.is_empty() {
                0.05
            } else {
                item.scales
                    .iter()
                    .map(|s| s[0].max(s[1]).max(s[2]))
                    .sum::<f32>()
                    / item.scales.len() as f32
            };
            let world_radius = mean_max_scale * 3.0;
            let center_w = model.transform_point3(glam::Vec3::ZERO);
            let p0_clip = ctx.view_proj * center_w.extend(1.0);
            let p1_clip = ctx.view_proj * (center_w + glam::Vec3::X * world_radius).extend(1.0);
            let radius_px = if p0_clip.w.abs() > 1e-6 && p1_clip.w.abs() > 1e-6 {
                let p0_ndc = glam::Vec2::new(p0_clip.x, p0_clip.y) / p0_clip.w;
                let p1_ndc = glam::Vec2::new(p1_clip.x, p1_clip.y) / p1_clip.w;
                ((p1_ndc - p0_ndc).length() * 0.5 * ctx.viewport_size.x.max(ctx.viewport_size.y))
                    .max(4.0)
            } else {
                world_radius * 100.0
            };
            if let Some(mut hit) = crate::interaction::query::picking::pick_gaussian_splat_cpu(
                ctx.click_pos,
                item.pick_id.0,
                &item.positions,
                model,
                ctx.view_proj,
                ctx.viewport_size,
                radius_px,
            ) {
                // pick_gaussian_splat_cpu returns SubObjectRef::Point; remap
                // to Splat, or clear it for object-level queries.
                let toi = (hit.world_pos - ray.origin).dot(ray.direction).max(0.0);
                if wants_splat {
                    if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                        hit.sub_object = Some(SubObjectRef::Splat(idx));
                    }
                } else {
                    hit.sub_object = None;
                }
                if best.as_ref().is_none_or(|(t, _)| toi < *t) {
                    best = Some((toi, hit));
                }
            }
        }
        best
    }

    fn pick_rect(&self, ctx: &RectPickContext) -> PickRectResult {
        let mut result = PickRectResult::default();
        let wants_splat = ctx.mask.intersects(PickMask::SPLAT);
        let wants_object = ctx.mask.intersects(PickMask::OBJECT);
        if !wants_splat && !wants_object {
            return result;
        }
        let in_rect = |p: glam::Vec2| {
            p.x >= ctx.rect_min.x
                && p.x <= ctx.rect_max.x
                && p.y >= ctx.rect_min.y
                && p.y <= ctx.rect_max.y
        };
        for item in &self.pick_items {
            if item.pick_id == PickId::NONE || item.positions.is_empty() {
                continue;
            }
            let model = glam::Mat4::from_cols_array_2d(&item.model);
            let mut item_hit = false;
            for (i, pos) in item.positions.iter().enumerate() {
                let world = model.transform_point3(glam::Vec3::from(*pos));
                if let Some(p) = project_to_screen(world, ctx.view_proj, ctx.viewport_size) {
                    if in_rect(p) {
                        if wants_splat {
                            result
                                .elements
                                .push((item.pick_id.0, SubObjectRef::Splat(i as u32)));
                        }
                        item_hit = true;
                    }
                }
            }
            if wants_object && item_hit {
                result.objects.push(item.pick_id.0);
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
        if !ctx.mask.intersects(PickMask::OBJECT | PickMask::SPLAT) {
            return;
        }
        let Some(gpu) = &self.gpu else { return };
        let mut bound = false;
        for fd in &self.frame {
            if fd.wireframe || fd.count == 0 {
                continue;
            }
            let Some((_, pick_bg)) = &fd.pick else {
                continue;
            };
            let Some(sort) = self.sort_state(fd) else {
                continue;
            };
            if !bound {
                pass.set_pipeline(&gpu.pick_pipeline);
                bound = true;
            }
            pass.set_bind_group(1, &sort.render_bg, &[]);
            pass.set_bind_group(2, pick_bg, &[]);
            pass.draw(0..6, 0..fd.count);
        }
    }

    /// Three orthogonal rings per splat, showing each Gaussian's anisotropy.
    ///
    /// This is sub-structure rather than bounds: the rings say what the set is
    /// made of. It is capped because a ring per splat is 99 line strips at 100
    /// splats and unbounded beyond that. Sets past the cap draw nothing, which
    /// is deliberate: the box that used to stand in for them was a single
    /// PCA-fitted hull around the whole cloud, which said less than the
    /// silhouette already did.
    fn wireframe_polylines(
        &self,
        items: &dyn PluginItemCollection,
        ctx: &ItemFrameContext<'_>,
    ) -> Vec<crate::renderer::PolylineItem> {
        /// Above this many splats a ring per splat stops being readable and
        /// starts being expensive.
        const MAX_RINGED_SPLATS: usize = 100;

        let Some(splats) = items.as_any().downcast_ref::<Vec<GaussianSplatItem>>() else {
            return Vec::new();
        };
        let store = &ctx.resources.content.gaussian_splat_store;
        splats
            .iter()
            .filter(|item| !item.settings.hidden && (ctx.wireframe_mode || item.settings.wireframe))
            .filter_map(|item| {
                let set = store.get(item.source)?;
                let count = (set.count as usize).min(set.cpu_positions.len());
                if count == 0 || count > MAX_RINGED_SPLATS {
                    return None;
                }
                Some(splat_rings_polyline(
                    &set.cpu_positions[..count],
                    &set.cpu_scales[..count],
                    glam::Mat4::from_cols_array_2d(&item.model),
                ))
            })
            .collect()
    }

    fn resolve_sub_object(
        &self,
        _pick_id: PickId,
        primitive_index: u32,
        _world_pos: glam::Vec3,
        mask: PickMask,
    ) -> Option<SubObjectRef> {
        // The pick fragment writes the splat's instance index into the
        // primitive channel; no device feature involved.
        mask.intersects(PickMask::SPLAT)
            .then_some(SubObjectRef::Splat(primitive_index))
    }
}

impl GaussianSplatPlugin {
    fn sort_state(&self, fd: &FrameDraw) -> Option<&pipeline::SortState> {
        self.sorts
            .get(&fd.source)?
            .viewports
            .get(fd.viewport_index)?
            .as_ref()
    }

    /// Build the outline mask coverage for this frame: whole-set discs for
    /// selected items, per-splat discs for sub-selected splats on items
    /// that are not themselves selected.
    fn build_outlines(
        &mut self,
        device: &crate::gpu::Device,
        ctx: &ItemFrameContext<'_>,
        items: &[GaussianSplatItem],
    ) {
        use crate::gpu::util::DeviceExt;
        let resources = ctx.resources;
        let view_proj = ctx.camera.view_proj();
        let (vp_w, vp_h) = (ctx.viewport_size.x, ctx.viewport_size.y);
        let cam_right = ctx.camera.view.row(0).truncate().normalize();

        // Project a world-space radius at `center_w` to a pixel half-size,
        // offsetting along the camera right vector so the measurement never
        // collapses when looking down an axis.
        let radius_px = |center_w: glam::Vec3, world_radius: f32| -> f32 {
            let p0_clip = view_proj * glam::Vec4::new(center_w.x, center_w.y, center_w.z, 1.0);
            let p1_world = center_w + cam_right * world_radius;
            let p1_clip = view_proj * glam::Vec4::new(p1_world.x, p1_world.y, p1_world.z, 1.0);
            if p0_clip.w.abs() > 1e-6 && p1_clip.w.abs() > 1e-6 {
                let p0_ndc = glam::Vec2::new(p0_clip.x, p0_clip.y) / p0_clip.w;
                let p1_ndc = glam::Vec2::new(p1_clip.x, p1_clip.y) / p1_clip.w;
                (p1_ndc - p0_ndc).length() * 0.5 * vp_w.max(vp_h)
            } else {
                world_radius * 100.0
            }
        };
        let make_bind_group = |uniform_buf: &crate::gpu::Buffer, label| {
            device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some(label),
                layout: &resources.outline.bind_group_layout,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: resources
                            .content
                            .fallback_position_override_buf
                            .as_entire_binding(),
                    },
                ],
            })
        };

        for item in items {
            let Some(gpu_set) = resources.content.gaussian_splat_store.get(item.source) else {
                continue;
            };
            if item.settings.selected && !gpu_set.cpu_positions.is_empty() {
                // Object-level: outline all splats. World-space radius covers
                // the visible Gaussian tail (~3 sigma).
                let mean_max_scale: f32 = if gpu_set.cpu_scales.is_empty() {
                    0.05
                } else {
                    gpu_set
                        .cpu_scales
                        .iter()
                        .map(|s| s[0].max(s[1]).max(s[2]))
                        .sum::<f32>()
                        / gpu_set.cpu_scales.len() as f32
                };
                let world_radius = mean_max_scale * 3.0;
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let center_w = model.transform_point3(glam::Vec3::ZERO);
                let pixel_radius = radius_px(center_w, world_radius).max(1.0);

                let position_buf =
                    device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                        label: Some("splat_outline_pos_buf"),
                        contents: bytemuck::cast_slice(gpu_set.cpu_positions.as_slice()),
                        usage: crate::gpu::BufferUsages::VERTEX,
                    });
                let uniform = SplatOutlineMaskUniform {
                    model: item.model,
                    viewport_w: vp_w,
                    viewport_h: vp_h,
                    pixel_radius,
                    _pad: [0.0; 9],
                };
                let uniform_buf =
                    device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                        label: Some("splat_outline_uniform_buf"),
                        contents: bytemuck::cast_slice(&[uniform]),
                        usage: crate::gpu::BufferUsages::UNIFORM,
                    });
                let bind_group = make_bind_group(&uniform_buf, "splat_outline_bg");
                let n = gpu_set.cpu_positions.len();
                let size_data: Vec<f32> = vec![pixel_radius; n];
                let size_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                    label: Some("splat_outline_size_buf"),
                    contents: bytemuck::cast_slice(&size_data),
                    usage: crate::gpu::BufferUsages::VERTEX,
                });
                self.outlines.push(pipeline::SplatOutlineEntry {
                    position_buf,
                    size_buf,
                    instance_count: n as u32,
                    _uniform_buf: uniform_buf,
                    bind_group,
                });
            } else if !item.settings.selected && item.settings.pick_id != PickId::NONE {
                // Per-splat sub-selection: outline only the selected splats.
                let selected_indices: Vec<u32> = ctx
                    .sub_selection
                    .iter()
                    .flat_map(|s| s.items.iter())
                    .filter_map(|(node_id, sub)| {
                        if *node_id == item.settings.pick_id.0 {
                            if let SubObjectRef::Splat(i) = sub {
                                return Some(*i);
                            }
                        }
                        None
                    })
                    .collect();
                if selected_indices.is_empty() {
                    continue;
                }

                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let mut positions: Vec<[f32; 3]> = Vec::with_capacity(selected_indices.len());
                let mut sizes: Vec<f32> = Vec::with_capacity(selected_indices.len());
                for &idx in &selected_indices {
                    let i = idx as usize;
                    if let Some(&pos) = gpu_set.cpu_positions.get(i) {
                        positions.push(pos);
                        let world_radius = if let Some(s) = gpu_set.cpu_scales.get(i) {
                            s[0].max(s[1]).max(s[2]) * 3.0
                        } else {
                            0.15
                        };
                        let center_w = model.transform_point3(glam::Vec3::from(pos));
                        sizes.push(radius_px(center_w, world_radius).max(1.0));
                    }
                }
                if positions.is_empty() {
                    continue;
                }

                let pixel_radius = sizes
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
                    .max(1.0);
                let uniform = SplatOutlineMaskUniform {
                    model: item.model,
                    viewport_w: vp_w,
                    viewport_h: vp_h,
                    pixel_radius,
                    _pad: [0.0; 9],
                };
                let uniform_buf =
                    device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                        label: Some("splat_sel_outline_uniform_buf"),
                        contents: bytemuck::cast_slice(&[uniform]),
                        usage: crate::gpu::BufferUsages::UNIFORM,
                    });
                let bind_group = make_bind_group(&uniform_buf, "splat_sel_outline_bg");
                let position_buf =
                    device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                        label: Some("splat_sel_outline_pos_buf"),
                        contents: bytemuck::cast_slice(&positions),
                        usage: crate::gpu::BufferUsages::VERTEX,
                    });
                let size_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                    label: Some("splat_sel_outline_size_buf"),
                    contents: bytemuck::cast_slice(&sizes),
                    usage: crate::gpu::BufferUsages::VERTEX,
                });
                self.outlines.push(pipeline::SplatOutlineEntry {
                    position_buf,
                    size_buf,
                    instance_count: positions.len() as u32,
                    _uniform_buf: uniform_buf,
                    bind_group,
                });
            }
        }
    }
}

/// Three orthogonal rings (XY, XZ, YZ) per splat, each scaled by that splat's
/// own scale and placed by the item's model.
fn splat_rings_polyline(
    positions: &[[f32; 3]],
    scales: &[[f32; 3]],
    model: glam::Mat4,
) -> crate::renderer::PolylineItem {
    const SEGMENTS: usize = 32;
    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    let mut strip_lengths: Vec<u32> = Vec::new();
    for (pos, scale) in positions.iter().zip(scales.iter()) {
        let centre = glam::Vec3::from(*pos);
        let [sx, sy, sz] = *scale;
        let rings: [(glam::Vec3, glam::Vec3, f32, f32); 3] = [
            (glam::Vec3::X, glam::Vec3::Y, sx, sy),
            (glam::Vec3::X, glam::Vec3::Z, sx, sz),
            (glam::Vec3::Y, glam::Vec3::Z, sy, sz),
        ];
        for (a1, a2, r1, r2) in &rings {
            for i in 0..=SEGMENTS {
                let t = std::f32::consts::TAU * i as f32 / SEGMENTS as f32;
                let local = centre + (*a1) * (r1 * t.cos()) + (*a2) * (r2 * t.sin());
                all_positions.push(model.transform_point3(local).to_array());
            }
            strip_lengths.push((SEGMENTS + 1) as u32);
        }
    }
    crate::renderer::PolylineItem {
        positions: all_positions,
        strip_lengths,
        default_colour: [0.75, 0.75, 0.75, 1.0].into(),
        line_width: 1.0,
        ..crate::renderer::PolylineItem::default()
    }
}
