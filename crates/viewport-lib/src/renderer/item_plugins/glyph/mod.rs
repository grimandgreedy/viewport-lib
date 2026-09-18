//! The glyph item type as an [`ItemTypePlugin`]: one instanced arrow, sphere or
//! cube per sample, oriented and scaled by the sample's vector. Consumers
//! submit [`GlyphItem`]s on `SceneFrame::glyphs`, or [`GlyphSetRefItem`]s on
//! `SceneFrame::glyph_set_refs` to draw a set uploaded once through
//! `upload_glyph_set`; the renderer routes both fields to this plugin.

pub(crate) mod pipeline;
pub(crate) mod store;
pub(crate) mod types;

use crate::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, RectPickContext,
};
use crate::renderer::{
    GlyphItem, GlyphSetRefItem, PickHit, PickId, PickMask, PickRectResult, SubObjectRef,
};
use crate::resources::HDR_COLOR_FORMAT;
use store::{GlyphGpuData, GlyphLayouts, GlyphSetStore, build_glyph_set, resolve_bindings};

pub(crate) use types::GlyphSetId;

pub(crate) const TYPE_NAME: &str = "viewport.glyph";

impl PluginItemCollection for Vec<GlyphItem> {
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

impl PluginItemCollection for Vec<GlyphSetRefItem> {
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
pub(crate) struct GlyphPlugin {
    /// The pre-uploaded sets, owned by the type that draws them.
    stored: GlyphSetStore,
    /// The two layouts every upload builds its bind groups against. Created on
    /// registration, because an upload can arrive before the first frame.
    layouts: Option<GlyphLayouts>,
    gpu: Option<pipeline::GlyphGpu>,
    /// Per drawn set, rebuilt each prepare: the inline items first, then the
    /// references.
    frame: Vec<pipeline::GlyphFrame>,
    /// Every inline item from the last prepared frame, hidden included,
    /// matching what the CPU pick cache used to retain. Reference items are not
    /// here: their instances live on the GPU, so they answer the GPU pick only.
    pick_items: Vec<GlyphItem>,
}

impl ItemTypePlugin for GlyphPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn init_gpu(
        &mut self,
        device: &crate::gpu::Device,
        _shared: &crate::plugin_api::SharedBindings<'_>,
    ) {
        self.layouts = Some(GlyphLayouts::new(device));
    }

    fn resident_bytes(&self) -> u64 {
        self.stored.allocated_bytes()
    }

    fn on_device_recreated(&mut self, device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {
        self.layouts = Some(GlyphLayouts::new(device));
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
            .downcast_ref::<Vec<GlyphItem>>()
            .expect("glyph collection is the SceneFrame field");
        let refs = ctx.refs_of::<GlyphSetRefItem>();
        self.pick_items.clear();
        self.pick_items.extend_from_slice(items);
        if items.is_empty() && refs.is_empty() {
            return Vec::new();
        }
        let layouts = self
            .layouts
            .get_or_insert_with(|| GlyphLayouts::new(device));
        let stored = &self.stored;
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::GlyphGpu::new(device, ctx.resources, layouts));

        for item in items {
            if item.settings.hidden || item.positions.is_empty() || item.vectors.is_empty() {
                continue;
            }
            let wireframe = ctx.wireframe_mode || item.settings.wireframe;
            let binds = resolve_bindings(device, ctx.resources, layouts, item);
            let gpu_data = build_glyph_set(device, queue, &binds, item, wireframe);
            let pick_bind_group = (gpu_data.pick_id != PickId::NONE).then(|| {
                gpu.pick_bind_group(device, queue, gpu_data.pick_id, &gpu_data._uniform_buf)
            });
            let outline = ctx
                .outline_selected
                .then(|| outline_for(&item.settings, ctx.sub_selection))
                .flatten();
            self.frame.push(pipeline::GlyphFrame {
                gpu: gpu_data,
                pick_bind_group,
                outline,
            });
        }

        // Pre-uploaded references. The model matrix lives at offset 0 of the
        // uniform, so a reference re-places a stored set without rebuilding its
        // instance buffer.
        for ref_item in refs {
            if ref_item.settings.hidden {
                continue;
            }
            let Some(entry) = stored.get(ref_item.source) else {
                continue;
            };
            let mut gpu_data = entry.clone();
            queue.write_buffer(
                &gpu_data._uniform_buf,
                0,
                bytemuck::bytes_of(&ref_item.model),
            );
            gpu_data.wireframe = ctx.wireframe_mode || ref_item.settings.wireframe;
            // The pick id comes from the reference, not from the upload: the
            // same stored set can be drawn twice under two ids.
            let pick_id = ref_item.settings.pick_id;
            let pick_bind_group = (pick_id != PickId::NONE)
                .then(|| gpu.pick_bind_group(device, queue, pick_id, &gpu_data._uniform_buf));
            gpu_data.pick_id = pick_id;
            self.frame.push(pipeline::GlyphFrame {
                gpu: gpu_data,
                pick_bind_group,
                outline: None,
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
            let pipeline = if entry.gpu.wireframe {
                gpu.wireframe_pipeline.for_format(is_hdr)
            } else {
                gpu.pipeline.for_format(is_hdr)
            };
            pass.set_pipeline(pipeline);
            pipeline::draw_set(pass, &entry.gpu, 0..entry.gpu.instance_count);
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
        let mut bound = false;
        for entry in &self.frame {
            let Some(instance_filter) = &entry.outline else {
                continue;
            };
            if !bound {
                pass.set_pipeline(&gpu.mask_pipeline);
                bound = true;
            }
            // The mask always draws the solid base mesh, never the edge list,
            // so the outline covers the glyph's silhouette even in wireframe.
            pass.set_bind_group(1, &entry.gpu.uniform_bind_group, &[]);
            pass.set_bind_group(2, &entry.gpu.instance_bind_group, &[]);
            pass.set_vertex_buffer(0, entry.gpu.mesh_vertex_buffer.slice(..));
            pass.set_index_buffer(
                entry.gpu.mesh_index_buffer.slice(..),
                crate::gpu::IndexFormat::Uint32,
            );
            match instance_filter {
                None => pass.draw_indexed(
                    0..entry.gpu.mesh_index_count,
                    0,
                    0..entry.gpu.instance_count,
                ),
                Some(indices) => {
                    for &i in indices {
                        pass.draw_indexed(0..entry.gpu.mesh_index_count, 0, i..i + 1);
                    }
                }
            }
        }
    }

    fn pick(&self, ray: &PickRay, ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
        let wants_instance = ctx.mask.intersects(PickMask::INSTANCE);
        if !wants_instance && !ctx.mask.intersects(PickMask::OBJECT) {
            return None;
        }
        let mut best: Option<(f32, PickHit)> = None;
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                continue;
            }
            let model = glam::Mat4::from_cols_array_2d(&item.model);
            // An arrow is picked about its midpoint rather than its base, so a
            // click anywhere along the shaft registers on the right instance.
            let (midpoints, full_len) = arrow_midpoints(item);
            if midpoints.is_empty() {
                continue;
            }
            let n = midpoints.len() as f32;
            let centroid = model.transform_point3(
                midpoints
                    .iter()
                    .map(|p| glam::Vec3::from(*p))
                    .sum::<glam::Vec3>()
                    / n,
            );
            let radius_px = crate::plugin_api::pick_helpers::world_radius_in_pixels(
                centroid,
                full_len * 0.5,
                ctx.view_proj,
                ctx.viewport_size,
            );
            let Some(mut hit) = crate::interaction::query::picking::pick_gaussian_splat_cpu(
                ctx.click_pos,
                item.settings.pick_id.0,
                &midpoints,
                model,
                ctx.view_proj,
                ctx.viewport_size,
                radius_px,
            ) else {
                continue;
            };
            // Report the instance's base position, not the midpoint the search
            // ran against.
            if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                if let Some(base) = item.positions.get(idx as usize) {
                    hit.world_pos = model.transform_point3(glam::Vec3::from(*base));
                }
                hit.sub_object = wants_instance.then_some(SubObjectRef::Instance(idx));
            }
            let toi = (hit.world_pos - ray.origin).dot(ray.direction).max(0.0);
            if best.as_ref().is_none_or(|(t, _)| toi < *t) {
                best = Some((toi, hit));
            }
        }
        best
    }

    fn pick_rect(&self, ctx: &RectPickContext<'_>) -> PickRectResult {
        let mut result = PickRectResult::default();
        let wants_instance = ctx.mask.intersects(PickMask::INSTANCE);
        let wants_object = ctx.mask.intersects(PickMask::OBJECT);
        if !wants_instance && !wants_object {
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
                if wants_instance {
                    result
                        .elements
                        .push((id, SubObjectRef::Instance(index as u32)));
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
        if !ctx.mask.intersects(PickMask::OBJECT | PickMask::INSTANCE) {
            return;
        }
        let Some(gpu) = &self.gpu else { return };
        let mut bound = false;
        for entry in &self.frame {
            let Some(pick_bg) = &entry.pick_bind_group else {
                continue;
            };
            if entry.gpu.instance_count == 0 {
                continue;
            }
            if !bound {
                pass.set_pipeline(&gpu.pick_pipeline);
                bound = true;
            }
            pass.set_bind_group(1, pick_bg, &[]);
            pass.set_bind_group(2, &entry.gpu.instance_bind_group, &[]);
            pass.set_vertex_buffer(0, entry.gpu.mesh_vertex_buffer.slice(..));
            pass.set_index_buffer(
                entry.gpu.mesh_index_buffer.slice(..),
                crate::gpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(
                0..entry.gpu.mesh_index_count,
                0,
                0..entry.gpu.instance_count,
            );
        }
    }

    fn resolve_sub_object(
        &self,
        _pick_id: PickId,
        primitive_index: u32,
        _world_pos: glam::Vec3,
        mask: PickMask,
    ) -> Option<SubObjectRef> {
        // The pick fragment writes the glyph's instance index into the
        // primitive channel; no device feature involved.
        mask.intersects(PickMask::INSTANCE)
            .then_some(SubObjectRef::Instance(primitive_index))
    }
}

/// This set's outline coverage for the frame: `None` when it is not outlined,
/// `Some(None)` for the whole set, `Some(Some(indices))` for a sub-selection.
fn outline_for(
    settings: &crate::scene::material::ItemSettings,
    sub_selection: Option<&crate::renderer::SubSelectionRef>,
) -> Option<Option<Vec<u32>>> {
    if settings.hidden {
        return None;
    }
    if settings.selected {
        return Some(None);
    }
    if settings.pick_id == PickId::NONE {
        return None;
    }
    let instances: Vec<u32> = sub_selection
        .iter()
        .flat_map(|s| s.items.iter())
        .filter_map(|(node_id, sub)| {
            if *node_id != settings.pick_id.0 {
                return None;
            }
            match sub {
                SubObjectRef::Instance(i) => Some(*i),
                _ => None,
            }
        })
        .collect();
    (!instances.is_empty()).then_some(Some(instances))
}

impl GlyphPlugin {
    /// Number of sets the last `prepare` produced draw data for.
    #[cfg(test)]
    pub(crate) fn drawn_count(&self) -> usize {
        self.frame.len()
    }

    /// The layouts, built on demand. The polyline vector decoration borrows
    /// them too: its arrows are glyphs and go through the same builder.
    pub(crate) fn layouts(&mut self, device: &crate::gpu::Device) -> &GlyphLayouts {
        self.layouts
            .get_or_insert_with(|| GlyphLayouts::new(device))
    }

    /// Build one set's GPU data against the plugin's layouts.
    fn build(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
        item: &GlyphItem,
    ) -> GlyphGpuData {
        let layouts = self
            .layouts
            .get_or_insert_with(|| GlyphLayouts::new(device));
        let binds = resolve_bindings(device, resources, layouts, item);
        build_glyph_set(device, queue, &binds, item, false)
    }

    /// Pre-upload a glyph set and return its handle.
    pub(crate) fn upload(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
        item: &GlyphItem,
    ) -> GlyphSetId {
        let gpu = self.build(device, queue, resources, item);
        self.stored.insert_sized(gpu)
    }

    /// Drop a stored set. `false` when the handle does not resolve.
    pub(crate) fn drop_stored(&mut self, id: GlyphSetId) -> bool {
        self.stored.remove(id).is_some()
    }

    /// Replace the samples behind a live handle, keeping the handle.
    pub(crate) fn replace(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
        id: GlyphSetId,
        item: &GlyphItem,
    ) -> bool {
        if !self.stored.contains(id) {
            return false;
        }
        let gpu = self.build(device, queue, resources, item);
        self.stored.replace_sized(id, gpu).is_some()
    }

    /// Build a set's buffers on a worker thread. The handle is minted when
    /// [`take_upload_result`](Self::take_upload_result) collects the job.
    ///
    /// The colourmap view, the shared sampler and the base mesh are resolved
    /// here and cloned into the worker: they are the renderer's and a worker
    /// has no `DeviceResources` borrow.
    pub(crate) fn begin_upload(
        &mut self,
        jobs: &crate::resources::Jobs<'_>,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
        item: GlyphItem,
    ) -> crate::resources::JobId {
        let layouts = self
            .layouts
            .get_or_insert_with(|| GlyphLayouts::new(device));
        let binds = resolve_bindings(device, resources, layouts, &item);
        let device = device.clone();
        let queue = queue.clone();
        jobs.submit_cpu(move || build_glyph_set(&device, &queue, &binds, &item, false))
    }

    /// Store the set a finished job built and hand back its handle.
    pub(crate) fn take_upload_result(
        &mut self,
        jobs: &crate::resources::Jobs<'_>,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<GlyphSetId> {
        match jobs.status(id) {
            crate::resources::UploadStatus::Pending { .. } => {
                Err(crate::error::ViewportError::JobNotReady)
            }
            crate::resources::UploadStatus::Failed(e) => Err(e),
            _ => match jobs.take::<GlyphGpuData>(id) {
                Some(gpu) => Ok(self.stored.insert_sized(gpu)),
                None => Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                }),
            },
        }
    }
}

/// Per-instance arrow midpoints (base plus half the vector) and the set's
/// representative full arrow length.
///
/// The proximity test runs against midpoints rather than base positions so the
/// hit circle does not extend a full arrow-length behind the base when the
/// arrow points away from the camera.
fn arrow_midpoints(item: &GlyphItem) -> (Vec<[f32; 3]>, f32) {
    let full_len = if item.scale_by_magnitude && !item.vectors.is_empty() {
        let mean_mag = item
            .vectors
            .iter()
            .map(|v| glam::Vec3::from(*v).length())
            .sum::<f32>()
            / item.vectors.len() as f32;
        (mean_mag * item.scale).max(0.01)
    } else {
        item.scale.max(0.01)
    };
    let has_vectors = item.vectors.len() == item.positions.len();
    let midpoints = item
        .positions
        .iter()
        .enumerate()
        .map(|(i, pos)| {
            if !has_vectors {
                return *pos;
            }
            let p = glam::Vec3::from(*pos);
            let v = glam::Vec3::from(item.vectors[i]);
            let len = if item.scale_by_magnitude {
                v.length() * item.scale
            } else {
                item.scale
            };
            (p + v.normalize_or_zero() * len * 0.5).to_array()
        })
        .collect();
    (midpoints, full_len)
}
