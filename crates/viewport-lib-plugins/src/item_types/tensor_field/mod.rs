//! The tensor field item type as an [`ItemTypePlugin`]: one instanced
//! ellipsoid per sample, oriented and scaled by the sample's eigenvectors and
//! eigenvalues. Consumers submit [`TensorFieldItem`]s with
//! `frame.scene.submit::<TensorFieldItem>(..)`, or [`TensorFieldRefItem`]s
//! to draw a set uploaded once through
//! [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload). Both forms
//! arrive at this plugin under the one type name.

pub mod channels;
mod pipeline;
mod store;
mod types;
mod uploads;

use store::{
    TensorFieldGpuData, TensorFieldResources, TensorFieldStore, build_tensor_field,
    resolve_bindings, sample_extents,
};
use viewport_lib::plugin_api::{
    ItemCollections, ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext,
    PickContext, PickPassContext, PickRay, PluginItem, RectPickContext,
};
use viewport_lib::renderer::{PickHit, PickId, PickMask, PickRectResult, SubObjectRef};
use viewport_lib::resources::HDR_COLOR_FORMAT;

pub(crate) use store::encode_samples;

pub use types::{TensorFieldId, TensorFieldItem, TensorFieldRefItem, TensorSource};

/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::item_types::shader::{lit_shader, scene_shader, wgsl_source};
    use viewport_lib::plugin_api::shared_wgsl;
    vec![
        (
            "tensor_field.wgsl",
            lit_shader(
                &[shared_wgsl::SHARED_CLIP_VOLUME_WGSL],
                wgsl_source!("tensor_field"),
            ),
        ),
        (
            "tensor_field_pick.wgsl",
            scene_shader(&[], wgsl_source!("tensor_field_pick")),
        ),
        (
            "tensor_field_outline_mask.wgsl",
            scene_shader(&[], wgsl_source!("tensor_field_outline_mask")),
        ),
    ]
}

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.tensor_field";

impl PluginItem for TensorFieldItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

impl PluginItem for TensorFieldRefItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

#[derive(Default)]
pub struct TensorFieldPlugin {
    /// The pre-uploaded fields, owned by the type that draws them.
    stored: TensorFieldStore,
    /// The two layouts every upload builds its bind groups against. Created on
    /// registration, because an upload can arrive before the first frame.
    layouts: Option<TensorFieldResources>,
    gpu: Option<pipeline::TensorFieldGpu>,
    /// Per drawn field, rebuilt each prepare: the inline items first, then the
    /// references.
    frame: Vec<pipeline::TensorFieldFrame>,
    /// Every inline item from the last prepared frame, hidden included.
    /// Reference items are not here: their instances live on the GPU, so they
    /// answer the GPU pick only.
    pick_items: Vec<TensorFieldItem>,
}

impl ItemTypePlugin for TensorFieldPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn init_gpu(
        &mut self,
        device: &viewport_lib::gpu::Device,
        _shared: &viewport_lib::plugin_api::SharedBindings<'_>,
    ) {
        self.layouts = Some(TensorFieldResources::new(device));
    }

    fn warm(
        &mut self,
        device: &viewport_lib::gpu::Device,
        resources: &viewport_lib::DeviceResources,
    ) {
        let layouts = self
            .layouts
            .get_or_insert_with(|| TensorFieldResources::new(device));
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::TensorFieldGpu::new(device, resources, layouts));
        gpu.pipelines.request_all();
    }

    fn resident_bytes(&self) -> u64 {
        self.stored.allocated_bytes()
    }

    fn on_device_recreated(
        &mut self,
        device: &viewport_lib::gpu::Device,
        _queue: &viewport_lib::gpu::Queue,
    ) {
        self.layouts = Some(TensorFieldResources::new(device));
        self.gpu = None;
        self.frame.clear();
    }

    fn prepare(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &ItemCollections<'_>,
    ) -> Vec<viewport_lib::gpu::CommandBuffer> {
        self.frame.clear();
        let items = items.of::<TensorFieldItem>();
        let refs = ctx.refs_of::<TensorFieldRefItem>();
        self.pick_items.clear();
        self.pick_items.extend_from_slice(items);
        if items.is_empty() && refs.is_empty() {
            return Vec::new();
        }
        let layouts = self
            .layouts
            .get_or_insert_with(|| TensorFieldResources::new(device));
        let stored = &self.stored;
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::TensorFieldGpu::new(device, ctx.resources, layouts));

        for item in items {
            if item.settings.hidden || item.positions.is_empty() {
                continue;
            }
            let binds = resolve_bindings(ctx.resources, layouts, item);
            let built = build_tensor_field(device, queue, &binds, item, 0);
            let draw = built.draw();
            let pick_bind_group = (draw.pick_id != PickId::NONE)
                .then(|| gpu.pick_bind_group(device, queue, draw.pick_id, built.uniform_buf()));
            let outline = ctx
                .outline_selected
                .then(|| outline_for(&item.settings, ctx.sub_selection))
                .flatten();
            self.frame.push(pipeline::TensorFieldFrame {
                draw,
                pick_bind_group,
                outline,
                settings: item.settings,
            });
        }

        // Pre-uploaded references. The model matrix lives at offset 0 of the
        // uniform and the shader composes it on top of each instance's
        // ellipsoid model, so a reference re-places a stored set without
        // rebuilding its instance buffer.
        for ref_item in refs {
            if ref_item.settings.hidden {
                continue;
            }
            let Some(entry) = stored.get(ref_item.source) else {
                continue;
            };
            queue.write_buffer(entry.uniform_buf(), 0, bytemuck::bytes_of(&ref_item.model));
            let mut draw = entry.draw();
            // The pick id comes from the reference, not from the upload: the
            // same stored field can be drawn twice under two ids.
            let pick_id = ref_item.settings.pick_id;
            let pick_bind_group = (pick_id != PickId::NONE)
                .then(|| gpu.pick_bind_group(device, queue, pick_id, entry.uniform_buf()));
            draw.pick_id = pick_id;
            self.frame.push(pipeline::TensorFieldFrame {
                draw,
                pick_bind_group,
                outline: None,
                settings: ref_item.settings,
            });
        }
        Vec::new()
    }

    fn paint(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let Some(gpu) = &self.gpu else { return };
        let is_hdr = ctx.target_format == HDR_COLOR_FORMAT;
        let mut bound = false;
        for entry in &self.frame {
            if entry.draw.instance_count == 0 {
                continue;
            }
            if !bound {
                // Still compiling: the fields draw next frame.
                let colour = if is_hdr {
                    pipeline::COLOUR_HDR
                } else {
                    pipeline::COLOUR_LDR
                };
                let Some(pl) = gpu.pipelines.get(colour) else {
                    return;
                };
                pass.set_pipeline(pl);
                bound = true;
            }
            pass.set_bind_group(1, &entry.draw.uniform_bind_group, &[]);
            pass.set_bind_group(2, &entry.draw.instance_bind_group, &[]);
            ctx.meshes
                .draw_indexed_instanced(pass, entry.draw.shape, entry.draw.instance_count);
        }
    }

    fn draws_ldr(&self) -> bool {
        true
    }

    fn outline_mask(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &OutlineMaskContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let Some(gpu) = self.gpu.as_ref().filter(|g| g.drawn()) else {
            return;
        };
        let mut bound = false;
        for entry in &self.frame {
            let Some(instance_filter) = &entry.outline else {
                continue;
            };
            if !bound {
                let Some(pl) = gpu.pipelines.get(pipeline::MASK) else {
                    return;
                };
                pass.set_pipeline(pl);
                bound = true;
            }
            pass.set_bind_group(1, &entry.draw.uniform_bind_group, &[]);
            pass.set_bind_group(2, &entry.draw.instance_bind_group, &[]);
            match instance_filter {
                None => {
                    ctx.meshes.draw_indexed_instanced(
                        pass,
                        entry.draw.shape,
                        entry.draw.instance_count,
                    );
                }
                Some(indices) => {
                    for &i in indices {
                        ctx.meshes
                            .draw_indexed_instance_range(pass, entry.draw.shape, i..i + 1);
                    }
                }
            }
        }
    }

    fn surface_mask(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &viewport_lib::plugin_api::SurfaceMaskContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let Some(gpu) = self.gpu.as_ref().filter(|g| g.drawn()) else {
            return;
        };
        let mut bound = false;
        for entry in &self.frame {
            let Some(value) = ctx.stamp_for(&entry.settings) else {
                continue;
            };
            if !bound {
                let Some(pl) = gpu.pipelines.get(pipeline::SURFACE_MASK) else {
                    return;
                };
                pass.set_pipeline(pl);
                bound = true;
            }
            pass.set_stencil_reference(value);
            pass.set_bind_group(1, &entry.draw.uniform_bind_group, &[]);
            pass.set_bind_group(2, &entry.draw.instance_bind_group, &[]);
            ctx.meshes
                .draw_indexed_instanced(pass, entry.draw.shape, entry.draw.instance_count);
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
            // Use the largest half-extent across the field so the biggest
            // instance is fully covered, and size it in pixels at the centroid
            // of the samples rather than at the model origin, which may be far
            // away.
            let world_r = sample_extents(item)
                .iter()
                .map(|e| e[0].max(e[1]).max(e[2]))
                .fold(0.0_f32, f32::max)
                .max(0.01);
            let n = item.positions.len() as f32;
            let centroid = model.transform_point3(
                item.positions
                    .iter()
                    .map(|p| glam::Vec3::from(*p))
                    .sum::<glam::Vec3>()
                    / n,
            );
            let radius_px = viewport_lib::plugin_api::pick_helpers::world_radius_in_pixels(
                centroid,
                world_r,
                ctx.view_proj,
                ctx.viewport_size,
            );
            let Some(mut hit) = viewport_lib::picking::pick_gaussian_splat_cpu(
                ctx.click_pos,
                item.settings.pick_id.0,
                &item.positions,
                model,
                ctx.view_proj,
                ctx.viewport_size,
                radius_px,
            ) else {
                continue;
            };
            let toi = (hit.world_pos - ray.origin).dot(ray.direction).max(0.0);
            if wants_instance {
                if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                    hit.sub_object = Some(SubObjectRef::Instance(idx));
                }
            } else {
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
                let Some(p) = viewport_lib::plugin_api::pick_helpers::project_to_screen(
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
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PickPassContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        if !ctx.mask.intersects(PickMask::OBJECT | PickMask::INSTANCE) {
            return;
        }
        let Some(gpu) = self.gpu.as_ref().filter(|g| g.drawn()) else {
            return;
        };
        let mut bound = false;
        for entry in &self.frame {
            let Some(pick_bg) = &entry.pick_bind_group else {
                continue;
            };
            if entry.draw.instance_count == 0 {
                continue;
            }
            if !bound {
                let Some(pl) = gpu.pipelines.get(pipeline::PICK) else {
                    return;
                };
                pass.set_pipeline(pl);
                bound = true;
            }
            pass.set_bind_group(1, pick_bg, &[]);
            pass.set_bind_group(2, &entry.draw.instance_bind_group, &[]);
            ctx.meshes
                .draw_indexed_instanced(pass, entry.draw.shape, entry.draw.instance_count);
        }
    }

    fn resolve_sub_object(
        &self,
        _pick_id: PickId,
        primitive_index: u32,
        _world_pos: glam::Vec3,
        mask: PickMask,
    ) -> Option<SubObjectRef> {
        // The pick fragment writes the sample index into the primitive
        // channel; no device feature involved.
        mask.intersects(PickMask::INSTANCE)
            .then_some(SubObjectRef::Instance(primitive_index))
    }
}

/// This field's outline coverage for the frame: `None` when it is not outlined,
/// `Some(None)` for the whole field, `Some(Some(indices))` for a sub-selection.
fn outline_for(
    settings: &viewport_lib::ItemSettings,
    sub_selection: Option<&viewport_lib::renderer::SubSelectionRef>,
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
        .flat_map(|s| s.items())
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

impl TensorFieldPlugin {
    /// Number of fields the last `prepare` produced draw data for.
    pub fn drawn_count(&self) -> usize {
        self.frame.len()
    }

    /// Build one field's GPU data against the plugin's layouts.
    fn build(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        item: &TensorFieldItem,
        capacity: u32,
    ) -> TensorFieldGpuData {
        let layouts = self
            .layouts
            .get_or_insert_with(|| TensorFieldResources::new(device));
        let binds = resolve_bindings(resources, layouts, item);
        build_tensor_field(device, queue, &binds, item, capacity)
    }

    /// Pre-upload a tensor field and return its handle.
    pub(crate) fn upload(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        item: &TensorFieldItem,
    ) -> TensorFieldId {
        self.upload_with_capacity(device, queue, resources, item, 0)
    }

    /// Pre-upload a tensor field with room for `capacity` samples.
    ///
    /// The field draws the samples it was given and holds the rest as headroom,
    /// so a feed reserves once and then only writes.
    pub fn upload_with_capacity(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        item: &TensorFieldItem,
        capacity: u32,
    ) -> TensorFieldId {
        let gpu = self.build(device, queue, resources, item, capacity);
        self.stored.insert_sized(gpu)
    }

    /// Write part of a stored field's sample buffer.
    pub(crate) fn write_samples(
        &mut self,
        queue: &viewport_lib::gpu::Queue,
        id: TensorFieldId,
        first_element: u32,
        data: &[u8],
    ) -> viewport_lib::error::ViewportResult<()> {
        let Some(gpu) = self.stored.get_mut(id) else {
            return Err(self.stored.stale(id));
        };
        gpu.write_samples(queue, first_element, data)
    }

    /// Grow a stored field to hold at least `capacity` samples.
    pub(crate) fn reserve_stored(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        id: TensorFieldId,
        capacity: u32,
    ) -> viewport_lib::error::ViewportResult<()> {
        let Some(gpu) = self.stored.get_mut(id) else {
            return Err(self.stored.stale(id));
        };
        if gpu.reserve(device, queue, capacity) {
            self.stored.bump_revision(id);
            self.stored.recharge(id);
        }
        Ok(())
    }

    /// Set how many of a stored field's samples draw.
    pub(crate) fn set_stored_len(
        &mut self,
        id: TensorFieldId,
        len: u32,
    ) -> viewport_lib::error::ViewportResult<()> {
        let Some(gpu) = self.stored.get_mut(id) else {
            return Err(self.stored.stale(id));
        };
        gpu.set_live_len(len)
    }

    /// What a stored field's sample buffer holds, and how much of it draws.
    pub(crate) fn stored_extent(
        &self,
        id: TensorFieldId,
    ) -> Option<viewport_lib::plugin_api::Extent> {
        self.stored.get(id).map(|gpu| gpu.extent())
    }

    /// Drop a stored field. `false` when the handle does not resolve.
    pub(crate) fn drop_stored(&mut self, id: TensorFieldId) -> bool {
        self.stored.remove(id).is_some()
    }

    /// Replace the tensors behind a live handle, keeping the handle.
    pub(crate) fn replace(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        id: TensorFieldId,
        item: &TensorFieldItem,
    ) -> viewport_lib::error::ViewportResult<()> {
        if !self.stored.contains(id) {
            return Err(self.stored.stale(id));
        }
        if let Some(gpu) = self.stored.get_mut(id)
            && store::try_replace_in_place(queue, gpu, item)
        {
            // Same bytes, different contents: the store's charge still holds,
            // the revision must not.
            self.stored.bump_revision(id);
            return Ok(());
        }
        let gpu = self.build(device, queue, resources, item, 0);
        self.stored.replace_sized(id, gpu);
        Ok(())
    }

    /// Build a field's buffers on a worker thread. The handle is minted when
    /// [`take_upload_result`](Self::take_upload_result) collects the job.
    ///
    /// The colourmap view and the shared sampler are resolved here and cloned
    /// into the worker: they are the renderer's and a worker has no
    /// `DeviceResources` borrow.
    pub(crate) fn begin_upload(
        &mut self,
        jobs: &viewport_lib::resources::Jobs<'_>,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        item: TensorFieldItem,
    ) -> viewport_lib::resources::JobId {
        let layouts = self
            .layouts
            .get_or_insert_with(|| TensorFieldResources::new(device));
        let binds = resolve_bindings(resources, layouts, &item);
        let device = device.clone();
        let queue = queue.clone();
        jobs.submit_cpu(move || build_tensor_field(&device, &queue, &binds, &item, 0))
    }

    /// Store the field a finished job built and hand back its handle.
    pub(crate) fn take_upload_result(
        &mut self,
        jobs: &viewport_lib::resources::Jobs<'_>,
        id: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<TensorFieldId> {
        match jobs.status(id) {
            viewport_lib::resources::UploadStatus::Pending { .. } => {
                Err(viewport_lib::error::ViewportError::JobNotReady)
            }
            viewport_lib::resources::UploadStatus::Failed(e) => Err(e),
            _ => match jobs.take::<TensorFieldGpuData>(id) {
                Some(gpu) => Ok(self.stored.insert_sized(gpu)),
                None => Err(viewport_lib::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                }),
            },
        }
    }
}
