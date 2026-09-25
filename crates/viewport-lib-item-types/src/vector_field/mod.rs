//! The vector field item type as an [`ItemTypePlugin`]: one instanced mesh per
//! sample, oriented along the sample's vector and sized and coloured from it.
//! Consumers submit [`VectorFieldItem`]s with
//! `frame.scene.items_mut::<VectorFieldItem>()`, or [`VectorFieldRefItem`]s to
//! draw a field uploaded once through
//! [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload). Both forms
//! arrive at this plugin under the one type name.

mod pipeline;
mod store;
mod types;

use store::{
    VectorFieldGpuData, VectorFieldResources, VectorFieldStore, build_vector_field, magnitudes,
    resolve_bindings, sample_sizes,
};
use viewport_lib::plugin_api::{
    ItemCollections, ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext,
    PickContext, PickPassContext, PickRay, PluginItem, RectPickContext,
};
use viewport_lib::renderer::{PickHit, PickId, PickMask, PickRectResult, SubObjectRef};
use viewport_lib::resources::HDR_COLOR_FORMAT;

pub use types::{VectorFieldId, VectorFieldItem, VectorFieldRefItem};

/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::shader::{lit_shader, scene_shader, wgsl_source};
    use viewport_lib::plugin_api::shared_wgsl;
    vec![
        (
            "vector_field.wgsl",
            lit_shader(
                &[shared_wgsl::SHARED_CLIP_VOLUME_WGSL],
                wgsl_source!("vector_field"),
            ),
        ),
        (
            "vector_field_pick.wgsl",
            scene_shader(&[], wgsl_source!("vector_field_pick")),
        ),
        (
            "vector_field_outline_mask.wgsl",
            scene_shader(&[], wgsl_source!("vector_field_outline_mask")),
        ),
    ]
}

pub(crate) const TYPE_NAME: &str = "vpl.vector_field";

impl PluginItem for VectorFieldItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

impl PluginItem for VectorFieldRefItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

#[derive(Default)]
pub struct VectorFieldPlugin {
    /// The pre-uploaded fields, owned by the type that draws them.
    stored: VectorFieldStore,
    /// The two layouts every upload builds its bind groups against. Created on
    /// registration, because an upload can arrive before the first frame.
    layouts: Option<VectorFieldResources>,
    gpu: Option<pipeline::VectorFieldGpu>,
    /// Per drawn field, rebuilt each prepare: the inline items first, then the
    /// references.
    frame: Vec<pipeline::VectorFieldFrame>,
    /// Every inline item from the last prepared frame, hidden included.
    /// Reference items are not here: their instances live on the GPU, so they
    /// answer the GPU pick only.
    pick_items: Vec<VectorFieldItem>,
}

impl ItemTypePlugin for VectorFieldPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn init_gpu(
        &mut self,
        device: &viewport_lib::gpu::Device,
        _shared: &viewport_lib::plugin_api::SharedBindings<'_>,
    ) {
        self.layouts = Some(VectorFieldResources::new(device));
    }

    fn resident_bytes(&self) -> u64 {
        self.stored.allocated_bytes()
    }

    fn on_device_recreated(
        &mut self,
        device: &viewport_lib::gpu::Device,
        _queue: &viewport_lib::gpu::Queue,
    ) {
        self.layouts = Some(VectorFieldResources::new(device));
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
        let items = items.of::<VectorFieldItem>();
        let refs = ctx.refs_of::<VectorFieldRefItem>();
        self.pick_items.clear();
        self.pick_items.extend_from_slice(items);
        if items.is_empty() && refs.is_empty() {
            return Vec::new();
        }
        let layouts = self
            .layouts
            .get_or_insert_with(|| VectorFieldResources::new(device));
        let stored = &self.stored;
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::VectorFieldGpu::new(device, ctx.resources, layouts));

        for item in items {
            if item.settings.hidden || item.positions.is_empty() {
                continue;
            }
            let binds = resolve_bindings(ctx.resources, layouts, item);
            let gpu_data = build_vector_field(device, queue, &binds, item);
            let pick_bind_group = (gpu_data.pick_id != PickId::NONE).then(|| {
                gpu.pick_bind_group(device, queue, gpu_data.pick_id, &gpu_data._uniform_buf)
            });
            let outline = ctx
                .outline_selected
                .then(|| outline_for(&item.settings, ctx.sub_selection))
                .flatten();
            self.frame.push(pipeline::VectorFieldFrame {
                gpu: gpu_data,
                pick_bind_group,
                outline,
            });
        }

        // Pre-uploaded references. The model matrix lives at offset 0 of the
        // uniform, so a reference re-places a stored field without rebuilding
        // its instance buffer.
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
            // The pick id comes from the reference, not from the upload: the
            // same stored field can be drawn twice under two ids.
            let pick_id = ref_item.settings.pick_id;
            let pick_bind_group = (pick_id != PickId::NONE)
                .then(|| gpu.pick_bind_group(device, queue, pick_id, &gpu_data._uniform_buf));
            gpu_data.pick_id = pick_id;
            self.frame.push(pipeline::VectorFieldFrame {
                gpu: gpu_data,
                pick_bind_group,
                outline: None,
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
            if entry.gpu.instance_count == 0 {
                continue;
            }
            if !bound {
                pass.set_pipeline(gpu.pipeline.for_format(is_hdr));
                bound = true;
            }
            pass.set_bind_group(1, &entry.gpu.uniform_bind_group, &[]);
            pass.set_bind_group(2, &entry.gpu.instance_bind_group, &[]);
            ctx.meshes
                .draw_indexed_instanced(pass, entry.gpu.shape, entry.gpu.instance_count);
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
            pass.set_bind_group(1, &entry.gpu.uniform_bind_group, &[]);
            pass.set_bind_group(2, &entry.gpu.instance_bind_group, &[]);
            match instance_filter {
                None => {
                    ctx.meshes.draw_indexed_instanced(
                        pass,
                        entry.gpu.shape,
                        entry.gpu.instance_count,
                    );
                }
                Some(indices) => {
                    for &i in indices {
                        ctx.meshes
                            .draw_indexed_instance_range(pass, entry.gpu.shape, i..i + 1);
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
            // The shape extends along its vector from the base position, so the
            // proximity test runs against midpoints: a hit circle centred on the
            // base would reach a full length behind it when the sample points
            // away from the camera.
            let (midpoints, mean_len) = midpoints_and_length(item);
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
            let radius_px = viewport_lib::plugin_api::pick_helpers::world_radius_in_pixels(
                centroid,
                mean_len * 0.5,
                ctx.view_proj,
                ctx.viewport_size,
            );
            let Some(mut hit) = viewport_lib::picking::pick_gaussian_splat_cpu(
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
            // Report the sample's base position, not the midpoint the search ran
            // against.
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
            ctx.meshes
                .draw_indexed_instanced(pass, entry.gpu.shape, entry.gpu.instance_count);
        }
    }

    fn resolve_sub_object(
        &self,
        _pick_id: PickId,
        primitive_index: u32,
        _world_pos: glam::Vec3,
        mask: PickMask,
    ) -> Option<SubObjectRef> {
        // The pick fragment writes the sample index into the primitive channel;
        // no device feature involved.
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

/// Per-sample midpoints (base plus half the drawn extent) and the field's mean
/// drawn length, both in the item's own space.
fn midpoints_and_length(item: &VectorFieldItem) -> (Vec<[f32; 3]>, f32) {
    let mags = magnitudes(item);
    let sizes = sample_sizes(item, &mags);
    let lengths: Vec<f32> = sizes.iter().map(|s| s * item.scale).collect();
    let mean = if lengths.is_empty() {
        0.0
    } else {
        lengths.iter().sum::<f32>() / lengths.len() as f32
    };
    let midpoints = item
        .positions
        .iter()
        .enumerate()
        .map(|(i, pos)| {
            let p = glam::Vec3::from(*pos);
            let Some(v) = item.vectors.get(i) else {
                return *pos;
            };
            let dir = glam::Vec3::from(*v).normalize_or_zero();
            (p + dir * lengths[i] * 0.5).to_array()
        })
        .collect();
    (midpoints, mean.max(0.01))
}

impl VectorFieldPlugin {
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
        item: &VectorFieldItem,
    ) -> VectorFieldGpuData {
        let layouts = self
            .layouts
            .get_or_insert_with(|| VectorFieldResources::new(device));
        let binds = resolve_bindings(resources, layouts, item);
        build_vector_field(device, queue, &binds, item)
    }

    /// Pre-upload a vector field and return its handle.
    pub(crate) fn upload(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        item: &VectorFieldItem,
    ) -> VectorFieldId {
        let gpu = self.build(device, queue, resources, item);
        self.stored.insert_sized(gpu)
    }

    /// Drop a stored field. `false` when the handle does not resolve.
    pub(crate) fn drop_stored(&mut self, id: VectorFieldId) -> bool {
        self.stored.remove(id).is_some()
    }

    /// Replace the samples behind a live handle, keeping the handle.
    pub(crate) fn replace(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        id: VectorFieldId,
        item: &VectorFieldItem,
    ) -> viewport_lib::error::ViewportResult<()> {
        if !self.stored.contains(id) {
            return Err(self.stored.stale(id));
        }
        let gpu = self.build(device, queue, resources, item);
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
        item: VectorFieldItem,
    ) -> viewport_lib::resources::JobId {
        let layouts = self
            .layouts
            .get_or_insert_with(|| VectorFieldResources::new(device));
        let binds = resolve_bindings(resources, layouts, &item);
        let device = device.clone();
        let queue = queue.clone();
        jobs.submit_cpu(move || build_vector_field(&device, &queue, &binds, &item))
    }

    /// Store the field a finished job built and hand back its handle.
    pub(crate) fn take_upload_result(
        &mut self,
        jobs: &viewport_lib::resources::Jobs<'_>,
        id: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<VectorFieldId> {
        match jobs.status(id) {
            viewport_lib::resources::UploadStatus::Pending { .. } => {
                Err(viewport_lib::error::ViewportError::JobNotReady)
            }
            viewport_lib::resources::UploadStatus::Failed(e) => Err(e),
            _ => match jobs.take::<VectorFieldGpuData>(id) {
                Some(gpu) => Ok(self.stored.insert_sized(gpu)),
                None => Err(viewport_lib::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                }),
            },
        }
    }
}
