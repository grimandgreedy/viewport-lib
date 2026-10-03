//! The external instance set item type as an [`ItemTypePlugin`]: one uploaded
//! mesh drawn once per element of a consumer-owned GPU buffer.
//!
//! The buffer holds tightly packed `[x, y, z]` `f32` triples, 12 bytes per
//! instance, and is bound directly as a read-only storage buffer: no CPU copy,
//! no per-frame upload. Whatever the consumer's compute passes last wrote is
//! what renders, and synchronisation is queue-submission order. Register the
//! buffer once with
//! [`ViewportRenderer::create_external_instance_set`](viewport_lib::renderer::ViewportRenderer::create_external_instance_set),
//! then submit an [`ExternalInstancesItem`] with
//! `frame.scene.items_mut::<ExternalInstancesItem>()`
//! per frame. `first_instance` / `instance_count` on the item select a window
//! of the buffer, so several items can render disjoint regions of one pool.
//!
//! The instances are opaque and depth-written, so they draw in `paint`, inside
//! the scene pass, and occlude like ordinary opaque geometry. HDR path only;
//! no shadows, no picking, no culling.

mod pipeline;
pub(crate) mod store;
mod types;

use store::{ExternalInstanceSetStore, ExternalInstancesGpuData};
pub use types::{ExternalInstanceSetConfig, ExternalInstanceSetId, ExternalInstancesItem};
use viewport_lib::gpu;
use viewport_lib::plugin_api::{ItemCollections, ItemFrameContext, ItemTypePlugin, PaintContext};
pub const TYPE_NAME: &str = "vpl.external_instances";

/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::shader::{scene_shader, wgsl_source};
    vec![(
        "external_instances.wgsl",
        scene_shader(&[], wgsl_source!("external_instances")),
    )]
}

impl viewport_lib::plugin_api::PluginItem for ExternalInstancesItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

#[derive(Default)]
pub struct ExternalInstancesPlugin {
    /// The registered sets, owned by the type that draws them.
    sets: ExternalInstanceSetStore,
    /// The group-1 layout every draw builds its bind group against. Created on
    /// registration, because a set can be registered before the first frame.
    bgl: Option<gpu::BindGroupLayout>,
    gpu: Option<pipeline::ExternalInstancesGpu>,
    /// Per submitted item, rebuilt each prepare.
    frame: Vec<ExternalInstancesGpuData>,
}

impl ExternalInstancesPlugin {
    /// Register a consumer-owned positions buffer and return its handle.
    ///
    /// # Errors
    ///
    /// [`ExternalBufferUsageMissing`](viewport_lib::error::ViewportError::ExternalBufferUsageMissing)
    /// when `config.positions` was created without `STORAGE` usage, or
    /// [`StaleHandle`](viewport_lib::error::ViewportError::StaleHandle) when
    /// `config.mesh_id` is not registered.
    pub fn create_set(
        &mut self,
        device: &gpu::Device,
        resources: &viewport_lib::resources::DeviceResources,
        config: &ExternalInstanceSetConfig,
    ) -> viewport_lib::error::ViewportResult<ExternalInstanceSetId> {
        if !config
            .positions
            .usage()
            .contains(gpu::BufferUsages::STORAGE)
        {
            return Err(
                viewport_lib::error::ViewportError::ExternalBufferUsageMissing {
                    missing: "STORAGE",
                },
            );
        }
        if resources.mesh_index_count(config.mesh_id).is_none() {
            return Err(viewport_lib::error::ViewportError::StaleHandle {
                index: config.mesh_id.index(),
                count: 0,
            });
        }
        self.bgl.get_or_insert_with(|| store::build_bgl(device));
        Ok(self.sets.insert_sized(store::ExternalInstanceSet {
            mesh_id: config.mesh_id,
            positions: config.positions.clone(),
        }))
    }

    /// Re-point a set at a new positions buffer, for a consumer whose pool was
    /// reallocated. Without this the renderer keeps rendering its clone of the
    /// old allocation's last contents.
    ///
    /// # Errors
    ///
    /// [`ExternalBufferUsageMissing`](viewport_lib::error::ViewportError::ExternalBufferUsageMissing)
    /// when `positions` lacks `STORAGE` usage, or
    /// [`StaleHandle`](viewport_lib::error::ViewportError::StaleHandle) when `id` does
    /// not resolve to a live set.
    pub fn set_buffer(
        &mut self,
        id: ExternalInstanceSetId,
        positions: gpu::Buffer,
    ) -> viewport_lib::error::ViewportResult<()> {
        if !positions.usage().contains(gpu::BufferUsages::STORAGE) {
            return Err(
                viewport_lib::error::ViewportError::ExternalBufferUsageMissing {
                    missing: "STORAGE",
                },
            );
        }
        let count = self.sets.slot_count();
        let set = self
            .sets
            .get_mut(id)
            .ok_or(viewport_lib::error::ViewportError::StaleHandle {
                index: id.index(),
                count,
            })?;
        set.positions = positions;
        Ok(())
    }

    /// Drop a set. Items still naming it are skipped. The renderer's clone of
    /// the consumer's buffer is released; the allocation lives as long as the
    /// consumer holds a handle to it.
    pub fn drop_set(&mut self, id: ExternalInstanceSetId) {
        self.sets.remove(id);
    }

    /// Whether a handle still resolves to a live set.
    pub fn contains(&self, id: ExternalInstanceSetId) -> bool {
        self.sets.contains(id)
    }

    /// The positions buffer a live set currently draws from.
    pub fn positions(&self, id: ExternalInstanceSetId) -> Option<&gpu::Buffer> {
        self.sets.get(id).map(|set| &set.positions)
    }
}

impl ItemTypePlugin for ExternalInstancesPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    /// Build the layout at registration rather than on the first frame that
    /// draws a set: a host registers its buffers at startup, and the handle has
    /// to be usable straight away.
    fn init_gpu(
        &mut self,
        device: &gpu::Device,
        _shared: &viewport_lib::plugin_api::SharedBindings<'_>,
    ) {
        self.bgl = Some(store::build_bgl(device));
    }

    fn warm(&mut self, device: &gpu::Device, resources: &viewport_lib::DeviceResources) {
        let bgl = self.bgl.get_or_insert_with(|| store::build_bgl(device));
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::ExternalInstancesGpu::new(device, resources, bgl));
        // The type does not draw into the LDR pass.
        gpu.pipelines.get(pipeline::COLOUR_HDR);
    }

    fn on_device_recreated(&mut self, device: &gpu::Device, _queue: &gpu::Queue) {
        self.bgl = Some(store::build_bgl(device));
        self.gpu = None;
        self.frame.clear();
    }

    fn prepare(
        &mut self,
        device: &gpu::Device,
        _queue: &gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &ItemCollections<'_>,
    ) -> Vec<gpu::CommandBuffer> {
        self.frame.clear();
        let items = items.of::<ExternalInstancesItem>();
        if items.is_empty() {
            return Vec::new();
        }
        let bgl = self.bgl.get_or_insert_with(|| store::build_bgl(device));
        self.gpu
            .get_or_insert_with(|| pipeline::ExternalInstancesGpu::new(device, ctx.resources, bgl));

        for item in items {
            if item.settings.hidden || item.instance_count == 0 {
                continue;
            }
            let Some(set) = self.sets.get(item.set_id) else {
                continue;
            };
            if let Some(data) = store::build_draw_data(device, bgl, set, item) {
                self.frame.push(data);
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
        let member = if ctx.target_format == viewport_lib::resources::HDR_COLOR_FORMAT {
            pipeline::COLOUR_HDR
        } else {
            pipeline::COLOUR_LDR
        };
        // Still compiling: the instances draw next frame.
        let Some(pl) = gpu.pipelines.get(member) else {
            return;
        };
        pass.set_pipeline(pl);
        for gd in &self.frame {
            pass.set_bind_group(1, &gd.bind_group, &[]);
            // The instance range is the buffer window: `instance_index` in the
            // shader starts at `first_instance` for direct draws.
            ctx.meshes.draw_indexed_instance_range(
                pass,
                gd.mesh_id,
                gd.first_instance..gd.first_instance + gd.instance_count,
            );
        }
    }
}
