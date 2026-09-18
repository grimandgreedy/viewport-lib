//! The external instance set item type as an [`ItemTypePlugin`]: one uploaded
//! mesh drawn once per element of a consumer-owned GPU buffer.
//!
//! The buffer holds tightly packed `[x, y, z]` `f32` triples, 12 bytes per
//! instance, and is bound directly as a read-only storage buffer: no CPU copy,
//! no per-frame upload. Whatever the consumer's compute passes last wrote is
//! what renders, and synchronisation is queue-submission order. Register the
//! buffer once with
//! [`ViewportRenderer::create_external_instance_set`](crate::renderer::ViewportRenderer::create_external_instance_set),
//! then submit an [`ExternalInstancesItem`] on `SceneFrame::external_instances`
//! per frame. `first_instance` / `instance_count` on the item select a window
//! of the buffer, so several items can render disjoint regions of one pool.
//!
//! The instances are opaque and depth-written, so they draw in `paint`, inside
//! the scene pass, and occlude like ordinary opaque geometry. HDR path only;
//! no shadows, no picking, no culling.

mod pipeline;
pub(crate) mod store;
pub(crate) mod types;

use crate::plugin_api::{ItemFrameContext, ItemTypePlugin, PaintContext, PluginItemCollection};
use crate::renderer::ExternalInstancesItem;
use store::{ExternalInstanceSetStore, ExternalInstancesGpuData};
use types::{ExternalInstanceSetConfig, ExternalInstanceSetId};

pub(crate) const TYPE_NAME: &str = "viewport.external_instances";

impl PluginItemCollection for Vec<ExternalInstancesItem> {
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
pub(crate) struct ExternalInstancesPlugin {
    /// The registered sets, owned by the type that draws them.
    sets: ExternalInstanceSetStore,
    /// The group-1 layout every draw builds its bind group against. Created on
    /// registration, because a set can be registered before the first frame.
    bgl: Option<crate::gpu::BindGroupLayout>,
    gpu: Option<pipeline::ExternalInstancesGpu>,
    /// Per submitted item, rebuilt each prepare.
    frame: Vec<ExternalInstancesGpuData>,
}

impl ExternalInstancesPlugin {
    /// Register a consumer-owned positions buffer and return its handle.
    ///
    /// # Errors
    ///
    /// [`ExternalBufferUsageMissing`](crate::error::ViewportError::ExternalBufferUsageMissing)
    /// when `config.positions` was created without `STORAGE` usage, or
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) when
    /// `config.mesh_id` is not registered.
    pub(crate) fn create_set(
        &mut self,
        device: &crate::gpu::Device,
        resources: &crate::resources::DeviceResources,
        config: &ExternalInstanceSetConfig,
    ) -> crate::error::ViewportResult<ExternalInstanceSetId> {
        if !config
            .positions
            .usage()
            .contains(crate::gpu::BufferUsages::STORAGE)
        {
            return Err(crate::error::ViewportError::ExternalBufferUsageMissing {
                missing: "STORAGE",
            });
        }
        if resources.mesh_index_count(config.mesh_id).is_none() {
            return Err(crate::error::ViewportError::StaleHandle {
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
    /// [`ExternalBufferUsageMissing`](crate::error::ViewportError::ExternalBufferUsageMissing)
    /// when `positions` lacks `STORAGE` usage, or
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) when `id` does
    /// not resolve to a live set.
    pub(crate) fn set_buffer(
        &mut self,
        id: ExternalInstanceSetId,
        positions: crate::gpu::Buffer,
    ) -> crate::error::ViewportResult<()> {
        if !positions
            .usage()
            .contains(crate::gpu::BufferUsages::STORAGE)
        {
            return Err(crate::error::ViewportError::ExternalBufferUsageMissing {
                missing: "STORAGE",
            });
        }
        let count = self.sets.slot_count();
        let set = self
            .sets
            .get_mut(id)
            .ok_or(crate::error::ViewportError::StaleHandle {
                index: id.index(),
                count,
            })?;
        set.positions = positions;
        Ok(())
    }

    /// Drop a set. Items still naming it are skipped. The renderer's clone of
    /// the consumer's buffer is released; the allocation lives as long as the
    /// consumer holds a handle to it.
    pub(crate) fn drop_set(&mut self, id: ExternalInstanceSetId) {
        self.sets.remove(id);
    }

    /// Whether a handle still resolves to a live set.
    #[cfg(test)]
    pub(crate) fn contains(&self, id: ExternalInstanceSetId) -> bool {
        self.sets.contains(id)
    }

    /// The positions buffer a live set currently draws from.
    #[cfg(test)]
    pub(crate) fn positions(&self, id: ExternalInstanceSetId) -> Option<&crate::gpu::Buffer> {
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
        device: &crate::gpu::Device,
        _shared: &crate::plugin_api::SharedBindings<'_>,
    ) {
        self.bgl = Some(store::build_bgl(device));
    }

    fn on_device_recreated(&mut self, device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {
        self.bgl = Some(store::build_bgl(device));
        self.gpu = None;
        self.frame.clear();
    }

    fn prepare(
        &mut self,
        device: &crate::gpu::Device,
        _queue: &crate::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<crate::gpu::CommandBuffer> {
        self.frame.clear();
        let items = items
            .as_any()
            .downcast_ref::<Vec<ExternalInstancesItem>>()
            .expect("external instances collection is the SceneFrame field");
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
                .for_format(ctx.target_format == crate::resources::HDR_COLOR_FORMAT),
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::primitives;

    fn positions_buffer(
        device: &crate::gpu::Device,
        elements: usize,
        usage: crate::gpu::BufferUsages,
    ) -> crate::gpu::Buffer {
        device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("test_positions"),
            size: (elements * 12) as u64,
            usage,
            mapped_at_creation: false,
        })
    }

    #[test]
    fn create_and_drop_roundtrip() {
        let Some((device, _queue, mut resources)) =
            crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let plane = primitives::grid_plane(1.0, 1.0, 2, 2);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let mut plugin = ExternalInstancesPlugin::default();

        let buf = positions_buffer(&device, 8, crate::gpu::BufferUsages::STORAGE);
        let id = plugin
            .create_set(
                &device,
                &resources,
                &ExternalInstanceSetConfig::new(mesh_id, buf.clone()),
            )
            .unwrap();
        assert!(plugin.contains(id));

        // Re-point works.
        let bigger = positions_buffer(&device, 16, crate::gpu::BufferUsages::STORAGE);
        plugin.set_buffer(id, bigger).unwrap();
        assert_eq!(plugin.positions(id).unwrap().size(), 16 * 12);

        plugin.drop_set(id);
        assert!(!plugin.contains(id));

        // Dropped slot is reused by the next create.
        let id2 = plugin
            .create_set(
                &device,
                &resources,
                &ExternalInstanceSetConfig::new(mesh_id, buf),
            )
            .unwrap();
        assert_eq!(id2.index(), id.index());
        // ... but the dropped handle must not follow the slot to its new
        // occupant. Before the handle carried a generation, it did.
        assert_ne!(id, id2, "the reused slot must carry a new generation");
        assert!(
            !plugin.contains(id),
            "the dropped handle must not resolve to the slot's new occupant"
        );

        // Re-pointing a dropped id fails.
        plugin.drop_set(id2);
        let other = positions_buffer(&device, 4, crate::gpu::BufferUsages::STORAGE);
        let err = plugin.set_buffer(id2, other);
        assert!(matches!(
            err,
            Err(crate::error::ViewportError::StaleHandle { .. })
        ));
    }

    #[test]
    fn create_rejects_non_storage_buffer() {
        let Some((device, _queue, mut resources)) =
            crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let plane = primitives::grid_plane(1.0, 1.0, 2, 2);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let mut plugin = ExternalInstancesPlugin::default();

        let buf = positions_buffer(&device, 8, crate::gpu::BufferUsages::VERTEX);
        let err = plugin.create_set(
            &device,
            &resources,
            &ExternalInstanceSetConfig::new(mesh_id, buf),
        );
        assert!(matches!(
            err,
            Err(crate::error::ViewportError::ExternalBufferUsageMissing { missing: "STORAGE" })
        ));
    }

    /// A set holds the consumer's buffer by handle, not by copy, so it must not
    /// be counted a second time in the renderer's working-set figure.
    #[test]
    fn a_set_charges_no_resident_bytes() {
        let Some((device, _queue, mut resources)) =
            crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let plane = primitives::grid_plane(1.0, 1.0, 2, 2);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let mut plugin = ExternalInstancesPlugin::default();

        let buf = positions_buffer(&device, 4096, crate::gpu::BufferUsages::STORAGE);
        let _id = plugin
            .create_set(
                &device,
                &resources,
                &ExternalInstanceSetConfig::new(mesh_id, buf),
            )
            .unwrap();
        assert_eq!(plugin.resident_bytes(), 0);
    }
}
