//! How a renderer uploads, writes and releases external instance sets.

use viewport_lib::gpu;
use viewport_lib::renderer::ViewportRenderer;

use super::*;
use crate::item_types::{host, plugin_mut};

/// The external instance set surface, on the renderer.
///
/// A set wraps a GPU buffer the consumer owns and writes: whatever their own
/// compute passes last left in it is what renders, with no CPU copy and no
/// per-frame upload.
pub trait ExternalInstanceUploads {
    /// Create an instance set drawn from a caller-owned positions buffer.
    fn create_external_instance_set(
        &mut self,
        device: &gpu::Device,
        config: &ExternalInstanceSetConfig,
    ) -> viewport_lib::error::ViewportResult<ExternalInstanceSetId>;

    /// Release a set. Items still naming it are skipped.
    fn drop_external_instance_set(&mut self, id: ExternalInstanceSetId);

    /// Re-point a set at a different positions buffer.
    fn set_external_instance_set_buffer(
        &mut self,
        id: ExternalInstanceSetId,
        positions: gpu::Buffer,
    ) -> viewport_lib::error::ViewportResult<()>;
}

impl ExternalInstanceUploads for ViewportRenderer {
    fn create_external_instance_set(
        &mut self,
        device: &gpu::Device,
        config: &ExternalInstanceSetConfig,
    ) -> viewport_lib::error::ViewportResult<ExternalInstanceSetId> {
        let host = host::<ExternalInstancesPlugin>(self, TYPE_NAME);
        host.plugin.create_set(device, host.resources, config)
    }

    fn drop_external_instance_set(&mut self, id: ExternalInstanceSetId) {
        plugin_mut::<ExternalInstancesPlugin>(self, TYPE_NAME).drop_set(id)
    }

    fn set_external_instance_set_buffer(
        &mut self,
        id: ExternalInstanceSetId,
        positions: gpu::Buffer,
    ) -> viewport_lib::error::ViewportResult<()> {
        plugin_mut::<ExternalInstancesPlugin>(self, TYPE_NAME).set_buffer(id, positions)
    }
}
