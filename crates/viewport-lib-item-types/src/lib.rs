//! The item types viewport-lib ships with, as an ordinary consumer crate.
//!
//! Each type here is a [`ItemTypePlugin`](viewport_lib::plugin_api::ItemTypePlugin)
//! built against viewport-lib's public API, on the same footing as
//! `viewport-lib-terrain` and `viewport-lib-mesh-assembly`. Nothing in the
//! renderer knows these types exist: they own their item structs, their
//! handles, their shaders and their GPU storage, and they submit and register
//! the way any other plugin does.
//!
//! ```no_run
//! # use viewport_lib_item_types::*;
//! # let mut renderer: viewport_lib::renderer::ViewportRenderer = unimplemented!();
//! # let device: &viewport_lib::gpu::Device = unimplemented!();
//! install(&mut renderer, device);
//!
//! // Each frame:
//! # let mut frame: viewport_lib::FrameData = Default::default();
//! frame.scene.items_mut::<PointCloudItem>().push(PointCloudItem::default());
//! ```
//!
//! Polylines are not here, and that is deliberate rather than pending. The
//! line substrate they submit through is core machinery: isolines, scatter
//! bounds, volume boxes, clip outlines and the wireframe of several item types
//! all draw through it, and
//! [`ItemTypePlugin::wireframe_polylines`](viewport_lib::plugin_api::ItemTypePlugin::wireframe_polylines)
//! returns that same submission form. `PolylineItem` is the consumer-facing
//! face of a renderer subsystem, so it stays in viewport-lib.

mod point_cloud;
mod shader;

pub use point_cloud::{
    PointCloudId, PointCloudItem, PointCloudPlugin, PointCloudRefItem, PointRenderMode,
};

/// The name the point cloud item type registers and submits under.
pub const POINT_CLOUD_TYPE_NAME: &str = point_cloud::TYPE_NAME;

/// Every shader this crate compiles, as `(name, source)`, with the shared
/// sections already spliced in front of each body.
///
/// A body on its own does not compile: it declares no group-0 bindings and
/// calls helpers it does not define. This returns what the pipelines actually
/// hand to `create_shader_module`, which is what a validation pass wants.
pub fn shader_sources() -> Vec<(&'static str, String)> {
    point_cloud::shader_sources()
}

use viewport_lib::gpu;
use viewport_lib::renderer::ViewportRenderer;

/// Register every item type in this crate with `renderer`.
///
/// Registering one on its own is
/// `renderer.with_item_type_plugin(device, Box::new(PointCloudPlugin::default()))`;
/// this is the same call for each type in turn.
pub fn install(renderer: &mut ViewportRenderer, device: &gpu::Device) {
    renderer.with_item_type_plugin(device, Box::new(PointCloudPlugin::default()));
}

/// The point cloud upload surface, on the renderer.
///
/// A pre-uploaded cloud is drawn by naming its [`PointCloudId`] from a
/// [`PointCloudRefItem`], which costs nothing per frame beyond the model
/// matrix. Inline [`PointCloudItem`]s rebuild their buffers every frame
/// instead, which is what you want for data that changes every frame and not
/// for data that does not.
pub trait PointCloudUploads {
    /// Upload a point cloud for reuse across frames, returning its handle.
    fn upload_point_cloud(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &PointCloudItem,
    ) -> PointCloudId;

    /// Start an off-thread upload. Poll the returned job with the renderer's
    /// `upload_status` and take the handle from
    /// [`upload_result_point_cloud`](Self::upload_result_point_cloud).
    fn begin_upload_point_cloud(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: PointCloudItem,
    ) -> viewport_lib::resources::JobId;

    /// Take the handle from a finished
    /// [`begin_upload_point_cloud`](Self::begin_upload_point_cloud) job.
    fn upload_result_point_cloud(
        &mut self,
        id: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<PointCloudId>;

    /// Replace the points behind a handle, keeping the handle valid. `false`
    /// when the handle does not resolve.
    fn replace_point_cloud(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: PointCloudId,
        item: &PointCloudItem,
    ) -> bool;

    /// Release a point cloud. `false` when the handle does not resolve.
    fn drop_point_cloud(&mut self, id: PointCloudId) -> bool;
}

impl PointCloudUploads for ViewportRenderer {
    fn upload_point_cloud(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &PointCloudItem,
    ) -> PointCloudId {
        let host = point_cloud_host(self);
        host.plugin.upload(device, queue, host.resources, item)
    }

    fn begin_upload_point_cloud(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: PointCloudItem,
    ) -> viewport_lib::resources::JobId {
        let host = point_cloud_host(self);
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    fn upload_result_point_cloud(
        &mut self,
        id: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<PointCloudId> {
        let host = point_cloud_host(self);
        host.plugin.take_upload_result(&host.jobs, id)
    }

    fn replace_point_cloud(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: PointCloudId,
        item: &PointCloudItem,
    ) -> bool {
        let host = point_cloud_host(self);
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    fn drop_point_cloud(&mut self, id: PointCloudId) -> bool {
        point_cloud_host(self).plugin.drop_stored(id)
    }
}

/// The registered point cloud plugin, together with the renderer-owned job
/// runner and content arenas an upload needs.
fn point_cloud_host(
    renderer: &mut ViewportRenderer,
) -> viewport_lib::plugin_api::ItemTypeHost<'_, PointCloudPlugin> {
    renderer
        .item_type_plugin_host(point_cloud::TYPE_NAME)
        .expect("the point cloud item type must be registered; call install() first")
}
