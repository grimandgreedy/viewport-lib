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

mod external_instances;
mod gaussian_splat;
mod gpu_implicit;
mod image_slice;
mod point_cloud;
mod helpers;
mod shader;
mod volume_surface_slice;

pub use external_instances::{
    ExternalInstanceSetConfig, ExternalInstancesItem, ExternalInstancesPlugin,
};
pub use gaussian_splat::{
    GaussianSplatData, GaussianSplatId, GaussianSplatItem, GaussianSplatPlugin, ShDegree,
};
pub use gpu_implicit::{
    GpuImplicitItem, GpuImplicitOptions, GpuImplicitPlugin, ImplicitBlendMode, ImplicitPrimitive,
};
pub use image_slice::{ImageSliceItem, ImageSlicePlugin, SliceAxis};
pub use point_cloud::{
    PointCloudId, PointCloudItem, PointCloudPlugin, PointCloudRefItem, PointRenderMode,
};
pub use volume_surface_slice::{VolumeSurfaceSliceItem, VolumeSurfaceSlicePlugin};

/// A handle the renderer's own id crate owns, re-exported so a consumer of this
/// crate does not have to name two crates to submit one item.
pub use viewport_lib_types::ids::ExternalInstanceSetId;

/// The name each item type registers and submits under.
pub const EXTERNAL_INSTANCES_TYPE_NAME: &str = external_instances::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const GAUSSIAN_SPLAT_TYPE_NAME: &str = gaussian_splat::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const GPU_IMPLICIT_TYPE_NAME: &str = gpu_implicit::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const IMAGE_SLICE_TYPE_NAME: &str = image_slice::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const POINT_CLOUD_TYPE_NAME: &str = point_cloud::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const VOLUME_SURFACE_SLICE_TYPE_NAME: &str = volume_surface_slice::TYPE_NAME;

/// Every shader this crate compiles, as `(name, source)`, with the shared
/// sections already spliced in front of each body.
///
/// A body on its own does not compile: it declares no group-0 bindings and
/// calls helpers it does not define. This returns what the pipelines actually
/// hand to `create_shader_module`, which is what a validation pass wants.
pub fn shader_sources() -> Vec<(&'static str, String)> {
    let mut all = Vec::new();
    all.extend(external_instances::shader_sources());
    all.extend(gaussian_splat::shader_sources());
    all.extend(gpu_implicit::shader_sources());
    all.extend(image_slice::shader_sources());
    all.extend(point_cloud::shader_sources());
    all.extend(volume_surface_slice::shader_sources());
    all.extend(helpers::shader_sources());
    all
}

use viewport_lib::gpu;
use viewport_lib::renderer::ViewportRenderer;

/// Register every item type in this crate with `renderer`.
///
/// Registering one on its own is
/// `renderer.with_item_type_plugin(device, Box::new(PointCloudPlugin::default()))`;
/// this is the same call for each type in turn.
pub fn install(renderer: &mut ViewportRenderer, device: &gpu::Device) {
    // Registration order is draw order, and it is the order the renderer used
    // when these types were built into it. Keep it.
    renderer.with_item_type_plugin(device, Box::new(ImageSlicePlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(VolumeSurfaceSlicePlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(PointCloudPlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(GaussianSplatPlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(GpuImplicitPlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(ExternalInstancesPlugin::default()));
}

impl viewport_lib::plugin_api::Uploads<PointCloudItem> for ViewportRenderer {
    type Id = PointCloudId;

    fn upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &PointCloudItem,
    ) -> viewport_lib::error::ViewportResult<PointCloudId> {
        let host = host::<PointCloudPlugin>(self, POINT_CLOUD_TYPE_NAME);
        Ok(host.plugin.upload(device, queue, host.resources, item))
    }

    fn begin_upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: PointCloudItem,
    ) -> viewport_lib::error::ViewportResult<viewport_lib::resources::JobId> {
        let host = host::<PointCloudPlugin>(self, POINT_CLOUD_TYPE_NAME);
        Ok(host
            .plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item))
    }

    fn replace(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: PointCloudId,
        item: &PointCloudItem,
    ) -> viewport_lib::error::ViewportResult<()> {
        let host = host::<PointCloudPlugin>(self, POINT_CLOUD_TYPE_NAME);
        host.plugin.replace(device, queue, host.resources, id, item)
    }

}

impl viewport_lib::plugin_api::Handles<PointCloudId> for ViewportRenderer {
    fn upload_result(
        &mut self,
        job: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<PointCloudId> {
        let host = host::<PointCloudPlugin>(self, POINT_CLOUD_TYPE_NAME);
        host.plugin.take_upload_result(&host.jobs, job)
    }

    fn release(&mut self, id: PointCloudId) -> bool {
        plugin_mut::<PointCloudPlugin>(self, POINT_CLOUD_TYPE_NAME).drop_stored(id)
    }
}

impl viewport_lib::plugin_api::Uploads<GaussianSplatData> for ViewportRenderer {
    type Id = GaussianSplatId;

    fn upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        data: &GaussianSplatData,
    ) -> viewport_lib::error::ViewportResult<GaussianSplatId> {
        plugin_mut::<GaussianSplatPlugin>(self, GAUSSIAN_SPLAT_TYPE_NAME).upload(device, queue, data)
    }

    fn begin_upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        data: GaussianSplatData,
    ) -> viewport_lib::error::ViewportResult<viewport_lib::resources::JobId> {
        let host = host::<GaussianSplatPlugin>(self, GAUSSIAN_SPLAT_TYPE_NAME);
        host.plugin.begin_upload(&host.jobs, device, queue, data)
    }

    fn replace(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: GaussianSplatId,
        data: &GaussianSplatData,
    ) -> viewport_lib::error::ViewportResult<()> {
        plugin_mut::<GaussianSplatPlugin>(self, GAUSSIAN_SPLAT_TYPE_NAME)
            .replace(device, queue, id, data)
    }
}

impl viewport_lib::plugin_api::Handles<GaussianSplatId> for ViewportRenderer {
    fn upload_result(
        &mut self,
        job: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<GaussianSplatId> {
        let host = host::<GaussianSplatPlugin>(self, GAUSSIAN_SPLAT_TYPE_NAME);
        host.plugin.take_upload_result(&host.jobs, job)
    }

    fn release(&mut self, id: GaussianSplatId) -> bool {
        plugin_mut::<GaussianSplatPlugin>(self, GAUSSIAN_SPLAT_TYPE_NAME).free(id)
    }
}

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
        let host = host::<ExternalInstancesPlugin>(self, EXTERNAL_INSTANCES_TYPE_NAME);
        host.plugin.create_set(device, host.resources, config)
    }

    fn drop_external_instance_set(&mut self, id: ExternalInstanceSetId) {
        plugin_mut::<ExternalInstancesPlugin>(self, EXTERNAL_INSTANCES_TYPE_NAME).drop_set(id)
    }

    fn set_external_instance_set_buffer(
        &mut self,
        id: ExternalInstanceSetId,
        positions: gpu::Buffer,
    ) -> viewport_lib::error::ViewportResult<()> {
        plugin_mut::<ExternalInstancesPlugin>(self, EXTERNAL_INSTANCES_TYPE_NAME)
            .set_buffer(id, positions)
    }
}

/// A registered plugin of this crate, borrowed back as its concrete type.
fn plugin_mut<'a, T: viewport_lib::plugin_api::ItemTypePlugin>(
    renderer: &'a mut ViewportRenderer,
    type_name: &str,
) -> &'a mut T {
    renderer
        .item_type_plugin_mut(type_name)
        .unwrap_or_else(|| panic!("{type_name} must be registered; call install() first"))
}

/// The same lookup, together with the renderer-owned job runner and content
/// arenas an upload needs.
fn host<'a, T: viewport_lib::plugin_api::ItemTypePlugin>(
    renderer: &'a mut ViewportRenderer,
    type_name: &str,
) -> viewport_lib::plugin_api::ItemTypeHost<'a, T> {
    renderer
        .item_type_plugin_host(type_name)
        .unwrap_or_else(|| panic!("{type_name} must be registered; call install() first"))
}
