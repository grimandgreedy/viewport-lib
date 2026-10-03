//! Item types: point clouds, sprites, curves, volumes, vector and tensor
//! fields and the rest.
//!
//! Each type is an [`ItemTypePlugin`](viewport_lib::plugin_api::ItemTypePlugin)
//! built against viewport-lib's public API, on the same footing as
//! `viewport-lib-terrain` and `viewport-lib-mesh-assembly`. Nothing in the
//! renderer knows these types exist: they own their item structs, their
//! handles, their shaders and their GPU storage, and they submit and register
//! the way any other plugin does.
//!
//! Each type has a module of its own holding its item struct, its plugin, its
//! handles, the name it registers under (`TYPE_NAME`) and, where it has any,
//! the markers for its writable channels (`channels`). [`install`] registers
//! them all.
//!
//! ```no_run
//! # use viewport_lib_plugins::item_types::{self, point_cloud::PointCloudItem};
//! # let mut renderer: viewport_lib::renderer::ViewportRenderer = unimplemented!();
//! # let device: &viewport_lib::gpu::Device = unimplemented!();
//! item_types::install(&mut renderer, device);
//!
//! // Each frame:
//! # let mut frame: viewport_lib::FrameData = Default::default();
//! frame.scene.items_mut::<PointCloudItem>().push(PointCloudItem::default());
//! ```
//!
//! Polylines are not here, and that is deliberate rather than pending. The
//! line substrate they submit through is core machinery: scatter bounds,
//! volume boxes, clip outlines and the wireframe of several item types all
//! draw through it, and
//! [`ItemTypePlugin::wireframe_polylines`](viewport_lib::plugin_api::ItemTypePlugin::wireframe_polylines)
//! returns that same submission form. `PolylineItem` is the consumer-facing
//! face of a renderer subsystem, so it stays in viewport-lib.

// The upload surface of most types is one of these two shapes. They come
// before the modules so each type's `uploads.rs` can expand them.
/// Implement the standard upload surface for one item type whose plugin
/// exposes the five store calls under their usual names.
macro_rules! standard_uploads {
    ($item:ty, $id:ty, $name:expr, $plugin:ty) => {
        impl viewport_lib::plugin_api::Uploads<$item> for viewport_lib::renderer::ViewportRenderer {
            type Id = $id;

            fn upload(
                &mut self,
                device: &viewport_lib::gpu::Device,
                queue: &viewport_lib::gpu::Queue,
                item: &$item,
            ) -> viewport_lib::error::ViewportResult<$id> {
                let host = $crate::item_types::host::<$plugin>(self, $name);
                Ok(host.plugin.upload(device, queue, host.resources, item))
            }

            fn begin_upload(
                &mut self,
                device: &viewport_lib::gpu::Device,
                queue: &viewport_lib::gpu::Queue,
                item: $item,
            ) -> viewport_lib::error::ViewportResult<viewport_lib::resources::JobId> {
                let host = $crate::item_types::host::<$plugin>(self, $name);
                Ok(host
                    .plugin
                    .begin_upload(&host.jobs, device, queue, host.resources, item))
            }

            fn replace(
                &mut self,
                device: &viewport_lib::gpu::Device,
                queue: &viewport_lib::gpu::Queue,
                id: $id,
                item: &$item,
            ) -> viewport_lib::error::ViewportResult<()> {
                let host = $crate::item_types::host::<$plugin>(self, $name);
                host.plugin.replace(device, queue, host.resources, id, item)
            }
        }

        impl viewport_lib::plugin_api::Handles<$id> for viewport_lib::renderer::ViewportRenderer {
            fn upload_result(
                &mut self,
                job: viewport_lib::resources::JobId,
            ) -> viewport_lib::error::ViewportResult<$id> {
                let host = $crate::item_types::host::<$plugin>(self, $name);
                host.plugin.take_upload_result(&host.jobs, job)
            }

            fn release(&mut self, id: $id) -> bool {
                $crate::item_types::plugin_mut::<$plugin>(self, $name).drop_stored(id)
            }
        }
    };
}

/// Ranged writes into a stored field's interleaved sample buffer.
///
/// One channel per type rather than one per component: the record's fields are
/// derived together, and the store keeps no CPU copy to read the untouched ones
/// back from. So a write supplies whole samples.
macro_rules! field_sample_writes {
    ($marker:ty, $id:ty, $name:expr, $plugin:ty, encode = $encode:expr) => {
        impl viewport_lib::plugin_api::Writes<$marker>
            for viewport_lib::renderer::ViewportRenderer
        {
            fn write_range(
                &mut self,
                _channel: $marker,
                queue: &viewport_lib::gpu::Queue,
                id: $id,
                first_element: u32,
                data: &[<$marker as viewport_lib::plugin_api::Channel>::Input],
            ) -> viewport_lib::error::ViewportResult<()> {
                let bytes = $encode(data);
                $crate::item_types::plugin_mut::<$plugin>(self, $name).write_samples(
                    queue,
                    id,
                    first_element,
                    &bytes,
                )
            }

            fn reserve(
                &mut self,
                _channel: $marker,
                device: &viewport_lib::gpu::Device,
                queue: &viewport_lib::gpu::Queue,
                id: $id,
                capacity: u32,
            ) -> viewport_lib::error::ViewportResult<()> {
                $crate::item_types::plugin_mut::<$plugin>(self, $name)
                    .reserve_stored(device, queue, id, capacity)
            }

            fn set_len(
                &mut self,
                _channel: $marker,
                id: $id,
                len: u32,
            ) -> viewport_lib::error::ViewportResult<()> {
                $crate::item_types::plugin_mut::<$plugin>(self, $name).set_stored_len(id, len)
            }

            fn extent(
                &self,
                _channel: $marker,
                id: $id,
            ) -> Option<viewport_lib::plugin_api::Extent> {
                self.item_type_plugin::<$plugin>($name)?.stored_extent(id)
            }
        }
    };
}

pub mod curves;
pub mod decal;
pub mod external_instances;
pub mod gaussian_splat;
pub mod gpu_implicit;
pub mod gpu_marching_cubes;
pub mod gpu_particles;
pub mod image_slice;
pub mod point_cloud;
pub mod scatter_volume;
pub mod sprite;
pub mod surface_contour;
pub mod surface_lic;
pub mod tensor_field;
pub mod vector_field;
pub mod volume;
pub mod volume_surface_slice;

mod helpers;
mod shader;
mod sources;

/// Every shader the item types compile, as `(name, source)`, with the shared
/// sections already spliced in front of each body.
///
/// A body on its own does not compile: it declares no group-0 bindings and
/// calls helpers it does not define. This returns what the pipelines actually
/// hand to `create_shader_module`, which is what a validation pass wants.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    let mut all = Vec::new();
    all.extend(curves::shader_sources());
    all.extend(decal::shader_sources());
    all.extend(external_instances::shader_sources());
    all.extend(gaussian_splat::shader_sources());
    all.extend(gpu_implicit::shader_sources());
    all.extend(gpu_marching_cubes::shader_sources());
    all.extend(gpu_particles::shader_sources());
    all.extend(image_slice::shader_sources());
    all.extend(point_cloud::shader_sources());
    all.extend(scatter_volume::shader_sources());
    all.extend(sprite::shader_sources());
    all.extend(surface_contour::shader_sources());
    all.extend(surface_lic::shader_sources());
    all.extend(tensor_field::shader_sources());
    all.extend(vector_field::shader_sources());
    all.extend(volume::shader_sources());
    all.extend(volume_surface_slice::shader_sources());
    all.extend(helpers::shader_sources());
    all
}

/// Register every item type with `renderer`.
///
/// Registering one on its own is
/// `renderer.with_item_type_plugin(device, Box::new(point_cloud::PointCloudPlugin::default()))`;
/// this is the same call for each type in turn.
pub fn install(
    renderer: &mut viewport_lib::renderer::ViewportRenderer,
    device: &viewport_lib::gpu::Device,
) {
    // Registration order is draw order, and it is the order the renderer used
    // when these types were built into it. Keep it.
    renderer.with_item_type_plugin(device, Box::new(decal::DecalPlugin::default()));
    // After decals, so a decal on a flow surface takes the streaks too.
    renderer.with_item_type_plugin(device, Box::new(surface_lic::SurfaceLicPlugin::default()));
    // Draws in the scene pass, so the decals and streaks above, which run
    // after it, land over the lines.
    renderer.with_item_type_plugin(
        device,
        Box::new(surface_contour::SurfaceContourPlugin::default()),
    );
    renderer.with_item_type_plugin(device, Box::new(image_slice::ImageSlicePlugin::default()));
    renderer.with_item_type_plugin(
        device,
        Box::new(volume_surface_slice::VolumeSurfaceSlicePlugin::default()),
    );
    renderer.with_item_type_plugin(device, Box::new(point_cloud::PointCloudPlugin::default()));
    renderer.with_item_type_plugin(
        device,
        Box::new(gaussian_splat::GaussianSplatPlugin::default()),
    );
    renderer.with_item_type_plugin(device, Box::new(gpu_implicit::GpuImplicitPlugin::default()));
    renderer.with_item_type_plugin(
        device,
        Box::new(gpu_marching_cubes::GpuMarchingCubesPlugin::default()),
    );
    renderer.with_item_type_plugin(device, Box::new(volume::VolumePlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(curves::StreamtubePlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(curves::TubePlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(tensor_field::TensorFieldPlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(vector_field::VectorFieldPlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(curves::RibbonPlugin::default()));
    renderer.with_item_type_plugin(
        device,
        Box::new(external_instances::ExternalInstancesPlugin::default()),
    );
    renderer.with_item_type_plugin(device, Box::new(sprite::SpritePlugin::default()));
    renderer.with_item_type_plugin(
        device,
        Box::new(gpu_particles::GpuParticlesPlugin::default()),
    );
    // Scatter composites over the finished scene, so it registers after every
    // type whose pixels it absorbs.
    renderer.with_item_type_plugin(
        device,
        Box::new(scatter_volume::ScatterVolumePlugin::default()),
    );
}

/// Elements that are already in the buffer's layout go across untouched.
pub(crate) fn cast_bytes<T: bytemuck::Pod>(data: &[T]) -> std::borrow::Cow<'_, [u8]> {
    std::borrow::Cow::Borrowed(bytemuck::cast_slice(data))
}

/// A registered item type plugin, borrowed back as its concrete type.
pub(crate) fn plugin_mut<'a, T: viewport_lib::plugin_api::ItemTypePlugin>(
    renderer: &'a mut viewport_lib::renderer::ViewportRenderer,
    type_name: &str,
) -> &'a mut T {
    renderer
        .item_type_plugin_mut(type_name)
        .unwrap_or_else(|| panic!("{type_name} must be registered; call install() first"))
}

/// The same lookup, together with the renderer-owned job runner and content
/// arenas an upload needs.
pub(crate) fn host<'a, T: viewport_lib::plugin_api::ItemTypePlugin>(
    renderer: &'a mut viewport_lib::renderer::ViewportRenderer,
    type_name: &str,
) -> viewport_lib::plugin_api::ItemTypeHost<'a, T> {
    renderer
        .item_type_plugin_host(type_name)
        .unwrap_or_else(|| panic!("{type_name} must be registered; call install() first"))
}
