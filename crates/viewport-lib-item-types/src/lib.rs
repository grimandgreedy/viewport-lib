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

mod curves;
mod external_instances;
mod gaussian_splat;
mod gpu_implicit;
mod gpu_marching_cubes;
mod gpu_particles;
mod helpers;
mod image_slice;
mod point_cloud;
mod shader;
mod sprite;
mod tensor_glyph;
mod volume;
mod volume_surface_slice;

pub use curves::{
    RibbonId, RibbonItem, RibbonPlugin, RibbonRefItem, StreamtubeId, StreamtubeItem,
    StreamtubePlugin, StreamtubeRefItem, TubeId, TubeItem, TubePlugin, TubeRefItem,
};
pub use external_instances::{
    ExternalInstanceSetConfig, ExternalInstancesItem, ExternalInstancesPlugin,
};
pub use gaussian_splat::{
    GaussianSplatData, GaussianSplatId, GaussianSplatItem, GaussianSplatPlugin, ShDegree,
};
pub use gpu_implicit::{
    GpuImplicitItem, GpuImplicitOptions, GpuImplicitPlugin, ImplicitBlendMode, ImplicitPrimitive,
};
pub use gpu_marching_cubes::{GpuMarchingCubesItem, GpuMarchingCubesPlugin, McVolumeId};
pub use gpu_particles::{
    EmitterConfig, ForceField, GpuParticleSystemConfig, GpuParticleSystemItem, GpuParticlesPlugin,
    ParticleMeshAlign, ParticleRender, SpawnShape, VelocityDist,
};
pub use image_slice::{ImageSliceItem, ImageSlicePlugin, SliceAxis};
pub use point_cloud::{
    PointCloudId, PointCloudItem, PointCloudPlugin, PointCloudRefItem, PointRenderMode,
};
pub use sprite::{
    SpriteInstanceSetId, SpriteInstanceSetRefItem, SpriteItem, SpriteLitParams, SpriteNormalMode,
    SpriteOrientation, SpritePlugin, SpriteSetId, SpriteSetRefItem, SpriteSizeMode,
};
pub use tensor_glyph::{
    TensorGlyphItem, TensorGlyphPlugin, TensorGlyphSetId, TensorGlyphSetRefItem,
};
pub use volume::{VolumeItem, VolumePlugin};
pub use volume_surface_slice::{VolumeSurfaceSliceItem, VolumeSurfaceSlicePlugin};

/// A handle the renderer's own id crate owns, re-exported so a consumer of this
/// crate does not have to name two crates to submit one item.
pub use viewport_lib_types::ids::{ExternalInstanceSetId, GpuParticleSystemId};

/// The name each item type registers and submits under.
pub const EXTERNAL_INSTANCES_TYPE_NAME: &str = external_instances::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const GPU_PARTICLES_TYPE_NAME: &str = gpu_particles::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const RIBBON_TYPE_NAME: &str = curves::RIBBON_TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const SPRITE_TYPE_NAME: &str = sprite::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const STREAMTUBE_TYPE_NAME: &str = curves::STREAMTUBE_TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const TENSOR_GLYPH_TYPE_NAME: &str = tensor_glyph::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const TUBE_TYPE_NAME: &str = curves::TUBE_TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const GAUSSIAN_SPLAT_TYPE_NAME: &str = gaussian_splat::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const GPU_IMPLICIT_TYPE_NAME: &str = gpu_implicit::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const GPU_MARCHING_CUBES_TYPE_NAME: &str = gpu_marching_cubes::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const IMAGE_SLICE_TYPE_NAME: &str = image_slice::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const POINT_CLOUD_TYPE_NAME: &str = point_cloud::TYPE_NAME;
/// See [`EXTERNAL_INSTANCES_TYPE_NAME`].
pub const VOLUME_TYPE_NAME: &str = volume::TYPE_NAME;
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
    all.extend(curves::shader_sources());
    all.extend(external_instances::shader_sources());
    all.extend(gaussian_splat::shader_sources());
    all.extend(gpu_implicit::shader_sources());
    all.extend(gpu_marching_cubes::shader_sources());
    all.extend(gpu_particles::shader_sources());
    all.extend(image_slice::shader_sources());
    all.extend(point_cloud::shader_sources());
    all.extend(sprite::shader_sources());
    all.extend(tensor_glyph::shader_sources());
    all.extend(volume::shader_sources());
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
    renderer.with_item_type_plugin(device, Box::new(GpuMarchingCubesPlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(VolumePlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(StreamtubePlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(TubePlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(TensorGlyphPlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(RibbonPlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(ExternalInstancesPlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(SpritePlugin::default()));
    renderer.with_item_type_plugin(device, Box::new(GpuParticlesPlugin::default()));
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
        plugin_mut::<GaussianSplatPlugin>(self, GAUSSIAN_SPLAT_TYPE_NAME)
            .upload(device, queue, data)
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

/// Implement the standard upload surface for one item type whose plugin
/// exposes the five store calls under their usual names.
macro_rules! standard_uploads {
    ($item:ty, $id:ty, $name:expr, $plugin:ty) => {
        impl viewport_lib::plugin_api::Uploads<$item> for ViewportRenderer {
            type Id = $id;

            fn upload(
                &mut self,
                device: &gpu::Device,
                queue: &gpu::Queue,
                item: &$item,
            ) -> viewport_lib::error::ViewportResult<$id> {
                let host = host::<$plugin>(self, $name);
                Ok(host.plugin.upload(device, queue, host.resources, item))
            }

            fn begin_upload(
                &mut self,
                device: &gpu::Device,
                queue: &gpu::Queue,
                item: $item,
            ) -> viewport_lib::error::ViewportResult<viewport_lib::resources::JobId> {
                let host = host::<$plugin>(self, $name);
                Ok(host
                    .plugin
                    .begin_upload(&host.jobs, device, queue, host.resources, item))
            }

            fn replace(
                &mut self,
                device: &gpu::Device,
                queue: &gpu::Queue,
                id: $id,
                item: &$item,
            ) -> viewport_lib::error::ViewportResult<()> {
                let host = host::<$plugin>(self, $name);
                host.plugin.replace(device, queue, host.resources, id, item)
            }
        }

        impl viewport_lib::plugin_api::Handles<$id> for ViewportRenderer {
            fn upload_result(
                &mut self,
                job: viewport_lib::resources::JobId,
            ) -> viewport_lib::error::ViewportResult<$id> {
                let host = host::<$plugin>(self, $name);
                host.plugin.take_upload_result(&host.jobs, job)
            }

            fn release(&mut self, id: $id) -> bool {
                plugin_mut::<$plugin>(self, $name).drop_stored(id)
            }
        }
    };
}

standard_uploads!(
    StreamtubeItem,
    StreamtubeId,
    STREAMTUBE_TYPE_NAME,
    StreamtubePlugin
);
standard_uploads!(TubeItem, TubeId, TUBE_TYPE_NAME, TubePlugin);
standard_uploads!(RibbonItem, RibbonId, RIBBON_TYPE_NAME, RibbonPlugin);
standard_uploads!(
    TensorGlyphItem,
    TensorGlyphSetId,
    TENSOR_GLYPH_TYPE_NAME,
    TensorGlyphPlugin
);

/// Sprites keep two stores behind one item struct: a batch drawn as one set,
/// and an instance set drawn per transform. `Uploads` carries one `Id` per
/// implementation, so it covers the plain set and the instance set keeps calls
/// of its own on [`SpriteInstanceUploads`].
impl viewport_lib::plugin_api::Uploads<SpriteItem> for ViewportRenderer {
    type Id = SpriteSetId;

    fn upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<SpriteSetId> {
        let host = host::<SpritePlugin>(self, SPRITE_TYPE_NAME);
        Ok(host.plugin.upload_set(device, queue, host.resources, item))
    }

    fn begin_upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: SpriteItem,
    ) -> viewport_lib::error::ViewportResult<viewport_lib::resources::JobId> {
        let host = host::<SpritePlugin>(self, SPRITE_TYPE_NAME);
        Ok(host
            .plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item))
    }

    fn replace(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: SpriteSetId,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<()> {
        let host = host::<SpritePlugin>(self, SPRITE_TYPE_NAME);
        host.plugin
            .replace_set(device, queue, host.resources, id, item)
    }
}

impl viewport_lib::plugin_api::Handles<SpriteSetId> for ViewportRenderer {
    fn upload_result(
        &mut self,
        job: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<SpriteSetId> {
        let host = host::<SpritePlugin>(self, SPRITE_TYPE_NAME);
        host.plugin.take_set_result(&host.jobs, job)
    }

    fn release(&mut self, id: SpriteSetId) -> bool {
        plugin_mut::<SpritePlugin>(self, SPRITE_TYPE_NAME).drop_set(id)
    }
}

/// The sprite instance-set store, which shares `SpriteItem` with the plain set
/// store and so cannot share its `Uploads` implementation. Its handle type is
/// its own, so taking a finished job and releasing a set go through
/// [`Handles`](viewport_lib::plugin_api::Handles) like every other store.
pub trait SpriteInstanceUploads {
    /// Upload an instance set, returning a handle valid until
    /// [`release`](viewport_lib::plugin_api::Handles::release).
    fn upload_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &SpriteItem,
    ) -> SpriteInstanceSetId;

    /// Start an off-thread upload of an instance set.
    fn begin_upload_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: SpriteItem,
    ) -> viewport_lib::resources::JobId;

    /// Replace the sprites behind an instance-set handle, keeping the handle.
    fn replace_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: SpriteInstanceSetId,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<()>;
}

impl SpriteInstanceUploads for ViewportRenderer {
    fn upload_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &SpriteItem,
    ) -> SpriteInstanceSetId {
        let host = host::<SpritePlugin>(self, SPRITE_TYPE_NAME);
        host.plugin
            .upload_instance_set(device, queue, host.resources, item)
    }

    fn begin_upload_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: SpriteItem,
    ) -> viewport_lib::resources::JobId {
        let host = host::<SpritePlugin>(self, SPRITE_TYPE_NAME);
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    fn replace_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: SpriteInstanceSetId,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<()> {
        let host = host::<SpritePlugin>(self, SPRITE_TYPE_NAME);
        host.plugin
            .replace_instance_set(device, queue, host.resources, id, item)
    }
}

impl viewport_lib::plugin_api::Handles<SpriteInstanceSetId> for ViewportRenderer {
    fn upload_result(
        &mut self,
        job: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<SpriteInstanceSetId> {
        let host = host::<SpritePlugin>(self, SPRITE_TYPE_NAME);
        host.plugin.take_instance_set_result(&host.jobs, job)
    }

    fn release(&mut self, id: SpriteInstanceSetId) -> bool {
        plugin_mut::<SpritePlugin>(self, SPRITE_TYPE_NAME).drop_instance_set(id)
    }
}

/// The marching-cubes volume surface, on the renderer.
///
/// `VolumeData` belongs to `viewport-lib-geometry` and this crate owns neither
/// it nor `ViewportRenderer`, so `Uploads<VolumeData>` cannot be written here:
/// no type in the implementation would be local to this crate. Two of these
/// calls are not uploads anyway, so the content-keyed half of the surface gets
/// verbs of its own. The handle is this crate's, so taking a finished job and
/// releasing a volume go through
/// [`Handles`](viewport_lib::plugin_api::Handles) as usual.
pub trait McVolumes {
    /// Upload a scalar field, pre-allocating every slab's buffers, and return
    /// its handle.
    fn upload_volume_for_mc(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        vol: &viewport_lib_geometry::marching_cubes::VolumeData,
    ) -> viewport_lib::error::ViewportResult<McVolumeId>;

    /// Start an off-thread upload of a scalar field. Ownership of `vol`
    /// transfers into the worker, and the handle is minted by
    /// [`upload_result`](viewport_lib::plugin_api::Handles::upload_result).
    fn begin_upload_volume_for_mc(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        vol: viewport_lib_geometry::marching_cubes::VolumeData,
    ) -> viewport_lib::resources::JobId;

    /// Feed a volume from a caller-supplied buffer, refreshed before every
    /// dispatch so the isosurface tracks it with no CPU upload.
    ///
    /// The buffer holds one `f32` per volume node in x-fastest order
    /// (`index = x + y * nx + z * nx * ny`), matching `VolumeData::data`,
    /// starting at `offset_bytes`. It needs `COPY_SRC` usage and
    /// `offset_bytes` must be a multiple of 4. The renderer keeps a clone of
    /// the buffer handle; if the consumer reallocates it, call this again with
    /// the new buffer.
    fn set_mc_scalar_source_buffer(
        &mut self,
        id: McVolumeId,
        buffer: gpu::Buffer,
        offset_bytes: u64,
    ) -> viewport_lib::error::ViewportResult<()>;

    /// Detach the external scalar source, freezing the isosurface at the last
    /// field copied in.
    fn clear_mc_scalar_source(&mut self, id: McVolumeId)
    -> viewport_lib::error::ViewportResult<()>;
}

impl McVolumes for ViewportRenderer {
    fn upload_volume_for_mc(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        vol: &viewport_lib_geometry::marching_cubes::VolumeData,
    ) -> viewport_lib::error::ViewportResult<McVolumeId> {
        plugin_mut::<GpuMarchingCubesPlugin>(self, GPU_MARCHING_CUBES_TYPE_NAME)
            .upload(device, queue, vol)
    }

    fn begin_upload_volume_for_mc(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        vol: viewport_lib_geometry::marching_cubes::VolumeData,
    ) -> viewport_lib::resources::JobId {
        let host = host::<GpuMarchingCubesPlugin>(self, GPU_MARCHING_CUBES_TYPE_NAME);
        host.plugin.begin_upload(&host.jobs, device, queue, vol)
    }

    fn set_mc_scalar_source_buffer(
        &mut self,
        id: McVolumeId,
        buffer: gpu::Buffer,
        offset_bytes: u64,
    ) -> viewport_lib::error::ViewportResult<()> {
        plugin_mut::<GpuMarchingCubesPlugin>(self, GPU_MARCHING_CUBES_TYPE_NAME).set_scalar_source(
            id,
            buffer,
            offset_bytes,
        )
    }

    fn clear_mc_scalar_source(
        &mut self,
        id: McVolumeId,
    ) -> viewport_lib::error::ViewportResult<()> {
        plugin_mut::<GpuMarchingCubesPlugin>(self, GPU_MARCHING_CUBES_TYPE_NAME)
            .clear_scalar_source(id)
    }
}

impl viewport_lib::plugin_api::Handles<McVolumeId> for ViewportRenderer {
    fn upload_result(
        &mut self,
        job: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<McVolumeId> {
        let host = host::<GpuMarchingCubesPlugin>(self, GPU_MARCHING_CUBES_TYPE_NAME);
        host.plugin.take_upload_result(&host.jobs, job)
    }

    fn release(&mut self, id: McVolumeId) -> bool {
        plugin_mut::<GpuMarchingCubesPlugin>(self, GPU_MARCHING_CUBES_TYPE_NAME).free(id)
    }
}

/// The GPU particle system surface, on the renderer.
///
/// A system owns a persistent particle buffer the simulation advances in
/// place, so creating one is not an upload of content and neither call fits
/// [`Uploads`](viewport_lib::plugin_api::Uploads).
pub trait GpuParticleSystems {
    /// Create a persistent GPU particle system, returning its handle.
    fn create_gpu_particle_system(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        config: &GpuParticleSystemConfig,
    ) -> GpuParticleSystemId;

    /// Release a system. The handle stops resolving and its buffers are freed.
    fn drop_gpu_particle_system(&mut self, id: GpuParticleSystemId);
}

impl GpuParticleSystems for ViewportRenderer {
    fn create_gpu_particle_system(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        config: &GpuParticleSystemConfig,
    ) -> GpuParticleSystemId {
        let host = host::<GpuParticlesPlugin>(self, GPU_PARTICLES_TYPE_NAME);
        host.plugin
            .create_system(device, queue, host.resources, config)
    }

    fn drop_gpu_particle_system(&mut self, id: GpuParticleSystemId) {
        plugin_mut::<GpuParticlesPlugin>(self, GPU_PARTICLES_TYPE_NAME).drop_system(id)
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
