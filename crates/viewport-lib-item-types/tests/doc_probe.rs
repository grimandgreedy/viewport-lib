//! Every name this crate promises a consumer, named once so a rename or a
//! dropped re-export fails the build here rather than in someone's project.
//!
//! Nothing is called: the point is that the paths resolve.

#![allow(unused_imports)]

use viewport_lib_item_types::{
    ColourSource, DensityRemap, Emission, EmissionCurve, EmitterConfig, ExternalInstanceSetConfig,
    ExternalInstanceSetId, ExternalInstanceUploads, ExternalInstancesItem, ForceField,
    GaussianSplatData, GaussianSplatId, GaussianSplatItem, GpuImplicitItem, GpuImplicitOptions,
    GpuMarchingCubesItem, GpuParticleSystemConfig, GpuParticleSystemId, GpuParticleSystemItem,
    GpuParticleSystems, ImageSliceItem, ImplicitBlendMode, ImplicitPrimitive, MAX_SCATTER_VOLUMES,
    McVolumeId, McVolumes, NoiseDriver, ParticleMeshAlign, ParticleRender, PointCloudId,
    PointCloudItem, PointCloudRefItem, PointRenderMode, RefractionParams, RibbonId, RibbonItem,
    RibbonRefItem, ScatterShape, ScatterVolume, ScatterVolumeItem, ShDegree, SliceAxis, SpawnShape,
    SpriteInstanceSetId, SpriteInstanceSetRefItem, SpriteInstanceUploads, SpriteItem,
    SpriteLitParams, SpriteNormalMode, SpriteOrientation, SpriteSetId, SpriteSetRefItem,
    SpriteSizeMode, StreamtubeId, StreamtubeItem, StreamtubeRefItem, TensorFieldId,
    TensorFieldItem, TensorFieldRefItem, TubeId, TubeItem, TubeRefItem, VectorFieldId,
    VectorFieldItem, VectorFieldRefItem, VelocityDist, VolumeItem, VolumeSurfaceSliceItem,
};

// The types that read as belonging to one of the above but stay in
// `viewport-lib`: `SpriteBlend` selects a pipeline for instanced mesh batches
// too, and the scatter settings are frame state rather than item data.
use viewport_lib::{ScatterQuality, ScatterSettings, SpriteBlend};

/// The upload surfaces, in the shape a consumer writes them.
#[allow(dead_code)]
fn upload_surfaces(
    renderer: &mut viewport_lib::renderer::ViewportRenderer,
    device: &viewport_lib::gpu::Device,
    queue: &viewport_lib::gpu::Queue,
    vol: viewport_lib_geometry::marching_cubes::VolumeData,
    cloud: &PointCloudItem,
) -> viewport_lib::error::ViewportResult<()> {
    use viewport_lib::plugin_api::{Handles, Uploads};

    // The standard pair, keyed on what is handed over and on the handle.
    let cloud_id: PointCloudId = renderer.upload(device, queue, cloud)?;
    renderer.release(cloud_id);

    // Marching cubes, whose content-keyed calls cannot go through `Uploads`.
    let mc_id = renderer.upload_volume_for_mc(device, queue, &vol)?;
    renderer.clear_mc_scalar_source(mc_id)?;
    renderer.release(mc_id);

    let job = renderer.begin_upload_volume_for_mc(device, queue, vol);
    let from_job: McVolumeId = renderer.upload_result(job)?;
    renderer.release(from_job);
    Ok(())
}

#[test]
fn the_public_names_resolve() {}
