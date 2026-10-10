//! CPU-side IBL precomputation and environment map upload.
//!
//! Produces:
//! - **Irradiance map** (64x32 equirect) : diffuse hemisphere integral.
//! - **Prefiltered specular map** (256x128 equirect, 5 mip levels) : split-sum approximation.
//! - **BRDF integration LUT** (128x128) : Schlick-GGX split-sum second integral.
//!
//! All textures are Rgba16Float for HDR correctness.
//!
//! The IBL shader helpers (`dir_to_equirect_uv`, `sample_ibl_irradiance`,
//! `ibl_ambient`, etc.) live in `src/shaders/helpers/ambient.wgsl`, shared by
//! the four lit mesh shaders via the build-time `// #include` preprocessor.

use crate::util::par::*;
use std::f32::consts::PI;

use crate::resources::upload_jobs::{ApplyFn, JobId, JobProduct, ProgressHandle, UploadStatus};

use super::ibl_compute::{
    IBL_ENV_CAPACITY, IBL_IRR_H, IBL_IRR_W, IBL_PREFILTER_H, IBL_PREFILTER_MIPS, IBL_PREFILTER_W,
};

/// Image-based-lighting and environment-map GPU resources.
///
/// The `*_view` slots are `None` until the first environment upload; the
/// `fallback_*` textures satisfy the lit-pass bind group in the meantime. Owned
/// array textures are kept alive alongside their views. Grouped off
/// `DeviceResources` as a plain data holder; upload logic lives in this module.
///
/// Array layer 0 is never handed out. It holds a copy of the environment that
/// lights the scene, because the shaders sample layer 0 outside every zone.
/// Uploaded environments take layers 1.. ([`EnvSlot`]).
pub(crate) struct IblResources {
    /// Irradiance equirect array view, all layers (binding 7). None until the
    /// first environment upload lands.
    pub(crate) irradiance_view: Option<crate::gpu::TextureView>,
    /// Prefiltered specular equirect array view, all layers (binding 8). None
    /// until the first environment upload lands.
    pub(crate) prefiltered_view: Option<crate::gpu::TextureView>,
    /// BRDF integration LUT texture view (binding 9). None until the first
    /// environment upload; cached across later uploads (the LUT is
    /// scene-independent: function of roughness x N.V only).
    pub(crate) brdf_lut_view: Option<crate::gpu::TextureView>,
    /// Linear-clamp sampler (binding 10).
    pub(crate) sampler: crate::gpu::Sampler,
    /// Full-resolution source of the lighting environment (binding 11). None
    /// while no environment lights the scene. The skybox draws from each
    /// viewport's own bind group instead ([`SkyboxBinding`]).
    pub(crate) skybox_view: Option<crate::gpu::TextureView>,
    /// Fallback 1x1 black Rgba16Float texture for the skybox slot (binding 11)
    /// when no environment is loaded.
    #[allow(dead_code)]
    pub(crate) fallback_texture: crate::gpu::Texture,
    /// View of `fallback_texture`.
    pub(crate) fallback_view: crate::gpu::TextureView,
    /// Fallback 1x1x1 black `2d-array` texture for the irradiance / prefiltered
    /// array slots (bindings 7-8) before any environment is uploaded.
    #[allow(dead_code)]
    pub(crate) fallback_array_texture: crate::gpu::Texture,
    /// `2d-array` view of `fallback_array_texture`.
    pub(crate) fallback_array_view: crate::gpu::TextureView,
    /// Fallback 1x1 BRDF LUT placeholder; swapped for the real 128x128 LUT
    /// by the first environment upload. Bound to satisfy the bind group layout
    /// when no environment map has been uploaded yet.
    #[allow(dead_code)]
    pub(crate) fallback_brdf_texture: crate::gpu::Texture,
    pub(crate) fallback_brdf_view: crate::gpu::TextureView,
    /// Irradiance array texture (owned, kept alive for the view). Holds every
    /// environment layer; created on the first environment upload.
    pub(crate) irradiance_texture: Option<crate::gpu::Texture>,
    /// Prefiltered specular array texture (owned). Holds every environment layer.
    pub(crate) prefiltered_texture: Option<crate::gpu::Texture>,
    /// One entry per array layer. Entry 0 stands for the internal lighting copy
    /// and is never live.
    pub(crate) env_slots: Vec<EnvSlot>,
    /// The environment currently copied into layer 0, if any.
    pub(crate) lighting: Option<EnvironmentMapId>,
    /// Environment uploads in flight, by job, with the handle each will return.
    pub(crate) env_jobs: std::collections::HashMap<JobId, EnvironmentMapId>,
    /// The zones last set, including any whose environment has not landed yet.
    pub(crate) zones: Vec<EnvironmentZone>,
    /// `zones` must be rewritten to the GPU: an environment landed or was freed.
    pub(crate) zones_dirty: bool,
    /// Number of active environment zones written to the env-zone region of
    /// `indirect_light_buf`. 0 = the lighting environment everywhere; the
    /// shaders skip the per-fragment zone loop.
    pub(crate) env_zone_count: u32,
    /// Uploaded BRDF LUT texture (owned).
    #[allow(dead_code)]
    pub(crate) brdf_lut_texture: Option<crate::gpu::Texture>,
    /// Skybox fullscreen render pipeline (renders equirect environment as
    /// background). `None` until a frame draws a skybox; see
    /// [`DeviceResources::ensure_skybox_pipeline`](crate::resources::DeviceResources::ensure_skybox_pipeline).
    pub(crate) skybox_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Layout of each viewport's skybox bind group (group 1): the background
    /// uniform and the environment's source. Built with the pipeline.
    pub(crate) skybox_bgl: Option<crate::gpu::BindGroupLayout>,
}

/// One layer of the environment set.
#[derive(Default)]
pub(crate) struct EnvSlot {
    /// Bumped each time the slot is released, so old handles stop resolving.
    pub(crate) generation: u32,
    /// Handed out and not yet freed.
    pub(crate) live: bool,
    /// The full-resolution source, kept for drawing as the skybox. Set when the
    /// bake lands; a live slot without one is still baking.
    pub(crate) source: Option<(crate::gpu::Texture, crate::gpu::TextureView)>,
}

impl EnvSlot {
    /// Release the slot and invalidate every handle issued for it.
    fn release(&mut self) {
        self.live = false;
        self.source = None;
        self.generation = (self.generation + 1) & 0x00FF_FFFF;
    }
}

impl IblResources {
    /// The skybox pipeline, built by
    /// [`DeviceResources::ensure_skybox_pipeline`](crate::resources::DeviceResources::ensure_skybox_pipeline)
    /// by the prepare of any frame that shows a skybox.
    pub(crate) fn skybox_pipeline(&self) -> &crate::gpu::RenderPipeline {
        self.skybox_pipeline
            .as_ref()
            .expect("skybox pipeline missing; the scene prepare builds it")
    }

    /// Empty slots for every array layer.
    pub(crate) fn empty_env_slots() -> Vec<EnvSlot> {
        (0..IBL_ENV_CAPACITY).map(|_| EnvSlot::default()).collect()
    }

    /// The slot `id` names, if `id` is still live.
    fn live_slot(&self, id: EnvironmentMapId) -> Option<&EnvSlot> {
        let slot = self.env_slots.get(id.index() as usize)?;
        (id.index() != 0 && slot.live && slot.generation == id.generation()).then_some(slot)
    }

    /// Whether `id` names a live environment whose bake has landed.
    pub(crate) fn is_baked(&self, id: EnvironmentMapId) -> bool {
        self.live_slot(id).is_some_and(|s| s.source.is_some())
    }
}

pub use viewport_lib_types::ids::EnvironmentMapId;

use viewport_lib_types::effects::environment::{
    BackgroundSource, EnvironmentBackground, EnvironmentIntensity, EnvironmentLighting,
};

/// GPU layout of a viewport's background (skybox group 1, binding 0), matching
/// the WGSL `Background`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BackgroundUniform {
    intensity: f32,
    rotation: f32,
    /// Prefiltered mip to sample, or negative for the sharp source.
    blur_lod: f32,
    layer: u32,
}

/// A viewport's skybox bind group, built for one environment.
pub(crate) struct SkyboxBinding {
    env: EnvironmentMapId,
    buf: crate::gpu::Buffer,
    pub(crate) bind_group: crate::gpu::BindGroup,
}

/// The factor `intensity` scales `env`'s stored radiance by.
pub(crate) fn environment_multiplier(
    _ibl: &IblResources,
    _env: EnvironmentMapId,
    intensity: EnvironmentIntensity,
) -> f32 {
    match intensity {
        EnvironmentIntensity::Multiplier(m) => m,
        _ => 1.0,
    }
}

/// The environment a viewport draws behind the scene and how, or `None` for
/// the flat background colour.
pub(crate) fn resolve_background(
    ibl: &IblResources,
    background: &EnvironmentBackground,
    lighting: Option<&EnvironmentLighting>,
    lighting_intensity: EnvironmentIntensity,
) -> Option<(EnvironmentMapId, BackgroundUniform)> {
    let env = match background.source {
        BackgroundSource::LightingEnvironment => lighting?.environment,
        BackgroundSource::Environment(env) => env,
        _ => return None,
    };
    if !ibl.is_baked(env) {
        return None;
    }
    let intensity = background.intensity.unwrap_or(lighting_intensity);
    let blur_lod = if background.blur > 0.0 {
        background.blur.min(1.0) * (IBL_PREFILTER_MIPS - 1) as f32
    } else {
        -1.0
    };
    Some((
        env,
        BackgroundUniform {
            intensity: environment_multiplier(ibl, env, intensity),
            rotation: background
                .rotation
                .unwrap_or(lighting.map_or(0.0, |l| l.rotation)),
            blur_lod,
            layer: env.index(),
        },
    ))
}

/// Make `binding` draw `background`, building the skybox pipeline and the bind
/// group as needed. Returns whether the viewport draws a skybox.
pub(crate) fn prepare_background(
    resources: &mut crate::resources::DeviceResources,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    binding: &mut Option<SkyboxBinding>,
    background: Option<(EnvironmentMapId, BackgroundUniform)>,
) -> bool {
    let Some((env, uniform)) = background else {
        *binding = None;
        return false;
    };
    resources.ensure_skybox_pipeline(device);
    if binding.as_ref().is_none_or(|b| b.env != env) {
        let ibl = &resources.ibl;
        let (_, source) = ibl.env_slots[env.index() as usize]
            .source
            .as_ref()
            .expect("a resolved background is baked");
        let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("skybox_background_buf"),
            size: std::mem::size_of::<BackgroundUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("skybox_background_bg"),
            layout: ibl
                .skybox_bgl
                .as_ref()
                .expect("built with the skybox pipeline"),
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(source),
                },
            ],
        });
        *binding = Some(SkyboxBinding {
            env,
            buf,
            bind_group,
        });
    }
    let b = binding.as_ref().expect("set above");
    queue.write_buffer(&b.buf, 0, bytemuck::bytes_of(&uniform));
    true
}

/// Options for an environment upload. Nothing to set yet; pass
/// `EnvironmentOptions::default()`.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct EnvironmentOptions {}

/// Maximum number of environment-selection zones uploaded to the GPU at once.
/// Extra zones past this are dropped (with a log). The per-fragment zone loop
/// bounds on the active count, so a modest cap keeps the shader loop short.
pub const MAX_ENV_ZONES: usize = 64;

/// Byte stride of one `EnvZone` in the GPU buffer (matches the WGSL struct).
pub const ENV_ZONE_STRIDE_BYTES: usize = 48;

/// A world-space box that selects an environment for fragments inside it.
///
/// Fragments inside `bounds` are lit by `environment`; fragments within
/// `fade_distance` of the box cross-fade to whatever else covers them (other
/// zones, or the lighting environment where coverage is incomplete). Overlapping
/// zones blend by influence weight, so there is no hard seam at a boundary. Feed
/// a set through `ViewportRenderer::set_environment_zones`.
///
/// A distant environment (a region-selected sky) sets `parallax = false`. A local
/// reflection probe, captured at the box centre, sets `parallax = true` so its
/// reflection is box-projected against `bounds`;
/// `ViewportRenderer::capture_reflection_probe` returns one already configured.
#[derive(Copy, Clone, Debug)]
pub struct EnvironmentZone {
    /// World-space box this zone covers (and, for a probe, the parallax proxy).
    pub bounds: crate::scene::aabb::Aabb,
    /// Environment selected inside the box (from `upload_environment`). A zone
    /// naming a freed environment is skipped.
    pub environment: EnvironmentMapId,
    /// Outer falloff band, in world units, over which influence fades to zero.
    pub fade_distance: f32,
    /// Box-project the reflection against `bounds` (a local reflection probe).
    pub parallax: bool,
}

/// GPU layout of one environment zone (binding 19), matching the WGSL `EnvZone`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct EnvZoneGpu {
    center: [f32; 3],
    layer: u32,
    half_extents: [f32; 3],
    fade: f32,
    parallax: u32,
    _pad_probe: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<EnvZoneGpu>() == ENV_ZONE_STRIDE_BYTES);

/// Byte offset of the environment-zone region inside `indirect_light_buf`; the
/// per-object light-probe SH region occupies the buffer before it. The shader's
/// `ENV_ZONE_BASE` (in vec4 elements) in `helpers/scene_lighting.wgsl` must equal
/// this divided by 16.
pub(crate) const ENV_ZONE_REGION_OFFSET_BYTES: u64 =
    (crate::resources::light_probes::MAX_LIGHT_PROBE_OBJECTS
        * crate::resources::light_probes::SH_GPU_STRIDE_BYTES) as u64;
// Guard the CPU/shader coupling: scene_lighting.wgsl hardcodes ENV_ZONE_BASE as
// this value in vec4 elements (bytes / 16). Update both together.
const _: () = assert!(ENV_ZONE_REGION_OFFSET_BYTES == 36864 * 16);

/// Set the environment-selection zones and upload them to the GPU buffer
/// (binding 19). Replaces any previous set; an empty slice clears zones (every
/// fragment reverts to the lighting environment). Zones past [`MAX_ENV_ZONES`]
/// are dropped. A zone whose environment is still baking is held back until it
/// lands; one whose environment is freed is dropped from the GPU set.
pub fn set_environment_zones(
    resources: &mut crate::resources::DeviceResources,
    queue: &crate::gpu::Queue,
    zones: &[EnvironmentZone],
) {
    let n = zones.len().min(MAX_ENV_ZONES);
    if zones.len() > MAX_ENV_ZONES {
        tracing::warn!(
            requested = zones.len(),
            max = MAX_ENV_ZONES,
            "environment zones exceed the cap; extra zones dropped"
        );
    }
    resources.ibl.zones = zones[..n].to_vec();
    write_environment_zones(resources, queue);
}

/// Write the zones whose environment is baked and record the count for the
/// `Lights` uniform.
pub(crate) fn write_environment_zones(
    resources: &mut crate::resources::DeviceResources,
    queue: &crate::gpu::Queue,
) {
    let ibl = &mut resources.ibl;
    ibl.zones_dirty = false;
    let gpu: Vec<EnvZoneGpu> = ibl
        .zones
        .iter()
        .filter(|z| ibl.is_baked(z.environment))
        .map(|z| EnvZoneGpu {
            center: z.bounds.center().into(),
            layer: z.environment.index(),
            half_extents: z.bounds.half_extents().into(),
            fade: z.fade_distance,
            parallax: u32::from(z.parallax),
            _pad_probe: [0; 3],
        })
        .collect();
    if !gpu.is_empty() {
        queue.write_buffer(
            &resources.lighting.indirect_buf,
            ENV_ZONE_REGION_OFFSET_BYTES,
            bytemuck::cast_slice(&gpu),
        );
    }
    resources.ibl.env_zone_count = gpu.len() as u32;
}

/// Clear all environment-selection zones. Fragments revert to the lighting
/// environment.
pub fn clear_environment_zones(resources: &mut crate::resources::DeviceResources) {
    resources.ibl.zones.clear();
    resources.ibl.zones_dirty = false;
    resources.ibl.env_zone_count = 0;
}

// -------------------------------------------------------------------------
// Public upload API
// -------------------------------------------------------------------------

/// Largest finite half float. Environment textures are `Rgba16Float`, so a
/// brighter texel (a sun disc in an HDRI can exceed it) would turn infinite and
/// spread through the convolutions.
const F16_MAX: f32 = 65504.0;

/// Upload an equirectangular environment, bake its lighting, and return its
/// handle. Blocks until the bake finishes.
///
/// `data` is a whole Z-up panorama. A float image (`TextureData::hdr`, linear
/// radiance as an `.hdr` or `.exr` file holds it) gives realistic light. An
/// 8-bit image is accepted and decoded by its colour space, sRGB through the
/// sRGB curve and linear divided by 255, but it tops out at 1.0, so the sun and
/// the bright sky lose their weight and the lighting comes out flat. Values are
/// clamped to 65504, the largest a half float holds.
///
/// The environment takes one slot of a fixed set and keeps its full-resolution
/// source, so it can be drawn as the skybox as well as light the scene. Release
/// it with [`free_environment`].
///
/// # Errors
///
/// `InvalidTextureData` or `InvalidTextureColourSpace` when `data` fails
/// validation, `UnsupportedTextureData` for a normal map or a compressed
/// payload, and `TooManyEnvironments` when every slot is in use.
pub fn upload_environment(
    resources: &mut crate::resources::DeviceResources,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    data: crate::TextureData,
    options: EnvironmentOptions,
) -> crate::error::ViewportResult<EnvironmentMapId> {
    let job = begin_upload_environment(resources, device, queue, data, options)?;
    let drained = drain_until_ready(resources, device, queue, job);
    let env = upload_result_environment(resources, job);
    drained.and(env)
}

/// Drive the upload-job runner until `id` is `Ready` (or `Failed`).
fn drain_until_ready(
    resources: &mut crate::resources::DeviceResources,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    id: JobId,
) -> crate::error::ViewportResult<()> {
    loop {
        // Retaining variant: this blocking drain pumps the runner many times,
        // and a reflection bake runs it per probe. Clearing the promotion window
        // here would reap a streaming consumer's in-flight mesh upload `Ready`
        // before their next poll observes it, stranding a deferred bind.
        resources.process_uploads_retaining(device, queue);
        match resources.upload_status(id) {
            UploadStatus::Ready => return Ok(()),
            UploadStatus::Failed(e) => return Err(e),
            UploadStatus::Pending { .. } => {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            UploadStatus::Unknown => {
                // The id was just issued and the only consumer of it is
                // this loop. Reaching Unknown means the runner reaped a
                // completed job between the previous Ready check and the
                // next status query, which the runner does not do.
                unreachable!("just-submitted job id disappeared");
            }
        }
    }
}

/// Start an asynchronous environment upload and return its job.
///
/// Validation, decoding and slot allocation happen here, so the errors listed
/// on [`upload_environment`] come back before anything is submitted. The bake
/// runs on a worker; once the job reports `Ready`, take the handle with
/// [`upload_result_environment`]. The next `prepare` picks the new textures up,
/// with no further call needed.
pub fn begin_upload_environment(
    resources: &mut crate::resources::DeviceResources,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    data: crate::TextureData,
    _options: EnvironmentOptions,
) -> crate::error::ViewportResult<JobId> {
    let (width, height, pixels) = environment_pixels(data)?;
    let layer =
        alloc_env_layer(resources).ok_or(crate::error::ViewportError::TooManyEnvironments {
            max: IBL_ENV_CAPACITY - 1,
        })?;
    let env =
        EnvironmentMapId::from_parts(layer, resources.ibl.env_slots[layer as usize].generation);
    let job = submit_bake(resources, device, queue, pixels, width, height, env);
    resources.ibl.env_jobs.insert(job, env);
    Ok(job)
}

/// Take the handle produced by a [`begin_upload_environment`] job.
///
/// # Errors
///
/// `JobNotReady` while the bake is running, the job's own error if it failed
/// (its slot is released), and `JobResultMissing` for an unknown job, one that
/// was not an environment upload, or one whose handle was already taken.
pub fn upload_result_environment(
    resources: &mut crate::resources::DeviceResources,
    job: JobId,
) -> crate::error::ViewportResult<EnvironmentMapId> {
    let Some(&env) = resources.ibl.env_jobs.get(&job) else {
        return Err(crate::error::ViewportError::JobResultMissing {
            reason: "unknown id or wrong upload type",
        });
    };
    if resources.ibl.is_baked(env) {
        resources.ibl.env_jobs.remove(&job);
        return Ok(env);
    }
    match resources.upload_status(job) {
        UploadStatus::Failed(e) => {
            resources.ibl.env_jobs.remove(&job);
            resources.ibl.env_slots[env.index() as usize].release();
            Err(e)
        }
        _ => Err(crate::error::ViewportError::JobNotReady),
    }
}

/// Release an environment's slot and its source. Returns `false` when `id` was
/// already freed.
///
/// A handle kept past this resolves to nothing: a zone naming it is skipped,
/// and if it lit the scene, the scene loses environment lighting until another
/// is selected. The slot is reused by a later upload, under a new handle.
pub fn free_environment(
    resources: &mut crate::resources::DeviceResources,
    id: EnvironmentMapId,
) -> bool {
    if resources.ibl.live_slot(id).is_none() {
        return false;
    }
    resources.ibl.env_slots[id.index() as usize].release();
    resources.ibl.zones_dirty = true;
    if resources.ibl.lighting == Some(id) {
        resources.ibl.lighting = None;
        resources.ibl.skybox_view = None;
        resources.camera_bind_groups_dirty = true;
    }
    true
}

/// Make array layer 0 hold `requested`, the environment that lights the scene,
/// and point binding 11 at its source. Returns whether it is baked and lights
/// the scene.
///
/// Runs in `prepare`. Copies only when the selection changes.
pub(crate) fn select_lighting_environment(
    resources: &mut crate::resources::DeviceResources,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    requested: EnvironmentMapId,
) -> bool {
    let selected = resources.ibl.is_baked(requested).then_some(requested);
    if selected == resources.ibl.lighting {
        return selected.is_some();
    }
    resources.ibl.lighting = selected;
    resources.camera_bind_groups_dirty = true;
    let Some(env) = selected else {
        resources.ibl.skybox_view = None;
        return false;
    };
    let ibl = &mut resources.ibl;
    let (Some(irradiance), Some(prefiltered)) = (&ibl.irradiance_texture, &ibl.prefiltered_texture)
    else {
        unreachable!("a baked environment implies the arrays exist");
    };
    let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
        label: Some("ibl_lighting_copy"),
    });
    let mut copy = |texture: &crate::gpu::Texture, mip: u32, width: u32, height: u32| {
        let at = |layer| crate::gpu::TexelCopyTextureInfo {
            texture,
            mip_level: mip,
            origin: crate::gpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: crate::gpu::TextureAspect::All,
        };
        encoder.copy_texture_to_texture(
            at(env.index()),
            at(0),
            crate::gpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    };
    copy(irradiance, 0, IBL_IRR_W, IBL_IRR_H);
    for mip in 0..IBL_PREFILTER_MIPS {
        copy(
            prefiltered,
            mip,
            (IBL_PREFILTER_W >> mip).max(1),
            (IBL_PREFILTER_H >> mip).max(1),
        );
    }
    queue.submit(std::iter::once(encoder.finish()));
    ibl.skybox_view = ibl.env_slots[env.index() as usize]
        .source
        .as_ref()
        .map(|(_, view)| view.clone());
    true
}

/// Validate `data` for an environment and decode it to linear RGBA f32,
/// clamped to the half-float range.
fn environment_pixels(
    data: crate::TextureData,
) -> crate::error::ViewportResult<(u32, u32, Vec<f32>)> {
    use crate::{ColourSpace, TexturePayload, TextureRejection, TextureRole, UploadSlot};
    data.validate()?;
    let reject = |reason| {
        Err(crate::error::ViewportError::UnsupportedTextureData {
            slot: UploadSlot::Environment,
            reason,
        })
    };
    if data.role() == TextureRole::NormalMap {
        return reject(TextureRejection::NormalMap);
    }
    if matches!(data.payload(), TexturePayload::Compressed { .. }) {
        return reject(TextureRejection::UnsupportedPayload);
    }
    let (width, height) = (data.width(), data.height());
    let srgb = data.colour_space() == ColourSpace::Srgb;
    let mut pixels = match data.into_payload() {
        TexturePayload::Rgba32F(pixels) => pixels,
        TexturePayload::Rgba8(bytes) => bytes
            .par_iter()
            .enumerate()
            .map(|(i, &b)| {
                let v = f32::from(b) / 255.0;
                if srgb && i % 4 != 3 {
                    viewport_lib_types::colour::srgb_to_linear(v)
                } else {
                    v
                }
            })
            .collect(),
        TexturePayload::Compressed { .. } => unreachable!("rejected above"),
    };
    pixels.par_iter_mut().for_each(|v| {
        // A NaN would poison every convolution it reaches; treat it as black.
        *v = if v.is_nan() {
            0.0
        } else {
            v.clamp(-F16_MAX, F16_MAX)
        };
    });
    Ok((width, height, pixels))
}

/// Submit a bake of `pixels` into `env`'s layer (GPU compute, or the CPU
/// fallback), creating the arrays on first use.
fn submit_bake(
    resources: &mut crate::resources::DeviceResources,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    pixels: Vec<f32>,
    width: u32,
    height: u32,
    env: EnvironmentMapId,
) -> JobId {
    let compute_supported = super::ibl_compute::compute_supported(device);
    let needs_brdf = resources.ibl.brdf_lut_texture.is_none();
    let (irr_array, pref_array) = ensure_ibl_arrays(resources, device, compute_supported);
    let layer = env.index();

    let mut runner = resources.jobs.lock().expect("upload job runner poisoned");
    if compute_supported {
        runner.submit_with_gpu(device, queue, move |dev, q, progress| {
            progress.set(0.1);
            let result = super::ibl_compute::bake_environment_layer(
                dev,
                q,
                &pixels,
                width,
                height,
                &irr_array,
                &pref_array,
                layer,
                needs_brdf,
            );
            progress.set(1.0);
            let source = (result.skybox_texture, result.skybox_view);
            let brdf = result.brdf_texture.zip(result.brdf_view);
            Ok(JobProduct::with_gpu_and_apply(
                result.submission,
                install_bake(env, source, brdf),
            ))
        })
    } else {
        runner.submit_with_gpu(device, queue, move |dev, q, progress| {
            run_cpu_path(
                dev,
                q,
                &pixels,
                width,
                height,
                needs_brdf,
                env,
                &irr_array,
                &pref_array,
                progress,
            )
        })
    }
}

/// Create the persistent irradiance / prefiltered arrays on first use and return
/// clonable handles for the worker to bake into.
fn ensure_ibl_arrays(
    resources: &mut crate::resources::DeviceResources,
    device: &crate::gpu::Device,
    compute: bool,
) -> (crate::gpu::Texture, crate::gpu::Texture) {
    if resources.ibl.irradiance_texture.is_none() {
        let (irr, pref) = super::ibl_compute::create_ibl_arrays(device, compute);
        resources.ibl.irradiance_texture = Some(irr);
        resources.ibl.prefiltered_texture = Some(pref);
    }
    (
        resources.ibl.irradiance_texture.clone().unwrap(),
        resources.ibl.prefiltered_texture.clone().unwrap(),
    )
}

/// Reserve the lowest free slot. Layer 0 is the internal lighting copy, so
/// allocation starts at 1. Returns `None` once every slot is in use.
fn alloc_env_layer(resources: &mut crate::resources::DeviceResources) -> Option<u32> {
    let (layer, slot) = resources
        .ibl
        .env_slots
        .iter_mut()
        .enumerate()
        .skip(1)
        .find(|(_, s)| !s.live)?;
    slot.live = true;
    Some(layer as u32)
}

/// Install a landed bake. The irradiance and prefiltered specular are already in
/// the arrays; this keeps the source in the slot, installs the shared BRDF LUT
/// if it was baked, and creates the array sampling views on the first bake.
fn install_bake(
    env: EnvironmentMapId,
    source: (crate::gpu::Texture, crate::gpu::TextureView),
    brdf: Option<(crate::gpu::Texture, crate::gpu::TextureView)>,
) -> ApplyFn {
    Box::new(move |resources: &mut crate::resources::DeviceResources| {
        let ibl = &mut resources.ibl;
        let mut rebind = false;
        if let Some((tex, view)) = brdf {
            ibl.brdf_lut_view = Some(view);
            ibl.brdf_lut_texture = Some(tex);
            rebind = true;
        }
        if ibl.irradiance_view.is_none() {
            ibl.irradiance_view = ibl
                .irradiance_texture
                .as_ref()
                .map(super::ibl_compute::array_binding_view);
            ibl.prefiltered_view = ibl
                .prefiltered_texture
                .as_ref()
                .map(super::ibl_compute::array_binding_view);
            rebind = true;
        }
        let slot = &mut ibl.env_slots[env.index() as usize];
        if slot.live && slot.generation == env.generation() {
            slot.source = Some(source);
            ibl.zones_dirty = true;
        }
        if rebind {
            resources.camera_bind_groups_dirty = true;
        }
    })
}

/// CPU IBL path executed on a worker thread.
///
/// Builds the irradiance, prefilter, and (optionally) BRDF LUT data on the CPU
/// and writes it into `env`'s layer of the shared arrays, then submits a flush so
/// the runner has a `SubmissionIndex` to gate on.
#[allow(clippy::too_many_arguments)]
fn run_cpu_path(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
    needs_brdf: bool,
    env: EnvironmentMapId,
    irradiance_array: &crate::gpu::Texture,
    prefilter_array: &crate::gpu::Texture,
    progress: &ProgressHandle,
) -> crate::error::ViewportResult<JobProduct> {
    let layer = env.index();
    progress.set(0.05);

    // 1. Full-resolution source, kept for the skybox.
    let skybox = upload_rgba16f(device, queue, pixels, width, height, "ibl_skybox");
    let skybox_view = skybox.create_view(&crate::gpu::TextureViewDescriptor::default());

    progress.set(0.15);

    // 2. Irradiance map, written into the target array layer.
    let irradiance_data = convolve_irradiance(pixels, width, height, IBL_IRR_W, IBL_IRR_H);
    write_layer_rgba16f(
        queue,
        irradiance_array,
        layer,
        0,
        &irradiance_data,
        IBL_IRR_W,
        IBL_IRR_H,
    );

    progress.set(0.55);

    // 3. Prefiltered specular mips, written into the same array layer.
    prefilter_specular(
        queue,
        pixels,
        width,
        height,
        IBL_PREFILTER_W,
        IBL_PREFILTER_H,
        IBL_PREFILTER_MIPS,
        prefilter_array,
        layer,
    );

    progress.set(0.9);

    // 4. BRDF integration LUT, only when no cached LUT exists. The LUT is
    // scene-independent so it is generated once and reused across env maps.
    let brdf = needs_brdf.then(|| {
        let brdf_size = super::ibl_compute::IBL_BRDF_SIZE;
        let brdf_data = generate_brdf_lut(brdf_size);
        let tex = upload_rgba16f(
            device,
            queue,
            &brdf_data,
            brdf_size,
            brdf_size,
            "ibl_brdf_lut",
        );
        let view = tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        (tex, view)
    });

    // 5. Flush so the runner has a submission to gate on. Implicit writes
    // queued above are folded into this submit by wgpu.
    let encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
        label: Some("ibl_flush"),
    });
    let submission = queue.submit(std::iter::once(encoder.finish()));

    progress.set(1.0);

    Ok(JobProduct::with_gpu_and_apply(
        submission,
        install_bake(env, (skybox, skybox_view), brdf),
    ))
}

/// Write f32 RGBA pixel data into one mip of one layer of an Rgba16Float array
/// texture (the CPU path's array-write helper).
fn write_layer_rgba16f(
    queue: &crate::gpu::Queue,
    texture: &crate::gpu::Texture,
    layer: u32,
    mip: u32,
    pixels: &[f32],
    width: u32,
    height: u32,
) {
    let half_data: Vec<u16> = pixels.iter().map(|&f| f32_to_f16(f)).collect();
    queue.write_texture(
        crate::gpu::TexelCopyTextureInfo {
            texture,
            mip_level: mip,
            origin: crate::gpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: crate::gpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&half_data),
        crate::gpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 8), // 4 x f16 = 8 bytes per pixel
            rows_per_image: Some(height),
        },
        crate::gpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

/// Upload f32 RGBA pixel data as an Rgba16Float GPU texture.
pub(crate) fn upload_rgba16f(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
    label: &str,
) -> crate::gpu::Texture {
    let mip_level_count = 1;
    let tex = device.create_texture(&crate::gpu::TextureDescriptor {
        label: Some(label),
        size: crate::gpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: crate::gpu::TextureDimension::D2,
        format: crate::gpu::TextureFormat::Rgba16Float,
        usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Convert f32 -> f16 for upload.
    let half_data: Vec<u16> = pixels.iter().map(|&f| f32_to_f16(f)).collect();
    queue.write_texture(
        crate::gpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: crate::gpu::Origin3d::ZERO,
            aspect: crate::gpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&half_data),
        crate::gpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 8), // 4 x f16 = 8 bytes per pixel
            rows_per_image: Some(height),
        },
        crate::gpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    tex
}

/// Sample an equirectangular HDR image at a Z-up world-space direction.
///
/// viewport-lib is Z-up: longitude is measured around the +Z axis in the XY
/// plane, latitude has +Z polar.
fn sample_equirect(pixels: &[f32], width: u32, height: u32, dir: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = dir;
    let phi = y.atan2(x); // -PI..PI (longitude around Z)
    let theta = z.clamp(-1.0, 1.0).asin(); // -PI/2..PI/2 (latitude: Z polar)
    let u = 0.5 + phi / (2.0 * PI);
    let v = 0.5 - theta / PI;
    let px = (u * width as f32).rem_euclid(width as f32);
    let py = (v * height as f32).clamp(0.0, height as f32 - 1.0);
    let ix = px as u32 % width;
    let iy = py as u32;
    let idx = (iy * width + ix) as usize * 4;
    if idx + 2 < pixels.len() {
        [pixels[idx], pixels[idx + 1], pixels[idx + 2]]
    } else {
        [0.0; 3]
    }
}

// -------------------------------------------------------------------------
// Irradiance convolution (hemisphere cosine-weighted sampling)
// -------------------------------------------------------------------------

fn convolve_irradiance(src: &[f32], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<f32> {
    let sample_delta = 0.05f32; // ~40 phi steps x ~20 theta steps = 800 samples
    let mut out = vec![0.0f32; (dst_w * dst_h * 4) as usize];

    // Per-row parallelism. Each row writes a disjoint slice of `out`, so
    // chunk by row stride and dispatch in parallel via rayon.
    let row_stride = (dst_w as usize) * 4;
    out.par_chunks_mut(row_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let v = y as f32 / dst_h as f32;
            let theta_n = PI * (0.5 - v); // latitude
            for x in 0..dst_w {
                let u = x as f32 / dst_w as f32;
                let phi_n = 2.0 * PI * (u - 0.5); // longitude

                // Normal direction for this texel (Z-up: latitude theta drives Z,
                // longitude phi spins around Z in the XY plane).
                let (st, ct) = theta_n.sin_cos();
                let (sp, cp) = phi_n.sin_cos();
                let normal = [ct * cp, ct * sp, st];

                // Build tangent frame.
                let up = if normal[2].abs() < 0.999 {
                    [0.0, 0.0, 1.0]
                } else {
                    [1.0, 0.0, 0.0]
                };
                let tangent = cross(up, normal);
                let tangent = normalize(tangent);
                let bitangent = cross(normal, tangent);

                let mut irr = [0.0f32; 3];
                let mut sample_count = 0.0f32;

                let mut s_phi = 0.0f32;
                while s_phi < 2.0 * PI {
                    let mut s_theta = 0.0f32;
                    while s_theta < 0.5 * PI {
                        let (sst, sct) = s_theta.sin_cos();
                        let (ssp, scp) = s_phi.sin_cos();
                        let ts = [sst * scp, sst * ssp, sct];
                        let dir = [
                            ts[0] * tangent[0] + ts[1] * bitangent[0] + ts[2] * normal[0],
                            ts[0] * tangent[1] + ts[1] * bitangent[1] + ts[2] * normal[1],
                            ts[0] * tangent[2] + ts[1] * bitangent[2] + ts[2] * normal[2],
                        ];
                        let c = sample_equirect(src, src_w, src_h, dir);
                        let w = sct * sst; // cos(theta) * sin(theta) for solid angle
                        irr[0] += c[0] * w;
                        irr[1] += c[1] * w;
                        irr[2] += c[2] * w;
                        sample_count += 1.0;
                        s_theta += sample_delta;
                    }
                    s_phi += sample_delta;
                }

                let scale = PI / sample_count;
                let idx = (x as usize) * 4;
                row[idx] = irr[0] * scale;
                row[idx + 1] = irr[1] * scale;
                row[idx + 2] = irr[2] * scale;
                row[idx + 3] = 1.0;
            }
        });
    out
}

// -------------------------------------------------------------------------
// Prefiltered specular (importance-sampled GGX)
// -------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn prefilter_specular(
    queue: &crate::gpu::Queue,
    src: &[f32],
    src_w: u32,
    src_h: u32,
    base_w: u32,
    base_h: u32,
    mip_levels: u32,
    dst: &crate::gpu::Texture,
    layer: u32,
) {
    let num_samples = 256u32;

    for mip in 0..mip_levels {
        let mip_w = (base_w >> mip).max(1);
        let mip_h = (base_h >> mip).max(1);
        let roughness = mip as f32 / (mip_levels - 1).max(1) as f32;
        let mut data = vec![0.0f32; (mip_w * mip_h * 4) as usize];

        let row_stride = (mip_w as usize) * 4;
        data.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(y, row)| {
                let v = y as f32 / mip_h as f32;
                let theta_n = PI * (0.5 - v);
                for x in 0..mip_w {
                    let u = x as f32 / mip_w as f32;
                    let phi_n = 2.0 * PI * (u - 0.5);
                    // Z-up: latitude theta drives Z, longitude phi spins around Z in the XY plane.
                    let (st, ct) = theta_n.sin_cos();
                    let (sp, cp) = phi_n.sin_cos();
                    let n = [ct * cp, ct * sp, st];
                    let r = n; // reflect = normal for prefilter
                    let v_dir = r;

                    let colour =
                        prefilter_sample(src, src_w, src_h, n, r, v_dir, roughness, num_samples);
                    let idx = (x as usize) * 4;
                    row[idx] = colour[0];
                    row[idx + 1] = colour[1];
                    row[idx + 2] = colour[2];
                    row[idx + 3] = 1.0;
                }
            });

        // Write this mip level into the target array layer.
        write_layer_rgba16f(queue, dst, layer, mip, &data, mip_w, mip_h);
    }
}

fn prefilter_sample(
    src: &[f32],
    src_w: u32,
    src_h: u32,
    n: [f32; 3],
    _r: [f32; 3],
    v: [f32; 3],
    roughness: f32,
    num_samples: u32,
) -> [f32; 3] {
    let mut colour = [0.0f32; 3];
    let mut total_weight = 0.0f32;
    let a = roughness * roughness;

    for i in 0..num_samples {
        let xi = hammersley(i, num_samples);
        let h = importance_sample_ggx(xi, n, a);
        let l = reflect(v, h);
        let n_dot_l = dot(n, l).max(0.0);

        if n_dot_l > 0.0 {
            let c = sample_equirect(src, src_w, src_h, l);
            colour[0] += c[0] * n_dot_l;
            colour[1] += c[1] * n_dot_l;
            colour[2] += c[2] * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    if total_weight > 0.0 {
        colour[0] /= total_weight;
        colour[1] /= total_weight;
        colour[2] /= total_weight;
    }
    colour
}

// -------------------------------------------------------------------------
// BRDF integration LUT (split-sum second integral)
// -------------------------------------------------------------------------

pub(crate) fn generate_brdf_lut(size: u32) -> Vec<f32> {
    let num_samples = 1024u32;
    let mut data = vec![0.0f32; (size * size * 4) as usize];

    let row_stride = (size as usize) * 4;
    data.par_chunks_mut(row_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let roughness = (y as f32 + 0.5) / size as f32;
            let roughness = roughness.max(0.01);
            for x in 0..size {
                let n_dot_v = (x as f32 + 0.5) / size as f32;
                let n_dot_v = n_dot_v.max(0.001);

                let (a, b) = integrate_brdf(n_dot_v, roughness, num_samples);
                let idx = (x as usize) * 4;
                row[idx] = a;
                row[idx + 1] = b;
                row[idx + 2] = 0.0;
                row[idx + 3] = 1.0;
            }
        });
    data
}

fn integrate_brdf(n_dot_v: f32, roughness: f32, num_samples: u32) -> (f32, f32) {
    let v = [(1.0 - n_dot_v * n_dot_v).sqrt(), 0.0, n_dot_v];
    let n = [0.0f32, 0.0, 1.0];
    let a = roughness * roughness;

    let mut a_out = 0.0f32;
    let mut b_out = 0.0f32;

    for i in 0..num_samples {
        let xi = hammersley(i, num_samples);
        let h = importance_sample_ggx(xi, n, a);
        let l = reflect(v, h);
        let n_dot_l = l[2].max(0.0);
        let n_dot_h = h[2].max(0.0);
        let v_dot_h = dot(v, h).max(0.0);

        if n_dot_l > 0.0 {
            let g = geometry_smith(n_dot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v).max(0.001);
            let fc = (1.0 - v_dot_h).powi(5);
            a_out += (1.0 - fc) * g_vis;
            b_out += fc * g_vis;
        }
    }
    let inv = 1.0 / num_samples as f32;
    (a_out * inv, b_out * inv)
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    let g1v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g1l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    g1v * g1l
}

// -------------------------------------------------------------------------
// Math utilities
// -------------------------------------------------------------------------

fn hammersley(i: u32, n: u32) -> [f32; 2] {
    [i as f32 / n as f32, radical_inverse_vdc(i)]
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    bits as f32 * 2.328_306_4e-10 // 0x100000000 as f32
}

fn importance_sample_ggx(xi: [f32; 2], n: [f32; 3], a: f32) -> [f32; 3] {
    let a2 = a * a;
    let phi = 2.0 * PI * xi[0];
    let cos_theta = ((1.0 - xi[1]) / (1.0 + (a2 - 1.0) * xi[1])).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    // Spherical to Cartesian (tangent space).
    let h_ts = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];

    // Build tangent frame from N.
    let up = if n[1].abs() < 0.999 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);

    normalize([
        h_ts[0] * tangent[0] + h_ts[1] * bitangent[0] + h_ts[2] * n[0],
        h_ts[0] * tangent[1] + h_ts[1] * bitangent[1] + h_ts[2] * n[1],
        h_ts[0] * tangent[2] + h_ts[1] * bitangent[2] + h_ts[2] * n[2],
    ])
}

fn reflect(v: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    let d = 2.0 * dot(v, n);
    [d * n[0] - v[0], d * n[1] - v[1], d * n[2] - v[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    if len < 1e-10 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

/// Convert f32 to IEEE 754 half-precision (f16) bits.
///
/// Wraps `half::f16::from_f32` which uses SIMD intrinsics and precomputed tables
/// where available. Called millions of times per environment upload, so the speed
/// of the underlying implementation matters.
#[inline]
fn f32_to_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::DeviceResources;

    fn make_solid_env(width: u32, height: u32, rgb: [f32; 3]) -> Vec<f32> {
        let mut v = Vec::with_capacity((width as usize) * (height as usize) * 4);
        for _ in 0..(width * height) {
            v.push(rgb[0]);
            v.push(rgb[1]);
            v.push(rgb[2]);
            v.push(1.0);
        }
        v
    }

    fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
                #[cfg(wgpu30)]
                apply_limit_buckets: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
    }

    fn make_resources(device: &crate::gpu::Device) -> DeviceResources {
        DeviceResources::new(device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1)
    }

    fn upload(
        resources: &mut DeviceResources,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        rgb: [f32; 3],
    ) -> crate::error::ViewportResult<EnvironmentMapId> {
        let data = crate::TextureData::hdr(8, 4, make_solid_env(8, 4, rgb));
        upload_environment(
            resources,
            device,
            queue,
            data,
            EnvironmentOptions::default(),
        )
    }

    /// Mean RGB of one layer of the irradiance array.
    fn irradiance_mean(
        resources: &DeviceResources,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        layer: u32,
    ) -> [f32; 3] {
        let texture = resources.ibl.irradiance_texture.as_ref().unwrap();
        let row = IBL_IRR_W * 8;
        let staging = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: None,
            size: (row * IBL_IRR_H) as u64,
            usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            crate::gpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: crate::gpu::TextureAspect::All,
            },
            crate::gpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: crate::gpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(IBL_IRR_H),
                },
            },
            crate::gpu::Extent3d {
                width: IBL_IRR_W,
                height: IBL_IRR_H,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        staging
            .slice(..)
            .map_async(crate::gpu::MapMode::Read, |_| {});
        device
            .poll(crate::gpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        let mut sum = [0.0f32; 3];
        {
            let mapped = crate::gpu::mapped_range(staging.slice(..));
            let halves: &[u16] = bytemuck::cast_slice(&mapped);
            for texel in halves.chunks_exact(4) {
                for c in 0..3 {
                    sum[c] += half::f16::from_bits(texel[c]).to_f32();
                }
            }
        }
        staging.unmap();
        let n = (IBL_IRR_W * IBL_IRR_H) as f32;
        sum.map(|v| v / n)
    }

    #[test]
    fn ibl_fallbacks_wired_before_any_upload() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let resources = make_resources(&device);
        let ibl = &resources.ibl;
        // Before any environment upload the view slots are empty but the
        // fallbacks and the slot table are wired, so the lit-pass bind group
        // stays valid.
        assert!(ibl.irradiance_view.is_none());
        assert!(ibl.prefiltered_view.is_none());
        assert!(ibl.brdf_lut_view.is_none());
        assert!(ibl.skybox_view.is_none());
        assert_eq!(ibl.env_slots.len(), IBL_ENV_CAPACITY as usize);
        assert!(ibl.env_slots.iter().all(|s| !s.live));
        assert_eq!(ibl.env_zone_count, 0);
        assert_eq!(
            ibl.fallback_array_texture.depth_or_array_layers(),
            1,
            "fallback array is a single black layer"
        );
    }

    #[test]
    fn invalid_size_returns_error_synchronously() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);

        // 2x2 image requires 16 floats. Pass 12 and confirm the error fires
        // before any job is submitted or any slot is taken.
        let data = crate::TextureData::hdr(2, 2, vec![0.0f32; 12]);
        let err =
            begin_upload_environment(&mut resources, &device, &queue, data, Default::default())
                .expect_err("invalid size should error");
        match err {
            crate::error::ViewportError::InvalidTextureData { expected, actual } => {
                assert_eq!(expected, 16);
                assert_eq!(actual, 12);
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(resources.uploads_pending(), 0);
        assert!(resources.ibl.env_slots.iter().all(|s| !s.live));
    }

    #[test]
    fn environment_rejects_normal_maps_and_compressed_payloads() {
        let cases = [
            (
                crate::TextureData::normal_map(2, 2, vec![128; 16]),
                crate::TextureRejection::NormalMap,
            ),
            (
                crate::TextureData::compressed(
                    4,
                    4,
                    crate::CompressedFormat::Bc6hRgb,
                    crate::ColourSpace::Linear,
                    vec![vec![0u8; 16]],
                ),
                crate::TextureRejection::UnsupportedPayload,
            ),
        ];
        for (data, expected) in cases {
            let err = environment_pixels(data).unwrap_err();
            assert!(
                matches!(
                    err,
                    crate::error::ViewportError::UnsupportedTextureData {
                        slot: crate::UploadSlot::Environment,
                        reason,
                    } if reason == expected
                ),
                "expected {expected:?}, got {err:?}"
            );
        }
    }

    /// 8-bit pixels decode by their colour space: sRGB through the curve,
    /// linear divided by 255, alpha always divided by 255.
    #[test]
    fn eight_bit_environment_decodes_by_colour_space() {
        let bytes = vec![128u8, 64, 255, 128];
        let (_, _, srgb) =
            environment_pixels(crate::TextureData::srgb(1, 1, bytes.clone())).unwrap();
        let (_, _, linear) = environment_pixels(crate::TextureData::linear(1, 1, bytes)).unwrap();
        let decode = viewport_lib_types::colour::srgb_to_linear;
        let expected_srgb = [
            decode(128.0 / 255.0),
            decode(64.0 / 255.0),
            1.0,
            128.0 / 255.0,
        ];
        let expected_linear = [128.0 / 255.0, 64.0 / 255.0, 1.0, 128.0 / 255.0];
        for c in 0..4 {
            assert!(
                (srgb[c] - expected_srgb[c]).abs() < 1e-6,
                "sRGB channel {c}"
            );
            assert!(
                (linear[c] - expected_linear[c]).abs() < 1e-6,
                "linear channel {c}"
            );
        }
    }

    /// A texel past the half-float range (a sun disc) is clamped rather than
    /// turning infinite, and a NaN becomes black, so the bake stays finite.
    #[test]
    fn hot_texels_bake_finite() {
        let mut px = make_solid_env(8, 4, [1.0, 1.0, 1.0]);
        px[0] = 1.0e6;
        px[5] = f32::INFINITY;
        px[10] = f32::NAN;
        let (w, h, px) = environment_pixels(crate::TextureData::hdr(8, 4, px)).unwrap();
        assert_eq!(px[0], F16_MAX);
        assert_eq!(px[5], F16_MAX);
        assert_eq!(px[10], 0.0);
        let irradiance = convolve_irradiance(&px, w, h, IBL_IRR_W, IBL_IRR_H);
        assert!(
            irradiance
                .iter()
                .all(|&v| half::f16::from_f32(v).to_f32().is_finite())
        );
    }

    #[test]
    fn begin_upload_completes_and_populates_ibl() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);
        assert!(resources.ibl.irradiance_view.is_none());

        let data = crate::TextureData::hdr(8, 4, make_solid_env(8, 4, [0.5, 0.6, 0.7]));
        let job =
            begin_upload_environment(&mut resources, &device, &queue, data, Default::default())
                .unwrap();
        assert_eq!(resources.uploads_pending(), 1);
        assert!(matches!(
            upload_result_environment(&mut resources, job),
            Err(crate::error::ViewportError::JobNotReady)
        ));

        // Drive the runner until the job lands. The CPU path takes around
        // 100 ms on this test image, so 100 iterations of 20 ms is plenty.
        let mut iterations = 0;
        let env = loop {
            resources.process_uploads(&device, &queue);
            match upload_result_environment(&mut resources, job) {
                Ok(env) => break env,
                Err(crate::error::ViewportError::JobNotReady) => {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                Err(e) => panic!("upload failed: {e:?}"),
            }
            iterations += 1;
            if iterations > 100 {
                panic!("environment upload did not complete in time");
            }
        };
        assert_eq!(env.index(), 1, "layer 0 is never handed out");

        assert!(resources.ibl.irradiance_view.is_some());
        assert!(resources.ibl.prefiltered_view.is_some());
        assert!(resources.ibl.brdf_lut_view.is_some());
        assert!(resources.camera_bind_groups_dirty);
        // The skybox follows the lighting selection, made in prepare.
        assert!(resources.ibl.skybox_view.is_none());
        assert!(select_lighting_environment(
            &mut resources,
            &device,
            &queue,
            env
        ));
        assert_eq!(resources.ibl.lighting, Some(env));
        assert!(resources.ibl.skybox_view.is_some());
        // The handle is taken once.
        assert!(matches!(
            upload_result_environment(&mut resources, job),
            Err(crate::error::ViewportError::JobResultMissing { .. })
        ));
    }

    #[test]
    fn uploads_take_the_next_layer_and_keep_the_brdf() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);

        let a = upload(&mut resources, &device, &queue, [0.5, 0.5, 0.5]).unwrap();
        assert!(resources.ibl.brdf_lut_texture.is_some());
        let b = upload(&mut resources, &device, &queue, [0.1, 0.9, 0.4]).unwrap();
        assert_eq!((a.index(), b.index()), (1, 2));
        assert!(resources.ibl.brdf_lut_texture.is_some());
        assert!(resources.all_uploads_complete());
    }

    /// Selecting an environment copies its bake into layer 0, which is what the
    /// shaders sample, and switching copies the other one in.
    #[test]
    fn selection_switches_the_lighting_layer() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);
        let red = upload(&mut resources, &device, &queue, [1.0, 0.0, 0.0]).unwrap();
        let green = upload(&mut resources, &device, &queue, [0.0, 1.0, 0.0]).unwrap();

        assert!(select_lighting_environment(
            &mut resources,
            &device,
            &queue,
            red
        ));
        let lit = irradiance_mean(&resources, &device, &queue, 0);
        assert!(lit[0] > 0.5 && lit[1] < 0.05, "red lights: {lit:?}");

        assert!(select_lighting_environment(
            &mut resources,
            &device,
            &queue,
            green
        ));
        let lit = irradiance_mean(&resources, &device, &queue, 0);
        assert!(lit[1] > 0.5 && lit[0] < 0.05, "selection switched: {lit:?}");

        assert!(select_lighting_environment(
            &mut resources,
            &device,
            &queue,
            red
        ));
        let lit = irradiance_mean(&resources, &device, &queue, 0);
        assert!(lit[0] > 0.5 && lit[1] < 0.05, "and back: {lit:?}");
    }

    #[test]
    fn freed_handles_resolve_to_nothing_and_slots_are_reused() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);
        let env = upload(&mut resources, &device, &queue, [0.5, 0.5, 0.5]).unwrap();
        assert!(select_lighting_environment(
            &mut resources,
            &device,
            &queue,
            env
        ));

        assert!(free_environment(&mut resources, env));
        assert!(!free_environment(&mut resources, env), "already freed");
        assert!(resources.ibl.skybox_view.is_none());
        assert!(!select_lighting_environment(
            &mut resources,
            &device,
            &queue,
            env
        ));

        let again = upload(&mut resources, &device, &queue, [0.2, 0.2, 0.2]).unwrap();
        assert_eq!(again.index(), env.index(), "the slot is reused");
        assert_ne!(again, env, "under a new handle");
        assert!(!select_lighting_environment(
            &mut resources,
            &device,
            &queue,
            env
        ));
        assert!(select_lighting_environment(
            &mut resources,
            &device,
            &queue,
            again
        ));
    }

    #[test]
    fn environment_set_is_capacity_bounded() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);

        // Layers 1..CAP-1 are the slots; the next request past the cap errors.
        let mut ids = Vec::new();
        for _ in 1..IBL_ENV_CAPACITY {
            ids.push(upload(&mut resources, &device, &queue, [0.3, 0.3, 0.3]).unwrap());
        }
        let err = upload(&mut resources, &device, &queue, [0.3, 0.3, 0.3])
            .expect_err("past-capacity upload should error");
        assert!(matches!(
            err,
            crate::error::ViewportError::TooManyEnvironments { max } if max == IBL_ENV_CAPACITY - 1
        ));
        assert!(free_environment(&mut resources, ids[3]));
        upload(&mut resources, &device, &queue, [0.3, 0.3, 0.3]).unwrap();
    }

    #[test]
    fn environment_zones_set_clear_and_cap() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);

        upload(&mut resources, &device, &queue, [0.5, 0.5, 0.5]).unwrap();
        let env = upload(&mut resources, &device, &queue, [0.9, 0.1, 0.1]).unwrap();

        let zone = EnvironmentZone {
            bounds: crate::scene::aabb::Aabb {
                min: glam::Vec3::splat(-1.0),
                max: glam::Vec3::splat(1.0),
            },
            environment: env,
            fade_distance: 0.5,
            parallax: true,
        };
        set_environment_zones(&mut resources, &queue, &[zone]);
        assert_eq!(resources.ibl.env_zone_count, 1);

        // Over the cap: the live count clamps to MAX_ENV_ZONES.
        let many = vec![zone; MAX_ENV_ZONES + 5];
        set_environment_zones(&mut resources, &queue, &many);
        assert_eq!(resources.ibl.env_zone_count, MAX_ENV_ZONES as u32);

        // Freeing the environment drops its zones on the next write.
        free_environment(&mut resources, env);
        assert!(resources.ibl.zones_dirty);
        write_environment_zones(&mut resources, &queue);
        assert_eq!(resources.ibl.env_zone_count, 0);

        set_environment_zones(&mut resources, &queue, &[zone]);
        clear_environment_zones(&mut resources);
        assert_eq!(resources.ibl.env_zone_count, 0);
    }
}
