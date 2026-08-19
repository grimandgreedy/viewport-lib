//! CPU-side IBL precomputation and environment map upload.
//!
//! Produces:
//! - **Irradiance map** (64x32 equirect) : diffuse hemisphere integral.
//! - **Prefiltered specular map** (128x64 equirect, 5 mip levels) : split-sum approximation.
//! - **BRDF integration LUT** (128x128) : Schlick-GGX split-sum second integral.
//!
//! All textures are Rgba16Float for HDR correctness.
//!
//! The IBL shader helpers (`dir_to_equirect_uv`, `sample_ibl_irradiance`,
//! `ibl_ambient`, etc.) live in `src/shaders/helpers/ambient.wgsl`, shared by
//! the four lit mesh shaders via the build-time `// #include` preprocessor.

use rayon::prelude::*;
use std::f32::consts::PI;

use crate::resources::upload_jobs::{ApplyFn, JobId, JobProduct, ProgressHandle, UploadStatus};

use super::ibl_compute::{
    IBL_ENV_CAPACITY, IBL_IRR_H, IBL_IRR_W, IBL_PREFILTER_H, IBL_PREFILTER_MIPS, IBL_PREFILTER_W,
};

/// Image-based-lighting and environment-map GPU resources.
///
/// The `*_view` slots are `None` until the first `upload_environment_map`; the
/// `fallback_*` textures satisfy the lit-pass bind group in the meantime. Owned
/// array textures are kept alive alongside their views. Grouped off
/// `DeviceResources` as a plain data holder; upload logic lives in this module.
pub(crate) struct IblResources {
    /// Irradiance equirect array view, all layers (binding 7). None until the
    /// default environment is uploaded (layer 0). Its `is_some()` gates `ibl_enabled`.
    pub(crate) irradiance_view: Option<crate::gpu::TextureView>,
    /// Prefiltered specular equirect array view, all layers (binding 8). None
    /// until the default environment is uploaded.
    pub(crate) prefiltered_view: Option<crate::gpu::TextureView>,
    /// BRDF integration LUT texture view (binding 9). None until the first
    /// `upload_environment_map`; cached across subsequent uploads (the LUT is
    /// scene-independent: function of roughness x N.V only).
    pub(crate) brdf_lut_view: Option<crate::gpu::TextureView>,
    /// Linear-clamp sampler (binding 10).
    pub(crate) sampler: crate::gpu::Sampler,
    /// Skybox / full-res environment equirect texture view (binding 11). None until uploaded.
    pub(crate) skybox_view: Option<crate::gpu::TextureView>,
    /// Fallback 1x1 black Rgba16Float texture for the skybox slot (binding 11)
    /// when no environment is loaded.
    #[allow(dead_code)]
    pub(crate) fallback_texture: crate::gpu::Texture,
    /// View of `fallback_texture`.
    pub(crate) fallback_view: crate::gpu::TextureView,
    /// Fallback 1x1x1 black `2d-array` texture for the irradiance / prefiltered
    /// array slots (bindings 7-8) before the default environment is uploaded.
    #[allow(dead_code)]
    pub(crate) fallback_array_texture: crate::gpu::Texture,
    /// `2d-array` view of `fallback_array_texture`.
    pub(crate) fallback_array_view: crate::gpu::TextureView,
    /// Fallback 1x1 BRDF LUT placeholder; swapped for the real 128x128 LUT
    /// on the first `upload_environment_map` call. Bound to satisfy the bind
    /// group layout when no environment map has been uploaded yet.
    #[allow(dead_code)]
    pub(crate) fallback_brdf_texture: crate::gpu::Texture,
    pub(crate) fallback_brdf_view: crate::gpu::TextureView,
    /// Irradiance array texture (owned, kept alive for the view). Holds every
    /// environment layer; created on the first environment upload.
    #[allow(dead_code)]
    pub(crate) irradiance_texture: Option<crate::gpu::Texture>,
    /// Prefiltered specular array texture (owned). Holds every environment layer.
    #[allow(dead_code)]
    pub(crate) prefiltered_texture: Option<crate::gpu::Texture>,
    /// Next free array layer for an extra environment (`upload_environment`).
    /// Starts at 1; layer 0 is reserved for the scene default.
    pub(crate) env_next_layer: u32,
    /// Number of active environment zones written to the env-zone region of
    /// `indirect_light_buf`. 0 = default environment everywhere; the shaders skip
    /// the per-fragment zone loop.
    pub(crate) env_zone_count: u32,
    /// Uploaded BRDF LUT texture (owned).
    #[allow(dead_code)]
    pub(crate) brdf_lut_texture: Option<crate::gpu::Texture>,
    /// Uploaded skybox equirect texture (owned).
    #[allow(dead_code)]
    pub(crate) skybox_texture: Option<crate::gpu::Texture>,
    /// Skybox fullscreen render pipeline (renders equirect environment as background).
    pub(crate) skybox_pipeline: crate::gpu::RenderPipeline,
}

/// Handle to one environment in the indexed set.
///
/// Layer 0 is the scene default (uploaded via [`upload_environment_map`]); extra
/// environments from [`upload_environment`] take layers 1.. up to the fixed
/// [`IBL_ENV_CAPACITY`]. The value is the array-texture layer the environment's
/// irradiance and prefiltered specular occupy.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnvironmentMapId(pub(crate) u32);

impl EnvironmentMapId {
    /// The scene default environment (array layer 0).
    pub const DEFAULT: Self = Self(0);

    /// The array layer this environment occupies.
    pub fn index(self) -> u32 {
        self.0
    }
}

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
/// zones, or the default environment where coverage is incomplete). Overlapping
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
    /// Environment selected inside the box (from `upload_environment`).
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

/// Upload the active environment-selection zones to the GPU buffer (binding 19)
/// and record the count for the `Lights` uniform. Replaces any previous set;
/// an empty slice clears zones (every fragment reverts to the default
/// environment). Zones past [`MAX_ENV_ZONES`] are dropped.
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
    if n > 0 {
        let gpu: Vec<EnvZoneGpu> = zones[..n]
            .iter()
            .map(|z| EnvZoneGpu {
                center: z.bounds.center().into(),
                layer: z.environment.0,
                half_extents: z.bounds.half_extents().into(),
                fade: z.fade_distance,
                parallax: u32::from(z.parallax),
                _pad_probe: [0; 3],
            })
            .collect();
        queue.write_buffer(
            &resources.lighting.indirect_buf,
            ENV_ZONE_REGION_OFFSET_BYTES,
            bytemuck::cast_slice(&gpu),
        );
    }
    resources.ibl.env_zone_count = n as u32;
}

/// Clear all environment-selection zones. Fragments revert to the default
/// environment (array layer 0).
pub fn clear_environment_zones(resources: &mut crate::resources::DeviceResources) {
    resources.ibl.env_zone_count = 0;
}

// -------------------------------------------------------------------------
// Public upload API
// -------------------------------------------------------------------------

/// Upload an equirectangular HDR environment map as the scene default and
/// precompute its IBL textures (array layer 0).
///
/// `pixels` is row-major RGBA f32 (4 floats per pixel), `width`x`height`.
/// After this call, the camera bind groups must be rebuilt so shaders see
/// the new textures: call `rebuild_camera_bind_groups` on the renderer.
///
/// This entry point blocks the calling thread until the upload finishes.
/// `begin_upload_environment_map` returns immediately and reports completion
/// through the upload-job runner.
pub fn upload_environment_map(
    resources: &mut crate::resources::DeviceResources,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
) -> crate::error::ViewportResult<()> {
    let id =
        begin_upload_environment_map(resources, device, queue, pixels.to_vec(), width, height)?;
    drain_until_ready(resources, device, queue, id)
}

/// Upload an extra environment into a new array layer and return its handle.
///
/// Unlike [`upload_environment_map`], this does not touch the scene default or
/// the skybox: the environment lives at its own layer, ready to be selected per
/// fragment once zone selection lands. Blocks until the upload finishes. Errors
/// with `TooManyEnvironments` once the fixed [`IBL_ENV_CAPACITY`] is reached.
pub fn upload_environment(
    resources: &mut crate::resources::DeviceResources,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
) -> crate::error::ViewportResult<EnvironmentMapId> {
    let (id, env) =
        begin_upload_environment(resources, device, queue, pixels.to_vec(), width, height)?;
    drain_until_ready(resources, device, queue, id)?;
    Ok(env)
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

/// Start an asynchronous default-environment upload (array layer 0).
///
/// Returns the `JobId` of the submitted upload. The caller is expected to
/// drive `process_uploads` from the renderer's prepare path each frame; once
/// the returned id reports `Ready`, the IBL textures are live and the
/// caller's next call to `rebuild_camera_bind_groups` will pick them up.
///
/// Ownership of `pixels` transfers into the background worker.
pub fn begin_upload_environment_map(
    resources: &mut crate::resources::DeviceResources,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    pixels: Vec<f32>,
    width: u32,
    height: u32,
) -> crate::error::ViewportResult<JobId> {
    begin_upload_layer(
        resources,
        device,
        queue,
        pixels,
        width,
        height,
        EnvironmentMapId::DEFAULT,
    )
}

/// Start an asynchronous extra-environment upload into a freshly allocated
/// layer. See [`upload_environment`]; returns the `JobId` and the new handle.
pub fn begin_upload_environment(
    resources: &mut crate::resources::DeviceResources,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    pixels: Vec<f32>,
    width: u32,
    height: u32,
) -> crate::error::ViewportResult<(JobId, EnvironmentMapId)> {
    let layer =
        alloc_env_layer(resources).ok_or(crate::error::ViewportError::TooManyEnvironments {
            max: IBL_ENV_CAPACITY,
        })?;
    let env = EnvironmentMapId(layer);
    let id = begin_upload_layer(resources, device, queue, pixels, width, height, env)?;
    Ok((id, env))
}

/// Shared body for the default and extra uploads: validate, ensure the arrays
/// exist, then submit a bake into `env`'s layer (GPU compute or CPU fallback).
fn begin_upload_layer(
    resources: &mut crate::resources::DeviceResources,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    pixels: Vec<f32>,
    width: u32,
    height: u32,
    env: EnvironmentMapId,
) -> crate::error::ViewportResult<JobId> {
    let expected = (width as usize) * (height as usize) * 4;
    if pixels.len() != expected {
        return Err(crate::error::ViewportError::InvalidTextureData {
            expected,
            actual: pixels.len(),
        });
    }

    let compute_supported = super::ibl_compute::compute_supported(device);
    let needs_brdf = resources.ibl.brdf_lut_texture.is_none();
    let (irr_array, pref_array) = ensure_ibl_arrays(resources, device, compute_supported);
    let is_default = env == EnvironmentMapId::DEFAULT;
    let layer = env.0;

    let mut runner = resources.jobs.lock().expect("upload job runner poisoned");
    let id = if compute_supported {
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
            Ok(JobProduct::with_gpu_and_apply(
                result.submission.clone(),
                apply_layer_bake(result, is_default),
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
                is_default,
                layer,
                &irr_array,
                &pref_array,
                progress,
            )
        })
    };
    Ok(id)
}

/// Create the persistent irradiance / prefiltered arrays on first use and return
/// clonable handles for the worker to bake into. Idempotent: a re-upload of the
/// default reuses the existing arrays (preserving any extra layers) and re-bakes
/// layer 0 in place.
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

/// Reserve the next free array layer for an extra environment. Layer 0 is the
/// default, so allocation starts at 1. Returns `None` once the cap is reached.
fn alloc_env_layer(resources: &mut crate::resources::DeviceResources) -> Option<u32> {
    let next = resources.ibl.env_next_layer.max(1);
    if next >= IBL_ENV_CAPACITY {
        return None;
    }
    resources.ibl.env_next_layer = next + 1;
    Some(next)
}

/// Install the results of a GPU layer bake. The irradiance and prefiltered
/// specular are already in the arrays; this installs the shared BRDF LUT (if it
/// was baked) and, for the default, the skybox and the array sampling views that
/// gate `ibl_enabled`.
fn apply_layer_bake(result: super::ibl_compute::LayerBakeResult, is_default: bool) -> ApplyFn {
    Box::new(move |resources: &mut crate::resources::DeviceResources| {
        if let (Some(brdf_tex), Some(brdf_view)) = (result.brdf_texture, result.brdf_view) {
            resources.ibl.brdf_lut_view = Some(brdf_view);
            resources.ibl.brdf_lut_texture = Some(brdf_tex);
        }
        if is_default {
            resources.ibl.skybox_texture = Some(result.skybox_texture);
            resources.ibl.skybox_view = Some(result.skybox_view);
            resources.ibl.irradiance_view = resources
                .ibl
                .irradiance_texture
                .as_ref()
                .map(super::ibl_compute::array_binding_view);
            resources.ibl.prefiltered_view = resources
                .ibl
                .prefiltered_texture
                .as_ref()
                .map(super::ibl_compute::array_binding_view);
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
    is_default: bool,
    layer: u32,
    irradiance_array: &crate::gpu::Texture,
    prefilter_array: &crate::gpu::Texture,
    progress: &ProgressHandle,
) -> crate::error::ViewportResult<JobProduct> {
    progress.set(0.05);

    // 1. Full-resolution skybox (default only; extra environments have no sky).
    let skybox = if is_default {
        let tex = upload_rgba16f(device, queue, pixels, width, height, "ibl_skybox");
        let view = tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        Some((tex, view))
    } else {
        None
    };

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
    let (brdf_tex, brdf_view) = if needs_brdf {
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
        (Some(tex), Some(view))
    } else {
        (None, None)
    };

    // 5. Flush so the runner has a submission to gate on. Implicit writes
    // queued above are folded into this submit by wgpu.
    let encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
        label: Some("ibl_flush"),
    });
    let submission = queue.submit(std::iter::once(encoder.finish()));

    progress.set(1.0);

    Ok(JobProduct::with_gpu_and_apply(
        submission,
        Box::new(move |resources: &mut crate::resources::DeviceResources| {
            if let (Some(tex), Some(view)) = (brdf_tex, brdf_view) {
                resources.ibl.brdf_lut_view = Some(view);
                resources.ibl.brdf_lut_texture = Some(tex);
            }
            if is_default {
                if let Some((tex, view)) = skybox {
                    resources.ibl.skybox_texture = Some(tex);
                    resources.ibl.skybox_view = Some(view);
                }
                resources.ibl.irradiance_view = resources
                    .ibl
                    .irradiance_texture
                    .as_ref()
                    .map(super::ibl_compute::array_binding_view);
                resources.ibl.prefiltered_view = resources
                    .ibl
                    .prefiltered_texture
                    .as_ref()
                    .map(super::ibl_compute::array_binding_view);
            }
        }),
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
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
    }

    fn make_resources(device: &crate::gpu::Device) -> DeviceResources {
        DeviceResources::new(device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1)
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
        // fallbacks and the environment-layer cursor are wired, so the lit-pass
        // bind group stays valid. Guards the init-assembly nesting for the
        // always-present IBL fields.
        assert!(ibl.irradiance_view.is_none());
        assert!(ibl.prefiltered_view.is_none());
        assert!(ibl.brdf_lut_view.is_none());
        assert!(ibl.skybox_view.is_none());
        assert_eq!(ibl.env_next_layer, 1, "layer 0 reserved for scene default");
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
        // before any job is submitted.
        let pixels = vec![0.0f32; 12];
        let err = begin_upload_environment_map(&mut resources, &device, &queue, pixels, 2, 2)
            .expect_err("invalid size should error");
        match err {
            crate::error::ViewportError::InvalidTextureData { expected, actual } => {
                assert_eq!(expected, 16);
                assert_eq!(actual, 12);
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(resources.uploads_pending(), 0);
    }

    #[test]
    fn begin_upload_completes_and_populates_ibl() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);
        assert!(resources.ibl.irradiance_view.is_none());

        let pixels = make_solid_env(8, 4, [0.5, 0.6, 0.7]);
        let id =
            begin_upload_environment_map(&mut resources, &device, &queue, pixels, 8, 4).unwrap();
        assert_eq!(resources.uploads_pending(), 1);

        // Drive the runner until the job lands. The CPU path takes around
        // 100 ms on this test image, so 100 iterations of 20 ms is plenty.
        let mut iterations = 0;
        loop {
            resources.process_uploads(&device, &queue);
            match resources.upload_status(id) {
                crate::resources::UploadStatus::Ready => break,
                crate::resources::UploadStatus::Failed(e) => panic!("upload failed: {e:?}"),
                crate::resources::UploadStatus::Pending { .. } => {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                crate::resources::UploadStatus::Unknown => {
                    panic!("job id disappeared before completion")
                }
            }
            iterations += 1;
            if iterations > 100 {
                panic!("env-map upload did not complete in time");
            }
        }

        assert!(resources.ibl.irradiance_view.is_some());
        assert!(resources.ibl.prefiltered_view.is_some());
        assert!(resources.ibl.skybox_view.is_some());
        assert!(resources.ibl.brdf_lut_view.is_some());
        assert_eq!(resources.uploads_pending(), 0);
    }

    #[test]
    fn sync_upload_blocks_until_ready() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);

        let pixels = make_solid_env(8, 4, [0.2, 0.4, 0.8]);
        upload_environment_map(&mut resources, &device, &queue, &pixels, 8, 4).unwrap();

        assert!(resources.ibl.irradiance_view.is_some());
        assert!(resources.ibl.prefiltered_view.is_some());
        assert!(resources.ibl.skybox_view.is_some());
        // BRDF LUT is scene-independent and computed on first upload.
        assert!(resources.ibl.brdf_lut_view.is_some());
        assert!(resources.all_uploads_complete());
    }

    #[test]
    fn second_upload_replaces_skybox_but_keeps_brdf() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);

        let pixels_a = make_solid_env(8, 4, [0.5, 0.5, 0.5]);
        upload_environment_map(&mut resources, &device, &queue, &pixels_a, 8, 4).unwrap();
        assert!(resources.ibl.brdf_lut_texture.is_some());

        // Second upload completes without falling over and leaves the BRDF
        // LUT present. The internal `needs_brdf` flag decides whether the
        // worker rebuilds the LUT or reuses the cached one; either way the
        // resulting state is "BRDF available".
        let pixels_b = make_solid_env(8, 4, [0.1, 0.9, 0.4]);
        upload_environment_map(&mut resources, &device, &queue, &pixels_b, 8, 4).unwrap();
        assert!(resources.ibl.brdf_lut_texture.is_some());
        assert!(resources.ibl.skybox_view.is_some());
    }

    #[test]
    fn extra_environment_takes_the_next_layer() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);

        // The default occupies layer 0.
        let default_px = make_solid_env(8, 4, [0.5, 0.5, 0.5]);
        upload_environment_map(&mut resources, &device, &queue, &default_px, 8, 4).unwrap();
        assert!(resources.ibl.irradiance_view.is_some());

        // An extra environment bakes into layer 1 and does not disturb the
        // default skybox / array views.
        let extra_px = make_solid_env(8, 4, [0.9, 0.2, 0.1]);
        let id = upload_environment(&mut resources, &device, &queue, &extra_px, 8, 4).unwrap();
        assert_eq!(id.index(), 1);
        assert_ne!(id, super::EnvironmentMapId::DEFAULT);
        assert!(resources.ibl.skybox_view.is_some());
        assert!(resources.all_uploads_complete());
    }

    #[test]
    fn environment_set_is_capacity_bounded() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);

        let px = make_solid_env(8, 4, [0.3, 0.3, 0.3]);
        upload_environment_map(&mut resources, &device, &queue, &px, 8, 4).unwrap();

        // Layers 1..CAP-1 are the extra slots; the next request past the cap errors.
        for _ in 1..super::IBL_ENV_CAPACITY {
            upload_environment(&mut resources, &device, &queue, &px, 8, 4).unwrap();
        }
        let err = upload_environment(&mut resources, &device, &queue, &px, 8, 4)
            .expect_err("past-capacity upload should error");
        assert!(matches!(
            err,
            crate::error::ViewportError::TooManyEnvironments { .. }
        ));
    }

    #[test]
    fn environment_zones_set_clear_and_cap() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);

        let default_px = make_solid_env(8, 4, [0.5, 0.5, 0.5]);
        upload_environment_map(&mut resources, &device, &queue, &default_px, 8, 4).unwrap();
        let env = upload_environment(
            &mut resources,
            &device,
            &queue,
            &make_solid_env(8, 4, [0.9, 0.1, 0.1]),
            8,
            4,
        )
        .unwrap();

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

        clear_environment_zones(&mut resources);
        assert_eq!(resources.ibl.env_zone_count, 0);
    }
}
