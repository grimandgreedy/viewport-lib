//! CPU-side IBL precomputation and environment map upload.
//!
//! Produces:
//! - **Irradiance map** (64x32 equirect) : diffuse hemisphere integral.
//! - **Prefiltered specular map** (128x64 equirect, 5 mip levels) : split-sum approximation.
//! - **BRDF integration LUT** (128x128) : Schlick-GGX split-sum second integral.
//!
//! All textures are Rgba16Float for HDR correctness.
//!
//! NOTE: The IBL shader helpers (`dir_to_equirect_uv`, `sample_ibl_irradiance`,
//! `ibl_ambient`, etc.) are duplicated in mesh.wgsl, mesh_instanced.wgsl,
//! mesh_oit.wgsl, and mesh_instanced_oit.wgsl. mesh.wgsl is the canonical copy.
//! Update all four when changing IBL shader code.

use rayon::prelude::*;
use std::f32::consts::PI;

use crate::resources::upload_jobs::{ApplyFn, JobId, JobProduct, ProgressHandle, UploadStatus};

// -------------------------------------------------------------------------
// Public upload API
// -------------------------------------------------------------------------

/// Upload an equirectangular HDR environment map and precompute IBL textures.
///
/// `pixels` is row-major RGBA f32 (4 floats per pixel), `width`x`height`.
/// After this call, the camera bind groups must be rebuilt so shaders see
/// the new textures: call `rebuild_camera_bind_groups` on the renderer.
///
/// This entry point blocks the calling thread until the upload finishes.
/// `begin_upload_environment_map` returns immediately and reports completion
/// through the upload-job runner.
pub fn upload_environment_map(
    resources: &mut crate::resources::ViewportGpuResources,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
) -> crate::error::ViewportResult<()> {
    let id =
        begin_upload_environment_map(resources, device, queue, pixels.to_vec(), width, height)?;
    loop {
        resources.process_uploads(device, queue);
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

/// Start an asynchronous environment-map upload.
///
/// Returns the `JobId` of the submitted upload. The caller is expected to
/// drive `process_uploads` from the renderer's prepare path each frame; once
/// the returned id reports `Ready`, the IBL textures are live and the
/// caller's next call to `rebuild_camera_bind_groups` will pick them up.
///
/// Ownership of `pixels` transfers into the background worker.
pub fn begin_upload_environment_map(
    resources: &mut crate::resources::ViewportGpuResources,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: Vec<f32>,
    width: u32,
    height: u32,
) -> crate::error::ViewportResult<JobId> {
    let expected = (width as usize) * (height as usize) * 4;
    if pixels.len() != expected {
        return Err(crate::error::ViewportError::InvalidTextureData {
            expected,
            actual: pixels.len(),
        });
    }

    let compute_supported = super::ibl_compute::compute_supported(device);
    let needs_brdf = resources.ibl_brdf_lut_texture.is_none();

    let mut runner = resources.jobs.lock().expect("upload job runner poisoned");
    let id = if compute_supported {
        runner.submit_with_gpu(device, queue, move |dev, q, progress| {
            progress.set(0.1);
            let result =
                super::ibl_compute::compute_ibl(dev, q, &pixels, width, height, needs_brdf);
            progress.set(1.0);
            Ok(JobProduct::with_gpu_and_apply(
                result.submission.clone(),
                apply_gpu_result(result),
            ))
        })
    } else {
        runner.submit_with_gpu(device, queue, move |dev, q, progress| {
            run_cpu_path(dev, q, &pixels, width, height, needs_brdf, progress)
        })
    };
    Ok(id)
}

fn apply_gpu_result(result: super::ibl_compute::IblComputeResult) -> ApplyFn {
    Box::new(
        move |resources: &mut crate::resources::ViewportGpuResources| {
            resources.ibl_irradiance_view = Some(result.irradiance_view);
            resources.ibl_prefiltered_view = Some(result.prefilter_view);
            resources.ibl_skybox_view = Some(result.skybox_view);
            resources.ibl_irradiance_texture = Some(result.irradiance_texture);
            resources.ibl_prefiltered_texture = Some(result.prefilter_texture);
            resources.ibl_skybox_texture = Some(result.skybox_texture);
            if let (Some(brdf_tex), Some(brdf_view)) = (result.brdf_texture, result.brdf_view) {
                resources.ibl_brdf_lut_view = Some(brdf_view);
                resources.ibl_brdf_lut_texture = Some(brdf_tex);
            }
        },
    )
}

/// CPU IBL path executed on a worker thread.
///
/// Builds the irradiance, prefilter, and (optionally) BRDF LUT data on the
/// CPU, creates GPU textures, queues their writes, and submits a
/// flush so the runner has a `SubmissionIndex` to gate on.
fn run_cpu_path(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
    needs_brdf: bool,
    progress: &ProgressHandle,
) -> crate::error::ViewportResult<JobProduct> {
    progress.set(0.05);

    // 1. Full-resolution skybox.
    let skybox_tex = upload_rgba16f(device, queue, pixels, width, height, "ibl_skybox");
    let skybox_view = skybox_tex.create_view(&wgpu::TextureViewDescriptor::default());

    progress.set(0.15);

    // 2. Irradiance map.
    let irr_w = 64u32;
    let irr_h = 32u32;
    let irradiance_data = convolve_irradiance(pixels, width, height, irr_w, irr_h);
    let irr_tex = upload_rgba16f(
        device,
        queue,
        &irradiance_data,
        irr_w,
        irr_h,
        "ibl_irradiance",
    );
    let irr_view = irr_tex.create_view(&wgpu::TextureViewDescriptor::default());

    progress.set(0.55);

    // 3. Prefiltered specular map.
    let spec_w = 128u32;
    let spec_h = 64u32;
    let mip_levels = 5u32;
    let (_spec_data_mips, spec_tex) = prefilter_specular(
        device, queue, pixels, width, height, spec_w, spec_h, mip_levels,
    );
    let spec_view = spec_tex.create_view(&wgpu::TextureViewDescriptor::default());

    progress.set(0.9);

    // 4. BRDF integration LUT, only when no cached LUT exists. The LUT is
    // scene-independent so it is generated once and reused across env maps.
    let (brdf_tex, brdf_view) = if needs_brdf {
        let brdf_size = 128u32;
        let brdf_data = generate_brdf_lut(brdf_size);
        let tex = upload_rgba16f(
            device,
            queue,
            &brdf_data,
            brdf_size,
            brdf_size,
            "ibl_brdf_lut",
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (Some(tex), Some(view))
    } else {
        (None, None)
    };

    // 5. Flush so the runner has a submission to gate on. Implicit writes
    // queued above are folded into this submit by wgpu.
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ibl_flush"),
    });
    let submission = queue.submit(std::iter::once(encoder.finish()));

    progress.set(1.0);

    Ok(JobProduct::with_gpu_and_apply(
        submission,
        Box::new(
            move |resources: &mut crate::resources::ViewportGpuResources| {
                resources.ibl_irradiance_view = Some(irr_view);
                resources.ibl_prefiltered_view = Some(spec_view);
                resources.ibl_skybox_view = Some(skybox_view);
                resources.ibl_irradiance_texture = Some(irr_tex);
                resources.ibl_prefiltered_texture = Some(spec_tex);
                resources.ibl_skybox_texture = Some(skybox_tex);
                if let (Some(tex), Some(view)) = (brdf_tex, brdf_view) {
                    resources.ibl_brdf_lut_view = Some(view);
                    resources.ibl_brdf_lut_texture = Some(tex);
                }
            },
        ),
    ))
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

/// Upload f32 RGBA pixel data as an Rgba16Float GPU texture.
pub(crate) fn upload_rgba16f(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
    label: &str,
) -> wgpu::Texture {
    let mip_level_count = 1;
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Convert f32 -> f16 for upload.
    let half_data: Vec<u16> = pixels.iter().map(|&f| f32_to_f16(f)).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&half_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 8), // 4 x f16 = 8 bytes per pixel
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
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

fn prefilter_specular(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &[f32],
    src_w: u32,
    src_h: u32,
    base_w: u32,
    base_h: u32,
    mip_levels: u32,
) -> (Vec<Vec<f32>>, wgpu::Texture) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ibl_prefiltered"),
        size: wgpu::Extent3d {
            width: base_w,
            height: base_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let num_samples = 256u32;
    let mut all_mips = Vec::new();

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

        // Upload this mip level.
        let half_data: Vec<u16> = data.iter().map(|&f| f32_to_f16(f)).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&half_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(mip_w * 8),
                rows_per_image: Some(mip_h),
            },
            wgpu::Extent3d {
                width: mip_w,
                height: mip_h,
                depth_or_array_layers: 1,
            },
        );

        all_mips.push(data);
    }

    (all_mips, tex)
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
    use crate::resources::ViewportGpuResources;

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

    fn try_make_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    fn make_resources(device: &wgpu::Device) -> ViewportGpuResources {
        ViewportGpuResources::new(device, wgpu::TextureFormat::Rgba8UnormSrgb, 1)
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
        assert!(resources.ibl_irradiance_view.is_none());

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

        assert!(resources.ibl_irradiance_view.is_some());
        assert!(resources.ibl_prefiltered_view.is_some());
        assert!(resources.ibl_skybox_view.is_some());
        assert!(resources.ibl_brdf_lut_view.is_some());
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

        assert!(resources.ibl_irradiance_view.is_some());
        assert!(resources.ibl_prefiltered_view.is_some());
        assert!(resources.ibl_skybox_view.is_some());
        // BRDF LUT is scene-independent and computed on first upload.
        assert!(resources.ibl_brdf_lut_view.is_some());
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
        assert!(resources.ibl_brdf_lut_texture.is_some());

        // Second upload completes without falling over and leaves the BRDF
        // LUT present. The internal `needs_brdf` flag decides whether the
        // worker rebuilds the LUT or reuses the cached one; either way the
        // resulting state is "BRDF available".
        let pixels_b = make_solid_env(8, 4, [0.1, 0.9, 0.4]);
        upload_environment_map(&mut resources, &device, &queue, &pixels_b, 8, 4).unwrap();
        assert!(resources.ibl_brdf_lut_texture.is_some());
        assert!(resources.ibl_skybox_view.is_some());
    }
}
