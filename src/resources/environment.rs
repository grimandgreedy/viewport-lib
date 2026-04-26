//! CPU-side IBL precomputation and environment map upload.
//!
//! Produces:
//! - **Irradiance map** (64×32 equirect) : diffuse hemisphere integral.
//! - **Prefiltered specular map** (128×64 equirect, 5 mip levels) : split-sum approximation.
//! - **BRDF integration LUT** (128×128) : Schlick-GGX split-sum second integral.
//!
//! All textures are Rgba16Float for HDR correctness.
//!
//! NOTE: The IBL shader helpers (`dir_to_equirect_uv`, `sample_ibl_irradiance`,
//! `ibl_ambient`, etc.) are duplicated in mesh.wgsl, mesh_instanced.wgsl,
//! mesh_oit.wgsl, and mesh_instanced_oit.wgsl. mesh.wgsl is the canonical copy.
//! Update all four when changing IBL shader code.

use std::f32::consts::PI;

// -------------------------------------------------------------------------
// Public upload API
// -------------------------------------------------------------------------

/// Upload an equirectangular HDR environment map and precompute IBL textures.
///
/// `pixels` is row-major RGBA f32 (4 floats per pixel), `width`×`height`.
/// After this call, the camera bind groups must be rebuilt so shaders see the
/// new textures : call `rebuild_camera_bind_groups` on the renderer.
///
/// **Performance:** This performs heavy CPU-side precomputation (irradiance
/// convolution, GGX specular prefilter, BRDF LUT). Call during asset loading
/// or on a background thread, not on the hot render path. The BRDF LUT is
/// scene-independent and could be cached across different environment maps.
pub fn upload_environment_map(
    resources: &mut super::ViewportGpuResources,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
) {
    assert_eq!(pixels.len(), (width * height * 4) as usize);

    // 1. Upload full-res skybox texture.
    let skybox_tex = upload_rgba16f(device, queue, pixels, width, height, "ibl_skybox");
    let skybox_view = skybox_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // 2. Generate irradiance map (diffuse).
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

    // 3. Generate prefiltered specular map (5 roughness levels).
    let spec_w = 128u32;
    let spec_h = 64u32;
    let mip_levels = 5u32;
    let (spec_data_mips, spec_tex) = prefilter_specular(
        device, queue, pixels, width, height, spec_w, spec_h, mip_levels,
    );
    let _ = spec_data_mips; // CPU data no longer needed
    let spec_view = spec_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // 4. Generate BRDF integration LUT.
    let brdf_size = 128u32;
    let brdf_data = generate_brdf_lut(brdf_size);
    let brdf_tex = upload_rgba16f(
        device,
        queue,
        &brdf_data,
        brdf_size,
        brdf_size,
        "ibl_brdf_lut",
    );
    let brdf_view = brdf_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // 5. Store on resources.
    resources.ibl_irradiance_view = Some(irr_view);
    resources.ibl_prefiltered_view = Some(spec_view);
    resources.ibl_brdf_lut_view = Some(brdf_view);
    resources.ibl_skybox_view = Some(skybox_view);
    resources.ibl_irradiance_texture = Some(irr_tex);
    resources.ibl_prefiltered_texture = Some(spec_tex);
    resources.ibl_brdf_lut_texture = Some(brdf_tex);
    resources.ibl_skybox_texture = Some(skybox_tex);
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

/// Upload f32 RGBA pixel data as an Rgba16Float GPU texture.
fn upload_rgba16f(
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
            bytes_per_row: Some(width * 8), // 4 × f16 = 8 bytes per pixel
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

/// Sample an equirectangular HDR image at a world-space direction.
fn sample_equirect(pixels: &[f32], width: u32, height: u32, dir: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = dir;
    // Longitude: atan2(z, x), latitude: asin(y)
    let phi = z.atan2(x); // -PI..PI
    let theta = y.clamp(-1.0, 1.0).asin(); // -PI/2..PI/2
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
    let sample_delta = 0.05f32; // ~40 phi steps × ~20 theta steps = 800 samples
    let mut out = vec![0.0f32; (dst_w * dst_h * 4) as usize];

    for y in 0..dst_h {
        let v = y as f32 / dst_h as f32;
        let theta_n = PI * (0.5 - v); // latitude
        for x in 0..dst_w {
            let u = x as f32 / dst_w as f32;
            let phi_n = 2.0 * PI * (u - 0.5); // longitude

            // Normal direction for this texel.
            let (st, ct) = theta_n.sin_cos();
            let (sp, cp) = phi_n.sin_cos();
            let normal = [ct * cp, st, ct * sp];

            // Build tangent frame.
            let up = if normal[1].abs() < 0.999 {
                [0.0, 1.0, 0.0]
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
                    // Tangent-space direction.
                    let ts = [sst * scp, sst * ssp, sct];
                    // World-space direction.
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
            let idx = (y * dst_w + x) as usize * 4;
            out[idx] = irr[0] * scale;
            out[idx + 1] = irr[1] * scale;
            out[idx + 2] = irr[2] * scale;
            out[idx + 3] = 1.0;
        }
    }
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

        for y in 0..mip_h {
            let v = y as f32 / mip_h as f32;
            let theta_n = PI * (0.5 - v);
            for x in 0..mip_w {
                let u = x as f32 / mip_w as f32;
                let phi_n = 2.0 * PI * (u - 0.5);
                let (st, ct) = theta_n.sin_cos();
                let (sp, cp) = phi_n.sin_cos();
                let n = [ct * cp, st, ct * sp];
                let r = n; // reflect = normal for prefilter
                let v_dir = r;

                let color =
                    prefilter_sample(src, src_w, src_h, n, r, v_dir, roughness, num_samples);
                let idx = (y * mip_w + x) as usize * 4;
                data[idx] = color[0];
                data[idx + 1] = color[1];
                data[idx + 2] = color[2];
                data[idx + 3] = 1.0;
            }
        }

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
    let mut color = [0.0f32; 3];
    let mut total_weight = 0.0f32;
    let a = roughness * roughness;

    for i in 0..num_samples {
        let xi = hammersley(i, num_samples);
        let h = importance_sample_ggx(xi, n, a);
        let l = reflect(v, h);
        let n_dot_l = dot(n, l).max(0.0);

        if n_dot_l > 0.0 {
            let c = sample_equirect(src, src_w, src_h, l);
            color[0] += c[0] * n_dot_l;
            color[1] += c[1] * n_dot_l;
            color[2] += c[2] * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    if total_weight > 0.0 {
        color[0] /= total_weight;
        color[1] /= total_weight;
        color[2] /= total_weight;
    }
    color
}

// -------------------------------------------------------------------------
// BRDF integration LUT (split-sum second integral)
// -------------------------------------------------------------------------

fn generate_brdf_lut(size: u32) -> Vec<f32> {
    let num_samples = 1024u32;
    let mut data = vec![0.0f32; (size * size * 4) as usize];

    for y in 0..size {
        let roughness = (y as f32 + 0.5) / size as f32;
        let roughness = roughness.max(0.01);
        for x in 0..size {
            let n_dot_v = (x as f32 + 0.5) / size as f32;
            let n_dot_v = n_dot_v.max(0.001);

            let (a, b) = integrate_brdf(n_dot_v, roughness, num_samples);
            let idx = (y * size + x) as usize * 4;
            data[idx] = a;
            data[idx + 1] = b;
            data[idx + 2] = 0.0;
            data[idx + 3] = 1.0;
        }
    }
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

/// Convert f32 to IEEE 754 half-precision (f16) bits with round-to-nearest.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;

    // NaN -> f16 NaN (preserve sign, quiet NaN).
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return (sign | 0x7E00) as u16;
    }

    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa = bits & 0x7F_FFFF;

    if exp > 15 {
        // Overflow -> infinity.
        (sign | 0x7C00) as u16
    } else if exp < -14 {
        // Underflow -> flush to zero (denormals ignored for simplicity).
        sign as u16
    } else {
        let rounded = mantissa + 0x0000_1000; // round to nearest
        if rounded & 0x0080_0000 != 0 {
            // Mantissa overflow : carry into exponent.
            let new_exp = (exp + 16) as u32;
            if new_exp > 30 {
                return (sign | 0x7C00) as u16; // overflow to infinity
            }
            (sign | (new_exp << 10)) as u16
        } else {
            let f16_exp = ((exp + 15) as u32) << 10;
            let f16_mantissa = (rounded >> 13) & 0x3FF;
            (sign | f16_exp | f16_mantissa) as u16
        }
    }
}
