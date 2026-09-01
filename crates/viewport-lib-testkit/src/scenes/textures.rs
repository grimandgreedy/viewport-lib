//! Texture corpus for the catalogue.
//!
//! Each function returns CPU-side RGBA8 pixels plus dimensions; scenes upload
//! them through `ViewportGpuResources::upload_texture` (sRGB albedo) or
//! `upload_normal_map` (linear). The set covers a high-frequency pattern
//! (checker), a smooth gradient, value noise, and a tangent-space normal map,
//! none of which the existing examples exercise.

/// CPU-side RGBA8 texture data ready to upload.
pub struct TextureData {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major RGBA8 pixels, `width * height * 4` bytes.
    pub rgba: Vec<u8>,
}

/// A two-colour checkerboard with `cells` squares per side. High-frequency
/// content for mip and sampling tests.
pub fn checker(size: u32, cells: u32, a: [u8; 3], b: [u8; 3]) -> TextureData {
    let size = size.max(2);
    let cells = cells.max(1);
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    let cell = size / cells;
    for y in 0..size {
        for x in 0..size {
            let on = ((x / cell.max(1)) + (y / cell.max(1))) % 2 == 0;
            let c = if on { a } else { b };
            rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    TextureData {
        width: size,
        height: size,
        rgba,
    }
}

/// A smooth diagonal RGB gradient.
pub fn gradient(size: u32) -> TextureData {
    let size = size.max(2);
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / (size - 1) as f32;
            let v = y as f32 / (size - 1) as f32;
            rgba.extend_from_slice(&[
                (u * 255.0) as u8,
                (v * 255.0) as u8,
                ((1.0 - u) * 255.0) as u8,
                255,
            ]);
        }
    }
    TextureData {
        width: size,
        height: size,
        rgba,
    }
}

/// Deterministic greyscale value noise (hash-based, no RNG state) so snapshots
/// are stable across runs.
pub fn value_noise(size: u32) -> TextureData {
    let size = size.max(2);
    let hash = |x: u32, y: u32| -> u8 {
        let mut h = x
            .wrapping_mul(374761393)
            .wrapping_add(y.wrapping_mul(668265263));
        h = (h ^ (h >> 13)).wrapping_mul(1274126177);
        (h ^ (h >> 16)) as u8
    };
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let v = hash(x, y);
            rgba.extend_from_slice(&[v, v, v, 255]);
        }
    }
    TextureData {
        width: size,
        height: size,
        rgba,
    }
}

/// A tangent-space normal map of rounded bumps. Encoded linearly: upload with
/// `upload_normal_map`, not `upload_texture`.
pub fn normal_bumps(size: u32, bumps: u32) -> TextureData {
    let size = size.max(2);
    let bumps = bumps.max(1) as f32;
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32 * std::f32::consts::TAU * bumps;
            let v = y as f32 / size as f32 * std::f32::consts::TAU * bumps;
            // Height field h = sin(u)*sin(v); normal from its gradient.
            let dhx = u.cos() * v.sin();
            let dhy = u.sin() * v.cos();
            let strength = 0.5;
            let nx = -dhx * strength;
            let ny = -dhy * strength;
            let nz = 1.0;
            let inv = 1.0 / (nx * nx + ny * ny + nz * nz).sqrt();
            let enc = |c: f32| (((c * inv) * 0.5 + 0.5) * 255.0) as u8;
            rgba.extend_from_slice(&[enc(nx), enc(ny), enc(nz), 255]);
        }
    }
    TextureData {
        width: size,
        height: size,
        rgba,
    }
}
