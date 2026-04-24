/// Procedurally generated built-in matcap textures.
///
/// Each matcap is 256×256 RGBA.  The texture is stored with the upper hemisphere
/// (normals pointing up in view space) at y = 0 (the first row in memory), so
/// that the fragment shader sampling `uv = (nx * 0.5 + 0.5, -ny * 0.5 + 0.5)`
/// produces the intuitively correct bright-at-top orientation.
///
/// Blendable matcaps use the alpha channel to blend with the base geometry color:
///   `output = matcap.rgb + matcap.a * base_color`
/// Static matcaps ignore alpha and override the color entirely.

const SIZE: usize = 256;

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Diffuse Lambertian factor N·L, clamped to [0,1].
fn diffuse(nx: f32, ny: f32, nz: f32, lx: f32, ly: f32, lz: f32) -> f32 {
    (nx * lx + ny * ly + nz * lz).max(0.0)
}

/// Blinn-Phong specular N·H^shininess, view direction is (0,0,1) in view space.
fn specular(nx: f32, ny: f32, nz: f32, lx: f32, ly: f32, lz: f32, shininess: f32) -> f32 {
    // View direction: (0, 0, 1)
    let hx = lx;
    let hy = ly;
    let hz = lz + 1.0;
    let hlen = (hx * hx + hy * hy + hz * hz).sqrt();
    if hlen < 1e-6 {
        return 0.0;
    }
    let hx = hx / hlen;
    let hy = hy / hlen;
    let hz = hz / hlen;
    let ndoth = (nx * hx + ny * hy + nz * hz).max(0.0);
    ndoth.powf(shininess)
}

/// Generate a 256×256 RGBA matcap texture from a per-pixel closure.
///
/// `f(nx, ny, nz)` receives the view-space surface normal at each pixel and
/// returns `[r, g, b, a]` in 0..=255.  Pixels outside the unit hemisphere
/// (radius > 1) receive `[0, 0, 0, 0]`.
fn generate<F: Fn(f32, f32, f32) -> [u8; 4]>(f: F) -> Vec<u8> {
    let mut data = Vec::with_capacity(SIZE * SIZE * 4);
    for y in 0..SIZE {
        for x in 0..SIZE {
            // Map pixel center to UV, then to view-space normal.
            // Convention: y=0 → ny=+1 (up-facing normals, top of image = bright).
            let u = (x as f32 + 0.5) / SIZE as f32; // [0, 1]
            let v = (y as f32 + 0.5) / SIZE as f32; // [0, 1], v=0 → top
            let nx = u * 2.0 - 1.0;
            let ny = 1.0 - v * 2.0; // flip: v=0 → ny=+1, v=1 → ny=-1
            let r2 = nx * nx + ny * ny;
            if r2 > 1.0 {
                // Outside the unit disc — no valid normal.
                data.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                let nz = (1.0 - r2).sqrt();
                data.extend_from_slice(&f(nx, ny, nz));
            }
        }
    }
    data
}

// ---------------------------------------------------------------------------
// Built-in matcap generators
// ---------------------------------------------------------------------------

/// Clay — warm orange-brown, RGBA blendable.
///
/// A single warm directional light from upper-left plus a cool fill from the right.
/// Alpha encodes how strongly the matcap tints the underlying geometry color.
pub fn clay() -> Vec<u8> {
    // Main light: warm, upper-left
    let (mlx, mly, mlz) = {
        let len = (0.4f32 * 0.4 + 0.9 * 0.9 + 0.4 * 0.4).sqrt();
        (-0.4 / len, 0.9 / len, 0.4 / len)
    };
    // Fill light: cool, lower-right, low intensity
    let (flx, fly, flz) = {
        let len = (0.6f32 * 0.6 + 0.2 * 0.2 + 0.3 * 0.3).sqrt();
        (0.6 / len, -0.2 / len, 0.3 / len)
    };

    let base = [srgb_to_linear(0.78), srgb_to_linear(0.47), srgb_to_linear(0.26)];
    let fill_color = [
        srgb_to_linear(0.45),
        srgb_to_linear(0.58),
        srgb_to_linear(0.72),
    ];

    generate(|nx, ny, nz| {
        let d = diffuse(nx, ny, nz, mlx, mly, mlz);
        let fd = diffuse(nx, ny, nz, flx, fly, flz) * 0.3;
        let s = specular(nx, ny, nz, mlx, mly, mlz, 18.0) * 0.25;
        let ambient = 0.18;

        let r = base[0] * (ambient + d * 0.75) + fill_color[0] * fd + s;
        let g = base[1] * (ambient + d * 0.75) + fill_color[1] * fd + s;
        let b = base[2] * (ambient + d * 0.75) + fill_color[2] * fd + s;
        // Alpha: blend weight — tints base geometry color at ~60 %
        let a = 0.58_f32;
        [to_u8(r), to_u8(g), to_u8(b), to_u8(a)]
    })
}

/// Wax — warm peach with a wide soft specular, RGBA blendable.
pub fn wax() -> Vec<u8> {
    let (lx, ly, lz) = {
        let len = (0.3f32 * 0.3 + 1.0 * 1.0 + 0.5 * 0.5).sqrt();
        (0.3 / len, 1.0 / len, 0.5 / len)
    };
    let base = [srgb_to_linear(0.95), srgb_to_linear(0.78), srgb_to_linear(0.65)];

    generate(|nx, ny, nz| {
        let d = diffuse(nx, ny, nz, lx, ly, lz);
        // Wide specular (subsurface scattering approximation)
        let s = specular(nx, ny, nz, lx, ly, lz, 8.0) * 0.55;
        let ambient = 0.22;
        let r = base[0] * (ambient + d * 0.68) + s * 0.95;
        let g = base[1] * (ambient + d * 0.68) + s * 0.88;
        let b = base[2] * (ambient + d * 0.68) + s * 0.78;
        let a = 0.45_f32;
        [to_u8(r), to_u8(g), to_u8(b), to_u8(a)]
    })
}

/// Candy — vivid colorful gradient, RGBA blendable.
///
/// Cycles through hue as a function of the upper hemisphere angle, giving
/// a rainbow-sphere appearance that blends with geometry color.
pub fn candy() -> Vec<u8> {
    let (lx, ly, lz) = {
        let len = (0.0f32 * 0.0 + 1.0 * 1.0 + 0.6 * 0.6).sqrt();
        (0.0 / len, 1.0 / len, 0.6 / len)
    };

    generate(|nx, ny, nz| {
        let d = diffuse(nx, ny, nz, lx, ly, lz);
        let s = specular(nx, ny, nz, lx, ly, lz, 32.0) * 0.7;
        let ambient = 0.25;

        // Hue from horizontal angle in view space
        let angle = nx.atan2(nz) / std::f32::consts::PI; // -1..+1
        let hue = (angle * 0.5 + 0.5 + ny * 0.12).fract(); // 0..1

        // HSV → RGB for vivid hue
        let h6 = hue * 6.0;
        let i = h6.floor() as u32 % 6;
        let f = h6 - h6.floor();
        let q = 1.0 - f;
        let (hr, hg, hb) = match i {
            0 => (1.0_f32, f, 0.0),
            1 => (q, 1.0, 0.0),
            2 => (0.0, 1.0, f),
            3 => (0.0, q, 1.0),
            4 => (f, 0.0, 1.0),
            _ => (1.0, 0.0, q),
        };

        let lit = (ambient + d * 0.65).min(1.0);
        let r = hr * lit + s;
        let g = hg * lit + s;
        let b = hb * lit + s;
        let a = 0.28_f32;
        [to_u8(r), to_u8(g), to_u8(b), to_u8(a)]
    })
}

/// Flat — neutral gray Lambertian, RGBA blendable.
///
/// Simple diffuse-only shading, no specular, no color contribution.  The alpha
/// channel controls how strongly the lighting gradient blends onto geometry.
pub fn flat() -> Vec<u8> {
    let (lx, ly, lz) = (0.0_f32, 1.0, 0.0);

    generate(|nx, ny, nz| {
        let d = diffuse(nx, ny, nz, lx, ly, lz);
        let ambient = 0.20;
        let gray = ambient + d * 0.78;
        let a = 0.65_f32;
        [to_u8(gray), to_u8(gray), to_u8(gray), to_u8(a)]
    })
}

/// Ceramic — clean white with sharp specular, RGB static.
pub fn ceramic() -> Vec<u8> {
    let (lx, ly, lz) = {
        let len = (-0.2f32 * -0.2 + 0.85 * 0.85 + 0.5 * 0.5).sqrt();
        (-0.2 / len, 0.85 / len, 0.5 / len)
    };
    let base = [srgb_to_linear(0.94), srgb_to_linear(0.95), srgb_to_linear(0.97)];

    generate(|nx, ny, nz| {
        let d = diffuse(nx, ny, nz, lx, ly, lz);
        let s = specular(nx, ny, nz, lx, ly, lz, 80.0) * 0.85;
        let ambient = 0.25;
        let r = base[0] * (ambient + d * 0.72) + s;
        let g = base[1] * (ambient + d * 0.72) + s;
        let b = base[2] * (ambient + d * 0.72) + s;
        [to_u8(r), to_u8(g), to_u8(b), 255]
    })
}

/// Jade — deep translucent green, RGB static.
pub fn jade() -> Vec<u8> {
    let (lx, ly, lz) = {
        let len = (0.3f32 * 0.3 + 0.9 * 0.9 + 0.4 * 0.4).sqrt();
        (0.3 / len, 0.9 / len, 0.4 / len)
    };
    let surface = [srgb_to_linear(0.15), srgb_to_linear(0.58), srgb_to_linear(0.36)];
    let deep = [srgb_to_linear(0.04), srgb_to_linear(0.28), srgb_to_linear(0.18)];
    let highlight = [srgb_to_linear(0.72), srgb_to_linear(0.90), srgb_to_linear(0.70)];

    generate(|nx, ny, nz| {
        let d = diffuse(nx, ny, nz, lx, ly, lz);
        let s = specular(nx, ny, nz, lx, ly, lz, 40.0) * 0.6;
        let rim = (1.0 - nz).powf(2.5) * 0.35; // rim light from behind
        let ambient = 0.18;

        let t = d * 0.80 + ambient;
        let r = deep[0] * (1.0 - t) + surface[0] * t + highlight[0] * s + 0.12 * rim;
        let g = deep[1] * (1.0 - t) + surface[1] * t + highlight[1] * s + 0.22 * rim;
        let b = deep[2] * (1.0 - t) + surface[2] * t + highlight[2] * s + 0.14 * rim;
        [to_u8(r), to_u8(g), to_u8(b), 255]
    })
}

/// Mud — dark brownish, low specular, rough look, RGB static.
pub fn mud() -> Vec<u8> {
    let (lx, ly, lz) = {
        let len = (0.1f32 * 0.1 + 0.8 * 0.8 + 0.3 * 0.3).sqrt();
        (0.1 / len, 0.8 / len, 0.3 / len)
    };
    let base = [srgb_to_linear(0.32), srgb_to_linear(0.20), srgb_to_linear(0.10)];
    let dark = [srgb_to_linear(0.08), srgb_to_linear(0.05), srgb_to_linear(0.02)];

    generate(|nx, ny, nz| {
        let d = diffuse(nx, ny, nz, lx, ly, lz);
        let s = specular(nx, ny, nz, lx, ly, lz, 4.0) * 0.12;
        let ambient = 0.12;

        // Add coarse directional variation to simulate rough texture
        let rough = 0.5 + 0.5 * ((nx * 7.0).sin() * (ny * 5.0).cos() * 0.5);
        let t = (ambient + d * 0.65) * rough;
        let r = dark[0] * (1.0 - t) + base[0] * t + s;
        let g = dark[1] * (1.0 - t) + base[1] * t + s;
        let b = dark[2] * (1.0 - t) + base[2] * t + s;
        [to_u8(r), to_u8(g), to_u8(b), 255]
    })
}

/// Normal — view-space normal visualization, RGB static.
///
/// R = nx mapped [−1, 1] → [0, 1], G = ny, B = nz.
pub fn normal() -> Vec<u8> {
    generate(|nx, ny, nz| {
        let r = nx * 0.5 + 0.5;
        let g = ny * 0.5 + 0.5;
        let b = nz * 0.5 + 0.5;
        [to_u8(r), to_u8(g), to_u8(b), 255]
    })
}
