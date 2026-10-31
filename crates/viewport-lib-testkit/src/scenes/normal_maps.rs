//! Normal-map decoding and the `Identical` back-face normal.
//!
//! One bump map is bound in several formats on identical spheres: RGB, two
//! channel `Rg8Unorm`, signed `Rg8Snorm`, and BC5 when the device samples it.
//! The shader reads only X and Y and rebuilds Z, so the RGB and RG8 spheres
//! must shade the same. Below them an open `Identical` sheet is seen from
//! behind with the sun on the camera's side, which lights it only when the
//! back-face normal is flipped toward the viewer.

use glam::{Mat4, Quat, Vec3};
use viewport_lib::wgpu;
use viewport_lib::{
    BackfacePolicy, Camera, Colour, CompressedFormat, LightKind, LightSource, LightingSettings,
    Material, MeshData, Projection, TextureId, primitives,
};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, item_model, textures, upload};

const MAP_SIZE: u32 = 256;
const SPHERE_RADIUS: f32 = 0.85;
/// Sphere spacing: exactly 100 pixels at the 400x300 snapshot size, so the
/// spheres land on the same sub-pixel offsets and compare pixel for pixel.
const SPACING: f32 = 2.0;
const SPHERE_Z: f32 = 1.1;

/// A full mip chain of one channel set, `channels` bytes per texel, by the same
/// 2x2 box filter (rounded) the library applies to an 8-bit upload.
fn mip_chain(
    base: Vec<u8>,
    size: u32,
    channels: usize,
    avg: impl Fn([u8; 4]) -> u8,
) -> Vec<Vec<u8>> {
    let mut levels = vec![base];
    let mut w = size as usize;
    while w > 1 {
        let nw = w / 2;
        let src = levels.last().expect("previous level");
        let mut dst = vec![0u8; nw * nw * channels];
        for y in 0..nw {
            for x in 0..nw {
                let at = |sx: usize, sy: usize, c: usize| src[(sy * w + sx) * channels + c];
                for c in 0..channels {
                    dst[(y * nw + x) * channels + c] = avg([
                        at(2 * x, 2 * y, c),
                        at(2 * x + 1, 2 * y, c),
                        at(2 * x, 2 * y + 1, c),
                        at(2 * x + 1, 2 * y + 1, c),
                    ]);
                }
            }
        }
        levels.push(dst);
        w = nw;
    }
    levels
}

fn unorm_avg(v: [u8; 4]) -> u8 {
    ((v.iter().map(|&b| b as u32).sum::<u32>() + 2) / 4) as u8
}

fn snorm_avg(v: [u8; 4]) -> u8 {
    let sum: f32 = v.iter().map(|&b| b as i8 as f32).sum();
    ((sum * 0.25).round() as i8) as u8
}

/// The red and green channels of an RGBA8 map.
fn red_green(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4).flat_map(|p| [p[0], p[1]]).collect()
}

/// A device texture holding `levels` (level 0 first), registered as a
/// material texture through its view.
fn register_view(
    ctx: &mut BuildCtx<'_>,
    format: wgpu::TextureFormat,
    bytes_per_texel: u32,
    levels: &[Vec<u8>],
) -> TextureId {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("normal_maps_view"),
        size: wgpu::Extent3d {
            width: MAP_SIZE,
            height: MAP_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, bytes) in levels.iter().enumerate() {
        let size = (MAP_SIZE >> level).max(1);
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * bytes_per_texel),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    ctx.renderer
        .resources_mut()
        .register_texture_view(ctx.device, &view, MAP_SIZE, MAP_SIZE)
        .expect("register normal map view")
}

/// One BC4 block from 16 values: the end points are the extremes, each texel
/// takes the nearest of the eight palette entries.
fn bc4_block(v: [u8; 16]) -> [u8; 8] {
    let (hi, lo) = (*v.iter().max().unwrap(), *v.iter().min().unwrap());
    let mut out = [0u8; 8];
    out[0] = hi;
    out[1] = lo;
    if hi == lo {
        return out;
    }
    // Code 0 is `hi`, 1 is `lo`, 2..=7 step from `hi` toward `lo`.
    let palette: [f32; 8] = std::array::from_fn(|code| match code {
        0 => hi as f32,
        1 => lo as f32,
        i => ((8 - i) as f32 * hi as f32 + (i - 1) as f32 * lo as f32) / 7.0,
    });
    let mut bits = 0u64;
    for (t, &x) in v.iter().enumerate() {
        let code = (0..8)
            .min_by(|&a, &b| {
                let da = (palette[a] - x as f32).abs();
                let db = (palette[b] - x as f32).abs();
                da.total_cmp(&db)
            })
            .unwrap() as u64;
        bits |= code << (3 * t);
    }
    out[2..8].copy_from_slice(&bits.to_le_bytes()[..6]);
    out
}

/// BC5 blocks for every level of a two-channel mip chain.
fn bc5_levels(rg_levels: &[Vec<u8>]) -> Vec<Vec<u8>> {
    rg_levels
        .iter()
        .enumerate()
        .map(|(level, rg)| {
            let size = (MAP_SIZE >> level).max(1) as usize;
            let blocks = size.div_ceil(4);
            let mut out = Vec::with_capacity(blocks * blocks * 16);
            for by in 0..blocks {
                for bx in 0..blocks {
                    for c in 0..2 {
                        let block: [u8; 16] = std::array::from_fn(|t| {
                            let x = (bx * 4 + t % 4).min(size - 1);
                            let y = (by * 4 + t / 4).min(size - 1);
                            rg[(y * size + x) * 2 + c]
                        });
                        out.extend_from_slice(&bc4_block(block));
                    }
                }
            }
            out
        })
        .collect()
}

/// An open sheet in the XZ plane whose front faces +Y.
fn sheet(half_width: f32, z0: f32, z1: f32) -> MeshData {
    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-half_width, 0.0, z0],
        [half_width, 0.0, z0],
        [half_width, 0.0, z1],
        [-half_width, 0.0, z1],
    ];
    mesh.normals = vec![[0.0, 1.0, 0.0]; 4];
    mesh.uvs = Some(vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
    mesh.indices = vec![0, 2, 1, 0, 3, 2];
    mesh
}

fn build_normalmap_bc5_and_backface(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let bumps = textures::normal_bumps(MAP_SIZE, 4);
    let rg = red_green(&bumps.rgba);

    let rgb_id = ctx
        .renderer
        .resources_mut()
        .upload_texture(
            ctx.device,
            ctx.queue,
            viewport_lib::TextureData::normal_map(MAP_SIZE, MAP_SIZE, bumps.rgba.clone()),
        )
        .expect("normal map upload");
    let rg_levels = mip_chain(rg.clone(), MAP_SIZE, 2, unorm_avg);
    let rg8_id = register_view(ctx, wgpu::TextureFormat::Rg8Unorm, 2, &rg_levels);
    let signed: Vec<u8> = rg
        .iter()
        .map(|&b| ((b as f32 / 255.0 * 2.0 - 1.0) * 127.0).round() as i8 as u8)
        .collect();
    let snorm_levels = mip_chain(signed, MAP_SIZE, 2, snorm_avg);
    let snorm_id = register_view(ctx, wgpu::TextureFormat::Rg8Snorm, 2, &snorm_levels);
    // BC5 needs a device feature; without it that sphere is left out.
    let bc5_id = viewport_lib::supports_texture_format(ctx.device, wgpu::TextureFormat::Bc5RgUnorm)
        .then(|| {
            ctx.renderer
                .resources_mut()
                .upload_texture(
                    ctx.device,
                    ctx.queue,
                    viewport_lib::TextureData::compressed_normal_map(
                        MAP_SIZE,
                        MAP_SIZE,
                        CompressedFormat::Bc5Rg,
                        bc5_levels(&rg_levels),
                    ),
                )
                .expect("bc5 normal map upload")
        });

    let sphere = upload(ctx, &primitives::sphere(SPHERE_RADIUS, 64, 32));
    let mut items = Vec::new();
    for (i, map) in [Some(rgb_id), Some(rg8_id), Some(snorm_id), bc5_id]
        .into_iter()
        .enumerate()
    {
        let Some(map) = map else { continue };
        let mut m = Material::pbr(Colour::linear_rgb(0.7, 0.7, 0.75), 0.0, 0.45);
        m.normal_map_id = Some(map);
        let x = (i as f32 - 1.5) * SPACING;
        let mut it = item_model(
            ctx,
            sphere,
            Mat4::from_translation(Vec3::new(x, 0.0, SPHERE_Z)),
            m,
        );
        // Shadow-map texels fall differently on each sphere; leave them
        // unshadowed so the decoded normal is the only thing that varies.
        it.settings.receive_shadows = false;
        it.settings.cast_shadows = false;
        items.push(it);
    }

    let sheet_mesh = upload(ctx, &sheet(3.6, -2.6, -0.6));
    let mut sheet_mat = Material::pbr(Colour::linear_rgb(0.75, 0.55, 0.35), 0.0, 0.6);
    sheet_mat.backface_policy = BackfacePolicy::Identical;
    items.push(item_model(ctx, sheet_mesh, Mat4::IDENTITY, sheet_mat));

    // The sun is on the camera's (-Y) side, above and to the left.
    let mut sun = LightSource::default();
    sun.kind = LightKind::Directional {
        direction: [-0.35, -1.0, 0.7],
    };
    sun.intensity = 1.2;
    let mut lighting = LightingSettings::default();
    lighting.lights = vec![sun];
    lighting.hemisphere_intensity = 0.3;

    BuiltScene {
        items,
        lighting,
        ..Default::default()
    }
}

/// Orthographic and square-on from -Y, so every sphere sees the same view
/// direction. The view is 6 units tall.
fn front_ortho() -> Camera {
    let distance = 10.0;
    Camera {
        projection: Projection::Orthographic,
        center: glam::DVec3::ZERO,
        distance,
        orientation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        fov_y: 2.0 * (3.0f32 / distance).atan(),
        // The snapshot's 400x300, so a pixel is 0.02 units both ways.
        aspect: 4.0 / 3.0,
        ..Camera::default()
    }
}

/// The normal-map decode and back-face scenes.
pub fn scenes() -> Vec<NamedScene> {
    vec![NamedScene {
        name: "normalmap_bc5_and_backface",
        cameras: vec![NamedCamera {
            name: "front",
            camera: front_ortho(),
        }],
        build: build_normalmap_bc5_and_backface,
    }]
}
