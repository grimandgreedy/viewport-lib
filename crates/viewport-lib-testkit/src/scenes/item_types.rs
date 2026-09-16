//! Catalogue scenes for the non-mesh item types.
//!
//! One scene per item type, so a golden mismatch names the type that moved.
//! Where a type supports the selection outline, one item in its scene is
//! selected so the outline pass is part of the pixels; where a type is
//! pickable, one item carries a pick id (invisible, but it keeps the scenes
//! honest as picking fixtures). Scenes with a nondeterministic pass pin the
//! settings that make a still frame repeatable (see the scatter scene).

use glam::{Mat4, Vec3};
use viewport_lib::{
    Aabb, AnchorX, AnchorY, ColourmapId, DecalBlendMode, DecalItem, GaussianSplatData,
    GaussianSplatItem, GpuImplicitItem, GpuMarchingCubesItem, ImageSliceItem, ImplicitBlendMode,
    ImplicitPrimitive, Material, MeshInstanceItem, PickId, RibbonItem, ScatterQuality,
    ScatterSettings, ScatterVolume, ScatterVolumeItem, ScreenImageItem, ShDegree, SliceAxis,
    SpriteItem, SpriteSizeMode, StreamtubeItem, TensorGlyphItem, TextureData, TubeItem, VolumeData,
    VolumeItem, VolumeSurfaceSliceItem, primitives,
};

use super::{BuildCtx, BuiltScene, NamedScene, rigs, standard_cameras};

/// The item-type scenes appended to the main catalogue.
pub fn scenes() -> Vec<NamedScene> {
    vec![
        NamedScene {
            name: "tensor_glyphs",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_tensor_glyphs,
        },
        NamedScene {
            name: "tubes",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_tubes,
        },
        NamedScene {
            name: "streamtubes",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_streamtubes,
        },
        NamedScene {
            name: "ribbons",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_ribbons,
        },
        NamedScene {
            name: "sprites",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_sprites,
        },
        NamedScene {
            name: "volume",
            cameras: standard_cameras(Vec3::ZERO, 5.0),
            build: build_volume,
        },
        NamedScene {
            name: "gaussian_splats",
            cameras: standard_cameras(Vec3::ZERO, 5.0),
            build: build_gaussian_splats,
        },
        NamedScene {
            name: "image_slice",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_image_slice,
        },
        NamedScene {
            name: "volume_surface_slice",
            cameras: standard_cameras(Vec3::ZERO, 5.0),
            build: build_volume_surface_slice,
        },
        NamedScene {
            name: "screen_image",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_screen_image,
        },
        NamedScene {
            name: "gpu_implicit",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_gpu_implicit,
        },
        NamedScene {
            name: "gpu_marching_cubes",
            cameras: standard_cameras(Vec3::ZERO, 10.0),
            build: build_gpu_marching_cubes,
        },
        NamedScene {
            name: "scatter_volume",
            cameras: standard_cameras(Vec3::new(0.0, 0.0, 1.0), 10.0),
            build: build_scatter_volume,
        },
        NamedScene {
            name: "decals",
            cameras: standard_cameras(Vec3::ZERO, 8.0),
            build: build_decals,
        },
        NamedScene {
            name: "mesh_instances",
            cameras: standard_cameras(Vec3::ZERO, 7.0),
            build: build_mesh_instances,
        },
    ]
}

// --- shared field builders ---------------------------------------------------

/// A smooth radial scalar field on a cube grid: positive inside a centred
/// blob, negative outside, so isovalue 0 and mid-range slices both show
/// structure. Index order is x-fastest, matching `upload_volume`.
fn radial_field(n: usize) -> (Vec<f32>, [u32; 3]) {
    let mut data = vec![0.0f32; n * n * n];
    let f = |i: usize| (i as f32 / (n - 1) as f32) * 2.0 - 1.0;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (fx, fy, fz) = (f(x), f(y), f(z));
                // Two lobes so slices are visibly asymmetric.
                let a = 0.65 - ((fx + 0.3).powi(2) + fy * fy + fz * fz).sqrt();
                let b = 0.45 - ((fx - 0.45).powi(2) + (fy - 0.2).powi(2) + fz * fz).sqrt();
                data[x + n * (y + n * z)] = a.max(b);
            }
        }
    }
    (data, [n as u32; 3])
}

/// A helix polyline used by the tube and ribbon scenes.
fn helix(n: usize, radius: f32, height: f32, turns: f32) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            let theta = t * std::f32::consts::TAU * turns;
            [
                radius * theta.cos(),
                radius * theta.sin(),
                (t - 0.5) * height,
            ]
        })
        .collect()
}

/// An 8x8 two-tone checker texture, uploaded as sRGB.
fn checker_texture(ctx: &mut BuildCtx<'_>, a: [u8; 3], b: [u8; 3]) -> viewport_lib::TextureId {
    let n = 8u32;
    let pixels: Vec<u8> = (0..n * n)
        .flat_map(|i| {
            let (x, y) = (i % n, i / n);
            let c = if (x + y) % 2 == 0 { a } else { b };
            [c[0], c[1], c[2], 255]
        })
        .collect();
    ctx.res
        .upload_texture(ctx.device, ctx.queue, TextureData::srgb(n, n, pixels))
        .expect("checker texture upload")
}

// --- scenes ------------------------------------------------------------------

fn build_tensor_glyphs(_ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A 4x4 grid of ellipsoids sweeping from needle-like to plate-like.
    let mut tg = TensorGlyphItem::default();
    for i in 0..4 {
        for j in 0..4 {
            let x = (i as f32 - 1.5) * 1.1;
            let y = (j as f32 - 1.5) * 1.1;
            tg.positions.push([x, y, 0.0]);
            let linear = 0.15 + i as f32 * 0.14;
            let planar = 0.15 + j as f32 * 0.14;
            tg.eigenvalues.push([0.55, planar, linear.min(planar)]);
            tg.eigenvectors
                .push([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        }
    }
    tg.settings.pick_id = PickId(1601);

    // A second, selected item so the outline pass has coverage.
    let mut sel = TensorGlyphItem::default();
    sel.positions.push([0.0, 0.0, 1.4]);
    sel.eigenvalues.push([0.5, 0.3, 0.2]);
    sel.eigenvectors
        .push([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    sel.settings.selected = true;

    BuiltScene {
        tensor_glyphs: vec![tg, sel],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_tubes(_ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A helix tube coloured along its length by scalars, plus a straight
    // selected tube beside it.
    let n = 96;
    let mut tube = TubeItem::default();
    tube.positions = helix(n, 1.0, 2.4, 3.0);
    tube.strip_lengths = vec![n as u32];
    tube.radius = 0.09;
    tube.scalars = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
    tube.scalar_range = Some((0.0, 1.0));
    tube.colourmap_id = Some(ColourmapId(0));
    tube.settings.pick_id = PickId(1602);

    let mut sel = TubeItem::default();
    sel.positions = vec![[1.8, 0.0, -1.2], [1.8, 0.0, 1.2]];
    sel.strip_lengths = vec![2];
    sel.radius = 0.12;
    sel.colour = [0.85, 0.35, 0.2, 1.0].into();
    sel.settings.selected = true;

    BuiltScene {
        tube_items: vec![tube, sel],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_streamtubes(_ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // Three arcs fanned around the origin, flat-coloured; one selected.
    let arc = |phase: f32, colour: [f32; 4], selected: bool| {
        let n = 48;
        let mut st = StreamtubeItem::default();
        st.positions = (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                let theta = phase + t * std::f32::consts::PI;
                [1.2 * theta.cos(), 1.2 * theta.sin(), (t - 0.5) * 1.2]
            })
            .collect();
        st.strip_lengths = vec![n as u32];
        st.radius = 0.08;
        st.colour = colour.into();
        st.settings.selected = selected;
        st
    };
    BuiltScene {
        streamtube_items: vec![
            arc(0.0, [0.1, 0.55, 0.5, 1.0], false),
            arc(2.1, [0.7, 0.45, 0.1, 1.0], false),
            arc(4.2, [0.4, 0.25, 0.65, 1.0], true),
        ],
        lighting: rigs::three_point(),
        ..Default::default()
    }
}

fn build_ribbons(_ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A twisted helical ribbon, plus a short selected ribbon.
    let n = 96;
    let mut rb = RibbonItem::default();
    rb.positions = helix(n, 1.1, 2.0, 2.0);
    rb.strip_lengths = vec![n as u32];
    rb.width = 0.3;
    rb.twist_attribute = Some(
        (0..n)
            .map(|i| {
                let theta = i as f32 / (n - 1) as f32 * std::f32::consts::TAU * 2.0;
                [theta.cos() * 0.2, theta.sin() * 0.2, 0.0]
            })
            .collect(),
    );
    rb.colour = [0.65, 0.3, 0.1, 1.0].into();
    rb.settings.pick_id = PickId(1604);

    let mut sel = RibbonItem::default();
    sel.positions = vec![[-0.4, 0.0, -1.5], [0.4, 0.0, -1.5]];
    sel.strip_lengths = vec![2];
    sel.width = 0.25;
    sel.colour = [0.2, 0.5, 0.75, 1.0].into();
    sel.settings.selected = true;

    BuiltScene {
        ribbon_items: vec![rb, sel],
        lighting: rigs::grazing(),
        ..Default::default()
    }
}

fn build_sprites(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // World-space textured billboards on a ring, plus one selected solid quad.
    let tex = checker_texture(ctx, [255, 200, 80], [60, 40, 160]);
    let mut ring = SpriteItem::default();
    ring.texture_id = Some(tex);
    ring.positions = (0..8)
        .map(|i| {
            let theta = i as f32 / 8.0 * std::f32::consts::TAU;
            [1.4 * theta.cos(), 1.4 * theta.sin(), 0.0]
        })
        .collect();
    ring.sizes = (0..8).map(|i| 0.4 + 0.05 * i as f32).collect();
    ring.default_colour = [1.0, 1.0, 1.0, 1.0].into();
    ring.size_mode = SpriteSizeMode::WorldSpace;
    ring.depth_write = true;
    ring.settings.pick_id = PickId(1605);

    let mut sel = SpriteItem::default();
    sel.positions = vec![[0.0, 0.0, 1.0]];
    sel.default_colour = [0.9, 0.25, 0.2, 1.0].into();
    sel.default_size = 0.5;
    sel.size_mode = SpriteSizeMode::WorldSpace;
    sel.depth_write = true;
    sel.settings.selected = true;

    BuiltScene {
        sprite_items: vec![ring, sel],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_volume(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let (data, dims) = radial_field(24);
    let vid = ctx.res.upload_volume(ctx.device, ctx.queue, &data, dims);
    let mut v = VolumeItem::default();
    v.volume_id = vid;
    v.colour_lut = Some(ColourmapId(0));
    v.scalar_range = (-0.4, 0.65);
    v.threshold_min = -0.4;
    v.threshold_max = 0.65;
    v.bbox_min = [-1.2, -1.2, -1.2];
    v.bbox_max = [1.2, 1.2, 1.2];
    v.enable_shading = true;
    v.settings.pick_id = PickId(1606);
    v.settings.selected = true;
    BuiltScene {
        volumes: vec![v],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_gaussian_splats(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A ring of degree-zero splats, coloured around the hue wheel-ish sweep.
    // SH coefficients are sRGB-referred; the shader decodes them.
    const SH0_C: f32 = 0.282_094_79;
    let mut sd = GaussianSplatData::default();
    sd.sh_degree = ShDegree::Zero;
    let n = 12;
    for i in 0..n {
        let t = i as f32 / n as f32;
        let theta = t * std::f32::consts::TAU;
        sd.positions.push([
            1.0 * theta.cos(),
            1.0 * theta.sin(),
            0.4 * (theta * 2.0).sin(),
        ]);
        sd.scales.push([0.28, 0.2, 0.14]);
        // Tilt each splat around the ring so anisotropy is visible.
        let half = theta * 0.5;
        sd.rotations.push([0.0, 0.0, half.sin(), half.cos()]);
        sd.opacities.push(0.85);
        let (r, g, b) = (0.2 + 0.7 * t, 0.25, 0.85 - 0.6 * t);
        sd.sh_coefficients.extend_from_slice(&[
            (r - 0.5) / SH0_C,
            (g - 0.5) / SH0_C,
            (b - 0.5) / SH0_C,
        ]);
    }
    let sid = ctx
        .res
        .upload_gaussian_splat(ctx.device, ctx.queue, &sd)
        .expect("splat upload");
    let mut item = GaussianSplatItem::default();
    item.source = sid;
    item.settings.pick_id = PickId(1607);
    item.settings.selected = true;
    BuiltScene {
        gaussian_splats: vec![item],
        lighting: rigs::from_above(),
        background: Some([0.02, 0.02, 0.03, 1.0]),
        ..Default::default()
    }
}

fn build_image_slice(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let (data, dims) = radial_field(24);
    let vid = ctx.res.upload_volume(ctx.device, ctx.queue, &data, dims);
    let slice = |axis: SliceAxis, offset: f32, selected: bool| {
        let mut s = ImageSliceItem::default();
        s.volume_id = vid;
        s.axis = axis;
        s.offset = offset;
        s.bbox_min = [-1.5, -1.5, -1.5];
        s.bbox_max = [1.5, 1.5, 1.5];
        s.scalar_range = (-0.4, 0.65);
        s.colour_lut = Some(ColourmapId(0));
        s.opacity = 1.0;
        s.settings.selected = selected;
        s.settings.pick_id = PickId(1608);
        s
    };
    BuiltScene {
        image_slices: vec![
            slice(SliceAxis::Z, 0.5, false),
            slice(SliceAxis::X, 0.35, true),
        ],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_volume_surface_slice(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let (data, dims) = radial_field(24);
    let vid = ctx.res.upload_volume(ctx.device, ctx.queue, &data, dims);
    // A bowl surface dipped through the field, sampling it per fragment.
    let bowl = super::meshes::bowl(1.1, 40, 12);
    let mesh_id = ctx
        .res
        .upload_mesh_data(ctx.device, &bowl)
        .expect("bowl upload");
    let mut ss = VolumeSurfaceSliceItem::default();
    ss.volume_id = vid;
    ss.mesh_id = mesh_id;
    ss.bbox_min = [-1.2, -1.2, -1.2];
    ss.bbox_max = [1.2, 1.2, 1.2];
    ss.scalar_range = (-0.4, 0.65);
    ss.colour_lut = Some(ColourmapId(0));
    ss.opacity = 1.0;
    ss.model = Mat4::from_translation(Vec3::new(0.0, 0.0, -0.4)).to_cols_array_2d();
    ss.settings.selected = true;
    ss.settings.pick_id = PickId(1609);
    BuiltScene {
        volume_surface_slices: vec![ss],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_screen_image(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A gradient panel centred on screen over a mesh, plus a corner-anchored
    // panel, so anchoring and compositing over geometry are both in the image.
    let sphere = ctx
        .res
        .upload_mesh_data(ctx.device, &primitives::sphere(1.0, 32, 16))
        .expect("sphere upload");
    let mut item = viewport_lib::SceneRenderItem::default();
    item.mesh_id = sphere;
    item.material = Material::pbr([0.4, 0.55, 0.7], 0.2, 0.5);

    let (w, h) = (64u32, 32u32);
    let gradient: Vec<[u8; 4]> = (0..w * h)
        .map(|i| {
            let t = (i % w) as f32 / (w - 1) as f32;
            [(255.0 * t) as u8, 60, (255.0 * (1.0 - t)) as u8, 220]
        })
        .collect();
    let mut centre = ScreenImageItem::default();
    centre.pixels = gradient.clone();
    centre.width = w;
    centre.height = h;
    centre.anchor_x = AnchorX::Middle;
    centre.anchor_y = AnchorY::Middle;
    centre.scale = 2.0;
    centre.alpha = 1.0;
    centre.settings.pick_id = PickId(1610);

    let mut corner = ScreenImageItem::default();
    corner.pixels = gradient;
    corner.width = w;
    corner.height = h;
    corner.scale = 1.0;
    corner.alpha = 0.8;

    BuiltScene {
        items: vec![item],
        screen_images: vec![centre, corner],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_gpu_implicit(_ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // Two spheres and a capsule fused with a smooth union.
    let sphere = |c: [f32; 3], r: f32, colour: [f32; 4]| {
        let mut p = ImplicitPrimitive::zeroed();
        p.kind = 1;
        p.blend = 0.35;
        p.params[..3].copy_from_slice(&c);
        p.params[3] = r;
        p.colour = colour.into();
        p
    };
    let mut capsule = ImplicitPrimitive::zeroed();
    capsule.kind = 4;
    capsule.blend = 0.35;
    capsule.params[..3].copy_from_slice(&[-1.0, 0.0, -0.6]);
    capsule.params[3] = 0.3;
    capsule.params[4..7].copy_from_slice(&[1.0, 0.0, -0.6]);
    capsule.colour = [0.7, 0.55, 0.15, 1.0].into();

    let mut item = GpuImplicitItem::default();
    item.primitives = vec![
        sphere([-0.7, 0.0, 0.3], 0.7, [0.38, 0.12, 0.62, 1.0]),
        sphere([0.7, 0.2, 0.4], 0.55, [0.12, 0.5, 0.55, 1.0]),
        capsule,
    ];
    item.blend_mode = ImplicitBlendMode::SmoothUnion;
    item.settings.selected = true;
    item.settings.pick_id = PickId(1611);
    BuiltScene {
        gpu_implicit: vec![item],
        lighting: rigs::three_point(),
        ..Default::default()
    }
}

fn build_gpu_marching_cubes(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A gyroid over a 32^3 grid, extracted on the GPU at isovalue 0.
    let n: u32 = 32;
    let origin = [-3.0f32; 3];
    let step = 6.0 / (n - 1) as f32;
    let mut data = Vec::with_capacity((n * n * n) as usize);
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let x = origin[0] + ix as f32 * step;
                let y = origin[1] + iy as f32 * step;
                let z = origin[2] + iz as f32 * step;
                data.push(x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos());
            }
        }
    }
    let vol = VolumeData {
        data,
        dims: [n, n, n],
        origin,
        spacing: [step; 3],
    };
    let volume_id = ctx
        .res
        .upload_volume_for_mc(ctx.device, ctx.queue, &vol)
        .expect("mc volume upload");
    let mut material = Material::from_colour([0.45, 0.48, 0.52]);
    material.roughness = 0.4;
    let mut settings = viewport_lib::ItemSettings::default();
    settings.pick_id = PickId(1612);
    settings.selected = true;
    BuiltScene {
        gpu_mc_items: vec![GpuMarchingCubesItem {
            volume_id,
            isovalue: 0.0,
            material,
            settings,
            cpu_data: None,
        }],
        lighting: rigs::grazing(),
        ..Default::default()
    }
}

fn build_scatter_volume(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // Ground fog over a slab and a sphere, so in-scattering has geometry
    // behind it. Temporal accumulation and jitter are pinned off so a still
    // frame is deterministic; quality High hides the banding jitter masks.
    let slab = ctx
        .res
        .upload_mesh_data(ctx.device, &primitives::cuboid(12.0, 12.0, 0.4))
        .expect("slab upload");
    let mut ground = viewport_lib::SceneRenderItem::default();
    ground.mesh_id = slab;
    ground.model = Mat4::from_translation(Vec3::new(0.0, 0.0, -0.2)).to_cols_array_2d();
    ground.material = Material::pbr([0.5, 0.52, 0.5], 0.0, 0.85);

    let sphere = ctx
        .res
        .upload_mesh_data(ctx.device, &primitives::sphere(1.0, 32, 16))
        .expect("sphere upload");
    let mut ball = viewport_lib::SceneRenderItem::default();
    ball.mesh_id = sphere;
    ball.model = Mat4::from_translation(Vec3::new(0.0, 0.0, 1.0)).to_cols_array_2d();
    ball.material = Material::pbr([0.75, 0.35, 0.25], 0.1, 0.4);

    let fog = ScatterVolume::box_uniform(
        Aabb {
            min: Vec3::new(-5.0, -5.0, 0.0),
            max: Vec3::new(5.0, 5.0, 2.2),
        },
        0.28,
        [0.75, 0.8, 0.9],
    );
    let mut fog_item = ScatterVolumeItem::new(fog);
    fog_item.settings.pick_id = PickId(1613);
    fog_item.settings.selected = true;

    let mut scatter_settings = ScatterSettings::default();
    scatter_settings.temporal = false;
    scatter_settings.blue_noise_jitter = false;
    scatter_settings.downsample = false;
    scatter_settings.quality = ScatterQuality::High;

    BuiltScene {
        items: vec![ground, ball],
        scatter_volumes: vec![fog_item],
        scatter_settings: Some(scatter_settings),
        lighting: rigs::grazing(),
        ..Default::default()
    }
}

fn build_decals(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // Two decals projected down onto a slab: an sRGB checker with Replace,
    // and an overlapping darker Multiply decal, so blend order is pinned by
    // sort_key.
    let slab = ctx
        .res
        .upload_mesh_data(ctx.device, &primitives::cuboid(8.0, 8.0, 0.5))
        .expect("slab upload");
    let mut ground = viewport_lib::SceneRenderItem::default();
    ground.mesh_id = slab;
    ground.model = Mat4::from_translation(Vec3::new(0.0, 0.0, -0.25)).to_cols_array_2d();
    ground.material = Material::pbr([0.62, 0.6, 0.58], 0.0, 0.8);

    let box_mesh = ctx
        .res
        .upload_mesh_data(ctx.device, &primitives::cube(1.2))
        .expect("cube upload");
    let mut cube = viewport_lib::SceneRenderItem::default();
    cube.mesh_id = box_mesh;
    cube.model = Mat4::from_translation(Vec3::new(-1.6, 1.2, 0.6)).to_cols_array_2d();
    cube.material = Material::pbr([0.45, 0.5, 0.65], 0.2, 0.5);

    let checker = checker_texture(ctx, [220, 70, 40], [240, 220, 200]);
    let dark = checker_texture(ctx, [70, 70, 80], [110, 110, 120]);

    let mut replace = DecalItem::default();
    replace.transform = (Mat4::from_translation(Vec3::new(0.4, -0.4, 0.0))
        * Mat4::from_scale(Vec3::new(3.0, 3.0, 2.0)))
    .to_cols_array_2d();
    replace.texture_id = checker;
    replace.blend_mode = DecalBlendMode::Replace;
    replace.alpha = 1.0;
    replace.sort_key = 0;

    let mut multiply = DecalItem::default();
    multiply.transform = (Mat4::from_translation(Vec3::new(1.6, 0.8, 0.0))
        * Mat4::from_rotation_z(0.6)
        * Mat4::from_scale(Vec3::new(2.2, 2.2, 2.0)))
    .to_cols_array_2d();
    multiply.texture_id = dark;
    multiply.blend_mode = DecalBlendMode::Multiply;
    multiply.alpha = 0.9;
    multiply.sort_key = 1;

    BuiltScene {
        items: vec![ground, cube],
        decals: vec![replace, multiply],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_mesh_instances(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A spiral of tinted cube instances in one batch draw.
    let cube = ctx
        .res
        .upload_mesh_data(ctx.device, &primitives::cube(0.35))
        .expect("cube upload");
    let n = 40;
    let mut item = MeshInstanceItem::default();
    item.mesh_id = cube;
    item.transforms = (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            let theta = t * std::f32::consts::TAU * 2.0;
            let r = 0.6 + 1.6 * t;
            (Mat4::from_translation(Vec3::new(r * theta.cos(), r * theta.sin(), (t - 0.5) * 1.6))
                * Mat4::from_rotation_z(theta))
            .to_cols_array_2d()
        })
        .collect();
    item.colours = (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            viewport_lib::Colour::from([0.9 - 0.6 * t, 0.3 + 0.5 * t, 0.25, 1.0])
        })
        .collect();
    BuiltScene {
        mesh_instances: vec![item],
        lighting: rigs::three_point(),
        ..Default::default()
    }
}
