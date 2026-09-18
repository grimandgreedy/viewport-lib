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
    ScatterSettings, ScatterVolume, ScatterVolumeItem, ShDegree, SliceAxis, SpriteBlend,
    SpriteItem, SpriteSizeMode, StreamtubeItem, TensorGlyphItem, TextureData, TubeItem, VolumeData,
    VolumeItem, VolumeSurfaceSliceItem, primitives,
};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, orbit_camera, rigs, standard_cameras};

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
            name: "sprites_soft",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_sprites_soft,
        },
        NamedScene {
            name: "sprites_oit",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_sprites_oit,
        },
        NamedScene {
            name: "sprites_refraction",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_sprites_refraction,
        },
        NamedScene {
            name: "supersampled_sprites",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_supersampled_sprites,
        },
        NamedScene {
            name: "supersampled_sprite_refraction",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_supersampled_sprite_refraction,
        },
        NamedScene {
            name: "gpu_particles",
            cameras: standard_cameras(Vec3::ZERO, 11.0),
            build: build_gpu_particles,
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
            name: "scatter_layered",
            cameras: standard_cameras(Vec3::new(0.0, 0.0, 1.0), 11.0),
            build: build_scatter_layered,
        },
        NamedScene {
            name: "scatter_textured",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_scatter_textured,
        },
        NamedScene {
            name: "scatter_animated",
            cameras: standard_cameras(Vec3::ZERO, 7.0),
            build: build_scatter_animated,
        },
        NamedScene {
            name: "item_wireframes",
            cameras: standard_cameras(Vec3::ZERO, 7.0),
            build: build_item_wireframes,
        },
        NamedScene {
            name: "decals",
            cameras: standard_cameras(Vec3::ZERO, 8.0),
            build: build_decals,
        },
        NamedScene {
            name: "supersampled_decals",
            cameras: standard_cameras(Vec3::ZERO, 8.0),
            build: build_supersampled_decals,
        },
        NamedScene {
            name: "refraction_over_soft_sprite",
            cameras: standard_cameras(Vec3::ZERO, 8.0),
            build: build_refraction_over_soft_sprite,
        },
        NamedScene {
            name: "decal_on_non_mesh",
            cameras: standard_cameras(Vec3::ZERO, 7.0),
            build: build_decal_on_non_mesh,
        },
        NamedScene {
            name: "decal_on_curves",
            cameras: standard_cameras(Vec3::ZERO, 8.0),
            build: build_decal_on_curves,
        },
        NamedScene {
            name: "decal_from_below",
            // Deliberately a low camera: the first entry is what the snapshot
            // renders, and this scene exists to pin what a decal on a curved
            // receiver does when the eye drops below the projection plane.
            cameras: vec![NamedCamera {
                name: "low",
                camera: orbit_camera(Vec3::ZERO, 7.0, 0.6, 1.95),
            }],
            build: build_decal_on_non_mesh,
        },
        NamedScene {
            name: "decal_under_soft_sprite",
            cameras: standard_cameras(Vec3::ZERO, 8.0),
            build: build_decal_under_soft_sprite,
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
    ctx.renderer
        .resources_mut()
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

/// Soft particles: sprites that fade out where they meet opaque geometry.
///
/// These draw in the read-only-depth half of the sprite work, which is a
/// different place in the frame from the depth-writing sprites in `sprites`
/// and reads the scene depth buffer rather than writing it. The slab and the
/// cube are here to be intersected: without geometry to fade against, the soft
/// distance has no effect and the scene would not tell the two halves apart.
/// One batch is lit so the lit variant of the same path is in the pixels too.
fn build_sprites_soft(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let slab = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cuboid(8.0, 8.0, 0.5))
        .expect("slab upload");
    let mut ground = viewport_lib::SceneRenderItem::default();
    ground.mesh_id = slab;
    ground.model = Mat4::from_translation(Vec3::new(0.0, 0.0, -0.25)).to_cols_array_2d();
    ground.material = Material::pbr([0.62, 0.6, 0.58], 0.0, 0.8);

    let box_mesh = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cube(1.6))
        .expect("cube upload");
    let mut cube = viewport_lib::SceneRenderItem::default();
    cube.mesh_id = box_mesh;
    cube.model = Mat4::from_translation(Vec3::new(-1.3, 0.9, 0.8)).to_cols_array_2d();
    cube.material = Material::pbr([0.45, 0.5, 0.65], 0.2, 0.5);

    let tex = checker_texture(ctx, [255, 200, 80], [60, 40, 160]);

    // A low sheet of billboards straddling the slab, so each one is partly
    // faded by the surface it intersects.
    let mut sheet = SpriteItem::default();
    sheet.texture_id = Some(tex);
    sheet.positions = (0..12)
        .map(|i| {
            let theta = i as f32 / 12.0 * std::f32::consts::TAU;
            [1.8 * theta.cos(), 1.8 * theta.sin(), 0.12]
        })
        .collect();
    sheet.default_size = 1.1;
    sheet.default_colour = [1.0, 0.95, 0.8, 0.75].into();
    sheet.size_mode = SpriteSizeMode::WorldSpace;
    sheet.depth_write = false;
    sheet.soft_particle_distance = Some(0.6);
    sheet.settings.pick_id = PickId(1606);

    // The same path with lighting on, which swaps in the lit pipeline and the
    // normal bind group (falling back to the shared one, since no normal map
    // is set here).
    let mut lit = SpriteItem::default();
    lit.positions = vec![[0.0, 0.0, 0.45], [1.1, -0.6, 0.45]];
    lit.default_size = 0.9;
    lit.default_colour = [0.4, 0.8, 1.0, 0.7].into();
    lit.size_mode = SpriteSizeMode::WorldSpace;
    lit.depth_write = false;
    lit.soft_particle_distance = Some(0.5);
    lit.lit = true;

    BuiltScene {
        items: vec![ground, cube],
        sprite_items: vec![sheet, lit],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

/// Order-independent transparency: sprites that composite through the OIT pass
/// instead of the sprite passes.
///
/// A sprite reaches OIT only when it blends, does not write depth, has no soft
/// distance and no refraction, so this scene pins that combination
/// deliberately: change any one of those fields and the batch leaves this pass
/// for another. Both blend modes that qualify are present, overlapping, and one
/// batch is lit so the lit OIT pipeline is covered as well.
fn build_sprites_oit(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let box_mesh = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cube(1.4))
        .expect("cube upload");
    let mut cube = viewport_lib::SceneRenderItem::default();
    cube.mesh_id = box_mesh;
    cube.material = Material::pbr([0.5, 0.45, 0.4], 0.1, 0.6);

    let tex = checker_texture(ctx, [255, 120, 90], [40, 70, 180]);

    // Overlapping alpha-blended quads at staggered depths, so the pass has
    // something to sort.
    let mut blended = SpriteItem::default();
    blended.texture_id = Some(tex);
    blended.positions = (0..6)
        .map(|i| {
            let t = i as f32 / 6.0;
            [1.2 * (t * 6.0).cos(), 1.2 * (t * 6.0).sin(), -0.8 + t * 1.6]
        })
        .collect();
    blended.default_size = 1.3;
    blended.default_colour = [1.0, 1.0, 1.0, 0.55].into();
    blended.size_mode = SpriteSizeMode::WorldSpace;
    blended.depth_write = false;
    blended.blend = SpriteBlend::AlphaBlend;
    blended.settings.pick_id = PickId(1607);

    let mut premultiplied = SpriteItem::default();
    premultiplied.positions = vec![[-0.9, 0.5, 0.3], [0.9, -0.5, -0.3]];
    premultiplied.default_size = 1.0;
    premultiplied.default_colour = [0.3, 0.55, 0.25, 0.55].into();
    premultiplied.size_mode = SpriteSizeMode::WorldSpace;
    premultiplied.depth_write = false;
    premultiplied.blend = SpriteBlend::Premultiplied;
    premultiplied.lit = true;

    BuiltScene {
        items: vec![cube],
        sprite_items: vec![blended, premultiplied],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

/// Refractive sprites: billboards that distort the image behind them.
///
/// These skip the ordinary sprite passes entirely and draw against a copy of
/// the scene colour taken after the opaque image is complete, which is a third
/// place in the frame again. The textured slab behind them is what makes the
/// distortion legible: over a flat colour a refractive sprite and a plain one
/// look the same.
/// The soft-particle scene again with supersampling on.
///
/// The sprite passes run before the SSAA resolve, so they draw into the
/// supersampled attachments and `clip_pos` is in supersampled pixels. Anything
/// that turns `clip_pos` back into a texture coordinate has to divide by the
/// size of the target it is drawing into, not by the scene viewport size. When
/// the soft fade divided by the viewport size instead, every sprite in this
/// scene sampled off the edge of the depth texture and faded to nothing,
/// leaving the cube and ground alone in the frame.
fn build_supersampled_sprites(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut scene = build_sprites_soft(ctx);
    let mut post = viewport_lib::PostProcessSettings::default();
    post.ssaa_factor = 2;
    scene.post_process = Some(post);
    scene
}

/// The refraction scene again with supersampling on.
///
/// Refractive sprites used to be skipped outright whenever supersampling was
/// active, because the pass they ran in had no resolve of its own at the
/// supersampled size. Drawing them from the item type's encoder hook puts them
/// after the resolve instead, where the scene colour they sample is a finished
/// image at scene resolution, so the factor no longer matters to them. This
/// scene is the gate on that: before, it rendered the ground with no bubbles.
fn build_supersampled_sprite_refraction(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut scene = build_sprites_refraction(ctx);
    let mut post = viewport_lib::PostProcessSettings::default();
    post.ssaa_factor = 2;
    scene.post_process = Some(post);
    scene
}

/// A GPU particle system emitting textured billboards.
///
/// The simulation lives on the GPU and carries state from frame to frame, so
/// this scene is only repeatable because nothing in it reads the clock: the
/// emit RNG is seeded from the system's own frame counter, which starts at zero
/// on a freshly created system, and `time_step` is a fixed number on the item
/// rather than a measured delta. The harness renders a fixed number of frames,
/// so the same pixels come back every run.
///
/// The emitter is an ordinary one, well under the system's capacity, so the
/// scene covers the case a consumer actually writes rather than a corner of the
/// parameter space. Two frames' worth of spawns land in two different windows
/// of the particle buffer, so the wrap-around in the emit kernel's slot
/// selection is part of what the reference pins.
///
/// The step is deliberately coarse. At a sixtieth of a second the particles
/// would barely have left the spawn volume by the frame that gets captured,
/// and the image would pin almost nothing.
fn build_gpu_particles(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let tex = checker_texture(ctx, [255, 170, 60], [90, 40, 150]);
    let mut config = viewport_lib::GpuParticleSystemConfig::default();
    config.capacity = 2048;
    config.render = viewport_lib::ParticleRender::Sprite {
        texture_id: Some(tex),
        blend: SpriteBlend::AlphaBlend,
        size_mode: SpriteSizeMode::WorldSpace,
        depth_write: false,
        lit: false,
        lit_params: Default::default(),
        normal_texture_id: None,
    };
    let system = ctx
        .renderer
        .create_gpu_particle_system(ctx.device, ctx.queue, &config);

    let mut item = viewport_lib::GpuParticleSystemItem::new(system, 0.4);
    item.emitter.rate = 400.0;
    item.emitter.lifetime = (4.0, 6.0);
    item.emitter.size = 0.3;
    item.emitter.colour = [1.0, 1.0, 1.0, 0.9].into();
    item.emitter.spawn_shape = viewport_lib::SpawnShape::Sphere {
        center: [0.0, 0.0, -1.8],
        radius: 0.25,
    };
    item.emitter.initial_velocity = viewport_lib::VelocityDist::UniformCone {
        axis: [0.0, 0.0, 1.0],
        half_angle: 0.6,
        min_speed: 2.5,
        max_speed: 5.5,
    };
    item.settings.pick_id = PickId(1609);

    BuiltScene {
        gpu_particle_systems: vec![item],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_sprites_refraction(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let slab = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cuboid(7.0, 7.0, 0.4))
        .expect("slab upload");
    let backdrop_tex = checker_texture(ctx, [230, 80, 60], [235, 225, 205]);
    let mut ground = viewport_lib::SceneRenderItem::default();
    ground.mesh_id = slab;
    ground.model = Mat4::from_translation(Vec3::new(0.0, 0.0, -1.2)).to_cols_array_2d();
    ground.material = Material::pbr([1.0, 1.0, 1.0], 0.0, 0.85);
    ground.material.texture_id = Some(backdrop_tex);

    // The sprite's own texture is the displacement map: red and green become a
    // signed screen-space offset, and the alpha gates how much of it shows. A
    // refractive sprite with no texture displaces by nothing and samples the
    // scene straight back, so it has to be textured to be visible at all.
    let warp_tex = checker_texture(ctx, [255, 40, 40], [40, 255, 40]);

    let mut bubbles = SpriteItem::default();
    bubbles.texture_id = Some(warp_tex);
    bubbles.positions = (0..5)
        .map(|i| {
            let t = i as f32 / 5.0 * std::f32::consts::TAU;
            [1.5 * t.cos(), 1.5 * t.sin(), 0.3]
        })
        .collect();
    bubbles.default_size = 1.2;
    bubbles.default_colour = [1.0, 1.0, 1.0, 1.0].into();
    bubbles.size_mode = SpriteSizeMode::WorldSpace;
    bubbles.depth_write = false;
    // In pixels of screen-space displacement, not a 0-to-1 fraction.
    bubbles.refraction_strength = Some(30.0);
    bubbles.settings.pick_id = PickId(1608);

    BuiltScene {
        items: vec![ground],
        sprite_items: vec![bubbles],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_volume(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let (data, dims) = radial_field(24);
    let vid = ctx
        .renderer
        .resources_mut()
        .upload_volume(ctx.device, ctx.queue, &data, dims);
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
        .renderer
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
    let vid = ctx
        .renderer
        .resources_mut()
        .upload_volume(ctx.device, ctx.queue, &data, dims);
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
    let vid = ctx
        .renderer
        .resources_mut()
        .upload_volume(ctx.device, ctx.queue, &data, dims);
    // A bowl surface dipped through the field, sampling it per fragment.
    let bowl = super::meshes::bowl(1.1, 40, 12);
    let mesh_id = ctx
        .renderer
        .resources_mut()
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
        .renderer
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
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cuboid(12.0, 12.0, 0.4))
        .expect("slab upload");
    let mut ground = viewport_lib::SceneRenderItem::default();
    ground.mesh_id = slab;
    ground.model = Mat4::from_translation(Vec3::new(0.0, 0.0, -0.2)).to_cols_array_2d();
    ground.material = Material::pbr([0.5, 0.52, 0.5], 0.0, 0.85);

    let sphere = ctx
        .renderer
        .resources_mut()
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

fn build_scatter_layered(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A wide fog box that fully contains a small dense sphere, which is the
    // arrangement the per-volume draw order has to get right: the two
    // centroids sit close together, so a centroid-distance sort flips as the
    // camera orbits while a far-corner sort keeps the container behind.
    // Downsampled, so the half-resolution target and the composite upscale
    // are in the picture too.
    let slab = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cuboid(14.0, 14.0, 0.4))
        .expect("slab upload");
    let mut ground = viewport_lib::SceneRenderItem::default();
    ground.mesh_id = slab;
    ground.model = Mat4::from_translation(Vec3::new(0.0, 0.0, -0.2)).to_cols_array_2d();
    ground.material = Material::pbr([0.42, 0.44, 0.46], 0.0, 0.85);

    let post = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cuboid(0.5, 0.5, 3.2))
        .expect("post upload");
    let mut pillar = viewport_lib::SceneRenderItem::default();
    pillar.mesh_id = post;
    pillar.model = Mat4::from_translation(Vec3::new(2.4, -1.4, 1.6)).to_cols_array_2d();
    pillar.material = Material::pbr([0.8, 0.45, 0.2], 0.1, 0.5);

    let fog = ScatterVolume::box_uniform(
        Aabb {
            min: Vec3::new(-6.0, -6.0, 0.0),
            max: Vec3::new(6.0, 6.0, 3.0),
        },
        0.16,
        [0.72, 0.78, 0.9],
    );

    // Forward-scattering core inside the fog, bright enough that compositing
    // it in the wrong order is obvious rather than subtle.
    let mut core = ScatterVolume::sphere_uniform([-0.8, 0.6, 1.3], 1.5, 0.9, [1.0, 0.72, 0.4]);
    core.anisotropy = 0.6;
    core.density_remap = viewport_lib::DensityRemap::Smoothstep { lo: 0.0, hi: 0.8 };
    core.emission = viewport_lib::Emission::Strength {
        strength: 0.8,
        curve: viewport_lib::EmissionCurve::Power(2.0),
    };

    let mut fog_item = ScatterVolumeItem::new(fog);
    fog_item.settings.pick_id = PickId(1620);
    let mut core_item = ScatterVolumeItem::new(core);
    core_item.settings.pick_id = PickId(1621);

    let mut scatter_settings = ScatterSettings::default();
    scatter_settings.temporal = false;
    scatter_settings.blue_noise_jitter = false;
    scatter_settings.downsample = true;
    scatter_settings.quality = ScatterQuality::High;

    BuiltScene {
        items: vec![ground, pillar],
        scatter_volumes: vec![fog_item, core_item],
        scatter_settings: Some(scatter_settings),
        lighting: rigs::grazing(),
        ..Default::default()
    }
}

fn build_scatter_textured(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // The two density sources that are not the flat constant: a 3D texture
    // read through a colourmap ramp, and static procedural noise. Both go
    // through the per-volume texture bind groups, which are cached by id and
    // so need a scene that actually binds two different ones.
    let (data, dims) = radial_field(24);
    let vid = ctx
        .renderer
        .resources_mut()
        .upload_volume(ctx.device, ctx.queue, &data, dims);

    let backdrop = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cuboid(8.0, 0.3, 5.0))
        .expect("backdrop upload");
    let mut wall = viewport_lib::SceneRenderItem::default();
    wall.mesh_id = backdrop;
    wall.model = Mat4::from_translation(Vec3::new(0.0, 2.6, 0.0)).to_cols_array_2d();
    wall.material = Material::pbr([0.55, 0.55, 0.6], 0.0, 0.9);

    let mut textured = ScatterVolume::box_uniform(
        Aabb {
            min: Vec3::new(-2.6, -1.2, -1.2),
            max: Vec3::new(-0.2, 1.2, 1.2),
        },
        1.1,
        [1.0, 1.0, 1.0],
    );
    textured.density_texture = Some(vid);
    textured.colour = viewport_lib::ColourSource::Ramp(ColourmapId(0));
    textured.density_remap = viewport_lib::DensityRemap::Smoothstep { lo: 0.1, hi: 0.7 };

    // Static noise: scroll velocity and time scale are both zero, so the
    // field does not move and a still frame repeats exactly.
    let mut noisy = ScatterVolume::box_uniform(
        Aabb {
            min: Vec3::new(0.2, -1.2, -1.2),
            max: Vec3::new(2.6, 1.2, 1.2),
        },
        0.85,
        [0.55, 0.85, 1.0],
    );
    let mut noise = viewport_lib::NoiseDriver::default();
    noise.scale = 1.4;
    noise.octaves = 4;
    noise.scroll_velocity = [0.0; 3];
    noise.time_scale = 0.0;
    noise.lacunarity = 2.0;
    noisy.noise = Some(noise);
    noisy.anisotropy = -0.3;

    let mut textured_item = ScatterVolumeItem::new(textured);
    textured_item.settings.pick_id = PickId(1622);
    textured_item.settings.selected = true;
    let noisy_item = ScatterVolumeItem::new(noisy);

    let mut scatter_settings = ScatterSettings::default();
    scatter_settings.temporal = false;
    scatter_settings.blue_noise_jitter = false;
    scatter_settings.downsample = false;
    scatter_settings.quality = ScatterQuality::High;

    BuiltScene {
        items: vec![wall],
        scatter_volumes: vec![textured_item, noisy_item],
        scatter_settings: Some(scatter_settings),
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_scatter_animated(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // The two scatter paths that are functions of the animation clock: noise
    // that scrolls with time, and the refraction shimmer, whose offset is
    // driven by sines of it. Both are pinned by `ScatterSettings::time_seconds`
    // below. The clock is a consumer input rather than something the renderer
    // reads off the wall, so a fixed value here renders the same frame every
    // time, which is what lets these be held to a golden at all.
    let backdrop = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cuboid(9.0, 0.3, 6.0))
        .expect("backdrop upload");
    let mut wall = viewport_lib::SceneRenderItem::default();
    wall.mesh_id = backdrop;
    wall.model = Mat4::from_translation(Vec3::new(0.0, 2.8, 0.0)).to_cols_array_2d();
    wall.material = Material::pbr([0.75, 0.3, 0.25], 0.0, 0.85);

    // Struts in front of the wall, so the refraction has hard edges to bend.
    let strut = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cuboid(0.35, 0.35, 4.5))
        .expect("strut upload");
    let mut posts = Vec::new();
    for (i, x) in [-2.2f32, -0.7, 0.8, 2.3].iter().enumerate() {
        let mut post = viewport_lib::SceneRenderItem::default();
        post.mesh_id = strut;
        post.model = Mat4::from_translation(Vec3::new(*x, 1.9, 0.0)).to_cols_array_2d();
        let t = i as f32 / 3.0;
        post.material = Material::pbr([0.3 + 0.5 * t, 0.55, 0.85 - 0.4 * t], 0.1, 0.5);
        posts.push(post);
    }

    // Smoke drifting along +X: at a non-zero clock the sample position has
    // moved, so a scene that ignored the clock would not match this image.
    let mut smoke = ScatterVolume::box_uniform(
        Aabb {
            min: Vec3::new(-3.2, -1.0, -1.6),
            max: Vec3::new(0.2, 1.0, 1.8),
        },
        0.9,
        [0.72, 0.76, 0.85],
    );
    let mut noise = viewport_lib::NoiseDriver::default();
    noise.scale = 1.1;
    noise.octaves = 3;
    noise.scroll_velocity = [0.9, 0.0, 0.35];
    noise.time_scale = 0.4;
    smoke.noise = Some(noise);

    // Heat haze: refraction strength well above the threshold so the shimmer
    // is a visible displacement rather than a sub-pixel wobble.
    let mut heat = ScatterVolume::box_uniform(
        Aabb {
            min: Vec3::new(0.6, -1.0, -1.6),
            max: Vec3::new(3.2, 1.0, 1.8),
        },
        0.35,
        [1.0, 0.85, 0.7],
    );
    let mut refraction = viewport_lib::RefractionParams::default();
    refraction.strength = 0.035;
    refraction.density_threshold = 0.0;
    refraction.noise_scale = 1.6;
    heat.refraction = Some(refraction);

    let mut smoke_item = ScatterVolumeItem::new(smoke);
    smoke_item.settings.pick_id = PickId(1623);
    let heat_item = ScatterVolumeItem::new(heat);

    let mut scatter_settings = ScatterSettings::default();
    scatter_settings.temporal = false;
    scatter_settings.blue_noise_jitter = false;
    scatter_settings.downsample = false;
    scatter_settings.quality = ScatterQuality::High;
    // Deliberately not zero: zero is the default, so it would not distinguish
    // a clock that is read from one that is ignored.
    scatter_settings.time_seconds = 3.75;

    let mut items = vec![wall];
    items.extend(posts);
    BuiltScene {
        items,
        scatter_volumes: vec![smoke_item, heat_item],
        scatter_settings: Some(scatter_settings),
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_item_wireframes(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // The item types whose wireframe is drawn as lines standing in for
    // geometry that has none: a volume's bounding box, a ring per Gaussian
    // splat, a quad outline per billboard. Each draws only under
    // `settings.wireframe`, which no other catalogue scene sets, so without
    // this scene all three paths are unrendered by the gate.
    let (data, dims) = radial_field(20);
    let vid = ctx
        .renderer
        .resources_mut()
        .upload_volume(ctx.device, ctx.queue, &data, dims);
    let mut volume = VolumeItem::default();
    volume.volume_id = vid;
    volume.colour_lut = Some(ColourmapId(0));
    volume.scalar_range = (-0.4, 0.65);
    volume.threshold_min = -0.4;
    volume.threshold_max = 0.65;
    volume.bbox_min = [-1.0, -1.0, -1.0];
    volume.bbox_max = [1.0, 1.0, 1.0];
    volume.model = Mat4::from_translation(Vec3::new(-2.6, 0.0, 0.0)).to_cols_array_2d();
    volume.settings.wireframe = true;

    // Eight splats, well under the ring cap, with distinct scales per axis so
    // the three rings of each are visibly different ellipses.
    const SH0_C: f32 = 0.282_094_79;
    let mut sd = GaussianSplatData::default();
    sd.sh_degree = ShDegree::Zero;
    for i in 0..8 {
        let t = i as f32 / 8.0;
        let theta = t * std::f32::consts::TAU;
        sd.positions.push([
            0.9 * theta.cos(),
            0.9 * theta.sin(),
            0.3 * (theta * 2.0).sin(),
        ]);
        sd.scales.push([0.34, 0.2 + 0.12 * t, 0.15]);
        sd.rotations.push([0.0, 0.0, 0.0, 1.0]);
        sd.opacities.push(0.9);
        sd.sh_coefficients.extend_from_slice(&[
            (0.8 - 0.4 * t - 0.5) / SH0_C,
            (0.4 - 0.5) / SH0_C,
            (0.3 + 0.5 * t - 0.5) / SH0_C,
        ]);
    }
    let splat_id = ctx
        .renderer
        .upload_gaussian_splat(ctx.device, ctx.queue, &sd)
        .expect("splat upload");
    let mut splats = GaussianSplatItem::default();
    splats.source = splat_id;
    splats.model = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)).to_cols_array_2d();
    splats.settings.wireframe = true;

    // Six world-space billboards, under the outline cap, at mixed sizes so the
    // quads differ from one another.
    let mut sprites = SpriteItem::default();
    sprites.positions = (0..6)
        .map(|i| {
            let theta = i as f32 / 6.0 * std::f32::consts::TAU;
            [2.8 + 0.7 * theta.cos(), 0.7 * theta.sin(), 0.0]
        })
        .collect();
    sprites.sizes = (0..6).map(|i| 0.3 + 0.07 * i as f32).collect();
    sprites.default_colour = [0.85, 0.85, 0.9, 1.0].into();
    sprites.size_mode = SpriteSizeMode::WorldSpace;
    sprites.depth_write = true;
    sprites.settings.wireframe = true;

    BuiltScene {
        volumes: vec![volume],
        gaussian_splats: vec![splats],
        sprite_items: vec![sprites],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_decals(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // Two decals projected down onto a slab: an sRGB checker with Replace,
    // and an overlapping darker Multiply decal, so blend order is pinned by
    // sort_key.
    let slab = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cuboid(8.0, 8.0, 0.5))
        .expect("slab upload");
    let mut ground = viewport_lib::SceneRenderItem::default();
    ground.mesh_id = slab;
    ground.model = Mat4::from_translation(Vec3::new(0.0, 0.0, -0.25)).to_cols_array_2d();
    ground.material = Material::pbr([0.62, 0.6, 0.58], 0.0, 0.8);

    let box_mesh = ctx
        .renderer
        .resources_mut()
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

/// The decal scene again with supersampling on.
///
/// Under SSAA the scene is drawn into the supersampled attachments and resolved
/// down partway through the frame, and everything after the resolve depth-tests
/// against the HDR depth buffer. This scene is the gate on that buffer being
/// written: when the resolve carried colour alone, the decals here rendered as
/// nothing at all while the rest of the frame looked correct. Decals are the
/// cheapest post-resolve content to put in shot; the sub-highlight, OIT,
/// scatter and foreground passes all depend on the same buffer.
fn build_supersampled_decals(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut scene = build_decals(ctx);
    let mut post = viewport_lib::PostProcessSettings::default();
    post.ssaa_factor = 2;
    scene.post_process = Some(post);
    scene
}

fn build_refraction_over_soft_sprite(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // Refractive sprites in front of a sheet of soft-particle billboards. The
    // two draw in different passes, so this scene pins whether the refraction
    // samples the soft particles or the bare backdrop behind them.
    let slab = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cuboid(7.0, 7.0, 0.4))
        .expect("slab upload");
    let backdrop_tex = checker_texture(ctx, [230, 80, 60], [235, 225, 205]);
    let mut ground = viewport_lib::SceneRenderItem::default();
    ground.mesh_id = slab;
    ground.model = Mat4::from_translation(Vec3::new(0.0, 0.0, -1.2)).to_cols_array_2d();
    ground.material = Material::pbr([1.0, 1.0, 1.0], 0.0, 0.85);
    ground.material.texture_id = Some(backdrop_tex);

    // Soft particles hugging the slab, so they fade where they intersect it.
    let soft_tex = checker_texture(ctx, [80, 140, 255], [20, 30, 90]);
    let mut haze = SpriteItem::default();
    haze.texture_id = Some(soft_tex);
    haze.positions = (0..6)
        .map(|i| {
            let t = i as f32 / 6.0 * std::f32::consts::TAU;
            [1.9 * t.cos(), 1.9 * t.sin(), -0.75]
        })
        .collect();
    haze.default_size = 2.2;
    haze.default_colour = [1.0, 1.0, 1.0, 0.85].into();
    haze.size_mode = SpriteSizeMode::WorldSpace;
    haze.depth_write = false;
    haze.soft_particle_distance = Some(0.8);

    // Refractive bubbles nearer the camera, overlapping the haze on screen.
    let warp_tex = checker_texture(ctx, [255, 40, 40], [40, 255, 40]);
    let mut bubbles = SpriteItem::default();
    bubbles.texture_id = Some(warp_tex);
    bubbles.positions = (0..5)
        .map(|i| {
            let t = i as f32 / 5.0 * std::f32::consts::TAU;
            [1.5 * t.cos(), 1.5 * t.sin(), 0.3]
        })
        .collect();
    bubbles.default_size = 1.2;
    bubbles.default_colour = [1.0, 1.0, 1.0, 1.0].into();
    bubbles.size_mode = SpriteSizeMode::WorldSpace;
    bubbles.depth_write = false;
    bubbles.refraction_strength = Some(30.0);

    BuiltScene {
        items: vec![ground],
        sprite_items: vec![haze, bubbles],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_decal_on_curves(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // Tube, streamtube and ribbon side by side under one decal box, all three
    // flat along X so the decal projects straight down onto them. The curve
    // types share a draw path, so a decal landing on some but not others is a
    // property of the type rather than of the placement.
    let line = |y: f32| -> Vec<[f32; 3]> {
        (0..24)
            .map(|i| {
                let t = i as f32 / 23.0;
                [-2.6 + t * 5.2, y, 0.0]
            })
            .collect()
    };

    let mut tube = TubeItem::default();
    tube.positions = line(-1.4);
    tube.strip_lengths = vec![24];
    tube.radius = 0.28;
    tube.colour = [0.75, 0.75, 0.78, 1.0].into();

    let mut st = StreamtubeItem::default();
    st.positions = line(0.0);
    st.strip_lengths = vec![24];
    st.radius = 0.28;
    st.colour = [0.75, 0.75, 0.78, 1.0].into();

    let mut rb = RibbonItem::default();
    rb.positions = line(1.4);
    rb.strip_lengths = vec![24];
    rb.width = 0.28;
    // twist_attribute sets the ribbon's width direction, not its normal: with
    // the tangent along X, a width along Y lays the face flat so it points at
    // the decal rather than standing edge-on to it.
    rb.twist_attribute = Some(vec![[0.0, 1.0, 0.0]; 24]);
    rb.colour = [0.75, 0.75, 0.78, 1.0].into();

    let checker = checker_texture(ctx, [220, 70, 40], [240, 220, 200]);
    let mut decal = DecalItem::default();
    decal.transform = (Mat4::from_translation(Vec3::ZERO)
        * Mat4::from_scale(Vec3::new(7.0, 7.0, 4.0)))
    .to_cols_array_2d();
    decal.texture_id = checker;
    decal.blend_mode = DecalBlendMode::Replace;
    decal.alpha = 1.0;

    BuiltScene {
        tube_items: vec![tube],
        streamtube_items: vec![st],
        ribbon_items: vec![rb],
        decals: vec![decal],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_decal_on_non_mesh(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A decal box enclosing both a mesh and a GPU implicit surface. The decal
    // pass reconstructs its receiver from the depth buffer, so it lands on
    // anything that wrote depth; the implicit surface does, and unlike the mesh
    // it has no `receives_decals` to decline with.
    let ball = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::sphere(1.0, 32, 16))
        .expect("sphere upload");
    let mut mesh = viewport_lib::SceneRenderItem::default();
    mesh.mesh_id = ball;
    mesh.model = Mat4::from_translation(Vec3::new(-1.6, 0.0, 0.0)).to_cols_array_2d();
    mesh.material = Material::pbr([0.55, 0.55, 0.58], 0.1, 0.6);
    // The mesh declines the decal. Nothing else in the library can: the flag
    // lives on `SceneRenderItem` alone, so the implicit surface beside it takes
    // the projection whether it wants to or not.
    mesh.receives_decals = false;

    let mut sphere = ImplicitPrimitive::zeroed();
    sphere.kind = 1;
    sphere.blend = 0.3;
    sphere.params[..3].copy_from_slice(&[1.6, 0.0, 0.0]);
    sphere.params[3] = 1.0;
    sphere.colour = [0.55, 0.55, 0.58, 1.0].into();
    let mut implicit = GpuImplicitItem::default();
    implicit.primitives = vec![sphere];
    implicit.blend_mode = ImplicitBlendMode::SmoothUnion;

    let checker = checker_texture(ctx, [220, 70, 40], [240, 220, 200]);
    let mut decal = DecalItem::default();
    decal.transform = (Mat4::from_translation(Vec3::ZERO)
        * Mat4::from_scale(Vec3::new(6.0, 6.0, 4.0)))
    .to_cols_array_2d();
    decal.texture_id = checker;
    decal.blend_mode = DecalBlendMode::Replace;
    decal.alpha = 1.0;

    BuiltScene {
        items: vec![mesh],
        gpu_implicit: vec![implicit],
        decals: vec![decal],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_decal_under_soft_sprite(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A decal on the slab with a sheet of soft-particle billboards hanging
    // directly over it. The two are drawn by different passes, so this scene
    // is what pins their order against each other.
    let slab = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cuboid(8.0, 8.0, 0.5))
        .expect("slab upload");
    let mut ground = viewport_lib::SceneRenderItem::default();
    ground.mesh_id = slab;
    ground.model = Mat4::from_translation(Vec3::new(0.0, 0.0, -0.25)).to_cols_array_2d();
    ground.material = Material::pbr([0.62, 0.6, 0.58], 0.0, 0.8);

    let checker = checker_texture(ctx, [220, 70, 40], [240, 220, 200]);
    let mut decal = DecalItem::default();
    decal.transform = (Mat4::from_translation(Vec3::ZERO)
        * Mat4::from_scale(Vec3::new(4.0, 4.0, 2.0)))
    .to_cols_array_2d();
    decal.texture_id = checker;
    decal.blend_mode = DecalBlendMode::Replace;
    decal.alpha = 1.0;

    let sprite_tex = checker_texture(ctx, [255, 200, 80], [60, 40, 160]);
    let mut sheet = SpriteItem::default();
    sheet.texture_id = Some(sprite_tex);
    sheet.positions = (0..8)
        .map(|i| {
            let theta = i as f32 / 8.0 * std::f32::consts::TAU;
            [1.4 * theta.cos(), 1.4 * theta.sin(), 0.10]
        })
        .collect();
    sheet.default_size = 1.3;
    sheet.default_colour = [1.0, 0.95, 0.8, 0.75].into();
    sheet.size_mode = SpriteSizeMode::WorldSpace;
    sheet.depth_write = false;
    sheet.soft_particle_distance = Some(0.6);

    BuiltScene {
        items: vec![ground],
        sprite_items: vec![sheet],
        decals: vec![decal],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_mesh_instances(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A spiral of tinted cube instances in one batch draw.
    let cube = ctx
        .renderer
        .resources_mut()
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
