//! Item types far from the world origin.
//!
//! The same content is built at any offset, so a test can draw it at the
//! world origin and again 2^20 m out under a snapped origin and compare the
//! two. Every coordinate is a multiple of 1/8: f32 holds those exactly out
//! to 2^21 m, so the far copy carries no rounding of its own and any
//! difference between the two renders is the renderer's.
//!
//! The content is split into parts, one per item type (or a few sharing a
//! surface), so a mismatch names the type that moved.

use glam::{Mat4, Vec3};
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{
    Aabb, Camera, Colour, ColourmapId, Material, PolylineItem, SceneRenderItem, VolumeData,
    WorldOrigin, primitives,
};
use viewport_lib::{ColourSource, SizeSource};
use viewport_lib_plugins::item_types::curves::{RibbonItem, StreamtubeItem, TubeItem};
use viewport_lib_plugins::item_types::decal::{DecalBlendMode, DecalItem};
use viewport_lib_plugins::item_types::gaussian_splat::{
    GaussianSplatData, GaussianSplatItem, ShDegree,
};
use viewport_lib_plugins::item_types::gpu_implicit::{
    GpuImplicitItem, ImplicitBlendMode, ImplicitPrimitive,
};
use viewport_lib_plugins::item_types::gpu_marching_cubes::{GpuMarchingCubesItem, McVolumes};
use viewport_lib_plugins::item_types::image_slice::{ImageSliceItem, SliceAxis};
use viewport_lib_plugins::item_types::point_cloud::PointCloudItem;
use viewport_lib_plugins::item_types::scatter_volume::{ScatterVolume, ScatterVolumeItem};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteSizeMode};
use viewport_lib_plugins::item_types::surface_contour::{ContourLevels, SurfaceContourItem};
use viewport_lib_plugins::item_types::surface_lic::SurfaceLicItem;
use viewport_lib_plugins::item_types::tensor_field::{TensorFieldItem, TensorSource};
use viewport_lib_plugins::item_types::vector_field::VectorFieldItem;
use viewport_lib_plugins::item_types::volume::VolumeItem;
use viewport_lib_plugins::item_types::volume_surface_slice::VolumeSurfaceSliceItem;

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, material, orbit_camera, rigs};

/// 2^20 m: where the far copy is built.
pub const FAR: f32 = 1_048_576.0;

/// The far offset, on two axes.
pub fn far_offset() -> Vec3 {
    Vec3::new(FAR, FAR, 0.0)
}

/// One part of the content: a name and a function adding it, at an offset,
/// to a scene.
pub struct Part {
    /// Stable name, reported when the part's renders differ.
    pub name: &'static str,
    /// Add the part's items to `scene`, with every position moved by `at`.
    pub add: fn(&mut BuildCtx<'_>, Vec3, &mut BuiltScene),
}

/// Round to a multiple of 1/8.
fn q(v: f32) -> f32 {
    (v * 8.0).round() / 8.0
}

fn qp(at: Vec3, p: [f32; 3]) -> [f32; 3] {
    [at.x + q(p[0]), at.y + q(p[1]), at.z + q(p[2])]
}

fn at_model(at: Vec3, local: Vec3) -> [[f32; 4]; 4] {
    Mat4::from_translation(at + local).to_cols_array_2d()
}

/// The parts the catalogue scene draws together.
pub fn mix_parts() -> Vec<Part> {
    vec![
        Part {
            name: "ground",
            add: ground,
        },
        Part {
            name: "point_cloud",
            add: point_cloud,
        },
        Part {
            name: "sprites",
            add: sprites,
        },
        Part {
            name: "curves",
            add: curves,
        },
        Part {
            name: "vector_field",
            add: vector_field,
        },
        Part {
            name: "decal",
            add: decal,
        },
        Part {
            name: "surface_contour",
            add: surface_contour,
        },
        Part {
            name: "polyline",
            add: polyline,
        },
        Part {
            name: "world_overlays",
            add: world_overlays,
        },
    ]
}

/// Parts drawn only one at a time, by the parity test.
pub fn extra_parts() -> Vec<Part> {
    vec![
        Part {
            name: "tensor_field",
            add: tensor_field,
        },
        Part {
            name: "streamtube",
            add: streamtube,
        },
        Part {
            name: "image_slice",
            add: image_slice,
        },
        Part {
            name: "volume",
            add: volume,
        },
        Part {
            name: "volume_surface_slice",
            add: volume_surface_slice,
        },
        Part {
            name: "gaussian_splats",
            add: gaussian_splats,
        },
        Part {
            name: "gpu_implicit",
            add: gpu_implicit,
        },
        Part {
            name: "gpu_marching_cubes",
            add: gpu_marching_cubes,
        },
        Part {
            name: "scatter_volume",
            add: scatter_volume,
        },
        Part {
            name: "surface_lic",
            add: surface_lic,
        },
        Part {
            name: "external_instances",
            add: external_instances,
        },
        Part {
            name: "gpu_particles",
            add: gpu_particles,
        },
    ]
}

/// Build `parts` at `at` into one scene, lit and framed the same way at any
/// offset.
pub fn build_parts(ctx: &mut BuildCtx<'_>, parts: &[Part], at: Vec3) -> BuiltScene {
    let mut scene = BuiltScene {
        lighting: rigs::from_above(),
        ..Default::default()
    };
    for part in parts {
        (part.add)(ctx, at, &mut scene);
    }
    scene
}

/// The camera every render of the content uses: 5 m from a point just above
/// the slab. The eye's offset from that point rounds to (0, -3, 4)
/// out at 2^20 m.
pub fn camera(at: Vec3) -> Camera {
    let mut camera = orbit_camera(at + Vec3::new(0.0, 0.5, 0.25), 5.0, 0.0, 0.8f32.acos());
    camera.fov_y = std::f32::consts::FRAC_PI_3;
    camera.aspect = 4.0 / 3.0;
    camera
}

/// The catalogue entry: the mixed parts at 2^20 m under a snapped origin.
pub fn scene() -> NamedScene {
    NamedScene {
        name: "item_types_far_origin",
        cameras: vec![NamedCamera {
            name: "near",
            camera: camera(far_offset()),
        }],
        build: build_far_mix,
    }
}

fn build_far_mix(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut scene = build_parts(ctx, &mix_parts(), far_offset());
    scene.world_origin = Some(WorldOrigin::snapped());
    scene
}

// --- parts -------------------------------------------------------------------

fn ground(ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let slab = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cuboid(6.0, 6.0, 0.5))
        .expect("slab upload");
    let mut item = SceneRenderItem::new(
        slab,
        material(
            ctx,
            Material::pbr(Colour::linear_rgb(0.6, 0.6, 0.58), 0.0, 0.8),
        ),
    );
    item.model = at_model(at, Vec3::new(0.0, 0.0, -0.25));
    scene.items.push(item);
}

fn point_cloud(_ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let mut cloud = PointCloudItem::default();
    for i in 0..6 {
        for j in 0..6 {
            let (x, y) = (-2.25 + i as f32 * 0.25, -2.25 + j as f32 * 0.25);
            cloud
                .positions
                .push(qp(at, [x, y, 0.25 + 0.125 * ((i + j) % 3) as f32]));
        }
    }
    cloud.colour = ColourSource::Solid(Colour::linear_rgb(0.9, 0.8, 0.1));
    cloud.size = SizeSource::Uniform(6.0);
    scene.point_clouds.push(cloud);
}

fn sprites(_ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let mut ring = SpriteItem::default();
    ring.positions = (0..6)
        .map(|i| {
            let theta = i as f32 / 6.0 * std::f32::consts::TAU;
            qp(at, [1.5 + 0.5 * theta.cos(), -1.5 + 0.5 * theta.sin(), 0.5])
        })
        .collect();
    ring.default_colour = Colour::linear(0.9, 0.3, 0.6, 1.0);
    ring.default_size = 0.25;
    ring.size_mode = SpriteSizeMode::WorldSpace;
    ring.depth_write = true;
    scene.sprite_items.push(ring);
}

fn curves(_ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let n = 32;
    let helix = |cx: f32, cy: f32| -> Vec<[f32; 3]> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                let theta = t * std::f32::consts::TAU * 1.5;
                qp(
                    at,
                    [
                        cx + 0.5 * theta.cos(),
                        cy + 0.5 * theta.sin(),
                        0.25 + 1.25 * t,
                    ],
                )
            })
            .collect()
    };
    let mut tube = TubeItem::default();
    tube.positions = helix(-1.5, 1.5);
    tube.strip_lengths = vec![n as u32];
    tube.radius = 0.08;
    tube.colour = Colour::linear(0.2, 0.7, 0.4, 1.0);
    scene.tube_items.push(tube);

    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![qp(at, [-2.5, -0.5, 0.25]), qp(at, [-0.5, -0.5, 0.25])];
    ribbon.strip_lengths = vec![2];
    ribbon.width = 0.25;
    ribbon.colour = Colour::linear(0.2, 0.45, 0.8, 1.0);
    scene.ribbon_items.push(ribbon);
}

fn vector_field(ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let shape = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::arrow(0.05, 0.12, 0.3, 12))
        .expect("arrow upload");
    let mut field = VectorFieldItem::new(shape);
    for i in 0..4 {
        for j in 0..4 {
            let (x, y) = (0.75 + i as f32 * 0.375, 0.75 + j as f32 * 0.375);
            field.positions.push(qp(at, [x, y, 0.125]));
            field.vectors.push([-(y - 1.25), x - 1.25, 0.5]);
        }
    }
    field.scale = 0.5;
    scene.vector_fields.push(field);
}

fn decal(ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let n = 8u32;
    let pixels: Vec<u8> = (0..n * n)
        .flat_map(|i| {
            let c = if (i % n + i / n) % 2 == 0 {
                [220, 70, 40]
            } else {
                [240, 220, 200]
            };
            [c[0], c[1], c[2], 255]
        })
        .collect();
    let texture = ctx
        .renderer
        .resources_mut()
        .upload_texture(
            ctx.device,
            ctx.queue,
            viewport_lib::TextureData::srgb(n, n, pixels),
        )
        .expect("checker upload");
    let mut decal = DecalItem::default();
    decal.transform = (Mat4::from_translation(at + Vec3::new(0.0, -1.0, 0.0))
        * Mat4::from_scale(Vec3::new(1.5, 1.5, 1.0)))
    .to_cols_array_2d();
    decal.texture_id = texture;
    decal.blend_mode = DecalBlendMode::Replace;
    decal.alpha = 1.0;
    scene.decals.push(decal);
}

fn surface_contour(ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let mut mesh = primitives::sphere(0.5, 32, 16);
    let values = mesh.positions.iter().map(|p| p[2] * 2.0).collect();
    mesh.attributes.insert(
        "height".to_string(),
        viewport_lib::AttributeData::Vertex(values),
    );
    let ball = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &mesh)
        .expect("sphere upload");
    let model = at_model(at, Vec3::new(0.0, 1.0, 0.5));
    let mut item = SceneRenderItem::new(
        ball,
        material(
            ctx,
            Material::pbr(Colour::linear_rgb(0.4, 0.5, 0.7), 0.0, 0.6),
        ),
    );
    item.model = model;
    scene.items.push(item);
    scene.surface_contours.push(SurfaceContourItem::new(
        ball,
        model,
        "height",
        ContourLevels::Values(vec![-0.5, 0.0, 0.5]),
    ));
}

fn polyline(_ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let mut line = PolylineItem::default();
    line.positions = (0..9)
        .map(|i| {
            let x = -2.0 + i as f32 * 0.5;
            qp(at, [x, 2.25, 0.125 + if i % 2 == 0 { 0.0 } else { 0.5 }])
        })
        .collect();
    line.strip_lengths = vec![9];
    line.default_colour = Colour::linear(1.0, 0.55, 0.1, 1.0);
    line.line_width = 3.0;
    scene.polylines.push(line);
}

fn tensor_field(ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let shape = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::icosphere(1.0, 2))
        .expect("icosphere upload");
    let mut field = TensorFieldItem::new(shape);
    let mut components = Vec::new();
    for i in 0..3 {
        for j in 0..3 {
            field
                .positions
                .push(qp(at, [i as f32 - 1.0, j as f32 - 1.0, 0.5]));
            components.push([0.4, 0.25 + 0.05 * i as f32, 0.15, 0.05 * j as f32, 0.0, 0.0]);
        }
    }
    field.tensors = TensorSource::Components(components);
    scene.tensor_fields.push(field);
}

fn streamtube(_ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let n = 24;
    let mut st = StreamtubeItem::default();
    st.positions = (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            let theta = t * std::f32::consts::PI;
            qp(at, [1.25 * theta.cos(), 1.25 * theta.sin(), 0.5])
        })
        .collect();
    st.strip_lengths = vec![n as u32];
    st.radius = 0.1;
    st.colour = Colour::linear(0.7, 0.45, 0.1, 1.0);
    scene.streamtube_items.push(st);
}

/// A two-lobed scalar field on an `n`-cube grid.
fn lobes(n: usize) -> (Vec<f32>, [u32; 3]) {
    let mut data = vec![0.0f32; n * n * n];
    let f = |i: usize| (i as f32 / (n - 1) as f32) * 2.0 - 1.0;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (fx, fy, fz) = (f(x), f(y), f(z));
                let a = 0.65 - ((fx + 0.3).powi(2) + fy * fy + fz * fz).sqrt();
                let b = 0.45 - ((fx - 0.45).powi(2) + (fy - 0.2).powi(2) + fz * fz).sqrt();
                data[x + n * (y + n * z)] = a.max(b);
            }
        }
    }
    (data, [n as u32; 3])
}

fn image_slice(ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let (data, dims) = lobes(16);
    let vid = ctx
        .renderer
        .resources_mut()
        .upload_volume(ctx.device, ctx.queue, &data, dims);
    let mut slice = ImageSliceItem::default();
    slice.volume_id = vid;
    slice.axis = SliceAxis::Y;
    slice.offset = 0.5;
    slice.bbox_min = qp(at, [-1.0, -1.0, 0.0]);
    slice.bbox_max = qp(at, [1.0, 1.0, 2.0]);
    slice.scalar_range = (-0.4, 0.65);
    slice.colour_lut = Some(ColourmapId(0));
    slice.opacity = 1.0;
    scene.image_slices.push(slice);
}

fn volume(ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let (data, dims) = lobes(16);
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
    v.bbox_min = qp(at, [-1.0, -1.0, 0.0]);
    v.bbox_max = qp(at, [1.0, 1.0, 2.0]);
    v.enable_shading = true;
    scene.volumes.push(v);
}

fn volume_surface_slice(ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let (data, dims) = lobes(16);
    let vid = ctx
        .renderer
        .resources_mut()
        .upload_volume(ctx.device, ctx.queue, &data, dims);
    let mesh_id = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::sphere(0.875, 32, 16))
        .expect("sphere upload");
    let mut ss = VolumeSurfaceSliceItem::default();
    ss.volume_id = vid;
    ss.mesh_id = mesh_id;
    ss.bbox_min = qp(at, [-1.0, -1.0, 0.0]);
    ss.bbox_max = qp(at, [1.0, 1.0, 2.0]);
    ss.scalar_range = (-0.4, 0.65);
    ss.colour_lut = Some(ColourmapId(0));
    ss.opacity = 1.0;
    ss.model = at_model(at, Vec3::new(0.0, 0.0, 1.0));
    scene.volume_surface_slices.push(ss);
}

fn gaussian_splats(ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    const SH0_C: f32 = 0.282_094_79;
    let mut sd = GaussianSplatData::default();
    sd.sh_degree = ShDegree::Zero;
    let n = 8;
    for i in 0..n {
        let t = i as f32 / n as f32;
        let theta = t * std::f32::consts::TAU;
        sd.positions.push(qp(at, [theta.cos(), theta.sin(), 0.75]));
        sd.scales.push([0.25, 0.2, 0.125]);
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
        .upload(ctx.device, ctx.queue, &sd)
        .expect("splat upload");
    let mut item = GaussianSplatItem::default();
    item.source = sid;
    scene.gaussian_splats.push(item);
}

fn gpu_implicit(_ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let sphere = |c: [f32; 3], r: f32, colour: [f32; 4]| {
        let mut p = ImplicitPrimitive::zeroed();
        p.kind = 1;
        p.blend = 0.375;
        p.params[..3].copy_from_slice(&qp(at, c));
        p.params[3] = r;
        p.colour = Colour::from_linear_array(colour);
        p
    };
    let mut item = GpuImplicitItem::default();
    item.primitives = vec![
        sphere([-0.5, 0.0, 0.75], 0.625, [0.38, 0.12, 0.62, 1.0]),
        sphere([0.625, 0.25, 0.75], 0.5, [0.12, 0.5, 0.55, 1.0]),
    ];
    item.blend_mode = ImplicitBlendMode::SmoothUnion;
    scene.gpu_implicit.push(item);
}

fn gpu_marching_cubes(ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let n: u32 = 16;
    let step = 0.125f32;
    let local = [-0.9375f32, -0.9375, 0.0];
    let mut data = Vec::with_capacity((n * n * n) as usize);
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let p = Vec3::new(
                    local[0] + ix as f32 * step,
                    local[1] + iy as f32 * step,
                    local[2] + iz as f32 * step,
                );
                data.push(0.75 - (p - Vec3::new(0.0, 0.0, 0.9375)).length());
            }
        }
    }
    let vol = VolumeData {
        data,
        dims: [n, n, n],
        origin: qp(at, local),
        spacing: [step; 3],
    };
    let volume_id = ctx
        .renderer
        .upload_volume_for_mc(ctx.device, ctx.queue, &vol)
        .expect("mc volume upload");
    let mut material = Material::from_colour(Colour::linear_rgb(0.45, 0.48, 0.52));
    material.roughness = 0.4;
    scene.gpu_mc_items.push(GpuMarchingCubesItem {
        volume_id,
        isovalue: 0.0,
        material,
        settings: viewport_lib::ItemSettings::default(),
        cpu_data: None,
    });
}

fn scatter_volume(_ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let fog = ScatterVolume::box_uniform(
        Aabb {
            min: at + Vec3::new(-2.0, -2.0, 0.0),
            max: at + Vec3::new(2.0, 2.0, 1.5),
        },
        0.375,
        Colour::linear_rgb(0.75, 0.8, 0.9),
    );
    scene.scatter_volumes.push(ScatterVolumeItem::new(fog));
    let mut settings = viewport_lib::ScatterSettings::default();
    settings.temporal = false;
    settings.blue_noise_jitter = false;
    settings.downsample = false;
    settings.quality = viewport_lib::ScatterQuality::High;
    scene.scatter_settings = Some(settings);
}

fn surface_lic(ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    let mut mesh = primitives::sphere(1.0, 32, 16);
    let flow = mesh
        .positions
        .iter()
        .map(|p| {
            Vec3::Z
                .cross(Vec3::from(*p))
                .normalize_or(Vec3::X)
                .to_array()
        })
        .collect();
    mesh.attributes.insert(
        "flow".to_string(),
        viewport_lib::AttributeData::VertexVector(flow),
    );
    let ball = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &mesh)
        .expect("sphere upload");
    let model = at_model(at, Vec3::new(0.0, 0.0, 1.0));
    let mut item = SceneRenderItem::new(
        ball,
        material(
            ctx,
            Material::pbr(Colour::linear_rgb(0.55, 0.6, 0.7), 0.0, 0.7),
        ),
    );
    item.model = model;
    scene.items.push(item);
    scene
        .surface_lics
        .push(SurfaceLicItem::new(ball, model, "flow"));
}

fn external_instances(ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    use viewport_lib::wgpu::util::DeviceExt as _;
    use viewport_lib_plugins::item_types::external_instances::{
        ExternalInstanceSetConfig, ExternalInstanceUploads, ExternalInstancesItem,
    };
    let cube = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cube(1.0))
        .expect("cube upload");
    let positions: Vec<u8> = (0..9)
        .flat_map(|i| qp(at, [(i % 3) as f32 - 1.0, (i / 3) as f32 - 1.0, 0.25]))
        .flat_map(f32::to_le_bytes)
        .collect();
    let buffer = ctx
        .device
        .create_buffer_init(&viewport_lib::wgpu::util::BufferInitDescriptor {
            label: Some("far_origin_instances"),
            contents: &positions,
            usage: viewport_lib::wgpu::BufferUsages::STORAGE,
        });
    let set = ctx
        .renderer
        .create_external_instance_set(ctx.device, &ExternalInstanceSetConfig::new(cube, buffer))
        .expect("external instance set");
    let mut item = ExternalInstancesItem::new(set, 9);
    item.scale = 0.375;
    item.colour = Colour::linear(0.8, 0.5, 0.3, 1.0);
    scene.external_instances.push(item);
}

fn gpu_particles(ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    use viewport_lib_plugins::item_types::gpu_particles::{
        GpuParticleSystemConfig, GpuParticleSystemItem, GpuParticleSystems, ParticleRender,
        SpawnShape, VelocityDist,
    };
    let mut config = GpuParticleSystemConfig::default();
    config.capacity = 512;
    config.render = ParticleRender::Sprite {
        texture_id: None,
        blend: viewport_lib::SpriteBlend::AlphaBlend,
        size_mode: SpriteSizeMode::WorldSpace,
        depth_write: false,
        lit: false,
        lit_params: Default::default(),
        normal_texture_id: None,
    };
    let system = ctx
        .renderer
        .create_gpu_particle_system(ctx.device, ctx.queue, &config)
        .expect("gpu particles are registered");
    let mut item = GpuParticleSystemItem::new(system, 0.25);
    item.emitter.rate = 400.0;
    item.emitter.lifetime = (4.0, 6.0);
    item.emitter.size = 0.125;
    item.emitter.colour = Colour::linear(1.0, 0.8, 0.4, 0.9);
    item.emitter.spawn_shape = SpawnShape::Sphere {
        center: qp(at, [0.0, 0.0, 0.25]),
        radius: 0.25,
    };
    item.emitter.initial_velocity = VelocityDist::UniformCone {
        axis: [0.0, 0.0, 1.0],
        half_angle: 0.5,
        min_speed: 1.0,
        max_speed: 2.0,
    };
    scene.gpu_particle_systems.push(item);
}

/// A label and a marker hung from world positions: the overlay anchors.
fn world_overlays(_ctx: &mut BuildCtx<'_>, at: Vec3, scene: &mut BuiltScene) {
    use viewport_lib::{Alignment, LabelItem, OverlayFill, OverlayShape, OverlayShapeItem};
    scene.overlays.labels.push(
        LabelItem::new("far")
            .with_world_anchor(qp(at, [1.5, 1.5, 0.5]))
            .with_font_size(14.0)
            .with_align(Alignment::CENTRE),
    );
    let mut marker = OverlayShapeItem::new(OverlayShape::Circle, [0.0, 0.0], [10.0, 10.0])
        .with_world_anchor(qp(at, [0.5, -0.5, 0.25]))
        .with_fill(OverlayFill::Solid(Colour::linear(0.1, 0.9, 0.9, 1.0)));
    marker.anchoring.align = Alignment::CENTRE;
    scene.overlays.shapes.push(marker);
}
