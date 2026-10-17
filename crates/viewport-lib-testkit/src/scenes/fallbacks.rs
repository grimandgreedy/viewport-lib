//! Scenes for every feature that keeps a mesh off the instanced path.
//!
//! Each column is one reason an item draws per-object today: a matcap, a
//! scalar or face-colour attribute, a warp, a per-material sampler, UV
//! visualisation, a styled back face, custom data, a wireframe item, normal
//! lines, a lightmap, a vertex override, or submesh material ranges. Every
//! column is drawn twice in the layout the material grids use (per-object
//! above, instanced below), so the lower copy records what the batched path
//! does with the feature: today it draws per-object too, so the rows match;
//! once the feature batches, the rows must still match.

use glam::{Mat4, Vec2, Vec3};
use viewport_lib::wgpu;
use viewport_lib::{
    AttributeData, AttributeKind, AttributeRef, BackfacePattern, BackfacePolicy, BuiltinMatcap,
    Colour, ColourmapId, Material, MeshData, ParamVis, ParamVisMode, PatternConfig, SamplerKey,
    SceneRenderItem, ShadingModel, SubmeshRange, TextureSlot, WrapMode, primitives,
    resources::{LightmapData, LightmapMode},
};

use super::materials::{card, grid_camera, grid_pos, place, upload_texture};
use super::{BuildCtx, BuiltScene, NamedScene, rigs, textures, upload};

const COL_STEP: f32 = 1.15;
const ROW_STEP: f32 = 0.95;

fn sphere() -> MeshData {
    primitives::sphere(0.42, 24, 12)
}

/// A sphere carrying a per-vertex height scalar, NaN over the top cap so the
/// NaN colour has somewhere to show.
fn sphere_scalar() -> MeshData {
    let mut m = sphere();
    let values = m
        .positions
        .iter()
        .map(|p| if p[2] > 0.3 { f32::NAN } else { p[2] / 0.42 })
        .collect();
    m.attributes
        .insert("height".into(), AttributeData::Vertex(values));
    m
}

/// A sphere with a colour per triangle.
fn sphere_faces() -> MeshData {
    let mut m = sphere();
    let colours = (0..m.indices.len() / 3)
        .map(|t| {
            let h = (t as f32 * 0.618) % 1.0;
            [h, 1.0 - h, 0.5, 1.0]
        })
        .collect();
    m.attributes
        .insert("faces".into(), AttributeData::FaceColour(colours));
    m
}

/// A sphere with a ripple vector along its normals.
fn sphere_warp() -> MeshData {
    let mut m = sphere();
    let ripple = m
        .normals
        .iter()
        .zip(&m.positions)
        .map(|(n, p)| {
            let k = (p[2] * 18.0).sin();
            [n[0] * k, n[1] * k, n[2] * k]
        })
        .collect();
    m.attributes
        .insert("ripple".into(), AttributeData::VertexVector(ripple));
    m
}

/// A sphere split into two index ranges, each with its own material.
fn sphere_submeshes() -> MeshData {
    let mut m = sphere();
    let half = (m.indices.len() / 2) / 3 * 3;
    m.submeshes = vec![
        SubmeshRange {
            first_index: 0,
            index_count: half as u32,
        },
        SubmeshRange {
            first_index: half as u32,
            index_count: (m.indices.len() - half) as u32,
        },
    ];
    m
}

fn grey() -> Material {
    Material::pbr(Colour::linear_rgb(0.8, 0.78, 0.74), 0.0, 0.6)
}

/// A column: a mesh, the material to draw it with, and a closure that sets
/// the item-level fields the feature needs.
struct Column {
    mesh: viewport_lib::MeshId,
    material: Material,
    /// Items with a per-material sampler of their own must not have it
    /// replaced by the one `place` uses to pin the per-object path.
    own_sampler: bool,
    set: fn(&mut SceneRenderItem),
    /// Model offset applied before the grid translation.
    model: Mat4,
}

fn no_op(_: &mut SceneRenderItem) {}

fn lay_out(items: &mut Vec<SceneRenderItem>, columns: &[Column], row_pairs: &[f32]) {
    for (pair, opacity) in row_pairs.iter().enumerate() {
        for (c, column) in columns.iter().enumerate() {
            for path in 0..2 {
                let pos = grid_pos(c, columns.len(), pair * 2 + path, COL_STEP, ROW_STEP);
                let before = items.len();
                place(
                    items,
                    column.mesh,
                    Mat4::from_translation(pos) * column.model,
                    column.material,
                    *opacity,
                    path == 0 && !column.own_sampler,
                );
                (column.set)(&mut items[before]);
            }
        }
    }
}

/// Material-level fallbacks: two matcaps, a scalar attribute with a NaN
/// colour, per-face colours, a warp, a clamped sampler on a tiled map, UV
/// visualisation, a hatched back face, custom-data emissive, a wireframe
/// item and normal lines. The second pair of rows draws the same columns at
/// opacity 0.6 through the transparent pass. Transparent matcap is a known
/// gap on every path (`matcap-ignored-at-transparent-opacity`), so those two
/// cells record the gap until it is fixed.
fn build_materials_fallbacks(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let plain = upload(ctx, &sphere());
    let scalar = upload(ctx, &sphere_scalar());
    let faces = upload(ctx, &sphere_faces());
    let warp = upload(ctx, &sphere_warp());
    let back_card = upload(ctx, &card(0.42, 0.84));
    let checker = textures::checker(256, 4, [230, 230, 230], [60, 60, 80]);
    let checker_tex = upload_texture(
        ctx,
        viewport_lib::TextureData::srgb(checker.width, checker.height, checker.rgba),
    );
    let clay = ctx
        .renderer
        .resources_mut()
        .builtin_matcap_id(BuiltinMatcap::Clay);
    let ceramic = ctx
        .renderer
        .resources_mut()
        .builtin_matcap_id(BuiltinMatcap::Ceramic);

    let mut tinted_matcap = Material::from_colour(Colour::linear_rgb(0.3, 0.6, 0.9));
    tinted_matcap.shading_model = ShadingModel::Matcap(clay);
    let mut static_matcap = Material::from_colour(Colour::linear_rgb(0.3, 0.6, 0.9));
    static_matcap.shading_model = ShadingModel::Matcap(ceramic);

    // The map is scaled so its UVs run past 0..1; under clamp the border
    // texels smear, under repeat the checker continues.
    let mut clamped = grey().with_texture(TextureSlot::Albedo, checker_tex);
    clamped.uv_scale = [2.5, 2.5];
    clamped.uv_offset = [-0.75, -0.75];
    clamped = clamped.with_sampler(
        TextureSlot::Albedo,
        SamplerKey {
            wrap_u: WrapMode::ClampToEdge,
            wrap_v: WrapMode::ClampToEdge,
            ..SamplerKey::default()
        },
    );

    let mut uv_vis = grey();
    uv_vis.param_vis = Some(ParamVis {
        mode: ParamVisMode::Checker,
        scale: 6.0,
    });

    let mut hatched = grey();
    hatched.backface_policy = BackfacePolicy::Pattern(PatternConfig {
        pattern: BackfacePattern::Hatching,
        colour: Colour::linear_rgb(0.9, 0.2, 0.2),
        scale: 8.0,
    });

    let plain_col = |material: Material, set: fn(&mut SceneRenderItem)| Column {
        mesh: plain,
        material,
        own_sampler: false,
        set,
        model: Mat4::IDENTITY,
    };
    let columns = [
        plain_col(tinted_matcap, no_op),
        plain_col(static_matcap, no_op),
        Column {
            mesh: scalar,
            material: grey(),
            own_sampler: false,
            set: |it| {
                it.active_attribute = Some(AttributeRef {
                    name: "height".into(),
                    kind: AttributeKind::Vertex,
                });
                it.colourmap_id = Some(ColourmapId(0));
                it.scalar_range = Some((-1.0, 0.3));
                it.nan_colour = Some(Colour::linear_rgb(1.0, 0.0, 1.0));
            },
            model: Mat4::IDENTITY,
        },
        Column {
            mesh: faces,
            material: grey(),
            own_sampler: false,
            set: |it| {
                it.active_attribute = Some(AttributeRef {
                    name: "faces".into(),
                    kind: AttributeKind::FaceColour,
                });
            },
            model: Mat4::IDENTITY,
        },
        Column {
            mesh: warp,
            material: grey(),
            own_sampler: false,
            set: |it| {
                it.warp_attribute = Some("ripple".into());
                it.warp_scale = 0.06;
            },
            model: Mat4::IDENTITY,
        },
        Column {
            mesh: plain,
            material: clamped,
            own_sampler: true,
            set: no_op,
            model: Mat4::IDENTITY,
        },
        plain_col(uv_vis, no_op),
        // The card is turned so its back faces the camera, and lowered so
        // its middle sits where a sphere's centre would.
        Column {
            mesh: back_card,
            material: hatched,
            own_sampler: false,
            set: no_op,
            model: Mat4::from_translation(Vec3::new(0.0, 0.0, -0.42))
                * Mat4::from_rotation_z(std::f32::consts::PI),
        },
        plain_col(grey(), |it| {
            it.settings.custom_data = [0.0, 1.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0];
        }),
        plain_col(grey(), |it| it.settings.wireframe = true),
        plain_col(grey(), |it| it.show_normals = true),
    ];

    let mut items = Vec::new();
    lay_out(&mut items, &columns, &[1.0, 0.6]);

    BuiltScene {
        items,
        lighting: rigs::three_point(),
        background: None,
        ..Default::default()
    }
}

fn linear_texture(
    ctx: &mut BuildCtx<'_>,
    size: u32,
    f: impl Fn(u32, u32) -> [u8; 4],
) -> viewport_lib::TextureId {
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            rgba.extend_from_slice(&f(x, y));
        }
    }
    upload_texture(ctx, viewport_lib::TextureData::linear(size, size, rgba))
}

fn storage_buffer(ctx: &mut BuildCtx<'_>, label: &str, data: &[f32]) -> wgpu::Buffer {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

/// Upload a sphere with a lightmap over its UVs.
fn lightmapped(
    ctx: &mut BuildCtx<'_>,
    data: LightmapData,
    mode: LightmapMode,
) -> viewport_lib::MeshId {
    let mesh = sphere();
    let uv1: Vec<Vec2> = mesh
        .uvs
        .as_ref()
        .expect("sphere has uvs")
        .iter()
        .map(|uv| Vec2::new(uv[0], uv[1]))
        .collect();
    let id = upload(ctx, &mesh);
    ctx.renderer
        .resources_mut()
        .set_lightmap(ctx.device, id, &uv1, data, mode)
        .expect("set lightmap");
    id
}

/// Per-mesh sidecars: a lightmap in each mode (replace, add, occlusion,
/// dominant direction, shadowmask), a position override that flattens the
/// sphere, a normal override that points every normal up, and submesh
/// ranges with two opaque materials and then with one blended range.
fn build_mesh_sidecars(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // Radiance runs warm on the left of the map to cool on the right.
    let radiance = linear_texture(ctx, 16, |x, _| {
        let t = x as f32 / 15.0;
        [
            (255.0 * (1.0 - t)) as u8,
            (90.0 + 60.0 * t) as u8,
            (255.0 * t) as u8,
            255,
        ]
    });
    let occlusion = linear_texture(ctx, 16, |x, y| {
        let v = if (x / 4 + y / 4) % 2 == 0 { 40 } else { 230 };
        [v, v, v, 255]
    });
    // A dominant direction leaning towards +X in tangent space.
    let direction = linear_texture(ctx, 4, |_, _| [220, 128, 180, 255]);
    // The map's left half is fully shadowed by the baked occluders.
    let shadowmask = linear_texture(ctx, 16, |x, _| {
        let v = if x < 8 { 0 } else { 255 };
        [v, v, v, 255]
    });

    let replace = lightmapped(
        ctx,
        LightmapData::NonDirectional { radiance },
        LightmapMode::Replace,
    );
    let add = lightmapped(
        ctx,
        LightmapData::NonDirectional { radiance },
        LightmapMode::Add,
    );
    let ao = lightmapped(
        ctx,
        LightmapData::AmbientOcclusion { occlusion },
        LightmapMode::AmbientOcclusion,
    );
    let directional = lightmapped(
        ctx,
        LightmapData::DominantDirection {
            radiance,
            direction,
        },
        LightmapMode::Replace,
    );
    let masked = lightmapped(
        ctx,
        LightmapData::Shadowmask {
            radiance,
            shadowmask,
        },
        LightmapMode::Subtractive,
    );

    let base = sphere();
    let flattened = upload(ctx, &base);
    let positions: Vec<f32> = base
        .positions
        .iter()
        .flat_map(|p| [p[0], p[1], p[2] * 0.5])
        .collect();
    let buffer = storage_buffer(ctx, "flattened positions", &positions);
    ctx.renderer
        .resources_mut()
        .set_position_override_buffer(flattened, buffer)
        .expect("position override");

    let upright = upload(ctx, &base);
    let normals: Vec<f32> = base.normals.iter().flat_map(|_| [0.0, 0.0, 1.0]).collect();
    let buffer = storage_buffer(ctx, "upright normals", &normals);
    ctx.renderer
        .resources_mut()
        .set_normal_override_buffer(upright, buffer)
        .expect("normal override");

    let ranges = upload(ctx, &sphere_submeshes());
    let white = Material::pbr(Colour::linear_rgb(1.0, 1.0, 1.0), 0.0, 0.6);
    let lm_col = |mesh| Column {
        mesh,
        material: white,
        own_sampler: false,
        set: no_op,
        model: Mat4::IDENTITY,
    };
    let columns = [
        lm_col(replace),
        lm_col(add),
        lm_col(ao),
        lm_col(directional),
        lm_col(masked),
        Column {
            mesh: flattened,
            material: grey(),
            own_sampler: false,
            set: no_op,
            model: Mat4::IDENTITY,
        },
        Column {
            mesh: upright,
            material: grey(),
            own_sampler: false,
            set: no_op,
            model: Mat4::IDENTITY,
        },
        Column {
            mesh: ranges,
            material: grey(),
            own_sampler: false,
            set: |it| {
                it.submesh_materials = Some(vec![
                    Material::pbr(Colour::linear_rgb(0.9, 0.3, 0.2), 0.0, 0.5),
                    Material::pbr(Colour::linear_rgb(0.2, 0.4, 0.9), 1.0, 0.3),
                ]);
            },
            model: Mat4::IDENTITY,
        },
        Column {
            mesh: ranges,
            material: grey(),
            own_sampler: false,
            set: |it| {
                let mut glass = Material::pbr(Colour::linear_rgb(0.2, 0.8, 0.4), 0.0, 0.3);
                glass.alpha_mode = viewport_lib::AlphaMode::Blend;
                glass.base_colour = glass.base_colour.with_alpha(0.45);
                it.submesh_materials = Some(vec![
                    Material::pbr(Colour::linear_rgb(0.9, 0.3, 0.2), 0.0, 0.5),
                    glass,
                ]);
            },
            model: Mat4::IDENTITY,
        },
    ];

    let mut items = Vec::new();
    lay_out(&mut items, &columns, &[1.0]);

    BuiltScene {
        items,
        lighting: rigs::three_point(),
        background: None,
        ..Default::default()
    }
}

pub fn scenes() -> Vec<NamedScene> {
    vec![
        NamedScene {
            name: "materials_fallbacks",
            cameras: grid_camera(-1.57, 12.0),
            build: build_materials_fallbacks,
        },
        NamedScene {
            name: "mesh_sidecars",
            cameras: grid_camera(-0.48, 10.0),
            build: build_mesh_sidecars,
        },
    ]
}
