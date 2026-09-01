//! The scene catalogue: named scenes, each built once into renderer resources.
//!
//! A scene is data, not code duplicated across drivers. Every test, benchmark,
//! and the `catalogue_viewer` example consumes the same `catalogue()` list, so
//! "the scene I looked at" and "the scene a check covers" are the same artifact.
//!
//! Each [`NamedScene`] uploads its meshes/textures through a `build` function and
//! returns a [`BuiltScene`] (surface items plus the lighting rig). Combine a
//! built scene with one of its [`NamedCamera`]s through [`frame_for`] to get a
//! `FrameData` ready to render.

// The corpus the catalogue builds from: procedural meshes, lighting rigs,
// textures, and (optionally) real model files. All behind the `scenes` feature
// with this module.
pub mod meshes;
#[cfg(feature = "real_models")]
pub mod real_models;
pub mod rigs;
pub mod textures;

use glam::{Mat4, Quat, Vec3};
use viewport_lib::wgpu;
use viewport_lib::{
    BackfacePolicy, Camera, CameraFrame, FrameData, GlyphItem, LightingSettings, Material,
    MeshData, MeshId, PointCloudItem, PolylineItem, SceneFrame, SceneRenderItem,
    ViewportGpuResources, primitives,
};

/// Resources a scene's `build` function may upload into.
pub struct BuildCtx<'a> {
    /// Long-lived GPU resources (mesh and texture stores).
    pub res: &'a mut ViewportGpuResources,
    /// The wgpu device.
    pub device: &'a wgpu::Device,
    /// The wgpu queue (needed for texture uploads).
    pub queue: &'a wgpu::Queue,
}

/// A camera viewpoint with a stable name (so a snapshot pins "grazing" and a
/// bench pins "iso").
pub struct NamedCamera {
    /// Stable identifier for this viewpoint.
    pub name: &'static str,
    /// The camera state.
    pub camera: Camera,
}

/// A scene after its assets have been uploaded: the surface items plus the
/// lighting rig and optional background colour.
#[derive(Default)]
pub struct BuiltScene {
    /// World-space surface (mesh) items. Rebuilt into a `SceneFrame` per frame by
    /// [`frame_for`] (`SceneFrame` is not `Clone`, so the items are kept instead).
    pub items: Vec<SceneRenderItem>,
    /// Point-cloud items.
    pub point_clouds: Vec<PointCloudItem>,
    /// Polyline items.
    pub polylines: Vec<PolylineItem>,
    /// Glyph (arrow/sphere/cube instance) items.
    pub glyphs: Vec<GlyphItem>,
    /// Lighting rig for this scene.
    pub lighting: LightingSettings,
    /// Optional background clear colour (linear RGBA).
    pub background: Option<[f32; 4]>,
}

/// A catalogue entry: a name, the cameras to view it from, and a function that
/// uploads its assets and returns the built scene.
pub struct NamedScene {
    /// Stable identifier for this scene.
    pub name: &'static str,
    /// Viewpoints worth rendering this scene from.
    pub cameras: Vec<NamedCamera>,
    /// Upload assets through the context and return the built scene.
    pub build: fn(&mut BuildCtx<'_>) -> BuiltScene,
}

/// Build an orbit camera looking at `centre` from `distance`, using the same
/// yaw/pitch convention as `Camera::default` (identity = top-down in Z-up).
pub fn orbit_camera(centre: Vec3, distance: f32, yaw: f32, pitch: f32) -> Camera {
    Camera {
        // `center` is viewport-lib's field name (external API keeps its spelling).
        center: centre,
        distance,
        orientation: Quat::from_rotation_z(yaw) * Quat::from_rotation_x(pitch),
        ..Camera::default()
    }
}

/// A standard set of viewpoints (iso, front, top, low grazing) around a target.
pub fn standard_cameras(centre: Vec3, distance: f32) -> Vec<NamedCamera> {
    vec![
        NamedCamera {
            name: "iso",
            camera: orbit_camera(centre, distance, 0.7, 1.0),
        },
        NamedCamera {
            name: "front",
            camera: orbit_camera(centre, distance, 0.0, 1.5708),
        },
        NamedCamera {
            name: "top",
            camera: orbit_camera(centre, distance, 0.0, 0.15),
        },
        NamedCamera {
            name: "low",
            camera: orbit_camera(centre, distance, 0.6, 1.95),
        },
    ]
}

/// Background clear colour every test frame pins explicitly, in linear light.
///
/// A scene that sets its own background overrides this; otherwise `frame_for`
/// uses it rather than leaving the background unset (which would inherit the
/// renderer's default). Pinning it here keeps the tests independent of the
/// renderer's default background, so a change to that default does not move
/// every golden. The value happens to match the renderer's current default, but
/// nothing depends on that: it just has to stay fixed here.
pub const TEST_BACKGROUND: [f32; 4] = [0.0437, 0.0437, 0.0513, 1.0];

/// Assemble a `FrameData` from a built scene and one camera.
///
/// The frame pins two pieces of viewport chrome so the tests do not depend on
/// renderer defaults: the background is set explicitly (to the scene's own
/// colour, or [`TEST_BACKGROUND`]), and the axes-orientation indicator is turned
/// off so it never draws over the scene.
pub fn frame_for(scene: &BuiltScene, camera: &Camera, viewport_size: [f32; 2]) -> FrameData {
    let mut sf = SceneFrame::from_surface_items(scene.items.clone());
    sf.point_clouds = scene.point_clouds.clone();
    sf.polylines = scene.polylines.clone();
    sf.glyphs = scene.glyphs.clone();
    let mut fd = FrameData::new(CameraFrame::from_camera(camera, viewport_size), sf);
    fd.effects.lighting = scene.lighting.clone();
    fd.viewport.background_colour = Some(scene.background.unwrap_or(TEST_BACKGROUND));
    fd.viewport.show_axes_indicator = false;
    fd
}

// --- small build helpers ---------------------------------------------------

fn upload(ctx: &mut BuildCtx<'_>, mesh: &MeshData) -> MeshId {
    ctx.res
        .upload_mesh_data(ctx.device, mesh)
        .expect("mesh upload")
}

fn item(mesh: MeshId, pos: Vec3, material: Material) -> SceneRenderItem {
    let mut it = SceneRenderItem::default();
    it.mesh_id = mesh;
    it.model = Mat4::from_translation(pos).to_cols_array_2d();
    it.material = material;
    it
}

fn item_model(mesh: MeshId, model: Mat4, material: Material) -> SceneRenderItem {
    let mut it = SceneRenderItem::default();
    it.mesh_id = mesh;
    it.model = model.to_cols_array_2d();
    it.material = material;
    it
}

fn two_sided(mut it: SceneRenderItem) -> SceneRenderItem {
    it.material.backface_policy = BackfacePolicy::Identical;
    it
}

fn with_opacity(mut it: SceneRenderItem, opacity: f32) -> SceneRenderItem {
    it.settings.opacity = opacity;
    it
}

/// A wide flat ground slab centred at the origin, top face at z = `top`.
fn ground(ctx: &mut BuildCtx<'_>, top: f32) -> SceneRenderItem {
    let mesh = upload(ctx, &primitives::cuboid(40.0, 40.0, 0.4));
    item(
        mesh,
        Vec3::new(0.0, 0.0, top - 0.2),
        Material::pbr([0.55, 0.55, 0.58], 0.0, 0.9),
    )
}

// --- scenes ----------------------------------------------------------------

fn build_primitives_trio(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let s = upload(ctx, &primitives::sphere(0.8, 32, 16));
    let c = upload(ctx, &primitives::cube(1.2));
    let t = upload(ctx, &primitives::torus(0.6, 0.22, 32, 16));
    BuiltScene {
        items: vec![
            item(
                s,
                Vec3::new(-2.5, 0.0, 0.0),
                Material::pbr([0.9, 0.5, 0.2], 0.2, 0.5),
            ),
            item(
                c,
                Vec3::new(0.0, 0.0, 0.0),
                Material::pbr([0.4, 0.6, 0.9], 0.6, 0.4),
            ),
            item(
                t,
                Vec3::new(2.5, 0.0, 0.0),
                Material::pbr([0.3, 0.8, 0.4], 0.3, 0.5),
            ),
        ],
        lighting: rigs::from_above(),
        background: None,
        ..Default::default()
    }
}

fn build_torus_knot(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let k = upload(ctx, &meshes::torus_knot(2, 3, 240, 16, 0.32));
    BuiltScene {
        items: vec![item_model(
            k,
            Mat4::from_scale(Vec3::splat(0.9)),
            Material::pbr([0.85, 0.45, 0.5], 0.4, 0.4),
        )],
        lighting: rigs::grazing(),
        background: None,
        ..Default::default()
    }
}

fn build_gear(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let g = upload(ctx, &meshes::gear(14, 1.0, 1.35, 0.45));
    BuiltScene {
        items: vec![item(
            g,
            Vec3::ZERO,
            Material::pbr([0.7, 0.7, 0.75], 0.9, 0.35),
        )],
        lighting: rigs::three_point(),
        background: None,
        ..Default::default()
    }
}

fn build_bowl(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let b = upload(ctx, &meshes::bowl(1.4, 48, 14));
    BuiltScene {
        items: vec![two_sided(item(
            b,
            Vec3::ZERO,
            Material::pbr([0.8, 0.78, 0.7], 0.1, 0.6),
        ))],
        lighting: rigs::from_above(),
        background: None,
        ..Default::default()
    }
}

fn build_castellated(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let g = ground(ctx, 0.0);
    let bar = upload(ctx, &meshes::castellated_bar(6, 5.0, 0.7, 1.0));
    BuiltScene {
        items: vec![
            g,
            item(
                bar,
                Vec3::new(0.0, 0.0, 0.5),
                Material::pbr([0.6, 0.4, 0.35], 0.0, 0.7),
            ),
        ],
        lighting: rigs::grazing(),
        background: None,
        ..Default::default()
    }
}

fn build_heightfield(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let h = upload(ctx, &meshes::heightfield(96, 96, 4.0, 0.8));
    BuiltScene {
        items: vec![item(
            h,
            Vec3::ZERO,
            Material::pbr([0.4, 0.55, 0.35], 0.0, 0.85),
        )],
        lighting: rigs::grazing(),
        background: None,
        ..Default::default()
    }
}

fn build_thin_sheet(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let sheet = upload(ctx, &meshes::thin_sheet(48, 48, 2.5, 0.18));
    BuiltScene {
        items: vec![two_sided(item(
            sheet,
            Vec3::ZERO,
            Material::pbr([0.85, 0.3, 0.35], 0.0, 0.5),
        ))],
        lighting: rigs::backlit(),
        background: None,
        ..Default::default()
    }
}

fn build_stress_sphere(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let s = upload(ctx, &meshes::stress_sphere(1.5, 6));
    BuiltScene {
        items: vec![item(
            s,
            Vec3::ZERO,
            Material::pbr([0.6, 0.6, 0.85], 0.3, 0.3),
        )],
        lighting: rigs::from_above(),
        background: None,
        ..Default::default()
    }
}

fn build_concave_shadows(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let g = ground(ctx, 0.0);
    let k = upload(ctx, &meshes::torus_knot(2, 5, 240, 14, 0.28));
    let b = upload(ctx, &meshes::bowl(1.2, 40, 12));
    BuiltScene {
        items: vec![
            g,
            item_model(
                k,
                Mat4::from_translation(Vec3::new(-1.6, 0.0, 1.4))
                    * Mat4::from_scale(Vec3::splat(0.7)),
                Material::pbr([0.8, 0.5, 0.3], 0.4, 0.4),
            ),
            two_sided(item(
                b,
                Vec3::new(1.8, 0.0, 0.6),
                Material::pbr([0.75, 0.75, 0.8], 0.1, 0.6),
            )),
        ],
        lighting: rigs::grazing(),
        background: None,
        ..Default::default()
    }
}

fn build_textured_checker(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let tex = textures::checker(512, 8, [230, 230, 230], [40, 40, 50]);
    let tex_id = ctx
        .res
        .upload_texture(ctx.device, ctx.queue, tex.width, tex.height, &tex.rgba)
        .expect("texture upload");
    let s = upload(ctx, &primitives::sphere(1.2, 48, 24));
    let mut mat = Material::pbr([1.0, 1.0, 1.0], 0.0, 0.6);
    mat.texture_id = Some(tex_id);
    BuiltScene {
        items: vec![item(s, Vec3::ZERO, mat)],
        lighting: rigs::from_above(),
        background: None,
        ..Default::default()
    }
}

fn build_textured_normalmap(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let nm = textures::normal_bumps(512, 8);
    let nm_id = ctx
        .res
        .upload_normal_map(ctx.device, ctx.queue, nm.width, nm.height, &nm.rgba)
        .expect("normal map upload");
    let s = upload(ctx, &primitives::sphere(1.3, 64, 32));
    let mut mat = Material::pbr([0.7, 0.7, 0.75], 0.1, 0.5);
    mat.normal_map_id = Some(nm_id);
    BuiltScene {
        items: vec![item(s, Vec3::ZERO, mat)],
        lighting: rigs::grazing(),
        background: None,
        ..Default::default()
    }
}

fn build_transparent(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let s = upload(ctx, &primitives::sphere(1.0, 32, 16));
    let cols = [[0.9, 0.3, 0.3], [0.3, 0.9, 0.3], [0.3, 0.3, 0.9]];
    let items = cols
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let x = (i as f32 - 1.0) * 1.1;
            with_opacity(
                item(s, Vec3::new(x, 0.0, 0.0), Material::pbr(c, 0.0, 0.4)),
                0.5,
            )
        })
        .collect();
    BuiltScene {
        items,
        lighting: rigs::from_above(),
        background: None,
        ..Default::default()
    }
}

fn build_materials_pbr(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let s = upload(ctx, &primitives::sphere(0.7, 48, 24));
    let mut items = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            let metallic = i as f32 / 4.0;
            let roughness = (j as f32 / 4.0).max(0.05);
            let pos = Vec3::new((i as f32 - 2.0) * 1.6, (j as f32 - 2.0) * 1.6, 0.0);
            items.push(item(
                s,
                pos,
                Material::pbr([0.85, 0.82, 0.78], metallic, roughness),
            ));
        }
    }
    BuiltScene {
        items,
        lighting: rigs::three_point(),
        background: None,
        ..Default::default()
    }
}

fn build_many_objects(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let c = upload(ctx, &primitives::cube(0.6));
    let s = upload(ctx, &primitives::sphere(0.4, 16, 8));
    let mut items = Vec::new();
    let n = 12;
    for i in 0..n {
        for j in 0..n {
            let x = (i as f32 - (n as f32 - 1.0) * 0.5) * 1.2;
            let y = (j as f32 - (n as f32 - 1.0) * 0.5) * 1.2;
            let (mesh, colour) = if (i + j) % 2 == 0 {
                (c, [0.8, 0.5, 0.3])
            } else {
                (s, [0.3, 0.6, 0.8])
            };
            items.push(item(
                mesh,
                Vec3::new(x, y, 0.0),
                Material::pbr(colour, 0.2, 0.5),
            ));
        }
    }
    BuiltScene {
        items,
        lighting: rigs::from_above(),
        background: None,
        ..Default::default()
    }
}

fn build_lights_eight(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let g = ground(ctx, -1.0);
    let s = upload(ctx, &primitives::sphere(0.8, 32, 16));
    let mut items = vec![g];
    for i in 0..3 {
        items.push(item(
            s,
            Vec3::new((i as f32 - 1.0) * 2.0, 0.0, 0.0),
            Material::pbr([0.85, 0.85, 0.85], 0.1, 0.4),
        ));
    }
    BuiltScene {
        items,
        lighting: rigs::eight_point_lights(),
        background: Some([0.02, 0.02, 0.03, 1.0]),
        ..Default::default()
    }
}

fn build_game_mix(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A small stand-in for the consumer's mixed scene: "buildings" (cuboids),
    // "characters" (capsules), and "props" (small cubes) on a ground plane.
    // Counts are modest here; the heavy million-triangle variant lives in the
    // frame benchmark.
    let g = ground(ctx, 0.0);
    let building = upload(ctx, &primitives::cuboid(1.4, 1.4, 3.0));
    let character = upload(ctx, &primitives::capsule(0.35, 1.2, 16, 8));
    let prop = upload(ctx, &primitives::cube(0.4));
    let mut items = vec![g];

    for i in 0..6 {
        for j in 0..6 {
            let x = (i as f32 - 2.5) * 3.0;
            let y = (j as f32 - 2.5) * 3.0;
            items.push(item(
                building,
                Vec3::new(x, y, 1.5),
                Material::pbr([0.6, 0.6, 0.62], 0.1, 0.8),
            ));
        }
    }
    for i in 0..8 {
        let a = i as f32 / 8.0 * std::f32::consts::TAU;
        items.push(item(
            character,
            Vec3::new(a.cos() * 6.0, a.sin() * 6.0, 0.6),
            Material::pbr([0.8, 0.5, 0.4], 0.0, 0.6),
        ));
    }
    for i in 0..40 {
        let a = i as f32 * 0.61;
        let r = 1.5 + (i as f32) * 0.18;
        items.push(item(
            prop,
            Vec3::new(a.cos() * r, a.sin() * r, 0.2),
            Material::pbr([0.7, 0.7, 0.3], 0.3, 0.5),
        ));
    }
    BuiltScene {
        items,
        lighting: rigs::three_point(),
        background: None,
        ..Default::default()
    }
}

// --- non-mesh item scenes --------------------------------------------------

fn build_point_cloud(_ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A Fibonacci sphere of points, coloured by height.
    let n = 3000usize;
    let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    let mut positions = Vec::with_capacity(n);
    let mut scalars = Vec::with_capacity(n);
    for i in 0..n {
        let y = 1.0 - (i as f32 / (n - 1) as f32) * 2.0;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden * i as f32;
        let p = [theta.cos() * r * 1.5, theta.sin() * r * 1.5, y * 1.5];
        scalars.push(p[2]);
        positions.push(p);
    }
    let mut pc = PointCloudItem::default();
    pc.positions = positions;
    pc.scalars = scalars;
    pc.point_size = 5.0;
    BuiltScene {
        point_clouds: vec![pc],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_polyline(_ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A single helix strip, coloured along its length.
    let n = 400usize;
    let mut positions = Vec::with_capacity(n);
    let mut scalars = Vec::with_capacity(n);
    for i in 0..n {
        let f = i as f32 / n as f32;
        let t = f * std::f32::consts::TAU * 5.0;
        let z = f * 3.0 - 1.5;
        positions.push([t.cos() * 1.2, t.sin() * 1.2, z]);
        scalars.push(z);
    }
    let mut pl = PolylineItem::default();
    pl.positions = positions;
    pl.strip_lengths = vec![n as u32];
    pl.scalars = scalars;
    pl.line_width = 4.0;
    BuiltScene {
        polylines: vec![pl],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_glyphs(_ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // An 8x8 grid of arrow glyphs following a simple swirl field.
    let mut positions = Vec::new();
    let mut vectors = Vec::new();
    for i in 0..8 {
        for j in 0..8 {
            let x = (i as f32 - 3.5) * 0.7;
            let y = (j as f32 - 3.5) * 0.7;
            positions.push([x, y, 0.0]);
            // Swirl: vector perpendicular to the radius, rising slightly.
            vectors.push([-y * 0.3, x * 0.3, 0.25]);
        }
    }
    let mut g = GlyphItem::default();
    g.positions = positions;
    g.vectors = vectors;
    g.scale = 0.6;
    BuiltScene {
        glyphs: vec![g],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

/// The full catalogue of named scenes. The same list drives the counter tests,
/// the snapshot tests, the benches, and the `catalogue_viewer` example.
pub fn catalogue() -> Vec<NamedScene> {
    vec![
        NamedScene {
            name: "primitives_trio",
            cameras: standard_cameras(Vec3::ZERO, 9.0),
            build: build_primitives_trio,
        },
        NamedScene {
            name: "torus_knot",
            cameras: standard_cameras(Vec3::ZERO, 9.0),
            build: build_torus_knot,
        },
        NamedScene {
            name: "gear",
            cameras: standard_cameras(Vec3::ZERO, 5.0),
            build: build_gear,
        },
        NamedScene {
            name: "bowl",
            cameras: standard_cameras(Vec3::ZERO, 5.0),
            build: build_bowl,
        },
        NamedScene {
            name: "castellated_bar",
            cameras: standard_cameras(Vec3::new(0.0, 0.0, 0.7), 9.0),
            build: build_castellated,
        },
        NamedScene {
            name: "heightfield",
            cameras: standard_cameras(Vec3::ZERO, 11.0),
            build: build_heightfield,
        },
        NamedScene {
            name: "thin_sheet",
            cameras: standard_cameras(Vec3::ZERO, 7.0),
            build: build_thin_sheet,
        },
        NamedScene {
            name: "stress_sphere",
            cameras: standard_cameras(Vec3::ZERO, 5.0),
            build: build_stress_sphere,
        },
        NamedScene {
            name: "concave_shadows",
            cameras: standard_cameras(Vec3::new(0.0, 0.0, 0.8), 9.0),
            build: build_concave_shadows,
        },
        NamedScene {
            name: "textured_checker",
            cameras: standard_cameras(Vec3::ZERO, 5.0),
            build: build_textured_checker,
        },
        NamedScene {
            name: "textured_normalmap",
            cameras: standard_cameras(Vec3::ZERO, 5.0),
            build: build_textured_normalmap,
        },
        NamedScene {
            name: "transparent",
            cameras: standard_cameras(Vec3::ZERO, 7.0),
            build: build_transparent,
        },
        NamedScene {
            name: "materials_pbr",
            cameras: standard_cameras(Vec3::ZERO, 14.0),
            build: build_materials_pbr,
        },
        NamedScene {
            name: "many_objects",
            cameras: standard_cameras(Vec3::ZERO, 22.0),
            build: build_many_objects,
        },
        NamedScene {
            name: "lights_eight",
            cameras: standard_cameras(Vec3::ZERO, 12.0),
            build: build_lights_eight,
        },
        NamedScene {
            name: "game_mix",
            cameras: standard_cameras(Vec3::new(0.0, 0.0, 1.0), 32.0),
            build: build_game_mix,
        },
        NamedScene {
            name: "point_cloud",
            cameras: standard_cameras(Vec3::ZERO, 6.0),
            build: build_point_cloud,
        },
        NamedScene {
            name: "polyline",
            cameras: standard_cameras(Vec3::ZERO, 7.0),
            build: build_polyline,
        },
        NamedScene {
            name: "glyphs",
            cameras: standard_cameras(Vec3::ZERO, 8.0),
            build: build_glyphs,
        },
    ]
}

/// Look up a scene by name.
pub fn scene_by_name(name: &str) -> Option<NamedScene> {
    catalogue().into_iter().find(|s| s.name == name)
}
