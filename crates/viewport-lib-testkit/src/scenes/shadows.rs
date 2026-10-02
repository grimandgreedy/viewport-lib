//! Shadow regression scenes: enclosed rooms, thin stacked receivers, clipped
//! solids and a long hall. The open platform under a high sun that most
//! examples use hides the shadow failure classes; these put the receivers at
//! grazing angles, inside closed geometry, and under cascades wide enough for
//! the texel size to matter. Interiors run with the hemisphere near zero so
//! the shadow term is the only thing lighting a receiver and a leak reads as
//! a bright pixel.

use glam::Vec3;
use viewport_lib::{
    BackfacePolicy, ClipObject, LightKind, LightSource, LightingSettings, Material,
    SceneRenderItem, primitives,
};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, item, orbit_camera, upload};

const PI: f32 = std::f32::consts::PI;

fn sun(direction: [f32; 3], intensity: f32) -> LightSource {
    let mut s = LightSource::default();
    s.kind = LightKind::Directional { direction };
    s.intensity = intensity;
    s
}

fn lamp(position: [f32; 3], candela: f32, range: f32) -> LightSource {
    let mut s = LightSource::point_candela(position, viewport_lib::Candela(candela), range, 0.05);
    s.colour = [1.0, 0.92, 0.8].into();
    s.cast_shadows = true;
    s
}

fn interior(lights: Vec<LightSource>) -> LightingSettings {
    let mut l = LightingSettings::default();
    l.lights = lights;
    l.hemisphere_intensity = 0.05;
    l
}

fn slab(ctx: &mut BuildCtx<'_>, size: [f32; 3], at: [f32; 3], colour: [f32; 3]) -> SceneRenderItem {
    let mesh = upload(ctx, &primitives::cuboid(size[0], size[1], size[2]));
    item(mesh, Vec3::from(at), Material::pbr(colour, 0.0, 0.9))
}

fn two_sided(mut it: SceneRenderItem) -> SceneRenderItem {
    it.material.backface_policy = BackfacePolicy::Identical;
    it
}

/// A closed 6 x 6 x 2.8 m room: floor, ceiling and four 0.2 m walls, each a
/// closed cuboid. The ceiling slab sits between the walls, which rise past
/// it, so no wall top lies in the ceiling's underside plane: a lit face in
/// that plane z-fights through the dark underside as a bright line along the
/// joint and looks like a shadow leak. The walls meet end to end at the
/// corners, which of the three corner layouts tried shows the least of the
/// vertical crease column recorded in `vertical-corner-crease-column`.
/// `open_x` leaves a 1.4 m doorway in the +X wall.
fn room(ctx: &mut BuildCtx<'_>, open_x: bool) -> Vec<SceneRenderItem> {
    const W: f32 = 6.0;
    const H: f32 = 2.8;
    const T: f32 = 0.2;
    let h = W / 2.0 + T / 2.0;
    let wh = H + T;
    let wall = [0.82, 0.8, 0.76];
    let mut items = vec![
        slab(
            ctx,
            [W + 2.0 * T, W + 2.0 * T, T],
            [0.0, 0.0, -T / 2.0],
            [0.7, 0.68, 0.64],
        ),
        slab(ctx, [W, W, T], [0.0, 0.0, H + T / 2.0], wall),
        slab(ctx, [W, T, wh], [0.0, h, wh / 2.0], wall),
        slab(ctx, [W, T, wh], [0.0, -h, wh / 2.0], wall),
        slab(ctx, [T, W, wh], [-h, 0.0, wh / 2.0], wall),
    ];
    if open_x {
        let seg = (W - 1.4) / 2.0;
        let y = 0.7 + seg / 2.0;
        items.push(slab(ctx, [T, seg, wh], [h, y, wh / 2.0], wall));
        items.push(slab(ctx, [T, seg, wh], [h, -y, wh / 2.0], wall));
    } else {
        items.push(slab(ctx, [T, W, wh], [h, 0.0, wh / 2.0], wall));
    }
    items
}

/// A table with four legs and a cube and a sphere resting on it.
fn table_set(ctx: &mut BuildCtx<'_>, at: Vec3) -> Vec<SceneRenderItem> {
    let top_z = at.z + 0.75;
    let mut items = vec![slab(
        ctx,
        [1.6, 0.8, 0.05],
        [at.x, at.y, top_z - 0.025],
        [0.6, 0.45, 0.3],
    )];
    for (sx, sy) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
        items.push(slab(
            ctx,
            [0.06, 0.06, 0.72],
            [at.x + sx * 0.72, at.y + sy * 0.32, at.z + 0.36],
            [0.5, 0.38, 0.25],
        ));
    }
    let cube = upload(ctx, &primitives::cube(0.3));
    let sphere = upload(ctx, &primitives::sphere(0.16, 32, 16));
    items.push(item(
        cube,
        Vec3::new(at.x - 0.4, at.y, top_z + 0.15),
        Material::pbr([0.7, 0.8, 0.95], 0.0, 0.6),
    ));
    items.push(item(
        sphere,
        Vec3::new(at.x + 0.4, at.y + 0.1, top_z + 0.16),
        Material::pbr([0.95, 0.75, 0.6], 0.0, 0.5),
    ));
    items
}

/// Camera standing inside the room near the -X, -Y corner at eye height,
/// looking across the table toward the far corner and the ceiling.
fn room_cameras() -> Vec<NamedCamera> {
    vec![
        NamedCamera {
            name: "inside",
            camera: orbit_camera(Vec3::new(0.4, 0.4, 1.6), 3.4, PI * 0.75, PI / 2.0 + 0.05),
        },
        NamedCamera {
            name: "ceiling",
            camera: orbit_camera(Vec3::new(0.0, 0.0, 2.8), 3.6, PI * 0.75, PI / 2.0 + 0.4),
        },
    ]
}

fn build_room_point(ctx: &mut BuildCtx<'_>, walls_two_sided: bool) -> BuiltScene {
    let mut items = room(ctx, false);
    if walls_two_sided {
        items = items.into_iter().map(two_sided).collect();
    }
    items.extend(table_set(ctx, Vec3::new(0.6, 0.4, 0.0)));
    BuiltScene {
        items,
        lighting: interior(vec![lamp([0.0, 0.0, 2.5], 120.0, 12.0)]),
        ..Default::default()
    }
}

fn build_room_point_light(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    build_room_point(ctx, false)
}

fn build_room_point_light_two_sided(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    build_room_point(ctx, true)
}

/// The room with a doorway in the +X wall and a sun 15 degrees above the
/// horizon shining through it onto the floor and the far wall.
fn build_room_doorway_sun(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut items = room(ctx, true);
    items.extend(table_set(ctx, Vec3::new(-0.6, 0.6, 0.0)));
    BuiltScene {
        items,
        lighting: interior(vec![sun([0.95, 0.12, 0.27], 2.5)]),
        ..Default::default()
    }
}

/// Closed solids with styled back-face policies, opened by a clip plane, on a
/// floor inside the doorway room under the sun and a lamp. The interiors must
/// shade as inside their own shadow: flat, no band, no speckle.
fn build_room_cut_solids(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // The clip plane is for the solids; the room itself stays whole.
    let mut items: Vec<SceneRenderItem> = room(ctx, true)
        .into_iter()
        .map(|mut it| {
            it.settings.ignore_clip = true;
            it
        })
        .collect();
    let sphere = upload(ctx, &primitives::sphere(0.6, 32, 16));
    let cube = upload(ctx, &primitives::cube(1.0));
    let torus = upload(ctx, &primitives::torus(0.5, 0.2, 32, 16));
    let styled = |policy: BackfacePolicy| {
        let mut m = Material::pbr([0.85, 0.85, 0.85], 0.0, 0.6);
        m.backface_policy = policy;
        m
    };
    items.push(item(
        sphere,
        Vec3::new(-1.2, 0.0, 0.6),
        styled(BackfacePolicy::Tint(0.4)),
    ));
    items.push(item(
        cube,
        Vec3::new(0.3, 0.0, 0.5),
        styled(BackfacePolicy::DifferentColour([0.8, 0.3, 0.25].into())),
    ));
    items.push(item(
        torus,
        Vec3::new(1.7, 0.0, 0.5),
        styled(BackfacePolicy::Tint(0.4)),
    ));
    items.push(item(
        sphere,
        Vec3::new(-1.2, 1.6, 0.6),
        styled(BackfacePolicy::Identical),
    ));
    BuiltScene {
        items,
        lighting: interior(vec![
            sun([0.95, 0.12, 0.27], 2.5),
            lamp([0.5, -1.0, 2.5], 80.0, 10.0),
        ]),
        clip: Some((vec![ClipObject::plane([0.0, 1.0, 0.0], 0.0)], false)),
        ..Default::default()
    }
}

/// Five 0.01 m plates stacked with 0.05 m gaps on the ground. The top face of
/// each plate must not self-shadow and no light may leak between plates.
fn slab_stack(ctx: &mut BuildCtx<'_>) -> Vec<SceneRenderItem> {
    let mut items = vec![super::ground(ctx, 0.0)];
    for i in 0..5 {
        let z = 0.3 + i as f32 * 0.06;
        items.push(slab(ctx, [2.0, 1.4, 0.01], [0.0, 0.0, z], [0.9, 0.85, 0.7]));
    }
    // Legs holding the bottom plate off the ground, so the gap under it is real.
    for (sx, sy) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
        items.push(slab(
            ctx,
            [0.05, 0.05, 0.3],
            [sx * 0.9, sy * 0.6, 0.15],
            [0.4, 0.4, 0.42],
        ));
    }
    items
}

fn stack_cameras() -> Vec<NamedCamera> {
    vec![NamedCamera {
        name: "low",
        camera: orbit_camera(Vec3::new(0.0, 0.0, 0.4), 4.0, 0.6, 1.35),
    }]
}

fn build_slab_stack_sun(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut l = LightingSettings::default();
    l.lights = vec![sun([0.87, 0.2, 0.5], 1.2)];
    l.hemisphere_intensity = 0.15;
    BuiltScene {
        items: slab_stack(ctx),
        lighting: l,
        ..Default::default()
    }
}

fn build_slab_stack_point(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    BuiltScene {
        items: slab_stack(ctx),
        lighting: interior(vec![lamp([1.5, -0.8, 2.0], 60.0, 10.0)]),
        ..Default::default()
    }
}

/// A 60 m open-topped hall with pillars every 5 m, lit by a low sun from the
/// side, seen from one end. The cascade splits fall on the pillars and the
/// floor in view, and the far cascades' texels are large.
fn build_long_hall(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    const L: f32 = 60.0;
    const W: f32 = 4.0;
    const H: f32 = 3.0;
    let wall = [0.8, 0.78, 0.74];
    let mut items = vec![
        slab(
            ctx,
            [L, W + 0.4, 0.2],
            [L / 2.0, 0.0, -0.1],
            [0.66, 0.65, 0.62],
        ),
        slab(ctx, [L, 0.2, H], [L / 2.0, W / 2.0 + 0.1, H / 2.0], wall),
        slab(ctx, [L, 0.2, H], [L / 2.0, -W / 2.0 - 0.1, H / 2.0], wall),
        slab(ctx, [0.2, W + 0.4, H], [L + 0.1, 0.0, H / 2.0], wall),
    ];
    let pillar = upload(ctx, &primitives::cuboid(0.4, 0.4, H));
    let pillar_of = |x: f32, y: f32| {
        item(
            pillar,
            Vec3::new(x, y, H / 2.0),
            Material::pbr([0.7, 0.7, 0.72], 0.0, 0.7),
        )
    };
    let mut x = 5.0;
    while x < L {
        items.push(pillar_of(x, W / 2.0 - 0.5));
        items.push(pillar_of(x, -W / 2.0 + 0.5));
        x += 5.0;
    }
    let mut l = LightingSettings::default();
    l.lights = vec![sun([0.3, 0.8, 0.45], 1.5)];
    l.hemisphere_intensity = 0.15;
    BuiltScene {
        items,
        lighting: l,
        ..Default::default()
    }
}

fn hall_cameras() -> Vec<NamedCamera> {
    vec![NamedCamera {
        name: "end",
        camera: orbit_camera(Vec3::new(14.0, 0.0, 1.4), 15.0, PI / 2.0, PI / 2.0 - 0.06),
    }]
}

/// The shadow regression scenes.
pub fn scenes() -> Vec<NamedScene> {
    vec![
        NamedScene {
            name: "room_point_light",
            cameras: room_cameras(),
            build: build_room_point_light,
        },
        NamedScene {
            name: "room_point_light_two_sided",
            cameras: room_cameras(),
            build: build_room_point_light_two_sided,
        },
        NamedScene {
            name: "room_doorway_sun",
            cameras: room_cameras(),
            build: build_room_doorway_sun,
        },
        NamedScene {
            name: "room_cut_solids",
            cameras: vec![NamedCamera {
                name: "inside",
                camera: orbit_camera(Vec3::new(0.2, 0.0, 0.5), 2.6, 0.0, PI / 2.0 - PI / 12.0),
            }],
            build: build_room_cut_solids,
        },
        NamedScene {
            name: "slab_stack_sun",
            cameras: stack_cameras(),
            build: build_slab_stack_sun,
        },
        NamedScene {
            name: "slab_stack_point",
            cameras: stack_cameras(),
            build: build_slab_stack_point,
        },
        NamedScene {
            name: "long_hall",
            cameras: hall_cameras(),
            build: build_long_hall,
        },
    ]
}
