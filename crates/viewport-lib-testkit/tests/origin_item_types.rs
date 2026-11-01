//! Item types 2^20 m from the world origin under a snapped origin draw as
//! the same content does at the world origin.
//!
//! Each part is built twice, at the world origin under `WorldOrigin::Zero`
//! and 2^20 m out under `WorldOrigin::snapped()`, and the two renders are
//! compared pixel by pixel. Every coordinate is a multiple of 1/8, exact in
//! f32 out there, and the camera's eye lands on the same offset from its
//! target in both, so the only differences left are rounding in the last
//! bits of the view matrix and of the relative positions.

use viewport_lib_testkit::scenes::far_origin::{
    self, FAR, Part, build_parts, extra_parts, mix_parts,
};
use viewport_lib_testkit::{BuildCtx, Harness, frame_for};

const W: u32 = 200;
const H: u32 = 150;

/// A pixel differs when any channel moves by more than this, out of 255.
const CHANNEL_TOLERANCE: u8 = 2;

/// At most this many pixels in 1000 may differ: a few edge pixels where a
/// last-bit difference in a projected position flips coverage.
const PIXELS_PER_MILLE: usize = 5;

fn render(harness: &mut Harness, parts: &[Part], far: bool, generation: u64) -> Vec<u8> {
    let at = if far {
        far_origin::far_offset()
    } else {
        glam::Vec3::ZERO
    };
    let mut ctx = BuildCtx {
        renderer: &mut harness.renderer,
        device: &harness.device,
        queue: &harness.queue,
    };
    let mut scene = build_parts(&mut ctx, parts, at);
    scene.generation = generation;
    scene.world_origin = Some(if far {
        viewport_lib::WorldOrigin::snapped()
    } else {
        viewport_lib::WorldOrigin::Zero
    });
    let frame = frame_for(&scene, &far_origin::camera(at), [W as f32, H as f32]);
    let _ = harness.render(&frame, W, H);
    harness.render(&frame, W, H)
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(x, y)| {
            x.iter()
                .zip(*y)
                .any(|(p, q)| p.abs_diff(*q) > CHANNEL_TOLERANCE)
        })
        .count()
}

#[test]
fn item_types_far_from_origin_match_the_world_origin() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(FAR, 1_048_576.0);
    let mut runs: Vec<(String, Vec<Part>)> = Vec::new();
    for part in mix_parts().into_iter().skip(1).chain(extra_parts()) {
        let name = part.name.to_string();
        let ground = mix_parts().into_iter().next().unwrap();
        runs.push((name, vec![ground, part]));
    }
    runs.push(("mix".to_string(), mix_parts()));

    let limit = (W * H) as usize * PIXELS_PER_MILLE / 1000;
    let mut failures = Vec::new();
    let mut generation = 10_000;
    for (name, parts) in &runs {
        generation += 2;
        let near = render(&mut harness, parts, false, generation);
        let far = render(&mut harness, parts, true, generation + 1);
        let blank = differing(&near, &vec![0u8; near.len()]) == 0;
        let n = differing(&near, &far);
        eprintln!("{name}: {n} of {} pixels differ", W * H);
        if std::env::var_os("ORIGIN_PARITY_DUMP").is_some() {
            let dir = std::env::temp_dir().join("origin_parity");
            let _ = std::fs::create_dir_all(&dir);
            for (tag, px) in [("near", &near), ("far", &far)] {
                viewport_lib_testkit::golden::RgbaImage::from_raw(W, H, px.clone())
                    .expect("image")
                    .save(dir.join(format!("{name}_{tag}.png")))
                    .expect("save");
            }
        }
        assert!(!blank, "{name} rendered nothing");
        if n > limit {
            failures.push(format!("{name}: {n} pixels differ (allowed {limit})"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The ground slab lit by a point light and cut by a clip plane, both placed
/// at `at` plus odd multiples of 1/16. f32 steps by 1/8 at 2^20 m and would
/// round each by 1/16 there, so only an f64 position draws the far copy in
/// place.
fn render_light_and_clip(harness: &mut Harness, far: bool, generation: u64) -> Vec<u8> {
    let at = if far {
        far_origin::far_offset()
    } else {
        glam::Vec3::ZERO
    };
    let mut ctx = BuildCtx {
        renderer: &mut harness.renderer,
        device: &harness.device,
        queue: &harness.queue,
    };
    let ground = mix_parts().into_iter().take(1).collect::<Vec<_>>();
    let mut scene = build_parts(&mut ctx, &ground, at);
    scene.generation = generation;
    scene.world_origin = Some(if far {
        viewport_lib::WorldOrigin::snapped()
    } else {
        viewport_lib::WorldOrigin::Zero
    });
    let at = at.as_dvec3();
    let mut light = viewport_lib::LightSource::point_candela(
        at + glam::DVec3::new(0.3125, -0.6875, 0.5625),
        viewport_lib::Candela(12.0),
        4.0,
        0.05,
    );
    light.colour = viewport_lib::Colour::linear_rgb(1.0, 0.6, 0.3);
    scene.lighting.lights.push(light);
    // Keep x >= at.x + 0.3125: `dot(p, n) + d >= 0` with n = +X.
    let plane = viewport_lib::ClipObject::plane([1.0, 0.0, 0.0], -(at.x + 0.3125));
    scene.clip = Some((vec![plane], false));
    let frame = frame_for(
        &scene,
        &far_origin::camera(at.as_vec3()),
        [W as f32, H as f32],
    );
    let _ = harness.render(&frame, W, H);
    harness.render(&frame, W, H)
}

#[test]
fn point_light_and_clip_plane_far_from_origin_match_the_world_origin() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let near = render_light_and_clip(&mut harness, false, 20_000);
    let far = render_light_and_clip(&mut harness, true, 20_001);
    let blank = differing(&near, &vec![0u8; near.len()]) == 0;
    assert!(!blank, "the light and clip scene rendered nothing");
    let n = differing(&near, &far);
    let limit = (W * H) as usize * PIXELS_PER_MILLE / 1000;
    eprintln!("light_and_clip: {n} of {} pixels differ", W * H);
    assert!(n <= limit, "{n} pixels differ (allowed {limit})");
}

/// 6 400 km: far enough out that an f32 world position steps by half a metre.
const ROOM_FAR: f64 = 6_400_000.0;

/// A small building, as (centre, size): a floor, four walls with a doorway
/// and a plinth.
const ROOM_BOXES: [([f64; 3], [f32; 3]); 8] = [
    ([0.0, 0.0, -0.1], [4.0, 4.0, 0.2]),
    ([0.0, 1.9, 1.2], [4.0, 0.2, 2.4]),
    ([-1.9, 0.0, 1.2], [0.2, 4.0, 2.4]),
    ([1.9, 0.0, 1.2], [0.2, 4.0, 2.4]),
    ([-1.25, -1.9, 1.2], [1.5, 0.2, 2.4]),
    ([1.25, -1.9, 1.2], [1.5, 0.2, 2.4]),
    ([0.0, -1.9, 2.1], [1.0, 0.2, 0.6]),
    ([0.6, 0.6, 0.25], [0.5, 0.5, 0.5]),
];

/// The building lit only by a shadowed point light inside it, seen from 7 m,
/// optionally cut open by a clip plane. At the world origin under
/// `WorldOrigin::Zero`, or 6 400 km out under a snapped origin.
fn render_room(harness: &mut Harness, far: bool, clip: bool, generation: u64) -> Vec<u8> {
    use glam::DVec3;
    use viewport_lib::{Colour, LightKind, LightSource, Material, SceneRenderItem, primitives};
    let base = DVec3::new(if far { ROOM_FAR } else { 0.0 }, 0.0, 0.0);
    let mut scene = viewport_lib_testkit::BuiltScene::default();
    let res = harness.renderer.resources_mut();
    let mat = res
        .upload_material(Material::pbr(Colour::linear_rgb(0.7, 0.7, 0.68), 0.0, 0.8))
        .expect("material upload");
    for (centre, size) in ROOM_BOXES {
        let mesh = res
            .upload_mesh_data(
                &harness.device,
                &primitives::cuboid(size[0], size[1], size[2]),
            )
            .expect("mesh upload");
        let mut item = SceneRenderItem::new(mesh, mat);
        item.model = viewport_lib::Transform::from_translation(base + DVec3::from(centre));
        scene.items.push(item);
    }
    let mut lamp = LightSource::default();
    lamp.kind = LightKind::Point {
        position: base + DVec3::new(-0.6, 0.4, 1.9),
        range: 8.0,
        radius: 0.05,
    };
    lamp.colour = Colour::linear_rgb(1.0, 0.75, 0.45);
    lamp.intensity = 6.0;
    lamp.cast_shadows = true;
    scene.lighting = viewport_lib::LightingSettings::default();
    scene.lighting.lights = vec![lamp];
    scene.lighting.hemisphere_intensity = 0.35;
    scene.lighting.shadows.enabled = true;
    scene.generation = generation;
    scene.world_origin = Some(if far {
        viewport_lib::WorldOrigin::snapped()
    } else {
        viewport_lib::WorldOrigin::Zero
    });
    if clip {
        // Keep x <= base.x + 0.4.
        let plane = viewport_lib::ClipObject::plane([-1.0, 0.0, 0.0], base.x + 0.4);
        scene.clip = Some((vec![plane], false));
    }
    let camera = viewport_lib::Camera::orbit(
        base + DVec3::new(0.0, 0.0, 1.0),
        7.0,
        glam::Quat::from_rotation_z(0.5) * glam::Quat::from_rotation_x(1.0),
    );
    let frame = frame_for(&scene, &camera, [W as f32, H as f32]);
    let _ = harness.render(&frame, W, H);
    harness.render(&frame, W, H)
}

#[test]
fn point_light_in_a_room_far_from_origin_matches_the_world_origin() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let limit = (W * H) as usize * PIXELS_PER_MILLE / 1000;
    let mut failures = Vec::new();
    let mut generation = 30_000;
    for clip in [false, true] {
        generation += 2;
        let near = render_room(&mut harness, false, clip, generation);
        let far = render_room(&mut harness, true, clip, generation + 1);
        assert!(
            differing(&near, &vec![0u8; near.len()]) > 0,
            "the room rendered nothing"
        );
        let n = differing(&near, &far);
        eprintln!("room, clip {clip}: {n} of {} pixels differ", W * H);
        if n > limit {
            failures.push(format!("clip {clip}: {n} pixels differ (allowed {limit})"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
