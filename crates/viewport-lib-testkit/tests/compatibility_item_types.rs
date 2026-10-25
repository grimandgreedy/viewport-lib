//! The item types that register on the compatibility tier, drawn from their
//! catalogue scenes on WebGL2 limits, with goldens of their own.
//!
//! The references live beside the compatibility mesh goldens in
//! `tests/snapshots/<backend>-compatibility/`, under the scene's name, and are
//! blessed the same way:
//!
//!     BLESS=1 cargo test --test compatibility_item_types

use std::path::Path;
use viewport_lib::Tier;
use viewport_lib_testkit::golden::{self, Outcome, RgbaImage, Tolerance};
use viewport_lib_testkit::{DeviceProfile, Harness, catalogue, frame_for};

const W: u32 = 256;
const H: u32 = 192;

/// The catalogue scenes of the types that draw on the compatibility tier, with the
/// type each one exercises.
const SCENES: &[(&str, &str)] = &[
    (
        "gpu_implicit",
        viewport_lib_plugins::item_types::gpu_implicit::TYPE_NAME,
    ),
    (
        "volume",
        viewport_lib_plugins::item_types::volume::TYPE_NAME,
    ),
    (
        "tubes",
        viewport_lib_plugins::item_types::curves::TUBE_TYPE_NAME,
    ),
    (
        "streamtubes",
        viewport_lib_plugins::item_types::curves::STREAMTUBE_TYPE_NAME,
    ),
    (
        "ribbons",
        viewport_lib_plugins::item_types::curves::RIBBON_TYPE_NAME,
    ),
    (
        "point_cloud",
        viewport_lib_plugins::item_types::point_cloud::TYPE_NAME,
    ),
    (
        "vector_fields",
        viewport_lib_plugins::item_types::vector_field::TYPE_NAME,
    ),
    (
        "gpu_marching_cubes",
        viewport_lib_plugins::item_types::gpu_marching_cubes::TYPE_NAME,
    ),
];

/// Compare against `<backend>-compatibility/<name>.png`, writing it when missing or
/// under `BLESS`.
fn check_golden(harness: &Harness, name: &str, pixels: Vec<u8>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/snapshots")
        .join(format!("{}-compatibility", harness.backend_dir_name()));
    std::fs::create_dir_all(&dir).expect("create snapshot dir");
    let tolerances = golden::load_tolerances(&dir.join("tolerances.txt"));
    let tolerance = tolerances.get(name).copied().unwrap_or(Tolerance::EXACT);
    let actual = RgbaImage::from_raw(W, H, pixels).expect("frame -> image");
    let bless = std::env::var_os("BLESS").is_some();
    match golden::check(&dir, name, &actual, tolerance, bless) {
        Outcome::Blessed => eprintln!("generated {name} in {}", dir.display()),
        Outcome::Match => {}
        Outcome::Mismatch { comparison, .. } => panic!(
            "{name}: max_channel_delta={} pixels_over={}/{}",
            comparison.max_channel_delta, comparison.pixels_over, comparison.total_pixels
        ),
    }
}

/// Each scene's items reach their plugin and draw: none is skipped, the frame
/// differs from one with the type's items left out, the selected item is
/// outlined, and a click or a box over a pickable item picks it on the CPU.
#[test]
fn compatibility_tier_draws_the_item_types_that_accept_it() {
    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::downlevel_webgl2("compatibility-item-types"))
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(harness.renderer.tier(), Tier::Compatibility);
    harness.renderer.set_cpu_pick_cache(true);
    let scenes = catalogue();
    for (name, type_name) in SCENES {
        assert!(
            harness.renderer.has_item_type_plugin(type_name),
            "{type_name} registers on the compatibility tier"
        );
        let scene = scenes
            .iter()
            .find(|s| s.name == *name)
            .unwrap_or_else(|| panic!("{name} is not in the catalogue"));
        let built = harness.build_scene(scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
        let items: usize = frame
            .scene
            .plugin_items
            .get(*type_name)
            .map(|c| c.iter().map(|c| c.len()).sum())
            .unwrap_or(0);
        assert!(items > 0, "{name} submits {type_name} items");

        // Each render runs twice: the first frame may still be compiling
        // pipelines.
        let mut render = |frame: &viewport_lib::FrameData| {
            let _ = harness.render(frame, W, H);
            harness.render(frame, W, H)
        };
        let mut without = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
        without.scene.plugin_items.remove(*type_name);
        without.interaction.outline_selected = false;
        let bare = render(&without);
        let mut plain = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
        plain.interaction.outline_selected = false;
        let unoutlined = render(&plain);
        let pixels = render(&frame);

        let differing = |a: &[u8], b: &[u8]| {
            a.chunks_exact(4)
                .zip(b.chunks_exact(4))
                .enumerate()
                .filter(|(_, (a, b))| a[..3] != b[..3])
                .map(|(i, _)| ((i as u32 % W) as f32 + 0.5, (i as u32 / W) as f32 + 0.5))
                .collect::<Vec<_>>()
        };
        let drawn = differing(&unoutlined, &bare);
        assert!(
            drawn.len() > (W * H / 50) as usize,
            "{name}: the {type_name} items must draw, {} pixels changed",
            drawn.len()
        );
        // Every one of these scenes selects one of its items.
        let ring = differing(&pixels, &unoutlined);
        assert!(
            ring.len() > 40,
            "{name}: the selected item must be outlined, {} pixels changed",
            ring.len()
        );

        let stats = harness.stats();
        let skippable = frame.scene.volume_meshes.len()
            + frame.scene.mesh_instances.len()
            + frame.scene.foreground_items.len()
            + frame
                .scene
                .plugin_items
                .iter()
                .filter(|(n, _)| !harness.renderer.has_item_type_plugin(n))
                .flat_map(|(_, c)| c)
                .map(|c| c.len())
                .sum::<usize>();
        assert_eq!(
            stats.items_skipped_by_tier as usize, skippable,
            "{name}: only items of unregistered types are skipped"
        );

        // Click the drawn pixel nearest the middle of what the type drew,
        // against the frame just prepared. The point cloud scene's large
        // cloud is not pickable, so there aim at the selected one, the middle
        // of its outline.
        let target = if name == &"point_cloud" {
            &ring
        } else {
            &drawn
        };
        let n = target.len() as f32;
        let centre = target
            .iter()
            .fold((0.0, 0.0), |(x, y), p| (x + p.0 / n, y + p.1 / n));
        let cursor = drawn
            .iter()
            .copied()
            .min_by(|a, b| {
                let d = |p: (f32, f32)| (p.0 - centre.0).powi(2) + (p.1 - centre.1).powi(2);
                d(*a).total_cmp(&d(*b))
            })
            .expect("drawn pixels");
        let pickable = name != &"streamtubes";
        let device = harness.device.clone();
        let queue = harness.queue.clone();
        let hit = harness.renderer.pick_object(
            viewport_lib::PickBackend::Auto,
            glam::Vec2::new(cursor.0, cursor.1),
            &frame,
            &device,
            &queue,
            viewport_lib::PickMask::OBJECT,
        );
        if pickable {
            let hit = hit.unwrap_or_else(|| panic!("{name}: a click at {cursor:?} picks nothing"));
            assert_ne!(hit.id, 0, "{name}");
            if name == &"point_cloud" {
                // The point under the cursor, as PlotScope reads a sample from.
                let point = harness.renderer.pick_object(
                    viewport_lib::PickBackend::Gpu,
                    glam::Vec2::new(cursor.0, cursor.1),
                    &frame,
                    &device,
                    &queue,
                    viewport_lib::PickMask::CLOUD_POINT,
                );
                assert!(
                    matches!(
                        point.and_then(|h| h.sub_object),
                        Some(viewport_lib::SubObjectRef::Point(_))
                    ),
                    "point_cloud: {point:?}"
                );
            }
            if name == &"volume" {
                // The cell under the cursor, as PlotScope reads a value from.
                let voxel = harness.renderer.pick_object(
                    viewport_lib::PickBackend::Gpu,
                    glam::Vec2::new(cursor.0, cursor.1),
                    &frame,
                    &device,
                    &queue,
                    viewport_lib::PickMask::VOXEL,
                );
                assert!(
                    matches!(
                        voxel.and_then(|h| h.sub_object),
                        Some(viewport_lib::SubObjectRef::Voxel(_))
                    ),
                    "volume: {voxel:?}"
                );
            }
            // A box over everything the type drew selects the clicked item.
            let (min, max) = drawn.iter().fold(
                (glam::Vec2::splat(f32::MAX), glam::Vec2::splat(f32::MIN)),
                |(lo, hi), p| {
                    let p = glam::Vec2::new(p.0, p.1);
                    (lo.min(p), hi.max(p))
                },
            );
            let rect = harness.renderer.pick_rect_objects(
                viewport_lib::PickBackend::Gpu,
                min,
                max,
                &frame,
                &device,
                &queue,
                viewport_lib::PickMask::OBJECT,
            );
            assert!(
                rect.objects.contains(&hit.id),
                "{name}: the box selects {:?}, not {}",
                rect.objects,
                hit.id
            );
        }
        check_golden(&harness, name, pixels);
    }
}

/// On WebGL2 limits a point cloud's channels are vertex streams, one pipeline
/// per channel set. Every set draws, and each channel shows: per-point colours
/// colour the points, scalars spread across the colourmap, per-point sizes
/// grow the discs and per-point transparency dims them.
#[test]
fn compatibility_point_clouds_draw_every_channel_set() {
    use viewport_lib::{Colour, ColourSource, SizeSource};
    use viewport_lib_plugins::item_types::point_cloud::PointCloudItem;
    use viewport_lib_testkit::BuiltScene;

    let Some(mut harness) = Harness::with_profile(&DeviceProfile::downlevel_webgl2(
        "compatibility-point-channels",
    )) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scenes = catalogue();
    let camera = scenes
        .iter()
        .find(|s| s.name == "point_cloud")
        .expect("point_cloud is in the catalogue")
        .cameras[0]
        .camera
        .clone();
    let positions: Vec<[f32; 3]> = (0..25)
        .map(|i| [(i % 5) as f32 * 0.6 - 1.2, 0.0, (i / 5) as f32 * 0.6 - 1.2])
        .collect();
    let n = positions.len();
    let mut render = |clouds: Vec<PointCloudItem>| {
        let built = BuiltScene {
            point_clouds: clouds,
            ..Default::default()
        };
        let frame = frame_for(&built, &camera, [W as f32, H as f32]);
        let _ = harness.render(&frame, W, H);
        harness.render(&frame, W, H)
    };
    let bare = render(Vec::new());
    // The pixels a frame changed against the empty one, as RGB.
    let drawn = |pixels: &[u8]| -> Vec<[u8; 3]> {
        pixels
            .chunks_exact(4)
            .zip(bare.chunks_exact(4))
            .filter(|(a, b)| a[..3] != b[..3])
            .map(|(a, _)| [a[0], a[1], a[2]])
            .collect()
    };
    // Mean distance from the background over the changed pixels.
    let contrast = |pixels: &[u8]| {
        let (sum, count) = pixels
            .chunks_exact(4)
            .zip(bare.chunks_exact(4))
            .filter(|(a, b)| a[..3] != b[..3])
            .fold((0u32, 0u32), |(sum, count), (a, b)| {
                let d: u32 = (0..3).map(|c| a[c].abs_diff(b[c]) as u32).sum();
                (sum + d, count + 1)
            });
        sum as f32 / count.max(1) as f32
    };

    for colour_mode in 0..3 {
        for per_point_size in [false, true] {
            for per_point_alpha in [false, true] {
                let mut cloud = PointCloudItem::default();
                cloud.positions = positions.clone();
                cloud.colour = match colour_mode {
                    0 => ColourSource::Solid(Colour::WHITE),
                    1 => ColourSource::Scalar {
                        values: (0..n).map(|i| i as f32).collect(),
                        range: None,
                        colourmap: None,
                    },
                    _ => ColourSource::PerSample(vec![Colour::srgb_rgb(1.0, 0.0, 0.0); n]),
                };
                cloud.size = if per_point_size {
                    SizeSource::PerSample(vec![14.0; n])
                } else {
                    SizeSource::Uniform(6.0)
                };
                if per_point_alpha {
                    cloud.transparencies = vec![0.3; n];
                }
                let label = format!(
                    "colour {colour_mode}, per-point size {per_point_size}, per-point alpha {per_point_alpha}"
                );
                let pixels = render(vec![cloud.clone()]);
                let px = drawn(&pixels);
                assert!(px.len() > n * 4, "{label}: {} pixels drawn", px.len());

                match colour_mode {
                    1 => {
                        let distinct: std::collections::HashSet<_> = px.iter().collect();
                        assert!(distinct.len() > 5, "{label}: scalars are colourmapped");
                    }
                    2 => assert!(
                        px.iter().all(|p| p[0] > p[1] && p[0] > p[2]),
                        "{label}: per-point colours are red"
                    ),
                    _ => {}
                }
                if per_point_size {
                    let mut small = cloud.clone();
                    small.size = SizeSource::Uniform(6.0);
                    let small_px = drawn(&render(vec![small]));
                    assert!(
                        px.len() > small_px.len() * 3,
                        "{label}: per-point sizes grow the discs, {} vs {}",
                        px.len(),
                        small_px.len()
                    );
                }
                if per_point_alpha {
                    let mut opaque = cloud.clone();
                    opaque.transparencies.clear();
                    let opaque_pixels = render(vec![opaque]);
                    assert!(
                        contrast(&pixels) < contrast(&opaque_pixels) * 0.6,
                        "{label}: per-point transparency dims the points"
                    );
                }
            }
        }
    }
}

/// Below the full tier a marching cubes item is ray-marched through its field,
/// so a new isovalue redraws the surface with no upload: the resident bytes
/// stay put and the surface moves.
#[test]
fn compatibility_marching_cubes_follows_the_isovalue() {
    use viewport_lib_plugins::item_types::gpu_marching_cubes::GpuMarchingCubesItem;

    let Some(mut harness) = Harness::with_profile(&DeviceProfile::downlevel_webgl2(
        "compatibility-mc-isovalue",
    )) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scenes = catalogue();
    let scene = scenes
        .iter()
        .find(|s| s.name == "gpu_marching_cubes")
        .expect("gpu_marching_cubes is in the catalogue");
    let mut built = harness.build_scene(scene);
    let mut render = |harness: &mut Harness, built: &viewport_lib_testkit::BuiltScene| {
        let frame = frame_for(built, &scene.cameras[0].camera, [W as f32, H as f32]);
        let _ = harness.render(&frame, W, H);
        harness.render(&frame, W, H)
    };
    let resident = |harness: &Harness| harness.renderer.resident_bytes().plugin_bytes;
    let at_zero = render(&mut harness, &built);
    let bytes = resident(&harness);
    assert!(
        bytes >= 32 * 32 * 32 * 4,
        "the field is charged, {bytes} bytes"
    );
    let item: &mut GpuMarchingCubesItem = &mut built.gpu_mc_items[0];
    item.isovalue = 0.8;
    let at_high = render(&mut harness, &built);
    assert_eq!(resident(&harness), bytes, "no upload for a new isovalue");
    let changed = at_zero
        .chunks_exact(4)
        .zip(at_high.chunks_exact(4))
        .filter(|(a, b)| a[..3] != b[..3])
        .count();
    assert!(
        changed > (W * H / 20) as usize,
        "the surface moves with the isovalue, {changed} pixels changed"
    );
}
