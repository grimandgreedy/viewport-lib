//! The item types that register on the compatibility tier, drawn from their
//! catalogue scenes on WebGL2 limits, with goldens of their own.
//!
//! The references live beside the compatibility mesh goldens in
//! `tests/snapshots/<backend>-compatibility/`, under the scene's name, and are
//! blessed the same way:
//!
//!     BLESS=1 cargo test --release --test compatibility_item_types

use std::path::Path;
use viewport_lib::Tier;
use viewport_lib_testkit::golden::{self, GoldenMode, Outcome, RgbaImage, Tolerance};
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
    (
        "gaussian_splats",
        viewport_lib_plugins::item_types::gaussian_splat::TYPE_NAME,
    ),
    // Polylines are core, registered by the renderer itself.
    ("polyline", "vpl.polyline"),
];

/// Check against `<adapter>-compatibility/<name>.png` in the mode the
/// environment names (see `golden::GoldenMode::from_env`).
fn check_golden(harness: &Harness, name: &str, pixels: Vec<u8>) {
    let snapshots = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots");
    let dir = snapshots.join(format!("{}-compatibility", harness.backend_dir_name()));
    let mode = GoldenMode::from_env();
    if mode != GoldenMode::Bless {
        // The driver stamp lives with the adapter's main reference set.
        let stamp_dir = snapshots.join(harness.backend_dir_name());
        if let Err(e) = golden::check_driver_stamp(&stamp_dir, harness.driver_info().unwrap_or(""))
        {
            panic!("{e}");
        }
    }
    let tolerances = golden::load_tolerances(&dir.join("tolerances.txt"));
    let tolerance = tolerances.get(name).copied().unwrap_or(Tolerance::EXACT);
    let actual = RgbaImage::from_raw(W, H, pixels).expect("frame -> image");
    match golden::check(&dir, name, &actual, tolerance, mode) {
        Outcome::Blessed => eprintln!("blessed {name} in {}", dir.display()),
        Outcome::Rendered { path } => eprintln!("render only: {} (not compared)", path.display()),
        Outcome::Match => {}
        Outcome::Missing { .. } => panic!("{}", golden::missing_message(name, &dir)),
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
            if name == &"gaussian_splats" {
                let splat = harness.renderer.pick_object(
                    viewport_lib::PickBackend::Gpu,
                    glam::Vec2::new(cursor.0, cursor.1),
                    &frame,
                    &device,
                    &queue,
                    viewport_lib::PickMask::SPLAT,
                );
                assert!(
                    matches!(
                        splat.and_then(|h| h.sub_object),
                        Some(viewport_lib::SubObjectRef::Splat(_))
                    ),
                    "gaussian_splats: {splat:?}"
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
    let render = |harness: &mut Harness, built: &viewport_lib_testkit::BuiltScene| {
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

/// Below the full tier a splat set is packed into data textures and sorted
/// back to front on the CPU. Two overlapping splats swap which one shows in
/// front when the eye moves to the other side, a ranged write that swaps
/// them is repacked and re-sorted, and a degree-1 set's colour follows the
/// view direction.
#[test]
fn compatibility_splats_sort_for_the_eye_and_follow_writes() {
    use viewport_lib::plugin_api::{Uploads, Writes};
    use viewport_lib_plugins::item_types::gaussian_splat::{
        GaussianSplatData, GaussianSplatItem, ShDegree, channels,
    };

    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::downlevel_webgl2("compatibility-splat-sort"))
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    const SH0: f32 = 0.282_094_79;
    let dc = |c: [f32; 3]| c.map(|v| (v - 0.5) / SH0);
    let device = harness.device.clone();
    let queue = harness.queue.clone();

    // The centre pixel's colour from an eye on one side of the origin or the
    // other, along the camera's own axis.
    let view = |yaw: f32| {
        viewport_lib::Camera::orbit(
            glam::DVec3::ZERO,
            4.0,
            glam::Quat::from_rotation_z(yaw) * glam::Quat::from_rotation_x(1.5708),
        )
    };
    let centre = |harness: &mut Harness, item: &GaussianSplatItem, yaw: f32| {
        let built = viewport_lib_testkit::BuiltScene {
            gaussian_splats: vec![item.clone()],
            ..Default::default()
        };
        let frame = frame_for(&built, &view(yaw), [W as f32, H as f32]);
        let _ = harness.render(&frame, W, H);
        let pixels = harness.render(&frame, W, H);
        let at = ((H / 2 * W + W / 2) * 4) as usize;
        [
            pixels[at] as i32,
            pixels[at + 1] as i32,
            pixels[at + 2] as i32,
        ]
    };
    let reddish = |c: [i32; 3]| c[0] > c[2];

    let mut pair = GaussianSplatData::default();
    pair.positions = vec![
        [0.0, -0.3, 0.0],
        [0.0, 0.3, 0.0],
        [-0.3, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ];
    pair.scales = vec![[0.45; 3]; 4];
    pair.rotations = vec![[0.0, 0.0, 0.0, 1.0]; 4];
    pair.opacities = vec![0.97; 4];
    // Red on the -y and -x sides, blue on the +y and +x sides, so whichever
    // horizontal axis the camera looks along, one colour is nearer.
    for c in [
        [1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0],
        [1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0],
    ] {
        pair.sh_coefficients.extend(dc(c));
    }
    let id = harness
        .renderer
        .upload(&device, &queue, &pair)
        .expect("upload");
    let mut item = GaussianSplatItem::default();
    item.source = id;

    let near_side = centre(&mut harness, &item, 0.0);
    let far_side = centre(&mut harness, &item, std::f32::consts::PI);
    assert_ne!(
        reddish(near_side),
        reddish(far_side),
        "the nearer splat shows from each side: {near_side:?} vs {far_side:?}"
    );

    // Swap every pair's centres: the same eye now sees the other colour.
    harness
        .renderer
        .write_range(
            channels::Positions,
            &queue,
            id,
            0,
            &[
                [0.0, 0.3, 0.0],
                [0.0, -0.3, 0.0],
                [0.3, 0.0, 0.0],
                [-0.3, 0.0, 0.0],
            ],
        )
        .expect("write");
    let swapped = centre(&mut harness, &item, 0.0);
    assert_ne!(
        reddish(swapped),
        reddish(near_side),
        "a write is repacked and re-sorted: {swapped:?} vs {near_side:?}"
    );

    // Degree 1: grey DC, and view-dependent terms on y and x that swing red
    // to blue as the view direction flips.
    let mut lit = GaussianSplatData::default();
    lit.positions = vec![[0.0; 3]];
    lit.scales = vec![[0.6; 3]];
    lit.rotations = vec![[0.0, 0.0, 0.0, 1.0]];
    lit.opacities = vec![0.97];
    lit.sh_degree = ShDegree::One;
    let swing = [2.0, 0.0, -2.0];
    lit.sh_coefficients = [[0.0; 3], swing, [0.0; 3], swing].concat();
    let lit_id = harness
        .renderer
        .upload(&device, &queue, &lit)
        .expect("upload");
    item.source = lit_id;
    let one_way = centre(&mut harness, &item, 0.0);
    let other_way = centre(&mut harness, &item, std::f32::consts::PI);
    assert_ne!(
        reddish(one_way),
        reddish(other_way),
        "degree-1 colour follows the view: {one_way:?} vs {other_way:?}"
    );
}

/// The line substrate draws below the full tier: an item type's wireframe
/// overlay (a splat set's rings, uploaded through the substrate each frame)
/// and a polyline drawn as thin wireframe lines through the LineList
/// pipeline.
#[test]
fn compatibility_tier_draws_wireframe_overlays_and_thin_lines() {
    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::downlevel_webgl2("compatibility-wireframes"))
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scenes = catalogue();
    let changed = |a: &[u8], b: &[u8]| {
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .filter(|(a, b)| a[..3] != b[..3])
            .count()
    };
    let render = |harness: &mut Harness,
                  name: &str,
                  edit: &dyn Fn(&mut viewport_lib_testkit::BuiltScene)| {
        let scene = scenes
            .iter()
            .find(|s| s.name == name)
            .expect("in the catalogue");
        let mut built = harness.build_scene(scene);
        edit(&mut built);
        let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
        let _ = harness.render(&frame, W, H);
        harness.render(&frame, W, H)
    };

    let empty = render(&mut harness, "gaussian_splats", &|b| {
        b.gaussian_splats.clear()
    });
    let rings = render(&mut harness, "gaussian_splats", &|b| {
        for item in &mut b.gaussian_splats {
            item.settings.wireframe = true;
            item.settings.selected = false;
        }
    });
    let ring_pixels = changed(&empty, &rings);
    assert!(
        ring_pixels > 200,
        "the splat rings draw through the line substrate, {ring_pixels} pixels"
    );

    let blank = render(&mut harness, "polyline", &|b| b.polylines.clear());
    let thin = render(&mut harness, "polyline", &|b| {
        b.polylines.truncate(1);
        b.polylines[0].settings.wireframe = true;
    });
    let thick = render(&mut harness, "polyline", &|b| b.polylines.truncate(1));
    let (thin_pixels, thick_pixels) = (changed(&blank, &thin), changed(&blank, &thick));
    assert!(
        thin_pixels > 200,
        "thin wireframe lines draw, {thin_pixels} pixels"
    );
    assert!(
        thin_pixels * 2 < thick_pixels,
        "the wireframe pipeline draws 1px lines: {thin_pixels} vs {thick_pixels} pixels"
    );
}
