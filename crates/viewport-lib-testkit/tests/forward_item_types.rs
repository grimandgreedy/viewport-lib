//! The item types that register on the forward tier, drawn from their
//! catalogue scenes on WebGL2 limits, with goldens of their own.
//!
//! The references live beside the forward mesh goldens in
//! `tests/snapshots/<backend>-forward/`, under the scene's name, and are
//! blessed the same way:
//!
//!     BLESS=1 cargo test --test forward_item_types

use std::path::Path;
use viewport_lib::Tier;
use viewport_lib_testkit::golden::{self, Outcome, RgbaImage, Tolerance};
use viewport_lib_testkit::{DeviceProfile, Harness, catalogue, frame_for};

const W: u32 = 256;
const H: u32 = 192;

/// The catalogue scenes of the types that draw on the forward tier, with the
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
];

/// Compare against `<backend>-forward/<name>.png`, writing it when missing or
/// under `BLESS`.
fn check_golden(harness: &Harness, name: &str, pixels: Vec<u8>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/snapshots")
        .join(format!("{}-forward", harness.backend_dir_name()));
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
/// outlined, and a click on a pickable item picks it on the CPU.
#[test]
fn forward_tier_draws_the_item_types_that_accept_it() {
    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::downlevel_webgl2("forward-item-types"))
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(harness.renderer.tier(), Tier::Forward);
    harness.renderer.set_cpu_pick_cache(true);
    let scenes = catalogue();
    for (name, type_name) in SCENES {
        assert!(
            harness.renderer.has_item_type_plugin(type_name),
            "{type_name} registers on the forward tier"
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
        // against the frame just prepared.
        let n = drawn.len() as f32;
        let centre = drawn
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
        // Streamtubes carry no pick id, and the volume no CPU copy to pick.
        let pickable = !matches!(*name, "streamtubes" | "volume");
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
        }
        check_golden(&harness, name, pixels);
    }
}
