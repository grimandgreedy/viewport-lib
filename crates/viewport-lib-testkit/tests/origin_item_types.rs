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
