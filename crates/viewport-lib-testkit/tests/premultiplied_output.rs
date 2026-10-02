//! A transparent background produces validly premultiplied pixels.
//!
//! The `transparent_background` golden catches a regression, but it cannot say
//! what the pixels are supposed to *be*: a reference that drifted to something
//! wrong would be blessed and the gate would go green again. This states the
//! invariant instead, over the whole frame, so the bless has something to be
//! checked against.
//!
//! Premultiplied means a covered pixel's colour never exceeds its own coverage.
//! Bloom is the exception, and not a narrow one: it carries colour with no
//! coverage of its own, so it can push any pixel it reaches past that bound,
//! not only the empty ones. That is correct, and under a premultiplied blend
//! (`src + dst * (1 - src.a)`) it reads as the additive glow it is. So the
//! invariant is checked with bloom off, where it holds strictly, and the glow
//! is checked separately for being there at all.

use viewport_lib_testkit::{Harness, frame_for, scene_by_name};

const W: u32 = 400;
const H: u32 = 300;

/// Decode one 0..=255 sRGB channel to linear. The target is sRGB, so the
/// readback is encoded and a comparison against linear coverage has to undo it:
/// sRGB lifts small values a long way, and comparing the encoded byte directly
/// reports violations that are not there.
fn to_linear(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[test]
fn transparent_background_output_is_premultiplied() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = scene_by_name("transparent_background").expect("scene in catalogue");
    let built = harness.build_scene(&scene);
    let mut frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    // Bloom off: additive glow is allowed to exceed coverage, so leaving it on
    // would make this assert nothing wherever the glow reaches. The golden
    // covers the scene as the catalogue renders it, bloom included.
    frame.effects.post_process.bloom.enabled = false;

    // Render twice and read the settled frame, as the snapshot gate does.
    let _ = harness.render(&frame, W, H);
    let pixels = harness.render(&frame, W, H);

    let mut empty = 0u32;
    let mut partial = 0u32;
    let mut opaque = 0u32;
    let mut worst: Option<([u8; 4], f32)> = None;

    for px in pixels.chunks_exact(4) {
        let rgba = [px[0], px[1], px[2], px[3]];
        let coverage = rgba[3] as f32 / 255.0;

        if rgba[3] == 0 {
            // Premultiplied by zero is zero. With no bloom in the frame there is
            // nothing else that can put colour at zero coverage, so a background
            // colour leaking through would show up right here.
            assert_eq!(
                [rgba[0], rgba[1], rgba[2]],
                [0, 0, 0],
                "an uncovered pixel carries colour: {rgba:?}"
            );
            empty += 1;
            continue;
        }
        if rgba[3] >= 250 {
            opaque += 1;
        } else {
            partial += 1;
        }

        // The invariant. The tolerance covers 8-bit quantisation of both the
        // colour and the coverage, which at low coverage is a whole step.
        let brightest = to_linear(rgba[0])
            .max(to_linear(rgba[1]))
            .max(to_linear(rgba[2]));
        let excess = brightest - coverage;
        if excess > 0.03 && worst.is_none_or(|(_, w)| excess > w) {
            worst = Some((rgba, excess));
        }
    }

    eprintln!("empty {empty}, partial {partial}, opaque {opaque}");
    assert!(
        worst.is_none(),
        "a covered pixel carries more colour than coverage, so the output is not \
         premultiplied: {worst:?}"
    );

    // The scene is only a gate if it actually contains all three regions. Each
    // takes a different path through the composite, and a scene that lost one
    // (a camera change that framed the transparent slab out, say) would pass the
    // invariant above while testing a third of what it is for.
    assert!(
        partial > 500,
        "expected a region of partial coverage, found {partial} pixels"
    );
    assert!(
        opaque > 500,
        "expected a region of full coverage, found {opaque} pixels"
    );
    assert!(
        empty > 10_000,
        "expected the frame to be mostly empty, found {empty} empty pixels"
    );
}

/// Bloom reaches past its own silhouette over nothing, and carries no coverage
/// there.
///
/// Colour at zero coverage is what makes a glow composite additively rather than
/// occluding the backdrop, so this is the behaviour rather than a leak. Folding
/// bloom into coverage would make a soft glow punch a hole in whatever the
/// viewport is drawn over.
#[test]
fn bloom_spills_over_the_background_without_coverage() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = scene_by_name("transparent_background").expect("scene in catalogue");
    let built = harness.build_scene(&scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    let _ = harness.render(&frame, W, H);
    let with_bloom = harness.render(&frame, W, H);

    let mut glow = 0u32;
    let mut brightest = 0u32;
    for px in with_bloom.chunks_exact(4) {
        if px[3] == 0 {
            let sum = px[0] as u32 + px[1] as u32 + px[2] as u32;
            if sum > 0 {
                glow += 1;
                brightest = brightest.max(sum);
            }
        }
    }

    eprintln!("glow pixels at zero coverage: {glow}, brightest sum {brightest}");
    assert!(
        glow > 100,
        "expected bloom spill over empty background, found {glow} pixels"
    );
    // Not just quantisation noise a stray unit of colour would also produce.
    assert!(
        brightest > 90,
        "the spill is too faint to be the glow, brightest sum {brightest}"
    );
}
