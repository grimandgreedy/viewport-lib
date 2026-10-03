//! The decal item type: its pipeline warm-up.
//!
//! One file per item type, so a type's coverage travels with it.

mod common;
use common::*;
use viewport_lib_plugins::item_types::decal::{DecalBlendMode, DecalItem, DecalPlugin};

/// Naming the decal type in a warm-up builds its pipelines, so the first frame
/// that projects, outlines and picks decals compiles none of them.
#[test]
fn a_warmed_decal_type_builds_nothing_on_its_first_frame() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.warm_pipelines(
        &device,
        &queue,
        &viewport_lib::PipelineSet::default().with_item_type::<DecalPlugin>(),
    );
    renderer.wait_for_pipelines(&device);

    viewport_lib::resources::build_log::enable();
    let _ = viewport_lib::resources::build_log::drain();
    for hdr in [true, false] {
        let mut frame = sub_object_pick_frame();
        if !hdr {
            frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
        }
        frame.interaction.outline_selected = true;
        let decals = [
            DecalBlendMode::Replace,
            DecalBlendMode::Multiply,
            DecalBlendMode::Additive,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, blend_mode)| {
            let mut decal = DecalItem::default();
            decal.blend_mode = blend_mode;
            decal.settings.pick_id = PickId(i as u64 + 1);
            decal.settings.selected = true;
            decal
        });
        frame.scene.items_mut::<DecalItem>().extend(decals);
        let _ = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
        let _ = renderer.pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::OBJECT,
        );
    }
    let decal_builds: Vec<String> = viewport_lib::resources::build_log::drain()
        .into_iter()
        .map(|(label, _)| label)
        .filter(|l| l.starts_with("decal") || l.starts_with("module decal"))
        .collect();
    assert!(
        decal_builds.is_empty(),
        "the first decal frames built pipelines after the warm-up: {decal_builds:?}"
    );
}
