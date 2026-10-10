//! The image slice item type: pipeline warm-up.
//!
//! One file per item type, so a type's coverage travels with it.

#![cfg(feature = "item-types")]

mod common;
use common::*;
use viewport_lib_plugins::item_types::image_slice::{ImageSliceItem, ImageSlicePlugin, SliceAxis};

/// Naming the image slice type in a warm-up builds its pipelines, so the first
/// frame that draws, outlines and picks a slice, in either format, compiles
/// none of them.
#[test]
fn a_warmed_image_slice_type_builds_nothing_on_its_first_frame() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let volume_id = renderer
        .resources_mut()
        .upload_volume(&device, &queue, &[0.5; 8], [2, 2, 2]);
    renderer.warm_pipelines(
        &device,
        &queue,
        &viewport_lib::PipelineSet::default().with_item_type::<ImageSlicePlugin>(),
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
        let mut slice = ImageSliceItem::default();
        slice.volume_id = volume_id;
        slice.axis = SliceAxis::Z;
        slice.bbox_min = [-1.0, -1.0, -1.0];
        slice.bbox_max = [1.0, 1.0, 1.0];
        slice.settings.pick_id = PickId(222);
        slice.settings.selected = true;
        frame.scene.items_mut::<ImageSliceItem>().push(slice);
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
    let builds: Vec<String> = viewport_lib::resources::build_log::drain()
        .into_iter()
        .map(|(label, _)| label)
        .filter(|l| l.starts_with("image_slice") || l.starts_with("module image_slice"))
        .collect();
    assert!(
        builds.is_empty(),
        "the first image slice frames built pipelines after the warm-up: {builds:?}"
    );
}
