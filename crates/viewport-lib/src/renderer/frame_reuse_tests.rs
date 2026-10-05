//! A frame that repeats the last one reuses the buffers and bind groups the
//! last one built: the overlay storage buffers and their bind groups, the
//! outline mask and x-ray bindings, and the tone-map bind group. Bind groups
//! compare by identity, so an equal handle means nothing was rebuilt.

use crate::renderer::{FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission};
use crate::{Camera, LabelItem, OverlayFill, OverlayShape, OverlayShapeItem, ViewportRenderer};

const SIZE: u32 = 64;

/// The handles a frame bound, read back after it rendered.
#[derive(PartialEq, Debug)]
struct Bound {
    label: crate::gpu::BindGroup,
    shape_shadow: crate::gpu::BindGroup,
    outline: crate::gpu::BindGroup,
    xray: crate::gpu::BindGroup,
    tone_map: crate::gpu::BindGroup,
}

fn bound(renderer: &ViewportRenderer) -> Bound {
    let slot = &renderer.viewport_slots[0];
    Bound {
        label: renderer.label_gpu_data.as_ref().unwrap().bind_group.clone(),
        shape_shadow: renderer
            .overlay_shape_gpu_data
            .as_ref()
            .and_then(|d| d.shadow_bind_group.clone())
            .unwrap(),
        outline: slot.selection_outlines.outline_object_buffers[0]
            .mask
            .bind_group
            .clone(),
        xray: slot.xray_object_buffers[0].1.bind_group.clone(),
        tone_map: slot.hdr.as_ref().unwrap().tone_map_bind_group.clone(),
    }
}

#[test]
fn a_repeated_frame_rebuilds_no_bind_group() {
    let Some((device, queue)) = crate::resources::test_support::try_make_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_pipeline_compilation(crate::PipelineCompilation::Blocking);
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &crate::geometry::primitives::cube(1.0))
        .unwrap();

    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.camera.pixels_per_point = 1.0;
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.settings.selected = true;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    frame.interaction.outline_selected = true;
    frame.interaction.xray_selected = true;
    frame.overlays.shapes = vec![
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 2.0 },
            [4.0, 4.0],
            [10.0, 6.0],
        )
        .with_fill(OverlayFill::Solid([0.8, 0.2, 0.2, 1.0].into())),
    ];
    frame.overlays.labels = vec![LabelItem::new("label").with_position([8.0, 20.0])];

    // The first frames build and grow; after that nothing should change.
    for _ in 0..2 {
        let _ = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    }
    let first = bound(&renderer);
    let _ = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert_eq!(bound(&renderer), first);
}
