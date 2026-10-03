//! GPU-pick and draw coverage for the item types in this crate that a single
//! per-type file does not already cover.

use viewport_lib::gpu;
use viewport_lib::plugin_api::Uploads;
use viewport_lib_plugins::item_types::{
    decal::DecalItem,
    external_instances::{
        ExternalInstanceSetConfig, ExternalInstanceUploads, ExternalInstancesItem,
    },
    gaussian_splat::{GaussianSplatData, GaussianSplatItem, ShDegree},
    image_slice::{ImageSliceItem, SliceAxis},
    sprite::{SpriteItem, SpriteSizeMode},
};

mod common;
use common::*;

// ---------------------------------------------------------------------------
// GPU pick: Gaussian splats, image slices and volume surface slices
// ---------------------------------------------------------------------------

#[test]
fn gpu_pick_splat_resolves_splat() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    // Three splats spread along X, large enough to cover the centre pixel.
    // The centre splat (index 1) sits at world origin, under the cursor.
    let mut data = GaussianSplatData::default();
    data.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    data.scales = vec![[0.5, 0.5, 0.5]; 3];
    data.rotations = vec![[0.0, 0.0, 0.0, 1.0]; 3];
    data.opacities = vec![1.0; 3];
    data.sh_coefficients = vec![0.0; 9];
    data.sh_degree = ShDegree::Zero;
    let splat_id = renderer
        .upload(&device, &queue, &data)
        .expect("upload splat set");

    let mut item = GaussianSplatItem::default();
    item.source = splat_id;
    item.settings.pick_id = PickId(777);
    frame.scene.items_mut::<GaussianSplatItem>().push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::SPLAT,
    );
    let hit = hit.expect("centre splat should be hit");
    assert_eq!(hit.id, 777);
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Splat(1)));
}

#[test]
fn gpu_pick_hits_image_slice() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    let volume_id = renderer
        .resources_mut()
        .upload_volume(&device, &queue, &[0.5; 8], [2, 2, 2]);

    let mut slice = ImageSliceItem::default();
    slice.volume_id = volume_id;
    slice.axis = SliceAxis::Z;
    slice.offset = 0.5;
    slice.bbox_min = [-1.0, -1.0, -1.0];
    slice.bbox_max = [1.0, 1.0, 1.0];
    slice.settings.pick_id = PickId(222);
    frame.scene.items_mut::<ImageSliceItem>().push(slice);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), Some(222));
}

/// An external instance set must draw exactly the window of the consumer's
/// positions buffer selected by the item's instance range. The buffer holds
/// four positions: elements 0..2 far behind the camera, elements 2..4 in
/// front of it. Drawing `first_instance = 2, instance_count = 2` must show
/// geometry; re-pointing the range at 0..2 must show none.
#[test]
fn external_instances_render_with_instance_range_slice() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    // Elements 0 and 1 behind the camera, 2 and 3 visible near the origin.
    let positions: Vec<f32> = vec![
        0.0, 0.0, -1000.0, // 0
        0.0, 0.0, -1000.0, // 1
        -0.6, 0.0, 0.0, // 2
        0.6, 0.0, 0.0, // 3
    ];
    let pos_buf = device.create_buffer(&gpu::BufferDescriptor {
        label: Some("test_external_positions"),
        size: (positions.len() * std::mem::size_of::<f32>()) as u64,
        usage: gpu::BufferUsages::STORAGE | gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&positions));

    let set_id = renderer
        .create_external_instance_set(&device, &ExternalInstanceSetConfig::new(mesh_id, pos_buf))
        .unwrap();

    let make_frame = |first: u32, count: u32| -> FrameData {
        let cam = Camera::default();
        let mut frame = FrameData::default();
        frame.camera.render_camera = {
            let mut rc = RenderCamera::from_camera(&cam);
            rc.aspect = 1.0;
            rc
        };
        frame.camera.viewport_size = [64.0, 64.0];
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        let mut item = ExternalInstancesItem::new(set_id, count);
        item.first_instance = first;
        item.scale = 0.4;
        item.colour = [1.0, 0.2, 0.2, 1.0].into();
        *frame.scene.items_mut::<ExternalInstancesItem>() = vec![item];
        frame
    };

    let empty = renderer.render_offscreen(&device, &queue, &make_frame(0, 0), 64, 64);
    let visible = renderer.render_offscreen(&device, &queue, &make_frame(2, 2), 64, 64);
    let hidden = renderer.render_offscreen(&device, &queue, &make_frame(0, 2), 64, 64);

    let diff_count = |a: &[u8], b: &[u8]| -> usize {
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .filter(|(pa, pb)| pa.iter().zip(pb.iter()).any(|(&x, &y)| x.abs_diff(y) > 8))
            .count()
    };

    assert!(
        diff_count(&visible, &empty) > 0,
        "instance range 2..4 selects the visible elements; the boxes must \
         render. If nothing shows, instance_index is not honouring the draw \
         call's first_instance or the storage window is wrong.",
    );
    assert_eq!(
        diff_count(&hidden, &empty),
        0,
        "instance range 0..2 selects only the behind-camera elements; the \
         image must match an empty scene.",
    );
}

/// A world-space sprite billboard resolves to its pick id. The billboard is
/// expanded in the vertex stage that the pick pipeline reuses, so prepare has
/// to run before the pick.
#[test]
fn gpu_pick_hits_sprite_set() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    let mut sprite = SpriteItem::default();
    sprite.positions = vec![[0.0, 0.0, 0.0]];
    sprite.default_size = 4.0;
    sprite.size_mode = SpriteSizeMode::WorldSpace;
    sprite.settings.pick_id = PickId(777);
    frame.scene.items_mut::<SpriteItem>().push(sprite);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(777)));
}

// ---------------------------------------------------------------------------
// GPU pick: decals
// ---------------------------------------------------------------------------

#[test]
fn gpu_pick_hits_decal_box() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    // A decal is the unit box [-0.5, 0.5]^3 mapped by `transform`; the default
    // transform places it at the origin. The decal item type rasterises that
    // box in the pick pass and reads back its pick_id.
    let mut decal = DecalItem::default();
    decal.settings.pick_id = PickId(77);
    frame.scene.items_mut::<DecalItem>().push(decal);

    // The decal's pick binding is built during prepare, like every other item
    // type that answers the id pass with geometry of its own.
    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), Some(77));
}
