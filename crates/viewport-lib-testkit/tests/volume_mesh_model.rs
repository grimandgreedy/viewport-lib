//! A transparent volume mesh follows `VolumeMeshItem::model`.
//!
//! The opaque mode draws the boundary through the mesh path, which has always
//! applied the matrix. The transparent mode draws projected tets, and its
//! wireframe overlay draws the boundary edges through a path of its own, so
//! each is checked here by moving the item and seeing the pixels move.

use viewport_lib::{
    CameraFrame, ColourmapId, FrameData, SceneFrame, VolumeMeshData, VolumeMeshItem,
    VolumeTransparency,
};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 200;
const H: u32 = 150;

fn frame(items: Vec<VolumeMeshItem>) -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 14.0, 0.0, 0.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        SceneFrame::from_surface_items(Vec::new()),
    );
    fd.scene.volume_meshes = items;
    fd.viewport.show_axes_indicator = false;
    fd.viewport.show_grid = false;
    fd
}

/// Mean column of the pixels that differ from `background`, or `None` when
/// nothing was drawn.
fn drawn_centre_x(pixels: &[u8], background: &[u8]) -> Option<f32> {
    let mut sum = 0.0;
    let mut count = 0.0;
    for (i, (a, b)) in pixels.chunks(4).zip(background.chunks(4)).enumerate() {
        if a.iter().zip(b).any(|(x, y)| x.abs_diff(*y) > 8) {
            sum += (i as u32 % W) as f32;
            count += 1.0;
        }
    }
    (count > 0.0).then(|| sum / count)
}

fn check(wireframe: bool) {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut cells = Vec::new();
    for k in 0..2u32 {
        for j in 0..2u32 {
            for i in 0..2u32 {
                cells.push([i, j, k]);
            }
        }
    }
    let mut grid = VolumeMeshData::from_grid_cells([-1.0; 3], [1.0; 3], &cells);
    grid.data
        .cell_scalars
        .insert("value".into(), (0..8).map(|i| i as f32).collect());
    let uploaded = h
        .renderer
        .resources_mut()
        .upload_volume_mesh_with_transparency(&h.device, grid.data, "value")
        .expect("volume mesh upload");

    let item_at = |x: f32| {
        let mut item = uploaded.clone();
        item.model = glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.0)).to_cols_array_2d();
        item.colourmap_id = Some(ColourmapId(0));
        item.transparency = Some(VolumeTransparency::default());
        item.settings.wireframe = wireframe;
        item
    };

    let background = h.render(&frame(Vec::new()), W, H);
    let mut centre = |x: f32| {
        // Twice: the first frame of a new configuration settles targets.
        let _ = h.render(&frame(vec![item_at(x)]), W, H);
        let pixels = h.render(&frame(vec![item_at(x)]), W, H);
        drawn_centre_x(&pixels, &background).expect("the volume drew nothing")
    };
    let middle = centre(0.0);
    let one_way = centre(3.0);
    let other_way = centre(-3.0);

    assert!(
        (middle - W as f32 / 2.0).abs() < 4.0,
        "untransformed volume is off centre: {middle}"
    );
    assert!(
        (one_way - middle).abs() > 20.0 && (other_way - middle).abs() > 20.0,
        "the volume did not move with its model: {other_way} {middle} {one_way}"
    );
    assert!(
        (one_way - middle).signum() != (other_way - middle).signum(),
        "opposite translations moved the same way: {other_way} {middle} {one_way}"
    );
}

#[test]
fn transparent_draw_follows_the_model() {
    check(false);
}

#[test]
fn transparent_wireframe_follows_the_model() {
    check(true);
}

/// Two transparent items sharing one upload are drawn with their own
/// wireframe records, not the last one written.
#[test]
fn wireframe_items_each_keep_their_own_model() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut grid = VolumeMeshData::from_grid_cells([-0.5; 3], [1.0; 3], &[[0, 0, 0]]);
    grid.data.cell_scalars.insert("value".into(), vec![1.0]);
    let uploaded = h
        .renderer
        .resources_mut()
        .upload_volume_mesh_with_transparency(&h.device, grid.data, "value")
        .expect("volume mesh upload");
    let item_at = |x: f32| {
        let mut item = uploaded.clone();
        item.model = glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.0)).to_cols_array_2d();
        item.transparency = Some(VolumeTransparency::default());
        item.settings.wireframe = true;
        item
    };
    let background = h.render(&frame(Vec::new()), W, H);
    let both = frame(vec![item_at(-3.0), item_at(3.0)]);
    let _ = h.render(&both, W, H);
    let pixels = h.render(&both, W, H);

    // Drawn pixels on both halves of the frame, and none in the middle band.
    let mut left = 0;
    let mut middle = 0;
    let mut right = 0;
    for (i, (a, b)) in pixels.chunks(4).zip(background.chunks(4)).enumerate() {
        if a.iter().zip(b).any(|(x, y)| x.abs_diff(*y) > 8) {
            match i as u32 % W {
                x if x < W * 2 / 5 => left += 1,
                x if x >= W * 3 / 5 => right += 1,
                _ => middle += 1,
            }
        }
    }
    assert!(left > 20 && right > 20, "left {left}, right {right}");
    assert_eq!(middle, 0, "something drew between the two items");
}
