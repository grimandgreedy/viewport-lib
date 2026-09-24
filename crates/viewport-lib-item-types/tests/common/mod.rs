//! Headless fixtures shared by this crate's item-type tests.

#![allow(dead_code)]

use viewport_lib::wgpu;

// Re-export the library types the test files reach for, so a single
// `use common::*;` covers the common set. Unused names from a glob import do
// not warn, so each file only pays for what it actually references.
pub use viewport_lib::{
    Camera, PickBackend, PickId, PickMask,
    renderer::{FrameData, RenderCamera, ViewportRenderer},
    resources::MeshData,
};

use viewport_lib_testkit::{DeviceProfile, headless_device_with};

/// A bare 64x64 frame with the grid and axes indicator off: the starting point
/// for pick tests, which want nothing in the scene but the item under test.
pub fn sub_object_pick_frame() -> FrameData {
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
    frame
}

/// Create a headless wgpu device + queue for testing.
pub fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(&DeviceProfile::low_power("test"))
}

/// A unit cube, the stand-in geometry for a draw that only has to put pixels
/// on the screen.
pub fn box_mesh() -> MeshData {
    let positions = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let normals = vec![
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    let indices = vec![
        0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 3, 7, 7, 4, 0, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7, 3,
        0, 4, 5, 5, 1, 0,
    ];
    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}
