//! Two OS windows driven by one runner, each with its own scene and camera.
//!
//! `ViewportAppV2` owns both windows, the shared wgpu device, and the event loop.
//! Each window drives its own `ViewportInstance`: orbit navigation (left/middle
//! drag), pan (right drag), and zoom (scroll) act only on the window they happen in.
//! The left window animates every frame (`RedrawMode::Continuous`); the right window
//! only redraws when you interact with it (`RedrawMode::OnDemand`), so it sits idle
//! otherwise.
//!
//! This is different from `winit-multi-viewport`, which draws several viewports into
//! one window's surface. Here each viewport is a separate OS window.

use std::cell::Cell;
use std::rc::Rc;

use viewport_lib as vpl;
use vpl::{AppConfigV2, Material, NodeId, ViewportAppV2, WindowConfig, primitives};
use vpl::runners::viewport_app::RedrawMode;

fn main() {
    // The left window's animated node id, shared between its setup and frame closures.
    let cube_id: Rc<Cell<Option<NodeId>>> = Rc::new(Cell::new(None));
    let left_setup_id = cube_id.clone();

    ViewportAppV2::new(AppConfigV2::default())
        .window(
            WindowConfig::default()
                .with_title("viewport-lib : left (animated)")
                .with_window_size(900, 700)
                .with_redraw_mode(RedrawMode::Continuous),
            move |vp, device| {
                let cube = vp
                    .resources_mut()
                    .upload_mesh_data(device, &primitives::cube(1.0))
                    .unwrap();
                let node = vp.scene_mut().add(
                    Some(cube),
                    glam::Mat4::IDENTITY,
                    Material::from_colour([0.12, 0.3, 0.7]),
                );
                left_setup_id.set(Some(node));
                vp.camera_mut().distance = 6.0;
            },
            move |ctx| {
                if let Some(id) = cube_id.get() {
                    // Z-up: spin the cube about the world up axis.
                    let spin = glam::Mat4::from_rotation_z(ctx.time());
                    ctx.scene_mut().set_local_transform(id, spin);
                }
            },
        )
        .window(
            WindowConfig::default()
                .with_title("viewport-lib : right (on demand)")
                .with_window_size(900, 700)
                .with_redraw_mode(RedrawMode::OnDemand),
            |vp, device| {
                let torus = vp
                    .resources_mut()
                    .upload_mesh_data(device, &primitives::torus(0.6, 0.22, 32, 16))
                    .unwrap();
                vp.scene_mut().add(
                    Some(torus),
                    glam::Mat4::IDENTITY,
                    Material::from_colour([0.1, 0.55, 0.2]),
                );
                vp.camera_mut().distance = 6.0;
            },
            |_ctx| {
                // Static scene: nothing to update per frame. Under OnDemand this
                // window only redraws in response to input or a resize.
            },
        )
        .run();
}
