//! `ViewportAppV2` on the wgpu 29 leg: the same runner as the other `app-*`
//! examples, built against wgpu 29 instead of the default wgpu 27.
//!
//! Run with:
//!   cargo run --release --example app-minimal-wgpu29 --no-default-features --features wgpu29,app
//!
//! It is deliberately minimal (one window, one spinning cube): its job is to prove
//! the `app` runners build and run on the wgpu 29 leg, the way `eframe-minimal-wgpu29`
//! and `slint-minimal` prove the framework integrations do.

use std::cell::Cell;
use std::rc::Rc;

use viewport_lib as vpl;
use vpl::{AppConfigV2, Material, NodeId, ViewportAppV2, WindowConfig, primitives};

fn main() {
    let cube_id: Rc<Cell<Option<NodeId>>> = Rc::new(Cell::new(None));
    let setup_id = cube_id.clone();

    ViewportAppV2::new(AppConfigV2::default())
        .window(
            WindowConfig::default()
                .with_title("viewport-lib : ViewportAppV2 on wgpu29")
                .with_window_size(1000, 700),
            move |vp, device| {
                let cube = vp
                    .resources_mut()
                    .upload_mesh_data(device, &primitives::cube(1.0))
                    .unwrap();
                let node = vp.scene_mut().add(
                    Some(cube),
                    glam::Mat4::IDENTITY,
                    Material::from_colour([0.85, 0.25, 0.2]),
                );
                setup_id.set(Some(node));
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
        .run();
}
