//! Minimal viewport-lib example using the built-in winit runner.
//!
//! `ViewportApp` owns the window, the wgpu device, and the event loop, and
//! drives a `ViewportInstance` each frame. Orbit navigation (left/middle drag),
//! pan (right drag), and zoom (scroll) work with no per-event code here: the
//! runner translates winit events and drives the camera. Compare the hand-written
//! event loop in `examples/winit_viewport` for the full-control version.

use std::cell::Cell;
use std::rc::Rc;

use viewport_lib as vpl;
use vpl::{AppConfig, Material, NodeId, ViewportApp, primitives};

fn main() {
    // The setup closure and the per-frame closure share the animated node's id.
    let cube_id: Rc<Cell<Option<NodeId>>> = Rc::new(Cell::new(None));
    let setup_id = cube_id.clone();

    ViewportApp::new(
        AppConfig::default()
            .with_title("viewport-lib : minimal")
            .with_window_size(1280, 720),
    )
    .setup(move |session, device| {
        // Upload the meshes once, with the device in hand.
        let sphere = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(0.6, 24, 12))
            .unwrap();
        let cube = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::cube(1.0))
            .unwrap();
        let torus = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::torus(0.5, 0.18, 32, 16))
            .unwrap();

        // Z-up scene: three primitives laid out along X.
        let scene = session.scene_mut();
        scene.add(
            Some(sphere),
            glam::Mat4::from_translation(glam::Vec3::new(-2.5, 0.0, 0.0)),
            Material::from_colour([0.75, 0.28, 0.05]),
        );
        let cube_node = scene.add(
            Some(cube),
            glam::Mat4::IDENTITY,
            Material::from_colour([0.12, 0.3, 0.7]),
        );
        scene.add(
            Some(torus),
            glam::Mat4::from_translation(glam::Vec3::new(2.5, 0.0, 0.0)),
            Material::from_colour([0.1, 0.55, 0.2]),
        );
        setup_id.set(Some(cube_node));

        session.camera_mut().distance = 10.0;
    })
    .run(move |ctx| {
        if let Some(id) = cube_id.get() {
            // Z-up: spin the cube about the world up axis.
            let spin = glam::Mat4::from_rotation_z(ctx.time());
            ctx.scene_mut().set_local_transform(id, spin);
        }
    });
}
