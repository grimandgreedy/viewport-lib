//! Several OS windows driven by one runner, each with its own scene and camera.
//!
//! `ViewportAppV2` owns every window, the shared wgpu device, and the event loop.
//! Each window drives its own `ViewportInstance`: orbit navigation (left/middle
//! drag), pan (right drag), and zoom (scroll) act only on the window they happen in.
//! The left window animates every frame (`RedrawMode::Continuous`); the right window
//! only redraws when you interact with it (`RedrawMode::OnDemand`), so it sits idle
//! otherwise.
//!
//! Runtime open/close: press N in any window to open a new one, and W to close the
//! focused window. Closing the last window ends the app.
//!
//! Tier-2 input: dropping a file on a window, or pressing the back/forward mouse
//! buttons, prints to stdout, showing those events reaching the callback.
//!
//! This is different from `winit-multi-viewport`, which draws several viewports into
//! one window's surface. Here each viewport is a separate OS window.

use std::cell::Cell;
use std::rc::Rc;

use viewport_lib as vpl;
use vpl::runners::viewport_app::RedrawMode;
use vpl::{
    AppConfigV2, ButtonState, FrameCtxV2, KeyCode, Material, MouseButton, NodeId, ViewportAppV2,
    ViewportEvent, WindowConfig, primitives,
};

/// N opens a new window, W closes the focused one. Called from every window's
/// per-frame callback (including windows opened at runtime), so the keys work no
/// matter which window has focus.
fn handle_open_close(ctx: &mut FrameCtxV2) {
    let mut open_new = false;
    let mut close_self = false;
    for ev in ctx.events() {
        match ev {
            ViewportEvent::Key { key, state, .. } if *state == ButtonState::Pressed => match key {
                KeyCode::N => open_new = true,
                KeyCode::W => close_self = true,
                _ => {}
            },
            // Tier-2 events surface on the raw event stream; here we just report them.
            ViewportEvent::MouseButton { button, state } if *state == ButtonState::Pressed => {
                match button {
                    MouseButton::Back => println!("window {:?}: back button", ctx.window_id()),
                    MouseButton::Forward => {
                        println!("window {:?}: forward button", ctx.window_id())
                    }
                    _ => {}
                }
            }
            ViewportEvent::FileDropped(path) => {
                println!("window {:?}: file dropped: {}", ctx.window_id(), path.display());
            }
            _ => {}
        }
    }
    if open_new {
        ctx.open_window(
            WindowConfig::default()
                .with_title("viewport-lib : opened at runtime")
                .with_window_size(700, 600),
            |vp, device| {
                let sphere = vp
                    .resources_mut()
                    .upload_mesh_data(device, &primitives::sphere(0.7, 24, 12))
                    .unwrap();
                vp.scene_mut().add(
                    Some(sphere),
                    glam::Mat4::IDENTITY,
                    Material::from_colour([0.8, 0.5, 0.1]),
                );
                vp.camera_mut().distance = 6.0;
            },
            handle_open_close,
        );
    }
    if close_self {
        let id = ctx.window_id();
        ctx.close_window(id);
    }
}

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
                handle_open_close(ctx);
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
            handle_open_close,
        )
        .run();
}
