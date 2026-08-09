//! Basic interaction with the ViewportApp runner: overlays, GPU picking, exit,
//! and an input handler that stops the viewport reacting to input under the UI.
//!
//! The `run` closure draws overlays and handles clicks, using what `FrameCtx`
//! exposes beyond the session it derefs to:
//!
//! - `ctx.overlays_mut()` draws a title, a live pick readout, and a quit button.
//!   Assembly clears the overlay frame each frame, so these are re-pushed every
//!   frame; the runner installs them after assembly and before render.
//! - `ctx.device()` / `ctx.queue()` make GPU picking reachable. A left click runs
//!   `pick_gpu` against the scene and the hit id shows up in the readout.
//! - Input is resolved before the callback, so `ctx.action_frame().pointer`
//!   reports the click on the same frame it lands.
//! - Clicking the quit button calls `ctx.request_exit()`.
//!
//! Navigation is handled by `with_input` rather than the built-in orbit. The
//! handler drives its own `OrbitCameraController` and, while the pointer is over
//! the quit button, withholds navigation so the wheel does not zoom and a drag
//! does not orbit under the button. Clicks are always forwarded so the button and
//! picking still work. This is the general seam a context menu or any modal
//! overlay uses to capture input.

use std::cell::Cell;
use std::rc::Rc;

use viewport_lib::{
    AppConfig, LabelAnchor, LabelItem, Material, OrbitCameraController, OverlayFill, OverlayShape,
    OverlayShapeItem, PickMask, ViewportApp, ViewportContext, ViewportEvent, primitives,
};

// Quit button rectangle in logical pixels (top-left corner of the window).
const QUIT_POS: [f32; 2] = [20.0, 20.0];
const QUIT_SIZE: [f32; 2] = [170.0, 44.0];

fn quit_button_hit(cursor: glam::Vec2) -> bool {
    cursor.x >= QUIT_POS[0]
        && cursor.x <= QUIT_POS[0] + QUIT_SIZE[0]
        && cursor.y >= QUIT_POS[1]
        && cursor.y <= QUIT_POS[1] + QUIT_SIZE[1]
}

fn main() {
    // Last picked object id, shared into the per-frame closure.
    let last_pick: Rc<Cell<Option<u64>>> = Rc::new(Cell::new(None));

    ViewportApp::new(
        AppConfig::default()
            .with_title("viewport-lib : app features")
            .with_window_size(1280, 720),
    )
    .setup(|session, device| {
        let cube = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::cube(1.0))
            .unwrap();
        let sphere = session
            .resources_mut()
            .upload_mesh_data(device, &primitives::sphere(0.6, 24, 12))
            .unwrap();

        let scene = session.scene_mut();
        scene.add(
            Some(cube),
            glam::Mat4::from_translation(glam::Vec3::new(-1.5, 0.0, 0.0)),
            Material::from_colour([0.4, 0.6, 0.9]),
        );
        scene.add(
            Some(sphere),
            glam::Mat4::from_translation(glam::Vec3::new(1.5, 0.0, 0.0)),
            Material::from_colour([0.9, 0.5, 0.2]),
        );

        session.camera_mut().distance = 8.0;
    })
    // Drive navigation ourselves so we can suppress it under the UI. The runner
    // stops auto-feeding events and driving orbit; this handler owns the input.
    .with_input({
        let mut orbit = OrbitCameraController::viewport_all();
        let mut cursor = glam::Vec2::ZERO;
        move |ictx| {
            let size = ictx.viewport_size();
            orbit.begin_frame(ViewportContext {
                hovered: true,
                focused: true,
                viewport_size: size,
            });
            // Cloned so we can forward (which borrows the instance) while reading.
            for ev in ictx.events().to_vec() {
                if let ViewportEvent::PointerMoved { position } = ev {
                    cursor = position;
                }
                // The viewport always sees the event: picking and click detection
                // in the run callback read the resolved frame.
                ictx.forward(ev.clone());
                // Navigate only when the pointer is not over the quit button, so
                // the wheel and drags do not move the camera under the UI.
                if !quit_button_hit(cursor) {
                    orbit.push_event(ev);
                }
            }
            orbit.apply_to_camera(ictx.camera_mut());
        }
    })
    .run(move |ctx| {
        // Handle this frame's click first: quit button, otherwise GPU pick.
        let pointer = ctx.action_frame().pointer;
        if pointer.clicked {
            if let Some(cursor) = pointer.cursor {
                if quit_button_hit(cursor) {
                    ctx.request_exit();
                    return;
                }
                // pick_gpu takes &mut self plus the device and queue; borrow the
                // handles first so the session borrow does not overlap them.
                let device = ctx.device().clone();
                let queue = ctx.queue().clone();
                let hit = ctx.pick_gpu(&device, &queue, cursor, PickMask::OBJECT);
                last_pick.set(hit.map(|h| h.id));
            }
        }

        // Overlays: re-pushed every frame because assembly clears them.
        let overlays = ctx.overlays_mut();
        overlays.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 6.0 },
                QUIT_POS,
                QUIT_SIZE,
            )
            .with_fill(OverlayFill::Solid([0.7, 0.2, 0.2, 0.9])),
        );
        // Center the label in the quit box: anchor at the box center, then
        // Center alignment centers horizontally and the anchor is already the
        // text's vertical center.
        let quit_center = [
            QUIT_POS[0] + QUIT_SIZE[0] * 0.5,
            QUIT_POS[1] + QUIT_SIZE[1] * 0.5,
        ];
        overlays.labels.push(
            LabelItem::new("click to quit")
                .with_screen_anchor(quit_center)
                .with_anchor_align(LabelAnchor::Middle)
                .with_font_size(22.0),
        );

        let readout = match last_pick.get() {
            Some(id) => format!("picked object {id}"),
            None => "click an object to pick it".to_string(),
        };
        overlays.labels.push(
            LabelItem::new(readout)
                .with_screen_anchor([20.0, 90.0])
                .with_font_size(22.0),
        );
    });
}
