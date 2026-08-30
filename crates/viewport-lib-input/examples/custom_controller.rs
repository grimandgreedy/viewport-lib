//! Build your own camera controller.
//!
//! Input handling is a replaceable layer: resolve events into an `ActionFrame`
//! with `ViewportInput`, then apply that frame to a `Camera` however you like.
//! This example does exactly that with a hand-rolled controller that has nothing
//! to do with the built-in `OrbitCameraController` - it needs no renderer and no
//! window, just the input crate.
//!
//! Run: `cargo run -p viewport-lib-input --example custom_controller`

use viewport_lib_input::input::{
    BindingPreset, ButtonState, MouseButton, ViewportContext, ViewportEvent, ViewportInput,
};
use viewport_lib_input::{Camera, CameraController};
use viewport_lib_types::input::ActionFrame;

/// A custom controller: orbit at half sensitivity and ignore pan/zoom entirely.
///
/// Replacing the built-in policy is this small - implement `apply`, read the
/// resolved navigation deltas, drive the camera your way.
struct HalfSpeedOrbit;

impl CameraController for HalfSpeedOrbit {
    fn apply(&mut self, camera: &mut Camera, frame: &ActionFrame) {
        let nav = &frame.navigation;
        if nav.orbit.x != 0.0 || nav.orbit.y != 0.0 {
            camera.orbit(nav.orbit.x * 0.5, nav.orbit.y * 0.5);
        }
        // A real custom controller might map zoom to dolly, add inertia, clamp
        // pitch differently, etc. The point is the app decides here.
    }
}

fn main() {
    // 1. RESOLVE: feed native-shaped events into ViewportInput for one frame.
    let mut input = ViewportInput::from_preset(BindingPreset::ViewportPrimitives);
    input.begin_frame(ViewportContext {
        hovered: true,
        focused: true,
        viewport_size: [800.0, 600.0],
    });

    // In the primitives preset, plain left-drag (no modifiers) orbits. Drive it
    // by hand: place the pointer, press the button there (that sets the drag
    // origin), then move across the viewport to accumulate the drag delta.
    input.push_event(ViewportEvent::PointerMoved {
        position: [400.0, 300.0].into(),
    });
    input.push_event(ViewportEvent::MouseButton {
        button: MouseButton::Left,
        state: ButtonState::Pressed,
    });
    input.push_event(ViewportEvent::PointerMoved {
        position: [460.0, 300.0].into(),
    });

    let frame: ActionFrame = input.resolve();

    // 2. APPLY: hand the resolved frame to your controller.
    let mut camera = Camera::default();
    let before = camera.orientation;

    let mut controller = HalfSpeedOrbit;
    controller.apply(&mut camera, &frame);

    println!("resolved orbit delta : {:?}", frame.navigation.orbit);
    println!("camera orientation before : {before:?}");
    println!("camera orientation after  : {:?}", camera.orientation);
    println!(
        "moved: {}",
        before.abs_diff_eq(camera.orientation, 1e-6) == false
    );
}
