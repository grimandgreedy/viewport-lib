//! Desktop runner for the mobile example.
//!
//! Handy for working on the scene or the camera without a device or simulator
//! in the loop. Mouse drags land on the same orbit controller the touch
//! handler feeds; there are no touches to map.
//!
//! On Android and iOS the entry point is in `src/lib.rs` instead, so this
//! binary's `main` does nothing there.

fn main() {
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    viewport_lib_examples_mobile::start();
}
