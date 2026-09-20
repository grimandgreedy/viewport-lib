//! `viewport-lib` example for Android and iOS: winit plus wgpu, driven by touch.
//!
//! One crate covers both platforms, the way cargo-mobile2 expects. `mobile.toml`
//! names the app, `cargo android run` and `cargo apple run` build and deploy it,
//! and the same code runs on the desktop through `src/main.rs`.
//!
//! | Gesture | Action |
//! | --- | --- |
//! | 1-finger drag | orbit |
//! | 2-finger drag | pan |
//! | pinch | zoom |
//! | 2-finger rotate | roll (iOS only) |
//!
//! The example owns its wgpu surface and calls `Surface::get_current_texture`,
//! which returns a `Result` on wgpu 27 and a `CurrentSurfaceTexture` enum from
//! 29 on, so it builds on the `wgpu27` leg only.
//!
//! See `README.md` for the cargo-mobile2 setup, which regenerates this file.

mod app;

pub use app::{run, start};

/// Entry point the Android runtime calls through the NDK.
#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
fn android_main(android_app: android_activity::AndroidApp) {
    use winit::error::EventLoopError;
    use winit::event_loop::EventLoop;
    use winit::platform::android::EventLoopBuilderExtAndroid;

    let event_loop = match EventLoop::builder().with_android_app(android_app).build() {
        Ok(el) => el,
        Err(EventLoopError::RecreationAttempt) => return,
        Err(e) => panic!("event loop: {e}"),
    };
    run(event_loop);
}

/// Entry point the generated `main.mm` calls on iOS.
#[cfg(target_os = "ios")]
#[unsafe(no_mangle)]
pub extern "C" fn start_app() {
    start();
}
