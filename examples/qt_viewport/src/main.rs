//! Qt/QML integration for `viewport-lib` via `cxx-qt`.
//!
//! The Rust side renders offscreen and publishes frames through a
//! `QQuickImageProvider`; QML handles presentation and input.
//!
//! # Prerequisites
//!
//! - Qt 6.5+ installed
//! - `qmake` in PATH (or set `QMAKE` environment variable)
//!
//! # Build and run
//!
//! ```sh
//! cd examples/qt_viewport
//! cargo run
//! ```

pub mod viewport_backend;
pub mod viewport_bridge;

use cxx_qt_lib::{QGuiApplication, QQmlApplicationEngine, QUrl};

unsafe extern "C" {
    fn register_viewport_provider(engine: *mut std::ffi::c_void);
}

fn main() {
    let mut app = QGuiApplication::new();
    let mut engine = QQmlApplicationEngine::new();

    if let Some(engine_ref) = engine.as_mut() {
        // Register the viewport image provider before loading QML.
        // Safety: we pass a valid QQmlApplicationEngine pointer; the C++ side
        // casts it back and calls engine->addImageProvider().
        unsafe {
            let ptr: *mut std::ffi::c_void =
                &*engine_ref as *const _ as *mut std::ffi::c_void;
            register_viewport_provider(ptr);
        }

        engine_ref.load(&QUrl::from(
            "qrc:/qt/qml/com/grimandgreedy/viewportlib/qml/main.qml",
        ));
    }

    if let Some(app) = app.as_mut() {
        app.exec();
    }
}
