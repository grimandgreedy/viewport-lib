//! Shared scene catalogue and headless harness for testing and benchmarking
//! `viewport-lib`.
//!
//! The catalogue defines scenes once, as data, so the same definitions feed
//! every driver: the counter-assertion tests, the golden-image snapshot tests,
//! the benches, and the `catalogue_viewer` example. It deliberately adds the
//! coverage the in-tree examples lack: concave geometry, grazing and below-angle
//! lighting, and real textures (checker, noise, normal maps).
//!
//! Typical use:
//!
//! ```no_run
//! use viewport_lib_testkit::{Harness, catalogue, frame_for};
//!
//! let mut h = Harness::new().expect("no GPU adapter");
//! for scene in catalogue() {
//!     let built = h.build_scene(&scene);
//!     let cam = &scene.cameras[0].camera;
//!     let frame = frame_for(&built, cam, [400.0, 300.0]);
//!     let stats = h.render_two_frames(&frame, 400, 300);
//!     println!("{}: {} draw calls", scene.name, stats.draw_calls);
//! }
//! ```

// The core: the headless device constructor and `Harness`. Always available; it
// carries no scene catalogue, so it builds under `--no-default-features`.
pub mod device;
pub mod harness;

// The scene corpus, behind the `scenes` feature so the core builds without it.
// The `scenes` module tree holds the catalogue plus the meshes, rigs, textures,
// and optional real-model loaders it builds from.
#[cfg(feature = "scenes")]
pub mod scenes;

// Checks: golden-image comparison and FrameStats counter snapshots. Each behind
// its own feature so a caller can pull the harness without them.
#[cfg(feature = "golden")]
pub mod golden;

#[cfg(feature = "counters")]
pub mod counters;

pub use device::{DeviceProfile, Limits, headless_device, headless_device_with};
pub use harness::Harness;

// Re-export the corpus submodules and catalogue items at the crate root so the
// public paths (`viewport_lib_testkit::meshes`, `::catalogue`, ...) stay stable
// regardless of the internal module layout.
#[cfg(feature = "real_models")]
pub use scenes::real_models;
#[cfg(feature = "scenes")]
pub use scenes::{
    BuildCtx, BuiltScene, NamedCamera, NamedScene, TEST_BACKGROUND, catalogue, frame_for,
    orbit_camera, scene_by_name, standard_cameras,
};
#[cfg(feature = "scenes")]
pub use scenes::{meshes, rigs, textures};
