//! Host crate for the eframe/egui `viewport-lib` examples, and the eframe
//! version shim they build against.
//!
//! Run one with:
//!
//! ```text
//! cargo run --release -p viewport-lib-examples-eframe --example eframe-showcase
//! ```
//!
//! # The leg shim
//!
//! eframe embeds wgpu, so an eframe version only works against the matching
//! viewport-lib leg: 0.33 with wgpu 27, 0.35 with 29, 0.36 with 30. The three
//! are aliased dependencies selected by this crate's `wgpu27` / `wgpu29` /
//! `wgpu30` features and re-exported here under one name, so an example writes
//!
//! ```ignore
//! use viewport_lib_examples_eframe::eframe;
//! ```
//!
//! and builds on whichever leg is selected, instead of needing one source per
//! version.
//!
//! # Per-example leg status
//!
//! What pins an example to one leg is eframe's and egui's own API, not
//! viewport-lib's. Two changes matter: eframe 0.35 replaced
//! `App::update(&Context, ..)` with `App::ui(&mut Ui, ..)`, and egui 0.35
//! replaced `SidePanel` / `TopBottomPanel` with a unified `Panel` that attaches
//! to a `Ui` rather than to the `Context`.
//!
//! An example carrying only a central panel works on every leg: it keeps both
//! entry points behind a `cfg` and shares one body, because a frameless central
//! panel on 0.33 produces the same margin-free `Ui` that 0.35 hands the app.
//! Examples that lay out side or top panels still need their layout code
//! migrated, so they stay on the default leg.
//!
//! | example | legs |
//! | --- | --- |
//! | `eframe-minimal`, `eframe-minimal-callback`, `eframe-primitives`, `eframe-retained-overlay`, `eframe-multi-viewport`, `debug-light` | 27, 29, 30 |
//! | `eframe-exposure`, `eframe-lighting-shadows`, `eframe-input-controllers`, `eframe-render-paths`, `eframe-testing`, `showcase`, `eframe-showcase` | 27 only (side/top panel layout) |
//!
//! Check a non-default leg with:
//!
//! ```text
//! cargo run --release -p viewport-lib-examples-eframe --no-default-features \
//!     --features wgpu30 --example eframe-minimal
//! ```

#[cfg(not(any(feature = "wgpu27", feature = "wgpu29", feature = "wgpu30")))]
compile_error!("enable exactly one wgpu leg: wgpu27, wgpu29, or wgpu30");

#[cfg(any(
    all(feature = "wgpu27", feature = "wgpu29"),
    all(feature = "wgpu27", feature = "wgpu30"),
    all(feature = "wgpu29", feature = "wgpu30"),
))]
compile_error!("enable exactly one wgpu leg: wgpu27, wgpu29, or wgpu30");

#[cfg(feature = "wgpu27")]
pub use eframe033 as eframe;
#[cfg(feature = "wgpu29")]
pub use eframe035 as eframe;
#[cfg(feature = "wgpu30")]
pub use eframe036 as eframe;

/// eframe options with a device the renderer can actually use.
///
/// eframe requests `wgpu::Limits::default()`, which grants eight storage buffers
/// per shader stage. That is the floor for the lit mesh path and no more, so a
/// build with the `raytrace` feature (this crate's `showcase`, on by default)
/// needs ten and `ViewportRenderer::new` rejects the device outright. Every
/// example goes through here rather than repeating the closure.
///
/// The existing configuration is edited rather than rebuilt, because
/// `WgpuSetupCreateNew` gained a field without a default in egui 0.35 and so
/// cannot be constructed the same way on all three legs.
///
/// `size` is the initial window size in logical points.
pub fn native_options(size: [f32; 2]) -> eframe::NativeOptions {
    use eframe::wgpu;

    let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.device_descriptor = std::sync::Arc::new(|adapter| {
            // WebGL cannot meet the recommended limits at all, so it keeps its own
            // downlevel set and gives up the features that need more.
            let base = if adapter.get_info().backend == wgpu::Backend::Gl {
                wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                viewport_lib::ViewportRenderer::recommended_device_limits(adapter)
            };
            wgpu::DeviceDescriptor {
                label: Some("viewport-lib example device"),
                required_features: viewport_lib::ViewportRenderer::recommended_device_features(
                    adapter,
                ),
                required_limits: wgpu::Limits {
                    max_texture_dimension_2d: 8192,
                    ..base
                },
                ..Default::default()
            }
        });
    }

    eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size(size),
        depth_buffer: 24,
        stencil_buffer: 8,
        wgpu_options,
        ..Default::default()
    }
}
