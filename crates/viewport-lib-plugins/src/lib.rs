//! Plugins for viewport-lib, built against its public plugin API.
//!
//! Each module holds one kind of plugin. [`item_types`] is the set of item
//! types viewport-lib ships with: point clouds, sprites, curves, volumes,
//! vector and tensor fields and the rest. Nothing is re-exported at the crate
//! root; a type is named by the module it belongs to.
//!
//! ```no_run
//! use viewport_lib_plugins::item_types::{self, point_cloud::PointCloudItem};
//! # let mut renderer: viewport_lib::renderer::ViewportRenderer = unimplemented!();
//! # let device: &viewport_lib::gpu::Device = unimplemented!();
//! item_types::install(&mut renderer, device);
//! ```

// Every version seam in this crate is spelled `#[cfg(feature = "wgpu29")]`, so
// it reads this crate's own features, while the wgpu that actually gets
// compiled is whatever cargo unified across the graph. A dependency edge that
// takes this crate with `default-features = false` and forwards no leg leaves
// the features unset while wgpu resolves to 29 or 30, and the 27 arm of every
// seam compiles against the wrong crate. Assert the two agree.
#[cfg(all(feature = "wgpu27", not(feature = "wgpu29"), not(feature = "wgpu30")))]
const OWN_WGPU_LEG: u32 = 27;
#[cfg(all(feature = "wgpu29", not(feature = "wgpu27"), not(feature = "wgpu30")))]
const OWN_WGPU_LEG: u32 = 29;
#[cfg(all(feature = "wgpu30", not(feature = "wgpu27"), not(feature = "wgpu29")))]
const OWN_WGPU_LEG: u32 = 30;
#[cfg(not(any(feature = "wgpu27", feature = "wgpu29", feature = "wgpu30")))]
const OWN_WGPU_LEG: u32 = 0;

const _: () = assert!(
    OWN_WGPU_LEG == viewport_lib::gpu::WGPU_LEG,
    "viewport-lib-plugins was compiled with a wgpu leg feature that does not match the \
     wgpu viewport-lib resolved to (0 below means no leg feature was enabled at all). \
     Whoever depends on this crate has to forward the same `wgpu27` / `wgpu29` / `wgpu30` \
     feature it gives viewport-lib."
);

pub mod item_types;

/// Every shader this crate compiles, as `(name, source)`, with the shared
/// sections already spliced in front of each body.
///
/// A body on its own does not compile: it declares no group-0 bindings and
/// calls helpers it does not define. This returns what the pipelines actually
/// hand to `create_shader_module`, which is what a validation pass wants.
pub fn shader_sources() -> Vec<(&'static str, String)> {
    item_types::shader_sources()
}
