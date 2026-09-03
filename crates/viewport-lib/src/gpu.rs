//! Internal alias for the wgpu version this build targets.
//!
//! The crate names `crate::gpu::*` instead of `wgpu::*` so that selecting a
//! wgpu version is this one module's concern. The `wgpu27` / `wgpu29` cargo
//! features pick the version by re-exporting the matching renamed dependency;
//! exactly one must be enabled, and only that one's crate is compiled.

#[cfg(all(feature = "wgpu27", feature = "wgpu29"))]
compile_error!("the `wgpu27` and `wgpu29` features are mutually exclusive: enable exactly one");

#[cfg(not(any(feature = "wgpu27", feature = "wgpu29")))]
compile_error!(
    "viewport-lib needs a wgpu version: enable exactly one of the `wgpu27` or `wgpu29` features (the default is `wgpu27`)"
);

// The 27 crate keeps its real name `wgpu`; only 29 is aliased.
#[cfg(all(feature = "wgpu27", not(feature = "wgpu29")))]
pub use wgpu::*;
#[cfg(all(feature = "wgpu29", not(feature = "wgpu27")))]
pub use wgpu29::*;

// The globs above do not bring in macros, so re-export by name the wgpu macros
// the crate uses.
#[cfg(all(feature = "wgpu27", not(feature = "wgpu29")))]
pub use wgpu::vertex_attr_array;
#[cfg(all(feature = "wgpu29", not(feature = "wgpu27")))]
pub use wgpu29::vertex_attr_array;

/// Construct a wgpu `Instance` with default options. This papers over the
/// `InstanceDescriptor` construction that differs across wgpu versions: 27
/// derives `Default`, while 29 gained a display-handle field and constructs
/// through `new_without_display_handle`. Used for headless setup and by the
/// test harness (in this crate and the testkit).
#[cfg(all(feature = "wgpu27", not(feature = "wgpu29")))]
#[doc(hidden)]
pub fn default_instance() -> Instance {
    Instance::new(&InstanceDescriptor::default())
}
// 29 also takes the descriptor by value rather than by reference.
#[cfg(all(feature = "wgpu29", not(feature = "wgpu27")))]
#[doc(hidden)]
pub fn default_instance() -> Instance {
    Instance::new(InstanceDescriptor::new_without_display_handle())
}

/// The outcome of acquiring the next surface texture, normalised across wgpu legs.
///
/// wgpu 27 returns `Result<SurfaceTexture, SurfaceError>`; wgpu 29 returns a
/// `CurrentSurfaceTexture` enum with no `SurfaceError`. [`acquire_surface`] maps both
/// to this, so a runner acquiring a surface does not need its own per-version branch.
#[cfg(any(feature = "wgpu27", feature = "wgpu29"))]
pub enum SurfaceFrame {
    /// The texture to render into and present this frame.
    Acquired(SurfaceTexture),
    /// The surface is lost or outdated: reconfigure it and skip this frame.
    Recreate,
    /// A transient failure (timeout, occlusion, ...): skip this frame and retry next.
    Skip,
}

/// Acquire the next surface texture, mapping each wgpu leg's result to
/// [`SurfaceFrame`]. Non-recoverable errors are logged here.
#[cfg(all(feature = "wgpu27", not(feature = "wgpu29")))]
pub fn acquire_surface(surface: &Surface<'_>) -> SurfaceFrame {
    match surface.get_current_texture() {
        Ok(texture) => SurfaceFrame::Acquired(texture),
        Err(SurfaceError::Lost | SurfaceError::Outdated) => SurfaceFrame::Recreate,
        Err(e) => {
            tracing::error!("surface error: {e:?}");
            SurfaceFrame::Skip
        }
    }
}

/// See the wgpu 27 sibling: on wgpu 29 `get_current_texture` returns a
/// `CurrentSurfaceTexture` enum rather than a `Result`.
#[cfg(all(feature = "wgpu29", not(feature = "wgpu27")))]
pub fn acquire_surface(surface: &Surface<'_>) -> SurfaceFrame {
    match surface.get_current_texture() {
        CurrentSurfaceTexture::Success(texture) | CurrentSurfaceTexture::Suboptimal(texture) => {
            SurfaceFrame::Acquired(texture)
        }
        CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Lost => SurfaceFrame::Recreate,
        // Timeout, Occluded, Validation, and any future variant: skip this frame.
        _ => SurfaceFrame::Skip,
    }
}

/// The device feature that enables `@builtin(primitive_index)` in fragment
/// shaders, named per wgpu leg: `SHADER_PRIMITIVE_INDEX` on wgpu 27,
/// `PRIMITIVE_INDEX` on wgpu 29. Used by the GPU pick pass to resolve the hit
/// triangle for sub-object picking.
#[cfg(all(feature = "wgpu27", not(feature = "wgpu29")))]
#[doc(hidden)]
pub const PRIMITIVE_INDEX_FEATURE: Features = Features::SHADER_PRIMITIVE_INDEX;
#[cfg(all(feature = "wgpu29", not(feature = "wgpu27")))]
#[doc(hidden)]
pub const PRIMITIVE_INDEX_FEATURE: Features = Features::PRIMITIVE_INDEX;

/// The device feature that enables the WGSL `ray_query` extension (hardware BVH
/// traversal). Named the same on both wgpu legs. The path tracer's hardware
/// backend requires it; the portable compute traversal does not.
#[cfg(any(feature = "wgpu27", feature = "wgpu29"))]
#[doc(hidden)]
pub const RAY_QUERY_FEATURE: Features = Features::EXPERIMENTAL_RAY_QUERY;

// Version-portability helpers, surfaced here (the version seam) so tests and
// consumers that build wgpu pipelines directly can do so without their own
// per-version `#[cfg]`. Each helper is the single place a new wgpu leg (e.g.
// wgpu 30) grows a branch. Defined in `resources::builders`; re-exported here to
// keep them findable next to the alias.
#[cfg(any(feature = "wgpu27", feature = "wgpu29"))]
pub use crate::resources::builders::{
    RenderPipelineDesc, dcompare, depth_stencil, dmipmap, dwrite, pipeline_layout, render_pipeline,
    scene_depth_stencil, write_mapped,
};
