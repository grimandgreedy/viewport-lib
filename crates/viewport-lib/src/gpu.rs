//! Internal alias for the wgpu version this build targets.
//!
//! The crate names `crate::gpu::*` instead of `wgpu::*` so that selecting a
//! wgpu version is this one module's concern. The `wgpu27` / `wgpu29` / `wgpu30`
//! cargo features pick the version by re-exporting the matching renamed
//! dependency; exactly one must be enabled, and only that one's crate is
//! compiled.

#[cfg(any(
    all(feature = "wgpu27", feature = "wgpu29"),
    all(feature = "wgpu27", feature = "wgpu30"),
    all(feature = "wgpu29", feature = "wgpu30"),
))]
compile_error!(
    "the `wgpu27`, `wgpu29`, and `wgpu30` features are mutually exclusive: enable exactly one"
);

#[cfg(not(any(feature = "wgpu27", feature = "wgpu29", feature = "wgpu30")))]
compile_error!(
    "viewport-lib needs a wgpu version: enable exactly one of `wgpu27`, `wgpu29`, or `wgpu30` (the default is `wgpu27`)"
);

// The 27 crate keeps its real name `wgpu`; 29 and 30 are aliased.
#[cfg(wgpu27)]
pub use wgpu::*;
#[cfg(wgpu29)]
pub use wgpu29::*;
#[cfg(wgpu30)]
pub use wgpu30::*;

// The globs above do not bring in macros, so re-export by name the wgpu macros
// the crate uses.
#[cfg(wgpu27)]
pub use wgpu::vertex_attr_array;
#[cfg(wgpu29)]
pub use wgpu29::vertex_attr_array;
#[cfg(wgpu30)]
pub use wgpu30::vertex_attr_array;

/// Construct a wgpu `Instance` with default options. This papers over the
/// `InstanceDescriptor` construction that differs across wgpu versions: 27
/// derives `Default`, while 29 and 30 gained a display-handle field and
/// construct through `new_without_display_handle`. Used for headless setup and
/// by the test harness (in this crate and the testkit).
#[cfg(wgpu27)]
#[doc(hidden)]
pub fn default_instance() -> Instance {
    Instance::new(&InstanceDescriptor::default())
}
// 29 and 30 also take the descriptor by value rather than by reference.
#[cfg(any(wgpu29, wgpu30))]
#[doc(hidden)]
pub fn default_instance() -> Instance {
    Instance::new(InstanceDescriptor::new_without_display_handle())
}

/// The outcome of acquiring the next surface texture, normalised across wgpu legs.
///
/// wgpu 27 returns `Result<SurfaceTexture, SurfaceError>`; wgpu 29 and 30 return a
/// `CurrentSurfaceTexture` enum with no `SurfaceError`. [`acquire_surface`] maps both
/// to this, so a runner acquiring a surface does not need its own per-version branch.
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
#[cfg(wgpu27)]
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

/// See the wgpu 27 sibling: on wgpu 29 and 30 `get_current_texture` returns a
/// `CurrentSurfaceTexture` enum rather than a `Result`.
#[cfg(any(wgpu29, wgpu30))]
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

/// Present an acquired surface texture. wgpu 27 and 29 present through the
/// texture (`SurfaceTexture::present`); wgpu 30 moved presentation to the queue
/// (`Queue::present`). Both signatures here take the queue so the runners call
/// this the same way on every leg.
#[cfg(any(wgpu27, wgpu29))]
pub fn present(_queue: &Queue, frame: SurfaceTexture) {
    frame.present();
}
#[cfg(wgpu30)]
pub fn present(queue: &Queue, frame: SurfaceTexture) {
    queue.present(frame);
}

/// Read-map a buffer slice, returning the mapped view. wgpu 27 and 29 return the
/// `BufferView` directly from `get_mapped_range`; wgpu 30 returns a `Result`. The
/// slice must already be mapped for reading (the caller awaited `map_async`), so
/// the 30 arm treats a failure as a bug rather than a runtime condition.
#[cfg(any(wgpu27, wgpu29))]
pub fn mapped_range(slice: BufferSlice<'_>) -> BufferView {
    slice.get_mapped_range()
}
#[cfg(wgpu30)]
pub fn mapped_range(slice: BufferSlice<'_>) -> BufferView {
    slice
        .get_mapped_range()
        .expect("buffer slice was not mapped for reading")
}

/// Build the swapchain [`SurfaceConfiguration`] the built-in winit runners use:
/// a render-attachment surface with empty view formats and a frame latency of 2.
/// The runners share this so the `SurfaceConfiguration` is constructed in one
/// place; the required colour space wgpu 30 added is absorbed here rather than at
/// each runner.
pub fn runner_surface_config(
    format: TextureFormat,
    width: u32,
    height: u32,
    present_mode: PresentMode,
    alpha_mode: CompositeAlphaMode,
) -> SurfaceConfiguration {
    SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode,
        alpha_mode,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        // wgpu 30 added a required surface colour space; earlier legs have no
        // such field. sRGB matches the sRGB swapchain format the runners pick.
        #[cfg(wgpu30)]
        color_space: SurfaceColorSpace::Srgb,
    }
}

/// Adapter-request options for headless device bring-up: no surface, no fallback,
/// the given power preference. Centralises the fields wgpu adds over time (wgpu 30
/// added `apply_limit_buckets`) so headless callers in crates without the version
/// cfg aliases (the testkit, the integration tests) do not spell a version-shaped
/// literal.
#[cfg(any(wgpu27, wgpu29))]
pub fn headless_adapter_options(
    power_preference: PowerPreference,
) -> RequestAdapterOptions<'static, 'static> {
    RequestAdapterOptions {
        power_preference,
        compatible_surface: None,
        force_fallback_adapter: false,
    }
}
#[cfg(wgpu30)]
pub fn headless_adapter_options(
    power_preference: PowerPreference,
) -> RequestAdapterOptions<'static, 'static> {
    RequestAdapterOptions {
        power_preference,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }
}

/// The device feature that enables `@builtin(primitive_index)` in fragment
/// shaders, named per wgpu leg: `SHADER_PRIMITIVE_INDEX` on wgpu 27,
/// `PRIMITIVE_INDEX` on wgpu 29 and 30. Used by the GPU pick pass to resolve the
/// hit triangle for sub-object picking.
#[cfg(wgpu27)]
#[doc(hidden)]
pub const PRIMITIVE_INDEX_FEATURE: Features = Features::SHADER_PRIMITIVE_INDEX;
#[cfg(any(wgpu29, wgpu30))]
#[doc(hidden)]
pub const PRIMITIVE_INDEX_FEATURE: Features = Features::PRIMITIVE_INDEX;

/// The device feature that enables the WGSL `ray_query` extension (hardware BVH
/// traversal). Named the same on every wgpu leg. The path tracer's hardware
/// backend requires it; the portable compute traversal does not.
#[doc(hidden)]
pub const RAY_QUERY_FEATURE: Features = Features::EXPERIMENTAL_RAY_QUERY;

/// The feature set the bindless material-texture path needs: a sampled-texture
/// `binding_array` and non-uniform (per-instance) indexing into it. Named the
/// same on every wgpu leg.
///
/// `PARTIALLY_BOUND_BINDING_ARRAY` is deliberately not in here. The renderer
/// fills every entry of the texture array, binding a neutral fallback view for
/// slots that hold no texture, so the array is never partially bound and a device
/// without that feature can still take this path.
///
/// Support is a property of the adapter, not of the backend: ask the adapter
/// rather than assuming from the platform. Apple silicon reports both through
/// Metal argument buffers, so a Mac takes the bindless path, and WebGPU does not.
/// `recommended_device_features` requests these when the adapter has them, which
/// is what selects
/// [`MaterialTextureBinding::Bindless`](crate::resources::mesh::instanced_bindless::MaterialTextureBinding);
/// [`ViewportRenderer::material_texture_binding`](crate::ViewportRenderer::material_texture_binding)
/// reports which path a renderer actually took.
#[doc(hidden)]
pub const BINDLESS_TEXTURE_FEATURES: Features = Features::TEXTURE_BINDING_ARRAY
    .union(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

// Version-portability helpers, surfaced here (the version seam) so tests and
// consumers that build wgpu pipelines directly can do so without their own
// per-version `#[cfg]`. Each helper is the single place a new wgpu leg grows a
// branch. Defined in `resources::builders`; re-exported here to keep them
// findable next to the alias.
pub use crate::resources::builders::{
    RenderPipelineDesc, dcompare, depth_stencil, dmipmap, dwrite, pipeline_layout, render_pipeline,
    scene_depth_stencil, write_mapped,
};
