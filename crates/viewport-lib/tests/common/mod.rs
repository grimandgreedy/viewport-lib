//! Shared helpers for the headless integration tests.
//!
//! The headless suite is split across several `tests/headless_*.rs` files, each
//! a separate test binary. This module holds the pieces they all need: the
//! wgpu device constructors and a unit box mesh. Every headless file pulls it in
//! with `mod common;` + `use common::*;`.

// Each headless_*.rs binary pulls in this whole module but only uses part of it:
// device constructors it does not call read as dead code, and re-exported names
// it does not name read as unused imports. Both are expected for a shared module.
#![allow(dead_code, unused_imports)]

// On the 27 leg the plain `wgpu` dependency is active and `wgpu::` resolves to
// it directly. On the 29 leg that dependency is inactive, so name wgpu through
// the library's re-export instead, which tracks whichever leg is built.
#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

// Re-export the library types the headless files reach for, so a single
// `use common::*;` covers the common set. Unused names from a glob import do not
// warn, so each file only pays for what it actually references.
pub use viewport_lib::{
    Aabb, AlphaMode, AnchorX, AnchorY, BackfacePolicy, Camera, DecalItem, GaussianSplatData,
    GaussianSplatItem, GlyphItem, GlyphType, ImageSliceItem, IndirectLightSource, ItemSettings,
    LightKind, LightSource, Material, MeshId, OverrideBufferSlice, PickBackend, PickId, PickMask,
    PickPoll, PointCloudItem, PolylineItem, RibbonItem, ScatterVolume, ScatterVolumeItem, Scene,
    ScreenImageItem, Selection, ShDegree, ShadingModel, SliceAxis, SpriteItem, SpriteSizeMode,
    VolumeItem, VolumeMeshItem, VolumeSurfaceSliceItem,
    error::ViewportError,
    plugin_api::{
        ItemTypePlugin, PickPassContext, PluginItemCollection, SharedBindings,
        shared_wgsl::SHARED_PICK_WGSL,
    },
    renderer::{FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission, ViewportRenderer},
    resources::{MeshData, PICK_COLOR_FORMAT, PICK_DEPTH_CHANNEL_FORMAT, SCENE_DEPTH_FORMAT},
};

// Every headless device below comes from the testkit's one parameterised
// constructor (`headless_device_with`), so there is a single copy of the adapter
// request, limits policy, and feature negotiation. Each function here just names
// the `DeviceProfile` its suite needs.
use viewport_lib_testkit::{DeviceProfile, headless_device_with};

/// Create a headless wgpu device + queue for testing.
pub fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(&DeviceProfile::low_power("test"))
}

/// Headless device with `INDIRECT_FIRST_INSTANCE` enabled (plus
/// `MULTI_DRAW_INDIRECT_COUNT` when the adapter offers it), or `None` when no
/// adapter is available or the adapter lacks `INDIRECT_FIRST_INSTANCE`. Used by
/// tests that exercise the GPU-culled indirect draw path and the multi-draw
/// collapse. Metal advertises `INDIRECT_FIRST_INSTANCE` but not the count
/// feature, so the collapse there runs emulated (still bit-identical output).
pub fn headless_device_with_indirect() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(
        &DeviceProfile::low_power("test-indirect")
            .require(wgpu::Features::INDIRECT_FIRST_INSTANCE)
            .optional(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT),
    )
}

/// Headless device with `SHADER_PRIMITIVE_INDEX` enabled, or `None` when no
/// adapter is available or the adapter does not support the feature. Used by the
/// GPU sub-object tests that read the pick pass's triangle-index channel.
pub fn headless_device_with_primitive_index() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(
        &DeviceProfile::low_power("test-primitive-index")
            .require(viewport_lib::gpu::PRIMITIVE_INDEX_FEATURE),
    )
}

/// Headless device with `max_bind_groups` capped at 2, or `None` when no
/// adapter is available. Mirrors the device iced_wgpu 0.14 requests (it
/// hardcodes this limit for WebGL2 portability and gives a consumer no way to
/// raise it), so any draw site that unconditionally binds group index 2 fails
/// wgpu validation against this device the same way it would against iced's.
/// See `docs/issues/iced-max-bind-groups-2-draw-path-incomplete.md` and
/// `docs/plans/iced-two-bind-group-support-plan.md`.
pub fn headless_device_limited_bind_groups() -> Option<(wgpu::Device, wgpu::Queue)> {
    // Recommended limits (viewport-lib needs more storage buffers per stage than
    // wgpu's default) with bind groups additionally capped at 2.
    headless_device_with(&DeviceProfile::low_power("test-2-bind-groups").max_bind_groups(2))
}

/// A headless device requested with exactly `ViewportRenderer::recommended_device_limits`
/// (not the adapter's full caps), so building a renderer here exercises the path
/// a limits-following consumer takes. `ViewportRenderer::new` asserts the device
/// meets its storage-buffer requirement, so a device under that requirement
/// panics with a clear message on any backend (Metal does not enforce the limit
/// at pipeline-layout creation, so the explicit assert is what catches it).
/// Returns `None` when no adapter is available.
pub fn headless_device_recommended_limits() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(
        &DeviceProfile::high_performance("test-recommended-limits").with_recommended_features(),
    )
}

/// A headless device/queue for the feature-gated bake and raytrace suites, or
/// `None` when no adapter is available. Requests a high-performance adapter and
/// recommended limits, named through `viewport_lib::gpu` to match those suites'
/// APIs (the same wgpu types the `wgpu` re-export names). Separate from
/// [`headless_device`], which the renderer tests use with a low-power adapter.
pub fn device_queue() -> Option<(viewport_lib::gpu::Device, viewport_lib::gpu::Queue)> {
    headless_device_with(&DeviceProfile::high_performance("test-device-queue"))
}

/// Simple unit box mesh data for testing.
pub fn box_mesh() -> MeshData {
    let positions = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let normals = vec![
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    let indices = vec![
        0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 3, 7, 7, 4, 0, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7, 3,
        0, 4, 5, 5, 1, 0,
    ];
    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}
