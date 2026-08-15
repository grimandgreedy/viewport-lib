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
    Aabb, AlphaMode, BackfacePolicy, Camera, DecalItem, GaussianSplatData, GaussianSplatItem,
    GlyphItem, GlyphType, ImageAnchor, ImageSliceItem, IndirectLightSource, ItemSettings,
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

/// Create a headless wgpu device + queue for testing.
pub fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = viewport_lib::wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// Headless device with `SHADER_PRIMITIVE_INDEX` enabled, or `None` when no
/// adapter is available or the adapter does not support the feature. Used by the
/// GPU sub-object tests that read the pick pass's triangle-index channel.
pub fn headless_device_with_primitive_index() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = viewport_lib::wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    if !adapter
        .features()
        .contains(viewport_lib::gpu::PRIMITIVE_INDEX_FEATURE)
    {
        return None;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("test-primitive-index"),
        required_features: viewport_lib::gpu::PRIMITIVE_INDEX_FEATURE,
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// Headless device with `max_bind_groups` capped at 2, or `None` when no
/// adapter is available. Mirrors the device iced_wgpu 0.14 requests (it
/// hardcodes this limit for WebGL2 portability and gives a consumer no way to
/// raise it), so any draw site that unconditionally binds group index 2 fails
/// wgpu validation against this device the same way it would against iced's.
/// See `docs/issues/iced-max-bind-groups-2-draw-path-incomplete.md` and
/// `docs/plans/iced-two-bind-group-support-plan.md`.
pub fn headless_device_limited_bind_groups() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = viewport_lib::wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let mut limits = wgpu::Limits::default();
    limits.max_bind_groups = 2;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("test-2-bind-groups"),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
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
