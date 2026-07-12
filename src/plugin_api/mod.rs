//! Plugin substrate: target descriptors, shared bind layouts, WGSL helpers,
//! and pipeline builders.
//!
//! This module publishes the pieces a plugin needs to build pipelines that
//! drop into the lib's existing render passes. A plugin reuses:
//!
//! - Target descriptors ([`OpaqueTargetDesc`], [`OitTargetDesc`],
//!   [`MaskTargetDesc`], [`PickTargetDesc`], [`ShadowTargetDesc`]) describe
//!   the render-target formats, blend states, and depth-stencil state each
//!   lib pass expects. Pipelines built against these are compatible with the
//!   corresponding pass.
//! - [`SharedBindings`] is the group-0 bind layout (camera, lights, shadows,
//!   clip, IBL) shared by every scene pipeline. Plugin pipeline layouts list
//!   it as group 0.
//! - [`shared_wgsl`] holds string constants with the standard bind
//!   declarations and shading helpers (`viewport_pbr_shade`,
//!   `viewport_oit_pack`, `viewport_sample_csm`, etc.) so plugin shaders stay
//!   in lockstep with the lib's lighting and transparency contracts.
//! - Pipeline builders on [`crate::resources::DeviceResources`]
//!   (`build_opaque_pipeline`, `build_oit_pipeline`, ...) construct the
//!   common variants in one call. Plugins ship one shader and call a
//!   builder per variant.
//!
//! All accessors live on [`crate::resources::DeviceResources`].
//!
//! # Compatibility policy
//!
//! Pre-1.0, this surface evolves more freely than a stable crate would, but
//! the policy below describes what plugins can rely on within a given minor
//! version and how breakage is signalled.
//!
//! Plugins are expected to track `viewport-lib` minor versions. The
//! convention is that a minor bump may rename or remove items the audit
//! flags as non-additive; a patch bump never does.
//!
//! - **Group-0 binding indices ([`SharedBindings`]) are additive-only.**
//!   The constants in `SharedBindings` (`CAMERA_BINDING`, ...) keep their
//!   numeric values forever. New bindings are appended at the next free
//!   index. This is the strongest guarantee in the API: a plugin pipeline
//!   built once stays valid as new bindings are added.
//!
//! - **WGSL helper strings in [`shared_wgsl`] are stable within a minor
//!   version.** Helper *function signatures* (`viewport_pbr_shade`,
//!   `viewport_oit_pack`, `viewport_sample_csm`, ...) and the struct
//!   layouts they declare may change with a minor bump. When a helper's
//!   signature changes in an incompatible way, the helper is renamed (the
//!   old name is removed) so a plugin shader that referenced the old name
//!   fails to compile loudly rather than silently producing wrong output.
//!   Additive changes (new helpers, new optional fields appended to
//!   structs) ride patch bumps.
//!
//! - **Target descriptors and pipeline builder signatures are stable within
//!   a minor version.** Format constants (`HDR_COLOR_FORMAT`, blend
//!   states, depth-stencil state) follow the same rule: any change rides a
//!   minor bump and is noted in the CHANGELOG.
//!
//! - **The deformer registry hook contract is additive-only.** The
//!   `DeformVertex` and `DeformContext` struct shapes (see
//!   [`crate::resources::mesh_sidecar::registry::DeformerDesc`]) keep
//!   existing fields stable across releases; new fields may be appended.
//!   The composition-order policy (ObjectSpace before WorldSpace, priority
//!   ascending within stage) is part of the contract and will not change.
//!
//! - **Builder methods on [`crate::resources::DeviceResources`] that
//!   construct pipelines for plugins** (`build_opaque_pipeline`,
//!   `build_oit_pipeline`, ...) keep their behaviour stable within a minor
//!   version. Signature changes ride minor bumps and are listed in the
//!   CHANGELOG.
//!
//! Anything not listed above is internal: a plugin that reaches past the
//! published surface (private modules, undocumented constants) may break at
//! any release.

pub mod cull;
pub mod item_type;
pub mod shared_wgsl;
pub mod target_desc;

pub use cull::{BatchMeta, CullSubmission, InstanceAabb, SingleMeshDraw};
pub use item_type::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickPassContext, PickRay,
    PluginItemCollection, ShadowCastContext,
};
pub use target_desc::{
    MaskTargetDesc, OIT_ACCUM_BLEND, OIT_REVEAL_BLEND, OitTargetDesc, OpaqueTargetDesc,
    PickTargetDesc, ShadowTargetDesc,
};

/// Group-0 bind layout shared by every scene pipeline.
///
/// Plugin pipeline layouts must list this layout as group 0 so the lib's bound
/// camera / lights / shadow / clip / IBL resources are visible to the plugin's
/// shader. Bindings 0-13 match [`shared_wgsl::SHARED_BINDINGS_WGSL`]; do not
/// re-declare them in plugin WGSL.
///
/// Obtain a reference via
/// [`DeviceResources::shared_bindings`](crate::resources::DeviceResources::shared_bindings).
pub struct SharedBindings<'a> {
    /// The group-0 `BindGroupLayout`. Pass by reference when calling
    /// `device.create_pipeline_layout`.
    pub group0_layout: &'a crate::gpu::BindGroupLayout,
}

impl<'a> SharedBindings<'a> {
    /// Binding indices inside group 0. Stable across releases: additive
    /// changes only. See [`shared_wgsl::SHARED_BINDINGS_WGSL`] for the WGSL
    /// declarations.
    pub const CAMERA_BINDING: u32 = 0;
    /// Shadow atlas depth texture.
    pub const SHADOW_ATLAS_BINDING: u32 = 1;
    /// Comparison sampler used for PCF shadow filtering.
    pub const SHADOW_SAMPLER_BINDING: u32 = 2;
    /// Lights header uniform (count, hemisphere, IBL toggles).
    pub const LIGHTS_HEADER_BINDING: u32 = 3;
    /// Clip planes uniform (section-view planes).
    pub const CLIP_PLANES_BINDING: u32 = 4;
    /// Shadow info uniform (CSM matrices, splits, PCSS params).
    pub const SHADOW_INFO_BINDING: u32 = 5;
    /// Clip volume uniform (box/sphere/cylinder regions).
    pub const CLIP_VOLUME_BINDING: u32 = 6;
    /// IBL irradiance equirect texture.
    pub const IBL_IRRADIANCE_BINDING: u32 = 7;
    /// IBL prefiltered specular equirect texture.
    pub const IBL_SPECULAR_BINDING: u32 = 8;
    /// BRDF integration LUT.
    pub const IBL_BRDF_LUT_BINDING: u32 = 9;
    /// IBL filtering sampler (linear, clamp-to-edge).
    pub const IBL_SAMPLER_BINDING: u32 = 10;
    /// Skybox / environment equirect texture (full-resolution).
    pub const SKYBOX_BINDING: u32 = 11;
    /// Per-fragment debug storage buffer.
    pub const DEBUG_FRAG_BINDING: u32 = 12;
    /// Lights array storage buffer.
    pub const LIGHTS_ARRAY_BINDING: u32 = 13;
}
