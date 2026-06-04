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
//! - Pipeline builders on [`crate::resources::ViewportGpuResources`]
//!   (`build_opaque_pipeline`, `build_oit_pipeline`, ...) construct the
//!   common variants in one call. Plugins ship one shader and call a
//!   builder per variant.
//!
//! All accessors live on [`crate::resources::ViewportGpuResources`].

pub mod cull;
pub mod item_type;
pub mod shared_wgsl;
pub mod target_desc;

pub use cull::{CullSubmission, InstanceAabb};
pub use item_type::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickRay,
    PluginItemCollection,
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
/// [`ViewportGpuResources::shared_bindings`](crate::resources::ViewportGpuResources::shared_bindings).
pub struct SharedBindings<'a> {
    /// The group-0 `BindGroupLayout`. Pass by reference when calling
    /// `device.create_pipeline_layout`.
    pub group0_layout: &'a wgpu::BindGroupLayout,
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
