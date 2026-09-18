//! Plugin substrate: target descriptors, shared bind layouts, WGSL helpers,
//! and pipeline builders.
//!
//! This module publishes the pieces a plugin needs to build pipelines that
//! drop into the lib's existing render passes. A plugin reuses:
//!
//! - Target descriptors ([`OpaqueTargetDesc`], [`ForegroundTargetDesc`],
//!   [`OitTargetDesc`], [`DepthReadTargetDesc`], [`MaskTargetDesc`],
//!   [`PickTargetDesc`], [`ShadowTargetDesc`]) describe
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
//! # Who owns what
//!
//! **A plugin owns its GPU data.** Buffers, textures, bind group layouts and
//! pipelines belong to the plugin, created from the `&Device` its hooks are
//! given and held in whatever structure suits the type. The library does not
//! supply a storage abstraction, and a plugin does not need one: a `HashMap`
//! keyed by the item's `PickId`, a `Vec` indexed by a version stamp, or a
//! slot map from crates.io all work, and which one is right depends on how the
//! type is submitted.
//!
//! What the library does supply is the part a plugin cannot build for itself:
//!
//! - **Off-thread work.** [`ItemFrameContext::jobs`](crate::plugin_api::ItemFrameContext#structfield.jobs) runs CPU work on a
//!   background worker and delivers the result to a later `prepare`, on the
//!   same runner the built-in uploads use. Reach for it rather than building
//!   geometry on the frame thread.
//! - **Content shared between item types.** Meshes, textures, 3D volumes and
//!   colourmaps are read by more than one type, so the library owns them and
//!   publishes readers (`texture_view`, `volume_view`, `colourmap_view`,
//!   `fallback_texture_view`, [`MeshDraw`](crate::resources::MeshDraw)).
//!   Upload through the renderer; do not copy them into the plugin.
//! - **Visibility to an eviction budget.** Report what a plugin holds from
//!   [`ItemTypePlugin::resident_bytes`](crate::plugin_api::ItemTypePlugin::resident_bytes), so a host sizing a working set can
//!   see it. Content a plugin holds is invisible to the library otherwise.
//! - **A route back from the host.**
//!   [`ViewportRenderer::item_type_plugin_mut`](crate::renderer::ViewportRenderer::item_type_plugin_mut)
//!   returns a registered plugin as its concrete type, so a host can upload
//!   into, configure, or read back from content the plugin stores itself.
//! - **Notice that a shared resource went away.** A bind group holding a
//!   `TextureView` keeps that texture alive after the host frees it, and does
//!   not see a replacement swapped in behind a live id. A plugin cannot work
//!   either event out from the ids it holds, so
//!   [`ResourceGate`](crate::resources::ResourceGate) answers it: poll it at
//!   the top of `prepare` and rebind the entries its
//!   [`Revalidate`](crate::resources::Revalidate) verdict names.
//!
//! Two submission shapes both work, and the choice is the plugin's. Carry the
//! geometry on the item behind a version stamp and rebuild when the stamp
//! changes, which needs no upload call at all; or take an upload call that
//! returns a handle and have per-frame items name the handle, which avoids
//! resubmitting geometry every frame. The built-in types use the second
//! because their items are submitted every frame.
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
//! - **[`PluginInstallCtx`] grows additively.** The struct is
//!   `#[non_exhaustive]`, so new borrows are appended as fields and existing
//!   ones keep their names and types within a minor version. Construct it
//!   through [`PluginInstallCtx::new`]; adding a field is a minor bump noted in
//!   the CHANGELOG. [`PluginInstaller::install`] returns a
//!   [`crate::ViewportResult`], so a feature that needs a piece the context did
//!   not carry fails with [`crate::ViewportError`] rather than panicking.
//!
//! Anything not listed above is internal: a plugin that reaches past the
//! published surface (private modules, undocumented constants) may break at
//! any release.
//!
//! # wgpu version
//!
//! This surface hands out raw wgpu types (`RenderPipeline`, `BindGroupLayout`,
//! `RenderPass`, ...) because a plugin builds its own pipelines. Those types
//! are the wgpu version the library was built against, which is selected by the
//! `wgpu27` / `wgpu29` cargo features. Name wgpu through
//! [`viewport_lib::wgpu`](crate::wgpu) rather than an external `wgpu` crate so a
//! plugin follows whichever version the build chose; a plugin that names its own
//! `wgpu` dependency is coupled to one version and must match the library's.

pub mod cull;
pub mod install;
pub mod item_type;
pub mod pick_helpers;
pub mod post_effect;
pub mod shared_wgsl;
pub mod target_desc;

pub use cull::{BatchMeta, CullSubmission, InstanceAabb, SingleMeshDraw};
pub use install::{PluginInstallCtx, PluginInstaller, install_plugin};
pub use item_type::{
    AsAnyItemTypePlugin, DepthReadContext, EncoderScope, EncoderScopeContext, ItemFrameContext,
    ItemTypeHost, ItemTypePlugin, LightContext, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, RectPickContext, ShadowCastContext,
};
pub use post_effect::{
    PostEffectContext, PostEffectProducer, PostEffectProducerId, PostEffectResizeContext,
    PostEffectSlot, PostEffectStage, PostEffectStageId, build_post_effect_pipeline,
};
pub use target_desc::{
    DepthReadTargetDesc, ForegroundTargetDesc, MaskTargetDesc, OIT_ACCUM_BLEND, OIT_REVEAL_BLEND,
    OitTargetDesc, OpaqueTargetDesc, PickTargetDesc, ShadowTargetDesc,
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
    /// MSAA sample count of the renderer's HDR scene target, as configured by
    /// [`ViewportRenderer::with_sample_count`](crate::renderer::ViewportRenderer::with_sample_count)
    /// (1 when MSAA is off).
    ///
    /// A plugin that hand-rolls a render pipeline (rather than building through
    /// the `DeviceResources::build_*_pipeline` helpers, which already apply
    /// this) must set its `multisample.count` to this value for the HDR colour
    /// passes: `paint`, `paint_transparent`, and `paint_depth_read`. The stock
    /// helpers read the same field, so pipelines built through them need no
    /// change.
    ///
    /// This applies to the HDR colour passes only. The shadow, pick, and
    /// outline-mask passes always render single-sampled (the library's own
    /// descriptors peg them to 1), so a plugin's `cast_shadow_pass`,
    /// `render_pick`, and `outline_mask` pipelines use `sample_count: 1`
    /// regardless of this value.
    pub sample_count: u32,
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
