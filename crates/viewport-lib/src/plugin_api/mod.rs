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
//! - [`PluginResources`] is what an item type's `init_gpu` builds against:
//!   the group 0 layout (camera, lights, shadows, clip, IBL) every scene
//!   pipeline lists first, the shadow pass's group 0, the sample count, the
//!   shared samplers and the pipeline builders.
//! - [`shared_wgsl`] holds string constants with the standard bind
//!   declarations and shading helpers (`viewport_pbr_shade`,
//!   `viewport_oit_pack`, `viewport_sample_csm`, etc.) so plugin shaders stay
//!   in lockstep with the lib's lighting and transparency contracts.
//! - [`depth`](crate::plugin_api::depth) holds the scene depth convention (reversed: near at 1, far
//!   at 0): the compare functions, clear value and bias a pipeline that
//!   touches scene depth uses. Its shader side is
//!   [`SHARED_DEPTH_WGSL`](crate::plugin_api::shared_wgsl::SHARED_DEPTH_WGSL).
//! - Pipeline builders (`build_opaque_pipeline`, `build_oit_pipeline`, ...)
//!   construct the common variants in one call. Plugins ship one shader and
//!   call a builder per variant.
//!
//! The accessors live on [`crate::resources::DeviceResources`], which the
//! per-frame hooks receive; `init_gpu` reaches them through
//! [`PluginResources`].
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
//! - **Group-0 binding indices are additive-only.** The bindings
//!   [`SHARED_BINDINGS_WGSL`](crate::plugin_api::shared_wgsl::SHARED_BINDINGS_WGSL)
//!   declares keep their numbers forever. New bindings are appended at the
//!   next free index. This is the strongest guarantee in the API: a plugin pipeline
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
//!
//! [`OpaqueTargetDesc`]: crate::plugin_api::OpaqueTargetDesc
//! [`ForegroundTargetDesc`]: crate::plugin_api::ForegroundTargetDesc
//! [`OitTargetDesc`]: crate::plugin_api::OitTargetDesc
//! [`DepthReadTargetDesc`]: crate::plugin_api::DepthReadTargetDesc
//! [`MaskTargetDesc`]: crate::plugin_api::MaskTargetDesc
//! [`PickTargetDesc`]: crate::plugin_api::PickTargetDesc
//! [`ShadowTargetDesc`]: crate::plugin_api::ShadowTargetDesc
//! [`PluginResources`]: crate::plugin_api::PluginResources
//! [`shared_wgsl`]: crate::plugin_api::shared_wgsl

pub mod builders;
pub mod cull;
pub mod depth;
pub mod install;
pub mod item_type;
pub mod overlay_fill;
mod per_viewport;
pub mod pick_helpers;
mod plugin_resources;
pub mod post_effect;
pub mod shared_wgsl;
pub mod target_desc;
pub mod uploads;
pub mod wgsl_compose;
pub mod writes;

pub use crate::resources::pipeline_slot::LazyFamily as LazyPipelines;
pub use crate::resources::pipeline_slot::LazyModule;
pub use crate::resources::{PipelineBuilder, PluginPipelineOpts, ShadowCaster};
pub use cull::{BatchMeta, CullSubmission, InstanceAabb, SingleMeshDraw};
pub use install::{
    PluginInstallCtx, PluginInstaller, install_plugin, reattach_plugin, uninstall_plugin,
};
pub use item_type::{
    AsAnyItemTypePlugin, DepthReadContext, EncoderScope, EncoderScopeContext, ItemCollections,
    ItemFrameContext, ItemTypeHost, ItemTypeId, ItemTypePlugin, LightContext, OutlineMaskContext,
    PaintContext, PainterBatch, PainterContext, PainterLightLayers, PainterMask,
    PainterMaskContext, PainterRun, PickContext, PickPassContext, PickRay, PluginItem,
    PluginItemCollection, RectPickContext, SURFACE_MASK_LAYERS, ShadowCastContext,
    SurfaceMaskContext, surface_mask_bits,
};
pub use overlay_fill::OverlayFillDesc;
pub use per_viewport::PerViewport;
pub use plugin_resources::PluginResources;
pub use post_effect::{
    PostEffectContext, PostEffectProducer, PostEffectProducerId, PostEffectResizeContext,
    PostEffectSlot, PostEffectStage, PostEffectStageId, build_post_effect_pipeline,
};
pub use target_desc::{
    DepthReadTargetDesc, ForegroundTargetDesc, MaskTargetDesc, OIT_ACCUM_BLEND, OIT_REVEAL_BLEND,
    OitTargetDesc, OpaqueTargetDesc, PickTargetDesc, SceneColourTargets, ShadowTargetDesc,
    indirect_fraction_target,
};
pub use uploads::{Handles, Uploads};
pub use writes::{Channel, Extent, Sourced, Span, Writes};
