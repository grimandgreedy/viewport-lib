//! Plugin-facing accessors and pipeline builders on [`DeviceResources`].
//!
//! See [`crate::plugin_api`] for the published types these methods return.

use crate::plugin_api::{
    DepthReadTargetDesc, ForegroundTargetDesc, MaskTargetDesc, OitTargetDesc, OpaqueTargetDesc,
    PickTargetDesc, ShadowTargetDesc,
    target_desc::{OIT_ACCUM_BLEND, OIT_REVEAL_BLEND},
};
use crate::resources::DeviceResources;
use crate::resources::mesh::mesh_store::MeshId;

const NO_LIT_GROUP0: &str = "item types register only where there is a lit group 0";

/// HDR colour format used by the scene buffer. Plugins targeting the HDR
/// path build pipelines against this format.
pub const HDR_COLOR_FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::Rgba16Float;

/// Format of the HDR scene pass's second colour target, the indirect
/// fraction: the indirect share of each opaque pixel's light in red and the
/// direct share in green (see
/// [`OpaqueTargetDesc`](crate::plugin_api::OpaqueTargetDesc)).
pub const INDIRECT_FRACTION_FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::Rg8Unorm;

/// The default scene depth-stencil format, `Depth24PlusStencil8`.
///
/// A renderer configured with
/// [`RendererConfig::scene_depth_format`](crate::RendererConfig::scene_depth_format)
/// (as `ViewportApp` and the testkit harness are on a device with
/// `Features::DEPTH32FLOAT_STENCIL8`) uses another one: build pipelines against
/// [`DeviceResources::scene_depth_format`] or a target descriptor's
/// `depth_format`, not this constant.
pub const SCENE_DEPTH_FORMAT: crate::gpu::TextureFormat =
    crate::gpu::TextureFormat::Depth24PlusStencil8;

/// Shadow atlas depth format.
pub const SHADOW_DEPTH_FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::Depth32Float;

/// Outline-mask colour format.
pub const MASK_COLOR_FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::R8Unorm;

/// Pick-id colour format, shared by the object-id (`@location(0)`) and
/// primitive-id (`@location(1)`) pick targets.
pub const PICK_COLOR_FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::R32Uint;

/// Pick depth-channel format (`@location(2)`): the fragment framebuffer `z`
/// written as a float so the renderer can reconstruct world position on
/// read-back.
pub const PICK_DEPTH_CHANNEL_FORMAT: crate::gpu::TextureFormat =
    crate::gpu::TextureFormat::R32Float;

impl DeviceResources {
    // ------------------------------------------------------------------
    // Group 0, target descriptors and the builder inputs
    // ------------------------------------------------------------------

    /// A set of `N` pipelines built on first read under this renderer's
    /// compilation policy, for an item type's own pipelines. `ctx` carries
    /// everything a build needs by value; `build` makes member `i` from it.
    ///
    /// An item type that keeps its pipelines here follows
    /// `ViewportRenderer::set_pipeline_compilation` like the built-in ones:
    /// under `Background` its draws skip until their pipeline is ready, and
    /// its compiles count in `pipelines_pending`.
    pub fn lazy_pipelines<C: Send + Sync + 'static, const N: usize>(
        &self,
        ctx: C,
        build: fn(&C, usize) -> crate::gpu::RenderPipeline,
    ) -> crate::plugin_api::LazyPipelines<C, N> {
        crate::resources::pipeline_slot::LazyFamily::new(
            ctx,
            std::sync::Arc::clone(&self.pipeline_compiler),
            build,
        )
    }

    /// [`lazy_pipelines`](Self::lazy_pipelines) for compute pipelines. A
    /// dispatch whose pipeline is still compiling is skipped, along with
    /// whatever reads its output that frame.
    pub fn lazy_compute_pipelines<C: Send + Sync + 'static, const N: usize>(
        &self,
        ctx: C,
        build: fn(&C, usize) -> crate::gpu::ComputePipeline,
    ) -> crate::plugin_api::LazyPipelines<C, N, crate::gpu::ComputePipeline> {
        crate::resources::pipeline_slot::LazyFamily::new(
            ctx,
            std::sync::Arc::clone(&self.pipeline_compiler),
            build,
        )
    }

    /// A shader module compiled the first time a pipeline build asks for it,
    /// so it can sit in a [`LazyPipelines`](crate::plugin_api::LazyPipelines)
    /// context and compile on the worker with the first member that reads
    /// it. Handles made from the same source share one compile, with each
    /// other and with the renderer's own.
    pub fn lazy_module(
        &self,
        device: &crate::gpu::Device,
        label: &str,
        source: impl AsRef<str>,
    ) -> crate::plugin_api::LazyModule {
        self.shared_module(device, label, source.as_ref())
    }

    /// The pipeline builders' inputs, by value, for a build that runs later
    /// or on another thread.
    pub fn pipeline_builder(&self) -> PipelineBuilder {
        PipelineBuilder {
            camera_bgl: self.group0_layout().clone(),
            shadow_camera_bgl: self.shadow_group0_layout().clone(),
            opaque: self.opaque_target_desc(),
            ldr_opaque: self.ldr_opaque_target_desc(),
            foreground: self.foreground_target_desc(),
            oit: self.oit_target_desc(),
            depth_read: self.depth_read_target_desc(),
            mask: self.mask_target_desc(),
            pick: self.pick_target_desc(),
            shadow: self.shadow_target_desc(),
            shadow_depth_clamp: self.capabilities.depth_clip_control,
        }
    }

    /// Group 0 of every scene pipeline: camera, lights, shadows, clip and
    /// IBL, from whichever tier the device runs. List it as group 0 of a
    /// pipeline layout; the WGSL declarations are
    /// [`SHARED_BINDINGS_WGSL`](crate::plugin_api::shared_wgsl::SHARED_BINDINGS_WGSL).
    ///
    /// Panics on the overlay-only tier, which has no lit group 0 and accepts
    /// no item types.
    pub fn group0_layout(&self) -> &crate::gpu::BindGroupLayout {
        self.lit_group0_layouts().expect(NO_LIT_GROUP0).0
    }

    /// Group 0 of the shadow pass: the cascade light matrix the renderer
    /// binds before calling
    /// [`cast_shadow_pass`](crate::plugin_api::ItemTypePlugin::cast_shadow_pass).
    /// List it as group 0 of a shadow pipeline layout instead of building a
    /// copy, which stops matching when the library adds a binding to it.
    ///
    /// Panics on the overlay-only tier, like [`group0_layout`](Self::group0_layout).
    pub fn shadow_group0_layout(&self) -> &crate::gpu::BindGroupLayout {
        self.lit_group0_layouts().expect(NO_LIT_GROUP0).1
    }

    /// Adapt a shader source that includes the shared group 0 declarations
    /// to this device, before handing it to `create_shader_module`.
    ///
    /// On the compatibility
    /// tier the light list at binding 13 is a uniform array of
    /// [`COMPATIBILITY_MAX_LIGHTS`](crate::resources::COMPATIBILITY_MAX_LIGHTS) and the
    /// cluster lookups in `apply_scene_lighting` walk every light, since that
    /// tier's group 0 has no storage buffers. Modules built through
    /// [`lazy_module`](Self::lazy_module) are adapted already.
    pub fn adapt_wgsl<'s>(&self, source: &'s str) -> std::borrow::Cow<'s, str> {
        crate::resources::builders::adapt_wgsl(source, &self.capabilities)
    }

    /// Render-target descriptor for the HDR opaque scene pass.
    pub fn opaque_target_desc(&self) -> OpaqueTargetDesc {
        OpaqueTargetDesc {
            color_format: HDR_COLOR_FORMAT,
            indirect_fraction_format: Some(INDIRECT_FRACTION_FORMAT),
            depth_format: self.scene_depth_format(),
            sample_count: self.sample_count,
        }
    }

    /// Render-target descriptor for the LDR scene pass
    /// (`PipelineMode::Direct`): the renderer's configured output format
    /// instead of the HDR scene format, otherwise identical to
    /// [`opaque_target_desc`](Self::opaque_target_desc). A plugin that opts
    /// into LDR painting via
    /// [`ItemTypePlugin::draws_ldr`](crate::plugin_api::ItemTypePlugin::draws_ldr)
    /// builds its second `paint` pipeline against this and selects it when
    /// [`PaintContext::target_format`](crate::plugin_api::PaintContext::target_format)
    /// matches.
    pub fn ldr_opaque_target_desc(&self) -> OpaqueTargetDesc {
        OpaqueTargetDesc {
            color_format: self.target_format,
            indirect_fraction_format: None,
            depth_format: self.scene_depth_format(),
            sample_count: self.sample_count,
        }
    }

    /// Render-target descriptor for the foreground pass.
    pub fn foreground_target_desc(&self) -> ForegroundTargetDesc {
        ForegroundTargetDesc {
            color_format: HDR_COLOR_FORMAT,
            indirect_fraction_format: INDIRECT_FRACTION_FORMAT,
            depth_format: self.scene_depth_format(),
            sample_count: 1,
        }
    }

    /// Render-target descriptor for the OIT pass (MRT: accum + reveal).
    pub fn oit_target_desc(&self) -> OitTargetDesc {
        OitTargetDesc {
            accum_format: HDR_COLOR_FORMAT,
            reveal_format: MASK_COLOR_FORMAT,
            depth_format: self.scene_depth_format(),
            accum_blend: OIT_ACCUM_BLEND,
            reveal_blend: OIT_REVEAL_BLEND,
            sample_count: self.sample_count,
        }
    }

    /// Render-target descriptor for the read-only-depth plugin pass.
    pub fn depth_read_target_desc(&self) -> DepthReadTargetDesc {
        DepthReadTargetDesc {
            color_format: HDR_COLOR_FORMAT,
            depth_format: self.scene_depth_format(),
            sample_count: self.sample_count,
        }
    }

    /// Render-target descriptor for the outline-mask pass.
    pub fn mask_target_desc(&self) -> MaskTargetDesc {
        MaskTargetDesc {
            color_format: MASK_COLOR_FORMAT,
            depth_format: self.scene_depth_format(),
            sample_count: 1,
        }
    }

    /// Render-target descriptor for the pick-id pass.
    pub fn pick_target_desc(&self) -> PickTargetDesc {
        PickTargetDesc {
            object_id_format: PICK_COLOR_FORMAT,
            primitive_id_format: PICK_COLOR_FORMAT,
            depth_channel_format: PICK_DEPTH_CHANNEL_FORMAT,
            depth_stencil_format: self.scene_depth_format(),
            sample_count: 1,
        }
    }

    /// Render-target descriptor for the shadow-atlas pass.
    pub fn shadow_target_desc(&self) -> ShadowTargetDesc {
        ShadowTargetDesc {
            depth_format: SHADOW_DEPTH_FORMAT,
            sample_count: 1,
        }
    }

    // ------------------------------------------------------------------
    // Texture-id namespace accessors
    // ------------------------------------------------------------------

    /// Borrow the `TextureView` for a texture previously uploaded via
    /// [`upload_texture`](Self::upload_texture) or
    /// [`upload_normal_map`](Self::upload_normal_map).
    ///
    /// Returns `None` if `id` does not refer to a live texture (a stale handle
    /// whose texture was freed, or one out of range).
    ///
    /// Lifetime contract: the returned view is valid until the texture is freed
    /// with [`free_texture`](Self::free_texture). Plugins that build a bind
    /// group from this view must rebuild it after any operation that could
    /// invalidate the texture (a free, device recreation). A safer pattern is to
    /// fetch the view each frame just before building / rebuilding the bind
    /// group.
    pub fn texture_view(
        &self,
        id: crate::resources::TextureId,
    ) -> Option<&crate::gpu::TextureView> {
        self.content.textures.get(id).map(|t| &t.view)
    }

    /// Borrow the sampler the texture was uploaded with.
    ///
    /// Most user textures are uploaded with a shared linear-repeat sampler;
    /// prefer [`material_sampler`](Self::material_sampler) when you need
    /// the shared lib sampler rather than the per-texture instance.
    pub fn texture_sampler(&self, id: crate::resources::TextureId) -> Option<&crate::gpu::Sampler> {
        self.content.textures.get(id).map(|t| &t.sampler)
    }

    /// The 1x1 neutral view the lib binds when a material slot names no
    /// texture: white for albedo and AO, a flat tangent-space normal, `[0, 1,
    /// 1]` for metallic-roughness so the scalar factors pass through, and black
    /// for emissive.
    ///
    /// Bind it wherever a pipeline layout requires a texture but the item has
    /// none, so the same layout is honoured either way and the slot contributes
    /// nothing of its own. Pair it with
    /// [`material_sampler`](Self::material_sampler).
    pub fn fallback_texture_view(
        &self,
        slot: crate::scene::material::TextureSlot,
    ) -> &crate::gpu::TextureView {
        self.scene_tier.material.slot_view(slot)
    }

    /// Shared linear-repeat sampler used by the lib's material pipelines.
    ///
    /// Use this when building a plugin bind group that samples user
    /// textures the same way `Material` does (linear filter, repeat wrap).
    pub fn material_sampler(&self) -> &crate::gpu::Sampler {
        match &self.compatibility_tier {
            Some(compat) => &compat.material_sampler,
            None => &self.scene_tier.material.sampler,
        }
    }

    /// Non-filtering clamp sampler for the read-only-depth pass.
    ///
    /// Pair it with the scene depth-only view to build a bind group matching
    /// [`depth_read_bind_group_layout`](Self::depth_read_bind_group_layout).
    /// The renderer already builds that bind group each frame and hands it
    /// over as
    /// [`DepthReadContext::scene_depth_bind_group`](crate::plugin_api::DepthReadContext::scene_depth_bind_group);
    /// use this only when building your own.
    pub fn depth_read_sampler(&self) -> &crate::gpu::Sampler {
        &self.scene_tier.material.depth_read_sampler
    }

    /// Bind group layout for the read-only-depth pass: binding 0 is the scene
    /// depth texture (`texture_depth_2d`), binding 1 the non-filtering sampler.
    /// Matches [`DepthReadContext::scene_depth_bind_group`](crate::plugin_api::DepthReadContext::scene_depth_bind_group).
    ///
    /// A convenience for plugins that have a spare bind group: place this at
    /// whatever free slot of `extra_bind_group_layouts` when building the
    /// pipeline with [`build_depth_read_pipeline`](Self::build_depth_read_pipeline),
    /// declare the matching `@group(K) @binding(0/1)` in the shader
    /// (`texture_2d<f32>` and `sampler`: binding 0 is unfilterable float), and bind
    /// the ready-made bind group there. A plugin already using all four bind
    /// groups skips this and folds the depth texture + sampler into an existing
    /// group instead (see
    /// [`SHARED_DEPTH_READ_WGSL`](crate::plugin_api::shared_wgsl::SHARED_DEPTH_READ_WGSL)).
    pub fn depth_read_bind_group_layout(&self) -> &crate::gpu::BindGroupLayout {
        &self.scene_tier.material.depth_read_bgl
    }

    /// Shared linear-clamp sampler used by the lib for colormap LUTs.
    ///
    /// Use this when sampling 1D LUT-style data (colourmaps, transfer
    /// functions) where the texture should not wrap.
    pub fn lut_sampler(&self) -> &crate::gpu::Sampler {
        match &self.compatibility_tier {
            Some(compat) => &compat.lut_sampler,
            None => &self.scene_tier.material.lut_sampler,
        }
    }

    /// Comparison sampler used for PCF shadow filtering.
    ///
    /// Plugins that sample the shadow atlas directly (rather than through
    /// `viewport_sample_csm`) use this sampler when binding the atlas.
    pub fn shadow_filter_sampler(&self) -> &crate::gpu::Sampler {
        match &self.compatibility_tier {
            Some(compat) => &compat.shadow_sampler,
            None => &self.scene_tier.shadow.sampler,
        }
    }

    /// Number of live user-uploaded textures.
    ///
    /// `id` values in `0..texture_count()` are addressable via
    /// [`texture_view`](Self::texture_view), with the caveat that promoted
    /// IDs from async uploads may sit at the high end.
    pub fn texture_count(&self) -> usize {
        self.content.textures.len()
    }

    /// `true` when `id` refers to a live texture slot.
    ///
    /// Cheaper than [`texture_view`](Self::texture_view) when only the
    /// liveness answer is needed, for example when deciding whether a cached
    /// bind group must be rebuilt against the fallback.
    pub fn has_texture(&self, id: crate::resources::TextureId) -> bool {
        self.content.textures.get(id).is_some()
    }

    /// Borrow the GPU LUT view for a colourmap uploaded via
    /// [`upload_colourmap`](Self::upload_colourmap), or a builtin id from
    /// [`builtin_colourmap_id`](Self::builtin_colourmap_id).
    ///
    /// Returns `None` when `id` is out of range (fall back to
    /// [`fallback_colourmap_view`](Self::fallback_colourmap_view)). Pair the
    /// view with [`lut_sampler`](Self::lut_sampler); the same lifetime
    /// contract as [`texture_view`](Self::texture_view) applies.
    pub fn colourmap_view(&self, id: crate::ColourmapId) -> Option<&crate::gpu::TextureView> {
        self.content.colourmap_views.get(id.0)
    }

    /// The 1x1 white LUT view the lib binds when an item names no colourmap
    /// (or a stale id). Bind it wherever a pipeline layout requires a LUT
    /// but the item has none, so plugin behaviour matches the built-in
    /// item types.
    pub fn fallback_colourmap_view(&self) -> &crate::gpu::TextureView {
        &self.content.fallback_lut_view
    }

    /// Borrow the GPU LUT view for a built-in colourmap preset.
    ///
    /// The built-in views exist from construction, so this resolves whenever it
    /// is called, including from an upload that runs before the first frame.
    /// The texels are written on the first `prepare`, before anything samples
    /// them.
    pub fn builtin_colourmap_view(
        &self,
        preset: crate::resources::BuiltinColourmap,
    ) -> &crate::gpu::TextureView {
        let ids = self.content.builtin_colourmap_ids;
        &self.content.colourmap_views[ids[preset as usize].0]
    }

    /// Index count of a mesh uploaded through
    /// [`upload_mesh_data`](Self::upload_mesh_data), or `None` when the id was
    /// never uploaded or has been freed.
    ///
    /// Use it during `prepare` to drop an item whose mesh is not resident,
    /// rather than building per-item state for geometry the draw hooks cannot
    /// bind.
    pub fn mesh_index_count(&self, mesh_id: MeshId) -> Option<u32> {
        self.mesh_store.get(mesh_id).map(|m| m.index_count)
    }

    /// Borrow the 3D texture view for a scalar field uploaded via
    /// [`upload_volume`](Self::upload_volume).
    ///
    /// `None` when `id` was never uploaded or has been freed. The texture is
    /// filterable (`R16Float`, or `R32Float` where `FLOAT32_FILTERABLE` is
    /// available), so a linear sampler reconstructs it trilinearly. The same
    /// lifetime contract as [`texture_view`](Self::texture_view) applies: bake
    /// the view into a bind group during `prepare` rather than holding the
    /// borrow.
    pub fn volume_view(&self, id: crate::resources::VolumeId) -> Option<&crate::gpu::TextureView> {
        self.content.volume_textures.get(id).map(|(_, view)| view)
    }

    /// Grid dimensions `[nx, ny, nz]` of an uploaded scalar field, or `None`
    /// when `id` was never uploaded or has been freed.
    pub fn volume_dims(&self, id: crate::resources::VolumeId) -> Option<[u32; 3]> {
        let (tex, _) = self.content.volume_textures.get(id)?;
        let size = tex.size();
        Some([size.width, size.height, size.depth_or_array_layers])
    }

    // ------------------------------------------------------------------
    // Pipeline builders
    // ------------------------------------------------------------------

    /// Build an opaque scene pipeline that draws into the HDR scene pass.
    ///
    /// Standard depth state: `opts.depth_compare` test
    /// ([`DEPTH_COMPARE`](crate::plugin_api::depth::DEPTH_COMPARE) by default),
    /// depth write on. The pipeline
    /// layout lists [`group0_layout`](Self::group0_layout) as group 0,
    /// then `extra_bind_group_layouts` as groups 1.., in order. The plugin
    /// owns all groups past 0.
    ///
    /// The pass has two colour targets, the scene colour and the indirect
    /// fraction (see [`OpaqueTargetDesc`]); the pipeline declares both. With
    /// [`writes_indirect_fraction`](PluginPipelineOpts::writes_indirect_fraction)
    /// off, the default, the fragment entry writes `@location(0)` only and the
    /// fraction is left as it was.
    pub fn build_opaque_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        self.pipeline_builder().build_opaque_pipeline(device, opts)
    }

    /// Build an opaque pipeline for the LDR scene pass: the same as
    /// [`build_opaque_pipeline`](Self::build_opaque_pipeline) against
    /// [`ldr_opaque_target_desc`](Self::ldr_opaque_target_desc). The compatibility
    /// tier draws only through this pass; on the full tier it is the pipeline
    /// a type that sets
    /// [`draws_ldr`](crate::plugin_api::ItemTypePlugin::draws_ldr) selects when
    /// [`PaintContext::target_format`](crate::plugin_api::PaintContext::target_format)
    /// is not the HDR format.
    pub fn build_ldr_opaque_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        self.pipeline_builder()
            .build_ldr_opaque_pipeline(device, opts)
    }

    /// Build a pipeline that draws into the foreground pass.
    ///
    /// Same shape as [`build_opaque_pipeline`](Self::build_opaque_pipeline)
    /// (group 0 = shared bindings, then `extra_bind_group_layouts`), but
    /// single-sampled: the foreground pass runs after the SSAA resolve. The
    /// bound group-0 camera carries the foreground projection and disabled
    /// clip planes; depth is tested against the pass's own cleared target.
    pub fn build_foreground_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        self.pipeline_builder()
            .build_foreground_pipeline(device, opts)
    }

    /// Build a transparent pipeline that draws into the OIT pass.
    ///
    /// The fragment shader must return [`OitOutput`](crate::plugin_api::shared_wgsl::SHARED_OIT_WGSL),
    /// writing both `@location(0)` (accum) and `@location(1)` (reveal).
    /// Depth state: `LessEqual` test, depth write off.
    ///
    /// `opts.primitive` sets the cull mode, which defaults to
    /// [`Face::Back`](crate::gpu::Face::Back). For a two-sided item (an open
    /// surface whose back faces should show through) set `cull_mode: None`:
    /// weighted-blended OIT is order-independent, so drawing both faces is
    /// correct, and the fragment stage can flip the normal on `@builtin(front_facing)`.
    /// With the default back-face culling the away-facing side is dropped.
    pub fn build_oit_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        self.pipeline_builder().build_oit_pipeline(device, opts)
    }

    /// Build a pipeline that draws into the read-only-depth pass.
    ///
    /// One colour target (the HDR scene buffer) with the caller's blend state,
    /// and the scene depth attachment bound read-only: the pipeline tests
    /// against opaque depth (`opts.depth_compare`, `DEPTH_COMPARE` by default) but
    /// never writes it, since the pass binds depth read-only. Set
    /// `opts.color_blend` to alpha blending for soft particles.
    ///
    /// The plugin lists its own bind group layouts in
    /// `opts.extra_bind_group_layouts` as usual. The scene depth read is not a
    /// fixed group: the plugin either adds
    /// [`depth_read_bind_group_layout`](Self::depth_read_bind_group_layout) at a
    /// spare slot, or folds the two depth bindings into one of its existing
    /// layouts. It reconstructs depth through
    /// [`SHARED_DEPTH_READ_WGSL`](crate::plugin_api::shared_wgsl::SHARED_DEPTH_READ_WGSL).
    pub fn build_depth_read_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        self.pipeline_builder()
            .build_depth_read_pipeline(device, opts)
    }

    /// Build a pipeline for the outline-mask pass (R8 target).
    ///
    /// Fragment shader must write `1.0` at `@location(0)` for any covered
    /// pixel; use [`SHARED_MASK_WGSL`](crate::plugin_api::shared_wgsl::SHARED_MASK_WGSL).
    /// Depth state: `LessEqual` test, no depth write.
    pub fn build_mask_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        self.pipeline_builder().build_mask_pipeline(device, opts)
    }

    /// Build a pipeline for the surface-mask pass, from the same options an
    /// outline-mask pipeline takes.
    ///
    /// The pass has no colour target, so whatever the fragment stage outputs
    /// is dropped: a type can hand over its outline-mask shader unchanged and
    /// what matters is where it discards. See
    /// [`build_surface_mask_pipeline`](crate::plugin_api::builders::build_surface_mask_pipeline)
    /// for the depth and stencil state, which is the same here.
    pub fn build_surface_mask_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        self.pipeline_builder()
            .build_surface_mask_pipeline(device, opts)
    }

    /// Build a pipeline for the pick-id pass.
    ///
    /// The pass has three colour targets (object id, primitive id, depth) plus a
    /// depth-stencil attachment; this matches the pipeline to all of them. The
    /// fragment shader must write all three: the item's `PickId` at
    /// `@location(0)`, a sub-object index (or 0) at `@location(1)`, and the
    /// framebuffer `z` at `@location(2)`. Use
    /// [`SHARED_PICK_WGSL`](crate::plugin_api::shared_wgsl::SHARED_PICK_WGSL),
    /// whose `viewport_pick_fs` produces exactly that output.
    pub fn build_pick_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        self.pipeline_builder().build_pick_pipeline(device, opts)
    }

    /// Build a depth-only pipeline for the shadow-atlas pass.
    ///
    /// No fragment output. The fragment entry is optional; pass an empty
    /// string to use a depth-only configuration with no fragment stage.
    /// Standard depth state: `LessEqual` test, depth write on, with the
    /// lib's standard depth bias. On a device with
    /// [`depth_clip_control`](crate::Capabilities::depth_clip_control) the
    /// primitive state's `unclipped_depth` is set, so casters in front of a
    /// cascade clamp to its near plane rather than vanish.
    ///
    /// Group 0 is the shadow pass's own camera, not the scene bind group the
    /// other builders use: the lib binds the cascade's light view-projection
    /// as a single dynamic-offset uniform before calling
    /// [`cast_shadow_pass`](crate::plugin_api::ItemTypePlugin::cast_shadow_pass).
    /// Declare it in the shader with
    /// [`SHARED_SHADOW_BINDINGS_WGSL`](crate::plugin_api::shared_wgsl::SHARED_SHADOW_BINDINGS_WGSL)
    /// rather than `SHARED_BINDINGS_WGSL`.
    pub fn build_shadow_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        self.pipeline_builder().build_shadow_pipeline(device, opts)
    }

    /// Build a fullscreen post-effect pipeline. Convenience alias for
    /// [`plugin_api::post_effect::build_post_effect_pipeline`]
    /// (a free function taking only the device, so it is also callable
    /// from a post effect's `init_gpu` / `on_viewport_resized`, where no
    /// `DeviceResources` is available).
    ///
    /// [`plugin_api::post_effect::build_post_effect_pipeline`]: crate::plugin_api::post_effect::build_post_effect_pipeline
    pub fn build_post_effect_pipeline(
        &self,
        device: &crate::gpu::Device,
        label: &str,
        shader: &crate::gpu::ShaderModule,
        bind_group_layout: &crate::gpu::BindGroupLayout,
        target_format: crate::gpu::TextureFormat,
        blend: Option<crate::gpu::BlendState>,
    ) -> crate::gpu::RenderPipeline {
        crate::plugin_api::post_effect::build_post_effect_pipeline(
            device,
            label,
            shader,
            bind_group_layout,
            target_format,
            blend,
        )
    }
}

/// What the plugin pipeline builders read from [`DeviceResources`], held by
/// value so a build can run on a worker thread, inside a
/// [`LazyPipelines`](crate::plugin_api::LazyPipelines) build function. Get
/// one with [`DeviceResources::pipeline_builder`]; its methods match the
/// `build_*_pipeline` methods there.
#[derive(Clone)]
pub struct PipelineBuilder {
    camera_bgl: crate::gpu::BindGroupLayout,
    shadow_camera_bgl: crate::gpu::BindGroupLayout,
    opaque: OpaqueTargetDesc,
    ldr_opaque: OpaqueTargetDesc,
    foreground: ForegroundTargetDesc,
    oit: OitTargetDesc,
    depth_read: DepthReadTargetDesc,
    mask: MaskTargetDesc,
    pick: PickTargetDesc,
    shadow: ShadowTargetDesc,
    /// Shadow pipelines clamp casters in front of a cascade to its near
    /// plane (`unclipped_depth`) rather than clip them.
    shadow_depth_clamp: bool,
}

impl PipelineBuilder {
    /// The scene depth-stencil format these pipelines are built against. See
    /// [`DeviceResources::scene_depth_format`].
    pub fn scene_depth_format(&self) -> crate::gpu::TextureFormat {
        self.opaque.depth_format
    }

    /// See [`DeviceResources::build_opaque_pipeline`].
    pub fn build_opaque_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        self.build_opaque_into(device, opts, self.opaque)
    }

    /// See [`DeviceResources::build_ldr_opaque_pipeline`].
    pub fn build_ldr_opaque_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        self.build_opaque_into(device, opts, self.ldr_opaque)
    }

    fn build_opaque_into(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
        desc: OpaqueTargetDesc,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(
            device,
            opts.label,
            &self.camera_bgl,
            opts.extra_bind_group_layouts,
        );
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex_module: opts.shader,
                vertex_entry: opts.vs_entry,
                vertex_buffers: opts.vertex_layouts,
                fragment: Some(crate::gpu::FragmentState {
                    module: opts.shader,
                    entry_point: Some(opts.fs_entry),
                    targets: desc
                        .colour_targets(opts.color_blend, opts.writes_indirect_fraction)
                        .as_slice(),
                    compilation_options: Default::default(),
                }),
                primitive: opts.primitive,
                depth_stencil: Some(crate::resources::builders::depth_stencil(
                    desc.depth_format,
                    opts.depth_write,
                    opts.depth_compare,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }

    /// See [`DeviceResources::build_foreground_pipeline`].
    pub fn build_foreground_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(
            device,
            opts.label,
            &self.camera_bgl,
            opts.extra_bind_group_layouts,
        );
        let desc = self.foreground;
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex_module: opts.shader,
                vertex_entry: opts.vs_entry,
                vertex_buffers: opts.vertex_layouts,
                fragment: Some(crate::gpu::FragmentState {
                    module: opts.shader,
                    entry_point: Some(opts.fs_entry),
                    targets: desc
                        .colour_targets(opts.color_blend, opts.writes_indirect_fraction)
                        .as_slice(),
                    compilation_options: Default::default(),
                }),
                primitive: opts.primitive,
                depth_stencil: Some(crate::resources::builders::depth_stencil(
                    desc.depth_format,
                    opts.depth_write,
                    opts.depth_compare,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }

    /// See [`DeviceResources::build_oit_pipeline`].
    pub fn build_oit_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(
            device,
            opts.label,
            &self.camera_bgl,
            opts.extra_bind_group_layouts,
        );
        let desc = self.oit;
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex_module: opts.shader,
                vertex_entry: opts.vs_entry,
                vertex_buffers: opts.vertex_layouts,
                fragment: Some(crate::gpu::FragmentState {
                    module: opts.shader,
                    entry_point: Some(opts.fs_entry),
                    targets: &[
                        Some(crate::gpu::ColorTargetState {
                            format: desc.accum_format,
                            blend: Some(desc.accum_blend),
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                        Some(crate::gpu::ColorTargetState {
                            format: desc.reveal_format,
                            blend: Some(desc.reveal_blend),
                            write_mask: crate::gpu::ColorWrites::RED,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: opts.primitive,
                depth_stencil: Some(crate::resources::builders::depth_stencil(
                    desc.depth_format,
                    false,
                    crate::plugin_api::depth::DEPTH_COMPARE,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }

    /// See [`DeviceResources::build_depth_read_pipeline`].
    pub fn build_depth_read_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(
            device,
            opts.label,
            &self.camera_bgl,
            opts.extra_bind_group_layouts,
        );
        let desc = self.depth_read;
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex_module: opts.shader,
                vertex_entry: opts.vs_entry,
                vertex_buffers: opts.vertex_layouts,
                fragment: Some(crate::gpu::FragmentState {
                    module: opts.shader,
                    entry_point: Some(opts.fs_entry),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: desc.color_format,
                        blend: opts.color_blend,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: opts.primitive,
                depth_stencil: Some(crate::resources::builders::depth_stencil(
                    desc.depth_format,
                    false,
                    opts.depth_compare,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }

    /// See [`DeviceResources::build_mask_pipeline`].
    pub fn build_mask_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(
            device,
            opts.label,
            &self.camera_bgl,
            opts.extra_bind_group_layouts,
        );
        let desc = self.mask;
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex_module: opts.shader,
                vertex_entry: opts.vs_entry,
                vertex_buffers: opts.vertex_layouts,
                fragment: Some(crate::gpu::FragmentState {
                    module: opts.shader,
                    entry_point: Some(opts.fs_entry),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: desc.color_format,
                        blend: None,
                        write_mask: crate::gpu::ColorWrites::RED,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: opts.primitive,
                depth_stencil: Some(crate::resources::builders::depth_stencil(
                    desc.depth_format,
                    false,
                    crate::plugin_api::depth::DEPTH_COMPARE,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }

    /// See [`DeviceResources::build_surface_mask_pipeline`].
    pub fn build_surface_mask_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(
            device,
            opts.label,
            &self.camera_bgl,
            opts.extra_bind_group_layouts,
        );
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex_module: opts.shader,
                vertex_entry: opts.vs_entry,
                vertex_buffers: opts.vertex_layouts,
                fragment: Some(crate::gpu::FragmentState {
                    module: opts.shader,
                    entry_point: Some(opts.fs_entry),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: opts.primitive,
                depth_stencil: Some(crate::resources::builders::surface_mask_depth_stencil(
                    self.opaque.depth_format,
                )),
                multisample: crate::gpu::MultisampleState::default(),
                cache: None,
            },
        )
    }

    /// See [`DeviceResources::build_pick_pipeline`].
    pub fn build_pick_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        let layout = build_layout(
            device,
            opts.label,
            &self.camera_bgl,
            opts.extra_bind_group_layouts,
        );
        let desc = self.pick;
        // Integer and float single-channel targets, no blending: a fragment
        // either writes an exact id/depth or leaves the attachment at its clear
        // value. Order and formats mirror the internal pick pipeline.
        let color_target = |format| {
            Some(crate::gpu::ColorTargetState {
                format,
                blend: None,
                write_mask: crate::gpu::ColorWrites::ALL,
            })
        };
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex_module: opts.shader,
                vertex_entry: opts.vs_entry,
                vertex_buffers: opts.vertex_layouts,
                fragment: Some(crate::gpu::FragmentState {
                    module: opts.shader,
                    entry_point: Some(opts.fs_entry),
                    targets: &[
                        color_target(desc.object_id_format),
                        color_target(desc.primitive_id_format),
                        color_target(desc.depth_channel_format),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: opts.primitive,
                depth_stencil: Some(crate::resources::builders::depth_stencil(
                    desc.depth_stencil_format,
                    true,
                    crate::plugin_api::depth::DEPTH_COMPARE,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }

    /// See [`DeviceResources::build_shadow_pipeline`].
    pub fn build_shadow_pipeline(
        &self,
        device: &crate::gpu::Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> crate::gpu::RenderPipeline {
        // Group 0 in the shadow pass is the cascade-space camera the lib binds
        // before calling `cast_shadow_pass`: a single dynamic-offset uniform,
        // not the scene bind group the other passes use.
        let mut bgls: Vec<&crate::gpu::BindGroupLayout> =
            Vec::with_capacity(1 + opts.extra_bind_group_layouts.len());
        bgls.push(&self.shadow_camera_bgl);
        bgls.extend(opts.extra_bind_group_layouts.iter().copied());
        let layout = crate::resources::builders::pipeline_layout(device, opts.label, &bgls);
        let desc = self.shadow;
        let fragment = if opts.fs_entry.is_empty() {
            None
        } else {
            Some(crate::gpu::FragmentState {
                module: opts.shader,
                entry_point: Some(opts.fs_entry),
                targets: &[],
                compilation_options: Default::default(),
            })
        };
        // The cascade fit relies on every caster clamping when the device
        // can, so a caster above the cascade's depth range still lands in it.
        let mut primitive = opts.primitive;
        primitive.unclipped_depth |= self.shadow_depth_clamp;
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: opts.label.unwrap_or_default(),
                layout: &layout,
                vertex_module: opts.shader,
                vertex_entry: opts.vs_entry,
                vertex_buffers: opts.vertex_layouts,
                fragment,
                primitive,
                depth_stencil: Some(crate::gpu::DepthStencilState {
                    format: desc.depth_format,
                    depth_write_enabled: crate::resources::builders::dwrite(true),
                    // `opts.depth_compare` is written for scene depth; the
                    // shadow map keeps its own convention.
                    depth_compare: crate::resources::builders::dcompare(
                        crate::plugin_api::depth::scene_to_shadow_compare(opts.depth_compare),
                    ),
                    stencil: crate::gpu::StencilState::default(),
                    bias: opts.depth_bias.unwrap_or(crate::gpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    }),
                }),
                multisample: crate::gpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    }
}

/// Inputs to a plugin pipeline builder. All builders take this struct; the
/// builder picks the target descriptor and blend state.
pub struct PluginPipelineOpts<'a> {
    /// Pipeline debug label. Forwarded to wgpu.
    pub label: Option<&'a str>,
    /// Shader module containing both the vertex and fragment entry points.
    pub shader: &'a crate::gpu::ShaderModule,
    /// Vertex-stage entry-point name (e.g. `"vs_main"`).
    pub vs_entry: &'a str,
    /// Fragment-stage entry-point name (e.g. `"fs_main"`). For
    /// `build_shadow_pipeline`, pass `""` to skip the fragment stage.
    pub fs_entry: &'a str,
    /// Vertex buffer layouts.
    pub vertex_layouts: &'a [crate::gpu::VertexBufferLayout<'a>],
    /// Bind group layouts for groups 1.. (the plugin's per-object data).
    /// Group 0 is supplied automatically from
    /// [`DeviceResources::group0_layout`].
    pub extra_bind_group_layouts: &'a [&'a crate::gpu::BindGroupLayout],
    /// Primitive topology, cull mode, polygon mode.
    pub primitive: crate::gpu::PrimitiveState,
    /// Optional blend state for the opaque builder. `None` = no blending
    /// (the default for opaque pipelines). Ignored by OIT / mask / pick /
    /// shadow builders, which use their pass-specific blend state.
    pub color_blend: Option<crate::gpu::BlendState>,
    /// Whether the opaque builder writes depth. Ignored by the other
    /// builders. Default `true`.
    pub depth_write: bool,
    /// Depth-compare function for the opaque and shadow builders, written
    /// for scene depth. Ignored by the others. Default
    /// [`DEPTH_COMPARE`](crate::plugin_api::depth::DEPTH_COMPARE); use the
    /// [`depth`](crate::plugin_api::depth) constants rather than naming a
    /// direction. The shadow builder translates it into the shadow map's own
    /// convention.
    pub depth_compare: crate::gpu::CompareFunction,
    /// Depth bias for the shadow builder. Ignored by the others.
    ///
    /// `None` uses a mild default (`constant: 2, slope_scale: 2.0`). Prefer
    /// [`shadow_caster`](Self::shadow_caster), which sets this and the cull
    /// mode to the pair the renderer's own meshes cast with.
    pub depth_bias: Option<crate::gpu::DepthBiasState>,
    /// Whether the fragment entry writes the HDR scene pass's indirect
    /// fraction at `@location(1)`, for the opaque and foreground builders (see
    /// [`indirect_fraction_target`](crate::plugin_api::target_desc::indirect_fraction_target)).
    /// Ignored by the others and on the LDR pass. Default `false`: the target
    /// is declared with an empty write mask.
    pub writes_indirect_fraction: bool,
}

/// How a shadow pipeline's triangles cast into the cascade atlas. Each case
/// pairs a cull mode with the depth bias the renderer's own meshes use for
/// it. The bias is calibrated against the receiver-side normal offset in the
/// shared shadow lookup, so the two are not independent settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowCaster {
    /// A closed solid. Back faces cast (`cull_mode: Front`), so a front face
    /// never compares against its own depth, and a small constant bias
    /// closes coplanar leaks. No acne, and contact shadows stay attached.
    Closed,
    /// Open or thin geometry whose back faces show. Both faces cast, with the
    /// large bias the renderer's two-sided surfaces use to keep a surface off
    /// its own depth. A contact shadow under it can lift by a texel or two.
    TwoSided,
}

impl ShadowCaster {
    /// The faces the shadow pass culls.
    pub fn cull_mode(self) -> Option<crate::gpu::Face> {
        match self {
            Self::Closed => Some(crate::gpu::Face::Front),
            Self::TwoSided => None,
        }
    }

    /// The caster's depth bias.
    pub fn depth_bias(self) -> crate::gpu::DepthBiasState {
        match self {
            Self::Closed => crate::resources::mesh::mesh_pipelines::CSM_SHADOW_BIAS,
            Self::TwoSided => crate::resources::mesh::mesh_pipelines::CSM_SHADOW_BIAS_TWO_SIDED,
        }
    }
}

impl<'a> PluginPipelineOpts<'a> {
    /// Construct an opts struct with sensible defaults for the variable
    /// fields (`TriangleList`, back-face cull, depth write on,
    /// `LessEqual`, no blend). Callers must supply `shader`, vertex / fragment
    /// entry points, and the vertex layout.
    pub fn new(
        label: Option<&'a str>,
        shader: &'a crate::gpu::ShaderModule,
        vs_entry: &'a str,
        fs_entry: &'a str,
        vertex_layouts: &'a [crate::gpu::VertexBufferLayout<'a>],
    ) -> Self {
        Self {
            label,
            shader,
            vs_entry,
            fs_entry,
            vertex_layouts,
            extra_bind_group_layouts: &[],
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(crate::gpu::Face::Back),
                ..Default::default()
            },
            color_blend: None,
            depth_write: true,
            depth_compare: crate::plugin_api::depth::DEPTH_COMPARE,
            depth_bias: None,
            writes_indirect_fraction: false,
        }
    }

    /// Set the cull mode and depth bias for a shadow pipeline from how its
    /// geometry casts. See [`ShadowCaster`].
    pub fn shadow_caster(mut self, caster: ShadowCaster) -> Self {
        self.primitive.cull_mode = caster.cull_mode();
        self.depth_bias = Some(caster.depth_bias());
        self
    }
}

fn build_layout(
    device: &crate::gpu::Device,
    label: Option<&str>,
    camera_bgl: &crate::gpu::BindGroupLayout,
    extras: &[&crate::gpu::BindGroupLayout],
) -> crate::gpu::PipelineLayout {
    let mut bgls: Vec<&crate::gpu::BindGroupLayout> = Vec::with_capacity(1 + extras.len());
    bgls.push(camera_bgl);
    bgls.extend(extras.iter().copied());
    crate::resources::builders::pipeline_layout(device, label, &bgls)
}

/// Draw handle for meshes the consumer uploaded through
/// [`upload_mesh_data`](DeviceResources::upload_mesh_data).
///
/// An item type whose geometry is a consumer-supplied [`MeshId`] rather than
/// buffers of its own cannot bind it from a draw hook: the hook contexts carry
/// no resources borrow, and the vertex and index data live in a shared arena
/// whose layout is the lib's business. This hands over the one operation that
/// needs, and nothing else.
///
/// Vertices use the lib's full 64-byte `Vertex` layout, so a pipeline that
/// consumes them declares that layout (or a prefix of it) as vertex buffer 0.
#[derive(Clone, Copy)]
pub struct MeshDraw<'a> {
    resources: &'a DeviceResources,
}

impl<'a> MeshDraw<'a> {
    pub(crate) fn new(resources: &'a DeviceResources) -> Self {
        Self { resources }
    }

    /// Bind the mesh's vertex and index buffers at slot 0 and issue the
    /// indexed draw for its full index range.
    ///
    /// Returns `false` without touching the pass when `mesh_id` was never
    /// uploaded or has been freed, so a plugin holding a stale id draws
    /// nothing instead of drawing the wrong geometry. The pipeline and any
    /// bind groups are the caller's to set first.
    pub fn draw_indexed(&self, pass: &mut crate::gpu::RenderPass<'_>, mesh_id: MeshId) -> bool {
        self.draw_indexed_instanced(pass, mesh_id, 1)
    }

    /// The same draw with an instance count, for an item type that draws one
    /// uploaded mesh many times and composes each instance's transform in its
    /// own vertex stage rather than from a per-instance vertex buffer.
    ///
    /// Returns `false` without touching the pass when `mesh_id` is stale, the
    /// same as [`draw_indexed`](Self::draw_indexed).
    pub fn draw_indexed_instanced(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        mesh_id: MeshId,
        instances: u32,
    ) -> bool {
        self.draw_indexed_instance_range(pass, mesh_id, 0..instances)
    }

    /// The same draw over an instance *range*, for an item type whose
    /// instances are a window into a larger buffer: `instance_index` in the
    /// vertex stage starts at the range's start, so several items can render
    /// disjoint regions of one pool without rebinding anything.
    ///
    /// Returns `false` without touching the pass when `mesh_id` is stale, the
    /// same as [`draw_indexed`](Self::draw_indexed).
    pub fn draw_indexed_instance_range(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        mesh_id: MeshId,
        instances: std::ops::Range<u32>,
    ) -> bool {
        let Some(mesh) = self.resources.mesh_store.get(mesh_id) else {
            return false;
        };
        pass.set_vertex_buffer(0, self.resources.geometry.vertex_slice(mesh.vertex_span));
        pass.set_index_buffer(
            self.resources.geometry.index_slice(mesh.index_span),
            crate::gpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(0..mesh.index_count, 0, instances);
        true
    }

    /// Bind one of the mesh's per-vertex vector attributes as the vertex
    /// buffer at `slot`.
    ///
    /// The attribute is one the mesh was uploaded with as
    /// `AttributeData::VertexVector` under `name`: three floats per vertex,
    /// in the same vertex order as slot 0. The pipeline declares it with
    /// [`vector_attribute_layout`](crate::plugin_api::builders::vector_attribute_layout).
    ///
    /// Returns `false` without touching the pass when the mesh is stale or
    /// carries no vector attribute of that name.
    pub fn bind_vector_attribute(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        slot: u32,
        mesh_id: MeshId,
        name: &str,
    ) -> bool {
        let Some((chunk, range)) = self.resources.mesh_store.get(mesh_id).and_then(|mesh| {
            let record = mesh.sidecar.as_ref()?;
            let stream = record.find(
                crate::resources::mesh::sidecar_pool::StreamKind::Vector,
                name,
            )?;
            Some((record.chunk(), record.byte_range(stream)))
        }) else {
            return false;
        };
        pass.set_vertex_buffer(
            slot,
            self.resources.sidecar.chunk_buffer(chunk).slice(range),
        );
        true
    }

    /// Bind one of the mesh's per-vertex scalar attributes as the vertex
    /// buffer at `slot`.
    ///
    /// The attribute is one the mesh was uploaded with under `name` as
    /// `AttributeData::Vertex`, or as `Cell` or `Edge`, which are averaged to
    /// the vertices at upload: one float per vertex, in the same vertex order
    /// as slot 0. `Face`, `Halfedge` and `Corner` attributes are stored per
    /// triangle corner and are not reached by this. The pipeline declares it
    /// with
    /// [`scalar_attribute_layout`](crate::plugin_api::builders::scalar_attribute_layout).
    ///
    /// Returns `false` without touching the pass when the mesh is stale or
    /// carries no per-vertex scalar attribute of that name.
    pub fn bind_scalar_attribute(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        slot: u32,
        mesh_id: MeshId,
        name: &str,
    ) -> bool {
        let Some((chunk, range)) = self.resources.mesh_store.get(mesh_id).and_then(|mesh| {
            let record = mesh.sidecar.as_ref()?;
            let stream = record.find(
                crate::resources::mesh::sidecar_pool::StreamKind::Scalar,
                name,
            )?;
            Some((record.chunk(), record.byte_range(stream)))
        }) else {
            return false;
        };
        pass.set_vertex_buffer(
            slot,
            self.resources.sidecar.chunk_buffer(chunk).slice(range),
        );
        true
    }

    /// Index count of an uploaded mesh, or `None` when the id is stale.
    pub fn index_count(&self, mesh_id: MeshId) -> Option<u32> {
        self.resources
            .mesh_store
            .get(mesh_id)
            .map(|m| m.index_count)
    }
}

/// Read handle for the CPU-side geometry of meshes the consumer uploaded
/// through [`upload_mesh_data`](DeviceResources::upload_mesh_data).
///
/// The counterpart of [`MeshDraw`] for the picking hooks: an item type whose
/// geometry is a consumer-supplied [`MeshId`] answers
/// [`pick`](crate::plugin_api::ItemTypePlugin::pick) and
/// [`pick_rect`](crate::plugin_api::ItemTypePlugin::pick_rect) against these
/// arrays rather than keeping a copy of its own.
///
/// Both accessors return `None` when the id is stale, and when the mesh was
/// uploaded without CPU-side geometry retained.
#[derive(Clone, Copy)]
pub struct MeshGeometry<'a> {
    resources: &'a DeviceResources,
}

impl std::fmt::Debug for MeshGeometry<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MeshGeometry")
    }
}

impl<'a> MeshGeometry<'a> {
    pub(crate) fn new(resources: &'a DeviceResources) -> Self {
        Self { resources }
    }

    /// Object-space vertex positions, in the mesh's own vertex order.
    pub fn positions(&self, mesh_id: MeshId) -> Option<&'a [[f32; 3]]> {
        self.resources
            .mesh_store
            .get(mesh_id)?
            .cpu_positions
            .as_deref()
    }

    /// Triangle indices into [`positions`](Self::positions), three per face.
    pub fn indices(&self, mesh_id: MeshId) -> Option<&'a [u32]> {
        self.resources
            .mesh_store
            .get(mesh_id)?
            .cpu_indices
            .as_deref()
    }
}
