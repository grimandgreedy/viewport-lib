//! [`PluginResources`]: what an item-type plugin builds its GPU state against.

use crate::gpu::{BindGroupLayout, Device, RenderPipeline, Sampler, TextureFormat, TextureView};
use crate::plugin_api::{
    DepthReadTargetDesc, ForegroundTargetDesc, LazyModule, MaskTargetDesc, OitTargetDesc,
    OpaqueTargetDesc, PickTargetDesc, PipelineBuilder, ShadowTargetDesc,
};
use crate::resources::{DeviceResources, PluginPipelineOpts};

mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::resources::DeviceResources {}
}

/// The renderer's layouts, target formats, pipeline builders and shared
/// samplers, as handed to
/// [`ItemTypePlugin::init_gpu`](crate::plugin_api::ItemTypePlugin::init_gpu).
///
/// Everything here reads the active device tier, so a pipeline built through
/// it matches the passes the renderer runs on that device. Build pipelines
/// with the `build_*` methods rather than copying formats or layouts into the
/// plugin: a copy stops matching when the library changes a pass.
///
/// What a tier does not have is an `Option`: the read-only-depth pass and the
/// material fallback textures exist only on [`Tier::Full`](crate::Tier::Full).
///
/// The trait is sealed; [`DeviceResources`] is its only implementation, and
/// its methods forward to the ones of the same name there.
pub trait PluginResources: sealed::Sealed {
    /// What the device can do. Read it to pick a path the device supports.
    fn capabilities(&self) -> &crate::capabilities::Capabilities;

    /// Group 0 of every scene pipeline: camera, lights, shadows, clip and
    /// IBL. List it as group 0 of a pipeline layout; the WGSL declarations
    /// are [`SHARED_BINDINGS_WGSL`](crate::plugin_api::shared_wgsl::SHARED_BINDINGS_WGSL).
    fn group0_layout(&self) -> &BindGroupLayout;

    /// Group 0 of the shadow pass: the cascade light matrix the renderer binds
    /// before calling
    /// [`cast_shadow_pass`](crate::plugin_api::ItemTypePlugin::cast_shadow_pass).
    fn shadow_group0_layout(&self) -> &BindGroupLayout;

    /// MSAA sample count of the HDR colour passes (`paint`,
    /// `paint_transparent`, `paint_depth_read`); 1 when MSAA is off. The
    /// shadow, pick and mask passes are always single-sampled.
    fn sample_count(&self) -> u32;

    /// The renderer's output format, which the LDR scene pass draws into.
    fn target_format(&self) -> TextureFormat;

    /// Adapt a source that includes the shared group 0 declarations to this
    /// device before compiling it: binding 17's texture type on a device
    /// without cube arrays, and on the forward tier a uniform light list.
    /// Modules from [`lazy_module`](Self::lazy_module) are adapted already.
    fn adapt_wgsl<'s>(&self, source: &'s str) -> std::borrow::Cow<'s, str>;

    /// See [`DeviceResources::lazy_module`].
    fn lazy_module(&self, device: &Device, label: &str, source: &str) -> LazyModule;

    /// The builders' inputs by value, for a pipeline built later or on another
    /// thread. See [`DeviceResources::pipeline_builder`].
    fn pipeline_builder(&self) -> PipelineBuilder;

    /// See [`DeviceResources::opaque_target_desc`].
    fn opaque_target_desc(&self) -> OpaqueTargetDesc;
    /// See [`DeviceResources::ldr_opaque_target_desc`].
    fn ldr_opaque_target_desc(&self) -> OpaqueTargetDesc;
    /// See [`DeviceResources::foreground_target_desc`].
    fn foreground_target_desc(&self) -> ForegroundTargetDesc;
    /// See [`DeviceResources::oit_target_desc`].
    fn oit_target_desc(&self) -> OitTargetDesc;
    /// See [`DeviceResources::depth_read_target_desc`].
    fn depth_read_target_desc(&self) -> DepthReadTargetDesc;
    /// See [`DeviceResources::mask_target_desc`].
    fn mask_target_desc(&self) -> MaskTargetDesc;
    /// See [`DeviceResources::pick_target_desc`].
    fn pick_target_desc(&self) -> PickTargetDesc;
    /// See [`DeviceResources::shadow_target_desc`].
    fn shadow_target_desc(&self) -> ShadowTargetDesc;

    /// See [`DeviceResources::build_opaque_pipeline`].
    fn build_opaque_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline;
    /// See [`DeviceResources::build_ldr_opaque_pipeline`].
    fn build_ldr_opaque_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline;
    /// See [`DeviceResources::build_foreground_pipeline`].
    fn build_foreground_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline;
    /// See [`DeviceResources::build_oit_pipeline`].
    fn build_oit_pipeline(&self, device: &Device, opts: &PluginPipelineOpts<'_>) -> RenderPipeline;
    /// See [`DeviceResources::build_depth_read_pipeline`].
    fn build_depth_read_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline;
    /// See [`DeviceResources::build_mask_pipeline`].
    fn build_mask_pipeline(&self, device: &Device, opts: &PluginPipelineOpts<'_>)
    -> RenderPipeline;
    /// See [`DeviceResources::build_surface_mask_pipeline`].
    fn build_surface_mask_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline;
    /// See [`DeviceResources::build_pick_pipeline`].
    fn build_pick_pipeline(&self, device: &Device, opts: &PluginPipelineOpts<'_>)
    -> RenderPipeline;
    /// See [`DeviceResources::build_shadow_pipeline`].
    fn build_shadow_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline;

    /// See [`DeviceResources::material_sampler`].
    fn material_sampler(&self) -> &Sampler;
    /// See [`DeviceResources::lut_sampler`].
    fn lut_sampler(&self) -> &Sampler;
    /// See [`DeviceResources::shadow_filter_sampler`].
    fn shadow_filter_sampler(&self) -> &Sampler;
    /// See [`DeviceResources::depth_read_sampler`]. `None` below the full
    /// tier, which runs no read-only-depth pass.
    fn depth_read_sampler(&self) -> Option<&Sampler>;
    /// See [`DeviceResources::depth_read_bind_group_layout`]. `None` below
    /// the full tier, which runs no read-only-depth pass.
    fn depth_read_bind_group_layout(&self) -> Option<&BindGroupLayout>;
    /// See [`DeviceResources::fallback_texture_view`]. `None` below the full
    /// tier.
    fn fallback_texture_view(
        &self,
        slot: crate::scene::material::TextureSlot,
    ) -> Option<&TextureView>;

    /// See [`DeviceResources::texture_view`].
    fn texture_view(&self, id: crate::resources::TextureId) -> Option<&TextureView>;
    /// See [`DeviceResources::texture_sampler`].
    fn texture_sampler(&self, id: crate::resources::TextureId) -> Option<&Sampler>;
    /// See [`DeviceResources::has_texture`].
    fn has_texture(&self, id: crate::resources::TextureId) -> bool;
    /// See [`DeviceResources::texture_count`].
    fn texture_count(&self) -> usize;
    /// See [`DeviceResources::colourmap_view`].
    fn colourmap_view(&self, id: crate::ColourmapId) -> Option<&TextureView>;
    /// See [`DeviceResources::fallback_colourmap_view`].
    fn fallback_colourmap_view(&self) -> &TextureView;
    /// See [`DeviceResources::builtin_colourmap_view`].
    fn builtin_colourmap_view(&self, preset: crate::resources::BuiltinColourmap) -> &TextureView;
    /// See [`DeviceResources::volume_view`].
    fn volume_view(&self, id: crate::resources::VolumeId) -> Option<&TextureView>;
    /// See [`DeviceResources::volume_dims`].
    fn volume_dims(&self, id: crate::resources::VolumeId) -> Option<[u32; 3]>;
    /// See [`DeviceResources::mesh_index_count`].
    fn mesh_index_count(&self, mesh_id: crate::resources::mesh::mesh_store::MeshId) -> Option<u32>;
}

impl PluginResources for DeviceResources {
    fn capabilities(&self) -> &crate::capabilities::Capabilities {
        &self.capabilities
    }
    fn group0_layout(&self) -> &BindGroupLayout {
        DeviceResources::group0_layout(self)
    }
    fn shadow_group0_layout(&self) -> &BindGroupLayout {
        DeviceResources::shadow_group0_layout(self)
    }
    fn sample_count(&self) -> u32 {
        DeviceResources::sample_count(self)
    }
    fn target_format(&self) -> TextureFormat {
        DeviceResources::target_format(self)
    }
    fn adapt_wgsl<'s>(&self, source: &'s str) -> std::borrow::Cow<'s, str> {
        DeviceResources::adapt_wgsl(self, source)
    }
    fn lazy_module(&self, device: &Device, label: &str, source: &str) -> LazyModule {
        DeviceResources::lazy_module(self, device, label, source)
    }
    fn pipeline_builder(&self) -> PipelineBuilder {
        DeviceResources::pipeline_builder(self)
    }

    fn opaque_target_desc(&self) -> OpaqueTargetDesc {
        DeviceResources::opaque_target_desc(self)
    }
    fn ldr_opaque_target_desc(&self) -> OpaqueTargetDesc {
        DeviceResources::ldr_opaque_target_desc(self)
    }
    fn foreground_target_desc(&self) -> ForegroundTargetDesc {
        DeviceResources::foreground_target_desc(self)
    }
    fn oit_target_desc(&self) -> OitTargetDesc {
        DeviceResources::oit_target_desc(self)
    }
    fn depth_read_target_desc(&self) -> DepthReadTargetDesc {
        DeviceResources::depth_read_target_desc(self)
    }
    fn mask_target_desc(&self) -> MaskTargetDesc {
        DeviceResources::mask_target_desc(self)
    }
    fn pick_target_desc(&self) -> PickTargetDesc {
        DeviceResources::pick_target_desc(self)
    }
    fn shadow_target_desc(&self) -> ShadowTargetDesc {
        DeviceResources::shadow_target_desc(self)
    }

    fn build_opaque_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline {
        DeviceResources::build_opaque_pipeline(self, device, opts)
    }
    fn build_ldr_opaque_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline {
        DeviceResources::build_ldr_opaque_pipeline(self, device, opts)
    }
    fn build_foreground_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline {
        DeviceResources::build_foreground_pipeline(self, device, opts)
    }
    fn build_oit_pipeline(&self, device: &Device, opts: &PluginPipelineOpts<'_>) -> RenderPipeline {
        DeviceResources::build_oit_pipeline(self, device, opts)
    }
    fn build_depth_read_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline {
        DeviceResources::build_depth_read_pipeline(self, device, opts)
    }
    fn build_mask_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline {
        DeviceResources::build_mask_pipeline(self, device, opts)
    }
    fn build_surface_mask_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline {
        DeviceResources::build_surface_mask_pipeline(self, device, opts)
    }
    fn build_pick_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline {
        DeviceResources::build_pick_pipeline(self, device, opts)
    }
    fn build_shadow_pipeline(
        &self,
        device: &Device,
        opts: &PluginPipelineOpts<'_>,
    ) -> RenderPipeline {
        DeviceResources::build_shadow_pipeline(self, device, opts)
    }

    fn material_sampler(&self) -> &Sampler {
        DeviceResources::material_sampler(self)
    }
    fn lut_sampler(&self) -> &Sampler {
        DeviceResources::lut_sampler(self)
    }
    fn shadow_filter_sampler(&self) -> &Sampler {
        DeviceResources::shadow_filter_sampler(self)
    }
    fn depth_read_sampler(&self) -> Option<&Sampler> {
        self.scene_tier
            .is_built()
            .then(|| DeviceResources::depth_read_sampler(self))
    }
    fn depth_read_bind_group_layout(&self) -> Option<&BindGroupLayout> {
        self.scene_tier
            .is_built()
            .then(|| DeviceResources::depth_read_bind_group_layout(self))
    }
    fn fallback_texture_view(
        &self,
        slot: crate::scene::material::TextureSlot,
    ) -> Option<&TextureView> {
        self.scene_tier
            .is_built()
            .then(|| DeviceResources::fallback_texture_view(self, slot))
    }

    fn texture_view(&self, id: crate::resources::TextureId) -> Option<&TextureView> {
        DeviceResources::texture_view(self, id)
    }
    fn texture_sampler(&self, id: crate::resources::TextureId) -> Option<&Sampler> {
        DeviceResources::texture_sampler(self, id)
    }
    fn has_texture(&self, id: crate::resources::TextureId) -> bool {
        DeviceResources::has_texture(self, id)
    }
    fn texture_count(&self) -> usize {
        DeviceResources::texture_count(self)
    }
    fn colourmap_view(&self, id: crate::ColourmapId) -> Option<&TextureView> {
        DeviceResources::colourmap_view(self, id)
    }
    fn fallback_colourmap_view(&self) -> &TextureView {
        DeviceResources::fallback_colourmap_view(self)
    }
    fn builtin_colourmap_view(&self, preset: crate::resources::BuiltinColourmap) -> &TextureView {
        DeviceResources::builtin_colourmap_view(self, preset)
    }
    fn volume_view(&self, id: crate::resources::VolumeId) -> Option<&TextureView> {
        DeviceResources::volume_view(self, id)
    }
    fn volume_dims(&self, id: crate::resources::VolumeId) -> Option<[u32; 3]> {
        DeviceResources::volume_dims(self, id)
    }
    fn mesh_index_count(&self, mesh_id: crate::resources::mesh::mesh_store::MeshId) -> Option<u32> {
        DeviceResources::mesh_index_count(self, mesh_id)
    }
}
