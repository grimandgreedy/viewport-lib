//! Construction-time renderer settings.

use crate::resources::PipelineCompilation;

/// Settings fixed when a [`ViewportRenderer`](crate::ViewportRenderer) is
/// built, passed to [`ViewportRenderer::with_config`](crate::ViewportRenderer::with_config).
///
/// Start from [`new`](Self::new) with the surface format and change what you
/// need:
///
/// ```no_run
/// # fn demo(device: &viewport_lib::gpu::Device, saved: Option<Vec<u8>>) {
/// use viewport_lib::{PipelineCompilation, RendererConfig, ViewportRenderer};
/// let config = RendererConfig::new(viewport_lib::gpu::TextureFormat::Bgra8UnormSrgb)
///     .with_sample_count(4)
///     .with_pipeline_cache_data(saved)
///     .with_pipeline_compilation(PipelineCompilation::Blocking);
/// let renderer = ViewportRenderer::with_config(device, &config);
/// # }
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RendererConfig {
    /// The colour format the renderer draws into. Must match the surface or
    /// texture the frame is presented to.
    pub target_format: crate::gpu::TextureFormat,
    /// MSAA sample count: 1, 2 or 4. Above 1 the caller provides multisampled
    /// colour and depth attachments with the final target as the resolve
    /// target. Default 1.
    pub sample_count: u32,
    /// Bytes from an earlier
    /// [`pipeline_cache_data`](crate::ViewportRenderer::pipeline_cache_data),
    /// so compiled pipelines are reused instead of rebuilt. Ignored on a device
    /// without `Features::PIPELINE_CACHE`, and stale or foreign data is
    /// discarded. Default `None`.
    pub pipeline_cache_data: Option<Vec<u8>>,
    /// A pipeline cache to share with other renderers on the same device, in
    /// place of one made from `pipeline_cache_data`. A host with several
    /// renderers on one device (one per window) makes one cache and gives it
    /// to each, so they reuse each other's pipelines and save one set of data.
    /// It must belong to the device the renderer is built on. Default `None`.
    pub pipeline_cache: Option<crate::gpu::PipelineCache>,
    /// How a pipeline is compiled the first time a frame needs it. `None`
    /// takes `VPL_PIPELINE_COMPILATION` from the environment if set, else
    /// [`PipelineCompilation::platform_default`]. Can be changed later with
    /// [`set_pipeline_compilation`](crate::ViewportRenderer::set_pipeline_compilation).
    /// Default `None`.
    pub pipeline_compilation: Option<PipelineCompilation>,
    /// Size of the first vertex and index chunk of the geometry store, in
    /// bytes. Later chunks double from it. `None` uses 16 MiB, or
    /// `VIEWPORT_SLAB_CHUNK_BYTES` from the environment if set and the
    /// `dev-knobs` feature is on. A smaller first
    /// chunk suits an application with little geometry. Default `None`.
    pub geometry_chunk_bytes: Option<u64>,
    /// The adapter's downlevel capabilities
    /// (`adapter.get_downlevel_capabilities()`). A device cannot report these,
    /// so without them the renderer assumes compute, storage buffers in both
    /// stages, cube-array textures and indirect execution are present wherever
    /// the limits allow. Pass them on wgpu's GL backend and on older Metal
    /// devices, where that assumption does not hold. Default `None`.
    pub downlevel_capabilities: Option<crate::gpu::DownlevelCapabilities>,
    /// The adapter's backend (`adapter.get_info().backend`). wgpu 29 and 30
    /// read it from the device; on wgpu 27 a device cannot report it, so pass
    /// it on the GL backend, whose shaders cannot load from depth textures.
    /// Default `None`.
    pub backend: Option<crate::gpu::Backend>,
    /// The highest tier the renderer runs at, whatever the device offers.
    /// `Some(Tier::Compatibility)` on a desktop GPU builds the renderer a WebGL2
    /// device gets, to see what that tier draws without one; see
    /// [`Capabilities::capped_at`](crate::Capabilities::capped_at) for what it
    /// does and does not model. Default `None`.
    pub max_tier: Option<crate::capabilities::Tier>,
    /// A device-lost watcher the host installed on the device, shared with
    /// this renderer. `None` has the renderer install its own, which replaces
    /// any lost callback set on the device before: wgpu keeps one per device.
    /// Pass one when several renderers share a device, or when the host sets
    /// the callback itself. Default `None`.
    pub device_lost_watcher: Option<crate::renderer::DeviceLostWatcher>,
    /// Frames a viewport's post-process, transparency or foreground targets
    /// stay allocated after the frames stop using them. A target released this
    /// way is allocated again by the next frame that needs it, so the value
    /// trades memory held by idle effects against reallocating when an effect
    /// is toggled back on. `Some(0)` releases after one idle frame. `None`
    /// keeps every target a viewport has used for as long as the viewport
    /// exists. Default `None`.
    pub target_release_frames: Option<u32>,
    /// The scene depth-stencil format. `None`, the default, is
    /// `Depth24PlusStencil8` on every device: the format any framework's render
    /// pass can carry, so a host that paints into a pass it does not allocate
    /// (an eframe callback with 24 depth and 8 stencil bits) matches without
    /// configuration. `Some(Depth32FloatStencil8)` draws the scene into a
    /// 32-bit float depth, which keeps reversed-Z precise on Vulkan and DX12,
    /// where the 24-bit format is unorm; it needs
    /// `Features::DEPTH32FLOAT_STENCIL8` (which
    /// [`ViewportRenderer::recommended_device_features`](crate::ViewportRenderer::recommended_device_features)
    /// requests) and without it falls back to `Depth24PlusStencil8` with a
    /// [`NoticeDetail::SceneDepthFormatFallback`](crate::NoticeDetail::SceneDepthFormatFallback).
    /// Any other format fails construction with
    /// [`ViewportError::InvalidSceneDepthFormat`](crate::ViewportError::InvalidSceneDepthFormat).
    ///
    /// Ask for the float format only where the renderer or your own code
    /// allocates the depth target: `owned()`, `render_offscreen`, or a pass you
    /// build yourself with an attachment of
    /// [`ViewportRenderer::scene_depth_format`](crate::ViewportRenderer::scene_depth_format).
    /// `ViewportApp` asks for it on its own when the device has the feature,
    /// since it owns the surface depth. A host that paints into a framework's
    /// pass ([`PassPath::paint`](crate::PassPath::paint), `paint_viewport`,
    /// `paint_hdr_blit`, the `*_with_depth` blits) leaves `None` unless it
    /// also made the framework allocate the float format.
    pub scene_depth_format: Option<crate::gpu::TextureFormat>,
    /// The point positions are measured from on their way to the GPU. A
    /// frame can replace it with
    /// [`SceneFrame::with_world_origin`](crate::SceneFrame::with_world_origin).
    /// Default [`WorldOrigin::default`](crate::WorldOrigin::default), which
    /// is [`snapped`](crate::WorldOrigin::snapped): the origin follows the
    /// camera, so precision does not fall away with distance from the world
    /// origin.
    pub world_origin: crate::renderer::WorldOrigin,
}

impl RendererConfig {
    /// The default settings for a renderer drawing into `target_format`.
    pub fn new(target_format: crate::gpu::TextureFormat) -> Self {
        Self {
            target_format,
            sample_count: 1,
            pipeline_cache_data: None,
            pipeline_cache: None,
            pipeline_compilation: None,
            geometry_chunk_bytes: None,
            downlevel_capabilities: None,
            backend: None,
            max_tier: None,
            device_lost_watcher: None,
            target_release_frames: None,
            scene_depth_format: None,
            world_origin: crate::renderer::WorldOrigin::default(),
        }
    }

    /// Set the colour format the renderer draws into.
    pub fn with_target_format(mut self, format: crate::gpu::TextureFormat) -> Self {
        self.target_format = format;
        self
    }

    /// Set the MSAA sample count (1, 2 or 4).
    pub fn with_sample_count(mut self, sample_count: u32) -> Self {
        self.sample_count = sample_count;
        self
    }

    /// Seed the pipeline cache from data an earlier run saved.
    pub fn with_pipeline_cache_data(mut self, data: Option<Vec<u8>>) -> Self {
        self.pipeline_cache_data = data;
        self
    }

    /// Share `cache` with the other renderers on the device instead of making
    /// a cache of this renderer's own. See [`pipeline_cache`](Self::pipeline_cache).
    pub fn with_pipeline_cache(mut self, cache: crate::gpu::PipelineCache) -> Self {
        self.pipeline_cache = Some(cache);
        self
    }

    /// Set the pipeline compilation policy the renderer starts with.
    pub fn with_pipeline_compilation(mut self, policy: PipelineCompilation) -> Self {
        self.pipeline_compilation = Some(policy);
        self
    }

    /// Set the size of the geometry store's first chunk, in bytes.
    pub fn with_geometry_chunk_bytes(mut self, bytes: u64) -> Self {
        self.geometry_chunk_bytes = Some(bytes);
        self
    }

    /// Pass the adapter's downlevel capabilities, so the renderer's
    /// [`Capabilities`](crate::Capabilities) read them rather than assume them.
    pub fn with_downlevel_capabilities(mut self, caps: crate::gpu::DownlevelCapabilities) -> Self {
        self.downlevel_capabilities = Some(caps);
        self
    }

    /// Run at `tier` at most, whatever the device offers. See
    /// [`max_tier`](Self::max_tier).
    pub fn with_max_tier(mut self, tier: crate::capabilities::Tier) -> Self {
        self.max_tier = Some(tier);
        self
    }

    /// Share a watcher the host installed on the device, rather than have the
    /// renderer install its own. See [`device_lost_watcher`](Self::device_lost_watcher).
    pub fn with_device_lost_watcher(mut self, watcher: crate::renderer::DeviceLostWatcher) -> Self {
        self.device_lost_watcher = Some(watcher);
        self
    }

    /// Release a viewport's unused targets after `frames` frames. See
    /// [`target_release_frames`](Self::target_release_frames).
    pub fn with_target_release_frames(mut self, frames: u32) -> Self {
        self.target_release_frames = Some(frames);
        self
    }

    /// Set the scene depth-stencil format. See
    /// [`scene_depth_format`](Self::scene_depth_format).
    pub fn with_scene_depth_format(mut self, format: crate::gpu::TextureFormat) -> Self {
        self.scene_depth_format = Some(format);
        self
    }

    /// Set the origin positions are drawn relative to. See
    /// [`world_origin`](Self::world_origin).
    pub fn with_world_origin(mut self, origin: crate::renderer::WorldOrigin) -> Self {
        self.world_origin = origin;
        self
    }

    /// Pass what the renderer reads from the adapter rather than assumes: its
    /// downlevel capabilities and its backend.
    pub fn with_adapter(mut self, adapter: &crate::gpu::Adapter) -> Self {
        self.downlevel_capabilities = Some(adapter.get_downlevel_capabilities());
        self.backend = Some(adapter.get_info().backend);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::RendererConfig;
    use crate::{PipelineCompilation, ViewportRenderer};

    /// The compilation policy and the geometry chunk size reach the renderer.
    #[test]
    fn with_config_applies_its_settings() {
        let Some((device, _queue)) = crate::resources::test_support::try_make_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let format = crate::gpu::TextureFormat::Rgba8UnormSrgb;
        let chunks = |config: &RendererConfig| {
            let mut renderer = ViewportRenderer::with_config(&device, config);
            for _ in 0..2 {
                renderer
                    .resources_mut()
                    .upload_mesh_data(&device, &crate::geometry::primitives::sphere(1.0, 32, 16))
                    .unwrap();
            }
            (
                renderer.pipeline_compilation(),
                renderer.resources.geometry.chunk_count(),
            )
        };
        for policy in [
            PipelineCompilation::Blocking,
            PipelineCompilation::Background,
        ] {
            let config = RendererConfig::new(format).with_pipeline_compilation(policy);
            assert_eq!(chunks(&config).0, policy);
        }
        let default_chunks = chunks(&RendererConfig::new(format)).1;
        let small_chunks = chunks(&RendererConfig::new(format).with_geometry_chunk_bytes(4096)).1;
        assert!(
            small_chunks > default_chunks,
            "a 4 KiB first chunk made {small_chunks} chunks, the default {default_chunks}"
        );
    }
}
