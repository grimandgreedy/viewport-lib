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
}

impl RendererConfig {
    /// The default settings for a renderer drawing into `target_format`.
    pub fn new(target_format: crate::gpu::TextureFormat) -> Self {
        Self {
            target_format,
            sample_count: 1,
            pipeline_cache_data: None,
            pipeline_compilation: None,
            geometry_chunk_bytes: None,
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
