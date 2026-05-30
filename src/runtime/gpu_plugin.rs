//! GPU plugin trait and lifecycle hooks for runtime extensions.
//!
//! `GpuPlugin` is the GPU-side counterpart to [`RuntimePlugin`](super::RuntimePlugin).
//! Where `RuntimePlugin` mutates the scene each frame in `step`, `GpuPlugin`
//! encodes wgpu command buffers at well-defined lifecycle points around the
//! renderer's own work. The two traits are independent: a plugin may implement
//! one, the other, or both and register itself separately with
//! [`ViewportRuntime::with_plugin`](super::ViewportRuntime::with_plugin) and
//! [`ViewportRuntime::with_gpu_plugin`](super::ViewportRuntime::with_gpu_plugin).
//!
//! # Frame integration
//!
//! ```text
//! output = runtime.step(scene, selection, &frame_ctx);
//! // host applies output (write_mesh_positions_normals, set_skin_palette, ...)
//!
//! let plugin_bufs  = runtime.pre_prepare(device, queue, &gpu_ctx);
//! let prepare_bufs = renderer.pass().prepare(device, queue, &frame);
//! queue.submit(plugin_bufs.into_iter().chain(prepare_bufs));
//! ```
//!
//! wgpu submit ordering guarantees plugin command buffers complete before
//! `prepare()`'s, so a storage buffer written by a plugin is observable by
//! the standard render passes in the same frame.

use crate::camera::Camera;

/// Priority bands for GPU plugin lifecycle points.
///
/// Mirrors [`super::plugin::phase`] for CPU plugins. Use a band constant as
/// the base of [`GpuPlugin::priority`] and offset within it (e.g.
/// `gpu_phase::PRE_PREPARE + 10`) to order plugins inside the same band.
pub mod gpu_phase {
    /// Compute that produces inputs read by `renderer.prepare()` and the
    /// standard render passes: cloth, hair, GPU particles, morph blends,
    /// audio-reactive displacement, probe capture.
    pub const PRE_PREPARE: i32 = 100;

    /// Reserved priority band between `PRE_PREPARE` and `POST_PAINT`.
    /// Currently unused; plugins cannot slot work between the renderer's own
    /// internal passes.
    pub const _RESERVED_INTERNAL: i32 = 500;

    /// Compute that samples rendered targets (custom AO, motion blur,
    /// screen-space outline, color grading).
    pub const POST_PAINT: i32 = 900;
}

/// Rendered target views handed to [`GpuPlugin::post_paint`].
///
/// The host owns these textures and is responsible for keeping them alive for
/// the duration of the `post_paint` call. A plugin that wants to *modify*
/// `color_view` should encode into a sibling target it owns; the host then
/// composites that overlay into the final image. The lib does not compose
/// plugin output back into the rendered color target for you.
pub struct PostPaintTargets<'a> {
    /// View of the just-rendered color target.
    pub color_view: &'a wgpu::TextureView,
    /// View of the depth target produced during paint.
    pub depth_view: &'a wgpu::TextureView,
    /// View of the pick-id target (`R32Uint`) if the host renders one. `None`
    /// when the active path does not produce a pick-id texture (e.g. an eframe
    /// callback that did not request GPU picking this frame).
    pub pick_id_view: Option<&'a wgpu::TextureView>,
    /// Format of `color_view`. Useful when a plugin builds a pipeline whose
    /// output format must match.
    pub color_format: wgpu::TextureFormat,
}

/// Per-frame, read-only context passed to a [`GpuPlugin`].
///
/// Mirrors the render-relevant fields of [`RuntimeFrameContext`](super::RuntimeFrameContext)
/// without the input or pick state, which GPU plugins do not consume.
pub struct GpuFrameContext<'a> {
    /// Active camera for this frame.
    pub camera: &'a Camera,
    /// Viewport size in physical pixels.
    pub viewport_size: glam::Vec2,
    /// Wall-clock seconds since the previous frame.
    pub dt: f32,
    /// Monotonically increasing frame counter, supplied by the host.
    pub frame_index: u64,
}

/// A plugin that encodes GPU work into the per-frame command stream.
///
/// Register with [`ViewportRuntime::with_gpu_plugin`](super::ViewportRuntime::with_gpu_plugin).
/// Each frame, after `runtime.step()` and before `renderer.prepare()`, the
/// host calls [`ViewportRuntime::pre_prepare`](super::ViewportRuntime::pre_prepare),
/// which invokes every registered plugin's `pre_prepare` in ascending priority
/// order and returns the concatenated command buffers for submission.
///
/// # Example
///
/// ```rust,ignore
/// use viewport_lib::runtime::{GpuFrameContext, GpuPlugin, gpu_phase};
///
/// struct WarpPlugin {
///     pipeline: Option<wgpu::ComputePipeline>,
///     bind_group: Option<wgpu::BindGroup>,
/// }
///
/// impl GpuPlugin for WarpPlugin {
///     fn priority(&self) -> i32 { gpu_phase::PRE_PREPARE }
///
///     fn init_gpu(&mut self, device: &wgpu::Device) {
///         // build pipeline and bind group, store on self
///     }
///
///     fn pre_prepare(
///         &mut self,
///         device: &wgpu::Device,
///         _queue: &wgpu::Queue,
///         _ctx: &GpuFrameContext<'_>,
///     ) -> Vec<wgpu::CommandBuffer> {
///         let mut enc = device.create_command_encoder(&Default::default());
///         {
///             let mut pass = enc.begin_compute_pass(&Default::default());
///             pass.set_pipeline(self.pipeline.as_ref().unwrap());
///             pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
///             pass.dispatch_workgroups(64, 1, 1);
///         }
///         vec![enc.finish()]
///     }
/// }
/// ```
pub trait GpuPlugin: Send + 'static {
    /// Ascending priority. Lower runs first. Use [`gpu_phase`] constants as
    /// base values. Defaults to [`gpu_phase::PRE_PREPARE`] since that is the
    /// only active band today.
    fn priority(&self) -> i32 {
        gpu_phase::PRE_PREPARE
    }

    /// Build pipelines, persistent buffers, and bind group layouts.
    ///
    /// Called once before the plugin's first `pre_prepare`. If a new GPU
    /// plugin is registered after the runtime has already run, every plugin's
    /// `init_gpu` is invoked again on the next frame; implementations should
    /// either be idempotent or guard their own one-time setup.
    fn init_gpu(&mut self, _device: &wgpu::Device) {}

    /// Encode work that runs before `renderer.prepare()`.
    ///
    /// CPU plugins have already stepped this frame; their outputs are visible
    /// through any shared state the consumer wired. The renderer has not
    /// started its own work yet, so anything written here is observable by
    /// `prepare()`'s passes (cluster build, shadow render, etc.).
    fn pre_prepare(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _ctx: &GpuFrameContext<'_>,
    ) -> Vec<wgpu::CommandBuffer> {
        Vec::new()
    }

    /// Encode work that runs after `renderer.paint_to()`.
    ///
    /// The plugin receives views of the just-rendered color, depth, and
    /// (optionally) pick-id targets and may sample them in a compute or
    /// fullscreen render pass. Typical uses: custom AO, motion blur,
    /// screen-space outlines, color grading LUTs.
    ///
    /// To *modify* the color target, write to a sibling texture the plugin
    /// owns and let the host composite it during the final blit. The lib
    /// does not loop a plugin's output back into the rendered color.
    fn post_paint(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _targets: &PostPaintTargets<'_>,
        _ctx: &GpuFrameContext<'_>,
    ) -> Vec<wgpu::CommandBuffer> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct RecordingPlugin {
        prio: i32,
        log: Arc<Mutex<Vec<i32>>>,
        post_log: Arc<Mutex<Vec<i32>>>,
        init_count: Arc<Mutex<u32>>,
    }

    impl GpuPlugin for RecordingPlugin {
        fn priority(&self) -> i32 {
            self.prio
        }

        fn init_gpu(&mut self, _device: &wgpu::Device) {
            *self.init_count.lock().unwrap() += 1;
        }

        fn pre_prepare(
            &mut self,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _ctx: &GpuFrameContext<'_>,
        ) -> Vec<wgpu::CommandBuffer> {
            self.log.lock().unwrap().push(self.prio);
            Vec::new()
        }

        fn post_paint(
            &mut self,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _targets: &PostPaintTargets<'_>,
            _ctx: &GpuFrameContext<'_>,
        ) -> Vec<wgpu::CommandBuffer> {
            self.post_log.lock().unwrap().push(self.prio);
            Vec::new()
        }
    }

    fn make_dummy_view(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post_paint_test_target"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn try_make_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    #[test]
    fn priority_order_and_init_once() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };

        let log = Arc::new(Mutex::new(Vec::new()));
        let post_log = Arc::new(Mutex::new(Vec::new()));
        let init_a = Arc::new(Mutex::new(0));
        let init_b = Arc::new(Mutex::new(0));

        let mut runtime = crate::runtime::ViewportRuntime::new()
            .with_gpu_plugin(RecordingPlugin {
                prio: 200,
                log: log.clone(),
                post_log: post_log.clone(),
                init_count: init_a.clone(),
            })
            .with_gpu_plugin(RecordingPlugin {
                prio: 100,
                log: log.clone(),
                post_log: post_log.clone(),
                init_count: init_b.clone(),
            });

        let camera = Camera::default();
        let ctx = GpuFrameContext {
            camera: &camera,
            viewport_size: glam::Vec2::new(800.0, 600.0),
            dt: 1.0 / 60.0,
            frame_index: 0,
        };

        let bufs = runtime.pre_prepare(&device, &queue, &ctx);
        assert!(bufs.is_empty(), "empty plugin returns should concatenate cleanly");
        assert_eq!(*log.lock().unwrap(), vec![100, 200], "ascending priority order");
        assert_eq!(*init_a.lock().unwrap(), 1);
        assert_eq!(*init_b.lock().unwrap(), 1);

        // Second frame: init_gpu is not called again.
        let _ = runtime.pre_prepare(&device, &queue, &ctx);
        assert_eq!(*init_a.lock().unwrap(), 1);
        assert_eq!(*init_b.lock().unwrap(), 1);
        assert_eq!(*log.lock().unwrap(), vec![100, 200, 100, 200]);

        // post_paint: same priority contract.
        let (_color_tex, color_view) = make_dummy_view(
            &device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let (_depth_tex, depth_view) = make_dummy_view(
            &device,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let targets = PostPaintTargets {
            color_view: &color_view,
            depth_view: &depth_view,
            pick_id_view: None,
            color_format: wgpu::TextureFormat::Rgba8UnormSrgb,
        };

        let bufs = runtime.post_paint(&device, &queue, &targets, &ctx);
        assert!(bufs.is_empty());
        assert_eq!(*post_log.lock().unwrap(), vec![100, 200]);
    }
}
