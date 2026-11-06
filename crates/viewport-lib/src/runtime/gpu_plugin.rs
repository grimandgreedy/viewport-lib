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
//! // host applies output (write_mesh_positions_normals, SkinningPlugin::attach_palette, ...)
//!
//! let plugin_bufs  = runtime.pre_prepare(device, queue, &gpu_ctx);
//! let prepare_bufs = renderer.pass().prepare(device, queue, &frame);
//! queue.submit(plugin_bufs.into_iter().chain(prepare_bufs));
//! ```
//!
//! wgpu submit ordering guarantees plugin command buffers complete before
//! `prepare()`'s, so a storage buffer written by a plugin is observable by
//! the standard render passes in the same frame.
//!
//! [`ViewportInstance`](crate::ViewportInstance) (and so `ViewportApp`) does
//! this for an attached runtime: its `render` and `prepare` run `pre_prepare`
//! and submit the plugins' buffers on the queue they are given before the
//! renderer's prepare, with the `dt` of the last `step_runtime`, a frame
//! counter, and the instance's [`viewport_id`](crate::ViewportInstance::viewport_id).
//! After the frame they run `post_paint` with the scene colour and depth:
//!
//! - HDR frames, owned or through a host's pass: the HDR scene colour and its
//!   depth. On the pass path the buffers come last in what `prepare` returns.
//! - LDR frames rendered into a target with `TEXTURE_BINDING` (an
//!   [`OffscreenViewportTarget`](crate::OffscreenViewportTarget)): that
//!   target and the LDR depth.
//! - Not called on an LDR frame painted into a host's render pass, rendered
//!   into a window surface or another target without `TEXTURE_BINDING`, or
//!   drawn at a reduced render scale or with backdrop-blur overlays (the
//!   scene then lands in an intermediate target). The renderer raises an
//!   info-level `NoticeDetail::PostPaintSkipped` notice instead. Under
//!   `ViewportApp` this means LDR windows.
//!
//! While GPU plugins are attached the renderer keeps the scene depth to the
//! end of the frame, which it otherwise discards. A host that renders
//! through the instance must not also call `pre_prepare` or `post_paint`
//! itself, or every plugin runs twice.

use crate::camera::Camera;
use crate::renderer::ViewportId;
use std::sync::{Arc, Mutex};

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

    /// Work that samples the rendered targets after paint: readbacks,
    /// capture and export, and effects rendered into a plugin-owned texture
    /// the host composites. This trait never slots work between the
    /// renderer's own internal passes.
    pub const POST_PAINT: i32 = 900;
}

/// Rendered target views handed to [`GpuPlugin::post_paint`].
///
/// The host owns these textures and is responsible for keeping them alive for
/// the duration of the `post_paint` call. A plugin that wants to *modify*
/// `color_view` should encode into a sibling target it owns; the host then
/// composites that overlay into the final image. The lib does not compose
/// plugin output back into the rendered color target for you.
#[non_exhaustive]
pub struct PostPaintTargets<'a> {
    /// View of the just-rendered color target.
    pub color_view: &'a crate::gpu::TextureView,
    /// View of the depth target produced during paint.
    pub depth_view: &'a crate::gpu::TextureView,
    /// View of the pick-id target (`R32Uint`) if the host renders one. `None`
    /// when the active path does not produce a pick-id texture (e.g. an eframe
    /// callback that did not request GPU picking this frame).
    pub pick_id_view: Option<&'a crate::gpu::TextureView>,
    /// Format of `color_view`. Useful when a plugin builds a pipeline whose
    /// output format must match.
    pub color_format: crate::gpu::TextureFormat,
}

impl<'a> PostPaintTargets<'a> {
    /// Construct targets without a pick-id view. A host that rendered a pick-id
    /// texture this frame chains [`with_pick_id_view`](Self::with_pick_id_view).
    pub fn new(
        color_view: &'a crate::gpu::TextureView,
        depth_view: &'a crate::gpu::TextureView,
        color_format: crate::gpu::TextureFormat,
    ) -> Self {
        Self {
            color_view,
            depth_view,
            pick_id_view: None,
            color_format,
        }
    }

    /// Supply the pick-id target view (`R32Uint`) for plugins that read it.
    pub fn with_pick_id_view(mut self, pick_id_view: &'a crate::gpu::TextureView) -> Self {
        self.pick_id_view = Some(pick_id_view);
        self
    }
}

/// Per-frame, read-only context passed to a [`GpuPlugin`].
///
/// Mirrors the render-relevant fields of [`RuntimeFrameContext`](super::RuntimeFrameContext)
/// without the input or pick state, which GPU plugins do not consume.
#[non_exhaustive]
pub struct GpuFrameContext<'a> {
    /// Active camera for this frame.
    pub camera: &'a Camera,
    /// Viewport size in physical pixels.
    pub viewport_size: glam::Vec2,
    /// Wall-clock seconds since the previous frame.
    pub dt: f32,
    /// Monotonically increasing frame counter, supplied by the host.
    pub frame_index: u64,
    /// Viewport this invocation is associated with, if the host renders
    /// multiple viewports per frame and calls `pre_prepare` / `post_paint`
    /// once per viewport. `None` for single-viewport hosts.
    ///
    /// Plugins registered via
    /// [`ViewportRuntime::with_gpu_plugin_for_viewport`](super::ViewportRuntime::with_gpu_plugin_for_viewport)
    /// only execute when this field matches the viewport they were scoped
    /// to. Unscoped plugins run on every call regardless of this field.
    pub viewport_id: Option<ViewportId>,
}

impl<'a> GpuFrameContext<'a> {
    /// Construct a context for a single-viewport host. Multi-viewport hosts that
    /// call `pre_prepare` / `post_paint` once per viewport chain
    /// [`with_viewport`](Self::with_viewport) to scope the call.
    pub fn new(camera: &'a Camera, viewport_size: glam::Vec2, dt: f32, frame_index: u64) -> Self {
        Self {
            camera,
            viewport_size,
            dt,
            frame_index,
            viewport_id: None,
        }
    }

    /// Associate this context with a specific viewport. Plugins registered via
    /// [`ViewportRuntime::with_gpu_plugin_for_viewport`](super::ViewportRuntime::with_gpu_plugin_for_viewport)
    /// run only when this id matches the one they were scoped to.
    pub fn with_viewport(mut self, viewport_id: ViewportId) -> Self {
        self.viewport_id = Some(viewport_id);
        self
    }
}

/// A plugin that encodes GPU work into the per-frame command stream.
///
/// Register with [`ViewportRuntime::with_gpu_plugin`](super::ViewportRuntime::with_gpu_plugin).
/// Each frame, after `runtime.step()` and before `renderer.prepare()`,
/// [`ViewportRuntime::pre_prepare`](super::ViewportRuntime::pre_prepare)
/// invokes every registered plugin's `pre_prepare` in ascending priority
/// order and returns the concatenated command buffers for submission. A
/// [`ViewportInstance`](crate::ViewportInstance) with the runtime attached
/// makes that call itself; a host driving a `ViewportRenderer` directly makes
/// it.
///
/// Registration model: GPU plugins are multi-instance, like
/// [`RuntimePlugin`](crate::runtime::RuntimePlugin) and unlike item-type
/// plugins. Registering two of the same type runs both in priority order; the
/// runtime does not deduplicate. This is intentional and frozen behavior.
///
/// [`ViewportRuntime::remove_gpu_plugin`](super::ViewportRuntime::remove_gpu_plugin)
/// hands the plugin back without calling any hook: dropping the returned box
/// releases the GPU resources it built in `init_gpu`.
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

    /// A stable name for this plugin, used to key per-plugin timing in
    /// [`RuntimeStats`](crate::RuntimeStats). Defaults to the concrete type
    /// name; override to give a friendlier label.
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// What the plugin needs from the device. Checked before the first
    /// `init_gpu`: a plugin whose requirements the device does not meet is
    /// removed and reported by
    /// [`ViewportRuntime::declined_gpu_plugins`](super::ViewportRuntime::declined_gpu_plugins).
    /// Defaults to the full tier.
    fn requires(&self) -> crate::capabilities::Requirements {
        crate::capabilities::Requirements::full_tier()
    }

    /// Build pipelines, persistent buffers, and bind group layouts.
    ///
    /// Called once per device, at the first `pre_prepare` after the plugin is
    /// registered and before any other hook. Registering another plugin does
    /// not call it again. After
    /// [`ViewportInstance::rebuild`](crate::ViewportInstance::rebuild) it runs
    /// again with the new device, and everything built on the old one has to
    /// be rebuilt here.
    fn init_gpu(&mut self, _device: &crate::gpu::Device) {}

    /// Encode work that runs before `renderer.prepare()`.
    ///
    /// CPU plugins have already stepped this frame; their outputs are visible
    /// through any shared state the consumer wired. The renderer has not
    /// started its own work yet, so anything written here is observable by
    /// `prepare()`'s passes (cluster build, shadow render, etc.).
    fn pre_prepare(
        &mut self,
        _device: &crate::gpu::Device,
        _queue: &crate::gpu::Queue,
        _ctx: &GpuFrameContext<'_>,
    ) -> Vec<crate::gpu::CommandBuffer> {
        Vec::new()
    }

    /// Encode work that runs after the frame is drawn.
    ///
    /// The plugin receives views of the just-rendered color, depth, and
    /// (optionally) pick-id targets and may sample them in a compute or
    /// fullscreen render pass. Typical uses: custom AO, motion blur,
    /// screen-space outlines, color grading LUTs.
    ///
    /// To *modify* the color target, write to a sibling texture the plugin
    /// owns and let the host composite it during the final blit. The lib
    /// does not loop a plugin's output back into the rendered color.
    ///
    /// Under a [`ViewportInstance`](crate::ViewportInstance) this is not
    /// called on frames that leave nothing readable; the
    /// [module docs](self) list them.
    fn post_paint(
        &mut self,
        _device: &crate::gpu::Device,
        _queue: &crate::gpu::Queue,
        _targets: &PostPaintTargets<'_>,
        _ctx: &GpuFrameContext<'_>,
    ) -> Vec<crate::gpu::CommandBuffer> {
        Vec::new()
    }
}

/// Handle returned by
/// [`ViewportRuntime::add_gpu_plugin`](super::ViewportRuntime::add_gpu_plugin);
/// pass it to
/// [`remove_gpu_plugin`](super::ViewportRuntime::remove_gpu_plugin) to
/// unregister the plugin. Ids count up from one per runtime, shared with
/// [`RuntimePluginId`](crate::RuntimePluginId)'s counter, and are never reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GpuPluginId(pub(crate) u64);

impl GpuPluginId {
    /// The raw id.
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Lets a plugin behind an `Arc<Mutex<P>>` register as a GPU plugin.
///
/// The GPU-side counterpart to the same impl on [`RuntimePlugin`](crate::RuntimePlugin). A feature
/// that implements both traits on one struct wraps it in `Arc<Mutex<..>>` and
/// registers the same object into the runtime and GPU lists by cloning the
/// `Arc`. `type_name` forwards to the inner plugin so `RuntimeStats` keys stay
/// meaningful. The host must not hold the lock across `runtime.step()` /
/// `pre_prepare()`: those run in sequence from the frame loop, so the mutex is
/// uncontended in that usage.
impl<P: GpuPlugin> GpuPlugin for Arc<Mutex<P>> {
    fn priority(&self) -> i32 {
        self.lock().unwrap().priority()
    }

    fn type_name(&self) -> &'static str {
        self.lock().unwrap().type_name()
    }

    fn init_gpu(&mut self, device: &crate::gpu::Device) {
        self.lock().unwrap().init_gpu(device);
    }

    fn pre_prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        ctx: &GpuFrameContext<'_>,
    ) -> Vec<crate::gpu::CommandBuffer> {
        self.lock().unwrap().pre_prepare(device, queue, ctx)
    }

    fn post_paint(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        targets: &PostPaintTargets<'_>,
        ctx: &GpuFrameContext<'_>,
    ) -> Vec<crate::gpu::CommandBuffer> {
        self.lock().unwrap().post_paint(device, queue, targets, ctx)
    }
}

/// Internal adapter that filters a [`GpuPlugin`] to a single viewport.
///
/// Constructed by
/// [`ViewportRuntime::with_gpu_plugin_for_viewport`](super::ViewportRuntime::with_gpu_plugin_for_viewport).
/// The wrapper forwards `priority` and `init_gpu` unconditionally, but only forwards `pre_prepare` / `post_paint` when
/// `ctx.viewport_id == Some(scoped_viewport)`.
pub(super) struct ViewportScopedPlugin<P: GpuPlugin> {
    pub(super) viewport: ViewportId,
    pub(super) inner: P,
}

impl<P: GpuPlugin> GpuPlugin for ViewportScopedPlugin<P> {
    fn priority(&self) -> i32 {
        self.inner.priority()
    }

    fn init_gpu(&mut self, device: &crate::gpu::Device) {
        self.inner.init_gpu(device);
    }

    fn pre_prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        ctx: &GpuFrameContext<'_>,
    ) -> Vec<crate::gpu::CommandBuffer> {
        if ctx.viewport_id == Some(self.viewport) {
            self.inner.pre_prepare(device, queue, ctx)
        } else {
            Vec::new()
        }
    }

    fn post_paint(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        targets: &PostPaintTargets<'_>,
        ctx: &GpuFrameContext<'_>,
    ) -> Vec<crate::gpu::CommandBuffer> {
        if ctx.viewport_id == Some(self.viewport) {
            self.inner.post_paint(device, queue, targets, ctx)
        } else {
            Vec::new()
        }
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

        fn init_gpu(&mut self, _device: &crate::gpu::Device) {
            *self.init_count.lock().unwrap() += 1;
        }

        fn pre_prepare(
            &mut self,
            _device: &crate::gpu::Device,
            _queue: &crate::gpu::Queue,
            _ctx: &GpuFrameContext<'_>,
        ) -> Vec<crate::gpu::CommandBuffer> {
            self.log.lock().unwrap().push(self.prio);
            Vec::new()
        }

        fn post_paint(
            &mut self,
            _device: &crate::gpu::Device,
            _queue: &crate::gpu::Queue,
            _targets: &PostPaintTargets<'_>,
            _ctx: &GpuFrameContext<'_>,
        ) -> Vec<crate::gpu::CommandBuffer> {
            self.post_log.lock().unwrap().push(self.prio);
            Vec::new()
        }
    }

    fn make_dummy_view(
        device: &crate::gpu::Device,
        format: crate::gpu::TextureFormat,
        usage: crate::gpu::TextureUsages,
    ) -> (crate::gpu::Texture, crate::gpu::TextureView) {
        let tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("post_paint_test_target"),
            size: crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::instance_from_env();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
                #[cfg(wgpu30)]
                apply_limit_buckets: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
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
        let ctx = GpuFrameContext::new(&camera, glam::Vec2::new(800.0, 600.0), 1.0 / 60.0, 0);

        let bufs = runtime.pre_prepare(&device, &queue, &ctx);
        assert!(
            bufs.is_empty(),
            "empty plugin returns should concatenate cleanly"
        );
        assert_eq!(
            *log.lock().unwrap(),
            vec![100, 200],
            "ascending priority order"
        );
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
            crate::gpu::TextureFormat::Rgba8UnormSrgb,
            crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
        );
        let (_depth_tex, depth_view) = make_dummy_view(
            &device,
            crate::gpu::TextureFormat::Depth32Float,
            crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
        );
        let targets = PostPaintTargets::new(
            &color_view,
            &depth_view,
            crate::gpu::TextureFormat::Rgba8UnormSrgb,
        );

        let bufs = runtime.post_paint(&device, &queue, &targets, &ctx);
        assert!(bufs.is_empty());
        assert_eq!(*post_log.lock().unwrap(), vec![100, 200]);
    }

    #[test]
    fn viewport_scoped_plugin_only_runs_for_matching_viewport() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };

        let log = Arc::new(Mutex::new(Vec::new()));
        let post_log = Arc::new(Mutex::new(Vec::new()));
        let init_count = Arc::new(Mutex::new(0));

        // The scoped viewport is constructed directly: ViewportId is opaque,
        // so we use `unsafe { std::mem::transmute }` would be wrong. Instead
        // rely on the `pub(crate)` constructor by going through the renderer's
        // `create_viewport` would need a renderer; for this test we synthesize
        // a ViewportId via the `Default` representation: it's a newtype around
        // `usize`, so `ViewportId(0)` is what `create_viewport` returns first.
        // ViewportId's inner field is pub(crate); in-crate test code can
        // construct synthetic ids without spinning up a full renderer.
        let vp_a = crate::renderer::ViewportId(0);
        let vp_b = crate::renderer::ViewportId(1);

        let mut runtime = crate::runtime::ViewportRuntime::new().with_gpu_plugin_for_viewport(
            vp_a,
            RecordingPlugin {
                prio: 100,
                log: log.clone(),
                post_log: post_log.clone(),
                init_count: init_count.clone(),
            },
        );

        let camera = Camera::default();
        let sz = glam::Vec2::new(1.0, 1.0);
        let ctx_a = GpuFrameContext::new(&camera, sz, 0.0, 0).with_viewport(vp_a);
        let ctx_b = GpuFrameContext::new(&camera, sz, 0.0, 0).with_viewport(vp_b);
        let ctx_none = GpuFrameContext::new(&camera, sz, 0.0, 0);

        let _ = runtime.pre_prepare(&device, &queue, &ctx_a);
        let _ = runtime.pre_prepare(&device, &queue, &ctx_b);
        let _ = runtime.pre_prepare(&device, &queue, &ctx_none);

        assert_eq!(*log.lock().unwrap(), vec![100], "only ran for vp_a");
        // init_gpu still runs unconditionally on first frame.
        assert_eq!(*init_count.lock().unwrap(), 1);
    }

    // One object implementing both traits, wrapped once and registered into
    // both lists via the Arc<Mutex<P>> blanket impls.
    struct HybridPlugin {
        step_calls: u32,
        pre_prepare_calls: u32,
    }

    impl crate::runtime::RuntimePlugin for HybridPlugin {
        fn priority(&self) -> i32 {
            crate::runtime::plugin::phase::ANIMATE
        }
        fn type_name(&self) -> &'static str {
            "hybrid_test_plugin"
        }
        fn step(&mut self, _ctx: &mut crate::runtime::RuntimeStepContext<'_>) {
            self.step_calls += 1;
        }
    }

    impl GpuPlugin for HybridPlugin {
        fn type_name(&self) -> &'static str {
            "hybrid_test_plugin"
        }
        fn pre_prepare(
            &mut self,
            _device: &crate::gpu::Device,
            _queue: &crate::gpu::Queue,
            _ctx: &GpuFrameContext<'_>,
        ) -> Vec<crate::gpu::CommandBuffer> {
            self.pre_prepare_calls += 1;
            Vec::new()
        }
    }

    #[test]
    fn arc_mutex_registers_as_both_kinds() {
        use crate::interaction::select::selection::Selection;
        use crate::runtime::RuntimeFrameContext;
        use crate::scene::scene::Scene;

        let hybrid = Arc::new(Mutex::new(HybridPlugin {
            step_calls: 0,
            pre_prepare_calls: 0,
        }));

        let mut runtime = crate::runtime::ViewportRuntime::new().with_plugin(hybrid.clone());
        runtime.add_gpu_plugin(hybrid.clone());

        // CPU path: the runtime step runs through the wrapper.
        let mut scene = Scene::new();
        let mut sel = Selection::new();
        let mut frame = RuntimeFrameContext::default();
        frame.dt = 1.0 / 60.0;
        runtime.step(&mut scene, &mut sel, &frame);

        assert_eq!(
            hybrid.lock().unwrap().step_calls,
            1,
            "runtime step ran through the Arc<Mutex<..>> wrapper"
        );
        assert!(
            runtime
                .last_stats()
                .step_ms
                .contains_key("hybrid_test_plugin"),
            "stats keyed by the inner type_name, got {:?}",
            runtime.last_stats().step_ms.keys().collect::<Vec<_>>()
        );

        // GPU path needs an adapter; skip the rest without one.
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping GPU-path assertions: no wgpu adapter available");
            return;
        };

        let camera = Camera::default();
        let ctx = GpuFrameContext::new(&camera, glam::Vec2::new(800.0, 600.0), 1.0 / 60.0, 0);
        let _ = runtime.pre_prepare(&device, &queue, &ctx);

        assert_eq!(
            hybrid.lock().unwrap().pre_prepare_calls,
            1,
            "gpu pre_prepare ran through the same wrapped object"
        );
        assert!(
            runtime
                .last_stats()
                .pre_prepare_ms
                .contains_key("hybrid_test_plugin"),
            "gpu stats keyed by the inner type_name"
        );
    }

    fn recording(
        prio: i32,
        log: &Arc<Mutex<Vec<i32>>>,
        post_log: &Arc<Mutex<Vec<i32>>>,
    ) -> (RecordingPlugin, Arc<Mutex<u32>>) {
        let init_count = Arc::new(Mutex::new(0));
        let plugin = RecordingPlugin {
            prio,
            log: log.clone(),
            post_log: post_log.clone(),
            init_count: init_count.clone(),
        };
        (plugin, init_count)
    }

    #[test]
    fn plugin_added_mid_session_is_initialised_alone() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let log = Arc::new(Mutex::new(Vec::new()));
        let post_log = Arc::new(Mutex::new(Vec::new()));
        let (first, init_first) = recording(100, &log, &post_log);
        let (second, init_second) = recording(50, &log, &post_log);

        let mut runtime = crate::runtime::ViewportRuntime::new().with_gpu_plugin(first);
        let camera = Camera::default();
        let ctx = GpuFrameContext::new(&camera, glam::Vec2::ONE, 0.0, 0);
        let _ = runtime.pre_prepare(&device, &queue, &ctx);
        assert!(runtime.gpu_initialized());

        runtime.add_gpu_plugin(second);
        assert!(!runtime.gpu_initialized());
        let _ = runtime.pre_prepare(&device, &queue, &ctx);
        assert!(runtime.gpu_initialized());
        assert_eq!(
            *init_first.lock().unwrap(),
            1,
            "earlier plugin re-initialised"
        );
        assert_eq!(*init_second.lock().unwrap(), 1);
        assert_eq!(*log.lock().unwrap(), vec![100, 50, 100]);
    }

    #[test]
    fn plugin_registered_before_post_paint_waits_for_pre_prepare() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let log = Arc::new(Mutex::new(Vec::new()));
        let post_log = Arc::new(Mutex::new(Vec::new()));
        let (first, _) = recording(100, &log, &post_log);
        let (late, init_late) = recording(200, &log, &post_log);

        let (_colour_tex, colour_view) = make_dummy_view(
            &device,
            crate::gpu::TextureFormat::Rgba8UnormSrgb,
            crate::gpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let (_depth_tex, depth_view) = make_dummy_view(
            &device,
            crate::gpu::TextureFormat::Depth32Float,
            crate::gpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let targets = PostPaintTargets::new(
            &colour_view,
            &depth_view,
            crate::gpu::TextureFormat::Rgba8UnormSrgb,
        );
        let camera = Camera::default();
        let ctx = GpuFrameContext::new(&camera, glam::Vec2::ONE, 0.0, 0);

        let mut runtime = crate::runtime::ViewportRuntime::new().with_gpu_plugin(first);
        let _ = runtime.pre_prepare(&device, &queue, &ctx);
        runtime.add_gpu_plugin(late);
        let _ = runtime.post_paint(&device, &queue, &targets, &ctx);
        assert_eq!(
            *post_log.lock().unwrap(),
            vec![100],
            "uninitialised plugin painted"
        );
        assert_eq!(*init_late.lock().unwrap(), 0);

        let _ = runtime.pre_prepare(&device, &queue, &ctx);
        let _ = runtime.post_paint(&device, &queue, &targets, &ctx);
        assert_eq!(*init_late.lock().unwrap(), 1);
        assert_eq!(*post_log.lock().unwrap(), vec![100, 100, 200]);
    }

    #[test]
    fn reset_gpu_init_initialises_every_plugin_again() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let log = Arc::new(Mutex::new(Vec::new()));
        let post_log = Arc::new(Mutex::new(Vec::new()));
        let (a, init_a) = recording(100, &log, &post_log);
        let (b, init_b) = recording(200, &log, &post_log);

        let mut runtime = crate::runtime::ViewportRuntime::new()
            .with_gpu_plugin(a)
            .with_gpu_plugin(b);
        let camera = Camera::default();
        let ctx = GpuFrameContext::new(&camera, glam::Vec2::ONE, 0.0, 0);
        let _ = runtime.pre_prepare(&device, &queue, &ctx);

        runtime.reset_gpu_init();
        assert!(!runtime.gpu_initialized());
        let _ = runtime.pre_prepare(&device, &queue, &ctx);
        assert!(runtime.gpu_initialized());
        assert_eq!(*init_a.lock().unwrap(), 2);
        assert_eq!(*init_b.lock().unwrap(), 2);

        let _ = runtime.pre_prepare(&device, &queue, &ctx);
        assert_eq!(*init_a.lock().unwrap(), 2);
        assert_eq!(*init_b.lock().unwrap(), 2);
    }
}
