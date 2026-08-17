//! Installing a feature in one call.
//!
//! A non-trivial feature usually spans several plugin kinds: a deformer, a
//! runtime plugin, a GPU plugin, an item type. Each kind registers on a
//! different object (the renderer, its resources, the runtime), so wiring a
//! feature in by hand means the host has to know which pieces the feature has
//! and which object each one registers on. Adding a piece to a feature then
//! breaks every host's setup code.
//!
//! [`ViewportPlugin`] moves that knowledge into the feature. The host builds a
//! [`PluginInstallCtx`] from the two objects it already owns (the renderer and,
//! if it uses one, the runtime) and makes one call:
//!
//! ```rust,ignore
//! let mut ctx = PluginInstallCtx::new(&device, &queue, Some(&mut runtime), &mut renderer);
//! let skinning = SkinnedActorFeature::default().install(&mut ctx)?;
//! ```
//!
//! Inside `install`, the feature calls the registration methods that already
//! exist (`renderer.with_item_type_plugin`,
//! `renderer.resources_mut().register_deformer`, `runtime.add_plugin`,
//! `runtime.add_gpu_plugin`) and returns whatever handle it wants the host to
//! keep. The context is a bundle of borrows, not a new registration surface.

use crate::error::{ViewportError, ViewportResult};
use crate::renderer::ViewportRenderer;
use crate::runtime::ViewportRuntime;

/// The objects a feature borrows at install time.
///
/// Constructed by the host from what it already owns and passed to
/// [`ViewportPlugin::install`]. The fields are borrows: nothing here owns the
/// renderer or the runtime, and the two stay independent of each other.
///
/// `runtime` is optional because the runtime is optional: the minimal hosts
/// never build one. A feature that needs it calls
/// [`require_runtime`](Self::require_runtime), which returns
/// [`ViewportError::PluginInstallMissing`] naming the feature rather than
/// panicking. A feature whose runtime half is optional installs what it can.
///
/// The struct is `#[non_exhaustive]`: new fields may be appended over time, so
/// construct it through [`new`](Self::new) rather than a struct literal.
#[non_exhaustive]
pub struct PluginInstallCtx<'a> {
    /// The device features build GPU resources against.
    pub device: &'a crate::gpu::Device,
    /// The queue features upload initial data through.
    pub queue: &'a crate::gpu::Queue,
    /// The runtime, if the host uses one. `None` on hosts that never build a
    /// [`ViewportRuntime`].
    pub runtime: Option<&'a mut ViewportRuntime>,
    /// The renderer. Always present: a host that installs a feature owns one.
    pub renderer: &'a mut ViewportRenderer,
}

impl<'a> PluginInstallCtx<'a> {
    /// Bundle the borrows a feature needs to install itself. Pass `None` for
    /// `runtime` on a host that does not use one.
    pub fn new(
        device: &'a crate::gpu::Device,
        queue: &'a crate::gpu::Queue,
        runtime: Option<&'a mut ViewportRuntime>,
        renderer: &'a mut ViewportRenderer,
    ) -> Self {
        Self {
            device,
            queue,
            runtime,
            renderer,
        }
    }

    /// Borrow the runtime, or fail with [`ViewportError::PluginInstallMissing`]
    /// naming what needs it. `needed` is a short human-readable label for the
    /// piece, e.g. `"a ViewportRuntime for the skeleton plugin"`.
    ///
    /// Call this before doing any renderer-side registration so a missing
    /// runtime does not leave the feature half-installed.
    pub fn require_runtime(
        &mut self,
        needed: &'static str,
    ) -> ViewportResult<&mut ViewportRuntime> {
        self.runtime
            .as_deref_mut()
            .ok_or(ViewportError::PluginInstallMissing { needed })
    }
}

/// A feature that wires itself into the renderer, and optionally the runtime,
/// in one call.
///
/// `install` registers every piece the feature needs and returns the handle the
/// host keeps around: an upload handle, a query handle, or `()` for a feature
/// with nothing to hand back. This does not change how any plugin kind runs;
/// it is one entry point for wiring a feature in.
///
/// Make `install` idempotent where a piece is keyed by name, following the
/// [`SkinningPlugin::install`](crate::plugins::skinning::SkinningPlugin::install)
/// convention: if a deformer with the feature's name is already registered,
/// reuse its id instead of erroring. Runtime and GPU plugins are multi-instance
/// by contract, so `install` is a once-per-feature call, not a deduplicating
/// one.
pub trait ViewportPlugin {
    /// What the host gets back and keeps: an upload handle, a query handle, or
    /// `()`.
    type Handle;

    /// Register the feature's pieces against `ctx` and return its handle.
    fn install(self, ctx: &mut PluginInstallCtx<'_>) -> ViewportResult<Self::Handle>;
}

/// Install a feature without writing out the [`PluginInstallCtx`] by hand.
///
/// ```rust,ignore
/// let handle = viewport_lib::install_plugin(
///     feature, &device, &queue, Some(&mut runtime), &mut renderer,
/// )?;
/// ```
///
/// Equivalent to building a [`PluginInstallCtx`] and calling
/// [`ViewportPlugin::install`]. Pass `None` for `runtime` on a host that does
/// not use one.
pub fn install_plugin<P: ViewportPlugin>(
    feature: P,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    runtime: Option<&mut ViewportRuntime>,
    renderer: &mut ViewportRenderer,
) -> ViewportResult<P::Handle> {
    let mut ctx = PluginInstallCtx::new(device, queue, runtime, renderer);
    feature.install(&mut ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::ViewportRenderer;
    use crate::resources::mesh_sidecar::registry::{DeformStage, DeformerDesc};
    use crate::runtime::plugin::phase;
    use crate::runtime::{RuntimePlugin, RuntimeStepContext, ViewportRuntime};
    use std::sync::{Arc, Mutex};

    fn headless() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
            required_limits: crate::renderer::ViewportRenderer::recommended_device_limits(&adapter),
            ..Default::default()
        }))
        .ok()
    }

    // Bumps a shared counter each step; lets the test see the runtime half ran.
    struct CounterPlugin {
        count: Arc<Mutex<u32>>,
    }

    impl RuntimePlugin for CounterPlugin {
        fn priority(&self) -> i32 {
            phase::ANIMATE
        }
        fn step(&mut self, _ctx: &mut RuntimeStepContext<'_>) {
            *self.count.lock().unwrap() += 1;
        }
    }

    const IDENTITY_BODY: &str =
        "fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {\n    return v;\n}\n";

    // Registers a deformer on the renderer and a runtime plugin, proving one
    // install call reaches both objects.
    struct TestFeature {
        count: Arc<Mutex<u32>>,
    }

    #[derive(Debug)]
    struct TestHandle {
        deformer_id: crate::resources::mesh_sidecar::registry::DeformerId,
    }

    impl ViewportPlugin for TestFeature {
        type Handle = TestHandle;

        fn install(self, ctx: &mut PluginInstallCtx<'_>) -> ViewportResult<TestHandle> {
            // Validate the runtime is present before touching the renderer, so
            // a missing runtime leaves nothing half-registered.
            ctx.require_runtime("a ViewportRuntime for the test feature")?
                .add_plugin(CounterPlugin {
                    count: self.count.clone(),
                });

            let device = ctx.device;
            let deformer_id = ctx.renderer.resources_mut().register_deformer(
                device,
                DeformerDesc {
                    name: "test_feature",
                    stage: DeformStage::ObjectSpace,
                    priority: 0,
                    wgsl_body: IDENTITY_BODY.to_string(),
                    per_vertex_stride: 4,
                },
            )?;

            Ok(TestHandle { deformer_id })
        }
    }

    #[test]
    fn install_wires_renderer_and_runtime_in_one_call() {
        let Some((device, queue)) = headless() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };

        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let mut runtime = ViewportRuntime::new();
        let count = Arc::new(Mutex::new(0));

        let handle = {
            let mut ctx = PluginInstallCtx::new(&device, &queue, Some(&mut runtime), &mut renderer);
            TestFeature {
                count: count.clone(),
            }
            .install(&mut ctx)
            .expect("install")
        };

        // Renderer side: the deformer is registered under its name.
        assert_eq!(
            renderer.resources().deformer_id_by_name("test_feature"),
            Some(handle.deformer_id),
            "install registered the deformer on the renderer"
        );

        // Runtime side: stepping runs the registered plugin.
        let mut scene = crate::scene::scene::Scene::new();
        let mut sel = crate::interaction::select::selection::Selection::new();
        let mut frame = crate::runtime::RuntimeFrameContext::default();
        frame.dt = 1.0 / 60.0;
        runtime.step(&mut scene, &mut sel, &frame);
        assert_eq!(
            *count.lock().unwrap(),
            1,
            "install registered the runtime plugin"
        );
    }

    #[test]
    fn install_without_runtime_names_the_missing_piece() {
        let Some((device, queue)) = headless() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };

        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let count = Arc::new(Mutex::new(0));

        let mut ctx = PluginInstallCtx::new(&device, &queue, None, &mut renderer);
        let result = TestFeature {
            count: count.clone(),
        }
        .install(&mut ctx);

        match result {
            Err(ViewportError::PluginInstallMissing { needed }) => {
                assert!(
                    needed.contains("ViewportRuntime"),
                    "names the missing piece"
                );
            }
            other => panic!("expected PluginInstallMissing, got {other:?}"),
        }

        // The feature checked the runtime first, so nothing was registered on
        // the renderer.
        assert_eq!(
            renderer.resources().deformer_id_by_name("test_feature"),
            None,
            "a missing runtime leaves the renderer untouched"
        );
    }
}
