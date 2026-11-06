//! Installing a feature in one call.
//!
//! A non-trivial feature usually spans several plugin kinds: a deformer, a
//! runtime plugin, a GPU plugin, an item type. Each kind registers on a
//! different object (the renderer, its resources, the runtime), so wiring a
//! feature in by hand means the host has to know which pieces the feature has
//! and which object each one registers on. Adding a piece to a feature then
//! breaks every host's setup code.
//!
//! [`PluginInstaller`] moves that knowledge into the feature. The host builds a
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
//!
//! A feature implements three calls:
//!
//! - [`install`](PluginInstaller::install) registers every piece and returns
//!   the handle.
//! - [`uninstall`](PluginInstaller::uninstall) takes the handle back and calls
//!   the matching `remove_*` / `unregister_*` methods, so the renderer and
//!   runtime read as if the feature had never been installed.
//! - [`reattach_renderer`](PluginInstaller::reattach_renderer) registers the
//!   renderer-side pieces again on a new renderer, for a host that rebuilt
//!   its renderer after a device loss and kept its runtime. The runtime-side
//!   pieces are still registered, so installing again would run them twice.
//!
//! ```rust,ignore
//! viewport_lib::uninstall_plugin::<MyFeature>(
//!     handle, &device, &queue, Some(&mut runtime), &mut renderer,
//! )?;
//!
//! // After building a new renderer, with the runtime kept:
//! viewport_lib::reattach_plugin::<MyFeature>(
//!     &mut handle, &device, &queue, Some(&mut runtime), &mut new_renderer,
//! )?;
//! ```

use crate::error::{ViewportError, ViewportResult};
use crate::renderer::ViewportRenderer;
use crate::runtime::ViewportRuntime;

/// The objects a feature borrows at install time.
///
/// Constructed by the host from what it already owns and passed to
/// [`PluginInstaller::install`]. The fields are borrows: nothing here owns the
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
/// one: after a renderer rebuild, bring the feature back with
/// [`reattach_renderer`](Self::reattach_renderer), not `install`.
pub trait PluginInstaller {
    /// What the host gets back and keeps: an upload handle, a query handle, or
    /// `()`.
    type Handle;

    /// Register the feature's pieces against `ctx` and return its handle.
    fn install(self, ctx: &mut PluginInstallCtx<'_>) -> ViewportResult<Self::Handle>;

    /// Remove what [`install`](Self::install) registered, consuming the
    /// handle it returned.
    ///
    /// Call the `remove_*` / `unregister_*` method of each piece the feature
    /// registered, so the renderer and runtime read as if the feature had
    /// never been installed. Attempt every piece even when one fails, and
    /// return the first error, so one failure does not leave the rest
    /// registered. A feature whose pieces cannot be removed returns
    /// [`ViewportError::PluginUninstallUnsupported`] rather than `Ok(())`.
    fn uninstall(handle: Self::Handle, ctx: &mut PluginInstallCtx<'_>) -> ViewportResult<()>
    where
        Self: Sized;

    /// Register the feature's renderer-side pieces again on `ctx.renderer`
    /// and update the ids in `handle` to the new ones.
    ///
    /// Called after the host replaced its renderer (for example with
    /// [`ViewportInstance::rebuild`](crate::ViewportInstance::rebuild) after a
    /// device loss) and kept its runtime. The runtime-side pieces are still
    /// registered: leave the runtime alone. This restores registrations, not
    /// per-mesh data such as deformer slot data or skin weights, which the
    /// host attaches again to the meshes it uploads again, as it did after
    /// `install`. A feature with nothing on the renderer returns `Ok(())`.
    ///
    /// Attempt every piece even when one fails, and return the first error.
    fn reattach_renderer(
        handle: &mut Self::Handle,
        ctx: &mut PluginInstallCtx<'_>,
    ) -> ViewportResult<()>
    where
        Self: Sized;
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
/// [`PluginInstaller::install`]. Pass `None` for `runtime` on a host that does
/// not use one.
pub fn install_plugin<P: PluginInstaller>(
    feature: P,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    runtime: Option<&mut ViewportRuntime>,
    renderer: &mut ViewportRenderer,
) -> ViewportResult<P::Handle> {
    let mut ctx = PluginInstallCtx::new(device, queue, runtime, renderer);
    feature.install(&mut ctx)
}

/// Uninstall a feature without writing out the [`PluginInstallCtx`] by hand:
/// [`PluginInstaller::uninstall`] for the feature type `P`.
///
/// ```rust,ignore
/// viewport_lib::uninstall_plugin::<MyFeature>(
///     handle, &device, &queue, Some(&mut runtime), &mut renderer,
/// )?;
/// ```
pub fn uninstall_plugin<P: PluginInstaller>(
    handle: P::Handle,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    runtime: Option<&mut ViewportRuntime>,
    renderer: &mut ViewportRenderer,
) -> ViewportResult<()> {
    let mut ctx = PluginInstallCtx::new(device, queue, runtime, renderer);
    P::uninstall(handle, &mut ctx)
}

/// Reattach a feature to a new renderer without writing out the
/// [`PluginInstallCtx`] by hand: [`PluginInstaller::reattach_renderer`] for the
/// feature type `P`.
///
/// ```rust,ignore
/// viewport_lib::reattach_plugin::<MyFeature>(
///     &mut handle, &device, &queue, Some(&mut runtime), &mut renderer,
/// )?;
/// ```
pub fn reattach_plugin<P: PluginInstaller>(
    handle: &mut P::Handle,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    runtime: Option<&mut ViewportRuntime>,
    renderer: &mut ViewportRenderer,
) -> ViewportResult<()> {
    let mut ctx = PluginInstallCtx::new(device, queue, runtime, renderer);
    P::reattach_renderer(handle, &mut ctx)
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
        step_plugin: crate::runtime::RuntimePluginId,
    }

    fn test_deformer() -> DeformerDesc {
        DeformerDesc {
            name: "test_feature".into(),
            stage: DeformStage::ObjectSpace,
            priority: 0,
            wgsl_body: IDENTITY_BODY.to_string(),
            per_vertex_stride: 4,
        }
    }

    impl PluginInstaller for TestFeature {
        type Handle = TestHandle;

        fn install(self, ctx: &mut PluginInstallCtx<'_>) -> ViewportResult<TestHandle> {
            // Validate the runtime is present before touching the renderer, so
            // a missing runtime leaves nothing half-registered.
            let step_plugin = ctx
                .require_runtime("a ViewportRuntime for the test feature")?
                .add_plugin(CounterPlugin {
                    count: self.count.clone(),
                });

            let device = ctx.device;
            let deformer_id = ctx
                .renderer
                .resources_mut()
                .register_deformer(device, test_deformer())?;

            Ok(TestHandle {
                deformer_id,
                step_plugin,
            })
        }

        // The deformer first, then the runtime plugin; each is attempted
        // whatever happened to the other.
        fn uninstall(handle: TestHandle, ctx: &mut PluginInstallCtx<'_>) -> ViewportResult<()> {
            let deformer = ctx
                .renderer
                .resources_mut()
                .remove_deformer(ctx.device, handle.deformer_id);
            let runtime = ctx
                .require_runtime("a ViewportRuntime for the test feature")
                .map(|runtime| {
                    runtime.remove_plugin(handle.step_plugin);
                });
            deformer.and(runtime)
        }

        fn reattach_renderer(
            handle: &mut TestHandle,
            ctx: &mut PluginInstallCtx<'_>,
        ) -> ViewportResult<()> {
            handle.deformer_id = ctx
                .renderer
                .resources_mut()
                .register_deformer(ctx.device, test_deformer())?;
            Ok(())
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

    fn renderer(device: &crate::gpu::Device) -> ViewportRenderer {
        ViewportRenderer::new(device, crate::gpu::TextureFormat::Bgra8UnormSrgb)
    }

    fn install_test_feature(
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        runtime: &mut ViewportRuntime,
        renderer: &mut ViewportRenderer,
    ) -> TestHandle {
        install_plugin(
            TestFeature {
                count: Arc::new(Mutex::new(0)),
            },
            device,
            queue,
            Some(runtime),
            renderer,
        )
        .expect("install")
    }

    #[test]
    fn built_in_runtime_features_uninstall_to_the_state_before_install() {
        let Some((device, queue)) = headless() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut renderer = renderer(&device);
        let mut runtime = ViewportRuntime::new();
        let deformers = renderer.resources().registered_deformer_count();

        fn round_trip<P: PluginInstaller<Handle = crate::runtime::RuntimePluginId>>(
            feature: P,
            device: &crate::gpu::Device,
            queue: &crate::gpu::Queue,
            runtime: &mut ViewportRuntime,
            renderer: &mut ViewportRenderer,
        ) {
            let id = install_plugin(feature, device, queue, Some(&mut *runtime), renderer)
                .expect("install");
            assert_eq!(runtime.plugin_ids().collect::<Vec<_>>(), vec![id]);
            uninstall_plugin::<P>(id, device, queue, Some(&mut *runtime), renderer)
                .expect("uninstall");
            assert_eq!(runtime.plugin_count(), 0);
        }
        round_trip(
            crate::plugins::physics_lite::PhysicsLitePlugin::new(),
            &device,
            &queue,
            &mut runtime,
            &mut renderer,
        );
        round_trip(
            crate::plugins::animation::AnimationPlugin::new(),
            &device,
            &queue,
            &mut runtime,
            &mut renderer,
        );
        round_trip(
            crate::plugins::constraint::ConstraintPlugin::new(),
            &device,
            &queue,
            &mut runtime,
            &mut renderer,
        );
        assert_eq!(renderer.resources().registered_deformer_count(), deformers);
    }

    #[test]
    fn uninstall_removes_every_piece_it_can() {
        let Some((device, queue)) = headless() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };

        // The runtime piece cannot be reached: the deformer still goes.
        let mut renderer = renderer(&device);
        let mut runtime = ViewportRuntime::new();
        let handle = install_test_feature(&device, &queue, &mut runtime, &mut renderer);
        let result = uninstall_plugin::<TestFeature>(handle, &device, &queue, None, &mut renderer);
        assert!(
            matches!(result, Err(ViewportError::PluginInstallMissing { .. })),
            "{result:?}"
        );
        assert_eq!(
            renderer.resources().deformer_id_by_name("test_feature"),
            None
        );

        // The deformer was already removed by hand: the runtime plugin still
        // goes.
        let mut renderer = self::renderer(&device);
        let mut runtime = ViewportRuntime::new();
        let handle = install_test_feature(&device, &queue, &mut runtime, &mut renderer);
        renderer
            .resources_mut()
            .remove_deformer(&device, handle.deformer_id)
            .expect("remove by hand");
        let result = uninstall_plugin::<TestFeature>(
            handle,
            &device,
            &queue,
            Some(&mut runtime),
            &mut renderer,
        );
        assert!(
            matches!(result, Err(ViewportError::StaleHandle { .. })),
            "{result:?}"
        );
        assert_eq!(runtime.plugin_count(), 0);
    }

    #[test]
    fn reattach_registers_the_renderer_pieces_without_a_second_runtime_plugin() {
        let Some((device, queue)) = headless() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut old = renderer(&device);
        let mut runtime = ViewportRuntime::new();
        let mut handle = install_test_feature(&device, &queue, &mut runtime, &mut old);
        let plugins = runtime.plugin_ids().collect::<Vec<_>>();
        drop(old);

        let mut new = renderer(&device);
        reattach_plugin::<TestFeature>(&mut handle, &device, &queue, Some(&mut runtime), &mut new)
            .expect("reattach");
        assert_eq!(
            new.resources().deformer_id_by_name("test_feature"),
            Some(handle.deformer_id),
            "the handle names the deformer on the new renderer"
        );
        assert_eq!(runtime.plugin_ids().collect::<Vec<_>>(), plugins);

        uninstall_plugin::<TestFeature>(handle, &device, &queue, Some(&mut runtime), &mut new)
            .expect("uninstall after reattach");
        assert_eq!(runtime.plugin_count(), 0);
        assert_eq!(new.resources().deformer_id_by_name("test_feature"), None);
    }
}
