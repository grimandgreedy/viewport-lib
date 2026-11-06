//! An installer that registers a deformer and a runtime plugin in one call.

use crate::fixtures::CallLog;
use crate::fixtures::deformer::{ConstantOffsetDeformer, constant_offset_deformer};
use crate::fixtures::runtime_plugin::LoggingRuntimePlugin;
use viewport_lib::plugin_api::{PluginInstallCtx, PluginInstaller};
use viewport_lib::runtime::plugin::phase;
use viewport_lib::{DeformerId, RuntimePluginId, ViewportResult};

/// What the host keeps after installing [`DeformAndStepInstaller`].
pub struct DeformAndStepInstallerHandle {
    /// Id of the registered constant-offset deformer, so the host can address
    /// its slot.
    pub deformer: DeformerId,
    /// Id of the registered [`LoggingRuntimePlugin`].
    pub step_plugin: RuntimePluginId,
    /// The log both halves record into.
    pub log: CallLog,
}

/// Installs the constant-offset deformer on the renderer's resources and a
/// [`LoggingRuntimePlugin`] on the runtime, returning both in one handle.
///
/// This is the cross-seam case: the pieces register on two different objects,
/// and the host makes one call. The runtime half is required, so installing
/// against a host with no runtime fails with
/// [`ViewportError::PluginInstallMissing`](viewport_lib::ViewportError::PluginInstallMissing)
/// before anything is registered. [`PluginInstaller::uninstall`] removes the
/// deformer and the runtime plugin again, and
/// [`PluginInstaller::reattach_renderer`] registers the deformer on a new
/// renderer without touching the runtime plugin.
pub struct DeformAndStepInstaller {
    log: CallLog,
}

impl DeformAndStepInstaller {
    /// An installer whose pieces both record into `log`.
    pub fn new(log: CallLog) -> Self {
        Self { log }
    }
}

impl PluginInstaller for DeformAndStepInstaller {
    type Handle = DeformAndStepInstallerHandle;

    fn install(self, ctx: &mut PluginInstallCtx<'_>) -> ViewportResult<Self::Handle> {
        // Claim the runtime first: a missing one must leave the renderer
        // untouched rather than half-installing the feature.
        let runtime = ctx.require_runtime("a ViewportRuntime for the deform-and-step fixture")?;
        let step_plugin = runtime.add_plugin(LoggingRuntimePlugin::new(
            self.log.clone(),
            "installed_step",
            phase::ANIMATE,
        ));

        let deformer = match ctx
            .renderer
            .resources_mut()
            .register_deformer(ctx.device, constant_offset_deformer())
        {
            Ok(id) => id,
            Err(e) => {
                // Take the runtime half back out so a failed install leaves
                // nothing behind.
                if let Some(runtime) = ctx.runtime.as_deref_mut() {
                    runtime.remove_plugin(step_plugin);
                }
                return Err(e);
            }
        };

        Ok(DeformAndStepInstallerHandle {
            deformer,
            step_plugin,
            log: self.log,
        })
    }

    /// Removes the runtime plugin and the deformer, attempting both and
    /// returning the first error.
    fn uninstall(
        handle: DeformAndStepInstallerHandle,
        ctx: &mut PluginInstallCtx<'_>,
    ) -> ViewportResult<()> {
        let runtime = ctx
            .require_runtime("a ViewportRuntime for the deform-and-step fixture")
            .map(|runtime| {
                runtime.remove_plugin(handle.step_plugin);
            });
        let deformer = ctx
            .renderer
            .resources_mut()
            .remove_deformer(ctx.device, handle.deformer);
        runtime.and(deformer)
    }

    /// Registers the deformer on the new renderer, reusing it if it is there
    /// already, and updates the handle. The runtime plugin stays.
    fn reattach_renderer(
        handle: &mut DeformAndStepInstallerHandle,
        ctx: &mut PluginInstallCtx<'_>,
    ) -> ViewportResult<()> {
        let resources = ctx.renderer.resources_mut();
        handle.deformer = match resources.deformer_id_by_name(ConstantOffsetDeformer::NAME) {
            Some(id) => id,
            None => resources.register_deformer(ctx.device, constant_offset_deformer())?,
        };
        Ok(())
    }
}
