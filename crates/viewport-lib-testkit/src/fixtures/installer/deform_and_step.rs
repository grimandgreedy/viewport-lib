//! An installer that registers a deformer and a runtime plugin in one call.

use crate::fixtures::CallLog;
use crate::fixtures::deformer::constant_offset_deformer;
use crate::fixtures::runtime_plugin::LoggingRuntimePlugin;
use viewport_lib::plugin_api::{PluginInstallCtx, PluginInstaller};
use viewport_lib::runtime::plugin::phase;
use viewport_lib::{DeformerId, ViewportResult};

/// What the host keeps after installing [`DeformAndStepInstaller`].
pub struct DeformAndStepInstallerHandle {
    /// Id of the registered constant-offset deformer, so the host can address
    /// its slot.
    pub deformer: DeformerId,
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
/// before anything is registered.
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
        runtime.add_plugin(LoggingRuntimePlugin::new(
            self.log.clone(),
            "installed_step",
            phase::ANIMATE,
        ));

        let deformer = ctx
            .renderer
            .resources_mut()
            .register_deformer(ctx.device, constant_offset_deformer())?;

        Ok(DeformAndStepInstallerHandle {
            deformer,
            log: self.log,
        })
    }
}
