//! Fixtures for [`PluginInstaller`](viewport_lib::PluginInstaller): the
//! one-call doorway a feature uses to register whatever pieces it is made of,
//! driven by `install_plugin` or a hand-built `PluginInstallCtx`.
//!
//! - [`DeformAndStepInstaller`]: installs a deformer and a runtime plugin, so
//!   one `install` spans a shader splice and a CPU phase.

mod deform_and_step;

pub use deform_and_step::{DeformAndStepInstaller, DeformAndStepInstallerHandle};
