//! Minimal plugin implementations, one folder per seam of the plugin API.
//!
//! A fixture is the smallest thing that can sit in a plugin seam and prove it
//! still works: it records which callbacks fired, in what order, with which
//! context values, and (where the seam is visible in the image) makes one
//! deliberate change a pixel assertion can see. The smoke tests in this
//! crate's `tests/` drive them through the real renderer and runtime.
//!
//! They live here, in a separate crate, because that is what makes them worth
//! having: a fixture can only reach `viewport_lib`'s public paths, so one that
//! compiles proves the seam is usable from outside the library rather than
//! only from in-crate code.
//!
//! # Rules
//!
//! - **Minimal, not realistic.** A fixture exercises the seam's core contract
//!   plus its assertion hooks. No believable effects, no tuning knobs. A
//!   fixture that grows features has stopped being a fixture; realistic
//!   implementations belong in the plugins that ship for real use.
//! - **Public surface only.** Fixtures import through `viewport_lib::` public
//!   paths and nothing else. A fixture that cannot be written without a
//!   private item has found a gap in the seam: that is a finding to record,
//!   not something to work around.
//! - **Editing a fixture is a signal.** A change that has to edit a fixture to
//!   keep it compiling has changed the plugin API, and owes a CHANGELOG entry
//!   (and a migration note when the change is breaking). The fixtures cannot
//!   fail the build on their own account, so this is how they earn their keep.
//!
//! # Layout and naming
//!
//! One folder per seam, named after the trait or descriptor it implements, so
//! a seam's fixtures stay together as more of them accumulate. Each folder's
//! `mod.rs` says what the seam is and lists what lives there; a new fixture is
//! a new file plus one line in that `mod.rs`.
//!
//! Fixtures are named `<Variety><Trait>`: the trait name says which seam it
//! sits in, and the prefix says what this one does. `Logging` fixtures record
//! their callbacks and do nothing else; the rest are named after the visible
//! change their tests assert on. Every fixture is also re-exported from this
//! module, so `fixtures::LoggingRuntimePlugin` is the import path and the
//! folders are for navigation.
//!
//! | Folder | Seam | Registered on |
//! | --- | --- | --- |
//! | [`runtime_plugin`] | [`RuntimePlugin`](viewport_lib::RuntimePlugin) | the runtime |
//! | [`gpu_plugin`] | [`GpuPlugin`](viewport_lib::GpuPlugin) | the runtime |
//! | [`item_type_plugin`] | [`ItemTypePlugin`](viewport_lib::plugin_api::ItemTypePlugin) | the renderer |
//! | [`post_effect_producer`] | [`PostEffectProducer`](viewport_lib::PostEffectProducer) | the renderer |
//! | [`post_effect_stage`] | [`PostEffectStage`](viewport_lib::PostEffectStage) | the renderer |
//! | [`deformer`] | [`DeformerDesc`](viewport_lib::DeformerDesc) | the renderer's resources |
//! | [`material_plugin`] | [`MaterialPlugin`](viewport_lib::MaterialPlugin) | the renderer's resources |
//! | [`installer`] | [`PluginInstaller`](viewport_lib::PluginInstaller) | spans the above |

pub mod call_log;
pub mod frame;

pub mod deformer;
pub mod gpu_plugin;
pub mod installer;
pub mod item_type_plugin;
pub mod material_plugin;
pub mod post_effect_producer;
pub mod post_effect_stage;
pub mod runtime_plugin;

pub use call_log::CallLog;
pub use frame::{probe_frame, probe_quad, probe_targets};

pub use deformer::{
    ConstantOffsetDeformer, PerVertexOffsetDeformer, constant_offset_deformer,
    per_vertex_offset_deformer,
};
pub use gpu_plugin::LoggingGpuPlugin;
pub use installer::{DeformAndStepInstaller, DeformAndStepInstallerHandle};
pub use item_type_plugin::{
    ConformanceItemTypePlugin, ConformanceItems, CountedItemCollection, LoggingItemTypePlugin,
    QuadId, StoredId, StoringItemTypePlugin, TriangleItemTypePlugin,
};
pub use material_plugin::{FlatColourMaterialPlugin, TexturedMaterialPlugin};
pub use post_effect_producer::LoggingPostEffectProducer;
pub use post_effect_stage::PassthroughPostEffectStage;
pub use runtime_plugin::{LoggingRuntimePlugin, ProbeEvent};
