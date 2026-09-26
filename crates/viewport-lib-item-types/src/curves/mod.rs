//! The three curve mesh item types as [`ItemTypePlugin`]s: streamtube, tube
//! and ribbon.
//!
//! All three take strips of world-space points and sweep them into a connected
//! triangle mesh on the CPU, so they share their pick, POLY_NODE pick and
//! outline-mask machinery ([`pipeline`]) and their screen-space CPU pick
//! helpers ([`cpu_pick`]). They stay three plugins with three type names,
//! because consumers submit them on three `SceneFrame` fields; each owns its
//! own pipelines, so pulling one out later does not disturb the others.
//!
//! Streamtube and tube shade through the same pipeline description and
//! therefore compile it once each. Ribbon differs: it is a flat two-sided quad
//! strip, so it keys its pipelines on blend mode, casts shadows, and routes
//! transparent draws through the OIT pass.

pub use types::{
    RibbonId, RibbonItem, RibbonRefItem, StreamtubeId, StreamtubeItem, StreamtubeRefItem, TubeId,
    TubeItem, TubeRefItem,
};

mod cpu_pick;
mod draw;
mod pipeline;
mod ribbon;
mod store;
mod streamtube;
mod tube;
mod types;

pub use ribbon::RibbonPlugin;
pub(crate) use ribbon::TYPE_NAME as RIBBON_TYPE_NAME;
pub use streamtube::StreamtubePlugin;
pub(crate) use streamtube::TYPE_NAME as STREAMTUBE_TYPE_NAME;
pub(crate) use tube::TYPE_NAME as TUBE_TYPE_NAME;
pub use tube::TubePlugin;

/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::shader::{lit_shader, scene_shader, wgsl_source};
    use viewport_lib::plugin_api::shared_wgsl;
    vec![
        (
            "curve_pick.wgsl",
            scene_shader(&[], wgsl_source!("curve_pick")),
        ),
        (
            "curve_pick_node.wgsl",
            scene_shader(&[], wgsl_source!("curve_pick_node")),
        ),
        (
            "curve_outline_mask.wgsl",
            scene_shader(&[], wgsl_source!("curve_outline_mask")),
        ),
        (
            "streamtube.wgsl",
            lit_shader(
                &[shared_wgsl::SHARED_CLIP_VOLUME_WGSL],
                wgsl_source!("streamtube"),
            ),
        ),
        (
            "ribbon.wgsl",
            lit_shader(
                &[
                    shared_wgsl::SHARED_CLIP_VOLUME_WGSL,
                    shared_wgsl::SHARED_CSM_WGSL,
                ],
                wgsl_source!("ribbon"),
            ),
        ),
        (
            "ribbon_oit.wgsl",
            lit_shader(
                &[
                    shared_wgsl::SHARED_CLIP_VOLUME_WGSL,
                    shared_wgsl::SHARED_CSM_WGSL,
                ],
                wgsl_source!("ribbon_oit"),
            ),
        ),
        (
            "ribbon_shadow.wgsl",
            wgsl_source!("ribbon_shadow").to_string(),
        ),
    ]
}

macro_rules! item_collection {
    ($ty:ty, $name:expr) => {
        impl viewport_lib::plugin_api::PluginItem for $ty {
            const TYPE_NAME: &'static str = $name;

            fn settings(&self) -> &viewport_lib::ItemSettings {
                &self.settings
            }
        }
    };
}

item_collection!(StreamtubeItem, STREAMTUBE_TYPE_NAME);
item_collection!(StreamtubeRefItem, STREAMTUBE_TYPE_NAME);
item_collection!(TubeItem, TUBE_TYPE_NAME);
item_collection!(TubeRefItem, TUBE_TYPE_NAME);
item_collection!(RibbonItem, RIBBON_TYPE_NAME);
item_collection!(RibbonRefItem, RIBBON_TYPE_NAME);
