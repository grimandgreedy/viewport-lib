//! The curves plugin, its pipelines and its shaders: built with
//! the `gpu` feature.

use super::*;
/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::item_types::shader::{lit_shader, scene_shader, wgsl_source};
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
