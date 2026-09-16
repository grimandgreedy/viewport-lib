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

mod cpu_pick;
mod draw;
mod pipeline;
mod ribbon;
mod streamtube;
mod tube;

pub(crate) use ribbon::{RibbonPlugin, TYPE_NAME as RIBBON_TYPE_NAME};
pub(crate) use streamtube::{StreamtubePlugin, TYPE_NAME as STREAMTUBE_TYPE_NAME};
pub(crate) use tube::{TYPE_NAME as TUBE_TYPE_NAME, TubePlugin};

use crate::plugin_api::PluginItemCollection;
use crate::renderer::{
    RibbonItem, RibbonRefItem, StreamtubeItem, StreamtubeRefItem, TubeItem, TubeRefItem,
};

macro_rules! item_collection {
    ($ty:ty) => {
        impl PluginItemCollection for Vec<$ty> {
            fn len(&self) -> usize {
                self.len()
            }
            fn item_settings(&self, index: usize) -> &crate::scene::material::ItemSettings {
                &self[index].settings
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
    };
}

item_collection!(StreamtubeItem);
item_collection!(StreamtubeRefItem);
item_collection!(TubeItem);
item_collection!(TubeRefItem);
item_collection!(RibbonItem);
item_collection!(RibbonRefItem);
